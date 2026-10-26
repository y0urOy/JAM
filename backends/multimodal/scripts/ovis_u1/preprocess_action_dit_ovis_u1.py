#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

from jam_multimodal.models.backbones.action_dit_yak import ActionDiTYak
from jam_multimodal.models.backbones.ovis_u1_imports import ensure_ovis_u1_remote_code_importable


def _interpolate_last_dim(tensor: torch.Tensor, new_size: int) -> torch.Tensor:
    if tensor.shape[-1] == new_size:
        return tensor
    flat = tensor.reshape(-1, 1, tensor.shape[-1]).to(torch.float32)
    flat = F.interpolate(flat, size=new_size, mode="linear", align_corners=True)
    return flat.reshape(*tensor.shape[:-1], new_size)


def _resize_tensor_to_shape(src: torch.Tensor, target_shape: tuple[int, ...]) -> torch.Tensor:
    if tuple(src.shape) == tuple(target_shape):
        return src
    out = src.to(torch.float32)
    while out.ndim < len(target_shape):
        out = out.unsqueeze(0)
    while out.ndim > len(target_shape):
        if out.shape[0] != 1:
            raise ValueError(f"Cannot reduce tensor rank: src={tuple(src.shape)} target={target_shape}")
        out = out.squeeze(0)
    for dim, new_size in enumerate(target_shape):
        if out.shape[dim] == new_size:
            continue
        perm = [i for i in range(out.ndim) if i != dim] + [dim]
        inv_perm = [0] * out.ndim
        for i, p in enumerate(perm):
            inv_perm[p] = i
        out = _interpolate_last_dim(out.permute(*perm).contiguous(), new_size).permute(*inv_perm).contiguous()
    return out.to(dtype=src.dtype)


def _is_unresolved(value: Any) -> bool:
    return isinstance(value, str) and "${" in value and "}" in value


def _load_action_cfg(path: str, action_dim: int, use_gradient_checkpointing: bool) -> dict[str, Any]:
    cfg = OmegaConf.load(path)
    action_cfg = cfg.model.action_dit_config if "model" in cfg else cfg.action_dit_config
    action_cfg = OmegaConf.to_container(action_cfg, resolve=False)
    if not isinstance(action_cfg, dict):
        raise ValueError(f"`action_dit_config` must resolve to a dict, got {type(action_cfg)}")
    if _is_unresolved(action_cfg.get("action_dim")):
        action_cfg["action_dim"] = int(action_dim)
    if _is_unresolved(action_cfg.get("use_gradient_checkpointing")):
        action_cfg["use_gradient_checkpointing"] = bool(use_gradient_checkpointing)
    unresolved = {key: value for key, value in action_cfg.items() if _is_unresolved(value)}
    if unresolved:
        raise ValueError(f"Unresolved action_dit_config values remain: {unresolved}")
    return action_cfg


def main() -> None:
    parser = argparse.ArgumentParser(description="Initialize Yak-shaped ActionDiT from Ovis-U1 Yak weights.")
    parser.add_argument("--model-config", required=True, help="Hydra model/task yaml containing action_dit_config.")
    parser.add_argument("--ovis-u1-model-path", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--action-dim", type=int, default=7)
    parser.add_argument("--use-gradient-checkpointing", action="store_true")
    args = parser.parse_args()

    # The loader of JAMPolicy.from_ovis_u1_pretrained: the vendored Ovis-U1 code as one regular
    # package. Every module in float32, as Transformers 4 loaded this call by default.
    ensure_ovis_u1_remote_code_importable()
    from ovis_u1_hf.modeling_ovis_u1 import OvisU1

    ovis_model = OvisU1.from_pretrained(args.ovis_u1_model_path, torch_dtype=torch.float32).to(args.device)
    yak = ovis_model.get_visual_generator().get_backbone()
    action_cfg = _load_action_cfg(
        args.model_config,
        action_dim=args.action_dim,
        use_gradient_checkpointing=args.use_gradient_checkpointing,
    )
    action_cfg["residual_dim"] = int(yak.hidden_size)
    action_cfg["num_heads"] = int(yak.num_heads)
    action_cfg["num_layers_double"] = int(len(yak.double_blocks))
    action_cfg["num_layers_single"] = int(len(yak.single_blocks))
    action_expert = ActionDiTYak(**action_cfg).to(args.device)

    src_state = yak.state_dict()
    dst_state = action_expert.state_dict()
    out_state = {}
    copied = 0
    resized = 0
    for key, dst in dst_state.items():
        if key.startswith(("action_encoder.", "head.")):
            continue
        src_key = key
        if src_key not in src_state:
            continue
        src = src_state[src_key]
        if tuple(src.shape) == tuple(dst.shape):
            value = src
            copied += 1
        else:
            value = _resize_tensor_to_shape(src, tuple(dst.shape))
            resized += 1
        out_state[key] = value.detach().cpu().to(dtype=dst.dtype).contiguous()

    payload = {
        "state_dict": out_state,
        "meta": {
            "source": args.ovis_u1_model_path,
            "copied": copied,
            "resized": resized,
            "policy": "copy matching Yak time/double/single blocks; randomly initialize action encoder/head",
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output)
    print(f"[ok] saved {output} copied={copied} resized={resized} keys={len(out_state)}")


if __name__ == "__main__":
    main()
