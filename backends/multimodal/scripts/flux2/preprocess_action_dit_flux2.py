#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from safetensors.torch import load_file as load_sft

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from jam_multimodal.models.backbones.action_dit_flux2 import ActionDiTFlux2
from jam_multimodal.models.backbones.flux2_imports import ensure_flux2_importable


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


def _parse_bool(value: str) -> bool:
    key = str(value).strip().lower()
    if key in {"1", "true", "yes", "y"}:
        return True
    if key in {"0", "false", "no", "n"}:
        return False
    raise ValueError(f"Cannot parse bool value: {value!r}")


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
    parser = argparse.ArgumentParser(description="Initialize slim FLUX.2 ActionDiT from FLUX.2 video weights.")
    parser.add_argument("--model-config", required=True, help="Hydra model/task yaml containing action_dit_config.")
    parser.add_argument(
        "--flux2-src-path",
        default=os.environ.get("FLUX2_SRC"),
        help="Path to a local FLUX.2 source checkout. Defaults to FLUX2_SRC.",
    )
    parser.add_argument("--flux2-model-path", required=True)
    parser.add_argument("--variant", default="klein-base-4b", choices=["klein-base-4b"])
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--action-dim", type=int, default=7)
    parser.add_argument("--use-gradient-checkpointing", action="store_true")
    parser.add_argument("--apply-alpha-scaling", default="true")
    args = parser.parse_args()
    apply_alpha_scaling = _parse_bool(args.apply_alpha_scaling)
    if not args.flux2_src_path:
        raise ValueError("Set --flux2-src-path or the FLUX2_SRC environment variable.")

    ensure_flux2_importable(args.flux2_src_path)
    from flux2.model import Flux2, Klein4BParams

    params = Klein4BParams()
    with torch.device("meta"):
        flux2 = Flux2(params).to(torch.bfloat16)
    flux2.load_state_dict(load_sft(args.flux2_model_path, device=args.device), strict=True, assign=True)
    flux2 = flux2.to(args.device)

    action_cfg = _load_action_cfg(
        args.model_config,
        action_dim=args.action_dim,
        use_gradient_checkpointing=args.use_gradient_checkpointing,
    )
    action_cfg["num_heads"] = int(flux2.num_heads)
    action_cfg["attn_head_dim"] = int(flux2.hidden_size) // int(flux2.num_heads)
    action_cfg["num_layers_double"] = int(len(flux2.double_blocks))
    action_cfg["num_layers_single"] = int(len(flux2.single_blocks))
    action_expert = ActionDiTFlux2(**action_cfg).to(args.device)

    src_state = flux2.state_dict()
    dst_state = action_expert.state_dict()
    out_state = {}
    copied = 0
    resized = 0
    for key, dst in dst_state.items():
        if key.startswith(("action_encoder.", "head.")):
            continue
        src_key = key
        if key.startswith("double_blocks."):
            parts = key.split(".")
            src_key = ".".join(["double_blocks", parts[1], *parts[2:]])
        if src_key not in src_state:
            continue
        src = src_state[src_key]
        if tuple(src.shape) == tuple(dst.shape):
            value = src
            copied += 1
        else:
            value = _resize_tensor_to_shape(src, tuple(dst.shape))
            if apply_alpha_scaling and src.ndim >= 2 and src.shape[-1] != dst.shape[-1]:
                alpha = (float(src.shape[-1]) / float(dst.shape[-1])) ** 0.5
                value = value.to(torch.float32) * alpha
            resized += 1
        out_state[key] = value.detach().cpu().to(dtype=dst.dtype).contiguous()

    payload = {
        "state_dict": out_state,
        "meta": {
            "source": args.flux2_model_path,
            "variant": args.variant,
            "copied": copied,
            "resized": resized,
            "policy": "copy FLUX.2 img/single branches into slim action blocks; interpolate mismatched axes with alpha scaling",
            "alpha_scaling": bool(apply_alpha_scaling),
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output)
    print(f"[ok] saved {output} copied={copied} resized={resized} keys={len(out_state)}")


if __name__ == "__main__":
    main()
