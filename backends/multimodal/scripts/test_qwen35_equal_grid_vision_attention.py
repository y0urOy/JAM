#!/usr/bin/env python
"""Bitwise CPU regression for Qwen equal-grid vision attention batching."""

from __future__ import annotations

import argparse
import copy
from types import MethodType, SimpleNamespace

import torch

from jam_multimodal.models.backbones.qwen35_vision_attention import (
    _equal_grid_attention_forward,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument(
        "--dtype", choices=("float32", "bfloat16"), default="float32"
    )
    return parser.parse_args()


def main() -> None:
    from transformers import AutoConfig
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5VisionAttention

    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA regression requested but CUDA is unavailable.")
    device = torch.device(args.device)
    dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16}[args.dtype]
    if device.type == "cpu" and dtype != torch.float32:
        raise ValueError("The deterministic CPU reference must use float32.")
    torch.manual_seed(3407)
    config = copy.deepcopy(
        AutoConfig.from_pretrained(
            "checkpoints/Qwen3.5-2B-Base", local_files_only=True
        ).vision_config
    )
    config._attn_implementation = "sdpa"
    reference = Qwen3_5VisionAttention(config).to(device=device, dtype=dtype).train()
    batched = copy.deepcopy(reference)

    tokens_per_image = (224 // int(config.patch_size)) ** 2
    batched._imagewam_equal_grid_tokens = tokens_per_image
    batched.forward = MethodType(_equal_grid_attention_forward, batched)

    num_images = 3
    seq_length = num_images * tokens_per_image
    hidden_reference = torch.randn(
        seq_length,
        int(config.hidden_size),
        device=device,
        dtype=dtype,
        requires_grad=True,
    )
    hidden_batched = hidden_reference.detach().clone().requires_grad_(True)
    angles = torch.randn(
        seq_length, reference.head_dim, device=device, dtype=torch.float32
    )
    position_embeddings = (angles.cos(), angles.sin())
    cu_seqlens = torch.arange(
        0,
        (num_images + 1) * tokens_per_image,
        tokens_per_image,
        device=device,
        dtype=torch.int32,
    )

    expected = reference(
        hidden_reference,
        cu_seqlens=cu_seqlens,
        position_embeddings=position_embeddings,
    )
    actual = batched(
        hidden_batched,
        cu_seqlens=cu_seqlens,
        position_embeddings=position_embeddings,
    )
    if not torch.equal(expected, actual):
        raise AssertionError(
            f"forward mismatch: max_abs={(expected - actual).abs().max().item()}"
        )

    upstream = torch.randn_like(expected)
    expected.backward(upstream)
    actual.backward(upstream)
    if not torch.equal(hidden_reference.grad, hidden_batched.grad):
        raise AssertionError(
            "input-gradient mismatch: max_abs="
            f"{(hidden_reference.grad - hidden_batched.grad).abs().max().item()}"
        )
    for (name_reference, parameter_reference), (name_batched, parameter_batched) in zip(
        reference.named_parameters(), batched.named_parameters(), strict=True
    ):
        if name_reference != name_batched:
            raise AssertionError((name_reference, name_batched))
        if not torch.equal(parameter_reference.grad, parameter_batched.grad):
            raise AssertionError(
                f"parameter-gradient mismatch for {name_reference}: max_abs="
                f"{(parameter_reference.grad - parameter_batched.grad).abs().max().item()}"
            )

    try:
        batched(
            hidden_batched.detach(),
            cu_seqlens=torch.tensor([0, tokens_per_image, seq_length - 1]),
            position_embeddings=position_embeddings,
        )
    except ValueError as exc:
        if "incompatible layout" not in str(exc):
            raise
    else:
        raise AssertionError("An incompatible vision layout must fail loudly.")

    print(
        "QWEN35_EQUAL_GRID_VISION_ATTENTION_BITWISE_PASS "
        f"images={num_images} tokens_per_image={tokens_per_image} "
        f"hidden={config.hidden_size} heads={config.num_heads} "
        f"device={device.type} dtype={dtype}"
    )


if __name__ == "__main__":
    main()
