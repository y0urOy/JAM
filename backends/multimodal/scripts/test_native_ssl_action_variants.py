#!/usr/bin/env python
"""CPU regressions for the DINOv2-L native action policy."""

from __future__ import annotations

import torch

from jam_multimodal.models.backbones.action_dit_dino import (
    ActionDiTDinoLarge24,
)
from jam_multimodal.models.backbones.external_ssl import DenseImageSSLBackbone


def _run_action_roundtrip(action, student, image_size: int) -> None:
    images = torch.randn(1, 3, image_size, 2 * image_size).clamp_(-1.0, 1.0)
    text = torch.randn(1, 5, 32)
    text_mask = torch.ones(1, 5, dtype=torch.bool)
    cache = action.prefill_dino_prefix(
        student,
        images,
        text,
        text_mask,
    )
    action_pre = action.pre_dit(
        torch.randn(1, 8, 7),
        torch.tensor([0.5]),
    )
    tokens = action.forward_with_prefix_cache(action_pre, cache)
    prediction = action.post_dit(tokens, action_pre)
    loss = prediction.float().square().mean()
    loss = loss + cache["final_dense"].float().square().mean()
    loss.backward()
    assert tuple(prediction.shape) == (1, 8, 7)
    assert bool(torch.isfinite(prediction).all())
    assert len(cache["layer_patch_kv"]) == action.num_layers


def test_tiny_dinov2_large_protocol() -> None:
    from transformers import Dinov2Config, Dinov2Model

    config = Dinov2Config(
        image_size=32,
        patch_size=8,
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        mlp_ratio=4,
        use_swiglu_ffn=False,
    )
    student = DenseImageSSLBackbone(
        Dinov2Model(config),
        image_size=32,
        num_views=2,
        pixel_mean=[0.5, 0.5, 0.5],
        pixel_std=[0.5, 0.5, 0.5],
    )
    action = ActionDiTDinoLarge24.from_dino_backbone(
        student.backbone,
        action_dim=7,
        text_dim=32,
        max_action_horizon=8,
        strict_shape=False,
    )
    assert (action.num_layers, action.hidden_size, action.num_heads) == (
        2,
        64,
        4,
    )
    _run_action_roundtrip(action, student, image_size=32)


def test_production_shapes() -> None:
    with torch.device("meta"):
        dino_large_action = ActionDiTDinoLarge24(
            action_dim=7,
            text_dim=4096,
            max_action_horizon=64,
        )
    assert sum(parameter.numel() for parameter in dino_large_action.parameters()) == 459_100_167


def main() -> None:
    test_tiny_dinov2_large_protocol()
    test_production_shapes()
    print("NATIVE SSL ACTION VARIANT CHECKS PASSED")


if __name__ == "__main__":
    main()
