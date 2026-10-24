#!/usr/bin/env python
"""CPU regressions for DINOv2-L and VideoMAE native action policies."""

from __future__ import annotations

import argparse
import copy

import torch

from jam_multimodal.models.backbones.action_dit_dino import (
    ActionDiTDinoLarge24,
)
from jam_multimodal.models.backbones.action_dit_videomae import (
    ActionDiTVideoMAE12,
)
from jam_multimodal.models.backbones.external_ssl import DenseImageSSLBackbone
from jam_multimodal.models.backbones.external_videomae import (
    SingleFrameVideoMAEBackbone,
)


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
        mlp_ratio=4.0,
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


def test_tiny_videomae_protocol() -> None:
    from transformers import VideoMAEConfig, VideoMAEModel

    config = VideoMAEConfig(
        image_size=32,
        patch_size=16,
        tubelet_size=2,
        num_frames=16,
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=256,
        qkv_bias=True,
        use_mean_pooling=False,
    )
    backbone = VideoMAEModel(config)
    source_projection = copy.deepcopy(
        backbone.embeddings.patch_embeddings.projection
    )
    student = SingleFrameVideoMAEBackbone(
        backbone,
        image_size=32,
        num_views=2,
        pixel_mean=[0.5, 0.5, 0.5],
        pixel_std=[0.5, 0.5, 0.5],
    )

    image = torch.randn(2, 3, 32, 32)
    pair = image[:, None].expand(-1, 2, -1, -1, -1)
    expected = source_projection(
        pair.permute(0, 2, 1, 3, 4)
    ).flatten(2).transpose(1, 2)
    actual = student.backbone.embeddings.projection(
        image
    ).flatten(2).transpose(1, 2)
    torch.testing.assert_close(actual, expected, atol=1.0e-5, rtol=1.0e-5)

    action = ActionDiTVideoMAE12.from_videomae_backbone(
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
        videomae_action = ActionDiTVideoMAE12(
            action_dim=7,
            text_dim=4096,
            max_action_horizon=64,
        )
    assert sum(parameter.numel() for parameter in dino_large_action.parameters()) == 459_100_167
    assert sum(parameter.numel() for parameter in videomae_action.parameters()) == 131_554_311


def test_real_videomae() -> None:
    student = SingleFrameVideoMAEBackbone.from_pretrained(
        "checkpoints/videomae-base",
        image_size=224,
        num_views=2,
        local_files_only=True,
        attn_implementation="sdpa",
        device="cpu",
        torch_dtype=torch.float32,
    )
    action = ActionDiTVideoMAE12.from_videomae_backbone(
        student.backbone,
        action_dim=7,
        text_dim=4096,
        max_action_horizon=64,
    )
    assert torch.equal(
        action.blocks[-1].key.weight,
        student.backbone.encoder.layer[-1].attention.attention.key.weight,
    )
    assert sum(parameter.numel() for parameter in action.parameters()) == 131_554_311


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--real-videomae",
        action="store_true",
        help="also load and audit the real VideoMAE-B checkpoint",
    )
    args = parser.parse_args()
    test_tiny_dinov2_large_protocol()
    test_tiny_videomae_protocol()
    test_production_shapes()
    if args.real_videomae:
        test_real_videomae()
    print("NATIVE SSL ACTION VARIANT CHECKS PASSED")


if __name__ == "__main__":
    main()
