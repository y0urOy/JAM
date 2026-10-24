#!/usr/bin/env python3
"""CPU checks for the native DINOv2-B / ActionDiTDino12 MoT path.

The tiny check deliberately uses the same public integration API as JAMPolicy:

``from_dino_backbone -> prefill_dino_prefix -> pre_dit ->
forward_with_prefix_cache -> post_dit``.

Passing ``--model-dir`` additionally checks the real local DINOv2-B weights.
No dataset, Ovis checkpoint, GPU, or distributed runtime is required.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from transformers import Dinov2Config, Dinov2Model

from jam_multimodal.models.backbones.action_dit_dino import ActionDiTDino12
from jam_multimodal.models.backbones.external_ssl import DenseImageSSLBackbone


def _assert_finite(tensor: torch.Tensor, name: str) -> None:
    if not bool(torch.isfinite(tensor).all()):
        raise AssertionError(f"{name} contains NaN or Inf.")


def _assert_nonzero_finite_grad(parameter: torch.Tensor, name: str) -> None:
    if parameter.grad is None:
        raise AssertionError(f"{name} did not receive a gradient.")
    _assert_finite(parameter.grad, f"{name}.grad")
    if float(parameter.grad.abs().sum()) == 0.0:
        raise AssertionError(f"{name}.grad is identically zero.")


def _make_tiny_models() -> tuple[DenseImageSSLBackbone, ActionDiTDino12]:
    config = Dinov2Config(
        image_size=16,
        patch_size=4,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        mlp_ratio=2.0,
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
        drop_path_rate=0.0,
        qkv_bias=True,
        layerscale_value=1.0,
        layer_norm_eps=1.0e-6,
        use_swiglu_ffn=False,
    )
    backbone = Dinov2Model(config)
    ssl_student = DenseImageSSLBackbone(
        backbone,
        image_size=16,
        num_views=2,
        pixel_mean=(0.485, 0.456, 0.406),
        pixel_std=(0.229, 0.224, 0.225),
    )
    action_expert = ActionDiTDino12.from_dino_backbone(
        backbone,
        action_dim=7,
        text_dim=24,
        max_action_horizon=5,
        timestep_embed_dim=16,
        strict_shape=False,
    )
    return ssl_student, action_expert


def _check_copied_dino_blocks(
    ssl_student: DenseImageSSLBackbone,
    action_expert: ActionDiTDino12,
) -> None:
    dino_layers = ssl_student.backbone.encoder.layer
    assert len(dino_layers) == len(action_expert.blocks)
    for layer_index, (source, target) in enumerate(
        zip(dino_layers, action_expert.blocks)
    ):
        pairs = (
            (
                source.attention.attention.query.weight,
                target.query.weight,
                "query",
            ),
            (
                source.attention.attention.key.weight,
                target.key.weight,
                "key",
            ),
            (
                source.attention.attention.value.weight,
                target.value.weight,
                "value",
            ),
            (
                source.attention.output.dense.weight,
                target.out.weight,
                "attention output",
            ),
            (source.mlp.fc1.weight, target.mlp.fc1.weight, "MLP fc1"),
            (source.mlp.fc2.weight, target.mlp.fc2.weight, "MLP fc2"),
        )
        for source_tensor, target_tensor, label in pairs:
            if not torch.equal(source_tensor, target_tensor):
                raise AssertionError(
                    f"Layer {layer_index} copied {label} weights differ."
                )
        if bool(target.modulation[-1].weight.count_nonzero()):
            raise AssertionError(
                f"Layer {layer_index} AdaLN modulation weight is not zero-initialized."
            )
        if bool(target.modulation[-1].bias.count_nonzero()):
            raise AssertionError(
                f"Layer {layer_index} AdaLN modulation bias is not zero-initialized."
            )


def check_tiny_native_forward_backward() -> None:
    torch.manual_seed(42)
    ssl_student, action_expert = _make_tiny_models()
    ssl_student.train()
    action_expert.train()
    _check_copied_dino_blocks(ssl_student, action_expert)

    batch_size = 2
    action_len = 5
    text_len = 3
    images = torch.randn(batch_size, 3, 16, 32).clamp_(-1.0, 1.0)
    text = torch.randn(
        batch_size,
        text_len,
        action_expert.text_dim,
        requires_grad=True,
    )
    text_mask = torch.tensor(
        [[True, True, False], [True, True, True]],
        dtype=torch.bool,
    )

    # Native prefill must leave DINO's established image path bit-identical.
    legacy_final = ssl_student(images)
    prefix_cache = action_expert.prefill_dino_prefix(
        ssl_student,
        images,
        text,
        text_mask,
    )
    native_final = prefix_cache["final_dense"]
    if not torch.equal(legacy_final, native_final):
        max_abs = float((legacy_final - native_final).abs().max())
        raise AssertionError(
            "Native DINO prefill changed the visual forward; "
            f"maximum absolute difference is {max_abs:.9g}."
        )

    expected_patch_tokens = 2 * (16 // 4) ** 2
    assert tuple(native_final.shape) == (
        batch_size,
        expected_patch_tokens,
        action_expert.hidden_size,
    )
    assert len(prefix_cache["layer_patch_kv"]) == action_expert.num_layers
    assert len(prefix_cache["layers"]) == action_expert.num_layers
    for layer_index, layer_kv in enumerate(prefix_cache["layer_patch_kv"]):
        expected_kv_shape = (
            batch_size,
            action_expert.num_heads,
            expected_patch_tokens,
            action_expert.head_dim,
        )
        assert tuple(layer_kv.key.shape) == expected_kv_shape, layer_index
        assert tuple(layer_kv.value.shape) == expected_kv_shape, layer_index
        _assert_finite(layer_kv.key, f"layer {layer_index} visual key")
        _assert_finite(layer_kv.value, f"layer {layer_index} visual value")

    noisy_action = torch.randn(batch_size, action_len, 7)
    timestep = torch.tensor([0.2, 0.8])
    action_pre = action_expert.pre_dit(noisy_action, timestep)
    assert tuple(action_pre["tokens"].shape) == (
        batch_size,
        action_len,
        action_expert.hidden_size,
    )
    assert tuple(action_pre["t_mod"].shape) == (
        batch_size,
        action_expert.hidden_size,
    )
    action_hidden = action_expert.forward_with_prefix_cache(
        action_pre,
        prefix_cache,
    )
    prediction = action_expert.post_dit(action_hidden, action_pre)
    assert tuple(prediction.shape) == (batch_size, action_len, 7)
    _assert_finite(action_hidden, "tiny action hidden states")
    _assert_finite(prediction, "tiny action prediction")

    # The action objective reaches DINO through per-layer visual K/V. The
    # final-dense term mirrors foresight supervision and reaches the final
    # native DINO layer as well.
    loss = prediction.float().square().mean()
    loss = loss + 0.01 * native_final.float().square().mean()
    loss.backward()
    _assert_finite(loss.detach(), "tiny loss")

    _assert_nonzero_finite_grad(
        ssl_student.backbone.embeddings.patch_embeddings.projection.weight,
        "DINO patch embedding",
    )
    _assert_nonzero_finite_grad(
        ssl_student.backbone.encoder.layer[0].attention.attention.key.weight,
        "DINO layer-0 key",
    )
    _assert_nonzero_finite_grad(
        ssl_student.backbone.encoder.layer[-1].mlp.fc2.weight,
        "DINO final-layer MLP",
    )
    _assert_nonzero_finite_grad(
        action_expert.action_encoder.weight,
        "Action encoder",
    )
    _assert_nonzero_finite_grad(
        action_expert.blocks[0].key.weight,
        "Action layer-0 key",
    )
    _assert_nonzero_finite_grad(
        action_expert.blocks[-1].mlp.fc2.weight,
        "Action final-layer MLP",
    )
    _assert_nonzero_finite_grad(action_expert.head.weight, "Action head")
    _assert_nonzero_finite_grad(text, "Text condition")


@torch.no_grad()
def check_real_dinov2_base(model_dir: Path) -> None:
    ssl_student = DenseImageSSLBackbone.from_pretrained(
        str(model_dir),
        image_size=224,
        num_views=2,
        local_files_only=True,
        attn_implementation="sdpa",
        device="cpu",
        torch_dtype=torch.float32,
    ).eval()
    action_expert = ActionDiTDino12.from_dino_backbone(
        ssl_student.backbone,
        action_dim=7,
        text_dim=4096,
        max_action_horizon=16,
    ).eval()
    _check_copied_dino_blocks(ssl_student, action_expert)

    assert action_expert.hidden_size == 768
    assert action_expert.num_heads == 12
    assert action_expert.head_dim == 64
    assert action_expert.num_layers == 12
    assert len(action_expert.blocks) == 12

    image = torch.zeros(1, 3, 224, 448)
    text = torch.zeros(1, 2, 4096)
    text_mask = torch.tensor([[True, False]])
    legacy_final = ssl_student(image)
    prefix_cache = action_expert.prefill_dino_prefix(
        ssl_student,
        image,
        text,
        text_mask,
    )
    native_final = prefix_cache["final_dense"]
    assert tuple(native_final.shape) == (1, 512, 768)
    if not torch.equal(legacy_final, native_final):
        max_abs = float((legacy_final - native_final).abs().max())
        raise AssertionError(
            "Real DINOv2-B native prefill is not bit-identical to the "
            f"established wrapper forward (max_abs={max_abs:.9g})."
        )
    assert len(prefix_cache["layer_patch_kv"]) == 12
    assert all(
        tuple(item.key.shape) == (1, 12, 512, 64)
        and tuple(item.value.shape) == (1, 12, 512, 64)
        for item in prefix_cache["layer_patch_kv"]
    )
    _assert_finite(native_final, "real DINOv2-B final dense tokens")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=None,
        help="Optionally validate the real local DINOv2-B checkpoint.",
    )
    args = parser.parse_args()

    check_tiny_native_forward_backward()
    if args.model_dir is not None:
        check_real_dinov2_base(args.model_dir)
    print(
        "DINO NATIVE ACTION CHECKS PASSED "
        f"tiny_forward_backward=True "
        f"real_dino={args.model_dir is not None}"
    )


if __name__ == "__main__":
    main()
