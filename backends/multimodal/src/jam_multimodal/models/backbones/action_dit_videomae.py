"""VideoMAE-native action expert for JAMPolicy.

The trainable visual student keeps VideoMAE-B's 12 transformer blocks.  A
matching 12-layer action expert copies each block's LayerNorm, Q/K/V/output,
and MLP weights, then adds action/timestep interfaces.  At depth ``i`` the
action queries attend one softmax over Ovis text, the two cameras' VideoMAE
patch K/V from layer ``i``, and action self K/V.  The visual stream never
reads action tokens.
"""

from __future__ import annotations

from typing import Any, Optional

import torch
import torch.nn as nn

from .action_dit_dino import (
    ActionDiTDino12,
    DinoActionBlock,
    DinoLayerPatchKV,
    DinoNativeOutput,
)


class VideoMAEActionBlock(DinoActionBlock):
    """VideoMAE-shaped action block without DINO LayerScale parameters."""

    def __init__(
        self,
        *,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float,
        layer_norm_eps: float,
        qkv_bias: bool,
    ) -> None:
        super().__init__(
            hidden_size=int(hidden_size),
            num_heads=int(num_heads),
            mlp_ratio=float(mlp_ratio),
            layer_norm_eps=float(layer_norm_eps),
            layerscale_value=1.0,
            qkv_bias=bool(qkv_bias),
        )
        # VideoMAE has plain residual branches rather than learned LayerScale.
        self.layer_scale1 = nn.Identity()
        self.layer_scale2 = nn.Identity()

    @torch.no_grad()
    def copy_from_hf_videomae_layer(self, source: nn.Module) -> None:
        required = (
            "layernorm_before",
            "attention",
            "layernorm_after",
            "intermediate",
            "output",
        )
        missing = [name for name in required if not hasattr(source, name)]
        if missing:
            raise TypeError(
                "Source is not a compatible HuggingFace VideoMAELayer; "
                f"missing {missing}."
            )
        self_attention = source.attention.attention
        self.norm1.load_state_dict(
            source.layernorm_before.state_dict(),
            strict=True,
        )
        self.query.load_state_dict(self_attention.query.state_dict(), strict=True)
        self.key.load_state_dict(self_attention.key.state_dict(), strict=True)
        self.value.load_state_dict(self_attention.value.state_dict(), strict=True)
        self.out.load_state_dict(
            source.attention.output.dense.state_dict(),
            strict=True,
        )
        self.norm2.load_state_dict(
            source.layernorm_after.state_dict(),
            strict=True,
        )
        self.mlp.fc1.load_state_dict(
            source.intermediate.dense.state_dict(),
            strict=True,
        )
        self.mlp.fc2.load_state_dict(
            source.output.dense.state_dict(),
            strict=True,
        )


class ActionDiTVideoMAE12(ActionDiTDino12):
    """12×768 action expert initialized one-to-one from VideoMAE-B."""

    EXPECTED_NUM_LAYERS = 12
    EXPECTED_HIDDEN_SIZE = 768
    EXPECTED_NUM_HEADS = 12

    def __init__(
        self,
        *,
        action_dim: int = 7,
        text_dim: int = 4096,
        hidden_size: int = EXPECTED_HIDDEN_SIZE,
        num_heads: int = EXPECTED_NUM_HEADS,
        num_layers: int = EXPECTED_NUM_LAYERS,
        mlp_ratio: float = 4.0,
        layer_norm_eps: float = 1.0e-12,
        qkv_bias: bool = False,
        max_action_horizon: int = 64,
        timestep_embed_dim: int = 256,
        time_factor: float = 1000.0,
        initializer_range: float = 0.02,
        strict_shape: bool = True,
    ) -> None:
        strict_shape = bool(strict_shape)
        hidden_size = int(hidden_size)
        num_heads = int(num_heads)
        num_layers = int(num_layers)
        if strict_shape:
            expected = (
                self.EXPECTED_HIDDEN_SIZE,
                self.EXPECTED_NUM_HEADS,
                self.EXPECTED_NUM_LAYERS,
            )
            actual = (hidden_size, num_heads, num_layers)
            if actual != expected:
                raise ValueError(
                    "ActionDiTVideoMAE12 production topology mismatch: "
                    f"actual={actual}, expected={expected}."
                )
        super().__init__(
            action_dim=int(action_dim),
            text_dim=int(text_dim),
            hidden_size=hidden_size,
            num_heads=num_heads,
            num_layers=num_layers,
            mlp_ratio=float(mlp_ratio),
            layer_norm_eps=float(layer_norm_eps),
            layerscale_value=1.0,
            qkv_bias=bool(qkv_bias),
            max_action_horizon=int(max_action_horizon),
            timestep_embed_dim=int(timestep_embed_dim),
            time_factor=float(time_factor),
            initializer_range=float(initializer_range),
            strict_shape=False,
        )
        self.blocks = nn.ModuleList(
            [
                VideoMAEActionBlock(
                    hidden_size=hidden_size,
                    num_heads=num_heads,
                    mlp_ratio=float(mlp_ratio),
                    layer_norm_eps=float(layer_norm_eps),
                    qkv_bias=bool(qkv_bias),
                )
                for _ in range(num_layers)
            ]
        )
        self.strict_shape = strict_shape
        self.reset_parameters()

    @staticmethod
    def _validate_hf_videomae(
        backbone: nn.Module,
        *,
        expected_hidden_size: Optional[int] = None,
        expected_num_layers: Optional[int] = None,
        expected_num_heads: Optional[int] = None,
    ) -> tuple[nn.ModuleList, Any]:
        config = getattr(backbone, "config", None)
        encoder = getattr(backbone, "encoder", None)
        layers = getattr(encoder, "layer", None)
        if config is None or layers is None:
            raise TypeError(
                "Expected a HuggingFace VideoMAEModel exposing config and "
                "encoder.layer."
            )
        actual = {
            "hidden_size": int(getattr(config, "hidden_size", -1)),
            "num_hidden_layers": int(
                getattr(config, "num_hidden_layers", -1)
            ),
            "num_attention_heads": int(
                getattr(config, "num_attention_heads", -1)
            ),
        }
        expected = {
            "hidden_size": expected_hidden_size,
            "num_hidden_layers": expected_num_layers,
            "num_attention_heads": expected_num_heads,
        }
        mismatches = {
            name: (actual[name], int(value))
            for name, value in expected.items()
            if value is not None and actual[name] != int(value)
        }
        if mismatches:
            raise ValueError(
                f"VideoMAE/action topology mismatch: {mismatches}."
            )
        if len(layers) != actual["num_hidden_layers"]:
            raise ValueError(
                "VideoMAE config/encoder depth mismatch: "
                f"{actual['num_hidden_layers']} versus {len(layers)}."
            )
        if any(value <= 0 for value in actual.values()):
            raise ValueError(f"Invalid VideoMAE dimensions: {actual}.")
        if actual["hidden_size"] % actual["num_attention_heads"] != 0:
            raise ValueError(
                "VideoMAE hidden size must divide evenly over attention "
                f"heads, got {actual}."
            )
        if not hasattr(backbone, "embeddings") or not isinstance(
            getattr(backbone, "layernorm", None),
            nn.Module,
        ):
            raise TypeError(
                "VideoMAE backbone must expose embeddings and a token-wise "
                "final layernorm (use_mean_pooling=false)."
            )
        return layers, config

    @classmethod
    def from_hf_videomae(
        cls,
        backbone: nn.Module,
        **kwargs: Any,
    ) -> "ActionDiTVideoMAE12":
        strict_shape = bool(kwargs.pop("strict_shape", True))
        layers, config = cls._validate_hf_videomae(
            backbone,
            expected_hidden_size=(
                cls.EXPECTED_HIDDEN_SIZE if strict_shape else None
            ),
            expected_num_layers=(
                cls.EXPECTED_NUM_LAYERS if strict_shape else None
            ),
            expected_num_heads=(
                cls.EXPECTED_NUM_HEADS if strict_shape else None
            ),
        )
        first_attention = layers[0].attention.attention
        qkv_bias = first_attention.query.bias is not None
        fixed = {
            "hidden_size": int(config.hidden_size),
            "num_heads": int(config.num_attention_heads),
            "num_layers": int(config.num_hidden_layers),
            "mlp_ratio": (
                float(config.intermediate_size)
                / float(config.hidden_size)
            ),
            "layer_norm_eps": float(config.layer_norm_eps),
            "qkv_bias": bool(qkv_bias),
            "initializer_range": float(config.initializer_range),
            "strict_shape": strict_shape,
        }
        overlaps = set(fixed).intersection(kwargs)
        incompatible = {
            name: (kwargs[name], fixed[name])
            for name in overlaps
            if kwargs[name] != fixed[name]
        }
        if incompatible:
            raise ValueError(
                "VideoMAE-derived ActionDiT shape overrides are "
                f"incompatible: {incompatible}."
            )
        fixed.update(kwargs)
        model = cls(**fixed)
        reference = next(backbone.parameters())
        model = model.to(device=reference.device, dtype=reference.dtype)
        model.copy_blocks_from_hf_videomae(backbone)
        return model

    @classmethod
    def from_videomae_backbone(
        cls,
        backbone: nn.Module,
        action_dim: int,
        text_dim: int = 4096,
        max_action_horizon: int = 64,
        **kwargs: Any,
    ) -> "ActionDiTVideoMAE12":
        return cls.from_hf_videomae(
            backbone,
            action_dim=int(action_dim),
            text_dim=int(text_dim),
            max_action_horizon=int(max_action_horizon),
            **kwargs,
        )

    @torch.no_grad()
    def copy_blocks_from_hf_videomae(self, backbone: nn.Module) -> None:
        layers, _ = self._validate_hf_videomae(
            backbone,
            expected_hidden_size=self.hidden_size,
            expected_num_layers=self.num_layers,
            expected_num_heads=self.num_heads,
        )
        for action_block, source_layer in zip(self.blocks, layers):
            action_block.copy_from_hf_videomae_layer(source_layer)

    def run_dino_native_layers(
        self,
        backbone: nn.Module,
        pixel_values: torch.Tensor,
        *,
        num_views: Optional[int] = None,
        prefix_tokens: int = 0,
    ) -> DinoNativeOutput:
        """Run the single-frame VideoMAE student and expose per-layer K/V."""

        layers, config = self._validate_hf_videomae(
            backbone,
            expected_hidden_size=self.hidden_size,
            expected_num_layers=self.num_layers,
            expected_num_heads=self.num_heads,
        )
        if int(prefix_tokens) != 0:
            raise ValueError(
                "VideoMAE has no CLS/register prefix tokens, "
                f"got prefix_tokens={prefix_tokens}."
            )
        flat_pixels, batch_size, num_views = self._resolve_view_batch(
            pixel_values,
            num_views,
        )
        hidden_states = backbone.embeddings(
            flat_pixels,
            bool_masked_pos=None,
        )
        layer_patch_kv: list[DinoLayerPatchKV] = []
        for layer_index, layer in enumerate(layers):
            layer_input = hidden_states
            layer_outputs = layer(
                layer_input,
                head_mask=None,
                output_attentions=False,
            )
            if not isinstance(layer_outputs, tuple) or not layer_outputs:
                raise RuntimeError(
                    f"VideoMAE layer {layer_index} returned an invalid output."
                )
            hidden_states = layer_outputs[0]
            normalized = layer.layernorm_before(layer_input)
            self_attention = layer.attention.attention
            layer_patch_kv.append(
                DinoLayerPatchKV(
                    key=self._reshape_patch_kv(
                        self_attention.key(normalized),
                        batch_size=batch_size,
                        num_views=num_views,
                        num_heads=self.num_heads,
                        head_dim=self.head_dim,
                        prefix_tokens=0,
                    ),
                    value=self._reshape_patch_kv(
                        self_attention.value(normalized),
                        batch_size=batch_size,
                        num_views=num_views,
                        num_heads=self.num_heads,
                        head_dim=self.head_dim,
                        prefix_tokens=0,
                    ),
                )
            )

        final_hidden_states = backbone.layernorm(hidden_states)
        patches_per_view = int(final_hidden_states.shape[1])
        patch_tokens_by_view = final_hidden_states.reshape(
            batch_size,
            num_views,
            patches_per_view,
            int(config.hidden_size),
        )
        packed_patch_tokens = self._pack_patch_tokens(patch_tokens_by_view)
        return DinoNativeOutput(
            final_hidden_states=final_hidden_states,
            patch_tokens_by_view=patch_tokens_by_view,
            packed_patch_tokens=packed_patch_tokens,
            layer_patch_kv=tuple(layer_patch_kv),
            batch_size=batch_size,
            num_views=num_views,
            patches_per_view=patches_per_view,
        )


__all__ = [
    "ActionDiTVideoMAE12",
    "VideoMAEActionBlock",
]
