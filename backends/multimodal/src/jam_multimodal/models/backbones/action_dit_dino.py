from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class DinoLayerPatchKV:
    """One native DINO layer's patch-only K/V tensors.

    Both tensors use ``[batch, heads, views * patches_per_view, head_dim]``.
    The per-view CLS token is deliberately excluded: it remains in the native
    visual stream, but is not exposed to the action policy.
    """

    key: torch.Tensor
    value: torch.Tensor


@dataclass(frozen=True)
class DinoNativeOutput:
    """Outputs from an unmodified, two-view HuggingFace DINOv2 forward."""

    final_hidden_states: torch.Tensor
    patch_tokens_by_view: torch.Tensor
    packed_patch_tokens: torch.Tensor
    layer_patch_kv: tuple[DinoLayerPatchKV, ...]
    batch_size: int
    num_views: int
    patches_per_view: int


@dataclass(frozen=True)
class DinoActionLayerCondition:
    """Static text/vision K/V consumed by one action layer."""

    key: torch.Tensor
    value: torch.Tensor
    key_mask: torch.Tensor


@dataclass(frozen=True)
class DinoActionConditionCache:
    """Condition cache shared by all action denoising evaluations.

    ``layers`` holds text followed by DINO patch K/V. Action self K/V is
    appended dynamically because it changes at every denoising step.
    """

    layers: tuple[DinoActionLayerCondition, ...]
    visual: DinoNativeOutput
    text_tokens: torch.Tensor
    text_attention_mask: torch.Tensor


class DinoLayerScale(nn.Module):
    """DINOv2 LayerScale with the same parameter layout as HuggingFace."""

    def __init__(self, hidden_size: int, initial_value: float = 1.0) -> None:
        super().__init__()
        self.lambda1 = nn.Parameter(
            torch.full((int(hidden_size),), float(initial_value))
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return hidden_states * self.lambda1


class DinoActionMLP(nn.Module):
    """The GELU MLP used by DINOv2-B."""

    def __init__(self, hidden_size: int, mlp_ratio: float) -> None:
        super().__init__()
        ffn_dim = int(int(hidden_size) * float(mlp_ratio))
        if ffn_dim <= 0:
            raise ValueError(f"Invalid DINO action FFN dimension: {ffn_dim}.")
        self.fc1 = nn.Linear(int(hidden_size), ffn_dim)
        self.activation = nn.GELU(approximate="none")
        self.fc2 = nn.Linear(ffn_dim, int(hidden_size))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.activation(self.fc1(hidden_states)))


class DinoActionBlock(nn.Module):
    """DINO-shaped action block with zero-initialized AdaLN deltas.

    The six modulation chunks are attention shift/scale/gate-delta followed by
    MLP shift/scale/gate-delta. The actual residual gates are ``1 + delta``.
    Consequently a zero-initialized modulation head leaves copied DINO block
    weights active instead of turning the block into an identity map.
    """

    def __init__(
        self,
        *,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float,
        layer_norm_eps: float,
        layerscale_value: float,
        qkv_bias: bool,
    ) -> None:
        super().__init__()
        hidden_size = int(hidden_size)
        num_heads = int(num_heads)
        if hidden_size <= 0 or num_heads <= 0 or hidden_size % num_heads != 0:
            raise ValueError(
                "Action DINO hidden size must be a positive multiple of heads, "
                f"got hidden={hidden_size}, heads={num_heads}."
            )
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        self.norm1 = nn.LayerNorm(hidden_size, eps=float(layer_norm_eps))
        self.query = nn.Linear(hidden_size, hidden_size, bias=bool(qkv_bias))
        self.key = nn.Linear(hidden_size, hidden_size, bias=bool(qkv_bias))
        self.value = nn.Linear(hidden_size, hidden_size, bias=bool(qkv_bias))
        self.out = nn.Linear(hidden_size, hidden_size)
        self.layer_scale1 = DinoLayerScale(hidden_size, layerscale_value)

        self.norm2 = nn.LayerNorm(hidden_size, eps=float(layer_norm_eps))
        self.mlp = DinoActionMLP(hidden_size, mlp_ratio)
        self.layer_scale2 = DinoLayerScale(hidden_size, layerscale_value)

        self.modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size),
        )
        nn.init.zeros_(self.modulation[-1].weight)
        nn.init.zeros_(self.modulation[-1].bias)

    def _split_heads(self, tensor: torch.Tensor) -> torch.Tensor:
        if tensor.ndim != 3 or int(tensor.shape[-1]) != self.hidden_size:
            raise ValueError(
                "Expected a [B,N,hidden] projection before head splitting, "
                f"got {tuple(tensor.shape)} with hidden={self.hidden_size}."
            )
        batch_size, seq_len = int(tensor.shape[0]), int(tensor.shape[1])
        return (
            tensor.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
            .contiguous()
        )

    def action_qkv(
        self,
        hidden_states: torch.Tensor,
        timestep_embedding: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        tuple[torch.Tensor, ...],
    ]:
        if hidden_states.ndim != 3:
            raise ValueError(
                f"Action hidden states must be [B,A,D], got {tuple(hidden_states.shape)}."
            )
        batch_size = int(hidden_states.shape[0])
        if tuple(timestep_embedding.shape) != (batch_size, self.hidden_size):
            raise ValueError(
                "Timestep embedding must be [B,hidden], "
                f"got {tuple(timestep_embedding.shape)} for B={batch_size}, "
                f"hidden={self.hidden_size}."
            )
        (
            shift_attn,
            scale_attn,
            gate_attn_delta,
            shift_mlp,
            scale_mlp,
            gate_mlp_delta,
        ) = self.modulation(timestep_embedding).chunk(6, dim=-1)
        normalized = self.norm1(hidden_states)
        attention_input = (
            normalized * (1.0 + scale_attn[:, None, :])
            + shift_attn[:, None, :]
        )
        return (
            self._split_heads(self.query(attention_input)),
            self._split_heads(self.key(attention_input)),
            self._split_heads(self.value(attention_input)),
            (
                gate_attn_delta,
                shift_mlp,
                scale_mlp,
                gate_mlp_delta,
            ),
        )

    def text_kv(
        self,
        text_tokens: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if text_tokens.ndim != 3 or int(text_tokens.shape[-1]) != self.hidden_size:
            raise ValueError(
                "Text tokens must be [B,L,hidden], "
                f"got {tuple(text_tokens.shape)} with hidden={self.hidden_size}."
            )
        normalized = self.norm1(text_tokens)
        return (
            self._split_heads(self.key(normalized)),
            self._split_heads(self.value(normalized)),
        )

    def apply_attention_and_mlp(
        self,
        hidden_states: torch.Tensor,
        mixed_attention: torch.Tensor,
        modulation_state: tuple[torch.Tensor, ...],
    ) -> torch.Tensor:
        if hidden_states.ndim != 3:
            raise ValueError(
                f"Action hidden states must be [B,A,D], got {tuple(hidden_states.shape)}."
            )
        if tuple(mixed_attention.shape) != tuple(hidden_states.shape):
            raise ValueError(
                "Mixed-attention output must match action hidden states, "
                f"got {tuple(mixed_attention.shape)} versus {tuple(hidden_states.shape)}."
            )
        if len(modulation_state) != 4:
            raise ValueError(
                f"Expected four post-attention modulation tensors, got {len(modulation_state)}."
            )
        gate_attn_delta, shift_mlp, scale_mlp, gate_mlp_delta = modulation_state
        hidden_states = hidden_states + (
            1.0 + gate_attn_delta[:, None, :]
        ) * self.layer_scale1(self.out(mixed_attention))

        normalized = self.norm2(hidden_states)
        mlp_input = (
            normalized * (1.0 + scale_mlp[:, None, :])
            + shift_mlp[:, None, :]
        )
        hidden_states = hidden_states + (
            1.0 + gate_mlp_delta[:, None, :]
        ) * self.layer_scale2(self.mlp(mlp_input))
        return hidden_states

    @torch.no_grad()
    def copy_from_hf_dinov2_layer(self, source: nn.Module) -> None:
        """Copy one HuggingFace ``Dinov2Layer`` without resizing."""

        required_paths = (
            "norm1",
            "attention",
            "layer_scale1",
            "norm2",
            "mlp",
            "layer_scale2",
        )
        missing = [name for name in required_paths if not hasattr(source, name)]
        if missing:
            raise TypeError(
                "Source is not a compatible HuggingFace Dinov2Layer; "
                f"missing {missing}."
            )
        attention = source.attention
        if not hasattr(attention, "attention") or not hasattr(attention, "output"):
            raise TypeError("Source DINO attention lacks `attention`/`output` modules.")
        self_attention = attention.attention
        source_mlp = source.mlp
        if not hasattr(source_mlp, "fc1") or not hasattr(source_mlp, "fc2"):
            raise TypeError(
                "ActionDiTDino12 currently supports the standard DINOv2 GELU "
                "MLP, not a SwiGLU source block."
            )

        self.norm1.load_state_dict(source.norm1.state_dict(), strict=True)
        self.query.load_state_dict(self_attention.query.state_dict(), strict=True)
        self.key.load_state_dict(self_attention.key.state_dict(), strict=True)
        self.value.load_state_dict(self_attention.value.state_dict(), strict=True)
        self.out.load_state_dict(attention.output.dense.state_dict(), strict=True)
        self.layer_scale1.load_state_dict(
            source.layer_scale1.state_dict(), strict=True
        )
        self.norm2.load_state_dict(source.norm2.state_dict(), strict=True)
        self.mlp.fc1.load_state_dict(source_mlp.fc1.state_dict(), strict=True)
        self.mlp.fc2.load_state_dict(source_mlp.fc2.state_dict(), strict=True)
        self.layer_scale2.load_state_dict(
            source.layer_scale2.state_dict(), strict=True
        )


class ActionDiTDino12(nn.Module):
    """Native DINOv2-B visual layers paired with a 12-layer action expert.

    This module intentionally does not implement the Yak block protocol. Native
    DINO layers remain independent for each camera and never consume text or
    action tokens. At every depth, the corresponding action block attends one
    shared softmax over static text, both cameras' DINO patch K/V, and action
    self K/V. That direction matches JAMPolicy's causal policy mask while
    preserving the visual branch's original HuggingFace forward exactly.
    """

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
        layer_norm_eps: float = 1.0e-6,
        layerscale_value: float = 1.0,
        qkv_bias: bool = True,
        max_action_horizon: int = 64,
        timestep_embed_dim: int = 256,
        time_factor: float = 1000.0,
        initializer_range: float = 0.02,
        strict_shape: bool = True,
    ) -> None:
        super().__init__()
        action_dim = int(action_dim)
        text_dim = int(text_dim)
        hidden_size = int(hidden_size)
        num_heads = int(num_heads)
        num_layers = int(num_layers)
        max_action_horizon = int(max_action_horizon)
        timestep_embed_dim = int(timestep_embed_dim)
        time_factor = float(time_factor)

        strict_shape = bool(strict_shape)
        if strict_shape and hidden_size != self.EXPECTED_HIDDEN_SIZE:
            raise ValueError(
                "ActionDiTDino12 is intentionally fixed to DINOv2-B hidden=768, "
                f"got {hidden_size}."
            )
        if strict_shape and num_heads != self.EXPECTED_NUM_HEADS:
            raise ValueError(
                "ActionDiTDino12 is intentionally fixed to 12 heads, "
                f"got {num_heads}."
            )
        if strict_shape and num_layers != self.EXPECTED_NUM_LAYERS:
            raise ValueError(
                "ActionDiTDino12 is intentionally fixed to 12 layers, "
                f"got {num_layers}."
            )
        if hidden_size <= 0 or num_heads <= 0 or hidden_size % num_heads != 0:
            raise ValueError(
                "hidden_size must be a positive multiple of num_heads, "
                f"got {hidden_size}/{num_heads}."
            )
        if num_layers <= 0:
            raise ValueError(f"num_layers must be positive, got {num_layers}.")
        if action_dim <= 0 or text_dim <= 0 or max_action_horizon <= 0:
            raise ValueError(
                "action_dim/text_dim/max_action_horizon must be positive, "
                f"got {action_dim}/{text_dim}/{max_action_horizon}."
            )
        if timestep_embed_dim <= 0:
            raise ValueError(
                f"timestep_embed_dim must be positive, got {timestep_embed_dim}."
            )
        if not math.isfinite(time_factor) or time_factor <= 0.0:
            raise ValueError(
                f"time_factor must be finite and positive, got {time_factor}."
            )

        self.action_dim = action_dim
        self.text_dim = text_dim
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.num_layers = num_layers
        self.max_action_horizon = max_action_horizon
        self.timestep_embed_dim = timestep_embed_dim
        self.time_factor = time_factor
        self.initializer_range = float(initializer_range)
        self.strict_shape = strict_shape

        self.action_encoder = nn.Linear(action_dim, hidden_size)
        self.action_position = nn.Parameter(
            torch.empty(1, max_action_horizon, hidden_size)
        )
        self.time_in = nn.Sequential(
            nn.Linear(timestep_embed_dim, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.text_norm = nn.LayerNorm(text_dim, eps=float(layer_norm_eps))
        self.text_in = nn.Linear(text_dim, hidden_size)

        self.blocks = nn.ModuleList(
            [
                DinoActionBlock(
                    hidden_size=hidden_size,
                    num_heads=num_heads,
                    mlp_ratio=float(mlp_ratio),
                    layer_norm_eps=float(layer_norm_eps),
                    layerscale_value=float(layerscale_value),
                    qkv_bias=bool(qkv_bias),
                )
                for _ in range(num_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(hidden_size, eps=float(layer_norm_eps))
        self.head = nn.Linear(hidden_size, action_dim)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """DINO-style initialization for parameters without pretrained values."""

        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(
                    module.weight,
                    mean=0.0,
                    std=self.initializer_range,
                )
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
        nn.init.trunc_normal_(
            self.action_position,
            mean=0.0,
            std=self.initializer_range,
        )
        # Preserve zero AdaLN deltas after the generic Linear initialization.
        for block in self.blocks:
            nn.init.zeros_(block.modulation[-1].weight)
            nn.init.zeros_(block.modulation[-1].bias)

    @staticmethod
    def timestep_embedding(
        timestep: torch.Tensor,
        dim: int,
        max_period: float = 10_000.0,
        time_factor: float = 1000.0,
    ) -> torch.Tensor:
        """Create a standard sinusoidal timestep embedding in fp32."""

        if timestep.ndim != 1:
            raise ValueError(
                f"Timestep must be a one-dimensional [B] tensor, got {tuple(timestep.shape)}."
            )
        dim = int(dim)
        half = dim // 2
        if half <= 0:
            raise ValueError(f"Timestep embedding dim must be >=2, got {dim}.")
        frequencies = torch.exp(
            -math.log(float(max_period))
            * torch.arange(half, device=timestep.device, dtype=torch.float32)
            / float(half)
        )
        time_factor = float(time_factor)
        if not math.isfinite(time_factor) or time_factor <= 0.0:
            raise ValueError(
                f"time_factor must be finite and positive, got {time_factor}."
            )
        angles = (
            timestep.float()[:, None]
            * time_factor
            * frequencies[None]
        )
        embedding = torch.cat([torch.cos(angles), torch.sin(angles)], dim=-1)
        if dim % 2:
            embedding = F.pad(embedding, (0, 1))
        return embedding

    def _parameter_device_dtype(self) -> tuple[torch.device, torch.dtype]:
        parameter = self.action_encoder.weight
        return parameter.device, parameter.dtype

    def preprocess_action(
        self,
        action_tokens: torch.Tensor,
        timestep: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Embed noisy actions and their diffusion timestep."""

        if action_tokens.ndim != 3:
            raise ValueError(
                f"Action tokens must be [B,A,{self.action_dim}], got {tuple(action_tokens.shape)}."
            )
        batch_size, action_len, action_dim = map(int, action_tokens.shape)
        if action_dim != self.action_dim:
            raise ValueError(
                f"Expected action_dim={self.action_dim}, got {action_dim}."
            )
        if action_len <= 0 or action_len > self.max_action_horizon:
            raise ValueError(
                f"Action length must be in [1,{self.max_action_horizon}], got {action_len}."
            )
        if timestep.ndim == 0:
            timestep = timestep[None]
        if timestep.ndim != 1:
            raise ValueError(
                f"Timestep must be scalar or [B], got {tuple(timestep.shape)}."
            )
        if int(timestep.shape[0]) == 1 and batch_size > 1:
            timestep = timestep.expand(batch_size)
        if int(timestep.shape[0]) != batch_size:
            raise ValueError(
                f"Timestep batch {timestep.shape[0]} does not match action batch {batch_size}."
            )

        device, dtype = self._parameter_device_dtype()
        action_tokens = action_tokens.to(device=device, dtype=dtype)
        hidden_states = self.action_encoder(action_tokens)
        hidden_states = (
            hidden_states
            + self.action_position[:, :action_len].to(
                device=hidden_states.device,
                dtype=hidden_states.dtype,
            )
        )
        time_features = self.timestep_embedding(
            timestep.to(device=device),
            self.timestep_embed_dim,
            time_factor=self.time_factor,
        )
        timestep_hidden = self.time_in(time_features.to(dtype=dtype))
        return {
            "hidden_states": hidden_states,
            "timestep_embedding": timestep_hidden,
        }

    def pre_dit(
        self,
        action_tokens: torch.Tensor,
        timestep: torch.Tensor,
    ) -> dict[str, Any]:
        """JAMPolicy-style wrapper around :meth:`preprocess_action`."""

        state = self.preprocess_action(action_tokens, timestep)
        return {
            "tokens": state["hidden_states"],
            "t_mod": state["timestep_embedding"],
            "hidden_states": state["hidden_states"],
            "timestep_embedding": state["timestep_embedding"],
            "meta": {
                "batch_size": int(action_tokens.shape[0]),
                "seq_len": int(action_tokens.shape[1]),
                "time_factor": self.time_factor,
            },
        }

    def encode_text(
        self,
        text_context: torch.Tensor,
        text_attention_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Map frozen 4096-D condition tokens to static 768-D tokens."""

        if text_context.ndim != 3:
            raise ValueError(
                f"Text context must be [B,L,{self.text_dim}], got {tuple(text_context.shape)}."
            )
        batch_size, text_len, text_dim = map(int, text_context.shape)
        if text_dim != self.text_dim:
            raise ValueError(
                f"Expected text_dim={self.text_dim}, got {text_dim}."
            )
        if text_len <= 0:
            raise ValueError("At least one text/context token is required.")
        if text_attention_mask is None:
            text_attention_mask = torch.ones(
                batch_size,
                text_len,
                device=text_context.device,
                dtype=torch.bool,
            )
        if tuple(text_attention_mask.shape) != (batch_size, text_len):
            raise ValueError(
                "Text mask must be [B,L] matching text context, "
                f"got {tuple(text_attention_mask.shape)} versus {(batch_size, text_len)}."
            )
        device, dtype = self._parameter_device_dtype()
        text_context = text_context.to(device=device, dtype=dtype)
        text_tokens = self.text_in(self.text_norm(text_context))
        text_attention_mask = text_attention_mask.to(
            device=device,
            dtype=torch.bool,
        )
        if not bool(text_attention_mask.any(dim=1).all().item()):
            raise ValueError("Every batch item must contain at least one valid text token.")
        return text_tokens, text_attention_mask

    @staticmethod
    def _validate_hf_dinov2(
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
                "Expected a HuggingFace Dinov2Model exposing config and encoder.layer."
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
        for name, value in expected.items():
            if value is not None and actual[name] != int(value):
                raise ValueError(
                    f"DINO/action {name} mismatch: expected {value}, got {actual[name]}."
                )
        if any(value <= 0 for value in actual.values()):
            raise ValueError(f"Invalid DINO config dimensions: {actual}.")
        if actual["hidden_size"] % actual["num_attention_heads"] != 0:
            raise ValueError(
                "DINO hidden size must be divisible by its head count, "
                f"got {actual}."
            )
        if len(layers) != actual["num_hidden_layers"]:
            raise ValueError(
                "DINO config/encoder depth mismatch: "
                f"{actual['num_hidden_layers']} versus {len(layers)}."
            )
        if not hasattr(backbone, "embeddings") or not hasattr(backbone, "layernorm"):
            raise TypeError("DINO backbone must expose embeddings and final layernorm.")
        return layers, config

    @classmethod
    def from_hf_dinov2(
        cls,
        backbone: nn.Module,
        **kwargs: Any,
    ) -> "ActionDiTDino12":
        """Construct the action expert and copy all 12 native DINO blocks."""

        strict_shape = bool(kwargs.pop("strict_shape", True))
        layers, config = cls._validate_hf_dinov2(
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
        fixed = {
            "hidden_size": int(config.hidden_size),
            "num_heads": int(config.num_attention_heads),
            "num_layers": int(config.num_hidden_layers),
            "mlp_ratio": float(config.mlp_ratio),
            "layer_norm_eps": float(config.layer_norm_eps),
            "layerscale_value": float(config.layerscale_value),
            "qkv_bias": bool(config.qkv_bias),
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
                f"DINO-derived ActionDiT shape overrides are incompatible: {incompatible}."
            )
        fixed.update(kwargs)
        model = cls(**fixed)
        reference = next(backbone.parameters())
        model = model.to(device=reference.device, dtype=reference.dtype)
        model.copy_blocks_from_hf_dinov2(backbone)
        return model

    @classmethod
    def from_dino_backbone(
        cls,
        backbone: nn.Module,
        action_dim: int,
        text_dim: int = 4096,
        max_action_horizon: int = 64,
        **kwargs: Any,
    ) -> "ActionDiTDino12":
        """Compatibility constructor used by JAMPolicy runtime integration."""

        return cls.from_hf_dinov2(
            backbone,
            action_dim=int(action_dim),
            text_dim=int(text_dim),
            max_action_horizon=int(max_action_horizon),
            **kwargs,
        )

    @torch.no_grad()
    def copy_blocks_from_hf_dinov2(self, backbone: nn.Module) -> None:
        """Copy DINO norm/QKV/out/LayerScale/MLP into action blocks."""

        layers, _ = self._validate_hf_dinov2(
            backbone,
            expected_hidden_size=self.hidden_size,
            expected_num_layers=self.num_layers,
            expected_num_heads=self.num_heads,
        )
        if len(layers) != len(self.blocks):
            raise ValueError(
                f"DINO/action depth mismatch: {len(layers)} versus {len(self.blocks)}."
            )
        for action_block, dino_layer in zip(self.blocks, layers):
            action_block.copy_from_hf_dinov2_layer(dino_layer)

    @staticmethod
    def _resolve_view_batch(
        pixel_values: torch.Tensor,
        num_views: Optional[int],
    ) -> tuple[torch.Tensor, int, int]:
        if pixel_values.ndim == 5:
            batch_size, inferred_views, channels, height, width = map(
                int, pixel_values.shape
            )
            if num_views is not None and int(num_views) != inferred_views:
                raise ValueError(
                    f"num_views={num_views} conflicts with pixel tensor views={inferred_views}."
                )
            num_views = inferred_views
            flat = pixel_values.reshape(
                batch_size * num_views,
                channels,
                height,
                width,
            )
        elif pixel_values.ndim == 4:
            if num_views is None:
                raise ValueError(
                    "Flattened [B*V,C,H,W] pixels require an explicit num_views."
                )
            num_views = int(num_views)
            if num_views <= 0 or int(pixel_values.shape[0]) % num_views != 0:
                raise ValueError(
                    f"Pixel batch {pixel_values.shape[0]} is not divisible by num_views={num_views}."
                )
            batch_size = int(pixel_values.shape[0]) // num_views
            flat = pixel_values
        else:
            raise ValueError(
                "Pixel values must be [B,V,C,H,W] or [B*V,C,H,W], "
                f"got {tuple(pixel_values.shape)}."
            )
        if batch_size <= 0 or int(num_views) <= 0:
            raise ValueError(
                f"Resolved invalid batch/views={batch_size}/{num_views}."
            )
        if int(flat.shape[1]) != 3:
            raise ValueError(
                f"DINO pixel inputs require three channels, got {flat.shape[1]}."
            )
        return flat, batch_size, int(num_views)

    @staticmethod
    def _reshape_patch_kv(
        projected: torch.Tensor,
        *,
        batch_size: int,
        num_views: int,
        num_heads: int,
        head_dim: int,
        prefix_tokens: int,
    ) -> torch.Tensor:
        if projected.ndim != 3:
            raise ValueError(
                f"DINO K/V projection must be [B*V,S,D], got {tuple(projected.shape)}."
            )
        flat_batch, seq_len, hidden_size = map(int, projected.shape)
        expected_batch = int(batch_size) * int(num_views)
        if flat_batch != expected_batch:
            raise ValueError(
                f"DINO K/V batch mismatch: expected {expected_batch}, got {flat_batch}."
            )
        if hidden_size != int(num_heads) * int(head_dim):
            raise ValueError(
                "DINO K/V hidden mismatch: "
                f"{hidden_size} != {num_heads}*{head_dim}."
            )
        if seq_len <= int(prefix_tokens):
            raise ValueError(
                f"DINO sequence length {seq_len} has no patch tokens after {prefix_tokens} prefixes."
            )
        patches = (
            projected.reshape(
                batch_size,
                num_views,
                seq_len,
                num_heads,
                head_dim,
            )[:, :, prefix_tokens:]
            .reshape(
                batch_size,
                num_views * (seq_len - prefix_tokens),
                num_heads,
                head_dim,
            )
            .permute(0, 2, 1, 3)
            .contiguous()
        )
        return patches

    @staticmethod
    def _pack_patch_tokens(
        patch_tokens_by_view: torch.Tensor,
    ) -> torch.Tensor:
        if patch_tokens_by_view.ndim != 4:
            raise ValueError(
                "Patch tokens by view must be [B,V,P,D], "
                f"got {tuple(patch_tokens_by_view.shape)}."
            )
        batch_size, num_views, patches_per_view, hidden_size = map(
            int, patch_tokens_by_view.shape
        )
        grid = math.isqrt(patches_per_view)
        if grid * grid != patches_per_view:
            raise ValueError(
                f"DINO patch count {patches_per_view} is not a square grid."
            )
        grids = patch_tokens_by_view.reshape(
            batch_size,
            num_views,
            grid,
            grid,
            hidden_size,
        )
        packed = torch.cat([grids[:, view] for view in range(num_views)], dim=2)
        return packed.reshape(
            batch_size,
            grid * grid * num_views,
            hidden_size,
        )

    def run_dino_native_layers(
        self,
        backbone: nn.Module,
        pixel_values: torch.Tensor,
        *,
        num_views: Optional[int] = None,
        prefix_tokens: int = 1,
    ) -> DinoNativeOutput:
        """Run native HF layers and separately expose each layer's patch K/V.

        Visual updates call ``Dinov2Layer.forward`` unchanged. K/V is recomputed
        from that layer's original input solely for the action path. This costs
        two extra linear projections per DINO layer, but guarantees that action
        integration cannot perturb the native visual output.
        """

        layers, config = self._validate_hf_dinov2(
            backbone,
            expected_hidden_size=self.hidden_size,
            expected_num_layers=self.num_layers,
            expected_num_heads=self.num_heads,
        )
        if int(prefix_tokens) != 1:
            raise ValueError(
                "DINOv2-B uses exactly one CLS prefix token; "
                f"got prefix_tokens={prefix_tokens}."
            )
        flat_pixels, batch_size, num_views = self._resolve_view_batch(
            pixel_values,
            num_views,
        )
        hidden_states = backbone.embeddings(flat_pixels)
        if hidden_states.ndim != 3:
            raise RuntimeError(
                f"DINO embeddings must be [B*V,S,D], got {tuple(hidden_states.shape)}."
            )
        if int(hidden_states.shape[-1]) != self.hidden_size:
            raise RuntimeError(
                f"DINO embedding hidden size must be {self.hidden_size}, got {hidden_states.shape[-1]}."
            )

        layer_patch_kv: list[DinoLayerPatchKV] = []
        for layer_idx, layer in enumerate(layers):
            layer_input = hidden_states
            # This is the exact HuggingFace visual branch.
            layer_outputs = layer(
                layer_input,
                head_mask=None,
                output_attentions=False,
            )
            if not isinstance(layer_outputs, tuple) or not layer_outputs:
                raise RuntimeError(
                    f"DINO layer {layer_idx} returned an invalid output type."
                )
            hidden_states = layer_outputs[0]

            # Recompute only K/V from the same pre-attention normalized input.
            normalized = layer.norm1(layer_input)
            self_attention = layer.attention.attention
            key = self_attention.key(normalized)
            value = self_attention.value(normalized)
            layer_patch_kv.append(
                DinoLayerPatchKV(
                    key=self._reshape_patch_kv(
                        key,
                        batch_size=batch_size,
                        num_views=num_views,
                        num_heads=self.num_heads,
                        head_dim=self.head_dim,
                        prefix_tokens=prefix_tokens,
                    ),
                    value=self._reshape_patch_kv(
                        value,
                        batch_size=batch_size,
                        num_views=num_views,
                        num_heads=self.num_heads,
                        head_dim=self.head_dim,
                        prefix_tokens=prefix_tokens,
                    ),
                )
            )

        if len(layer_patch_kv) != self.num_layers:
            raise RuntimeError(
                f"Expected {self.num_layers} DINO K/V layers, got {len(layer_patch_kv)}."
            )
        final_hidden_states = backbone.layernorm(hidden_states)
        seq_len = int(final_hidden_states.shape[1])
        if seq_len <= int(prefix_tokens):
            raise RuntimeError(
                f"DINO final sequence has no patches: shape={tuple(final_hidden_states.shape)}."
            )
        patches_per_view = seq_len - int(prefix_tokens)
        patch_tokens_by_view = final_hidden_states.reshape(
            batch_size,
            num_views,
            seq_len,
            int(config.hidden_size),
        )[:, :, prefix_tokens:]
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

    def build_condition_cache(
        self,
        backbone: nn.Module,
        pixel_values: torch.Tensor,
        text_context: torch.Tensor,
        text_attention_mask: Optional[torch.Tensor] = None,
        *,
        num_views: Optional[int] = None,
    ) -> DinoActionConditionCache:
        """Run native DINO and prepare layer-specific static text/visual K/V."""

        visual = self.run_dino_native_layers(
            backbone,
            pixel_values,
            num_views=num_views,
        )
        text_tokens, text_attention_mask = self.encode_text(
            text_context,
            text_attention_mask,
        )
        if int(text_tokens.shape[0]) != visual.batch_size:
            raise ValueError(
                "Text/visual batch mismatch: "
                f"{text_tokens.shape[0]} versus {visual.batch_size}."
            )

        visual_key_mask = torch.ones(
            visual.batch_size,
            visual.num_views * visual.patches_per_view,
            device=text_attention_mask.device,
            dtype=torch.bool,
        )
        condition_mask = torch.cat(
            [text_attention_mask, visual_key_mask],
            dim=1,
        ).contiguous()
        layers: list[DinoActionLayerCondition] = []
        for layer_idx, (block, visual_kv) in enumerate(
            zip(self.blocks, visual.layer_patch_kv)
        ):
            text_key, text_value = block.text_kv(text_tokens)
            if (
                text_key.shape[:2] != visual_kv.key.shape[:2]
                or int(text_key.shape[-1]) != int(visual_kv.key.shape[-1])
            ):
                raise RuntimeError(
                    "Text/visual K/V head mismatch at layer "
                    f"{layer_idx}: text={tuple(text_key.shape)}, "
                    f"visual={tuple(visual_kv.key.shape)}."
                )
            layers.append(
                DinoActionLayerCondition(
                    key=torch.cat([text_key, visual_kv.key], dim=2),
                    value=torch.cat([text_value, visual_kv.value], dim=2),
                    key_mask=condition_mask,
                )
            )
        if len(layers) != self.num_layers:
            raise RuntimeError(
                f"Expected {self.num_layers} condition layers, got {len(layers)}."
            )
        return DinoActionConditionCache(
            layers=tuple(layers),
            visual=visual,
            text_tokens=text_tokens,
            text_attention_mask=text_attention_mask,
        )

    def prefill_dino_prefix(
        self,
        ssl_student: nn.Module,
        images: torch.Tensor,
        text_hidden_states: torch.Tensor,
        text_attention_mask: Optional[torch.Tensor] = None,
    ) -> dict[str, Any]:
        """JAMPolicy-style condition prefill from a dense SSL wrapper.

        ``ssl_student`` may be the repository's ``DenseImageSSLBackbone`` or a
        bare HuggingFace ``Dinov2Model``. Packed JAMPolicy camera tensors are
        accepted only through the dense wrapper, whose existing preprocessing
        splits and normalizes both views before this module runs native layers.
        """

        backbone = getattr(ssl_student, "backbone", ssl_student)
        prepare_views = getattr(ssl_student, "_prepare_views", None)
        configured_views = getattr(ssl_student, "num_views", None)
        if callable(prepare_views):
            pixel_values = prepare_views(images)
            if configured_views is None:
                raise ValueError(
                    "Dense SSL wrapper exposes `_prepare_views` but no `num_views`."
                )
            num_views = int(configured_views)
        else:
            pixel_values = images
            if images.ndim == 5:
                num_views = int(images.shape[1])
            elif configured_views is not None:
                num_views = int(configured_views)
            else:
                raise ValueError(
                    "A bare DINO backbone requires [B,V,C,H,W] images or an "
                    "object exposing `num_views`."
                )

        condition_cache = self.build_condition_cache(
            backbone,
            pixel_values,
            text_hidden_states,
            text_attention_mask,
            num_views=num_views,
        )
        return {
            "condition_cache": condition_cache,
            "final_dense": condition_cache.visual.packed_patch_tokens,
            "visual_final_hidden_states": (
                condition_cache.visual.final_hidden_states
            ),
            "layer_patch_kv": condition_cache.visual.layer_patch_kv,
            "layers": condition_cache.layers,
            "text_tokens": condition_cache.text_tokens,
            "text_attention_mask": condition_cache.text_attention_mask,
            "batch_size": condition_cache.visual.batch_size,
            "num_views": condition_cache.visual.num_views,
            "patches_per_view": condition_cache.visual.patches_per_view,
        }

    def forward_action_layer(
        self,
        hidden_states: torch.Tensor,
        timestep_embedding: torch.Tensor,
        *,
        layer_index: int,
        condition: DinoActionLayerCondition,
    ) -> torch.Tensor:
        """Run one action layer against static text and native DINO K/V."""

        layer_index = int(layer_index)
        if layer_index < 0 or layer_index >= self.num_layers:
            raise IndexError(
                f"Action layer index must be in [0,{self.num_layers - 1}], got {layer_index}."
            )
        if hidden_states.ndim != 3 or int(hidden_states.shape[-1]) != self.hidden_size:
            raise ValueError(
                "Action hidden states must be [B,A,768], "
                f"got {tuple(hidden_states.shape)}."
            )
        batch_size, action_len = int(hidden_states.shape[0]), int(hidden_states.shape[1])
        if condition.key.ndim != 4 or condition.value.ndim != 4:
            raise ValueError("Condition K/V must be [B,H,K,head_dim].")
        if tuple(condition.key.shape) != tuple(condition.value.shape):
            raise ValueError(
                "Condition K/V shapes differ: "
                f"{tuple(condition.key.shape)} versus {tuple(condition.value.shape)}."
            )
        expected_prefix = (
            batch_size,
            self.num_heads,
        )
        if tuple(condition.key.shape[:2]) != expected_prefix:
            raise ValueError(
                "Condition K/V batch/head mismatch: "
                f"expected {expected_prefix}, got {tuple(condition.key.shape[:2])}."
            )
        if int(condition.key.shape[-1]) != self.head_dim:
            raise ValueError(
                f"Condition head_dim must be {self.head_dim}, got {condition.key.shape[-1]}."
            )
        condition_len = int(condition.key.shape[2])
        if tuple(condition.key_mask.shape) != (batch_size, condition_len):
            raise ValueError(
                "Condition key mask must be [B,K], "
                f"got {tuple(condition.key_mask.shape)} versus {(batch_size, condition_len)}."
            )

        block = self.blocks[layer_index]
        action_q, action_k, action_v, modulation_state = block.action_qkv(
            hidden_states,
            timestep_embedding,
        )
        key = torch.cat([condition.key, action_k], dim=2)
        value = torch.cat([condition.value, action_v], dim=2)
        action_valid = torch.ones(
            batch_size,
            action_len,
            device=condition.key_mask.device,
            dtype=torch.bool,
        )
        key_mask = torch.cat(
            [condition.key_mask, action_valid],
            dim=1,
        )
        attention_mask = (
            key_mask[:, None, None, :]
            .expand(batch_size, 1, action_len, condition_len + action_len)
            .contiguous()
        )
        mixed = F.scaled_dot_product_attention(
            action_q,
            key,
            value,
            attn_mask=attention_mask,
            dropout_p=0.0,
            is_causal=False,
            scale=self.head_dim**-0.5,
        )
        mixed = (
            mixed.transpose(1, 2)
            .contiguous()
            .reshape(batch_size, action_len, self.hidden_size)
        )
        return block.apply_attention_and_mlp(
            hidden_states,
            mixed,
            modulation_state,
        )

    def postprocess_action(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if hidden_states.ndim != 3 or int(hidden_states.shape[-1]) != self.hidden_size:
            raise ValueError(
                "Action hidden states must be [B,A,768], "
                f"got {tuple(hidden_states.shape)}."
            )
        return self.head(self.final_norm(hidden_states))

    def forward_with_prefix_cache(
        self,
        action_pre: dict[str, Any],
        prefix_cache: dict[str, Any] | DinoActionConditionCache,
    ) -> torch.Tensor:
        """Run action blocks from JAMPolicy-style pre-DiT/cache payloads."""

        if not isinstance(action_pre, dict):
            raise TypeError(
                f"`action_pre` must be a dict, got {type(action_pre)}."
            )
        hidden_states = action_pre.get(
            "tokens",
            action_pre.get("hidden_states"),
        )
        timestep_embedding = action_pre.get(
            "t_mod",
            action_pre.get("timestep_embedding"),
        )
        if not torch.is_tensor(hidden_states) or not torch.is_tensor(
            timestep_embedding
        ):
            raise ValueError(
                "`action_pre` must contain tensor `tokens`/`hidden_states` and "
                "`t_mod`/`timestep_embedding`."
            )
        if isinstance(prefix_cache, DinoActionConditionCache):
            condition_cache = prefix_cache
        elif isinstance(prefix_cache, dict):
            condition_cache = prefix_cache.get("condition_cache")
            if not isinstance(condition_cache, DinoActionConditionCache):
                raise ValueError(
                    "Prefix-cache dict must contain a "
                    "`DinoActionConditionCache` under `condition_cache`."
                )
        else:
            raise TypeError(
                "`prefix_cache` must be a dict or DinoActionConditionCache, "
                f"got {type(prefix_cache)}."
            )
        if len(condition_cache.layers) != self.num_layers:
            raise ValueError(
                f"Prefix cache has {len(condition_cache.layers)} layers, "
                f"expected {self.num_layers}."
            )
        if int(hidden_states.shape[0]) != condition_cache.visual.batch_size:
            raise ValueError(
                "Action/prefix batch mismatch: "
                f"{hidden_states.shape[0]} versus "
                f"{condition_cache.visual.batch_size}."
            )
        for layer_index, condition in enumerate(condition_cache.layers):
            hidden_states = self.forward_action_layer(
                hidden_states,
                timestep_embedding,
                layer_index=layer_index,
                condition=condition,
            )
        return hidden_states

    def post_dit(
        self,
        tokens: torch.Tensor,
        action_pre: Optional[dict[str, Any]] = None,
    ) -> torch.Tensor:
        """JAMPolicy-style action head; ``action_pre`` is shape-audit metadata."""

        if action_pre is not None:
            if not isinstance(action_pre, dict):
                raise TypeError(
                    f"`action_pre` must be a dict or None, got {type(action_pre)}."
                )
            meta = action_pre.get("meta")
            if isinstance(meta, dict):
                expected = (
                    int(meta.get("batch_size", tokens.shape[0])),
                    int(meta.get("seq_len", tokens.shape[1])),
                )
                if tuple(tokens.shape[:2]) != expected:
                    raise ValueError(
                        "Post-DiT token shape does not match pre-DiT metadata: "
                        f"{tuple(tokens.shape[:2])} versus {expected}."
                    )
        return self.postprocess_action(tokens)

    def forward_action_with_cache(
        self,
        action_tokens: torch.Tensor,
        timestep: torch.Tensor,
        condition_cache: DinoActionConditionCache,
    ) -> dict[str, torch.Tensor]:
        """Run the 12-layer action expert against a reusable condition cache."""

        if len(condition_cache.layers) != self.num_layers:
            raise ValueError(
                f"Condition cache has {len(condition_cache.layers)} layers; "
                f"expected {self.num_layers}."
            )
        pre_state = self.preprocess_action(action_tokens, timestep)
        hidden_states = pre_state["hidden_states"]
        timestep_hidden = pre_state["timestep_embedding"]
        if int(hidden_states.shape[0]) != condition_cache.visual.batch_size:
            raise ValueError(
                "Action/condition batch mismatch: "
                f"{hidden_states.shape[0]} versus {condition_cache.visual.batch_size}."
            )
        for layer_index, condition in enumerate(condition_cache.layers):
            hidden_states = self.forward_action_layer(
                hidden_states,
                timestep_hidden,
                layer_index=layer_index,
                condition=condition,
            )
        prediction = self.postprocess_action(hidden_states)
        return {
            "pred_action": prediction,
            "hidden_states": hidden_states,
            "timestep_embedding": timestep_hidden,
        }

    def forward(
        self,
        *,
        dino_backbone: nn.Module,
        pixel_values: torch.Tensor,
        action_tokens: torch.Tensor,
        timestep: torch.Tensor,
        text_context: torch.Tensor,
        text_attention_mask: Optional[torch.Tensor] = None,
        num_views: Optional[int] = None,
        return_condition_cache: bool = False,
    ) -> dict[str, Any]:
        """Training convenience API: native DINO cache followed by ActionDiT."""

        condition_cache = self.build_condition_cache(
            dino_backbone,
            pixel_values,
            text_context,
            text_attention_mask,
            num_views=num_views,
        )
        outputs: dict[str, Any] = self.forward_action_with_cache(
            action_tokens,
            timestep,
            condition_cache,
        )
        outputs["visual_patch_tokens"] = (
            condition_cache.visual.packed_patch_tokens
        )
        outputs["visual_final_hidden_states"] = (
            condition_cache.visual.final_hidden_states
        )
        if return_condition_cache:
            outputs["condition_cache"] = condition_cache
        return outputs


class ActionDiTDino24(ActionDiTDino12):
    """Parameter-matched DINOv2-B action expert.

    The visual branch remains the native 12-layer, 768-wide DINOv2-B.  Each
    normalized DINO layer state is widened once through a learned 768->1536
    adapter and conditions two consecutive 1536-wide action blocks.  The
    action expert therefore has 24 layers, 12 heads, and head_dim=128 while
    preserving the directional JAMPolicy mask: action reads text/vision/action,
    and DINO never reads action.

    Production defaults are deliberately fixed to the parameter-matched
    topology.  ``strict_shape=False`` exists only for small CPU tests and keeps
    the same 2x-width/2x-depth construction.
    """

    EXPECTED_VISUAL_HIDDEN_SIZE = 768
    EXPECTED_VISUAL_NUM_HEADS = 12
    EXPECTED_VISUAL_NUM_LAYERS = 12
    EXPECTED_HIDDEN_SIZE = 1536
    EXPECTED_NUM_HEADS = 12
    EXPECTED_NUM_LAYERS = 24

    def __init__(
        self,
        *,
        action_dim: int = 7,
        text_dim: int = 4096,
        visual_hidden_size: int = EXPECTED_VISUAL_HIDDEN_SIZE,
        visual_num_heads: int = EXPECTED_VISUAL_NUM_HEADS,
        visual_num_layers: int = EXPECTED_VISUAL_NUM_LAYERS,
        hidden_size: int = EXPECTED_HIDDEN_SIZE,
        num_heads: int = EXPECTED_NUM_HEADS,
        num_layers: int = EXPECTED_NUM_LAYERS,
        mlp_ratio: float = 4.0,
        layer_norm_eps: float = 1.0e-6,
        layerscale_value: float = 1.0,
        qkv_bias: bool = True,
        max_action_horizon: int = 64,
        timestep_embed_dim: int = 256,
        time_factor: float = 1000.0,
        initializer_range: float = 0.02,
        strict_shape: bool = True,
    ) -> None:
        visual_hidden_size = int(visual_hidden_size)
        visual_num_heads = int(visual_num_heads)
        visual_num_layers = int(visual_num_layers)
        hidden_size = int(hidden_size)
        num_heads = int(num_heads)
        num_layers = int(num_layers)
        strict_shape = bool(strict_shape)

        if strict_shape:
            expected = {
                "visual_hidden_size": self.EXPECTED_VISUAL_HIDDEN_SIZE,
                "visual_num_heads": self.EXPECTED_VISUAL_NUM_HEADS,
                "visual_num_layers": self.EXPECTED_VISUAL_NUM_LAYERS,
                "hidden_size": self.EXPECTED_HIDDEN_SIZE,
                "num_heads": self.EXPECTED_NUM_HEADS,
                "num_layers": self.EXPECTED_NUM_LAYERS,
            }
            actual = {
                "visual_hidden_size": visual_hidden_size,
                "visual_num_heads": visual_num_heads,
                "visual_num_layers": visual_num_layers,
                "hidden_size": hidden_size,
                "num_heads": num_heads,
                "num_layers": num_layers,
            }
            mismatches = {
                name: (actual[name], value)
                for name, value in expected.items()
                if actual[name] != value
            }
            if mismatches:
                raise ValueError(
                    "ActionDiTDino24 production topology mismatch: "
                    f"{mismatches}."
                )
        if hidden_size != 2 * visual_hidden_size:
            raise ValueError(
                "ActionDiTDino24 requires action hidden size = 2 * visual "
                f"hidden size, got {hidden_size} versus {visual_hidden_size}."
            )
        if num_layers != 2 * visual_num_layers:
            raise ValueError(
                "ActionDiTDino24 requires two action blocks per visual layer, "
                f"got {num_layers} versus {visual_num_layers}."
            )
        if num_heads != visual_num_heads:
            raise ValueError(
                "ActionDiTDino24 keeps the DINO head count while doubling head "
                f"dimension, got {num_heads} versus {visual_num_heads}."
            )

        super().__init__(
            action_dim=action_dim,
            text_dim=text_dim,
            hidden_size=hidden_size,
            num_heads=num_heads,
            num_layers=num_layers,
            mlp_ratio=mlp_ratio,
            layer_norm_eps=layer_norm_eps,
            layerscale_value=layerscale_value,
            qkv_bias=qkv_bias,
            max_action_horizon=max_action_horizon,
            timestep_embed_dim=timestep_embed_dim,
            time_factor=time_factor,
            initializer_range=initializer_range,
            strict_shape=False,
        )
        self.strict_shape = strict_shape
        self.visual_hidden_size = visual_hidden_size
        self.visual_num_heads = visual_num_heads
        self.visual_head_dim = visual_hidden_size // visual_num_heads
        self.visual_num_layers = visual_num_layers
        self.action_layer_map = tuple(
            visual_layer
            for visual_layer in range(visual_num_layers)
            for _ in range(2)
        )
        self.visual_adapters = nn.ModuleList(
            [
                nn.Linear(visual_hidden_size, hidden_size)
                for _ in range(visual_num_layers)
            ]
        )
        self._reset_visual_adapters_as_exact_duplication()

    @torch.no_grad()
    def _reset_visual_adapters_as_exact_duplication(self) -> None:
        """Initialize each 2x adapter as ``repeat_interleave(x, 2)``."""

        source_index = torch.arange(self.visual_hidden_size)
        target_even = 2 * source_index
        target_odd = target_even + 1
        for adapter in self.visual_adapters:
            adapter.weight.zero_()
            adapter.bias.zero_()
            adapter.weight[target_even, source_index] = 1.0
            adapter.weight[target_odd, source_index] = 1.0

    @staticmethod
    @torch.no_grad()
    def _copy_vector_2x(
        target: torch.Tensor,
        source: torch.Tensor,
        *,
        scale: float = 1.0,
    ) -> None:
        expanded = source.detach().repeat_interleave(2) * float(scale)
        if tuple(target.shape) != tuple(expanded.shape):
            raise ValueError(
                "2x vector inflation shape mismatch: "
                f"target={tuple(target.shape)} source={tuple(source.shape)}."
            )
        target.copy_(expanded.to(device=target.device, dtype=target.dtype))

    @staticmethod
    @torch.no_grad()
    def _copy_linear_net2wider(
        target: nn.Linear,
        source: nn.Linear,
        *,
        output_scale: float = 1.0,
    ) -> None:
        """Function-preserving 2x-width inflation for a dense linear layer.

        Input features are duplicated in adjacent pairs.  Averaging each input
        pair and duplicating each output gives ``E @ W @ P``.  Q/K use an
        additional ``2**(-1/4)`` scale so 128-D attention logits initially
        match the source 64-D scaled dot product.
        """

        expanded_weight = (
            source.weight.detach()
            .repeat_interleave(2, dim=0)
            .repeat_interleave(2, dim=1)
            .mul(0.5 * float(output_scale))
        )
        if tuple(target.weight.shape) != tuple(expanded_weight.shape):
            raise ValueError(
                "2x linear weight inflation shape mismatch: "
                f"target={tuple(target.weight.shape)} "
                f"source={tuple(source.weight.shape)}."
            )
        target.weight.copy_(
            expanded_weight.to(
                device=target.weight.device,
                dtype=target.weight.dtype,
            )
        )
        if (target.bias is None) != (source.bias is None):
            raise ValueError("2x linear inflation requires matching bias layout.")
        if target.bias is not None:
            expanded_bias = (
                source.bias.detach().repeat_interleave(2)
                * float(output_scale)
            )
            target.bias.copy_(
                expanded_bias.to(
                    device=target.bias.device,
                    dtype=target.bias.dtype,
                )
            )

    @torch.no_grad()
    def _copy_block_from_hf_dinov2_layer_2x(
        self,
        target: DinoActionBlock,
        source: nn.Module,
    ) -> None:
        attention = source.attention
        self_attention = attention.attention
        source_mlp = source.mlp
        if not hasattr(source_mlp, "fc1") or not hasattr(source_mlp, "fc2"):
            raise TypeError(
                "ActionDiTDino24 requires the standard DINOv2 GELU MLP."
            )

        self._copy_vector_2x(target.norm1.weight, source.norm1.weight)
        self._copy_vector_2x(target.norm1.bias, source.norm1.bias)
        qk_scale = 2.0 ** (-0.25)
        self._copy_linear_net2wider(
            target.query,
            self_attention.query,
            output_scale=qk_scale,
        )
        self._copy_linear_net2wider(
            target.key,
            self_attention.key,
            output_scale=qk_scale,
        )
        self._copy_linear_net2wider(target.value, self_attention.value)
        self._copy_linear_net2wider(target.out, attention.output.dense)
        # Two consecutive action blocks represent one DINO depth.  Splitting
        # each copied LayerScale residual equally avoids doubling its initial
        # residual magnitude solely because depth was duplicated.
        self._copy_vector_2x(
            target.layer_scale1.lambda1,
            source.layer_scale1.lambda1,
            scale=0.5,
        )
        self._copy_vector_2x(target.norm2.weight, source.norm2.weight)
        self._copy_vector_2x(target.norm2.bias, source.norm2.bias)
        self._copy_linear_net2wider(target.mlp.fc1, source_mlp.fc1)
        self._copy_linear_net2wider(target.mlp.fc2, source_mlp.fc2)
        self._copy_vector_2x(
            target.layer_scale2.lambda1,
            source.layer_scale2.lambda1,
            scale=0.5,
        )
        target.modulation[-1].weight.zero_()
        target.modulation[-1].bias.zero_()

    @classmethod
    def from_hf_dinov2(
        cls,
        backbone: nn.Module,
        **kwargs: Any,
    ) -> "ActionDiTDino24":
        """Build the 24x1536 expert from a 12x768 DINOv2 backbone."""

        strict_shape = bool(kwargs.pop("strict_shape", True))
        layers, config = cls._validate_hf_dinov2(
            backbone,
            expected_hidden_size=(
                cls.EXPECTED_VISUAL_HIDDEN_SIZE if strict_shape else None
            ),
            expected_num_layers=(
                cls.EXPECTED_VISUAL_NUM_LAYERS if strict_shape else None
            ),
            expected_num_heads=(
                cls.EXPECTED_VISUAL_NUM_HEADS if strict_shape else None
            ),
        )
        visual_hidden_size = int(config.hidden_size)
        visual_num_layers = int(config.num_hidden_layers)
        visual_num_heads = int(config.num_attention_heads)
        fixed = {
            "visual_hidden_size": visual_hidden_size,
            "visual_num_layers": visual_num_layers,
            "visual_num_heads": visual_num_heads,
            "hidden_size": 2 * visual_hidden_size,
            "num_layers": 2 * visual_num_layers,
            "num_heads": visual_num_heads,
            "mlp_ratio": float(config.mlp_ratio),
            "layer_norm_eps": float(config.layer_norm_eps),
            "layerscale_value": float(config.layerscale_value),
            "qkv_bias": bool(config.qkv_bias),
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
                "DINO-derived ActionDiTDino24 shape overrides are "
                f"incompatible: {incompatible}."
            )
        fixed.update(kwargs)
        model = cls(**fixed)
        reference = next(backbone.parameters())
        model = model.to(device=reference.device, dtype=reference.dtype)
        model.copy_blocks_from_hf_dinov2(backbone)
        return model

    @torch.no_grad()
    def copy_blocks_from_hf_dinov2(self, backbone: nn.Module) -> None:
        """Inflate every DINO block into its two matching action blocks."""

        layers, _ = self._validate_hf_dinov2(
            backbone,
            expected_hidden_size=self.visual_hidden_size,
            expected_num_layers=self.visual_num_layers,
            expected_num_heads=self.visual_num_heads,
        )
        for action_index, visual_index in enumerate(self.action_layer_map):
            self._copy_block_from_hf_dinov2_layer_2x(
                self.blocks[action_index],
                layers[visual_index],
            )

    @staticmethod
    def _reshape_patch_tokens(
        normalized: torch.Tensor,
        *,
        batch_size: int,
        num_views: int,
        prefix_tokens: int,
        hidden_size: int,
    ) -> torch.Tensor:
        if normalized.ndim != 3:
            raise ValueError(
                "DINO normalized tokens must be [B*V,S,D], "
                f"got {tuple(normalized.shape)}."
            )
        flat_batch, seq_len, actual_hidden = map(int, normalized.shape)
        if flat_batch != int(batch_size) * int(num_views):
            raise ValueError(
                "DINO normalized-token batch mismatch: "
                f"{flat_batch} versus {batch_size}*{num_views}."
            )
        if actual_hidden != int(hidden_size):
            raise ValueError(
                "DINO normalized-token hidden mismatch: "
                f"{actual_hidden} versus {hidden_size}."
            )
        if seq_len <= int(prefix_tokens):
            raise ValueError(
                f"DINO sequence {seq_len} has no patches after prefix."
            )
        return (
            normalized.reshape(
                batch_size,
                num_views,
                seq_len,
                hidden_size,
            )[:, :, prefix_tokens:]
            .reshape(
                batch_size,
                num_views * (seq_len - prefix_tokens),
                hidden_size,
            )
            .contiguous()
        )

    def run_dino_native_layers(
        self,
        backbone: nn.Module,
        pixel_values: torch.Tensor,
        *,
        num_views: Optional[int] = None,
        prefix_tokens: int = 1,
    ) -> DinoNativeOutput:
        """Run DINO once and expose 24 widened, layer-specific action K/Vs."""

        layers, config = self._validate_hf_dinov2(
            backbone,
            expected_hidden_size=self.visual_hidden_size,
            expected_num_layers=self.visual_num_layers,
            expected_num_heads=self.visual_num_heads,
        )
        if int(prefix_tokens) != 1:
            raise ValueError(
                "DINOv2 uses exactly one CLS prefix token; "
                f"got prefix_tokens={prefix_tokens}."
            )
        flat_pixels, batch_size, num_views = self._resolve_view_batch(
            pixel_values,
            num_views,
        )
        hidden_states = backbone.embeddings(flat_pixels)
        layer_patch_kv: list[DinoLayerPatchKV] = []
        for visual_index, layer in enumerate(layers):
            layer_input = hidden_states
            layer_outputs = layer(
                layer_input,
                head_mask=None,
                output_attentions=False,
            )
            if not isinstance(layer_outputs, tuple) or not layer_outputs:
                raise RuntimeError(
                    f"DINO layer {visual_index} returned an invalid output."
                )
            hidden_states = layer_outputs[0]

            normalized = layer.norm1(layer_input)
            patch_tokens = self._reshape_patch_tokens(
                normalized,
                batch_size=batch_size,
                num_views=num_views,
                prefix_tokens=prefix_tokens,
                hidden_size=self.visual_hidden_size,
            )
            widened = self.visual_adapters[visual_index](patch_tokens)
            action_start = 2 * visual_index
            for action_index in (action_start, action_start + 1):
                key, value = self.blocks[action_index].text_kv(widened)
                layer_patch_kv.append(
                    DinoLayerPatchKV(key=key, value=value)
                )

        if len(layer_patch_kv) != self.num_layers:
            raise RuntimeError(
                "Expected one widened visual K/V per action layer, "
                f"got {len(layer_patch_kv)} versus {self.num_layers}."
            )
        final_hidden_states = backbone.layernorm(hidden_states)
        seq_len = int(final_hidden_states.shape[1])
        patches_per_view = seq_len - int(prefix_tokens)
        if patches_per_view <= 0:
            raise RuntimeError(
                f"DINO final sequence has no patches: {tuple(final_hidden_states.shape)}."
            )
        patch_tokens_by_view = final_hidden_states.reshape(
            batch_size,
            num_views,
            seq_len,
            int(config.hidden_size),
        )[:, :, prefix_tokens:]
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


class ActionDiTDinoLarge24(ActionDiTDino12):
    """Native action expert paired one-to-one with DINOv2-L/14.

    Unlike :class:`ActionDiTDino24`, this class does not widen or repeat a
    12-layer DINOv2-B stream.  DINOv2-L is already 24 layers deep and 1024
    wide, so every action block copies the matching native DINOv2-L block and
    reads that layer's original 16-head, 64-d/head patch K/V.
    """

    EXPECTED_NUM_LAYERS = 24
    EXPECTED_HIDDEN_SIZE = 1024
    EXPECTED_NUM_HEADS = 16

    def __init__(
        self,
        *,
        action_dim: int = 7,
        text_dim: int = 4096,
        hidden_size: int = EXPECTED_HIDDEN_SIZE,
        num_heads: int = EXPECTED_NUM_HEADS,
        num_layers: int = EXPECTED_NUM_LAYERS,
        mlp_ratio: float = 4.0,
        layer_norm_eps: float = 1.0e-6,
        layerscale_value: float = 1.0,
        qkv_bias: bool = True,
        max_action_horizon: int = 64,
        timestep_embed_dim: int = 256,
        time_factor: float = 1000.0,
        initializer_range: float = 0.02,
        strict_shape: bool = True,
    ) -> None:
        super().__init__(
            action_dim=int(action_dim),
            text_dim=int(text_dim),
            hidden_size=int(hidden_size),
            num_heads=int(num_heads),
            num_layers=int(num_layers),
            mlp_ratio=float(mlp_ratio),
            layer_norm_eps=float(layer_norm_eps),
            layerscale_value=float(layerscale_value),
            qkv_bias=bool(qkv_bias),
            max_action_horizon=int(max_action_horizon),
            timestep_embed_dim=int(timestep_embed_dim),
            time_factor=float(time_factor),
            initializer_range=float(initializer_range),
            strict_shape=bool(strict_shape),
        )


# Short generic name used by runtime code; production defaults remain strict
# DINOv2-B 12x768 unless callers explicitly pass ``strict_shape=False``.
ActionDiTDino = ActionDiTDino12


__all__ = [
    "ActionDiTDino",
    "ActionDiTDino12",
    "ActionDiTDino24",
    "ActionDiTDinoLarge24",
    "DinoActionBlock",
    "DinoActionConditionCache",
    "DinoActionLayerCondition",
    "DinoLayerPatchKV",
    "DinoNativeOutput",
]
