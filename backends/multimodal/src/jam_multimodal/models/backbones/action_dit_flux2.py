from __future__ import annotations

from typing import Any, Dict, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from jam_multimodal.utils.logging_config import get_logger

from .flux2_imports import ensure_flux2_importable

logger = get_logger(__name__)


class SlimFlux2SelfAttention(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, attn_head_dim: int):
        super().__init__()
        ensure_flux2_importable()
        from flux2.model import QKNorm

        self.hidden_dim = int(hidden_dim)
        self.num_heads = int(num_heads)
        self.attn_head_dim = int(attn_head_dim)
        self.attn_dim = self.num_heads * self.attn_head_dim
        self.qkv = nn.Linear(self.hidden_dim, 3 * self.attn_dim, bias=False)
        self.norm = QKNorm(self.attn_head_dim)
        self.proj = nn.Linear(self.attn_dim, self.hidden_dim, bias=False)


class SlimFlux2DoubleBlock(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, attn_head_dim: int, mlp_ratio: float = 4.0):
        super().__init__()
        ensure_flux2_importable()
        from flux2.model import SiLUActivation

        self.hidden_size = int(hidden_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_heads = int(num_heads)
        self.attn_head_dim = int(attn_head_dim)
        self.attn_dim = self.num_heads * self.attn_head_dim
        self.mlp_hidden_dim = int(round(self.hidden_dim * float(mlp_ratio)))
        self.mlp_mult_factor = 2
        self.img_norm1 = nn.LayerNorm(self.hidden_dim, elementwise_affine=False, eps=1e-6)
        self.img_attn = SlimFlux2SelfAttention(self.hidden_dim, self.num_heads, self.attn_head_dim)
        self.img_norm2 = nn.LayerNorm(self.hidden_dim, elementwise_affine=False, eps=1e-6)
        self.img_mlp = nn.Sequential(
            nn.Linear(self.hidden_dim, self.mlp_hidden_dim * self.mlp_mult_factor, bias=False),
            SiLUActivation(),
            nn.Linear(self.mlp_hidden_dim, self.hidden_dim, bias=False),
        )

    def prepare_qkv(self, x: torch.Tensor, pe: torch.Tensor, mod_img):
        from einops import rearrange
        from flux2.model import apply_rope

        mod1, mod2 = mod_img
        mod1_shift, mod1_scale, mod1_gate = mod1
        mod2_shift, mod2_scale, mod2_gate = mod2
        x_mod = (1 + mod1_scale) * self.img_norm1(x) + mod1_shift
        qkv = self.img_attn.qkv(x_mod)
        q, k, v = rearrange(qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads)
        q, k = self.img_attn.norm(q, k, v)
        q, k = apply_rope(q, k, pe)
        return {
            "q": q.transpose(1, 2).reshape(x.shape[0], x.shape[1], self.attn_dim),
            "k": k.transpose(1, 2).reshape(x.shape[0], x.shape[1], self.attn_dim),
            "v": v.transpose(1, 2).reshape(x.shape[0], x.shape[1], self.attn_dim),
            "residual_x": x,
            "mod2_shift": mod2_shift,
            "mod2_scale": mod2_scale,
            "mod1_gate": mod1_gate,
            "mod2_gate": mod2_gate,
        }

    def apply_post(self, mixed_attn_out: torch.Tensor, state: dict) -> torch.Tensor:
        x = state["residual_x"] + state["mod1_gate"] * self.img_attn.proj(mixed_attn_out)
        x = x + state["mod2_gate"] * self.img_mlp(
            (1 + state["mod2_scale"]) * self.img_norm2(x) + state["mod2_shift"]
        )
        return x


class SlimFlux2SingleBlock(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, attn_head_dim: int, mlp_ratio: float = 4.0):
        super().__init__()
        ensure_flux2_importable()
        from flux2.model import QKNorm, SiLUActivation

        self.hidden_size = int(hidden_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_heads = int(num_heads)
        self.attn_head_dim = int(attn_head_dim)
        self.attn_dim = self.num_heads * self.attn_head_dim
        self.mlp_hidden_dim = int(round(self.hidden_dim * float(mlp_ratio)))
        self.mlp_mult_factor = 2
        self.linear1 = nn.Linear(
            self.hidden_dim,
            3 * self.attn_dim + self.mlp_hidden_dim * self.mlp_mult_factor,
            bias=False,
        )
        self.linear2 = nn.Linear(self.attn_dim + self.mlp_hidden_dim, self.hidden_dim, bias=False)
        self.norm = QKNorm(self.attn_head_dim)
        self.pre_norm = nn.LayerNorm(self.hidden_dim, elementwise_affine=False, eps=1e-6)
        self.mlp_act = SiLUActivation()

    def prepare_qkv(self, x: torch.Tensor, pe: torch.Tensor, mod):
        from einops import rearrange
        from flux2.model import apply_rope

        mod_shift, mod_scale, mod_gate = mod
        x_mod = (1 + mod_scale) * self.pre_norm(x) + mod_shift
        qkv, mlp = torch.split(
            self.linear1(x_mod),
            [3 * self.attn_dim, self.mlp_hidden_dim * self.mlp_mult_factor],
            dim=-1,
        )
        q, k, v = rearrange(qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads)
        q, k = self.norm(q, k, v)
        q, k = apply_rope(q, k, pe)
        return {
            "q": q.transpose(1, 2).reshape(x.shape[0], x.shape[1], self.attn_dim),
            "k": k.transpose(1, 2).reshape(x.shape[0], x.shape[1], self.attn_dim),
            "v": v.transpose(1, 2).reshape(x.shape[0], x.shape[1], self.attn_dim),
            "mlp": mlp,
            "gate": mod_gate,
            "residual_x": x,
        }

    def apply_post(self, mixed_attn_out: torch.Tensor, state: dict) -> torch.Tensor:
        output = self.linear2(torch.cat((mixed_attn_out, self.mlp_act(state["mlp"])), dim=2))
        return state["residual_x"] + state["gate"] * output


class Flux2ActionHead(nn.Module):
    def __init__(self, hidden_dim: int, action_dim: int):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_dim, action_dim, bias=False)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_dim, 2 * hidden_dim, bias=False))

    def forward(self, x: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
        shift, scale = self.adaLN_modulation(vec).chunk(2, dim=-1)
        return self.linear((1 + scale[:, None, :]) * self.norm_final(x) + shift[:, None, :])


class ActionDiTFlux2(nn.Module):
    """Slim action expert with FLUX.2-compatible attention dimensions."""

    block_protocol = "flux2"

    def __init__(
        self,
        action_dim: int,
        hidden_dim: int = 1024,
        num_heads: int = 24,
        attn_head_dim: int = 128,
        num_layers_double: int = 5,
        num_layers_single: int = 20,
        mlp_ratio: float = 4.0,
        max_action_horizon: int = 64,
        use_gradient_checkpointing: bool = False,
    ) -> None:
        super().__init__()
        ensure_flux2_importable()
        from flux2.model import MLPEmbedder, Modulation
        from flux2.model import EmbedND

        self.action_dim = int(action_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_heads = int(num_heads)
        self.num_kv_heads = self.num_heads
        self.attn_head_dim = int(attn_head_dim)
        self.attn_dim = self.num_heads * self.attn_head_dim
        self.double_layers = int(num_layers_double)
        self.single_layers = int(num_layers_single)
        self.max_action_horizon = int(max_action_horizon)
        self.use_gradient_checkpointing = bool(use_gradient_checkpointing)

        # FLUX.2 positional encoding is parameter-free, so keeping an identical
        # embedder on the slim action expert does not change checkpoint format.
        # It also lets external visual backbones build the same 4-axis RoPE
        # coordinates without constructing the 4B FLUX video transformer.
        self.pe_embedder = EmbedND(
            dim=self.attn_head_dim,
            theta=2000,
            axes_dim=[32, 32, 32, 32],
        )

        self.action_encoder = nn.Linear(self.action_dim, self.hidden_dim)
        self.time_in = MLPEmbedder(in_dim=256, hidden_dim=self.hidden_dim, disable_bias=True)
        self.double_stream_modulation_img = Modulation(self.hidden_dim, double=True, disable_bias=True)
        self.single_stream_modulation = Modulation(self.hidden_dim, double=False, disable_bias=True)
        self.double_blocks = nn.ModuleList(
            [
                SlimFlux2DoubleBlock(
                    hidden_dim=self.hidden_dim,
                    num_heads=self.num_heads,
                    attn_head_dim=self.attn_head_dim,
                    mlp_ratio=mlp_ratio,
                )
                for _ in range(self.double_layers)
            ]
        )
        self.single_blocks = nn.ModuleList(
            [
                SlimFlux2SingleBlock(
                    hidden_dim=self.hidden_dim,
                    num_heads=self.num_heads,
                    attn_head_dim=self.attn_head_dim,
                    mlp_ratio=mlp_ratio,
                )
                for _ in range(self.single_layers)
            ]
        )
        self.head = Flux2ActionHead(self.hidden_dim, self.action_dim)

    @property
    def blocks(self):
        return list(self.double_blocks) + list(self.single_blocks)

    @classmethod
    def from_pretrained(
        cls,
        action_dit_config: dict[str, Any],
        action_dit_pretrained_path: str | None = None,
        skip_dit_load_from_pretrain: bool = False,
        device: str = "cuda",
        torch_dtype: torch.dtype = torch.bfloat16,
    ) -> "ActionDiTFlux2":
        if action_dit_config is None:
            raise ValueError("`action_dit_config` is required for ActionDiTFlux2.from_pretrained().")
        model = cls(**action_dit_config).to(device=device, dtype=torch_dtype)
        if skip_dit_load_from_pretrain or not action_dit_pretrained_path:
            logger.info("Initializing ActionDiTFlux2 without pretrained action weights.")
            return model
        payload = torch.load(action_dit_pretrained_path, map_location="cpu")
        state_dict = payload.get("state_dict", payload) if isinstance(payload, dict) else payload
        if not isinstance(state_dict, dict):
            raise ValueError(f"Invalid ActionDiTFlux2 checkpoint type: {type(payload)}")
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            logger.warning("ActionDiTFlux2 missing keys when loading: %s", missing[:20])
        if unexpected:
            logger.warning("ActionDiTFlux2 unexpected keys when loading: %s", unexpected[:20])
        return model

    @staticmethod
    def build_action_ids(
        batch_size: int,
        seq_len: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        ids = torch.zeros(batch_size, seq_len, 4, device=device, dtype=dtype)
        ids[..., 0] = 2.0
        ids[..., 1] = torch.arange(seq_len, device=device, dtype=dtype)[None, :]
        return ids

    def pre_dit(
        self,
        action_tokens: torch.Tensor,
        timestep: torch.Tensor,
        context: torch.Tensor | None = None,
        context_mask: torch.Tensor | None = None,
    ) -> Dict[str, Any]:
        del context, context_mask
        if action_tokens.ndim != 3:
            raise ValueError(f"`action_tokens` must be [B,T,D], got {tuple(action_tokens.shape)}")
        batch_size, seq_len, action_dim = action_tokens.shape
        if action_dim != self.action_dim:
            raise ValueError(f"Expected action_dim={self.action_dim}, got {action_dim}")
        if seq_len > self.max_action_horizon:
            raise ValueError(f"Action length {seq_len} exceeds max_action_horizon={self.max_action_horizon}")
        if timestep.ndim != 1:
            raise ValueError(f"`timestep` must be [B], got {tuple(timestep.shape)}")
        if timestep.shape[0] == 1 and batch_size > 1:
            timestep = timestep.expand(batch_size)
        if timestep.shape[0] != batch_size:
            raise ValueError(f"`timestep` length must match batch size {batch_size}, got {timestep.shape[0]}")

        from flux2.model import timestep_embedding

        tokens = self.action_encoder(action_tokens)
        vec = self.time_in(timestep_embedding(timestep, 256)).to(dtype=tokens.dtype)
        double_mod_img = self.double_stream_modulation_img(vec)
        single_mod, _ = self.single_stream_modulation(vec)
        ids = self.build_action_ids(batch_size, seq_len, device=tokens.device, dtype=tokens.dtype)
        return {
            "tokens": tokens,
            "ids": ids,
            "t_mod": {"vec": vec, "double_img": double_mod_img, "single": single_mod},
            "context": None,
            "context_mask": None,
            "meta": {"batch_size": batch_size, "seq_len": seq_len},
        }

    def post_dit(self, tokens: torch.Tensor, pre_state: Dict[str, Any]) -> torch.Tensor:
        return self.head(tokens, pre_state["t_mod"]["vec"])

    @staticmethod
    def _flatten_heads(tensor: torch.Tensor) -> torch.Tensor:
        return tensor.transpose(1, 2).reshape(
            tensor.shape[0], tensor.shape[2], tensor.shape[1] * tensor.shape[3]
        )

    def _mixed_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, query_len = query.shape[:2]
        key_len = key.shape[1]
        expected_mask = (batch_size, query_len, key_len)
        if tuple(attention_mask.shape) != expected_mask:
            raise ValueError(
                f"FLUX.2 external-prefix mask must be {expected_mask}, "
                f"got {tuple(attention_mask.shape)}."
            )
        query = query.view(
            batch_size, query_len, self.num_heads, self.attn_head_dim
        ).transpose(1, 2)
        key = key.view(
            batch_size, key_len, self.num_heads, self.attn_head_dim
        ).transpose(1, 2)
        value = value.view(
            batch_size, key_len, self.num_heads, self.attn_head_dim
        ).transpose(1, 2)
        output = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attention_mask[:, None].to(
                device=query.device, dtype=torch.bool
            ),
        )
        return output.transpose(1, 2).reshape(
            batch_size, query_len, self.attn_dim
        )

    @staticmethod
    def _build_text_ids(
        batch_size: int,
        seq_len: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        ids = torch.zeros(batch_size, seq_len, 4, device=device, dtype=dtype)
        ids[..., 3] = torch.arange(seq_len, device=device, dtype=dtype)[None, :]
        return ids

    @staticmethod
    def _build_external_visual_ids(
        *,
        batch_size: int,
        grid_height: int,
        grid_width: int,
        num_views: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if min(grid_height, grid_width, num_views) <= 0:
            raise ValueError(
                "External visual grid and view count must be positive, got "
                f"{grid_height}x{grid_width}, views={num_views}."
            )
        per_view = []
        for view_idx in range(int(num_views)):
            ids = torch.zeros(
                grid_height, grid_width, 4, device=device, dtype=dtype
            )
            # Match the official FLUX.2 ImageWAM current/reference image ID.
            ids[..., 0] = 10.0
            ids[..., 1] = torch.arange(
                grid_height, device=device, dtype=dtype
            )[:, None]
            ids[..., 2] = (
                torch.arange(grid_width, device=device, dtype=dtype)[None, :]
                + view_idx * grid_width
            )
            per_view.append(ids.reshape(1, grid_height * grid_width, 4))
        return torch.cat(per_view, dim=1).expand(batch_size, -1, -1)

    def prepare_layerwise_prefix(
        self,
        *,
        layerwise_text_tokens: Sequence[torch.Tensor],
        layerwise_visual_tokens: Sequence[torch.Tensor],
        text_mask: torch.Tensor,
        visual_grid_height: int,
        visual_grid_width: int,
        num_views: int,
        fusion_type: str,
        source_name: str,
    ) -> dict[str, Any]:
        """Build per-layer K/V from an external visual/text prefix.

        The action blocks, attention dimensions, timestep modulation, RoPE and
        pretrained weights remain the official ActionDiTFlux2-25 design.  The
        absent FLUX video expert is replaced by projected external text and
        visual states; action queries read those states unidirectionally in the
        same SDPA softmax as action self-attention.
        """

        source_name = str(source_name)
        fusion_type = str(fusion_type)
        if not source_name or not fusion_type:
            raise ValueError("`source_name` and `fusion_type` must be non-empty.")
        total_layers = len(self.blocks)
        if len(layerwise_text_tokens) != total_layers:
            raise ValueError(
                f"{source_name} text/action depth mismatch: expected "
                f"{total_layers}, got {len(layerwise_text_tokens)}."
            )
        if len(layerwise_visual_tokens) != total_layers:
            raise ValueError(
                f"{source_name} visual/action depth mismatch: expected "
                f"{total_layers}, got {len(layerwise_visual_tokens)}."
            )
        reference_text = layerwise_text_tokens[0]
        reference_visual = layerwise_visual_tokens[0]
        if reference_text.ndim != 3 or reference_visual.ndim != 3:
            raise ValueError(
                f"{source_name} text/visual prefixes must both be [B,L,D]."
            )
        batch_size, text_len, hidden_dim = reference_text.shape
        visual_len = int(reference_visual.shape[1])
        if hidden_dim != self.hidden_dim or reference_visual.shape[-1] != self.hidden_dim:
            raise ValueError(
                f"{source_name} prefixes must use ActionDiTFlux2 hidden="
                f"{self.hidden_dim}, got text={hidden_dim}, "
                f"visual={reference_visual.shape[-1]}."
            )
        expected_visual_len = (
            int(num_views) * int(visual_grid_height) * int(visual_grid_width)
        )
        if visual_len != expected_visual_len:
            raise ValueError(
                f"{source_name} visual prefix length does not match its view "
                f"grid: tokens={visual_len}, expected={expected_visual_len}."
            )
        if tuple(text_mask.shape) != (batch_size, text_len):
            raise ValueError(
                f"{source_name} text mask must be {(batch_size, text_len)}, "
                f"got {tuple(text_mask.shape)}."
            )
        for layer_idx, (text, visual) in enumerate(
            zip(layerwise_text_tokens, layerwise_visual_tokens)
        ):
            if tuple(text.shape) != tuple(reference_text.shape):
                raise ValueError(
                    f"{source_name} text prefix shape changed at action layer "
                    f"{layer_idx}: {tuple(text.shape)} vs "
                    f"{tuple(reference_text.shape)}."
                )
            if tuple(visual.shape) != tuple(reference_visual.shape):
                raise ValueError(
                    f"{source_name} visual prefix shape changed at action layer "
                    f"{layer_idx}: {tuple(visual.shape)} vs "
                    f"{tuple(reference_visual.shape)}."
                )

        text_ids = self._build_text_ids(
            batch_size,
            text_len,
            device=reference_text.device,
            dtype=reference_text.dtype,
        )
        visual_ids = self._build_external_visual_ids(
            batch_size=batch_size,
            grid_height=int(visual_grid_height),
            grid_width=int(visual_grid_width),
            num_views=int(num_views),
            device=reference_visual.device,
            dtype=reference_visual.dtype,
        )
        text_pe = self.pe_embedder(text_ids)
        visual_pe = self.pe_embedder(visual_ids)
        prefix_pe = torch.cat((text_pe, visual_pe), dim=2)

        from flux2.model import timestep_embedding

        clean_timestep = torch.zeros(
            batch_size,
            device=reference_text.device,
            dtype=reference_text.dtype,
        )
        prefix_vec = self.time_in(timestep_embedding(clean_timestep, 256)).to(
            dtype=reference_text.dtype
        )
        prefix_double_mod = self.double_stream_modulation_img(prefix_vec)
        prefix_single_mod, _ = self.single_stream_modulation(prefix_vec)

        double_cache = []
        for layer_idx, block in enumerate(self.double_blocks):
            prefix = torch.cat(
                (
                    layerwise_text_tokens[layer_idx],
                    layerwise_visual_tokens[layer_idx],
                ),
                dim=1,
            )
            state = block.prepare_qkv(prefix, prefix_pe, prefix_double_mod)
            double_cache.append({"k": state["k"], "v": state["v"]})

        single_cache = []
        double_depth = len(self.double_blocks)
        for single_idx, block in enumerate(self.single_blocks):
            layer_idx = double_depth + single_idx
            prefix = torch.cat(
                (
                    layerwise_text_tokens[layer_idx],
                    layerwise_visual_tokens[layer_idx],
                ),
                dim=1,
            )
            state = block.prepare_qkv(prefix, prefix_pe, prefix_single_mod)
            single_cache.append({"k": state["k"], "v": state["v"]})

        return {
            "double": double_cache,
            "single": single_cache,
            "text_mask": text_mask.to(
                device=reference_text.device, dtype=torch.bool
            ),
            "text_len": int(text_len),
            "visual_len": visual_len,
            "batch_size": int(batch_size),
            "fusion_type": fusion_type,
            "source_name": source_name,
        }

    def forward_layerwise_prefix_mot(
        self,
        *,
        action_tokens: torch.Tensor,
        timestep: torch.Tensor,
        prefix_cache: dict[str, Any],
        expected_fusion_type: str,
        source_name: str,
    ) -> torch.Tensor:
        """Run ActionDiTFlux2-25 against cached external prefix K/V."""

        if prefix_cache.get("fusion_type") != str(expected_fusion_type):
            raise ValueError(
                f"Expected a {source_name}->ActionDiTFlux2-25 prefix cache with "
                f"fusion_type={expected_fusion_type!r}."
            )
        pre_state = self.pre_dit(action_tokens, timestep)
        action = pre_state["tokens"]
        action_pe = self.pe_embedder(
            pre_state["ids"].to(device=action.device, dtype=action.dtype)
        )
        batch_size, action_len = action.shape[:2]
        if batch_size != int(prefix_cache["batch_size"]):
            raise ValueError(f"{source_name} prefix/action batch size mismatch.")
        text_len = int(prefix_cache["text_len"])
        visual_len = int(prefix_cache["visual_len"])
        text_valid = prefix_cache["text_mask"]
        visual_valid = torch.ones(
            batch_size, visual_len, device=action.device, dtype=torch.bool
        )
        action_valid = torch.ones(
            batch_size, action_len, device=action.device, dtype=torch.bool
        )
        key_valid = torch.cat(
            (text_valid, visual_valid, action_valid), dim=1
        )
        action_mask = key_valid[:, None, :].expand(-1, action_len, -1)

        if len(prefix_cache["double"]) != len(self.double_blocks):
            raise ValueError(f"{source_name} prefix double-block cache depth mismatch.")
        if len(prefix_cache["single"]) != len(self.single_blocks):
            raise ValueError(f"{source_name} prefix single-block cache depth mismatch.")

        for layer_idx, cache in enumerate(prefix_cache["double"]):
            block = self.double_blocks[layer_idx]
            state = block.prepare_qkv(
                action, action_pe, pre_state["t_mod"]["double_img"]
            )
            mixed = self._mixed_attention(
                state["q"],
                torch.cat((cache["k"], state["k"]), dim=1),
                torch.cat((cache["v"], state["v"]), dim=1),
                action_mask,
            )
            action = block.apply_post(mixed, state)

        for layer_idx, cache in enumerate(prefix_cache["single"]):
            block = self.single_blocks[layer_idx]
            state = block.prepare_qkv(
                action, action_pe, pre_state["t_mod"]["single"]
            )
            mixed = self._mixed_attention(
                state["q"],
                torch.cat((cache["k"], state["k"]), dim=1),
                torch.cat((cache["v"], state["v"]), dim=1),
                action_mask,
            )
            action = block.apply_post(mixed, state)
        return self.post_dit(action, pre_state)
