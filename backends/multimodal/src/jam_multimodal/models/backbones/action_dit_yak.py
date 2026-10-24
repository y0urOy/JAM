from __future__ import annotations

import os
from typing import Any, Dict, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from jam_multimodal.utils.logging_config import get_logger

from .ovis_u1_imports import ensure_ovis_u1_remote_code_importable

ensure_ovis_u1_remote_code_importable()

from ovis_u1_hf.modeling_yak import (  # noqa: E402
    DoubleStreamXBlock,
    EmbedND,
    LastLayer,
    MLPEmbedder,
    SingleStreamBlock,
    apply_rope,
    timestep_embedding,
)

logger = get_logger(__name__)


class ActionDiTYak(nn.Module):
    """Action expert shaped like Ovis-U1's Yak MMDiT."""

    block_protocol = "yak"

    def __init__(
        self,
        action_dim: int,
        residual_dim: int = 1536,
        num_heads: int = 12,
        num_layers_double: int = 6,
        num_layers_single: int = 12,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        max_action_horizon: int = 64,
        use_gradient_checkpointing: bool = False,
    ) -> None:
        super().__init__()
        self.action_dim = int(action_dim)
        self.hidden_dim = int(residual_dim)
        self.num_heads = int(num_heads)
        self.num_kv_heads = self.num_heads
        self.attn_head_dim = self.hidden_dim // self.num_heads
        self.double_layers = int(num_layers_double)
        self.single_layers = int(num_layers_single)
        self.max_action_horizon = int(max_action_horizon)
        self.use_gradient_checkpointing = bool(use_gradient_checkpointing)

        self.action_encoder = nn.Linear(self.action_dim, self.hidden_dim)
        self.time_in = MLPEmbedder(in_dim=256, hidden_dim=self.hidden_dim)
        self.double_blocks = nn.ModuleList(
            [
                DoubleStreamXBlock(
                    self.hidden_dim,
                    self.num_heads,
                    mlp_ratio=float(mlp_ratio),
                    qkv_bias=bool(qkv_bias),
                )
                for _ in range(self.double_layers)
            ]
        )
        self.single_blocks = nn.ModuleList(
            [
                SingleStreamBlock(
                    self.hidden_dim,
                    self.num_heads,
                    mlp_ratio=float(mlp_ratio),
                )
                for _ in range(self.single_layers)
            ]
        )
        # Yak's 3-axis RoPE has no learned parameters.  Keeping an independent
        # instance here lets the action expert consume Qwen prefixes without
        # constructing Ovis, its VAE, or the Yak video expert.
        self.pe_embedder = EmbedND(
            dim=self.attn_head_dim,
            theta=10000,
            axes_dim=[16, 56, 56],
        )
        self.head = LastLayer(self.hidden_dim, 1, self.action_dim)
        if os.environ.get("JAM_ZERO_INIT_HEAD") == "1":
            self._zero_init_head()

    def _zero_init_head(self):
        # Optional (env-gated, default OFF = official random init): zero-init the
        # output head so the initial action prediction is ~0. Standard DiT/Flux
        # practice; it was used as a mitigation while DeepSpeed 0.18.5 corrupted
        # training on aarch64.
        # With deepspeed>=0.19.2 the official random init trains fine.
        nn.init.zeros_(self.head.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.head.adaLN_modulation[-1].bias)
        nn.init.zeros_(self.head.linear.weight)
        nn.init.zeros_(self.head.linear.bias)

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
    ) -> "ActionDiTYak":
        if action_dit_config is None:
            raise ValueError("`action_dit_config` is required for ActionDiTYak.from_pretrained().")
        model = cls(**action_dit_config).to(device=device, dtype=torch_dtype)
        if skip_dit_load_from_pretrain or not action_dit_pretrained_path:
            logger.info("Initializing ActionDiTYak without pretrained action weights.")
            return model

        payload = torch.load(action_dit_pretrained_path, map_location="cpu")
        state_dict = payload.get("state_dict", payload) if isinstance(payload, dict) else payload
        if not isinstance(state_dict, dict):
            raise ValueError(f"Invalid ActionDiTYak checkpoint type: {type(payload)}")
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            logger.warning("ActionDiTYak missing keys when loading: %s", missing[:20])
        if unexpected:
            logger.warning("ActionDiTYak unexpected keys when loading: %s", unexpected[:20])
        return model

    def pre_dit(
        self,
        action_tokens: torch.Tensor,
        timestep: torch.Tensor,
        context: torch.Tensor | None = None,
        context_mask: torch.Tensor | None = None,
    ) -> Dict[str, Any]:
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

        tokens = self.action_encoder(action_tokens)
        vec = self.time_in(timestep_embedding(timestep, 256)).to(dtype=tokens.dtype)
        ids = torch.zeros(
            batch_size,
            seq_len,
            3,
            device=tokens.device,
            dtype=tokens.dtype,
        )
        ids[..., 0] = 2.0
        ids[..., 1] = torch.arange(
            seq_len, device=tokens.device, dtype=tokens.dtype
        )[None, :]
        return {
            "tokens": tokens,
            "ids": ids,
            "t_mod": vec,
            "context": None,
            "context_mask": context_mask,
            "meta": {"batch_size": batch_size, "seq_len": seq_len},
        }

    def post_dit(self, tokens: torch.Tensor, pre_state: Dict[str, Any]) -> torch.Tensor:
        return self.head(tokens, pre_state["t_mod"])

    @staticmethod
    def _flatten_heads(tensor: torch.Tensor) -> torch.Tensor:
        return tensor.transpose(1, 2).reshape(
            tensor.shape[0], tensor.shape[2], tensor.shape[1] * tensor.shape[3]
        )

    def _qkv(
        self,
        attn: nn.Module,
        tokens: torch.Tensor,
        pe: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, seq_len = tokens.shape[:2]
        qkv = attn.qkv(tokens).view(
            batch_size,
            seq_len,
            3,
            self.num_heads,
            self.attn_head_dim,
        )
        query, key, value = qkv.permute(2, 0, 3, 1, 4)
        query, key = attn.norm(query, key, value)
        query, key = apply_rope(query, key, pe)
        return (
            self._flatten_heads(query),
            self._flatten_heads(key),
            self._flatten_heads(value),
        )

    def _attn_kv(
        self,
        attn: nn.Module,
        tokens: torch.Tensor,
        pe: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Evaluate only the K/V slices of a Yak SelfAttention projection."""

        hidden_size = self.hidden_dim
        bias = attn.qkv.bias
        key_value = F.linear(
            tokens,
            attn.qkv.weight[hidden_size : 3 * hidden_size],
            None if bias is None else bias[hidden_size : 3 * hidden_size],
        )
        batch_size, seq_len = tokens.shape[:2]
        key_value = key_value.view(
            batch_size,
            seq_len,
            2,
            self.num_heads,
            self.attn_head_dim,
        )
        key, value = key_value.permute(2, 0, 3, 1, 4)
        key = attn.norm.key_norm(key).to(value)
        _, key = apply_rope(key, key, pe)
        return self._flatten_heads(key), self._flatten_heads(value)

    def _single_io(
        self,
        block: nn.Module,
        tokens: torch.Tensor,
        t_mod: torch.Tensor,
        pe: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        modulation, _ = block.modulation(t_mod)
        modulated = (1 + modulation.scale) * block.pre_norm(tokens) + modulation.shift
        qkv, mlp = torch.split(
            block.linear1(modulated),
            [3 * block.hidden_size, block.mlp_hidden_dim],
            dim=-1,
        )
        batch_size, seq_len = tokens.shape[:2]
        qkv = qkv.view(
            batch_size,
            seq_len,
            3,
            self.num_heads,
            self.attn_head_dim,
        )
        query, key, value = qkv.permute(2, 0, 3, 1, 4)
        query, key = block.norm(query, key, value)
        query, key = apply_rope(query, key, pe)
        return {
            "q": self._flatten_heads(query),
            "k": self._flatten_heads(key),
            "v": self._flatten_heads(value),
            "mlp": mlp,
            "gate": modulation.gate,
            "residual": tokens,
        }

    def _single_kv(
        self,
        block: nn.Module,
        tokens: torch.Tensor,
        t_mod: torch.Tensor,
        pe: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        modulation, _ = block.modulation(t_mod)
        modulated = (1 + modulation.scale) * block.pre_norm(tokens) + modulation.shift
        hidden_size = int(block.hidden_size)
        bias = block.linear1.bias
        key_value = F.linear(
            modulated,
            block.linear1.weight[hidden_size : 3 * hidden_size],
            None if bias is None else bias[hidden_size : 3 * hidden_size],
        )
        batch_size, seq_len = tokens.shape[:2]
        key_value = key_value.view(
            batch_size,
            seq_len,
            2,
            self.num_heads,
            self.attn_head_dim,
        )
        key, value = key_value.permute(2, 0, 3, 1, 4)
        key = block.norm.key_norm(key).to(value)
        _, key = apply_rope(key, key, pe)
        return self._flatten_heads(key), self._flatten_heads(value)

    @staticmethod
    def _single_post(
        block: nn.Module,
        mixed_attention: torch.Tensor,
        state: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        output = block.linear2(
            torch.cat((mixed_attention, block.mlp_act(state["mlp"])), dim=2)
        )
        return state["residual"] + state["gate"] * output

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
                f"Yak mixed-attention mask must be {expected_mask}, "
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
            attn_mask=attention_mask[:, None].to(device=query.device, dtype=torch.bool),
        )
        return output.transpose(1, 2).reshape(batch_size, query_len, self.hidden_dim)

    def _build_external_visual_ids(
        self,
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
                grid_height, grid_width, 3, device=device, dtype=dtype
            )
            ids[..., 0] = 1.0
            ids[..., 1] = torch.arange(
                grid_height, device=device, dtype=dtype
            )[:, None]
            ids[..., 2] = (
                torch.arange(grid_width, device=device, dtype=dtype)[None, :]
                + view_idx * grid_width
            )
            per_view.append(ids.reshape(1, grid_height * grid_width, 3))
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
        """Build ActionDiTYak K/V prefixes directly from external layer states.

        No Ovis condition encoder, VAE, or Yak video block is constructed or
        executed. Prefix modulation uses Yak's clean-condition timestep zero,
        matching the stable context stream in the original ImageWAM policy.
        """

        source_name = str(source_name)
        fusion_type = str(fusion_type)
        if not source_name or not fusion_type:
            raise ValueError("`source_name` and `fusion_type` must be non-empty.")
        total_layers = len(self.blocks)
        if len(layerwise_text_tokens) != total_layers:
            raise ValueError(
                f"{source_name} text/action depth mismatch: "
                f"expected {total_layers}, got {len(layerwise_text_tokens)}."
            )
        if len(layerwise_visual_tokens) != total_layers:
            raise ValueError(
                f"{source_name} visual/action depth mismatch: "
                f"expected {total_layers}, got {len(layerwise_visual_tokens)}."
            )
        reference_text = layerwise_text_tokens[0]
        reference_visual = layerwise_visual_tokens[0]
        if reference_text.ndim != 3 or reference_visual.ndim != 3:
            raise ValueError(
                f"{source_name} text/visual prefixes must both be [B,L,D]."
            )
        batch_size, text_len, hidden_dim = reference_text.shape
        visual_len = reference_visual.shape[1]
        if hidden_dim != self.hidden_dim or reference_visual.shape[-1] != self.hidden_dim:
            raise ValueError(
                f"{source_name} prefixes must use ActionDiTYak hidden={self.hidden_dim}, got "
                f"text={hidden_dim}, visual={reference_visual.shape[-1]}."
            )
        expected_visual_len = int(num_views) * int(visual_grid_height) * int(visual_grid_width)
        if visual_len != expected_visual_len:
            raise ValueError(
                f"{source_name} visual prefix length does not match its view grid: "
                f"tokens={visual_len}, expected={expected_visual_len}."
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
                    f"{source_name} text prefix shape changed at action layer {layer_idx}: "
                    f"{tuple(text.shape)} vs {tuple(reference_text.shape)}."
                )
            if tuple(visual.shape) != tuple(reference_visual.shape):
                raise ValueError(
                    f"{source_name} visual prefix shape changed at action layer {layer_idx}: "
                    f"{tuple(visual.shape)} vs {tuple(reference_visual.shape)}."
                )

        text_ids = torch.zeros(
            batch_size,
            text_len,
            3,
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
        clean_timestep = torch.zeros(
            batch_size,
            device=reference_text.device,
            dtype=reference_text.dtype,
        )
        prefix_t_mod = self.time_in(timestep_embedding(clean_timestep, 256)).to(
            dtype=reference_text.dtype
        )

        double_cache = []
        for layer_idx, block in enumerate(self.double_blocks):
            text = layerwise_text_tokens[layer_idx]
            visual = layerwise_visual_tokens[layer_idx]
            image_mod1, _, image_mod3 = block.img_mod(prefix_t_mod)
            text_mod1, _ = block.txt_mod(prefix_t_mod)
            text_input = (
                (1 + text_mod1.scale) * block.txt_norm1(text) + text_mod1.shift
            )
            text_key, text_value = self._attn_kv(
                block.txt_attn, text_input, text_pe
            )
            visual_norm = block.img_norm1(visual)
            visual_joint_input = (
                (1 + image_mod1.scale) * visual_norm + image_mod1.shift
            )
            visual_key, visual_value = self._attn_kv(
                block.img_attn, visual_joint_input, visual_pe
            )
            visual_self_input = (
                (1 + image_mod3.scale) * visual_norm + image_mod3.shift
            )
            image_key, image_value = self._attn_kv(
                block.img_self_attn, visual_self_input, visual_pe
            )
            double_cache.append(
                {
                    "joint_k": torch.cat((text_key, visual_key), dim=1),
                    "joint_v": torch.cat((text_value, visual_value), dim=1),
                    "image_k": image_key,
                    "image_v": image_value,
                }
            )

        single_cache = []
        double_depth = len(self.double_blocks)
        for single_idx, block in enumerate(self.single_blocks):
            layer_idx = double_depth + single_idx
            prefix = torch.cat(
                (layerwise_text_tokens[layer_idx], layerwise_visual_tokens[layer_idx]),
                dim=1,
            )
            prefix_pe = torch.cat((text_pe, visual_pe), dim=2)
            key, value = self._single_kv(
                block, prefix, prefix_t_mod, prefix_pe
            )
            single_cache.append({"k": key, "v": value})

        return {
            "double": double_cache,
            "single": single_cache,
            "text_mask": text_mask.to(device=reference_text.device, dtype=torch.bool),
            "text_len": text_len,
            "visual_len": visual_len,
            "batch_size": batch_size,
            "fusion_type": fusion_type,
            "source_name": source_name,
        }

    def prepare_qwen_layerwise_prefix(
        self,
        *,
        layerwise_text_tokens: Sequence[torch.Tensor],
        layerwise_visual_tokens: Sequence[torch.Tensor],
        text_mask: torch.Tensor,
        visual_grid_height: int,
        visual_grid_width: int,
        num_views: int,
    ) -> dict[str, Any]:
        """Backward-compatible Qwen wrapper around the generic prefix path."""

        return self.prepare_layerwise_prefix(
            layerwise_text_tokens=layerwise_text_tokens,
            layerwise_visual_tokens=layerwise_visual_tokens,
            text_mask=text_mask,
            visual_grid_height=visual_grid_height,
            visual_grid_width=visual_grid_width,
            num_views=num_views,
            fusion_type="qwen_yak18_layerwise_mot",
            source_name="Qwen",
        )

    def forward_layerwise_prefix_mot(
        self,
        *,
        action_tokens: torch.Tensor,
        timestep: torch.Tensor,
        prefix_cache: dict[str, Any],
        expected_fusion_type: str,
        source_name: str,
    ) -> torch.Tensor:
        """Run ActionDiTYak18 while reading cached external text/visual K/V."""

        if prefix_cache.get("fusion_type") != str(expected_fusion_type):
            raise ValueError(
                f"Expected a {source_name}→ActionDiTYak18 prefix cache with "
                f"fusion_type={expected_fusion_type!r}."
            )
        pre_state = self.pre_dit(action_tokens, timestep)
        action = pre_state["tokens"]
        action_pe = self.pe_embedder(pre_state["ids"])
        batch_size, action_len = action.shape[:2]
        if batch_size != int(prefix_cache["batch_size"]):
            raise ValueError(f"{source_name} prefix/action batch size mismatch.")
        text_len = int(prefix_cache["text_len"])
        visual_len = int(prefix_cache["visual_len"])
        text_valid = prefix_cache["text_mask"]
        action_valid = torch.ones(
            batch_size, action_len, device=action.device, dtype=torch.bool
        )
        visual_valid = torch.ones(
            batch_size, visual_len, device=action.device, dtype=torch.bool
        )
        joint_key_valid = torch.cat(
            (text_valid, visual_valid, action_valid), dim=1
        )
        image_key_valid = torch.cat((visual_valid, action_valid), dim=1)
        joint_mask = joint_key_valid[:, None, :].expand(-1, action_len, -1)
        image_mask = image_key_valid[:, None, :].expand(-1, action_len, -1)

        if len(prefix_cache["double"]) != len(self.double_blocks):
            raise ValueError(f"{source_name} prefix double-block cache depth mismatch.")
        if len(prefix_cache["single"]) != len(self.single_blocks):
            raise ValueError(f"{source_name} prefix single-block cache depth mismatch.")

        for layer_idx, cache in enumerate(prefix_cache["double"]):
            block = self.double_blocks[layer_idx]
            action_mod1, action_mod2, action_mod3 = block.img_mod(pre_state["t_mod"])
            action_norm = block.img_norm1(action)
            joint_input = (
                (1 + action_mod1.scale) * action_norm + action_mod1.shift
            )
            self_input = (
                (1 + action_mod3.scale) * action_norm + action_mod3.shift
            )
            query, key, value = self._qkv(block.img_attn, joint_input, action_pe)
            joint = self._mixed_attention(
                query,
                torch.cat((cache["joint_k"], key), dim=1),
                torch.cat((cache["joint_v"], value), dim=1),
                joint_mask,
            )
            self_query, self_key, self_value = self._qkv(
                block.img_self_attn, self_input, action_pe
            )
            image_self = self._mixed_attention(
                self_query,
                torch.cat((cache["image_k"], self_key), dim=1),
                torch.cat((cache["image_v"], self_value), dim=1),
                image_mask,
            )
            action = action + action_mod1.gate * block.img_attn.proj(joint)
            action = action + action_mod3.gate * block.img_self_attn.proj(image_self)
            action = action + action_mod2.gate * block.img_mlp(
                (1 + action_mod2.scale) * block.img_norm2(action)
                + action_mod2.shift
            )

        for layer_idx, cache in enumerate(prefix_cache["single"]):
            block = self.single_blocks[layer_idx]
            state = self._single_io(
                block, action, pre_state["t_mod"], action_pe
            )
            mixed = self._mixed_attention(
                state["q"],
                torch.cat((cache["k"], state["k"]), dim=1),
                torch.cat((cache["v"], state["v"]), dim=1),
                joint_mask,
            )
            action = self._single_post(block, mixed, state)
        return self.post_dit(action, pre_state)

    def forward_qwen_layerwise_mot(
        self,
        *,
        action_tokens: torch.Tensor,
        timestep: torch.Tensor,
        prefix_cache: dict[str, Any],
    ) -> torch.Tensor:
        """Backward-compatible Qwen wrapper around the generic action path."""

        return self.forward_layerwise_prefix_mot(
            action_tokens=action_tokens,
            timestep=timestep,
            prefix_cache=prefix_cache,
            expected_fusion_type="qwen_yak18_layerwise_mot",
            source_name="Qwen",
        )
