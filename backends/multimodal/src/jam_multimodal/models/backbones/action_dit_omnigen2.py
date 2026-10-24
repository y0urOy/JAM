from __future__ import annotations

from typing import Any, Dict

import torch
import torch.nn as nn
from diffusers.models.embeddings import Timesteps, get_1d_rotary_pos_embed

from jam_multimodal.utils.logging_config import get_logger

from omnigen2.models.embeddings import TimestepEmbedding
from omnigen2.models.transformers.block_lumina2 import LuminaLayerNormContinuous
from omnigen2.models.transformers.transformer_omnigen2 import OmniGen2TransformerBlock

logger = get_logger(__name__)


class ActionDiTOmnigen2(nn.Module):
    """Action expert using OmniGen2-style transformer blocks."""

    block_protocol = "omnigen2"

    def __init__(
        self,
        action_dim: int,
        residual_dim: int = 1024,
        num_heads: int = 24,
        num_kv_heads: int = 8,
        attn_head_dim: int = 96,
        num_layers: int = 26,
        multiple_of: int = 256,
        ffn_dim_multiplier: float | None = None,
        norm_eps: float = 1e-5,
        frequency_embedding_size: int = 256,
        timestep_scale: float = 1.0,
        max_action_horizon: int = 64,
        rope_theta: float = 10000.0,
        use_gradient_checkpointing: bool = False,
    ) -> None:
        super().__init__()
        self.hidden_dim = int(residual_dim)
        self.action_dim = int(action_dim)
        self.num_heads = int(num_heads)
        self.num_kv_heads = int(num_kv_heads)
        self.attn_head_dim = int(attn_head_dim)
        self.cond_dim = min(self.hidden_dim, 1024)
        self.use_gradient_checkpointing = bool(use_gradient_checkpointing)

        self.action_encoder = nn.Linear(self.action_dim, self.hidden_dim)
        self.time_proj = Timesteps(
            num_channels=frequency_embedding_size,
            flip_sin_to_cos=True,
            downscale_freq_shift=0.0,
            scale=timestep_scale,
        )
        self.timestep_embedder = TimestepEmbedding(
            in_channels=frequency_embedding_size,
            time_embed_dim=self.cond_dim,
        )

        self.blocks = nn.ModuleList(
            [
                OmniGen2TransformerBlock(
                    dim=self.hidden_dim,
                    num_attention_heads=self.num_heads,
                    num_kv_heads=self.num_kv_heads,
                    multiple_of=multiple_of,
                    ffn_dim_multiplier=ffn_dim_multiplier,
                    norm_eps=norm_eps,
                    modulation=True,
                    attn_head_dim=self.attn_head_dim,
                )
                for _ in range(num_layers)
            ]
        )
        self.head = LuminaLayerNormContinuous(
            embedding_dim=self.hidden_dim,
            conditioning_embedding_dim=self.cond_dim,
            elementwise_affine=False,
            eps=1e-6,
            out_dim=self.action_dim,
        )

        freqs_cis = get_1d_rotary_pos_embed(
            self.attn_head_dim,
            max_action_horizon,
            theta=rope_theta,
            freqs_dtype=torch.float64,
        )
        # Keep complex RoPE as a plain attribute. Registering it as a buffer
        # would make module.to(dtype=bf16) cast complex values to real and drop
        # the imaginary component.
        self.action_freqs_cis = freqs_cis

    @classmethod
    def from_pretrained(
        cls,
        action_dit_config: dict[str, Any],
        action_dit_pretrained_path: str | None = None,
        skip_dit_load_from_pretrain: bool = False,
        device: str = "cuda",
        torch_dtype: torch.dtype = torch.bfloat16,
    ) -> "ActionDiTOmnigen2":
        if action_dit_config is None:
            raise ValueError("`action_dit_config` is required for ActionDiTOmnigen2.from_pretrained().")
        model = cls(**action_dit_config).to(device=device, dtype=torch_dtype)
        if skip_dit_load_from_pretrain or not action_dit_pretrained_path:
            logger.info("Initializing ActionDiTOmnigen2 without pretrained action weights.")
            return model

        payload = torch.load(action_dit_pretrained_path, map_location="cpu")
        state_dict = payload.get("state_dict", payload) if isinstance(payload, dict) else payload
        if not isinstance(state_dict, dict):
            raise ValueError(f"Invalid ActionDiTOmnigen2 checkpoint type: {type(payload)}")
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            logger.warning("ActionDiTOmnigen2 missing keys when loading: %s", missing[:20])
        if unexpected:
            logger.warning("ActionDiTOmnigen2 unexpected keys when loading: %s", unexpected[:20])
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
        if timestep.ndim != 1:
            raise ValueError(f"`timestep` must be [B], got {tuple(timestep.shape)}")
        if timestep.shape[0] not in (1, batch_size):
            raise ValueError(f"`timestep` length must be 1 or batch size {batch_size}, got {timestep.shape[0]}")
        if timestep.shape[0] == 1 and batch_size > 1:
            timestep = timestep.expand(batch_size)
        if seq_len > self.action_freqs_cis.shape[0]:
            raise ValueError(f"Action length {seq_len} exceeds RoPE cache {self.action_freqs_cis.shape[0]}")

        tokens = self.action_encoder(action_tokens)
        time_emb = self.time_proj(timestep).to(device=tokens.device, dtype=tokens.dtype)
        temb = self.timestep_embedder(time_emb)
        freqs = self.action_freqs_cis[:seq_len].unsqueeze(0).expand(batch_size, seq_len, -1).to(tokens.device)
        return {
            "tokens": tokens,
            "freqs": freqs,
            "t_mod": temb,
            "context": None,
            "context_mask": None,
            "meta": {"batch_size": batch_size, "seq_len": seq_len},
        }

    def post_dit(self, tokens: torch.Tensor, pre_state: Dict[str, Any]) -> torch.Tensor:
        return self.head(tokens, pre_state["t_mod"])

    def forward(self, action_tokens: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        pre_state = self.pre_dit(action_tokens=action_tokens, timestep=timestep)
        hidden_states = pre_state["tokens"]
        for block in self.blocks:
            hidden_states = block(hidden_states, torch.ones(hidden_states.shape[:2], dtype=torch.bool, device=hidden_states.device), pre_state["freqs"], pre_state["t_mod"])
        return self.post_dit(hidden_states, pre_state)
