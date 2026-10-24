from __future__ import annotations

from typing import Any, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.models.embeddings import Timesteps

from jam_multimodal.utils.logging_config import get_logger

logger = get_logger(__name__)


class _ActionSanaBlock(nn.Module):
    """SANA-shaped action block with a smaller residual stream.

    The residual/action hidden size can be smaller than SANA's image hidden size,
    while the self-attention Q/K/V space matches SANA's linear-attention geometry.
    """

    def __init__(
        self,
        residual_dim: int,
        attn_dim: int,
        num_heads: int,
        context_dim: int,
        mlp_ratio: float = 2.5,
        qk_norm: bool = True,
    ) -> None:
        super().__init__()
        self.residual_dim = int(residual_dim)
        self.attn_dim = int(attn_dim)
        self.num_heads = int(num_heads)
        self.head_dim = self.attn_dim // self.num_heads
        if self.attn_dim % self.num_heads != 0:
            raise ValueError(f"attn_dim={attn_dim} must be divisible by num_heads={num_heads}.")

        self.norm1 = nn.LayerNorm(self.residual_dim, elementwise_affine=False, eps=1e-6)
        self.qkv = nn.Linear(self.residual_dim, 3 * self.attn_dim, bias=False)
        self.q_norm = nn.LayerNorm(self.head_dim, elementwise_affine=False, eps=1e-6) if qk_norm else nn.Identity()
        self.k_norm = nn.LayerNorm(self.head_dim, elementwise_affine=False, eps=1e-6) if qk_norm else nn.Identity()
        self.proj = nn.Linear(self.attn_dim, self.residual_dim, bias=False)

        self.cross_q = nn.Linear(self.residual_dim, self.attn_dim)
        self.cross_kv = nn.Linear(int(context_dim), 2 * self.attn_dim)
        self.cross_q_norm = nn.LayerNorm(self.head_dim, elementwise_affine=False, eps=1e-6) if qk_norm else nn.Identity()
        self.cross_k_norm = nn.LayerNorm(self.head_dim, elementwise_affine=False, eps=1e-6) if qk_norm else nn.Identity()
        self.cross_proj = nn.Linear(self.attn_dim, self.residual_dim)

        hidden = int(self.residual_dim * float(mlp_ratio))
        self.norm2 = nn.LayerNorm(self.residual_dim, elementwise_affine=False, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(self.residual_dim, hidden, bias=False),
            nn.SiLU(),
            nn.Linear(hidden, self.residual_dim, bias=False),
        )
        self.scale_shift_table = nn.Parameter(torch.randn(6, self.residual_dim) / self.residual_dim**0.5)

    @staticmethod
    def _modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        return x * (1 + scale) + shift

    def split_modulation(self, t_mod: torch.Tensor) -> tuple[torch.Tensor, ...]:
        if t_mod.ndim != 2 or t_mod.shape[-1] != 6 * self.residual_dim:
            raise ValueError(
                f"`t_mod` must be [B,{6 * self.residual_dim}], got {tuple(t_mod.shape)}"
            )
        return (self.scale_shift_table[None].to(t_mod.dtype) + t_mod.reshape(t_mod.shape[0], 6, -1)).chunk(6, dim=1)

    def build_self_attention_io(self, x: torch.Tensor, t_mod: torch.Tensor) -> Dict[str, torch.Tensor]:
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.split_modulation(t_mod)
        attn_in = self._modulate(self.norm1(x), shift_msa, scale_msa)
        batch_size, seq_len = x.shape[:2]
        qkv = self.qkv(attn_in).view(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q = self.q_norm(q)
        k = self.k_norm(k)
        return {
            "q": q.transpose(1, 2).transpose(-1, -2),  # [B,H,D,N]
            "k": k.transpose(1, 2).transpose(-1, -2),
            "v": v.transpose(1, 2).transpose(-1, -2),
            "residual_x": x,
            "gate_msa": gate_msa,
            "shift_mlp": shift_mlp,
            "scale_mlp": scale_mlp,
            "gate_mlp": gate_mlp,
        }

    def linear_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        eps: float = 1e-8,
    ) -> torch.Tensor:
        q = F.relu(q)
        k = F.relu(k)
        v_pad = F.pad(v, (0, 0, 0, 1), mode="constant", value=1.0)
        vk = torch.matmul(v_pad, k.transpose(-1, -2))
        out = torch.matmul(vk, q)
        if out.dtype in (torch.float16, torch.bfloat16):
            out = out.float()
        out = out[:, :, :-1] / (out[:, :, -1:] + eps)
        return out.to(dtype=v.dtype).reshape(q.shape[0], self.attn_dim, q.shape[-1]).transpose(1, 2)

    def cross_attention(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        if context.ndim != 3:
            raise ValueError(f"`context` must be [B,L,D], got {tuple(context.shape)}")
        batch_size, seq_len = x.shape[:2]
        q = self.cross_q(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        kv = self.cross_kv(context).view(batch_size, context.shape[1], 2, self.num_heads, self.head_dim)
        k, v = kv.unbind(dim=2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        q = self.cross_q_norm(q)
        k = self.cross_k_norm(k)
        attn_mask = None
        if context_mask is not None:
            attn_mask = context_mask.to(device=x.device, dtype=torch.bool)[:, None, None, :]
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        out = out.transpose(1, 2).reshape(batch_size, seq_len, self.attn_dim)
        return self.cross_proj(out)

    def apply_post(
        self,
        attn_out: torch.Tensor,
        state: Dict[str, torch.Tensor],
        context: torch.Tensor | None,
        context_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        x = state["residual_x"] + state["gate_msa"] * self.proj(attn_out)
        if context is not None:
            x = x + self.cross_attention(x, context, context_mask)
        mlp_input = self._modulate(self.norm2(x), state["shift_mlp"], state["scale_mlp"])
        return x + state["gate_mlp"] * self.mlp(mlp_input)


class ActionDiTSana(nn.Module):
    """Action expert aligned to DIM/SANA linear-attention space."""

    block_protocol = "sana"

    def __init__(
        self,
        action_dim: int,
        residual_dim: int = 1536,
        attn_dim: int = 2240,
        num_heads: int = 70,
        num_layers: int = 20,
        context_dim: int = 2240,
        mlp_ratio: float = 2.5,
        qk_norm: bool = True,
        frequency_embedding_size: int = 256,
        timestep_scale: float = 1.0,
        use_gradient_checkpointing: bool = False,
    ) -> None:
        super().__init__()
        self.action_dim = int(action_dim)
        self.hidden_dim = int(residual_dim)
        self.attn_dim = int(attn_dim)
        self.num_heads = int(num_heads)
        self.num_kv_heads = self.num_heads
        self.attn_head_dim = self.attn_dim // self.num_heads
        self.context_dim = int(context_dim)
        self.use_gradient_checkpointing = bool(use_gradient_checkpointing)

        self.action_encoder = nn.Linear(self.action_dim, self.hidden_dim)
        self.time_proj = Timesteps(
            num_channels=frequency_embedding_size,
            flip_sin_to_cos=True,
            downscale_freq_shift=0.0,
            scale=timestep_scale,
        )
        self.timestep_embedder = nn.Sequential(
            nn.Linear(frequency_embedding_size, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, 6 * self.hidden_dim),
        )
        self.blocks = nn.ModuleList(
            [
                _ActionSanaBlock(
                    residual_dim=self.hidden_dim,
                    attn_dim=self.attn_dim,
                    num_heads=self.num_heads,
                    context_dim=self.context_dim,
                    mlp_ratio=mlp_ratio,
                    qk_norm=qk_norm,
                )
                for _ in range(int(num_layers))
            ]
        )
        self.head_norm = nn.LayerNorm(self.hidden_dim, elementwise_affine=False, eps=1e-6)
        self.head = nn.Linear(self.hidden_dim, self.action_dim)

    @classmethod
    def from_pretrained(
        cls,
        action_dit_config: dict[str, Any],
        action_dit_pretrained_path: str | None = None,
        skip_dit_load_from_pretrain: bool = False,
        device: str = "cuda",
        torch_dtype: torch.dtype = torch.bfloat16,
    ) -> "ActionDiTSana":
        if action_dit_config is None:
            raise ValueError("`action_dit_config` is required for ActionDiTSana.from_pretrained().")
        model = cls(**action_dit_config).to(device=device, dtype=torch_dtype)
        if skip_dit_load_from_pretrain or not action_dit_pretrained_path:
            logger.info("Initializing ActionDiTSana without pretrained action weights.")
            return model
        payload = torch.load(action_dit_pretrained_path, map_location="cpu")
        state_dict = payload.get("state_dict", payload) if isinstance(payload, dict) else payload
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            logger.warning("ActionDiTSana missing keys when loading: %s", missing[:20])
        if unexpected:
            logger.warning("ActionDiTSana unexpected keys when loading: %s", unexpected[:20])
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
        if timestep.shape[0] == 1 and batch_size > 1:
            timestep = timestep.expand(batch_size)
        if timestep.shape[0] != batch_size:
            raise ValueError(f"`timestep` length must match batch size {batch_size}, got {timestep.shape[0]}")
        tokens = self.action_encoder(action_tokens)
        time_emb = self.time_proj(timestep).to(device=tokens.device, dtype=tokens.dtype)
        t_mod = self.timestep_embedder(time_emb)
        return {
            "tokens": tokens,
            "t_mod": t_mod,
            "context": context,
            "context_mask": context_mask,
            "meta": {"batch_size": batch_size, "seq_len": seq_len},
        }

    def post_dit(self, tokens: torch.Tensor, pre_state: Dict[str, Any]) -> torch.Tensor:
        return self.head(self.head_norm(tokens))

    def forward(self, action_tokens: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        pre = self.pre_dit(action_tokens, timestep)
        x = pre["tokens"]
        for block in self.blocks:
            state = block.build_self_attention_io(x, pre["t_mod"])
            attn = block.linear_attention(state["q"], state["k"], state["v"])
            x = block.apply_post(attn, state, None, None)
        return self.post_dit(x, pre)
