"""Exact equal-grid batching for Qwen3.5 vision self-attention.

Transformers' SDPA implementation splits a flattened vision sequence and
launches one attention call per image.  JAMPolicy fixes every camera image to
the same square resolution, so the independent image sequences can instead be
represented by the batch dimension of one SDPA call.  This changes execution
only: images remain isolated and the module parameters/state dict are
untouched.

The helper is deliberately opt-in.  Call :func:`enable_equal_grid_attention`
after loading a Qwen3.5 backbone.  Unsupported or non-uniform inputs fail
loudly rather than silently changing attention semantics.
"""

from __future__ import annotations

from types import MethodType
from typing import Any

import torch


def _equal_grid_attention_forward(
    self,
    hidden_states: torch.Tensor,
    cu_seqlens: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
    **kwargs: Any,
) -> torch.Tensor:
    """Batch equal-length image sequences without cross-image attention."""

    from transformers.models.qwen3_5.modeling_qwen3_5 import (
        ALL_ATTENTION_FUNCTIONS,
        apply_rotary_pos_emb_vision,
        eager_attention_forward,
    )

    seq_length = int(hidden_states.shape[0])
    num_images = int(cu_seqlens.numel()) - 1
    expected_tokens = int(self._imagewam_equal_grid_tokens)
    if num_images <= 0 or seq_length != num_images * expected_tokens:
        raise ValueError(
            "Equal-grid Qwen vision attention received an incompatible layout: "
            f"sequence={seq_length}, images={num_images}, "
            f"tokens_per_image={expected_tokens}."
        )
    if position_embeddings is None:
        raise ValueError("Qwen vision attention requires rotary position embeddings.")

    query_states, key_states, value_states = (
        self.qkv(hidden_states)
        .reshape(seq_length, 3, self.num_heads, self.head_dim)
        .permute(1, 0, 2, 3)
        .unbind(0)
    )
    cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb_vision(
        query_states, key_states, cos, sin
    )

    # [images * tokens, heads, dim] -> [images, heads, tokens, dim].
    # The image index is the SDPA batch dimension, so there is no cross-image
    # attention and the result is mathematically identical to the upstream
    # per-image Python loop.
    def as_attention_batch(states: torch.Tensor) -> torch.Tensor:
        return states.reshape(
            num_images, expected_tokens, self.num_heads, self.head_dim
        ).transpose(1, 2)

    query_states = as_attention_batch(query_states)
    key_states = as_attention_batch(key_states)
    value_states = as_attention_batch(value_states)

    attention_interface = ALL_ATTENTION_FUNCTIONS.get_interface(
        self.config._attn_implementation, eager_attention_forward
    )
    attn_output, _ = attention_interface(
        self,
        query_states,
        key_states,
        value_states,
        attention_mask=None,
        scaling=self.scaling,
        dropout=0.0 if not self.training else self.attention_dropout,
        is_causal=False,
        **kwargs,
    )
    attn_output = attn_output.reshape(seq_length, self.dim).contiguous()
    return self.proj(attn_output)


def enable_equal_grid_attention(backbone, *, image_size: int) -> int:
    """Enable exact batched vision attention on a loaded Qwen3.5 backbone.

    Returns the number of patched vision blocks.  The fixed token count is
    derived from the real Qwen patch size; spatial merging occurs after these
    attention blocks and therefore is intentionally not part of this count.
    """

    visual = getattr(backbone, "visual", None)
    if visual is None or not hasattr(visual, "blocks"):
        raise TypeError("Expected a Qwen3.5 backbone with `visual.blocks`.")
    config = visual.config
    if str(getattr(config, "_attn_implementation", "")) != "sdpa":
        raise ValueError(
            "Equal-grid batching is currently validated only for Qwen SDPA; "
            f"got {getattr(config, '_attn_implementation', None)!r}."
        )
    image_size = int(image_size)
    patch_size = int(config.patch_size)
    if image_size <= 0 or image_size % patch_size != 0:
        raise ValueError(
            f"image_size={image_size} must be divisible by patch_size={patch_size}."
        )
    expected_tokens = (image_size // patch_size) ** 2

    patched = 0
    for block in visual.blocks:
        attention = block.attn
        attention._imagewam_equal_grid_tokens = expected_tokens
        attention.forward = MethodType(_equal_grid_attention_forward, attention)
        patched += 1
    if patched <= 0:
        raise RuntimeError("Qwen vision tower contains no attention blocks.")
    return patched

