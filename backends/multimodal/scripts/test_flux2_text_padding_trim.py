#!/usr/bin/env python3
"""CPU checks for the exact cached-text right-padding optimization."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from jam_multimodal.models.backbones.jam import JAMPolicy


def _policy(trim: bool) -> JAMPolicy:
    policy = JAMPolicy.__new__(JAMPolicy)
    torch.nn.Module.__init__(policy)
    policy.device = torch.device("cpu")
    policy.torch_dtype = torch.float32
    policy.trim_cached_text_padding = trim
    return policy


def main() -> None:
    generator = torch.Generator().manual_seed(42)
    context = torch.randn(3, 8, 16, generator=generator)
    lengths = torch.tensor([5, 3, 7])
    positions = torch.arange(8).unsqueeze(0)
    mask = positions < lengths.unsqueeze(1)
    sample = {
        "text_hidden_states": context.clone(),
        "text_attention_mask": mask.clone(),
    }

    full_context, full_mask = _policy(False)._encode_flux2_text(sample)
    trim_context, trim_mask = _policy(True)._encode_flux2_text(sample)
    assert trim_context.shape == (3, 7, 16)
    assert trim_mask.shape == (3, 7)
    torch.testing.assert_close(trim_context, full_context[:, :7], rtol=0, atol=0)
    assert torch.equal(trim_mask, full_mask[:, :7])

    # Masked padding removal must preserve every valid-query attention output.
    q = torch.randn(3, 2, 4, 8, generator=generator)
    k = torch.randn(3, 2, 8, 8, generator=generator)
    v = torch.randn(3, 2, 8, 8, generator=generator)
    full_attn_mask = full_mask[:, None, None, :].expand(3, 1, 4, 8)
    trim_attn_mask = trim_mask[:, None, None, :].expand(3, 1, 4, 7)
    full_out = F.scaled_dot_product_attention(q, k, v, attn_mask=full_attn_mask)
    trim_out = F.scaled_dot_product_attention(
        q,
        k[:, :, :7],
        v[:, :, :7],
        attn_mask=trim_attn_mask,
    )
    torch.testing.assert_close(trim_out, full_out, rtol=1e-6, atol=1e-6)

    invalid_mask = mask.clone()
    invalid_mask[0, 2] = False
    try:
        _policy(True)._encode_flux2_text(
            {
                "text_hidden_states": context,
                "text_attention_mask": invalid_mask,
            }
        )
    except ValueError as exc:
        assert "right-padded" in str(exc)
    else:
        raise AssertionError("hole-padded masks must fail closed")

    print("FLUX2 TEXT PADDING TRIM EQUIVALENCE PASSED")


if __name__ == "__main__":
    main()
