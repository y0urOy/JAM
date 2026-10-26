"""Training-only foresight supervision for JAMWan.

The foresight target of a sample is its recorded future in the frozen
pretrained VAE latent space ("VAE-direct"): the clean latents of the future
frames are cut into the spatial patches of Wan's patch embedding and lifted by
a fixed isometric map to the video hidden width. A lightweight shared
projector maps the policy's mid-depth observation-token hidden states to each
future frame's target, pulled toward it with a cosine loss. No teacher network
runs, and the projector is discarded at inference, so the deployed model and
its latency are unchanged.
"""

from __future__ import annotations

from typing import Optional, Sequence

import torch
import torch.nn as nn

from .wan_video_dit import flash_attention


class ForesightBlock(nn.Module):
    """Pre-LN transformer block (token-mixing self-attention + MLP)."""

    def __init__(self, hidden_dim: int, num_heads: int, ffn_dim: int, eps: float = 1e-6):
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError(
                f"`hidden_dim` must be divisible by `num_heads`, got {hidden_dim} and {num_heads}"
            )
        self.num_heads = num_heads
        self.norm1 = nn.LayerNorm(hidden_dim, eps=eps)
        self.q = nn.Linear(hidden_dim, hidden_dim)
        self.k = nn.Linear(hidden_dim, hidden_dim)
        self.v = nn.Linear(hidden_dim, hidden_dim)
        self.o = nn.Linear(hidden_dim, hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim, eps=eps)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, ffn_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(ffn_dim, hidden_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        attn = flash_attention(q=self.q(h), k=self.k(h), v=self.v(h), num_heads=self.num_heads)
        x = x + self.o(attn)
        x = x + self.ffn(self.norm2(x))
        return x


class SharedForesightProjector(nn.Module):
    """One shared transformer backbone + lightweight per-target linear heads.

    Each (student layer, future offset) target gets its own Linear head on top
    of one shared token-mixing trunk. `keys` are "{layer}_{offset}".
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        keys: Sequence[str],
        hidden_dim: Optional[int] = None,
        num_layers: int = 3,
        num_heads: int = 24,
        ffn_dim: Optional[int] = None,
        eps: float = 1e-6,
    ):
        super().__init__()
        hidden_dim = int(hidden_dim) if hidden_dim is not None else int(in_dim)
        ffn_dim = int(ffn_dim) if ffn_dim is not None else hidden_dim * 4
        self.in_proj = nn.Identity() if hidden_dim == int(in_dim) else nn.Linear(int(in_dim), hidden_dim)
        self.blocks = nn.ModuleList(
            [ForesightBlock(hidden_dim, int(num_heads), ffn_dim, eps=eps) for _ in range(int(num_layers))]
        )
        self.norm = nn.LayerNorm(hidden_dim, eps=eps)
        self.heads = nn.ModuleDict({k: nn.Linear(hidden_dim, int(out_dim)) for k in keys})

    def backbone(self, x: torch.Tensor) -> torch.Tensor:
        """Shared token-mixing trunk; run once per student layer, reuse across its frame heads."""
        x = self.in_proj(x)
        for block in self.blocks:
            x = block(x)
        return self.norm(x)

    def head(self, feat: torch.Tensor, key: str) -> torch.Tensor:
        return self.heads[key](feat)


FORESIGHT_TEACHER_MODES = frozenset({"frozen_pretrained"})

FORESIGHT_TARGET_TYPES = frozenset({"vae_direct"})


def normalize_foresight_target_type(target_type: Optional[str]) -> str:
    """Return the canonical foresight target representation.

    ``vae_direct`` (deterministic spatial patches of the frozen pretrained VAE
    latent) is the only target this release implements.
    """
    canonical = "" if target_type is None else str(target_type).strip().lower()
    aliases = {
        "vae": "vae_direct",
        "latent": "vae_direct",
    }
    canonical = aliases.get(canonical, canonical)
    if canonical not in FORESIGHT_TARGET_TYPES:
        raise ValueError(
            f"Unsupported `foresight.target_type` {target_type!r}; expected one of "
            f"{sorted(FORESIGHT_TARGET_TYPES)}."
        )
    return canonical


def normalize_foresight_teacher_mode(mode: Optional[str]) -> str:
    """Return the canonical target-encoder mode used by Wan foresight."""
    canonical = "frozen_pretrained" if mode is None else str(mode).strip().lower()
    aliases = {
        "pretrained": "frozen_pretrained",
        "fixed": "frozen_pretrained",
    }
    canonical = aliases.get(canonical, canonical)
    if canonical not in FORESIGHT_TEACHER_MODES:
        raise ValueError(
            f"Unsupported `foresight.teacher_mode` {mode!r}; expected one of "
            f"{sorted(FORESIGHT_TEACHER_MODES)}."
        )
    return canonical


@torch.no_grad()
def patch_foresight_vae_latents(
    clean_latents: torch.Tensor,
    *,
    patch_size: Sequence[int],
) -> tuple[torch.Tensor, int]:
    """Patch clean VAE latents in the exact order used by Wan's Conv3d.

    Latent patches follow the exact Conv3d kernel order ``C, pt, ph, pw`` used
    by Wan's patch embedding. The raw target is 48*1*2*2=192-D for the current
    model. Keeping this operation separate from the fixed 192->3072 lift lets
    the cache store the compact raw representation without changing the loss.
    """
    if clean_latents.ndim != 5:
        raise ValueError(
            f"`clean_latents` must be [B, C, F, H, W], got {tuple(clean_latents.shape)}."
        )
    if len(patch_size) != 3:
        raise ValueError(f"`patch_size` must contain (pt, ph, pw), got {patch_size}.")
    pt, ph, pw = (int(value) for value in patch_size)
    if min(pt, ph, pw) <= 0:
        raise ValueError(f"Patch dimensions must be positive, got {(pt, ph, pw)}.")

    batch, channels, frames, height, width = clean_latents.shape
    if frames % pt or height % ph or width % pw:
        raise ValueError(
            "VAE latent shape must be divisible by the target patch size: "
            f"shape={(frames, height, width)} patch={(pt, ph, pw)}."
        )

    # unfold appends each kernel dimension at the end, producing
    # [B,C,F',H',W',pt,ph,pw].  Permuting places tokens in Wan's F/H/W order
    # and values in Conv3d's C/pt/ph/pw order.
    patches = (
        clean_latents.unfold(2, pt, pt)
        .unfold(3, ph, ph)
        .unfold(4, pw, pw)
    )
    target_frames = frames // pt
    target_height = height // ph
    target_width = width // pw
    raw_dim = channels * pt * ph * pw
    patches = patches.permute(0, 2, 3, 4, 1, 5, 6, 7).reshape(
        batch, target_frames * target_height * target_width, raw_dim
    ).contiguous()

    return patches, int(target_height * target_width)


@torch.no_grad()
def lift_foresight_vae_patches(
    patches: torch.Tensor,
    *,
    output_dim: int,
) -> torch.Tensor:
    """Apply the deterministic isometric raw-VAE-patch feature lift.

    Repeating each raw vector and scaling by ``1/sqrt(repeats)`` exactly
    preserves target norms, dot products and cosine geometry. It contains no
    learned or random parameters and is therefore safe to apply after reading
    compact raw patches from the target cache.
    """
    if patches.ndim != 3:
        raise ValueError(
            f"`patches` must be [B, tokens, raw_dim], got {tuple(patches.shape)}."
        )

    raw_dim = int(patches.shape[-1])
    output_dim = int(output_dim)
    if output_dim < raw_dim or output_dim % raw_dim != 0:
        raise ValueError(
            "`output_dim` must be an integer multiple of the raw VAE patch dim "
            f"{raw_dim}, got {output_dim}."
        )
    repeats = output_dim // raw_dim
    if repeats > 1:
        patches = patches.repeat(1, 1, repeats) / (repeats ** 0.5)
    return patches


@torch.no_grad()
def encode_foresight_vae_direct(
    clean_latents: torch.Tensor,
    *,
    patch_size: Sequence[int],
    output_dim: int,
) -> tuple[torch.Tensor, int]:
    """Turn clean pretrained-VAE latents into deterministic spatial targets.

    This is the online VAE-direct path. It is deliberately composed from the
    compact raw patching operation and the fixed isometric lift so the online
    and cached paths share exactly the same implementation.
    """
    patches, tokens_per_frame = patch_foresight_vae_latents(
        clean_latents,
        patch_size=patch_size,
    )
    return (
        lift_foresight_vae_patches(patches, output_dim=output_dim),
        tokens_per_frame,
    )


def per_frame_targets(
    hidden: torch.Tensor,
    tokens_per_frame: int,
    frame_valid: Optional[torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-frame target maps, preserving temporal structure.

    Returns:
        targets: [B, num_frames, tokens_per_frame, D] — one spatial target map per latent frame.
        valid:   [B, num_frames] bool — per-frame validity (False for padded frames).
    """
    batch_size, seq_len, dim = hidden.shape
    num_frames = seq_len // tokens_per_frame
    targets = hidden.view(batch_size, num_frames, tokens_per_frame, dim)
    if frame_valid is None:
        valid = torch.ones((batch_size, num_frames), dtype=torch.bool, device=hidden.device)
    else:
        valid = frame_valid
    return targets, valid
