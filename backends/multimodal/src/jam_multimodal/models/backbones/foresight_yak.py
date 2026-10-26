"""Foresight alignment for the Ovis-U1 (Yak) JAMPolicy stack.

Ports the Wan JAM recipe (foresight-only, gen OFF):
  - The policy's CURRENT-observation (cond) token hiddens at a mid-depth
    video-expert block are pulled (cosine) toward a FROZEN pretrained Yak
    expert run on the CLEAN frames at denoise timestep t=0.
  - Offsets: 0 = current frame feature, 1 = future (target) frame feature,
    equal weights, per-target Linear heads on a shared pre-LN transformer
    trunk (Wan-recipe projector, dims scaled 3072->1536 proportionally).
  - Loss = 1 - cos(proj(student), teacher.detach()); teacher detached
    (anti-collapse). Teacher + projector are training-only and discarded at
    inference.

Layer convention mirrors JAMWan: `layer: k` = capture after the k-th video
block (1-indexed) out of double_layers + single_layers total (Yak: 6+12=18;
the Wan recipe's 18/30=60% depth -> 11/18 here).
"""

from __future__ import annotations

import copy
import os
from typing import Any, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from jam_multimodal.utils.logging_config import get_logger

logger = get_logger(__name__)


class _PreLNBlock(nn.Module):
    """Pre-LN transformer block: SDPA self-attention token mixing + GELU FFN.

    Hand-rolled qkv + F.scaled_dot_product_attention (the same op family the
    Yak backbone uses everywhere) instead of nn.MultiheadAttention, whose
    fused paths were not verified on aarch64.
    Same parameter count/shape as MHA (qkv 3d^2 + proj d^2).
    """

    def __init__(self, dim: int, num_heads: int, ffn_dim: int):
        super().__init__()
        self.num_heads = int(num_heads)
        self.head_dim = int(dim) // self.num_heads
        self.norm1 = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, ffn_dim),
            nn.GELU(),
            nn.Linear(ffn_dim, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, dim = x.shape
        h = self.norm1(x)
        q, k, v = (
            self.qkv(h)
            .reshape(batch, seq_len, 3, self.num_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
        )
        attn = F.scaled_dot_product_attention(q, k, v)
        attn = attn.transpose(1, 2).reshape(batch, seq_len, dim)
        x = x + self.proj(attn)
        return x + self.mlp(self.norm2(x))


class ForesightProjector(nn.Module):
    """Foresight projector: in-proj -> shared trunk -> per-target Linear heads.

    Wan JAM recipe: transformer trunk at half model dim (3072->1536), 2
    layers, head_dim 64, shared across targets with light Linear heads.
    Scaled here: 1536 -> 768 trunk, 12 heads (head_dim 64), ffn 4x.
    """

    def __init__(
        self,
        in_dim: int = 1536,
        hidden_dim: int = 768,
        ffn_dim: int = 3072,
        num_layers: int = 2,
        num_heads: int = 12,
        num_targets: int = 2,
        out_dim: int | None = None,
    ):
        super().__init__()
        out_dim = int(out_dim if out_dim is not None else in_dim)
        self.in_proj = nn.Linear(int(in_dim), int(hidden_dim))
        self.blocks = nn.ModuleList(
            [_PreLNBlock(int(hidden_dim), int(num_heads), int(ffn_dim)) for _ in range(int(num_layers))]
        )
        self.heads = nn.ModuleList([nn.Linear(int(hidden_dim), out_dim) for _ in range(int(num_targets))])

    def trunk(self, x: torch.Tensor) -> torch.Tensor:
        x = self.in_proj(x)
        for block in self.blocks:
            x = block(x)
        return x

    def head(self, trunk_out: torch.Tensor, idx: int) -> torch.Tensor:
        return self.heads[idx](trunk_out)


def cosine_align_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Token-wise (1 - cosine) alignment, mean over batch/tokens.

    Caller detaches the teacher target (anti-collapse, as in the Wan recipe).
    """
    return (1.0 - F.cosine_similarity(pred.float(), target.float(), dim=-1)).mean()


class YakForesightTeacher(nn.Module):
    """Frozen copy of the Ovis video expert (Yak). Eval-locked.

    Kept OUT of the model's registered module tree (attached via
    object.__setattr__, mirroring `ovis_condition_encoder`) so checkpoints
    stay teacher-free and the trainer's freeze/optimizer logic never sees it.
    """

    def __init__(self, expert: nn.Module):
        super().__init__()
        self.expert = expert
        self.expert.requires_grad_(False)
        self.expert.eval()

    def train(self, mode: bool = True):  # noqa: ARG002 - eval-locked by design
        return super().train(False)


_TEACHER_MODES = {"frozen_pretrained"}
_TEACHER_TYPES = {"yak"}


def _normalize_teacher_mode(value: Any) -> str:
    mode = str(value if value is not None else "frozen_pretrained").strip().lower()
    if mode not in _TEACHER_MODES:
        raise ValueError(
            f"Unsupported foresight.teacher_mode={value!r}; expected one of "
            f"{sorted(_TEACHER_MODES)}."
        )
    return mode


def attach_ovis_foresight(model, cfg: Dict[str, Any]) -> None:
    """Attach a frozen Yak future teacher and trainable projector to Ovis-JAMPolicy.

    The teacher is a frozen copy of the pretrained Yak video expert.  It is
    intentionally kept out of the registered module tree (no
    checkpoint/optimizer traffic); the projector is registered under
    ``model.mot`` and trained with the policy.
    """
    if not cfg or not bool(cfg.get("enabled", False)):
        return

    # Inference/eval never uses the teacher ("discarded at inference" is the
    # method's contract) — skip the 2.2GB deepcopy when the flag is set.
    # Training must NOT set this flag (training_loss requires the teacher);
    # attach-time construction (not lazy) also keeps resume-correctness: the
    # teacher must be fixed before any policy checkpoint load.
    teacher_type = str(cfg.get("teacher_type", "yak")).strip().lower()
    if teacher_type not in _TEACHER_TYPES:
        raise ValueError(
            f"Unsupported foresight.teacher_type={teacher_type!r}; expected one of "
            f"{sorted(_TEACHER_TYPES)}."
        )
    teacher_mode = _normalize_teacher_mode(cfg.get("teacher_mode", "frozen_pretrained"))
    teacher_hidden_dim = int(model.video_expert.hidden_dim)
    skip_teacher = os.environ.get("JAM_FORESIGHT_NO_TEACHER") == "1"
    if skip_teacher:
        logger.info(
            "Foresight attach: teacher SKIPPED "
            "(inference mode; configured type=%s mode=%s).",
            teacher_type,
            teacher_mode,
        )
    else:
        teacher = YakForesightTeacher(copy.deepcopy(model.video_expert))
        object.__setattr__(model, "foresight_teacher", teacher)

    ref_param = next(model.video_expert.parameters())
    offsets = [int(o) for o in cfg.get("offsets", [0, 1])]
    projector = ForesightProjector(
        in_dim=int(model.video_expert.hidden_dim),
        hidden_dim=int(cfg.get("projector_hidden_dim", 768)),
        ffn_dim=int(cfg.get("projector_ffn_dim", 3072)),
        num_layers=int(cfg.get("projector_num_layers", 2)),
        num_heads=int(cfg.get("projector_num_heads", 12)),
        num_targets=len(offsets),
        out_dim=teacher_hidden_dim,
    ).to(device=ref_param.device, dtype=ref_param.dtype)
    model.mot.foresight_projector = projector

    model.foresight_enabled = True
    model.foresight_teacher_type = teacher_type
    model.foresight_teacher_mode = teacher_mode
    model.foresight_layer = int(cfg.get("layer", 11))
    model.foresight_encoder_layer = int(cfg.get("encoder_layer", cfg.get("layer", 11)))
    model.foresight_teacher_early_exit = bool(cfg.get("teacher_early_exit", True))
    model.foresight_offsets = offsets
    model.foresight_lambda = float(cfg.get("lambda_foresight", 0.6))
    model.foresight_drop_target = bool(cfg.get("drop_target_tokens", True))

    total_blocks = int(model.video_expert.double_layers) + int(model.video_expert.single_layers)
    n_proj = sum(p.numel() for p in projector.parameters())
    logger.info(
        "Foresight attached: layer=%d/%d encoder_layer=%d "
        "teacher_type=%s teacher_early_exit=%s offsets=%s lambda=%.3f "
        "projector=%.1fM teacher_mode=%s teacher_hidden=%d",
        model.foresight_layer,
        total_blocks,
        model.foresight_encoder_layer,
        model.foresight_teacher_type,
        model.foresight_teacher_early_exit,
        model.foresight_offsets,
        model.foresight_lambda,
        n_proj / 1e6,
        model.foresight_teacher_mode,
        teacher_hidden_dim,
    )
