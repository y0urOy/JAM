"""Foresight alignment for the FLUX.2 JAMPolicy stack.

The training-only objective mirrors the Ovis recipe: generation loss off, a
future-blind student, and a frozen pretrained image expert that supplies
clean future-frame features.  Layer indices are 1-based over the FLUX.2
double-stream and single-stream blocks.
"""

from __future__ import annotations

import copy
import os
from typing import Any, Dict

import torch
import torch.nn as nn

from jam_multimodal.utils.logging_config import get_logger

from .foresight_yak import ForesightProjector

logger = get_logger(__name__)


class Flux2ForesightTeacher(nn.Module):
    """Frozen, eval-locked FLUX.2 video expert used only during training."""

    def __init__(self, expert: nn.Module):
        super().__init__()
        self.expert = expert
        self.expert.requires_grad_(False)
        self.expert.eval()

    def train(self, mode: bool = True):  # noqa: ARG002 - eval-locked by design
        return super().train(False)


def attach_flux2_foresight(model, cfg: Dict[str, Any]) -> None:
    """Attach the frozen FLUX.2 teacher and trainable foresight projector.

    The teacher is deliberately attached outside the registered module tree,
    so it is absent from optimizer state and policy checkpoints.  The
    projector is registered under ``model.mot`` and is therefore handled by
    the existing DiT-only trainer policy.
    """
    if not cfg or not bool(cfg.get("enabled", False)):
        return
    if str(getattr(model, "stack", "")) != "flux2":
        raise ValueError("FLUX.2 foresight can only be attached to stack='flux2'.")

    teacher_mode = str(cfg.get("teacher_mode", "frozen_pretrained")).strip().lower()
    if teacher_mode != "frozen_pretrained":
        raise ValueError(
            "FLUX.2 foresight currently supports teacher_mode='frozen_pretrained' only; "
            f"got {teacher_mode!r}."
        )

    total_blocks = int(model.video_expert.double_layers) + int(model.video_expert.single_layers)
    layer = int(cfg.get("layer", round(total_blocks * 0.6)))
    encoder_layer = int(cfg.get("encoder_layer", layer))
    for name, value in (("layer", layer), ("encoder_layer", encoder_layer)):
        if value < 1 or value > total_blocks:
            raise ValueError(f"foresight.{name} must be in [1, {total_blocks}], got {value}.")

    offsets = [int(offset) for offset in cfg.get("offsets", [1])]
    if not offsets:
        raise ValueError("foresight.offsets must not be empty.")
    if len(set(offsets)) != len(offsets) or any(offset not in {0, 1} for offset in offsets):
        raise ValueError(
            "FLUX.2 image-pair foresight offsets must be unique values from {0, 1} "
            f"(0=current, 1=future), got {offsets}."
        )

    # Evaluation disables foresight before model construction.  Keep the
    # environment escape hatch as a second guard for direct/custom evaluators.
    if os.environ.get("JAM_FORESIGHT_NO_TEACHER") == "1":
        logger.info("FLUX.2 foresight teacher skipped (inference-only mode).")
    else:
        teacher = Flux2ForesightTeacher(copy.deepcopy(model.video_expert))
        object.__setattr__(model, "foresight_teacher", teacher)

    ref_param = next(model.video_expert.parameters())
    in_dim = int(model.video_expert.hidden_dim)
    hidden_dim = int(cfg.get("projector_hidden_dim", in_dim // 2))
    num_heads = int(cfg.get("projector_num_heads", max(1, hidden_dim // 64)))
    if hidden_dim % num_heads != 0:
        raise ValueError(
            "foresight.projector_hidden_dim must be divisible by "
            f"projector_num_heads, got {hidden_dim} and {num_heads}."
        )
    projector = ForesightProjector(
        in_dim=in_dim,
        hidden_dim=hidden_dim,
        ffn_dim=int(cfg.get("projector_ffn_dim", hidden_dim * 4)),
        num_layers=int(cfg.get("projector_num_layers", 2)),
        num_heads=num_heads,
        num_targets=len(offsets),
        out_dim=in_dim,
    ).to(device=ref_param.device, dtype=ref_param.dtype)
    model.mot.foresight_projector = projector

    model.foresight_enabled = True
    model.foresight_teacher_mode = teacher_mode
    model.foresight_layer = layer
    model.foresight_encoder_layer = encoder_layer
    model.foresight_teacher_early_exit = bool(cfg.get("teacher_early_exit", True))
    model.foresight_offsets = offsets
    model.foresight_lambda = float(cfg.get("lambda_foresight", 0.3))
    model.foresight_drop_target = bool(cfg.get("drop_target_tokens", True))

    n_projector = sum(parameter.numel() for parameter in projector.parameters())
    logger.info(
        "FLUX.2 foresight attached: layer=%d/%d encoder_layer=%d "
        "teacher_early_exit=%s offsets=%s lambda=%.3f drop_target=%s "
        "projector=%.1fM teacher_mode=%s",
        layer,
        total_blocks,
        encoder_layer,
        model.foresight_teacher_early_exit,
        offsets,
        model.foresight_lambda,
        model.foresight_drop_target,
        n_projector / 1e6,
        teacher_mode,
    )
