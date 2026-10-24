"""Foresight alignment for the Ovis-U1 (Yak) JAMPolicy stack.

Ports the Wan JAM recipe (foresight-only, gen OFF):
  - The policy's CURRENT-observation (cond) token hiddens at a mid-depth
    video-expert block are pulled (cosine) toward a FROZEN Yak expert run on
    the CLEAN frames at denoise timestep t=0. The teacher can use either the
    pretrained weights (historical default) or a native random initialization.
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
from typing import Any, Dict, Optional, Sequence

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


class VideoMAEClipTeacher(nn.Module):
    """Frozen VideoMAE encoder for a real future clip.

    The teacher receives packed two-camera frames ``[B,3,T,H,2W]``.  Each
    camera is encoded as a real VideoMAE clip; the final temporal tubelet is
    selected after full bidirectional clip encoding and the two spatial grids
    are stitched back into the dataset's horizontal camera layout.

    Selecting the final tubelet avoids a temporal-mean shortcut: the target
    corresponds specifically to the end of the future clip while still
    carrying context from every preceding frame.
    """

    def __init__(
        self,
        backbone: nn.Module,
        *,
        image_size: int,
        num_views: int,
        clip_frames: int,
        pixel_mean: Sequence[float],
        pixel_std: Sequence[float],
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.image_size = int(image_size)
        self.num_views = int(num_views)
        self.clip_frames = int(clip_frames)

        cfg = backbone.config
        self.patch_size = int(cfg.patch_size)
        self.tubelet_size = int(cfg.tubelet_size)
        self.hidden_size = int(cfg.hidden_size)
        if int(cfg.image_size) != self.image_size:
            raise ValueError(
                "VideoMAE teacher image-size mismatch: "
                f"config={cfg.image_size}, requested={self.image_size}."
            )
        if int(cfg.num_frames) != self.clip_frames:
            raise ValueError(
                "VideoMAE teacher clip-length mismatch: "
                f"config={cfg.num_frames}, requested={self.clip_frames}."
            )
        if self.image_size % self.patch_size != 0:
            raise ValueError(
                f"image_size={self.image_size} is not divisible by patch_size={self.patch_size}."
            )
        if self.clip_frames % self.tubelet_size != 0:
            raise ValueError(
                f"clip_frames={self.clip_frames} is not divisible by tubelet_size={self.tubelet_size}."
            )
        if self.num_views <= 0:
            raise ValueError(f"`num_views` must be positive, got {self.num_views}.")

        self.grid_size = self.image_size // self.patch_size
        self.temporal_grid = self.clip_frames // self.tubelet_size
        mean = torch.as_tensor(pixel_mean, dtype=torch.float32)
        std = torch.as_tensor(pixel_std, dtype=torch.float32)
        if mean.shape != (3,) or std.shape != (3,) or bool((std <= 0).any().item()):
            raise ValueError(
                f"`pixel_mean/std` must contain three valid values, got {mean}/{std}."
            )
        self.register_buffer(
            "pixel_mean",
            mean.view(1, 1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "pixel_std",
            std.view(1, 1, 3, 1, 1),
            persistent=False,
        )
        self.requires_grad_(False)
        self.eval()

    @classmethod
    def from_pretrained(
        cls,
        model_id: str,
        *,
        image_size: int,
        num_views: int,
        clip_frames: int,
        local_files_only: bool,
        attn_implementation: Optional[str],
        device: torch.device | str,
        torch_dtype: torch.dtype,
    ) -> "VideoMAEClipTeacher":
        from transformers import AutoImageProcessor, VideoMAEModel

        processor = AutoImageProcessor.from_pretrained(
            model_id,
            local_files_only=bool(local_files_only),
            use_fast=False,
        )
        load_kwargs: dict[str, Any] = {
            "local_files_only": bool(local_files_only),
            "torch_dtype": torch_dtype,
            "low_cpu_mem_usage": True,
        }
        if attn_implementation is not None:
            load_kwargs["attn_implementation"] = str(attn_implementation)
        backbone = VideoMAEModel.from_pretrained(model_id, **load_kwargs)
        teacher = cls(
            backbone,
            image_size=int(image_size),
            num_views=int(num_views),
            clip_frames=int(clip_frames),
            pixel_mean=list(processor.image_mean),
            pixel_std=list(processor.image_std),
        )
        return teacher.to(device=device, dtype=torch_dtype)

    @classmethod
    def from_random(
        cls,
        model_id: str,
        *,
        seed: int,
        image_size: int,
        num_views: int,
        clip_frames: int,
        local_files_only: bool,
        attn_implementation: Optional[str],
        device: torch.device | str,
        torch_dtype: torch.dtype,
    ) -> "VideoMAEClipTeacher":
        """Build a deterministic frozen-random VideoMAE with pretrained topology.

        The checkpoint contributes only its architecture and image-normalization
        metadata. Model weights are initialized natively under an isolated CPU
        RNG fork, so every distributed rank gets the same teacher without
        shifting the caller's training RNG stream.
        """
        from transformers import AutoConfig, AutoImageProcessor

        processor = AutoImageProcessor.from_pretrained(
            model_id,
            local_files_only=bool(local_files_only),
            use_fast=False,
        )
        config = AutoConfig.from_pretrained(
            model_id,
            local_files_only=bool(local_files_only),
        )
        if attn_implementation is not None:
            config._attn_implementation = str(attn_implementation)
        backbone = _build_random_videomae_backbone(config, seed=int(seed))
        teacher = cls(
            backbone,
            image_size=int(image_size),
            num_views=int(num_views),
            clip_frames=int(clip_frames),
            pixel_mean=list(processor.image_mean),
            pixel_std=list(processor.image_std),
        )
        return teacher.to(device=device, dtype=torch_dtype)

    def train(self, mode: bool = True):  # noqa: ARG002 - eval-locked by design
        return super().train(False)

    def _prepare_views(self, video: torch.Tensor) -> torch.Tensor:
        if video.ndim != 5 or int(video.shape[1]) != 3:
            raise ValueError(
                "VideoMAE future clip must be [B,3,T,H,W], got "
                f"{tuple(video.shape)}."
            )
        batch_size, _, frames, height, width = video.shape
        expected_width = self.image_size * self.num_views
        if int(frames) != self.clip_frames:
            raise ValueError(
                f"Expected {self.clip_frames} future frames, got {frames}."
            )
        if (int(height), int(width)) != (self.image_size, expected_width):
            raise ValueError(
                "VideoMAE future clip must preserve horizontally packed views: "
                f"expected {self.image_size}x{expected_width}, got {height}x{width}."
            )
        views = (
            video.reshape(
                batch_size,
                3,
                frames,
                self.image_size,
                self.num_views,
                self.image_size,
            )
            .permute(0, 4, 2, 1, 3, 5)
            .reshape(
                batch_size * self.num_views,
                frames,
                3,
                self.image_size,
                self.image_size,
            )
        )
        views = ((views.float() + 1.0) * 0.5).clamp_(0.0, 1.0)
        views = (views - self.pixel_mean) / self.pixel_std
        param = next(self.backbone.parameters())
        return views.to(device=param.device, dtype=param.dtype, non_blocking=True)

    @torch.no_grad()
    def forward(self, video: torch.Tensor) -> torch.Tensor:
        batch_size = int(video.shape[0])
        pixel_values = self._prepare_views(video)
        output = self.backbone(
            pixel_values=pixel_values,
            output_hidden_states=False,
            return_dict=True,
        )
        expected_tokens = self.temporal_grid * self.grid_size * self.grid_size
        tokens = output.last_hidden_state
        expected_shape = (
            batch_size * self.num_views,
            expected_tokens,
            self.hidden_size,
        )
        if tuple(tokens.shape) != expected_shape:
            raise RuntimeError(
                "Unexpected VideoMAE teacher token shape: "
                f"expected={expected_shape}, got={tuple(tokens.shape)}."
            )
        patches = tokens.reshape(
            batch_size,
            self.num_views,
            self.temporal_grid,
            self.grid_size,
            self.grid_size,
            self.hidden_size,
        )
        # The final tubelet represents frames t+15/t+16 for a 16-frame clip.
        final_tubelet = patches[:, :, -1]
        packed = torch.cat(
            [final_tubelet[:, view] for view in range(self.num_views)],
            dim=2,
        )
        return packed.reshape(
            batch_size,
            self.grid_size * self.grid_size * self.num_views,
            self.hidden_size,
        )


_TEACHER_MODES = {"frozen_pretrained", "frozen_random"}
_TEACHER_TYPES = {"yak", "videomae_clip", "vae_direct"}


def _normalize_teacher_mode(value: Any) -> str:
    mode = str(value if value is not None else "frozen_pretrained").strip().lower()
    if mode not in _TEACHER_MODES:
        raise ValueError(
            f"Unsupported foresight.teacher_mode={value!r}; expected one of "
            f"{sorted(_TEACHER_MODES)}."
        )
    return mode


def lift_vae_direct_tokens(
    tokens: torch.Tensor,
    *,
    output_dim: int,
) -> torch.Tensor:
    """Lift packed VAE tokens without changing their cosine geometry.

    Ovis-U1 packs each 2x2 spatial patch of its four-channel VAE latent into
    a 16-D token, whereas the historical Yak teacher and projector heads use
    1536-D targets. Repeating the complete token vector and scaling by
    ``1/sqrt(repeats)`` exactly preserves norms, dot products, and cosine
    similarities. The lift is fixed: it adds no learned/random target
    parameters and keeps the Ovis recipe's projector topology unchanged.
    """
    if tokens.ndim != 3:
        raise ValueError(
            f"Packed VAE tokens must be [B,N,D], got {tuple(tokens.shape)}."
        )
    input_dim = int(tokens.shape[-1])
    output_dim = int(output_dim)
    if input_dim <= 0 or output_dim < input_dim or output_dim % input_dim != 0:
        raise ValueError(
            "VAE-direct output_dim must be an integer multiple of the packed "
            f"token dim {input_dim}, got {output_dim}."
        )
    repeats = output_dim // input_dim
    if repeats == 1:
        return tokens
    return tokens.repeat(1, 1, repeats) / (repeats ** 0.5)


def _build_random_yak_expert(pretrained_expert: nn.Module, seed: int) -> nn.Module:
    """Construct a fresh Yak expert with its native module initializers.

    Construction happens on CPU under an isolated RNG fork. This makes the
    teacher deterministic across distributed ranks without shifting the RNG
    stream used later for the trainable projector, data, or optimization.
    """
    pretrained_transformer = pretrained_expert.transformer
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(int(seed))
        random_transformer = type(pretrained_transformer)(
            copy.deepcopy(pretrained_transformer.config)
        )
    random_expert = type(pretrained_expert)(random_transformer)
    ref_param = next(pretrained_expert.parameters())
    return random_expert.to(device=ref_param.device, dtype=ref_param.dtype)


def _build_random_videomae_backbone(config: Any, seed: int) -> nn.Module:
    """Construct a VideoMAE backbone with native deterministic initialization."""
    from transformers import VideoMAEModel

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(int(seed))
        backbone = VideoMAEModel(copy.deepcopy(config))
    return backbone


def attach_ovis_foresight(model, cfg: Dict[str, Any]) -> None:
    """Attach a frozen future teacher and trainable projector to Ovis-JAMPolicy.

    Teacher choices are a pretrained/random Yak self-teacher, a
    pretrained/random VideoMAE real future-clip teacher, or the already
    computed frozen pretrained Ovis VAE tokens directly. Neural teachers are
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
    if teacher_type == "vae_direct" and teacher_mode != "frozen_pretrained":
        raise ValueError(
            "foresight.teacher_type=vae_direct uses the policy's frozen "
            "pretrained Ovis VAE and requires teacher_mode=frozen_pretrained."
        )
    teacher_random_seed = int(cfg.get("teacher_random_seed", 42))
    if teacher_random_seed < 0:
        raise ValueError(
            "foresight.teacher_random_seed must be non-negative, "
            f"got {teacher_random_seed}."
        )
    teacher_hidden_dim = int(model.video_expert.hidden_dim)
    teacher_grid_size = None
    skip_teacher = os.environ.get("JAM_FORESIGHT_NO_TEACHER") == "1"
    if teacher_type == "vae_direct":
        logger.info(
            "Foresight attach: using packed frozen pretrained Ovis VAE "
            "tokens directly; no neural foresight teacher is constructed."
        )
    elif skip_teacher:
        logger.info(
            "Foresight attach: teacher SKIPPED "
            "(inference mode; configured type=%s mode=%s).",
            teacher_type,
            teacher_mode,
        )
    else:
        if teacher_type == "yak":
            if teacher_mode == "frozen_pretrained":
                expert_copy = copy.deepcopy(model.video_expert)
            else:
                expert_copy = _build_random_yak_expert(
                    model.video_expert,
                    seed=teacher_random_seed,
                )
            teacher = YakForesightTeacher(expert_copy)
        elif teacher_type == "videomae_clip":
            teacher_model_id = str(cfg.get("teacher_model_id", "")).strip()
            if not teacher_model_id:
                raise ValueError(
                    "`foresight.teacher_model_id` is required for teacher_type=videomae_clip."
                )
            ref_param = next(model.video_expert.parameters())
            teacher_kwargs = {
                "image_size": int(cfg.get("teacher_image_size", 224)),
                "num_views": int(cfg.get("teacher_num_views", 2)),
                "clip_frames": int(cfg.get("teacher_clip_frames", 16)),
                "local_files_only": bool(cfg.get("teacher_local_files_only", True)),
                "attn_implementation": cfg.get("teacher_attn_implementation", "sdpa"),
                "device": ref_param.device,
                "torch_dtype": ref_param.dtype,
            }
            if teacher_mode == "frozen_pretrained":
                teacher = VideoMAEClipTeacher.from_pretrained(
                    teacher_model_id,
                    **teacher_kwargs,
                )
            else:
                teacher = VideoMAEClipTeacher.from_random(
                    teacher_model_id,
                    seed=teacher_random_seed,
                    **teacher_kwargs,
                )
            teacher_hidden_dim = int(teacher.hidden_size)
            teacher_grid_size = int(teacher.grid_size)
        object.__setattr__(model, "foresight_teacher", teacher)

    ref_param = next(model.video_expert.parameters())
    offsets = [int(o) for o in cfg.get("offsets", [0, 1])]
    if teacher_type == "vae_direct" and offsets != [1]:
        raise ValueError(
            "Ovis VAE-direct is the strict future-only control of the Ovis recipe and "
            f"requires offsets=[1], got {offsets}."
        )
    if teacher_type == "videomae_clip" and offsets != [1]:
        raise ValueError(
            "VideoMAE clip teacher provides one strict future target and requires "
            f"offsets=[1], got {offsets}."
        )
    if teacher_type == "videomae_clip":
        declared_hidden = cfg.get("teacher_hidden_dim")
        declared_grid = cfg.get("teacher_grid_size")
        if skip_teacher and (declared_hidden is None or declared_grid is None):
            raise ValueError(
                "VideoMAE inference with JAM_FORESIGHT_NO_TEACHER=1 requires "
                "both `foresight.teacher_hidden_dim` and "
                "`foresight.teacher_grid_size` to reconstruct the projector."
            )
        if not skip_teacher:
            if declared_hidden is not None and int(declared_hidden) != teacher_hidden_dim:
                raise ValueError(
                    "Configured VideoMAE hidden size does not match the loaded teacher: "
                    f"declared={declared_hidden}, loaded={teacher_hidden_dim}."
                )
            if declared_grid is not None and int(declared_grid) != teacher_grid_size:
                raise ValueError(
                    "Configured VideoMAE grid size does not match the loaded teacher: "
                    f"declared={declared_grid}, loaded={teacher_grid_size}."
                )
        teacher_hidden_dim = int(
            declared_hidden if declared_hidden is not None else teacher_hidden_dim
        )
        teacher_grid_size = int(
            declared_grid if declared_grid is not None else teacher_grid_size
        )
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
    model.foresight_teacher_random_seed = teacher_random_seed
    model.foresight_layer = int(cfg.get("layer", 11))
    model.foresight_encoder_layer = int(cfg.get("encoder_layer", cfg.get("layer", 11)))
    model.foresight_teacher_early_exit = bool(cfg.get("teacher_early_exit", True))
    model.foresight_offsets = offsets
    model.foresight_lambda = float(cfg.get("lambda_foresight", 0.6))
    model.foresight_drop_target = bool(cfg.get("drop_target_tokens", True))
    if teacher_type == "vae_direct":
        model.foresight_vae_direct_output_dim = int(teacher_hidden_dim)
    if teacher_type == "videomae_clip":
        model.foresight_teacher_model_id = str(cfg["teacher_model_id"])
        model.foresight_teacher_clip_frames = int(cfg.get("teacher_clip_frames", 16))
        model.foresight_teacher_grid_size = int(teacher_grid_size)
        model.foresight_teacher_hidden_dim = int(teacher_hidden_dim)
        model.foresight_teacher_temporal_target = "last_tubelet_after_full_clip"

    total_blocks = int(model.video_expert.double_layers) + int(model.video_expert.single_layers)
    n_proj = sum(p.numel() for p in projector.parameters())
    logger.info(
        "Foresight attached: layer=%d/%d encoder_layer=%d "
        "teacher_type=%s teacher_early_exit=%s offsets=%s lambda=%.3f "
        "projector=%.1fM teacher_mode=%s teacher_random_seed=%d "
        "teacher_grid=%s teacher_hidden=%d",
        model.foresight_layer,
        total_blocks,
        model.foresight_encoder_layer,
        model.foresight_teacher_type,
        model.foresight_teacher_early_exit,
        model.foresight_offsets,
        model.foresight_lambda,
        n_proj / 1e6,
        model.foresight_teacher_mode,
        model.foresight_teacher_random_seed,
        teacher_grid_size,
        teacher_hidden_dim,
    )
