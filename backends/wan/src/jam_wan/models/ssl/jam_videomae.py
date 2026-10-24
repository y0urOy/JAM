"""Matched VideoMAE student/teacher policy for JAMWan.

The trainable VideoMAE student is causal: it receives the current observation
repeated to the 16-frame shape expected by the pretrained backbone.  A frozen
copy of the same checkpoint receives the real future clip ``t+1..t+16`` and
provides a training-only foresight target.  The official 30-layer Wan
ActionDiT remains the action expert.  Its blocks receive the 12 VideoMAE layer
features through an endpoint-aligned 12->30 depth map.
"""

from __future__ import annotations

import copy
import hashlib
from pathlib import Path
from typing import Any, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from jam_wan.utils.logging_config import get_logger

from ..wan22.action_dit import ActionDiT
from ..wan22.foresight import ForesightProjector
from ..wan22.schedulers.scheduler_continuous import WanContinuousFlowMatchScheduler

logger = get_logger(__name__)


VIDEOMAE_JAM_CHECKPOINT_FORMAT = "videomae_fastwam_v1"
VIDEOMAE_TEMPORAL_POOLING = "mean_after_full_clip_encoding"


def uniform_depth_map(source_depth: int, target_depth: int) -> tuple[int, ...]:
    """Endpoint-aligned source-layer index for every target block."""

    source_depth = int(source_depth)
    target_depth = int(target_depth)
    if source_depth <= 0 or target_depth <= 0:
        raise ValueError(
            f"Depths must be positive, got source={source_depth}, target={target_depth}."
        )
    if target_depth == 1:
        return (source_depth - 1,)
    return tuple(
        round(index * (source_depth - 1) / (target_depth - 1))
        for index in range(target_depth)
    )


class JAMVideoMAE(nn.Module):
    """VideoMAE visual policy with an online frozen future-clip teacher."""

    def __init__(
        self,
        *,
        student_backbone: nn.Module,
        teacher_backbone: Optional[nn.Module],
        action_expert: ActionDiT,
        pixel_mean: Sequence[float],
        pixel_std: Sequence[float],
        model_id: str,
        teacher_model_id: str,
        text_embedding_cache_dir: Optional[str],
        context_len: int,
        proprio_dim: Optional[int],
        image_size: int,
        num_views: int,
        clip_frames: int,
        use_gradient_checkpointing: bool,
        action_train_shift: float,
        action_infer_shift: float,
        action_num_train_timesteps: int,
        loss_lambda_action: float,
        loss_lambda_foresight: float,
        foresight_cfg: Optional[dict[str, Any]],
        device: str | torch.device,
        torch_dtype: torch.dtype,
    ) -> None:
        super().__init__()
        foresight_cfg = dict(foresight_cfg or {})

        self.model_id = str(model_id)
        self.teacher_model_id = str(teacher_model_id)
        self.text_embedding_cache_dir = (
            None
            if text_embedding_cache_dir is None
            else str(text_embedding_cache_dir)
        )
        self.context_len = int(context_len)
        self.proprio_dim = None if proprio_dim is None else int(proprio_dim)
        self.image_size = int(image_size)
        self.num_views = int(num_views)
        self.clip_frames = int(clip_frames)
        self.torch_dtype = torch_dtype
        self.device = torch.device(device)

        cfg = student_backbone.config
        config_image_size = int(cfg.image_size)
        patch_size = int(cfg.patch_size)
        config_frames = int(cfg.num_frames)
        tubelet_size = int(cfg.tubelet_size)
        if self.image_size != config_image_size:
            raise ValueError(
                f"VideoMAE image size mismatch: requested={self.image_size}, "
                f"checkpoint={config_image_size}."
            )
        if self.clip_frames != config_frames:
            raise ValueError(
                f"VideoMAE frame-count mismatch: requested={self.clip_frames}, "
                f"checkpoint={config_frames}."
            )
        if self.image_size % patch_size != 0:
            raise ValueError(
                f"image_size={self.image_size} must be divisible by patch_size={patch_size}."
            )
        if self.clip_frames % tubelet_size != 0:
            raise ValueError(
                f"clip_frames={self.clip_frames} must be divisible by tubelet_size={tubelet_size}."
            )
        if self.num_views <= 0:
            raise ValueError(f"`num_views` must be positive, got {self.num_views}.")

        self.patch_size = patch_size
        self.tubelet_size = tubelet_size
        self.spatial_grid = self.image_size // self.patch_size
        self.temporal_grid = self.clip_frames // self.tubelet_size
        self.hidden_size = int(cfg.hidden_size)
        self.student_num_layers = int(cfg.num_hidden_layers)
        self.attn_implementation = str(
            getattr(cfg, "_attn_implementation", "eager")
        )
        self.action_num_layers = len(action_expert.blocks)
        self.action_layer_map = uniform_depth_map(
            self.student_num_layers,
            self.action_num_layers,
        )
        self.visual_tokens = (
            self.spatial_grid * self.spatial_grid * self.num_views
        )

        mean = torch.as_tensor(pixel_mean, dtype=torch.float32)
        std = torch.as_tensor(pixel_std, dtype=torch.float32)
        if mean.shape != (3,) or std.shape != (3,) or bool((std <= 0).any()):
            raise ValueError(
                f"`pixel_mean/std` must contain three valid values, got {mean}/{std}."
            )
        self.register_buffer(
            "pixel_mean",
            mean.reshape(1, 1, 3, 1, 1).to(device=self.device),
            persistent=False,
        )
        self.register_buffer(
            "pixel_std",
            std.reshape(1, 1, 3, 1, 1).to(device=self.device),
            persistent=False,
        )

        if int(action_expert.text_dim) <= 0:
            raise ValueError("ActionDiT must expose a positive text dimension.")
        self.text_dim = int(action_expert.text_dim)

        projector_hidden = int(
            foresight_cfg.get("projector_hidden_dim", self.hidden_size // 2)
        )
        projector_heads = int(
            foresight_cfg.get(
                "projector_num_heads",
                max(1, int(cfg.num_attention_heads) // 2),
            )
        )
        projector_ffn = int(
            foresight_cfg.get("projector_ffn_dim", projector_hidden * 4)
        )
        projector_layers = int(foresight_cfg.get("projector_num_layers", 2))
        if projector_hidden % projector_heads != 0:
            raise ValueError(
                "Foresight projector hidden dimension must be divisible by heads: "
                f"{projector_hidden} vs {projector_heads}."
            )

        visual_projector = nn.Sequential(
            nn.LayerNorm(self.hidden_size),
            nn.Linear(self.hidden_size, action_expert.hidden_dim),
        )
        proprio_projector: nn.Module
        if self.proprio_dim is None:
            proprio_projector = nn.Identity()
        else:
            proprio_projector = nn.Sequential(
                nn.Linear(self.proprio_dim, self.hidden_size),
                nn.SiLU(),
                nn.Linear(self.hidden_size, self.text_dim),
            )

        self.foresight_enabled = bool(foresight_cfg.get("enabled", True))
        self.foresight_load_teacher = bool(
            foresight_cfg.get("load_teacher", True)
        )
        self.future_clip_indices = tuple(
            int(index)
            for index in foresight_cfg.get(
                "future_clip_indices",
                range(1, self.clip_frames + 1),
            )
        )
        expected_future_indices = tuple(range(1, self.clip_frames + 1))
        if self.future_clip_indices != expected_future_indices:
            raise ValueError(
                "JAMVideoMAE currently implements the strict consecutive "
                f"future clip {expected_future_indices}, got "
                f"{self.future_clip_indices}."
            )
        modules: dict[str, nn.Module] = {
            "videomae": student_backbone,
            "visual_projector": visual_projector,
            "action_expert": action_expert,
            "proprio_projector": proprio_projector,
        }
        if self.foresight_enabled:
            modules["foresight_projector"] = ForesightProjector(
                in_dim=self.hidden_size,
                out_dim=self.hidden_size,
                hidden_dim=projector_hidden,
                num_layers=projector_layers,
                num_heads=projector_heads,
                ffn_dim=projector_ffn,
            )
        self.dit = nn.ModuleDict(modules)
        self.dit.to(device=self.device, dtype=self.torch_dtype)

        self._teacher_ref: Optional[list[nn.Module]] = None
        if teacher_backbone is not None:
            teacher_backbone.eval()
            teacher_backbone.requires_grad_(False)
            teacher_backbone.to(device=self.device, dtype=self.torch_dtype)
            self._teacher_ref = [teacher_backbone]

        self.train_action_scheduler = WanContinuousFlowMatchScheduler(
            num_train_timesteps=int(action_num_train_timesteps),
            shift=float(action_train_shift),
        )
        self.infer_action_scheduler = WanContinuousFlowMatchScheduler(
            num_train_timesteps=int(action_num_train_timesteps),
            shift=float(action_infer_shift),
        )
        self.loss_lambda_action = float(loss_lambda_action)
        self.loss_lambda_foresight = float(loss_lambda_foresight)
        self.use_gradient_checkpointing = bool(use_gradient_checkpointing)

        logger.info(
            "JAMVideoMAE: model=%s student=trainable teacher_loaded=%s "
            "clip=current-repeat%d->future[1..%d] views=%d grid=%dx%d "
            "temporal_tokens=%d visual_tokens=%d depth_map=%d->%d "
            "pooling=%s attention=%s lambda=%.3f",
            self.model_id,
            self.teacher_backbone is not None,
            self.clip_frames,
            self.clip_frames,
            self.num_views,
            self.spatial_grid,
            self.spatial_grid * self.num_views,
            self.temporal_grid,
            self.visual_tokens,
            self.student_num_layers,
            self.action_num_layers,
            VIDEOMAE_TEMPORAL_POOLING,
            self.attn_implementation,
            self.loss_lambda_foresight,
        )

    @property
    def student_backbone(self) -> nn.Module:
        return self.dit["videomae"]

    @property
    def teacher_backbone(self) -> Optional[nn.Module]:
        if self._teacher_ref is None:
            return None
        return self._teacher_ref[0]

    @property
    def visual_projector(self) -> nn.Module:
        return self.dit["visual_projector"]

    @property
    def action_expert(self) -> ActionDiT:
        return self.dit["action_expert"]

    @property
    def proprio_projector(self) -> nn.Module:
        return self.dit["proprio_projector"]

    @property
    def foresight_projector(self) -> Optional[nn.Module]:
        return self.dit["foresight_projector"] if "foresight_projector" in self.dit else None

    @classmethod
    def from_pretrained(
        cls,
        *,
        model_id: str,
        teacher_model_id: Optional[str] = None,
        text_embedding_cache_dir: Optional[str] = None,
        context_len: int = 128,
        proprio_dim: Optional[int] = None,
        image_size: int = 224,
        num_views: int = 2,
        clip_frames: int = 16,
        local_files_only: bool = True,
        attn_implementation: Optional[str] = "sdpa",
        load_text_encoder: bool = False,
        use_gradient_checkpointing: bool = True,
        action_dit_config: Optional[dict[str, Any]] = None,
        action_dit_pretrained_path: Optional[str] = None,
        skip_dit_load_from_pretrain: bool = False,
        action_train_shift: float = 5.0,
        action_infer_shift: float = 5.0,
        action_num_train_timesteps: int = 1000,
        loss_lambda_action: float = 1.0,
        loss_lambda_foresight: float = 0.3,
        foresight_cfg: Optional[dict[str, Any]] = None,
        device: str = "cuda",
        torch_dtype: torch.dtype = torch.bfloat16,
    ) -> "JAMVideoMAE":
        try:
            from transformers import AutoImageProcessor, VideoMAEModel
        except (ImportError, AttributeError) as exc:
            raise ImportError(
                "JAMVideoMAE requires Transformers with VideoMAE support."
            ) from exc

        if load_text_encoder:
            logger.info(
                "JAMVideoMAE ignores `load_text_encoder=true`; LIBERO text "
                "conditioning is read from the established Wan T5 cache."
            )

        foresight_cfg = dict(foresight_cfg or {})
        teacher_model_id = str(teacher_model_id or model_id)
        processor = AutoImageProcessor.from_pretrained(
            model_id,
            local_files_only=bool(local_files_only),
            use_fast=False,
        )
        logger.info("Loading trainable VideoMAE student from %s", model_id)
        load_kwargs = {
            "local_files_only": bool(local_files_only),
            "torch_dtype": torch_dtype,
            "low_cpu_mem_usage": True,
        }
        if attn_implementation is not None:
            load_kwargs["attn_implementation"] = str(attn_implementation)
        student = VideoMAEModel.from_pretrained(
            model_id,
            **load_kwargs,
        )

        load_teacher = bool(foresight_cfg.get("enabled", True)) and bool(
            foresight_cfg.get("load_teacher", True)
        )
        teacher = None
        if load_teacher:
            if teacher_model_id == str(model_id):
                logger.info(
                    "Creating frozen VideoMAE teacher from the exact student checkpoint."
                )
                teacher = copy.deepcopy(student)
            else:
                logger.info(
                    "Loading frozen VideoMAE teacher from %s", teacher_model_id
                )
                teacher = VideoMAEModel.from_pretrained(
                    teacher_model_id,
                    **load_kwargs,
                )
            disable = getattr(teacher, "gradient_checkpointing_disable", None)
            if callable(disable):
                disable()
            teacher.eval()
            teacher.requires_grad_(False)

        if use_gradient_checkpointing:
            student.gradient_checkpointing_enable()
        student.to(device=device, dtype=torch_dtype)

        action_expert = ActionDiT.from_pretrained(
            action_dit_config=dict(action_dit_config or {}),
            action_dit_pretrained_path=action_dit_pretrained_path,
            skip_dit_load_from_pretrain=bool(skip_dit_load_from_pretrain),
            device=device,
            torch_dtype=torch_dtype,
        )
        return cls(
            student_backbone=student,
            teacher_backbone=teacher,
            action_expert=action_expert,
            pixel_mean=list(processor.image_mean),
            pixel_std=list(processor.image_std),
            model_id=str(model_id),
            teacher_model_id=teacher_model_id,
            text_embedding_cache_dir=text_embedding_cache_dir,
            context_len=int(context_len),
            proprio_dim=proprio_dim,
            image_size=int(image_size),
            num_views=int(num_views),
            clip_frames=int(clip_frames),
            use_gradient_checkpointing=bool(use_gradient_checkpointing),
            action_train_shift=float(action_train_shift),
            action_infer_shift=float(action_infer_shift),
            action_num_train_timesteps=int(action_num_train_timesteps),
            loss_lambda_action=float(loss_lambda_action),
            loss_lambda_foresight=float(loss_lambda_foresight),
            foresight_cfg=foresight_cfg,
            device=device,
            torch_dtype=torch_dtype,
        )

    def train(self, mode: bool = True):
        super().train(mode)
        if self.teacher_backbone is not None:
            self.teacher_backbone.eval()
        return self

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        if self.teacher_backbone is not None:
            self.teacher_backbone.to(*args, **kwargs)
        if args and isinstance(args[0], (str, torch.device)):
            self.device = torch.device(args[0])
        elif kwargs.get("device") is not None:
            self.device = torch.device(kwargs["device"])
        if kwargs.get("dtype") is not None:
            self.torch_dtype = kwargs["dtype"]
        return self

    def _prepare_video_views(self, video: torch.Tensor) -> torch.Tensor:
        """Normalize packed cameras and return ``[B*V,T,3,H,W]``."""

        if video.ndim == 4:
            video = video.unsqueeze(2)
        if video.ndim != 5 or int(video.shape[1]) != 3:
            raise ValueError(
                f"VideoMAE input must be [B,3,T,H,W], got {tuple(video.shape)}."
            )
        batch, _, frames, height, width = video.shape
        expected_width = self.image_size * self.num_views
        if (int(height), int(width)) != (self.image_size, expected_width):
            raise ValueError(
                "VideoMAE expects horizontally packed square camera views: "
                f"expected HxW={self.image_size}x{expected_width}, "
                f"got {height}x{width}."
            )
        if int(frames) != self.clip_frames:
            raise ValueError(
                f"VideoMAE expects {self.clip_frames} frames, got {frames}."
            )
        views = (
            video.reshape(
                batch,
                3,
                frames,
                self.image_size,
                self.num_views,
                self.image_size,
            )
            .permute(0, 4, 2, 1, 3, 5)
            .reshape(
                batch * self.num_views,
                frames,
                3,
                self.image_size,
                self.image_size,
            )
        )
        views = ((views.float() + 1.0) * 0.5).clamp(0.0, 1.0)
        views = (views - self.pixel_mean) / self.pixel_std
        param = next(self.student_backbone.parameters())
        return views.to(
            device=param.device,
            dtype=param.dtype,
            non_blocking=True,
        )

    def _pool_and_stitch(
        self,
        tokens: torch.Tensor,
        *,
        batch_size: int,
    ) -> torch.Tensor:
        """Pool all encoded tubelets, then stitch camera patch grids."""

        expected_per_view = (
            self.temporal_grid * self.spatial_grid * self.spatial_grid
        )
        expected_shape = (
            batch_size * self.num_views,
            expected_per_view,
            self.hidden_size,
        )
        if tuple(tokens.shape) != expected_shape:
            raise ValueError(
                f"Unexpected VideoMAE token shape: expected={expected_shape}, "
                f"got={tuple(tokens.shape)}."
            )
        patches = tokens.reshape(
            batch_size,
            self.num_views,
            self.temporal_grid,
            self.spatial_grid,
            self.spatial_grid,
            self.hidden_size,
        )
        # Every future tubelet participates in the teacher representation.  The
        # temporal mean keeps ActionDiT's visual K/V length tractable while
        # preserving the motion-aware features produced by the full clip.
        patches = patches.mean(dim=2)
        packed = torch.cat(
            [patches[:, view] for view in range(self.num_views)],
            dim=2,
        )
        return packed.reshape(batch_size, self.visual_tokens, self.hidden_size)

    def _encode_student(
        self,
        current_image: torch.Tensor,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
        batch_size = int(current_image.shape[0])
        current_clip = current_image.unsqueeze(2).expand(
            -1,
            -1,
            self.clip_frames,
            -1,
            -1,
        )
        pixel_values = self._prepare_video_views(current_clip)
        output = self.student_backbone(
            pixel_values=pixel_values,
            output_hidden_states=True,
            return_dict=True,
        )
        if output.hidden_states is None:
            raise RuntimeError("VideoMAE did not return requested hidden states.")
        raw_layers = tuple(output.hidden_states[1:])
        if len(raw_layers) != self.student_num_layers:
            raise RuntimeError(
                f"Expected {self.student_num_layers} VideoMAE layers, "
                f"got {len(raw_layers)}."
            )
        pooled_layers = tuple(
            self._pool_and_stitch(hidden, batch_size=batch_size)
            for hidden in raw_layers
        )
        projected_layers = tuple(
            self.visual_projector(hidden) for hidden in pooled_layers
        )
        layerwise_visual = tuple(
            projected_layers[source_index]
            for source_index in self.action_layer_map
        )
        student_visual = self._pool_and_stitch(
            output.last_hidden_state,
            batch_size=batch_size,
        )
        return student_visual, layerwise_visual

    @torch.no_grad()
    def _encode_teacher(self, future_clip: torch.Tensor) -> torch.Tensor:
        if self.teacher_backbone is None:
            raise RuntimeError(
                "VideoMAE foresight training requires the frozen teacher; "
                "`foresight.load_teacher=false` is evaluation-only."
            )
        batch_size = int(future_clip.shape[0])
        # Normalization values are shared by the matched checkpoint.  The
        # helper resolves dtype/device from the student, so explicitly move the
        # prepared tensor to the teacher afterwards.
        pixel_values = self._prepare_video_views(future_clip)
        teacher_param = next(self.teacher_backbone.parameters())
        pixel_values = pixel_values.to(
            device=teacher_param.device,
            dtype=teacher_param.dtype,
            non_blocking=True,
        )
        output = self.teacher_backbone(
            pixel_values=pixel_values,
            output_hidden_states=False,
            return_dict=True,
        )
        return self._pool_and_stitch(
            output.last_hidden_state,
            batch_size=batch_size,
        )

    def _prepare_context(
        self,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        proprio: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if context.ndim == 2:
            context = context.unsqueeze(0)
        if context_mask.ndim == 1:
            context_mask = context_mask.unsqueeze(0)
        if context.ndim != 3 or context_mask.ndim != 2:
            raise ValueError(
                "`context/context_mask` must be [B,L,D]/[B,L], got "
                f"{tuple(context.shape)}/{tuple(context_mask.shape)}."
            )
        if int(context.shape[-1]) != self.text_dim:
            raise ValueError(
                f"Text context dim must be {self.text_dim}, got {context.shape[-1]}."
            )
        if tuple(context.shape[:2]) != tuple(context_mask.shape):
            raise ValueError(
                f"Context/mask shape mismatch: {tuple(context.shape)} vs "
                f"{tuple(context_mask.shape)}."
            )
        context = context.to(
            device=self.device,
            dtype=self.torch_dtype,
            non_blocking=True,
        )
        context_mask = context_mask.to(
            device=self.device,
            dtype=torch.bool,
            non_blocking=True,
        )

        if self.proprio_dim is None:
            if proprio is not None:
                raise ValueError(
                    "`proprio` was provided but the model has `proprio_dim=null`."
                )
            return context, context_mask
        if proprio is None:
            raise ValueError(
                "JAMVideoMAE requires proprio when `proprio_dim` is configured."
            )
        if proprio.ndim == 3:
            proprio = proprio[:, 0]
        elif proprio.ndim == 1:
            proprio = proprio.unsqueeze(0)
        expected = (int(context.shape[0]), self.proprio_dim)
        if tuple(proprio.shape) != expected:
            raise ValueError(
                f"`proprio` must resolve to {expected}, got {tuple(proprio.shape)}."
            )
        token = self.proprio_projector(
            proprio.to(
                device=self.device,
                dtype=self.torch_dtype,
                non_blocking=True,
            )
        ).unsqueeze(1)
        token_mask = torch.ones(
            (context.shape[0], 1),
            device=self.device,
            dtype=torch.bool,
        )
        return (
            torch.cat((context, token), dim=1),
            torch.cat((context_mask, token_mask), dim=1),
        )

    def encode_prompt(self, prompt: str) -> tuple[torch.Tensor, torch.Tensor]:
        """Read the established Wan T5 cache used by LIBERO training."""

        if self.text_embedding_cache_dir is None:
            raise ValueError(
                "JAMVideoMAE has no online text encoder. Configure "
                "`text_embedding_cache_dir` and precompute Wan T5 contexts."
            )
        digest = hashlib.sha256(str(prompt).encode("utf-8")).hexdigest()
        cache_path = (
            Path(self.text_embedding_cache_dir)
            / f"{digest}.t5_len{self.context_len}.wan22ti2v5b.pt"
        )
        if not cache_path.is_file():
            raise FileNotFoundError(
                f"Missing cached text context: {cache_path}. "
                "Run scripts/precompute_text_embeds.py first."
            )
        payload = torch.load(cache_path, map_location="cpu")
        context = payload["context"]
        mask = payload["mask"].bool()
        if tuple(context.shape[:1]) != (self.context_len,) or tuple(mask.shape) != (
            self.context_len,
        ):
            raise ValueError(
                f"Cached context length mismatch in {cache_path}: "
                f"context={tuple(context.shape)} mask={tuple(mask.shape)}."
            )
        context = context.clone()
        context[~mask] = 0
        # Match RobotVideoDataset: Wan historically exposes zero-padded tokens
        # to cross-attention under an all-valid mask.
        mask = torch.ones_like(mask)
        return (
            context.unsqueeze(0).to(
                device=self.device,
                dtype=self.torch_dtype,
            ),
            mask.unsqueeze(0).to(device=self.device),
        )

    def _compute_action_loss(
        self,
        *,
        action: torch.Tensor,
        action_is_pad: Optional[torch.Tensor],
        context: torch.Tensor,
        context_mask: torch.Tensor,
        layerwise_visual: Sequence[torch.Tensor],
    ) -> torch.Tensor:
        batch_size = int(action.shape[0])
        noise = torch.randn_like(action)
        timestep = self.train_action_scheduler.sample_training_t(
            batch_size=batch_size,
            device=self.device,
            dtype=action.dtype,
        )
        noisy_action = self.train_action_scheduler.add_noise(
            action,
            noise,
            timestep,
        )
        target = self.train_action_scheduler.training_target(
            action,
            noise,
            timestep,
        )
        prediction = self.action_expert.forward_layerwise_mot(
            action_tokens=noisy_action,
            timestep=timestep,
            context=context,
            context_mask=context_mask,
            layerwise_visual_tokens=layerwise_visual,
        )
        token_loss = F.mse_loss(
            prediction.float(),
            target.float(),
            reduction="none",
        ).mean(dim=-1)
        if action_is_pad is None:
            per_sample = token_loss.mean(dim=1)
        else:
            if tuple(action_is_pad.shape) != tuple(token_loss.shape):
                raise ValueError(
                    f"`action_is_pad` must be {tuple(token_loss.shape)}, "
                    f"got {tuple(action_is_pad.shape)}."
                )
            valid = (
                ~action_is_pad.to(device=self.device, dtype=torch.bool)
            ).to(token_loss.dtype)
            per_sample = (token_loss * valid).sum(dim=1) / valid.sum(
                dim=1
            ).clamp(min=1.0)
        weight = self.train_action_scheduler.training_weight(timestep).to(
            device=per_sample.device,
            dtype=per_sample.dtype,
        )
        return (per_sample * weight).mean()

    def _compute_foresight_loss(
        self,
        *,
        student_visual: torch.Tensor,
        teacher_visual: torch.Tensor,
        ssl_image_is_pad: Optional[torch.Tensor],
    ) -> torch.Tensor:
        projector = self.foresight_projector
        if projector is None:
            raise RuntimeError("Foresight is enabled but its projector is missing.")
        prediction = projector(student_visual)
        if tuple(prediction.shape) != tuple(teacher_visual.shape):
            raise RuntimeError(
                "VideoMAE student/teacher target mismatch: "
                f"{tuple(prediction.shape)} vs {tuple(teacher_visual.shape)}."
            )
        per_sample = (
            1.0
            - F.cosine_similarity(
                prediction.float(),
                teacher_visual.detach().float(),
                dim=-1,
            )
        ).mean(dim=1)
        if ssl_image_is_pad is None:
            valid = torch.ones_like(per_sample)
        else:
            if ssl_image_is_pad.ndim != 2 or tuple(
                ssl_image_is_pad.shape
            ) != (int(per_sample.shape[0]), self.clip_frames):
                raise ValueError(
                    "`ssl_image_is_pad` must be "
                    f"[B,{self.clip_frames}], got {tuple(ssl_image_is_pad.shape)}."
                )
            valid = (~ssl_image_is_pad.to(
                device=self.device,
                dtype=torch.bool,
            ).any(dim=1)).to(per_sample.dtype)
        if bool(valid.any().item()):
            return (per_sample * valid).sum() / valid.sum()
        return prediction.float().sum() * 0.0

    def _validate_training_sample(self, sample: dict[str, Any]) -> None:
        required = {
            "video",
            "action",
            "proprio",
            "context",
            "context_mask",
        }
        if self.foresight_enabled:
            required.add("ssl_video")
        missing = required - set(sample)
        if missing:
            raise ValueError(
                f"JAMVideoMAE sample is missing: {sorted(missing)}."
            )
        video = sample["video"]
        future = sample.get("ssl_video")
        action = sample["action"]
        if video.ndim != 5 or tuple(video.shape[1:3]) != (3, 1):
            raise ValueError(
                f"`video` must be causal current-only [B,3,1,H,W], got {tuple(video.shape)}."
            )
        if self.foresight_enabled:
            if future.ndim != 5 or tuple(future.shape[1:3]) != (
                3,
                self.clip_frames,
            ):
                raise ValueError(
                    f"`ssl_video` must be [B,3,{self.clip_frames},H,W], "
                    f"got {tuple(future.shape)}."
                )
        if action.ndim != 3 or int(action.shape[-1]) != self.action_expert.action_dim:
            raise ValueError(
                f"`action` must be [B,T,{self.action_expert.action_dim}], "
                f"got {tuple(action.shape)}."
            )
        if int(video.shape[0]) != int(action.shape[0]):
            raise ValueError(
                "Video/action batch sizes do not match: "
                f"{video.shape[0]}/{action.shape[0]}."
            )
        if self.foresight_enabled and int(future.shape[0]) != int(video.shape[0]):
            raise ValueError(
                "Video/future batch sizes do not match: "
                f"{video.shape[0]}/{future.shape[0]}."
            )

    def training_loss(self, sample: dict[str, Any]):
        self._validate_training_sample(sample)
        video = sample["video"]
        action = sample["action"].to(
            device=self.device,
            dtype=self.torch_dtype,
            non_blocking=True,
        )
        context, context_mask = self._prepare_context(
            sample["context"],
            sample["context_mask"],
            sample.get("proprio"),
        )
        student_visual, layerwise_visual = self._encode_student(video[:, :, 0])
        action_is_pad = sample.get("action_is_pad")
        if action_is_pad is not None:
            action_is_pad = action_is_pad.to(
                device=self.device,
                dtype=torch.bool,
                non_blocking=True,
            )
        loss_action = self._compute_action_loss(
            action=action,
            action_is_pad=action_is_pad,
            context=context,
            context_mask=context_mask,
            layerwise_visual=layerwise_visual,
        )

        loss_foresight = loss_action.new_zeros(())
        if self.foresight_enabled:
            teacher_visual = self._encode_teacher(sample["ssl_video"])
            loss_foresight = self._compute_foresight_loss(
                student_visual=student_visual,
                teacher_visual=teacher_visual,
                ssl_image_is_pad=sample.get("ssl_image_is_pad"),
            )
        loss_total = (
            self.loss_lambda_action * loss_action
            + self.loss_lambda_foresight * loss_foresight
        )
        return loss_total, {
            "loss_action": self.loss_lambda_action
            * float(loss_action.detach().item()),
            "loss_foresight": self.loss_lambda_foresight
            * float(loss_foresight.detach().item()),
        }

    @torch.no_grad()
    def infer_action(
        self,
        prompt: Optional[str],
        input_image: torch.Tensor,
        action_horizon: int,
        proprio: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
        negative_prompt: Optional[str] = None,
        text_cfg_scale: float = 1.0,
        num_inference_steps: int = 20,
        sigma_shift: Optional[float] = None,
        seed: Optional[int] = None,
        rand_device: str = "cpu",
        tiled: bool = False,
        **_unused,
    ) -> dict[str, torch.Tensor]:
        del negative_prompt, text_cfg_scale, tiled
        self.eval()
        if input_image.ndim == 3:
            input_image = input_image.unsqueeze(0)
        if input_image.ndim != 4 or tuple(input_image.shape[:2]) != (1, 3):
            raise ValueError(
                "`input_image` must be [1,3,H,W] or [3,H,W], got "
                f"{tuple(input_image.shape)}."
            )
        if int(action_horizon) <= 0:
            raise ValueError(
                f"`action_horizon` must be positive, got {action_horizon}."
            )
        if context is None or context_mask is None:
            if context is not None or context_mask is not None:
                raise ValueError(
                    "`context` and `context_mask` must be provided together."
                )
            if prompt is None:
                raise ValueError(
                    "Provide either prompt or cached context/context_mask."
                )
            context, context_mask = self.encode_prompt(str(prompt))
        context, context_mask = self._prepare_context(
            context,
            context_mask,
            proprio,
        )
        _, layerwise_visual = self._encode_student(input_image)

        generator = (
            None
            if seed is None
            else torch.Generator(device=rand_device).manual_seed(int(seed))
        )
        action = torch.randn(
            (1, int(action_horizon), self.action_expert.action_dim),
            generator=generator,
            device=rand_device,
            dtype=torch.float32,
        ).to(device=self.device, dtype=self.torch_dtype)
        timesteps, deltas = (
            self.infer_action_scheduler.build_inference_schedule(
                num_inference_steps=int(num_inference_steps),
                device=self.device,
                dtype=action.dtype,
                shift_override=sigma_shift,
            )
        )
        for step_t, step_delta in zip(timesteps, deltas):
            timestep = step_t.reshape(1).to(
                device=self.device,
                dtype=action.dtype,
            )
            prediction = self.action_expert.forward_layerwise_mot(
                action_tokens=action,
                timestep=timestep,
                context=context,
                context_mask=context_mask,
                layerwise_visual_tokens=layerwise_visual,
            )
            action = self.infer_action_scheduler.step(
                prediction,
                step_delta,
                action,
            )
        return {
            "action": action[0].detach().to(
                device="cpu",
                dtype=torch.float32,
            )
        }

    def save_checkpoint(self, path, optimizer=None, step=None) -> None:
        payload = {
            "format": VIDEOMAE_JAM_CHECKPOINT_FORMAT,
            "dit": self.dit.state_dict(),
            "step": step,
            "model_id": self.model_id,
            "teacher_model_id": self.teacher_model_id,
            "image_size": self.image_size,
            "num_views": self.num_views,
            "clip_frames": self.clip_frames,
            "student_num_layers": self.student_num_layers,
            "action_num_layers": self.action_num_layers,
            "action_layer_map": self.action_layer_map,
            "future_clip_indices": self.future_clip_indices,
            "temporal_pooling": VIDEOMAE_TEMPORAL_POOLING,
            "attn_implementation": self.attn_implementation,
            "foresight_enabled": self.foresight_enabled,
        }
        if optimizer is not None:
            payload["optimizer"] = optimizer.state_dict()
        torch.save(payload, path)

    def load_checkpoint(self, path, optimizer=None, mmap: bool = False):
        load_kwargs = {"map_location": "cpu"}
        if mmap:
            load_kwargs["mmap"] = True
        payload = torch.load(path, **load_kwargs)
        if payload.get("format") != VIDEOMAE_JAM_CHECKPOINT_FORMAT:
            raise ValueError(
                f"Unsupported JAMVideoMAE checkpoint: "
                f"{payload.get('format')!r}."
            )
        expected = {
            "image_size": self.image_size,
            "num_views": self.num_views,
            "clip_frames": self.clip_frames,
            "student_num_layers": self.student_num_layers,
            "action_num_layers": self.action_num_layers,
            "future_clip_indices": self.future_clip_indices,
            "temporal_pooling": VIDEOMAE_TEMPORAL_POOLING,
            "attn_implementation": self.attn_implementation,
            "foresight_enabled": self.foresight_enabled,
        }
        mismatches = {
            key: (payload.get(key), value)
            for key, value in expected.items()
            if payload.get(key) != value
        }
        saved_map = tuple(int(value) for value in payload.get("action_layer_map", ()))
        if saved_map != self.action_layer_map:
            mismatches["action_layer_map"] = (
                saved_map,
                self.action_layer_map,
            )
        if mismatches:
            raise ValueError(
                f"JAMVideoMAE checkpoint topology mismatch: {mismatches}."
            )
        self.dit.load_state_dict(payload["dit"], strict=True)
        if optimizer is not None and "optimizer" in payload:
            optimizer.load_state_dict(payload["optimizer"])
        return payload

    def forward(self, *args, **kwargs):
        return self.training_loss(*args, **kwargs)


__all__ = [
    "VIDEOMAE_FASTWAM_CHECKPOINT_FORMAT",
    "VIDEOMAE_TEMPORAL_POOLING",
    "JAMVideoMAE",
    "uniform_depth_map",
]


# Compatibility aliases for historical configs and imports.
VideoMAEFastWAM = JAMVideoMAE

# Historical constant aliases; serialized values are unchanged.
VIDEOMAE_FASTWAM_CHECKPOINT_FORMAT = VIDEOMAE_JAM_CHECKPOINT_FORMAT

__all__ = list(dict.fromkeys(__all__ + ['JAMVideoMAE', 'VIDEOMAE_FASTWAM_CHECKPOINT_FORMAT', 'VIDEOMAE_JAM_CHECKPOINT_FORMAT', 'VideoMAEFastWAM']))
