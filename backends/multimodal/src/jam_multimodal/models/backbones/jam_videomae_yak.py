"""Strict VideoMAE replacement for the visual backbone of the Ovis-U1 policy.

The trainable student consumes only the current image through the exact
single-frame equivalent of VideoMAE's two-frame tubelet stem.  Its 12 visual
states are mapped uniformly into the unchanged pretrained 18-layer
ActionDiTYak.  A separate frozen VideoMAE teacher consumes the real future
clip ``t+1...t+16`` and supplies the final ``t+15/t+16`` tubelet target.

No Ovis condition encoder, image VAE, or Yak video expert is constructed.
Language uses the same cached Wan-UMT5 features as the strict DINO variant.
"""

from __future__ import annotations

import os
from typing import Any, Optional, Sequence

import torch

from jam_multimodal.utils.logging_config import get_logger

from .action_dit_yak import ActionDiTYak
from .jam_dinov2_yak import JAMDINOv2Yak
from .external_videomae import SingleFrameVideoMAEBackbone
from .foresight_yak import VideoMAEClipTeacher

logger = get_logger(__name__)


VIDEOMAE_YAK18_CHECKPOINT_FORMAT = "videomae_imagewam_yak18_strict_v1"
VIDEOMAE_YAK18_FUSION_TYPE = "videomae_yak18_layerwise_mot"
VIDEOMAE_STUDENT_INPUT = "single_frame_folded_tubelet2_first_position"
VIDEOMAE_TEACHER_INPUT = "real_future_clip_t1_t16_last_tubelet_t15_t16"


class JAMVideoMAEYak(JAMDINOv2Yak):
    """VideoMAE-B visual policy with the original ImageWAM ActionDiTYak18."""

    # The input video contains every raw frame t...t+16 (stride one).
    expected_frame_offsets = (16,)

    def __init__(
        self,
        *,
        student_backbone: SingleFrameVideoMAEBackbone,
        teacher_backbone: Optional[VideoMAEClipTeacher],
        action_expert: ActionDiTYak,
        model_id: str,
        text_embedding_cache_dir: str,
        text_context_len: int,
        text_hidden_dim: int,
        proprio_dim: Optional[int],
        teacher_clip_frames: int,
        predictor_hidden_dim: int,
        predictor_ffn_dim: int,
        predictor_num_layers: int,
        predictor_num_heads: int,
        action_train_shift: float,
        action_infer_shift: float,
        action_num_train_timesteps: int,
        loss_lambda_action: float,
        loss_lambda_foresight: float,
        foresight_cfg: dict[str, Any],
        device: str | torch.device,
        torch_dtype: torch.dtype,
    ) -> None:
        if not isinstance(student_backbone, SingleFrameVideoMAEBackbone):
            raise TypeError(
                "Strict VideoMAE replacement requires "
                f"SingleFrameVideoMAEBackbone, got {type(student_backbone).__name__}."
            )
        if teacher_backbone is not None and not isinstance(
            teacher_backbone, VideoMAEClipTeacher
        ):
            raise TypeError(
                "Strict VideoMAE replacement requires VideoMAEClipTeacher, got "
                f"{type(teacher_backbone).__name__}."
            )
        self.teacher_clip_frames = int(teacher_clip_frames)
        if self.teacher_clip_frames != 16:
            raise ValueError(
                "Strict VideoMAE LIBERO semantics require a 16-frame future clip, "
                f"got {self.teacher_clip_frames}."
            )
        if student_backbone.student_input_mode != VIDEOMAE_STUDENT_INPUT:
            raise ValueError(
                "Unexpected VideoMAE student input mode: "
                f"{student_backbone.student_input_mode!r}."
            )

        super().__init__(
            student_backbone=student_backbone,
            teacher_backbone=teacher_backbone,
            action_expert=action_expert,
            model_id=model_id,
            text_embedding_cache_dir=text_embedding_cache_dir,
            text_context_len=text_context_len,
            text_hidden_dim=text_hidden_dim,
            proprio_dim=proprio_dim,
            predictor_hidden_dim=predictor_hidden_dim,
            predictor_ffn_dim=predictor_ffn_dim,
            predictor_num_layers=predictor_num_layers,
            predictor_num_heads=predictor_num_heads,
            action_train_shift=action_train_shift,
            action_infer_shift=action_infer_shift,
            action_num_train_timesteps=action_num_train_timesteps,
            loss_lambda_action=loss_lambda_action,
            loss_lambda_foresight=loss_lambda_foresight,
            foresight_cfg=foresight_cfg,
            device=device,
            torch_dtype=torch_dtype,
        )

        # Give the standalone checkpoint an accurate, family-specific module
        # name. All inherited policy methods access the dynamic property below.
        self.dit["videomae"] = self.dit.pop("dinov2")

        logger.info(
            "JAMVideoMAEYak strict replacement: VideoMAE=%s layers=%d "
            "hidden=%d grid=%dx%d views=%d ActionDiTYak=%d hidden=%d map=%s "
            "student=t-only teacher=t+1...t+16:last-tubelet "
            "text=Wan-UMT5-cache Ovis/VAE/Yak-video=absent lambda=%.3f",
            self.model_id,
            self.visual_num_layers,
            self.visual_hidden_dim,
            self.visual_grid_size,
            self.visual_grid_size,
            self.num_views,
            self.action_num_layers,
            self.action_expert.hidden_dim,
            self.action_visual_layer_map,
            self.loss_lambda_foresight,
        )

    @property
    def student_backbone(self) -> SingleFrameVideoMAEBackbone:
        return self.dit["videomae"]

    @property
    def teacher_backbone(self) -> Optional[VideoMAEClipTeacher]:
        return None if self._teacher_ref is None else self._teacher_ref[0]

    def apply_trainable_policy(self) -> None:
        """Keep the DINO-only mask-token policy out of the VideoMAE subclass.

        ``JAMDINOv2Yak.__init__`` calls this hook before this subclass
        renames the registered visual module from ``dinov2`` to ``videomae``.
        VideoMAE has no structurally unused DINO mask token, so every dense
        student parameter remains trainable and this policy is intentionally
        a no-op.
        """

        return None

    @classmethod
    def from_pretrained(
        cls,
        *,
        model_id: str,
        text_embedding_cache_dir: str,
        text_context_len: int = 128,
        text_hidden_dim: int = 4096,
        proprio_dim: Optional[int] = None,
        image_size: int = 224,
        num_views: int = 2,
        local_files_only: bool = True,
        attn_implementation: Optional[str] = "sdpa",
        gradient_checkpointing: bool = False,
        action_dit_config: Optional[dict[str, Any]] = None,
        action_dit_pretrained_path: Optional[str] = None,
        skip_dit_load_from_pretrain: bool = False,
        teacher_clip_frames: int = 16,
        predictor_hidden_dim: int = 384,
        predictor_ffn_dim: int = 1536,
        predictor_num_layers: int = 2,
        predictor_num_heads: int = 6,
        action_train_shift: float = 5.0,
        action_infer_shift: float = 5.0,
        action_num_train_timesteps: int = 1000,
        loss_lambda_action: float = 1.0,
        loss_lambda_foresight: float = 0.3,
        foresight_cfg: Optional[dict[str, Any]] = None,
        device: str = "cuda",
        torch_dtype: torch.dtype = torch.bfloat16,
    ) -> "JAMVideoMAEYak":
        logger.info("Loading strict trainable VideoMAE student from %s", model_id)
        student = SingleFrameVideoMAEBackbone.from_pretrained(
            model_id,
            image_size=int(image_size),
            num_views=int(num_views),
            local_files_only=bool(local_files_only),
            attn_implementation=attn_implementation,
            device=device,
            torch_dtype=torch_dtype,
        )
        if bool(gradient_checkpointing):
            enable = getattr(student.backbone, "gradient_checkpointing_enable", None)
            if callable(enable):
                enable()

        foresight_cfg = dict(foresight_cfg or {})
        load_teacher = (
            bool(foresight_cfg.get("enabled", True))
            and bool(foresight_cfg.get("load_teacher", True))
            and os.environ.get("JAM_FORESIGHT_NO_TEACHER") != "1"
        )
        teacher = None
        if load_teacher:
            teacher = VideoMAEClipTeacher.from_pretrained(
                model_id,
                image_size=int(image_size),
                num_views=int(num_views),
                clip_frames=int(teacher_clip_frames),
                local_files_only=bool(local_files_only),
                attn_implementation=attn_implementation,
                device=device,
                torch_dtype=torch_dtype,
            )

        action_expert = ActionDiTYak.from_pretrained(
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
            model_id=model_id,
            text_embedding_cache_dir=text_embedding_cache_dir,
            text_context_len=text_context_len,
            text_hidden_dim=text_hidden_dim,
            proprio_dim=proprio_dim,
            teacher_clip_frames=teacher_clip_frames,
            predictor_hidden_dim=predictor_hidden_dim,
            predictor_ffn_dim=predictor_ffn_dim,
            predictor_num_layers=predictor_num_layers,
            predictor_num_heads=predictor_num_heads,
            action_train_shift=action_train_shift,
            action_infer_shift=action_infer_shift,
            action_num_train_timesteps=action_num_train_timesteps,
            loss_lambda_action=loss_lambda_action,
            loss_lambda_foresight=loss_lambda_foresight,
            foresight_cfg=foresight_cfg,
            device=device,
            torch_dtype=torch_dtype,
        )

    @torch.no_grad()
    def _encode_teacher(self, future_clip: torch.Tensor) -> torch.Tensor:
        if self.teacher_backbone is None:
            raise RuntimeError(
                "Foresight training requires the frozen VideoMAE clip teacher."
            )
        target = self.teacher_backbone(future_clip)
        return self._packed_to_view_major(target).detach()

    def _prepare_action_prefix(
        self,
        layerwise_text: Sequence[torch.Tensor],
        layerwise_visual: Sequence[torch.Tensor],
        text_mask: torch.Tensor,
    ) -> dict[str, Any]:
        return self.action_expert.prepare_layerwise_prefix(
            layerwise_text_tokens=layerwise_text,
            layerwise_visual_tokens=layerwise_visual,
            text_mask=text_mask,
            visual_grid_height=self.visual_grid_size,
            visual_grid_width=self.visual_grid_size,
            num_views=self.num_views,
            fusion_type=VIDEOMAE_YAK18_FUSION_TYPE,
            source_name="VideoMAE-B",
        )

    def _predict_action(
        self,
        action_tokens: torch.Tensor,
        timestep: torch.Tensor,
        prefix_cache: dict[str, Any],
    ) -> torch.Tensor:
        return self.action_expert.forward_layerwise_prefix_mot(
            action_tokens=action_tokens,
            timestep=timestep / float(self.action_num_train_timesteps),
            prefix_cache=prefix_cache,
            expected_fusion_type=VIDEOMAE_YAK18_FUSION_TYPE,
            source_name="VideoMAE-B",
        )

    def training_loss(self, sample: dict[str, Any]):
        required = {"video", "action", "proprio", "context", "context_mask"}
        missing = required - set(sample)
        if missing:
            raise ValueError(
                f"Strict VideoMAE training sample is missing {sorted(missing)}."
            )
        video = sample["video"]
        action = sample["action"]
        if video.ndim != 5 or int(video.shape[1]) != 3 or int(video.shape[2]) != 17:
            raise ValueError(
                "Strict VideoMAE video must be [B,3,17,H,2W] for t...t+16, "
                f"got {tuple(video.shape)}."
            )
        if action.ndim != 3 or int(action.shape[-1]) != self.action_expert.action_dim:
            raise ValueError(
                f"Action must be [B,T,{self.action_expert.action_dim}], "
                f"got {tuple(action.shape)}."
            )
        batch_size = int(video.shape[0])
        current_image = video[:, :, 0].to(
            device=self.device, dtype=self.torch_dtype, non_blocking=True
        )
        future_clip = video[:, :, 1:].to(
            device=self.device, dtype=self.torch_dtype, non_blocking=True
        )
        current_tokens, layerwise_visual = self._encode_student(current_image)
        _, text_mask, layerwise_text = self._prepare_text(
            batch_size=batch_size,
            prompt=sample.get("prompt"),
            context=sample.get("context"),
            context_mask=sample.get("context_mask"),
            proprio=sample.get("proprio"),
        )
        prefix_cache = self._prepare_action_prefix(
            layerwise_text, layerwise_visual, text_mask
        )

        action = action.to(
            device=self.device, dtype=self.torch_dtype, non_blocking=True
        )
        action_is_pad = sample.get("action_is_pad")
        if action_is_pad is not None:
            action_is_pad = action_is_pad.to(
                device=self.device, dtype=torch.bool, non_blocking=True
            )
        loss_action = self._action_loss(action, action_is_pad, prefix_cache)

        loss_foresight = loss_action.new_zeros(())
        if self.foresight_enabled:
            future_tokens = self._encode_teacher(future_clip)
            image_is_pad = sample.get("image_is_pad")
            if image_is_pad is None:
                target_valid = torch.ones(
                    batch_size, device=self.device, dtype=torch.bool
                )
            else:
                image_is_pad = image_is_pad.to(
                    device=self.device, dtype=torch.bool, non_blocking=True
                )
                if tuple(image_is_pad.shape) != (batch_size, 17):
                    raise ValueError(
                        "Strict VideoMAE image_is_pad must be [B,17], got "
                        f"{tuple(image_is_pad.shape)}."
                    )
                # A padded member changes the bidirectional clip representation;
                # supervise only clips whose complete future is real.
                target_valid = ~image_is_pad[:, 1:].any(dim=1)
            loss_foresight = self._foresight_loss(
                current_tokens, future_tokens, target_valid
            )

        total = (
            self.loss_lambda_action * loss_action
            + self.loss_lambda_foresight * loss_foresight
        )
        return total, {
            "loss_action": self.loss_lambda_action
            * float(loss_action.detach().item()),
            "loss_foresight": self.loss_lambda_foresight
            * float(loss_foresight.detach().item()),
        }

    def save_checkpoint(self, path, optimizer=None, step=None):
        payload = {
            "format": VIDEOMAE_YAK18_CHECKPOINT_FORMAT,
            "dit": self.dit.state_dict(),
            "step": step,
            "model_id": self.model_id,
            "action_fusion_type": VIDEOMAE_YAK18_FUSION_TYPE,
            "visual_num_layers": self.visual_num_layers,
            "action_num_layers": self.action_num_layers,
            "action_visual_layer_map": self.action_visual_layer_map,
            "action_expert_family": type(self.action_expert).__name__,
            "text_condition_source": "wan_umt5_xxl_cached",
            "text_context_len": self.text_context_len,
            "text_hidden_dim": self.text_hidden_dim,
            "future_frame_offsets": self.future_frame_offsets,
            "target_horizon": 16,
            "student_input": VIDEOMAE_STUDENT_INPUT,
            "teacher_input": VIDEOMAE_TEACHER_INPUT,
            "teacher_clip_frames": self.teacher_clip_frames,
            "loss_lambda_foresight": self.loss_lambda_foresight,
            "foresight_predictor_config": self.foresight_predictor_config,
            "strict_absent_modules": [
                "ovis_condition_encoder",
                "image_vae",
                "yak_video_expert",
            ],
        }
        if optimizer is not None:
            payload["optimizer"] = optimizer.state_dict()
        torch.save(payload, path)

    def load_checkpoint(self, path, optimizer=None, mmap: bool = False):
        load_kwargs: dict[str, Any] = {"map_location": "cpu"}
        if mmap:
            load_kwargs["mmap"] = True
        payload = torch.load(path, **load_kwargs)
        if payload.get("format") != VIDEOMAE_YAK18_CHECKPOINT_FORMAT:
            raise ValueError(
                "Unsupported strict-VideoMAE checkpoint format: "
                f"{payload.get('format')!r}."
            )
        checks = {
            "action_fusion_type": VIDEOMAE_YAK18_FUSION_TYPE,
            "visual_num_layers": self.visual_num_layers,
            "action_num_layers": self.action_num_layers,
            "action_visual_layer_map": self.action_visual_layer_map,
            "text_condition_source": "wan_umt5_xxl_cached",
            "future_frame_offsets": self.future_frame_offsets,
            "target_horizon": 16,
            "student_input": VIDEOMAE_STUDENT_INPUT,
            "teacher_input": VIDEOMAE_TEACHER_INPUT,
            "teacher_clip_frames": self.teacher_clip_frames,
        }
        for key, expected in checks.items():
            saved = payload.get(key)
            if key in {"action_visual_layer_map", "future_frame_offsets"}:
                saved = tuple(int(value) for value in saved or ())
            if saved != expected:
                raise ValueError(
                    f"Strict-VideoMAE checkpoint {key} mismatch: "
                    f"checkpoint={saved!r}, model={expected!r}."
                )
        self.dit.load_state_dict(payload["dit"], strict=True)
        if optimizer is not None and "optimizer" in payload:
            optimizer.load_state_dict(payload["optimizer"])
        return payload


__all__ = [
    "VIDEOMAE_STUDENT_INPUT",
    "VIDEOMAE_TEACHER_INPUT",
    "VIDEOMAE_YAK18_CHECKPOINT_FORMAT",
    "VIDEOMAE_YAK18_FUSION_TYPE",
    "JAMVideoMAEYak",
]


# Compatibility aliases for historical configs and imports.
VideoMAEYakImageWAM = JAMVideoMAEYak

__all__ = list(dict.fromkeys(__all__ + ['JAMVideoMAEYak', 'VideoMAEYakImageWAM']))
