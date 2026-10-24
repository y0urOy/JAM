"""Strict DINOv2 replacement for the visual backbone of the Ovis-U1 policy.

This variant deliberately contains no Ovis condition encoder, image VAE, or
Yak video expert.  A trainable DINOv2 student supplies current-image states to
the unchanged pretrained 18-layer ActionDiTYak through directed layerwise MoT.
Language comes from the already established frozen Wan-UMT5 embedding cache;
the online frozen DINOv2 teacher sees only the real t+16 endpoint.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from jam_multimodal.utils.logging_config import get_logger

from .action_dit_yak import ActionDiTYak
from .external_ssl import DenseImageSSLBackbone
from .foresight_yak import ForesightProjector
from .jam_qwen35 import (
    future_target_valid_mask,
    uniform_endpoint_layer_map,
)
from .schedulers.scheduler_continuous import WanContinuousFlowMatchScheduler

logger = get_logger(__name__)


DINO_YAK18_CHECKPOINT_FORMAT = "dinov2_imagewam_yak18_strict_v1"
DINO_YAK18_FUSION_TYPE = "dinov2_yak18_layerwise_mot"


class JAMDINOv2Yak(nn.Module):
    """DINOv2 visual policy with the original ImageWAM ActionDiTYak18."""

    expected_frame_offsets = (4,)

    def __init__(
        self,
        *,
        student_backbone: DenseImageSSLBackbone,
        teacher_backbone: Optional[DenseImageSSLBackbone],
        action_expert: ActionDiTYak,
        model_id: str,
        text_embedding_cache_dir: str,
        text_context_len: int,
        text_hidden_dim: int,
        proprio_dim: Optional[int],
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
        expected_action_hidden_dim: int = 1536,
    ) -> None:
        super().__init__()
        self.model_id = str(model_id)
        self.text_embedding_cache_dir = str(
            Path(os.path.expandvars(os.path.expanduser(text_embedding_cache_dir))).resolve()
        )
        self.text_context_len = int(text_context_len)
        self.text_hidden_dim = int(text_hidden_dim)
        self.proprio_dim = None if proprio_dim is None else int(proprio_dim)
        self.device = torch.device(device)
        self.torch_dtype = torch_dtype
        self.action_num_train_timesteps = int(action_num_train_timesteps)
        self.loss_lambda_action = float(loss_lambda_action)
        self.loss_lambda_foresight = float(loss_lambda_foresight)

        if not isinstance(action_expert, ActionDiTYak) and not (
            getattr(action_expert, "block_protocol", None) == "flux2"
            and callable(getattr(action_expert, "prepare_layerwise_prefix", None))
            and callable(getattr(action_expert, "forward_layerwise_prefix_mot", None))
        ):
            raise TypeError(
                "Strict DINO replacement requires ActionDiTYak or a compatible "
                "ActionDiTFlux2 external-prefix expert, got "
                f"{type(action_expert).__name__}."
            )
        if student_backbone.num_views != 2:
            raise ValueError(
                "The current LIBERO strict DINO recipe requires two views, got "
                f"{student_backbone.num_views}."
            )
        if action_expert.hidden_dim != int(expected_action_hidden_dim):
            raise ValueError(
                "Strict DINO replacement action width mismatch: expected "
                f"{int(expected_action_hidden_dim)}, got {action_expert.hidden_dim}."
            )

        self.visual_num_layers = int(student_backbone.num_layers)
        self.action_num_layers = len(action_expert.blocks)
        self.action_visual_layer_map = uniform_endpoint_layer_map(
            self.visual_num_layers, self.action_num_layers
        )
        self.visual_grid_size = int(student_backbone.grid_size)
        self.num_views = int(student_backbone.num_views)
        self.image_size = int(student_backbone.image_size)
        self.visual_hidden_dim = int(student_backbone.hidden_size)

        foresight_cfg = dict(foresight_cfg or {})
        self.foresight_enabled = bool(foresight_cfg.get("enabled", True))
        self.foresight_load_teacher = bool(foresight_cfg.get("load_teacher", True))
        self.foresight_teacher_mode = str(
            foresight_cfg.get("teacher_mode", "frozen_pretrained")
        ).strip().lower()
        if self.foresight_teacher_mode != "frozen_pretrained":
            raise ValueError(
                "Strict DINO currently supports only teacher_mode='frozen_pretrained'."
            )
        expected_offsets = tuple(int(value) for value in self.expected_frame_offsets)
        offsets = tuple(
            int(value)
            for value in foresight_cfg.get("frame_offsets", expected_offsets)
        )
        if offsets != expected_offsets:
            raise ValueError(
                "Strict ImageWAM target semantics require "
                f"frame_offsets={list(expected_offsets)}, got {offsets}."
            )
        self.future_frame_offsets = offsets

        visual_projector = nn.Sequential(
            nn.LayerNorm(self.visual_hidden_dim),
            nn.Linear(self.visual_hidden_dim, action_expert.hidden_dim),
        )
        text_projector = nn.Sequential(
            nn.LayerNorm(self.text_hidden_dim),
            nn.Linear(self.text_hidden_dim, action_expert.hidden_dim),
        )
        proprio_projector: nn.Module
        if self.proprio_dim is None:
            proprio_projector = nn.Identity()
        else:
            proprio_projector = nn.Sequential(
                nn.Linear(self.proprio_dim, action_expert.hidden_dim),
                nn.SiLU(),
                nn.Linear(action_expert.hidden_dim, action_expert.hidden_dim),
            )
        # Always construct the projector. Evaluation disables teacher/loss but
        # still needs to load a training checkpoint strictly.
        foresight_projector = ForesightProjector(
            in_dim=self.visual_hidden_dim,
            out_dim=self.visual_hidden_dim,
            hidden_dim=int(predictor_hidden_dim),
            ffn_dim=int(predictor_ffn_dim),
            num_layers=int(predictor_num_layers),
            num_heads=int(predictor_num_heads),
            num_targets=1,
        )
        self.foresight_predictor_config = {
            "hidden_dim": int(predictor_hidden_dim),
            "ffn_dim": int(predictor_ffn_dim),
            "num_layers": int(predictor_num_layers),
            "num_heads": int(predictor_num_heads),
        }

        # Trainer compatibility: every trainable component lives below `dit`.
        self.dit = nn.ModuleDict(
            {
                "dinov2": student_backbone,
                "visual_projector": visual_projector,
                "text_projector": text_projector,
                "proprio_projector": proprio_projector,
                "action_expert": action_expert,
                "foresight_projector": foresight_projector,
            }
        ).to(device=self.device, dtype=self.torch_dtype)
        # DINO's learned mask token is only consumed by masked-image
        # pretraining.  The dense LIBERO forward never supplies a mask, so the
        # parameter is structurally unreachable and must not be advertised to
        # the optimizer/DeepSpeed as trainable.
        self.apply_trainable_policy()

        # Keep the frozen teacher outside the registered module graph so it is
        # absent from optimizer partitioning and weights-only checkpoints.
        self._teacher_ref = None
        if teacher_backbone is not None:
            teacher_backbone.eval().requires_grad_(False)
            teacher_backbone.to(device=self.device, dtype=self.torch_dtype)
            self._teacher_ref = [teacher_backbone]

        self.train_action_scheduler = WanContinuousFlowMatchScheduler(
            num_train_timesteps=self.action_num_train_timesteps,
            shift=float(action_train_shift),
        )
        self.infer_action_scheduler = WanContinuousFlowMatchScheduler(
            num_train_timesteps=self.action_num_train_timesteps,
            shift=float(action_infer_shift),
        )
        self._text_cache: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}

        logger.info(
            "%s strict replacement: DINO=%s layers=%d hidden=%d "
            "grid=%dx%d views=%d action=%s layers=%d hidden=%d map=%s "
            "text_cache=[%d,%d] teacher_loaded=%s teacher_input=t+16 "
            "lambda=%.3f",
            type(self).__name__,
            self.model_id,
            self.visual_num_layers,
            self.visual_hidden_dim,
            self.visual_grid_size,
            self.visual_grid_size,
            self.num_views,
            type(self.action_expert).__name__,
            self.action_num_layers,
            self.action_expert.hidden_dim,
            self.action_visual_layer_map,
            self.text_context_len,
            self.text_hidden_dim,
            self.teacher_backbone is not None,
            self.loss_lambda_foresight,
        )

    @property
    def student_backbone(self) -> DenseImageSSLBackbone:
        return self.dit["dinov2"]

    @property
    def teacher_backbone(self) -> Optional[DenseImageSSLBackbone]:
        return None if self._teacher_ref is None else self._teacher_ref[0]

    @property
    def action_expert(self) -> ActionDiTYak:
        return self.dit["action_expert"]

    @property
    def visual_projector(self) -> nn.Module:
        return self.dit["visual_projector"]

    @property
    def text_projector(self) -> nn.Module:
        return self.dit["text_projector"]

    @property
    def proprio_projector(self) -> nn.Module:
        return self.dit["proprio_projector"]

    @property
    def foresight_projector(self) -> ForesightProjector:
        return self.dit["foresight_projector"]

    def apply_trainable_policy(self) -> None:
        """Keep structurally unused DINO pretraining parameters frozen.

        Trainer calls this again after its generic ``dit.requires_grad_(True)``
        step, so the policy holds both before optimizer construction and while
        training under Accelerate/DeepSpeed.
        """

        embeddings = getattr(self.student_backbone.backbone, "embeddings", None)
        mask_token = getattr(embeddings, "mask_token", None)
        if isinstance(mask_token, torch.Tensor):
            mask_token.requires_grad_(False)

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
    ) -> "JAMDINOv2Yak":
        logger.info("Loading strict trainable DINOv2 student from %s", model_id)
        student = DenseImageSSLBackbone.from_pretrained(
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
            # DINOv2-B is small enough that a matched frozen copy is both
            # unambiguous and cheaper than constructing unrelated machinery.
            import copy

            teacher = copy.deepcopy(student).eval().requires_grad_(False)

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

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        if self.teacher_backbone is not None:
            self.teacher_backbone.to(*args, **kwargs)
        if args and isinstance(args[0], (str, torch.device)):
            self.device = torch.device(args[0])
        elif kwargs.get("device") is not None:
            self.device = torch.device(kwargs["device"])
        return self

    def _packed_to_view_major(self, tokens: torch.Tensor) -> torch.Tensor:
        """Convert DenseImageSSL's row-packed cameras to view-major tokens."""

        batch_size, token_count, hidden_dim = tokens.shape
        expected = self.num_views * self.visual_grid_size * self.visual_grid_size
        if token_count != expected or hidden_dim != self.visual_hidden_dim:
            raise ValueError(
                "Unexpected DINO patch tensor: "
                f"{tuple(tokens.shape)}, expected [B,{expected},{self.visual_hidden_dim}]."
            )
        packed = tokens.reshape(
            batch_size,
            self.visual_grid_size,
            self.num_views * self.visual_grid_size,
            hidden_dim,
        )
        return torch.cat(
            [
                view.reshape(
                    batch_size,
                    self.visual_grid_size * self.visual_grid_size,
                    hidden_dim,
                )
                for view in packed.split(self.visual_grid_size, dim=2)
            ],
            dim=1,
        )

    def _encode_student(
        self, current_image: torch.Tensor
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
        final_tokens, hidden_states = self.student_backbone(
            current_image, output_hidden_states=True
        )
        final_tokens = self._packed_to_view_major(final_tokens)
        selected = tuple(
            self.visual_projector(
                self._packed_to_view_major(hidden_states[layer_idx])
            )
            for layer_idx in self.action_visual_layer_map
        )
        return final_tokens, selected

    @torch.no_grad()
    def _encode_teacher(self, future_image: torch.Tensor) -> torch.Tensor:
        if self.teacher_backbone is None:
            raise RuntimeError("Foresight training requires the frozen DINOv2 teacher.")
        return self._packed_to_view_major(self.teacher_backbone(future_image)).detach()

    def _read_text_cache(self, prompt: str) -> tuple[torch.Tensor, torch.Tensor]:
        prompt = str(prompt)
        cached = self._text_cache.get(prompt)
        if cached is not None:
            return cached
        digest = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        cache_path = Path(self.text_embedding_cache_dir) / (
            f"{digest}.t5_len{self.text_context_len}.wan22ti2v5b.pt"
        )
        if not cache_path.is_file():
            raise FileNotFoundError(
                f"Missing strict-DINO UMT5 cache: {cache_path}. "
                "Use the same precomputed LIBERO text cache as JAMWan."
            )
        payload = torch.load(cache_path, map_location="cpu")
        context = payload["context"]
        mask = payload["mask"].to(dtype=torch.bool)
        if tuple(context.shape) != (self.text_context_len, self.text_hidden_dim):
            raise ValueError(
                f"Cached context shape mismatch in {cache_path}: {tuple(context.shape)}."
            )
        if tuple(mask.shape) != (self.text_context_len,):
            raise ValueError(
                f"Cached mask shape mismatch in {cache_path}: {tuple(mask.shape)}."
            )
        context = context.clone()
        context[~mask] = 0
        # Match Wan/FastWAM training semantics: padded embeddings are zeros but
        # the fixed-length context remains visible to attention.
        mask = torch.ones_like(mask)
        self._text_cache[prompt] = (context, mask)
        return context, mask

    def _prepare_text(
        self,
        *,
        batch_size: int,
        prompt: str | Sequence[str] | None,
        context: Optional[torch.Tensor],
        context_mask: Optional[torch.Tensor],
        proprio: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, tuple[torch.Tensor, ...]]:
        if context is None or context_mask is None:
            if prompt is None:
                raise ValueError("Strict DINO requires prompt or cached context/context_mask.")
            prompts = [prompt] if isinstance(prompt, str) else list(prompt)
            if len(prompts) == 1 and batch_size > 1:
                prompts = prompts * batch_size
            if len(prompts) != batch_size:
                raise ValueError(
                    f"Prompt batch must contain {batch_size} entries, got {len(prompts)}."
                )
            pairs = [self._read_text_cache(str(item)) for item in prompts]
            context = torch.stack([pair[0] for pair in pairs], dim=0)
            context_mask = torch.stack([pair[1] for pair in pairs], dim=0)
        else:
            if context.ndim == 2:
                context = context.unsqueeze(0)
            if context_mask.ndim == 1:
                context_mask = context_mask.unsqueeze(0)
        expected_context = (batch_size, self.text_context_len, self.text_hidden_dim)
        if tuple(context.shape) != expected_context:
            raise ValueError(
                f"UMT5 context must be {expected_context}, got {tuple(context.shape)}."
            )
        if tuple(context_mask.shape) != expected_context[:2]:
            raise ValueError(
                f"UMT5 context mask must be {expected_context[:2]}, "
                f"got {tuple(context_mask.shape)}."
            )
        text = self.text_projector(
            context.to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
        )
        text_mask = context_mask.to(
            device=self.device, dtype=torch.bool, non_blocking=True
        )

        if self.proprio_dim is not None:
            if proprio is None:
                raise ValueError("Strict DINO requires proprio when proprio_dim is configured.")
            if proprio.ndim == 3:
                proprio = proprio[:, 0]
            elif proprio.ndim == 1:
                proprio = proprio.unsqueeze(0)
            if tuple(proprio.shape) != (batch_size, self.proprio_dim):
                raise ValueError(
                    f"Proprio must resolve to [B,{self.proprio_dim}], got {tuple(proprio.shape)}."
                )
            proprio_token = self.proprio_projector(
                proprio.to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
            ).unsqueeze(1)
            text = torch.cat((text, proprio_token), dim=1)
            text_mask = torch.cat(
                (
                    text_mask,
                    torch.ones(batch_size, 1, device=self.device, dtype=torch.bool),
                ),
                dim=1,
            )
        return text, text_mask, tuple(text for _ in range(self.action_num_layers))

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
            fusion_type=DINO_YAK18_FUSION_TYPE,
            source_name="DINOv2",
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
            expected_fusion_type=DINO_YAK18_FUSION_TYPE,
            source_name="DINOv2",
        )

    def _action_loss(
        self,
        action: torch.Tensor,
        action_is_pad: Optional[torch.Tensor],
        prefix_cache: dict[str, Any],
    ) -> torch.Tensor:
        batch_size = int(action.shape[0])
        noise = torch.randn_like(action)
        timestep = self.train_action_scheduler.sample_training_t(
            batch_size=batch_size, device=self.device, dtype=action.dtype
        )
        noisy_action = self.train_action_scheduler.add_noise(action, noise, timestep)
        target = self.train_action_scheduler.training_target(action, noise, timestep)
        prediction = self._predict_action(noisy_action, timestep, prefix_cache)
        token_loss = F.mse_loss(
            prediction.float(), target.float(), reduction="none"
        ).mean(dim=-1)
        if action_is_pad is None:
            per_sample = token_loss.mean(dim=1)
        else:
            if tuple(action_is_pad.shape) != tuple(token_loss.shape):
                raise ValueError(
                    f"action_is_pad must be {tuple(token_loss.shape)}, "
                    f"got {tuple(action_is_pad.shape)}."
                )
            valid = (~action_is_pad).to(device=self.device, dtype=token_loss.dtype)
            per_sample = (token_loss * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1)
        weight = self.train_action_scheduler.training_weight(timestep).to(
            device=self.device, dtype=per_sample.dtype
        )
        return (per_sample * weight).mean()

    def _foresight_loss(
        self,
        current_tokens: torch.Tensor,
        future_tokens: torch.Tensor,
        valid: torch.Tensor,
    ) -> torch.Tensor:
        prediction = self.foresight_projector.head(
            self.foresight_projector.trunk(current_tokens), 0
        )
        per_sample = (
            1.0
            - F.cosine_similarity(
                prediction.float(), future_tokens.float(), dim=-1
            )
        ).mean(dim=1)
        valid = valid.to(device=self.device, dtype=per_sample.dtype)
        if bool(valid.any().item()):
            return (per_sample * valid).sum() / valid.sum()
        return prediction.float().sum() * 0.0

    def training_loss(self, sample: dict[str, Any]):
        required = {"video", "action", "proprio", "context", "context_mask"}
        missing = required - set(sample)
        if missing:
            raise ValueError(f"Strict DINO training sample is missing {sorted(missing)}.")
        video = sample["video"]
        action = sample["action"]
        if video.ndim != 5 or int(video.shape[1]) != 3 or int(video.shape[2]) != 2:
            raise ValueError(
                "Strict DINO endpoint-only video must be [B,3,2,H,2W], got "
                f"{tuple(video.shape)}."
            )
        if action.ndim != 3 or int(action.shape[-1]) != self.action_expert.action_dim:
            raise ValueError(
                f"Action must be [B,T,{self.action_expert.action_dim}], got {tuple(action.shape)}."
            )
        batch_size = int(video.shape[0])
        current_image = video[:, :, 0].to(
            device=self.device, dtype=self.torch_dtype, non_blocking=True
        )
        future_image = video[:, :, 1].to(
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
            future_tokens = self._encode_teacher(future_image)
            target_valid = future_target_valid_mask(
                sample.get("image_is_pad"),
                (1,),
                batch_size=batch_size,
                device=self.device,
            )[:, 0]
            loss_foresight = self._foresight_loss(
                current_tokens, future_tokens, target_valid
            )
        total = (
            self.loss_lambda_action * loss_action
            + self.loss_lambda_foresight * loss_foresight
        )
        return total, {
            "loss_action": self.loss_lambda_action * float(loss_action.detach().item()),
            "loss_foresight": self.loss_lambda_foresight
            * float(loss_foresight.detach().item()),
        }

    @torch.no_grad()
    def infer_action(
        self,
        prompt: str,
        input_image: torch.Tensor,
        action_horizon: int,
        proprio: Optional[torch.Tensor] = None,
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
        if input_image.ndim != 4 or int(input_image.shape[1]) != 3:
            raise ValueError(
                f"input_image must be [B,3,H,2W], got {tuple(input_image.shape)}."
            )
        batch_size = int(input_image.shape[0])
        current = input_image.to(
            device=self.device, dtype=self.torch_dtype, non_blocking=True
        )
        _, layerwise_visual = self._encode_student(current)
        _, text_mask, layerwise_text = self._prepare_text(
            batch_size=batch_size,
            prompt=prompt,
            context=None,
            context_mask=None,
            proprio=proprio,
        )
        prefix_cache = self._prepare_action_prefix(
            layerwise_text, layerwise_visual, text_mask
        )

        generator = (
            None
            if seed is None
            else torch.Generator(device=rand_device).manual_seed(int(seed))
        )
        action = torch.randn(
            (batch_size, int(action_horizon), self.action_expert.action_dim),
            generator=generator,
            device=rand_device,
            dtype=torch.float32,
        ).to(device=self.device, dtype=self.torch_dtype)
        timesteps, deltas = self.infer_action_scheduler.build_inference_schedule(
            num_inference_steps=int(num_inference_steps),
            device=self.device,
            dtype=action.dtype,
            shift_override=sigma_shift,
        )
        for timestep, delta in zip(timesteps, deltas):
            prediction = self._predict_action(
                action,
                timestep.expand(batch_size),
                prefix_cache,
            )
            action = self.infer_action_scheduler.step(prediction, delta, action)
        action = action.detach().to(device="cpu", dtype=torch.float32)
        return {"action": action[0] if batch_size == 1 else action}

    def save_checkpoint(self, path, optimizer=None, step=None):
        payload = {
            "format": DINO_YAK18_CHECKPOINT_FORMAT,
            "dit": self.dit.state_dict(),
            "step": step,
            "model_id": self.model_id,
            "action_fusion_type": DINO_YAK18_FUSION_TYPE,
            "visual_num_layers": self.visual_num_layers,
            "action_num_layers": self.action_num_layers,
            "action_visual_layer_map": self.action_visual_layer_map,
            "action_expert_family": type(self.action_expert).__name__,
            "text_condition_source": "wan_umt5_xxl_cached",
            "text_context_len": self.text_context_len,
            "text_hidden_dim": self.text_hidden_dim,
            "future_frame_offsets": self.future_frame_offsets,
            "teacher_input": "future_endpoint_t16",
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
        if payload.get("format") != DINO_YAK18_CHECKPOINT_FORMAT:
            raise ValueError(
                f"Unsupported strict-DINO checkpoint format: {payload.get('format')!r}."
            )
        checks = {
            "action_fusion_type": DINO_YAK18_FUSION_TYPE,
            "visual_num_layers": self.visual_num_layers,
            "action_num_layers": self.action_num_layers,
            "action_visual_layer_map": self.action_visual_layer_map,
            "text_condition_source": "wan_umt5_xxl_cached",
            "future_frame_offsets": self.future_frame_offsets,
        }
        for key, expected in checks.items():
            saved = payload.get(key)
            if key in {"action_visual_layer_map", "future_frame_offsets"}:
                saved = tuple(int(value) for value in saved or ())
            if saved != expected:
                raise ValueError(
                    f"Strict-DINO checkpoint {key} mismatch: "
                    f"checkpoint={saved!r}, model={expected!r}."
                )
        self.dit.load_state_dict(payload["dit"], strict=True)
        if optimizer is not None and "optimizer" in payload:
            optimizer.load_state_dict(payload["optimizer"])
        return payload

    def forward(self, *args, **kwargs):
        return self.training_loss(*args, **kwargs)


__all__ = [
    "DINO_YAK18_CHECKPOINT_FORMAT",
    "DINO_YAK18_FUSION_TYPE",
    "JAMDINOv2Yak",
]


# Compatibility aliases for historical configs and imports.
DinoYakImageWAM = JAMDINOv2Yak

__all__ = list(dict.fromkeys(__all__ + ['DinoYakImageWAM', 'JAMDINOv2Yak']))
