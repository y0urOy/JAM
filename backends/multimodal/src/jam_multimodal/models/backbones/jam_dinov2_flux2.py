"""DINOv2 visual replacement paired with official ActionDiTFlux2-25.

The trainable student and frozen endpoint teacher are DINOv2-B/14.  No FLUX
video transformer, image VAE, or image-generation loss is constructed.  The
official slim FLUX.2 action expert keeps its 5 double + 20 single blocks and
pretrained initialization; projected Qwen3 text and DINO layer states replace
the FLUX video prefix that action queries read through directed layerwise MoT.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any, Optional

import torch

from .action_dit_flux2 import ActionDiTFlux2
from .jam_dinov2_yak import JAMDINOv2Yak
from .external_ssl import DenseImageSSLBackbone


DINO_FLUX2_CHECKPOINT_FORMAT = "dinov2_imagewam_flux2_25_strict_v1"
DINO_FLUX2_FUSION_TYPE = "dinov2_flux2_25_layerwise_mot"


class JAMDINOv2Flux2(JAMDINOv2Yak):
    """DINOv2-B visual policy with the official FLUX.2 ActionDiT-25."""

    @classmethod
    def from_pretrained(
        cls,
        *,
        model_id: str,
        qwen_text_cache_dir: str,
        qwen_context_len: int = 512,
        qwen_hidden_dim: int = 7680,
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
    ) -> "JAMDINOv2Flux2":
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
            import copy

            teacher = copy.deepcopy(student).eval().requires_grad_(False)

        action_expert = ActionDiTFlux2.from_pretrained(
            action_dit_config=dict(action_dit_config or {}),
            action_dit_pretrained_path=action_dit_pretrained_path,
            skip_dit_load_from_pretrain=bool(skip_dit_load_from_pretrain),
            device=device,
            torch_dtype=torch_dtype,
        )
        if len(action_expert.double_blocks) != 5 or len(action_expert.single_blocks) != 20:
            raise ValueError(
                "Strict FLUX.2 comparison requires ActionDiTFlux2 5+20 blocks, "
                f"got {len(action_expert.double_blocks)}+"
                f"{len(action_expert.single_blocks)}."
            )
        return cls(
            student_backbone=student,
            teacher_backbone=teacher,
            action_expert=action_expert,
            model_id=model_id,
            text_embedding_cache_dir=qwen_text_cache_dir,
            text_context_len=int(qwen_context_len),
            text_hidden_dim=int(qwen_hidden_dim),
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
            expected_action_hidden_dim=1024,
        )

    @property
    def action_expert(self) -> ActionDiTFlux2:
        return self.dit["action_expert"]

    def _read_text_cache(self, prompt: str) -> tuple[torch.Tensor, torch.Tensor]:
        prompt = str(prompt)
        cached = self._text_cache.get(prompt)
        if cached is not None:
            return cached
        digest = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        cache_path = Path(self.text_embedding_cache_dir) / (
            f"{digest}.qwen3_flux2_len{self.text_context_len}.pt"
        )
        if not cache_path.is_file():
            raise FileNotFoundError(
                f"Missing FLUX.2 Qwen3 text cache: {cache_path}. Run "
                "scripts/flux2/precompute_flux2_qwen3_embeds.py first."
            )
        payload = torch.load(cache_path, map_location="cpu")
        context = payload["text_hidden_states"]
        mask = payload["text_attention_mask"].to(dtype=torch.bool)
        if tuple(context.shape) != (self.text_context_len, self.text_hidden_dim):
            raise ValueError(
                f"Cached Qwen3 context shape mismatch in {cache_path}: "
                f"{tuple(context.shape)}."
            )
        if tuple(mask.shape) != (self.text_context_len,):
            raise ValueError(
                f"Cached Qwen3 mask shape mismatch in {cache_path}: "
                f"{tuple(mask.shape)}."
            )
        self._text_cache[prompt] = (context, mask)
        return context, mask

    def _prepare_action_prefix(
        self,
        layerwise_text,
        layerwise_visual,
        text_mask,
    ) -> dict[str, Any]:
        return self.action_expert.prepare_layerwise_prefix(
            layerwise_text_tokens=layerwise_text,
            layerwise_visual_tokens=layerwise_visual,
            text_mask=text_mask,
            visual_grid_height=self.visual_grid_size,
            visual_grid_width=self.visual_grid_size,
            num_views=self.num_views,
            fusion_type=DINO_FLUX2_FUSION_TYPE,
            source_name="DINOv2",
        )

    def _predict_action(
        self,
        action_tokens: torch.Tensor,
        timestep: torch.Tensor,
        prefix_cache: dict[str, Any],
    ) -> torch.Tensor:
        # FLUX.2 uses unit interval timesteps, unlike the legacy Yak action
        # helper that embeds scheduler indices directly.
        return self.action_expert.forward_layerwise_prefix_mot(
            action_tokens=action_tokens,
            timestep=timestep / float(self.action_num_train_timesteps),
            prefix_cache=prefix_cache,
            expected_fusion_type=DINO_FLUX2_FUSION_TYPE,
            source_name="DINOv2",
        )

    def training_loss(self, sample: dict[str, Any]):
        if "text_hidden_states" not in sample or "text_attention_mask" not in sample:
            raise ValueError(
                "DINO->ActionDiTFlux2 training requires cached Qwen3 "
                "text_hidden_states/text_attention_mask."
            )
        remapped = dict(sample)
        remapped["context"] = sample["text_hidden_states"]
        remapped["context_mask"] = sample["text_attention_mask"]
        return super().training_loss(remapped)

    def save_checkpoint(self, path, optimizer=None, step=None):
        payload = {
            "format": DINO_FLUX2_CHECKPOINT_FORMAT,
            "dit": self.dit.state_dict(),
            "step": step,
            "model_id": self.model_id,
            "action_fusion_type": DINO_FLUX2_FUSION_TYPE,
            "visual_num_layers": self.visual_num_layers,
            "action_num_layers": self.action_num_layers,
            "action_visual_layer_map": self.action_visual_layer_map,
            "action_expert_family": type(self.action_expert).__name__,
            "action_expert_blocks": {"double": 5, "single": 20},
            "text_condition_source": "qwen3_4b_flux2_cached",
            "text_context_len": self.text_context_len,
            "text_hidden_dim": self.text_hidden_dim,
            "future_frame_offsets": self.future_frame_offsets,
            "teacher_input": "future_endpoint_t16",
            "loss_lambda_foresight": self.loss_lambda_foresight,
            "foresight_predictor_config": self.foresight_predictor_config,
            "strict_absent_modules": [
                "flux2_video_expert",
                "flux2_image_vae",
                "image_generation_head",
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
        if payload.get("format") != DINO_FLUX2_CHECKPOINT_FORMAT:
            raise ValueError(
                "Unsupported DINO->FLUX2 checkpoint format: "
                f"{payload.get('format')!r}."
            )
        checks = {
            "action_fusion_type": DINO_FLUX2_FUSION_TYPE,
            "visual_num_layers": self.visual_num_layers,
            "action_num_layers": self.action_num_layers,
            "action_visual_layer_map": self.action_visual_layer_map,
            "text_condition_source": "qwen3_4b_flux2_cached",
            "future_frame_offsets": self.future_frame_offsets,
        }
        for key, expected in checks.items():
            saved = payload.get(key)
            if key in {"action_visual_layer_map", "future_frame_offsets"}:
                saved = tuple(int(value) for value in saved or ())
            if saved != expected:
                raise ValueError(
                    f"DINO->FLUX2 checkpoint {key} mismatch: "
                    f"checkpoint={saved!r}, model={expected!r}."
                )
        self.dit.load_state_dict(payload["dit"], strict=True)
        if optimizer is not None and "optimizer" in payload:
            optimizer.load_state_dict(payload["optimizer"])
        return payload


__all__ = [
    "DINO_FLUX2_CHECKPOINT_FORMAT",
    "DINO_FLUX2_FUSION_TYPE",
    "JAMDINOv2Flux2",
]


# Compatibility aliases for historical configs and imports.
DinoFlux2ImageWAM = JAMDINOv2Flux2

__all__ = list(dict.fromkeys(__all__ + ['DinoFlux2ImageWAM', 'JAMDINOv2Flux2']))
