"""DINOv2 visual student/teacher for the native DINO JAM policies.

The policy-side visual student and the frozen future-image teacher are loaded
from the same DINOv2 checkpoint.  The trainable student encodes only the
current observation; its per-layer patch K/V condition a native ActionDiTDINO
expert copied from the DINO blocks, and its final dense tokens feed the
foresight projector.  The frozen teacher encodes only the future endpoint.
Language enters as frozen cached FLUX.2 Qwen3-4B instruction tokens.
"""

from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from jam_multimodal.utils.logging_config import get_logger

from .foresight_yak import ForesightProjector

logger = get_logger(__name__)


class NativeDinoActionContainer(nn.Module):
    """Train/checkpoint container for the opt-in DINO-native policy.

    The generic JAMPolicy ``MoT`` requires two experts with an identical block
    protocol.  HuggingFace DINO and ActionDiTDINO intentionally use a dedicated
    directed-attention implementation instead, so this container exposes only
    modules that actually train or need to be checkpointed.
    """

    def __init__(
        self,
        *,
        ssl_student: nn.Module,
        action_expert: nn.Module,
        foresight_projector: Optional[nn.Module],
    ) -> None:
        super().__init__()
        self.ssl_student = ssl_student
        self.action_expert = action_expert
        if foresight_projector is not None:
            self.foresight_projector = foresight_projector


class DenseImageSSLBackbone(nn.Module):
    """Tensor-native dense image SSL encoder for horizontally packed cameras."""

    def __init__(
        self,
        backbone: nn.Module,
        *,
        image_size: int,
        num_views: int,
        pixel_mean: list[float] | tuple[float, ...],
        pixel_std: list[float] | tuple[float, ...],
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.image_size = int(image_size)
        self.num_views = int(num_views)
        if self.image_size <= 0 or self.num_views <= 0:
            raise ValueError(
                f"`image_size/num_views` must be positive, got {image_size}/{num_views}."
            )

        patch_size = getattr(backbone.config, "patch_size", None)
        if isinstance(patch_size, (tuple, list)):
            if len(patch_size) != 2 or int(patch_size[0]) != int(patch_size[1]):
                raise ValueError(f"Only square SSL patches are supported, got {patch_size}.")
            patch_size = int(patch_size[0])
        if patch_size is None:
            raise ValueError("The SSL backbone config must expose `patch_size`.")
        self.patch_size = int(patch_size)
        if self.image_size % self.patch_size != 0:
            raise ValueError(
                f"image_size={self.image_size} must be divisible by patch_size={self.patch_size}."
            )
        self.grid_size = self.image_size // self.patch_size
        self.hidden_size = int(backbone.config.hidden_size)
        self.num_layers = int(backbone.config.num_hidden_layers)
        self.tokens_per_view = self.grid_size * self.grid_size

        mean = torch.as_tensor(pixel_mean, dtype=torch.float32)
        std = torch.as_tensor(pixel_std, dtype=torch.float32)
        if mean.shape != (3,) or std.shape != (3,) or bool((std <= 0).any().item()):
            raise ValueError(
                f"`pixel_mean/std` must be three valid channel values, got {mean}/{std}."
            )
        self.register_buffer("pixel_mean", mean.view(1, 3, 1, 1), persistent=True)
        self.register_buffer("pixel_std", std.view(1, 3, 1, 1), persistent=True)

    @classmethod
    def from_pretrained(
        cls,
        model_id: str,
        *,
        image_size: int,
        num_views: int,
        local_files_only: bool,
        attn_implementation: Optional[str],
        device: torch.device | str,
        torch_dtype: torch.dtype,
    ) -> "DenseImageSSLBackbone":
        from transformers import AutoImageProcessor, AutoModel

        processor = AutoImageProcessor.from_pretrained(
            model_id,
            local_files_only=bool(local_files_only),
            use_fast=False,
        )
        load_kwargs = {"local_files_only": bool(local_files_only)}
        if attn_implementation is not None:
            load_kwargs["attn_implementation"] = str(attn_implementation)
        backbone = AutoModel.from_pretrained(model_id, **load_kwargs)
        model = cls(
            backbone,
            image_size=int(image_size),
            num_views=int(num_views),
            pixel_mean=list(processor.image_mean),
            pixel_std=list(processor.image_std),
        )
        return model.to(device=device, dtype=torch_dtype)

    def _prepare_views(self, images: torch.Tensor) -> torch.Tensor:
        if images.ndim == 3:
            images = images.unsqueeze(0)
        if images.ndim != 4 or int(images.shape[1]) != 3:
            raise ValueError(
                f"`images` must be [B,3,H,W] or [3,H,W], got {tuple(images.shape)}."
            )
        batch_size, _, height, width = images.shape
        expected_width = self.image_size * self.num_views
        if (int(height), int(width)) != (self.image_size, expected_width):
            raise ValueError(
                "External SSL input must preserve the packed camera layout: "
                f"expected HxW=({self.image_size},{expected_width}), "
                f"got ({height},{width})."
            )
        views = (
            images.reshape(
                batch_size,
                3,
                self.image_size,
                self.num_views,
                self.image_size,
            )
            .permute(0, 3, 1, 2, 4)
            .reshape(batch_size * self.num_views, 3, self.image_size, self.image_size)
        )
        views = ((views.float() + 1.0) * 0.5).clamp_(0.0, 1.0)
        views = (views - self.pixel_mean) / self.pixel_std
        dtype = next(self.backbone.parameters()).dtype
        return views.to(device=next(self.backbone.parameters()).device, dtype=dtype)

    def _stitch_dense_tokens(
        self,
        tokens: torch.Tensor,
        *,
        batch_size: int,
    ) -> torch.Tensor:
        expected = self.tokens_per_view
        if tokens.ndim != 3 or int(tokens.shape[0]) != batch_size * self.num_views:
            raise ValueError(
                "Unexpected SSL token batch: "
                f"got {tuple(tokens.shape)}, batch={batch_size}, views={self.num_views}."
            )
        if int(tokens.shape[1]) < expected:
            raise ValueError(
                f"SSL output has {tokens.shape[1]} tokens, fewer than the {expected} patch tokens."
            )
        # DINO-style encoders prepend CLS/register tokens.  Dense tokens are
        # always the trailing HxW patch grid.
        patches = tokens[:, -expected:, :]
        patches = patches.reshape(
            batch_size,
            self.num_views,
            self.grid_size,
            self.grid_size,
            self.hidden_size,
        )
        # Match the dataset's horizontal camera packing: for every grid row,
        # camera 0 occupies the left columns and camera 1 the right columns.
        packed = torch.cat([patches[:, view] for view in range(self.num_views)], dim=2)
        return packed.reshape(
            batch_size,
            self.grid_size * self.grid_size * self.num_views,
            self.hidden_size,
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        batch_size = 1 if images.ndim == 3 else int(images.shape[0])
        pixel_values = self._prepare_views(images)
        outputs = self.backbone(
            pixel_values=pixel_values,
            output_hidden_states=False,
            return_dict=True,
        )
        return self._stitch_dense_tokens(outputs.last_hidden_state, batch_size=batch_size)


def get_external_ssl_teacher(model) -> Optional[DenseImageSSLBackbone]:
    ref = getattr(model, "_external_ssl_teacher_ref", None)
    if ref is None:
        return None
    return ref[0]


def attach_ovis_external_ssl(
    model,
    ssl_cfg: Dict[str, Any],
    foresight_cfg: Optional[Dict[str, Any]] = None,
) -> None:
    """Replace the Ovis visual/action path with a native DINOv2 student/teacher pair."""

    ssl_cfg = dict(ssl_cfg or {})
    foresight_cfg = dict(foresight_cfg or {})
    if not bool(ssl_cfg.get("enabled", False)):
        return
    if str(getattr(model, "stack", "")) != "ovis_u1":
        raise ValueError("External image SSL is currently implemented only for Ovis-U1 JAMPolicy.")

    model_id = str(ssl_cfg.get("model_id", "")).strip()
    if not model_id:
        raise ValueError("`ssl_vision.model_id` is required.")
    image_size = int(ssl_cfg.get("image_size", 224))
    num_views = int(ssl_cfg.get("num_views", 2))
    local_files_only = bool(ssl_cfg.get("local_files_only", True))
    attn_implementation = ssl_cfg.get("attn_implementation", "sdpa")
    backend = str(ssl_cfg.get("backend", "huggingface"))
    action_fusion_type = str(
        ssl_cfg.get("action_fusion_type", "")
    ).strip().lower()
    valid_fusion_types = {
        "native_dino12_mot",
        "native_dinov2_large24_mot",
    }
    if action_fusion_type not in valid_fusion_types:
        raise ValueError(
            "`ssl_vision.action_fusion_type` must be one of "
            f"{sorted(valid_fusion_types)}, got {action_fusion_type!r}."
        )
    if backend != "huggingface":
        raise ValueError(
            "Native DINO/ActionDINO MoT currently requires the "
            "HuggingFace DINOv2 backend."
        )
    student = DenseImageSSLBackbone.from_pretrained(
        model_id,
        image_size=image_size,
        num_views=num_views,
        local_files_only=local_files_only,
        attn_implementation=attn_implementation,
        device=model.device,
        torch_dtype=model.torch_dtype,
    )
    if bool(ssl_cfg.get("gradient_checkpointing", False)):
        enable = getattr(student.backbone, "gradient_checkpointing_enable", None)
        if callable(enable):
            enable()

    if hasattr(model.mot, "ssl_student"):
        raise RuntimeError("The Ovis MoT already has an `ssl_student` module.")
    text_condition_source = str(
        ssl_cfg.get("text_condition_source", "")
    ).strip().lower()
    if text_condition_source != "qwen3_4b_flux2_cached":
        raise ValueError(
            "`ssl_vision.text_condition_source` must be 'qwen3_4b_flux2_cached', "
            f"got {text_condition_source!r}."
        )
    if getattr(model, "ovis_condition_encoder", None) is not None:
        raise ValueError(
            "Native SSL + Qwen3 text must set "
            "`load_condition_encoder=false`; "
            "otherwise Ovis would remain a second multimodal conditioner."
        )
    qwen_text_cache_dir = str(
        ssl_cfg.get("qwen_text_cache_dir", "")
    ).strip()
    if not qwen_text_cache_dir:
        raise ValueError(
            "`ssl_vision.qwen_text_cache_dir` is required for cached "
            "Qwen3-4B text."
        )
    qwen_text_cache_dir = str(
        Path(
            os.path.expandvars(os.path.expanduser(qwen_text_cache_dir))
        ).resolve()
    )
    qwen_context_len = int(ssl_cfg.get("qwen_context_len", 512))
    qwen_hidden_dim = int(ssl_cfg.get("qwen_hidden_dim", 7680))
    if qwen_context_len <= 0 or qwen_hidden_dim <= 0:
        raise ValueError(
            "Qwen context length/hidden dimension must be positive, got "
            f"{qwen_context_len}/{qwen_hidden_dim}."
        )
    if not Path(qwen_text_cache_dir).is_dir():
        raise FileNotFoundError(
            f"Qwen3 text cache directory does not exist: {qwen_text_cache_dir}"
        )

    foresight_enabled = bool(foresight_cfg.get("enabled", False))
    load_teacher = foresight_enabled and bool(foresight_cfg.get("load_teacher", True))
    if os.environ.get("JAM_FORESIGHT_NO_TEACHER") == "1":
        load_teacher = False
    teacher = None
    if load_teacher:
        # The frozen future-endpoint teacher is an exact copy of the pretrained
        # student, taken before any student update.
        teacher = copy.deepcopy(student)
        teacher.eval()
        teacher.requires_grad_(False)
        teacher.to(device=model.device, dtype=model.torch_dtype)
    object.__setattr__(
        model,
        "_external_ssl_teacher_ref",
        None if teacher is None else [teacher],
    )

    projector = None
    if foresight_enabled:
        offsets = [int(value) for value in foresight_cfg.get("offsets", [1])]
        if offsets != [1]:
            raise ValueError(
                "External future-image SSL currently supports strict future-only offsets=[1], "
                f"got {offsets}."
            )
        projector = ForesightProjector(
            in_dim=student.hidden_size,
            out_dim=student.hidden_size,
            hidden_dim=int(
                foresight_cfg.get("projector_hidden_dim", student.hidden_size)
            ),
            ffn_dim=int(
                foresight_cfg.get("projector_ffn_dim", 4 * student.hidden_size)
            ),
            num_layers=int(foresight_cfg.get("projector_num_layers", 2)),
            num_heads=int(foresight_cfg.get("projector_num_heads", 12)),
            num_targets=1,
        ).to(device=model.device, dtype=model.torch_dtype)
    else:
        offsets = []

    native_cfg = dict(ssl_cfg.get("native_action") or {})
    configured_text_dim = int(
        native_cfg.get("text_dim", qwen_hidden_dim)
    )
    if configured_text_dim != qwen_hidden_dim:
        raise ValueError(
            "ActionDiTDino text_dim must equal the cached Qwen3 hidden "
            f"dimension: {configured_text_dim} vs {qwen_hidden_dim}."
        )
    native_cfg["text_dim"] = qwen_hidden_dim
    initialize_from_backbone = native_cfg.pop(
        "initialize_from_backbone",
        native_cfg.pop("initialize_from_dino", True),
    )
    if not bool(initialize_from_backbone):
        raise ValueError(
            "Native SSL action recipes require initialization from the "
            "matching visual backbone."
        )
    from .action_dit_dino import ActionDiTDino12, ActionDiTDinoLarge24

    old_action = model.action_expert
    action_kwargs = {
        "action_dim": int(old_action.action_dim),
        "text_dim": int(native_cfg.pop("text_dim", 4096)),
        "max_action_horizon": int(
            native_cfg.pop(
                "max_action_horizon",
                getattr(old_action, "max_action_horizon", 64),
            )
        ),
        **native_cfg,
    }
    action_class = {
        "native_dino12_mot": ActionDiTDino12,
        "native_dinov2_large24_mot": ActionDiTDinoLarge24,
    }[action_fusion_type]
    action_expert = action_class.from_dino_backbone(
        student.backbone,
        **action_kwargs,
    )
    native_container = NativeDinoActionContainer(
        ssl_student=student,
        action_expert=action_expert,
        foresight_projector=projector,
    ).to(device=model.device, dtype=model.torch_dtype)
    # Keep all three aliases synchronized: Trainer optimizes `dit`,
    # checkpointing reads `mot`, and policy forward reads `action_expert`.
    model.mot = native_container
    model.dit = native_container
    model.action_expert = action_expert
    del old_action

    # Preserve the established proprio-as-one-condition-token protocol:
    # proprio is first embedded into the source text width, then the
    # shared ActionDiTDino text adapter maps every condition token to
    # the matching action hidden width (768 for DINO-B, 1024 for
    # DINO-L). The source width is the frozen FLUX.2 Qwen3 context
    # width 7680.
    model.text_dim = qwen_hidden_dim
    if model.proprio_dim is not None:
        model.proprio_encoder = nn.Linear(
            int(model.proprio_dim), qwen_hidden_dim
        ).to(device=model.device, dtype=model.torch_dtype)

    model.external_ssl_enabled = True
    model.external_ssl_backend = backend
    model.external_ssl_model_id = model_id
    model.external_ssl_image_size = image_size
    model.external_ssl_num_views = num_views
    model.external_ssl_grid_size = student.grid_size
    model.external_ssl_hidden_size = student.hidden_size
    model.external_ssl_num_layers = int(student.num_layers)
    model.external_ssl_action_fusion_type = action_fusion_type
    model.external_ssl_action_layer_map = []
    model.external_ssl_action_num_layers = len(model.action_expert.blocks)
    model.external_ssl_text_condition_source = text_condition_source
    model.external_ssl_qwen_text_cache_dir = qwen_text_cache_dir
    model.external_ssl_qwen_context_len = qwen_context_len
    model.external_ssl_qwen_hidden_dim = qwen_hidden_dim
    object.__setattr__(model, "_external_ssl_text_cache", {})
    model.external_ssl_hidden_state_source = "pre_block_norm1_native_qkv"
    model.external_ssl_hidden_state_norm = "native_final_layernorm_only"
    model.external_ssl_layer_map_algorithm = {
        "native_dino12_mot": "identity_12_to_12",
        "native_dinov2_large24_mot": "identity_24_to_24",
    }[action_fusion_type]
    model.external_ssl_dense_token_selection = "trailing_patch_grid"
    model.external_ssl_attn_implementation = str(
        getattr(student.backbone.config, "_attn_implementation", "eager")
    )
    model.foresight_enabled = foresight_enabled
    model.foresight_load_teacher = load_teacher
    model.foresight_offsets = offsets
    model.foresight_lambda = float(foresight_cfg.get("lambda_foresight", 0.3))
    model.foresight_drop_target = True
    action_expert = model.action_expert
    model.external_ssl_native_action_class = type(action_expert).__name__
    model.external_ssl_native_action_hidden_size = int(
        action_expert.hidden_size
    )
    model.external_ssl_native_action_num_heads = int(
        action_expert.num_heads
    )
    model.external_ssl_native_action_head_dim = int(action_expert.head_dim)
    model.external_ssl_native_action_mlp_hidden_size = int(
        action_expert.blocks[0].mlp.fc1.out_features
    )
    model.external_ssl_native_action_text_dim = int(action_expert.text_dim)
    model.external_ssl_native_action_max_horizon = int(
        action_expert.max_action_horizon
    )
    model.external_ssl_native_visual_kv_source = "pre_block_norm1_native_qkv"
    model.external_ssl_native_text_prefix = "static_projected_qwen3_4b_7680"
    model.external_ssl_native_action_initialization = "copy_matching_dino_blocks_v1"
    model.external_ssl_native_action_position = "learned_1d_no_rope"
    model.external_ssl_native_action_time_factor = float(
        action_expert.time_factor
    )
    model.external_ssl_native_visual_hidden_size = int(
        getattr(action_expert, "visual_hidden_size", student.hidden_size)
    )
    model.external_ssl_native_visual_num_layers = int(
        getattr(action_expert, "visual_num_layers", student.num_layers)
    )
    model.external_ssl_native_action_layer_map = [
        int(index)
        for index in getattr(
            action_expert,
            "action_layer_map",
            range(len(action_expert.blocks)),
        )
    ]
    model.external_ssl_native_action_parameter_count = sum(
        int(parameter.numel())
        for parameter in action_expert.parameters()
    )
    model.external_ssl_native_camera_encoding = "independent_with_cls"
    model.external_ssl_native_action_visual_tokens = "patch_only"
    model.external_ssl_teacher_input = "future_endpoint_t16"
    model.external_ssl_student_input = str(
        getattr(student, "student_input_mode", "current_endpoint_t")
    )
    if hasattr(model, "model_paths"):
        model.model_paths["external_ssl_student"] = model_id
        model.model_paths["external_ssl_teacher"] = model_id

    logger.info(
        "External SSL Ovis policy attached: backend=%s model=%s student=trainable "
        "teacher_loaded=%s dense_grid=%dx%d views=%d hidden=%d layers=%d "
        "fusion=%s future_only=%s attention=%s lambda=%.3f text_source=%s",
        backend,
        model_id,
        teacher is not None,
        student.grid_size,
        student.grid_size,
        student.num_views,
        student.hidden_size,
        student.num_layers,
        action_fusion_type,
        foresight_enabled,
        model.external_ssl_attn_implementation,
        model.foresight_lambda,
        text_condition_source,
    )


__all__ = [
    "DenseImageSSLBackbone",
    "NativeDinoActionContainer",
    "attach_ovis_external_ssl",
    "get_external_ssl_teacher",
]
