"""External self-supervised visual student/teacher for the Ovis-U1 policy.

The policy-side visual student and the frozen future-image teacher are loaded
from the same dense image SSL checkpoint (DINOv2 by default).  The trainable
student encodes only the current observation and its patch tokens replace the
VAE patch input to the existing Yak video/action MoT.  The frozen teacher
encodes only the future endpoint.  This keeps the Ovis language stack and the
official 18-layer ActionDiTYak fixed while avoiding a cross-representation
Yak-to-SSL distillation target.
"""

from __future__ import annotations

import copy
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from jam_multimodal.utils.logging_config import get_logger

from .foresight_yak import ForesightProjector

logger = get_logger(__name__)


VIDEOMAE_CLIP_TEACHER_INPUT = (
    "real_future_clip_t1_to_t16_last_tubelet"
)
VIDEOMAE_ENDPOINT_TEACHER_INPUT = (
    "future_endpoint_t16_folded_tubelet2"
)


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


class OfficialDINOv3BackboneAdapter(nn.Module):
    """Expose Meta's official DINOv3 ViT through the HF-like interface below.

    Ovis-U1 currently requires Transformers 4.51.3, while native DINOv3 support
    starts in Transformers 4.56 and conflicts with Ovis' custom AIMv2
    registration.  Keeping the official DINOv3 implementation behind this
    small adapter lets both models coexist without changing the Ovis
    environment.
    """

    def __init__(self, backbone: nn.Module) -> None:
        super().__init__()
        self.backbone = backbone
        self.config = SimpleNamespace(
            patch_size=int(backbone.patch_size),
            hidden_size=int(backbone.embed_dim),
            num_hidden_layers=int(backbone.n_blocks),
            _attn_implementation="sdpa_official",
        )

    def forward(
        self,
        *,
        pixel_values: torch.Tensor,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ):
        if not return_dict:
            raise ValueError("OfficialDINOv3BackboneAdapter requires return_dict=True.")
        if output_hidden_states:
            layer_indices = list(range(int(self.config.num_hidden_layers)))
            layers = self.backbone.get_intermediate_layers(
                pixel_values,
                n=layer_indices,
                norm=True,
            )
            if len(layers) != int(self.config.num_hidden_layers):
                raise RuntimeError(
                    "Official DINOv3 returned an unexpected hidden-state count: "
                    f"{len(layers)}."
                )
            # DenseImageSSLBackbone follows the Transformers convention that
            # hidden_states[0] is the embedding output and layers start at 1.
            # The embedding value is not consumed by the policy.
            hidden_states = (layers[0], *layers)
            dense = layers[-1]
        else:
            features = self.backbone.forward_features(pixel_values)
            dense = features["x_norm_patchtokens"]
            hidden_states = None
        return SimpleNamespace(
            last_hidden_state=dense,
            hidden_states=hidden_states,
        )


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
        intermediate_hidden_states_normalized: bool = False,
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
        self.intermediate_hidden_states_normalized = bool(
            intermediate_hidden_states_normalized
        )
        self.intermediate_hidden_state_norm = (
            "already_normalized"
            if self.intermediate_hidden_states_normalized
            else "backbone_final_layernorm_each"
        )
        if (
            not self.intermediate_hidden_states_normalized
            and not isinstance(getattr(backbone, "layernorm", None), nn.Module)
        ):
            raise ValueError(
                "The SSL backbone must expose `layernorm` when intermediate "
                "hidden states are not already normalized."
            )

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
        backend: str = "huggingface",
        official_repo_path: Optional[str] = None,
        pixel_mean: Optional[list[float] | tuple[float, ...]] = None,
        pixel_std: Optional[list[float] | tuple[float, ...]] = None,
    ) -> "DenseImageSSLBackbone":
        backend = str(backend).strip().lower()
        if backend == "huggingface":
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
            resolved_mean = list(processor.image_mean)
            resolved_std = list(processor.image_std)
        elif backend == "dinov3_official":
            repo_path = Path(
                official_repo_path
                or os.environ.get("JAM_DINOV3_REPO", "./third_party/dinov3")
            ).expanduser().resolve()
            if not (repo_path / "dinov3" / "__init__.py").is_file():
                raise FileNotFoundError(
                    f"Official DINOv3 source tree is missing: {repo_path}."
                )
            weights_path = Path(model_id).expanduser().resolve()
            if not weights_path.is_file():
                raise FileNotFoundError(
                    f"Official DINOv3 weights are missing: {weights_path}."
                )
            if str(repo_path) not in sys.path:
                sys.path.insert(0, str(repo_path))
            from dinov3.hub.backbones import dinov3_vitl16

            official = dinov3_vitl16(
                pretrained=True,
                weights=str(weights_path),
            )
            backbone = OfficialDINOv3BackboneAdapter(official)
            if pixel_mean is None or pixel_std is None:
                raise ValueError(
                    "The official DINOv3 backend requires explicit pixel_mean/pixel_std."
                )
            resolved_mean = list(pixel_mean)
            resolved_std = list(pixel_std)
        else:
            raise ValueError(
                "Unsupported external SSL backend "
                f"{backend!r}; expected 'huggingface' or 'dinov3_official'."
            )
        model = cls(
            backbone,
            image_size=int(image_size),
            num_views=int(num_views),
            pixel_mean=resolved_mean,
            pixel_std=resolved_std,
            intermediate_hidden_states_normalized=(
                backend == "dinov3_official"
            ),
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

    def forward(
        self,
        images: torch.Tensor,
        *,
        output_hidden_states: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
        batch_size = 1 if images.ndim == 3 else int(images.shape[0])
        pixel_values = self._prepare_views(images)
        outputs = self.backbone(
            pixel_values=pixel_values,
            output_hidden_states=bool(output_hidden_states),
            return_dict=True,
        )
        dense = self._stitch_dense_tokens(outputs.last_hidden_state, batch_size=batch_size)
        if not output_hidden_states:
            return dense
        if outputs.hidden_states is None:
            raise RuntimeError("SSL backbone did not return requested hidden states.")
        raw_layers = tuple(outputs.hidden_states[1:])
        if self.intermediate_hidden_states_normalized:
            normalized_layers = raw_layers
        else:
            # HuggingFace DINOv2 exposes post-block hidden states before its
            # final LayerNorm.  Apply that pretrained norm to every depth and
            # reuse `last_hidden_state` for the final layer so the layerwise
            # route ends bit-identically to the established final-token path.
            normalized_layers = tuple(
                self.backbone.layernorm(hidden)
                for hidden in raw_layers[:-1]
            ) + (outputs.last_hidden_state,)
        layers = tuple(
            self._stitch_dense_tokens(hidden, batch_size=batch_size)
            for hidden in normalized_layers
        )
        if len(layers) != self.num_layers:
            raise RuntimeError(
                f"Expected {self.num_layers} SSL layers, received {len(layers)}."
            )
        return dense, layers


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
    """Replace Ovis VAE visual inputs with a matched SSL student/teacher pair."""

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
        ssl_cfg.get("action_fusion_type", "layerwise_mot")
    ).strip().lower()
    valid_fusion_types = {
        "layerwise_mot",
        "direct_static_kv",
        "direct_layerwise_kv",
        "native_dino12_mot",
        "native_dino24_mot",
        "native_dinov2_large24_mot",
        "native_videomae12_mot",
    }
    if action_fusion_type not in valid_fusion_types:
        raise ValueError(
            "`ssl_vision.action_fusion_type` must be one of "
            f"{sorted(valid_fusion_types)}, got {action_fusion_type!r}."
        )
    native_videomae_action = action_fusion_type == "native_videomae12_mot"
    if native_videomae_action:
        if backend != "videomae_huggingface":
            raise ValueError(
                "Native VideoMAE/ActionVideoMAE requires "
                "`ssl_vision.backend=videomae_huggingface`."
            )
        from .external_videomae import SingleFrameVideoMAEBackbone

        student = SingleFrameVideoMAEBackbone.from_pretrained(
            model_id,
            image_size=image_size,
            num_views=num_views,
            local_files_only=local_files_only,
            attn_implementation=attn_implementation,
            device=model.device,
            torch_dtype=model.torch_dtype,
        )
    else:
        student = DenseImageSSLBackbone.from_pretrained(
            model_id,
            image_size=image_size,
            num_views=num_views,
            local_files_only=local_files_only,
            attn_implementation=attn_implementation,
            device=model.device,
            torch_dtype=model.torch_dtype,
            backend=backend,
            official_repo_path=ssl_cfg.get("official_repo_path"),
            pixel_mean=ssl_cfg.get("pixel_mean"),
            pixel_std=ssl_cfg.get("pixel_std"),
        )
    if bool(ssl_cfg.get("gradient_checkpointing", False)):
        enable = getattr(student.backbone, "gradient_checkpointing_enable", None)
        if callable(enable):
            enable()

    if hasattr(model.mot, "ssl_student"):
        raise RuntimeError("The Ovis MoT already has an `ssl_student` module.")
    native_dino_action = action_fusion_type in {
        "native_dino12_mot",
        "native_dino24_mot",
        "native_dinov2_large24_mot",
        "native_videomae12_mot",
    }
    text_condition_source = str(
        ssl_cfg.get("text_condition_source", "ovis_multimodal")
    ).strip().lower()
    valid_text_condition_sources = {
        "ovis_multimodal",
        "qwen3_4b_flux2_cached",
    }
    if text_condition_source not in valid_text_condition_sources:
        raise ValueError(
            "`ssl_vision.text_condition_source` must be one of "
            f"{sorted(valid_text_condition_sources)}, got "
            f"{text_condition_source!r}."
        )
    qwen_native_ssl_fusions = {
        "native_dino12_mot",
        "native_dino24_mot",
        "native_dinov2_large24_mot",
        "native_videomae12_mot",
    }
    if (
        text_condition_source == "qwen3_4b_flux2_cached"
        and action_fusion_type not in qwen_native_ssl_fusions
    ):
        raise ValueError(
            "Cached Qwen3-4B text requires a native DINO or VideoMAE "
            "action expert; "
            f"got action_fusion_type={action_fusion_type!r}."
        )
    qwen_text_cache_dir = None
    qwen_context_len = 0
    qwen_hidden_dim = 0
    if text_condition_source == "qwen3_4b_flux2_cached":
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
    if not native_dino_action:
        model.mot.add_module("ssl_student", student)
    if not native_dino_action:
        model.video_expert.enable_external_ssl_input(student.hidden_size)
    action_layer_map: list[int] = []
    if action_fusion_type == "direct_layerwise_kv":
        action_depth = len(model.action_expert.blocks)
        ssl_depth = int(student.num_layers)
        if action_depth <= 0 or ssl_depth <= 0:
            raise ValueError(
                "DINO layerwise action fusion requires positive SSL/action depths, "
                f"got ssl={ssl_depth}, action={action_depth}."
            )
        configured_map = ssl_cfg.get("action_layer_map", "uniform")
        if configured_map in (None, "uniform"):
            if action_depth == 1:
                action_layer_map = [ssl_depth - 1]
            else:
                action_layer_map = [
                    int(
                        layer_idx * (ssl_depth - 1) / (action_depth - 1)
                        + 0.5
                    )
                    for layer_idx in range(action_depth)
                ]
        elif isinstance(configured_map, (list, tuple)):
            action_layer_map = [int(value) for value in configured_map]
        else:
            raise ValueError(
                "`ssl_vision.action_layer_map` must be 'uniform' or a list "
                f"of zero-based DINO layer indices, got {configured_map!r}."
            )
        if len(action_layer_map) != action_depth:
            raise ValueError(
                "DINO/action layer map length mismatch: "
                f"expected {action_depth}, got {len(action_layer_map)}."
            )
        if any(index < 0 or index >= ssl_depth for index in action_layer_map):
            raise ValueError(
                f"DINO/action layer map must stay in [0,{ssl_depth - 1}], "
                f"got {action_layer_map}."
            )
        if action_layer_map != sorted(action_layer_map):
            raise ValueError(
                "DINO/action layer map must be monotonically non-decreasing, "
                f"got {action_layer_map}."
            )
        if (
            action_depth > 1
            and (
                action_layer_map[0] != 0
                or action_layer_map[-1] != ssl_depth - 1
            )
        ):
            raise ValueError(
                "DINO/action layer map must cover the first and last DINO "
                f"layers, got {action_layer_map}."
            )

    foresight_enabled = bool(foresight_cfg.get("enabled", False))
    load_teacher = foresight_enabled and bool(foresight_cfg.get("load_teacher", True))
    if os.environ.get("JAM_FORESIGHT_NO_TEACHER") == "1":
        load_teacher = False
    videomae_teacher_input = str(
        foresight_cfg.get(
            "teacher_input",
            VIDEOMAE_CLIP_TEACHER_INPUT,
        )
    ).strip().lower()
    if native_videomae_action:
        valid_videomae_teacher_inputs = {
            VIDEOMAE_CLIP_TEACHER_INPUT,
            VIDEOMAE_ENDPOINT_TEACHER_INPUT,
        }
        if videomae_teacher_input not in valid_videomae_teacher_inputs:
            raise ValueError(
                "Native VideoMAE foresight teacher_input must be one of "
                f"{sorted(valid_videomae_teacher_inputs)}, got "
                f"{videomae_teacher_input!r}."
            )
    teacher = None
    if load_teacher:
        if (
            native_videomae_action
            and videomae_teacher_input == VIDEOMAE_CLIP_TEACHER_INPUT
        ):
            from .foresight_yak import VideoMAEClipTeacher

            teacher = VideoMAEClipTeacher.from_pretrained(
                model_id,
                image_size=image_size,
                num_views=num_views,
                clip_frames=int(foresight_cfg.get("teacher_clip_frames", 16)),
                local_files_only=local_files_only,
                attn_implementation=attn_implementation,
                device=model.device,
                torch_dtype=model.torch_dtype,
            )
        else:
            # Strict endpoint supervision uses the exact same folded
            # VideoMAE image adapter as the current-frame student. At the
            # patch stem this is Conv3d([t+16,t+16]) without constructing a
            # repeated clip; deepcopy freezes the pretrained initialization
            # before any student update.
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
        if not native_dino_action:
            model.mot.foresight_projector = projector
    else:
        offsets = []

    if native_dino_action:
        if not native_videomae_action and backend != "huggingface":
            raise ValueError(
                "Native DINO/ActionDINO MoT currently requires the "
                "HuggingFace DINOv2 backend."
            )
        native_cfg = dict(ssl_cfg.get("native_action") or {})
        if text_condition_source == "qwen3_4b_flux2_cached":
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
        from .action_dit_dino import (
            ActionDiTDino12,
            ActionDiTDino24,
            ActionDiTDinoLarge24,
        )

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
        if native_videomae_action:
            from .action_dit_videomae import ActionDiTVideoMAE12

            action_expert = ActionDiTVideoMAE12.from_videomae_backbone(
                student.backbone,
                **action_kwargs,
            )
        else:
            action_class = {
                "native_dino12_mot": ActionDiTDino12,
                "native_dino24_mot": ActionDiTDino24,
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

        if text_condition_source == "qwen3_4b_flux2_cached":
            # Preserve the established proprio-as-one-condition-token protocol:
            # proprio is first embedded into the source text width, then the
            # shared ActionDiTDino text adapter maps every condition token to
            # the matching action hidden width (768 for DINO-B, 1024 for
            # DINO-L). Only the source width changes from Ovis 4096 to the
            # frozen FLUX.2 Qwen3 context width 7680.
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
    model.external_ssl_action_layer_map = action_layer_map
    model.external_ssl_action_num_layers = len(model.action_expert.blocks)
    model.external_ssl_text_condition_source = text_condition_source
    model.external_ssl_qwen_text_cache_dir = qwen_text_cache_dir
    model.external_ssl_qwen_context_len = qwen_context_len
    model.external_ssl_qwen_hidden_dim = qwen_hidden_dim
    object.__setattr__(model, "_external_ssl_text_cache", {})
    model.external_ssl_hidden_state_source = (
        "pre_block_norm1_native_qkv"
        if native_dino_action
        else f"post_block_1_to_{student.num_layers}"
    )
    model.external_ssl_hidden_state_norm = (
        "native_final_layernorm_only"
        if native_dino_action
        else student.intermediate_hidden_state_norm
    )
    model.external_ssl_layer_map_algorithm = (
        (
            {
                "native_dino24_mot": "repeat_each_visual_layer_twice_12_to_24",
                "native_dino12_mot": "identity_12_to_12",
                "native_dinov2_large24_mot": "identity_24_to_24",
                "native_videomae12_mot": "identity_12_to_12",
            }[action_fusion_type]
        )
        if native_dino_action
        else "endpoint_nearest_v1"
    )
    model.external_ssl_dense_token_selection = "trailing_patch_grid"
    model.external_ssl_attn_implementation = str(
        getattr(student.backbone.config, "_attn_implementation", "eager")
    )
    model.foresight_enabled = foresight_enabled
    model.foresight_load_teacher = load_teacher
    model.foresight_offsets = offsets
    model.foresight_lambda = float(foresight_cfg.get("lambda_foresight", 0.3))
    model.foresight_drop_target = True
    if native_dino_action:
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
        model.external_ssl_native_visual_kv_source = (
            "pre_block_norm1_768_to_1536_adapter_action_kv"
            if action_fusion_type == "native_dino24_mot"
            else (
                "pre_block_layernorm_before_native_qkv_no_cls"
                if native_videomae_action
                else "pre_block_norm1_native_qkv"
            )
        )
        model.external_ssl_native_text_prefix = (
            "static_projected_qwen3_4b_7680"
            if text_condition_source == "qwen3_4b_flux2_cached"
            else "static_projected_ovis_4096"
        )
        model.external_ssl_native_action_initialization = (
            "net2wider_2x_repeat_depth_layerscale_half_v1"
            if action_fusion_type == "native_dino24_mot"
            else (
                "copy_matching_videomae_blocks_v1"
                if native_videomae_action
                else "copy_matching_dino_blocks_v1"
            )
        )
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
        model.external_ssl_native_camera_encoding = (
            "independent_no_cls_single_frame_folded_tubelet2"
            if native_videomae_action
            else "independent_with_cls"
        )
        model.external_ssl_native_action_visual_tokens = "patch_only"
        model.external_ssl_teacher_input = (
            videomae_teacher_input
            if native_videomae_action
            else "future_endpoint_t16"
        )
        model.external_ssl_student_input = str(
            getattr(student, "student_input_mode", "current_endpoint_t")
        )
    if hasattr(model, "model_paths"):
        model.model_paths["external_ssl_student"] = model_id
        model.model_paths["external_ssl_teacher"] = model_id

    logger.info(
        "External SSL Ovis policy attached: backend=%s model=%s student=trainable "
        "teacher_loaded=%s dense_grid=%dx%d views=%d hidden=%d layers=%d "
        "fusion=%s layer_map=%s "
        "future_only=%s attention=%s lambda=%.3f text_source=%s",
        backend,
        model_id,
        teacher is not None,
        student.grid_size,
        student.grid_size,
        student.num_views,
        student.hidden_size,
        student.num_layers,
        action_fusion_type,
        action_layer_map if action_layer_map else "n/a",
        foresight_enabled,
        model.external_ssl_attn_implementation,
        model.foresight_lambda,
        text_condition_source,
    )


__all__ = [
    "DenseImageSSLBackbone",
    "NativeDinoActionContainer",
    "VIDEOMAE_CLIP_TEACHER_INPUT",
    "VIDEOMAE_ENDPOINT_TEACHER_INPUT",
    "attach_ovis_external_ssl",
    "get_external_ssl_teacher",
]
