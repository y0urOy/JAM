from __future__ import annotations

import logging
from typing import Any, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..wan22.action_dit import ActionDiT
from ..wan22.schedulers.scheduler_continuous import WanContinuousFlowMatchScheduler

logger = logging.getLogger(__name__)


QWEN_LEGACY_CHECKPOINT_FORMAT = "qwen35_fastwam_v1"
QWEN_CHECKPOINT_FORMAT = "qwen35_fastwam_v2"
QWEN_ACTION_FUSION_FINAL = "final_context"
QWEN_ACTION_FUSION_LAYERWISE_MOT = "layerwise_mot"


def resolve_teacher_frame_layout(
    frame_offsets: Sequence[int],
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Return teacher frame order and each offset's group index.

    Frame zero is always present as causal context. Offsets are indices on the
    already-subsampled RGB video grid, not action/state timestep indices.
    """

    offsets = tuple(int(offset) for offset in frame_offsets)
    if not offsets:
        raise ValueError("`foresight.frame_offsets` must contain at least one offset.")
    if any(offset < 0 for offset in offsets):
        raise ValueError(f"`foresight.frame_offsets` must be non-negative, got {offsets}.")
    if len(set(offsets)) != len(offsets):
        raise ValueError(f"`foresight.frame_offsets` must not contain duplicates, got {offsets}.")

    frame_indices = tuple(dict.fromkeys((0, *offsets)))
    group_by_frame = {frame_idx: group_idx for group_idx, frame_idx in enumerate(frame_indices)}
    target_group_indices = tuple(group_by_frame[offset] for offset in offsets)
    return frame_indices, target_group_indices


def future_target_valid_mask(
    image_is_pad: Optional[torch.Tensor],
    frame_offsets: Sequence[int],
    *,
    batch_size: Optional[int] = None,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Build a [B, num_offsets] mask that excludes copied future frames."""

    offsets = tuple(int(offset) for offset in frame_offsets)
    if image_is_pad is None:
        if batch_size is None:
            raise ValueError("`batch_size` is required when `image_is_pad` is None.")
        return torch.ones((batch_size, len(offsets)), dtype=torch.bool, device=device)
    if image_is_pad.ndim != 2:
        raise ValueError(
            f"`image_is_pad` must be 2D [B,T], got shape {tuple(image_is_pad.shape)}."
        )
    if offsets and max(offsets) >= image_is_pad.shape[1]:
        raise ValueError(
            "Foresight offset exceeds the available RGB video grid: "
            f"max_offset={max(offsets)}, num_video_frames={image_is_pad.shape[1]}."
        )
    offset_index = torch.as_tensor(offsets, dtype=torch.long, device=image_is_pad.device)
    return ~image_is_pad.to(dtype=torch.bool).index_select(1, offset_index)


def extract_grouped_visual_tokens(
    hidden_states: torch.Tensor,
    input_ids: torch.Tensor,
    *,
    image_token_id: int,
    num_images_per_sample: int,
    visual_tokens_per_image: int,
) -> torch.Tensor:
    """Select image placeholder states and reshape them to [B, images, tokens, D]."""

    if hidden_states.ndim != 3 or input_ids.ndim != 2:
        raise ValueError(
            "`hidden_states/input_ids` must be [B,L,D]/[B,L], got "
            f"{tuple(hidden_states.shape)} and {tuple(input_ids.shape)}."
        )
    if hidden_states.shape[:2] != input_ids.shape:
        raise ValueError(
            "Hidden/input token shapes do not match: "
            f"{tuple(hidden_states.shape[:2])} vs {tuple(input_ids.shape)}."
        )

    expected = int(num_images_per_sample) * int(visual_tokens_per_image)
    image_mask = input_ids.eq(int(image_token_id))
    counts = image_mask.sum(dim=1)
    if not bool(torch.all(counts == expected).item()):
        raise ValueError(
            "Qwen visual token count mismatch: "
            f"expected {expected} ({num_images_per_sample} images x "
            f"{visual_tokens_per_image}), got {counts.detach().cpu().tolist()}."
        )
    selected = torch.stack(
        [hidden_states[b, image_mask[b]] for b in range(hidden_states.shape[0])],
        dim=0,
    )
    return selected.reshape(
        hidden_states.shape[0],
        int(num_images_per_sample),
        int(visual_tokens_per_image),
        hidden_states.shape[-1],
    )


def forward_with_decoder_layer_capture(
    backbone: nn.Module,
    *,
    layer: Optional[int],
    model_inputs: dict[str, torch.Tensor],
    return_decoder_inputs: bool = False,
):
    """Run Qwen while retaining only one decoder layer's output.

    ``layer`` is one-based. ``None`` uses the final normalized hidden state and
    installs no hook.
    """

    captured: dict[str, torch.Tensor] = {}
    decoder_inputs: dict[int, torch.Tensor] = {}
    handles = []
    decoder_layers = backbone.language_model.layers
    if return_decoder_inputs:
        for layer_idx, decoder_layer in enumerate(decoder_layers):
            def capture_input(_module, args, _layer_idx=layer_idx):
                if not args or not isinstance(args[0], torch.Tensor):
                    raise TypeError(
                        "Captured Qwen decoder input must be the first positional tensor."
                    )
                decoder_inputs[_layer_idx] = args[0]

            handles.append(decoder_layer.register_forward_pre_hook(capture_input))

    if layer is not None:
        layer = int(layer)
        if not 1 <= layer <= len(decoder_layers):
            raise ValueError(
                f"`foresight.layer` must be in [1, {len(decoder_layers)}] or null, got {layer}."
            )

        def capture(_module, _args, output):
            if isinstance(output, (tuple, list)):
                output = output[0]
            if not isinstance(output, torch.Tensor):
                raise TypeError(
                    "Captured Qwen decoder output must be a tensor, "
                    f"got {type(output)}."
                )
            captured["hidden"] = output

        handles.append(decoder_layers[layer - 1].register_forward_hook(capture))

    try:
        output = backbone(
            **model_inputs,
            use_cache=False,
            return_dict=True,
        )
    finally:
        for handle in handles:
            handle.remove()

    layer_hidden = output.last_hidden_state if layer is None else captured.get("hidden")
    if layer_hidden is None:
        raise RuntimeError(f"Qwen decoder layer {layer} hook did not capture an output.")
    if not return_decoder_inputs:
        return output, layer_hidden
    missing = [idx for idx in range(len(decoder_layers)) if idx not in decoder_inputs]
    if missing:
        raise RuntimeError(f"Qwen decoder input hooks missed layers: {missing}.")
    return output, layer_hidden, tuple(
        decoder_inputs[idx] for idx in range(len(decoder_layers))
    )


def truncate_decoder_for_exact_early_exit(
    backbone: nn.Module,
    *,
    layer: int,
) -> int:
    """Permanently keep decoder layers ``1..layer`` on a frozen Qwen backbone.

    Qwen's text forward iterates over ``language_model.layers``. Replacing that
    ModuleList before moving the teacher to CUDA both stops computation exactly
    after the requested one-based layer and releases the unused later-layer
    weights. The layer hook still captures the same pre-final-norm tensor as a
    full forward.

    Returns the original decoder depth.
    """

    layer = int(layer)
    language_model = getattr(backbone, "language_model", None)
    decoder_layers = getattr(language_model, "layers", None)
    if not isinstance(decoder_layers, nn.ModuleList):
        raise TypeError(
            "Qwen exact early-exit requires `backbone.language_model.layers` "
            f"to be nn.ModuleList, got {type(decoder_layers)}."
        )

    original_depth = len(decoder_layers)
    if not 1 <= layer <= original_depth:
        raise ValueError(
            f"Teacher early-exit layer must be in [1, {original_depth}], got {layer}."
        )

    if layer < original_depth:
        language_model.layers = nn.ModuleList(list(decoder_layers[:layer]))

    # Qwen3.5 slices the ModuleList by config.num_hidden_layers in forward.
    # Keep both config views consistent for introspection and future versions.
    language_config = getattr(language_model, "config", None)
    if language_config is not None and hasattr(language_config, "num_hidden_layers"):
        language_config.num_hidden_layers = layer
    backbone_config = getattr(backbone, "config", None)
    text_config = getattr(backbone_config, "text_config", None)
    if text_config is not None and hasattr(text_config, "num_hidden_layers"):
        text_config.num_hidden_layers = layer

    return original_depth


class ForesightResidualMLP(nn.Module):
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, dim)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return tokens + self.fc2(F.silu(self.fc1(self.norm(tokens))))


class JAMQwen35(nn.Module):
    """Qwen3.5 RGB policy with online frozen-Qwen foresight targets."""

    def __init__(
        self,
        *,
        student_backbone: nn.Module,
        teacher_backbone: Optional[nn.Module],
        processor,
        action_expert: ActionDiT,
        model_id: str,
        teacher_model_id: str,
        proprio_dim: Optional[int],
        image_size: int,
        num_views: int,
        context_dim: int,
        predictor_hidden_dim: int,
        action_fusion_type: str = QWEN_ACTION_FUSION_FINAL,
        action_train_shift: float,
        action_infer_shift: float,
        action_num_train_timesteps: int,
        loss_lambda_action: float,
        loss_lambda_foresight: float,
        foresight_cfg: Optional[dict[str, Any]],
        device: str | torch.device,
        torch_dtype: torch.dtype,
    ):
        super().__init__()
        foresight_cfg = dict(foresight_cfg or {})

        self.model_id = str(model_id)
        self.teacher_model_id = str(teacher_model_id)
        self.processor = processor
        self.image_size = int(image_size)
        self.num_views = int(num_views)
        self.context_dim = int(context_dim)
        self.proprio_dim = None if proprio_dim is None else int(proprio_dim)
        self.torch_dtype = torch_dtype
        self.device = torch.device(device)
        self.action_fusion_type = str(action_fusion_type).strip().lower()
        if self.action_fusion_type not in {
            QWEN_ACTION_FUSION_FINAL,
            QWEN_ACTION_FUSION_LAYERWISE_MOT,
        }:
            raise ValueError(
                "`action_fusion_type` must be 'final_context' or 'layerwise_mot', "
                f"got {action_fusion_type!r}."
            )

        qwen_hidden_dim = int(student_backbone.config.text_config.hidden_size)
        vision_cfg = student_backbone.config.vision_config
        visual_stride = int(vision_cfg.patch_size) * int(vision_cfg.spatial_merge_size)
        if self.image_size % visual_stride != 0:
            raise ValueError(
                f"Qwen image_size={self.image_size} must be divisible by visual stride={visual_stride}."
            )
        self.visual_tokens_per_view = (self.image_size // visual_stride) ** 2
        self.image_token_id = int(student_backbone.config.image_token_id)
        self.qwen_num_layers = len(student_backbone.language_model.layers)
        self.action_num_layers = len(action_expert.blocks)
        if int(action_expert.text_dim) != self.context_dim:
            raise ValueError(
                "ActionDiT text_dim must match Qwen context_dim: "
                f"{action_expert.text_dim} vs {self.context_dim}."
            )
        if (
            self.action_fusion_type == QWEN_ACTION_FUSION_LAYERWISE_MOT
            and self.action_num_layers != self.qwen_num_layers
        ):
            raise ValueError(
                "Qwen layerwise MoT requires a 1:1 decoder/action depth match: "
                f"qwen={self.qwen_num_layers}, action={self.action_num_layers}."
            )

        self.foresight_enabled = bool(foresight_cfg.get("enabled", True))
        self.foresight_load_teacher = bool(foresight_cfg.get("load_teacher", True))
        self.teacher_early_exit = bool(foresight_cfg.get("teacher_early_exit", True))
        layer = foresight_cfg.get("layer", None)
        self.foresight_layer = None if layer is None else int(layer)
        self.future_frame_offsets = tuple(
            int(offset) for offset in foresight_cfg.get("frame_offsets", (1, 2, 4))
        )
        (
            self.teacher_frame_indices,
            self.teacher_target_group_indices,
        ) = resolve_teacher_frame_layout(self.future_frame_offsets)

        raw_offset_weights = foresight_cfg.get("offset_weights", None)
        if raw_offset_weights is None:
            self.offset_weights = tuple(1.0 for _ in self.future_frame_offsets)
        else:
            self.offset_weights = tuple(float(weight) for weight in raw_offset_weights)
            if len(self.offset_weights) != len(self.future_frame_offsets):
                raise ValueError(
                    "`foresight.offset_weights` must match `frame_offsets`: "
                    f"{self.offset_weights} vs {self.future_frame_offsets}."
                )
            if any(weight < 0 for weight in self.offset_weights) or sum(self.offset_weights) <= 0:
                raise ValueError(
                    f"`foresight.offset_weights` must be non-negative with positive sum, got {self.offset_weights}."
                )

        if self.foresight_layer is not None:
            num_decoder_layers = len(student_backbone.language_model.layers)
            if not 1 <= self.foresight_layer <= num_decoder_layers:
                raise ValueError(
                    f"`foresight.layer` must be in [1, {num_decoder_layers}] or null, "
                    f"got {self.foresight_layer}."
                )

        teacher_original_depth = None
        if (
            teacher_backbone is not None
            and self.teacher_early_exit
            and self.foresight_layer is not None
        ):
            teacher_original_depth = truncate_decoder_for_exact_early_exit(
                teacher_backbone,
                layer=self.foresight_layer,
            )

        context_projector = nn.Sequential(
            nn.LayerNorm(qwen_hidden_dim),
            nn.Linear(qwen_hidden_dim, self.context_dim),
        )
        if self.action_fusion_type == QWEN_ACTION_FUSION_LAYERWISE_MOT:
            mot_visual_projector = nn.Sequential(
                nn.LayerNorm(qwen_hidden_dim),
                nn.Linear(qwen_hidden_dim, action_expert.hidden_dim),
            )
        else:
            mot_visual_projector = nn.Identity()
        if self.proprio_dim is None:
            proprio_projector = nn.Identity()
        else:
            proprio_projector = nn.Sequential(
                nn.Linear(self.proprio_dim, qwen_hidden_dim),
                nn.SiLU(),
                nn.Linear(qwen_hidden_dim, self.context_dim),
            )

        # Trainer compatibility: every trainable component must live under `dit`.
        # Predictors are always constructed, even when foresight is disabled for
        # evaluation, so a training checkpoint can still be loaded strictly.
        self.dit = nn.ModuleDict(
            {
                "qwen": student_backbone,
                "context_projector": context_projector,
                "mot_visual_projector": mot_visual_projector,
                "action_expert": action_expert,
                "foresight_predictors": nn.ModuleList(
                    [
                        ForesightResidualMLP(qwen_hidden_dim, int(predictor_hidden_dim))
                        for _ in self.future_frame_offsets
                    ]
                ),
                "proprio_projector": proprio_projector,
            }
        )
        self.dit.to(device=self.device, dtype=torch_dtype)

        # A plain list deliberately bypasses nn.Module registration. The frozen
        # teacher is therefore absent from optimizer, DeepSpeed, and checkpoints.
        self._teacher_ref = None
        if teacher_backbone is not None:
            teacher_backbone.eval()
            teacher_backbone.requires_grad_(False)
            teacher_backbone.to(device=self.device)
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

        logger.info(
            "JAMQwen35: visual_tokens/view=%d views=%d offsets=%s "
            "teacher_frames=%s foresight_layer=%s teacher_loaded=%s "
            "teacher_early_exit=%s teacher_decoder_depth=%s "
            "action_fusion=%s qwen/action_depth=%d/%d",
            self.visual_tokens_per_view,
            self.num_views,
            self.future_frame_offsets,
            self.teacher_frame_indices,
            "final" if self.foresight_layer is None else self.foresight_layer,
            self.teacher_backbone is not None,
            self.teacher_early_exit,
            (
                "none"
                if self.teacher_backbone is None
                else f"{len(self.teacher_backbone.language_model.layers)}/{teacher_original_depth or len(self.teacher_backbone.language_model.layers)}"
            ),
            self.action_fusion_type,
            self.qwen_num_layers,
            self.action_num_layers,
        )

    @property
    def student_backbone(self) -> nn.Module:
        return self.dit["qwen"]

    @property
    def context_projector(self) -> nn.Module:
        return self.dit["context_projector"]

    @property
    def mot_visual_projector(self) -> nn.Module:
        return self.dit["mot_visual_projector"]

    @property
    def action_expert(self) -> ActionDiT:
        return self.dit["action_expert"]

    @property
    def foresight_predictors(self) -> nn.ModuleList:
        return self.dit["foresight_predictors"]

    @property
    def proprio_projector(self) -> nn.Module:
        return self.dit["proprio_projector"]

    @property
    def teacher_backbone(self) -> Optional[nn.Module]:
        if self._teacher_ref is None:
            return None
        return self._teacher_ref[0]

    @classmethod
    def from_pretrained(
        cls,
        *,
        model_id: str,
        teacher_model_id: Optional[str] = None,
        proprio_dim: Optional[int] = None,
        image_size: int = 224,
        num_views: int = 2,
        context_dim: int = 4096,
        predictor_hidden_dim: int = 4096,
        use_gradient_checkpointing: bool = True,
        qwen_attn_implementation: Optional[str] = "sdpa",
        action_dit_config: Optional[dict[str, Any]] = None,
        action_dit_pretrained_path: Optional[str] = None,
        action_dit_pretrained_layer_mapping: Optional[str] = None,
        skip_dit_load_from_pretrain: bool = False,
        action_fusion_type: str = QWEN_ACTION_FUSION_FINAL,
        action_train_shift: float = 5.0,
        action_infer_shift: float = 5.0,
        action_num_train_timesteps: int = 1000,
        loss_lambda_action: float = 1.0,
        loss_lambda_foresight: float = 0.2,
        foresight_cfg: Optional[dict[str, Any]] = None,
        device: str = "cuda",
        torch_dtype: torch.dtype = torch.bfloat16,
    ) -> "JAMQwen35":
        try:
            from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration
        except (ImportError, AttributeError) as exc:
            raise ImportError(
                "JAMQwen35 requires Transformers with Qwen3.5 support. "
                "Run `bash scripts/setup_qwen35.sh` and use that venv."
            ) from exc

        foresight_cfg = dict(foresight_cfg or {})
        teacher_model_id = str(teacher_model_id or model_id)

        processor = AutoProcessor.from_pretrained(
            model_id,
            min_pixels=int(image_size) * int(image_size),
            max_pixels=int(image_size) * int(image_size),
        )

        logger.info("Loading trainable Qwen3.5 student from %s", model_id)
        qwen_load_kwargs = {
            "dtype": torch_dtype,
            "low_cpu_mem_usage": True,
        }
        if qwen_attn_implementation is not None:
            qwen_load_kwargs["attn_implementation"] = str(qwen_attn_implementation)
        student_full = Qwen3_5ForConditionalGeneration.from_pretrained(
            model_id,
            **qwen_load_kwargs,
        )
        student_backbone = student_full.model
        logger.info(
            "Qwen attention implementation: requested=%s resolved_text=%s resolved_vision=%s",
            qwen_attn_implementation,
            getattr(student_backbone.config.text_config, "_attn_implementation", None),
            getattr(student_backbone.config.vision_config, "_attn_implementation", None),
        )
        del student_full
        if use_gradient_checkpointing:
            student_backbone.gradient_checkpointing_enable()
        student_backbone.config.text_config.use_cache = False
        student_backbone.to(device=device)

        teacher_backbone = None
        load_teacher = bool(foresight_cfg.get("enabled", True)) and bool(
            foresight_cfg.get("load_teacher", True)
        )
        if load_teacher:
            logger.info("Loading frozen online Qwen3.5 teacher from %s", teacher_model_id)
            teacher_full = Qwen3_5ForConditionalGeneration.from_pretrained(
                teacher_model_id,
                **qwen_load_kwargs,
            )
            teacher_backbone = teacher_full.model
            del teacher_full
            teacher_backbone.config.text_config.use_cache = False
            teacher_backbone.eval()
            teacher_backbone.requires_grad_(False)

        action_expert = ActionDiT.from_pretrained(
            action_dit_config=dict(action_dit_config or {}),
            action_dit_pretrained_path=action_dit_pretrained_path,
            pretrained_layer_mapping=action_dit_pretrained_layer_mapping,
            skip_dit_load_from_pretrain=bool(skip_dit_load_from_pretrain),
            device=device,
            torch_dtype=torch_dtype,
        )

        return cls(
            student_backbone=student_backbone,
            teacher_backbone=teacher_backbone,
            processor=processor,
            action_expert=action_expert,
            model_id=str(model_id),
            teacher_model_id=teacher_model_id,
            proprio_dim=proprio_dim,
            image_size=image_size,
            num_views=num_views,
            context_dim=context_dim,
            predictor_hidden_dim=predictor_hidden_dim,
            action_fusion_type=action_fusion_type,
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

    def _split_views(self, frame: torch.Tensor) -> torch.Tensor:
        if frame.ndim != 4 or frame.shape[1] != 3:
            raise ValueError(f"RGB frame must be [B,3,H,W], got {tuple(frame.shape)}.")
        expected_h = self.image_size
        expected_w = self.image_size * self.num_views
        if tuple(frame.shape[-2:]) != (expected_h, expected_w):
            raise ValueError(
                "JAMQwen35 expects horizontally concatenated square camera views: "
                f"expected HxW={expected_h}x{expected_w}, got {tuple(frame.shape[-2:])}."
            )
        return torch.stack(frame.split(self.image_size, dim=-1), dim=1)

    @staticmethod
    def _to_uint8_image(view: torch.Tensor) -> torch.Tensor:
        return (
            ((view.detach().float().cpu() + 1.0) * 127.5)
            .round()
            .clamp_(0, 255)
            .to(torch.uint8)
        )

    @staticmethod
    def _vision_placeholder() -> str:
        return "<|vision_start|><|image_pad|><|vision_end|>"

    def _student_texts(self, prompts: Sequence[str]) -> list[str]:
        labels = ["agent image", "wrist image"]
        if self.num_views != len(labels):
            labels = [f"camera {idx + 1}" for idx in range(self.num_views)]
        image_lines = "\n".join(
            f"  {label}: {self._vision_placeholder()}" for label in labels
        )
        return [
            f"{prompt}\n\nCurrent observation:\n{image_lines}" for prompt in prompts
        ]

    def _teacher_texts(self, prompts: Sequence[str]) -> list[str]:
        texts = []
        for prompt in prompts:
            sections = []
            for frame_idx in self.teacher_frame_indices:
                title = (
                    "Current observation:"
                    if frame_idx == 0
                    else f"Future observation at video offset +{frame_idx}:"
                )
                lines = [title]
                for view_idx in range(self.num_views):
                    lines.append(
                        f"  camera {view_idx + 1}: {self._vision_placeholder()}"
                    )
                sections.append("\n".join(lines))
            texts.append(f"{prompt}\n\n" + "\n\n".join(sections))
        return texts

    def _prepare_qwen_inputs(
        self,
        views: torch.Tensor,
        texts: Sequence[str],
    ) -> dict[str, torch.Tensor]:
        if views.ndim != 5:
            raise ValueError(f"Qwen views must be [B,N,3,H,W], got {tuple(views.shape)}.")
        if views.shape[0] != len(texts):
            raise ValueError(f"Text/image batch mismatch: {len(texts)} vs {views.shape[0]}.")

        images = [
            self._to_uint8_image(views[batch_idx, image_idx])
            for batch_idx in range(views.shape[0])
            for image_idx in range(views.shape[1])
        ]
        processed = self.processor(
            images=images,
            text=list(texts),
            padding=True,
            return_tensors="pt",
            return_mm_token_type_ids=True,
        )
        allowed_keys = {
            "input_ids",
            "attention_mask",
            "pixel_values",
            "image_grid_thw",
            "mm_token_type_ids",
        }
        model_inputs = {}
        for key, value in processed.items():
            if key not in allowed_keys or not isinstance(value, torch.Tensor):
                continue
            if torch.is_floating_point(value):
                value = value.to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
            else:
                value = value.to(device=self.device, non_blocking=True)
            model_inputs[key] = value

        required = {"input_ids", "attention_mask", "pixel_values", "image_grid_thw", "mm_token_type_ids"}
        missing = required - set(model_inputs)
        if missing:
            raise ValueError(f"Qwen processor output is missing required keys: {sorted(missing)}.")
        return model_inputs

    def _encode_student(
        self,
        current_views: torch.Tensor,
        prompts: Sequence[str],
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        Optional[tuple[torch.Tensor, ...]],
    ]:
        model_inputs = self._prepare_qwen_inputs(current_views, self._student_texts(prompts))
        if self.action_fusion_type == QWEN_ACTION_FUSION_LAYERWISE_MOT:
            output, foresight_hidden, decoder_inputs = forward_with_decoder_layer_capture(
                self.student_backbone,
                layer=self.foresight_layer,
                model_inputs=model_inputs,
                return_decoder_inputs=True,
            )
        else:
            output, foresight_hidden = forward_with_decoder_layer_capture(
                self.student_backbone,
                layer=self.foresight_layer,
                model_inputs=model_inputs,
            )
            decoder_inputs = None
        grouped_visual = extract_grouped_visual_tokens(
            foresight_hidden,
            model_inputs["input_ids"],
            image_token_id=self.image_token_id,
            num_images_per_sample=self.num_views,
            visual_tokens_per_image=self.visual_tokens_per_view,
        )
        current_visual = grouped_visual.flatten(1, 2)
        context = self.context_projector(output.last_hidden_state)
        context_mask = model_inputs["attention_mask"].to(dtype=torch.bool)
        layerwise_visual_tokens = None
        if decoder_inputs is not None:
            embedded_visual_tokens = []
            for decoder_input in decoder_inputs:
                layer_visual = extract_grouped_visual_tokens(
                    decoder_input,
                    model_inputs["input_ids"],
                    image_token_id=self.image_token_id,
                    num_images_per_sample=self.num_views,
                    visual_tokens_per_image=self.visual_tokens_per_view,
                ).flatten(1, 2)
                embedded_visual_tokens.append(self.mot_visual_projector(layer_visual))
            layerwise_visual_tokens = tuple(embedded_visual_tokens)
            # Mixed attention owns the image stream. Keep only instruction text
            # (plus the proprio token appended later) in ActionDiT cross-attn.
            text_mask = context_mask & ~model_inputs["input_ids"].eq(self.image_token_id)
            text_lengths = text_mask.sum(dim=1)
            max_text_len = int(text_lengths.max().item())
            if max_text_len <= 0:
                raise RuntimeError("Qwen layerwise MoT found no instruction tokens.")
            context = torch.stack(
                [
                    F.pad(
                        context[batch_idx, text_mask[batch_idx]],
                        (0, 0, 0, max_text_len - int(text_lengths[batch_idx].item())),
                    )
                    for batch_idx in range(context.shape[0])
                ],
                dim=0,
            )
            context_mask = (
                torch.arange(max_text_len, device=context.device).unsqueeze(0)
                < text_lengths.unsqueeze(1)
            )
        return context, context_mask, current_visual, layerwise_visual_tokens

    @torch.no_grad()
    def _encode_teacher_targets(
        self,
        teacher_views: torch.Tensor,
        prompts: Sequence[str],
    ) -> list[torch.Tensor]:
        teacher = self.teacher_backbone
        if teacher is None:
            raise RuntimeError(
                "Foresight training requires the frozen teacher, but `foresight.load_teacher=false`."
            )
        model_inputs = self._prepare_qwen_inputs(teacher_views, self._teacher_texts(prompts))
        _, foresight_hidden = forward_with_decoder_layer_capture(
            teacher,
            layer=self.foresight_layer,
            model_inputs=model_inputs,
        )
        grouped_visual = extract_grouped_visual_tokens(
            foresight_hidden,
            model_inputs["input_ids"],
            image_token_id=self.image_token_id,
            num_images_per_sample=len(self.teacher_frame_indices) * self.num_views,
            visual_tokens_per_image=self.visual_tokens_per_view,
        )
        grouped_visual = grouped_visual.reshape(
            grouped_visual.shape[0],
            len(self.teacher_frame_indices),
            self.num_views,
            self.visual_tokens_per_view,
            grouped_visual.shape[-1],
        )
        return [
            grouped_visual[:, group_idx].flatten(1, 2).detach()
            for group_idx in self.teacher_target_group_indices
        ]

    @staticmethod
    def _normalize_prompts(prompt, batch_size: int) -> list[str]:
        if isinstance(prompt, str):
            prompts = [prompt]
        elif isinstance(prompt, tuple):
            prompts = list(prompt)
        elif isinstance(prompt, list):
            prompts = prompt
        else:
            raise TypeError(f"`prompt` must be str/list[str], got {type(prompt)}.")
        if len(prompts) != batch_size or not all(isinstance(item, str) for item in prompts):
            raise ValueError(
                f"Prompt batch must contain {batch_size} strings, got {len(prompts)} entries."
            )
        return prompts

    def _append_proprio_context(
        self,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        proprio: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.proprio_dim is None:
            if proprio is not None:
                raise ValueError("`proprio` was provided but the model has `proprio_dim=null`.")
            return context, context_mask
        if proprio is None:
            raise ValueError("JAMQwen35 requires `proprio` when `proprio_dim` is configured.")
        if proprio.ndim == 3:
            proprio = proprio[:, 0]
        elif proprio.ndim == 1:
            proprio = proprio.unsqueeze(0)
        if proprio.ndim != 2 or proprio.shape != (context.shape[0], self.proprio_dim):
            raise ValueError(
                f"`proprio` must resolve to [B,{self.proprio_dim}], got {tuple(proprio.shape)}."
            )
        proprio_token = self.proprio_projector(
            proprio.to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
        ).unsqueeze(1)
        proprio_mask = torch.ones(
            (context.shape[0], 1), dtype=torch.bool, device=context_mask.device
        )
        return (
            torch.cat([context, proprio_token], dim=1),
            torch.cat([context_mask, proprio_mask], dim=1),
        )

    def _validate_training_sample(self, sample: dict[str, Any]):
        required = {"video", "action", "prompt", "proprio"}
        missing = required - set(sample)
        if missing:
            raise ValueError(f"JAMQwen35 training sample is missing: {sorted(missing)}.")
        video = sample["video"]
        action = sample["action"]
        if video.ndim != 5 or video.shape[1] != 3:
            raise ValueError(f"`video` must be [B,3,T,H,W], got {tuple(video.shape)}.")
        if action.ndim != 3:
            raise ValueError(f"`action` must be [B,T,D], got {tuple(action.shape)}.")
        if video.shape[0] != action.shape[0]:
            raise ValueError(f"Video/action batch mismatch: {video.shape[0]} vs {action.shape[0]}.")
        max_frame = max(self.teacher_frame_indices)
        if video.shape[2] <= max_frame:
            raise ValueError(
                "Foresight offset exceeds the RGB video window: "
                f"max requested={max_frame}, available indices=0..{video.shape[2] - 1}."
            )
        if action.shape[-1] != self.action_expert.action_dim:
            raise ValueError(
                f"Action dim must be {self.action_expert.action_dim}, got {action.shape[-1]}."
            )

    def _compute_action_loss(
        self,
        action: torch.Tensor,
        action_is_pad: Optional[torch.Tensor],
        context: torch.Tensor,
        context_mask: torch.Tensor,
        layerwise_visual_tokens: Optional[Sequence[torch.Tensor]],
    ) -> torch.Tensor:
        batch_size = action.shape[0]
        noise = torch.randn_like(action)
        timestep = self.train_action_scheduler.sample_training_t(
            batch_size=batch_size,
            device=self.device,
            dtype=action.dtype,
        )
        noisy_action = self.train_action_scheduler.add_noise(action, noise, timestep)
        target = self.train_action_scheduler.training_target(action, noise, timestep)
        prediction = self._run_action_expert(
            action_tokens=noisy_action,
            timestep=timestep,
            context=context,
            context_mask=context_mask,
            layerwise_visual_tokens=layerwise_visual_tokens,
        )
        token_loss = F.mse_loss(prediction.float(), target.float(), reduction="none").mean(dim=-1)
        if action_is_pad is None:
            per_sample = token_loss.mean(dim=1)
        else:
            if action_is_pad.shape != token_loss.shape:
                raise ValueError(
                    f"`action_is_pad` must be {tuple(token_loss.shape)}, got {tuple(action_is_pad.shape)}."
                )
            valid = (~action_is_pad.to(device=self.device, dtype=torch.bool)).to(token_loss.dtype)
            per_sample = (token_loss * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1.0)
        weight = self.train_action_scheduler.training_weight(timestep).to(
            device=per_sample.device, dtype=per_sample.dtype
        )
        return (per_sample * weight).mean()

    def _run_action_expert(
        self,
        *,
        action_tokens: torch.Tensor,
        timestep: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        layerwise_visual_tokens: Optional[Sequence[torch.Tensor]],
    ) -> torch.Tensor:
        if self.action_fusion_type == QWEN_ACTION_FUSION_LAYERWISE_MOT:
            if layerwise_visual_tokens is None:
                raise RuntimeError("Layerwise MoT action fusion is missing Qwen visual tokens.")
            return self.action_expert.forward_layerwise_mot(
                action_tokens=action_tokens,
                timestep=timestep,
                context=context,
                context_mask=context_mask,
                layerwise_visual_tokens=layerwise_visual_tokens,
            )
        if layerwise_visual_tokens is not None:
            raise RuntimeError("Legacy final-context fusion received unexpected layerwise tokens.")
        return self.action_expert(
            action_tokens=action_tokens,
            timestep=timestep,
            context=context,
            context_mask=context_mask,
        )

    def _compute_foresight_loss(
        self,
        student_visual: torch.Tensor,
        teacher_targets: Sequence[torch.Tensor],
        target_valid: torch.Tensor,
    ) -> torch.Tensor:
        weighted_losses = []
        for predictor, target, valid, weight in zip(
            self.foresight_predictors,
            teacher_targets,
            target_valid.unbind(dim=1),
            self.offset_weights,
        ):
            prediction = predictor(student_visual)
            per_sample = (
                1.0 - F.cosine_similarity(prediction.float(), target.float(), dim=-1)
            ).mean(dim=1)
            valid = valid.to(device=per_sample.device, dtype=per_sample.dtype)
            if bool(valid.any().item()):
                offset_loss = (per_sample * valid).sum() / valid.sum()
            else:
                offset_loss = prediction.float().sum() * 0.0
            weighted_losses.append(float(weight) * offset_loss)
        return torch.stack(weighted_losses).sum() / float(sum(self.offset_weights))

    def training_loss(self, sample: dict[str, Any]):
        self._validate_training_sample(sample)
        video = sample["video"]
        batch_size = video.shape[0]
        prompts = self._normalize_prompts(sample["prompt"], batch_size)

        current_frame = video[:, :, 0]
        current_views = self._split_views(current_frame)
        context, context_mask, student_visual, layerwise_visual_tokens = self._encode_student(
            current_views, prompts
        )
        context, context_mask = self._append_proprio_context(
            context,
            context_mask,
            sample.get("proprio"),
        )

        action = sample["action"].to(
            device=self.device, dtype=self.torch_dtype, non_blocking=True
        )
        action_is_pad = sample.get("action_is_pad")
        if action_is_pad is not None:
            action_is_pad = action_is_pad.to(device=self.device, dtype=torch.bool, non_blocking=True)
        loss_action = self._compute_action_loss(
            action=action,
            action_is_pad=action_is_pad,
            context=context,
            context_mask=context_mask,
            layerwise_visual_tokens=layerwise_visual_tokens,
        )

        loss_foresight = loss_action.new_zeros(())
        if self.foresight_enabled:
            teacher_views = torch.cat(
                [self._split_views(video[:, :, frame_idx]) for frame_idx in self.teacher_frame_indices],
                dim=1,
            )
            teacher_targets = self._encode_teacher_targets(teacher_views, prompts)
            target_valid = future_target_valid_mask(
                sample.get("image_is_pad"),
                self.future_frame_offsets,
                batch_size=batch_size,
                device=self.device,
            ).to(device=self.device)
            loss_foresight = self._compute_foresight_loss(
                student_visual=student_visual,
                teacher_targets=teacher_targets,
                target_valid=target_valid,
            )

        loss_total = (
            self.loss_lambda_action * loss_action
            + self.loss_lambda_foresight * loss_foresight
        )
        return loss_total, {
            "loss_action": self.loss_lambda_action * float(loss_action.detach().item()),
            "loss_foresight": self.loss_lambda_foresight * float(loss_foresight.detach().item()),
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
        if input_image.ndim != 4 or input_image.shape[0] != 1:
            raise ValueError(
                f"`input_image` must be [1,3,H,W] or [3,H,W], got {tuple(input_image.shape)}."
            )
        if int(action_horizon) <= 0:
            raise ValueError(f"`action_horizon` must be positive, got {action_horizon}.")

        views = self._split_views(input_image)
        context, context_mask, _, layerwise_visual_tokens = self._encode_student(
            views, [str(prompt)]
        )
        context, context_mask = self._append_proprio_context(context, context_mask, proprio)

        generator = None if seed is None else torch.Generator(device=rand_device).manual_seed(int(seed))
        action = torch.randn(
            (1, int(action_horizon), self.action_expert.action_dim),
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
        for step_t, step_delta in zip(timesteps, deltas):
            timestep = step_t.reshape(1).to(device=self.device, dtype=action.dtype)
            prediction = self._run_action_expert(
                action_tokens=action,
                timestep=timestep,
                context=context,
                context_mask=context_mask,
                layerwise_visual_tokens=layerwise_visual_tokens,
            )
            action = self.infer_action_scheduler.step(prediction, step_delta, action)
        return {"action": action[0].detach().to(device="cpu", dtype=torch.float32)}

    def save_checkpoint(self, path, optimizer=None, step=None):
        payload = {
            "format": QWEN_CHECKPOINT_FORMAT,
            "dit": self.dit.state_dict(),
            "step": step,
            "model_id": self.model_id,
            "teacher_model_id": self.teacher_model_id,
            "future_frame_offsets": self.future_frame_offsets,
            "foresight_layer": self.foresight_layer,
            "teacher_early_exit": self.teacher_early_exit,
            "action_fusion_type": self.action_fusion_type,
            "qwen_num_layers": self.qwen_num_layers,
            "action_num_layers": self.action_num_layers,
        }
        if optimizer is not None:
            payload["optimizer"] = optimizer.state_dict()
        torch.save(payload, path)

    def load_checkpoint(self, path, optimizer=None, mmap: bool = False):
        load_kwargs = {"map_location": "cpu"}
        if mmap:
            load_kwargs["mmap"] = True
        payload = torch.load(path, **load_kwargs)
        checkpoint_format = payload.get("format")
        if checkpoint_format not in {
            QWEN_LEGACY_CHECKPOINT_FORMAT,
            QWEN_CHECKPOINT_FORMAT,
        }:
            raise ValueError(
                f"Unsupported JAMQwen35 checkpoint format in {path}: {payload.get('format')!r}."
            )
        saved_action_fusion = payload.get(
            "action_fusion_type", QWEN_ACTION_FUSION_FINAL
        )
        if saved_action_fusion != self.action_fusion_type:
            raise ValueError(
                "Action fusion mismatch while resuming: "
                f"checkpoint={saved_action_fusion!r}, model={self.action_fusion_type!r}."
            )
        saved_action_layers = int(
            payload.get("action_num_layers", len(self.action_expert.blocks))
        )
        if saved_action_layers != self.action_num_layers:
            raise ValueError(
                "Action depth mismatch while resuming: "
                f"checkpoint={saved_action_layers}, model={self.action_num_layers}."
            )
        saved_offsets = tuple(int(offset) for offset in payload.get("future_frame_offsets", ()))
        if saved_offsets != self.future_frame_offsets:
            raise ValueError(
                "Foresight offset mismatch while resuming: "
                f"checkpoint={saved_offsets}, model={self.future_frame_offsets}."
            )
        saved_layer = payload.get("foresight_layer", None)
        saved_layer = None if saved_layer is None else int(saved_layer)
        if saved_layer != self.foresight_layer:
            raise ValueError(
                "Foresight layer mismatch while resuming: "
                f"checkpoint={saved_layer}, model={self.foresight_layer}."
            )
        self.dit.load_state_dict(payload["dit"], strict=True)
        if optimizer is not None and "optimizer" in payload:
            optimizer.load_state_dict(payload["optimizer"])
        return payload

    def forward(self, *args, **kwargs):
        return self.training_loss(*args, **kwargs)


__all__ = [
    "ForesightResidualMLP",
    "JAMQwen35",
    "QWEN_ACTION_FUSION_FINAL",
    "QWEN_ACTION_FUSION_LAYERWISE_MOT",
    "QWEN_CHECKPOINT_FORMAT",
    "QWEN_LEGACY_CHECKPOINT_FORMAT",
    "extract_grouped_visual_tokens",
    "forward_with_decoder_layer_capture",
    "future_target_valid_mask",
    "resolve_teacher_frame_layout",
    "truncate_decoder_for_exact_early_exit",
]


# Compatibility aliases for historical configs and imports.
QwenFastWAM = JAMQwen35

__all__ = list(dict.fromkeys(__all__ + ['JAMQwen35', 'QwenFastWAM']))
