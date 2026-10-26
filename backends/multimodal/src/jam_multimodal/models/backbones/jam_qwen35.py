from __future__ import annotations

import logging
import os
from collections import OrderedDict
from functools import partial
from typing import Any, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .action_dit import ActionDiT
from .action_dit_qwen35 import (
    ActionDiTQwen35Native,
    Qwen35ActionCondition,
    QWEN_ACTION_ATTENTION_CAUSAL,
)
from .foresight_yak import ForesightProjector as JAMForesightProjector
from .qwen35_vision_attention import enable_equal_grid_attention
from .schedulers.scheduler_continuous import WanContinuousFlowMatchScheduler

logger = logging.getLogger(__name__)


# The legacy residual-MLP variant remains loadable for old FastWAM-Qwen
# checkpoints. The default ImageWAM projector changes state-dict structure, so
# it uses an explicit v2 format instead of pretending to be losslessly
# interchangeable with the FastWAM branch.
QWEN_WAN_LEGACY_CHECKPOINT_FORMAT = "qwen35_fastwam_v1"
QWEN_WAN_CHECKPOINT_FORMAT = "qwen35_fastwam_v2"
QWEN_JAM_LEGACY_CHECKPOINT_FORMAT = "qwen35_imagewam_v2"
QWEN_JAM_CHECKPOINT_FORMAT = "qwen35_imagewam_v3"
QWEN_JAM_YAK_CHECKPOINT_FORMAT = "qwen35_imagewam_yak18_v1"
QWEN_JAM_NATIVE_CHECKPOINT_FORMAT = "qwen35_imagewam_native_qwen24_v1"
QWEN_NATIVE_COPY_INITIALIZATION = "exact_qwen35_decoder_block_copy_v1"
QWEN_CHECKPOINT_FORMAT = QWEN_JAM_CHECKPOINT_FORMAT
QWEN_PREDICTOR_JAM = "imagewam_transformer"
QWEN_PREDICTOR_LEGACY = "residual_mlp"
QWEN_ACTION_FUSION_FINAL = "final_context"
QWEN_ACTION_FUSION_LAYERWISE_MOT = "layerwise_mot"
QWEN_ACTION_FUSION_YAK_LAYERWISE_MOT = "yak_layerwise_mot"
QWEN_ACTION_FUSION_NATIVE_QWEN = "qwen_native_mot"
QWEN_TEACHER_MODES = frozenset({"frozen_pretrained"})


def _uniform_block_indices(depth: int, count: int) -> tuple[int, ...]:
    if depth <= 0 or count <= 0 or count > depth:
        raise ValueError(
            f"Checkpoint block count must be in [1, {depth}], got {count}."
        )
    if count == 1:
        return (depth // 2,)
    return tuple(
        int(block_idx * (depth - 1) / (count - 1) + 0.5)
        for block_idx in range(count)
    )


def enable_qwen_vision_gradient_checkpointing(
    backbone: nn.Module,
    *,
    num_blocks: Optional[int] = None,
) -> tuple[int, ...]:
    """Checkpoint only the trainable Qwen vision blocks.

    The generic Transformers switch also checkpoints all 24 language layers.
    Here the memory target is the fixed-grid vision activation that prevents
    RoboTwin b48/GA1 from fitting, so leave the language and ActionQwen paths
    untouched. Non-reentrant checkpointing preserves parameter gradients and
    supports the tensor kwargs used by the Qwen3.5 vision blocks.
    """

    visual = getattr(backbone, "visual", None)
    blocks = getattr(visual, "blocks", None)
    if blocks is None:
        raise TypeError("Qwen backbone has no visual.blocks to checkpoint.")
    depth = len(blocks)
    selected = _uniform_block_indices(
        depth,
        depth if num_blocks is None else int(num_blocks),
    )
    selected_set = set(selected)
    checkpoint_fn = partial(checkpoint, use_reentrant=False)
    for block_idx, block in enumerate(blocks):
        if not hasattr(block, "gradient_checkpointing"):
            raise TypeError(
                f"Vision block {type(block).__name__} does not support checkpointing."
            )
        block.gradient_checkpointing = block_idx in selected_set
        if block.gradient_checkpointing:
            block._gradient_checkpointing_func = checkpoint_fn
    return selected


def uniform_endpoint_layer_map(source_depth: int, target_depth: int) -> tuple[int, ...]:
    """Map target blocks over the full source depth, including both endpoints."""

    source_depth = int(source_depth)
    target_depth = int(target_depth)
    if source_depth <= 0 or target_depth <= 0:
        raise ValueError(
            f"Layer depths must be positive, got source={source_depth}, target={target_depth}."
        )
    if target_depth == 1:
        return (source_depth - 1,)
    return tuple(
        int(layer_idx * (source_depth - 1) / (target_depth - 1) + 0.5)
        for layer_idx in range(target_depth)
    )


def normalize_qwen_teacher_mode(value: Any) -> str:
    mode = str(value if value is not None else "frozen_pretrained").strip().lower()
    if mode not in QWEN_TEACHER_MODES:
        raise ValueError(
            f"Unsupported Qwen foresight teacher mode {value!r}; expected one of "
            f"{sorted(QWEN_TEACHER_MODES)}."
        )
    return mode


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


def resolve_decoded_frame_positions(
    logical_frame_indices: Sequence[int],
    num_decoded_frames: int,
) -> tuple[int, ...]:
    """Map logical RGB-grid indices to positions in the decoded tensor.

    The normal input keeps the full subsampled grid, so logical and physical
    positions are identical. ``RobotVideoDataset(endpoint_frames_only=true)``
    instead decodes only the first and last observations. A single-future
    layout such as logical ``(0, 4)`` is therefore stored at tensor positions
    ``(0, 1)``. Multi-future layouts still require the full grid because two
    endpoint frames cannot represent intermediate targets.
    """

    logical = tuple(int(frame_idx) for frame_idx in logical_frame_indices)
    num_decoded_frames = int(num_decoded_frames)
    if not logical:
        raise ValueError("`logical_frame_indices` must not be empty.")
    if any(frame_idx < 0 for frame_idx in logical):
        raise ValueError(f"Logical frame indices must be non-negative, got {logical}.")
    if len(set(logical)) != len(logical):
        raise ValueError(f"Logical frame indices must be unique, got {logical}.")
    if num_decoded_frames <= 0:
        raise ValueError(f"`num_decoded_frames` must be positive, got {num_decoded_frames}.")

    if max(logical) < num_decoded_frames:
        return logical
    if num_decoded_frames == 2 and len(logical) == 2 and logical[0] == 0:
        return (0, 1)
    raise ValueError(
        "Decoded RGB layout cannot represent the requested foresight frames: "
        f"logical={logical}, num_decoded_frames={num_decoded_frames}. "
        "Use the full RGB grid for multi-future targets, or endpoint-only "
        "decoding for a single current/future pair."
    )


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
    token_indices: Optional[torch.Tensor] = None,
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
    if token_indices is not None:
        if tuple(token_indices.shape) != (hidden_states.shape[0], expected):
            raise ValueError(
                "Qwen cached visual indices must be [B,images*tokens], got "
                f"{tuple(token_indices.shape)} vs "
                f"{(hidden_states.shape[0], expected)}."
            )
        selected = hidden_states.gather(
            1,
            token_indices[:, :, None].expand(
                -1, -1, hidden_states.shape[-1]
            ),
        )
        return selected.reshape(
            hidden_states.shape[0],
            int(num_images_per_sample),
            int(visual_tokens_per_image),
            hidden_states.shape[-1],
        )

    image_mask = input_ids.eq(int(image_token_id))
    counts = image_mask.sum(dim=1)
    counts_ok = torch.all(counts == expected)
    if counts_ok.device.type == "cuda" and hasattr(torch, "_assert_async"):
        # Avoid a device-wide synchronization in every student and teacher
        # forward.  A malformed batch still fails on the CUDA stream before
        # its result can be consumed.
        torch._assert_async(
            counts_ok,
            "Qwen visual token count does not match the fixed image layout.",
        )
    elif not bool(counts_ok.item()):
        raise ValueError(
            "Qwen visual token count mismatch: "
            f"expected {expected} ({num_images_per_sample} images x "
            f"{visual_tokens_per_image}), got {counts.detach().cpu().tolist()}."
        )

    # Boolean indexing is batch-major.  Since every row has the same verified
    # image-token count, one indexing kernel is exactly equivalent to the old
    # B separate indexing kernels followed by stack, including its gradient.
    selected = hidden_states[image_mask].reshape(
        hidden_states.shape[0], expected, hidden_states.shape[-1]
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
    return_decoder_outputs: bool = False,
):
    """Run Qwen while retaining only one decoder layer's output.

    ``layer`` is one-based. ``None`` uses the final normalized hidden state and
    installs no hook.
    """

    captured: dict[str, torch.Tensor] = {}
    decoder_inputs: dict[int, torch.Tensor] = {}
    decoder_outputs: dict[int, torch.Tensor] = {}
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

    if return_decoder_outputs:
        for layer_idx, decoder_layer in enumerate(decoder_layers):
            def capture_output(_module, _args, output, _layer_idx=layer_idx):
                if isinstance(output, (tuple, list)):
                    output = output[0]
                if not isinstance(output, torch.Tensor):
                    raise TypeError(
                        "Captured Qwen decoder output must be a tensor, "
                        f"got {type(output)}."
                    )
                decoder_outputs[_layer_idx] = output

            handles.append(decoder_layer.register_forward_hook(capture_output))

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
    if not return_decoder_inputs and not return_decoder_outputs:
        return output, layer_hidden
    returned = [output, layer_hidden]
    if return_decoder_inputs:
        missing = [idx for idx in range(len(decoder_layers)) if idx not in decoder_inputs]
        if missing:
            raise RuntimeError(f"Qwen decoder input hooks missed layers: {missing}.")
        returned.append(tuple(decoder_inputs[idx] for idx in range(len(decoder_layers))))
    if return_decoder_outputs:
        missing = [idx for idx in range(len(decoder_layers)) if idx not in decoder_outputs]
        if missing:
            raise RuntimeError(f"Qwen decoder output hooks missed layers: {missing}.")
        returned.append(tuple(decoder_outputs[idx] for idx in range(len(decoder_layers))))
    return tuple(returned)


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
        action_expert: nn.Module,
        model_id: str,
        teacher_model_id: str,
        proprio_dim: Optional[int],
        image_size: int,
        num_views: int,
        context_dim: int,
        predictor_type: str,
        predictor_hidden_dim: int,
        predictor_ffn_dim: int,
        predictor_num_layers: int,
        predictor_num_heads: int,
        tensor_native_preprocessing: bool = False,
        batch_equal_grid_vision_attention: bool = False,
        vision_gradient_checkpointing: bool = False,
        vision_gradient_checkpointing_blocks: Optional[int] = None,
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
        self.tensor_native_preprocessing = bool(tensor_native_preprocessing)
        self.batch_equal_grid_vision_attention = bool(
            batch_equal_grid_vision_attention
        )
        self.vision_gradient_checkpointing = bool(vision_gradient_checkpointing)
        self.vision_gradient_checkpointing_blocks = (
            None
            if vision_gradient_checkpointing_blocks is None
            else int(vision_gradient_checkpointing_blocks)
        )
        self._tokenized_cache_max_entries = int(
            os.environ.get("JAM_QWEN_TOKEN_CACHE_ENTRIES", "2048")
        )
        if self._tokenized_cache_max_entries < 0:
            raise ValueError(
                "JAM_QWEN_TOKEN_CACHE_ENTRIES must be non-negative."
            )
        # RoboTwin has 921,032 distinct instructions.  An unbounded cache of
        # expanded token/M-RoPE rows would eventually consume many GiB on both
        # host and GPU.  Each entry keeps compact immutable CPU tuples plus a
        # tiny per-device tensor tuple; LRU bounds the long-run footprint.
        self._tokenized_layout_cache: OrderedDict[
            str,
            tuple[
                tuple[int, ...],
                tuple[tuple[int, ...], ...],
                dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
            ],
        ] = OrderedDict()
        self.action_fusion_type = str(action_fusion_type).strip().lower()
        if self.action_fusion_type not in {
            QWEN_ACTION_FUSION_FINAL,
            QWEN_ACTION_FUSION_LAYERWISE_MOT,
            QWEN_ACTION_FUSION_YAK_LAYERWISE_MOT,
            QWEN_ACTION_FUSION_NATIVE_QWEN,
        }:
            raise ValueError(
                "`action_fusion_type` must be 'final_context', 'layerwise_mot', "
                "'yak_layerwise_mot', or 'qwen_native_mot', "
                f"got {action_fusion_type!r}."
            )

        qwen_hidden_dim = int(student_backbone.config.text_config.hidden_size)
        self.foresight_predictor_type = str(predictor_type).strip().lower()
        if self.foresight_predictor_type not in {
            QWEN_PREDICTOR_JAM,
            QWEN_PREDICTOR_LEGACY,
        }:
            raise ValueError(
                "`predictor_type` must be 'imagewam_transformer' or 'residual_mlp', "
                f"got {predictor_type!r}."
            )
        self.foresight_predictor_config = {
            "hidden_dim": int(predictor_hidden_dim),
            "ffn_dim": int(predictor_ffn_dim),
            "num_layers": int(predictor_num_layers),
            "num_heads": int(predictor_num_heads),
        }
        if (
            self.foresight_predictor_type == QWEN_PREDICTOR_JAM
            and int(predictor_hidden_dim) % int(predictor_num_heads) != 0
        ):
            raise ValueError(
                "JAMPolicy predictor hidden dim must be divisible by its head count: "
                f"{predictor_hidden_dim} vs {predictor_num_heads}."
            )
        vision_cfg = student_backbone.config.vision_config
        batched_vision_blocks = (0, 0)
        if self.batch_equal_grid_vision_attention:
            student_vision_blocks = enable_equal_grid_attention(
                student_backbone, image_size=self.image_size
            )
            teacher_vision_blocks = 0
            if teacher_backbone is not None:
                teacher_vision_blocks = enable_equal_grid_attention(
                    teacher_backbone, image_size=self.image_size
                )
            batched_vision_blocks = (student_vision_blocks, teacher_vision_blocks)
        visual_stride = int(vision_cfg.patch_size) * int(vision_cfg.spatial_merge_size)
        if self.image_size % visual_stride != 0:
            raise ValueError(
                f"Qwen image_size={self.image_size} must be divisible by visual stride={visual_stride}."
            )
        self.visual_tokens_per_view = (self.image_size // visual_stride) ** 2
        self.image_token_id = int(student_backbone.config.image_token_id)
        self.qwen_num_layers = len(student_backbone.language_model.layers)
        self.action_num_layers = len(action_expert.blocks)
        self.action_visual_layer_map = uniform_endpoint_layer_map(
            self.qwen_num_layers, self.action_num_layers
        )
        if self.action_fusion_type == QWEN_ACTION_FUSION_NATIVE_QWEN:
            if not isinstance(action_expert, ActionDiTQwen35Native):
                raise TypeError(
                    "Qwen native MoT requires ActionDiTQwen35Native, got "
                    f"{type(action_expert).__name__}."
                )
            if int(action_expert.hidden_dim) != qwen_hidden_dim:
                raise ValueError(
                    "Native Qwen action hidden dim must match the Qwen backbone: "
                    f"{action_expert.hidden_dim} vs {qwen_hidden_dim}."
                )
            if self.context_dim != qwen_hidden_dim:
                raise ValueError(
                    "Native Qwen action requires context_dim=Qwen hidden size: "
                    f"{self.context_dim} vs {qwen_hidden_dim}."
                )
            self.action_attention_mode = str(action_expert.action_attention_mode)
        elif self.action_fusion_type == QWEN_ACTION_FUSION_YAK_LAYERWISE_MOT:
            self.action_attention_mode = None
            if type(action_expert).__name__ != "ActionDiTYak":
                raise TypeError(
                    "Qwen Yak layerwise MoT requires ActionDiTYak, got "
                    f"{type(action_expert).__name__}."
                )
            if int(action_expert.hidden_dim) != self.context_dim:
                raise ValueError(
                    "ActionDiTYak hidden dim must match Qwen context_dim: "
                    f"{action_expert.hidden_dim} vs {self.context_dim}."
                )
        else:
            self.action_attention_mode = None
            if not isinstance(action_expert, ActionDiT):
                raise TypeError(
                    "Qwen final/Wan layerwise fusion requires ActionDiT, got "
                    f"{type(action_expert).__name__}."
                )
            if int(action_expert.text_dim) != self.context_dim:
                raise ValueError(
                    "ActionDiT text_dim must match Qwen context_dim: "
                    f"{action_expert.text_dim} vs {self.context_dim}."
                )
        if (
            self.action_fusion_type in {
                QWEN_ACTION_FUSION_LAYERWISE_MOT,
                QWEN_ACTION_FUSION_NATIVE_QWEN,
            }
            and self.action_num_layers != self.qwen_num_layers
        ):
            raise ValueError(
                "Qwen layerwise MoT requires a 1:1 decoder/action depth match: "
                f"qwen={self.qwen_num_layers}, action={self.action_num_layers}."
            )

        self.foresight_enabled = bool(foresight_cfg.get("enabled", True))
        self.foresight_load_teacher = bool(foresight_cfg.get("load_teacher", True))
        self.foresight_teacher_mode = normalize_qwen_teacher_mode(
            foresight_cfg.get("teacher_mode", "frozen_pretrained")
        )
        self.foresight_teacher_random_seed = int(
            foresight_cfg.get("teacher_random_seed", 42)
        )
        self.teacher_early_exit = bool(foresight_cfg.get("teacher_early_exit", True))
        # The frozen teacher has no RNG side effects and does not participate
        # in autograd.  Running it before the trainable student keeps only the
        # compact target tokens alive while the student/action graph is built,
        # instead of overlapping the teacher's temporary activations with the
        # complete trainable graph.  This changes execution order only; the
        # samples, forward equations, loss, and gradients are identical.
        self.teacher_first = bool(foresight_cfg.get("teacher_first", False))
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

        if self.action_fusion_type == QWEN_ACTION_FUSION_NATIVE_QWEN:
            context_projector = nn.Identity()
        else:
            context_projector = nn.Sequential(
                nn.LayerNorm(qwen_hidden_dim),
                nn.Linear(qwen_hidden_dim, self.context_dim),
            )
        if self.action_fusion_type in {
            QWEN_ACTION_FUSION_LAYERWISE_MOT,
            QWEN_ACTION_FUSION_YAK_LAYERWISE_MOT,
        }:
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
        dit_modules = {
            "qwen": student_backbone,
            "context_projector": context_projector,
            "mot_visual_projector": mot_visual_projector,
            "action_expert": action_expert,
            "proprio_projector": proprio_projector,
        }
        if self.foresight_predictor_type == QWEN_PREDICTOR_JAM:
            dit_modules["foresight_projector"] = JAMForesightProjector(
                in_dim=qwen_hidden_dim,
                hidden_dim=int(predictor_hidden_dim),
                ffn_dim=int(predictor_ffn_dim),
                num_layers=int(predictor_num_layers),
                num_heads=int(predictor_num_heads),
                num_targets=len(self.future_frame_offsets),
                out_dim=qwen_hidden_dim,
            )
        else:
            dit_modules["foresight_predictors"] = nn.ModuleList(
                [
                    ForesightResidualMLP(qwen_hidden_dim, int(predictor_hidden_dim))
                    for _ in self.future_frame_offsets
                ]
            )
        self.dit = nn.ModuleDict(dit_modules)
        self.dit.to(device=self.device, dtype=torch_dtype)

        # A plain list deliberately bypasses nn.Module registration. The frozen
        # teacher is therefore absent from optimizer, DeepSpeed, and checkpoints.
        self._teacher_ref = None
        if teacher_backbone is not None:
            teacher_backbone.eval()
            teacher_backbone.requires_grad_(False)
            teacher_backbone.to(device=self.device)
            self._teacher_ref = [teacher_backbone]

        self.action_num_train_timesteps = int(action_num_train_timesteps)
        self.train_action_scheduler = WanContinuousFlowMatchScheduler(
            num_train_timesteps=self.action_num_train_timesteps,
            shift=float(action_train_shift),
        )
        self.infer_action_scheduler = WanContinuousFlowMatchScheduler(
            num_train_timesteps=self.action_num_train_timesteps,
            shift=float(action_infer_shift),
        )
        self.loss_lambda_action = float(loss_lambda_action)
        self.loss_lambda_foresight = float(loss_lambda_foresight)

        logger.info(
            "JAMQwen35: visual_tokens/view=%d views=%d offsets=%s "
            "teacher_frames=%s foresight_layer=%s teacher_loaded=%s "
            "teacher_mode=%s teacher_random_seed=%d "
            "teacher_early_exit=%s teacher_first=%s teacher_decoder_depth=%s "
            "predictor=%s predictor_config=%s "
            "lambda_action=%.3f lambda_foresight=%.3f "
            "action_fusion=%s action_attention_mode=%s "
            "qwen/action_depth=%d/%d layer_map=%s tensor_native_preprocessing=%s "
            "batch_equal_grid_vision_attention=%s vision_blocks=%s "
            "vision_gradient_checkpointing=%s vision_checkpoint_blocks=%s",
            self.visual_tokens_per_view,
            self.num_views,
            self.future_frame_offsets,
            self.teacher_frame_indices,
            "final" if self.foresight_layer is None else self.foresight_layer,
            self.teacher_backbone is not None,
            self.foresight_teacher_mode,
            self.foresight_teacher_random_seed,
            self.teacher_early_exit,
            self.teacher_first,
            (
                "none"
                if self.teacher_backbone is None
                else f"{len(self.teacher_backbone.language_model.layers)}/{teacher_original_depth or len(self.teacher_backbone.language_model.layers)}"
            ),
            self.foresight_predictor_type,
            self.foresight_predictor_config,
            self.loss_lambda_action,
            self.loss_lambda_foresight,
            self.action_fusion_type,
            self.action_attention_mode,
            self.qwen_num_layers,
            self.action_num_layers,
            self.action_visual_layer_map,
            self.tensor_native_preprocessing,
            self.batch_equal_grid_vision_attention,
            batched_vision_blocks,
            self.vision_gradient_checkpointing,
            self.vision_gradient_checkpointing_blocks,
        )
        if self.action_fusion_type == QWEN_ACTION_FUSION_NATIVE_QWEN:
            # ``parameters()`` deduplicates shared tensors, so these are the
            # true trainable sizes of the policy Qwen and its action expert.
            qwen_params = sum(p.numel() for p in self.dit["qwen"].parameters())
            expert_ids = {id(p) for p in self.dit["qwen"].parameters()}
            expert_own = sum(
                p.numel()
                for p in self.dit["action_expert"].parameters()
                if id(p) not in expert_ids
            )
            total_trainable = sum(p.numel() for p in self.dit.parameters())
            logger.info(
                "JAMQwen35 native action: qwen_params=%d action_expert_own_params=%d "
                "dit_unique_params=%d",
                qwen_params,
                expert_own,
                total_trainable,
            )

    def _native_initialization_tag(self) -> str:
        return QWEN_NATIVE_COPY_INITIALIZATION

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
    def action_expert(self) -> nn.Module:
        return self.dit["action_expert"]

    @property
    def foresight_predictors(self) -> nn.ModuleList:
        if self.foresight_predictor_type != QWEN_PREDICTOR_LEGACY:
            raise AttributeError("The JAMPolicy predictor uses `foresight_projector`, not a predictor list.")
        return self.dit["foresight_predictors"]

    @property
    def foresight_projector(self) -> JAMForesightProjector:
        if self.foresight_predictor_type != QWEN_PREDICTOR_JAM:
            raise AttributeError("The legacy predictor uses `foresight_predictors`, not a shared projector.")
        return self.dit["foresight_projector"]

    @property
    def proprio_projector(self) -> nn.Module:
        return self.dit["proprio_projector"]

    @property
    def teacher_backbone(self) -> Optional[nn.Module]:
        if self._teacher_ref is None:
            return None
        return self._teacher_ref[0]

    def apply_trainable_policy(self) -> None:
        """Freeze the training-only foresight head when foresight is disabled.

        The projector/predictors are always constructed so that checkpoints
        load strictly at evaluation.  With ``foresight.enabled=false`` they never
        receive gradients, so they are excluded from the optimizer and from the
        trainable-gradient audit.  The trainer calls this hook right after
        ``model.dit.requires_grad_(True)``.
        """
        if self.foresight_enabled:
            return
        frozen = 0
        for key in ("foresight_projector", "foresight_predictors"):
            if key not in self.dit:
                continue
            module = self.dit[key]
            module.eval()
            module.requires_grad_(False)
            frozen += sum(parameter.numel() for parameter in module.parameters())
        if frozen:
            logger.info(
                "Foresight disabled (action-only training): froze %d training-only "
                "predictor parameters.",
                frozen,
            )

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
        predictor_type: str = QWEN_PREDICTOR_JAM,
        predictor_hidden_dim: int = 1024,
        predictor_ffn_dim: int = 4096,
        predictor_num_layers: int = 2,
        predictor_num_heads: int = 16,
        tensor_native_preprocessing: bool = False,
        batch_equal_grid_vision_attention: bool = False,
        vision_gradient_checkpointing: bool = False,
        vision_gradient_checkpointing_blocks: Optional[int] = None,
        use_gradient_checkpointing: bool = False,
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
        loss_lambda_foresight: float = 0.3,
        foresight_cfg: Optional[dict[str, Any]] = None,
        device: str = "cuda",
        torch_dtype: torch.dtype = torch.bfloat16,
    ) -> "JAMQwen35":
        try:
            from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration
        except (ImportError, AttributeError) as exc:
            raise ImportError(
                "JAMQwen35 requires Transformers with Qwen3.5 support. "
                "Install the environment of tools/bootstrap.py (Transformers 5.13.1) and use it."
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
        elif vision_gradient_checkpointing:
            checkpointed_vision_blocks = enable_qwen_vision_gradient_checkpointing(
                student_backbone,
                num_blocks=vision_gradient_checkpointing_blocks,
            )
            logger.info(
                "Enabled selective gradient checkpointing for %d/%d student vision "
                "blocks: %s",
                len(checkpointed_vision_blocks),
                len(student_backbone.visual.blocks),
                checkpointed_vision_blocks,
            )
        student_backbone.config.text_config.use_cache = False
        normalized_action_fusion = str(action_fusion_type).strip().lower()
        if normalized_action_fusion != QWEN_ACTION_FUSION_NATIVE_QWEN:
            student_backbone.to(device=device)

        teacher_backbone = None
        load_teacher = bool(foresight_cfg.get("enabled", True)) and bool(
            foresight_cfg.get("load_teacher", True)
        )
        # Validate the teacher mode before loading any teacher weights.
        normalize_qwen_teacher_mode(
            foresight_cfg.get("teacher_mode", "frozen_pretrained")
        )
        if load_teacher:
            logger.info("Loading frozen pretrained Qwen3.5 teacher from %s", teacher_model_id)
            teacher_full = Qwen3_5ForConditionalGeneration.from_pretrained(
                teacher_model_id,
                **qwen_load_kwargs,
            )
            teacher_backbone = teacher_full.model
            del teacher_full
            teacher_backbone.config.text_config.use_cache = False
            teacher_backbone.eval()
            teacher_backbone.requires_grad_(False)

        if normalized_action_fusion == QWEN_ACTION_FUSION_NATIVE_QWEN:
            if action_dit_pretrained_path not in {None, "", "none"}:
                raise ValueError(
                    "Native Qwen action blocks are initialized from the policy Qwen; "
                    "`action_dit_pretrained_path` must be null."
                )
            if action_dit_pretrained_layer_mapping not in {None, "", "none"}:
                raise ValueError(
                    "Native Qwen action uses an exact 24-to-24 copy; "
                    "`action_dit_pretrained_layer_mapping` must be null."
                )
            if bool(skip_dit_load_from_pretrain):
                raise ValueError(
                    "Native Qwen action requires source-Qwen initialization; "
                    "`skip_dit_load_from_pretrain` must be false."
                )
            action_expert = ActionDiTQwen35Native.from_qwen_backbone(
                student_backbone,
                device=device,
                torch_dtype=torch_dtype,
                **dict(action_dit_config or {}),
            )
            student_backbone.to(device=device)
        elif normalized_action_fusion == QWEN_ACTION_FUSION_YAK_LAYERWISE_MOT:
            from .action_dit_yak import ActionDiTYak

            if action_dit_pretrained_layer_mapping not in {None, "", "none"}:
                raise ValueError(
                    "ActionDiTYak loads its native 18-layer checkpoint directly; "
                    "`action_dit_pretrained_layer_mapping` must be null."
                )
            action_expert = ActionDiTYak.from_pretrained(
                action_dit_config=dict(action_dit_config or {}),
                action_dit_pretrained_path=action_dit_pretrained_path,
                skip_dit_load_from_pretrain=bool(skip_dit_load_from_pretrain),
                device=device,
                torch_dtype=torch_dtype,
            )
        else:
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
            predictor_type=predictor_type,
            predictor_hidden_dim=predictor_hidden_dim,
            predictor_ffn_dim=predictor_ffn_dim,
            predictor_num_layers=predictor_num_layers,
            predictor_num_heads=predictor_num_heads,
            tensor_native_preprocessing=tensor_native_preprocessing,
            batch_equal_grid_vision_attention=batch_equal_grid_vision_attention,
            vision_gradient_checkpointing=vision_gradient_checkpointing,
            vision_gradient_checkpointing_blocks=vision_gradient_checkpointing_blocks,
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
        # Horizontal camera panels are already adjacent in storage.  The old
        # ``split + stack`` path copied the complete RGB tensor once for the
        # current frame and once for the future frame.  Reshape/permute is an
        # exact strided view of the same elements; the following Qwen pixel
        # arithmetic materializes its own output anyway.
        return frame.reshape(
            frame.shape[0],
            frame.shape[1],
            self.image_size,
            self.num_views,
            self.image_size,
        ).permute(0, 3, 1, 2, 4)

    @staticmethod
    def _to_uint8_image(view: torch.Tensor) -> torch.Tensor:
        return (
            ((view.detach().float().cpu() + 1.0) * 127.5)
            .round()
            .clamp_(0, 255)
            .to(torch.uint8)
        )

    @staticmethod
    def _to_uint8_images(views: torch.Tensor) -> torch.Tensor:
        """Convert an image batch with one device-to-host transfer.

        ``Accelerator`` places the collated video on the GPU before the model
        sees it, while Qwen's image processor runs on the CPU.  Converting one
        view at a time caused ``B * N`` synchronizing D2H copies.  The batched
        form uses the identical float32 arithmetic and rounding, but performs
        a single transfer for the whole contiguous tensor.
        """

        return (
            ((views.detach().float().cpu() + 1.0) * 127.5)
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
        *,
        prepared_pixels: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> dict[str, torch.Tensor]:
        if views.ndim != 5:
            raise ValueError(f"Qwen views must be [B,N,3,H,W], got {tuple(views.shape)}.")
        if views.shape[0] != len(texts):
            raise ValueError(f"Text/image batch mismatch: {len(texts)} vs {views.shape[0]}.")

        if self.tensor_native_preprocessing:
            return self._prepare_qwen_inputs_tensor_native(
                views,
                texts,
                prepared_pixels=prepared_pixels,
            )

        if prepared_pixels is not None:
            raise ValueError(
                "Prepared Qwen pixels are supported only by tensor-native preprocessing."
            )

        uint8_views = self._to_uint8_images(views)
        images = [
            uint8_views[batch_idx, image_idx]
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

    def _prepare_qwen_pixels_tensor_native(
        self,
        views: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Pack fixed-size RGB views exactly as the official Qwen processor.

        This image-only helper lets training reuse the identical current-frame
        pixels between the trainable student and frozen teacher.  The two
        backbones still execute independently; only deterministic input
        quantization/patch packing is shared.
        """

        if views.ndim != 5:
            raise ValueError(f"Qwen views must be [B,N,3,H,W], got {tuple(views.shape)}.")
        batch_size, num_images, channels, height, width = map(int, views.shape)
        if channels != 3 or height != self.image_size or width != self.image_size:
            raise ValueError(
                "Tensor-native Qwen preprocessing requires fixed RGB image_size: "
                f"expected [B,N,3,{self.image_size},{self.image_size}], got "
                f"{tuple(views.shape)}."
            )
        image_processor = self.processor.image_processor
        patch_size = int(image_processor.patch_size)
        temporal_patch_size = int(image_processor.temporal_patch_size)
        merge_size = int(image_processor.merge_size)
        if (
            patch_size != 16
            or temporal_patch_size != 2
            or merge_size != 2
            or tuple(float(x) for x in image_processor.image_mean) != (0.5, 0.5, 0.5)
            or tuple(float(x) for x in image_processor.image_std) != (0.5, 0.5, 0.5)
        ):
            raise ValueError(
                "Tensor-native preprocessing was validated only for Qwen3.5's "
                "patch=16 temporal=2 merge=2 mean/std=0.5 image processor."
            )
        grid_h, grid_w = height // patch_size, width // patch_size
        if grid_h % merge_size or grid_w % merge_size:
            raise ValueError(
                f"Qwen image grid {(grid_h, grid_w)} is not divisible by merge={merge_size}."
            )

        if views.dtype == torch.uint8:
            # The optional Qwen DataLoader path has already applied the exact
            # official uint8 round/clamp on CPU.  Transfer only one byte per
            # channel and normalize once on device.
            quantized_views = views.detach().to(dtype=torch.float32)
            quantized_views.sub_(127.5).div_(127.5)
        else:
            # Preserve the official arithmetic order but reuse storage after
            # each elementwise operation. ``add`` deliberately allocates
            # because the input video belongs to the training batch; all
            # subsequent operations are safe in-place transformations of that
            # private buffer.
            quantized_views = views.detach().to(dtype=torch.float32).add(1.0)
            quantized_views.mul_(127.5).round_().clamp_(0, 255)
            quantized_views.sub_(127.5).div_(127.5)
        patches = quantized_views.to(dtype=self.torch_dtype).reshape(
            batch_size * num_images,
            channels,
            grid_h // merge_size,
            merge_size,
            patch_size,
            grid_w // merge_size,
            merge_size,
            patch_size,
        ).permute(0, 2, 5, 3, 6, 1, 4, 7)
        pixel_values = (
            patches.unsqueeze(6)
            .expand(-1, -1, -1, -1, -1, -1, temporal_patch_size, -1, -1)
            .reshape(
                batch_size * num_images * grid_h * grid_w,
                channels * temporal_patch_size * patch_size * patch_size,
            )
        )
        image_grid_thw = torch.empty(
            (batch_size * num_images, 3),
            device=views.device,
            dtype=torch.long,
        )
        image_grid_thw[:, 0] = 1
        image_grid_thw[:, 1] = grid_h
        image_grid_thw[:, 2] = grid_w
        return pixel_values, image_grid_thw

    def _prepare_qwen_inputs_tensor_native(
        self,
        views: torch.Tensor,
        texts: Sequence[str],
        *,
        prepared_pixels: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> dict[str, torch.Tensor]:
        """Exact fixed-224 Qwen preprocessing without a GPU→CPU→GPU image trip.

        ``Qwen2VLImageProcessor`` first reconstructs uint8 RGB, applies the
        fused torchvision normalization ``(x - 127.5) / 127.5``, duplicates
        the single image for temporal patch size two, and packs 16x16 patches.
        All RoboTwin Qwen views are already 224x224, so resize is an identity.
        Keeping those operations tensor-native produces the same BF16
        ``pixel_values`` while avoiding hundreds of CPU image objects per
        rank/step.  Token IDs are immutable functions of the prompt and fixed
        image grid and are cached per exact expanded string.
        """

        batch_size, num_images, channels, height, width = map(int, views.shape)
        if channels != 3 or height != self.image_size or width != self.image_size:
            raise ValueError(
                "Tensor-native Qwen preprocessing requires fixed RGB image_size: "
                f"expected [B,N,3,{self.image_size},{self.image_size}], got "
                f"{tuple(views.shape)}."
            )
        image_processor = self.processor.image_processor
        patch_size = int(image_processor.patch_size)
        temporal_patch_size = int(image_processor.temporal_patch_size)
        merge_size = int(image_processor.merge_size)
        if (
            patch_size != 16
            or temporal_patch_size != 2
            or merge_size != 2
            or tuple(float(x) for x in image_processor.image_mean) != (0.5, 0.5, 0.5)
            or tuple(float(x) for x in image_processor.image_std) != (0.5, 0.5, 0.5)
        ):
            raise ValueError(
                "Tensor-native preprocessing was validated only for Qwen3.5's "
                "patch=16 temporal=2 merge=2 mean/std=0.5 image processor."
            )
        grid_h, grid_w = height // patch_size, width // patch_size
        if grid_h % merge_size or grid_w % merge_size:
            raise ValueError(
                f"Qwen image grid {(grid_h, grid_w)} is not divisible by merge={merge_size}."
            )

        if prepared_pixels is None:
            pixel_values, image_grid_thw = self._prepare_qwen_pixels_tensor_native(
                views
            )
        else:
            pixel_values, image_grid_thw = prepared_pixels
            expected_pixel_shape = (
                batch_size * num_images * grid_h * grid_w,
                channels * temporal_patch_size * patch_size * patch_size,
            )
            expected_grid_shape = (batch_size * num_images, 3)
            if tuple(pixel_values.shape) != expected_pixel_shape:
                raise ValueError(
                    "Prepared Qwen pixel shape mismatch: "
                    f"expected {expected_pixel_shape}, got {tuple(pixel_values.shape)}."
                )
            if tuple(image_grid_thw.shape) != expected_grid_shape:
                raise ValueError(
                    "Prepared Qwen grid shape mismatch: "
                    f"expected {expected_grid_shape}, got {tuple(image_grid_thw.shape)}."
                )
            if pixel_values.device != views.device or image_grid_thw.device != views.device:
                raise ValueError("Prepared Qwen pixels must be on the same device as views.")
            if pixel_values.dtype != self.torch_dtype or image_grid_thw.dtype != torch.long:
                raise ValueError(
                    "Prepared Qwen pixels must use model dtype/long grid, got "
                    f"{pixel_values.dtype}/{image_grid_thw.dtype}."
                )

        image_tokens_per_view = grid_h * grid_w // (merge_size * merge_size)
        if image_tokens_per_view != self.visual_tokens_per_view:
            raise RuntimeError(
                "Tensor-native Qwen token-grid mismatch: "
                f"processor={image_tokens_per_view}, model={self.visual_tokens_per_view}."
            )
        expanded_texts = [
            text.replace(
                "<|image_pad|>",
                "<|image_pad|>" * image_tokens_per_view,
            )
            for text in texts
        ]
        tokenizer = self.processor.tokenizer
        resolved_entries = {}
        missing_texts = []
        for text in dict.fromkeys(expanded_texts):
            entry = self._tokenized_layout_cache.get(text)
            if entry is None:
                missing_texts.append(text)
            else:
                self._tokenized_layout_cache.move_to_end(text)
                resolved_entries[text] = entry

        # Fast-tokenizer batching is exactly equivalent to invoking the same
        # tokenizer once per string, while avoiding B Python/Rust crossings on
        # RoboTwin batches whose instructions are usually unique.
        if missing_texts:
            tokenized_batch = tokenizer(
                missing_texts,
                add_special_tokens=True,
                padding=False,
                return_attention_mask=False,
                return_token_type_ids=False,
            )["input_ids"]
            if len(tokenized_batch) != len(missing_texts):
                raise RuntimeError(
                    "Qwen tokenizer batch-size mismatch: "
                    f"{len(tokenized_batch)} vs {len(missing_texts)}."
                )
            for text, tokenized in zip(missing_texts, tokenized_batch):
                token_ids = tuple(int(token) for token in tokenized)
                positions = self._build_fixed_image_position_row(
                    token_ids,
                    image_token_id=self.image_token_id,
                    grid_h=grid_h,
                    grid_w=grid_w,
                    merge_size=merge_size,
                    expected_num_images=num_images,
                )
                entry = (token_ids, positions, {})
                resolved_entries[text] = entry
                self._tokenized_layout_cache[text] = entry

        encoded_rows = []
        position_rows = []
        image_index_rows = []
        device_name = str(views.device)
        for text in expanded_texts:
            cached, cached_positions, device_entries = resolved_entries[text]
            cached_device = device_entries.get(device_name)
            if cached_device is None:
                id_tensor = torch.tensor(cached, device=views.device, dtype=torch.long)
                # pad_sequence operates on the leading dimension, hence [L,3].
                position_tensor = torch.tensor(
                    cached_positions, device=views.device, dtype=torch.long
                ).transpose(0, 1)
                image_indices = torch.tensor(
                    [
                        idx
                        for idx, token in enumerate(cached)
                        if token == self.image_token_id
                    ],
                    device=views.device,
                    dtype=torch.long,
                )
                cached_device = (id_tensor, position_tensor, image_indices)
                device_entries[device_name] = cached_device
            encoded_rows.append(cached_device[0])
            position_rows.append(cached_device[1])
            image_index_rows.append(cached_device[2])

        while (
            len(self._tokenized_layout_cache)
            > self._tokenized_cache_max_entries
        ):
            self._tokenized_layout_cache.popitem(last=False)

        input_ids = torch.nn.utils.rnn.pad_sequence(
            encoded_rows,
            batch_first=True,
            padding_value=int(tokenizer.pad_token_id),
        )
        lengths = torch.tensor(
            [int(row.shape[0]) for row in encoded_rows],
            device=views.device,
            dtype=torch.long,
        )
        attention_mask = (
            torch.arange(input_ids.shape[1], device=views.device).unsqueeze(0)
            < lengths.unsqueeze(1)
        ).to(dtype=torch.long)
        position_ids = torch.nn.utils.rnn.pad_sequence(
            position_rows,
            batch_first=True,
            padding_value=0,
        ).permute(2, 0, 1).contiguous()
        mm_token_type_ids = input_ids.eq(self.image_token_id).to(dtype=torch.long)
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "pixel_values": pixel_values,
            "image_grid_thw": image_grid_thw,
            "mm_token_type_ids": mm_token_type_ids,
            "position_ids": position_ids,
            # Consumed by JAMQwen35 before calling the HF backbone.  Fixed
            # gather indices avoid dynamic-size CUDA boolean indexing when
            # extracting layer-13 visual tokens.
            "_image_token_indices": torch.stack(image_index_rows, dim=0),
        }

    @staticmethod
    def _build_fixed_image_position_row(
        input_ids: Sequence[int],
        *,
        image_token_id: int,
        grid_h: int,
        grid_w: int,
        merge_size: int,
        expected_num_images: int,
    ) -> tuple[tuple[int, ...], ...]:
        """Reproduce Qwen3.5 M-RoPE positions once for a fixed image grid.

        Transformers' generic routine iterates over every batch row and calls
        ``.tolist()``/``.item()`` on CUDA tensors.  RoboTwin always uses the
        same 224x224 image grid, while token rows are immutable functions of
        the prompt, so the exact positions can be cached per expanded prompt.
        """

        grid_h = int(grid_h)
        grid_w = int(grid_w)
        merge_size = int(merge_size)
        merged_h = grid_h // merge_size
        merged_w = grid_w // merge_size
        expected_image_tokens = merged_h * merged_w
        positions = [[], [], []]
        current_position = 0
        image_groups = 0
        start = 0
        tokens = tuple(int(token) for token in input_ids)
        while start < len(tokens):
            is_image = tokens[start] == int(image_token_id)
            end = start + 1
            while end < len(tokens) and (
                tokens[end] == int(image_token_id)
            ) == is_image:
                end += 1
            group_len = end - start
            if not is_image:
                row = list(range(current_position, current_position + group_len))
                for axis in positions:
                    axis.extend(row)
                current_position += group_len
            else:
                if group_len != expected_image_tokens:
                    raise ValueError(
                        "Fixed-grid Qwen image-token run mismatch: "
                        f"expected {expected_image_tokens}, got {group_len}."
                    )
                positions[0].extend([current_position] * expected_image_tokens)
                positions[1].extend(
                    current_position + h
                    for h in range(merged_h)
                    for _ in range(merged_w)
                )
                positions[2].extend(
                    current_position + w
                    for _ in range(merged_h)
                    for w in range(merged_w)
                )
                current_position += max(grid_h, grid_w) // merge_size
                image_groups += 1
            start = end
        if image_groups != int(expected_num_images):
            raise ValueError(
                "Fixed-grid Qwen image-group mismatch: "
                f"expected {expected_num_images}, got {image_groups}."
            )
        return tuple(tuple(int(value) for value in axis) for axis in positions)

    def _encode_student(
        self,
        current_views: torch.Tensor,
        prompts: Sequence[str],
        *,
        prepared_pixels: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        Optional[tuple[torch.Tensor, ...]],
        Optional[tuple[torch.Tensor, ...]],
        Optional[dict[str, Any]],
    ]:
        model_inputs = self._prepare_qwen_inputs(
            current_views,
            self._student_texts(prompts),
            prepared_pixels=prepared_pixels,
        )
        image_token_indices = model_inputs.pop("_image_token_indices", None)
        decoder_inputs = None
        decoder_outputs = None
        native_action_inputs = None
        if self.action_fusion_type == QWEN_ACTION_FUSION_NATIVE_QWEN:
            position_ids = model_inputs.get("position_ids")
            if position_ids is None:
                position_ids = self.student_backbone.compute_3d_position_ids(
                    input_ids=model_inputs["input_ids"],
                    inputs_embeds=None,
                    image_grid_thw=model_inputs["image_grid_thw"],
                    video_grid_thw=None,
                    attention_mask=model_inputs["attention_mask"],
                    past_key_values=None,
                    mm_token_type_ids=model_inputs["mm_token_type_ids"],
                )
            if position_ids is None:
                raise RuntimeError("Qwen native action could not compute multimodal position IDs.")
            model_inputs["position_ids"] = position_ids
            output, foresight_hidden, decoder_outputs = forward_with_decoder_layer_capture(
                self.student_backbone,
                layer=self.foresight_layer,
                model_inputs=model_inputs,
                return_decoder_outputs=True,
            )
            # The final policy prefix uses Qwen's final RMSNorm so every source
            # Qwen parameter participates in action training, just as it does
            # in the ordinary final-context policy.
            decoder_outputs = (*decoder_outputs[:-1], output.last_hidden_state)
            native_action_inputs = {
                "layerwise_prefix_tokens": decoder_outputs,
                "prefix_attention_mask": model_inputs["attention_mask"].to(
                    dtype=torch.bool
                ),
                "prefix_position_ids": position_ids,
            }
        elif self.action_fusion_type in {
            QWEN_ACTION_FUSION_LAYERWISE_MOT,
            QWEN_ACTION_FUSION_YAK_LAYERWISE_MOT,
        }:
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
        grouped_visual = extract_grouped_visual_tokens(
            foresight_hidden,
            model_inputs["input_ids"],
            image_token_id=self.image_token_id,
            num_images_per_sample=self.num_views,
            visual_tokens_per_image=self.visual_tokens_per_view,
            token_indices=image_token_indices,
        )
        current_visual = grouped_visual.flatten(1, 2)
        context = self.context_projector(output.last_hidden_state)
        context_mask = model_inputs["attention_mask"].to(dtype=torch.bool)
        layerwise_visual_tokens = None
        layerwise_text_tokens = None
        if decoder_inputs is not None:
            if self.action_fusion_type == QWEN_ACTION_FUSION_YAK_LAYERWISE_MOT:
                selected_decoder_inputs = tuple(
                    decoder_inputs[layer_idx]
                    for layer_idx in self.action_visual_layer_map
                )
            else:
                selected_decoder_inputs = decoder_inputs
            embedded_visual_tokens = []
            for decoder_input in selected_decoder_inputs:
                layer_visual = extract_grouped_visual_tokens(
                    decoder_input,
                    model_inputs["input_ids"],
                    image_token_id=self.image_token_id,
                    num_images_per_sample=self.num_views,
                    visual_tokens_per_image=self.visual_tokens_per_view,
                ).flatten(1, 2)
                embedded_visual_tokens.append(self.mot_visual_projector(layer_visual))
            layerwise_visual_tokens = tuple(embedded_visual_tokens)
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
            if self.action_fusion_type == QWEN_ACTION_FUSION_YAK_LAYERWISE_MOT:
                embedded_text_tokens = []
                for decoder_input in selected_decoder_inputs:
                    projected_text = self.context_projector(decoder_input)
                    embedded_text_tokens.append(
                        torch.stack(
                            [
                                F.pad(
                                    projected_text[batch_idx, text_mask[batch_idx]],
                                    (
                                        0,
                                        0,
                                        0,
                                        max_text_len
                                        - int(text_lengths[batch_idx].item()),
                                    ),
                                )
                                for batch_idx in range(projected_text.shape[0])
                            ],
                            dim=0,
                        )
                    )
                layerwise_text_tokens = tuple(embedded_text_tokens)
        return (
            context,
            context_mask,
            current_visual,
            layerwise_visual_tokens,
            layerwise_text_tokens,
            native_action_inputs,
        )

    @torch.no_grad()
    def _encode_teacher_targets(
        self,
        teacher_views: torch.Tensor,
        prompts: Sequence[str],
        *,
        prepared_pixels: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> list[torch.Tensor]:
        teacher = self.teacher_backbone
        if teacher is None:
            raise RuntimeError(
                "Foresight training requires the frozen teacher, but `foresight.load_teacher=false`."
            )
        model_inputs = self._prepare_qwen_inputs(
            teacher_views,
            self._teacher_texts(prompts),
            prepared_pixels=prepared_pixels,
        )
        image_token_indices = model_inputs.pop("_image_token_indices", None)
        _, foresight_hidden = forward_with_decoder_layer_capture(
            teacher,
            layer=self.foresight_layer,
            model_inputs=model_inputs,
        )
        if image_token_indices is not None:
            image_token_indices = image_token_indices.reshape(
                image_token_indices.shape[0],
                len(self.teacher_frame_indices),
                self.num_views,
                self.visual_tokens_per_view,
            ).index_select(
                1,
                torch.as_tensor(
                    self.teacher_target_group_indices,
                    device=image_token_indices.device,
                    dtype=torch.long,
                ),
            ).flatten(1, 3)
            grouped_visual = extract_grouped_visual_tokens(
                foresight_hidden,
                model_inputs["input_ids"],
                image_token_id=self.image_token_id,
                num_images_per_sample=(
                    len(self.teacher_target_group_indices) * self.num_views
                ),
                visual_tokens_per_image=self.visual_tokens_per_view,
                token_indices=image_token_indices,
            ).reshape(
                foresight_hidden.shape[0],
                len(self.teacher_target_group_indices),
                self.num_views,
                self.visual_tokens_per_view,
                foresight_hidden.shape[-1],
            )
            return [
                grouped_visual[:, target_idx].flatten(1, 2).detach()
                for target_idx in range(len(self.teacher_target_group_indices))
            ]

        grouped_visual = extract_grouped_visual_tokens(
            foresight_hidden,
            model_inputs["input_ids"],
            image_token_id=self.image_token_id,
            num_images_per_sample=len(self.teacher_frame_indices) * self.num_views,
            visual_tokens_per_image=self.visual_tokens_per_view,
            token_indices=image_token_indices,
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
        layerwise_text_tokens: Optional[Sequence[torch.Tensor]] = None,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        Optional[tuple[torch.Tensor, ...]],
    ]:
        if self.proprio_dim is None:
            if proprio is not None:
                raise ValueError("`proprio` was provided but the model has `proprio_dim=null`.")
            return (
                context,
                context_mask,
                None
                if layerwise_text_tokens is None
                else tuple(layerwise_text_tokens),
            )
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
        if self.action_fusion_type == QWEN_ACTION_FUSION_NATIVE_QWEN:
            if layerwise_text_tokens is not None:
                raise RuntimeError(
                    "Native Qwen action fusion received legacy layerwise text tokens."
                )
            # Native ActionQwen consumes the Qwen prefix directly from
            # ``native_action_inputs``.  Carry only the live proprio token
            # instead of allocating a second [B,prefix+1,D] context whose
            # prefix portion is immediately discarded by prepare_condition.
            return proprio_token, proprio_mask, None
        updated_layerwise_text = None
        if layerwise_text_tokens is not None:
            updated_layerwise_text = tuple(
                torch.cat((layer_text, proprio_token), dim=1)
                for layer_text in layerwise_text_tokens
            )
        return (
            torch.cat([context, proprio_token], dim=1),
            torch.cat([context_mask, proprio_mask], dim=1),
            updated_layerwise_text,
        )

    def _validate_training_sample(self, sample: dict[str, Any]) -> tuple[int, ...]:
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
        teacher_frame_positions = resolve_decoded_frame_positions(
            self.teacher_frame_indices,
            num_decoded_frames=int(video.shape[2]),
        )
        if action.shape[-1] != self.action_expert.action_dim:
            raise ValueError(
                f"Action dim must be {self.action_expert.action_dim}, got {action.shape[-1]}."
            )
        return teacher_frame_positions

    def _compute_action_loss(
        self,
        action: torch.Tensor,
        action_is_pad: Optional[torch.Tensor],
        context: torch.Tensor,
        context_mask: torch.Tensor,
        layerwise_visual_tokens: Optional[Sequence[torch.Tensor]],
        layerwise_text_tokens: Optional[Sequence[torch.Tensor]],
        native_action_inputs: Optional[dict[str, Any]],
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
        action_prefix_cache = self._prepare_action_prefix(
            context=context,
            context_mask=context_mask,
            layerwise_visual_tokens=layerwise_visual_tokens,
            layerwise_text_tokens=layerwise_text_tokens,
            native_action_inputs=native_action_inputs,
        )
        prediction = self._run_action_expert(
            action_tokens=noisy_action,
            timestep=timestep,
            context=context,
            context_mask=context_mask,
            layerwise_visual_tokens=layerwise_visual_tokens,
            action_prefix_cache=action_prefix_cache,
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

    def _prepare_action_prefix(
        self,
        *,
        context_mask: torch.Tensor,
        layerwise_visual_tokens: Optional[Sequence[torch.Tensor]],
        layerwise_text_tokens: Optional[Sequence[torch.Tensor]],
        context: torch.Tensor,
        native_action_inputs: Optional[dict[str, Any]],
    ) -> Optional[Any]:
        if self.action_fusion_type == QWEN_ACTION_FUSION_NATIVE_QWEN:
            if native_action_inputs is None:
                raise RuntimeError("Native Qwen action fusion is missing policy prefixes.")
            if layerwise_visual_tokens is not None or layerwise_text_tokens is not None:
                raise RuntimeError(
                    "Native Qwen action fusion received legacy projected layerwise tokens."
                )
            proprio_token = None
            prefix_width = int(native_action_inputs["prefix_attention_mask"].shape[1])
            if self.proprio_dim is not None:
                if int(context.shape[1]) != 1:
                    raise RuntimeError(
                        "Native Qwen proprio packing expected its one live token, got "
                        f"context width {context.shape[1]}."
                    )
                proprio_token = context
            elif int(context.shape[1]) != prefix_width:
                raise RuntimeError(
                    "Native Qwen context/prefix width mismatch without proprio: "
                    f"context={context.shape[1]}, prefix={prefix_width}."
                )
            return self.action_expert.prepare_condition(
                **native_action_inputs,
                proprio_token=proprio_token,
            )
        if self.action_fusion_type != QWEN_ACTION_FUSION_YAK_LAYERWISE_MOT:
            if layerwise_text_tokens is not None:
                raise RuntimeError(
                    "Only Qwen→ActionDiTYak18 accepts layerwise Qwen text tokens."
                )
            return None
        if layerwise_visual_tokens is None or layerwise_text_tokens is None:
            raise RuntimeError(
                "Qwen→ActionDiTYak18 is missing layerwise text or visual tokens."
            )
        visual_grid = int(self.visual_tokens_per_view**0.5)
        if visual_grid * visual_grid != self.visual_tokens_per_view:
            raise ValueError(
                "Qwen→ActionDiTYak18 requires a square per-view visual grid, "
                f"got {self.visual_tokens_per_view} tokens."
            )
        return self.action_expert.prepare_qwen_layerwise_prefix(
            layerwise_text_tokens=layerwise_text_tokens,
            layerwise_visual_tokens=layerwise_visual_tokens,
            text_mask=context_mask,
            visual_grid_height=visual_grid,
            visual_grid_width=visual_grid,
            num_views=self.num_views,
        )

    def _run_action_expert(
        self,
        *,
        action_tokens: torch.Tensor,
        timestep: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        layerwise_visual_tokens: Optional[Sequence[torch.Tensor]],
        action_prefix_cache: Optional[Any] = None,
    ) -> torch.Tensor:
        if self.action_fusion_type == QWEN_ACTION_FUSION_NATIVE_QWEN:
            if not isinstance(action_prefix_cache, Qwen35ActionCondition):
                raise RuntimeError(
                    "Native Qwen action fusion is missing its Qwen35ActionCondition."
                )
            return self.action_expert.forward_native_mot(
                action_tokens=action_tokens,
                timestep=timestep,
                condition=action_prefix_cache,
            )
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
        if self.action_fusion_type == QWEN_ACTION_FUSION_YAK_LAYERWISE_MOT:
            if action_prefix_cache is None:
                raise RuntimeError(
                    "Qwen→ActionDiTYak18 action fusion is missing its prefix cache."
                )
            # ImageWAM's Yak time embedding is defined on [0,1], whereas the
            # shared Qwen scheduler exposes the equivalent [0,1000] index.
            unit_timestep = timestep / float(self.action_num_train_timesteps)
            return self.action_expert.forward_qwen_layerwise_mot(
                action_tokens=action_tokens,
                timestep=unit_timestep,
                prefix_cache=action_prefix_cache,
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
        if self.foresight_predictor_type == QWEN_PREDICTOR_JAM:
            trunk_out = self.foresight_projector.trunk(student_visual)
            predictions = [
                self.foresight_projector.head(trunk_out, idx)
                for idx in range(len(teacher_targets))
            ]
        else:
            predictions = [
                predictor(student_visual) for predictor in self.foresight_predictors
            ]

        weighted_losses = []
        for prediction, target, valid, weight in zip(
            predictions,
            teacher_targets,
            target_valid.unbind(dim=1),
            self.offset_weights,
        ):
            per_sample = (
                1.0 - F.cosine_similarity(prediction.float(), target.float(), dim=-1)
            ).mean(dim=1)
            valid = valid.to(device=per_sample.device, dtype=per_sample.dtype)
            valid_count = valid.sum()
            valid_loss = (per_sample * valid).sum() / valid_count.clamp(min=1.0)
            # Keep the old explicit zero-gradient path for an all-padded
            # target, but select it on device instead of synchronizing the CPU
            # with ``valid.any().item()`` every training step.
            offset_loss = torch.where(
                valid_count > 0,
                valid_loss,
                prediction.float().sum() * 0.0,
            )
            weighted_losses.append(float(weight) * offset_loss)
        return torch.stack(weighted_losses).sum() / float(sum(self.offset_weights))

    def training_loss(self, sample: dict[str, Any]):
        teacher_frame_positions = self._validate_training_sample(sample)
        video = sample["video"]
        batch_size = video.shape[0]
        prompts = self._normalize_prompts(sample["prompt"], batch_size)

        current_frame = video[:, :, 0]
        current_views = self._split_views(current_frame)
        teacher_views = None
        teacher_prepared_pixels = None
        student_prepared_pixels = None
        teacher_targets = None
        if self.foresight_enabled:
            split_views_by_position = {0: current_views}
            for frame_position in teacher_frame_positions:
                if frame_position not in split_views_by_position:
                    split_views_by_position[frame_position] = self._split_views(
                        video[:, :, frame_position]
                    )
            teacher_views = torch.cat(
                [
                    split_views_by_position[frame_position]
                    for frame_position in teacher_frame_positions
                ],
                dim=1,
            )
            if self.tensor_native_preprocessing:
                teacher_prepared_pixels = self._prepare_qwen_pixels_tensor_native(
                    teacher_views
                )
                teacher_pixel_values, teacher_grid_thw = teacher_prepared_pixels
                total_teacher_images = int(teacher_views.shape[1])
                patch_rows_per_image = int(teacher_pixel_values.shape[0]) // (
                    int(batch_size) * total_teacher_images
                )
                if (
                    patch_rows_per_image * int(batch_size) * total_teacher_images
                    != int(teacher_pixel_values.shape[0])
                ):
                    raise RuntimeError("Qwen prepared-pixel rows are not image-aligned.")
                # Current is always the first teacher frame by
                # resolve_teacher_frame_layout().  Select those already packed
                # rows for the student instead of quantizing/patchifying the
                # same augmented RGB views a second time.
                student_pixel_values = teacher_pixel_values.reshape(
                    int(batch_size),
                    total_teacher_images,
                    patch_rows_per_image,
                    int(teacher_pixel_values.shape[-1]),
                )[:, : self.num_views].clone().reshape(
                    int(batch_size) * self.num_views * patch_rows_per_image,
                    int(teacher_pixel_values.shape[-1]),
                )
                student_grid_thw = teacher_grid_thw.reshape(
                    int(batch_size), total_teacher_images, 3
                )[:, : self.num_views].clone().reshape(
                    int(batch_size) * self.num_views, 3
                )
                student_prepared_pixels = (
                    student_pixel_values,
                    student_grid_thw,
                )
            if self.teacher_first:
                teacher_targets = self._encode_teacher_targets(
                    teacher_views,
                    prompts,
                    prepared_pixels=teacher_prepared_pixels,
                )
                # Do not retain the full six-view packed input while building
                # the trainable student/action graph.  The cloned three-view
                # student rows and compact target tokens remain live.
                teacher_views = None
                teacher_prepared_pixels = None
                if self.tensor_native_preprocessing:
                    del teacher_pixel_values, teacher_grid_thw
        (
            context,
            context_mask,
            student_visual,
            layerwise_visual_tokens,
            layerwise_text_tokens,
            native_action_inputs,
        ) = self._encode_student(
            current_views,
            prompts,
            prepared_pixels=student_prepared_pixels,
        )
        context, context_mask, layerwise_text_tokens = self._append_proprio_context(
            context,
            context_mask,
            sample.get("proprio"),
            layerwise_text_tokens,
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
            layerwise_text_tokens=layerwise_text_tokens,
            native_action_inputs=native_action_inputs,
        )

        loss_foresight = loss_action.new_zeros(())
        if self.foresight_enabled:
            if teacher_targets is None:
                if teacher_views is None:
                    raise RuntimeError("Foresight teacher views were not prepared.")
                teacher_targets = self._encode_teacher_targets(
                    teacher_views,
                    prompts,
                    prepared_pixels=teacher_prepared_pixels,
                )
            target_frame_positions = tuple(
                teacher_frame_positions[group_idx]
                for group_idx in self.teacher_target_group_indices
            )
            target_valid = future_target_valid_mask(
                sample.get("image_is_pad"),
                target_frame_positions,
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
        metrics = {
            # Keep metrics on device. Trainer gathers/materializes them only
            # on logging steps when gather_metrics_every_step=false, avoiding
            # two metric-only CUDA synchronizations on every optimizer step.
            "loss_action": self.loss_lambda_action * loss_action.detach(),
            "loss_foresight": self.loss_lambda_foresight * loss_foresight.detach(),
        }
        return loss_total, metrics

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
        (
            context,
            context_mask,
            _,
            layerwise_visual_tokens,
            layerwise_text_tokens,
            native_action_inputs,
        ) = self._encode_student(views, [str(prompt)])
        context, context_mask, layerwise_text_tokens = self._append_proprio_context(
            context,
            context_mask,
            proprio,
            layerwise_text_tokens,
        )
        action_prefix_cache = self._prepare_action_prefix(
            context=context,
            context_mask=context_mask,
            layerwise_visual_tokens=layerwise_visual_tokens,
            layerwise_text_tokens=layerwise_text_tokens,
            native_action_inputs=native_action_inputs,
        )

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
                action_prefix_cache=action_prefix_cache,
            )
            action = self.infer_action_scheduler.step(prediction, step_delta, action)
        return {"action": action[0].detach().to(device="cpu", dtype=torch.float32)}

    def save_checkpoint(self, path, optimizer=None, step=None):
        if self.action_fusion_type == QWEN_ACTION_FUSION_NATIVE_QWEN:
            checkpoint_format = QWEN_JAM_NATIVE_CHECKPOINT_FORMAT
        elif self.action_fusion_type == QWEN_ACTION_FUSION_YAK_LAYERWISE_MOT:
            checkpoint_format = QWEN_JAM_YAK_CHECKPOINT_FORMAT
        else:
            checkpoint_format = (
                QWEN_JAM_CHECKPOINT_FORMAT
                if self.foresight_predictor_type == QWEN_PREDICTOR_JAM
                else QWEN_WAN_CHECKPOINT_FORMAT
            )
        payload = {
            "format": checkpoint_format,
            "dit": self.dit.state_dict(),
            "step": step,
            "model_id": self.model_id,
            "teacher_model_id": self.teacher_model_id,
            "teacher_mode": self.foresight_teacher_mode,
            "teacher_random_seed": self.foresight_teacher_random_seed,
            "future_frame_offsets": self.future_frame_offsets,
            "foresight_layer": self.foresight_layer,
            "teacher_early_exit": self.teacher_early_exit,
            "loss_lambda_foresight": self.loss_lambda_foresight,
            "foresight_predictor_type": self.foresight_predictor_type,
            "foresight_predictor_config": self.foresight_predictor_config,
            "action_fusion_type": self.action_fusion_type,
            "qwen_num_layers": self.qwen_num_layers,
            "action_num_layers": self.action_num_layers,
            "action_visual_layer_map": self.action_visual_layer_map,
            "action_expert_family": type(self.action_expert).__name__,
        }
        if self.action_fusion_type == QWEN_ACTION_FUSION_NATIVE_QWEN:
            payload.update(
                {
                    "action_initialization": self._native_initialization_tag(),
                    # Retained so the native checkpoint payload keeps its
                    # established key set: the action expert owns block copies
                    # and is a separate MoT stream.
                    "action_shares_backbone_weights": False,
                    "action_single_stream": False,
                    "action_layer_types": tuple(self.action_expert.layer_types),
                    "action_hidden_dim": int(self.action_expert.hidden_dim),
                    "action_attention_mode": self.action_expert.action_attention_mode,
                }
            )
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
            QWEN_WAN_LEGACY_CHECKPOINT_FORMAT,
            QWEN_WAN_CHECKPOINT_FORMAT,
            QWEN_JAM_LEGACY_CHECKPOINT_FORMAT,
            QWEN_JAM_CHECKPOINT_FORMAT,
            QWEN_JAM_YAK_CHECKPOINT_FORMAT,
            QWEN_JAM_NATIVE_CHECKPOINT_FORMAT,
        }:
            raise ValueError(
                f"Unsupported JAMQwen35 checkpoint format in {path}: {payload.get('format')!r}."
            )
        saved_predictor_type = payload.get("foresight_predictor_type", None)
        if saved_predictor_type is None:
            saved_predictor_type = (
                QWEN_PREDICTOR_LEGACY
                if checkpoint_format in {
                    QWEN_WAN_LEGACY_CHECKPOINT_FORMAT,
                    QWEN_WAN_CHECKPOINT_FORMAT,
                }
                else QWEN_PREDICTOR_JAM
            )
        if saved_predictor_type != self.foresight_predictor_type:
            raise ValueError(
                "Foresight predictor mismatch while loading: "
                f"checkpoint={saved_predictor_type!r}, model={self.foresight_predictor_type!r}. "
                "Use predictor_type='residual_mlp' only for legacy JAMWan-Qwen checkpoints."
            )
        saved_predictor_config = payload.get("foresight_predictor_config", None)
        if saved_predictor_config is not None and dict(saved_predictor_config) != self.foresight_predictor_config:
            raise ValueError(
                "Foresight predictor config mismatch while loading: "
                f"checkpoint={dict(saved_predictor_config)}, model={self.foresight_predictor_config}."
            )
        saved_action_fusion = payload.get(
            "action_fusion_type", QWEN_ACTION_FUSION_FINAL
        )
        if saved_action_fusion != self.action_fusion_type:
            raise ValueError(
                "Action fusion mismatch while loading: "
                f"checkpoint={saved_action_fusion!r}, model={self.action_fusion_type!r}."
            )
        saved_action_layers = int(
            payload.get("action_num_layers", len(self.action_expert.blocks))
        )
        if saved_action_layers != self.action_num_layers:
            raise ValueError(
                "Action depth mismatch while loading: "
                f"checkpoint={saved_action_layers}, model={self.action_num_layers}."
            )
        saved_layer_map = tuple(
            int(layer_idx)
            for layer_idx in payload.get(
                "action_visual_layer_map", self.action_visual_layer_map
            )
        )
        if saved_layer_map != self.action_visual_layer_map:
            raise ValueError(
                "Qwen/action layer map mismatch while loading: "
                f"checkpoint={saved_layer_map}, model={self.action_visual_layer_map}."
            )
        if self.action_fusion_type == QWEN_ACTION_FUSION_NATIVE_QWEN:
            expected_native_format = QWEN_JAM_NATIVE_CHECKPOINT_FORMAT
            if checkpoint_format != expected_native_format:
                raise ValueError(
                    "Native Qwen action checkpoint format/mask mismatch: "
                    f"checkpoint={checkpoint_format!r}, expected={expected_native_format!r}."
                )
            saved_initialization = payload.get("action_initialization")
            expected_initialization = self._native_initialization_tag()
            if saved_initialization != expected_initialization:
                raise ValueError(
                    "Native Qwen action initialization metadata mismatch: "
                    f"checkpoint={saved_initialization!r}, "
                    f"model={expected_initialization!r}."
                )
            saved_layer_types = tuple(payload.get("action_layer_types", ()))
            if saved_layer_types != tuple(self.action_expert.layer_types):
                raise ValueError(
                    "Native Qwen action layer topology mismatch: "
                    f"checkpoint={saved_layer_types}, "
                    f"model={tuple(self.action_expert.layer_types)}."
                )
            if int(payload.get("action_hidden_dim", -1)) != int(
                self.action_expert.hidden_dim
            ):
                raise ValueError(
                    "Native Qwen action hidden dimension mismatch: "
                    f"checkpoint={payload.get('action_hidden_dim')}, "
                    f"model={self.action_expert.hidden_dim}."
                )
            saved_action_attention_mode = payload.get(
                "action_attention_mode", QWEN_ACTION_ATTENTION_CAUSAL
            )
            if saved_action_attention_mode != self.action_expert.action_attention_mode:
                raise ValueError(
                    "Native Qwen action attention-mode mismatch: "
                    f"checkpoint={saved_action_attention_mode!r}, "
                    f"model={self.action_expert.action_attention_mode!r}."
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
        saved_lambda = payload.get("loss_lambda_foresight", None)
        # The loss weight only matters while the foresight objective is active;
        # evaluation disables foresight and must load action-only checkpoints
        # (lambda 0) with the unchanged task config.
        if (
            self.foresight_enabled
            and saved_lambda is not None
            and float(saved_lambda) != self.loss_lambda_foresight
        ):
            raise ValueError(
                "Foresight lambda mismatch while resuming: "
                f"checkpoint={float(saved_lambda)}, model={self.loss_lambda_foresight}."
            )
        saved_teacher_mode = normalize_qwen_teacher_mode(
            payload.get("teacher_mode", "frozen_pretrained")
        )
        if saved_teacher_mode != self.foresight_teacher_mode:
            raise ValueError(
                "Qwen teacher mode mismatch while loading: "
                f"checkpoint={saved_teacher_mode!r}, model={self.foresight_teacher_mode!r}."
            )
        self.dit.load_state_dict(payload["dit"], strict=True)
        if optimizer is not None and "optimizer" in payload:
            optimizer.load_state_dict(payload["optimizer"])
        return payload

    def forward(self, *args, **kwargs):
        return self.training_loss(*args, **kwargs)


__all__ = [
    "ForesightResidualMLP",
    "QWEN_ACTION_FUSION_FINAL",
    "QWEN_ACTION_FUSION_LAYERWISE_MOT",
    "QWEN_ACTION_FUSION_YAK_LAYERWISE_MOT",
    "QWEN_ACTION_FUSION_NATIVE_QWEN",
    "QWEN_FASTWAM_CHECKPOINT_FORMAT",
    "QWEN_FASTWAM_LEGACY_CHECKPOINT_FORMAT",
    "QWEN_IMAGEWAM_CHECKPOINT_FORMAT",
    "QWEN_IMAGEWAM_LEGACY_CHECKPOINT_FORMAT",
    "QWEN_IMAGEWAM_YAK_CHECKPOINT_FORMAT",
    "QWEN_IMAGEWAM_NATIVE_CHECKPOINT_FORMAT",
    "QWEN_PREDICTOR_IMAGEWAM",
    "QWEN_PREDICTOR_LEGACY",
    "QWEN_TEACHER_MODES",
    "JAMQwen35",
    "extract_grouped_visual_tokens",
    "forward_with_decoder_layer_capture",
    "future_target_valid_mask",
    "uniform_endpoint_layer_map",
    "normalize_qwen_teacher_mode",
    "resolve_decoded_frame_positions",
    "resolve_teacher_frame_layout",
    "truncate_decoder_for_exact_early_exit",
]


# Compatibility aliases for historical configs and imports.
QwenImageWAM = JAMQwen35

# Historical constant aliases; serialized values are unchanged.
QWEN_FASTWAM_LEGACY_CHECKPOINT_FORMAT = QWEN_WAN_LEGACY_CHECKPOINT_FORMAT
QWEN_FASTWAM_CHECKPOINT_FORMAT = QWEN_WAN_CHECKPOINT_FORMAT
QWEN_IMAGEWAM_LEGACY_CHECKPOINT_FORMAT = QWEN_JAM_LEGACY_CHECKPOINT_FORMAT
QWEN_IMAGEWAM_CHECKPOINT_FORMAT = QWEN_JAM_CHECKPOINT_FORMAT
QWEN_IMAGEWAM_YAK_CHECKPOINT_FORMAT = QWEN_JAM_YAK_CHECKPOINT_FORMAT
QWEN_IMAGEWAM_NATIVE_CHECKPOINT_FORMAT = QWEN_JAM_NATIVE_CHECKPOINT_FORMAT
QWEN_PREDICTOR_IMAGEWAM = QWEN_PREDICTOR_JAM

__all__ = list(dict.fromkeys(__all__ + ['JAMQwen35', 'QWEN_FASTWAM_CHECKPOINT_FORMAT', 'QWEN_FASTWAM_LEGACY_CHECKPOINT_FORMAT', 'QWEN_IMAGEWAM_CHECKPOINT_FORMAT', 'QWEN_IMAGEWAM_LEGACY_CHECKPOINT_FORMAT', 'QWEN_IMAGEWAM_NATIVE_CHECKPOINT_FORMAT', 'QWEN_IMAGEWAM_YAK_CHECKPOINT_FORMAT', 'QWEN_JAM_CHECKPOINT_FORMAT', 'QWEN_JAM_LEGACY_CHECKPOINT_FORMAT', 'QWEN_JAM_NATIVE_CHECKPOINT_FORMAT', 'QWEN_JAM_YAK_CHECKPOINT_FORMAT', 'QWEN_PREDICTOR_IMAGEWAM', 'QWEN_PREDICTOR_JAM', 'QWEN_WAN_CHECKPOINT_FORMAT', 'QWEN_WAN_LEGACY_CHECKPOINT_FORMAT', 'QwenImageWAM']))
