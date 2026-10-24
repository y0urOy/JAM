"""Single-frame VideoMAE visual student for the native VideoMAE policy.

The pretrained VideoMAE patch stem consumes two-frame tubelets.  A policy
student must remain current-only, so feeding a fabricated 16-frame constant
clip would be both wasteful and semantically misleading.  This module folds
the temporal kernel into a Conv2d:

``W_image = W_tubelet[:, :, 0] + W_tubelet[:, :, 1]``.

That is exactly the native tubelet projection of the minimal duplicated pair
``[t, t]`` without constructing or processing any repeated video.  The first
pretrained temporal position block is retained, while all 12 transformer
layers stay bit-for-bit initialized from VideoMAE-B.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


class SingleFrameVideoMAEEmbeddings(nn.Module):
    """Image patch embedding initialized from a two-frame VideoMAE tubelet."""

    def __init__(
        self,
        projection: nn.Conv2d,
        position_embeddings: torch.Tensor,
    ) -> None:
        super().__init__()
        self.projection = projection
        if position_embeddings.ndim != 3 or int(position_embeddings.shape[0]) != 1:
            raise ValueError(
                "Single-frame VideoMAE position embeddings must be [1,P,D], "
                f"got {tuple(position_embeddings.shape)}."
            )
        self.register_buffer(
            "position_embeddings",
            position_embeddings.detach().clone(),
            persistent=True,
        )

    @classmethod
    def from_videomae_embeddings(
        cls,
        embeddings: nn.Module,
        *,
        tokens_per_frame: int,
    ) -> "SingleFrameVideoMAEEmbeddings":
        patch_embeddings = getattr(embeddings, "patch_embeddings", None)
        source = getattr(patch_embeddings, "projection", None)
        if not isinstance(source, nn.Conv3d):
            raise TypeError(
                "Expected VideoMAE embeddings with a Conv3d patch projection."
            )
        if tuple(source.kernel_size[:1]) != (2,) or tuple(source.stride[:1]) != (2,):
            raise ValueError(
                "The current-frame adapter requires VideoMAE tubelet_size=2, "
                f"got kernel/stride={source.kernel_size}/{source.stride}."
            )
        projection = nn.Conv2d(
            in_channels=int(source.in_channels),
            out_channels=int(source.out_channels),
            kernel_size=tuple(int(value) for value in source.kernel_size[1:]),
            stride=tuple(int(value) for value in source.stride[1:]),
            padding=tuple(int(value) for value in source.padding[1:]),
            dilation=tuple(int(value) for value in source.dilation[1:]),
            groups=int(source.groups),
            bias=source.bias is not None,
            padding_mode=str(source.padding_mode),
            device=source.weight.device,
            dtype=source.weight.dtype,
        )
        with torch.no_grad():
            # Exact Conv3d([t,t]) equivalence, with the bias applied once.
            projection.weight.copy_(source.weight.sum(dim=2))
            if projection.bias is not None:
                projection.bias.copy_(source.bias)

        positions = getattr(embeddings, "position_embeddings", None)
        if not torch.is_tensor(positions):
            raise TypeError("VideoMAE embeddings do not expose position_embeddings.")
        tokens_per_frame = int(tokens_per_frame)
        if tokens_per_frame <= 0 or int(positions.shape[1]) < tokens_per_frame:
            raise ValueError(
                "VideoMAE positional table is too short for one spatial grid: "
                f"positions={tuple(positions.shape)}, P={tokens_per_frame}."
            )
        return cls(
            projection=projection,
            position_embeddings=positions[:, :tokens_per_frame],
        )

    def forward(
        self,
        pixel_values: torch.Tensor,
        bool_masked_pos: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if pixel_values.ndim != 4 or int(pixel_values.shape[1]) != 3:
            raise ValueError(
                "Single-frame VideoMAE pixels must be [B,3,H,W], "
                f"got {tuple(pixel_values.shape)}."
            )
        embeddings = self.projection(pixel_values).flatten(2).transpose(1, 2)
        if int(embeddings.shape[1]) != int(self.position_embeddings.shape[1]):
            raise ValueError(
                "Single-frame VideoMAE patch/position mismatch: "
                f"{tuple(embeddings.shape)} versus "
                f"{tuple(self.position_embeddings.shape)}."
            )
        embeddings = embeddings + self.position_embeddings.detach().to(
            device=embeddings.device,
            dtype=embeddings.dtype,
        )
        if bool_masked_pos is not None:
            if tuple(bool_masked_pos.shape) != tuple(embeddings.shape[:2]):
                raise ValueError(
                    "VideoMAE mask shape must match [B,P], "
                    f"got {tuple(bool_masked_pos.shape)} versus "
                    f"{tuple(embeddings.shape[:2])}."
                )
            batch_size, _, hidden_size = embeddings.shape
            embeddings = embeddings[~bool_masked_pos].reshape(
                batch_size,
                -1,
                hidden_size,
            )
        return embeddings


class SingleFrameVideoMAEBackbone(nn.Module):
    """Tensor-native two-camera VideoMAE student that consumes only frame ``t``."""

    def __init__(
        self,
        backbone: nn.Module,
        *,
        image_size: int,
        num_views: int,
        pixel_mean: list[float] | tuple[float, ...],
        pixel_std: list[float] | tuple[float, ...],
        fold_patch_stem: bool = True,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.image_size = int(image_size)
        self.num_views = int(num_views)
        config = backbone.config
        patch_size = int(config.patch_size)
        if self.image_size <= 0 or self.num_views <= 0:
            raise ValueError(
                f"image_size/num_views must be positive, got "
                f"{self.image_size}/{self.num_views}."
            )
        if self.image_size % patch_size != 0:
            raise ValueError(
                f"image_size={self.image_size} must be divisible by "
                f"patch_size={patch_size}."
            )
        if int(config.tubelet_size) != 2:
            raise ValueError(
                "Single-frame VideoMAE currently requires tubelet_size=2, "
                f"got {config.tubelet_size}."
            )
        self.patch_size = patch_size
        self.grid_size = self.image_size // patch_size
        self.tokens_per_view = self.grid_size * self.grid_size
        self.hidden_size = int(config.hidden_size)
        self.num_layers = int(config.num_hidden_layers)
        self.student_input_mode = "single_frame_folded_tubelet2_first_position"

        if fold_patch_stem:
            backbone.embeddings = (
                SingleFrameVideoMAEEmbeddings.from_videomae_embeddings(
                    backbone.embeddings,
                    tokens_per_frame=self.tokens_per_view,
                )
            )

        mean = torch.as_tensor(pixel_mean, dtype=torch.float32)
        std = torch.as_tensor(pixel_std, dtype=torch.float32)
        if mean.shape != (3,) or std.shape != (3,) or bool((std <= 0).any().item()):
            raise ValueError(
                f"`pixel_mean/std` must be valid RGB triplets, got {mean}/{std}."
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
    ) -> "SingleFrameVideoMAEBackbone":
        from transformers import AutoImageProcessor, VideoMAEModel

        processor = AutoImageProcessor.from_pretrained(
            model_id,
            local_files_only=bool(local_files_only),
            use_fast=False,
        )
        load_kwargs = {"local_files_only": bool(local_files_only)}
        if attn_implementation is not None:
            load_kwargs["attn_implementation"] = str(attn_implementation)
        backbone = VideoMAEModel.from_pretrained(model_id, **load_kwargs)
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
                f"`images` must be [B,3,H,W] or [3,H,W], "
                f"got {tuple(images.shape)}."
            )
        batch_size, _, height, width = map(int, images.shape)
        expected_width = self.image_size * self.num_views
        if (height, width) != (self.image_size, expected_width):
            raise ValueError(
                "VideoMAE input must preserve the packed camera layout: "
                f"expected ({self.image_size},{expected_width}), "
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
            .reshape(
                batch_size * self.num_views,
                3,
                self.image_size,
                self.image_size,
            )
        )
        views = ((views.float() + 1.0) * 0.5).clamp_(0.0, 1.0)
        views = (views - self.pixel_mean) / self.pixel_std
        reference = next(self.backbone.parameters())
        return views.to(device=reference.device, dtype=reference.dtype)

    def _stitch_dense_tokens(
        self,
        tokens: torch.Tensor,
        *,
        batch_size: int,
    ) -> torch.Tensor:
        expected_batch = batch_size * self.num_views
        if tuple(tokens.shape[:2]) != (expected_batch, self.tokens_per_view):
            raise ValueError(
                "Unexpected single-frame VideoMAE token shape: "
                f"got {tuple(tokens.shape)}, expected "
                f"({expected_batch},{self.tokens_per_view},{self.hidden_size})."
            )
        patches = tokens.reshape(
            batch_size,
            self.num_views,
            self.grid_size,
            self.grid_size,
            self.hidden_size,
        )
        packed = torch.cat(
            [patches[:, view] for view in range(self.num_views)],
            dim=2,
        )
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
        outputs = self.backbone(
            pixel_values=self._prepare_views(images),
            output_hidden_states=bool(output_hidden_states),
            return_dict=True,
        )
        dense = self._stitch_dense_tokens(
            outputs.last_hidden_state,
            batch_size=batch_size,
        )
        if not output_hidden_states:
            return dense
        if outputs.hidden_states is None:
            raise RuntimeError("VideoMAE did not return requested hidden states.")
        layers = tuple(
            self._stitch_dense_tokens(
                self.backbone.layernorm(hidden),
                batch_size=batch_size,
            )
            for hidden in outputs.hidden_states[1:-1]
        ) + (dense,)
        if len(layers) != self.num_layers:
            raise RuntimeError(
                f"Expected {self.num_layers} VideoMAE layers, got {len(layers)}."
            )
        return dense, layers


__all__ = [
    "SingleFrameVideoMAEBackbone",
    "SingleFrameVideoMAEEmbeddings",
]
