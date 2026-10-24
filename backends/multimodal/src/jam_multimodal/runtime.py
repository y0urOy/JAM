import logging
import os
import inspect
from pathlib import Path

import torch
from hydra.utils import instantiate
from omegaconf import DictConfig
from PIL import Image
import numpy as np
from einops import repeat
from omegaconf import OmegaConf

from .trainer import Wan22Trainer
from .utils.logging_config import get_logger, setup_logging
from .utils.video_io import save_mp4
from .utils import misc

logger = get_logger(__name__)


def _normalize_mixed_precision(mixed_precision: str) -> str:
    if not isinstance(mixed_precision, str):
        raise ValueError(f"`mixed_precision` must be str, got {type(mixed_precision)}")
    key = mixed_precision.strip().lower()
    if key not in {"no", "fp16", "bf16"}:
        raise ValueError(
            f"Unsupported mixed_precision: {mixed_precision}. "
            "Expected one of: ['no', 'fp16', 'bf16']."
        )
    return key


def _mixed_precision_to_model_dtype(mixed_precision: str) -> torch.dtype:
    precision = _normalize_mixed_precision(mixed_precision)
    if precision == "no":
        return torch.float32
    if precision == "fp16":
        return torch.float16
    return torch.bfloat16


def create_wan22_model(
    model_id: str,
    tokenizer_model_id: str,
    dit_config,
    tokenizer_max_len: int = 512,
    train_shift: float = 5.0,
    infer_shift: float = 5.0,
    num_train_timesteps: int = 1000,
    redirect_common_files: bool = True,
    model_dtype: torch.dtype = torch.bfloat16,
    device: str = "cuda",
):
    from .models.backbones.wan22 import Wan22Core

    if isinstance(dit_config, DictConfig):
        dit_config = OmegaConf.to_container(dit_config, resolve=True)
    if not isinstance(dit_config, dict):
        raise ValueError(f"`dit_config` must resolve to a dict, got {type(dit_config)}")

    return Wan22Core.from_wan22_pretrained(
        device=device,
        torch_dtype=model_dtype,
        model_id=model_id,
        tokenizer_model_id=tokenizer_model_id,
        tokenizer_max_len=int(tokenizer_max_len),
        redirect_common_files=bool(redirect_common_files),
        dit_config=dit_config,
        train_shift=float(train_shift),
        infer_shift=float(infer_shift),
        num_train_timesteps=int(num_train_timesteps),
    )


def create_jam(
    model_id: str,
    tokenizer_model_id: str,
    video_dit_config,
    tokenizer_max_len: int = 512,
    load_text_encoder: bool = True,
    proprio_dim: int | None = None,
    action_dit_config=None,
    action_dit_pretrained_path: str | None = None,
    skip_dit_load_from_pretrain: bool = False,
    video_scheduler=None,
    action_scheduler=None,
    loss=None,
    mot_checkpoint_mixed_attn: bool = True,
    redirect_common_files: bool = True,
    model_dtype: torch.dtype = torch.bfloat16,
    device: str = "cuda",
):
    from .models.backbones.jam import JAMPolicy

    if isinstance(video_dit_config, DictConfig):
        video_dit_config = OmegaConf.to_container(video_dit_config, resolve=True)
    if not isinstance(video_dit_config, dict):
        raise ValueError(f"`video_dit_config` must resolve to a dict, got {type(video_dit_config)}")

    if isinstance(action_dit_config, DictConfig):
        action_dit_config = OmegaConf.to_container(action_dit_config, resolve=True)
    if action_dit_config is None:
        action_dit_config = {}
    if not isinstance(action_dit_config, dict):
        raise ValueError(f"`action_dit_config` must resolve to a dict, got {type(action_dit_config)}")

    if isinstance(video_scheduler, DictConfig):
        video_scheduler = OmegaConf.to_container(video_scheduler, resolve=True)
    if video_scheduler is None:
        video_scheduler = {}
    if not isinstance(video_scheduler, dict):
        raise ValueError(f"`video_scheduler` must be dict-like, got {type(video_scheduler)}")

    if isinstance(action_scheduler, DictConfig):
        action_scheduler = OmegaConf.to_container(action_scheduler, resolve=True)
    if action_scheduler is None:
        raise ValueError("`action_scheduler` is required for JAMPolicy.")
    if not isinstance(action_scheduler, dict):
        raise ValueError(f"`action_scheduler` must be dict-like, got {type(action_scheduler)}")
    required_action_scheduler_keys = {"train_shift", "infer_shift", "num_train_timesteps"}
    missing_keys = required_action_scheduler_keys - set(action_scheduler.keys())
    if missing_keys:
        raise ValueError(
            f"`action_scheduler` missing required keys: {sorted(missing_keys)}. "
            "Expected keys: train_shift, infer_shift, num_train_timesteps."
        )

    if isinstance(loss, DictConfig):
        loss = OmegaConf.to_container(loss, resolve=True)
    if loss is None:
        loss = {}
    if not isinstance(loss, dict):
        raise ValueError(f"`loss` must be dict-like, got {type(loss)}")

    return JAMPolicy.from_wan22_pretrained(
        device=device,
        torch_dtype=model_dtype,
        model_id=model_id,
        tokenizer_model_id=tokenizer_model_id,
        tokenizer_max_len=int(tokenizer_max_len),
        load_text_encoder=bool(load_text_encoder),
        proprio_dim=(None if proprio_dim is None else int(proprio_dim)),
        redirect_common_files=bool(redirect_common_files),
        video_dit_config=video_dit_config,
        action_dit_config=action_dit_config,
        action_dit_pretrained_path=action_dit_pretrained_path,
        skip_dit_load_from_pretrain=bool(skip_dit_load_from_pretrain),
        mot_checkpoint_mixed_attn=bool(mot_checkpoint_mixed_attn),
        video_train_shift=float(video_scheduler.get("train_shift", 5.0)),
        video_infer_shift=float(video_scheduler.get("infer_shift", 5.0)),
        video_num_train_timesteps=int(video_scheduler.get("num_train_timesteps", 1000)),
        action_train_shift=float(action_scheduler["train_shift"]),
        action_infer_shift=float(action_scheduler["infer_shift"]),
        action_num_train_timesteps=int(action_scheduler["num_train_timesteps"]),
        loss_lambda_video=float(loss.get("lambda_video", 1.0)),
        loss_lambda_action=float(loss.get("lambda_action", 1.0)),
    )


def create_jam_omnigen2(
    omnigen2_model_path: str,
    omnigen2_vae_path: str,
    qwen_path: str | None,
    action_dit_config=None,
    action_dit_pretrained_path: str | None = None,
    proprio_dim: int | None = None,
    transformer_subfolder: str | None = "transformer",
    vae_subfolder: str | None = "vae",
    load_text_encoder: bool = True,
    video_scheduler=None,
    action_scheduler=None,
    loss=None,
    mot_checkpoint_mixed_attn: bool = True,
    mot_gqa_implementation: str = "repeat",
    mot_force_flash_attention: bool = False,
    omnigen2_online_text_cache_compatible: bool = False,
    qwen_context_len: int = 128,
    pack_proprio_after_text: bool = False,
    model_dtype: torch.dtype = torch.bfloat16,
    device: str = "cuda",
):
    from .models.backbones.jam import JAMPolicy

    if isinstance(action_dit_config, DictConfig):
        action_dit_config = OmegaConf.to_container(action_dit_config, resolve=True)
    if action_dit_config is None:
        action_dit_config = {}
    if not isinstance(action_dit_config, dict):
        raise ValueError(f"`action_dit_config` must resolve to a dict, got {type(action_dit_config)}")

    if isinstance(video_scheduler, DictConfig):
        video_scheduler = OmegaConf.to_container(video_scheduler, resolve=True)
    if video_scheduler is None:
        video_scheduler = {}
    if not isinstance(video_scheduler, dict):
        raise ValueError(f"`video_scheduler` must be dict-like, got {type(video_scheduler)}")

    if isinstance(action_scheduler, DictConfig):
        action_scheduler = OmegaConf.to_container(action_scheduler, resolve=True)
    if action_scheduler is None:
        raise ValueError("`action_scheduler` is required for JAMPolicy OmniGen2 stack.")
    if not isinstance(action_scheduler, dict):
        raise ValueError(f"`action_scheduler` must be dict-like, got {type(action_scheduler)}")

    if isinstance(loss, DictConfig):
        loss = OmegaConf.to_container(loss, resolve=True)
    if loss is None:
        loss = {}
    if not isinstance(loss, dict):
        raise ValueError(f"`loss` must be dict-like, got {type(loss)}")

    return JAMPolicy.from_omnigen2_pretrained(
        omnigen2_model_path=omnigen2_model_path,
        omnigen2_vae_path=omnigen2_vae_path,
        qwen_path=qwen_path,
        action_dit_config=action_dit_config,
        action_dit_pretrained_path=action_dit_pretrained_path,
        proprio_dim=(None if proprio_dim is None else int(proprio_dim)),
        transformer_subfolder=transformer_subfolder,
        vae_subfolder=vae_subfolder,
        load_text_encoder=bool(load_text_encoder),
        device=device,
        torch_dtype=model_dtype,
        mot_checkpoint_mixed_attn=bool(mot_checkpoint_mixed_attn),
        video_train_shift=float(video_scheduler.get("train_shift", 5.0)),
        video_infer_shift=float(video_scheduler.get("infer_shift", 5.0)),
        video_num_train_timesteps=int(video_scheduler.get("num_train_timesteps", 1000)),
        action_train_shift=float(action_scheduler["train_shift"]),
        action_infer_shift=float(action_scheduler["infer_shift"]),
        action_num_train_timesteps=int(action_scheduler["num_train_timesteps"]),
        loss_lambda_video=float(loss.get("lambda_video", 1.0)),
        loss_lambda_action=float(loss.get("lambda_action", 1.0)),
        mot_gqa_implementation=str(mot_gqa_implementation),
        mot_force_flash_attention=bool(mot_force_flash_attention),
        omnigen2_online_text_cache_compatible=bool(omnigen2_online_text_cache_compatible),
        qwen_context_len=int(qwen_context_len),
        pack_proprio_after_text=bool(pack_proprio_after_text),
    )


def create_jam_qwen35(
    model_id: str,
    teacher_model_id: str | None = None,
    load_text_encoder: bool = False,
    proprio_dim: int | None = None,
    image_size: int = 224,
    num_views: int = 2,
    context_dim: int = 4096,
    predictor_type: str = "imagewam_transformer",
    predictor_hidden_dim: int = 1024,
    predictor_ffn_dim: int = 4096,
    predictor_num_layers: int = 2,
    predictor_num_heads: int = 16,
    tensor_native_preprocessing: bool = False,
    batch_equal_grid_vision_attention: bool = False,
    vision_gradient_checkpointing: bool = False,
    vision_gradient_checkpointing_blocks: int | None = None,
    use_gradient_checkpointing: bool = False,
    qwen_attn_implementation: str | None = "sdpa",
    action_dit_config=None,
    action_dit_pretrained_path: str | None = None,
    action_dit_pretrained_layer_mapping: str | None = None,
    skip_dit_load_from_pretrain: bool = False,
    action_fusion_type: str = "final_context",
    action_scheduler=None,
    loss=None,
    foresight=None,
    model_dtype: torch.dtype = torch.bfloat16,
    device: str = "cuda",
):
    """Create the isolated Qwen3.5 JAMPolicy policy variant.

    The import stays lazy because Ovis/OmniGen2 and FLUX.2 intentionally use
    older, mutually incompatible Transformers versions. Qwen3.5 is launched
    from its own Transformers-5 environment.
    """
    from .models.backbones.jam_qwen35 import JAMQwen35

    del load_text_encoder  # accepted by shared train/eval configuration

    def as_plain_dict(value, *, name: str, required: bool = False):
        if isinstance(value, DictConfig):
            value = OmegaConf.to_container(value, resolve=True)
        if value is None:
            if required:
                raise ValueError(f"`{name}` is required for JAMQwen35.")
            value = {}
        if not isinstance(value, dict):
            raise ValueError(f"`{name}` must be dict-like, got {type(value)}.")
        return value

    action_dit_config = as_plain_dict(
        action_dit_config, name="action_dit_config", required=True
    )
    action_scheduler = as_plain_dict(
        action_scheduler, name="action_scheduler", required=True
    )
    loss = as_plain_dict(loss, name="loss")
    foresight = as_plain_dict(foresight, name="foresight")

    required_scheduler_keys = {"train_shift", "infer_shift", "num_train_timesteps"}
    missing_scheduler_keys = required_scheduler_keys - set(action_scheduler)
    if missing_scheduler_keys:
        raise ValueError(
            "`action_scheduler` missing required keys: "
            f"{sorted(missing_scheduler_keys)}."
        )

    return JAMQwen35.from_pretrained(
        model_id=str(model_id),
        teacher_model_id=None if teacher_model_id is None else str(teacher_model_id),
        proprio_dim=None if proprio_dim is None else int(proprio_dim),
        image_size=int(image_size),
        num_views=int(num_views),
        context_dim=int(context_dim),
        predictor_type=str(predictor_type),
        predictor_hidden_dim=int(predictor_hidden_dim),
        predictor_ffn_dim=int(predictor_ffn_dim),
        predictor_num_layers=int(predictor_num_layers),
        predictor_num_heads=int(predictor_num_heads),
        tensor_native_preprocessing=bool(tensor_native_preprocessing),
        batch_equal_grid_vision_attention=bool(batch_equal_grid_vision_attention),
        vision_gradient_checkpointing=bool(vision_gradient_checkpointing),
        vision_gradient_checkpointing_blocks=(
            None
            if vision_gradient_checkpointing_blocks is None
            else int(vision_gradient_checkpointing_blocks)
        ),
        use_gradient_checkpointing=bool(use_gradient_checkpointing),
        qwen_attn_implementation=qwen_attn_implementation,
        action_dit_config=action_dit_config,
        action_dit_pretrained_path=action_dit_pretrained_path,
        action_dit_pretrained_layer_mapping=action_dit_pretrained_layer_mapping,
        skip_dit_load_from_pretrain=bool(skip_dit_load_from_pretrain),
        action_fusion_type=str(action_fusion_type),
        action_train_shift=float(action_scheduler["train_shift"]),
        action_infer_shift=float(action_scheduler["infer_shift"]),
        action_num_train_timesteps=int(action_scheduler["num_train_timesteps"]),
        loss_lambda_action=float(loss.get("lambda_action", 1.0)),
        loss_lambda_foresight=float(loss.get("lambda_foresight", 0.3)),
        foresight_cfg=foresight,
        device=device,
        torch_dtype=model_dtype,
    )


def create_jam_dinov2_yak18_strict(
    model_id: str,
    text_embedding_cache_dir: str,
    text_context_len: int = 128,
    text_hidden_dim: int = 4096,
    proprio_dim: int | None = None,
    image_size: int = 224,
    num_views: int = 2,
    local_files_only: bool = True,
    attn_implementation: str | None = "sdpa",
    gradient_checkpointing: bool = False,
    action_dit_config=None,
    action_dit_pretrained_path: str | None = None,
    skip_dit_load_from_pretrain: bool = False,
    predictor_hidden_dim: int = 384,
    predictor_ffn_dim: int = 1536,
    predictor_num_layers: int = 2,
    predictor_num_heads: int = 6,
    action_scheduler=None,
    loss=None,
    foresight=None,
    model_dtype: torch.dtype = torch.bfloat16,
    device: str = "cuda",
):
    """Create DINOv2 + ActionDiTYak18 without an image-editing stack."""

    from .models.backbones.jam_dinov2_yak import JAMDINOv2Yak

    def as_plain_dict(value, *, name: str, required: bool = False):
        if isinstance(value, DictConfig):
            value = OmegaConf.to_container(value, resolve=True)
        if value is None:
            if required:
                raise ValueError(f"`{name}` is required for strict DINO JAMPolicy.")
            value = {}
        if not isinstance(value, dict):
            raise ValueError(f"`{name}` must be dict-like, got {type(value)}.")
        return value

    action_dit_config = as_plain_dict(
        action_dit_config, name="action_dit_config", required=True
    )
    action_scheduler = as_plain_dict(
        action_scheduler, name="action_scheduler", required=True
    )
    loss = as_plain_dict(loss, name="loss")
    foresight = as_plain_dict(foresight, name="foresight")
    required_scheduler_keys = {"train_shift", "infer_shift", "num_train_timesteps"}
    missing = required_scheduler_keys - set(action_scheduler)
    if missing:
        raise ValueError(
            f"`action_scheduler` is missing required keys: {sorted(missing)}."
        )

    return JAMDINOv2Yak.from_pretrained(
        model_id=str(model_id),
        text_embedding_cache_dir=str(text_embedding_cache_dir),
        text_context_len=int(text_context_len),
        text_hidden_dim=int(text_hidden_dim),
        proprio_dim=None if proprio_dim is None else int(proprio_dim),
        image_size=int(image_size),
        num_views=int(num_views),
        local_files_only=bool(local_files_only),
        attn_implementation=attn_implementation,
        gradient_checkpointing=bool(gradient_checkpointing),
        action_dit_config=action_dit_config,
        action_dit_pretrained_path=action_dit_pretrained_path,
        skip_dit_load_from_pretrain=bool(skip_dit_load_from_pretrain),
        predictor_hidden_dim=int(predictor_hidden_dim),
        predictor_ffn_dim=int(predictor_ffn_dim),
        predictor_num_layers=int(predictor_num_layers),
        predictor_num_heads=int(predictor_num_heads),
        action_train_shift=float(action_scheduler["train_shift"]),
        action_infer_shift=float(action_scheduler["infer_shift"]),
        action_num_train_timesteps=int(action_scheduler["num_train_timesteps"]),
        loss_lambda_action=float(loss.get("lambda_action", 1.0)),
        loss_lambda_foresight=float(loss.get("lambda_foresight", 0.3)),
        foresight_cfg=foresight,
        device=device,
        torch_dtype=model_dtype,
    )


def create_jam_dinov2_flux2_25_strict(
    model_id: str,
    qwen_text_cache_dir: str,
    qwen_context_len: int = 512,
    qwen_hidden_dim: int = 7680,
    proprio_dim: int | None = None,
    image_size: int = 224,
    num_views: int = 2,
    local_files_only: bool = True,
    attn_implementation: str | None = "sdpa",
    gradient_checkpointing: bool = False,
    flux2_src_path: str | None = None,
    action_dit_config=None,
    action_dit_pretrained_path: str | None = None,
    skip_dit_load_from_pretrain: bool = False,
    predictor_hidden_dim: int = 384,
    predictor_ffn_dim: int = 1536,
    predictor_num_layers: int = 2,
    predictor_num_heads: int = 6,
    action_scheduler=None,
    loss=None,
    foresight=None,
    model_dtype: torch.dtype = torch.bfloat16,
    device: str = "cuda",
):
    """Create DINOv2 + official ActionDiTFlux2-25 without FLUX video/VAE."""

    from .models.backbones.jam_dinov2_flux2 import JAMDINOv2Flux2
    from .models.backbones.flux2_imports import ensure_flux2_importable

    ensure_flux2_importable(flux2_src_path)

    def as_plain_dict(value, *, name: str, required: bool = False):
        if isinstance(value, DictConfig):
            value = OmegaConf.to_container(value, resolve=True)
        if value is None:
            if required:
                raise ValueError(
                    f"`{name}` is required for strict DINO/FLUX2 JAMPolicy."
                )
            value = {}
        if not isinstance(value, dict):
            raise ValueError(f"`{name}` must be dict-like, got {type(value)}.")
        return value

    action_dit_config = as_plain_dict(
        action_dit_config, name="action_dit_config", required=True
    )
    action_scheduler = as_plain_dict(
        action_scheduler, name="action_scheduler", required=True
    )
    loss = as_plain_dict(loss, name="loss")
    foresight = as_plain_dict(foresight, name="foresight")
    required_scheduler_keys = {"train_shift", "infer_shift", "num_train_timesteps"}
    missing = required_scheduler_keys - set(action_scheduler)
    if missing:
        raise ValueError(
            f"`action_scheduler` is missing required keys: {sorted(missing)}."
        )

    return JAMDINOv2Flux2.from_pretrained(
        model_id=str(model_id),
        qwen_text_cache_dir=str(qwen_text_cache_dir),
        qwen_context_len=int(qwen_context_len),
        qwen_hidden_dim=int(qwen_hidden_dim),
        proprio_dim=None if proprio_dim is None else int(proprio_dim),
        image_size=int(image_size),
        num_views=int(num_views),
        local_files_only=bool(local_files_only),
        attn_implementation=attn_implementation,
        gradient_checkpointing=bool(gradient_checkpointing),
        action_dit_config=action_dit_config,
        action_dit_pretrained_path=action_dit_pretrained_path,
        skip_dit_load_from_pretrain=bool(skip_dit_load_from_pretrain),
        predictor_hidden_dim=int(predictor_hidden_dim),
        predictor_ffn_dim=int(predictor_ffn_dim),
        predictor_num_layers=int(predictor_num_layers),
        predictor_num_heads=int(predictor_num_heads),
        action_train_shift=float(action_scheduler["train_shift"]),
        action_infer_shift=float(action_scheduler["infer_shift"]),
        action_num_train_timesteps=int(action_scheduler["num_train_timesteps"]),
        loss_lambda_action=float(loss.get("lambda_action", 1.0)),
        loss_lambda_foresight=float(loss.get("lambda_foresight", 0.3)),
        foresight_cfg=foresight,
        device=device,
        torch_dtype=model_dtype,
    )


def create_jam_videomae_yak18_strict(
    model_id: str,
    text_embedding_cache_dir: str,
    text_context_len: int = 128,
    text_hidden_dim: int = 4096,
    proprio_dim: int | None = None,
    image_size: int = 224,
    num_views: int = 2,
    local_files_only: bool = True,
    attn_implementation: str | None = "sdpa",
    gradient_checkpointing: bool = False,
    action_dit_config=None,
    action_dit_pretrained_path: str | None = None,
    skip_dit_load_from_pretrain: bool = False,
    predictor_hidden_dim: int = 384,
    predictor_ffn_dim: int = 1536,
    predictor_num_layers: int = 2,
    predictor_num_heads: int = 6,
    action_scheduler=None,
    loss=None,
    foresight=None,
    model_dtype: torch.dtype = torch.bfloat16,
    device: str = "cuda",
):
    """Create VideoMAE-B + ActionDiTYak18 without an image-editing stack."""

    from .models.backbones.jam_videomae_yak import JAMVideoMAEYak

    def as_plain_dict(value, *, name: str, required: bool = False):
        if isinstance(value, DictConfig):
            value = OmegaConf.to_container(value, resolve=True)
        if value is None:
            if required:
                raise ValueError(
                    f"`{name}` is required for strict VideoMAE JAMPolicy."
                )
            value = {}
        if not isinstance(value, dict):
            raise ValueError(f"`{name}` must be dict-like, got {type(value)}.")
        return value

    action_dit_config = as_plain_dict(
        action_dit_config, name="action_dit_config", required=True
    )
    action_scheduler = as_plain_dict(
        action_scheduler, name="action_scheduler", required=True
    )
    loss = as_plain_dict(loss, name="loss")
    foresight = as_plain_dict(foresight, name="foresight")
    required_scheduler_keys = {"train_shift", "infer_shift", "num_train_timesteps"}
    missing = required_scheduler_keys - set(action_scheduler)
    if missing:
        raise ValueError(
            f"`action_scheduler` is missing required keys: {sorted(missing)}."
        )

    return JAMVideoMAEYak.from_pretrained(
        model_id=str(model_id),
        text_embedding_cache_dir=str(text_embedding_cache_dir),
        text_context_len=int(text_context_len),
        text_hidden_dim=int(text_hidden_dim),
        proprio_dim=None if proprio_dim is None else int(proprio_dim),
        image_size=int(image_size),
        num_views=int(num_views),
        local_files_only=bool(local_files_only),
        attn_implementation=attn_implementation,
        gradient_checkpointing=bool(gradient_checkpointing),
        action_dit_config=action_dit_config,
        action_dit_pretrained_path=action_dit_pretrained_path,
        skip_dit_load_from_pretrain=bool(skip_dit_load_from_pretrain),
        teacher_clip_frames=int(foresight.get("teacher_clip_frames", 16)),
        predictor_hidden_dim=int(predictor_hidden_dim),
        predictor_ffn_dim=int(predictor_ffn_dim),
        predictor_num_layers=int(predictor_num_layers),
        predictor_num_heads=int(predictor_num_heads),
        action_train_shift=float(action_scheduler["train_shift"]),
        action_infer_shift=float(action_scheduler["infer_shift"]),
        action_num_train_timesteps=int(action_scheduler["num_train_timesteps"]),
        loss_lambda_action=float(loss.get("lambda_action", 1.0)),
        loss_lambda_foresight=float(loss.get("lambda_foresight", 0.3)),
        foresight_cfg=foresight,
        device=device,
        torch_dtype=model_dtype,
    )


def create_jam_ovis_u1(
    ovis_u1_model_path: str,
    action_dit_config=None,
    action_dit_pretrained_path: str | None = None,
    proprio_dim: int | None = None,
    load_condition_encoder: bool = False,
    video_scheduler=None,
    action_scheduler=None,
    loss=None,
    foresight=None,
    ssl_vision=None,
    mot_checkpoint_mixed_attn: bool = True,
    mot_gqa_implementation: str = "repeat",
    mot_force_flash_attention: bool = False,
    model_dtype: torch.dtype = torch.bfloat16,
    device: str = "cuda",
):
    from .models.backbones.jam import JAMPolicy

    if isinstance(action_dit_config, DictConfig):
        action_dit_config = OmegaConf.to_container(action_dit_config, resolve=True)
    if action_dit_config is None:
        action_dit_config = {}
    if not isinstance(action_dit_config, dict):
        raise ValueError(f"`action_dit_config` must resolve to a dict, got {type(action_dit_config)}")

    if isinstance(video_scheduler, DictConfig):
        video_scheduler = OmegaConf.to_container(video_scheduler, resolve=True)
    if video_scheduler is None:
        video_scheduler = {}
    if not isinstance(video_scheduler, dict):
        raise ValueError(f"`video_scheduler` must be dict-like, got {type(video_scheduler)}")

    if isinstance(action_scheduler, DictConfig):
        action_scheduler = OmegaConf.to_container(action_scheduler, resolve=True)
    if action_scheduler is None:
        raise ValueError("`action_scheduler` is required for JAMPolicy Ovis-U1 stack.")
    if not isinstance(action_scheduler, dict):
        raise ValueError(f"`action_scheduler` must be dict-like, got {type(action_scheduler)}")

    if isinstance(loss, DictConfig):
        loss = OmegaConf.to_container(loss, resolve=True)
    if loss is None:
        loss = {}
    if not isinstance(loss, dict):
        raise ValueError(f"`loss` must be dict-like, got {type(loss)}")

    if isinstance(ssl_vision, DictConfig):
        ssl_vision = OmegaConf.to_container(ssl_vision, resolve=True)
    if ssl_vision is None:
        ssl_vision = {}
    if not isinstance(ssl_vision, dict):
        raise ValueError(f"`ssl_vision` must be dict-like, got {type(ssl_vision)}")

    # Debug-only: env-gated dtype override to bisect bf16 numerical issues.
    _envdt = os.environ.get("JAM_MODEL_DTYPE")
    if _envdt:
        model_dtype = {
            "float32": torch.float32, "fp32": torch.float32,
            "bfloat16": torch.bfloat16, "bf16": torch.bfloat16,
            "float16": torch.float16, "fp16": torch.float16,
        }.get(_envdt.strip().lower(), model_dtype)

    ssl_action_fusion = str(
        ssl_vision.get("action_fusion_type", "layerwise_mot")
    ).strip().lower()
    effective_action_dit_pretrained_path = action_dit_pretrained_path
    if (
        bool(ssl_vision.get("enabled", False))
        and ssl_action_fusion
        in {
            "native_dino12_mot",
            "native_dino24_mot",
            "native_dinov2_large24_mot",
            "native_videomae12_mot",
        }
    ):
        # The temporary Yak-shaped action expert is replaced immediately after
        # loading DINO.  Loading the 1B ActionDiTYak checkpoint here would add
        # startup I/O without initializing a single native ActionDiTDINO tensor.
        effective_action_dit_pretrained_path = None
        logger.info(
            "Skipping legacy ActionDiTYak checkpoint load for native DINO MoT; "
            "ActionDiTDINO is initialized from the DINO student blocks."
        )

    model = JAMPolicy.from_ovis_u1_pretrained(
        ovis_u1_model_path=ovis_u1_model_path,
        action_dit_config=action_dit_config,
        action_dit_pretrained_path=effective_action_dit_pretrained_path,
        proprio_dim=(None if proprio_dim is None else int(proprio_dim)),
        load_condition_encoder=bool(load_condition_encoder),
        device=device,
        torch_dtype=model_dtype,
        mot_checkpoint_mixed_attn=bool(mot_checkpoint_mixed_attn),
        video_train_shift=float(video_scheduler.get("train_shift", 5.0)),
        video_infer_shift=float(video_scheduler.get("infer_shift", 5.0)),
        video_num_train_timesteps=int(video_scheduler.get("num_train_timesteps", 1000)),
        action_train_shift=float(action_scheduler["train_shift"]),
        action_infer_shift=float(action_scheduler["infer_shift"]),
        action_num_train_timesteps=int(action_scheduler["num_train_timesteps"]),
        loss_lambda_video=float(loss.get("lambda_video", 1.0)),
        loss_lambda_action=float(loss.get("lambda_action", 1.0)),
        mot_gqa_implementation=str(mot_gqa_implementation),
        mot_force_flash_attention=bool(mot_force_flash_attention),
    )

    # With IMAGEWAM_MODEL_DTYPE=float32 (fp32 trainable DiT), the FROZEN
    # conditioning stack must stay bf16: the AIMv2 visual tokenizer calls
    # flash_attn directly (fp16/bf16 only). Cast just visual_tokenizer/vte/llm
    # back to bf16 — they are no_grad-only, and `_encode_ovis_u1_condition_online`
    # already casts pixel inputs to the tokenizer's dtype and the resulting
    # text_hidden_states to the model dtype. The visual_generator (= trainable
    # video expert) is untouched and stays in model_dtype.
    if (
        os.environ.get("JAM_COND_STACK_BF16") == "1"
        and getattr(model, "ovis_condition_encoder", None) is not None
    ):
        enc = model.ovis_condition_encoder
        enc.visual_tokenizer.to(torch.bfloat16)
        enc.vte.to(torch.bfloat16)
        enc.get_llm().to(torch.bfloat16)

    if isinstance(foresight, DictConfig):
        foresight = OmegaConf.to_container(foresight, resolve=True)
    if foresight is None:
        foresight = {}
    if not isinstance(foresight, dict):
        raise ValueError(f"`foresight` must be dict-like, got {type(foresight)}")

    if bool(ssl_vision.get("enabled", False)):
        from .models.backbones.external_ssl import attach_ovis_external_ssl

        attach_ovis_external_ssl(model, ssl_vision, foresight)
    elif foresight:
        if not isinstance(foresight, dict):
            raise ValueError(f"`foresight` must be dict-like, got {type(foresight)}")
        from .models.backbones.foresight_yak import attach_ovis_foresight

        attach_ovis_foresight(model, foresight)

    return model


def create_jam_flux2_klein(
    flux2_model_path: str,
    ae_model_path: str,
    flux2_src_path: str | None = None,
    variant: str = "klein-base-4b",
    qwen3_model_spec: str | None = None,
    qwen_context_len: int = 512,
    trim_cached_text_padding: bool = False,
    action_dit_config=None,
    action_dit_pretrained_path: str | None = None,
    proprio_dim: int | None = None,
    load_text_encoder: bool = False,
    video_scheduler=None,
    action_scheduler=None,
    loss=None,
    foresight=None,
    mot_checkpoint_mixed_attn: bool = True,
    mot_gqa_implementation: str = "repeat",
    mot_force_flash_attention: bool = False,
    pack_proprio_after_text: bool = True,
    flux2_lora_config=None,
    model_dtype: torch.dtype = torch.bfloat16,
    device: str = "cuda",
):
    from .models.backbones.jam import JAMPolicy

    if isinstance(action_dit_config, DictConfig):
        action_dit_config = OmegaConf.to_container(action_dit_config, resolve=True)
    if action_dit_config is None:
        action_dit_config = {}
    if not isinstance(action_dit_config, dict):
        raise ValueError(f"`action_dit_config` must resolve to a dict, got {type(action_dit_config)}")

    if isinstance(video_scheduler, DictConfig):
        video_scheduler = OmegaConf.to_container(video_scheduler, resolve=True)
    if video_scheduler is None:
        video_scheduler = {}
    if not isinstance(video_scheduler, dict):
        raise ValueError(f"`video_scheduler` must be dict-like, got {type(video_scheduler)}")

    if isinstance(action_scheduler, DictConfig):
        action_scheduler = OmegaConf.to_container(action_scheduler, resolve=True)
    if action_scheduler is None:
        raise ValueError("`action_scheduler` is required for JAMPolicy FLUX.2 stack.")
    if not isinstance(action_scheduler, dict):
        raise ValueError(f"`action_scheduler` must be dict-like, got {type(action_scheduler)}")

    if isinstance(loss, DictConfig):
        loss = OmegaConf.to_container(loss, resolve=True)
    if loss is None:
        loss = {}
    if not isinstance(loss, dict):
        raise ValueError(f"`loss` must be dict-like, got {type(loss)}")
    if isinstance(foresight, DictConfig):
        foresight = OmegaConf.to_container(foresight, resolve=True)
    if foresight is not None and not isinstance(foresight, dict):
        raise ValueError(f"`foresight` must be dict-like, got {type(foresight)}")
    if isinstance(flux2_lora_config, DictConfig):
        flux2_lora_config = OmegaConf.to_container(flux2_lora_config, resolve=True)
    if flux2_lora_config is None:
        flux2_lora_config = {}
    if not isinstance(flux2_lora_config, dict):
        raise ValueError(f"`flux2_lora_config` must be dict-like, got {type(flux2_lora_config)}")
    model = JAMPolicy.from_flux2_klein_pretrained(
        flux2_model_path=flux2_model_path,
        ae_model_path=ae_model_path,
        flux2_src_path=flux2_src_path,
        variant=str(variant),
        qwen3_model_spec=qwen3_model_spec,
        qwen_context_len=int(qwen_context_len),
        trim_cached_text_padding=bool(trim_cached_text_padding),
        action_dit_config=action_dit_config,
        action_dit_pretrained_path=action_dit_pretrained_path,
        proprio_dim=(None if proprio_dim is None else int(proprio_dim)),
        load_text_encoder=bool(load_text_encoder),
        device=device,
        torch_dtype=model_dtype,
        mot_checkpoint_mixed_attn=bool(mot_checkpoint_mixed_attn),
        video_train_shift=float(video_scheduler.get("train_shift", 5.0)),
        video_infer_shift=float(video_scheduler.get("infer_shift", 5.0)),
        video_num_train_timesteps=int(video_scheduler.get("num_train_timesteps", 1000)),
        action_train_shift=float(action_scheduler["train_shift"]),
        action_infer_shift=float(action_scheduler["infer_shift"]),
        action_num_train_timesteps=int(action_scheduler["num_train_timesteps"]),
        loss_lambda_video=float(loss.get("lambda_video", 1.0)),
        loss_lambda_action=float(loss.get("lambda_action", 1.0)),
        mot_gqa_implementation=str(mot_gqa_implementation),
        mot_force_flash_attention=bool(mot_force_flash_attention),
        pack_proprio_after_text=bool(pack_proprio_after_text),
        flux2_lora_config=flux2_lora_config,
    )
    if foresight:
        from .models.backbones.foresight_flux2 import attach_flux2_foresight

        attach_flux2_foresight(model, foresight)
    return model


def create_jam_dim(
    dim_model_path: str,
    sana_config_path: str,
    qwen_path: str,
    action_dit_config=None,
    action_dit_pretrained_path: str | None = None,
    proprio_dim: int | None = None,
    max_condition_length: int = 8192,
    with_latents_condition: bool = True,
    load_mllm: bool = True,
    qwen_attn_implementation: str = "flash_attention_2",
    video_scheduler=None,
    action_scheduler=None,
    loss=None,
    mot_checkpoint_mixed_attn: bool = True,
    mot_gqa_implementation: str = "repeat",
    mot_force_flash_attention: bool = False,
    pack_proprio_after_text: bool = True,
    model_dtype: torch.dtype = torch.bfloat16,
    device: str = "cuda",
):
    from .models.backbones.jam import JAMPolicy

    if isinstance(action_dit_config, DictConfig):
        action_dit_config = OmegaConf.to_container(action_dit_config, resolve=True)
    if action_dit_config is None:
        action_dit_config = {}
    if not isinstance(action_dit_config, dict):
        raise ValueError(f"`action_dit_config` must resolve to a dict, got {type(action_dit_config)}")

    if isinstance(video_scheduler, DictConfig):
        video_scheduler = OmegaConf.to_container(video_scheduler, resolve=True)
    if video_scheduler is None:
        video_scheduler = {}
    if not isinstance(video_scheduler, dict):
        raise ValueError(f"`video_scheduler` must be dict-like, got {type(video_scheduler)}")

    if isinstance(action_scheduler, DictConfig):
        action_scheduler = OmegaConf.to_container(action_scheduler, resolve=True)
    if action_scheduler is None:
        raise ValueError("`action_scheduler` is required for JAMPolicy DIM stack.")
    if not isinstance(action_scheduler, dict):
        raise ValueError(f"`action_scheduler` must be dict-like, got {type(action_scheduler)}")

    if isinstance(loss, DictConfig):
        loss = OmegaConf.to_container(loss, resolve=True)
    if loss is None:
        loss = {}
    if not isinstance(loss, dict):
        raise ValueError(f"`loss` must be dict-like, got {type(loss)}")

    return JAMPolicy.from_dim_pretrained(
        dim_model_path=dim_model_path,
        sana_config_path=sana_config_path,
        qwen_path=qwen_path,
        action_dit_config=action_dit_config,
        action_dit_pretrained_path=action_dit_pretrained_path,
        proprio_dim=(None if proprio_dim is None else int(proprio_dim)),
        max_condition_length=int(max_condition_length),
        with_latents_condition=bool(with_latents_condition),
        load_mllm=bool(load_mllm),
        qwen_attn_implementation=str(qwen_attn_implementation),
        device=device,
        torch_dtype=model_dtype,
        mot_checkpoint_mixed_attn=bool(mot_checkpoint_mixed_attn),
        video_train_shift=float(video_scheduler.get("train_shift", 5.0)),
        video_infer_shift=float(video_scheduler.get("infer_shift", 5.0)),
        video_num_train_timesteps=int(video_scheduler.get("num_train_timesteps", 1000)),
        action_train_shift=float(action_scheduler["train_shift"]),
        action_infer_shift=float(action_scheduler["infer_shift"]),
        action_num_train_timesteps=int(action_scheduler["num_train_timesteps"]),
        loss_lambda_video=float(loss.get("lambda_video", 1.0)),
        loss_lambda_action=float(loss.get("lambda_action", 1.0)),
        mot_gqa_implementation=str(mot_gqa_implementation),
        mot_force_flash_attention=bool(mot_force_flash_attention),
        pack_proprio_after_text=bool(pack_proprio_after_text),
    )


def create_jam_joint(
    model_id: str,
    tokenizer_model_id: str,
    video_dit_config,
    tokenizer_max_len: int = 512,
    load_text_encoder: bool = True,
    proprio_dim: int | None = None,
    action_dit_config=None,
    action_dit_pretrained_path: str | None = None,
    skip_dit_load_from_pretrain: bool = False,
    video_scheduler=None,
    action_scheduler=None,
    loss=None,
    mot_checkpoint_mixed_attn: bool = True,
    redirect_common_files: bool = True,
    model_dtype: torch.dtype = torch.bfloat16,
    device: str = "cuda",
):
    raise NotImplementedError("This historical joint/IDM prototype is not included in the paper release. Use a recipe from tools/jam.py list.")


def create_jam_idm(
    model_id: str,
    tokenizer_model_id: str,
    video_dit_config,
    tokenizer_max_len: int = 512,
    load_text_encoder: bool = True,
    proprio_dim: int | None = None,
    action_dit_config=None,
    action_dit_pretrained_path: str | None = None,
    skip_dit_load_from_pretrain: bool = False,
    video_scheduler=None,
    action_scheduler=None,
    loss=None,
    mot_checkpoint_mixed_attn: bool = True,
    redirect_common_files: bool = True,
    model_dtype: torch.dtype = torch.bfloat16,
    device: str = "cuda",
):
    raise NotImplementedError("This historical joint/IDM prototype is not included in the paper release. Use a recipe from tools/jam.py list.")


def create_jam_noise_idm(
    model_id: str,
    tokenizer_model_id: str,
    video_dit_config,
    tokenizer_max_len: int = 512,
    load_text_encoder: bool = True,
    proprio_dim: int | None = None,
    action_dit_config=None,
    action_dit_pretrained_path: str | None = None,
    skip_dit_load_from_pretrain: bool = False,
    video_scheduler=None,
    action_scheduler=None,
    loss=None,
    mot_checkpoint_mixed_attn: bool = True,
    noise_idm_train_schedule: str = "joint",
    noise_idm_alternate_loss_scale: float = 2.0,
    noise_idm_ddp_zero_anchor: bool = False,
    redirect_common_files: bool = True,
    model_dtype: torch.dtype = torch.bfloat16,
    device: str = "cuda",
):
    raise NotImplementedError("This historical joint/IDM prototype is not included in the paper release. Use a recipe from tools/jam.py list.")


def create_jam_noise_idm_omnigen2(
    omnigen2_model_path: str,
    omnigen2_vae_path: str,
    qwen_path: str | None,
    action_dit_config=None,
    action_dit_pretrained_path: str | None = None,
    proprio_dim: int | None = None,
    transformer_subfolder: str | None = "transformer",
    vae_subfolder: str | None = "vae",
    load_text_encoder: bool = True,
    video_scheduler=None,
    action_scheduler=None,
    loss=None,
    mot_checkpoint_mixed_attn: bool = True,
    mot_gqa_implementation: str = "repeat",
    mot_force_flash_attention: bool = False,
    noise_idm_train_schedule: str = "joint",
    noise_idm_alternate_loss_scale: float = 2.0,
    noise_idm_ddp_zero_anchor: bool = False,
    omnigen2_online_text_cache_compatible: bool = False,
    qwen_context_len: int = 128,
    pack_proprio_after_text: bool = False,
    model_dtype: torch.dtype = torch.bfloat16,
    device: str = "cuda",
):
    raise NotImplementedError("This historical joint/IDM prototype is not included in the paper release. Use a recipe from tools/jam.py list.")


def create_jam_cache_idm(
    model_id: str,
    tokenizer_model_id: str,
    video_dit_config,
    tokenizer_max_len: int = 512,
    load_text_encoder: bool = True,
    proprio_dim: int | None = None,
    action_dit_config=None,
    action_dit_pretrained_path: str | None = None,
    skip_dit_load_from_pretrain: bool = False,
    video_scheduler=None,
    action_scheduler=None,
    loss=None,
    mot_checkpoint_mixed_attn: bool = True,
    cache_idm_train_schedule: str = "sequential_micro",
    cache_idm_alternate_loss_scale: float = 1.0,
    cache_idm_ddp_zero_anchor: bool = False,
    redirect_common_files: bool = True,
    model_dtype: torch.dtype = torch.bfloat16,
    device: str = "cuda",
):
    raise NotImplementedError("This historical joint/IDM prototype is not included in the paper release. Use a recipe from tools/jam.py list.")


def create_jam_cache_idm_omnigen2(
    omnigen2_model_path: str,
    omnigen2_vae_path: str,
    qwen_path: str | None,
    action_dit_config=None,
    action_dit_pretrained_path: str | None = None,
    proprio_dim: int | None = None,
    transformer_subfolder: str | None = "transformer",
    vae_subfolder: str | None = "vae",
    load_text_encoder: bool = True,
    video_scheduler=None,
    action_scheduler=None,
    loss=None,
    mot_checkpoint_mixed_attn: bool = True,
    mot_gqa_implementation: str = "repeat",
    mot_force_flash_attention: bool = False,
    cache_idm_train_schedule: str = "sequential_micro",
    cache_idm_alternate_loss_scale: float = 1.0,
    cache_idm_ddp_zero_anchor: bool = False,
    omnigen2_online_text_cache_compatible: bool = False,
    qwen_context_len: int = 128,
    pack_proprio_after_text: bool = False,
    model_dtype: torch.dtype = torch.bfloat16,
    device: str = "cuda",
):
    raise NotImplementedError("This historical joint/IDM prototype is not included in the paper release. Use a recipe from tools/jam.py list.")


def build_datasets(data_cfg: DictConfig, *, build_val: bool = True):
    train_ds = instantiate(data_cfg.train)
    if not build_val:
        val_ds = None
        logger.info("Skipping val dataset construction because eval_every=0.")
    elif data_cfg.get("val") is None:
        val_ds = train_ds
    else:
        train_stats_path = data_cfg.train.get("pretrained_norm_stats")
        default_stats_path = os.path.join(misc.get_work_dir(), "dataset_stats.json")
        val_stats_path = data_cfg.val.get("pretrained_norm_stats")
        pretrained_norm_stats = val_stats_path or train_stats_path or default_stats_path
        logger.info("Building val dataset with pretrained_norm_stats: %s", pretrained_norm_stats)
        val_ds = instantiate(data_cfg.val, pretrained_norm_stats=pretrained_norm_stats)
    return train_ds, val_ds


def _resolve_train_device() -> str:
    if not torch.cuda.is_available():
        return "cpu"
    device_count = torch.cuda.device_count()
    if device_count <= 1:
        return "cuda:0"
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if local_rank < 0 or local_rank >= device_count:
        return "cuda:0"
    return f"cuda:{local_rank}"


def run_training(cfg: DictConfig):
    setup_logging(
        log_level=logging.INFO,
        is_main_process=torch.distributed.get_rank() == 0 if torch.distributed.is_initialized() else True,
    )
    misc.register_work_dir(cfg.output_dir)
    config_payload = OmegaConf.to_container(cfg, resolve=True)
    with open(Path(cfg.output_dir) / "config.yaml", "w") as f:
        OmegaConf.save(config_payload, f)

    model_device = _resolve_train_device()
    mixed_precision = _normalize_mixed_precision(cfg.mixed_precision)
    model_dtype = _mixed_precision_to_model_dtype(mixed_precision)
    model = instantiate(cfg.model, model_dtype=model_dtype, device=model_device)
    train_ds, val_ds = build_datasets(
        cfg.data,
        build_val=int(cfg.eval_every) > 0,
    )

    trainer = Wan22Trainer(
        cfg=cfg,
        model=model,
        train_dataset=train_ds,
        val_dataset=val_ds,
    )
    trainer.train()

def run_inference(cfg: DictConfig):
    setup_logging(log_level=logging.INFO)
    inference_cfg = cfg.inference
    mixed_precision = _normalize_mixed_precision(cfg.mixed_precision)
    model_dtype = _mixed_precision_to_model_dtype(mixed_precision)

    model = instantiate(cfg.model, model_dtype=model_dtype, device=str(inference_cfg.device))
    checkpoint_path = inference_cfg.get("checkpoint_path")
    if checkpoint_path:
        ckpt = Path(checkpoint_path)
        if ckpt.exists():
            logger.info("Loading finetuned checkpoint: %s", checkpoint_path)
            model.load_checkpoint(checkpoint_path)
        else:
            logger.warning("Checkpoint not found, skipping load: %s", checkpoint_path)
    model.eval()
    
    def center_crop_resize(img: Image, width: int, height: int) -> Image.Image:
        src_w, src_h = img.size
        scale = max(width / src_w, height / src_h)
        resized = img.resize((round(src_w * scale), round(src_h * scale)), resample=Image.BILINEAR)
        rw, rh = resized.size
        left = max((rw - width) // 2, 0)
        top = max((rh - height) // 2, 0)
        return resized.crop((left, top, left + width, top + height))

    input_image = Image.open(str(inference_cfg.input_image_path)).convert("RGB")
    input_image = center_crop_resize(input_image, width=inference_cfg.width, height=inference_cfg.height)
    arr = np.array(input_image, dtype=np.float32)
    x = torch.from_numpy(arr)
    x = x.to(device=model.device, dtype=model.torch_dtype)
    x = x * (2.0 / 255.0) - 1.0
    x = repeat(x, "H W C -> B C H W", B=1)
    output_mp4 = str(inference_cfg.output_mp4)

    infer_kwargs = {
        "prompt": str(inference_cfg.prompt),
        "negative_prompt": str(inference_cfg.negative_prompt),
        "text_cfg_scale": float(inference_cfg.text_cfg_scale),
        "action_cfg_scale": float(inference_cfg.action_cfg_scale),
        "input_image": x,
        "num_frames": int(inference_cfg.num_frames),
        "num_inference_steps": int(inference_cfg.num_inference_steps),
        "sigma_shift": None if inference_cfg.get("sigma_shift") is None else float(inference_cfg.sigma_shift),
        "seed": int(inference_cfg.seed),
        "rand_device": str(inference_cfg.rand_device),
        "tiled": bool(inference_cfg.tiled),
    }

    infer_out = model.infer(**infer_kwargs)
    video = infer_out["video"]
    save_mp4(video, output_mp4, fps=15)
    logger.info("Saved inference video to %s", output_mp4)
    return output_mp4


# Compatibility aliases for historical configs and imports.
create_imagewam = create_jam
create_imagewam_omnigen2 = create_jam_omnigen2
create_imagewam_qwen35 = create_jam_qwen35
create_imagewam_dinov2_yak18_strict = create_jam_dinov2_yak18_strict
create_imagewam_dinov2_flux2_25_strict = create_jam_dinov2_flux2_25_strict
create_imagewam_videomae_yak18_strict = create_jam_videomae_yak18_strict
create_imagewam_ovis_u1 = create_jam_ovis_u1
create_imagewam_flux2_klein = create_jam_flux2_klein
create_imagewam_dim = create_jam_dim
create_imagewam_joint = create_jam_joint
create_imagewam_idm = create_jam_idm
create_imagewam_noise_idm = create_jam_noise_idm
create_imagewam_noise_idm_omnigen2 = create_jam_noise_idm_omnigen2
create_imagewam_cache_idm = create_jam_cache_idm
create_imagewam_cache_idm_omnigen2 = create_jam_cache_idm_omnigen2
