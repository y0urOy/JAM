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
from .utils.pytorch_utils import set_global_seed
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
    from .models.wan22.wan22 import Wan22Core

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


def create_jam_wan(
    model_id: str,
    tokenizer_model_id: str,
    video_dit_config,
    tokenizer_max_len: int = 512,
    load_text_encoder: bool = True,
    proprio_dim: int | None = None,
    action_dit_config=None,
    action_dit_pretrained_path: str | None = None,
    skip_dit_load_from_pretrain: bool = False,
    skip_video_dit_load_from_pretrain: bool | None = None,
    video_dit_random_seed: int | None = None,
    video_scheduler=None,
    action_scheduler=None,
    loss=None,
    foresight=None,
    mot_checkpoint_mixed_attn: bool = True,
    redirect_common_files: bool = True,
    model_dtype: torch.dtype = torch.bfloat16,
    device: str = "cuda",
):
    from .models.wan22.jam_wan import JAMWan

    # The model config keeps these keys (null) for config compatibility; a
    # separately random video expert is not part of this release.
    if skip_video_dit_load_from_pretrain is not None or video_dit_random_seed is not None:
        raise ValueError(
            "`skip_video_dit_load_from_pretrain` and `video_dit_random_seed` must be null; "
            "use `skip_dit_load_from_pretrain` to skip both pretrained expert loads."
        )

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
        raise ValueError("`action_scheduler` is required for JAMWan.")
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

    if isinstance(foresight, DictConfig):
        foresight = OmegaConf.to_container(foresight, resolve=True)
    if foresight is None:
        foresight = {}
    if not isinstance(foresight, dict):
        raise ValueError(f"`foresight` must be dict-like, got {type(foresight)}")

    return JAMWan.from_wan22_pretrained(
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
        foresight_cfg=foresight,
    )


def build_datasets(data_cfg: DictConfig):
    train_ds = instantiate(data_cfg.train)
    if data_cfg.get("val") is None:
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
    # Seed before model instantiation so randomly initialized modules (proprio encoder,
    # foresight projector) are identical across runs given the same `cfg.seed`. The
    # trainer re-seeds with the same value afterwards, so the training stream
    # (data order, noise sampling) is unchanged.
    set_global_seed(int(cfg.seed))
    model = instantiate(cfg.model, model_dtype=model_dtype, device=model_device)
    train_ds, val_ds = build_datasets(cfg.data)

    # Precomputed foresight-target cache (bf16+zstd, bit-exact): attach the cached
    # VAE-direct targets (and optionally the current-frame VAE latents) to every
    # training sample instead of computing them from the decoded future frames.
    fs_cfg = OmegaConf.select(cfg, "model.foresight")
    cache_roots = (
        fs_cfg.get("target_cache_roots")
        if fs_cfg is not None and fs_cfg.get("enabled", False)
        else None
    )
    input_latent_roots = (
        fs_cfg.get("input_latent_cache_roots")
        if fs_cfg is not None and fs_cfg.get("enabled", False)
        else None
    )
    val_cache_roots = (
        fs_cfg.get("val_target_cache_roots")
        if fs_cfg is not None and fs_cfg.get("enabled", False)
        else None
    )
    val_input_latent_roots = (
        fs_cfg.get("val_input_latent_cache_roots")
        if fs_cfg is not None and fs_cfg.get("enabled", False)
        else None
    )
    if cache_roots:
        from .datasets.foresight_cache import ForesightCacheDataset

        # Same precedence as JAMWan.__init__: encoder_layers > layers > encoder_layer > layer.
        enc_layers = (
            fs_cfg.get("encoder_layers") or fs_cfg.get("layers")
            or fs_cfg.get("encoder_layer") or fs_cfg.get("layer", 15)
        )
        if not isinstance(enc_layers, (list, tuple)) and not OmegaConf.is_list(enc_layers):
            enc_layers = [enc_layers]
        enc_layers = [int(x) for x in enc_layers]
        target_type = str(fs_cfg.get("target_type", "vae_direct"))
        offsets = fs_cfg.get("offsets")
        offsets = None if offsets is None else [int(value) for value in offsets]
        train_ds = ForesightCacheDataset(
            train_ds,
            [str(r) for r in cache_roots],
            layers=enc_layers,
            target_type=target_type,
            offsets=offsets,
            input_latent_roots=(
                [str(r) for r in input_latent_roots]
                if input_latent_roots else None
            ),
        )
        logger.info(
            "Foresight train caches attached: target_roots=%s input_latent_roots=%s "
            "layers=%s target_type=%s offsets=%s",
            list(cache_roots),
            list(input_latent_roots) if input_latent_roots else None,
            enc_layers,
            target_type,
            offsets,
        )

        train_uses_cached_input = bool(
            getattr(getattr(train_ds, "base", None), "use_input_latent_cache", False)
        )
        if bool(input_latent_roots) != train_uses_cached_input:
            raise RuntimeError(
                "Training input-latent cache and dataset decode mode must be enabled together: "
                f"input_latent_roots={bool(input_latent_roots)} "
                f"data.train.use_input_latent_cache={train_uses_cached_input}."
            )

        if val_cache_roots or val_input_latent_roots:
            if not val_cache_roots or not val_input_latent_roots:
                raise RuntimeError(
                    "Validation must provide both val_target_cache_roots and "
                    "val_input_latent_cache_roots when RGB decoding is disabled."
                )
            val_ds = ForesightCacheDataset(
                val_ds,
                [str(r) for r in val_cache_roots],
                layers=enc_layers,
                target_type=target_type,
                offsets=offsets,
                input_latent_roots=[str(r) for r in val_input_latent_roots],
            )
            val_uses_cached_input = bool(
                getattr(getattr(val_ds, "base", None), "use_input_latent_cache", False)
            )
            if not val_uses_cached_input:
                raise RuntimeError(
                    "Validation input-latent caches require "
                    "data.val.use_input_latent_cache=true."
                )
            logger.info(
                "Foresight val caches attached: target_roots=%s input_latent_roots=%s",
                list(val_cache_roots),
                list(val_input_latent_roots),
            )
    elif input_latent_roots or val_cache_roots or val_input_latent_roots:
        raise RuntimeError(
            "Input/validation latent caches require model.foresight.target_cache_roots."
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


# Compatibility alias for historical configs and imports.
create_fastwam = create_jam_wan
