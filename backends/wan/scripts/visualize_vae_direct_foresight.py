#!/usr/bin/env python3
"""Diagnose what a trained Fast-WAM VAE-direct foresight head predicts.

The VAE-direct objective aligns a 3072-D policy prediction with a deterministic
isometric lift of a 192-D Wan-VAE latent patch.  The training loss is cosine
similarity, so it does *not* identify the prediction magnitude.  This script
therefore reports two faithful diagnostics and one explicitly approximate one:

1. patch-wise cosine similarity to the recorded future target;
2. projected-to-original feature-norm ratio for the deterministic lifted-VAE subspace; and
3. a decoded visualization after projecting into that subspace and applying a
   single, future-blind scale estimated from the current observation latent.

The decoded frames are diagnostics, not samples emitted by the deployed policy.
No recorded future target is used to calibrate their scale.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import sys
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from hydra.utils import instantiate
from omegaconf import OmegaConf
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from jam_wan.models.wan22.foresight import (  # noqa: E402
    lift_foresight_vae_patches,
    patch_foresight_vae_latents,
)
from jam_wan.utils import misc  # noqa: E402


DEFAULT_DATASET_ROOT = ROOT / "data/libero_mujoco3.3.2"
SUITE_DIR_NAMES = (
    "libero_spatial_no_noops_lerobot",
    "libero_object_no_noops_lerobot",
    "libero_goal_no_noops_lerobot",
    "libero_10_no_noops_lerobot",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-config", type=Path, required=True, help="config.yaml of a trained VAE-direct run")
    parser.add_argument("--checkpoint", type=Path, required=True, help="weights file of that run")
    parser.add_argument("--stats", type=Path, required=True, help="dataset_stats.json of that run")
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument(
        "--text-cache-dir",
        type=Path,
        default=ROOT / "data/text_embeds_cache/libero",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "evaluate_results/foresight_decode/vae_direct_off12_step020000",
    )
    parser.add_argument(
        "--indices",
        type=int,
        nargs="*",
        default=None,
        help="Explicit concatenated-dataset indices. Default: one valid midpoint per suite.",
    )
    parser.add_argument("--samples-per-suite", type=int, default=1)
    parser.add_argument("--max-valid-search", type=int, default=96)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument(
        "--geometry-self-test",
        action="store_true",
        help="Run patch/lift inverse tests on CPU and exit without loading data or a model.",
    )
    return parser.parse_args()


def _dtype(name: str) -> torch.dtype:
    return {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }[name]


def inverse_isometric_lift(prediction: torch.Tensor, raw_dim: int) -> torch.Tensor:
    """Orthogonally project a lifted-space vector back to its raw patch vector."""
    output_dim = int(prediction.shape[-1])
    raw_dim = int(raw_dim)
    if output_dim % raw_dim:
        raise ValueError(f"output_dim={output_dim} is not divisible by raw_dim={raw_dim}")
    repeats = output_dim // raw_dim
    return prediction.reshape(*prediction.shape[:-1], repeats, raw_dim).mean(dim=-2) * math.sqrt(repeats)


def unpatch_vae_latents(
    patches: torch.Tensor,
    *,
    channels: int,
    frames: int,
    height: int,
    width: int,
    patch_size: Sequence[int],
) -> torch.Tensor:
    """Invert ``patch_foresight_vae_latents`` exactly."""
    pt, ph, pw = (int(value) for value in patch_size)
    if frames % pt or height % ph or width % pw:
        raise ValueError("Requested latent geometry is not patch divisible.")
    fp, hp, wp = frames // pt, height // ph, width // pw
    expected_tokens = fp * hp * wp
    expected_dim = channels * pt * ph * pw
    if tuple(patches.shape[-2:]) != (expected_tokens, expected_dim):
        raise ValueError(
            f"Patch shape {tuple(patches.shape)} is incompatible with "
            f"tokens={expected_tokens}, raw_dim={expected_dim}."
        )
    batch = int(patches.shape[0])
    return (
        patches.reshape(batch, fp, hp, wp, channels, pt, ph, pw)
        .permute(0, 4, 1, 5, 2, 6, 3, 7)
        .reshape(batch, channels, frames, height, width)
        .contiguous()
    )


def geometry_self_test() -> None:
    generator = torch.Generator().manual_seed(17)
    latents = torch.randn((2, 48, 3, 8, 10), generator=generator)
    patch_size = (1, 2, 2)
    raw, tokens_per_frame = patch_foresight_vae_latents(latents, patch_size=patch_size)
    assert tokens_per_frame == 20
    restored = unpatch_vae_latents(
        raw,
        channels=48,
        frames=3,
        height=8,
        width=10,
        patch_size=patch_size,
    )
    torch.testing.assert_close(restored, latents, rtol=0, atol=0)
    lifted = lift_foresight_vae_patches(raw, output_dim=3072)
    inverse = inverse_isometric_lift(lifted, raw_dim=raw.shape[-1])
    torch.testing.assert_close(inverse, raw, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(
        lifted.norm(dim=-1), raw.norm(dim=-1), rtol=1e-6, atol=1e-6
    )
    print("GEOMETRY_SELF_TEST_OK", flush=True)


def tensor_frame_to_pil(frame: torch.Tensor) -> Image.Image:
    """Convert a normalized [3,H,W] frame in [-1,1] to RGB PIL."""
    array = (
        ((frame.detach().float().clamp(-1, 1) + 1.0) * 127.5)
        .round()
        .to(torch.uint8)
        .permute(1, 2, 0)
        .cpu()
        .numpy()
    )
    return Image.fromarray(array, mode="RGB")


def split_views(image: Image.Image) -> dict[str, Image.Image]:
    """Return full two-camera frame and its main/wrist square views."""
    width, height = image.size
    views = {"both": image}
    if width == 2 * height:
        views["main"] = image.crop((0, 0, height, height))
        views["wrist"] = image.crop((height, 0, width, height))
    return views


def save_frame_views(image: Image.Image, stem: Path) -> None:
    for view_name, view in split_views(image).items():
        view.save(stem.with_name(f"{stem.name}_{view_name}.png"))


def cosine_to_rgb(cosine: np.ndarray) -> np.ndarray:
    """Small dependency-free blue-white-red map for cosine values in [-1,1]."""
    value = np.clip((cosine + 1.0) * 0.5, 0.0, 1.0)[..., None]
    blue = np.array([36.0, 90.0, 170.0])
    white = np.array([246.0, 246.0, 242.0])
    red = np.array([190.0, 42.0, 52.0])
    lower = blue + (white - blue) * (value * 2.0)
    upper = white + (red - white) * ((value - 0.5) * 2.0)
    rgb = np.where(value <= 0.5, lower, upper)
    return np.clip(rgb, 0, 255).astype(np.uint8)


def save_cosine_map(
    cosine: torch.Tensor,
    *,
    grid_height: int,
    grid_width: int,
    target_image: Image.Image,
    stem: Path,
) -> None:
    values = cosine.detach().float().cpu().reshape(grid_height, grid_width).numpy()
    heat = Image.fromarray(cosine_to_rgb(values), mode="RGB").resize(
        target_image.size, resample=Image.Resampling.NEAREST
    )
    heat.save(stem.with_name(f"{stem.name}_heat_both.png"))
    overlay = Image.blend(target_image.convert("RGB"), heat, alpha=0.48)
    overlay.save(stem.with_name(f"{stem.name}_overlay_both.png"))
    for name, view in split_views(heat).items():
        if name != "both":
            view.save(stem.with_name(f"{stem.name}_heat_{name}.png"))
    for name, view in split_views(overlay).items():
        if name != "both":
            view.save(stem.with_name(f"{stem.name}_overlay_{name}.png"))


def _font(size: int = 22) -> ImageFont.ImageFont:
    for candidate in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans.ttf",
    ):
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, size=size)
    return ImageFont.load_default()


def make_contact_sheet(sample_dir: Path, prompt: str) -> None:
    """Create a compact main-camera diagnostic sheet for manual inspection."""
    columns = (
        ("recorded_t0_main.png", "Current"),
        ("recorded_t4_main.png", "Recorded +4"),
        ("predicted_t4_main.png", "Decoded pred. +4"),
        ("recorded_t8_main.png", "Recorded +8"),
        ("predicted_t8_main.png", "Decoded pred. +8"),
        ("offset2_overlay_main.png", "Cosine map +8"),
    )
    images = [(Image.open(sample_dir / filename).convert("RGB"), label) for filename, label in columns]
    tile_w, tile_h = images[0][0].size
    top = 82
    label_h = 34
    sheet = Image.new("RGB", (len(images) * tile_w, top + tile_h + label_h), "white")
    draw = ImageDraw.Draw(sheet)
    title_font = _font(21)
    label_font = _font(18)
    prompt_text = prompt.replace("A video recorded from a robot's point of view executing the following instruction: ", "")
    if len(prompt_text) > 150:
        prompt_text = prompt_text[:147] + "..."
    draw.text((10, 8), prompt_text, fill="black", font=title_font)
    draw.text(
        (10, 45),
        "Projected decoder view is diagnostic; magnitude uses current-frame statistics only.",
        fill=(78, 78, 78),
        font=label_font,
    )
    for column, (image, label) in enumerate(images):
        x = column * tile_w
        sheet.paste(image, (x, top))
        draw.text((x + 8, top + tile_h + 5), label, fill="black", font=label_font)
    sheet.save(sample_dir / "contact_sheet_main.png")


def configure_dataset(cfg, args: argparse.Namespace):
    data_cfg = OmegaConf.create(OmegaConf.to_container(cfg.data.train, resolve=True))
    dataset_dirs = [args.dataset_root / name for name in SUITE_DIR_NAMES]
    missing = [str(path) for path in dataset_dirs if not path.is_dir()]
    if missing:
        raise FileNotFoundError(f"Missing LIBERO dataset directories: {missing}")
    data_cfg.dataset_dirs = [str(path) for path in dataset_dirs]
    data_cfg.text_embedding_cache_dir = str(args.text_cache_dir)
    data_cfg.pretrained_norm_stats = str(args.stats)
    data_cfg.val_set_proportion = 0.0
    data_cfg.is_training_set = False
    data_cfg.first_frame_only = False
    data_cfg.use_input_latent_cache = False
    return instantiate(data_cfg)


def suite_boundaries(dataset) -> list[tuple[str, int, int]]:
    multi = dataset.lerobot_dataset.multi_dataset
    boundaries = []
    start = 0
    for name, sub_dataset in zip(SUITE_DIR_NAMES, multi._datasets, strict=True):
        end = start + int(sub_dataset.num_frames)
        boundaries.append((name.replace("_no_noops_lerobot", ""), start, end))
        start = end
    if start != len(dataset):
        raise RuntimeError(f"Sub-dataset total {start} differs from dataset length {len(dataset)}")
    return boundaries


def infer_suite(index: int, boundaries: Iterable[tuple[str, int, int]]) -> str:
    for name, start, end in boundaries:
        if start <= index < end:
            return name
    raise IndexError(index)


def choose_indices(dataset, args: argparse.Namespace) -> tuple[list[int], list[tuple[str, int, int]]]:
    boundaries = suite_boundaries(dataset)
    if args.indices:
        requested = [int(index) for index in args.indices]
    else:
        count = max(1, int(args.samples_per_suite))
        requested = []
        for _, start, end in boundaries:
            width = end - start
            for rank in range(count):
                fraction = (rank + 1) / (count + 1)
                requested.append(start + int(width * fraction))

    chosen = []
    for initial in requested:
        suite = infer_suite(initial, boundaries)
        _, suite_start, suite_end = next(item for item in boundaries if item[0] == suite)
        selected = None
        for delta in range(max(1, int(args.max_valid_search))):
            candidate = min(initial + delta, suite_end - 1)
            if candidate < suite_start:
                continue
            sample = dataset._get(candidate)
            if not bool(sample["image_is_pad"].any().item()):
                selected = candidate
                break
        if selected is None:
            raise RuntimeError(f"No unpadded sample found near index {initial} in {suite}")
        chosen.append(selected)
    return chosen, boundaries


@torch.inference_mode()
def predict_sample(model, sample: dict, model_dtype: torch.dtype) -> dict:
    batch = torch.utils.data.default_collate([sample])
    device_type = torch.device(model.device).type
    autocast_enabled = device_type == "cuda" and model_dtype != torch.float32
    with torch.autocast(device_type=device_type, dtype=model_dtype, enabled=autocast_enabled):
        inputs = model.build_inputs(batch)
        latents = inputs["input_latents"]
        if latents.shape[2] != 3:
            raise ValueError(f"Expected 9 RGB frames -> 3 Wan latent frames, got {latents.shape}")
        batch_size = int(latents.shape[0])
        video_input = inputs["first_frame_latents"]
        if video_input is None:
            video_input = latents[:, :, :1]
        timestep_video = torch.zeros((batch_size,), device=model.device, dtype=latents.dtype)
        video_pre = model.video_expert.pre_dit(
            x=video_input,
            timestep=timestep_video,
            context=inputs["context"],
            context_mask=inputs["context_mask"],
            action=None,
            fuse_vae_embedding_in_latents=inputs["fuse_vae_embedding_in_latents"],
        )

        action = inputs["action"]
        timestep_action = torch.zeros((batch_size,), device=model.device, dtype=action.dtype)
        action_pre = model.action_expert.pre_dit(
            action_tokens=action,
            timestep=timestep_action,
            context=inputs["context"],
            context_mask=inputs["context_mask"],
        )
        attention_mask = model._build_mot_attention_mask(
            video_seq_len=video_pre["tokens"].shape[1],
            action_seq_len=action_pre["tokens"].shape[1],
            video_tokens_per_frame=int(video_pre["meta"]["tokens_per_frame"]),
            device=video_pre["tokens"].device,
        )
        _, video_hidden = model.mot(
            embeds_all={"video": video_pre["tokens"], "action": action_pre["tokens"]},
            attention_mask=attention_mask,
            freqs_all={"video": video_pre["freqs"], "action": action_pre["freqs"]},
            context_all={
                "video": {"context": video_pre["context"], "mask": video_pre["context_mask"]},
                "action": {"context": action_pre["context"], "mask": action_pre["context_mask"]},
            },
            t_mod_all={"video": video_pre["t_mod"], "action": action_pre["t_mod"]},
            return_video_hidden_layers=model.foresight_layers,
        )

        student_layer = int(model.foresight_layers[0])
        tokens_per_frame = int(video_pre["meta"]["tokens_per_frame"])
        h_obs = video_hidden[student_layer][:, :tokens_per_frame]
        shared = model.foresight_projectors["shared"]
        features = shared.backbone(h_obs)
        predicted_lifted = {
            int(offset): shared.head(features, f"{student_layer}_{int(offset)}")
            for offset in model.foresight_offsets
        }

        patch_size = tuple(int(value) for value in model.video_expert.patch_size)
        target_raw, target_tpf = patch_foresight_vae_latents(latents, patch_size=patch_size)
        if target_tpf != tokens_per_frame:
            raise RuntimeError(f"Target/policy token grids differ: {target_tpf} vs {tokens_per_frame}")
        latent_frames = int(latents.shape[2] // patch_size[0])
        raw_dim = int(target_raw.shape[-1])
        target_raw = target_raw.reshape(batch_size, latent_frames, tokens_per_frame, raw_dim)
        target_lifted = lift_foresight_vae_patches(
            target_raw.reshape(batch_size, -1, raw_dim),
            output_dim=int(model.video_expert.hidden_dim),
        ).reshape(batch_size, latent_frames, tokens_per_frame, -1)

    metrics = {}
    predicted_raw_scaled = []
    current_norm = target_raw[:, 0].float().norm(dim=-1).median(dim=1).values
    for offset in model.foresight_offsets:
        offset = int(offset)
        prediction = predicted_lifted[offset].float()
        target = target_lifted[:, offset].float()
        raw_prediction = inverse_isometric_lift(prediction, raw_dim=raw_dim)
        reprojection = lift_foresight_vae_patches(
            raw_prediction, output_dim=int(model.video_expert.hidden_dim)
        ).float()
        predicted_norm = raw_prediction.norm(dim=-1).median(dim=1).values.clamp_min(1e-8)
        scale = current_norm / predicted_norm
        predicted_raw_scaled.append(raw_prediction * scale[:, None, None])
        cosine = F.cosine_similarity(prediction, target, dim=-1)
        subspace_norm_fraction = reprojection.norm(dim=-1) / prediction.norm(dim=-1).clamp_min(1e-8)
        metrics[offset] = {
            "cosine": cosine,
            "cosine_mean": float(cosine.mean().item()),
            "cosine_median": float(cosine.median().item()),
            "subspace_norm_fraction_mean": float(subspace_norm_fraction.mean().item()),
            "subspace_norm_fraction_median": float(subspace_norm_fraction.median().item()),
            "future_blind_scale": float(scale.item()),
            "prediction_raw_median_norm": float(predicted_norm.item()),
            "current_raw_median_norm": float(current_norm.item()),
        }

    predicted_raw = torch.stack(predicted_raw_scaled, dim=1).reshape(
        batch_size, len(predicted_raw_scaled) * tokens_per_frame, raw_dim
    )
    _, channels, _, latent_height, latent_width = latents.shape
    predicted_future_latents = unpatch_vae_latents(
        predicted_raw,
        channels=int(channels),
        frames=len(predicted_raw_scaled),
        height=int(latent_height),
        width=int(latent_width),
        patch_size=patch_size,
    ).to(device=model.device, dtype=latents.dtype)
    predicted_clip_latents = torch.cat([latents[:, :, :1], predicted_future_latents], dim=2)

    with torch.autocast(device_type=device_type, dtype=model_dtype, enabled=autocast_enabled):
        predicted_frames = model._decode_latents(predicted_clip_latents)
        reconstructed_frames = model._decode_latents(latents)
    return {
        "batch": batch,
        "latents": latents,
        "predicted_frames": predicted_frames,
        "reconstructed_frames": reconstructed_frames,
        "metrics": metrics,
        "grid_height": int(latent_height // patch_size[1]),
        "grid_width": int(latent_width // patch_size[2]),
    }


def write_sample_outputs(
    *,
    output_dir: Path,
    index: int,
    suite: str,
    sample: dict,
    result: dict,
) -> dict:
    sample_dir = output_dir / f"{suite}_{index:07d}"
    sample_dir.mkdir(parents=True, exist_ok=True)
    prompt = str(sample["prompt"])
    recorded = [tensor_frame_to_pil(sample["video"][:, frame]) for frame in (0, 4, 8)]
    reconstructed = [result["reconstructed_frames"][frame] for frame in (0, 4, 8)]
    predicted = [result["predicted_frames"][frame] for frame in (0, 4, 8)]
    for timestep, image in zip((0, 4, 8), recorded, strict=True):
        save_frame_views(image, sample_dir / f"recorded_t{timestep}")
    for timestep, image in zip((0, 4, 8), reconstructed, strict=True):
        save_frame_views(image, sample_dir / f"reconstruction_t{timestep}")
    for timestep, image in zip((0, 4, 8), predicted, strict=True):
        save_frame_views(image, sample_dir / f"predicted_t{timestep}")

    serializable_metrics = {}
    for offset, metric in result["metrics"].items():
        target_image = recorded[offset]
        save_cosine_map(
            metric["cosine"][0],
            grid_height=result["grid_height"],
            grid_width=result["grid_width"],
            target_image=target_image,
            stem=sample_dir / f"offset{offset}",
        )
        serializable_metrics[str(offset)] = {
            key: value for key, value in metric.items() if key != "cosine"
        }
    metadata = {
        "dataset_index": int(index),
        "suite": suite,
        "prompt": prompt,
        "image_is_pad": sample["image_is_pad"].tolist(),
        "offset_metrics": serializable_metrics,
        "interpretation": (
            "Decoded predictions are diagnostic projections. The cosine-trained head does not "
            "identify latent magnitude; scale is estimated only from the current observation."
        ),
    }
    (sample_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    make_contact_sheet(sample_dir, prompt)
    return metadata


def main() -> None:
    args = parse_args()
    geometry_self_test()
    if args.geometry_self_test:
        return
    for required in (args.run_config, args.checkpoint, args.stats):
        if not required.is_file():
            raise FileNotFoundError(required)
    if not torch.cuda.is_available() and str(args.device).startswith("cuda"):
        raise RuntimeError("CUDA was requested but is unavailable.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    misc.register_work_dir(str(args.output_dir))
    cfg = OmegaConf.load(args.run_config)
    dataset = configure_dataset(cfg, args)
    indices, boundaries = choose_indices(dataset, args)
    print("DATASET_BOUNDARIES", boundaries, flush=True)
    print("SELECTED_INDICES", indices, flush=True)

    model_dtype = _dtype(args.dtype)
    model_cfg = OmegaConf.create(OmegaConf.to_container(cfg.model, resolve=True))
    model = instantiate(model_cfg, model_dtype=model_dtype, device=args.device)
    model.eval()
    model.requires_grad_(False)
    for expert_name in ("video_expert", "action_expert"):
        expert = getattr(model, expert_name, None)
        if expert is not None and hasattr(expert, "use_gradient_checkpointing"):
            expert.use_gradient_checkpointing = False
    print(f"LOADING_CHECKPOINT {args.checkpoint}", flush=True)
    payload = model.load_checkpoint(str(args.checkpoint), mmap=True)
    print(f"CHECKPOINT_KEYS {sorted(payload.keys())}", flush=True)
    del payload
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    all_metadata = []
    for ordinal, index in enumerate(indices, start=1):
        suite = infer_suite(index, boundaries)
        print(f"SAMPLE_START {ordinal}/{len(indices)} suite={suite} index={index}", flush=True)
        sample = dataset._get(index)
        if bool(sample["image_is_pad"].any().item()):
            raise RuntimeError(f"Selected sample {index} unexpectedly contains padding")
        result = predict_sample(model, sample, model_dtype)
        metadata = write_sample_outputs(
            output_dir=args.output_dir,
            index=index,
            suite=suite,
            sample=sample,
            result=result,
        )
        all_metadata.append(metadata)
        means = {
            offset: round(values["cosine_mean"], 4)
            for offset, values in metadata["offset_metrics"].items()
        }
        norm_fractions = {
            offset: round(values["subspace_norm_fraction_mean"], 4)
            for offset, values in metadata["offset_metrics"].items()
        }
        print(
            f"SAMPLE_DONE index={index} cosine={means} "
            f"subspace_norm_fraction={norm_fractions}",
            flush=True,
        )

    manifest = {
        "run_config": str(args.run_config.resolve()),
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_step": 20000,
        "target_type": "vae_direct",
        "foresight_offsets": [1, 2],
        "calibration": "per-offset global median patch norm matched to current latent only",
        "samples": all_metadata,
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"VISUALIZATION_COMPLETE {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
