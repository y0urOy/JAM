#!/usr/bin/env python3
"""Resolve and audit the Ovis foresight RoboTwin training configuration."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import OmegaConf


REPO_ROOT = Path(__file__).resolve().parents[1]


def _assert_equal(name: str, actual, expected) -> None:
    if actual != expected:
        raise AssertionError(f"{name}: expected {expected!r}, got {actual!r}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task",
        default="robotwin_ovis_u1_jam_foresight_paper30k",
    )
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--norm-stats", required=True)
    parser.add_argument("--nonidle-filter", required=True)
    parser.add_argument("--world-size", type=int, default=8)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--gradient-accumulation-steps", type=int)
    parser.add_argument("--action-init")
    parser.add_argument(
        "--video-backend",
        choices=("auto", "pyav", "torchcodec", "video_reader"),
        default="auto",
    )
    parser.add_argument(
        "--sample",
        action="store_true",
        help="Decode one real sample in addition to constructing both splits.",
    )
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root).resolve()
    norm_stats = Path(args.norm_stats).resolve()
    nonidle_filter = Path(args.nonidle_filter).resolve()
    for label, path in (
        ("dataset root", dataset_root),
        ("normalization statistics", norm_stats),
        ("non-idle filter", nonidle_filter),
    ):
        if not path.exists():
            raise FileNotFoundError(f"Missing {label}: {path}")
    if args.action_init and not Path(args.action_init).exists():
        raise FileNotFoundError(f"Missing ActionDiT initialization: {args.action_init}")

    overrides = [
        f"task={args.task}",
        f"data.train.dataset_dirs=[{dataset_root}]",
        f"data.val.dataset_dirs=[{dataset_root}]",
        f"data.train.pretrained_norm_stats={norm_stats}",
        f"data.val.pretrained_norm_stats={norm_stats}",
        f"data.train.nonidle_filter_path={nonidle_filter}",
    ]
    if args.batch_size is not None:
        overrides.append(f"batch_size={args.batch_size}")
    if args.gradient_accumulation_steps is not None:
        overrides.append(
            f"gradient_accumulation_steps={args.gradient_accumulation_steps}"
        )
    if args.video_backend != "auto":
        overrides.extend(
            [
                f"+data.train.lerobot_video_backend={args.video_backend}",
                f"+data.val.lerobot_video_backend={args.video_backend}",
            ]
        )

    with initialize_config_dir(
        version_base=None,
        config_dir=str(REPO_ROOT / "configs"),
    ):
        cfg = compose(config_name="train", overrides=overrides)

    _assert_equal("data.num_frames", int(cfg.data.train.num_frames), 17)
    _assert_equal("data.video_size", list(cfg.data.train.video_size), [288, 256])
    _assert_equal("data.cameras", int(cfg.data.train.processor.num_output_cameras), 3)
    _assert_equal("data.action_dim", int(cfg.data.train.processor.action_output_dim), 14)
    _assert_equal("data.proprio_dim", int(cfg.data.train.processor.proprio_output_dim), 14)
    _assert_equal("data.endpoint_frames_only", bool(cfg.data.train.endpoint_frames_only), True)
    _assert_equal("model.proprio_dim", int(cfg.model.proprio_dim), 14)
    _assert_equal("model.lambda_video", float(cfg.model.loss.lambda_video), 0.0)
    _assert_equal("model.lambda_action", float(cfg.model.loss.lambda_action), 1.0)
    _assert_equal("foresight.enabled", bool(cfg.model.foresight.enabled), True)
    _assert_equal("foresight.teacher_mode", str(cfg.model.foresight.teacher_mode), "frozen_pretrained")
    _assert_equal("foresight.drop_target_tokens", bool(cfg.model.foresight.drop_target_tokens), True)
    _assert_equal("foresight.layer", int(cfg.model.foresight.layer), 11)
    _assert_equal("foresight.encoder_layer", int(cfg.model.foresight.encoder_layer), 11)
    _assert_equal("foresight.offsets", list(cfg.model.foresight.offsets), [1])
    _assert_equal("foresight.lambda", float(cfg.model.foresight.lambda_foresight), 0.3)
    if args.batch_size is not None:
        _assert_equal("train.batch_size", int(cfg.batch_size), args.batch_size)
    if args.gradient_accumulation_steps is not None:
        _assert_equal(
            "train.gradient_accumulation_steps",
            int(cfg.gradient_accumulation_steps),
            args.gradient_accumulation_steps,
        )
    _assert_equal("train.num_epochs", int(cfg.num_epochs), 5)
    _assert_equal("train.precision", str(cfg.mixed_precision), "bf16")
    _assert_equal("train.learning_rate", float(cfg.learning_rate), 1e-4)
    _assert_equal("train.weight_decay", float(cfg.weight_decay), 1e-2)

    train_dataset = instantiate(cfg.data.train)
    val_dataset = instantiate(cfg.data.val)
    train_len = len(train_dataset)
    val_len = len(val_dataset)
    global_batch = int(cfg.batch_size) * args.world_size * int(cfg.gradient_accumulation_steps)
    repo_steps_per_epoch = math.ceil(train_len / global_batch)
    repo_five_epoch_steps = repo_steps_per_epoch * int(cfg.num_epochs)
    selected_steps = int(cfg.max_steps) if cfg.max_steps is not None else repo_five_epoch_steps
    selected_samples = selected_steps * global_batch

    if args.sample:
        sample = train_dataset[0]
        video = sample["video"]
        action = sample["action"]
        print(f"[sample] video={tuple(video.shape)} action={tuple(action.shape)}")

    print("ROBOTWIN OVIS FORESIGHT CONFIG AUDIT PASSED")
    print(OmegaConf.to_yaml(cfg.model.foresight, resolve=True).rstrip())
    print(
        f"[data] train={train_len:,} val={val_len:,} world_size={args.world_size} "
        f"per_rank_batch={int(cfg.batch_size)} global_batch={global_batch}"
    )
    print(
        f"[budget] selected_steps={selected_steps:,} selected_samples={selected_samples:,} "
        f"effective_epochs={selected_samples / train_len:.4f}"
    )
    print(
        f"[repo-5ep] steps_per_epoch={repo_steps_per_epoch:,} "
        f"five_epoch_steps={repo_five_epoch_steps:,}"
    )
    if int(cfg.max_steps or 0) == 30000 and repo_five_epoch_steps != 30000:
        print(
            "[protocol-note] paper main text selects 30,000 steps, while the public "
            f"5-epoch YAML resolves to {repo_five_epoch_steps:,} steps on this dataset."
        )


if __name__ == "__main__":
    main()
