#!/usr/bin/env python3
"""Fail-closed audit for the Qwen3.5 RoboTwin foresight recipe."""

from __future__ import annotations

import argparse
from math import ceil
from pathlib import Path

import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate


ROOT = Path(__file__).resolve().parents[1]
TASK = "robotwin_qwen35_foresight_3cam224"
EXPECTED_DATASET_SIZE = 5_368_277
EXPECTED_FINAL_STEP = 69_900


def _compose(args: argparse.Namespace):
    overrides = [
        f"task={TASK}",
        f"data.train.dataset_dirs=[{args.dataset_root}]",
        f"data.val.dataset_dirs=[{args.dataset_root}]",
        f"data.train.pretrained_norm_stats={args.dataset_stats}",
        f"data.val.pretrained_norm_stats={args.dataset_stats}",
        f"data.train.nonidle_filter_path={args.nonidle_filter}",
        f"batch_size={args.batch_size}",
        f"gradient_accumulation_steps={args.gradient_accumulation_steps}",
    ]
    with initialize_config_dir(config_dir=str(ROOT / "configs"), version_base="1.3"):
        return compose(config_name="train", overrides=overrides)


def _audit_config(cfg, args: argparse.Namespace) -> None:
    assert cfg.model._target_ == "jam_multimodal.runtime.create_jam_qwen35"
    assert cfg.model.action_fusion_type == "qwen_native_mot"
    assert int(cfg.model.num_views) == 3
    assert int(cfg.model.image_size) == 224
    assert int(cfg.model.context_dim) == 2048
    assert cfg.model.action_dit_pretrained_path is None
    assert cfg.model.action_dit_pretrained_layer_mapping is None
    assert not bool(cfg.model.skip_dit_load_from_pretrain)
    assert int(cfg.model.action_dit_config.action_dim) == 14
    assert int(cfg.model.action_dit_config.max_action_horizon) == 64
    assert int(cfg.model.action_dit_config.timestep_embed_dim) == 256
    assert cfg.model.action_dit_config.initializer_range is None
    assert cfg.model.action_dit_config.action_attention_mode == "causal"
    assert not bool(cfg.model.use_gradient_checkpointing)
    assert not bool(cfg.model.action_dit_config.use_gradient_checkpointing)
    assert float(cfg.model.loss.lambda_action) == 1.0
    assert float(cfg.model.loss.lambda_foresight) == 0.3
    assert int(cfg.model.foresight.layer) == 13
    assert list(cfg.model.foresight.frame_offsets) == [16]
    assert bool(cfg.model.foresight.teacher_early_exit)
    assert cfg.model.foresight.teacher_mode == "frozen_pretrained"

    train = cfg.data.train
    assert int(train.num_frames) == 17
    assert int(train.action_video_freq_ratio) == 1
    assert bool(train.endpoint_frames_only)
    assert list(train.video_size) == [224, 672]
    assert train.concat_multi_camera == "horizontal"
    assert train.robotwin_camera_layout == "qwen_horizontal_224"
    assert int(train.processor.image_obs_steps) == 2
    assert int(train.processor.num_output_cameras) == 3
    assert not bool(train.require_text_cache)
    assert train.qwen_text_cache_dir is None
    assert float(train.video_augmentation.p) == 0.5

    assert int(cfg.num_epochs) == 5
    assert cfg.max_steps is None
    assert int(cfg.save_every) == 5000
    assert bool(cfg.save_full_state)
    assert bool(cfg.keep_latest_state_only)
    assert str(cfg.mixed_precision) == "bf16"
    assert int(cfg.seed) == 42
    assert str(cfg.lr_scheduler_type) == "cosine"
    assert float(cfg.learning_rate) == 1.0e-4
    assert float(cfg.weight_decay) == 1.0e-2
    assert not bool(cfg.gather_metrics_every_step)
    assert not bool(cfg.persistent_workers)
    assert str(cfg.dataloader_multiprocessing_context) == "spawn"

    global_batch = (
        int(args.world_size)
        * int(args.batch_size)
        * int(args.gradient_accumulation_steps)
    )
    assert global_batch == 384, global_batch
    micro_steps = ceil(EXPECTED_DATASET_SIZE / (args.world_size * args.batch_size))
    steps_per_epoch = ceil(micro_steps / args.gradient_accumulation_steps)
    assert steps_per_epoch * int(cfg.num_epochs) == EXPECTED_FINAL_STEP


def _audit_dataset(cfg, *, sample: bool) -> None:
    dataset = instantiate(cfg.data.train)
    assert len(dataset) == EXPECTED_DATASET_SIZE, len(dataset)
    assert dataset.video_sample_indices == list(range(17))
    if sample:
        item = dataset[0]
        assert tuple(item["video"].shape) == (3, 2, 224, 672)
        assert tuple(item["action"].shape) == (16, 14)
        assert tuple(item["proprio"].shape) == (16, 14)
        assert tuple(item["image_is_pad"].shape) == (2,)
        assert tuple(item["action_is_pad"].shape) == (16,)
        assert isinstance(item["prompt"], str) and item["prompt"]


def _audit_checkpoint(path: Path, expected_step: int) -> None:
    payload = torch.load(path, map_location="cpu", mmap=True)
    assert payload["format"] == "qwen35_imagewam_native_qwen24_v1"
    assert int(payload["step"]) == int(expected_step)
    assert tuple(payload["future_frame_offsets"]) == (16,)
    assert int(payload["foresight_layer"]) == 13
    assert float(payload["loss_lambda_foresight"]) == 0.3
    assert payload["teacher_mode"] == "frozen_pretrained"
    assert bool(payload["teacher_early_exit"])
    assert payload["action_fusion_type"] == "qwen_native_mot"
    assert payload["action_expert_family"] == "ActionDiTQwen35Native"
    assert payload["action_initialization"] == "exact_qwen35_decoder_block_copy_v1"
    assert int(payload["action_hidden_dim"]) == 2048
    assert payload["action_attention_mode"] == "causal"
    assert tuple(payload["action_layer_types"]) == (
        "linear_attention",
        "linear_attention",
        "linear_attention",
        "full_attention",
    ) * 6
    assert int(payload["qwen_num_layers"]) == 24
    assert int(payload["action_num_layers"]) == 24
    assert tuple(payload["action_visual_layer_map"]) == tuple(range(24))
    state = payload["dit"]
    assert tuple(state["action_expert.action_encoder.weight"].shape) == (2048, 14)
    assert tuple(state["action_expert.head.weight"].shape) == (14, 2048)
    assert not any(key.startswith("teacher") for key in state)
    bad = [
        key
        for key, value in state.items()
        if torch.is_tensor(value)
        and (value.is_floating_point() or value.is_complex())
        and not bool(torch.isfinite(value).all().item())
    ]
    assert not bad, bad[:20]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--dataset-stats", type=Path, required=True)
    parser.add_argument("--nonidle-filter", type=Path, required=True)
    parser.add_argument("--qwen-model", type=Path, required=True)
    parser.add_argument("--world-size", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--instantiate-data", action="store_true")
    parser.add_argument("--sample", action="store_true")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--expected-step", type=int, default=EXPECTED_FINAL_STEP)
    args = parser.parse_args()

    for path in (
        args.dataset_root,
        args.dataset_stats,
        args.nonidle_filter,
        args.qwen_model,
    ):
        if not path.exists():
            raise FileNotFoundError(path)
    cfg = _compose(args)
    _audit_config(cfg, args)
    if args.instantiate_data:
        _audit_dataset(cfg, sample=args.sample)
    if args.checkpoint is not None:
        if not args.checkpoint.is_file():
            raise FileNotFoundError(args.checkpoint)
        _audit_checkpoint(args.checkpoint, args.expected_step)
    print(
        "QWEN35 ROBOTWIN FORESIGHT AUDIT PASSED "
        f"world={args.world_size} batch={args.batch_size} "
        f"ga={args.gradient_accumulation_steps} global_batch=384 "
        f"final_step={EXPECTED_FINAL_STEP} checkpoint={args.checkpoint}",
        flush=True,
    )


if __name__ == "__main__":
    main()
