#!/usr/bin/env python3
"""Resolve and audit the FLUX.2-4B foresight RoboTwin recipe."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import OmegaConf


REPO_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_UNIQUE_PROMPTS = 921_032
EXPECTED_TRAIN_WINDOWS = 5_368_277
EXPECTED_GLOBAL_BATCH = 384
EXPECTED_FINAL_STEP = 69_900


def _equal(name: str, actual, expected) -> None:
    if actual != expected:
        raise AssertionError(f"{name}: expected {expected!r}, got {actual!r}")


def _cache_name(prompt: str, context_len: int) -> str:
    digest = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    return f"{digest}.qwen3_flux2_len{context_len}.pt"


def _audit_prompt_cache(
    dataset_root: Path,
    cache_dir: Path,
    context_len: int,
    override_instruction: str | None = None,
) -> None:
    tasks_path = dataset_root / "meta" / "tasks.jsonl"
    if not tasks_path.is_file():
        raise FileNotFoundError(f"missing RoboTwin task metadata: {tasks_path}")
    from jam_multimodal.datasets.lerobot.robot_video_dataset import DEFAULT_PROMPT

    if override_instruction is not None:
        unique_prompts = {DEFAULT_PROMPT.format(task=str(override_instruction))}
        expected_prompts = 1
    else:
        unique_prompts: set[str] = set()
        with tasks_path.open("r", encoding="utf-8") as stream:
            for line_number, raw_line in enumerate(stream, start=1):
                if not raw_line.strip():
                    continue
                record = json.loads(raw_line)
                if "task" not in record:
                    raise KeyError(f"missing task at {tasks_path}:{line_number}")
                unique_prompts.add(DEFAULT_PROMPT.format(task=str(record["task"])))
        expected_prompts = EXPECTED_UNIQUE_PROMPTS
    _equal("unique RoboTwin prompts", len(unique_prompts), expected_prompts)

    missing = []
    for prompt in unique_prompts:
        path = cache_dir / _cache_name(prompt, context_len)
        if not path.is_file() or path.stat().st_size == 0:
            missing.append(path.name)
            if len(missing) >= 10:
                break
    if missing:
        raise FileNotFoundError(
            f"Qwen3 cache is incomplete under {cache_dir}; first missing entries: {missing}"
        )
    print(
        f"[cache] prompts={len(unique_prompts):,} context_len={context_len} "
        f"dir={cache_dir} complete=true"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task",
        default="robotwin_flux2_klein_4b_base_jam_foresight",
    )
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--norm-stats", required=True, type=Path)
    parser.add_argument("--nonidle-filter", required=True, type=Path)
    parser.add_argument("--qwen-cache-dir", required=True, type=Path)
    parser.add_argument("--flux2-src", required=True, type=Path)
    parser.add_argument("--flux2-model", required=True, type=Path)
    parser.add_argument("--ae-model", required=True, type=Path)
    parser.add_argument("--action-init", type=Path)
    parser.add_argument("--world-size", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--check-cache", action="store_true")
    parser.add_argument("--instantiate-data", action="store_true")
    parser.add_argument("--sample", action="store_true")
    parser.add_argument(
        "--override-instruction",
        help="Use one deterministic instruction for a real-data memory smoke.",
    )
    parser.add_argument(
        "--skip-budget-check",
        action="store_true",
        help="Allow a reduced-world-size smoke while retaining all science checks.",
    )
    args = parser.parse_args()

    paths = {
        "dataset root": args.dataset_root,
        "normalization statistics": args.norm_stats,
        "non-idle filter": args.nonidle_filter,
        "FLUX.2 source": args.flux2_src,
        "FLUX.2 weights": args.flux2_model,
        "FLUX.2 autoencoder": args.ae_model,
    }
    if args.action_init is not None:
        paths["ActionDiT initialization"] = args.action_init
    for label, raw_path in paths.items():
        path = raw_path.expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"missing {label}: {path}")

    dataset_root = args.dataset_root.expanduser().resolve()
    norm_stats = args.norm_stats.expanduser().resolve()
    nonidle_filter = args.nonidle_filter.expanduser().resolve()
    qwen_cache_dir = args.qwen_cache_dir.expanduser().resolve()
    overrides = [
        f"task={args.task}",
        f"data.train.dataset_dirs=[{dataset_root}]",
        f"data.val.dataset_dirs=[{dataset_root}]",
        f"data.train.pretrained_norm_stats={norm_stats}",
        f"data.val.pretrained_norm_stats={norm_stats}",
        f"data.train.nonidle_filter_path={nonidle_filter}",
        f"data.train.qwen_text_cache_dir={qwen_cache_dir}",
        f"data.val.qwen_text_cache_dir={qwen_cache_dir}",
        f"model.flux2_src_path={args.flux2_src.expanduser().resolve()}",
        f"model.flux2_model_path={args.flux2_model.expanduser().resolve()}",
        f"model.ae_model_path={args.ae_model.expanduser().resolve()}",
        f"batch_size={args.batch_size}",
        f"gradient_accumulation_steps={args.gradient_accumulation_steps}",
    ]
    if args.action_init is not None:
        overrides.append(
            f"model.action_dit_pretrained_path={args.action_init.expanduser().resolve()}"
        )
    if args.override_instruction is not None:
        quoted_instruction = json.dumps(str(args.override_instruction))
        overrides.extend(
            [
                f"+data.train.override_instruction={quoted_instruction}",
                f"+data.val.override_instruction={quoted_instruction}",
            ]
        )

    with initialize_config_dir(
        version_base=None,
        config_dir=str(REPO_ROOT / "configs"),
    ):
        cfg = compose(config_name="train", overrides=overrides)

    _equal("data.num_frames", int(cfg.data.train.num_frames), 17)
    _equal("data.video_size", list(cfg.data.train.video_size), [288, 256])
    _equal("data.cameras", int(cfg.data.train.processor.num_output_cameras), 3)
    _equal("data.action_dim", int(cfg.data.train.processor.action_output_dim), 14)
    _equal("data.proprio_dim", int(cfg.data.train.processor.proprio_output_dim), 14)
    _equal("data.endpoint_frames_only", bool(cfg.data.train.endpoint_frames_only), True)
    # `require_text_cache` is the unrelated legacy UMT5/Wan switch.  FLUX.2
    # consumes qwen_text_cache_dir directly.
    _equal("data.require_text_cache", bool(cfg.data.train.require_text_cache), False)
    _equal("data.qwen_context_len", int(cfg.data.train.qwen_context_len), 128)
    _equal("data.qwen_cache_format", str(cfg.data.train.qwen_text_cache_format), "qwen3_flux2")
    _equal("model.variant", str(cfg.model.variant), "klein-base-4b")
    _equal("model.qwen_context_len", int(cfg.model.qwen_context_len), 128)
    _equal(
        "model.trim_cached_text_padding",
        bool(cfg.model.trim_cached_text_padding),
        True,
    )
    _equal("model.proprio_dim", int(cfg.model.proprio_dim), 14)
    _equal("model.lambda_video", float(cfg.model.loss.lambda_video), 0.0)
    _equal("model.lambda_action", float(cfg.model.loss.lambda_action), 1.0)
    _equal("foresight.enabled", bool(cfg.model.foresight.enabled), True)
    _equal("foresight.teacher_mode", str(cfg.model.foresight.teacher_mode), "frozen_pretrained")
    _equal("foresight.drop_target_tokens", bool(cfg.model.foresight.drop_target_tokens), True)
    _equal("foresight.layer", int(cfg.model.foresight.layer), 15)
    _equal("foresight.encoder_layer", int(cfg.model.foresight.encoder_layer), 15)
    _equal("foresight.teacher_early_exit", bool(cfg.model.foresight.teacher_early_exit), True)
    _equal("foresight.offsets", list(cfg.model.foresight.offsets), [1])
    _equal("foresight.lambda", float(cfg.model.foresight.lambda_foresight), 0.3)
    _equal("projector.hidden", int(cfg.model.foresight.projector_hidden_dim), 1536)
    _equal("projector.ffn", int(cfg.model.foresight.projector_ffn_dim), 6144)
    _equal("projector.layers", int(cfg.model.foresight.projector_num_layers), 2)
    _equal("projector.heads", int(cfg.model.foresight.projector_num_heads), 24)
    _equal("train.num_epochs", int(cfg.num_epochs), 5)
    _equal("train.max_steps", cfg.max_steps, None)
    _equal("train.precision", str(cfg.mixed_precision), "bf16")
    _equal("train.learning_rate", float(cfg.learning_rate), 1e-4)
    _equal("train.weight_decay", float(cfg.weight_decay), 1e-2)
    _equal("train.seed", int(cfg.seed), 42)

    global_batch = args.world_size * args.batch_size * args.gradient_accumulation_steps
    expected_steps = math.ceil(EXPECTED_TRAIN_WINDOWS / global_batch) * int(cfg.num_epochs)
    if not args.skip_budget_check:
        _equal("global batch", global_batch, EXPECTED_GLOBAL_BATCH)
        _equal("five-epoch optimizer steps", expected_steps, EXPECTED_FINAL_STEP)

    if args.check_cache:
        if not qwen_cache_dir.is_dir():
            raise FileNotFoundError(f"missing Qwen3 cache directory: {qwen_cache_dir}")
        _audit_prompt_cache(
            dataset_root,
            qwen_cache_dir,
            int(cfg.data.train.qwen_context_len),
            override_instruction=args.override_instruction,
        )

    if args.instantiate_data or args.sample:
        train_dataset = instantiate(cfg.data.train)
        val_dataset = instantiate(cfg.data.val)
        _equal("instantiated train windows", len(train_dataset), EXPECTED_TRAIN_WINDOWS)
        if args.sample:
            sample = train_dataset[0]
            print(
                f"[sample] video={tuple(sample['video'].shape)} "
                f"action={tuple(sample['action'].shape)}"
            )
        print(f"[data] train={len(train_dataset):,} val={len(val_dataset):,}")

    print("ROBOTWIN FLUX2 FORESIGHT CONFIG AUDIT PASSED")
    print(OmegaConf.to_yaml(cfg.model.foresight, resolve=True).rstrip())
    print(
        f"[budget] world_size={args.world_size} batch/rank={args.batch_size} "
        f"ga={args.gradient_accumulation_steps} global_batch={global_batch} "
        f"steps={expected_steps:,} samples={expected_steps * global_batch:,}"
    )


if __name__ == "__main__":
    main()
