"""Precompute Wan UMT5 contexts for every LIBERO-Plus instruction.

The strict DINO/VideoMAE policies intentionally reuse JAMWan's frozen Wan
UMT5 condition space.  The ordinary LIBERO dataset has only 40 instructions,
whereas the pinned LIBERO-Plus benchmark exposes 10,030 tasks (10,002 unique
language strings).  Evaluation must therefore populate the extra cache entries
before allocating multi-node rollout jobs.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import os
import sys
import uuid
from pathlib import Path

import torch
from tqdm import tqdm


SUITE_ORDER = ("libero_spatial", "libero_object", "libero_goal", "libero_10")
EXPECTED_SUITE_SIZES = {
    "libero_spatial": 2402,
    "libero_object": 2518,
    "libero_goal": 2591,
    "libero_10": 2519,
}
DEFAULT_PROMPT = (
    "A video recorded from a robot's point of view executing the following "
    "instruction: {task}"
)
ENCODER_ID = "wan22ti2v5b"


def _expanded_path(value: str) -> Path:
    return Path(os.path.expandvars(os.path.expanduser(value))).resolve()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--libero-plus-root",
        type=_expanded_path,
        default=_expanded_path(
            "./third_party/LIBERO-plus"
        ),
    )
    parser.add_argument(
        "--libero-config-path",
        type=_expanded_path,
        default=_expanded_path("./.libero_plus"),
    )
    parser.add_argument(
        "--cache-dir",
        type=_expanded_path,
        default=_expanded_path(
            "./data/text_embeds_cache/libero"
        ),
    )
    parser.add_argument(
        "--text-encoder-path",
        type=_expanded_path,
        default=_expanded_path(
            "./checkpoints/Wan-AI/"
            "Wan2.2-TI2V-5B/models_t5_umt5-xxl-enc-bf16.pth"
        ),
    )
    parser.add_argument(
        "--tokenizer-path",
        type=_expanded_path,
        default=_expanded_path(
            "./checkpoints/Wan-AI/"
            "Wan2.1-T2V-1.3B/google/umt5-xxl"
        ),
    )
    parser.add_argument("--context-len", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument(
        "--payload-check-count",
        type=int,
        default=64,
        help="Number of deterministic cache payloads to load during validation.",
    )
    return parser.parse_args()


def _collect_prompts(
    libero_plus_root: Path, libero_config_path: Path
) -> tuple[list[str], dict[str, int]]:
    if not libero_plus_root.is_dir():
        raise FileNotFoundError(f"LIBERO-Plus checkout not found: {libero_plus_root}")
    os.environ["LIBERO_CONFIG_PATH"] = str(libero_config_path)
    sys.path.insert(0, str(libero_plus_root))

    from libero.libero import benchmark

    benchmark_dict = benchmark.get_benchmark_dict()
    prompts: list[str] = []
    seen: set[str] = set()
    suite_sizes: dict[str, int] = {}
    for suite in SUITE_ORDER:
        # LIBERO-Plus prints every task order while constructing a suite.  That
        # is useful interactively but creates megabytes of scheduler logs here.
        with contextlib.redirect_stdout(io.StringIO()):
            suite_obj = benchmark_dict[suite]()
        size = int(suite_obj.get_num_tasks())
        suite_sizes[suite] = size
        expected = EXPECTED_SUITE_SIZES[suite]
        if size != expected:
            raise RuntimeError(
                f"Wrong LIBERO-Plus benchmark for {suite}: {size}, expected {expected}."
            )
        for task_id in range(size):
            language = str(suite_obj.get_task(task_id).language)
            prompt = DEFAULT_PROMPT.format(task=language)
            if prompt not in seen:
                seen.add(prompt)
                prompts.append(prompt)

    if sum(suite_sizes.values()) != 10030:
        raise RuntimeError(f"Expected 10,030 LIBERO-Plus tasks, got {suite_sizes}.")
    if len(prompts) != 10002:
        raise RuntimeError(
            f"Expected 10,002 unique LIBERO-Plus prompts, got {len(prompts)}."
        )
    return prompts, suite_sizes


def _cache_path(cache_dir: Path, prompt: str, context_len: int) -> Path:
    digest = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    return cache_dir / f"{digest}.t5_len{context_len}.{ENCODER_ID}.pt"


def _validate_payload(path: Path, context_len: int) -> None:
    payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict) or set(payload) < {"context", "mask"}:
        raise RuntimeError(f"Invalid UMT5 payload keys: {path}")
    context = payload["context"]
    mask = payload["mask"]
    if tuple(context.shape) != (context_len, 4096):
        raise RuntimeError(
            f"Invalid UMT5 context shape in {path}: {tuple(context.shape)}"
        )
    if context.dtype != torch.bfloat16:
        raise RuntimeError(f"Invalid UMT5 context dtype in {path}: {context.dtype}")
    if tuple(mask.shape) != (context_len,) or mask.dtype != torch.bool:
        raise RuntimeError(
            f"Invalid UMT5 mask in {path}: shape={tuple(mask.shape)} dtype={mask.dtype}"
        )
    if not torch.isfinite(context).all():
        raise RuntimeError(f"Non-finite UMT5 context in {path}")


def _validate_coverage(
    prompts: list[str], cache_dir: Path, context_len: int, payload_check_count: int
) -> dict[str, int]:
    paths = [_cache_path(cache_dir, prompt, context_len) for prompt in prompts]
    missing = [path for path in paths if not path.is_file()]
    empty = [path for path in paths if path.is_file() and path.stat().st_size == 0]
    if missing or empty:
        preview = [str(path) for path in (missing + empty)[:5]]
        raise RuntimeError(
            "Incomplete LIBERO-Plus UMT5 cache: "
            f"expected={len(paths)} missing={len(missing)} empty={len(empty)} "
            f"first={preview}"
        )

    check_count = min(max(int(payload_check_count), 0), len(paths))
    if check_count:
        if check_count == 1:
            indices = [0]
        else:
            indices = sorted(
                {
                    round(index * (len(paths) - 1) / (check_count - 1))
                    for index in range(check_count)
                }
            )
        for index in tqdm(indices, desc="Validating cache payloads", unit="file"):
            _validate_payload(paths[index], context_len)
    return {"expected": len(paths), "missing": 0, "payloads_checked": check_count}


def _atomic_torch_save(payload: dict[str, torch.Tensor], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(
        f".{output_path.name}.tmp.{os.getpid()}.{uuid.uuid4().hex}"
    )
    torch.save(payload, temporary)
    os.replace(temporary, output_path)


def _write_manifest(
    cache_dir: Path,
    *,
    prompts: list[str],
    suite_sizes: dict[str, int],
    context_len: int,
    new_entries: int,
    existing_entries: int,
) -> None:
    payload = {
        "format": "libero_plus_wan_umt5_cache_v1",
        "num_tasks": sum(suite_sizes.values()),
        "num_unique_prompts": len(prompts),
        "suite_sizes": suite_sizes,
        "prompt_template": DEFAULT_PROMPT,
        "context_len": context_len,
        "hidden_dim": 4096,
        "dtype": "bfloat16",
        "encoder_id": ENCODER_ID,
        "new_entries_this_run": new_entries,
        "existing_entries_this_run": existing_entries,
    }
    output = cache_dir / "libero_plus_umt5_manifest.json"
    temporary = output.with_name(f".{output.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, output)


def main() -> None:
    args = _parse_args()
    if args.context_len <= 0 or args.batch_size <= 0:
        raise ValueError("context-len and batch-size must be positive")

    prompts, suite_sizes = _collect_prompts(
        args.libero_plus_root, args.libero_config_path
    )
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    paths = [_cache_path(args.cache_dir, prompt, args.context_len) for prompt in prompts]
    existing = sum(path.is_file() for path in paths)
    missing_indices = [
        index
        for index, path in enumerate(paths)
        if args.overwrite or not path.is_file()
    ]
    print(
        "LIBERO-Plus UMT5 inventory: "
        f"tasks={sum(suite_sizes.values())} unique_prompts={len(prompts)} "
        f"cached={existing} to_encode={len(missing_indices)} cache={args.cache_dir}"
    )

    if args.validate_only:
        result = _validate_coverage(
            prompts,
            args.cache_dir,
            args.context_len,
            args.payload_check_count,
        )
        print(f"LIBERO-PLUS UMT5 CACHE VALIDATION PASSED: {result}")
        return

    if not args.text_encoder_path.is_file():
        raise FileNotFoundError(f"Wan UMT5 checkpoint not found: {args.text_encoder_path}")
    if not args.tokenizer_path.is_dir():
        raise FileNotFoundError(f"UMT5 tokenizer not found: {args.tokenizer_path}")
    if not torch.cuda.is_available() and str(args.device).startswith("cuda"):
        raise RuntimeError("CUDA is required to precompute the full UMT5 cache")

    if missing_indices:
        from jam_multimodal.models.backbones.helpers.loader import _load_registered_model
        from jam_multimodal.models.backbones.wan_video_text_encoder import (
            HuggingfaceTokenizer,
        )

        device = torch.device(args.device)
        text_encoder = _load_registered_model(
            str(args.text_encoder_path),
            "wan_video_text_encoder",
            torch_dtype=torch.bfloat16,
            device=str(device),
        ).eval()
        tokenizer = HuggingfaceTokenizer(
            name=str(args.tokenizer_path),
            seq_len=args.context_len,
            clean="whitespace",
        )

        with torch.inference_mode():
            for start in tqdm(
                range(0, len(missing_indices), args.batch_size),
                desc="Encoding LIBERO-Plus prompts",
                unit="batch",
            ):
                indices = missing_indices[start : start + args.batch_size]
                batch_prompts = [prompts[index] for index in indices]
                ids, mask = tokenizer(
                    batch_prompts, return_mask=True, add_special_tokens=True
                )
                ids = ids.to(device=device, non_blocking=True)
                mask = mask.to(device=device, dtype=torch.bool, non_blocking=True)
                context = text_encoder(ids, mask)
                if tuple(context.shape) != (
                    len(indices),
                    args.context_len,
                    4096,
                ):
                    raise RuntimeError(
                        f"Unexpected batched UMT5 shape: {tuple(context.shape)}"
                    )
                if not torch.isfinite(context).all():
                    raise RuntimeError("UMT5 produced non-finite embeddings")

                for batch_index, prompt_index in enumerate(indices):
                    payload = {
                        "context": context[batch_index]
                        .detach()
                        .to(device="cpu", dtype=torch.bfloat16)
                        .contiguous(),
                        "mask": mask[batch_index]
                        .detach()
                        .to(device="cpu", dtype=torch.bool)
                        .contiguous(),
                    }
                    _validate_payload_tensors = (
                        tuple(payload["context"].shape) == (args.context_len, 4096)
                        and tuple(payload["mask"].shape) == (args.context_len,)
                    )
                    if not _validate_payload_tensors:
                        raise RuntimeError("Invalid UMT5 payload before cache write")
                    _atomic_torch_save(payload, paths[prompt_index])

        del text_encoder
        torch.cuda.empty_cache()

    result = _validate_coverage(
        prompts,
        args.cache_dir,
        args.context_len,
        args.payload_check_count,
    )
    _write_manifest(
        args.cache_dir,
        prompts=prompts,
        suite_sizes=suite_sizes,
        context_len=args.context_len,
        new_entries=len(missing_indices),
        existing_entries=existing,
    )
    print(f"LIBERO-PLUS UMT5 CACHE PRECOMPUTE PASSED: {result}")


if __name__ == "__main__":
    main()
