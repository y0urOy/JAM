#!/usr/bin/env python3
"""Cache FLUX.2 Qwen3-4B contexts for the fixed ImageWAM paper15 tasks.

The strict DINO -> ActionDiTFlux2 policy has no online text encoder.  Its
ordinary LIBERO run therefore contains only the 40 training prompts in the
Qwen3 cache.  This utility deterministically selects the canonical 15%/seed42
LIBERO-Plus manifest, resolves the 1,502 unique formatted prompts, and writes
the exact 512-token Qwen3 payload format used during training.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import re
import sys
import time
import uuid
from pathlib import Path

import torch
from einops import rearrange
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from experiments.libero.prepare_libero_plus_tasks import load_tasks, sample_tasks
from jam_multimodal.datasets.lerobot.robot_video_dataset import DEFAULT_PROMPT
from jam_multimodal.models.backbones.flux2_imports import ensure_flux2_importable


EXPECTED_PLUS_COMMIT = "4976dc30028e805ff8094b55501d532c48fec182"
EXPECTED_MANIFEST_SHA256 = (
    "50a1eb797ebdb027356458b91f00ff1e064a25df9819ffbe83d916da84fd9277"
)
EXPECTED_SUITE_COUNTS = {
    "libero_spatial": 361,
    "libero_object": 378,
    "libero_goal": 389,
    "libero_10": 378,
}
EXPECTED_SELECTED_TASKS = 1506
EXPECTED_UNIQUE_PROMPTS = 1502


def _expanded_path(value: str) -> Path:
    return Path(os.path.expandvars(os.path.expanduser(value))).resolve()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--libero-plus-root",
        type=_expanded_path,
        default=_expanded_path("./third_party/LIBERO-plus"),
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
            "./data/libero_mujoco3.3.2/"
            "flux2_qwen3_cache_4b"
        ),
    )
    parser.add_argument(
        "--flux2-src",
        type=_expanded_path,
        default=_expanded_path("./third_party/flux2"),
    )
    parser.add_argument("--model-spec", default="Qwen/Qwen3-4B")
    parser.add_argument("--context-len", type=int, default=512)
    parser.add_argument("--hidden-dim", type=int, default=7680)
    parser.add_argument("--sample-ratio", type=float, default=0.15)
    parser.add_argument("--sample-seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--save-workers", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--payload-check-count", type=int, default=64)
    return parser.parse_args()


def _git_commit(path: Path) -> str:
    import subprocess

    return subprocess.check_output(
        ["git", "-C", str(path), "rev-parse", "HEAD"], text=True
    ).strip()


def _language_from_task_name(
    libero_plus_root: Path, suite: str, task_name: str
) -> str:
    """Reproduce LIBERO-Plus ``grab_language_from_filename`` without MuJoCo.

    Importing ``libero.libero.benchmark`` eagerly imports the rendering stack,
    including Wand/ImageMagick.  Text-cache generation only needs the task
    language, so parsing it directly keeps the Qwen training environment
    independent of EGL and renderer packages.
    """

    filename = f"{task_name}.bddl"
    if "_language_" not in filename:
        if filename[0].isupper():
            marker = filename.find("SCENE")
            offset = 8 if "SCENE10" in filename else 7
            language = " ".join(filename[marker + offset :].split("_"))
        else:
            language = " ".join(filename.split("_"))
        return language[: language.find(".bddl")]

    bddl_stem = task_name.split("_view_", 1)[0]
    bddl_file = (
        libero_plus_root
        / "libero/libero/bddl_files"
        / suite
        / f"{bddl_stem}.bddl"
    )
    if not bddl_file.is_file():
        raise FileNotFoundError(f"LIBERO-Plus language BDDL not found: {bddl_file}")
    match = re.search(
        r"\(\s*:language\s+([^)]*?)\s*\)",
        bddl_file.read_text(encoding="utf-8"),
        flags=re.IGNORECASE | re.DOTALL,
    )
    if match is None:
        raise RuntimeError(f"No :language field found in {bddl_file}")
    # bddl.parsing.scan_tokens lower-cases tokens before get_problem_info joins
    # them; match that exact benchmark behavior.
    return " ".join(match.group(1).split()).lower()


def _collect_paper15_prompts(
    libero_plus_root: Path,
    libero_config_path: Path,
    sample_ratio: float,
    sample_seed: int,
) -> tuple[list[str], dict[str, int], str]:
    if not libero_plus_root.is_dir():
        raise FileNotFoundError(f"LIBERO-Plus checkout not found: {libero_plus_root}")
    actual_commit = _git_commit(libero_plus_root)
    if actual_commit != EXPECTED_PLUS_COMMIT:
        raise RuntimeError(
            "LIBERO-Plus commit mismatch: "
            f"actual={actual_commit} expected={EXPECTED_PLUS_COMMIT}"
        )

    classification = (
        libero_plus_root / "libero/libero/benchmark/task_classification.json"
    )
    selected = sample_tasks(load_tasks(classification), sample_ratio, sample_seed)
    manifest = "".join(f"{task.suite},{task.task_id}\n" for task in selected)
    manifest_sha = hashlib.sha256(manifest.encode("utf-8")).hexdigest()
    suite_counts: dict[str, int] = {}
    for task in selected:
        suite_counts[task.suite] = suite_counts.get(task.suite, 0) + 1

    if (
        sample_ratio != 0.15
        or sample_seed != 42
        or len(selected) != EXPECTED_SELECTED_TASKS
        or suite_counts != EXPECTED_SUITE_COUNTS
        or manifest_sha != EXPECTED_MANIFEST_SHA256
    ):
        raise RuntimeError(
            "Selection is not the canonical ImageWAM paper15 manifest: "
            f"ratio={sample_ratio} seed={sample_seed} tasks={len(selected)} "
            f"counts={suite_counts} sha256={manifest_sha}"
        )

    prompts: list[str] = []
    seen: set[str] = set()
    for spec in selected:
        language = _language_from_task_name(
            libero_plus_root, spec.suite, spec.name
        )
        prompt = DEFAULT_PROMPT.format(task=language)
        if prompt not in seen:
            seen.add(prompt)
            prompts.append(prompt)

    if len(prompts) != EXPECTED_UNIQUE_PROMPTS:
        raise RuntimeError(
            f"Expected {EXPECTED_UNIQUE_PROMPTS} unique paper15 prompts, "
            f"got {len(prompts)}."
        )
    return prompts, suite_counts, manifest_sha


def _cache_path(cache_dir: Path, prompt: str, context_len: int) -> Path:
    digest = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    return cache_dir / f"{digest}.qwen3_flux2_len{context_len}.pt"


def _validate_payload(path: Path, context_len: int, hidden_dim: int) -> None:
    payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict) or set(payload) < {
        "text_hidden_states",
        "text_attention_mask",
    }:
        raise RuntimeError(f"Invalid Qwen3 cache payload keys: {path}")
    context = payload["text_hidden_states"]
    mask = payload["text_attention_mask"]
    if tuple(context.shape) != (context_len, hidden_dim):
        raise RuntimeError(
            f"Invalid Qwen3 context shape in {path}: {tuple(context.shape)}"
        )
    if context.dtype != torch.bfloat16 or not torch.isfinite(context).all():
        raise RuntimeError(
            f"Invalid Qwen3 context values in {path}: dtype={context.dtype}"
        )
    if tuple(mask.shape) != (context_len,) or mask.dtype != torch.bool:
        raise RuntimeError(
            f"Invalid Qwen3 mask in {path}: shape={tuple(mask.shape)} "
            f"dtype={mask.dtype}"
        )
    if not bool(mask.any()):
        raise RuntimeError(f"Qwen3 cache mask contains no valid token: {path}")


def _validate_coverage(
    prompts: list[str],
    cache_dir: Path,
    context_len: int,
    hidden_dim: int,
    payload_check_count: int,
) -> dict[str, int]:
    paths = [_cache_path(cache_dir, prompt, context_len) for prompt in prompts]
    missing = [path for path in paths if not path.is_file()]
    empty = [path for path in paths if path.is_file() and path.stat().st_size == 0]
    if missing or empty:
        raise RuntimeError(
            "Incomplete paper15 FLUX.2 Qwen3 cache: "
            f"expected={len(paths)} missing={len(missing)} empty={len(empty)} "
            f"first={[str(path) for path in (missing + empty)[:5]]}"
        )

    count = min(max(int(payload_check_count), 0), len(paths))
    if count == 1:
        indices = [0]
    elif count > 1:
        indices = sorted(
            {
                round(index * (len(paths) - 1) / (count - 1))
                for index in range(count)
            }
        )
    else:
        indices = []
    for index in tqdm(indices, desc="Validating Qwen3 cache", unit="file"):
        _validate_payload(paths[index], context_len, hidden_dim)
    return {"expected": len(paths), "missing": 0, "payloads_checked": len(indices)}


def _atomic_torch_save(payload: dict[str, torch.Tensor], path: Path) -> None:
    temporary = path.with_name(
        f".{path.name}.tmp.{os.getpid()}.{uuid.uuid4().hex}"
    )
    torch.save(payload, temporary)
    os.replace(temporary, path)


def _write_manifest(
    cache_dir: Path,
    *,
    prompts: list[str],
    suite_counts: dict[str, int],
    manifest_sha: str,
    context_len: int,
    hidden_dim: int,
    model_spec: str,
    existing_entries: int,
    new_entries: int,
) -> None:
    payload = {
        "format": "libero_plus_flux2_qwen3_paper15_cache_v1",
        "protocol": "jam_multimodal-paper15",
        "sample_ratio": 0.15,
        "sample_seed": 42,
        "selected_tasks": EXPECTED_SELECTED_TASKS,
        "unique_prompts": len(prompts),
        "suite_counts": suite_counts,
        "ordered_manifest_sha256": manifest_sha,
        "prompt_template": DEFAULT_PROMPT,
        "context_len": context_len,
        "hidden_dim": hidden_dim,
        "dtype": "bfloat16",
        "model_spec": model_spec,
        "existing_entries_this_run": existing_entries,
        "new_entries_this_run": new_entries,
    }
    output = cache_dir / "libero_plus_qwen3_flux2_paper15_manifest.json"
    temporary = output.with_name(f".{output.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, output)


def main() -> None:
    args = _parse_args()
    if args.context_len != 512 or args.hidden_dim != 7680:
        raise ValueError("Strict FLUX.2-4B cache requires [512,7680].")
    if args.batch_size <= 0 or args.save_workers <= 0:
        raise ValueError("batch-size and save-workers must be positive.")

    prompts, suite_counts, manifest_sha = _collect_paper15_prompts(
        args.libero_plus_root,
        args.libero_config_path,
        args.sample_ratio,
        args.sample_seed,
    )
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    paths = [_cache_path(args.cache_dir, prompt, args.context_len) for prompt in prompts]
    existing = sum(path.is_file() and path.stat().st_size > 0 for path in paths)
    missing_indices = [
        index
        for index, path in enumerate(paths)
        if args.overwrite or not path.is_file() or path.stat().st_size == 0
    ]
    print(
        "LIBERO-Plus Qwen3 paper15 inventory: "
        f"tasks={EXPECTED_SELECTED_TASKS} unique_prompts={len(prompts)} "
        f"cached={existing} to_encode={len(missing_indices)} "
        f"manifest_sha256={manifest_sha} cache={args.cache_dir}"
    )

    if args.validate_only:
        result = _validate_coverage(
            prompts,
            args.cache_dir,
            args.context_len,
            args.hidden_dim,
            args.payload_check_count,
        )
        print(f"LIBERO-PLUS QWEN3 CACHE VALIDATION PASSED: {result}")
        return

    if str(args.device).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to generate the Qwen3 paper15 cache.")

    ensure_flux2_importable(str(args.flux2_src))
    from flux2.text_encoder import OUTPUT_LAYERS_QWEN3

    device = torch.device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_spec, local_files_only=not args.allow_download
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model_spec,
        torch_dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
        local_files_only=not args.allow_download,
    ).to(device).eval()

    written = 0
    total_forward_seconds = 0.0
    with torch.inference_mode(), concurrent.futures.ThreadPoolExecutor(
        max_workers=args.save_workers
    ) as pool:
        for start in tqdm(
            range(0, len(missing_indices), args.batch_size),
            desc="Encoding paper15 Qwen3 prompts",
            unit="batch",
        ):
            indices = missing_indices[start : start + args.batch_size]
            if not indices:
                continue
            batch_prompts = [prompts[index] for index in indices]
            rendered = [
                tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt}],
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
                for prompt in batch_prompts
            ]
            encoded = tokenizer(
                rendered,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=args.context_len,
            )
            input_ids = encoded.input_ids.to(device)
            attention_mask = encoded.attention_mask.to(device)
            started = time.perf_counter()
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                use_cache=False,
            )
            hidden = torch.stack(
                [outputs.hidden_states[index] for index in OUTPUT_LAYERS_QWEN3],
                dim=1,
            )
            hidden = rearrange(hidden, "b c l d -> b l (c d)")
            total_forward_seconds += time.perf_counter() - started
            hidden_cpu = hidden.to(device="cpu", dtype=torch.bfloat16).contiguous()
            mask_cpu = attention_mask.to(device="cpu", dtype=torch.bool).contiguous()
            if tuple(hidden_cpu.shape[1:]) != (args.context_len, args.hidden_dim):
                raise RuntimeError(
                    f"Unexpected Qwen3 hidden shape: {tuple(hidden_cpu.shape)}"
                )

            futures = []
            for batch_index, prompt_index in enumerate(indices):
                payload = {
                    "text_hidden_states": hidden_cpu[batch_index].clone(),
                    "text_attention_mask": mask_cpu[batch_index].clone(),
                }
                futures.append(
                    pool.submit(_atomic_torch_save, payload, paths[prompt_index])
                )
            for future in futures:
                future.result()
            written += len(indices)

    result = _validate_coverage(
        prompts,
        args.cache_dir,
        args.context_len,
        args.hidden_dim,
        args.payload_check_count,
    )
    _write_manifest(
        args.cache_dir,
        prompts=prompts,
        suite_counts=suite_counts,
        manifest_sha=manifest_sha,
        context_len=args.context_len,
        hidden_dim=args.hidden_dim,
        model_spec=args.model_spec,
        existing_entries=existing,
        new_entries=written,
    )
    print(
        "LIBERO-PLUS QWEN3 CACHE BUILD PASSED: "
        f"{result} written={written} gpu_forward_seconds={total_forward_seconds:.2f}"
    )


if __name__ == "__main__":
    main()
