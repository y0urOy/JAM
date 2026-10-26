"""Chunked, resumable multi-GPU manager for LIBERO-Plus evaluation."""

from __future__ import annotations

import hashlib
import json
import math
import os
import queue
import random
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from statistics import median
from typing import Any

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf


@dataclass(frozen=True)
class TaskSpec:
    suite: str
    task_id: int
    name: str
    category: str
    difficulty: int | None
    estimated_cost: float = 1.0

    @property
    def key(self) -> tuple[str, int]:
        return (self.suite, self.task_id)


def _expanded_path(value: Any) -> Path:
    return Path(os.path.expanduser(os.path.expandvars(str(value)))).resolve()


def _classification_path() -> Path:
    explicit = os.environ.get("LIBERO_PLUS_CLASSIFICATION_PATH")
    if explicit:
        return _expanded_path(explicit)
    config_root = Path(os.environ.get("LIBERO_CONFIG_PATH", ".libero_plus"))
    if not config_root.is_absolute():
        config_root = Path.cwd() / config_root
    config_file = config_root / "config.yaml"
    if config_file.exists():
        config = OmegaConf.load(config_file)
        benchmark_root = _expanded_path(config.benchmark_root)
        return benchmark_root / "benchmark" / "task_classification.json"
    return Path("third_party/LIBERO-plus/libero/libero/benchmark/task_classification.json").resolve()


def load_tasks(cfg: DictConfig) -> list[TaskSpec]:
    path = _classification_path()
    if not path.exists():
        raise FileNotFoundError(
            f"LIBERO-Plus classification file not found: {path}. "
            "Run python tools/setup_simulators.py plus first."
        )
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected an object in {path}, got {type(payload).__name__}")

    manager = cfg.MULTIRUN
    suite_names = [str(x) for x in manager.task_suite_names]
    expected_sizes = {str(k): int(v) for k, v in manager.expected_suite_sizes.items()}
    suite_weights = {str(k): float(v) for k, v in manager.suite_cost_weights.items()}
    category_weights = {str(k): float(v) for k, v in manager.category_cost_weights.items()}
    difficulty_step = float(manager.difficulty_cost_step)
    tasks: list[TaskSpec] = []

    for suite in suite_names:
        entries = payload.get(suite)
        if not isinstance(entries, list):
            raise ValueError(f"Classification has no task list for suite {suite!r}")
        expected = expected_sizes.get(suite)
        if expected is not None and len(entries) != expected:
            raise ValueError(
                f"Unexpected {suite} size: classification has {len(entries)}, expected {expected}"
            )
        ids = [int(entry["id"]) for entry in entries]
        if ids != list(range(1, len(entries) + 1)):
            raise ValueError(f"{suite} classification IDs must be contiguous and 1-based")

        for entry in entries:
            difficulty_raw = entry.get("difficulty_level")
            difficulty = None if difficulty_raw is None else int(difficulty_raw)
            category = str(entry["category"])
            difficulty_multiplier = (
                1.0 if difficulty is None else 1.0 + difficulty_step * max(difficulty - 1, 0)
            )
            cost = (
                suite_weights.get(suite, 1.0)
                * category_weights.get(category, 1.0)
                * difficulty_multiplier
            )
            tasks.append(
                TaskSpec(
                    suite=suite,
                    task_id=int(entry["id"]) - 1,
                    name=str(entry.get("name", "")),
                    category=category,
                    difficulty=difficulty,
                    estimated_cost=round(cost, 6),
                )
            )
    return tasks


def apply_duration_reference(
    tasks: list[TaskSpec], reference_value: Any
) -> tuple[list[TaskSpec], int, Path | None]:
    """Replace scheduling-only costs with measured per-task durations.

    Task identities and ordering are untouched. Missing measurements retain the
    classification heuristic, scaled onto the measured-seconds range.
    """
    if reference_value is None or str(reference_value).strip().lower() in {"", "null", "none"}:
        return tasks, 0, None

    reference_path = _expanded_path(reference_value)
    if reference_path.is_dir():
        reference_path = reference_path / "libero_plus_summary.json"
    if not reference_path.is_file():
        raise FileNotFoundError(f"Duration reference not found: {reference_path}")
    with reference_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    rows = payload.get("per_task") if isinstance(payload, dict) else None
    if not isinstance(rows, list):
        raise ValueError(f"Duration reference has no per_task list: {reference_path}")

    measured: dict[tuple[str, int], float] = {}
    for row in rows:
        try:
            key = (str(row["suite"]), int(row["task_id"]))
            duration = float(row["duration"])
        except (KeyError, TypeError, ValueError):
            continue
        if math.isfinite(duration) and duration > 0.0:
            measured[key] = duration

    matched_ratios = [
        measured[task.key] / task.estimated_cost
        for task in tasks
        if task.key in measured and task.estimated_cost > 0.0
    ]
    if not matched_ratios:
        raise ValueError(
            f"Duration reference does not match any configured LIBERO-Plus task: {reference_path}"
        )
    fallback_scale = median(matched_ratios)
    adjusted = [
        replace(
            task,
            estimated_cost=round(
                measured.get(task.key, task.estimated_cost * fallback_scale), 6
            ),
        )
        for task in tasks
    ]
    return adjusted, len(matched_ratios), reference_path


def sample_tasks(
    tasks: list[TaskSpec],
    ratio: float,
    seed: int,
    suite_names: list[str],
    strategy: str = "sha256",
) -> list[TaskSpec]:
    if not 0.0 < ratio <= 1.0:
        raise ValueError(f"MULTIRUN.task_sample_ratio must be in (0, 1], got {ratio}")
    strategy = str(strategy).strip().lower()
    if strategy not in {"sha256", "jam_multimodal"}:
        raise ValueError(
            "MULTIRUN.task_sample_strategy must be sha256 or jam_multimodal, "
            f"got {strategy!r}"
        )
    if ratio == 1.0:
        return list(tasks)

    selected: list[TaskSpec] = []
    for suite in suite_names:
        suite_tasks = [task for task in tasks if task.suite == suite]
        count = min(len(suite_tasks), max(1, math.ceil(len(suite_tasks) * ratio)))
        if strategy == "jam_multimodal":
            # Match ImageWAM's public manager exactly so cross-repository
            # comparisons use the same task IDs, not merely the same ratio.
            expected_ids = list(range(len(suite_tasks)))
            actual_ids = [task.task_id for task in suite_tasks]
            if actual_ids != expected_ids:
                raise ValueError(
                    f"{suite} task IDs must be contiguous for ImageWAM sampling"
                )
            rng = random.Random(f"{seed}:{suite}")
            sampled_ids = set(rng.sample(expected_ids, count))
            chosen = [task for task in suite_tasks if task.task_id in sampled_ids]
        else:
            def sample_key(task: TaskSpec) -> bytes:
                value = f"{seed}:{task.suite}:{task.task_id}".encode("utf-8")
                return hashlib.sha256(value).digest()

            chosen = sorted(suite_tasks, key=sample_key)[:count]
        selected.extend(sorted(chosen, key=lambda task: task.task_id))
    return selected


def _valid_result(path: Path, expected: TaskSpec) -> bool:
    try:
        with path.open("r", encoding="utf-8") as handle:
            result = json.load(handle)
        episodes = int(result["total_episodes"])
        successes = int(result["successes"])
        return (
            str(result["task_suite"]) == expected.suite
            and int(result["task_id"]) == expected.task_id
            and episodes > 0
            and 0 <= successes <= episodes
        )
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
        return False


def _archive_result(path: Path, suffix: str) -> Path:
    stamp = time.strftime("%Y%m%d_%H%M%S")
    candidate = path.with_name(f"{path.name}.{suffix}.{stamp}")
    counter = 1
    while candidate.exists():
        candidate = path.with_name(f"{path.name}.{suffix}.{stamp}.{counter}")
        counter += 1
    path.rename(candidate)
    return candidate


def scan_completed(
    output_dir: Path, selected: list[TaskSpec], *, skip_completed: bool
) -> set[tuple[str, int]]:
    completed: set[tuple[str, int]] = set()
    for task in selected:
        suite_dir = output_dir / task.suite
        candidates = sorted(suite_dir.glob(f"*task{task.task_id}_results.json"))
        valid_candidates: list[Path] = []
        for path in candidates:
            if _valid_result(path, task):
                valid_candidates.append(path)
            else:
                archived = _archive_result(path, "invalid")
                print(f"Invalid result archived: {path} -> {archived.name}")

        if skip_completed and valid_candidates:
            completed.add(task.key)
            continue
        if not skip_completed:
            for path in valid_candidates:
                archived = _archive_result(path, "previous")
                print(f"Previous result archived: {path} -> {archived.name}")
    return completed


def make_chunks(tasks: list[TaskSpec], chunk_size: int, balance: bool) -> list[list[TaskSpec]]:
    if chunk_size <= 0:
        raise ValueError(f"MULTIRUN.chunk_size must be positive, got {chunk_size}")
    if not tasks:
        return []
    if not balance:
        return [tasks[start : start + chunk_size] for start in range(0, len(tasks), chunk_size)]

    num_chunks = math.ceil(len(tasks) / chunk_size)
    chunks: list[list[TaskSpec]] = [[] for _ in range(num_chunks)]
    costs = [0.0] * num_chunks
    for task in sorted(tasks, key=lambda item: (-item.estimated_cost, item.suite, item.task_id)):
        candidates = [idx for idx, chunk in enumerate(chunks) if len(chunk) < chunk_size]
        target = min(candidates, key=lambda idx: (costs[idx], len(chunks[idx]), idx))
        chunks[target].append(task)
        costs[target] += task.estimated_cost
    for chunk in chunks:
        chunk.sort(key=lambda item: (item.suite, item.task_id))
    return chunks


def _is_blocked_override(raw_override: str) -> bool:
    key = raw_override.split("=", 1)[0].lstrip("+~")
    blocked = {
        "task",
        "ckpt",
        "gpu_id",
        "EVALUATION.output_dir",
        "EVALUATION.task_file",
        "EVALUATION.task_suite_name",
        "EVALUATION.task_id",
        "EVALUATION.num_trials",
        "EVALUATION.dataset_stats_path",
    }
    return key in blocked or key.startswith("MULTIRUN.") or key.startswith("hydra.")


def collect_worker_overrides() -> list[str]:
    return [
        override
        for override in HydraConfig.get().overrides.task
        if not _is_blocked_override(override)
    ]


def _task_choice() -> str:
    choice = HydraConfig.get().runtime.choices.get("task")
    if not choice:
        raise ValueError("Hydra task choice is empty; pass task=<evaluation task config>.")
    return str(choice)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=False)
        handle.write("\n")
    os.replace(temp, path)


def write_manifests(
    output_dir: Path,
    selected: list[TaskSpec],
    completed: set[tuple[str, int]],
    chunks: list[list[TaskSpec]],
) -> list[Path]:
    with (output_dir / "tasks.txt").open("w", encoding="utf-8") as handle:
        for task in selected:
            handle.write(f"{task.suite},{task.task_id}\n")

    chunks_dir = output_dir / "chunks"
    chunks_dir.mkdir(parents=True, exist_ok=True)
    chunk_files: list[Path] = []
    manifest_chunks = []
    for index, chunk in enumerate(chunks):
        path = chunks_dir / f"chunk_{index:05d}.txt"
        with path.open("w", encoding="utf-8") as handle:
            for task in chunk:
                handle.write(f"{task.suite},{task.task_id}\n")
        chunk_files.append(path.resolve())
        manifest_chunks.append(
            {
                "chunk_id": index,
                "task_file": str(path.resolve()),
                "num_tasks": len(chunk),
                "estimated_cost": round(sum(task.estimated_cost for task in chunk), 6),
                "tasks": [asdict(task) for task in chunk],
            }
        )

    _write_json(
        output_dir / "chunk_manifest.json",
        {
            "selected_tasks": len(selected),
            "already_completed_tasks": len(completed),
            "pending_tasks": sum(len(chunk) for chunk in chunks),
            "num_chunks": len(chunks),
            "chunks": manifest_chunks,
        },
    )
    return chunk_files


def run_chunk(
    *,
    chunk_id: int,
    chunk_file: Path,
    gpu_id: int,
    task_choice: str,
    ckpt: Path,
    dataset_stats: Path,
    num_trials: int,
    output_dir: Path,
    log_dir: Path,
    worker_overrides: list[str],
) -> tuple[int, int, int, Path]:
    log_path = log_dir / f"chunk_{chunk_id:05d}_gpu{gpu_id}.log"
    command = [
        sys.executable,
        "experiments/libero/eval_libero_single.py",
        "--config-name=sim_libero_plus.yaml",
        f"task={task_choice}",
        f"ckpt={ckpt}",
        f"gpu_id={gpu_id}",
        f"EVALUATION.output_dir={output_dir}",
        f"EVALUATION.dataset_stats_path={dataset_stats}",
        f"EVALUATION.num_trials={num_trials}",
        f"EVALUATION.task_file={chunk_file}",
        *worker_overrides,
    ]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    env["MUJOCO_EGL_DEVICE_ID"] = str(gpu_id)
    env.setdefault("MUJOCO_GL", "egl")
    env.setdefault("PYOPENGL_PLATFORM", "egl")
    with log_path.open("w", encoding="utf-8") as log:
        log.write("Command: " + " ".join(command) + "\n")
        log.flush()
        result = subprocess.run(
            command,
            cwd=Path.cwd(),
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    return chunk_id, gpu_id, result.returncode, log_path


@hydra.main(version_base="1.3", config_path="../../configs", config_name="sim_libero_plus.yaml")
def main(cfg: DictConfig) -> None:
    if cfg.ckpt is None:
        raise ValueError("ckpt must not be null")
    if cfg.EVALUATION.dataset_stats_path is None:
        raise ValueError("EVALUATION.dataset_stats_path must not be null")

    ckpt = _expanded_path(cfg.ckpt)
    dataset_stats = _expanded_path(cfg.EVALUATION.dataset_stats_path)
    if not ckpt.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")
    if not dataset_stats.is_file():
        raise FileNotFoundError(f"Dataset stats not found: {dataset_stats}")

    output_dir = _expanded_path(cfg.EVALUATION.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir = output_dir / "task_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    failed_file = output_dir / "failed_chunks.txt"
    if failed_file.exists():
        failed_file.unlink()

    manager = cfg.MULTIRUN
    num_gpus = int(manager.num_gpus)
    slots_per_gpu = int(manager.max_tasks_per_gpu)
    if num_gpus <= 0 or slots_per_gpu <= 0:
        raise ValueError("num_gpus and max_tasks_per_gpu must both be positive")

    all_tasks = load_tasks(cfg)
    all_tasks, duration_matches, duration_reference = apply_duration_reference(
        all_tasks, manager.get("duration_reference", None)
    )
    suite_names = [str(name) for name in manager.task_suite_names]
    selected = sample_tasks(
        all_tasks,
        ratio=float(manager.task_sample_ratio),
        seed=int(manager.task_sample_seed),
        suite_names=suite_names,
        strategy=str(manager.get("task_sample_strategy", "sha256")),
    )
    if float(manager.task_sample_ratio) == 1.0 and len(selected) != 10030:
        raise RuntimeError(f"Full LIBERO-Plus evaluation must select 10030 tasks, got {len(selected)}")

    completed = scan_completed(
        output_dir,
        selected,
        skip_completed=bool(manager.skip_completed_tasks),
    )
    pending = [task for task in selected if task.key not in completed]
    configured_chunk_size = int(manager.chunk_size)
    target_chunks_per_slot = int(manager.get("target_chunks_per_slot", 0))
    effective_chunk_size = configured_chunk_size
    if pending and target_chunks_per_slot > 0:
        target_chunk_count = num_gpus * slots_per_gpu * target_chunks_per_slot
        adaptive_chunk_size = math.ceil(len(pending) / target_chunk_count)
        effective_chunk_size = min(configured_chunk_size, max(1, adaptive_chunk_size))
    chunks = make_chunks(
        pending,
        chunk_size=effective_chunk_size,
        balance=bool(manager.balance_chunks),
    )
    chunk_files = write_manifests(output_dir, selected, completed, chunks)
    OmegaConf.save(config=cfg, f=str(output_dir / "manager_config.yaml"))

    print(
        f"LIBERO-Plus tasks: selected={len(selected)} completed={len(completed)} "
        f"pending={len(pending)} chunks={len(chunks)} "
        f"chunk_size={effective_chunk_size} (configured_max={configured_chunk_size}) "
        f"sample_strategy={manager.get('task_sample_strategy', 'sha256')}"
    )
    if duration_reference is not None:
        print(
            f"Scheduling costs: measured durations from {duration_reference} "
            f"(matched={duration_matches}/{len(all_tasks)}; result protocol unchanged)"
        )
    if bool(manager.get("create_only", False)):
        print("MULTIRUN.create_only=true: manifests created; worker launch skipped.")
        return
    if not chunks:
        print("All selected tasks already have valid results.")
        return

    task_choice = _task_choice()
    worker_overrides = collect_worker_overrides()
    available_slots: queue.Queue[int] = queue.Queue()
    for gpu_id in range(num_gpus):
        for _ in range(slots_per_gpu):
            available_slots.put(gpu_id)
    print_lock = threading.Lock()

    def scheduled_run(chunk_id: int, chunk_file: Path):
        gpu_id = available_slots.get()
        try:
            with print_lock:
                print(f"START chunk={chunk_id} gpu={gpu_id} tasks={len(chunks[chunk_id])}")
            return run_chunk(
                chunk_id=chunk_id,
                chunk_file=chunk_file,
                gpu_id=gpu_id,
                task_choice=task_choice,
                ckpt=ckpt,
                dataset_stats=dataset_stats,
                num_trials=int(cfg.EVALUATION.num_trials),
                output_dir=output_dir,
                log_dir=log_dir,
                worker_overrides=worker_overrides,
            )
        finally:
            available_slots.put(gpu_id)

    failures: list[tuple[int, int, int, Path]] = []
    max_workers = num_gpus * slots_per_gpu
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(scheduled_run, index, path): index
            for index, path in enumerate(chunk_files)
        }
        for future in as_completed(futures):
            result = future.result()
            chunk_id, gpu_id, returncode, log_path = result
            with print_lock:
                status = "DONE" if returncode == 0 else "FAILED"
                print(
                    f"{status} chunk={chunk_id} gpu={gpu_id} returncode={returncode} "
                    f"log={log_path}"
                )
            if returncode != 0:
                failures.append(result)

    if failures:
        with failed_file.open("w", encoding="utf-8") as handle:
            for chunk_id, gpu_id, returncode, log_path in sorted(failures):
                handle.write(
                    f"chunk={chunk_id} gpu={gpu_id} returncode={returncode} "
                    f"task_file={chunk_files[chunk_id]} log={log_path}\n"
                )
        raise RuntimeError(
            f"{len(failures)} chunk(s) failed; see {failed_file}. "
            "Valid per-task JSON files will be reused on the next run."
        )


if __name__ == "__main__":
    main()
