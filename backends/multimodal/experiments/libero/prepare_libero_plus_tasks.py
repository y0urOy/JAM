"""Prepare validated, balanced, resumable LIBERO-Plus task manifests."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


SUITE_ORDER = ["libero_spatial", "libero_object", "libero_goal", "libero_10"]
EXPECTED_SUITE_SIZES = {
    "libero_spatial": 2402,
    "libero_object": 2518,
    "libero_goal": 2591,
    "libero_10": 2519,
}
SUITE_COST_WEIGHTS = {"libero_10": 1.75}
CATEGORY_COST_WEIGHTS = {"Sensor Noise": 2.0}
DIFFICULTY_COST_STEP = 0.15


@dataclass(frozen=True)
class TaskSpec:
    suite: str
    task_id: int
    name: str
    category: str
    difficulty: int | None
    estimated_cost: float

    @property
    def key(self) -> tuple[str, int]:
        return self.suite, self.task_id


def _expanded_path(value: str | os.PathLike[str]) -> Path:
    return Path(os.path.expanduser(os.path.expandvars(str(value)))).resolve()


def load_tasks(classification_file: Path) -> list[TaskSpec]:
    with classification_file.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(
            f"Expected a JSON object in {classification_file}, got {type(payload).__name__}"
        )

    tasks: list[TaskSpec] = []
    for suite in SUITE_ORDER:
        entries = payload.get(suite)
        if not isinstance(entries, list):
            raise ValueError(f"Classification has no task list for {suite!r}")
        expected_size = EXPECTED_SUITE_SIZES[suite]
        if len(entries) != expected_size:
            raise ValueError(
                f"Unexpected {suite} size: classification={len(entries)}, expected={expected_size}"
            )
        ids = [int(entry["id"]) for entry in entries]
        if ids != list(range(1, expected_size + 1)):
            raise ValueError(f"{suite} classification IDs must be contiguous and 1-based")

        for entry in entries:
            difficulty_value = entry.get("difficulty_level")
            difficulty = None if difficulty_value is None else int(difficulty_value)
            category = str(entry["category"])
            difficulty_weight = (
                1.0
                if difficulty is None
                else 1.0 + DIFFICULTY_COST_STEP * max(difficulty - 1, 0)
            )
            cost = (
                SUITE_COST_WEIGHTS.get(suite, 1.0)
                * CATEGORY_COST_WEIGHTS.get(category, 1.0)
                * difficulty_weight
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

    if len(tasks) != 10030:
        raise ValueError(f"Expected exactly 10030 LIBERO-Plus tasks, got {len(tasks)}")
    return tasks


def sample_tasks(tasks: list[TaskSpec], ratio: float, seed: int) -> list[TaskSpec]:
    if not 0.0 < ratio <= 1.0:
        raise ValueError(f"sample ratio must be in (0, 1], got {ratio}")
    if ratio == 1.0:
        return list(tasks)

    selected: list[TaskSpec] = []
    for suite in SUITE_ORDER:
        suite_tasks = [task for task in tasks if task.suite == suite]
        count = min(len(suite_tasks), max(1, math.ceil(len(suite_tasks) * ratio)))
        # Match ImageWAM's public run_libero_manager.py exactly: seed an
        # independent Python RNG with "<seed>:<suite>", sample task IDs without
        # replacement, then restore task-ID order in the manifest.
        rng = random.Random(f"{seed}:{suite}")
        sampled_ids = set(rng.sample(range(len(suite_tasks)), count))
        chosen = [task for task in suite_tasks if task.task_id in sampled_ids]
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


def _archive(path: Path, reason: str) -> Path:
    stamp = time.strftime("%Y%m%d_%H%M%S")
    archived = path.with_name(f"{path.name}.{reason}.{stamp}")
    suffix = 1
    while archived.exists():
        archived = path.with_name(f"{path.name}.{reason}.{stamp}.{suffix}")
        suffix += 1
    path.rename(archived)
    return archived


def scan_completed(
    output_dir: Path, selected: list[TaskSpec], *, skip_completed: bool
) -> set[tuple[str, int]]:
    completed: set[tuple[str, int]] = set()
    for task in selected:
        suite_dir = output_dir / task.suite
        candidates = sorted(suite_dir.glob(f"*task{task.task_id}_results.json"))
        valid: list[Path] = []
        for candidate in candidates:
            if _valid_result(candidate, task):
                valid.append(candidate)
            else:
                archived = _archive(candidate, "invalid")
                print(f"Archived invalid result: {candidate} -> {archived.name}")
        if skip_completed and valid:
            completed.add(task.key)
        elif not skip_completed:
            for candidate in valid:
                archived = _archive(candidate, "previous")
                print(f"Archived previous result: {candidate} -> {archived.name}")
    return completed


def make_chunks(
    pending: list[TaskSpec], *, num_slots: int, target_chunks_per_slot: int, max_chunk_size: int
) -> tuple[list[list[TaskSpec]], int]:
    if num_slots <= 0 or target_chunks_per_slot <= 0 or max_chunk_size <= 0:
        raise ValueError("num-slots, target-chunks-per-slot, and max-chunk-size must be positive")
    if not pending:
        return [], max_chunk_size

    target_chunks = min(len(pending), num_slots * target_chunks_per_slot)
    adaptive_size = math.ceil(len(pending) / target_chunks)
    effective_chunk_size = min(max_chunk_size, adaptive_size)
    num_chunks = math.ceil(len(pending) / effective_chunk_size)
    chunks: list[list[TaskSpec]] = [[] for _ in range(num_chunks)]
    costs = [0.0] * num_chunks
    for task in sorted(pending, key=lambda item: (-item.estimated_cost, item.suite, item.task_id)):
        candidates = [idx for idx, chunk in enumerate(chunks) if len(chunk) < effective_chunk_size]
        target = min(candidates, key=lambda idx: (costs[idx], len(chunks[idx]), idx))
        chunks[target].append(task)
        costs[target] += task.estimated_cost
    for chunk in chunks:
        chunk.sort(key=lambda item: (SUITE_ORDER.index(item.suite), item.task_id))
    return chunks, effective_chunk_size


def _atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(content, encoding="utf-8")
    os.replace(temporary, path)


def _atomic_write_json(path: Path, payload: Any) -> None:
    _atomic_write_text(path, json.dumps(payload, indent=2) + "\n")


def write_manifests(
    output_dir: Path,
    selected: list[TaskSpec],
    completed: set[tuple[str, int]],
    chunks: list[list[TaskSpec]],
    effective_chunk_size: int,
    args: argparse.Namespace,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    tasks_text = "".join(f"{task.suite},{task.task_id}\n" for task in selected)
    _atomic_write_text(output_dir / "tasks.txt", tasks_text)

    chunks_dir = output_dir / "chunks"
    chunks_dir.mkdir(parents=True, exist_ok=True)
    for stale in chunks_dir.glob("chunk_*.txt"):
        stale.unlink()

    manifest_chunks: list[dict[str, Any]] = []
    for index, chunk in enumerate(chunks):
        chunk_file = chunks_dir / f"chunk_{index:05d}.txt"
        content = "".join(f"{task.suite},{task.task_id}\n" for task in chunk)
        _atomic_write_text(chunk_file, content)
        manifest_chunks.append(
            {
                "chunk_id": index,
                "task_file": str(chunk_file.resolve()),
                "num_tasks": len(chunk),
                "estimated_cost": round(sum(task.estimated_cost for task in chunk), 6),
                "tasks": [asdict(task) for task in chunk],
            }
        )

    _atomic_write_json(
        output_dir / "chunk_manifest.json",
        {
            "classification_file": str(args.classification_file),
            "sample_ratio": args.sample_ratio,
            "sample_seed": args.sample_seed,
            "selected_tasks": len(selected),
            "already_completed_tasks": len(completed),
            "pending_tasks": sum(len(chunk) for chunk in chunks),
            "num_slots": args.num_slots,
            "target_chunks_per_slot": args.target_chunks_per_slot,
            "max_chunk_size": args.max_chunk_size,
            "effective_chunk_size": effective_chunk_size,
            "num_chunks": len(chunks),
            "chunks": manifest_chunks,
        },
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=_expanded_path)
    parser.add_argument("--classification-file", required=True, type=_expanded_path)
    parser.add_argument("--num-slots", required=True, type=int)
    parser.add_argument("--sample-ratio", type=float, default=1.0)
    parser.add_argument("--sample-seed", type=int, default=42)
    parser.add_argument("--max-chunk-size", type=int, default=640)
    parser.add_argument("--target-chunks-per-slot", type=int, default=2)
    parser.add_argument("--skip-completed", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.classification_file.is_file():
        raise FileNotFoundError(f"Classification file not found: {args.classification_file}")
    tasks = load_tasks(args.classification_file)
    selected = sample_tasks(tasks, args.sample_ratio, args.sample_seed)
    if args.sample_ratio == 1.0 and len(selected) != 10030:
        raise RuntimeError(f"Full LIBERO-Plus manifest must have 10030 tasks, got {len(selected)}")
    completed = scan_completed(args.output_dir, selected, skip_completed=args.skip_completed)
    pending = [task for task in selected if task.key not in completed]
    chunks, effective_chunk_size = make_chunks(
        pending,
        num_slots=args.num_slots,
        target_chunks_per_slot=args.target_chunks_per_slot,
        max_chunk_size=args.max_chunk_size,
    )
    write_manifests(
        args.output_dir,
        selected,
        completed,
        chunks,
        effective_chunk_size,
        args,
    )
    print(
        "LIBERO-Plus manifest: "
        f"selected={len(selected)} completed={len(completed)} pending={len(pending)} "
        f"chunks={len(chunks)} chunk_size<={effective_chunk_size}"
    )


if __name__ == "__main__":
    main()
