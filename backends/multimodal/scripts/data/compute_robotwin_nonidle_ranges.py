#!/usr/bin/env python3
"""Compute OpenPI-style non-idle keep ranges for a RoboTwin LeRobot dataset.

The output JSON does not modify the dataset. It can be passed to the JAMPolicy
LeRobot dataloader via `data.train.nonidle_filter_path=...`.

Filtering follows the OpenPI DROID recipe:
- Detect idle timesteps.
- Remove idle segments whose length is at least `min_idle_len`.
- Keep contiguous non-idle ranges with length at least `min_non_idle_len`.

RoboTwin terminal idle/no-op frames are intentionally kept. They often contain
the release/open/settle tail needed to learn successful episode endings. Kept
ranges are not shortened; the dataloader may still concatenate separate kept
ranges into one filtered timeline.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


ACTION_CANDIDATES = ("action", "action.default")
STATE_CANDIDATES = ("observation.state", "observation.state.default")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _load_episodes(dataset_dir: Path) -> dict[int, dict[str, Any]]:
    episodes: dict[int, dict[str, Any]] = {}
    for row in _read_jsonl(dataset_dir / "meta" / "episodes.jsonl"):
        if "episode_index" in row:
            episodes[int(row["episode_index"])] = row
    return episodes


def _episode_file_from_info(dataset_dir: Path, episode_index: int) -> Path:
    info_path = dataset_dir / "meta" / "info.json"
    if info_path.exists():
        info = json.loads(info_path.read_text(encoding="utf-8"))
        data_path = info.get("data_path")
        if isinstance(data_path, str) and "episode_index" in data_path:
            return dataset_dir / data_path.format(
                episode_index=episode_index,
                episode_chunk=episode_index // 1000,
            )
    return dataset_dir / "data" / f"chunk-{episode_index // 1000:03d}" / f"episode_{episode_index:06d}.parquet"


def _find_episode_files(dataset_dir: Path, max_episodes: int | None) -> list[tuple[int, Path]]:
    episodes = _load_episodes(dataset_dir)
    if episodes:
        pairs = [
            (episode_index, _episode_file_from_info(dataset_dir, episode_index))
            for episode_index in sorted(episodes)
        ]
    else:
        pairs = [
            (int(path.stem.split("_")[-1]), path)
            for path in sorted((dataset_dir / "data").glob("chunk-*/episode_*.parquet"))
        ]
    pairs = [(episode_index, path) for episode_index, path in pairs if path.exists()]
    if max_episodes is not None:
        pairs = pairs[:max_episodes]
    return pairs


def _select_column(schema_names: list[str], candidates: tuple[str, ...], label: str) -> str:
    for candidate in candidates:
        if candidate in schema_names:
            return candidate
    raise KeyError(f"Could not find {label} column. Tried {candidates}; available columns: {schema_names}")


def _column_to_numpy(table, column_name: str) -> np.ndarray:
    arr = np.asarray(table[column_name].to_pylist(), dtype=np.float64)
    if arr.ndim == 1:
        arr = arr[:, None]
    return arr


def _split_arm_gripper(delta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if delta.shape[1] >= 14:
        arm = np.concatenate([delta[:, :6], delta[:, 7:13]], axis=1)
        gripper = delta[:, [6, 13]]
        return arm, gripper
    return delta, np.zeros((delta.shape[0], 0), dtype=delta.dtype)


def _idle_mask(
    action: np.ndarray,
    state: np.ndarray,
    *,
    idle_l2_threshold: float,
    idle_arm_l2_threshold: float,
    idle_gripper_l2_threshold: float,
) -> np.ndarray:
    delta = action - state
    target_l2 = np.linalg.norm(delta, axis=1)
    arm_delta, gripper_delta = _split_arm_gripper(delta)
    arm_l2 = np.linalg.norm(arm_delta, axis=1)
    gripper_l2 = np.linalg.norm(gripper_delta, axis=1) if gripper_delta.shape[1] else np.zeros_like(target_l2)
    return (
        (target_l2 <= idle_l2_threshold)
        | (
            (arm_l2 <= idle_arm_l2_threshold)
            & (gripper_l2 <= idle_gripper_l2_threshold)
        )
    )


def _true_spans(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    padded = np.concatenate([[False], mask.astype(bool), [False]])
    diff = np.diff(padded.astype(np.int8))
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]
    return starts, ends


def _compute_keep_ranges(
    idle: np.ndarray,
    *,
    min_idle_len: int,
    min_non_idle_len: int,
    filter_last_n_in_ranges: int,
) -> list[list[int]]:
    idle_starts, idle_ends = _true_spans(idle)
    idle_lengths = idle_ends - idle_starts
    remove_mask = np.zeros(len(idle), dtype=bool)
    for start, end, length in zip(idle_starts, idle_ends, idle_lengths, strict=True):
        if int(end) == len(idle):
            continue
        if int(length) >= min_idle_len:
            remove_mask[int(start):int(end)] = True

    keep_mask = ~remove_mask
    keep_starts, keep_ends = _true_spans(keep_mask)
    ranges: list[list[int]] = []
    for start, end in zip(keep_starts, keep_ends, strict=True):
        start_i = int(start)
        end_i = int(end)
        if end_i - start_i < min_non_idle_len:
            continue
        ranges.append([start_i, end_i])
    return ranges


def _load_action_state(path: Path) -> tuple[np.ndarray, np.ndarray]:
    import pyarrow.parquet as pq

    table = pq.read_table(path)
    names = table.schema.names
    action_col = _select_column(names, ACTION_CANDIDATES, "action")
    state_col = _select_column(names, STATE_CANDIDATES, "state")
    action = _column_to_numpy(table, action_col)
    state = _column_to_numpy(table, state_col)
    if action.shape != state.shape:
        raise ValueError(f"Action/state shape mismatch in {path}: action={action.shape}, state={state.shape}")
    return action, state


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--idle-l2-threshold", type=float, default=1e-3)
    parser.add_argument("--idle-arm-l2-threshold", type=float, default=1e-3)
    parser.add_argument("--idle-gripper-l2-threshold", type=float, default=1e-3)
    parser.add_argument("--min-idle-len", type=int, default=5)
    parser.add_argument("--min-non-idle-len", type=int, default=1)
    parser.add_argument(
        "--filter-last-n-in-ranges",
        type=int,
        default=0,
        help="Deprecated compatibility option; kept ranges are no longer shortened.",
    )
    args = parser.parse_args()

    dataset_dir = args.dataset_dir.expanduser().resolve()
    episode_files = _find_episode_files(dataset_dir, args.max_episodes)
    if not episode_files:
        raise FileNotFoundError(f"No episode parquet files found under {dataset_dir}")

    episode_ranges: dict[str, list[list[int]]] = {}
    total_steps = 0
    kept_steps = 0
    total_idle_steps = 0

    for count, (episode_index, path) in enumerate(episode_files, start=1):
        action, state = _load_action_state(path)
        idle = _idle_mask(
            action,
            state,
            idle_l2_threshold=args.idle_l2_threshold,
            idle_arm_l2_threshold=args.idle_arm_l2_threshold,
            idle_gripper_l2_threshold=args.idle_gripper_l2_threshold,
        )
        ranges = _compute_keep_ranges(
            idle,
            min_idle_len=args.min_idle_len,
            min_non_idle_len=args.min_non_idle_len,
            filter_last_n_in_ranges=args.filter_last_n_in_ranges,
        )
        episode_ranges[str(episode_index)] = ranges
        total_steps += int(action.shape[0])
        kept_steps += sum(end - start for start, end in ranges)
        total_idle_steps += int(idle.sum())
        if count % 100 == 0:
            print(f"processed {count}/{len(episode_files)} episodes...", flush=True)

    payload = {
        "format": "imagewam_nonidle_ranges_v1",
        "dataset_dir": str(dataset_dir),
        "thresholds": {
            "idle_l2_threshold": args.idle_l2_threshold,
            "idle_arm_l2_threshold": args.idle_arm_l2_threshold,
            "idle_gripper_l2_threshold": args.idle_gripper_l2_threshold,
            "min_idle_len": args.min_idle_len,
            "min_non_idle_len": args.min_non_idle_len,
            "filter_last_n_in_ranges": args.filter_last_n_in_ranges,
        },
        "summary": {
            "episodes": len(episode_files),
            "total_steps": total_steps,
            "idle_steps": total_idle_steps,
            "idle_rate": total_idle_steps / max(total_steps, 1),
            "kept_steps": kept_steps,
            "kept_rate": kept_steps / max(total_steps, 1),
        },
        "episodes": episode_ranges,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote keep-ranges JSON: {args.output}")
    print(
        f"episodes={len(episode_files)} total_steps={total_steps} "
        f"idle_rate={payload['summary']['idle_rate'] * 100:.2f}% "
        f"kept_rate={payload['summary']['kept_rate'] * 100:.2f}%"
    )


if __name__ == "__main__":
    main()
