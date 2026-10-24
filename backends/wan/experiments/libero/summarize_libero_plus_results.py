"""Summarize LIBERO-Plus per-task JSON results by robustness taxonomy."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import yaml


CATEGORY_ORDER = [
    "Camera Viewpoints",
    "Robot Initial States",
    "Language Instructions",
    "Light Conditions",
    "Background Textures",
    "Sensor Noise",
    "Objects Layout",
]
SUITE_ORDER = ["libero_10", "libero_goal", "libero_spatial", "libero_object"]
JAM_SAMPLE_SUITE_ORDER = [
    "libero_spatial",
    "libero_object",
    "libero_goal",
    "libero_10",
]
JAM_PAPER_RATIO = 0.15
JAM_PAPER_SEED = 42
JAM_PAPER_TASKS = 1506
JAM_PAPER_SUITE_COUNTS = {
    "libero_spatial": 361,
    "libero_object": 378,
    "libero_goal": 389,
    "libero_10": 378,
}
JAM_PAPER_CATEGORY_COUNTS = {
    "Camera Viewpoints": 240,
    "Robot Initial States": 197,
    "Language Instructions": 244,
    "Light Conditions": 161,
    "Background Textures": 186,
    "Sensor Noise": 258,
    "Objects Layout": 220,
}


def _expanded_path(value: str | os.PathLike[str]) -> Path:
    return Path(os.path.expanduser(os.path.expandvars(str(value)))).resolve()


def resolve_classification_path(explicit: str | None = None) -> Path:
    if explicit:
        return _expanded_path(explicit)
    env_path = os.environ.get("LIBERO_PLUS_CLASSIFICATION_PATH")
    if env_path:
        return _expanded_path(env_path)
    config_dir = _expanded_path(os.environ.get("LIBERO_CONFIG_PATH", ".libero_plus"))
    config_file = config_dir / "config.yaml"
    if config_file.is_file():
        with config_file.open("r", encoding="utf-8") as handle:
            config = yaml.safe_load(handle)
        return _expanded_path(config["benchmark_root"]) / "benchmark" / "task_classification.json"
    return Path("third_party/LIBERO-plus/libero/libero/benchmark/task_classification.json").resolve()


def load_classification(
    path: Path, category_index_mode: str = "correct"
) -> dict[tuple[str, int], dict[str, Any]]:
    if category_index_mode not in {"correct", "paper"}:
        raise ValueError(
            "category_index_mode must be 'correct' or 'paper', "
            f"got {category_index_mode!r}"
        )
    with path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    metadata: dict[tuple[str, int], dict[str, Any]] = {}
    for suite, entries in raw.items():
        ids = [int(entry["id"]) for entry in entries]
        if ids != list(range(1, len(entries) + 1)):
            raise ValueError(f"{suite} classification IDs must be contiguous and 1-based")
        for entry in entries:
            # Official classification IDs are 1-based; LIBERO benchmark IDs are 0-based.
            key = (str(suite), int(entry["id"]) - 1)
            if key in metadata:
                raise ValueError(f"Duplicate classification entry: {key}")
            metadata[key] = {
                "suite": str(suite),
                "task_id": key[1],
                "name": str(entry.get("name", "")),
                "category": str(entry["category"]),
                "difficulty_level": (
                    None
                    if entry.get("difficulty_level") is None
                    else int(entry["difficulty_level"])
                ),
            }

        if category_index_mode == "paper":
            # Reproduce the released ImageWAM category CSV exactly. Its category
            # join indexes the 1-based classification list with (task_id - 1),
            # where LIBERO task_id is 0-based. Python's -1 therefore maps task 0
            # to the final classification entry. This affects only the seven
            # category columns; task identity, suite scores, and overall AVG stay
            # tied to the real task_id.
            for task_id in range(len(entries)):
                metadata[(str(suite), task_id)]["category"] = str(
                    entries[task_id - 1]["category"]
                )
    return metadata


def load_task_manifest(tasks_file: Path) -> list[tuple[str, int]]:
    if not tasks_file.is_file():
        raise FileNotFoundError(f"Expected task manifest not found: {tasks_file}")
    tasks: list[tuple[str, int]] = []
    seen: set[tuple[str, int]] = set()
    with tasks_file.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                suite, task_id = line.split(",", 1)
                key = (suite.strip(), int(task_id))
            except ValueError as exc:
                raise ValueError(f"Invalid line in {tasks_file}:{line_number}: {line!r}") from exc
            if key in seen:
                raise ValueError(f"Duplicate task in {tasks_file}:{line_number}: {key}")
            seen.add(key)
            tasks.append(key)
    if not tasks:
        raise ValueError(f"Task manifest is empty: {tasks_file}")
    return tasks


def load_expected_tasks(output_dir: Path) -> list[tuple[str, int]]:
    return load_task_manifest(output_dir / "tasks.txt")


def imagewam_paper_tasks(classification_path: Path) -> list[tuple[str, int]]:
    """Return ImageWAM's fixed 15%/seed-42 paper task set."""
    metadata = load_classification(classification_path, category_index_mode="correct")
    selected: list[tuple[str, int]] = []
    for suite in JAM_SAMPLE_SUITE_ORDER:
        task_ids = sorted(task_id for candidate, task_id in metadata if candidate == suite)
        if task_ids != list(range(len(task_ids))):
            raise ValueError(f"{suite} task IDs must be contiguous and 0-based")
        count = math.ceil(len(task_ids) * JAM_PAPER_RATIO)
        rng = random.Random(f"{JAM_PAPER_SEED}:{suite}")
        sampled = set(rng.sample(task_ids, count))
        selected.extend((suite, task_id) for task_id in task_ids if task_id in sampled)
    if len(selected) != JAM_PAPER_TASKS:
        raise RuntimeError(
            f"ImageWAM paper manifest must contain {JAM_PAPER_TASKS} tasks, "
            f"got {len(selected)}"
        )
    suite_counts = Counter(suite for suite, _ in selected)
    if dict(suite_counts) != JAM_PAPER_SUITE_COUNTS:
        raise RuntimeError(
            "ImageWAM paper suite counts changed: "
            f"actual={dict(suite_counts)}, expected={JAM_PAPER_SUITE_COUNTS}"
        )
    return selected


def _load_result(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            result = json.load(handle)
        episodes = int(result["total_episodes"])
        successes = int(result["successes"])
        if episodes <= 0 or not 0 <= successes <= episodes:
            raise ValueError(
                f"invalid successes/episodes ({successes}/{episodes})"
            )
        result["task_suite"] = str(result["task_suite"])
        result["task_id"] = int(result["task_id"])
        result["total_episodes"] = episodes
        result["successes"] = successes
        result["duration"] = float(result.get("duration", 0.0))
        result["_source_file"] = str(path)
        return result
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid result JSON {path}: {exc}") from exc


def collect_results(output_dir: Path) -> tuple[dict[tuple[str, int], dict[str, Any]], list[str]]:
    candidates: dict[tuple[str, int], list[tuple[Path, dict[str, Any]]]] = defaultdict(list)
    for suite_dir in output_dir.iterdir():
        if not suite_dir.is_dir():
            continue
        for path in suite_dir.glob("*_results.json"):
            result = _load_result(path)
            key = (result["task_suite"], result["task_id"])
            candidates[key].append((path, result))

    chosen: dict[tuple[str, int], dict[str, Any]] = {}
    ignored_duplicates: list[str] = []
    for key, values in candidates.items():
        canonical_name = f"task{key[1]}_results.json"
        values.sort(
            key=lambda item: (
                item[0].name == canonical_name,
                item[0].stat().st_mtime_ns,
            ),
            reverse=True,
        )
        chosen[key] = values[0][1]
        ignored_duplicates.extend(str(path) for path, _ in values[1:])
    return chosen, ignored_duplicates


def group_statistics(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    materialized = list(rows)
    episodes = sum(int(row["total_episodes"]) for row in materialized)
    successes = sum(int(row["successes"]) for row in materialized)
    duration = sum(float(row.get("duration", 0.0)) for row in materialized)
    return {
        "completed_tasks": len(materialized),
        "total_episodes": episodes,
        "successes": successes,
        "success_rate": (100.0 * successes / episodes) if episodes else None,
        "total_duration_seconds": duration,
        "average_task_duration_seconds": duration / len(materialized) if materialized else None,
    }


def _ordered_group(
    rows: list[dict[str, Any]], field: str, order: list[Any] | None = None
) -> dict[str, dict[str, Any]]:
    grouped: dict[Any, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        value = row[field]
        if value is not None:
            grouped[value].append(row)
    keys = order if order is not None else sorted(grouped)
    return {str(key): group_statistics(grouped[key]) for key in keys if key in grouped}


def summarize(
    output_dir: Path,
    classification_path: Path,
    allow_partial: bool = False,
    *,
    protocol: str = "manifest",
    category_index_mode: str | None = None,
    output_file: Path | None = None,
) -> dict:
    if protocol not in {"manifest", "jam_multimodal-paper15"}:
        raise ValueError(f"Unknown LIBERO-Plus summary protocol: {protocol}")
    if category_index_mode is None:
        category_index_mode = "paper" if protocol == "jam_multimodal-paper15" else "correct"
    metadata = load_classification(
        classification_path, category_index_mode=category_index_mode
    )
    if protocol == "jam_multimodal-paper15":
        expected_list = imagewam_paper_tasks(classification_path)
        manifest_source = "ImageWAM paper: 15% per suite, seed 42"
    else:
        expected_list = load_expected_tasks(output_dir)
        manifest_source = str(output_dir / "tasks.txt")
    expected = set(expected_list)
    manifest_text = "".join(f"{suite},{task_id}\n" for suite, task_id in expected_list)
    manifest_sha256 = hashlib.sha256(manifest_text.encode("utf-8")).hexdigest()
    expected_suite_counts = Counter(suite for suite, _ in expected_list)
    expected_category_counts = Counter(metadata[key]["category"] for key in expected_list)
    if (
        protocol == "jam_multimodal-paper15"
        and category_index_mode == "paper"
        and dict(expected_category_counts) != JAM_PAPER_CATEGORY_COUNTS
    ):
        raise RuntimeError(
            "ImageWAM paper category counts changed: "
            f"actual={dict(expected_category_counts)}, "
            f"expected={JAM_PAPER_CATEGORY_COUNTS}"
        )
    raw_results, ignored_duplicates = collect_results(output_dir)
    actual = set(raw_results)
    missing = sorted(expected - actual)
    unexpected = sorted(actual - expected)
    if unexpected and protocol == "manifest":
        raise RuntimeError(
            f"Results directory contains {len(unexpected)} task(s) outside tasks.txt; "
            f"first entries: {unexpected[:10]}"
        )
    if missing and not allow_partial:
        raise RuntimeError(
            f"Incomplete LIBERO-Plus results: completed={len(expected & actual)}/{len(expected)}, "
            f"missing first entries={missing[:10]}. Use --allow-partial only for diagnostics."
        )

    rows: list[dict[str, Any]] = []
    for key in expected_list:
        if key not in raw_results:
            continue
        if key not in metadata:
            raise KeyError(f"No LIBERO-Plus classification metadata for task {key}")
        result = raw_results[key]
        meta = metadata[key]
        rows.append(
            {
                **meta,
                "task_description": str(result.get("task_description", "")),
                "total_episodes": int(result["total_episodes"]),
                "successes": int(result["successes"]),
                "success_rate": 100.0 * result["successes"] / result["total_episodes"],
                "duration": float(result.get("duration", 0.0)),
                "source_file": result["_source_file"],
            }
        )

    overall = group_statistics(rows)
    by_suite = _ordered_group(rows, "suite", SUITE_ORDER)
    by_category = _ordered_group(rows, "category", CATEGORY_ORDER)
    # Null difficulty tasks remain in rows, overall, suite, and category. They are
    # intentionally omitted only here, matching the official classification gap.
    by_difficulty = _ordered_group(rows, "difficulty_level", [1, 2, 3, 4, 5])
    category_rates = [
        stats["success_rate"]
        for stats in by_category.values()
        if stats["success_rate"] is not None
    ]
    macro = sum(category_rates) / len(category_rates) if category_rates else None

    summary = {
        "output_dir": str(output_dir),
        "classification_file": str(classification_path),
        "protocol": protocol,
        "protocol_parameters": (
            {
                "sample_ratio_per_suite": JAM_PAPER_RATIO,
                "sample_seed": JAM_PAPER_SEED,
                "num_trials_per_task": 1,
            }
            if protocol == "jam_multimodal-paper15"
            else None
        ),
        "task_manifest_source": manifest_source,
        "task_manifest_sha256": manifest_sha256,
        "category_index_mode": category_index_mode,
        "expected_tasks": len(expected),
        "expected_tasks_by_suite": dict(expected_suite_counts),
        "expected_tasks_by_category": dict(expected_category_counts),
        "completed_tasks": len(rows),
        "missing_tasks": [f"{suite}:{task_id}" for suite, task_id in missing],
        "found_result_tasks": len(actual),
        "ignored_extra_result_tasks": len(unexpected),
        "ignored_extra_result_task_examples": [
            f"{suite}:{task_id}" for suite, task_id in unexpected[:20]
        ],
        "ignored_duplicate_result_files": ignored_duplicates,
        "overall_micro_success_rate": overall["success_rate"],
        "overall_macro_category_success_rate": macro,
        "category_macro_denominator": len(category_rates),
        "overall": overall,
        "by_suite": by_suite,
        "by_category": by_category,
        "by_difficulty": by_difficulty,
        "per_task": rows,
    }
    if output_file is None:
        output_file = output_dir / "libero_plus_summary.json"
    elif not output_file.is_absolute():
        output_file = output_dir / output_file
    output_file = output_file.resolve()
    output_file.parent.mkdir(parents=True, exist_ok=True)
    temp_file = output_file.with_suffix(output_file.suffix + ".tmp")
    with temp_file.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
        handle.write("\n")
    os.replace(temp_file, output_file)
    return summary


def _format_rate(value: float | None) -> str:
    return "—" if value is None else f"{value:.2f}"


def write_paper_table(summary: dict[str, Any], output_file: Path, label: str) -> None:
    suite_keys = ["libero_spatial", "libero_object", "libero_goal", "libero_10"]
    suite_rates = [summary["by_suite"].get(key, {}).get("success_rate") for key in suite_keys]
    category_rates = [
        summary["by_category"].get(key, {}).get("success_rate") for key in CATEGORY_ORDER
    ]
    overall = summary["overall_micro_success_rate"]
    lines = [
        "| Variant | Spatial | Object | Goal | Long | AVG |",
        "|---|---:|---:|---:|---:|---:|",
        f"| {label} | "
        + " | ".join(_format_rate(rate) for rate in suite_rates)
        + f" | {_format_rate(overall)} |",
        "",
        "| Variant | Camera | Robot | Language | Light | Background | Noise | Layout | AVG |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        f"| {label} | "
        + " | ".join(_format_rate(rate) for rate in category_rates)
        + f" | {_format_rate(overall)} |",
        "",
        f"Protocol: {summary['protocol']}",
        f"Tasks: {summary['completed_tasks']}/{summary['expected_tasks']}",
        f"Category index mode: {summary['category_index_mode']}",
        "AVG: task-weighted overall success rate",
    ]
    output_file.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_file.with_suffix(output_file.suffix + ".tmp")
    temporary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    os.replace(temporary, output_file)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--classification-file")
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument(
        "--protocol",
        choices=["manifest", "jam_multimodal-paper15"],
        default="manifest",
        help=(
            "manifest: summarize tasks.txt strictly; jam_multimodal-paper15: select the fixed "
            "15%%/seed-42 paper subset and ignore result files outside that subset"
        ),
    )
    parser.add_argument(
        "--category-index-mode",
        choices=["correct", "paper"],
        help=(
            "correct uses classification[task_id]; paper reproduces the released "
            "ImageWAM CSV's classification[task_id-1] category join"
        ),
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        default=Path("libero_plus_summary.json"),
        help="Summary JSON path (relative paths are resolved under --output-dir)",
    )
    parser.add_argument(
        "--table-file",
        type=Path,
        help="Markdown table path; defaults beside --output-file with _table.md suffix",
    )
    parser.add_argument("--label", help="Variant label used in the Markdown table")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    classification_path = resolve_classification_path(args.classification_file)
    summary_path = args.output_file
    if not summary_path.is_absolute():
        summary_path = output_dir / summary_path
    summary_path = summary_path.resolve()
    summary = summarize(
        output_dir,
        classification_path,
        allow_partial=args.allow_partial,
        protocol=args.protocol,
        category_index_mode=args.category_index_mode,
        output_file=summary_path,
    )
    if args.table_file is None:
        table_name = summary_path.stem
        if table_name.endswith("_summary"):
            table_name = table_name[: -len("_summary")]
        table_path = summary_path.with_name(f"{table_name}_table.md")
    else:
        table_path = args.table_file
        if not table_path.is_absolute():
            table_path = output_dir / table_path
        table_path = table_path.resolve()
    write_paper_table(summary, table_path, args.label or output_dir.name)
    micro = summary["overall_micro_success_rate"]
    macro = summary["overall_macro_category_success_rate"]
    print(
        f"LIBERO-Plus summary: completed={summary['completed_tasks']}/"
        f"{summary['expected_tasks']} micro={_format_rate(micro)}% "
        f"category_macro={_format_rate(macro)}%"
    )
    print(f"Saved: {summary_path}")
    print(f"Saved: {table_path}")


if __name__ == "__main__":
    main()

# Historical constant aliases; serialized values are unchanged.
IMAGEWAM_SAMPLE_SUITE_ORDER = JAM_SAMPLE_SUITE_ORDER
IMAGEWAM_PAPER_RATIO = JAM_PAPER_RATIO
IMAGEWAM_PAPER_SEED = JAM_PAPER_SEED
IMAGEWAM_PAPER_TASKS = JAM_PAPER_TASKS
IMAGEWAM_PAPER_SUITE_COUNTS = JAM_PAPER_SUITE_COUNTS
IMAGEWAM_PAPER_CATEGORY_COUNTS = JAM_PAPER_CATEGORY_COUNTS
