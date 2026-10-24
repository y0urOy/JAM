"""Paired comparison of two LIBERO-Plus evaluation result directories."""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable


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


def load_summary(directory: Path) -> dict:
    path = directory / "libero_plus_summary.json"
    if not path.is_file():
        raise FileNotFoundError(f"LIBERO-Plus summary not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def task_map(summary: dict) -> dict[tuple[str, int], dict[str, Any]]:
    mapped: dict[tuple[str, int], dict[str, Any]] = {}
    for row in summary.get("per_task", []):
        key = (str(row["suite"]), int(row["task_id"]))
        if key in mapped:
            raise ValueError(f"Duplicate task in summary: {key}")
        mapped[key] = row
    return mapped


def paired_metrics(pairs: list[tuple[dict, dict]]) -> dict[str, Any]:
    baseline_episodes = sum(int(b["total_episodes"]) for b, _ in pairs)
    champion_episodes = sum(int(c["total_episodes"]) for _, c in pairs)
    baseline_successes = sum(int(b["successes"]) for b, _ in pairs)
    champion_successes = sum(int(c["successes"]) for _, c in pairs)
    baseline_rate = (
        100.0 * baseline_successes / baseline_episodes if baseline_episodes else None
    )
    champion_rate = (
        100.0 * champion_successes / champion_episodes if champion_episodes else None
    )
    champion_wins = sum(float(c["success_rate"]) > float(b["success_rate"]) for b, c in pairs)
    baseline_wins = sum(float(c["success_rate"]) < float(b["success_rate"]) for b, c in pairs)
    ties = len(pairs) - champion_wins - baseline_wins
    return {
        "matched_tasks": len(pairs),
        "baseline_total_episodes": baseline_episodes,
        "champion_total_episodes": champion_episodes,
        "baseline_successes": baseline_successes,
        "champion_successes": champion_successes,
        "baseline_success_rate": baseline_rate,
        "champion_success_rate": champion_rate,
        "delta_percentage_points": (
            champion_rate - baseline_rate
            if baseline_rate is not None and champion_rate is not None
            else None
        ),
        "champion_wins": champion_wins,
        "ties": ties,
        "baseline_wins": baseline_wins,
    }


def grouped_metrics(
    pairs: list[tuple[dict, dict]],
    key_fn: Callable[[dict], Any],
    order: list[Any] | None = None,
    *,
    omit_none: bool = False,
) -> dict[str, dict[str, Any]]:
    grouped: dict[Any, list[tuple[dict, dict]]] = defaultdict(list)
    for pair in pairs:
        value = key_fn(pair[0])
        if omit_none and value is None:
            continue
        grouped[value].append(pair)
    keys = order if order is not None else sorted(grouped)
    return {str(key): paired_metrics(grouped[key]) for key in keys if key in grouped}


def category_macro(by_category: dict[str, dict[str, Any]]) -> dict[str, Any]:
    rows = [value for value in by_category.values() if value["matched_tasks"] > 0]
    if not rows:
        return {
            "categories": 0,
            "baseline_success_rate": None,
            "champion_success_rate": None,
            "delta_percentage_points": None,
        }
    baseline = sum(row["baseline_success_rate"] for row in rows) / len(rows)
    champion = sum(row["champion_success_rate"] for row in rows) / len(rows)
    return {
        "categories": len(rows),
        "baseline_success_rate": baseline,
        "champion_success_rate": champion,
        "delta_percentage_points": champion - baseline,
    }


def _atomic_json(path: Path, payload: dict) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
    os.replace(temp, path)


def compare(
    baseline_dir: Path,
    champion_dir: Path,
    output_dir: Path,
    baseline_label: str,
    champion_label: str,
) -> dict:
    baseline_summary = load_summary(baseline_dir)
    champion_summary = load_summary(champion_dir)
    baseline = task_map(baseline_summary)
    champion = task_map(champion_summary)
    baseline_keys = set(baseline)
    champion_keys = set(champion)
    matched_keys = sorted(baseline_keys & champion_keys)
    baseline_only = sorted(baseline_keys - champion_keys)
    champion_only = sorted(champion_keys - baseline_keys)
    pairs = [(baseline[key], champion[key]) for key in matched_keys]

    for key, (base_row, champ_row) in zip(matched_keys, pairs):
        for field in ("category", "difficulty_level"):
            if base_row.get(field) != champ_row.get(field):
                raise ValueError(
                    f"Classification mismatch for {key}, field={field}: "
                    f"{base_row.get(field)!r} != {champ_row.get(field)!r}"
                )

    micro = paired_metrics(pairs)
    baseline_is_full = (
        int(baseline_summary.get("expected_tasks", -1)) == 10030
        and int(baseline_summary.get("completed_tasks", -1)) == 10030
    )
    champion_is_full = (
        int(champion_summary.get("expected_tasks", -1)) == 10030
        and int(champion_summary.get("completed_tasks", -1)) == 10030
    )
    if baseline_is_full and champion_is_full and len(matched_keys) != 10030:
        raise RuntimeError(
            f"Two full evaluations must pair all 10030 tasks, got {len(matched_keys)}"
        )
    by_suite = grouped_metrics(pairs, lambda row: row["suite"], SUITE_ORDER)
    by_category = grouped_metrics(pairs, lambda row: row["category"], CATEGORY_ORDER)
    by_difficulty = grouped_metrics(
        pairs,
        lambda row: row.get("difficulty_level"),
        [1, 2, 3, 4, 5],
        omit_none=True,
    )
    macro = category_macro(by_category)
    comparison = {
        "baseline_label": baseline_label,
        "champion_label": champion_label,
        "baseline_dir": str(baseline_dir),
        "champion_dir": str(champion_dir),
        "matched_tasks": len(matched_keys),
        "baseline_only_tasks": len(baseline_only),
        "champion_only_tasks": len(champion_only),
        "champion_wins": micro["champion_wins"],
        "ties": micro["ties"],
        "baseline_wins": micro["baseline_wins"],
        "baseline_only_task_keys": [f"{suite}:{task_id}" for suite, task_id in baseline_only],
        "champion_only_task_keys": [f"{suite}:{task_id}" for suite, task_id in champion_only],
        "micro": micro,
        "category_macro": macro,
        "by_suite": by_suite,
        "by_category": by_category,
        "by_difficulty": by_difficulty,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    _atomic_json(output_dir / "comparison.json", comparison)

    per_task_path = output_dir / "per_task_comparison.csv"
    with per_task_path.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = [
            "suite",
            "task_id",
            "name",
            "category",
            "difficulty_level",
            "baseline_successes",
            "baseline_total_episodes",
            "baseline_success_rate",
            "champion_successes",
            "champion_total_episodes",
            "champion_success_rate",
            "delta_percentage_points",
            "outcome",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for key, (base_row, champ_row) in zip(matched_keys, pairs):
            delta = float(champ_row["success_rate"]) - float(base_row["success_rate"])
            outcome = "champion_win" if delta > 0 else "baseline_win" if delta < 0 else "tie"
            writer.writerow(
                {
                    "suite": key[0],
                    "task_id": key[1],
                    "name": base_row.get("name", ""),
                    "category": base_row["category"],
                    "difficulty_level": base_row.get("difficulty_level"),
                    "baseline_successes": base_row["successes"],
                    "baseline_total_episodes": base_row["total_episodes"],
                    "baseline_success_rate": base_row["success_rate"],
                    "champion_successes": champ_row["successes"],
                    "champion_total_episodes": champ_row["total_episodes"],
                    "champion_success_rate": champ_row["success_rate"],
                    "delta_percentage_points": delta,
                    "outcome": outcome,
                }
            )

    group_path = output_dir / "group_comparison.csv"
    with group_path.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = [
            "group_type",
            "group",
            "matched_tasks",
            "baseline_success_rate",
            "champion_success_rate",
            "delta_percentage_points",
            "champion_wins",
            "ties",
            "baseline_wins",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()

        def write_group(group_type: str, group: str, metrics: dict[str, Any]) -> None:
            writer.writerow(
                {
                    "group_type": group_type,
                    "group": group,
                    **{key: metrics.get(key) for key in fieldnames[2:]},
                }
            )

        write_group("overall", "micro", micro)
        for group, metrics in by_suite.items():
            write_group("suite", group, metrics)
        for group, metrics in by_category.items():
            write_group("category", group, metrics)
        for group, metrics in by_difficulty.items():
            write_group("difficulty", group, metrics)

    markdown = [
        "# LIBERO-Plus paired comparison",
        "",
        f"- Baseline: `{baseline_label}`",
        f"- Champion: `{champion_label}`",
        f"- Matched tasks: {len(matched_keys)}",
        f"- Baseline-only tasks: {len(baseline_only)}",
        f"- Champion-only tasks: {len(champion_only)}",
        "",
        "## Overall",
        "",
        "| Metric | Baseline | Champion | Delta (pp) |",
        "|---|---:|---:|---:|",
        (
            f"| Micro success rate | {micro['baseline_success_rate']:.4f}% | "
            f"{micro['champion_success_rate']:.4f}% | {micro['delta_percentage_points']:+.4f} |"
        ),
        (
            f"| 7-category macro | {macro['baseline_success_rate']:.4f}% | "
            f"{macro['champion_success_rate']:.4f}% | {macro['delta_percentage_points']:+.4f} |"
        ),
        "",
        (
            f"Task outcomes: champion wins **{micro['champion_wins']}**, "
            f"ties **{micro['ties']}**, baseline wins **{micro['baseline_wins']}**."
        ),
    ]

    def add_table(title: str, groups: dict[str, dict[str, Any]]) -> None:
        markdown.extend(
            [
                "",
                f"## {title}",
                "",
                "| Group | Tasks | Baseline | Champion | Delta (pp) | W/T/L |",
                "|---|---:|---:|---:|---:|---:|",
            ]
        )
        for group, metrics in groups.items():
            markdown.append(
                f"| {group} | {metrics['matched_tasks']} | "
                f"{metrics['baseline_success_rate']:.4f}% | "
                f"{metrics['champion_success_rate']:.4f}% | "
                f"{metrics['delta_percentage_points']:+.4f} | "
                f"{metrics['champion_wins']}/{metrics['ties']}/{metrics['baseline_wins']} |"
            )

    add_table("By suite", by_suite)
    add_table("By category", by_category)
    add_table("By difficulty", by_difficulty)
    (output_dir / "comparison_summary.md").write_text("\n".join(markdown) + "\n", encoding="utf-8")
    return comparison


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-dir", required=True, type=Path)
    parser.add_argument("--champion-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--baseline-label", default="baseline")
    parser.add_argument("--champion-label", default="champion")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = compare(
        baseline_dir=args.baseline_dir.resolve(),
        champion_dir=args.champion_dir.resolve(),
        output_dir=args.output_dir.resolve(),
        baseline_label=args.baseline_label,
        champion_label=args.champion_label,
    )
    print(
        f"Paired comparison: matched={result['matched_tasks']} "
        f"baseline_only={result['baseline_only_tasks']} "
        f"champion_only={result['champion_only_tasks']}"
    )
    print(f"Saved: {args.output_dir.resolve() / 'comparison_summary.md'}")


if __name__ == "__main__":
    main()
