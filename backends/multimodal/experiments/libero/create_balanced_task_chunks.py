#!/usr/bin/env python3
"""Create deterministic LPT-balanced LIBERO task chunks for eval workers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")

# Per-task seconds measured from complete 50-trial evaluations. They are
# scheduling priors only: evaluation behavior and metrics do not depend on
# them. A previous result directory can override either profile.
OVIS_DURATION_PRIORS = {
    "libero_spatial": (397.4, 533.7, 469.2, 410.0, 832.7, 497.5, 516.1, 551.4, 475.8, 569.4),
    "libero_object": (614.4, 519.2, 496.2, 519.2, 557.4, 505.6, 659.3, 576.6, 615.2, 496.5),
    "libero_goal": (584.8, 391.3, 533.2, 765.8, 401.0, 532.1, 441.4, 337.0, 371.4, 715.4),
    "libero_10": (1177.9, 1082.4, 1014.1, 945.7, 1105.8, 829.3, 1023.8, 982.9, 1907.3, 1763.4),
}

QWEN35_DURATION_PRIORS = {
    "libero_spatial": (431.8, 631.0, 451.1, 476.0, 699.7, 498.6, 500.6, 570.6, 465.7, 568.2),
    "libero_object": (629.9, 487.2, 514.1, 542.0, 574.7, 574.9, 592.4, 577.1, 590.2, 494.8),
    "libero_goal": (565.6, 424.1, 488.1, 759.3, 395.0, 577.8, 430.5, 351.6, 403.5, 672.8),
    "libero_10": (1100.2, 962.1, 1099.7, 798.1, 998.5, 693.4, 869.3, 942.5, 1644.9, 1522.1),
}

DURATION_PROFILES = {
    "ovis": OVIS_DURATION_PRIORS,
    "qwen35": QWEN35_DURATION_PRIORS,
}


def _load_history(history_dir: Path | None) -> dict[tuple[str, int], float]:
    durations: dict[tuple[str, int], float] = {}
    if history_dir is None:
        return durations
    if not history_dir.is_dir():
        raise FileNotFoundError(f"History result directory not found: {history_dir}")

    for result_file in sorted(history_dir.rglob("*results.json")):
        try:
            payload = json.loads(result_file.read_text(encoding="utf-8"))
            suite = str(payload["task_suite"])
            task_id = int(payload["task_id"])
            duration = float(payload["duration"])
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            continue
        if suite in SUITES and 0 <= task_id < 10 and duration > 0:
            durations[(suite, task_id)] = duration
    return durations


def _task_weights(
    history_dir: Path | None,
    prior_profile: str,
) -> list[tuple[str, int, float]]:
    history = _load_history(history_dir)
    priors = DURATION_PROFILES[prior_profile]
    tasks = []
    for suite in SUITES:
        for task_id, prior in enumerate(priors[suite]):
            tasks.append((suite, task_id, history.get((suite, task_id), prior)))
    return tasks


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-ranks", type=int, required=True)
    parser.add_argument("--workers-per-rank", type=int, default=1)
    parser.add_argument("--history-dir", type=Path)
    parser.add_argument(
        "--prior-profile",
        choices=tuple(DURATION_PROFILES),
        default="ovis",
        help="Built-in task-duration profile used when --history-dir has no measurement.",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Only schedule libero_10 tasks 8 and 9 for a short worker comparison.",
    )
    parser.add_argument(
        "--task-list",
        type=Path,
        help="Only schedule the tasks listed in this file (one 'suite,task_id' per line), e.g. to rerun missing tasks.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.num_ranks <= 0:
        raise ValueError("--num-ranks must be positive")
    if args.workers_per_rank <= 0:
        raise ValueError("--workers-per-rank must be positive")

    tasks = _task_weights(args.history_dir, args.prior_profile)
    if args.smoke:
        tasks = [task for task in tasks if task[:2] in {("libero_10", 8), ("libero_10", 9)}]
    if args.task_list is not None:
        wanted = set()
        for line in args.task_list.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                suite, task_id = line.split(",")
                wanted.add((suite.strip(), int(task_id)))
        tasks = [task for task in tasks if task[:2] in wanted]
        if len(tasks) != len(wanted):
            raise ValueError(f"--task-list names unknown tasks: {wanted - {t[:2] for t in tasks}}")

    slots = [
        {"rank": rank, "worker": worker, "load": 0.0, "tasks": []}
        for rank in range(args.num_ranks)
        for worker in range(args.workers_per_rank)
    ]
    suite_order = {suite: idx for idx, suite in enumerate(SUITES)}
    for suite, task_id, duration in sorted(
        tasks,
        key=lambda item: (-item[2], suite_order[item[0]], item[1]),
    ):
        slot = min(slots, key=lambda item: (item["load"], item["rank"], item["worker"]))
        slot["tasks"].append((suite, task_id, duration))
        slot["load"] += duration

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "num_ranks": args.num_ranks,
        "workers_per_rank": args.workers_per_rank,
        "prior_profile": args.prior_profile,
        "history_dir": str(args.history_dir) if args.history_dir is not None else None,
        "slots": [],
    }
    for slot in slots:
        chunk_file = args.output_dir / f"rank{slot['rank']}_w{slot['worker']}_tasks.txt"
        chunk_file.write_text(
            "".join(f"{suite},{task_id}\n" for suite, task_id, _ in slot["tasks"]),
            encoding="utf-8",
        )
        manifest["slots"].append(
            {
                "rank": slot["rank"],
                "worker": slot["worker"],
                "estimated_seconds": round(slot["load"], 3),
                "tasks": [f"{suite},{task_id}" for suite, task_id, _ in slot["tasks"]],
            }
        )

    # A per-rank aggregate remains useful for inspection and compatibility with
    # tooling that predates multiple workers per rank.
    for rank in range(args.num_ranks):
        rank_tasks = []
        for slot in slots:
            if slot["rank"] == rank:
                rank_tasks.extend(slot["tasks"])
        (args.output_dir / f"rank{rank}_tasks.txt").write_text(
            "".join(f"{suite},{task_id}\n" for suite, task_id, _ in rank_tasks),
            encoding="utf-8",
        )

    manifest_path = args.output_dir / "task_balance_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    for slot in manifest["slots"]:
        print(
            f"rank={slot['rank']} worker={slot['worker']} "
            f"estimated={slot['estimated_seconds'] / 60:.1f}min tasks={len(slot['tasks'])}"
        )
    print(f"Wrote balanced chunks: {manifest_path}")


if __name__ == "__main__":
    main()
