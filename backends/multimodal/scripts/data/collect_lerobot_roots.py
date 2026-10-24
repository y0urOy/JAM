#!/usr/bin/env python3
"""Collect LeRobot dataset root directories without scanning episode files.

A directory is treated as a LeRobot root when it contains `data/`, `meta/`,
and `videos/` subdirectories. Once such a root is found, the traversal does
not descend into it.
"""

from __future__ import annotations

import argparse
import os
from collections import deque
from pathlib import Path


def _has_child_dirs(path: Path, required: set[str]) -> bool:
    seen: set[str] = set()
    try:
        with os.scandir(path) as entries:
            for entry in entries:
                if entry.is_dir(follow_symlinks=False):
                    seen.add(entry.name)
                    if required.issubset(seen):
                        return True
    except OSError:
        return False
    return False


def collect_lerobot_roots(root: Path, *, max_depth: int | None = None) -> list[Path]:
    required = {"data", "meta", "videos"}
    roots: list[Path] = []
    queue: deque[tuple[Path, int]] = deque([(root, 0)])

    while queue:
        path, depth = queue.popleft()
        if _has_child_dirs(path, required):
            roots.append(path.resolve())
            continue
        if max_depth is not None and depth >= max_depth:
            continue

        try:
            with os.scandir(path) as entries:
                children = [
                    Path(entry.path)
                    for entry in entries
                    if entry.is_dir(follow_symlinks=False) and not entry.name.startswith(".")
                ]
        except OSError:
            continue

        for child in sorted(children):
            queue.append((child, depth + 1))

    return sorted(roots)


def _yaml_quote(path: Path) -> str:
    text = str(path)
    return "'" + text.replace("'", "''") + "'"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, help="Directory to traverse.")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output YAML file. Prints to stdout when omitted.",
    )
    parser.add_argument(
        "--key",
        default="dataset_dirs",
        help="Top-level YAML key to emit. Use an empty string to emit only list items.",
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=None,
        help="Optional maximum directory depth below root.",
    )
    args = parser.parse_args()

    roots = collect_lerobot_roots(args.root.expanduser(), max_depth=args.max_depth)
    lines: list[str] = []
    if args.key:
        lines.append(f"{args.key}:")
        lines.extend(f"  - {_yaml_quote(path)}" for path in roots)
    else:
        lines.extend(f"- {_yaml_quote(path)}" for path in roots)
    text = "\n".join(lines) + ("\n" if lines else "")

    if args.output is None:
        print(text, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
        print(f"Wrote {len(roots)} dataset roots to {args.output}")


if __name__ == "__main__":
    main()
