#!/usr/bin/env python3
"""Download files under one Hugging Face repo folder without allow_patterns."""

from __future__ import annotations

import argparse
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RemoteFile:
    path: str
    size: int | None


@dataclass(frozen=True)
class DownloadPlan:
    remote: RemoteFile
    local_path: Path
    status: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Recursively list one folder in a Hugging Face repo and download the files "
            "one by one. This avoids allow_patterns over very large dataset repos."
        )
    )
    parser.add_argument("--repo-id", required=True, help="Hugging Face repo id.")
    parser.add_argument(
        "--repo-type",
        default="dataset",
        choices=["dataset", "model", "space"],
        help="Hugging Face repo type.",
    )
    parser.add_argument(
        "--remote-folder",
        required=True,
        help="Folder path inside the repo, for example sim_updated/foo.",
    )
    parser.add_argument(
        "--local-dir",
        required=True,
        type=Path,
        help="Local root. Remote paths are preserved under this directory.",
    )
    parser.add_argument("--revision", default=None, help="Optional repo revision.")
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Parallel downloads.",
    )
    parser.add_argument(
        "--regex",
        default=None,
        help="Optional regex filter applied to full remote file paths.",
    )
    parser.add_argument(
        "--suffix",
        action="append",
        default=[],
        help="Optional suffix filter, repeatable. Example: --suffix .tar.gz --suffix .json",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Only download the first N matched files. Useful for smoke tests.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the plan without downloading.",
    )
    parser.add_argument(
        "--include-skipped",
        action="store_true",
        help="Also print files that are already complete locally.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Download even if a same-size local file already exists.",
    )
    return parser.parse_args()


def normalize_folder(path: str) -> str:
    return path.strip("/")


def remote_file_size(item: object) -> int | None:
    size = getattr(item, "size", None)
    if isinstance(size, int):
        return size

    lfs = getattr(item, "lfs", None)
    if isinstance(lfs, dict) and isinstance(lfs.get("size"), int):
        return lfs["size"]

    return None


def list_remote_files(
    repo_id: str,
    repo_type: str,
    remote_folder: str,
    revision: str | None,
) -> list[RemoteFile]:
    from huggingface_hub import HfApi

    api = HfApi()
    items = api.list_repo_tree(
        repo_id=repo_id,
        repo_type=repo_type,
        path_in_repo=remote_folder,
        recursive=True,
        revision=revision,
    )
    files = [
        RemoteFile(path=item.path, size=remote_file_size(item))
        for item in items
        if item.__class__.__name__ == "RepoFile"
    ]
    return sorted(files, key=lambda item: item.path)


def filter_remote_files(
    files: list[RemoteFile],
    regex: str | None,
    suffixes: list[str],
    max_files: int | None,
) -> list[RemoteFile]:
    pattern = re.compile(regex) if regex else None
    filtered: list[RemoteFile] = []
    for item in files:
        if pattern and not pattern.search(item.path):
            continue
        if suffixes and not any(item.path.endswith(suffix) for suffix in suffixes):
            continue
        filtered.append(item)
        if max_files is not None and len(filtered) >= max_files:
            break
    return filtered


def classify_file(remote: RemoteFile, local_dir: Path, force: bool) -> DownloadPlan:
    local_path = local_dir / remote.path
    if force:
        return DownloadPlan(remote, local_path, "download: forced")

    if local_path.is_file():
        local_size = local_path.stat().st_size
        if remote.size is None:
            return DownloadPlan(remote, local_path, "skip: local file exists")
        if local_size == remote.size:
            return DownloadPlan(remote, local_path, "skip: local file already complete")
        return DownloadPlan(
            remote,
            local_path,
            f"download: size mismatch local={local_size} remote={remote.size}",
        )

    return DownloadPlan(remote, local_path, "download: missing")


def should_download(plan: DownloadPlan) -> bool:
    return plan.status.startswith("download:")


def download_one(
    repo_id: str,
    repo_type: str,
    revision: str | None,
    local_dir: Path,
    plan: DownloadPlan,
) -> tuple[DownloadPlan, str]:
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(
        repo_id=repo_id,
        repo_type=repo_type,
        revision=revision,
        filename=plan.remote.path,
        local_dir=str(local_dir),
    )
    return plan, path


def print_plan(plans: list[DownloadPlan], include_skipped: bool) -> None:
    for plan in plans:
        if not should_download(plan) and not include_skipped:
            continue
        size_text = "unknown-size" if plan.remote.size is None else str(plan.remote.size)
        print(f"[{plan.status}] {plan.remote.path} ({size_text})")


def main() -> None:
    args = parse_args()
    remote_folder = normalize_folder(args.remote_folder)
    local_dir = args.local_dir.expanduser().resolve()

    print(f"Listing {args.repo_type} repo {args.repo_id}:{remote_folder}", flush=True)
    remote_files = list_remote_files(
        repo_id=args.repo_id,
        repo_type=args.repo_type,
        remote_folder=remote_folder,
        revision=args.revision,
    )
    matched_files = filter_remote_files(
        remote_files,
        regex=args.regex,
        suffixes=args.suffix,
        max_files=args.max_files,
    )
    plans = [classify_file(item, local_dir, args.force) for item in matched_files]

    print(f"Remote files under folder: {len(remote_files)}")
    print(f"Matched files: {len(matched_files)}")
    print(f"Need download: {sum(1 for plan in plans if should_download(plan))}")
    print(f"Already complete/existing: {sum(1 for plan in plans if not should_download(plan))}")
    print_plan(plans, include_skipped=args.include_skipped or args.dry_run)

    if args.dry_run:
        return

    local_dir.mkdir(parents=True, exist_ok=True)
    to_download = [plan for plan in plans if should_download(plan)]
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [
            executor.submit(
                download_one,
                args.repo_id,
                args.repo_type,
                args.revision,
                local_dir,
                plan,
            )
            for plan in to_download
        ]
        for future in as_completed(futures):
            plan, path = future.result()
            print(f"[done] {plan.remote.path} -> {path}", flush=True)


if __name__ == "__main__":
    main()
