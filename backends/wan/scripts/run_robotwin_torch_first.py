#!/usr/bin/env python3
"""Run RoboTwin's evaluation entry after initializing CUDA before SAPIEN.

Some NVIDIA driver versions segfault when SAPIEN/Vulkan is loaded before
PyTorch/CUDA in the same process.  This wrapper establishes the safe library
initialization order and executes the evaluation entry with its original
arguments.  It uses ``os._exit`` after both successful and failed attempts:
CUDA/Vulkan interpreter teardown can otherwise hang after rendering and
keep the calling evaluation process waiting for the subprocess to exit.
"""

from __future__ import annotations

import os
from pathlib import Path
import runpy
import sys
import traceback


def _hard_exit(code: int) -> None:
    sys.stdout.flush()
    sys.stderr.flush()
    # Never run Python/CUDA/Vulkan finalizers.  Successful renderer teardown is
    # not required because every evaluation attempt is an isolated process.
    os._exit(code)


def _execute_entry(entry: Path) -> None:
    try:
        runpy.run_path(str(entry), run_name="__main__")
    except SystemExit as error:
        code = error.code
        if code is None:
            _hard_exit(0)
        if isinstance(code, int):
            _hard_exit(code)
        print(code, file=sys.stderr, flush=True)
        _hard_exit(1)
    except BaseException:
        traceback.print_exc()
        _hard_exit(1)
    else:
        _hard_exit(0)


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit("usage: run_robotwin_torch_first.py ENTRY [ARGS ...]")

    entry = Path(sys.argv[1]).expanduser().resolve()
    if not entry.is_file():
        raise FileNotFoundError(entry)

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("RoboTwin evaluation requires a visible CUDA device")
    torch.cuda.init()
    print(
        f"ROBOTWIN_TORCH_FIRST_READY device={torch.cuda.get_device_name(0)} entry={entry}",
        flush=True,
    )

    sys.path.insert(0, str(entry.parent))
    sys.argv = [str(entry), *sys.argv[2:]]
    _execute_entry(entry)


if __name__ == "__main__":
    main()
