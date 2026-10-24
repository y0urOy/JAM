"""Memory tuning + on-demand reclaim utilities.

Why this exists
---------------
PyTorch DataLoader workers under fork accumulate a lot of small Python-object
allocations per __getitem__ (HF datasets row dicts, list comprehensions,
intermediate torch tensors via numpy, pyav decoded buffers, ...). After the
caller drops the batch, those allocations end up as free chunks inside glibc
malloc arenas. glibc only returns memory to the OS when the *top* chunk grows
past `M_TRIM_THRESHOLD` (default 128 KB) and only mmap's allocations that are
larger than `M_MMAP_THRESHOLD` (dynamic 128KB-32MB).

Under our workload that means each worker's RSS tends to ratchet upwards even
though the live working set stays roughly the same. Occasionally glibc auto-
trims and the RSS drops on its own -- which is the fall-back the user
observed; we just want to trigger it deterministically.

Tools provided here:
  - `malloc_trim(0)`              : force glibc to return free pages now
  - `mallopt(opt, val)`           : tune glibc thresholds at runtime
  - `trim_now(do_gc=True)`        : `gc.collect()` then `malloc_trim(0)`
  - `apply_glibc_tuning_from_env`: read JAM_M_TRIM_THRESHOLD /
                                    JAM_M_MMAP_THRESHOLD env vars and
                                    apply via mallopt
  - `install_periodic_trim`      : helper to call trim_now every N invocations
"""

from __future__ import annotations

import ctypes
import gc
import os
from typing import Optional

# glibc mallopt option codes (see /usr/include/malloc.h)
M_TRIM_THRESHOLD = -1
M_TOP_PAD = -2
M_MMAP_THRESHOLD = -3
M_MMAP_MAX = -4
M_ARENA_TEST = -7
M_ARENA_MAX = -8


_libc: Optional[ctypes.CDLL] = None
_libc_resolved = False


def _get_libc() -> Optional[ctypes.CDLL]:
    global _libc, _libc_resolved
    if _libc_resolved:
        return _libc
    _libc_resolved = True
    try:
        _libc = ctypes.CDLL("libc.so.6", use_errno=True)
    except OSError:
        _libc = None
    return _libc


def malloc_trim(pad: int = 0) -> bool:
    """Force glibc to return free top-of-heap memory back to the OS.

    Returns True if any memory was actually released.
    """
    libc = _get_libc()
    if libc is None:
        return False
    try:
        return bool(libc.malloc_trim(pad))
    except (AttributeError, OSError):
        return False


def mallopt(option: int, value: int) -> bool:
    libc = _get_libc()
    if libc is None:
        return False
    try:
        return bool(libc.mallopt(option, value))
    except (AttributeError, OSError):
        return False


def trim_now(do_gc: bool = True) -> None:
    """gc.collect() then malloc_trim(0). Cheap to call every few hundred samples,
    expensive to call every sample.
    """
    if do_gc:
        gc.collect()
    malloc_trim(0)


def apply_glibc_tuning_from_env(verbose: bool = False) -> dict:
    """Apply glibc mallopt tuning from environment variables.

    Recognized vars (all bytes; 0 / unset = leave default):
      - JAM_M_TRIM_THRESHOLD : auto-trim top chunk when free > this
                                   (default glibc: 131072 = 128 KB).
                                   Smaller -> more aggressive auto-return.
                                   Try 65536 or 16384 for very tight RSS.
      - JAM_M_MMAP_THRESHOLD : allocations >= this go through mmap and
                                   are returned to OS on free() (no arena
                                   fragmentation). Default glibc: dynamic
                                   128KB..32MB. Set e.g. 65536 to force
                                   most large allocs to mmap.
      - JAM_M_TOP_PAD        : pad bytes above top chunk on sbrk(); set
                                   small (e.g. 0) to keep heap tight.
      - JAM_M_ARENA_MAX      : equivalent to MALLOC_ARENA_MAX env var
                                   when set after libc init.

    Returns a dict {opt_name: applied_bool}.
    """
    spec = [
        ("JAM_M_TRIM_THRESHOLD", M_TRIM_THRESHOLD, "M_TRIM_THRESHOLD"),
        ("JAM_M_MMAP_THRESHOLD", M_MMAP_THRESHOLD, "M_MMAP_THRESHOLD"),
        ("JAM_M_TOP_PAD", M_TOP_PAD, "M_TOP_PAD"),
        ("JAM_M_ARENA_MAX", M_ARENA_MAX, "M_ARENA_MAX"),
    ]
    applied = {}
    for env, opt, name in spec:
        raw = os.environ.get(env)
        if raw is None or raw == "":
            continue
        try:
            val = int(raw)
        except ValueError:
            if verbose:
                print(f"[mem_tools] ignored {env}={raw!r} (not int)")
            continue
        ok = mallopt(opt, val)
        applied[name] = ok
        if verbose:
            print(f"[mem_tools] mallopt({name}, {val}) -> {ok}")
    return applied


class PeriodicTrim:
    """Call `trim_now` every `every` invocations of `tick`.

    Designed to be cheap when disabled (`every<=0`) so it can sit in a hot loop.
    """

    __slots__ = ("every", "_n", "do_gc")

    def __init__(self, every: int = 0, do_gc: bool = True) -> None:
        self.every = int(every)
        self._n = 0
        self.do_gc = bool(do_gc)

    def tick(self) -> bool:
        if self.every <= 0:
            return False
        self._n += 1
        if self._n % self.every == 0:
            trim_now(do_gc=self.do_gc)
            return True
        return False
