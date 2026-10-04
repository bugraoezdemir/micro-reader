"""One process-wide thread pool for decoding and reading.

Codecs (zlib, LZW, zstd, JPEG, ...) and file reads release the GIL, so a
single read can decode and read its pieces on all cores instead of only on
the caller's thread.
Shared by every scene, so the total number of decoding threads stays fixed
however many reads run at once.

Size: ``set_decode_threads(n)``, else the ``MICRO_READER_DECODE_THREADS``
environment variable, else the number of CPUs.  1 decodes on the caller's
thread (no pool).  Callers that already run many reads in parallel (like
eubi-bridge's writer) can lower it to avoid oversubscribing the CPU.
"""
from __future__ import annotations

import itertools
import os
import threading
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Iterable, Optional

_lock = threading.Lock()
_pool: Optional[ThreadPoolExecutor] = None
_threads: Optional[int] = None


def decode_threads() -> int:
    """The number of decoding threads in use."""
    global _threads
    if _threads is None:
        env = os.environ.get("MICRO_READER_DECODE_THREADS")
        _threads = max(1, int(env)) if env else (os.cpu_count() or 1)
    return _threads


def set_decode_threads(n: int) -> None:
    """Set the number of decoding threads (1: decode on the caller's thread)."""
    global _pool, _threads
    with _lock:
        old, _pool = _pool, None
        _threads = max(1, int(n))
    if old is not None:
        old.shutdown(wait=False)


def _get_pool() -> Optional[ThreadPoolExecutor]:
    global _pool
    if decode_threads() < 2:
        return None
    with _lock:
        if _pool is None:
            _pool = ThreadPoolExecutor(decode_threads(), thread_name_prefix="micro-reader-decode")
        return _pool


def run(tasks: Iterable[Callable[[], int]]) -> int:
    """Run *tasks* (each returning a count) on the pool; return the sum.

    The iterable is consumed lazily and at most two tasks per thread are in
    flight, so whatever the tasks hold (compressed bytes, decoded segments)
    stays bounded however many there are.  Preparation done by the iterable
    itself -- reading bytes from a file handle that is not thread-safe --
    happens on the caller's thread.
    """
    pool = _get_pool()
    if pool is None:
        return sum(task() for task in tasks)
    tasks = iter(tasks)
    first, second = next(tasks, None), next(tasks, None)
    if second is None:                  # nothing to spread: skip the hand-off
        return first() if first is not None else 0
    tasks = itertools.chain((first, second), tasks)
    window, pending, total = 2 * decode_threads(), deque(), 0
    try:
        for task in tasks:
            pending.append(pool.submit(task))
            if len(pending) >= window:
                total += pending.popleft().result()
        while pending:
            total += pending.popleft().result()
    finally:
        for future in pending:          # after an error: don't leave work running
            future.cancel()
    return total
