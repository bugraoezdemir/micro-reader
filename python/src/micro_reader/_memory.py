"""The memory budget for decoding (docs/PLAN.md, "memory-bounded reads of huge
compressed blocks").

Compressed data is decoded per block (a TIFF strip or tile, a CZI subblock,
an ND2 "lossless" frame, a whole PNG ...).

- A block too large for the decoded-block caches (``STREAM_THRESHOLD``) is
  streamed where its codec allows it: decoded from its start, keeping only
  the rows a read needs.  Decoding it whole would buy nothing -- it cannot
  be kept for the next read -- and costs the whole block for every read.
- Every other block is decoded whole, and reserves its size (``reserve``)
  while it is being decoded: the blocks in flight never exceed
  ``memory_limit()`` together, whichever threads decode them (a caller's
  own threads as well as the decode pool); a decode that does not fit
  waits until others finish.  A single block above the limit raises
  ``MemoryLimitError``.

Size: ``set_memory_limit(n)`` (bytes, or a string such as ``"8GB"``), else
the ``MICRO_READER_MEMORY_LIMIT`` environment variable, else a quarter of
the computer's memory.  The limit is per process: each worker process of a
caller such as eubi-bridge sets its own.
"""
from __future__ import annotations

import os
import re
import threading
from typing import Optional, Union

from ._errors import MicroReaderError

_lock = threading.Lock()
_limit: Optional[int] = None
#: when the computer's memory cannot be found
FALLBACK_LIMIT = 4 << 30
#: never below this: a budget smaller than one camera frame helps nobody
MINIMUM_LIMIT = 64 << 20
_UNITS = {"": 1, "b": 1, "k": 1 << 10, "kb": 1 << 10, "kib": 1 << 10, "m": 1 << 20,
          "mb": 1 << 20, "mib": 1 << 20, "g": 1 << 30, "gb": 1 << 30, "gib": 1 << 30,
          "t": 1 << 40, "tb": 1 << 40, "tib": 1 << 40}


class MemoryLimitError(MicroReaderError):
    """A block that must be decoded whole does not fit the memory budget."""


def parse_size(value: Union[int, float, str]) -> int:
    """Bytes from 4294967296, "4GB", "512 MB", "1.5g" (binary units)."""
    if isinstance(value, (int, float)):
        return int(value)
    match = re.fullmatch(r"\s*([0-9]*\.?[0-9]+)\s*([a-zA-Z]*)\s*", str(value))
    if not match or match.group(2).lower() not in _UNITS:
        raise ValueError(f"memory size {value!r}: a number of bytes or e.g. '8GB'")
    return int(float(match.group(1)) * _UNITS[match.group(2).lower()])


def physical_memory() -> Optional[int]:
    """The computer's memory in bytes, or None if it cannot be found."""
    try:
        if os.name == "nt":
            import ctypes

            class _Status(ctypes.Structure):
                _fields_ = [("length", ctypes.c_ulong), ("load", ctypes.c_ulong),
                            ("total", ctypes.c_ulonglong), ("free", ctypes.c_ulonglong),
                            ("page_total", ctypes.c_ulonglong), ("page_free", ctypes.c_ulonglong),
                            ("virtual_total", ctypes.c_ulonglong),
                            ("virtual_free", ctypes.c_ulonglong), ("extended", ctypes.c_ulonglong)]

            status = _Status()
            status.length = ctypes.sizeof(_Status)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
                return int(status.total)
            return None
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except (AttributeError, OSError, ValueError):
        return None


def memory_limit() -> int:
    """The decoding memory budget of this process, in bytes."""
    global _limit
    if _limit is None:
        env = os.environ.get("MICRO_READER_MEMORY_LIMIT")
        if env:
            _limit = max(MINIMUM_LIMIT, parse_size(env))
        else:
            total = physical_memory()
            _limit = max(MINIMUM_LIMIT, total // 4) if total else FALLBACK_LIMIT
    return _limit


def set_memory_limit(limit: Union[int, float, str, None]) -> None:
    """Set the decoding memory budget (bytes or e.g. "8GB"); None restores
    the default (environment variable, else a quarter of the memory)."""
    global _limit
    with _lock:
        _limit = None if limit is None else max(MINIMUM_LIMIT, parse_size(limit))


#: blocks larger than this are streamed where their codec allows: the
#: decoded-block caches keep blocks up to a quarter of their 64 MB
STREAM_THRESHOLD = 16 << 20

_reserved = 0
_peak_reserved = 0
_room = threading.Condition()


def should_stream(nbytes: int) -> bool:
    """Whether a block of *nbytes* is better streamed than decoded whole
    (when its codec can be streamed)."""
    return nbytes > STREAM_THRESHOLD


def fits(nbytes: int) -> bool:
    """Whether a block of *nbytes* can ever be decoded whole."""
    return nbytes <= memory_limit()


class reserve:
    """``with reserve(nbytes):`` -- hold *nbytes* of the budget while a block
    is decoded whole; waits while the blocks in flight leave too little."""

    def __init__(self, nbytes: int, what: str = "a block"):
        self.nbytes, self.what = int(nbytes), what

    def __enter__(self):
        global _reserved, _peak_reserved
        if not fits(self.nbytes):
            raise too_large(self.what, self.nbytes)
        with _room:
            while _reserved and _reserved + self.nbytes > memory_limit():
                _room.wait()
            _reserved += self.nbytes
            _peak_reserved = max(_peak_reserved, _reserved)
        return self

    def __exit__(self, *exc):
        global _reserved
        with _room:
            _reserved -= self.nbytes
            _room.notify_all()


def peak_reserved(reset: bool = False) -> int:
    """The most bytes reserved at once (for tests and diagnostics)."""
    global _peak_reserved
    with _room:
        peak = _peak_reserved
        if reset:
            _peak_reserved = _reserved
    return peak


def too_large(what: str, nbytes: int) -> MemoryLimitError:
    """The error for a block that cannot be decoded within the budget."""
    return MemoryLimitError(
        f"{what} must be decoded whole ({nbytes / 2**20:,.0f} MB), more than the memory "
        f"limit ({memory_limit() / 2**20:,.0f} MB).  Raise it with "
        "micro_reader.set_memory_limit(...) or MICRO_READER_MEMORY_LIMIT")
