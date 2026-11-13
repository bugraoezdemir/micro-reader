"""The native read engine (micro-reader-core, Workplan MR23), when installed.

A backend plans a read as before -- which stored segments, which of their rows,
columns and samples go where in the output array -- and hands the pieces it can
to a :class:`Batch` instead of making a pool task of each.  The engine then
reads, decodes and places all of them in one call, on its own threads without
Python's lock: one handle per file, positional reads, each segment decoded once
however many pieces need it, a cache of decoded segments of its own.  Whatever
a backend does not hand over (large segments decoded in parts, codecs the
engine lacks) runs on the decode pool as before, in the same call.

Off when micro-reader-core is not installed, with ``MICRO_READER_ENGINE=0``, or
after ``set_engine(False)``; the Python path is then the only one, and stays the
reference the tests compare the engine with.
"""
from __future__ import annotations

import itertools
import os
import threading

import numpy as np

from . import _memory, _pool

try:
    import micro_reader_core as _core
except ImportError:                                   # not installed: the Python path
    _core = None

#: the columns of a plan row, as the engine numbers them
COLUMNS = tuple(_core.PIECE_COLUMNS) if _core is not None else ()
NCOL = len(COLUMNS)

_lock = threading.Lock()
_engine = None
_enabled = None
_cache_limit = None
_codecs: dict = {}


def available() -> bool:
    """Whether micro-reader-core is installed (whatever ``set_engine`` says)."""
    return _core is not None


def enabled() -> bool:
    """Whether reads go through the engine."""
    global _enabled
    if _enabled is None:
        _enabled = _core is not None and os.environ.get("MICRO_READER_ENGINE", "1") != "0"
    return _enabled


def set_engine(on) -> None:
    """Read through the engine (True), on the Python path (False), or as the
    ``MICRO_READER_ENGINE`` environment variable says (None).  True without
    micro-reader-core installed is refused."""
    global _enabled
    if on and _core is None:
        raise RuntimeError("set_engine(True): micro-reader-core is not installed "
                           "(pip install micro-reader-core)")
    _enabled = None if on is None else bool(on)


def engine():
    """The process' engine (made on first use), or None when reads do not use it."""
    global _engine, _cache_limit
    if not enabled():
        return None
    if _engine is None:
        with _lock:
            if _engine is None:
                _engine = _core.Engine()
    from ._io import cache_limit
    limit = cache_limit()
    if limit != _cache_limit:                    # set_cache_limit() applies here too
        _engine.set_cache_limit(limit)
        _cache_limit = limit
    return _engine


def cached_bytes() -> int:
    """What the engine's cache holds (0 without an engine)."""
    return _engine.cached_bytes() if _engine is not None else 0


def codec(kind: str, itemsize: int, samples: int, **kwargs):
    """The engine's spec of a codec (made once per distinct spec)."""
    key = (kind, itemsize, samples, tuple(sorted(kwargs.items())))
    spec = _codecs.get(key)
    if spec is None:
        spec = _codecs.setdefault(key, _core.Codec(kind, itemsize, samples, **kwargs))
    return spec


class Batch:
    """The pieces of one read for the engine, from one image or many (a file
    set's planes): rows of ``COLUMNS`` plus the codecs and tables they name."""

    def __init__(self, engine_):
        self.engine = engine_
        self.rows: list = []
        self.codecs: list = []
        self._codec_at: dict = {}
        self.tables: list = []
        self._table_at: dict = {}
        #: the largest decoded segment, for the memory budget
        self.max_segment = 0

    def _codec(self, spec) -> int:
        at = self._codec_at.get(id(spec))
        if at is None:
            at = self._codec_at[id(spec)] = len(self.codecs)
            self.codecs.append(spec)
        return at

    def _table(self, data) -> int:
        if not data:
            return -1
        data = bytes(data)
        at = self._table_at.get(data)
        if at is None:
            at = self._table_at[data] = len(self.tables)
            self.tables.append(data)
        return at

    def add(self, handles, offset: int, count: int, spec, table, seg_shape: tuple,
            rows: tuple, cols: tuple, samples: tuple, dst: np.ndarray) -> None:
        """Rows, columns and samples (``(start, stop)`` each) of the segment of
        *seg_shape* (rows, cols) stored at *offset* in *handles*' file, into
        *dst* (rows, cols, samples: a view of the output array)."""
        r0, r1 = rows
        if r1 <= r0 or cols[1] <= cols[0] or samples[1] <= samples[0]:
            return
        s = dst.strides
        self.rows.append((handles.engine_file(self.engine), offset, count, self._codec(spec),
                          self._table(table), seg_shape[0], seg_shape[1], r0, r1, cols[0],
                          cols[1], samples[0], samples[1],
                          dst.__array_interface__["data"][0], s[0], s[1], s[2]))

    def run(self, out: np.ndarray) -> int:
        """Run the pieces into *out*, the array every destination is part of."""
        if not self.rows:
            return 0
        pieces = np.array(self.rows, dtype=np.int64)
        threads = _pool.decode_threads()
        # at most one decoded segment per thread at once, plus a raw read buffer each
        hold = min(threads, len(self.rows)) * (self.max_segment + (4 << 20))
        hold = min(hold, _memory.memory_limit())
        with _memory.reserve(hold, "segments decoded by the read engine"):
            return self.engine.read(pieces, self.codecs, self.tables, out, threads)


def run_plans(plans, out: np.ndarray) -> int:
    """Run the plans of images (``[(image, plan), ...]``, a backend's
    ``_plan_series``) into *out*: on the engine what each image hands it
    (``image._engine_split(batch, *plan)``), the rest as the images' pool
    tasks (``image._tasks``).  Without the engine, all of it on the pool."""
    eng = engine()
    if eng is None:
        return _pool.run(itertools.chain.from_iterable(
            image._tasks(*plan) for image, plan in plans))
    batch = Batch(eng)
    rest = []
    for image, plan in plans:
        split = getattr(image, "_engine_split", None)
        rest.append((image, split(batch, *plan) if split is not None else plan))
    try:
        n = batch.run(out)
    except ValueError as exc:                # the engine's message names file and offset
        from ._errors import CorruptFileError
        raise CorruptFileError(str(exc)) from exc
    except OSError as exc:
        from ._errors import CorruptFileError
        raise CorruptFileError(str(exc)) from exc
    return n + _pool.run(itertools.chain.from_iterable(
        image._tasks(*plan) for image, plan in rest))
