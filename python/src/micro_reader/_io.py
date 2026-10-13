"""Pieces every backend that reads segments on the decode pool needs."""
from __future__ import annotations

import builtins
import threading
from collections import OrderedDict

import numpy as np

from ._errors import CorruptFileError


class ThreadHandles:
    """One read-only file handle per thread for one file (seek + read on a
    shared handle is not thread-safe), kept open until :meth:`close`."""

    def __init__(self, path: str):
        self.path = path
        self._local = threading.local()
        self._lock = threading.Lock()
        self._all: list = []

    def get(self):
        fh = getattr(self._local, "fh", None)
        if fh is None:
            fh = self._local.fh = builtins.open(self.path, "rb", buffering=0)
            with self._lock:
                self._all.append(fh)
        return fh

    def read(self, offset: int, size: int) -> bytes:
        fh = self.get()
        fh.seek(offset)
        data = fh.read(size)
        if len(data) != size:
            raise CorruptFileError(f"{self.path!r}: data at byte {offset} ends early "
                                   "(truncated file?)")
        return data

    def close(self) -> None:
        with self._lock:
            handles, self._all = self._all, []
        for fh in handles:
            fh.close()


#: what all decoded-block caches of a process may hold together, by default
DEFAULT_CACHE_LIMIT = 256 << 20

# One least-recently-used order over every cache of the process, so the number of
# open files does not multiply the memory the caches hold: (cache id, key) ->
# (cache, nbytes). Reentrant: a cache that is garbage-collected releases its bytes
# from __del__, which may run while this thread holds the lock.
_shared_lock = threading.RLock()
_shared: OrderedDict = OrderedDict()
_shared_bytes = 0
_cache_limit = None
_ids = iter(range(1 << 62))


def cache_limit() -> int:
    """What all decoded-block caches of this process may hold together, in bytes."""
    global _cache_limit
    if _cache_limit is None:
        import os
        from ._memory import parse_size
        env = os.environ.get("MICRO_READER_CACHE_LIMIT")
        _cache_limit = parse_size(env) if env else DEFAULT_CACHE_LIMIT
    return _cache_limit


def set_cache_limit(limit) -> None:
    """Set what all decoded-block caches of this process may hold together (bytes or
    e.g. "512MB"; 0 keeps nothing); None restores the default (the
    ``MICRO_READER_CACHE_LIMIT`` environment variable, else 256 MB). Least recently
    used blocks are dropped at once if the caches hold more."""
    global _cache_limit
    from ._memory import parse_size
    with _shared_lock:
        _cache_limit = None if limit is None else max(0, parse_size(limit))
        _evict_shared(cache_limit())


def cached_bytes() -> int:
    """What the decoded-block caches of this process hold now, in bytes."""
    with _shared_lock:
        return _shared_bytes


def _evict_shared(limit: int) -> None:
    global _shared_bytes
    while _shared_bytes > limit and _shared:
        (_, key), (cache, nbytes) = _shared.popitem(last=False)
        cache._drop(key)
        _shared_bytes -= nbytes


class DecodeCache:
    """Decoded segments (tiles, strips, chunks), least recently used out.

    Keeps a segment shared by neighbouring reads from being decoded twice.
    Segments above a quarter of ``max_bytes`` are not kept (a whole plane from
    a single-strip file would evict everything else). ``max_bytes`` caps this
    cache (one file or scene); every cache of the process together stays within
    :func:`cache_limit`, so many open files do not hold many times the memory."""

    def __init__(self, max_bytes: int):
        self.max_bytes = max_bytes
        self._items: OrderedDict = OrderedDict()
        self._bytes = 0
        self._id = next(_ids)

    def get(self, key):
        with _shared_lock:
            seg = self._items.get(key)
            if seg is not None:
                self._items.move_to_end(key)
                _shared.move_to_end((self._id, key))
            return seg

    def put(self, key, seg: np.ndarray) -> None:
        global _shared_bytes
        if seg.nbytes > self.max_bytes // 4 or seg.nbytes > cache_limit():
            return
        seg.setflags(write=False)
        with _shared_lock:
            if key in self._items:
                return
            self._items[key] = seg
            self._bytes += seg.nbytes
            _shared[(self._id, key)] = (self, seg.nbytes)
            _shared_bytes += seg.nbytes
            while self._bytes > self.max_bytes:
                old_key = next(iter(self._items))
                _shared.pop((self._id, old_key))
                _shared_bytes -= self._items[old_key].nbytes
                self._drop(old_key)
            _evict_shared(cache_limit())

    def _drop(self, key) -> None:
        """Forget *key* (the shared order is updated by the caller)."""
        seg = self._items.pop(key, None)
        if seg is not None:
            self._bytes -= seg.nbytes

    def clear(self) -> None:
        global _shared_bytes
        with _shared_lock:
            for key, seg in self._items.items():
                if _shared.pop((self._id, key), None) is not None:
                    _shared_bytes -= seg.nbytes
            self._items.clear()
            self._bytes = 0

    def __del__(self):
        try:
            self.clear()
        except Exception:                           # noqa: BLE001 - interpreter exit
            pass
