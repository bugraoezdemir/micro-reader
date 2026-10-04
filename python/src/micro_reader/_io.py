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


class DecodeCache:
    """Decoded segments (tiles, strips, chunks), least recently used out.

    Keeps a segment shared by neighbouring reads from being decoded twice.
    Segments above a quarter of the budget are not kept (a whole plane from
    a single-strip file would evict everything else)."""

    def __init__(self, max_bytes: int):
        self.max_bytes = max_bytes
        self._items: OrderedDict = OrderedDict()
        self._bytes = 0
        self._lock = threading.Lock()

    def get(self, key):
        with self._lock:
            seg = self._items.get(key)
            if seg is not None:
                self._items.move_to_end(key)
            return seg

    def put(self, key, seg: np.ndarray) -> None:
        if seg.nbytes > self.max_bytes // 4:
            return
        seg.setflags(write=False)
        with self._lock:
            if key in self._items:
                return
            self._items[key] = seg
            self._bytes += seg.nbytes
            while self._bytes > self.max_bytes:
                _, old = self._items.popitem(last=False)
                self._bytes -= old.nbytes

    def clear(self) -> None:
        with self._lock:
            self._items.clear()
            self._bytes = 0
