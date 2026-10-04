"""Streaming decode of compressed blocks too large to decode whole
(``_memory``; docs/PLAN.md, "memory-bounded reads of huge compressed blocks").

A block is decoded from its start, keeping only the bytes a read asks for;
the compressed bytes are read from the file in small pieces.  Each decoder
keeps its place, and reads are handed the decoder of their block that is
closest before their start, so regions read top to bottom (as eubi-bridge's
writer reads) continue where the last one stopped: reading a whole plane
stays linear.  Several reads of one block at once each get a decoder of
their own.

Checkpoints (deflate / zlib): the decoders of a block share an index of the
inflater's state every ``every`` decoded bytes (about 1/128 of the block, at
least 1 MB; ~43 KB each), so a read before a decoder's place -- a viewer
jumping around -- resumes from the nearest checkpoint instead of starting
the block again.  Indexes of the ``MAX_INDEXES`` most recent blocks are kept.

Native: ``micro_reader_codecs.StreamDecoder`` (zlib / deflate, LZW, zstd,
PackBits, none).  Reference, in Python: zlib / deflate (stdlib), LZMA
(stdlib), zstd (Python 3.14 or ``zstandard``), PackBits, none -- LZMA
always, the others when the native package is missing.
"""
from __future__ import annotations

import threading
import zlib
from collections import OrderedDict

from . import _codecs
from ._errors import CorruptFileError

#: codecs that can be streamed
STREAMABLE = ("zlib", "deflate", "lzw", "zstd", "packbits", "lzma", "none", "png")
NATIVE = ("zlib", "deflate", "lzw", "zstd", "packbits", "none", "png")
#: decoders kept for later reads (each holds a file handle and its window)
MAX_IDLE = 32
#: blocks whose checkpoint indexes are kept; checkpoints per block, spacing
MAX_INDEXES = 16
MAX_CHECKPOINTS = 128
MIN_CHECKPOINT_EVERY = 1 << 20
CHECKPOINTED = ("zlib", "deflate")
_PIECE = 1 << 16


def streamable(codec: str) -> bool:
    if codec not in STREAMABLE:
        return False
    if codec in NATIVE and _native_stream() is not None:
        return True
    return codec in ("zlib", "deflate", "lzma", "packbits", "none") or \
        (codec == "zstd" and _reference_zstd_stream() is not None)


def _native_stream():
    native = _codecs._native
    return getattr(native, "StreamDecoder", None) if native is not None else None


def _reference_zstd_stream():
    try:
        from compression import zstd                  # Python >= 3.14
        return lambda: zstd.ZstdDecompressor()
    except ImportError:
        pass
    try:
        import zstandard
        return lambda: zstandard.ZstdDecompressor().decompressobj()
    except ImportError:
        return None


class _PackBits:
    """PackBits as a decompressor object (``decompress(data)``)."""

    def __init__(self):
        self.pending = b""

    def decompress(self, data: bytes) -> bytes:
        data, out, i = self.pending + data, bytearray(), 0
        while i < len(data):
            n = data[i] - 256 if data[i] > 127 else data[i]
            if n >= 0:
                if i + 1 + n + 1 > len(data):
                    break
                out += data[i + 1:i + 2 + n]
                i += n + 2
            elif n != -128:
                if i + 1 >= len(data):
                    break
                out += bytes([data[i + 1]]) * (1 - n)
                i += 2
            else:
                i += 1
        self.pending = data[i:]
        return bytes(out)


class _ReferenceIndex:
    """Checkpoints of one zlib / deflate block for the reference decoders:
    (decoded position, compressed bytes consumed, decompressor copy)."""

    def __init__(self, every: int, max: int = MAX_CHECKPOINTS):
        self.every, self.max, self.points = every, max, []
        self._lock = threading.Lock()

    def best(self, start: int):
        with self._lock:
            before = [p for p in self.points if p[0] <= start]
            return max(before, key=lambda p: p[0]) if before else None

    def record(self, position: int, consumed: int, decompressor) -> None:
        with self._lock:
            if len(self.points) >= self.max or \
                    any(abs(p[0] - position) < self.every for p in self.points):
                return
            self.points.append((position, consumed, decompressor.copy()))

    def __len__(self):
        return len(self.points)


class _ReferenceDecoder:
    """The native decoder's interface, decoded with Python's decompressor objects."""

    def __init__(self, path: str, offset: int, length: int, codec: str, index=None):
        self.path, self.offset, self.length, self.codec = path, offset, length, codec
        self.position, self.restarts, self.resumes = 0, 0, 0
        self._index = index if codec in CHECKPOINTED else None
        self._fh = None
        self._lock = threading.Lock()

    def _resume(self, point) -> None:
        position, consumed, decompressor = point
        if self._fh is not None:
            self._fh.close()
        self._fh = open(self.path, "rb")                   # noqa: SIM115 - kept open
        self._fh.seek(self.offset + consumed)
        self._left = self.length - consumed
        self._pending = b""
        self.position = position
        self._decompressor = decompressor.copy()
        self.resumes += 1

    def _start(self):
        import lzma
        if self._fh is not None:
            self.restarts += 1
            self._fh.close()
        self._fh = open(self.path, "rb")                   # noqa: SIM115 - kept open
        self._fh.seek(self.offset)
        self._left = self.length
        self._pending = b""
        self.position = 0
        self._decompressor = {
            "zlib": lambda: zlib.decompressobj(),
            "deflate": lambda: zlib.decompressobj(-15),
            "lzma": lambda: lzma.LZMADecompressor(),
            "zstd": _reference_zstd_stream(),
            "packbits": _PackBits,
            "none": lambda: None,
        }[self.codec]()

    def _more(self) -> bytes:
        """The next decoded bytes; b"" at the end of the block."""
        while True:
            if not self._left:
                return b""
            data = self._fh.read(min(_PIECE, self._left))
            self._left -= len(data)
            if not data:
                self._left = 0
                return b""
            if self._decompressor is None:
                return data
            try:
                out = self._decompressor.decompress(data)
            except Exception as exc:                    # noqa: BLE001 - codec specific
                raise CorruptFileError(f"{self.path!r}: corrupt {self.codec} data: {exc}") from exc
            if out:
                return out

    def read_into(self, start: int, out) -> int:
        view = memoryview(out).cast("B")
        with self._lock:
            behind = self._fh is None or start < self.position
            point = self._index.best(start) if self._index is not None else None
            if point is not None and (behind or point[0] > self.position):
                self._resume(point)
            elif behind:
                self._start()
            written = 0
            while written < len(view):
                if not self._pending:
                    if self._index is not None:          # all decoded so far is delivered
                        self._index.record(self.position, self.length - self._left,
                                           self._decompressor)
                    self._pending = self._more()
                    if not self._pending:
                        break
                skip = start + written - self.position
                if skip >= len(self._pending):             # all before the start
                    self.position += len(self._pending)
                    self._pending = b""
                    continue
                if skip > 0:
                    self.position += skip
                    self._pending = self._pending[skip:]
                n = min(len(view) - written, len(self._pending))
                view[written:written + n] = self._pending[:n]
                self._pending = self._pending[n:]
                self.position += n
                written += n
            return written

    def close(self) -> None:
        with self._lock:
            if self._fh is not None:
                self._fh.close()
                self._fh = None

    def __del__(self):
        if getattr(self, "_fh", None) is not None:
            self._fh.close()


_lock = threading.Lock()
#: (path, offset, codec) -> decoders not in use; least recently used first
_idle: "OrderedDict[tuple, list]" = OrderedDict()
#: (path, offset, codec) -> the block's checkpoint index; least recently used first
_indexes: "OrderedDict[tuple, object]" = OrderedDict()


def _index(key: tuple, size) -> object:
    """The checkpoint index of block *key* (zlib / deflate), created on first use."""
    if key[2] not in CHECKPOINTED:
        return None
    with _lock:
        index = _indexes.get(key)
        if index is None:
            every = max(MIN_CHECKPOINT_EVERY, int(size or 0) // MAX_CHECKPOINTS)
            native = _codecs._native
            index = (native.BlockIndex(every, MAX_CHECKPOINTS)
                     if native is not None and hasattr(native, "BlockIndex")
                     else _ReferenceIndex(every))
            _indexes[key] = index
            while len(_indexes) > MAX_INDEXES:
                _indexes.popitem(last=False)
        else:
            _indexes.move_to_end(key)
        return index


def _new(path: str, offset: int, length: int, codec: str, index=None):
    native = _native_stream()
    if native is not None and codec in NATIVE:
        if index is not None and not isinstance(index, _ReferenceIndex):
            return native(path, offset, length, codec, index)
        return native(path, offset, length, codec)
    return _ReferenceDecoder(path, offset, length, codec,
                             index if isinstance(index, _ReferenceIndex) else None)


def _acquire(key: tuple, length: int, start: int, size=None):
    with _lock:
        decoders = _idle.get(key, [])
        before = [d for d in decoders if d.position <= start]
        if before:
            decoder = max(before, key=lambda d: d.position)
            decoders.remove(decoder)
            if not decoders:
                _idle.pop(key, None)
            return decoder
        if decoders:                     # all past *start*: one resumes from a checkpoint
            decoder = decoders.pop(0)
            if not decoders:
                _idle.pop(key, None)
            return decoder
    return _new(key[0], key[1], length, key[2], _index(key, size))


def _release(key: tuple, decoder) -> None:
    with _lock:
        _idle.setdefault(key, []).append(decoder)
        _idle.move_to_end(key)
        total = sum(len(v) for v in _idle.values())
        while total > MAX_IDLE:
            oldest = next(iter(_idle))
            _idle[oldest].pop(0)
            if not _idle[oldest]:
                del _idle[oldest]
            total -= 1


def read(path: str, offset: int, length: int, codec: str, start: int, out,
         size: int = None) -> int:
    """Decoded bytes *start*... of the block (*length* compressed bytes at
    *offset* of *path*, *codec*; *size* decoded bytes, if known) into *out*;
    returns the bytes written (fewer at the end of the block)."""
    key = (str(path), int(offset), codec)
    decoder = _acquire(key, int(length), int(start), size)
    try:
        try:
            return decoder.read_into(int(start), out)
        except ValueError as exc:
            raise CorruptFileError(f"{path!r}: corrupt {codec} data: {exc}") from exc
    finally:
        _release(key, decoder)


def forget(path: str) -> None:
    """Close and drop the decoders of *path* (its file was closed)."""
    with _lock:
        dropped = [d for key in [k for k in _idle if k[0] == str(path)] for d in _idle.pop(key)]
        for key in [k for k in _indexes if k[0] == str(path)]:
            del _indexes[key]
    for decoder in dropped:
        getattr(decoder, "close", lambda: None)()
