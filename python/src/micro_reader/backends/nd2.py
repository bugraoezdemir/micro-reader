"""Nikon ND2 (version 3 and later; legacy versions 1 / 2 in ``_nd2_legacy``), own reader.

A frame is one (time, position, Z) plane of all channels, stored in a chunk
``ImageDataSeq|<n>!`` after an 8-byte timestamp: ``height`` rows of
``width_bytes`` bytes (rows may be padded), each row ``width`` pixels of all
channels and RGB components interleaved.  Frame numbers run over the
experiment loops, outermost first.  Structure parsing is ported from nd2
(Talley Lambert, BSD-3), see ``_nd2_structure``.

- Uncompressed frames: only the requested rows are read (all channels
  come along: they are interleaved per pixel).
- "Lossless" frames are one zlib stream each: decoded whole, cached.

Images are the XY positions, indexed by ``series`` (bioio-nd2 calls them scenes); axes ``tczyx``, plus ``s``
for RGB components.  Legacy JPEG 2000 ND2 (versions 1 / 2): ``_nd2_legacy``.
Not supported:
lossy compression, tiled frames, custom loops.
"""
from __future__ import annotations

import builtins
import itertools
import os
import threading
import zlib

import numpy as np

from .. import _memory, _pool, _stream
from .._base import File, Image
from .._errors import CorruptFileError, UnsupportedFormatError
from .._io import DecodeCache, ThreadHandles
from .._spatial import CoordinateSystem
from . import _nd2_legacy as legacy
from . import _nd2_structure as structure

NAME = "nd2"
DECODE_CACHE_BYTES = 64 << 20
RAW_TASK_BYTES = 8 << 20          # uncompressed frames are read in row pieces of this size
SEPARATE_READ_BYTES = 32 << 10    # bytes worth skipping rather than issuing one more read
STREAM_PIECE_BYTES = 8 << 20      # rows of a streamed "lossless" frame decoded at a time
TIMESTAMP_BYTES = 8               # each frame chunk starts with a float64 timestamp


def sniff(head: bytes, path: str) -> bool:
    if head[:4] == structure.CHUNK_MAGIC.to_bytes(4, "little"):
        return True
    # legacy ND2 is a JPEG 2000 box file: the JP2 signature alone would also
    # claim plain .jp2 images, so the extension decides (the box map at the
    # end is checked on opening)
    return legacy.is_legacy(head) and path.lower().endswith(".nd2")


class Nd2Parser:
    """What the images of one file share: layout, frame offsets, handles."""

    def __init__(self, path: str):
        self.path = path
        with builtins.open(path, "rb") as fh:
            ver = structure.version(fh.read(structure.START_CHUNK.size))
            if ver < (3, 0):
                raise UnsupportedFormatError(f"{path!r}: legacy ND2 version {ver}")
            cmap = structure.chunkmap(fh)

            def decoded(name):
                if name not in cmap:
                    return {}
                return structure.decode_clx_lite(structure.read_chunk(fh, cmap[name][0]))

            picture = decoded(b"ImageMetadataSeqLV|0!")
            experiment = decoded(b"ImageMetadataLV!")
            self.attrs = structure.attributes(decoded(b"ImageAttributesLV!"), picture)
            self.loops = structure.experiment(experiment)
            #: raw picture metadata and experiment, for ``_nd2_metadata``
            self.picture = picture.get("SLxPictureMetadata", picture)
            self.experiment = experiment
        if self.attrs.compression == "lossy":
            raise UnsupportedFormatError(f"{path!r}: lossy ND2 compression")
        a = self.attrs
        if a.components % a.channels:
            raise UnsupportedFormatError(f"{path!r}: {a.components} components "
                                         f"for {a.channels} channels")
        self.samples = a.components // a.channels
        self.dtype = np.dtype(a.dtype)
        self.pixel_bytes = a.components * self.dtype.itemsize
        if a.width_bytes < a.width * self.pixel_bytes:
            raise CorruptFileError(f"{path!r}: rows of {a.width_bytes} bytes cannot hold "
                                   f"{a.width} pixels of {self.pixel_bytes} bytes")
        kinds = [loop.kind for loop in self.loops]
        if len(set(kinds)) != len(kinds):
            raise UnsupportedFormatError(f"{path!r}: repeated experiment loops {kinds}")
        # frame chunk -> offset of its header; data offsets are read lazily
        self.frames = {int(name[13:-1]): offset for name, (offset, _) in cmap.items()
                       if name.startswith(b"ImageDataSeq|")}
        self._data_at: dict = {}
        self._lock = threading.Lock()
        self.handles = ThreadHandles(path)

    def size(self, kind: str) -> int:
        return next((loop.count for loop in self.loops if loop.kind == kind), 1)

    def frame_number(self, t: int, p: int, z: int) -> int:
        index = {"T": t, "P": p, "Z": z}
        number = 0
        for loop in self.loops:
            number = number * loop.count + index[loop.kind]
        return number

    def frame_data(self, number: int):
        """(offset of the frame's pixels or compressed stream, its length),
        or None for a frame that was never written."""
        where = self._data_at.get(number)
        if where is None:
            header = self.frames.get(number)
            if header is None:
                return None
            fh = self.handles.get()
            start, length = structure.data_offset(fh, header)
            where = self._data_at[number] = (start + TIMESTAMP_BYTES, length - TIMESTAMP_BYTES)
        return where


class _Nd2Image(Image):
    """One XY position: axes 'tczyx' (+ 's' for RGB components)."""

    def __init__(self, nd2: Nd2Parser, position: int):
        a = nd2.attrs
        self._f = nd2
        self._p = position
        self._rgb = nd2.samples > 1
        shape = (nd2.size("T"), a.channels, nd2.size("Z"), a.height, a.width)
        axes = "tczyx"
        if self._rgb:
            shape, axes = shape + (nd2.samples,), axes + "s"
        #: decoded "lossless" frames kept for reuse
        self.decode_cache = DecodeCache(DECODE_CACHE_BYTES)
        self._closed = False
        unit = (1, a.channels, 1, a.height, a.width) + ((nd2.samples,) if self._rgb else ()) \
            if a.compression == "lossless" else None
        super().__init__(position, axes, shape, nd2.dtype,
                         name=f"{os.path.basename(nd2.path)} position {position}",
                         read_unit=unit)

    def _read_block(self, block):
        if self._closed:
            raise ValueError("this ND2 image has been closed: its file was closed "
                             "(File.close() or the end of a with block)")
        ts, cs, zs, ys, xs = block[:5]
        ss = block[5] if self._rgb else slice(0, 1)
        out = np.empty(tuple(s.stop - s.start for s in block), dtype=self.dtype)
        view = out if self._rgb else out[..., None]          # always (..., s)
        tasks = []
        for t, z in itertools.product(range(ts.start, ts.stop), range(zs.start, zs.stop)):
            dst = view[t - ts.start, :, z - zs.start]         # (c, y, x, s)
            where = self._f.frame_data(self._f.frame_number(t, self._p, z))
            if where is None:
                dst[...] = 0
            elif self._f.attrs.compression == "lossless":
                tasks.append(lambda a=(t, z, where, dst, cs, ys, xs, ss): self._decoded_into(*a))
            else:
                tasks.extend(self._raw_tasks(where[0], dst, cs, ys, xs, ss))
        return out, _pool.run(tasks)

    def _frame_view(self, buf: np.ndarray, rows: int) -> np.ndarray:
        """Bytes of *rows* whole rows -> (rows, x, channel, sample)."""
        f, a = self._f, self._f.attrs
        pixels = buf.reshape(rows, a.width_bytes)[:, :a.width * f.pixel_bytes]
        return pixels.view(f.dtype).reshape(rows, a.width, a.channels, f.samples)

    def _raw_tasks(self, start, dst, cs, ys, xs, ss):
        f, a = self._f, self._f.attrs
        per_task = max(1, RAW_TASK_BYTES // a.width_bytes)
        # whole rows when the columns skipped per row are few: fewer bytes than another read
        # call costs (as in LIF), or at most half the row; else only the needed bytes per row
        skipped = a.width_bytes - (xs.stop - xs.start) * f.pixel_bytes
        whole_rows = (xs.stop - xs.start) * 2 >= a.width or skipped <= SEPARATE_READ_BYTES
        for y0 in range(ys.start, ys.stop, per_task):
            y1 = min(ys.stop, y0 + per_task)
            part = dst[:, y0 - ys.start:y1 - ys.start]
            yield lambda args=(start, y0, y1, part, cs, xs, ss, whole_rows): self._read_rows(*args)

    def _read_rows(self, start, y0, y1, dst, cs, xs, ss, whole_rows) -> int:
        """Rows y0:y1 of an uncompressed frame into *dst* (c, y, x, s)."""
        f, a = self._f, self._f.attrs
        fh = f.handles.get()
        n = y1 - y0
        if whole_rows:
            buf = np.empty(n * a.width_bytes, np.uint8)
            fh.seek(start + y0 * a.width_bytes)
            if fh.readinto(memoryview(buf)) != buf.nbytes:
                raise CorruptFileError(f"{f.path!r}: frame data ends early")
            pixels = self._frame_view(buf, n)[:, xs]
            read = n * a.width * a.components
        else:
            width = xs.stop - xs.start
            buf = np.empty((n, width * f.pixel_bytes), np.uint8)
            for i in range(n):
                fh.seek(start + (y0 + i) * a.width_bytes + xs.start * f.pixel_bytes)
                if fh.readinto(memoryview(buf[i])) != buf.shape[1]:
                    raise CorruptFileError(f"{f.path!r}: frame data ends early")
            pixels = buf.view(f.dtype).reshape(n, width, a.channels, f.samples)
            read = n * width * a.components
        dst[...] = pixels[:, :, cs, ss].transpose(2, 0, 1, 3)
        return read

    def _decoded_into(self, t, z, where, dst, cs, ys, xs, ss) -> int:
        frame, read = self.decode_cache.get((t, z)), 0
        a = self._f.attrs
        if frame is None and _memory.should_stream(a.height * a.width_bytes):
            return self._streamed_into(where, dst, cs, ys, xs, ss)
        if frame is None:
            f, a = self._f, self._f.attrs
            with _memory.reserve(a.height * a.width_bytes, f"{f.path!r}: an ND2 frame"):
                data = zlib.decompress(f.handles.read(*where), 15, a.height * a.width_bytes)
            if len(data) < a.height * a.width_bytes:
                raise CorruptFileError(f"{f.path!r}: frame decodes to {len(data)} bytes")
            frame = self._frame_view(np.frombuffer(data, np.uint8)[:a.height * a.width_bytes],
                                     a.height)
            self.decode_cache.put((t, z), frame)
            read = frame.size
        dst[...] = frame[ys, xs][:, :, cs, ss].transpose(2, 0, 1, 3)
        return read

    def _read_metadata(self):
        from ._nd2_metadata import nd2_metadata
        return nd2_metadata(self._f.picture, self._f.experiment, self._p,
                            dict(zip(self.axes, self.shape)))

    def _streamed_into(self, where, dst, cs, ys, xs, ss) -> int:
        """Rows *ys* of a "lossless" frame too large to decode whole: its zlib
        stream decoded as a stream, in pieces of rows (``_stream``)."""
        f, a = self._f, self._f.attrs
        per_piece = max(1, min(STREAM_PIECE_BYTES, _memory.memory_limit() // 4) // a.width_bytes)
        for y0 in range(ys.start, ys.stop, per_piece):
            y1 = min(ys.stop, y0 + per_piece)
            with _memory.reserve((y1 - y0) * a.width_bytes, f"{f.path!r}: rows of a frame"):
                buf = np.empty((y1 - y0) * a.width_bytes, np.uint8)
                n = _stream.read(f.path, where[0], where[1], "zlib", y0 * a.width_bytes, buf,
                                 size=a.height * a.width_bytes)
                if n < buf.nbytes:
                    raise CorruptFileError(f"{f.path!r}: frame ends at {n} of {buf.nbytes} bytes")
                rows = self._frame_view(buf, y1 - y0)
                dst[:, y0 - ys.start:y1 - ys.start] =                     rows[:, xs][:, :, cs, ss].transpose(2, 0, 1, 3)
        return (ys.stop - ys.start) * a.width * a.components

    def close(self) -> None:
        self._closed = True
        self.decode_cache.clear()
        _stream.forget(self._f.path)


class _Nd2File(File):
    backend = NAME

    def __init__(self, path, images, nd2: Nd2Parser):
        super().__init__(path, images)
        self._nd2 = nd2

    def _declare_coordinate_systems(self):
        # the positions' one stage frame (plan 4.5), when they can have a
        # position at all: none without the camera-to-stage matrix
        from ._nd2_metadata import STAGE, placed
        return (CoordinateSystem(STAGE),) if placed(self._nd2.picture) else ()

    def close(self) -> None:
        super().close()
        self._nd2.handles.close()


def open(path: str, tiles: str = "separate", resolutions: str = "base"):  # noqa: A001 - backend protocol name
    with builtins.open(path, "rb") as fh:
        if legacy.is_legacy(fh.read(12)):
            return legacy._LegacyNd2File(path)
    nd2 = Nd2Parser(path)
    return _Nd2File(path, [_Nd2Image(nd2, p) for p in range(nd2.size("P"))], nd2)
