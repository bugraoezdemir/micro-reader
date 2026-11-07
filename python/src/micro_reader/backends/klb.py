"""KLB (Keller Lab Block file type; Amat et al., Nature Protocols 2015), own reader.

Written from the format's public description and klb_imageHeader.h: a header of a version
byte, xyzct sizes (uint32), the sampling per axis (float32: um, um, um, index count,
seconds), data type and compression bytes, 256 bytes of free text and the block size per
axis (uint32); then one uint64 per block, each the end of that block's bytes counted from
the end of this table.  Blocks run with x fastest, then y, z, c, t, as do the pixels inside
a block.  All little endian.

A rectangle decodes only the blocks it touches, on the decode pool (bzip2 and zlib release
the GIL); decoded blocks are cached, since the usual block (96 x 96 x 8) is thinner than a
reader's chunk of z.  Without magic number the extension decides, as for an old MRC file.
Bio-Formats reports one time point for a file of several (gap hunt 2026-11-02, N-01).
A value of 1 in the header is its default, no measurement: see ``_sizes``.
"""
from __future__ import annotations

import builtins
import bz2
import itertools
import math
import os
import struct
import zlib

import numpy as np

from .. import _memory, _pool
from .._base import File, Image
from .._errors import CorruptFileError, UnsupportedFormatError
from .._io import DecodeCache, ThreadHandles
from .._metadata import Channel, Metadata, quantity

NAME = "klb"
EXTENSIONS = (".klb",)
VERSION = 2
HEADER = struct.Struct("<B5I5fBB256s5I")
DECODE_CACHE_BYTES = 64 << 20
#: data type byte -> numpy dtype
DTYPES = {0: "u1", 1: "u2", 2: "u4", 3: "u8", 4: "i1", 5: "i2", 6: "i4", 7: "i8", 8: "f4", 9: "f8"}
RAW, BZIP2, ZLIB = 0, 1, 2
#: the axes of the sampling a header records, and their units; c is an index count
UNITS = {"x": "micrometer", "y": "micrometer", "z": "micrometer", "t": "second"}


def sniff(head: bytes, path: str) -> bool:
    return path.lower().endswith(EXTENSIONS)


def _decoded(data: bytes, compression: int) -> bytes:
    if compression == BZIP2:
        return bz2.decompress(data)
    if compression == ZLIB:
        return zlib.decompress(data)
    return data


class _KlbImage(Image):
    """The whole file: axes 'tczyx', one read unit per block."""

    def __init__(self, path: str, shape, dtype, compression: int, block, offsets, start: int,
                 sizes: dict):
        super().__init__(0, "tczyx", shape, dtype, name=os.path.basename(path), read_unit=block)
        self._path = path
        self._file_dtype = np.dtype(dtype).newbyteorder("<")
        self._compression = compression
        self._block = block
        self._blocks = tuple(math.ceil(n / b) for n, b in zip(shape, block))
        self._offsets = offsets
        self._start = start
        self._sizes = sizes
        self._handles = ThreadHandles(path)
        # room for a block however large, or each read of one decodes it again
        block_bytes = int(np.prod(block)) * self._file_dtype.itemsize
        self.decode_cache = DecodeCache(max(DECODE_CACHE_BYTES, 4 * block_bytes))
        self._closed = False

    def _read_metadata(self):
        return Metadata(pixel_sizes=self._sizes, channels=(Channel(),) * self.shape[1])

    def _read_block(self, block):
        if self._closed:
            raise ValueError("this KLB image has been closed: its file was closed "
                             "(File.close() or the end of a with block)")
        out = np.empty(tuple(s.stop - s.start for s in block), self.dtype)
        spans = [range(s.start // b, (s.stop - 1) // b + 1) for s, b in zip(block, self._block)]
        tasks = (lambda at=at: self._into(at, block, out) for at in itertools.product(*spans))
        return out, _pool.run(tasks)

    def _into(self, at, block, out) -> int:
        """The part of block *at* (t, c, z, y, x block numbers) inside *block*, into *out*."""
        number = int(np.ravel_multi_index(at, self._blocks))
        data, read = self.decode_cache.get(number), 0
        if data is None:
            data = self._decode(number, at)
            self.decode_cache.put(number, data)
            read = data.size
        src, dst = [], []
        for k, span, size in zip(at, block, self._block):
            low, high = max(span.start, k * size), min(span.stop, (k + 1) * size)
            src.append(slice(low - k * size, high - k * size))
            dst.append(slice(low - span.start, high - span.start))
        out[tuple(dst)] = data[tuple(src)]
        return read

    def _decode(self, number: int, at) -> np.ndarray:
        shape = tuple(min(b, n - k * b) for k, b, n in zip(at, self._block, self.shape))
        size = int(np.prod(shape)) * self._file_dtype.itemsize
        first = int(self._offsets[number - 1]) if number else 0
        stored = self._handles.read(self._start + first, int(self._offsets[number]) - first)
        with _memory.reserve(size, f"{self._path!r}: a KLB block"):
            try:
                data = _decoded(stored, self._compression)
            except (OSError, ValueError, zlib.error) as exc:
                raise CorruptFileError(f"{self._path!r}: block {number}: {exc}") from exc
            if len(data) != size:
                raise CorruptFileError(f"{self._path!r}: block {number} decodes to {len(data)} "
                                       f"bytes, its shape {shape} holds {size}")
            return np.frombuffer(data, self._file_dtype).reshape(shape)

    def close(self) -> None:
        self._closed = True
        self.decode_cache.clear()
        self._handles.close()


class _KlbFile(File):
    backend = NAME


def open(path: str, tiles: str = "separate", resolutions: str = "base") -> _KlbFile:  # noqa: A001 - backend protocol name
    file_size = os.path.getsize(path)
    if file_size < HEADER.size:
        raise CorruptFileError(f"{path!r} is smaller than a KLB header")
    with builtins.open(path, "rb") as fh:
        fields = HEADER.unpack(fh.read(HEADER.size))
        version, xyzct, sampling = fields[0], fields[1:6], fields[6:11]
        data_type, compression, block_xyzct = fields[11], fields[12], fields[14:19]
        if version != VERSION:
            raise UnsupportedFormatError(f"{path!r}: KLB header version {version}: only version "
                                         f"{VERSION} is known")
        if data_type not in DTYPES:
            raise CorruptFileError(f"unknown KLB data type {data_type} in {path!r}")
        if compression not in (RAW, BZIP2, ZLIB):
            raise UnsupportedFormatError(f"{path!r}: KLB compression {compression}: known are "
                                         f"{RAW} (none), {BZIP2} (bzip2) and {ZLIB} (zlib)")
        if min(xyzct) < 1 or min(block_xyzct) < 1:
            raise CorruptFileError(f"invalid KLB sizes {xyzct} or blocks {block_xyzct} in {path!r}")
        count = math.prod(math.ceil(n / b) for n, b in zip(xyzct, block_xyzct))
        if HEADER.size + 8 * count > file_size:
            raise CorruptFileError(f"{path!r} is truncated: its block table of {count} entries "
                                   f"does not fit in {file_size} bytes")
        offsets = np.frombuffer(fh.read(8 * count), "<u8")
    start = HEADER.size + 8 * count
    if start + int(offsets[-1]) > file_size or np.any(np.diff(offsets.astype(np.int64)) < 0):
        raise CorruptFileError(f"{path!r} is truncated or its block table is damaged: the blocks "
                               f"end at byte {start + int(offsets[-1])} of {file_size}")
    return _KlbFile(path, [_KlbImage(path, xyzct[::-1], DTYPES[data_type], compression,   # t c z y x
                                     block_xyzct[::-1], offsets, start, _sizes(sampling, xyzct))])


def _sizes(sampling, xyzct) -> dict:
    """Pixel sizes of the header's sampling.  1 is what a header holds when nobody set a size
    and no unit tells it from a real 1: unset on t, and on z over a set x or y, and on
    x, y and z when none of the three is set.  A z of its own keeps x and y of 1."""
    x, y, z, _, t = sampling
    given = {}
    if (x, y, z) != (1, 1, 1):
        given.update(x=x, y=y)
        if z != 1:
            given["z"] = z
    if t != 1 and xyzct[4] > 1:
        given["t"] = t
    found = {axis: quantity(value, UNITS[axis]) for axis, value in given.items()}
    return {axis: q for axis, q in found.items() if q}
