"""Olympus / Evident ETS files: the tiled pixel stores of a cellSens VSI
(``_<name>_/stack<N>/frame_t*.ets``).

Layout as slideio reads it (Stanislav Melnikov, BSD-3: ``etsfile.cpp``,
``vsistruct.hpp``, ``pyramid.cpp``):

- a 64-byte ``SIS`` header (little endian): magic ``SIS\\0``, header size 64,
  version, number of dimensions, file position and size of the ``ETS``
  header, position and number of the used chunks (tiles);
- the ``ETS`` header: magic ``ETS\\0``, version, component type (the VSI
  value type: 2 uint8, 4 uint16, ...), component count, colour space (1
  grey, 3 RGB, 4 BGR), compression (0 raw, 2 JPEG, 3 JPEG 2000, 5 lossless
  JPEG, 8 PNG, 9 BMP), quality, tile x / y / z size, 17 pixel-info hints, 10
  background values (40 bytes: a missing tile's value per component, packed as
  Bio-Formats reads them), component order, "use pyramid";
- one record per tile: 4 bytes, the tile's coordinates (one int32 per
  dimension: tile column, tile row, then the image's other dimensions in an
  order the VSI's metadata gives, the pyramid level last when the file has a
  pyramid), its file offset (int64), its size (uint32), 4 bytes.

Level ``k`` is level 0 halved ``k`` times, rounded up (its tiles have the same size).
Which coordinate is channel, z or time is not in the ETS: ``read`` takes the
values of the coordinates between the tile row and the level as they are.
"""
from __future__ import annotations

import struct
import threading

import numpy as np

from .. import _codecs, _pool
from .._errors import CorruptFileError, UnsupportedFormatError
from .._io import DecodeCache, ThreadHandles

SIS = struct.Struct("<4sIIIQII QII QII")
#: the VSI value type of a component -> numpy dtype
COMPONENT_TYPES = {1: "i1", 2: "u1", 3: "<i2", 4: "<u2", 5: "<i4", 6: "<u4", 7: "<i8", 8: "<u8",
                   9: "<f4", 10: "<f8"}
RAW, JPEG, JPEG_2000, JPEG_LOSSLESS, PNG, BMP = 0, 2, 3, 5, 8, 9
COMPRESSIONS = {RAW: "raw", JPEG: "JPEG", JPEG_2000: "JPEG 2000", JPEG_LOSSLESS: "lossless JPEG",
                PNG: "PNG", BMP: "BMP"}
GREY, RGB, BGR = 1, 3, 4
DECODE_CACHE_BYTES = 64 << 20


class EtsFile:
    """The header and tile table of an ETS file; ``read`` fills rectangles."""

    def __init__(self, path: str):
        self.path = path
        with open(path, "rb") as fh:
            head = fh.read(SIS.size)
            if len(head) < SIS.size or head[:3] != b"SIS":
                raise CorruptFileError(f"{path!r}: no ETS file (no SIS header)")
            (_magic, header_size, self.version, self.ndim, extra_at, extra_size, _u1, chunks_at,
             n_chunks, _u2, _free_at, _n_free, _u3) = SIS.unpack(head)
            if header_size != 64 or self.ndim < 2:
                raise CorruptFileError(f"{path!r}: ETS header size {header_size}, "
                                       f"{self.ndim} dimensions")
            fh.seek(extra_at)
            extra = fh.read(max(extra_size, 228))
            if extra[:3] != b"ETS":
                raise CorruptFileError(f"{path!r}: no ETS header at byte {extra_at}")
            (_v, component_type, self.components, self.color_space, self.compression,
             self.quality, tile_x, tile_y, tile_z) = struct.unpack_from("<9I", extra, 4)
            self.background = struct.unpack_from("<10I", extra, 40 + 17 * 4)
            self.pyramid = bool(struct.unpack_from("<I", extra, 40 + 27 * 4 + 4)[0])
            if component_type not in COMPONENT_TYPES:
                raise UnsupportedFormatError(f"{path!r}: ETS component type {component_type}")
            if self.compression not in COMPRESSIONS or self.compression == BMP:
                raise UnsupportedFormatError(
                    f"{path!r}: ETS compression {COMPRESSIONS.get(self.compression, self.compression)}"
                    " is not supported")
            self.dtype = np.dtype(COMPONENT_TYPES[component_type])
            #: what a tile the table lacks reads as: the background block's first values,
            #: one per component, as Bio-Formats fills such tiles (CellSensReader.decodeTile;
            #: an RGB VS200 slide stores 0xDCDCDC there: light grey around the tissue)
            n = min(self.components, 40 // self.dtype.itemsize)
            self.fill = np.frombuffer(struct.pack("<10I", *self.background),
                                      self.dtype.newbyteorder("<"), count=n).copy()
            self.tile = (tile_y, tile_x)
            fh.seek(chunks_at)
            record = 4 + 4 * self.ndim + 8 + 4 + 4
            table = fh.read(record * n_chunks)
            if len(table) != record * n_chunks:
                raise CorruptFileError(f"{path!r}: the ETS tile table ends early")
        entry = struct.Struct(f"<4x{self.ndim}iqI4x")
        self.tiles = {}                       # (level, coordinates without level) -> (offset, size)
        top = [0] * self.ndim
        for k in range(n_chunks):
            *coords, offset, size = entry.unpack_from(table, k * record)
            level = coords[-1] if self.pyramid else 0
            key = tuple(coords[:-1] if self.pyramid else coords)
            self.tiles[(level,) + key] = (offset, size)
            top = [max(a, b) for a, b in zip(top, coords)]
        #: the largest value of each coordinate
        self.top = tuple(top)
        self.levels = (top[-1] + 1) if self.pyramid else 1
        #: tiles' other coordinates, between tile row and level
        self.extra_dims = self.ndim - 2 - (1 if self.pyramid else 0)
        self._handles = ThreadHandles(path)
        self.decode_cache = DecodeCache(DECODE_CACHE_BYTES)
        self._lock = threading.Lock()

    def size_from_tiles(self) -> tuple:
        """(rows, cols) of level 0 if its last tiles were full."""
        return (self.top[1] + 1) * self.tile[0], (self.top[0] + 1) * self.tile[1]

    def read(self, level: int, extra: tuple, ys: slice, xs: slice, out: np.ndarray,
             fill=0) -> int:
        """Rows *ys*, columns *xs* of *level* at the other coordinates *extra*
        into *out* (rows, cols, components); a tile the table lacks is
        *fill*.  Returns the number of values decoded."""
        th, tw = self.tile

        def tasks():
            for ty in range(ys.start // th, (ys.stop - 1) // th + 1):
                for tx in range(xs.start // tw, (xs.stop - 1) // tw + 1):
                    y0, y1 = max(ys.start, ty * th), min(ys.stop, (ty + 1) * th)
                    x0, x1 = max(xs.start, tx * tw), min(xs.stop, (tx + 1) * tw)
                    dest = out[y0 - ys.start:y1 - ys.start, x0 - xs.start:x1 - xs.start]
                    key = (level, tx, ty) + tuple(extra)
                    part = (slice(y0 - ty * th, y1 - ty * th), slice(x0 - tx * tw, x1 - tx * tw))
                    cached = self.decode_cache.get(key)
                    if cached is not None:
                        dest[...] = cached[part]
                        continue
                    if key not in self.tiles:
                        dest[...] = fill
                        continue
                    yield lambda key=key, dest=dest, part=part: self._decode_into(key, dest, part)
        return _pool.run(tasks())

    def _decode_into(self, key, dest, part) -> int:
        offset, size = self.tiles[key]
        data = self._handles.read(offset, size)
        th, tw = self.tile
        tile = np.empty((th, tw, self.components), self.dtype.newbyteorder("="))
        if self.compression == RAW:
            raw = np.frombuffer(data, self.dtype, count=th * tw * self.components)
            tile[...] = raw.reshape(th, tw, self.components)
        else:
            decode = {JPEG: _codecs.jpeg_decode_into, JPEG_LOSSLESS: _codecs.jpeg_decode_into,
                      JPEG_2000: _codecs.jpeg2k_decode_into, PNG: _codecs.png_decode_into}
            got = decode[self.compression](data, tile if self.components > 1 else tile[..., 0])
            if (got[1], got[0]) != (th, tw):
                raise CorruptFileError(f"{self.path!r}: a tile decodes to {got[0]} x {got[1]}, "
                                       f"the ETS header says {tw} x {th}")
        dest[...] = tile[part]
        self.decode_cache.put(key, tile)
        return tile.size

    def close(self) -> None:
        self._handles.close()
        self.decode_cache.clear()
