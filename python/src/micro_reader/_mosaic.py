"""Tiles placed on one canvas: a ``Mosaic``, the ``tiles="mosaic"`` image of a
format whose tiles are separate images (docs/PROCESSED.md, "Spatial
operations").

Each tile is placed at its stored (nominal) position, a whole number of
pixels from the canvas corner; nothing is registered, resampled or blended.
Overlaps are fused by "overwrite": the later tile in the given order supplies
the pixel (the same rule as CZI's mosaics, libCZI's).  Canvas pixels no tile
covers read as *fill*.  Every output pixel is a copy of one tile pixel.
"""
from __future__ import annotations

import numpy as np

from ._base import Image


class Mosaic(Image):
    """*tiles*: ``[(image, y, x)]`` in drawing order (later on top), with each
    tile's corner in canvas pixels.  The tiles share axes, pixel type and
    every size but ``y`` / ``x``.  The canvas starts at the smallest corner
    (``origin``: that corner, in the tiles' pixel grid)."""

    def __init__(self, position: int, tiles: list, name: str, index: dict, fill=0):
        first = tiles[0][0]
        if "y" not in first.axes or "x" not in first.axes:
            raise ValueError(f"{name!r}: mosaic tiles need y and x axes ({first.axes})")
        yi, xi = first.axes.index("y"), first.axes.index("x")
        other = [k for k in range(first.ndim) if k not in (yi, xi)]
        for tile, _y, _x in tiles[1:]:
            if tile.axes != first.axes or tile.dtype != first.dtype or \
                    [tile.shape[k] for k in other] != [first.shape[k] for k in other]:
                raise ValueError(f"{name!r}: mosaic tiles differ ({first.axes} {first.shape} "
                                 f"{first.dtype}, {tile.axes} {tile.shape} {tile.dtype})")
        y0 = min(y for _t, y, _x in tiles)
        x0 = min(x for _t, _y, x in tiles)
        height = max(y + t.shape[yi] for t, y, _x in tiles) - y0
        width = max(x + t.shape[xi] for t, _y, x in tiles) - x0
        shape = list(first.shape)
        shape[yi], shape[xi] = height, width
        super().__init__(position, first.axes, shape, first.dtype, name=name, index=index)
        self._yi, self._xi = yi, xi
        self._tiles = [(t, y - y0, x - x0) for t, y, x in tiles]
        #: the canvas corner in the tiles' pixel grid
        self.origin = (y0, x0)
        self._fill = fill
        self._closed = False

    @property
    def tiles(self) -> list:
        """``[(image, y, x)]``: the tiles and their corners on this canvas."""
        return list(self._tiles)

    def _read_block(self, block):
        if self._closed:
            raise ValueError(f"{self.name!r} has been closed: its file was closed "
                             "(File.close() or the end of a with block)")
        yi, xi = self._yi, self._xi
        ys, xs = block[yi], block[xi]
        out = np.full(tuple(s.stop - s.start for s in block), self._fill, dtype=self.dtype)
        read = 0
        for tile, ty, tx in self._tiles:                 # drawing order: later on top
            y0, y1 = max(ys.start, ty), min(ys.stop, ty + tile.shape[yi])
            x0, x1 = max(xs.start, tx), min(xs.stop, tx + tile.shape[xi])
            if y0 >= y1 or x0 >= x1:
                continue
            key = list(block)
            key[yi], key[xi] = slice(y0 - ty, y1 - ty), slice(x0 - tx, x1 - tx)
            data, count = tile._read_block(tuple(key))
            dest = [slice(None)] * out.ndim
            dest[yi] = slice(y0 - ys.start, y1 - ys.start)
            dest[xi] = slice(x0 - xs.start, x1 - xs.start)
            out[tuple(dest)] = data
            read += count
        return out, read

    def close(self) -> None:
        self._closed = True
        for tile, _y, _x in self._tiles:
            tile.close()
