"""An image whose channels come from several images of other pixel types: a
``ChannelStack``, the whole of a mixed-type image (docs/AXES.md, ragged images).

A file may store its channels in different pixel types (a CZI with a gray8 and
a gray16 channel).  Every channel stays readable exactly, in its own type, as a
*part* (``image.parts``: one image per pixel type, index ``part``); the whole
reads as one array in the type that holds every part's values unchanged
(``numpy.result_type``: uint8 + uint16 -> uint16, uint16 + float32 -> float32).

This is the interim model (Bugra, 2026-10-18): mixed types and ragged sampling
are becoming common enough to deserve a richer Image object, designed to be the
model for dyna too (format tracker, roadmap "Image model for ragged and
mixed-type images").
"""
from __future__ import annotations

import numpy as np

from ._base import Image
from ._metadata import Metadata


class ChannelStack(Image):
    """*parts* (images with the same axes and shape apart from ``c``) stacked
    along ``c`` in the order *channels*: one ``(part, channel of that part)``
    per channel of the whole."""

    def __init__(self, position: int, parts: list, channels: list, name: str, index: dict):
        first = parts[0]
        if "c" not in first.axes:
            raise ValueError("a channel stack needs a c axis")
        ci = first.axes.index("c")
        for part in parts[1:]:
            other = tuple(n for k, n in enumerate(part.shape) if k != ci)
            if part.axes != first.axes or other != tuple(n for k, n in enumerate(first.shape)
                                                         if k != ci):
                raise ValueError(f"{name!r}: parts differ in axes or shape "
                                 f"({first.axes} {first.shape}, {part.axes} {part.shape})")
        self._ci = ci
        self._channels = list(channels)
        shape = list(first.shape)
        shape[ci] = len(self._channels)
        dtype = np.result_type(*[p.dtype for p in parts])
        super().__init__(position, first.axes, shape, dtype, name=name, index=index)
        self._parts = list(parts)
        for part in parts:
            part._whole = self

    def _read_block(self, block):
        ci = self._ci
        cs = block[ci]
        out = np.empty(tuple(s.stop - s.start for s in block), dtype=self.dtype)
        read, k = 0, cs.start
        while k < cs.stop:                         # runs of consecutive channels of one part
            part, j = self._channels[k]
            n = 1
            while k + n < cs.stop and self._channels[k + n] == (part, j + n):
                n += 1
            key = list(block)
            key[ci] = slice(j, j + n)
            data, count = part._read_block(tuple(key))
            dest = [slice(None)] * out.ndim
            dest[ci] = slice(k - cs.start, k - cs.start + n)
            out[tuple(dest)] = data                 # widened, values unchanged
            read += count
            k += n
        return out, read

    @property
    def metadata(self):
        """The first part's metadata, with the channels in the whole's order."""
        if self._metadata is None:
            base = self._parts[0].metadata
            channels = tuple(part.metadata.channels[j] for part, j in self._channels)
            self._metadata = Metadata(pixel_sizes=base.pixel_sizes, position=base.position,
                                      channels=channels, index_values=base.index_values)
        return self._metadata

    def close(self) -> None:
        for part in self._parts:
            part.close()
