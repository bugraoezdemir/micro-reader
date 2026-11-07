"""Lambda (spectral) scans: one image per scan in every format (decision MR-5).

Leica, Olympus and OME record a lambda scan as an axis of its own, crossed with
the channels; a backend reads each band as an image (index ``emission``, its
wavelength in ``metadata.index_values``).  ``open`` gathers them here, before
images are named, so no backend has to know:

- by default the bands lie along ``c``, a channel's bands next to each other in
  band order, each channel with its ``emission_wavelength`` and its wavelength
  in its name (``LambdaStack``): NGFF has no room for a sixth axis, and it is
  what Zeiss and Bio-Formats do with them;
- with ``axes="native"``, or axes that name ``l``, the image has a real axis
  ``l`` before ``c`` (``LambdaImage``), and ``metadata.index_values["emission"]``
  lists the wavelengths.

Images of one scan that differ in axes, shape or type stay apart, as do those with
stored lower resolutions.  An excitation scan stays an index: its bands are not a
detection sampling.  Reads go band by band through the backends' own reads.
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np

from ._base import Image
from ._stack import ChannelStack

EMISSION = "emission"


def _values(band) -> dict:
    return {k: v for k, v in band.metadata.index_values.items() if k != EMISSION}


def _copy_names(whole, band) -> None:
    for attr in ("_own_name", "_label", "_main_scan"):
        if hasattr(band, attr):
            setattr(whole, attr, getattr(band, attr))


class LambdaStack(ChannelStack):
    """*bands* along ``c``: a channel's bands next to each other."""

    def __init__(self, bands: list, index: dict):
        first = bands[0]
        n = first.shape[first.axes.index("c")]
        super().__init__(first._position, bands, [(b, j) for j in range(n) for b in bands],
                         first.name, index)
        _copy_names(self, first)

    @property
    def metadata(self):
        if self._metadata is None:
            channels = []
            for band, j in self._channels:
                channel = band.metadata.channels[j]
                value = band.metadata.index_values.get(EMISSION)
                label = f"{value.value:g} nm" if value is not None \
                    else f"{EMISSION} {band.index[EMISSION]}"
                many = len(band.metadata.channels) > 1
                detector = channel.name or (f"Channel {j}" if many else None)
                channels.append(replace(channel, name=f"{detector} {label}" if detector else label,
                                        emission_wavelength=value))
            base = self._parts[0]
            self._metadata = replace(base.metadata, channels=tuple(channels),
                                     index_values=_values(base))
        return self._metadata


class LambdaImage(Image):
    """*bands* along an axis ``l`` before ``c`` (before ``z y x`` without one)."""

    def __init__(self, bands: list, index: dict):
        first = bands[0]
        at = first.axes.index("c") if "c" in first.axes else \
            next((i for i, a in enumerate(first.axes) if a in "zyx"), 0)
        self._at = at
        self._bands = list(bands)
        axes = first.axes[:at] + "l" + first.axes[at:]
        shape = first.shape[:at] + (len(bands),) + first.shape[at:]
        unit = None if first.read_unit is None else \
            first.read_unit[:at] + (1,) + first.read_unit[at:]
        super().__init__(first._position, axes, shape, first.dtype, name=first.name,
                         read_unit=unit, index=index)
        _copy_names(self, first)

    def _read_block(self, block):
        at, ls = self._at, block[self._at]
        rest = block[:at] + block[at + 1:]
        out = np.empty(tuple(s.stop - s.start for s in block), dtype=self.dtype)
        read = 0
        for k in range(ls.start, ls.stop):
            data, count = self._bands[k]._read_block(rest)
            dest = [slice(None)] * out.ndim
            dest[at] = k - ls.start
            out[tuple(dest)] = data
            read += count
        return out, read

    @property
    def metadata(self):
        if self._metadata is None:
            base = self._bands[0]
            values = _values(base)
            values[EMISSION] = tuple(b.metadata.index_values.get(EMISSION) for b in self._bands)
            self._metadata = replace(base.metadata, index_values=values)
        return self._metadata

    def close(self) -> None:
        for band in self._bands:
            band.close()


def _one_layout(bands: list) -> bool:
    first = bands[0]
    return all(b.axes == first.axes and b.shape == first.shape and b.dtype == first.dtype
               and b.level == 0 and not b.is_pyramidal for b in bands)


def gather(file, native: bool) -> None:
    """Replace the images of each lambda scan of *file* by one image (module
    docstring), the scan's first band's place in ``file.images``."""
    scans: dict = {}
    for image in file.images:
        index = dict(dict.items(image.index))
        if EMISSION in index:
            key = tuple((k, v) for k, v in index.items() if k != EMISSION)
            scans.setdefault(key, []).append(image)
    wholes, gone = {}, set()
    for key, bands in scans.items():
        if len(bands) < 2 or not _one_layout(bands):
            continue
        bands = sorted(bands, key=lambda b: dict.get(b.index, EMISSION))
        first = bands[0]
        gone.update(id(b) for b in bands)
        if native:
            wholes[id(first)] = LambdaImage(bands, dict(key))
            continue
        if "c" not in first.axes:
            from ._axes import AxesImage
            bands = [AxesImage(b, "canonical") for b in bands]
        wholes[id(first)] = LambdaStack(bands, dict(key))
    if wholes:
        file.images = [wholes.get(id(image), image) for image in file.images
                       if id(image) not in gone or id(image) in wholes]
        file._bind_plates()
