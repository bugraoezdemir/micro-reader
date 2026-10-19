"""Hamamatsu NDPIS: a NanoZoomer fluorescence scan, one ``.ndpi`` per
channel, listed by an ``.ndpis`` set file (an INI text)::

    [NanoZoomer Digital Pathology Image Set]
    NoImages=3
    Image0=test3-DAPI 2 (387) .ndpi
    Image1=...

A pointed set, index only (docs/COMPOSITE.md): the ``.ndpis`` names its
members, which say nothing back and open alone as ordinary NDPI slides.
Opened, the ``.ndpis`` is one slide image with one channel per member, at
every resolution level the members store; the members' associated pictures
(Macro, Map) follow as further scenes, from the first member (NDP.scan writes
the same ones into every file).

NDP.scan stores each fluorescence channel as an RGB rendering in the colour
of its emission (checked on OME's Hamamatsu-NDPI/manuel set: the DAPI file's
signal is in its blue samples, FITC's in green, TRITC's in red); NDPI tag
65451 records the emission wavelength and tag 65434 the filter set (the
channel's name).  A channel is the member's band of its wavelength -- blue
380-490 nm, green 490-580, red 580-780 -- as Bio-Formats reads it (equal to
its planes on that set).  A member without a fluorescence wavelength (0: a
colour image) is refused: no sample yet.
"""
from __future__ import annotations

import builtins
import os
import warnings
from typing import Optional

import numpy as np

from .. import _parts
from .._base import File, Image
from .._errors import CorruptFileError, MissingImagesWarning, UnsupportedFormatError
from .._metadata import Channel, Metadata, quantity

NAME = "ndpis"
#: group_files: the set file is an index naming its members
SET_INDEX = True
MAGIC = b"[NanoZoomer Digital Pathology Image Set]"
#: NDPI private tags: the filter set (channel name), the emission wavelength (nm)
TAG_CHANNEL, TAG_EMISSION = 65434, 65451
#: emission wavelength (nm) -> the RGB sample NDP.scan renders it into
BANDS = ((380.0, 490.0, 2), (490.0, 580.0, 1), (580.0, 780.0, 0))


def may_index(name: str) -> bool:
    return name.lower().endswith(".ndpis")


def sniff(head: bytes, path: str) -> bool:
    return head.lstrip(b"\xef\xbb\xbf").startswith(MAGIC)


def read_set(path: str) -> list:
    """The member file names, in channel order."""
    with builtins.open(path, "rb") as fh:
        raw = fh.read(1 << 20)
    text = raw.lstrip(b"\xef\xbb\xbf").decode("utf-8", "replace")
    values = {}
    for line in text.splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            values[key.strip().lower()] = value.strip()
    try:
        n = int(values.get("noimages", ""))
    except ValueError as exc:
        raise CorruptFileError(f"{path!r}: an NDPIS set file without a whole-number NoImages") from exc
    names = []
    for k in range(n):
        name = values.get(f"image{k}")
        if not name:
            raise CorruptFileError(f"{path!r}: NoImages={n}, but Image{k} is missing")
        names.append(name)
    if not names:
        raise CorruptFileError(f"{path!r}: an NDPIS set file listing no images")
    return names


def _band(wavelength: Optional[float], path: str) -> int:
    if wavelength:
        for low, high, band in BANDS:
            if low < wavelength <= high:
                return band
    raise UnsupportedFormatError(
        f"{path!r}: an NDPIS member whose emission wavelength ({wavelength}) names no colour band "
        "(a colour image in a fluorescence set?): not read yet")


class _Member:
    """One channel's ``.ndpi``: opened lazily through the TIFF backend."""

    def __init__(self, path: str):
        import tifffile
        self.path = path
        with tifffile.TiffFile(path) as tif:                 # tag values are read while open
            tags = tif.pages[0].tags
            name = tags.get(TAG_CHANNEL)
            wave = tags.get(TAG_EMISSION)
            self.name = (str(name.value).strip() or None) if name is not None else None
            self.wavelength = float(wave.value) if wave is not None else None
        self.band = _band(self.wavelength, path)
        from . import tiff
        self.file = tiff.open(path)

    @property
    def slide(self) -> Image:
        return self.file.images[0]


class _NdpisImage(Image):
    """One resolution level of the slide: channel k = member k's band."""

    def __init__(self, members: list, level: int, path: str, index: dict, name: str):
        first = next(m for m in members if m is not None)
        base = first.slide.resolutions[level]
        if base.axes != "yxs":
            raise UnsupportedFormatError(f"{path!r}: NDPIS members of axes {base.axes!r} "
                                         "(RGB renderings, 'yxs', expected)")
        super().__init__(0, "cyx", (len(members),) + base.shape[:2], base.dtype, name=name, index=index)
        self._members, self._level, self._path = members, level, path
        for m in members:
            if m is not None and m.slide.resolutions[level].shape != base.shape:
                raise CorruptFileError(f"{path!r}: member {os.path.basename(m.path)!r} is "
                                       f"{m.slide.resolutions[level].shape} at level {level}, "
                                       f"{os.path.basename(first.path)!r} {base.shape}")

    def _read_block(self, block):
        cs, ys, xs = block
        out = np.zeros(tuple(s.stop - s.start for s in block), dtype=self.dtype)
        read = 0
        for i, c in enumerate(range(cs.start, cs.stop)):
            member = self._members[c]
            if member is None:                     # missing: zeros (warned at open)
                continue
            level = member.slide.resolutions[self._level]
            out[i] = level[ys, xs, member.band]
            read += out[i].size
        return out, read

    def _read_metadata(self):
        first = next(m for m in self._members if m is not None)
        base = first.slide.resolutions[self._level].metadata
        channels = tuple(Channel() if m is None else
                         Channel(name=m.name, emission_wavelength=quantity(m.wavelength, "nanometer"))
                         for m in self._members)
        return Metadata(pixel_sizes=base.pixel_sizes, position=base.position, channels=channels)

    def _member_paths(self) -> set:
        return {m.path for m in self._members if m is not None}


class _NdpisFile(File):
    backend = NAME

    def __init__(self, path: str, images: list, members: list):
        super().__init__(path, images)
        self._members = members

    def close(self) -> None:
        super().close()
        for m in self._members:
            if m is not None:
                m.file.close()


def open(path: str, tiles: str = "separate", resolutions: str = "base") -> _NdpisFile:  # noqa: A001 - backend protocol
    folder = os.path.dirname(os.path.abspath(path))
    members, missing = [], []
    try:
        for name in read_set(path):
            found = _parts.find(folder, name)
            if found is None:
                members.append(None)
                missing.append(name)
            else:
                members.append(_Member(found))
        if all(m is None for m in members):
            raise CorruptFileError(f"{path!r}: none of its NDPI files is there ({missing})")
        if missing:
            warnings.warn(MissingImagesWarning(
                f"{path!r}: {len(missing)} of its {len(members)} NDPI files are missing "
                f"({missing}): their channels read as zeros"), stacklevel=4)
        first = next(m for m in members if m is not None)
        stem = os.path.splitext(os.path.basename(path))[0]
        n_levels = len(first.slide.resolutions)
        index = {"scene": 0}
        pyramid = [_NdpisImage(members, k, path, dict(index, resolution=k) if (k or resolutions == "all")
                               else dict(index), stem) for k in range(n_levels)]
        for level in pyramid:
            level._resolutions = pyramid
        images = pyramid if resolutions == "all" else [pyramid[0]]
        # the associated pictures, from the first member, as further scenes
        for k, extra in enumerate(first.file.images[1:], 1):
            extra._position = len(images)
            extra.index = dict(extra.index, scene=k)
            images.append(extra)
        return _NdpisFile(path, images, members)
    except BaseException:
        for m in members:
            if m is not None:
                m.file.close()
        raise
