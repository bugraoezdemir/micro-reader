"""Imaris (.ims): HDF5 with a fixed layout (``[hdf5]`` extra, h5py).

    /DataSet/ResolutionLevel r/TimePoint t/Channel c/Data      3D (Z, Y, X), chunked
    /DataSet/ResolutionLevel r/TimePoint t/Channel c  .attrs   ImageSizeZ/Y/X

Two things differ from reading the datasets through h5py:

- **Padding is cropped.**  Imaris pads every ``Data`` dataset to whole HDF5
  chunks (a 424 x 424 image stored as 512 x 512, zeros beyond); the real size
  is in the ``ImageSize`` attributes.  Images have the real size, as
  Bio-Formats and Imaris show it.
- **Chunks are read and decoded here, in parallel** (``_hdf5.ChunkReader``).

One image, full resolution, axes ``tczyx``; every ``ResolutionLevel r`` is in
``image.resolutions``, and an image of its own with ``resolutions="all"``
(index ``{"series": 0, "resolution": r}``).

This backend also answers for HDF5 files of no layout micro-reader reads (it is
the last HDF5 backend tried): they are refused with the datasets they hold, and
without h5py every HDF5 file comes here to say which extra to install.
"""
from __future__ import annotations

import os
import re
import warnings

import numpy as np

from .. import _parts
from .._base import File, Image
from .._errors import CorruptFileError, MissingImagesWarning, UnsupportedFormatError
from .._metadata import Channel, Metadata, Quantity, color_hex, quantity
from . import _hdf5
from ._hdf5 import attr_text as _attr_str

NAME = "ims"


def sniff(head: bytes, path: str) -> bool:
    # every HDF5 file the other HDF5 backends (tried before this one) did not take
    return _hdf5.is_hdf5(head)


def _image_size(ds) -> tuple:
    """Real (Z, Y, X) of a ``Data`` dataset: its Channel group's ImageSize
    attributes, else the dataset shape."""
    sizes = []
    for axis, stored in zip("ZYX", ds.shape):
        try:
            size = int(_attr_str(ds.parent.attrs, f"ImageSize{axis}"))
        except (TypeError, ValueError):
            size = stored
        sizes.append(size if 0 < size <= stored else stored)
    return tuple(sizes)


def _numbered(group, prefix) -> list:
    """Indices of ``'{prefix} {n}'`` members, which must be 0..n-1."""
    found = sorted(int(k[len(prefix) + 1:]) for k in group
                   if k.startswith(prefix + " ") and k[len(prefix) + 1:].isdigit())
    if found != list(range(len(found))) or not found:
        raise UnsupportedFormatError(f"Imaris file with {prefix} groups {found}")
    return found


def _linked(group, name: str):
    """The dataset *name*, or None when it is missing -- also when a link on
    the way leads to a file that is not there."""
    try:
        return group[name] if name in group else None
    except (KeyError, OSError):
        return None


def _link_target(group, t: int, c: int) -> str:
    """Where ``TimePoint t / Channel c`` (or its Data) links to, for messages."""
    import h5py
    try:
        for name in (f"TimePoint {t}/Channel {c}", f"TimePoint {t}/Channel {c}/Data"):
            parent, last = name.rsplit("/", 1)
            link = group[parent].get(last, getlink=True)
            if isinstance(link, h5py.ExternalLink):
                return link.filename
    except (KeyError, OSError):
        pass
    return f"TimePoint {t}/Channel {c}"


#: FileConverter's multi-file Imaris: <name>.ims (or .h5) links each time
#: point / channel to <name>--C<cc>--T<ttttt>.h5
PART_NAME = re.compile(r"^(?P<stem>.+)--C\d+--T\d+\.h5$", re.IGNORECASE)


def may_own(folder: str, name: str, folders) -> bool:
    """``group_files``' cheap test: a part file of a multi-file Imaris set."""
    m = PART_NAME.match(name)
    names = folders.names(folder) if m else ()
    return bool(m) and any(m["stem"].lower() + s in names for s in (".ims", ".h5"))


def owner(path: str, head):
    """A part file ``<name>--Cxx--Txxxxx.h5`` belongs to ``<name>.ims`` (or the
    ``.h5`` copy FileConverter also writes); it is no image on its own."""
    m = PART_NAME.match(os.path.basename(path))
    if not m or head is None:
        return None
    folder = os.path.dirname(path)
    for suffix in (".ims", ".h5"):
        main = _parts.find(folder, m["stem"] + suffix)
        if main is not None:
            return _parts.Part(main, main)
    return _parts.Part(None, os.path.join(folder, m["stem"] + ".ims"))


class _ImsImage(Image):
    """One resolution level: all time points and channels, axes 'tczyx'."""

    def __init__(self, h5, reader: _hdf5.ChunkReader, path: str, level: int = 0,
                 key: dict = None):
        base = f"DataSet/ResolutionLevel {level}"
        if base not in h5:
            raise UnsupportedFormatError(f"{path!r}: HDF5 file without Imaris {base!r}")
        group = h5[base]
        times = _numbered(group, "TimePoint")
        channels = _numbered(group[f"TimePoint {times[0]}"], "Channel")
        #: [t][c] -> dataset, or None: a part of a multi-file Imaris set (an
        #: external link: FileConverter's <name>--Cxx--Txxxxx.h5, Luxendo's
        #: .lux.h5) that is missing reads as zeros
        self._datasets = [[_linked(group, f"TimePoint {t}/Channel {c}/Data") for c in channels]
                          for t in times]
        self.missing = [(t, c) for t, row in enumerate(self._datasets)
                        for c, ds in enumerate(row) if ds is None]
        all_ds = [ds for row in self._datasets for ds in row if ds is not None]
        if not all_ds:
            raise CorruptFileError(f"{path!r}: none of its Imaris data is there (missing "
                                   f"parts, e.g. {_link_target(group, times[0], channels[0])!r})")
        sizes = {_image_size(ds) for ds in all_ds}
        dtypes = {ds.dtype.newbyteorder("=") for ds in all_ds}
        if len(sizes) != 1 or len(dtypes) != 1:
            raise UnsupportedFormatError(
                f"{path!r}: channels / time points differ in size {sizes} or type {dtypes}")
        self._path = path
        self._h5 = h5
        self._channel_ids = channels
        self._reader = reader
        self._level = level
        self._closed = False
        chunks = all_ds[0].chunks
        super().__init__(level, "tczyx", (len(times), len(channels), *sizes.pop()),
                         dtypes.pop(), name=os.path.basename(path),
                         read_unit=(1, 1, *chunks) if chunks else None,
                         index=key if key is not None else {"series": 0})

    def _layout(self, t: int, c: int) -> _hdf5.Layout:
        """How the dataset of time point *t*, channel *c* is stored."""
        return self._reader.layout((self._level, t, c), self._datasets[t][c])

    @property
    def decode_cache(self):
        """Decoded chunks kept for reuse, shared by the file's levels."""
        return self._reader.cache

    def _read_block(self, block):
        if self._closed:
            raise ValueError("this Imaris image has been closed: its file was closed "
                             "(File.close() or the end of a with block)")
        ts, cs, zs, ys, xs = block
        make = np.zeros if self.missing else np.empty
        out = make(tuple(s.stop - s.start for s in block), dtype=self.dtype)
        read, jobs = 0, []
        for t in range(ts.start, ts.stop):
            for c in range(cs.start, cs.stop):
                if self._datasets[t][c] is None:
                    continue                    # a missing part: zeros
                read += self._reader.plan((self._level, t, c), self._datasets[t][c],
                                          (zs, ys, xs), out[t - ts.start, c - cs.start], jobs)
        return out, read + self._reader.run(jobs)

    def close(self) -> None:
        self._closed = True
        self._reader.close()

    def _read_metadata(self):
        with self._reader.lock:
            return _metadata(self._h5, self.shape, self._channel_ids)


def _metadata(h5, shape: tuple, channel_ids: list):
    """``DataSetInfo``: the volume's extent (``ExtMin`` / ``ExtMax`` 0-2 = x,
    y, z, edges of the outer pixels, in ``Unit``) over the image size gives
    the pixel sizes and the position of the first pixel; ``TimeInfo`` the
    time points; ``Channel c`` name, colour (``r g b`` in 0-1) and
    wavelengths (nm)."""
    if "DataSetInfo" not in h5:
        return None
    info = h5["DataSetInfo"]
    image = info["Image"].attrs if "Image" in info else {}
    unit = _attr_str(image, "Unit") or "um"
    sizes, position = {}, {}
    nz, ny, nx = shape[2:]
    for k, (axis, n) in enumerate((("x", nx), ("y", ny), ("z", nz))):
        lo, hi = (quantity(_attr_str(image, f"Ext{e}{k}"), unit, positive=False)
                  for e in ("Min", "Max"))
        if lo is None or hi is None:
            continue
        size = quantity((hi.value - lo.value) / n, lo.unit)
        if size:
            sizes[axis] = size
            if axis != "z":
                position[axis] = Quantity(lo.value + size.value / 2, lo.unit)
    times = _time_points(info["TimeInfo"].attrs) if "TimeInfo" in info else []
    if len(times) > 1 and shape[0] > 1:
        step = quantity((times[-1] - times[0]).total_seconds() / (len(times) - 1), "second")
        if step:
            sizes["t"] = step
    channels = []
    for c in channel_ids:
        attrs = info[f"Channel {c}"].attrs if f"Channel {c}" in info else {}
        name = _attr_str(attrs, "Name")
        color = None
        rgb = (_attr_str(attrs, "Color") or "").split()
        if len(rgb) >= 3:
            try:
                color = color_hex(tuple(float(v) for v in rgb[:3]))
            except ValueError:
                pass
        channels.append(Channel(
            name=None if not name or name == "(name not specified)" else name,
            color=color,
            emission_wavelength=quantity(_attr_str(attrs, "LSMEmissionWavelength"), "nanometer"),
            excitation_wavelength=quantity(_attr_str(attrs, "LSMExcitationWavelength"),
                                           "nanometer")))
    return Metadata(pixel_sizes=sizes, position=position, channels=tuple(channels))


def _time_points(attrs) -> list:
    """``TimePoint1`` .. ``TimePointN`` ('2020-02-29 00:07:15.969') as datetimes."""
    import datetime
    out = []
    for n in range(1, len(attrs) + 1):
        text = _attr_str(attrs, f"TimePoint{n}")
        if text is None:
            break
        try:
            out.append(datetime.datetime.fromisoformat(text.strip()))
        except ValueError:
            return []
    return out


class _ImsFile(File):
    backend = NAME

    def __init__(self, path, images, h5, reader):
        super().__init__(path, images)
        self._h5 = h5
        self._reader = reader

    def close(self) -> None:
        super().close()
        self._reader.close()
        self._h5.close()


#: the HDF5 layouts another backend reads (``_hdf5.kind``): what the file is, and
#: which backend reads it -- for ``backend="ims"`` on such a file
OTHER_LAYOUTS = {"bdv": ("a BigDataViewer HDF5 file", "bdv"),
                 "ilastik": ("an ilastik HDF5 file", "ilastik"),
                 "luxendo": ("a Luxendo HDF5 file", "hdf5"),
                 "fiji-hdf5": ("a Fiji HDF5 file", "hdf5")}


def _refuse(h5, path: str, found) -> UnsupportedFormatError:
    """The error for an HDF5 file that is not Imaris: the layout it is and the
    backend that reads it, or that micro-reader reads none of its layout."""
    if found == "ilastik-project":
        return UnsupportedFormatError(
            f"{path!r} is an ilastik project (.ilp), not an image: open the data it was "
            "made from, or the files ilastik exported")
    if found in OTHER_LAYOUTS:
        what, backend = OTHER_LAYOUTS[found]
        return UnsupportedFormatError(
            f"{path!r} is {what}, not an Imaris file: the {backend!r} backend reads it "
            f"(backend={backend!r}, or no backend= to choose by content)")
    shown = "; ".join(_hdf5.datasets(h5, 8)) or "no datasets"
    return UnsupportedFormatError(
        f"{path!r}: an HDF5 file of no layout micro-reader reads (Imaris, ilastik, "
        f"BigDataViewer, Luxendo, Fiji HDF5). Its datasets: {shown}")


def open(path: str, tiles: str = "separate", resolutions: str = "base") -> _ImsFile:  # noqa: A001 - backend protocol name
    h5py = _hdf5.h5py_module("reading HDF5 files (Imaris, ilastik, BigDataViewer)")
    try:
        h5 = h5py.File(path, "r")
    except OSError as exc:
        raise CorruptFileError(f"{path!r}: {exc}") from exc
    reader = _hdf5.ChunkReader(path)
    try:
        found = _hdf5.classify(h5)
        if found != "ims":
            raise _refuse(h5, path, found)
        levels = sorted(int(name.rsplit(" ", 1)[1]) for name in h5.get("DataSet", {})
                        if name.startswith("ResolutionLevel ")
                        and name.rsplit(" ", 1)[1].isdigit())
        if 0 not in levels:                     # e.g. only "ResolutionLevel x" groups
            shown = sorted(name for name in h5["DataSet"] if name.startswith("ResolutionLevel "))
            raise CorruptFileError(f"{path!r}: Imaris DataSet without 'ResolutionLevel 0' "
                                   f"(its levels: {shown[:8]})")
        listed = resolutions == "all"
        pyramid = [_ImsImage(h5, reader, path, r, key={"series": 0, "resolution": r}
                             if (listed or r) else {"series": 0}) for r in levels]
        for image in pyramid:
            image._resolutions = pyramid
        images = pyramid if listed else pyramid[:1]
        for n, image in enumerate(images):
            image._position = n
        lost = pyramid[0].missing
        if lost:
            t, c = lost[0]
            group = h5["DataSet/ResolutionLevel 0"]
            warnings.warn(MissingImagesWarning(
                f"{path!r}: {len(lost)} of {pyramid[0].shape[0] * pyramid[0].shape[1]} time "
                f"point / channel parts are missing (e.g. "
                f"{_link_target(group, t, pyramid[0]._channel_ids[c])!r}): read as zeros"),
                stacklevel=4)
    except BaseException:
        reader.close()
        h5.close()
        raise
    return _ImsFile(path, images, h5, reader)
