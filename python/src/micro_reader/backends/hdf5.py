"""HDF5 stacks described by an ``element_size_um`` attribute (``[hdf5]`` extra,
h5py): Luxendo / Bruker light-sheet files and Fiji's HDF5 plugin (Vibez).

**Luxendo** (``.lux.h5``, github.com/Luxendo/luxendo-image): one stack per file
-- one time point, channel, camera and objective -- in the dataset ``Data``
(z, y, x), lower resolutions as ``Data_<a>_<b>_<c>``, ``element_size_um``
(z, y, x) and a JSON ``metadata`` dataset (``processingInformation``: channel,
voxel size, ...; ``metaData``: lasers, filter wheels).  One image, ``tczyx``
with t = c = 1.  Channel name: ``channel_description``; excitation: the one
laser switched on; emission: the centre of the detection filter's band
(``BP 459-499``, ``BP 525/50``).  No position: the stage positions and the
transform to sample coordinates disagree on the axes, unchecked on overlapping
stacks.  A whole acquisition (several time points, channels, cameras) is
opened through the Imaris file Luxendo writes next to it, whose time points and
channels link to the ``.lux.h5`` files (the ``ims`` backend).

**Fiji HDF5** (the Vibez plugin, ``File > Save As > HDF5``): datasets with
``element_size_um`` (z, y, x).  Datasets named ``t<n>/channel<n>`` (the
plugin's default) make one ``tczyx`` image; any other such dataset is an image
of its own (index ``scene``), ``zyx`` or ``yx`` -- more axes have no recorded
order and are refused.  An ``element_size_um`` of exactly 1, 1, 1 is what Fiji
writes for an image without calibration: no pixel sizes then (never 1.0).
"""
from __future__ import annotations

import json
import os
import re
import warnings

import numpy as np

from .._base import File, Image
from .._errors import CorruptFileError, MissingImagesWarning, UnsupportedFormatError
from .._metadata import Channel, Metadata, quantity
from . import _hdf5

NAME = "hdf5"
FORMATS = ("luxendo", "fiji-hdf5")
LEVEL = re.compile(r"^Data_(\d+)_(\d+)_(\d+)$")
TIME_CHANNEL = re.compile(r"^/?t(\d+)/channel(\d+)$")
#: a band-pass filter as filter wheels name it: "BP 459-499" (edges) or "BP 525/50" (centre / width)
BAND = re.compile(r"(\d+(?:\.\d+)?)\s*([-/])\s*(\d+(?:\.\d+)?)")


def sniff(head: bytes, path: str) -> bool:
    return _hdf5.is_hdf5(head) and _hdf5.kind(path) in FORMATS


def _sizes(ds) -> dict:
    """``element_size_um`` (z, y, x) as pixel sizes of the dataset's spatial
    axes; empty when absent, not positive, or exactly 1, 1, 1 (uncalibrated)."""
    raw = ds.attrs.get("element_size_um")
    if raw is None:
        return {}
    # str() first: a float32 attribute (Luxendo, Fiji) gives 0.26, not 0.25999999
    values = [float(str(v)) for v in np.asarray(raw).ravel()]
    if not values or all(v == 1.0 for v in values):
        return {}
    axes = "zyx"[-min(len(values), 3):]
    out = {}
    for axis, value in zip(axes, values[-3:]):
        q = quantity(value, "micrometer")
        if q is not None:
            out[axis] = q
    return out


class _StackImage(Image):
    """Datasets laid out as [t][c] (None: missing), each (z, y, x) or (y, x)."""

    def __init__(self, position, name, index, axes, datasets, shape, dtype, reader, key,
                 metadata):
        self._datasets = datasets
        self._reader = reader
        self._key = key
        self._meta = metadata
        self._missing = any(ds is None for row in datasets for ds in row)
        self._closed = False
        present = next(ds for row in datasets for ds in row if ds is not None)
        unit = present.chunks
        if unit is not None and "t" in axes:
            unit = (1, 1) + tuple(unit)
        super().__init__(position, axes, shape, dtype, name=name, read_unit=unit, index=index)

    def _read_block(self, block):
        if self._closed:
            raise ValueError("this HDF5 image has been closed: its file was closed "
                             "(File.close() or the end of a with block)")
        out = (np.zeros if self._missing else np.empty)(
            tuple(s.stop - s.start for s in block), dtype=self.dtype)
        read, jobs = 0, []
        if "t" not in self.axes:                         # one dataset, its own axes
            read += self._reader.plan(self._key, self._datasets[0][0], block, out, jobs)
            return out, read + self._reader.run(jobs)
        ts, cs, *spatial = block
        for t in range(ts.start, ts.stop):
            for c in range(cs.start, cs.stop):
                ds = self._datasets[t][c]
                if ds is None:
                    continue                             # missing: zeros
                dest = out[t - ts.start, c - cs.start]
                if ds.ndim == 2:                         # a plane: the image's z of 1
                    read += self._reader.plan((self._key, t, c), ds, tuple(spatial[1:]),
                                              dest[0], jobs)
                else:
                    read += self._reader.plan((self._key, t, c), ds, tuple(spatial), dest, jobs)
        return out, read + self._reader.run(jobs)

    def _read_metadata(self):
        return self._meta

    def close(self) -> None:
        self._closed = True


# -- Luxendo --------------------------------------------------------------------

def _luxendo_metadata(h5, ds) -> Metadata:
    try:
        text = h5["metadata"][()]
        info = json.loads(text.decode() if isinstance(text, bytes) else text)
    except (KeyError, ValueError, TypeError, AttributeError):
        info = {}
    process = info.get("processingInformation", {}) if isinstance(info, dict) else {}
    sizes = _sizes(ds)
    if not sizes:                                        # the JSON says the same, in words
        voxel = process.get("voxel_size_um") or {}
        for axis, key in (("x", "width"), ("y", "height"), ("z", "depth")):
            q = quantity(voxel.get(key), "micrometer")
            if q is not None:
                sizes[axis] = q
    meta = info.get("metaData", {}) if isinstance(info, dict) else {}
    lasers = [l for l in meta.get("lasers", []) if isinstance(l, dict) and l.get("on")
              and str(l.get("name", "")).upper() != "LED"]
    excitation = quantity(lasers[0].get("wavelength_nm"), "nanometer") if len(lasers) == 1 else None
    emission = None
    branch = (info.get("imagingBranch") or {}).get("detection", {}) if isinstance(info, dict) else {}
    wheels = {w.get("name"): w.get("selection") for w in meta.get("filterwheels", [])
              if isinstance(w, dict)}
    selected = [wheels.get(name) for name in branch.get("filterwheels", []) if wheels.get(name)]
    if len(selected) == 1:
        m = BAND.search(str(selected[0]))
        if m:
            a, sep, b = float(m.group(1)), m.group(2), float(m.group(3))
            emission = quantity((a + b) / 2 if sep == "-" else a, "nanometer")
    name = process.get("channel_description") or None
    return Metadata(pixel_sizes=sizes, channels=(Channel(
        name=name, emission_wavelength=emission, excitation_wavelength=excitation),))


def _luxendo(h5, path, reader, resolutions) -> list:
    base = h5["Data"]
    if base.ndim != 3:
        raise UnsupportedFormatError(f"{path!r}: Luxendo Data with {base.ndim} axes (z, y, x expected)")
    levels = sorted((n for n in h5 if LEVEL.match(n)),
                    key=lambda n: -int(np.prod(h5[n].shape)))
    meta = _luxendo_metadata(h5, base)
    listed = resolutions == "all"
    pyramid = []
    for r, name in enumerate(["Data"] + levels):
        ds = h5[name]
        index = {"scene": 0, "resolution": r} if (listed or r) else {"scene": 0}
        pyramid.append(_StackImage(0, os.path.basename(path), index, "tczyx", [[ds]],
                                   (1, 1) + ds.shape, ds.dtype.newbyteorder("="), reader, name,
                                   meta if r == 0 else None))
    for image in pyramid:
        image._resolutions = pyramid
        if image is not pyramid[0]:
            image._meta = None                           # scaled from level 0 by the base class
    images = pyramid if listed else pyramid[:1]
    for n, image in enumerate(images):
        image._position = n
    return images


# -- Fiji HDF5 (Vibez) ----------------------------------------------------------

def _datasets_with_sizes(h5) -> list:
    import h5py
    found = []
    h5.visititems(lambda n, o: found.append(n) if isinstance(o, h5py.Dataset)
                  and "element_size_um" in o.attrs else None)
    return sorted(found)


def _fiji(h5, path, reader) -> list:
    names = _datasets_with_sizes(h5)
    where = os.path.basename(path)
    matched = {n: TIME_CHANNEL.match(n) for n in names}
    if names and all(matched.values()):
        grid = {(int(m.group(1)), int(m.group(2))): n for n, m in matched.items()}
        times = sorted({t for t, _ in grid})
        channels = sorted({c for _, c in grid})
        shapes = {h5[n].shape for n in names}
        dtypes = {h5[n].dtype.newbyteorder("=") for n in names}
        if len(shapes) == 1 and len(dtypes) == 1 and len(next(iter(shapes))) in (2, 3):
            shape = next(iter(shapes))
            spatial = "zyx"[-len(shape):]
            datasets = [[h5[grid[(t, c)]] if (t, c) in grid else None for c in channels]
                        for t in times]
            absent = sum(ds is None for row in datasets for ds in row)
            if absent:
                warnings.warn(MissingImagesWarning(
                    f"{where}: {absent} of {len(times) * len(channels)} t / channel datasets "
                    "are missing: read as zeros"), stacklevel=4)
            first = h5[names[0]]
            meta = Metadata(pixel_sizes=_sizes(first),
                            channels=tuple(Channel() for _ in channels))
            full = (len(times), len(channels)) + (shape if len(shape) == 3 else (1,) + shape)
            return [_StackImage(0, where, {"scene": 0}, "tczyx", datasets, full,
                                dtypes.pop(), reader, "fiji", meta)]
    images = []
    for n, name in enumerate(names):
        ds = h5[name]
        if ds.ndim not in (2, 3):
            raise UnsupportedFormatError(
                f"{where}: dataset {name!r} has {ds.ndim} axes; a Fiji HDF5 dataset records "
                "no axis order beyond z, y, x")
        meta = Metadata(pixel_sizes=_sizes(ds), channels=(Channel(),))
        images.append(_StackImage(n, f"{where} {name}", {"scene": n}, "zyx"[-ds.ndim:], [[ds]],
                                  ds.shape, ds.dtype.newbyteorder("="), reader, name, meta))
    return images


class _Hdf5File(File):
    backend = NAME

    def __init__(self, path, images, h5, reader, fmt):
        self.format = fmt
        super().__init__(path, images)
        self._h5 = h5
        self._reader = reader

    def close(self) -> None:
        super().close()
        self._reader.close()
        self._h5.close()


def open(path: str, tiles: str = "separate", resolutions: str = "base") -> _Hdf5File:  # noqa: A001 - backend protocol
    h5py = _hdf5.h5py_module("reading HDF5 files")
    try:
        h5 = h5py.File(path, "r")
    except OSError as exc:
        raise CorruptFileError(f"{path!r}: {exc}") from exc
    reader = _hdf5.ChunkReader(path)
    try:
        fmt = _hdf5.classify(h5)
        if fmt == "luxendo":
            images = _luxendo(h5, path, reader, resolutions)
        elif fmt == "fiji-hdf5":
            images = _fiji(h5, path, reader)
        else:
            raise UnsupportedFormatError(f"{path!r}: not a Luxendo or Fiji HDF5 file")
    except BaseException:
        reader.close()
        h5.close()
        raise
    return _Hdf5File(path, images, h5, reader, fmt)
