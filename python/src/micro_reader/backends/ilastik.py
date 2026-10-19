"""ilastik HDF5 (``[hdf5]`` extra, h5py): datasets that carry an ``axistags``
attribute -- ilastik's exports (``exported_data``: raw data, probabilities,
segmentations), its input data files, and tools that write the same layout
(steinbock's ``img``).

``axistags`` is vigra's JSON description of the dataset's axes, outermost
first::

    {"axes": [{"key": "z", "typeFlags": 2, "resolution": 0, "description": ""},
              {"key": "y", ...}, {"key": "x", ...}, {"key": "c", "typeFlags": 1, ...}]}

Each such dataset is one image (index ``scene``, in name order), with the
dataset's own axis order (``zyxc``, ``tzyxc``, ``cyx``, ...: ``as_axes``
gives another).  The ``resolution`` of an axis has no unit, so no pixel sizes
are given (docs/METADATA.md: never guess).  An ilastik project (``.ilp``) is
not an image and is refused by the ``ims`` backend's catch-all.

Pixels are read through ``_hdf5.ChunkReader``: chunks decoded in parallel,
contiguous datasets read as runs of bytes.
"""
from __future__ import annotations

import json
import os

import numpy as np

from .._base import File, Image
from .._errors import CorruptFileError, UnsupportedFormatError
from . import _hdf5

NAME = "ilastik"
#: vigra axis keys micro-reader takes (vigra also knows e.g. frequency-domain
#: keys, which no image file of ilastik carries)
AXIS_KEYS = "tczyx"


def sniff(head: bytes, path: str) -> bool:
    return _hdf5.is_hdf5(head) and _hdf5.kind(path) == NAME


def parse_axistags(text, ndim: int, where: str) -> str:
    """The axis letters of an ``axistags`` attribute, outermost first."""
    try:
        tags = json.loads(text)
        keys = [str(axis["key"]).lower() for axis in tags["axes"]]
    except (TypeError, ValueError, KeyError) as exc:
        raise CorruptFileError(f"{where}: axistags is not vigra's JSON ({exc})") from exc
    stray = [k for k in keys if k not in AXIS_KEYS]
    if stray:
        raise UnsupportedFormatError(f"{where}: axistags has axes {stray}; micro-reader "
                                     f"reads {list(AXIS_KEYS)}")
    if len(set(keys)) != len(keys):
        raise CorruptFileError(f"{where}: axistags names an axis twice ({keys})")
    if len(keys) != ndim:
        raise CorruptFileError(f"{where}: axistags names {len(keys)} axes ({''.join(keys)}) "
                               f"for a {ndim}-D dataset")
    return "".join(keys)


class _IlastikImage(Image):
    """One dataset, in its own axis order."""

    def __init__(self, position: int, path: str, name: str, ds, reader: _hdf5.ChunkReader):
        where = f"{os.path.basename(path)}:{name}"
        axes = parse_axistags(_hdf5.attr_text(ds.attrs, "axistags"), ds.ndim, where)
        if ds.dtype.kind not in "biuf":
            raise UnsupportedFormatError(f"{where}: pixel type {ds.dtype} is not an image type")
        self._ds = ds
        self._key = name
        self._reader = reader
        self._closed = False
        super().__init__(position, axes, ds.shape, ds.dtype,
                         name=name if name != "exported_data" else os.path.basename(path),
                         read_unit=ds.chunks, index={"scene": position})

    def _read_block(self, block):
        if self._closed:
            raise ValueError("this ilastik image has been closed: its file was closed "
                             "(File.close() or the end of a with block)")
        out = np.empty(tuple(s.stop - s.start for s in block), dtype=self.dtype)
        jobs = []
        read = self._reader.plan(self._key, self._ds, block, out, jobs)
        return out, read + self._reader.run(jobs)

    def close(self) -> None:
        self._closed = True


class _IlastikFile(File):
    backend = NAME
    format = NAME

    def __init__(self, path, images, h5, reader):
        super().__init__(path, images)
        self._h5 = h5
        self._reader = reader

    def close(self) -> None:
        super().close()
        self._reader.close()
        self._h5.close()


def open(path: str, tiles: str = "separate", resolutions: str = "base") -> _IlastikFile:  # noqa: A001 - backend protocol
    """Every dataset with ``axistags`` as an image (*tiles*, *resolutions*:
    nothing to do, ilastik files have neither)."""
    h5py = _hdf5.h5py_module("reading ilastik HDF5 files")
    try:
        h5 = h5py.File(path, "r")
    except OSError as exc:
        raise CorruptFileError(f"{path!r}: {exc}") from exc
    reader = _hdf5.ChunkReader(path)
    try:
        names = _hdf5.axistags_datasets(h5)
        if not names:
            raise UnsupportedFormatError(f"{path!r}: no dataset with ilastik axistags")
        images = [_IlastikImage(n, path, name, h5[name], reader) for n, name in enumerate(names)]
    except BaseException:
        reader.close()
        h5.close()
        raise
    return _IlastikFile(path, images, h5, reader)
