"""Backend selection: content first, file extension only as a fallback."""
from __future__ import annotations

import builtins
import os
import warnings
from typing import Optional

from ._base import SAMPLED_INDICES, File
from . import _parts
from ._errors import CorruptFileError, OpenInstead, UnsupportedFormatError
from .backends import (bdv, companion, czi, cv7000, dv, harmony, hdf5, ilastik, ims, incell, lif,
                       metamorph, metaxpress, mrc, nd2, oif, oir, omp2info, raster, tiff, vsi)

#: Tried in order: VSI before TIFF (a .vsi is a TIFF with a metadata tree);
#: magic numbers before MRC, which may also match by
#: extension alone (DeltaVision: an MRC-like header with its own ID); index files of plates (harmony) by their XML namespace;
#: OME companion files by their extension and root element.
#: Each backend module provides ``NAME``,
#: ``sniff(head: bytes, path: str) -> bool`` and
#: ``open(path, tiles="separate", resolutions="base") -> File``; a backend
#: that declares ``RENAMES`` / ``PLATES`` also takes ``rename=`` / ``plates=``.
#: HDF5 files by their structure (_hdf5.kind): BigDataViewer (also its XML),
#: ilastik, Luxendo / Fiji HDF5 (hdf5), then ims, which also refuses every other HDF5
#: file with what it holds.  MetaMorph ``.nd`` sets by their ``"NDInfoFile"`` line;
#: their TIFF / STK files are parts (``metamorph.owner``, asked after ``tiff.owner``
#: in this order, docs/COMPOSITE.md rule 5).
BACKENDS = [vsi, tiff, czi, bdv, ilastik, hdf5, ims, nd2, oir, omp2info, oif, lif, harmony,
            metaxpress, metamorph, incell, cv7000, companion, raster, dv, mrc]

#: Bytes read from the start of the file for sniffing.
HEAD_BYTES = 1024

#: mosaic tiles as separate images (default) or placed as one mosaic, where
#: the format defines how (CZI); see docs/AXES.md.  "mosaic", not "stitched":
#: tiles are put at their stored positions, nothing is registered or blended
#: (docs/PROCESSED.md, "Spatial operations")
TILE_POLICIES = ("separate", "mosaic")
#: old names still accepted, with a DeprecationWarning
DEPRECATED_TILE_POLICIES = {"stitched": "mosaic"}

#: resolution levels: full resolution only (default), or every level as its
#: own image indexed by "resolution" (image.resolutions gives a pyramid either way)
RESOLUTION_POLICIES = ("base", "all")


def open(path, backend: Optional[str] = None, *, tiles: str = "separate",  # noqa: A001
         resolutions: str = "base", rename: Optional[dict] = None, plates: bool = False,
         multifile: bool = True, axes: Optional[str] = None, samples: str = "keep",
         squeeze: bool = False) -> File:
    """Open *path* with the backend that recognises it (or the one named).

    Each image of the file is one spatial frame with axes ``t c z y x`` (+ ``s``),
    and an index saying which: scene, tile, rotation, view, and the
    dimensions sampled at the same points (illumination, phase, block,
    emission, excitation, lifetime, sequence).  *tiles*: ``"separate"`` gives
    each mosaic tile as its own image; ``"mosaic"`` places a scene's tiles
    at their stored positions as one image (no registration, no blending),
    for formats that define how (CZI), and is an error for tiled files of
    other formats.  ``"stitched"``, its old name, still works and warns.

    *rename* makes an index an array axis, for files that do not say what a
    dimension is: ``rename={"sequence": "z"}`` reads a plain multi-page TIFF
    as one z-stack.  The target is ``t``, ``c`` or ``z``; a format that
    cannot do it for that index raises.

    *plates* (opt-in for now, docs/HCS.md): ``True`` reads a file that is a
    plate as one: ``File.plates`` describes it, and its images are indexed by
    well and field instead of scene.  For a file that is not a plate (or a
    format without plates) it changes nothing: ``File.plates`` stays empty.

    *multifile*: a file of a multi-file OME set opens as the whole set
    (``True``, the default) or alone (``False``):

    - an OME-TIFF whose OME-XML lists planes in other files: the whole set,
      or only the planes stored in it, as the block of the set's image they
      form, its metadata from its own XML;
    - a member whose OME-XML is a ``BinaryOnly`` stub pointing to a metadata
      file (an OME companion, or another OME-TIFF): the set, as if that file
      were opened -- ``File.path`` is the metadata file and its metadata wins;
      a metadata file that cannot be followed (missing, unreadable, not
      listing the file) is a ``CorruptFileError``.  ``False``: the file alone,
      as its block of the set's image, with the metadata file's metadata
      (docs/COMPANION.md, CO-2; a loud warning when it cannot be followed).

    Missing members of a set warn and the rest reads, as for every file set.
    Not affected: parts of composite formats (an OIR continuation, an OIF
    plane, a VSI ``.ets``: pieces of one image, docs/COMPOSITE.md);
    Micro-Manager stacks, though OME-TIFFs, because they split their files by
    size, mid-timepoint, so a file is no block of the image; single-file
    OME-TIFFs.

    *axes* (``"canonical"``, or letters such as ``"tczyx"``), *samples* and
    *squeeze* present every image as ``Image.as_axes`` does.  Each image's
    ``.source`` is then its native image.

    A part of a multi-file (composite) format -- an OIR continuation, a
    DeltaVision log, NDTiff's index -- opens the whole image; a folder opens
    the image it is a part of, or the one image it holds (docs/COMPOSITE.md).
    ``File.path`` is then the main file, ``File.opened`` the *path* given.
    A *backend* named explicitly reads *path* itself."""
    if tiles in DEPRECATED_TILE_POLICIES:
        warnings.warn(f"tiles={tiles!r} is deprecated: use tiles="
                      f"{DEPRECATED_TILE_POLICIES[tiles]!r} (the tiles are placed at their "
                      "stored positions, not stitched)", DeprecationWarning, stacklevel=2)
        tiles = DEPRECATED_TILE_POLICIES[tiles]
    if tiles not in TILE_POLICIES:
        raise ValueError(f"tiles={tiles!r}: one of {TILE_POLICIES}")
    if resolutions not in RESOLUTION_POLICIES:
        raise ValueError(f"resolutions={resolutions!r}: one of {RESOLUTION_POLICIES}")
    rename = _check_rename(rename)
    opened = os.fspath(path)
    main = opened if backend is not None else _parts.main_file(opened, BACKENDS)
    image = _open(main, backend, tiles, resolutions, rename, bool(plates), bool(multifile))
    image.opened = opened
    left = sorted({name for index in image.indices for name in index if name in rename})
    if left:
        image.close()
        raise UnsupportedFormatError(
            f"rename: {path!r}: the {image.backend} backend cannot make {left} an axis")
    if axes is not None or samples != "keep" or squeeze:
        from ._axes import AxesImage
        try:
            image.images = [AxesImage(s, axes, samples=samples, squeeze=squeeze)
                            for s in image.images]
            image._bind_plates()            # plate navigation reaches the new images
        except BaseException:
            image.close()
            raise
    return image


def _check_rename(rename: Optional[dict]) -> dict:
    rename = {str(k).lower(): str(v).lower() for k, v in (rename or {}).items()}
    for name, axis in rename.items():
        if name not in SAMPLED_INDICES:
            raise ValueError(f"rename: {name!r} is not an index that can become an axis "
                             f"(those are {SAMPLED_INDICES})")
        if axis not in ("t", "c", "z"):
            raise ValueError(f"rename {name!r}: {axis!r} is not 't', 'c' or 'z'")
    if len(set(rename.values())) != len(rename):
        raise ValueError(f"rename {rename}: two indices onto one axis")
    return rename


def _open(path: str, backend: Optional[str], tiles: str, resolutions: str,
          rename: dict, plates: bool = False, multifile: bool = True,
          instead_of: Optional[str] = None) -> File:
    def call(module):
        options = {}
        if rename and getattr(module, "RENAMES", False):
            options["rename"] = rename
        if plates and getattr(module, "PLATES", False):
            options["plates"] = True
        if not multifile and getattr(module, "MULTIFILE", False):
            options["multifile"] = False
        try:
            return module.open(path, tiles, resolutions, **options)
        except OpenInstead as instead:
            # a member of a set: the set's metadata file opens it (multifile=True),
            # with the same options; once -- a metadata file is no member itself
            if instead_of is not None:
                raise CorruptFileError(
                    f"{instead_of!r} points to {path!r} for its metadata, which points on to "
                    f"{instead.path!r}: a chain of metadata files is not supported. "
                    f"multifile=False opens {instead_of!r} alone.") from None
            return _open(instead.path, None, tiles, resolutions, rename, plates, multifile,
                         instead_of=path)

    if backend is not None:
        for module in BACKENDS:
            if module.NAME == backend:
                return call(module)
        raise UnsupportedFormatError(
            f"unknown backend {backend!r}; available: {[m.NAME for m in BACKENDS]}")
    with builtins.open(path, "rb") as fh:
        head = fh.read(HEAD_BYTES)
    for module in BACKENDS:
        if module.sniff(head, path):
            return call(module)
    raise UnsupportedFormatError(f"no micro-reader backend recognises {path!r}")
