"""Backend selection: content first, file extension only as a fallback."""
from __future__ import annotations

import builtins
import os
from typing import Optional

from ._base import SAMPLED_INDICES, File
from ._errors import UnsupportedFormatError
from .backends import (companion, czi, cv7000, harmony, ims, incell, lif, metaxpress, mrc,
                       nd2, oir, raster, tiff)

#: Tried in order: magic numbers before MRC, which may also match by
#: extension alone; index files of plates (harmony) by their XML namespace;
#: OME companion files by their extension and root element.
#: Each backend module provides ``NAME``,
#: ``sniff(head: bytes, path: str) -> bool`` and
#: ``open(path, tiles="separate", resolutions="base") -> File``; a backend
#: that declares ``RENAMES`` / ``PLATES`` also takes ``rename=`` / ``plates=``.
BACKENDS = [tiff, czi, ims, nd2, oir, lif, harmony, metaxpress, incell, cv7000, companion,
            raster, mrc]

#: Bytes read from the start of the file for sniffing.
HEAD_BYTES = 1024

#: mosaic tiles as separate images (default) or composed, where the format
#: defines how (CZI); see docs/AXES.md
TILE_POLICIES = ("separate", "stitched")

#: resolution levels: full resolution only (default), or every level as its
#: own image indexed by "resolution" (image.resolutions gives a pyramid either way)
RESOLUTION_POLICIES = ("base", "all")


def open(path, backend: Optional[str] = None, *, tiles: str = "separate",  # noqa: A001
         resolutions: str = "base", rename: Optional[dict] = None, plates: bool = False,
         axes: Optional[str] = None, samples: str = "keep", squeeze: bool = False) -> File:
    """Open *path* with the backend that recognises it (or the one named).

    Each image of the file is one spatial frame with axes ``t c z y x`` (+ ``s``),
    and an index saying which: scene, tile, rotation, view, and the
    dimensions sampled at the same points (illumination, phase, block,
    emission, excitation, lifetime, sequence).  *tiles*: ``"separate"`` gives
    each mosaic tile as its own image; ``"stitched"`` composes a scene's
    tiles, for formats that define how (CZI), and is an error for tiled files
    of other formats.

    *rename* makes an index an array axis, for files that do not say what a
    dimension is: ``rename={"sequence": "z"}`` reads a plain multi-page TIFF
    as one z-stack.  The target is ``t``, ``c`` or ``z``; a format that
    cannot do it for that index raises.

    *plates* (opt-in for now, docs/HCS.md): ``True`` reads a file that is a
    plate as one: ``File.plates`` describes it, and its images are indexed by
    well and field instead of scene.  For a file that is not a plate (or a
    format without plates) it changes nothing: ``File.plates`` stays empty.

    *axes* (``"canonical"``, or letters such as ``"tczyx"``), *samples* and
    *squeeze* present every image as ``Image.as_axes`` does.  Each image's
    ``.source`` is then its native image."""
    if tiles not in TILE_POLICIES:
        raise ValueError(f"tiles={tiles!r}: one of {TILE_POLICIES}")
    if resolutions not in RESOLUTION_POLICIES:
        raise ValueError(f"resolutions={resolutions!r}: one of {RESOLUTION_POLICIES}")
    rename = _check_rename(rename)
    image = _open(os.fspath(path), backend, tiles, resolutions, rename, bool(plates))
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
          rename: dict, plates: bool = False) -> File:
    def call(module):
        options = {}
        if rename and getattr(module, "RENAMES", False):
            options["rename"] = rename
        if plates and getattr(module, "PLATES", False):
            options["plates"] = True
        return module.open(path, tiles, resolutions, **options)

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
