"""Backend selection: content first, file extension only as a fallback."""
from __future__ import annotations

import builtins
import os
import warnings
from typing import Optional

from ._base import SAMPLED_INDICES, File
from . import _parts
from ._errors import CorruptFileError, OpenInstead, UnsupportedFormatError
from .backends import (bdv, cellomics, columbus, companion, czi, cv7000, dv, flex, harmony, hdf5, ics, ilastik, ims, incell,
                       jdce, klb, lif, luxendo, metamorph, metaxpress, micromanager, mrc, nd2, ndpis, oif, oir,
                       omp2info, raster, scanr, tiff, vsi)

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
#: in this order, docs/COMPOSITE.md rule 5).  ICS headers by their ``ics_version`` line;
#: ``.ids`` pixel files and Huygens Remote Manager's side files are their parts.  Hamamatsu
#: ``.ndpis`` set files by their ``[NanoZoomer Digital Pathology Image Set]`` line.
#: Luxendo experiment folders (a folder holding ``raw/``) before BigDataViewer and Imaris:
#: their ``bdv.xml`` / ``.ims`` link files, ``raw/`` and its ``.lux.h5`` stacks are parts
#: (``luxendo.owner``); the experiment is read as a folder (``sniff_folder``).
#: Micro-Manager image-per-file acquisitions by a ``metadata.txt`` starting with its
#: ``"Summary"``; their plane TIFFs, other positions' ``metadata.txt`` and folders are
#: parts (``micromanager.owner``, after ``tiff.owner``: MMStack names never collide).
BACKENDS = [vsi, flex, tiff, czi, luxendo, bdv, ilastik, hdf5, ims, nd2, oir, omp2info, oif, lif, harmony, columbus, scanr, cellomics, jdce,
            metaxpress, metamorph, micromanager, incell, cv7000, companion, ndpis, ics, raster, klb, dv, mrc]

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
#: the canonical array axes (``_axes.STANDARD``), the default since 0.4.0
STANDARD_AXES = "tczyx"


def open(path, backend: Optional[str] = None, *, tiles: str = "separate",  # noqa: A001
         resolutions: str = "base", rename: Optional[dict] = None, plates: bool = True,
         multifile: bool = True, axes: Optional[str] = "canonical", samples: str = "channels",
         squeeze: bool = False) -> File:
    """Open *path* with the backend that recognises it (or the one named).

    Each image of the file is one spatial frame with axes ``t c z y x`` (+ ``s``),
    and an index saying which: series, tile, rotation, view, and the
    dimensions sampled at the same points (illumination, phase, block,
    emission, excitation, lifetime, sequence).  *tiles*: ``"separate"`` gives
    each mosaic tile as its own image; ``"mosaic"`` places a series' tiles
    at their stored positions as one image (no registration, no blending),
    for formats that define how (CZI), and is an error for tiled files of
    other formats.  ``"stitched"``, its old name, still works and warns.

    *rename* makes an index an array axis, for files that do not say what a
    dimension is: ``rename={"sequence": "z"}`` reads a plain multi-page TIFF
    as one z-stack.  The target is ``t``, ``c`` or ``z``; a format that
    cannot do it for that index raises.

    *plates* (docs/HCS.md): ``True``, the default, reads a file that is a plate
    as one: ``File.plates`` describes it, and its images are indexed by well and
    field instead of series.  For a file that is not a plate (or a format
    without plates) it changes nothing: ``File.plates`` stays empty.  ``False``
    reads a plate acquired in a generic format (CZI, ND2, OME-TIFF) by its series;
    plate formats are plates either way.

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

    *axes*: ``"canonical"`` (the default since 0.4.0, decision MR-4) gives every
    image ``t c z y x`` (then ``s``), axes the file lacks with size 1;
    ``"native"`` (or None) keeps the order the backend reports (ICS ``ctzyx``,
    Micro-Manager OME-TIFF ``tzcyx``, a 2-D TIFF ``yx``); letters such as
    ``"zyx"`` ask for exactly those.  *axes*, *samples* and *squeeze* present
    every image as ``Image.as_axes`` does; an image already in the order asked
    for is not wrapped.  A presented image's ``.source`` is its native image.  *samples*: ``"channels"`` (the
    default since 0.3.0) folds RGB samples into the channels -- an RGB image
    has ``c`` 3, channels "<name> R" / "G" / "B" -- and only images with
    samples are presented anew; ``"keep"`` keeps them as a trailing ``s``
    axis (``Image.as_axes``' default).

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
    from ._axes import SAMPLE_POLICIES
    if samples not in SAMPLE_POLICIES:
        raise ValueError(f"samples={samples!r}: one of {SAMPLE_POLICIES}")
    opened = os.path.normpath(os.fspath(path))
    main = opened if backend is not None else _parts.main_file(opened, BACKENDS, bool(multifile),
                                                                bool(plates))
    try:
        image = _open(main, backend, tiles, resolutions, rename, bool(plates), bool(multifile))
    except (CorruptFileError, UnsupportedFormatError) as exc:
        # a folder opened the file it holds that is chosen first (an index
        # file over images): when that one fails, name the others
        if backend is None and main != opened and os.path.isdir(opened) and \
                os.path.dirname(os.path.abspath(main)) == os.path.abspath(opened):
            others = _parts.other_images(opened, main, BACKENDS)
            if others:
                raise type(exc)(f"{exc} (the folder also holds {', '.join(others[:10])}: "
                                "open one of them)") from exc
        raise
    if backend is None and multifile and main != opened and _unlisted_image(image, opened):
        # a file named like a member that its set does not list (a TIFF named like a
        # Harmony plane, no part of the index): it opens as itself, as group_files lists it
        # (a single), not as the set, which would leave its pixels out (review cycle 1, N-03)
        image.close()
        return open(path, tiles=tiles, resolutions=resolutions, rename=rename, plates=plates,
                    multifile=False, axes=axes, samples=samples, squeeze=squeeze)
    image.opened = opened
    left = sorted({name for index in image.indices for name in index if name in rename})
    if left:
        image.close()
        raise UnsupportedFormatError(
            f"rename: {path!r}: the {image.backend} backend cannot make {left} an axis")
    if axes is not None and str(axes).lower() == "native":
        axes = None
    from ._lambda import gather
    from ._naming import assign_names
    # one image per lambda scan (MR-5): a real l axis only when asked for
    gather(image, axes is None or "l" in str(axes).lower().replace("canonical", ""))
    assign_names(image)                     # one naming rule for every format (MR-2)
    # an image is presented anew when asked to, or when its samples fold; one
    # already in the order asked for stays the backend's own (no wrapper)

    def as_asked(s) -> bool:
        if squeeze or (samples == "channels" and "s" in s.axes):
            return False
        if axes is None:
            return True
        wanted = STANDARD_AXES if str(axes).lower() == "canonical" else str(axes).lower()
        return s.axes == wanted

    if any(not as_asked(s) for s in image.images):
        from ._axes import AxesImage
        try:
            image.images = [s if as_asked(s) else
                            AxesImage(s, axes, samples=samples, squeeze=squeeze)
                            for s in image.images]
            image._bind_plates()            # plate navigation reaches the new images
        except BaseException:
            image.close()
            raise
    return image


def _check_rename(rename: Optional[dict]) -> dict:
    if rename is not None and not isinstance(rename, dict):
        raise TypeError(f"rename={rename!r}: a dict from index to axis, such as "
                        "{'sequence': 'z'}")
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


def _unlisted_image(file: File, opened: str) -> bool:
    """Whether *opened*, a file that opened the set *file* by its name, is an image the set
    does not list: neither one of its members nor one of its side files. A side file that is
    no image (an index's text or XML companion) still opens the set."""
    if not os.path.isfile(opened):
        return False
    module = next((m for m in BACKENDS if getattr(m, "NAME", None) == file.backend), None)
    if not getattr(module, "SET_INDEX", False):
        return False                # not a set an index lists (a Luxendo experiment's parts)
    head = _parts.read_head(opened)
    part = _parts.part_of(opened, head, BACKENDS)
    if part is None or not part.member:
        return False                # a structural part (a side file, an experiment's .xml)
    key = os.path.normcase(os.path.abspath(opened))
    try:
        listed = file._member_files() | file._side_files()
    except Exception:                                   # noqa: BLE001 - a set that cannot say
        return False
    if not listed or key in {os.path.normcase(os.path.abspath(p)) for p in listed}:
        return False
    return any(module.sniff(head, opened) for module in BACKENDS
               if getattr(module, "NAME", None) != file.backend)


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
            # the metadata file may itself be a part of a composite (scanR's
            # data/metadata.ome.xml is its plate descriptor's): that opens
            return _open(_parts.main_file(instead.path, BACKENDS), None, tiles, resolutions,
                         rename, plates, multifile, instead_of=path)

    if backend is not None:
        for module in BACKENDS:
            if module.NAME == backend:
                return call(module)
        raise UnsupportedFormatError(
            f"unknown backend {backend!r}; available: {[m.NAME for m in BACKENDS]}")
    if os.path.isdir(path):              # a main file that is a folder (a Luxendo experiment)
        for module in BACKENDS:
            sniff_folder = getattr(module, "sniff_folder", None)
            if sniff_folder is not None and sniff_folder(path):
                return call(module)
        raise UnsupportedFormatError(f"no micro-reader backend reads the folder {path!r} "
                                     "as one image")
    with builtins.open(path, "rb") as fh:
        head = fh.read(HEAD_BYTES)
    for module in BACKENDS:
        if module.sniff(head, path):
            return call(module)
    name = os.path.basename(path).lower()
    folder = os.path.dirname(os.path.abspath(path))
    if name in _parts.ZARR_MARKERS or _parts.is_zarr_store(folder):
        raise _parts._zarr_refusal(folder)
    for suffix, what in _parts.NOT_READ.items():
        if name.endswith(suffix):
            raise UnsupportedFormatError(f"{path!r} is {what}, which micro-reader does not "
                                         f"read yet")
    raise UnsupportedFormatError(f"no micro-reader backend recognises {path!r}")


