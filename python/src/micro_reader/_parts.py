"""Files made of several files (docs/COMPOSITE.md).

A *composite* format fixes where its other files are -- OIR continuations,
the DeltaVision log, NDTiff's index, OIF's storage folder, VSI's ETS files
-- so they are found by the format's rule, never by a search.  Opening any
part opens the whole image (Bugra, 2026-10-15):

- ``owner(path, head)``, offered by a composite backend, says from the path
  (and the first bytes, ``head``; None for a folder) whether *path* is a
  part, and of which main file: a ``Part``.  ``open()`` asks every backend
  before sniffing content, because a part may be a valid file of another
  format (an OIF plane is a plain TIFF).
- A part whose main file is missing: one that is an image on its own is read
  as itself, with a ``CompositeWarning``; one that is not raises.
- A folder opens the image it is a part of (a format's root: OIF storage,
  VSI ``_<name>_``, a Luxendo experiment); else the one image its direct
  contents hold (a plate or companion index first); else, when it holds none
  of its own, the one dataset found in its sub-folders (decision MR-6, down
  to ``FOLDER_DEPTH`` levels, each sub-folder taken as ``open`` would take
  it).  Several is an error listing them; nothing is merged.  An OME-Zarr
  store is refused, naming ome_zarr_pyramid.  A main "file" may be a folder
  itself (a Luxendo experiment): the backend whose ``sniff_folder`` claims
  it reads it.
- ``open(multifile=False)`` reads a part that is a ``member`` (``Part``) as
  itself.

Also shared here: the case-insensitive file lookup (files written on Windows
and copied elsewhere) and per-path file handles.
"""
from __future__ import annotations

import builtins
import os
import warnings
from typing import NamedTuple, Optional
from urllib.parse import unquote

from ._errors import CompositeWarning, CorruptFileError, MicroReaderError, UnsupportedFormatError
from ._io import ThreadHandles

#: bytes read from the start of a file for ``owner`` and sniffing (as _registry)
HEAD_BYTES = 1024
#: names shown in an error about a folder's candidates
LISTED = 10


class Part(NamedTuple):
    """*path* is a part of the composite whose main file is *main* (None:
    missing; *expected* names it); *standalone*: the part is an image on its
    own, read as itself when the main file is missing; *member*: a file
    that is read alone with ``open(multifile=False)`` (a Luxendo stack or
    link file: the whole otherwise); *hint*: what can be opened when *main*
    is missing."""
    main: Optional[str]
    expected: str
    standalone: bool = False
    member: bool = False
    hint: str = ""


#: the last folder ``only_file`` listed: (folder, modification time) -> its names
_last_listing: dict = {}


def only_file(folder: str, suffix: str) -> Optional[str]:
    """The one file of *folder* ending in *suffix* (lower case), None for none or
    several.  For ``owner`` on a file named like a plate's plane: the folder is
    listed once however many of its planes are asked about."""
    try:
        key = (folder, os.stat(folder).st_mtime_ns)
        if key not in _last_listing:
            names = os.listdir(folder)
            _last_listing.clear()
            _last_listing[key] = names
    except OSError:
        return None
    found = [n for n in _last_listing[key] if n.lower().endswith(suffix)]
    return os.path.join(folder, found[0]) if len(found) == 1 else None


def find(dirname: str, name: str) -> Optional[str]:
    """*name* in *dirname*: the exact name, else the one entry whose name
    differs only in letter case; None if there is none.  Two such entries:
    a ``CompositeWarning``, and None -- the rule names one file.  The path
    returned is spelled as the folder spells it: on a file system that ignores
    case, a name asked for in other letters (a BigDataViewer XML looked up
    from its lower-cased stem) still names the file as it is, so names built
    from it keep their case (gap hunt B-01: 'hisyfp-spim.xml angle 0')."""
    try:
        entries = os.listdir(dirname or ".")
    except OSError:
        exact = os.path.join(dirname, name)
        return exact if os.path.exists(exact) else None
    if name in entries:
        return os.path.join(dirname, name)
    matches = [e for e in entries if e.lower() == name.lower()]
    if len(matches) > 1:
        warnings.warn(CompositeWarning(
            f"{dirname!r}: {len(matches)} files match {name!r} apart from letter case "
            f"({sorted(matches)}): none is used"), stacklevel=3)
        return None
    return os.path.join(dirname, matches[0]) if matches else None


def resolve(dirname: str, ref: str, where: str) -> str:
    """A file a format references by a relative path (Leica XLIF / XLEF /
    XLCF): URL-quoted, possibly with backslashes and another letter case
    (written on Windows); each level looked up with ``find``.  Missing:
    ``CorruptFileError``."""
    rel = os.path.normpath(unquote(ref).replace("\\", "/"))
    path = os.path.join(dirname, rel)
    if os.path.exists(path):
        return path
    current = dirname
    for part in rel.split(os.sep):
        found = find(current, part)
        if found is None:
            raise CorruptFileError(f"{where}: referenced file {ref!r} not found")
        current = found
    return current


class Handles:
    """``ThreadHandles`` per file of a composite, opened on first use and
    closed together."""

    def __init__(self):
        self._by_path: dict = {}

    def get(self, path: str) -> ThreadHandles:
        handles = self._by_path.get(path)
        if handles is None:
            handles = self._by_path.setdefault(path, ThreadHandles(path))
        return handles

    def discard(self, path: str) -> None:
        """Close the handles of *path* (opened again on next use)."""
        handles = self._by_path.pop(path, None)
        if handles is not None:
            handles.close()

    def __iter__(self):
        return iter(list(self._by_path.values()))

    def __len__(self) -> int:
        return len(self._by_path)

    def close(self) -> None:
        for handles in list(self._by_path.values()):
            handles.close()
        self._by_path.clear()


def read_head(path: str) -> bytes:
    with builtins.open(path, "rb") as fh:
        return fh.read(HEAD_BYTES)


def part_of(path: str, head: Optional[bytes], backends) -> Optional[Part]:
    """The ``Part`` *path* is, by the first composite backend that owns it."""
    for module in backends:
        owner = getattr(module, "owner", None)
        part = owner(path, head) if owner is not None else None
        if part is not None:
            return part
    return None


def main_file(path: str, backends, multifile: bool = True, plates: bool = False) -> str:
    """The file ``open(path)`` reads: *path* itself, or the main file of the
    composite it is a part of, or -- for a folder -- the image the folder
    is a part of or holds (module docstring).  *multifile* False: a part
    that is a ``member`` is read itself.  *plates*: a folder of one
    NIS-Elements well-plate run is one dataset (``open(plates=True)``)."""
    if os.path.isdir(path):
        return _folder_image(path, backends, plates)
    part = part_of(path, read_head(path), backends)
    if part is None or (part.member and not multifile):
        return path
    if part.main is not None:
        return part.main
    if part.standalone:
        warnings.warn(CompositeWarning(
            f"{path!r} is a part of {part.expected!r}, which is missing: reading this "
            "part on its own"), stacklevel=4)
        return path
    raise CorruptFileError(f"{path!r} is a part of {part.expected!r}, which is missing"
                           + (f"; {part.hint}" if part.hint else ""))


def _sniffed(paths, backends) -> tuple:
    """(index files, images) among *paths*: each file's head read, parts
    left out (their main file is listed itself)."""
    indices, images = [], []
    for path in paths:
        try:
            head = read_head(path)
        except OSError:
            continue
        if part_of(path, head, backends) is not None:
            continue
        module = next((m for m in backends if m.sniff(head, path)), None)
        if module is None:
            continue
        (indices if getattr(module, "SET_INDEX", False) else images).append(path)
    return indices, images


#: formats micro-reader knows of and does not read: named in the error (gap hunt
#: B-07: they got the generic "no backend recognises"), and in a warning when a
#: folder opens the images beside them
NOT_READ = {
    ".ptu": "PicoQuant time-tagged (TTTR) photon data",
    ".phu": "a PicoQuant histogram file",
    ".sptw": "a PicoQuant SymPhoTime workspace",
}

#: how deep, and for how many datasets, a folder without an image of its own is
#: searched (decision MR-6): enough for a sample folder over its export
#: (plate > measurement > Images), bounded for a folder of many samples
FOLDER_DEPTH = 6
FOLDER_CANDIDATES = 25
#: what makes a folder an OME-Zarr store: ome_zarr_pyramid's ground, not micro-reader's
ZARR_MARKERS = ("zarr.json", ".zattrs", ".zgroup")


def is_zarr_store(folder: str) -> bool:
    """Whether *folder* is a Zarr store (an OME-Zarr image, plate or collection)."""
    if folder.rstrip("/\\").lower().endswith(".zarr"):
        return True
    return any(os.path.isfile(os.path.join(folder, m)) for m in ZARR_MARKERS)


def _zarr_refusal(folder: str) -> UnsupportedFormatError:
    return UnsupportedFormatError(
        f"{folder!r} is an OME-Zarr store: micro-reader reads microscopes' own formats; "
        f"read OME-Zarr with ome_zarr_pyramid (ome_zarr_pyramid.io.read)")


def _own_images(folder: str, backends) -> tuple:
    """(index files, images) of *folder* itself, not its sub-folders, and the
    folder's sub-folders.  A plate folder holds every plane: index files are
    looked for by name first and only those are read; every file is sniffed
    only when there is none."""
    with os.scandir(folder) as entries:              # the listing says file / folder: no stat
        listed = sorted(entries, key=lambda e: e.name)
        files = [e.path for e in listed if e.is_file()]
        subfolders = [e.path for e in listed if e.is_dir() and not e.name.startswith(".")]
    named = [p for p in files if any(getattr(m, "SET_INDEX", False) and
                                     getattr(m, "may_index", lambda _n: False)(os.path.basename(p))
                                     for m in backends)]
    indices, images = _sniffed(named, backends)
    if not indices:
        indices, images = _sniffed(files, backends)
    skipped = [os.path.basename(p) for p in files if p.lower().endswith(tuple(NOT_READ))]
    if skipped and (indices or images):
        warnings.warn(CompositeWarning(
            f"{folder!r}: micro-reader does not read {_listed(skipped, '.')} beside the images "
            "it opens here"), stacklevel=5)
    return indices, images, subfolders


def _one_run(folder: str, found: list, plates: bool) -> list:
    """*found* (a folder's images) as one dataset when, with *plates*, they are
    the files of one NIS-Elements well-plate run: its first file, which opens
    the run as a plate (gap hunt C-05: the folder listed its 360 files)."""
    if not plates or len(found) < 2 or not all(p.lower().endswith(".nd2") for p in found):
        return found
    from .backends._nd2_run import run_files
    run = {os.path.normcase(p) for p in run_files(folder)}
    return found[:1] if all(os.path.normcase(p) in run for p in found) else found


def _datasets_below(folder: str, backends, depth: int, out: list, stores: list,
                    plates: bool = False) -> None:
    """The datasets in *folder*'s sub-folders, each sub-folder as ``open``
    would take it: the one a format defines as its root (a part's main
    file), else its own index files or images; only a folder with neither is
    searched further.  At most ``FOLDER_CANDIDATES``; Zarr stores apart."""
    for sub in _own_images(folder, backends)[2]:
        if len(out) > FOLDER_CANDIDATES:
            return
        if is_zarr_store(sub):
            stores.append(sub)
            continue
        part = part_of(sub, None, backends)
        if part is not None:
            out.append(part.main if part.main is not None else sub)
            continue
        if any(getattr(m, "sniff_folder", lambda _p: False)(sub) for m in backends):
            out.append(sub)
            continue
        indices, images, _ = _own_images(sub, backends)
        if indices or images:
            out.extend(indices or _one_run(sub, images, plates))
        elif depth > 1:
            _datasets_below(sub, backends, depth - 1, out, stores, plates)


def _listed(paths, folder: str) -> str:
    names = [os.path.relpath(p, folder) if os.path.isabs(p) else p for p in paths]
    shown = ", ".join(names[:LISTED])
    more = len(names) > LISTED
    return shown + (f", ... ({len(names)}{'+' if len(names) > FOLDER_CANDIDATES else ''} in all)"
                    if more else "")


def _nd2_hint(found, plates: bool) -> str:
    """The way out for ND2 files that may be one NIS-Elements well-plate run."""
    if plates or not all(p.lower().endswith(".nd2") for p in found):
        return ""
    return ("; the files of one NIS-Elements well-plate run open as one plate with "
            "plates=True")


def _folder_image(folder: str, backends, plates: bool = False) -> str:
    """The image ``open(folder)`` reads (decision MR-6): the one the folder
    is a part of (a format's root), else the folder's own index file or
    image, else -- for a folder with none of its own, a sample folder over
    its export -- the one dataset found in its sub-folders.  Several: the
    error lists them."""
    part = part_of(folder, None, backends)
    if part is not None:
        if part.main is None:
            raise CorruptFileError(f"{folder!r} is a part of {part.expected!r}, which is missing")
        return part.main
    if is_zarr_store(folder):
        raise _zarr_refusal(folder)
    indices, images, _ = _own_images(folder, backends)
    found = indices or _one_run(folder, images, plates)
    if len(found) == 1:
        return found[0]
    if found:
        kind = "index files (plates, companions, image sets)" if indices else "images"
        raise MicroReaderError(f"{folder!r} holds {len(found)} {kind}: {_listed(found, folder)}; "
                               f"open one of them{_nd2_hint(found, plates)}")
    below, stores = [], []
    _datasets_below(folder, backends, FOLDER_DEPTH, below, stores, plates)
    if len(below) == 1:
        return below[0]
    if below:
        raise MicroReaderError(f"{folder!r} holds no image itself, and {len(below)}"
                               f"{'+' if len(below) > FOLDER_CANDIDATES else ''} datasets in "
                               f"its sub-folders: {_listed(below, folder)}; open one of them"
                               f"{_nd2_hint(below, plates)}")
    if stores:
        raise UnsupportedFormatError(
            f"{folder!r}: no image micro-reader reads; it holds OME-Zarr stores "
            f"({_listed(stores, folder)}), which ome_zarr_pyramid reads")
    raise UnsupportedFormatError(f"{folder!r}: no file in this folder or its sub-folders is an "
                                 "image micro-reader reads")


def other_images(folder: str, chosen: str, backends) -> list:
    """The names of the images in *folder* other than *chosen*, the one a
    folder opened (an index file is chosen over images): named when
    *chosen* fails, so the error shows the way out."""
    try:
        with os.scandir(folder) as entries:
            files = sorted(e.path for e in entries if e.is_file())
    except OSError:
        return []
    indices, images = _sniffed(files, backends)
    own = os.path.normcase(os.path.abspath(chosen))
    return [os.path.basename(p) for p in indices + images
            if os.path.normcase(os.path.abspath(p)) != own]
