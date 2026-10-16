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
- A folder opens the image it is a part of; else the one image its direct
  contents hold (a plate or companion index first); none or several is an
  error listing them.  No search in subfolders, no merging.

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
    own, read as itself when the main file is missing."""
    main: Optional[str]
    expected: str
    standalone: bool = False


def find(dirname: str, name: str) -> Optional[str]:
    """*name* in *dirname*: the exact name, else the one entry whose name
    differs only in letter case; None if there is none.  Two such entries:
    a ``CompositeWarning``, and None -- the rule names one file."""
    exact = os.path.join(dirname, name)
    if os.path.exists(exact):
        return exact
    try:
        entries = os.listdir(dirname or ".")
    except OSError:
        return None
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


def main_file(path: str, backends) -> str:
    """The file ``open(path)`` reads: *path* itself, or the main file of the
    composite it is a part of, or -- for a folder -- the image the folder
    is a part of or holds (module docstring)."""
    if os.path.isdir(path):
        return _folder_image(path, backends)
    part = part_of(path, read_head(path), backends)
    if part is None:
        return path
    if part.main is not None:
        return part.main
    if part.standalone:
        warnings.warn(CompositeWarning(
            f"{path!r} is a part of {part.expected!r}, which is missing: reading this "
            "part on its own"), stacklevel=4)
        return path
    raise CorruptFileError(f"{path!r} is a part of {part.expected!r}, which is missing")


def _folder_image(folder: str, backends) -> str:
    part = part_of(folder, None, backends)
    if part is not None:
        if part.main is None:
            raise CorruptFileError(f"{folder!r} is a part of {part.expected!r}, which is missing")
        return part.main
    indices, images = [], []
    for name in sorted(os.listdir(folder)):
        path = os.path.join(folder, name)
        if not os.path.isfile(path):
            continue
        try:
            head = read_head(path)
        except OSError:
            continue
        if part_of(path, head, backends) is not None:
            continue                                 # a part: its main file is listed itself
        module = next((m for m in backends if m.sniff(head, path)), None)
        if module is None:
            continue
        (indices if getattr(module, "SET_INDEX", False) else images).append(path)
    found = indices or images
    if len(found) == 1:
        return found[0]
    if not found:
        raise UnsupportedFormatError(f"{folder!r}: no file in this folder is an image "
                                     "micro-reader reads")
    names = [os.path.basename(p) for p in found]
    shown = ", ".join(names[:LISTED]) + (f", ... ({len(names)} in all)" if len(names) > LISTED else "")
    kind = "plate / companion index files" if indices else "images"
    raise MicroReaderError(f"{folder!r} holds {len(found)} {kind}: {shown}; open one of them")
