"""Which of many files make one image (docs/COMPOSITE.md, "Grouping files").

``group_files`` takes files (or folders, walked down) and returns:

- a ``FileGroup`` for each *composite* image of which some part was given --
  the main file and its parts, by the format's own rules (OIF storage
  folders, VSI ``_<name>_`` folders, DeltaVision logs, OIR continuations,
  Micro-Manager and NDTiff stack files);
- a ``FileGroup`` for each *index* file -- a plate index, an OME companion,
  a Leica XLEF / XLCF / XLIF -- with the files its images are read from as
  ``members``.  Members are images on their own as well; a given file that
  an index names is listed in that index's group only (Bugra, 2026-10-15);
- every other file as a plain path (``singles``), untouched.

Cost (a folder walk may hold millions of files): index files are recognised
by name first and only those are read and opened; every other file is
checked against the composite formats' name rules (``may_own``: no reading;
folder listings cached), and only a file a rule may apply to has its first
bytes read for the full rule (``owner``).  A plain file is never read.
Whether a member's content fits its index is checked when it is read
(``File.check_members`` checks them all up front)."""
from __future__ import annotations

import os
import warnings
from dataclasses import dataclass, field
from typing import Iterable, Optional, Union

from . import _parts

INDEX, COMPOSITE = "index", "composite"


@dataclass(frozen=True)
class FileGroup:
    """Files that make one image (``group_files``).

    *kind*: ``"composite"`` -- a main file and its parts, found by the
    format's rules -- or ``"index"`` -- an index file and the files its
    images are read from.  *main*: the file to open; None when a
    composite's main file is missing (``open()`` of a part that is an image
    on its own still reads that part alone, with a warning).  *expected*:
    where the main file is, or should be.  *format*: the backend that reads
    the group.  *parts*: the given files that belong to *main* (not *main*
    itself); for an index, other index files of the same images (CellVoyager
    ``.wpi`` / ``.mrf`` / ``.mlf``) and its side files (CellVoyager correction
    images; they still open alone).  *members* (index): every file its
    images are read from.  *notes* (index): what opening it said -- missing
    files, or why it could not be read."""
    kind: str
    main: Optional[str]
    expected: str
    format: Optional[str] = None
    parts: tuple = ()
    members: frozenset = field(default_factory=frozenset)
    notes: tuple = ()


@dataclass(frozen=True)
class Grouping:
    """``group_files``' result: *groups* (``FileGroup``), *singles* (paths of
    every other file, as given)."""
    groups: list
    singles: list


def _key(path: str) -> str:
    return os.path.normcase(os.path.abspath(path))


class _Folders:
    """What ``may_own`` asks about a folder (absolute), worked out once per
    folder: the lower-case names in it, and the names of it and its parents."""

    def __init__(self):
        self._names: dict = {}
        self._bases: dict = {}

    def names(self, folder: str) -> frozenset:
        found = self._names.get(folder)
        if found is None:
            try:
                found = frozenset(e.lower() for e in os.listdir(folder))
            except OSError:
                found = frozenset()
            self._names[folder] = found
        return found

    def base(self, folder: str, up: int = 0) -> str:
        """The name of *folder* (*up* = 0), of its parent (1), ..."""
        found = self._bases.get((folder, up))
        if found is None:
            path = folder
            for _ in range(up):
                path = os.path.dirname(path)
            found = self._bases[(folder, up)] = os.path.basename(path)
        return found


def _expand(paths) -> dict:
    """key -> (path as given, its folder (absolute), its name) of every file
    in *paths* (folders walked down), the first spelling kept.  Paths are
    joined once per folder: Windows path functions cost microseconds each."""
    if isinstance(paths, (str, os.PathLike)):
        paths = [paths]
    out = {}
    for given in paths:
        given = os.fspath(given)
        if os.path.isdir(given):
            for root, dirs, files in os.walk(given):
                dirs.sort()
                folder = os.path.abspath(root)
                key_folder = os.path.normcase(folder) + os.sep
                for name in sorted(files):
                    out.setdefault(key_folder + os.path.normcase(name),
                                   (root + os.sep + name, folder, name))
        elif os.path.exists(given):
            folder, name = os.path.split(os.path.abspath(given))
            out.setdefault(_key(given), (given, folder, name))
        else:
            raise FileNotFoundError(given)
    return out


def group_files(paths: Union[str, os.PathLike, Iterable]) -> Grouping:
    """Group *paths* (files, or folders, walked down with all their files)
    into the images they make: see the module docstring."""
    from ._registry import BACKENDS, open as open_file
    files = _expand(paths)
    rank = {m.NAME: k for k, m in enumerate(BACKENDS)}

    # 1. index files, recognised by name, then by content, then opened -- unless
    #    a composite rule makes one a part (a CellVoyager folder's Harmony export)
    indexers = [m for m in BACKENDS if hasattr(m, "may_index")]
    owners = [m for m in BACKENDS if hasattr(m, "may_own")]
    folders = _Folders()
    found = []                                   # (rank, path, key, module, member keys, members, notes)
    for key, (path, folder, name) in files.items():
        modules = [m for m in indexers if m.may_index(name)]
        if not modules:
            continue
        try:
            head = _parts.read_head(path)
        except OSError:
            continue
        if any(m.owner(path, head) is not None for m in owners
               if m.may_own(folder, name, folders)):
            continue                             # a part: pass 2 places it
        module = next((m for m in modules if m.sniff(head, path)), None)
        if module is None:
            continue
        members, sides, notes = frozenset(), frozenset(), []
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            try:
                with open_file(path, backend=module.NAME) as f:
                    members = frozenset(f._member_files())
                    sides = frozenset(f._side_files())
            except Exception as exc:             # noqa: BLE001 - reported on the group
                notes.append(f"cannot be read: {type(exc).__name__}: {exc}")
        notes = [str(w.message) for w in caught] + notes
        found.append((rank[module.NAME], path, key, module, frozenset(map(_key, members)),
                      members, tuple(notes), sides))
    member_keys = set().union(*(f[4] for f in found)) if found else set()
    consumed = set(member_keys)
    by_members: dict = {}                        # the same members: one set of images
    groups = []
    for entry in sorted(found, key=lambda f: (f[0], f[2])):
        _rank, path, key, module, keys, members, notes, sides = entry
        consumed.add(key)
        if key in member_keys:
            continue                             # named by another index: one of its members
        # side files (correction images) given here are the set's parts, not singles
        side_parts = [files[k][0] for k in map(_key, sides) if k in files and k not in member_keys]
        consumed.update(map(_key, side_parts))
        twin = by_members.get(keys) if keys else None
        if twin is not None:
            twin["parts"] += [path] + [p for p in side_parts if p not in twin["parts"]]
            continue
        group = {"kind": INDEX, "main": path, "expected": path, "format": module.NAME,
                 "parts": side_parts, "members": members, "notes": notes}
        if keys:
            by_members[keys] = group
        groups.append(group)

    # 2. composite parts: name rules first, the full rule only where one may apply
    composites: dict = {}                        # expected key -> group
    singles = []                                 # (key, path)
    for key, (path, folder, name) in files.items():
        if key in consumed:
            continue
        modules = [m for m in owners if m.may_own(folder, name, folders)]
        part, module = None, None
        if modules:
            try:
                head = _parts.read_head(path)
            except OSError:
                head = None
            for module in modules:
                part = module.owner(path, head)
                if part is not None:
                    break
        if part is None:
            singles.append((key, path))
            continue
        group = composites.get(_key(part.expected))
        if group is None:
            group = composites[_key(part.expected)] = {
                "kind": COMPOSITE, "main": part.main, "expected": part.expected,
                "format": module.NAME, "parts": [], "members": frozenset(), "notes": ()}
        group["parts"].append(path)
    singles = [p for k, p in singles if k not in composites]       # main files with parts
    indexes = {_key(g["main"]): g for g in groups}
    for key, composite in composites.items():
        if key in indexes:                       # an index file with parts of its own
            indexes[key]["parts"] += composite["parts"]
        else:
            groups.append(composite)
    groups.sort(key=lambda g: _key(g["expected"]))
    return Grouping([FileGroup(**{**g, "parts": tuple(sorted(g["parts"]))}) for g in groups],
                    sorted(singles))
