"""Luxendo / Bruker light-sheet experiments (TruLive3D, InVi, MuVi, LCS SPIM): an
experiment *folder* read as one file (``[hdf5]`` extra; github.com/Luxendo/
luxendo-image; docs_private/COMPOSITE.md, "Luxendo experiment folders").

The spec fixes the places, not the names::

    <experiment>/                       a timestamp, renamed by users
        raw/
            stack_0-Zstack_channel_0-Cyan_obj_bottom/     one per stack, channel, objective
                Cam_long_00000.lux.h5    one stack per camera and time point (``hdf5`` reads one)
                Cam_long_00000.json      its JSON, the same as the file's ``metadata``
        processed/                      Image Processor output: not part of it, opens alone
        main_raw.lux.h5                 link files: datasets that are external links into raw/
        imaris.ims, ims/stacks/imaris_<n>.ims, bdv.xml + bdv.h5   (seen next to it, not in the spec)

- **Any part opens the experiment** (``owner``): the experiment folder, ``raw/``,
  a stack / channel folder, a ``.lux.h5`` or ``.json`` in one, a link file.  The
  walk up stops at the folder holding ``raw/``.  ``multifile=False``: a
  ``.lux.h5`` or a link file reads alone, by its own backend.
- **Pixels always come from raw/**, found by place: one image per stack,
  objective and camera, ``tczyx`` -- ``t`` the file number, ``c`` the channels by
  Luxendo channel number.  Identity from each stack folder's first file's JSON
  (``stack``, ``channel``, ``channel_description``, ``objective``, ``camera``),
  else from the folder and file names.  Every file's JSON is checked when it is
  first read (time point, channel, stack, objective, camera; size and type): a
  file that does not fit its place raises ``CorruptFileError`` naming it.
- **Link files say what should exist** -- the time points and channels no raw
  file knows -- never where pixels come from: the first of ``main_raw.lux.h5``
  (``main_processed.lux.h5``, ``main.lux.h5``), the experiment folder's ``.ims``
  files, the ``ims/stacks`` ones, a ``bdv.xml`` with its ``.h5`` that links any
  raw file.  A file one names that raw/ lacks reads as zeros (one
  ``MissingImagesWarning``); without one only gaps in the time numbering are
  found.  A ``bdv.xml`` whose ``.h5`` is missing (BioImage Archive S-BIAD3686)
  is passed over with a ``CompositeWarning``.
- Several stacks / objectives / cameras: index ``tile`` (stack) and ``view``
  (objective and camera), as Luxendo's BigDataViewer XML names them
  (provisional: no sample with more than one yet).  ``tiles="mosaic"`` is
  refused: nothing places the stacks.
- Pixel sizes, channel names and wavelengths as one ``.lux.h5`` gives them
  (``hdf5``); the time step from the first and last stack's time stamps.  No
  position (as ``hdf5``).
- The ``.lux.h5`` files are opened when read, at most ``OPEN_FILES`` at once.
"""
from __future__ import annotations

import builtins
import datetime
import json
import os
import re
import threading
import warnings
import xml.etree.ElementTree as ET
from collections import OrderedDict
from typing import Optional

import numpy as np

from .. import _parts
from .._base import File, Image
from .._errors import CompositeWarning, CorruptFileError, MissingImagesWarning, UnsupportedFormatError
from .._metadata import Channel, Metadata, quantity
from . import _hdf5
from .hdf5 import LEVEL, _luxendo_metadata

NAME = "luxendo"
RAW = "raw"
SUFFIX = ".lux.h5"
#: a stack file: <camera>_<time point>.lux.h5 (Cam_long_00000.lux.h5)
STACK_FILE = re.compile(r"^(?P<prefix>.+?)_(?P<t>\d+)\.lux\.h5$", re.IGNORECASE)
#: the parts of a stack folder's name, each looked for alone (no grammar in the spec:
#: stack_0-Zstack_channel_0-Cyan_obj_bottom, stack_0_channel_0-Magenta_obj_bottom,
#: stack_0-x00-y01_channel_0_obj_left)
FOLDER_STACK = re.compile(r"(?:^|_)stack_(\d+)", re.IGNORECASE)
FOLDER_CHANNEL = re.compile(r"(?:^|_)channel_(\d+)(?:-([^_]+))?", re.IGNORECASE)
FOLDER_OBJECTIVE = re.compile(r"(?:^|_)obj_([^_]+)", re.IGNORECASE)
#: the spec's link files in the experiment folder, asked first
MAIN_FILES = ("main_raw.lux.h5", "main_processed.lux.h5", "main.lux.h5")
#: link files in the experiment folder (and Imaris ones in ims/stacks/)
LINK_SUFFIXES = (".ims", ".h5", ".xml")
#: .lux.h5 files kept open at once (a long time series has thousands)
OPEN_FILES = 64
#: groups looked at in a link file before giving up (a link file is small)
MAX_LINK_GROUPS = 1_000_000


# -- which files belong ------------------------------------------------------------

def _holds_stacks(folder: str) -> bool:
    try:
        with os.scandir(folder) as entries:
            return any(e.name.lower().endswith(SUFFIX) and e.is_file() for e in entries)
    except OSError:
        return False


def _is_raw(folder: str) -> bool:
    """*folder* is an experiment's ``raw/``: named so, holding a folder of stacks."""
    if os.path.basename(folder).lower() != RAW:
        return False
    try:
        with os.scandir(folder) as entries:
            subfolders = sorted(e.path for e in entries if e.is_dir())
    except OSError:
        return False
    return any(_holds_stacks(s) for s in subfolders)


def _is_root(folder: str) -> bool:
    raw = os.path.join(folder, RAW)
    return os.path.isdir(raw) and _is_raw(raw)


def experiment_folder(path: str) -> Optional[str]:
    """The experiment *path* is a part of (the folder holding ``raw/``), or None."""
    path = os.path.abspath(path)
    if os.path.isdir(path):
        if _is_root(path):
            return path
        parent = os.path.dirname(path)
        if _is_raw(path):
            return parent
        if os.path.basename(parent).lower() == RAW and _holds_stacks(path) and _is_raw(parent):
            return os.path.dirname(parent)
        return None
    folder, name = os.path.split(path)
    low = name.lower()
    if low.endswith(SUFFIX) or low.endswith(".json"):
        raw = os.path.dirname(folder)
        if os.path.basename(raw).lower() == RAW and (
                low.endswith(SUFFIX) or os.path.exists(path[:-len(".json")] + SUFFIX)):
            return os.path.dirname(raw)
    if low.endswith(LINK_SUFFIXES) and _is_root(folder):
        return folder
    if low.endswith(".ims") and os.path.basename(folder).lower() == "stacks":
        ims = os.path.dirname(folder)
        if _is_ims_folder(os.path.basename(ims)) and _is_root(os.path.dirname(ims)):
            return os.path.dirname(ims)
    return None


def _is_ims_folder(name: str) -> bool:
    return name.lower() == "ims" or name.lower().endswith("_ims")


def may_own(folder: str, name: str, folders) -> bool:
    """``group_files``' cheap test (names and cached listings only)."""
    low = name.lower()
    if low.endswith(SUFFIX) or low.endswith(".json"):
        return folders.base(folder, 1).lower() == RAW
    if low.endswith(LINK_SUFFIXES):
        return RAW in folders.names(folder) or (
            folders.base(folder).lower() == "stacks" and _is_ims_folder(folders.base(folder, 1)))
    return False


def owner(path: str, head):
    """Any part of an experiment (module docstring) -> the experiment folder.
    A ``.lux.h5`` or link file is a ``member``: ``multifile=False`` reads it
    alone."""
    root = experiment_folder(path)
    if root is None:
        return None
    member = os.path.isfile(path) and not path.lower().endswith(".json")
    return _parts.Part(root, root, member=member)


def sniff(head: bytes, path: str) -> bool:
    return False                                     # an experiment is a folder


def sniff_folder(path: str) -> bool:
    return experiment_folder(path) == os.path.abspath(path)


# -- what the names say ------------------------------------------------------------

def _int(value) -> Optional[int]:
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return None


def _text(value) -> Optional[str]:
    return str(value).strip() if value is not None and str(value).strip() else None


def _camera(prefix: str) -> str:
    """``Cam_long`` -> ``long``."""
    return prefix[4:] if prefix.lower().startswith("cam_") and len(prefix) > 4 else prefix


def _named(folder: str, prefix: str) -> dict:
    """What a stack folder's and file's names say (None: they do not)."""
    stack, channel, objective = (FOLDER_STACK.search(folder), FOLDER_CHANNEL.search(folder),
                                 FOLDER_OBJECTIVE.search(folder))
    return {"stack": int(stack.group(1)) if stack else None,
            "channel": int(channel.group(1)) if channel else None,
            "channel_description": channel.group(2) if channel and channel.group(2) else None,
            "objective": objective.group(1) if objective else None,
            "camera": _camera(prefix)}


def _same(a, b) -> bool:
    return str(a).lower() == str(b).lower()


# -- one stack folder's files of one camera -----------------------------------------

class _Group:
    """The files of one camera in one stack folder: ``files`` {t: path} on disk,
    ``expected`` the time points a link file names."""

    def __init__(self, folder: str, prefix: str):
        self.folder = folder                         # its name in raw/
        self.prefix = prefix
        self.files: dict = {}
        self.expected: set = set()
        self.digits = 5
        self.identity: dict = {}
        self.shape = self.dtype = None
        self.levels: dict = {}                       # level name -> shape
        self.chunks: dict = {}                       # dataset name -> its chunks (None: contiguous)
        self.metadata: Optional[Metadata] = None

    def name(self, t: int) -> str:
        return f"{self.prefix}_{t:0{self.digits}d}{SUFFIX}"

    def key(self) -> tuple:
        i = self.identity
        return (i["stack"], str(i["objective"]).lower(), str(i["camera"]).lower())


def _info(h5) -> dict:
    """The JSON ``metadata`` dataset of a ``.lux.h5`` ({} when absent or not JSON)."""
    try:
        text = h5["metadata"][()]
        info = json.loads(text.decode() if isinstance(text, bytes) else text)
    except (KeyError, ValueError, TypeError, AttributeError, OSError):
        return {}
    return info if isinstance(info, dict) else {}


def _stated(info: dict) -> dict:
    """What a file's JSON says of its place (missing keys left out)."""
    process = info.get("processingInformation") or {}
    out = {"time_point": _int(process.get("time_point")), "stack": _int(process.get("stack")),
           "channel": _int(process.get("channel")),
           "channel_description": _text(process.get("channel_description")),
           "objective": _text(process.get("objective")), "camera": _text(process.get("camera"))}
    return {k: v for k, v in out.items() if v is not None}


def _identify(group: _Group, h5) -> None:
    """The group's place (JSON first, else names) and what its first file holds."""
    path = group.files[min(group.files)]
    if "Data" not in h5 or not hasattr(h5["Data"], "shape") or h5["Data"].ndim != 3:
        raise CorruptFileError(f"{path!r}: not a Luxendo stack (no 3-D Data dataset)")
    named = _named(group.folder, group.prefix)
    stated = _stated(_info(h5))
    for key in ("stack", "channel", "objective", "camera"):
        if key in stated and named[key] is not None and not _same(stated[key], named[key]):
            warnings.warn(CompositeWarning(
                f"{path!r}: its metadata says {key} {stated[key]!r}, its name "
                f"{named[key]!r}: the metadata is used"), stacklevel=5)
    identity = {**named, **{k: v for k, v in stated.items() if k != "time_point"}}
    identity["stack"] = identity["stack"] if identity["stack"] is not None else 0
    identity["objective"] = identity["objective"] or ""
    group.identity = identity
    base = h5["Data"]
    group.shape, group.dtype = base.shape, base.dtype.newbyteorder("=")
    group.levels = {n: h5[n].shape for n in h5 if LEVEL.match(n)}
    group.chunks = {n: h5[n].chunks for n in ["Data"] + list(group.levels)}
    group.metadata = _luxendo_metadata(h5, base)


# -- link files: what should exist -------------------------------------------------

def _linked(filename: str):
    """An external link's file -> (stack folder, file name) in raw/, or None."""
    parts = [p for p in re.split(r"[\\/]+", filename) if p not in ("", ".")]
    for k in range(len(parts) - 1, -1, -1):
        if parts[k].lower() == RAW:
            rest = parts[k + 1:]
            return (rest[0], rest[1]) if len(rest) == 2 else None
    return None


def _links(path: str) -> list:
    """The raw stack files an HDF5 link file's external links name (followed
    nowhere: the files are not opened)."""
    h5py = _hdf5.h5py_module("reading Luxendo experiments")
    found, seen = [], [0]

    def walk(group):
        for name in group:
            seen[0] += 1
            if seen[0] > MAX_LINK_GROUPS:
                return
            link = group.get(name, getlink=True)
            if isinstance(link, h5py.ExternalLink):
                target = _linked(link.filename)
                if target is not None:
                    found.append(target)
            elif isinstance(link, h5py.HardLink):
                obj = group.get(name)
                if isinstance(obj, h5py.Group):
                    walk(obj)
    try:
        with h5py.File(path, "r") as h5:
            walk(h5)
    except (OSError, KeyError) as exc:
        warnings.warn(CompositeWarning(f"{path!r}: cannot be read as a link file ({exc}): "
                                       "not used"), stacklevel=4)
        return []
    return found


def _bdv_target(xml: str) -> Optional[str]:
    """The ``.h5`` a BigDataViewer XML names (None: not one)."""
    try:
        tree = ET.parse(xml)
    except (ET.ParseError, OSError):
        return None
    loader = tree.getroot().find(".//ImageLoader")
    hdf5 = loader.find("hdf5") if loader is not None else None
    if hdf5 is None or not (hdf5.text or "").strip():
        return None
    return os.path.join(os.path.dirname(xml), hdf5.text.strip())


def _link_files(root: str) -> list:
    """Sources of what should exist, in order; each a list of link files read
    together (the ``ims/stacks`` files: one per stack)."""
    with os.scandir(root) as entries:
        files = sorted(e.path for e in entries if e.is_file())
    by_name = {os.path.basename(p).lower(): p for p in files}
    sources = [[by_name[n]] for n in MAIN_FILES if n in by_name]
    sources.append([p for p in files if p.lower().endswith(".ims")])
    stacks = []
    with os.scandir(root) as entries:
        folders = sorted(e.path for e in entries if e.is_dir() and _is_ims_folder(e.name))
    for folder in folders:
        inner = os.path.join(folder, "stacks")
        if os.path.isdir(inner):
            stacks += sorted(os.path.join(inner, n) for n in os.listdir(inner)
                             if n.lower().endswith(".ims"))
    sources.append(stacks)
    for xml in (p for p in files if p.lower().endswith(".xml")):
        target = _bdv_target(xml)
        if target is None:
            continue
        if not os.path.exists(target):
            sources.append([("missing", xml, target)])
        else:
            sources.append([target])
    return [s for s in sources if s]


def _expected(root: str) -> tuple:
    """{(stack folder, file name)} the first usable link source names, and
    that source's files (empty: none)."""
    for source in _link_files(root):
        if isinstance(source[0], tuple):
            _, xml, target = source[0]
            warnings.warn(CompositeWarning(
                f"{xml!r} names {os.path.basename(target)!r}, which is missing: the time "
                "points and channels come from raw/ alone"), stacklevel=4)
            continue
        found = set()
        for path in source:
            found.update(_links(path))
        if found:
            return found, source
    return set(), []


# -- reading -----------------------------------------------------------------------

class _Files:
    """The experiment's ``.lux.h5`` files open at once: at most ``limit`` not
    being read, the least recently used closed first.  A file's JSON and
    datasets are checked against its place when it is first opened."""

    def __init__(self, reader: _hdf5.ChunkReader, limit: Optional[int] = None):
        self._h5py = _hdf5.h5py_module("reading Luxendo experiments")
        self._reader = reader
        self._limit = OPEN_FILES if limit is None else limit
        self._open: OrderedDict = OrderedDict()
        self._use: dict = {}
        self._checked: set = set()
        self._lock = threading.Lock()

    def acquire(self, path: str, check) -> object:
        with self._lock:
            h5 = self._open.get(path)
            if h5 is None:
                try:
                    h5 = self._h5py.File(path, "r")
                except OSError as exc:
                    raise CorruptFileError(f"{path!r}: {exc}") from exc
                self._open[path] = h5
            self._open.move_to_end(path)
            self._use[path] = self._use.get(path, 0) + 1
            try:
                if path not in self._checked:
                    check(h5)
                    self._checked.add(path)
            except BaseException:
                self._use[path] -= 1
                raise
            self._evict()
            return h5

    def release(self, paths) -> None:
        with self._lock:
            for path in paths:
                self._use[path] -= 1
            self._evict()

    def _evict(self) -> None:
        for path in list(self._open):
            if len(self._open) <= self._limit:
                return
            if self._use.get(path, 0) == 0:
                h5 = self._open.pop(path)
                self._reader.forget(h5.filename)
                h5.close()

    def close(self) -> None:
        with self._lock:
            for h5 in self._open.values():
                h5.close()
            self._open.clear()


class _ExperimentImage(Image):
    """One stack / objective / camera, one resolution level: ``grid[t][c]``
    is the stack file's path, or None (missing: zeros)."""

    def __init__(self, position, name, index, grid, channels, t0, level, shape, dtype,
                 files: _Files, reader, metadata, chunks=None):
        self._grid = grid
        self._channels = channels                    # per c: the group's identity
        self._t0 = t0
        self._level = level
        self._files = files
        self._reader = reader
        self._meta = metadata
        self._missing = any(p is None for row in grid for p in row)
        self._closed = False
        super().__init__(position, "tczyx", shape, dtype, name=name, index=index,
                         read_unit=(1, 1, *chunks) if chunks else None)

    def _check(self, t: int, c: int):
        expected, base = self._channels[c], self.shape[2:]

        def check(h5):
            path = h5.filename
            stated = _stated(_info(h5))
            place = {**{k: expected[k] for k in ("stack", "channel", "objective", "camera")
                        if expected.get(k) is not None}, "time_point": t + self._t0}
            for key, value in place.items():
                if key in stated and not _same(stated[key], value):
                    raise CorruptFileError(
                        f"{path!r}: its metadata says {key} {stated[key]!r}, its place in the "
                        f"experiment {value!r}")
            ds = h5.get(self._level)
            if ds is None or not hasattr(ds, "shape") or tuple(ds.shape) != tuple(base) or \
                    ds.dtype.newbyteorder("=") != self.dtype:
                found = "none" if ds is None else f"{getattr(ds, 'shape', '?')} {getattr(ds, 'dtype', '?')}"
                raise CorruptFileError(
                    f"{path!r}: {self._level} is {found}, the experiment's {tuple(base)} "
                    f"{self.dtype}")
        return check

    def _read_block(self, block):
        if self._closed:
            raise ValueError("this Luxendo image has been closed: its file was closed "
                             "(File.close() or the end of a with block)")
        ts, cs, zs, ys, xs = block
        out = (np.zeros if self._missing else np.empty)(
            tuple(s.stop - s.start for s in block), dtype=self.dtype)
        read, jobs, held = 0, [], []
        try:
            for t in range(ts.start, ts.stop):
                for c in range(cs.start, cs.stop):
                    path = self._grid[t][c]
                    if path is None:
                        continue                     # missing: zeros
                    h5 = self._files.acquire(path, self._check(t, c))
                    held.append(path)
                    read += self._reader.plan((path, self._level), h5[self._level],
                                              (zs, ys, xs), out[t - ts.start, c - cs.start], jobs)
            return out, read + self._reader.run(jobs)
        finally:
            self._files.release(held)

    def _read_metadata(self):
        return self._meta() if callable(self._meta) else self._meta

    def _member_paths(self) -> list:
        return [p for row in self._grid for p in row if p is not None]

    def close(self) -> None:
        self._closed = True


def _time_step(paths: list, t0_t1: tuple):
    """The mean time step between the first and the last of *paths* (their
    JSON's first time stamp), or None."""
    h5py = _hdf5.h5py_module()
    stamps = []
    for path in paths:
        sidecar = path[:-len(SUFFIX)] + ".json"
        info = {}
        try:
            with builtins.open(sidecar, encoding="utf-8") as fh:
                info = json.load(fh)
        except (OSError, ValueError):
            try:
                with h5py.File(path, "r") as h5:
                    info = _info(h5)
            except OSError:
                return None
        try:
            stamp = info["acquisition"][0]["time_stamps"][0]
            stamps.append(datetime.datetime.fromisoformat(stamp))
        except (KeyError, IndexError, TypeError, ValueError):
            return None
    steps = t0_t1[1] - t0_t1[0]
    if len(stamps) != 2 or steps <= 0:
        return None
    return quantity((stamps[1] - stamps[0]).total_seconds() / steps, "second")


class _ExperimentFile(File):
    backend = NAME
    format = "luxendo"

    def __init__(self, path, images, files: _Files, reader, links: list):
        super().__init__(path, images)
        self._files = files
        self._reader = reader
        #: the link files that said which files should exist (empty: none)
        self.link_files = list(links)

    def close(self) -> None:
        super().close()
        self._files.close()
        self._reader.close()


def _scan(raw: str) -> dict:
    """(stack folder, camera prefix) lower case -> _Group of the files on disk."""
    groups = {}
    with os.scandir(raw) as entries:
        folders = sorted((e.name, e.path) for e in entries if e.is_dir())
    for folder, folder_path in folders:
        with os.scandir(folder_path) as entries:
            names = sorted(e.name for e in entries if e.is_file())
        for name in names:
            m = STACK_FILE.match(name)
            if not m:
                continue
            key = (folder.lower(), m["prefix"].lower())
            group = groups.get(key)
            if group is None:
                group = groups[key] = _Group(folder, m["prefix"])
                group.digits = len(m["t"])
            group.files[int(m["t"])] = os.path.join(folder_path, name)
    return groups


def open(path: str, tiles: str = "separate", resolutions: str = "base") -> _ExperimentFile:  # noqa: A001
    h5py = _hdf5.h5py_module("reading Luxendo experiments")
    root = experiment_folder(path)
    if root is None:
        raise UnsupportedFormatError(f"{path!r} is not a part of a Luxendo experiment (a folder "
                                     "holding raw/ with folders of .lux.h5 stacks)")
    raw = os.path.join(root, RAW)
    groups = _scan(raw)
    expected, links = _expected(root)
    for folder, name in expected:
        m = STACK_FILE.match(name)
        if not m:
            continue
        key = (folder.lower(), m["prefix"].lower())
        group = groups.get(key)
        if group is None:
            group = groups[key] = _Group(folder, m["prefix"])
            group.digits = len(m["t"])
        group.expected.add(int(m["t"]))

    reader = _hdf5.ChunkReader(root)
    try:
        present = [g for g in groups.values() if g.files]
        for group in present:
            first = group.files[min(group.files)]
            try:
                with h5py.File(first, "r") as h5:
                    _identify(group, h5)
            except OSError as exc:
                raise CorruptFileError(f"{first!r}: {exc}") from exc
        images = _images(root, groups, present, tiles, resolutions, reader)
    except BaseException:
        reader.close()
        raise
    files = _Files(reader)
    for image in images:
        for member in image.resolutions:
            member._files = files
    return _ExperimentFile(root, images, files, reader, links)


def _images(root, groups, present, tiles, resolutions, reader) -> list:
    if not present:
        raise CorruptFileError(f"{root!r}: no stack file of the experiment is there (raw/ "
                               "holds no .lux.h5)")
    # groups only a link file names take their place from their names
    for group in groups.values():
        if not group.files:
            named = _named(group.folder, group.prefix)
            named["stack"] = named["stack"] if named["stack"] is not None else 0
            named["objective"] = named["objective"] or ""
            group.identity = named
    by_image: dict = {}
    for group in groups.values():
        by_image.setdefault(group.key(), []).append(group)
    # an image only a link file names (no file of it in raw/) is left out
    keys = sorted(k for k in by_image if any(g.files for g in by_image[k]))
    stacks = sorted({k[0] for k in keys})
    views = sorted({k[1:] for k in keys})
    if tiles == "mosaic" and len(stacks) > 1:
        raise UnsupportedFormatError(f"{root!r}: tiles='mosaic': a Luxendo experiment does not "
                                     "say how its stacks are placed")
    name = os.path.basename(root)
    images, missing, total, example = [], 0, 0, None
    for key in keys:
        members = by_image[key]
        on_disk = [g for g in members if g.files]
        shapes = {(g.shape, g.dtype) for g in on_disk}
        if len(shapes) != 1:
            raise UnsupportedFormatError(
                f"{root!r}: the channels of stack {key[0]}, objective {key[1]!r}, camera "
                f"{key[2]!r} differ in size or type: {sorted(map(str, shapes))}")
        shape, dtype = shapes.pop()
        members.sort(key=lambda g: (g.identity["channel"] if g.identity["channel"] is not None
                                    else 1 << 30, g.folder.lower()))
        seen = {}
        for g in members:
            c = g.identity["channel"]
            if c is not None and c in seen:
                raise UnsupportedFormatError(
                    f"{root!r}: channel {c} twice for stack {key[0]}, objective {key[1]!r}, "
                    f"camera {key[2]!r}: raw/{seen[c]} and raw/{g.folder}")
            seen[c] = g.folder
        times = set()
        for g in members:
            times.update(g.files)
            times.update(g.expected)
        t0, t1 = min(times), max(times)
        grid = [[g.files.get(t) for g in members] for t in range(t0, t1 + 1)]
        for t, row in enumerate(grid):
            for g, p in zip(members, row):
                if p is None:
                    missing += 1
                    example = example or f"raw/{g.folder}/{g.name(t + t0)}"
        total += len(grid) * len(members)
        level_names = set.intersection(*(set(g.levels) for g in on_disk))
        levels = sorted(level_names, key=lambda n: -int(np.prod(on_disk[0].levels[n])))
        index = {"series": 0}
        if len(stacks) > 1:
            index["tile"] = stacks.index(key[0])
        if len(views) > 1:
            index["view"] = views.index(key[1:])
        ident = on_disk[0].identity
        label = name if len(keys) == 1 else \
            f"{name} st:{ident['stack']} obj:{ident['objective']} cam:{ident['camera']}"
        firsts = sorted(on_disk[0].files.items())
        span = (firsts[0][0], firsts[-1][0])
        channels = [dict(g.identity) for g in members]

        def metadata(on_disk=on_disk, members=members, firsts=firsts, span=span):
            meta = on_disk[0].metadata
            sizes = dict(meta.pixel_sizes)
            if span[1] > span[0]:
                step = _time_step([firsts[0][1], firsts[-1][1]], span)
                if step is not None:
                    sizes["t"] = step
            return Metadata(pixel_sizes=sizes, channels=tuple(
                g.metadata.channels[0] if g.metadata is not None
                else Channel(name=g.identity.get("channel_description")) for g in members))

        listed = resolutions == "all"
        pyramid = []
        for r, level in enumerate(["Data"] + levels):
            level_shape = shape if r == 0 else on_disk[0].levels[level]
            level_index = dict(index, resolution=r) if (listed or r) else dict(index)
            pyramid.append(_ExperimentImage(
                len(images), label, level_index, grid, channels, t0, level,
                (len(grid), len(members)) + tuple(level_shape), dtype, None, reader,
                metadata if r == 0 else None, on_disk[0].chunks.get(level)))
        for image in pyramid:
            image._resolutions = pyramid
            # _naming: the files' stack, objective and camera labels tell several apart
            image._label = (f"st:{ident['stack']} obj:{ident['objective']} "
                            f"cam:{ident['camera']}")
        images.extend(pyramid if listed else pyramid[:1])
    for n, image in enumerate(images):
        image._position = n
    if missing:
        source = "a link file" if any(g.expected for g in groups.values()) else "the time numbering"
        warnings.warn(MissingImagesWarning(
            f"{root!r}: {missing} of {total} stack files are missing (e.g. {example!r}, by "
            f"{source}): read as zeros"), stacklevel=4)
    return images
