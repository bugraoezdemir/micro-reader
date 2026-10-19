"""What every file-set backend needs (docs/HCS.md): a plate whose planes are
separate 2-D files (Harmony, MetaXpress, ...), listed by an index file.

- ``Members``: the member files, opened on demand by the TIFF backend; at
  most ``OPEN_MEMBERS`` at once (or the set's own limit), never one that a
  read is using; as plain pages when the index file says what every page is
  (OME companion files, docs/COMPANION.md).
- ``PlaneImage``: one field, its planes stacked to ``t c z y x``; a missing
  plane reads as zeros.
- ``keep_fields``: the rule for missing files (Bugra, 2026-10-13): a field
  missing some planes is kept (zeros there), a field missing all of them is
  left out; one ``MissingImagesWarning`` says how many of each.
- ``ragged_field``: a field whose channels have different z samplings, as a
  ragged image (docs/AXES.md): its ``parts`` are the channels that share
  their z indices; the whole reads them on the deepest part's grid, each
  channel's planes at the slices nearest their own focus positions, zeros
  elsewhere.  A single plane always has a nearest slice (within half a
  step); a channel with several planes must lie on the grid (within
  ``GRID_TOLERANCE`` of a step), otherwise reading the whole raises: read
  its parts."""
from __future__ import annotations

import itertools
import os
import threading
import warnings
from collections import OrderedDict

import numpy as np

from ._base import Image
from ._errors import CorruptFileError, MissingImagesWarning, UnsupportedFormatError

#: member files kept open at once (each holds one OS handle per reading thread)
OPEN_MEMBERS = 32
#: how far (as a fraction of the z step) a ragged field's channel with
#: several planes may sit off the whole's grid
GRID_TOLERANCE = 0.1


class Members:
    """The file set's member files, opened on demand: at most *limit* at once
    (None: ``OPEN_MEMBERS``), and never one that a read is using.

    *pages*: open each member as plain pages (``tiff.open_pages``), for index
    files that say what every page is (a plane is then (path, IFD)); otherwise
    each member is one 2-D image. *check(path, file)*: run once per member, the
    first time it is opened; it raises when the member does not belong."""

    def __init__(self, limit: int = None, pages: bool = False, check=None):
        self._open: OrderedDict = OrderedDict()   # path -> [micro_reader File, users]
        self._lock = threading.Lock()
        self._limit, self._pages, self._check = limit, bool(pages), check
        self._checked: set = set()

    def acquire(self, path: str):
        from .backends import tiff
        with self._lock:
            entry = self._open.get(path)
            if entry is None:
                member = (tiff.open_pages(path) if self._pages
                          else tiff.open(path, "separate", "base", companion=False))
                if self._check is not None and path not in self._checked:
                    try:
                        self._check(path, member)
                    except BaseException:
                        member.close()
                        raise
                    self._checked.add(path)
                entry = self._open[path] = [member, 0]
            entry[1] += 1
            self._open.move_to_end(path)
            return entry

    def adopt(self, path: str, member) -> None:
        """Take over member *path* already opened (and checked) by the caller."""
        with self._lock:
            if path in self._open:
                member.close()
                return
            self._open[path] = [member, 0]
            self._checked.add(path)

    def release(self, entry) -> None:
        with self._lock:
            entry[1] -= 1
            for path in list(self._open):
                if len(self._open) <= (OPEN_MEMBERS if self._limit is None else self._limit):
                    break
                old = self._open[path]
                if old[1] == 0:
                    del self._open[path]
                    old[0].close()

    def plane_info(self, path: str, shape=None) -> tuple:
        """(dtype, read_unit) of the 2-D image in member *path*; checks its
        shape against *shape* (y, x) when given."""
        entry = self.acquire(path)
        try:
            image = entry[0].images[0]
            if image.axes != "yx" or (shape is not None and image.shape != tuple(shape)):
                want = f"one {shape[0]} x {shape[1]} plane" if shape else "a 2-D plane"
                raise CorruptFileError(f"{path!r}: a {image.axes} {image.shape} image, "
                                       f"the file set expects {want}")
            return image.dtype, image.read_unit, image.shape
        finally:
            self.release(entry)

    def page_info(self, path: str, ifd: int) -> tuple:
        """(dtype, read_unit (y, x), shape (y, x)) of page *ifd* of member
        *path* (opened as pages); a page that is not one 2-D plane raises."""
        entry = self.acquire(path)
        try:
            image, _k = _page(entry[0], ifd, path)
            unit = image.read_unit[-2:] if image.read_unit is not None else None
            return image.dtype, unit, image.shape[-2:]
        finally:
            self.release(entry)

    def close(self) -> None:
        with self._lock:
            for member, _users in self._open.values():
                member.close()
            self._open.clear()


def _page(member, ifd: int, path: str):
    """(image, page in it) of IFD *ifd* of a member opened as pages."""
    found = member.pages.get(int(ifd))
    if found is None:
        raise CorruptFileError(f"{path!r} has no page {ifd} (IFD)")
    image, k = found
    if image.axes not in ("yx", "zyx"):
        raise UnsupportedFormatError(
            f"{path!r} page {ifd}: a {image.axes} image, not one plane (RGB / multi-sample "
            f"pages in a file set are not read yet)")
    return image, k


def _read_page(member, ifd: int, ys, xs, path: str):
    """Rows *ys*, columns *xs* of page *ifd* of a member opened as pages."""
    image, k = _page(member, ifd, path)
    if image.axes == "yx":
        return image._read_block((ys, xs))
    plane, n = image._read_block((slice(k, k + 1), ys, xs))
    return plane[0], n


class PlaneImage(Image):
    """One field of a file set: 2-D member files stacked to t c z y x.
    *planes*: (t, c, z) position -> member path, or (member path, IFD) for
    members opened as pages (a missing plane reads as zeros).  The backend
    gives the metadata (``_read_metadata``).  A ragged whole whose parts share
    no grid is not *readable*."""

    def __init__(self, members: Members, position: int, planes: dict, shape, dtype,
                 name: str, read_unit, index: dict, readable: bool = True):
        super().__init__(position, "tczyx", shape, dtype, name=name,
                         read_unit=None if read_unit is None else (1, 1, 1) + tuple(read_unit),
                         index=index)
        self._members = members
        self._planes = planes
        self._readable = readable
        self._closed = False

    def _read_block(self, block):
        if self._closed:
            # its members would reopen on acquire: a closed file set stays closed
            raise ValueError(f"{self.name!r} has been closed: its file was closed "
                             "(File.close() or the end of a with block)")
        if not self._readable:
            raise UnsupportedFormatError(
                f"{self.name!r} is ragged and its parts do not share a z grid: read its "
                f"parts (image.parts)")
        (t0, t1), (c0, c1), (z0, z1) = ((s.start, s.stop) for s in block[:3])
        ys, xs = block[3], block[4]
        out = np.empty((t1 - t0, c1 - c0, z1 - z0, ys.stop - ys.start, xs.stop - xs.start),
                       dtype=self.dtype)
        if self._members._pages:
            return out, self._read_pages(out, (t0, t1), (c0, c1), (z0, z1), ys, xs)
        read = 0
        for t in range(t0, t1):
            for c in range(c0, c1):
                for z in range(z0, z1):
                    member = self._planes.get((t, c, z))
                    if member is None:                  # a missing plane reads as zeros
                        out[t - t0, c - c0, z - z0] = 0
                        continue
                    path, ifd = member if isinstance(member, tuple) else (member, None)
                    entry = self._members.acquire(path)
                    try:
                        if ifd is None:
                            image = entry[0].images[0]
                            self._fits(path, image)
                            plane, n = image._read_block((ys, xs))
                        else:
                            plane, n = _read_page(entry[0], ifd, ys, xs, path)
                    finally:
                        self._members.release(entry)
                    out[t - t0, c - c0, z - z0] = plane
                    read += n
        return out, read

    def _read_pages(self, out, ts, cs, zs, ys, xs) -> int:
        """Fill *out* from members opened as pages: every plane's reads are
        planned straight into *out* first, then all of them run in one batch on
        the decode pool, as a multi-page TIFF's pages do (no copy per plane, the
        pool busy across planes). The members stay acquired until it is done."""
        from . import _pool
        entries, plans = [], []
        try:
            for t in range(*ts):
                for c in range(*cs):
                    for z in range(*zs):
                        dest = out[t - ts[0], c - cs[0], z - zs[0]]
                        member = self._planes.get((t, c, z))
                        if member is None:              # a missing plane reads as zeros
                            dest[...] = 0
                            continue
                        path, ifd = member
                        entry = self._members.acquire(path)
                        entries.append(entry)
                        image, k = _page(entry[0], ifd, path)
                        self._fits(path, image, ifd)
                        if image.axes == "yx":
                            plan = image._plan_series((ys, xs), dest)
                        else:
                            plan = image._plan_series((slice(k, k + 1), ys, xs), dest[None])
                        plans.append((image, plan))
            return _pool.run(itertools.chain.from_iterable(
                image._tasks(*plan) for image, plan in plans))
        finally:
            for entry in entries:
                self._members.release(entry)

    def _fits(self, path: str, image, ifd=None) -> None:
        """Raise unless member *path* (its first image, or page *ifd*) is one
        plane of this image's size and type: the index decides what a plane
        is, a file that differs would be cropped or cast without a word.  Run
        on every read (a comparison; the member is open anyway)."""
        y, x = self.shape[-2:]
        if ifd is not None:
            fits = tuple(image.shape[-2:]) == (y, x)
        else:
            fits = image.axes == "yx" and tuple(image.shape) == (y, x)
        if fits and image.dtype == self.dtype:
            return
        what = f"page {ifd}: " if ifd is not None else ""
        raise CorruptFileError(f"{path!r}: {what}a {image.axes} {tuple(image.shape)} "
                               f"{image.dtype} image, {self.name!r} expects one {y} x {x} "
                               f"{self.dtype} plane")

    def _member_paths(self) -> set:
        """The member files this image reads (``File._member_files``)."""
        return {m[0] if isinstance(m, tuple) else m for m in self._planes.values()}

    def _check_members(self, seen: set):
        """Tasks checking each member not in *seen* (see ``File.check_members``);
        each returns 1, or the error text of a member that does not fit."""
        for member in self._planes.values():
            path, ifd = member if isinstance(member, tuple) else (member, None)
            key = (path, ifd)
            if key in seen:
                continue
            seen.add(key)
            yield lambda path=path, ifd=ifd: self._check_one(path, ifd)

    def _check_one(self, path: str, ifd):
        entry = self._members.acquire(path)
        try:
            if ifd is None:
                self._fits(path, entry[0].images[0])
            else:
                self._fits(path, _page(entry[0], ifd, path)[0], ifd)
        except (CorruptFileError, UnsupportedFormatError) as exc:
            return str(exc)
        finally:
            self._members.release(entry)
        return 1

    def close(self) -> None:
        self._closed = True                     # the file closes the members


def keep_fields(fields: dict, wanted, path: str) -> dict:
    """*fields*: field key -> {plane: member path, (member path, IFD), or None
    when the index lists a plane without a usable file}; *wanted*: every plane
    a field should have.  -> {field key: {plane: path or (path, IFD)}} for the
    fields with at least one file, after one MissingImagesWarning about what is
    missing."""
    kept, partial, empty, missing, example = {}, 0, 0, 0, None
    for key, listed in fields.items():
        files = {}
        for plane in wanted:
            p = listed.get(plane)
            file = p[0] if isinstance(p, tuple) else p
            if file and os.path.isfile(file):
                files[plane] = p
            else:
                missing += 1
                example = example or (os.path.basename(file) if file else f"{key}: no file listed")
        if not files:
            empty += 1
            continue
        partial += len(files) < len(wanted)
        kept[key] = files
    if missing:
        told = []
        if partial:
            told.append(f"{partial} fields miss some planes, which read as zeros")
        if empty:
            told.append(f"{empty} fields have no files and are left out")
        warnings.warn(MissingImagesWarning(
            f"{path!r}: {missing} image files are missing (e.g. {example!r}): "
            + "; ".join(told) + f" (of {len(fields)} fields)"), stacklevel=4)
    if not kept:
        raise CorruptFileError(f"{path!r}: none of the fields it lists has any of its files")
    return kept


def check_times(wanted, path: str) -> None:
    """Refuse channels imaged at different timepoints (ragged t: not read yet).
    *wanted*: the (t, channel, z) planes of the plate."""
    channels = {c for _t, c, _z in wanted}
    for t in {t for t, _c, _z in wanted}:
        if {c for tt, c, _z in wanted if tt == t} != channels:
            raise UnsupportedFormatError(
                f"{path!r}: channels imaged at different timepoints are not read yet")


def z_parts(wanted) -> list:
    """The channels that share their z indices: [(z indices, [channel, ...])],
    by first channel.  *wanted*: the (t, channel, z) planes of the plate."""
    z_of: dict = {}
    for _t, c, z in wanted:
        z_of.setdefault(c, set()).add(z)
    groups: dict = {}
    for c in sorted(z_of):
        groups.setdefault(tuple(sorted(z_of[c])), []).append(c)
    return sorted(groups.items(), key=lambda g: g[1][0])


def z_step(z_at: dict, t, channel, zs) -> "float | None":
    """The z step of *channel*'s planes *zs* at timepoint *t*, from their focus
    positions *z_at* ((t, channel, z) -> z), or None for one plane."""
    if len(zs) < 2:
        return None
    a, b = z_at.get((t, channel, zs[0])), z_at.get((t, channel, zs[1]))
    return abs(b - a) if a is not None and b is not None and b != a else None


def ragged_field(make, files: dict, z_at: dict, times: list, parts: list, index: dict):
    """One field as an image, ragged when *parts* (``z_parts``) has several.

    *files*: (t, channel, z) -> member path; *z_at*: (t, channel, z) -> focus
    position (one length unit, any origin).  *make(planes, shape_tcz, channels,
    dz, index, readable)* builds an image of the backend (``PlaneImage``):
    *planes* maps (t, c, z) positions to member paths, *dz* is its z step in
    the focus positions' unit (None: one plane).  -> the whole, its
    ``_parts`` set when ragged."""
    channels = [c for _zs, members in parts for c in members]
    channels.sort()
    deepest = max(parts, key=lambda g: len(g[0]))
    t0 = times[0]
    grid_c, grid_zs = deepest[1][0], deepest[0]
    grid = [z_at.get((t0, grid_c, z)) for z in grid_zs]
    step = z_step(z_at, t0, grid_c, grid_zs)
    planes, readable = {}, True
    for zs, members in parts:
        for ch in members:              # each channel by its own focus positions
            slot = _slots(z_at, t0, ch, zs, grid, step, (zs, members) == deepest)
            if slot is None:
                readable = False
                continue
            for (t, c, z), p in files.items():
                if c == ch and z in slot:
                    planes[(times.index(t), channels.index(ch), slot[z])] = p
    whole = make(planes, (len(times), len(channels), len(grid_zs)), channels, step, index,
                 readable)
    if len(parts) > 1:
        whole._parts = []
        for k, (zs, members) in enumerate(parts):
            part = make({(times.index(t), members.index(c), zs.index(z)): p
                         for (t, c, z), p in files.items() if c in members},
                        (len(times), len(members), len(zs)), members,
                        z_step(z_at, t0, members[0], zs), {**index, "part": k}, True)
            part._whole = whole
            whole._parts.append(part)
    return whole


def _slots(z_at, t0, ch, zs, grid, step, on_grid) -> "dict | None":
    """{z index: slice of the whole} for channel *ch*'s planes *zs*, or None
    when they do not lie on the *grid* (the deepest part's focus positions)."""
    if on_grid or len(grid) == 1:
        return {z: n for n, z in enumerate(zs)}
    if None in grid or not step:
        return None
    slot = {}
    for z in zs:
        here = z_at.get((t0, ch, z))
        if here is None:
            return None
        n = min(range(len(grid)), key=lambda i: abs(grid[i] - here))
        # one plane: the nearest slice (within half a step of the grid);
        # several: they must lie on the grid
        if (abs(grid[n] - here) / step > (0.5 if len(zs) == 1 else GRID_TOLERANCE)
                or n in slot.values()):
            return None
        slot[z] = n
    return slot
