"""NIS-Elements well-plate runs: a folder of ND2 files, one per well or per point
(docs/HCS.md).

NIS-Elements (JOBS, and ND acquisitions over a well plate) can write each well,
or each point of a well, to an ND2 file of its own, in one folder per run:

- ``WellB05_ChannelDIA_Seq0000.nd2``: well B05, one file per well;
- ``WellA01_PointA01_0003_ChannelFITC BP_Seq0003.nd2``: well A01, its point
  ``A01_0003`` (the well's name and a counter), one file per point;

``Seq`` numbers the files in acquisition order over the run.  The files hold no
well metadata: the plate is in their names only.

With ``plates=True`` (and ``multifile=True``, the default), such a file opens
its run folder as a plate: every file of the folder named so is a member.

- Plate ``0``: named after the run folder.  The layout is the well plate of the
  job that acquired the files (``JobDefinitionV1_0``: ``CLxWellplate``'s
  ``uiYCount`` x ``uiXCount``; "384 well 1.5H Cellvis" is 16 x 24), else the
  smallest standard plate holding the wells, with a PlateLayoutWarning.
- A well's fields: its points in counter order (one field when its files name
  no point); a field's files with different ``Channel`` names are its
  channels, in ``Seq`` order, stacked along ``c`` (same t, z, y, x).
- Each file must hold one XY position; a field with two files of one channel,
  or files of different shapes, is refused (UnsupportedFormatError naming
  ``multifile=False`` / ``plates=False``, which read the one file).
- Metadata of a field: its first file's (pixel sizes, stage position, channel
  names; a channel without a name takes the file name's).
"""
from __future__ import annotations

import dataclasses
import os
import re
import threading
from collections import OrderedDict

import numpy as np

from .._base import File, Image
from .._errors import UnsupportedFormatError
from .._metadata import Channel
from .._plate import Plate, Well, layout, row_label, standard_layout, well_of
from .._spatial import CoordinateSystem

#: Well<row><column>[_Point<name>_<n>]_Channel<name>_Seq<n>.nd2
RUN_FILE = re.compile(r"^Well([A-Za-z]{1,2}\d{1,3})(?:_Point([^_]+)_(\d+))?_Channel(.+)_Seq(\d+)\.nd2$",
                      re.IGNORECASE)
#: member files whose handles stay open at once
OPEN_FILES = 32


#: the JOBS definition NIS-Elements stores in a file it acquired with a job
JOB_CHUNK = b"CustomData|JobDefinitionV1_0!"


def jobs_wellplate(path: str):
    """(rows, columns, model name) of the well plate the NIS-Elements job that
    acquired *path* defined (its ``CLxWellplate``: ``uiYCount`` x ``uiXCount``,
    ``wsModelName``), or None: no job, or a plate the definition does not hold."""
    import builtins
    import zlib
    from . import _nd2_structure as structure
    try:
        with builtins.open(path, "rb") as fh:
            cmap = structure.chunkmap(fh)
            if JOB_CHUNK not in cmap:
                return None
            raw = structure.read_chunk(fh, cmap[JOB_CHUNK][0])
        data = zlib.decompress(raw[12:]) if raw[:4] == b"LV0Z" else raw
        job = structure.decode_clx_lite(data)
    except Exception:                       # noqa: BLE001 - a job we cannot read names no plate
        return None

    def find(node):
        if isinstance(node, dict):
            plate = node.get("CLxWellplate")
            if isinstance(plate, dict) and plate.get("uiXCount") and plate.get("uiYCount"):
                return plate
            for value in node.values():
                hit = find(value)
                if hit is not None:
                    return hit
        elif isinstance(node, (list, tuple)):
            for value in node:
                hit = find(value)
                if hit is not None:
                    return hit
        return None

    plate = find(job)
    if plate is None:
        return None
    return int(plate["uiYCount"]), int(plate["uiXCount"]), str(plate.get("wsModelName") or "")


def plate_size(path: str, wells, where: str) -> tuple:
    """(rows, columns) of a NIS-Elements plate: its job's well plate, else the
    smallest standard plate holding *wells*, with a PlateLayoutWarning (a guess)."""
    import warnings
    from .._errors import PlateLayoutWarning
    found = jobs_wellplate(path)
    if found is not None:
        return found[:2]
    size = standard_layout(wells) or (max(r for r, _ in wells) + 1, max(c for _, c in wells) + 1)
    warnings.warn(PlateLayoutWarning(
        f"{where!r}: the file names no well plate (no NIS-Elements job definition): laid out "
        f"as the smallest standard plate holding its wells, {size[0]} x {size[1]}"), stacklevel=5)
    return size


def run_files(folder: str) -> list:
    """The files of *folder* named as a well-plate run's, sorted by name."""
    with os.scandir(folder) as entries:
        return sorted(e.path for e in entries if e.is_file() and RUN_FILE.match(e.name))


class _OpenLimit:
    """At most OPEN_FILES member files keep their handles; the least recently
    used one that no read is using closes first (it reopens on its next read)."""

    def __init__(self):
        self._lock = threading.Lock()
        self._order: OrderedDict = OrderedDict()        # parser id -> [parser, users]

    def enter(self, nd2):
        with self._lock:
            entry = self._order.setdefault(id(nd2), [nd2, 0])
            entry[1] += 1
            self._order.move_to_end(id(nd2))

    def leave(self, nd2):
        with self._lock:
            self._order[id(nd2)][1] -= 1
            for key in list(self._order):
                if len(self._order) <= OPEN_FILES:
                    break
                parser, users = self._order[key]
                if users == 0:
                    del self._order[key]
                    parser.handles.close()


class _RunField(Image):
    """One field: its files' images stacked along c."""

    def __init__(self, position, parts, names, name, index, limit):
        first = parts[0]
        if any(len(p.shape) != 5 for p in parts):
            raise UnsupportedFormatError(
                f"{name}: its files hold RGB samples, which a well-plate run does not stack "
                "yet; open one file with multifile=False")
        t, _c, z, y, x = first.shape
        for p in parts[1:]:
            if (p.shape[0], p.shape[2:]) != (t, (z, y, x)) or p.dtype != first.dtype:
                raise UnsupportedFormatError(
                    f"{name}: its channel files differ in shape or type ({first.shape} "
                    f"{first.dtype}, {p.shape} {p.dtype}); open one file with "
                    "multifile=False")
        channels = sum(p.shape[1] for p in parts)
        super().__init__(position, "tczyx", (t, channels, z, y, x), first.dtype, name=name,
                         read_unit=first.read_unit if len(parts) == 1 else None, index=index)
        self._files, self._names, self._limit = parts, names, limit
        self._at = [(k, c) for k, p in enumerate(parts) for c in range(p.shape[1])]

    def _read_block(self, block):
        ts, cs, zs, ys, xs = block
        out = np.empty(tuple(s.stop - s.start for s in block), dtype=self.dtype)
        read = 0
        c = cs.start
        while c < cs.stop:
            k, local = self._at[c]
            part = self._files[k]
            stop = min(cs.stop, c + part.shape[1] - local)      # this file's channels
            self._limit.enter(part._f)
            try:
                data, n = part._read_block((ts, slice(local, local + stop - c), zs, ys, xs))
            finally:
                self._limit.leave(part._f)
            out[:, c - cs.start:stop - cs.start] = data
            read += n
            c = stop
        return out, read

    def _read_metadata(self):
        channels = []
        for part, name in zip(self._files, self._names):
            own = list(part.metadata.channels) or [Channel()] * part.shape[1]
            channels += [ch if ch.name else dataclasses.replace(ch, name=name) for ch in own]
        return dataclasses.replace(self._files[0].metadata, channels=tuple(channels))

    def _member_paths(self) -> set:
        return {os.path.normcase(os.path.abspath(p._f.path)) for p in self._files}

    def close(self) -> None:
        for part in self._files:
            part.close()


class _Nd2RunFile(File):
    backend = "nd2"
    format = "nd2-run"

    def __init__(self, path: str):
        from .nd2 import Nd2Parser, _Nd2Image
        folder = os.path.dirname(os.path.abspath(path))
        paths = run_files(folder)
        self._parsers = []
        fields: dict = {}          # (well rc, point counter or None) -> [(seq, channel, path)]
        for p in paths:
            m = RUN_FILE.match(os.path.basename(p))
            well, _point, counter, channel, seq = m.groups()
            rc = well_of(well)
            if rc is None:
                raise UnsupportedFormatError(f"{p!r}: {well!r} is not a well name")
            fields.setdefault((rc, None if counter is None else int(counter)), []).append(
                (int(seq), channel, p))
        wells = sorted({rc for rc, _ in fields})
        number = {rc: w for w, rc in enumerate(wells)}
        size = plate_size(paths[0], wells, folder)
        self._limit = _OpenLimit()
        images, field_of = [], {}
        try:
            for (rc, counter) in sorted(fields, key=lambda k: (number[k[0]], k[1] or 0)):
                files = sorted(fields[(rc, counter)])
                names = [ch for _seq, ch, _p in files]
                if len(set(n.lower() for n in names)) != len(names):
                    raise UnsupportedFormatError(
                        f"{path!r}: well {row_label(rc[0])}{rc[1] + 1} has several files of one "
                        f"channel{'' if counter is None else f' at point {counter}'} "
                        f"({[os.path.basename(p) for _s, _c, p in files][:3]}): not a run "
                        "micro-reader reads as a plate; open one file with multifile=False "
                        "or plates=False")
                parts = []
                for _seq, _ch, p in files:
                    nd2 = Nd2Parser(p)
                    self._parsers.append(nd2)
                    if nd2.size("P") != 1:
                        raise UnsupportedFormatError(
                            f"{p!r} holds {nd2.size('P')} XY positions; a run file micro-reader "
                            "reads as a plate holds one. Open it with multifile=False or "
                            "plates=False")
                    parts.append(_Nd2Image(nd2, 0))
                f = field_of[rc] = field_of.get(rc, -1) + 1
                label = f"{row_label(rc[0])}{rc[1] + 1}"
                images.append(_RunField(
                    len(images), parts, names, f"{os.path.basename(folder)} well {label} field {f}",
                    {"plate": 0, "acquisition": 0, "well": number[rc], "field": f}, self._limit))
        except BaseException:
            for nd2 in self._parsers:
                nd2.handles.close()
            raise
        row_names, column_names = layout(*size, wells, folder)
        plate = Plate(name=os.path.basename(folder), row_names=row_names,
                      column_names=column_names,
                      wells=tuple(Well(r, c, f"{row_label(r)}{c + 1}") for r, c in wells))
        super().__init__(path, images, plates=[plate])

    def _declare_coordinate_systems(self):
        from ._nd2_metadata import STAGE, placed
        return (CoordinateSystem(STAGE),) if any(placed(n.picture) for n in self._parsers) else ()

    def close(self) -> None:
        super().close()
        for nd2 in self._parsers:
            nd2.handles.close()
