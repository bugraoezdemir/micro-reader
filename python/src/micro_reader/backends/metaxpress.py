"""Molecular Devices MetaXpress / ImageXpress plates (MetaMorph HTS): docs/HCS.md.

A plate is a folder of TIFFs (MetaMorph STK, one plane each) and a text index
file, ``<plate>.HTD``.  Open the ``.HTD`` file.  Unlike Harmony's index it
does not list the images; it gives:

- the plate: ``XWells`` columns x ``YWells`` rows, and the wells imaged
  (``WellsSelection<row>``: one TRUE / FALSE per column);
- the sites (fields) per well: ``Sites`` TRUE, ``XSites`` x ``YSites`` and
  ``SiteSelection<row>``; the selected grid cells, row by row, are sites 1,
  2, ... (as Bio-Formats numbers them);
- the channels: ``Waves`` TRUE, ``NWavelengths``, ``WaveName<n>``;
- ``TimePoints``: more than one -> folders ``TimePoint_<t>``;
- ``ZSteps``: more than one -> folders ``ZStep_<z>`` (in the time point's
  folder when there are several), ``ZStep_1`` .. ``ZStep_<ZSteps>`` the
  planes; ``ZStep_0`` holds the projection when ``ZProjection`` is TRUE and
  is left out (not read; its TIFFs are parts of the plate, so they do not
  turn up as images of their own).  Each ``TimePoint_`` and ``ZStep_``
  folder carries a copy of the ``.HTD``: a part of the plate's, the outermost
  one (a single-timepoint export may still have a ``TimePoint_1`` folder and
  copies at three levels; a time series' copies list fields whose files are
  in the plate's other timepoint folders).

The files are named ``<plate>_<well>[_s<site>][_w<wave>]<anything>.TIF``
(``_s`` / ``_w`` only when the plate has sites / waves; some versions add a
GUID; ``_thumb`` files are previews and skipped).  The previews are parts of
the ``.HTD`` (docs/COMPOSITE.md): opening one opens the plate.

- One micro-reader image per site, axes ``t c z y x``.
- Indices: ``plate`` 0, ``acquisition`` 0, ``well`` (row then column),
  ``field`` = site - 1.
- Missing files: as for every file set (``_fileset.keep_fields``).
- Metadata: channel names (``WaveName``); pixel sizes from the TIFFs'
  MetaMorph calibration.  Not yet: the z step (the HTD gives none), stage
  positions.
"""
from __future__ import annotations

import csv
import io
import os
import re

from .. import _parts
from .._base import File
from .._errors import CorruptFileError, UnsupportedFormatError
from .._fileset import Members, PlaneImage, keep_fields
from .._metadata import Channel, Metadata
from .._plate import Plate, Well, layout, row_label

NAME = "metaxpress"
#: the index file of a set: a folder holding it opens it (docs/COMPOSITE.md)
SET_INDEX = True
MAGIC = b'"HTSInfoFile"'
NAME_RE = re.compile(r"^(?P<base>.+?)_(?P<well>[A-Za-z]{1,2})(?P<col>\d{1,3})"
                     r"(?:_s(?P<site>\d+))?(?:_w(?P<wave>\d+))?(?P<rest>[^.]*)\.tiff?$",
                     re.IGNORECASE)


TIMEPOINT_FOLDER = re.compile(r"^TimePoint_\d+$", re.IGNORECASE)
ZSTEP_FOLDER = re.compile(r"^ZStep_\d+$", re.IGNORECASE)


def _projection_folder(name: str) -> bool:
    """Whether a folder named *name* is ``ZStep_0``, a z series' projection."""
    return ZSTEP_FOLDER.match(name) is not None and int(name.split("_", 1)[1]) == 0


def may_own(folder: str, name: str, folders) -> bool:
    """``group_files``' cheap test (reads nothing): a ``_thumb`` TIFF may be
    a plate's preview, a TIFF in ``ZStep_0`` its projection; an ``.HTD`` in a
    ``ZStep_`` / ``TimePoint_`` folder a copy of the plate's."""
    lower = name.lower()
    base = folders.base(folder)
    if lower.endswith(".htd"):
        return ZSTEP_FOLDER.match(base) is not None or TIMEPOINT_FOLDER.match(base) is not None
    return lower.endswith((".tif", ".tiff")) and ("thumb" in lower or _projection_folder(base))


def _plate_htd(folder: str, base: str):
    """The plate's ``.HTD`` named *base* for a file in *folder*: the outermost
    of *folder*, the folder above a ``ZStep_`` folder and the one above a
    ``TimePoint_`` folder that holds one (exact names: no listing) -- MetaXpress
    copies the plate's ``.HTD`` into its ``TimePoint_`` and ``ZStep_`` folders,
    and the plate is the one around them; None when none exists."""
    places = [folder]
    if ZSTEP_FOLDER.match(os.path.basename(folder)):
        folder = os.path.dirname(folder)
        places.append(folder)
    if TIMEPOINT_FOLDER.match(os.path.basename(folder)):
        places.append(os.path.dirname(folder))
    for place in reversed(places):
        for ext in (".HTD", ".htd"):
            htd = os.path.join(place, base + ext)
            if os.path.isfile(htd):
                return htd
    return None


def owner(path: str, head):
    """A plate's ``_thumb`` previews, the projection TIFFs in ``ZStep_0`` (not
    read: not planes of the z series) and the copies of its ``.HTD`` in its
    ``TimePoint_`` / ``ZStep_`` folders are parts of the plate's ``.HTD`` --
    only when that file exists (exact names: no folder listing), so a lone
    file named like one is not taken for a part."""
    if os.path.isdir(path):
        return None
    name = os.path.basename(path)
    folder = os.path.dirname(os.path.abspath(path))
    if name.lower().endswith(".htd"):
        if not (ZSTEP_FOLDER.match(os.path.basename(folder))
                or TIMEPOINT_FOLDER.match(os.path.basename(folder))):
            return None
        htd = _plate_htd(folder, os.path.splitext(name)[0])
        same = htd is not None and os.path.normcase(htd) == os.path.normcase(os.path.abspath(path))
        return _parts.Part(htd, htd, standalone=True) if htd and not same else None
    match = NAME_RE.match(name)
    if match is None or not ("thumb" in match["rest"].lower()
                             or _projection_folder(os.path.basename(folder))):
        return None
    htd = _plate_htd(folder, match["base"])
    return _parts.Part(htd, htd, standalone=True) if htd else None


def may_index(name: str) -> bool:
    """``group_files``' cheap test (reads nothing): the plate is an ``.HTD`` file."""
    return name.lower().endswith(('.htd',))


def sniff(head: bytes, path: str) -> bool:
    return MAGIC in head[:64]


def _row_number(letters: str) -> int:
    """0-based row of well letters: A -> 0, Z -> 25, AA -> 26."""
    n = 0
    for ch in letters.upper():
        n = n * 26 + (ord(ch) - ord("A") + 1)
    return n - 1


def parse_htd(text: str) -> dict:
    """``{key: [values]}``: TRUE / FALSE as bool, whole numbers as int."""
    out = {}
    for row in csv.reader(io.StringIO(text), skipinitialspace=True):
        if not row:
            continue
        values = []
        for v in row[1:]:
            v = v.strip()
            if v.upper() in ("TRUE", "FALSE"):
                values.append(v.upper() == "TRUE")
            else:
                try:
                    values.append(int(v))
                except ValueError:
                    values.append(v)
        out[row[0].strip()] = values
    return out


def _one(htd: dict, key: str, default=None):
    values = htd.get(key)
    return values[0] if values else default


def _grid(htd: dict, prefix: str, rows: int, columns: int) -> list:
    """The (row, column) cells set TRUE in ``<prefix>1`` .. ``<prefix><rows>``;
    every cell when the file has no such grid."""
    if f"{prefix}1" not in htd:
        return [(r, c) for r in range(rows) for c in range(columns)]
    cells = []
    for r in range(rows):
        flags = htd.get(f"{prefix}{r + 1}", [])
        cells += [(r, c) for c in range(columns) if c < len(flags) and flags[c] is True]
    return cells


class _MetaXpressImage(PlaneImage):
    def __init__(self, *args, channel_names, first, **kwargs):
        super().__init__(*args, **kwargs)
        self._channel_names = channel_names
        self._first = first                     # a member file: pixel sizes

    def _read_metadata(self):
        entry = self._members.acquire(self._first)
        try:
            sizes = dict(entry[0].images[0].metadata.pixel_sizes)
        finally:
            self._members.release(entry)
        return Metadata(pixel_sizes={a: q for a, q in sizes.items() if a in "xy"},
                        channels=tuple(Channel(name=n) for n in self._channel_names))


class _MetaXpressFile(File):
    backend = NAME
    format = "metaxpress"

    def __init__(self, path: str):
        self.path = path
        self._members = Members()
        with open_text(path) as fh:
            htd = parse_htd(fh.read())
        rows, columns = _one(htd, "YWells"), _one(htd, "XWells")
        if not isinstance(rows, int) or not isinstance(columns, int) or rows < 1 or columns < 1:
            raise CorruptFileError(f"{path!r}: the HTD file gives no plate size (XWells / YWells)")
        n_z = _one(htd, "ZSteps", 1) or 1
        if not isinstance(n_z, int) or n_z < 1:
            n_z = 1
        wells = _grid(htd, "WellsSelection", rows, columns)
        has_sites = _one(htd, "Sites") is True
        sites = (_grid(htd, "SiteSelection", _one(htd, "YSites", 1), _one(htd, "XSites", 1))
                 if has_sites else [(0, 0)])
        has_waves = _one(htd, "Waves") is True
        n_waves = _one(htd, "NWavelengths", 1) if has_waves else 1
        names = [(_one(htd, f"WaveName{w}") or None) if has_waves else None
                 for w in range(1, n_waves + 1)]
        n_times = _one(htd, "TimePoints", 1) or 1

        folder = os.path.dirname(os.path.abspath(path))
        stem = os.path.splitext(os.path.basename(path))[0]
        found = self._find(folder, stem, n_times, n_z, has_sites, has_waves)
        def expected(r, c, s, w, t, z):
            """A missing plane's file as it would be named (for the warning)."""
            name = (f"{stem}_{row_label(r)}{c + 1:02d}" + (f"_s{s}" if has_sites else "")
                    + (f"_w{w}" if has_waves else "") + ".TIF")
            return os.path.join(folder, f"TimePoint_{t}" if n_times > 1 else "",
                                f"ZStep_{z}" if n_z > 1 else "", name)

        listed = {}
        for r, c in wells:
            for s in range(1, len(sites) + 1):
                listed[(r, c, s)] = {(t, w, z): found.get((r, c, s, w, t, z)) or expected(r, c, s, w, t, z)
                                     for t in range(1, n_times + 1)
                                     for w in range(1, n_waves + 1)
                                     for z in range(1, n_z + 1)}
        wanted = [(t, w, z) for t in range(1, n_times + 1) for w in range(1, n_waves + 1)
                  for z in range(1, n_z + 1)]
        kept = keep_fields(listed, wanted, path)

        wells_rc = sorted({(r, c) for r, c, _s in kept})
        number = {rc: n for n, rc in enumerate(wells_rc)}
        first = next(iter(next(iter(kept.values())).values()))
        dtype, unit, shape_yx = self._members.plane_info(first)
        shape = (n_times, n_waves, n_z) + tuple(shape_yx)
        images = []
        for r, c, s in sorted(kept, key=lambda k: (number[k[:2]], k[2])):
            planes = {(t - 1, w - 1, z - 1): p for (t, w, z), p in kept[(r, c, s)].items()}
            images.append(_MetaXpressImage(
                self._members, len(images), planes, shape, dtype,
                f"{stem} well {row_label(r)}{c + 1} field {s - 1}", unit,
                {"plate": 0, "acquisition": 0, "well": number[(r, c)], "field": s - 1},
                channel_names=names, first=first))
        row_names, column_names = layout(rows, columns, wells_rc, path)
        plate = Plate(name=stem, row_names=row_names, column_names=column_names,
                      wells=tuple(Well(r, c, f"{row_label(r)}{c + 1}") for r, c in wells_rc))
        super().__init__(path, images, plates=[plate])

    @staticmethod
    def _find(folder: str, stem: str, n_times: int, n_z: int, has_sites: bool,
              has_waves: bool) -> dict:
        """{(row, col, site, wave, timepoint, z): path} of the plate's TIFFs (z from 1;
        ``ZStep_0``, the projection, is not a plane)."""
        found = {}
        for t in range(1, n_times + 1):
            sub = os.path.join(folder, f"TimePoint_{t}")
            at_t = sub if os.path.isdir(sub) else (folder if n_times == 1 else None)
            if at_t is None:
                continue
            for z in range(1, n_z + 1):
                sub = os.path.join(at_t, f"ZStep_{z}")
                where = sub if os.path.isdir(sub) else (at_t if n_z == 1 else None)
                if where is None:
                    continue
                for name in os.listdir(where):
                    m = NAME_RE.match(name)
                    if not m or m["base"].lower() != stem.lower() or "thumb" in m["rest"].lower():
                        continue
                    if (m["site"] is not None) != has_sites or (m["wave"] is not None) != has_waves:
                        continue
                    key = (_row_number(m["well"]), int(m["col"]) - 1,
                           int(m["site"] or 1), int(m["wave"] or 1), t, z)
                    if key in found:
                        raise CorruptFileError(
                            f"{os.path.join(where, name)!r} and {found[key]!r} are the same plane")
                    found[key] = os.path.join(where, name)
        return found

    def close(self) -> None:
        super().close()
        self._members.close()


def open_text(path: str):
    """The HTD file as text (Latin-1: some descriptions hold Windows-1252
    bytes, and every byte decodes)."""
    return io.open(path, "r", encoding="latin-1", newline="")


def open(path: str, tiles: str = "separate", resolutions: str = "base") -> _MetaXpressFile:  # noqa: A001 - backend protocol
    """A MetaXpress plate from its ``.HTD`` file (*tiles* and *resolutions*
    change nothing: sites are not tiled, and the TIFFs hold no pyramids)."""
    return _MetaXpressFile(path)
