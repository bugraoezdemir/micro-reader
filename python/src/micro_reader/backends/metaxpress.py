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
- ``TimePoints``: more than one -> folders ``TimePoint_<t>``.

The files are named ``<plate>_<well>[_s<site>][_w<wave>]<anything>.TIF``
(``_s`` / ``_w`` only when the plate has sites / waves; some versions add a
GUID; ``_thumb`` files are previews and skipped).  The previews are parts of
the ``.HTD`` (docs/COMPOSITE.md): opening one opens the plate.

- One micro-reader image per site, axes ``t c z y x`` (z = 1).
- Indices: ``plate`` 0, ``acquisition`` 0, ``well`` (row then column),
  ``field`` = site - 1.
- Missing files: as for every file set (``_fileset.keep_fields``).
- Metadata: channel names (``WaveName``); pixel sizes from the TIFFs'
  MetaMorph calibration.  Not yet: z-series (``ZSteps``), stage positions.
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


def may_own(folder: str, name: str, folders) -> bool:
    """``group_files``' cheap test (reads nothing): a ``_thumb`` TIFF may be
    a plate's preview."""
    lower = name.lower()
    return "thumb" in lower and lower.endswith((".tif", ".tiff"))


def owner(path: str, head):
    """A plate's ``_thumb`` previews are parts of its ``.HTD`` -- only when
    that file exists (exact names: no folder listing), so a lone file named
    like one is not taken for a part."""
    match = NAME_RE.match(os.path.basename(path))
    if match is None or "thumb" not in match["rest"].lower() or os.path.isdir(path):
        return None
    folder = os.path.dirname(os.path.abspath(path))
    places = [folder]
    if TIMEPOINT_FOLDER.match(os.path.basename(folder)):
        places.append(os.path.dirname(folder))
    for place in places:
        for ext in (".HTD", ".htd"):
            htd = os.path.join(place, match["base"] + ext)
            if os.path.isfile(htd):
                return _parts.Part(htd, htd, standalone=True)
    return None


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
        if (_one(htd, "ZSteps", 1) or 1) > 1:
            raise UnsupportedFormatError(f"{path!r}: z-series (ZSteps) are not read yet")
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
        found = self._find(folder, stem, n_times, has_sites, has_waves)
        def expected(r, c, s, w, t):
            """A missing plane's file as it would be named (for the warning)."""
            name = (f"{stem}_{row_label(r)}{c + 1:02d}" + (f"_s{s}" if has_sites else "")
                    + (f"_w{w}" if has_waves else "") + ".TIF")
            return os.path.join(folder, f"TimePoint_{t}" if n_times > 1 else "", name)

        listed = {}
        for r, c in wells:
            for s in range(1, len(sites) + 1):
                listed[(r, c, s)] = {(t, w, 1): found.get((r, c, s, w, t)) or expected(r, c, s, w, t)
                                     for t in range(1, n_times + 1)
                                     for w in range(1, n_waves + 1)}
        wanted = [(t, w, 1) for t in range(1, n_times + 1) for w in range(1, n_waves + 1)]
        kept = keep_fields(listed, wanted, path)

        wells_rc = sorted({(r, c) for r, c, _s in kept})
        number = {rc: n for n, rc in enumerate(wells_rc)}
        first = next(iter(next(iter(kept.values())).values()))
        dtype, unit, shape_yx = self._members.plane_info(first)
        shape = (n_times, n_waves, 1) + tuple(shape_yx)
        images = []
        for r, c, s in sorted(kept, key=lambda k: (number[k[:2]], k[2])):
            planes = {(t - 1, w - 1, 0): p for (t, w, _z), p in kept[(r, c, s)].items()}
            images.append(_MetaXpressImage(
                self._members, len(images), planes, shape, dtype,
                f"{stem} well {row_label(r)}{c + 1} field {s - 1}", unit,
                {"plate": 0, "acquisition": 0, "well": number[(r, c)], "field": s - 1},
                channel_names=names, first=first))
        row_names, column_names = layout(rows, columns, wells_rc, path)
        plate = Plate(name=stem, rows=row_names, columns=column_names,
                      wells=tuple(Well(r, c, f"{row_label(r)}{c + 1}") for r, c in wells_rc))
        super().__init__(path, images, plates=[plate])

    @staticmethod
    def _find(folder: str, stem: str, n_times: int, has_sites: bool, has_waves: bool) -> dict:
        """{(row, col, site, wave, timepoint): path} of the plate's TIFFs."""
        found = {}
        for t in range(1, n_times + 1):
            sub = os.path.join(folder, f"TimePoint_{t}")
            where = sub if os.path.isdir(sub) else (folder if n_times == 1 else None)
            if where is None:
                continue
            for name in os.listdir(where):
                m = NAME_RE.match(name)
                if not m or m["base"].lower() != stem.lower() or "thumb" in m["rest"].lower():
                    continue
                if (m["site"] is not None) != has_sites or (m["wave"] is not None) != has_waves:
                    continue
                key = (_row_number(m["well"]), int(m["col"]) - 1,
                       int(m["site"] or 1), int(m["wave"] or 1), t)
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
