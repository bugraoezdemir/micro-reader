"""Olympus / Evident scanR plates (``experiment_descriptor.xml``): docs/HCS.md.

A scanR acquisition is a plate folder holding

- ``experiment_descriptor.xml`` (and ``.dat``): the settings, as LabVIEW
  ``LVData`` (named values in nested clusters): the plate format (``Rows`` /
  ``Columns``), the plate name, the selected wells;
- ``AcquisitionLog.dat``: each image's stage position and time;
- ``data/``: one TIFF per plane, named ``<well>--W<well no>--P<position>--
  Z<z>--T<t>--<channel>.tif`` (``A1--W00001--P00001--Z00000--T00000--488nm.tif``),
  and ``data/metadata.ome.xml``: an OME companion of every image the
  acquisition planned (one per well and position, files or not), without a
  ``Plate``.

Open the descriptor, the plate folder, the ``data`` folder, a ``.dat`` file or
``metadata.ome.xml``: micro-reader reads the plate (a data TIFF, a member of
the companion, opens it too: ``multifile=False`` reads the one TIFF).

- Read as the companion (backends/companion.py: axes ``t c z y x``, channels,
  pixel sizes, plane positions in the stage system, missing files warned
  once), with its plate from the descriptor's format and the file names: an
  image's well is the ``<well>`` of its files, its field the position - 1.
  Indices ``plate`` 0, ``acquisition`` 0, ``well`` (wells with images, row then
  column), ``field``.
- The plate's name: the descriptor's ``plate name``, else the plate folder's.

Older scanR versions (2007 to about 2015) write no ``data/metadata.ome.xml``:
the plate comes from the descriptor and the file names alone.

- Files ``[<well>]--W<n>--P<n>--Z<n>--T<n>--<channel>.tif`` (scanR 2007 names no
  well); plain 2-D TIFFs.  A data file opens the plate too (``multifile=False``
  reads the one TIFF).
- The layout: the format's ``Rows`` / ``Columns``, else the custom format
  (``custom row`` / ``custom column``: idr0009's 32 x 12 cell array) when it
  holds every well number, else the smallest of 96 / 384 / 1536 wells that
  does.  A well number counts along the rows (``W`` - 1 = row x columns +
  column), as the named files show even with ``meander wells`` set.  Checked
  against what else the descriptor says: a file name or a name of the well
  table (number, name pairs since about 2015) that another number has is
  refused; a ``Format description`` whose number ("384 Well") differs from
  ``Rows`` x ``Columns`` warns (PlateLayoutWarning), and so does a guessed
  layout.  scanR 2007 writes neither names nor positions: its custom format
  is all there is.  Bio-Formats lays idr0009 out as 16 x 24: it ignores the
  custom format.
- Channels: the names in the files, in the descriptor's order of the channel
  configs that are not idle.  Pixel size: ``conversion factor um/pixel``.
  Positions: ``AcquisitionLog.dat``'s IMAGEX / IMAGEY / IMAGEZ of a field's
  first plane, in micrometres (the field spacing equals ``subpos spacer``), in
  the stage system as written; older logs record none.  Not read: the z step
  and the time step (no sample with several of either).
- Pixels: 12-bit values stored with the top bit set (every pixel of the first
  plane is 0x8000 + 0..4095, zenodo 3357662) are read without it, with one
  StoredValuesWarning; any other plate as stored (idr0009's carry no flag; a
  16-bit camera's use bits 12-14).  Bio-Formats keeps the low 12 bits of every
  scanR pixel, which would cut 16-bit data.
- The descriptor is LabVIEW XML that names no encoding and may hold Latin-1
  bytes (``interpolation spacing [µm]``): read as UTF-8, else Latin-1.
"""
from __future__ import annotations

import builtins
import os
import re
import warnings
import xml.etree.ElementTree as ET

import numpy as np

from .. import _parts
from .._base import File
from .._errors import CorruptFileError, PlateLayoutWarning, StoredValuesWarning
from .._fileset import Members, PlaneImage, keep_fields
from .._metadata import Channel, Metadata, quantity
from .._plate import Plate, Well, layout, row_label, well_of
from .._spatial import CoordinateSystem
from ._tiff_metadata import STAGE
from .companion import _CompanionFile, load

NAME = "scanr"
#: the index file of a set: a folder holding it opens it (docs/COMPOSITE.md)
SET_INDEX = True

DESCRIPTOR = "experiment_descriptor.xml"
#: the other files of the plate folder, and the companion in data/: parts
SIDE_FILES = ("experiment_descriptor.dat", "acquisitionlog.dat")
COMPANION = "metadata.ome.xml"
#: a data file: <well>--W<n>--P<n>--Z<n>--T<n>--<channel>.tif
FILE_NAME = re.compile(r"^([A-Za-z]+)(\d+)--W(\d+)--P(\d+)--Z(\d+)--T(\d+)--(.+)\.tiff?$")
#: a data file of an older plate, whose name may start without the well:
#: --W<n>--P<n>--Z<n>--T<n>--<channel>.tif (scanR 2007)
OLD_FILE_NAME = re.compile(r"^(?:([A-Za-z]+)(\d+))?--W(\d+)--P(\d+)--Z(\d+)--T(\d+)--(.+)\.tiff?$",
                           re.IGNORECASE)
#: the acquisition log's line for one plane: IMAGEPOS W=.. P=.. T=.. Z=.. [IMAGEX=.. ...]
LOG_LINE = re.compile(r"^IMAGEPOS\b(.*)$")
#: the plate formats scanR offers, smallest first, when the descriptor names none
#: (Bio-Formats guesses by the number of wells the same way)
GUESSED_LAYOUTS = ((8, 12), (16, 24), (32, 48))


def may_index(name: str) -> bool:
    """``group_files``' cheap test (reads nothing): the descriptor's name."""
    return name.lower() == DESCRIPTOR


def may_own(folder: str, name: str, folders) -> bool:
    """``group_files``' cheap test: the plate folder's other files may be parts."""
    return name.lower() in SIDE_FILES + (COMPANION,)


def sniff(head: bytes, path: str) -> bool:
    return os.path.basename(path).lower() == DESCRIPTOR and b"LVData" in head


def _named(folder: str, lower: str):
    """The file in *folder* whose name is *lower* in any letter case, or None."""
    try:
        with os.scandir(folder) as entries:
            return next((e.path for e in entries if e.is_file() and e.name.lower() == lower), None)
    except OSError:
        return None


def owner(path: str, head):
    """The plate folder, its ``data`` folder, its ``.dat`` files and
    ``data/metadata.ome.xml`` are parts of the descriptor (module docstring)."""
    path = os.path.abspath(path)
    name = os.path.basename(path).lower()
    if os.path.isdir(path):
        folder = os.path.dirname(path) if name == "data" else path
    elif name in SIDE_FILES:
        folder = os.path.dirname(path)
    elif name == COMPANION and os.path.basename(os.path.dirname(path)).lower() == "data":
        folder = os.path.dirname(os.path.dirname(path))
    else:
        return None
    descriptor = _named(folder, DESCRIPTOR)
    return _parts.Part(descriptor, descriptor) if descriptor is not None else None


def _root(path: str):
    """The descriptor's XML.  LabVIEW writes it without an encoding and may put
    Latin-1 bytes in it (``interpolation spacing [µm]``): read as UTF-8, else as
    Latin-1, as Bio-Formats does; UTF-16 when it starts with that byte-order mark."""
    with builtins.open(path, "rb") as fh:
        raw = fh.read()
    if raw[:2] in (b"\xff\xfe", b"\xfe\xff"):
        text = raw.decode("utf-16")
    else:
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            text = raw.decode("latin-1")
    if text.startswith("<?"):
        text = text[text.index("?>") + 2:]          # the declaration names no encoding
    try:
        return ET.fromstring(text)
    except ET.ParseError as exc:
        raise CorruptFileError(f"{path!r}: not readable XML ({exc})") from None


def _settings(path: str, root=None) -> dict:
    """{name: value} of the descriptor's named values (the first of each name)."""
    root = _root(path) if root is None else root
    out = {}
    for e in root.iter():
        name, val = e.find("{*}Name"), e.find("{*}Val")
        if name is not None and val is not None and name.text:
            out.setdefault(name.text.strip(), (val.text or "").strip())
    return out


def _row(letters: str) -> int:
    n = 0
    for ch in letters.upper():
        n = n * 26 + ord(ch) - ord("A") + 1
    return n - 1


def _plan(path: str, core) -> list:
    """The plate as ``Core.plates`` gives one: layout and name from the
    descriptor, each image's well and field from its files' names."""
    settings = _settings(path)
    try:
        rows, columns = int(settings.get("Rows") or 0), int(settings.get("Columns") or 0)
    except ValueError:
        raise CorruptFileError(f"{path!r}: the plate format's Rows / Columns are not whole "
                               "numbers") from None
    places, wells = {}, set()
    for s in core.series:
        found = {FILE_NAME.match(os.path.basename(member))
                 for member, _ifd in s.planes.values()}
        keys = {(m.group(1), int(m.group(2)), int(m.group(4))) for m in found if m}
        if not keys:
            continue                                  # not a scanR data file: beside the plate
        if len(keys) > 1 or None in found:
            raise CorruptFileError(f"{path!r}: image {s.name!r} has files of several wells or "
                                   f"positions: {sorted(keys)[:4]}")
        letters, column, position = keys.pop()
        at = (_row(letters), column - 1)
        wells.add(at)
        places[s.number] = (*at, 0, position - 1)
    return [{"name": settings.get("plate name") or os.path.basename(os.path.dirname(path)),
             "rows": rows or max([r + 1 for r, _ in wells] + [1]),
             "columns": columns or max([c + 1 for _, c in wells] + [1]),
             "wells": wells, "acquisitions": [], "places": places}]


class _ScanrFile(_CompanionFile):
    backend = NAME
    format = "scanr"

    def __init__(self, path: str, companion: str):
        super().__init__(path, plates=True, companion=companion,
                         plan=_plan(path, load(companion)))


# -- older plates: no data/metadata.ome.xml ----------------------------------------------

def _listing(folder: str) -> dict:
    """*folder*'s entries, listed once: {name in lower case: (path, is a folder)}."""
    with os.scandir(folder) as entries:
        return {e.name.lower(): (e.path, e.is_dir()) for e in entries}


def _data_folder(listing: dict):
    """The ``data`` folder (any letter case) of a plate folder's *listing*, or None."""
    path, is_dir = listing.get("data", (None, False))
    return path if is_dir else None


def old_plate_of(path: str):
    """The descriptor of the older scanR plate (no ``data/metadata.ome.xml``) whose
    data file *path* is, or None: what a data TIFF opens with ``multifile=True``
    (backends/tiff.py)."""
    if not OLD_FILE_NAME.match(os.path.basename(path)):
        return None
    data = os.path.dirname(os.path.abspath(path))
    if os.path.basename(data).lower() != "data" or _named(data, COMPANION) is not None:
        return None
    return _named(os.path.dirname(data), DESCRIPTOR)


def _channel_order(root) -> list:
    """The descriptor's channel names in its order: each ``channel config``
    that is not idle (Bio-Formats' rule)."""
    names = []
    for cluster in root.iter("Cluster"):
        label = cluster.find("Name")
        if label is None or (label.text or "").strip() != "channel config":
            continue
        values = {(e.findtext("Name") or "").strip(): (e.findtext("Val") or "").strip()
                  for e in cluster}
        if values.get("idle", "0") == "0" and values.get("name"):
            names.append(values["name"])
    return names


def _well_table(root) -> dict:
    """{well number: well name} from the descriptor's ``well selection table +
    cDNA`` (number, name pairs; scanR 2007 writes the numbers only: {})."""
    for e in root.iter("Array"):
        if (e.findtext("Name") or "").strip() != "well selection table + cDNA":
            continue
        values = [(v.text or "").strip() for v in e.iter("Val")]
        return {int(n): name for n, name in zip(values[::2], values[1::2])
                if n.isdigit() and name and not name[0].isdigit()}
    return {}


def _layout_of(path: str, settings: dict, numbers, table: dict) -> tuple:
    """(rows, columns) of an older plate: the format's ``Rows`` / ``Columns``,
    else the custom format (``custom row`` / ``custom column``) when it holds
    every well number, else the smallest of GUESSED_LAYOUTS that does (with a
    PlateLayoutWarning: a guess).  Checked against what else the descriptor
    says: the number in ``Format description`` ("384 Well": a PlateLayoutWarning
    when it differs) and the names of the well table (a CorruptFileError when a
    number names another well)."""
    def whole(name):
        try:
            return int(settings.get(name) or 0)
        except ValueError:
            raise CorruptFileError(f"{path!r}: the descriptor's {name!r} is not a whole "
                                   "number") from None
    most = max(list(numbers) + list(table))
    chosen = None
    for source, (rows, columns) in (("plate format", (whole("Rows"), whole("Columns"))),
                                    ("custom format", (whole("custom row"),
                                                       whole("custom column")))):
        if rows > 0 and columns > 0 and rows * columns >= most:
            chosen = source, rows, columns
            break
    if chosen is None:
        rows, columns = next((rc for rc in GUESSED_LAYOUTS if rc[0] * rc[1] >= most), (most, 1))
        chosen = "guess", rows, columns
        warnings.warn(PlateLayoutWarning(
            f"{path!r}: the descriptor gives no plate format that holds well number {most}: "
            f"laid out as the smallest standard plate that does, {rows} x {columns}"),
            stacklevel=4)
    source, rows, columns = chosen
    described = re.match(r"\s*(\d+)", settings.get("Format description") or "")
    if source == "plate format" and described and int(described.group(1)) != rows * columns:
        warnings.warn(PlateLayoutWarning(
            f"{path!r}: the plate format is {rows} x {columns}, but its description says "
            f"{settings['Format description']!r}: read as {rows} x {columns}"), stacklevel=4)
    for number, name in sorted(table.items()):
        at = divmod(number - 1, columns)
        if well_of(name) != at:
            raise CorruptFileError(
                f"{path!r}: the descriptor's well table names well number {number} {name!r}, "
                f"but in the {rows} x {columns} {source} it is well "
                f"{row_label(at[0])}{at[1] + 1}")
    return rows, columns


def _log_positions(folder: str, listing: dict = None) -> dict:
    """{(well no, position, t, z): {"x", "y", "z": micrometres}} from the plate's
    ``AcquisitionLog.dat`` (scanR 2015 and later write IMAGEX / IMAGEY / IMAGEZ;
    older logs only the plane's numbers: {}).  *listing*: the plate folder's
    (``_listing``), when it has been listed already."""
    if listing is None:
        log = _named(folder, "acquisitionlog.dat")
    else:
        log, is_dir = listing.get("acquisitionlog.dat", (None, True))
        log = None if is_dir else log
    out = {}
    if log is None:
        return out
    with builtins.open(log, encoding="latin-1") as fh:
        for line in fh:
            m = LOG_LINE.match(line.strip())
            if not m:
                continue
            fields = dict(kv.split("=", 1) for kv in m.group(1).split() if "=" in kv)
            try:
                key = tuple(int(fields[k]) for k in ("W", "P", "T", "Z"))
                at = {a: float(fields[f"IMAGE{a.upper()}"]) for a in "xyz"
                      if f"IMAGE{a.upper()}" in fields}
            except (KeyError, ValueError):
                continue
            if at:
                out[key] = at
    return out


def _flagged(path: str, dtype) -> bool:
    """Whether the plate's 16-bit pixels are 12-bit values stored with the top bit
    set (0x8000 + 0..4095), from its first plane: every pixel has the top bit and
    none of bits 12-14 (zenodo 3357662).  A 16-bit camera's plane fails the test."""
    if np.dtype(dtype) != np.uint16:
        return False
    import tifffile
    plane = tifffile.imread(path)
    return bool(plane.size) and bool(np.all((plane & 0xF000) == 0x8000))


class _OldScanrImage(PlaneImage):
    """One field of an older scanR plate: its TIFF planes, with the plate's pixel
    size, the channel names and the field's stage position; *flagged*: the top
    bit of every pixel is cleared (``_flagged``)."""

    def __init__(self, *args, sizes: dict, channels: list, position: dict,
                 flagged: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self._sizes, self._channels, self._stage = sizes, channels, position
        self._flag = flagged

    def _read_block(self, block):
        out, read = super()._read_block(block)
        if self._flag:
            np.bitwise_and(out, 0x7FFF, out=out)       # the stored top bit is no signal
        return out, read

    def _read_metadata(self):
        return Metadata(pixel_sizes=self._sizes,
                        channels=tuple(Channel(name=n) for n in self._channels),
                        position=self._stage,
                        coordinate_system=STAGE if self._stage else None)


class _OldScanrFile(File):
    """An older scanR plate: the plate from the descriptor and the data file names
    alone (module docstring)."""

    backend = NAME
    format = "scanr"

    def __init__(self, path: str, data: str, data_listing: dict = None, listing: dict = None):
        """*data_listing*, *listing*: the data folder's and the plate folder's
        (``_listing``), when ``open`` has listed them (each folder once)."""
        self.path = path
        root = _root(path)
        settings = _settings(path, root)
        files = {}                      # (row or None, col or None, W, P) -> {(t, ch, z): path}
        for entry, is_dir in (data_listing if data_listing is not None else _listing(data)).values():
            m = None if is_dir else OLD_FILE_NAME.match(os.path.basename(entry))
            if m:
                letters, column, w, p, z, t, ch = m.groups()
                files.setdefault((letters, column, int(w), int(p)), {})[
                    (int(t), ch, int(z))] = entry
        if not files:
            raise CorruptFileError(f"{path!r}: its data folder holds no scanR data files "
                                   "(--W<n>--P<n>--Z<n>--T<n>--<channel>.tif)")
        rows, columns = _layout_of(path, settings, [w for _l, _c, w, _p in files],
                                   _well_table(root))
        known = _channel_order(root)
        found = {ch for planes in files.values() for _t, ch, _z in planes}
        channels = [n for n in known if n in found] + sorted(found - set(known))
        times = sorted({t for planes in files.values() for t, _c, _z in planes})
        zs = sorted({z for planes in files.values() for _t, _c, z in planes})

        listed = {}                     # (row, col, field) -> {(t, c, z): path}
        number_of = {}                  # (row, col) -> the well number of its files
        for (letters, column, w, p), planes in files.items():
            at = divmod(w - 1, columns)
            if letters is not None:
                named = (_row(letters), int(column) - 1)
                if named != at:
                    raise CorruptFileError(
                        f"{path!r}: data files of well {letters}{column} are numbered W{w:05d}, "
                        f"which is well {row_label(at[0])}{at[1] + 1} of the {rows} x {columns} "
                        "plate format the descriptor gives")
            if at[0] >= rows:
                raise CorruptFileError(f"{path!r}: well number {w} lies outside the {rows} x "
                                       f"{columns} plate format")
            number_of[at] = w
            if at + (p - 1,) in listed:
                raise CorruptFileError(f"{path!r}: two sets of data files name well {w} "
                                       f"position {p}; open one data file with multifile=False")
            listed[at + (p - 1,)] = {(times.index(t), channels.index(ch), zs.index(z)): f
                                     for (t, ch, z), f in planes.items()}
        wanted = [(t, c, z) for t in range(len(times)) for c in range(len(channels))
                  for z in range(len(zs))]
        kept = keep_fields(listed, wanted, path)

        wells = sorted({(r, c) for r, c, _f in kept})
        number = {rc: n for n, rc in enumerate(wells)}
        self._members = Members()
        first = next(iter(next(iter(kept.values())).values()))
        dtype, unit, shape_yx = self._members.plane_info(first)
        flagged = _flagged(first, dtype)
        if flagged:
            warnings.warn(StoredValuesWarning(
                f"{path!r}: the plate's pixels are 12-bit values stored with the top bit set "
                f"(0x8000 + value, e.g. {os.path.basename(first)!r}); read without it, as the "
                "measured values"), stacklevel=3)
        try:
            size = float(settings.get("conversion factor um/pixel") or 0)
        except ValueError:
            size = 0.0
        sizes = {a: quantity(size, "micrometer") for a in "xy"} if size > 0 else {}
        log = _log_positions(os.path.dirname(os.path.abspath(path)), listing)
        name = settings.get("plate name") or os.path.basename(os.path.dirname(os.path.abspath(path)))
        images = []
        for r, c, f in sorted(kept, key=lambda k: (number[k[:2]], k[2])):
            at = log.get((number_of[(r, c)], f + 1, times[0], zs[0]), {})
            position = {a: quantity(v, "micrometer") for a, v in at.items()}
            images.append(_OldScanrImage(
                self._members, len(images), kept[(r, c, f)],
                (len(times), len(channels), len(zs)) + tuple(shape_yx), dtype,
                f"{name} well {row_label(r)}{c + 1} field {f}", unit,
                {"plate": 0, "acquisition": 0, "well": number[(r, c)], "field": f},
                sizes=sizes, channels=channels, position=position, flagged=flagged))
        self._placed = any(image._stage for image in images)
        row_names, column_names = layout(rows, columns, wells, path)
        plate = Plate(name=name, row_names=row_names, column_names=column_names,
                      wells=tuple(Well(r, c, f"{row_label(r)}{c + 1}") for r, c in wells))
        super().__init__(path, images, plates=[plate])

    def _declare_coordinate_systems(self):
        return (CoordinateSystem(STAGE),) if self._placed else ()

    def close(self) -> None:
        super().close()
        self._members.close()


def open(path: str, tiles: str = "separate", resolutions: str = "base"):  # noqa: A001 - backend protocol
    """The plate of a scanR descriptor (*tiles* and *resolutions* change nothing):
    read through ``data/metadata.ome.xml`` when the plate has one, else from the
    descriptor and the data file names (older scanR versions)."""
    folder = os.path.dirname(os.path.abspath(path))
    listing = _listing(folder)                  # each folder listed once
    data = _data_folder(listing)
    data_listing = _listing(data) if data is not None else {}
    companion, is_dir = data_listing.get(COMPANION, (None, True))
    if companion is not None and not is_dir:
        return _ScanrFile(path, companion)
    if data is None:
        raise CorruptFileError(f"{path!r}: the scanR plate folder {folder!r} has no data "
                               "folder (its images)")
    return _OldScanrFile(path, data, data_listing, listing)
