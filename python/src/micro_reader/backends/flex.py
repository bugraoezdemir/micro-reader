"""PerkinElmer Opera Flex plates (``.flex``): docs/HCS.md.

A ``.flex`` file is a TIFF holding the images of one well (or one field of
it): its pages are planes, described by the XML in its first page's FlexXML
tag (65200).  A plate is the folder of its ``.flex`` files; there is no index
file.  Open any ``.flex`` file, or the folder: micro-reader reads the plate.

- The set: every ``.flex`` file in the folder (``multifile=True``, the
  default; as Bio-Formats groups them), one plate per ``Barcode``;
  ``multifile=False`` reads the one file.  Subfolders (Opera's ``Meas_01``,
  ``Meas_02``, ... measurements) are not searched: each folder is a plate.
- Each ``<Image>`` is one page (``BufferNo``): its field (``Sublayout``), z
  (``Stack``), time point (``Timepoint``, 1 when absent) and channel (the
  pair ``ExposureNo``, ``CameraRef``).  The well: ``WellCoordinate`` Row / Col
  (1-based); the layout: the Plate's ``XSize`` rows x ``YSize`` columns.
- One micro-reader image per field, axes ``t c z y x``, in the order of the
  values (channels by exposure, then camera, as listed).  Indices: ``plate``
  (plates by barcode, in name order), ``acquisition`` 0, ``well`` (the wells
  with images, row then column), ``field`` = Sublayout - 1.  A field whose
  planes do not fill its t x c x z grid is refused (ragged; no sample has
  one), as is a scaled intensity (``ImageFactor`` other than 1).
- The plate's name: its ``Barcode`` (``PlateName`` is the plate type), else
  the folder's name.
- Metadata: pixel sizes (``ImageResolutionX/Y``, metres; z from the planes'
  ``PositionZ`` steps, else ``ImageResolutionZ``), channel names
  (``ChannelName``, else the page's ``Array`` name, e.g. ``Exp1Cam1``, as
  Bio-Formats names them), and the position of the field's first pixel
  relative to its WELL'S CENTRE: ``PositionX`` / ``PositionY`` are the
  field's centre, read as written (y not flipped), as Bio-Formats 8.0.1
  reports them; the fields of the samples do not overlap, so the y
  direction is not checked against the pixels.  z: the first plane's
  ``PositionZ``.
- Coordinate systems: one per well (``well-B2``; plan 4.9), as for Harmony.
- The parsed files are cached per process (``FILE_CACHE`` files, by path,
  modification time and size): every file of a plate opens the plate.
"""
from __future__ import annotations

import os
import threading
import xml.etree.ElementTree as ET
from collections import OrderedDict

from .. import _parts
from .._base import File
from .._errors import CorruptFileError, UnsupportedFormatError
from .._fileset import Members, PlaneImage
from .._metadata import Channel, Metadata, quantity
from .._plate import Plate, Well, layout, row_label, well_system, well_systems

NAME = "flex"
#: takes multifile=False: the one file instead of its folder's plate
MULTIFILE = True
SUFFIX = ".flex"
#: the TIFF tag holding the FlexXML
FLEX_TAG = 65200
#: the first bytes of a TIFF / BigTIFF, either byte order
TIFF_MAGIC = (b"II*\x00", b"MM\x00*", b"II+\x00", b"MM\x00+")
#: parsed .flex files kept per process (a plate opened from each of its files)
FILE_CACHE = 4096
#: member files kept open at once
OPEN_MEMBERS = 64


def may_index(name: str) -> bool:
    """``group_files``' cheap test (reads nothing): a ``.flex`` file, which
    opens its folder's plate."""
    return name.lower().endswith(SUFFIX)


def sniff(head: bytes, path: str) -> bool:
    return path.lower().endswith(SUFFIX) and head[:4] in TIFF_MAGIC


def owner(path: str, head):
    """A folder holding ``.flex`` files is a part of the plate they make: it
    opens the first of them, which opens the folder (module docstring)."""
    if head is not None or not os.path.isdir(path):
        return None
    files = _flex_files(path)
    return _parts.Part(files[0], files[0]) if files else None


def _flex_files(folder: str) -> list:
    with os.scandir(folder) as entries:
        return sorted(e.path for e in entries if e.is_file() and may_index(e.name))


# -- one file, parsed ----------------------------------------------------------------

class _Parsed:
    """What one ``.flex`` file's XML says: its plate, well and pages."""

    def __init__(self, path: str):
        self.path = path
        root = _flex_root(path)
        plate = _first(root, "Plate")
        well = _first(root, "Well")
        coord = _first(well, "WellCoordinate") if well is not None else None
        if coord is None:
            raise CorruptFileError(f"{path!r}: its FlexXML names no well (WellCoordinate)")
        try:
            self.well = (int(coord.get("Row")) - 1, int(coord.get("Col")) - 1)
        except (TypeError, ValueError):
            raise CorruptFileError(f"{path!r}: WellCoordinate without a whole-number Row / "
                                   "Col") from None
        if min(self.well) < 0:
            raise CorruptFileError(f"{path!r}: WellCoordinate {coord.attrib} (they count from 1)")
        self.barcode = _text(plate, "Barcode") or ""
        self.plate_type = _text(plate, "PlateName") or ""
        self.rows, self.columns = (_whole(plate, "XSize", path, 0), _whole(plate, "YSize", path, 0))
        arrays = [e for e in root.iter() if _local(e.tag) == "Array"]
        #: page (BufferNo) -> {"field", "t", "z", "channel": (exposure, camera), "name", ...}
        self.pages = {}
        for e in (e for e in root.iter() if _local(e.tag) == "Image" and e.get("BufferNo")):
            page = int(e.get("BufferNo"))
            if page in self.pages:
                raise CorruptFileError(f"{path!r}: page {page} (BufferNo) is listed twice")
            factor = _text(e, "ImageFactor")
            if factor not in (None, "") and float(factor) != 1:
                raise UnsupportedFormatError(
                    f"{path!r} page {page}: ImageFactor {factor} (scaled intensities) is not "
                    "read yet")
            array = arrays[page].get("Name") if page < len(arrays) else None
            exposure = _whole(e, "ExposureNo", path, 1)
            camera = _text(e, "CameraRef") or ""
            self.pages[page] = {
                "field": _whole(e, "Sublayout", path, 1) - 1,
                "t": _whole(e, "Timepoint", path, 1),
                "z": _whole(e, "Stack", path, 1),
                "channel": (exposure, camera),
                "name": _text(e, "ChannelName") or array or f"Exp{exposure}{camera}",
                "element": e,
            }
        if not self.pages:
            raise CorruptFileError(f"{path!r}: its FlexXML lists no images")


def _flex_root(path: str):
    import tifffile
    try:
        with tifffile.TiffFile(path) as tif:
            tag = tif.pages.first.tags.get(FLEX_TAG)
            value = tag.value if tag is not None else None
    except tifffile.TiffFileError as exc:
        raise CorruptFileError(f"{path!r}: {exc}") from None
    if value is None:
        raise CorruptFileError(f"{path!r}: a .flex file without its FlexXML (tag {FLEX_TAG})")
    if isinstance(value, bytes):
        value = value.decode("utf-8", "replace")
    try:
        return ET.fromstring(value.strip("\x00"))
    except ET.ParseError as exc:
        raise CorruptFileError(f"{path!r}: its FlexXML is not readable XML ({exc})") from None


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _first(e, name):
    return None if e is None else next((x for x in e.iter() if _local(x.tag) == name), None)


def _text(e, name):
    child = None if e is None else next((x for x in e if _local(x.tag) == name), None)
    return None if child is None or child.text is None else child.text.strip()


def _whole(e, name, path, default):
    value = _text(e, name)
    if value in (None, ""):
        return default
    try:
        return int(value)
    except ValueError:
        raise CorruptFileError(f"{path!r}: <{name}> {value!r} is not a whole number") from None


def _value(e, name, positive=True):
    """A length / time in the unit its element names (``Unit``, metres)."""
    child = None if e is None else next((x for x in e if _local(x.tag) == name), None)
    if child is None:
        return None
    return quantity(child.text, child.get("Unit") or "m", positive=positive)


_cache: "OrderedDict[tuple, _Parsed]" = OrderedDict()
_cache_lock = threading.Lock()


def parse(path: str) -> _Parsed:
    """The parsed ``.flex`` file at *path*, parsed once per process while it
    does not change (by path, modification time and size)."""
    path = os.path.abspath(path)
    stat = os.stat(path)
    key = (os.path.normcase(path), stat.st_mtime_ns, stat.st_size)
    with _cache_lock:
        found = _cache.get(key)
        if found is not None:
            _cache.move_to_end(key)
            return found
    parsed = _Parsed(path)
    with _cache_lock:
        _cache[key] = parsed
        while len(_cache) > FILE_CACHE:
            _cache.popitem(last=False)
    return parsed


# -- the images ------------------------------------------------------------------------

class _FlexImage(PlaneImage):
    """One field; metadata from its pages' ``<Image>`` entries."""

    def __init__(self, *args, entries: dict, names: list, system: str, **kwargs):
        super().__init__(*args, **kwargs)
        #: (t, c, z) position -> its <Image> element
        self._entries = entries
        self._names = names
        #: the id of the coordinate system the position is in: its well's (4.9)
        self._system = system

    def _read_metadata(self):
        first = self._entries[(0, 0, 0)]
        sizes = {axis: _value(first, name) for axis, name in
                 (("x", "ImageResolutionX"), ("y", "ImageResolutionY"))}
        nt, _nc, nz = self.shape[:3]
        if nz > 1:
            a, b = (_value(self._entries[(0, 0, z)], "PositionZ", False) for z in (0, 1))
            if a is not None and b is not None and a.unit == b.unit and b.value != a.value:
                sizes["z"] = quantity(abs(b.value - a.value), a.unit)
            else:
                sizes["z"] = _value(first, "ImageResolutionZ")
        if nt > 1:
            for name in ("TimeOffset", "MeasurementTimeOffset"):
                a, b = (_value(self._entries[(t, 0, 0)], name, False) for t in (0, 1))
                if a is not None and b is not None and a.unit == b.unit:
                    sizes["t"] = quantity(b.value - a.value, a.unit)
                    break
        sizes = {a: q for a, q in sizes.items() if q}
        position = self._field_position(first, sizes)
        return Metadata(pixel_sizes=sizes, position=position,
                        coordinate_system=self._system if position else None,
                        channels=tuple(Channel(name=n) for n in self._names))

    def _field_position(self, e, sizes) -> dict:
        """The centre of the field's first pixel relative to the well's centre:
        ``PositionX`` / ``PositionY`` are the field's centre, y as written (as
        Bio-Formats reports them), so x0 = X - (W - 1) / 2 * px and
        y0 = Y - (H - 1) / 2 * py; z is the first plane's ``PositionZ``."""
        out = {}
        h, w = self.shape[3:5]
        for axis, name, n in (("x", "PositionX", w), ("y", "PositionY", h)):
            q, px = _value(e, name, positive=False), sizes.get(axis)
            if q is None or px is None:
                continue
            px = px.to(q.unit)
            out[axis] = quantity(q.value - (n - 1) / 2 * px.value, q.unit, positive=False)
        z = _value(e, "PositionZ", positive=False)
        if z is not None:
            out["z"] = z
        return {a: q for a, q in out.items() if q is not None}


class _FlexFile(File):
    backend = NAME
    format = "opera-flex"

    def __init__(self, path: str, multifile: bool = True):
        folder = os.path.dirname(os.path.abspath(path))
        paths = _flex_files(folder) if multifile else [os.path.abspath(path)]
        parsed = [parse(p) for p in paths]
        # plates by barcode (else the folder's name), in name order
        by_plate: dict = {}
        for f in parsed:
            by_plate.setdefault(f.barcode or os.path.basename(folder), []).append(f)
        self._members = Members(limit=OPEN_MEMBERS, pages=True)
        plates, fields = [], []
        try:
            for p, name in enumerate(sorted(by_plate)):
                files = by_plate[name]
                listed: dict = {}                # (row, col, field) -> [(file, page, info)]
                for f in files:
                    for page, info in sorted(f.pages.items()):
                        listed.setdefault(f.well + (info["field"],), []).append((f, page, info))
                wells = sorted({k[:2] for k in listed})
                rows = max(f.rows for f in files)
                columns = max(f.columns for f in files)
                row_names, column_names = layout(rows, columns, wells, f"{folder} plate {name}")
                plates.append(Plate(name=name, row_names=row_names, column_names=column_names,
                                    wells=tuple(Well(r, c, f"{row_label(r)}{c + 1}")
                                                for r, c in wells)))
                number = {rc: n for n, rc in enumerate(wells)}
                for key in sorted(listed, key=lambda k: (number[k[:2]], k[2])):
                    fields.append((p, number[key[:2]], key, listed[key]))
            images = []
            for p, w, key, planes in fields:
                images.append(self._field(plates, p, w, key, planes, len(images)))
        except BaseException:
            self._members.close()
            raise
        super().__init__(path, images, plates=plates)

    def _field(self, plates, p, w, key, planes, position):
        r, c, field = key
        where = f"{planes[0][0].path!r} well {row_label(r)}{c + 1} field {field}"
        ts = sorted({info["t"] for _f, _pg, info in planes})
        zs = sorted({info["z"] for _f, _pg, info in planes})
        channels, names = [], []
        for _f, _pg, info in sorted(planes, key=lambda x: x[2]["channel"][0]):
            if info["channel"] not in channels:
                channels.append(info["channel"])
                names.append(info["name"])
        grid, entries = {}, {}
        for f, page, info in planes:
            at = (ts.index(info["t"]), channels.index(info["channel"]), zs.index(info["z"]))
            if at in grid:
                raise CorruptFileError(f"{where}: time point {info['t']}, channel "
                                       f"{info['channel']}, z {info['z']} is listed twice")
            grid[at] = (f.path, page)
            entries[at] = info["element"]
        n = len(ts) * len(channels) * len(zs)
        if len(grid) != n:
            raise UnsupportedFormatError(
                f"{where}: {len(grid)} planes do not fill its {len(ts)} time points x "
                f"{len(channels)} channels x {len(zs)} z (channels sampled differently are "
                "not read yet)")
        member, ifd = grid[(0, 0, 0)]
        dtype, unit, shape_yx = self._members.page_info(member, ifd)
        plate = plates[p]
        name = f"{plate.name} well {row_label(r)}{c + 1} field {field}"
        return _FlexImage(self._members, position, grid,
                          (len(ts), len(channels), len(zs)) + tuple(shape_yx), dtype, name, unit,
                          {"plate": p, "acquisition": 0, "well": w, "field": field},
                          entries=entries, names=names, system=well_system(plates, p, w))

    def _declare_coordinate_systems(self):
        """One system per well (plan 4.9): x / y relative to the well's centre."""
        return well_systems(self.plates)

    def close(self) -> None:
        super().close()
        self._members.close()


def open(path: str, tiles: str = "separate", resolutions: str = "base",  # noqa: A001 - backend protocol
         multifile: bool = True) -> _FlexFile:
    """The plate of the folder holding the ``.flex`` file *path* (*multifile*
    False: the one file); *tiles* and *resolutions* change nothing."""
    return _FlexFile(path, multifile)
