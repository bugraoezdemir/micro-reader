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
- Not read yet: a plate folder without ``data/metadata.ome.xml`` (older
  scanR versions; the structure would come from the file names alone).
"""
from __future__ import annotations

import os
import re
import xml.etree.ElementTree as ET

from .. import _parts
from .._errors import CorruptFileError, UnsupportedFormatError
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


def _settings(path: str) -> dict:
    """{name: value} of the descriptor's named values (the first of each name)."""
    try:
        root = ET.parse(path).getroot()
    except ET.ParseError as exc:
        raise CorruptFileError(f"{path!r}: not readable XML ({exc})") from None
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

    def __init__(self, path: str):
        companion = os.path.join(os.path.dirname(os.path.abspath(path)), "data", COMPANION)
        found = _named(os.path.dirname(companion), COMPANION)
        if found is None:
            raise UnsupportedFormatError(
                f"{path!r}: a scanR plate without data/metadata.ome.xml (older scanR "
                "versions) is not read yet")
        super().__init__(path, plates=True, companion=found, plan=_plan(path, load(found)))


def open(path: str, tiles: str = "separate", resolutions: str = "base") -> _ScanrFile:  # noqa: A001 - backend protocol
    """The plate of a scanR descriptor (*tiles* and *resolutions* change nothing)."""
    return _ScanrFile(path)
