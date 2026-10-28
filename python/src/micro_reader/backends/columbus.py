"""PerkinElmer / Revvity Columbus exports (``ImageIndex.ColumbusIDX.xml``): docs/HCS.md.

Columbus (the image database of Harmony / Opera instruments) exports a
measurement as a folder of TIFFs with two index files:

- ``ImageIndex.ColumbusIDX.xml``: Harmony's index (``EvaluationInputData``,
  one ``<Image>`` per plane with its well, field, plane, time point and
  channel), in the ``http://www.perkinelmer.com/Columbus`` namespace.  A
  plane's ``URL`` carries a ``BufferNo``: its page in the TIFF, which may
  hold all planes of a well (6 in the OME sample) or one.  It is read as a
  Harmony index (backends/harmony.py: same indices, metadata, positions
  relative to the well's centre with the same ``OrientationMatrix``, one
  coordinate system per well); ``File.backend`` and ``format`` are
  ``"columbus"``.
- ``MeasurementIndex.ColumbusIDX.xml`` (``ColumbusMeasurementIndex``): the
  measurement, and where its image index is (``Reference Format="Native
  XML"``, relative to it; exports flattened by hand keep the image index
  beside it, which is used when the reference does not exist).  It is a part
  of the image index, as are ``ImageIndex.ColumbusIDX.csv`` (the same table
  as CSV) and the folder holding either: each opens the plate.
"""
from __future__ import annotations

import os
import xml.etree.ElementTree as ET

from .. import _parts
from .harmony import _HarmonyFile

NAME = "columbus"
#: the index file of a set: a folder holding it opens it (docs/COMPOSITE.md)
SET_INDEX = True

IMAGE_INDEX = "imageindex.columbusidx.xml"
MEASUREMENT_INDEX = "measurementindex.columbusidx.xml"
SUFFIXES = (".columbusidx.xml", ".columbusidx.csv")
NAMESPACE = b"http://www.perkinelmer.com/Columbus"


def may_index(name: str) -> bool:
    """``group_files``' cheap test (reads nothing): the image index's name."""
    return name.lower() == IMAGE_INDEX


def may_own(folder: str, name: str, folders) -> bool:
    """``group_files``' cheap test: Columbus's other index files may be parts."""
    return name.lower().endswith(SUFFIXES)


def sniff(head: bytes, path: str) -> bool:
    return b"EvaluationInputData" in head and NAMESPACE in head


def _named(folder: str, lower: str):
    """The file in *folder* whose name is *lower* in any letter case, or None."""
    try:
        with os.scandir(folder) as entries:
            return next((e.path for e in entries if e.is_file() and e.name.lower() == lower), None)
    except OSError:
        return None


def _referenced(measurement_index: str):
    """(the image index *measurement_index* names, as found; as expected): its
    ``Native XML`` reference, relative to it, else the image index beside it."""
    folder = os.path.dirname(os.path.abspath(measurement_index))
    expected = None
    try:
        root = ET.parse(measurement_index).getroot()
    except (ET.ParseError, OSError):
        root = None
    if root is not None:
        for ref in root.iter():
            if ref.tag.endswith("}Reference") and ref.get("Format") == "Native XML" and ref.text:
                expected = os.path.normpath(os.path.join(folder, ref.text.strip()))
                if os.path.isfile(expected):
                    return expected, expected
                break
    beside = _named(folder, IMAGE_INDEX)
    return beside, beside or expected or os.path.join(folder, "ImageIndex.ColumbusIDX.xml")


def owner(path: str, head):
    """``MeasurementIndex.ColumbusIDX.xml``, ``ImageIndex.ColumbusIDX.csv`` and
    a folder holding an index are parts of the image index (module docstring)."""
    if os.path.isdir(path):
        index = _named(path, IMAGE_INDEX)
        if index is not None:
            return _parts.Part(index, index)
        measurement = _named(path, MEASUREMENT_INDEX)
        if measurement is None:
            return None
        main, expected = _referenced(measurement)
        return _parts.Part(main, expected)
    name = os.path.basename(path).lower()
    if name == MEASUREMENT_INDEX:
        main, expected = _referenced(path)
        return _parts.Part(main, expected)
    if name.endswith(".columbusidx.csv"):
        index = _named(os.path.dirname(os.path.abspath(path)), IMAGE_INDEX)
        return _parts.Part(index, index or path[:-4] + ".xml")
    return None


def open(path: str, tiles: str = "separate", resolutions: str = "base") -> _HarmonyFile:  # noqa: A001 - backend protocol
    """The plate of a Columbus image index (*tiles* and *resolutions* change nothing)."""
    return _HarmonyFile(path, columbus=True)
