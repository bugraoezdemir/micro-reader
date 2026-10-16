"""Yokogawa CellVoyager plates (CV7000; CV8000 writes the same files):
docs/HCS.md.

A measurement is a folder of TIFFs, one per plane, described by XML files in
the BTS namespace (``http://www.yokogawa.co.jp/BTS/BTSSchema/1.0``).  Open
any of ``<plate>.wpi``, ``MeasurementDetail.mrf`` or ``MeasurementData.mlf``:
micro-reader reads the measurement in that folder.

- ``MeasurementDetail.mrf``: the plate (``RowCount`` x ``ColumnCount``,
  ``MeasurementSamplePlate`` name), per camera channel ``Ch`` its pixel size
  (um) and image size, and the settings file (``.mes``: channel names
  ``Target``, colours, light sources, emission filters ``BP<centre>/<width>``).
- ``MeasurementData.mlf``: one ``MeasurementRecord`` per plane: ``Type``
  ``IMG`` with 1-based ``Row`` / ``Column`` / ``TimePoint`` / ``FieldIndex``
  / ``ZIndex``, ``ActionIndex``, ``Ch``, stage ``X`` / ``Y`` / ``Z`` (um) and
  the file name; ``ERR`` records log instrument errors (one warning).
- One micro-reader image per field, axes ``t c z y x``.  A channel is an
  (action, camera channel) pair, as Bio-Formats counts them.  Indices:
  ``plate`` 0, ``acquisition`` 0, ``well`` (row then column), ``field`` =
  FieldIndex - 1.
- Position: as for Harmony (``harmony._HarmonyImage._field_position``),
  the first pixel's centre relative to the well centre, image coordinates:
  x0 = X - (W - 1) / 2 * px, y0 = -Y - (H - 1) / 2 * py, z = the first
  plane's Z.  The CV7000's own Harmony-format export (``Index.idx.xml``)
  stores the same X / Y / Z, so both routes give the same positions, up to
  the ``.mlf`` truncating X / Y to 0.1 um (checked on idr0088: < 0.1 um).
- Channels: name = ``Target``, excitation = the light source's wavelength,
  emission = the centre of the band-pass filter (Bio-Formats 8.0.0 reports
  that centre as the excitation).
- Actions with different z (a 3-D action next to 2-D ones): a RAGGED image
  (docs/AXES.md, ``_fileset.ragged_field``), placed by each record's ``Z``.

The measurement folder is also a composite (docs/COMPOSITE.md): its main
file is ``<plate>.wpi`` (else ``MeasurementDetail.mrf``, else
``MeasurementData.mlf``); the other entry files, the instrument's own
Harmony-format export (``Index.idx.xml``) and the settings files (``.wpp``,
``.mes``, ``.ppp``, ``.ppf``, ``.icr``, ``*_parameter.xml``) are its parts and
open it.  The plane TIFFs are the index's members (they open alone).

Missing files: as for every file set (``_fileset.keep_fields``).  Refused:
several timelines, channels imaged at different timepoints, channels with
different pixel sizes or image sizes.
"""
from __future__ import annotations

import os
import warnings
import xml.etree.ElementTree as ET
from datetime import datetime

from .. import _parts
from .._base import File
from .._errors import CorruptFileError, UnsupportedFormatError
from .._fileset import (Members, PlaneImage, check_times, keep_fields, ragged_field,
                        z_parts)
from .._metadata import Channel, Metadata, color_hex, quantity
from .._plate import Plate, Well, row_label

NAME = "cv7000"
#: the index file of a set: a folder holding it opens it (docs/COMPOSITE.md)
SET_INDEX = True
BTS = "{http://www.yokogawa.co.jp/BTS/BTSSchema/1.0}"
DETAIL, DATA = "MeasurementDetail.mrf", "MeasurementData.mlf"
#: the root elements of the files that open a measurement (not .wpp / .mes)
ENTRY_ROOTS = (b"bts:WellPlate ", b"bts:MeasurementDetail ", b"bts:MeasurementData ")


#: files of a measurement folder that are parts of its main file (lower case)
SIDE_EXTENSIONS = (".wpi", ".mrf", ".mlf", ".wpp", ".mes", ".ppp", ".ppf", ".icr", ".xml")


#: folder -> (its modification time, its main entry): a measurement folder
#: holds every plane, so it is listed once (again only when it changes)
_ENTRIES: dict = {}


def _main_entry(folder: str):
    """The measurement's main file: ``<plate>.wpi``, else ``.mrf``, else ``.mlf``."""
    try:
        stamp = os.stat(folder or ".").st_mtime_ns
    except OSError:
        return None
    cached = _ENTRIES.get(folder)
    if cached is not None and cached[0] == stamp:
        return cached[1]
    try:
        names = os.listdir(folder or ".")
    except OSError:
        return None
    wpi = sorted(e for e in names if e.lower().endswith(".wpi"))
    lower = {e.lower(): e for e in names}
    main = os.path.join(folder, wpi[0]) if len(wpi) == 1 else next(
        (os.path.join(folder, lower[n.lower()]) for n in (DETAIL, DATA) if n.lower() in lower), None)
    if len(_ENTRIES) > 256:
        _ENTRIES.clear()
    _ENTRIES[folder] = (stamp, main)
    return main


def may_own(folder: str, name: str, folders) -> bool:
    """``group_files``' cheap test (reads nothing but the folder listing
    *folders* caches): a settings or entry file next to a ``.mlf``."""
    return name.lower().endswith(SIDE_EXTENSIONS) and DATA.lower() in folders.names(folder)


def owner(path: str, head):
    """The entry, export and settings files of a measurement folder are
    parts of its main file (module docstring)."""
    if os.path.isdir(path) or not path.lower().endswith(SIDE_EXTENSIONS):
        return None
    folder = os.path.dirname(path)
    if not os.path.isfile(os.path.join(folder, DATA)):     # exact name: no folder listing
        return None
    main = _main_entry(folder)
    if main is None or os.path.normcase(os.path.abspath(main)) ==             os.path.normcase(os.path.abspath(path)):
        return None
    return _parts.Part(main, main)


def may_index(name: str) -> bool:
    """``group_files``' cheap test (reads nothing): a measurement is entered by its ``.wpi``, ``.mrf`` or ``.mlf``."""
    return name.lower().endswith(('.wpi', '.mrf', '.mlf'))


def sniff(head: bytes, path: str) -> bool:
    return b"yokogawa.co.jp/BTS" in head and any(r in head for r in ENTRY_ROOTS)


def _a(e, name):
    return e.get(BTS + name)


def _int(e, name):
    try:
        return int(_a(e, name))
    except (TypeError, ValueError):
        raise CorruptFileError(f"CV7000 record without a whole-number {name}: {e.attrib}")


def _float(e, name):
    try:
        return float(_a(e, name))
    except (TypeError, ValueError):
        return None


def _parse(path):
    try:
        return ET.parse(path).getroot()
    except ET.ParseError as exc:
        raise CorruptFileError(f"{path!r}: not readable XML ({exc})") from None
    except FileNotFoundError:
        raise CorruptFileError(f"{path!r} is missing: a CV7000 measurement needs it") from None


class _CVImage(PlaneImage):
    """A field (or a part of a ragged one); *channels*: its (action, Ch)
    keys, *records*: its planes' (t, key, z) -> (X, Y, Z, Time)."""

    def __init__(self, *args, info: dict, channels: list, dz, records: dict, times: list,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self._info, self._channels, self._dz = info, channels, dz
        self._records, self._times = records, times

    def _record(self, t: int, key, z=None):
        """The record of the first listed plane at timepoint *t* of channel *key*."""
        for (tt, kk, zz), rec in sorted(self._records.items()):
            if tt == t and kk == key and (z is None or zz == z):
                return rec
        return None

    def _read_metadata(self):
        sizes = dict(self._info["pixel_sizes"])
        if self._dz:
            sizes["z"] = quantity(self._dz, "micrometer")
        if len(self._times) > 1:
            a = self._record(self._times[0], self._channels[0])
            b = self._record(self._times[1], self._channels[0])
            if a and b and a[3] and b[3]:
                sizes["t"] = quantity((b[3] - a[3]).total_seconds(), "second")
        first = self._record(self._times[0], self._channels[0])
        position = {}
        if first:
            h, w = self.shape[-2:]
            for axis, value, sign, n in (("x", first[0], 1, w), ("y", first[1], -1, h)):
                px = sizes.get(axis)
                if value is not None and px is not None:
                    position[axis] = quantity(sign * value - (n - 1) / 2 * px.to("micrometer").value,
                                              "micrometer", positive=False)
            if first[2] is not None:
                position["z"] = quantity(first[2], "micrometer", positive=False)
        names = self._info["channels"]
        return Metadata(pixel_sizes={a: q for a, q in sizes.items() if q},
                        position={a: q for a, q in position.items() if q is not None},
                        channels=tuple(names.get(ch, Channel(name=f"Action {act}, Ch {ch}"))
                                       for act, ch in self._channels))


class _CVFile(File):
    backend = NAME
    format = "cv7000"

    def __init__(self, path: str):
        self.path = path
        self._members = Members()
        folder = os.path.dirname(os.path.abspath(path))
        detail = _parse(os.path.join(folder, DETAIL))
        data = _parse(os.path.join(folder, DATA))
        rows, columns = _int(detail, "RowCount"), _int(detail, "ColumnCount")
        sample = detail.find(BTS + "MeasurementSamplePlate")
        plate_name = (_a(sample, "Name") if sample is not None else None) or os.path.basename(folder)
        cameras = {_int(e, "Ch"): e for e in detail.findall(BTS + "MeasurementChannel")}
        settings = _a(detail, "MeasurementSettingFileName")
        channel_info = self._channels(os.path.join(folder, settings)) if settings else {}

        listed: dict = {}     # (row, col, field) -> {(t, (action, ch), z): path}
        records: dict = {}    # (row, col, field) -> {(t, (action, ch), z): (X, Y, Z, Time)}
        timelines, errors = set(), []
        for e in data.findall(BTS + "MeasurementRecord"):
            kind = _a(e, "Type")
            if kind == "ERR":
                errors.append((e.text or "").strip())
                continue
            if kind != "IMG":
                continue
            timelines.add(_a(e, "TimelineIndex"))
            key = (_int(e, "Row") - 1, _int(e, "Column") - 1, _int(e, "FieldIndex"))
            plane = (_int(e, "TimePoint"), (_int(e, "ActionIndex"), _int(e, "Ch")),
                     _int(e, "ZIndex"))
            name = (e.text or "").strip()
            listed.setdefault(key, {})[plane] = os.path.join(folder, name) if name else None
            records.setdefault(key, {})[plane] = (_float(e, "X"), _float(e, "Y"), _float(e, "Z"),
                                                  self._time(_a(e, "Time")))
        if not listed:
            raise CorruptFileError(f"{path!r}: the measurement lists no images")
        if len(timelines) > 1:
            raise UnsupportedFormatError(
                f"{path!r}: {len(timelines)} timelines in one measurement are not read yet")
        if errors:
            warnings.warn(f"{path!r}: the instrument logged {len(errors)} error"
                          f"{'' if len(errors) == 1 else 's'} (e.g. {errors[0]!r})", stacklevel=3)

        wanted = sorted({p for field in listed.values() for p in field})
        times = sorted({t for t, _k, _z in wanted})
        check_times(wanted, path)
        used = sorted({ch for _t, (_a_, ch), _z in wanted})
        sizes = {(_float(cameras[ch], "HorizontalPixelDimension"),
                  _float(cameras[ch], "VerticalPixelDimension"),
                  _a(cameras[ch], "HorizontalPixels"), _a(cameras[ch], "VerticalPixels"))
                 for ch in used if ch in cameras}
        if len(sizes) > 1:
            raise UnsupportedFormatError(
                f"{path!r}: channels with different pixel or image sizes are not read yet")
        kept = keep_fields(listed, wanted, path)
        parts = z_parts(wanted)
        z_at = {key: {p: rec[2] for p, rec in field.items()} for key, field in records.items()}

        first = next(iter(next(iter(kept.values())).values()))
        camera = cameras.get(used[0])
        shape_hint = None
        if camera is not None and _a(camera, "VerticalPixels") and _a(camera, "HorizontalPixels"):
            shape_hint = (_int(camera, "VerticalPixels"), _int(camera, "HorizontalPixels"))
        dtype, unit, shape_yx = self._members.plane_info(first, shape_hint)
        info = {"pixel_sizes": {} if camera is None else {
                    "x": quantity(_a(camera, "HorizontalPixelDimension"), "micrometer"),
                    "y": quantity(_a(camera, "VerticalPixelDimension"), "micrometer")},
                "channels": channel_info}

        wells_rc = sorted({(r, c) for r, c, _f in kept})
        number = {rc: n for n, rc in enumerate(wells_rc)}
        images = []
        for r, c, f in sorted(kept, key=lambda k: (number[k[:2]], k[2])):
            name = f"{plate_name} well {row_label(r)}{c + 1} field {f - 1}"
            recs = records[(r, c, f)]

            def make(planes, shape_tcz, chs, dz, index, readable, name=name, recs=recs):
                label = name if "part" not in index else f"{name} part {index['part']}"
                return _CVImage(self._members, len(images), planes,
                                shape_tcz + tuple(shape_yx), dtype, label, unit, index,
                                readable=readable, info=info, channels=chs, dz=dz,
                                records=recs, times=times)

            images.append(ragged_field(
                make, kept[(r, c, f)], z_at[(r, c, f)], times, parts,
                {"plate": 0, "acquisition": 0, "well": number[(r, c)], "field": f - 1}))
        plate = Plate(name=plate_name, rows=tuple(row_label(r) for r in range(rows)),
                      columns=tuple(str(c + 1) for c in range(columns)),
                      wells=tuple(Well(r, c, f"{row_label(r)}{c + 1}") for r, c in wells_rc))
        super().__init__(path, images, plates=[plate])

    @staticmethod
    def _time(value):
        try:
            return datetime.fromisoformat(value) if value else None
        except ValueError:
            return None

    @staticmethod
    def _channels(mes: str) -> dict:
        """Ch -> Channel from the settings file (.mes); {} when it is missing."""
        if not os.path.isfile(mes):
            return {}
        root = _parse(mes)
        lights = {e.get(BTS + "Name"): _float(e, "WaveLength")
                  for e in root.iter(BTS + "LightSource")}
        out = {}
        for e in root.iter(BTS + "Channel"):
            if _a(e, "Ch") is None:
                continue
            light = e.find(BTS + "LightSourceName")
            ex = lights.get(light.text.strip()) if light is not None and light.text else None
            em = None
            band = (_a(e, "Acquisition") or "").upper()
            if band.startswith("BP") and "/" in band:      # "BP445/45": centre / width
                em = quantity(band[2:].split("/", 1)[0], "nm")
            out[_int(e, "Ch")] = Channel(
                name=_a(e, "Target") or None, color=color_hex(_a(e, "Color")),
                excitation_wavelength=quantity(ex, "nm") if ex else None,
                emission_wavelength=em)
        return out

    def close(self) -> None:
        super().close()
        self._members.close()


def open(path: str, tiles: str = "separate", resolutions: str = "base") -> _CVFile:  # noqa: A001 - backend protocol
    """A CV7000 measurement from its ``.wpi``, ``.mrf`` or ``.mlf`` file
    (*tiles* and *resolutions* change nothing: fields are not tiled, and the
    TIFFs hold no pyramids)."""
    return _CVFile(path)
