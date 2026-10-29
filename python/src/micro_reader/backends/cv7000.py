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
  the file name; ``ERR`` records log instrument errors (one warning).  A
  plane listed twice is read from the first record whose file exists
  (Bio-Formats' rule: a metadata-only record never shadows a real one).
- The files name their elements and attributes in the BTS namespace with
  any prefix (``bts:``, ``record:``, a default namespace) or none at all;
  they are matched by namespace URI or local name, never by prefix.  Numbers
  may use a decimal comma (``0,65``).
- One micro-reader image per field, axes ``t c z y x``.  A channel is an
  (action, camera channel) pair, as Bio-Formats counts them.  Indices:
  ``plate`` 0, ``acquisition`` 0, ``well`` (row then column), ``field`` =
  FieldIndex - 1.
- Position: as for Harmony (``harmony._HarmonyImage._field_position``),
  the first pixel's centre relative to the well centre, image coordinates:
  x0 = X - (W - 1) / 2 * px, y0 = -Y - (H - 1) / 2 * py, z = the Z of
  slice 0 (a ragged whole: its deepest channel's first plane).  The CV7000's own Harmony-format export (``Index.idx.xml``)
  stores the same X / Y / Z, so both routes give the same positions, up to
  the ``.mlf`` truncating X / Y to 0.1 um (checked on idr0088: < 0.1 um).
  Coordinate systems: one per well, as for Harmony (``well-B2``; plan 4.9).
- Channels: name = ``Target`` (when it is empty or the software's default
  "Microscope Image": the light source, e.g. "405nm", plus the filter when two
  channels share one), excitation = the light source's wavelength,
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

Missing files: as for every file set (``_fileset.keep_fields``).  The
timelines of the settings file (``TimelineIndex``: each images its own wells,
fields and actions, its TimePoint numbers running on from the timeline
before) are merged, as Bio-Formats merges them (checked against its expected
results for the 14 synthetic packages of Zenodo 21334895): each timeline's
timepoints are counted from its own first one, a field holds every plane
recorded for it, a well imaged by two timelines holds the channels of both,
and the plate's channels are the (action, camera channel) pairs of all its
records; a plane a second timeline images with the channel and timepoint of
the first becomes a channel of its own, named after its timeline (Bio-Formats'
"timeline-qualified overflow").  Refused: channels imaged at different
timepoints, channels with different pixel sizes or image sizes.
"""
from __future__ import annotations

import os
import re
import warnings
import xml.etree.ElementTree as ET
from dataclasses import replace
from datetime import datetime

from .. import _parts
from .._base import File
from .._errors import CorruptFileError, UnsupportedFormatError
from .._fileset import (Members, PlaneImage, check_times, keep_fields, ragged_field,
                        z_parts)
from .._metadata import Channel, Metadata, color_hex, quantity
from .._plate import Plate, Well, layout, row_label, well_system, well_systems

NAME = "cv7000"
#: the index file of a set: a folder holding it opens it (docs/COMPOSITE.md)
SET_INDEX = True
BTS = "{http://www.yokogawa.co.jp/BTS/BTSSchema/1.0}"
DETAIL, DATA = "MeasurementDetail.mrf", "MeasurementData.mlf"
#: the root element of a file that opens a measurement (not .wpp / .mes),
#: with any namespace prefix or none (``<bts:WellPlate ``, ``<plate:WellPlate ``,
#: ``<WellPlate>``; not ``<bts:WellPlateProduct``)
ENTRY_ROOT = re.compile(rb"<(?:[A-Za-z_][\w.-]*:)?(?:WellPlate|MeasurementDetail|MeasurementData)[\s/>]")


#: the CellVoyager software's default channel ``Target``: it names no channel
DEFAULT_TARGET = "Microscope Image"
#: correction files in a measurement folder: shading, dark current, OTF (upper case)
CORRECTION_PREFIXES = ("SC_", "DC_", "OTF_")
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
    return b"yokogawa.co.jp/BTS" in head and ENTRY_ROOT.search(head) is not None


def _a(e, name):
    """Attribute *name* of *e*: in the BTS namespace (any prefix), else
    unprefixed (no namespace, also under a default namespace)."""
    value = e.get(BTS + name)
    return e.get(name) if value is None else value


def _number(value):
    """A number's text with a decimal comma (``0,65``) made a point."""
    if isinstance(value, str) and value.count(",") == 1 and "." not in value:
        return value.replace(",", ".")
    return value


def _int(e, name):
    try:
        return int(_a(e, name))
    except (TypeError, ValueError):
        raise CorruptFileError(f"CV7000 record without a whole-number {name}: {e.attrib}")


def _float(e, name):
    try:
        return float(_number(_a(e, name)))
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
    """A field (or a part of a ragged one); *channels*: its (action, Ch, run)
    keys, *records*: its planes' (t, key, z) -> (X, Y, Z, Time)."""

    def __init__(self, *args, info: dict, channels: list, dz, records: dict, times: list,
                 system: str, **kwargs):
        super().__init__(*args, **kwargs)
        self._info, self._channels, self._dz = info, channels, dz
        self._records, self._times = records, times
        #: the id of the coordinate system the position is in: its well's (4.9)
        self._system = system

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
            # z: slice 0, the first plane of the channel with the most planes
            # (a ragged whole's grid is its deepest part's; JUMP's 3-D brightfield
            # next to 2-D fluorescence)
            planes = {}
            for tt, kk, _zz in self._records:
                if tt == self._times[0]:
                    planes[kk] = planes.get(kk, 0) + 1
            deepest = max(self._channels, key=lambda k: planes.get(k, 0))
            origin = self._record(self._times[0], deepest)
            if origin and origin[2] is not None:
                position["z"] = quantity(origin[2], "micrometer", positive=False)
        names = self._info["channels"]
        position = {a: q for a, q in position.items() if q is not None}
        return Metadata(pixel_sizes={a: q for a, q in sizes.items() if q}, position=position,
                        coordinate_system=self._system if position else None,
                        channels=tuple(self._channel(names, act, ch, run)
                                       for act, ch, run in self._channels))

    @staticmethod
    def _channel(names: dict, act: int, ch: int, run: int) -> Channel:
        """The channel's description: the settings file's for camera channel *ch*,
        else a name from the action and channel; a channel a second timeline
        added (*run*: its TimelineIndex, 0 for the first) says so."""
        channel = names.get(ch, Channel(name=f"Action {act}, Ch {ch}"))
        if run:
            channel = replace(channel, name=f"{channel.name or f'Action {act}, Ch {ch}'} "
                                            f"(timeline {run})")
        return channel


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
        sample = detail.find("{*}MeasurementSamplePlate")
        plate_name = (_a(sample, "Name") if sample is not None else None) or os.path.basename(folder)
        cameras = {_int(e, "Ch"): e for e in detail.findall("{*}MeasurementChannel")}
        settings = _a(detail, "MeasurementSettingFileName")
        channel_info = self._channels(os.path.join(folder, settings)) if settings else {}

        # every IMG record of the measurement; the timelines of the settings file
        # (TimelineIndex) are merged as Bio-Formats merges them: each timeline's
        # timepoints counted from its own first one (TimePoint numbers run on from
        # one timeline to the next), then the planes of a field put together
        # (module docstring)
        found: list = []      # (timeline, (row, col, field), (TimePoint, (action, ch), z), path, rec)
        errors = []
        for e in data.findall("{*}MeasurementRecord"):
            kind = _a(e, "Type")
            if kind == "ERR":
                errors.append((e.text or "").strip())
                continue
            if kind != "IMG":
                continue
            timeline = _int(e, "TimelineIndex") if _a(e, "TimelineIndex") is not None else 1
            key = (_int(e, "Row") - 1, _int(e, "Column") - 1, _int(e, "FieldIndex"))
            plane = (_int(e, "TimePoint"), (_int(e, "ActionIndex"), _int(e, "Ch")),
                     _int(e, "ZIndex"))
            name = (e.text or "").strip()
            found.append((timeline, key, plane, os.path.join(folder, name) if name else None,
                          (_float(e, "X"), _float(e, "Y"), _float(e, "Z"),
                           self._time(_a(e, "Time")))))
        ranks = {}            # timeline -> {TimePoint: its rank in the timeline}
        for timeline, _key, (tp, _ch, _z), _p, _rec in found:
            ranks.setdefault(timeline, set()).add(tp)
        ranks = {line: {tp: k for k, tp in enumerate(sorted(tps))} for line, tps in ranks.items()}
        listed: dict = {}     # (row, col, field) -> {(t, (action, ch), z): path}
        records: dict = {}    # (row, col, field) -> {(t, (action, ch), z): (X, Y, Z, Time)}
        owner: dict = {}      # (row, col, field), plane -> the timeline it came from
        for timeline, key, (tp, (action, ch), z), file, rec in found:
            # a channel is (action, camera channel, 0); a plane another timeline
            # imaged already with the same channel and timepoint becomes a channel
            # of its own, qualified by its timeline (Bio-Formats' "overflow")
            plane = (ranks[timeline][tp], (action, ch, 0), z)
            field = listed.setdefault(key, {})
            if plane in field and owner[(key, plane)] != timeline and field[plane] != file:
                plane = (plane[0], (action, ch, timeline), z)
            if plane in field:
                # a plane listed again: the first record whose file exists is kept
                # (a metadata-only record never shadows a real one, module docstring)
                if field[plane] and os.path.isfile(field[plane]):
                    continue
            field[plane] = file
            owner[(key, plane)] = timeline
            records.setdefault(key, {})[plane] = rec
        if not listed:
            raise CorruptFileError(f"{path!r}: the measurement lists no images")
        if errors:
            warnings.warn(f"{path!r}: the instrument logged {len(errors)} error"
                          f"{'' if len(errors) == 1 else 's'} (e.g. {errors[0]!r})", stacklevel=3)

        wanted = sorted({p for field in listed.values() for p in field})
        times = sorted({t for t, _k, _z in wanted})
        check_times(wanted, path)
        used = sorted({ch for _t, (_a_, ch, _run), _z in wanted})
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
                    "x": quantity(_float(camera, "HorizontalPixelDimension"), "micrometer"),
                    "y": quantity(_float(camera, "VerticalPixelDimension"), "micrometer")},
                "channels": channel_info}

        wells_rc = sorted({(r, c) for r, c, _f in kept})
        number = {rc: n for n, rc in enumerate(wells_rc)}
        row_names, column_names = layout(rows, columns, wells_rc, path)
        plates = [Plate(name=plate_name, row_names=row_names, column_names=column_names,
                        wells=tuple(Well(r, c, f"{row_label(r)}{c + 1}") for r, c in wells_rc))]
        images = []
        for r, c, f in sorted(kept, key=lambda k: (number[k[:2]], k[2])):
            name = f"{plate_name} well {row_label(r)}{c + 1} field {f - 1}"
            recs = records[(r, c, f)]
            system = well_system(plates, 0, number[(r, c)])

            def make(planes, shape_tcz, chs, dz, index, readable, name=name, recs=recs,
                     system=system):
                label = name if "part" not in index else f"{name} part {index['part']}"
                return _CVImage(self._members, len(images), planes,
                                shape_tcz + tuple(shape_yx), dtype, label, unit, index,
                                readable=readable, info=info, channels=chs, dz=dz,
                                records=recs, times=times, system=system)

            images.append(ragged_field(
                make, kept[(r, c, f)], z_at[(r, c, f)], times, parts,
                {"plate": 0, "acquisition": 0, "well": number[(r, c)], "field": f - 1}))
        super().__init__(path, images, plates=plates)
        self._folder = folder

    def _declare_coordinate_systems(self):
        """One system per well (plan 4.9): x / y relative to the well's
        centre, z as the instrument gives it, so wells share no frame."""
        return well_systems(self.plates)

    def _side_files(self) -> set:
        """The correction files the instrument writes into the measurement folder
        (``SC_`` shading, ``DC_`` dark current, ``OTF_`` optical transfer images and
        parameters): side files of the plate for ``group_files``, images on their
        own when opened (decided 2026-10-18)."""
        try:
            names = os.listdir(self._folder)
        except OSError:
            return set()
        return {os.path.join(self._folder, n) for n in names if n.upper().startswith(CORRECTION_PREFIXES)}

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
        lights = {_a(e, "Name"): _float(e, "WaveLength")
                  for e in root.iterfind(".//{*}LightSource")}
        out, fallback = {}, {}
        for e in root.iterfind(".//{*}Channel"):
            if _a(e, "Ch") is None:
                continue
            light = e.find("{*}LightSourceName")
            light = light.text.strip() if light is not None and light.text else None
            ex = lights.get(light) if light else None
            em = None
            band = (_a(e, "Acquisition") or "").upper()
            if band.startswith("BP") and "/" in band:      # "BP445/45": centre / width
                em = quantity(band[2:].split("/", 1)[0], "nm")
            ch = _int(e, "Ch")
            name = (_a(e, "Target") or "").strip()
            if name in ("", DEFAULT_TARGET):
                # the software's default target names nothing: the light source does
                # (decided 2026-10-18; BIA S-BIAD1013 / S-BIAD3934 name every channel so)
                fallback[ch] = (light, _a(e, "Acquisition"))
                name = light or ""
            out[ch] = Channel(
                name=name or None, color=color_hex(_a(e, "Color")),
                excitation_wavelength=quantity(ex, "nm") if ex else None,
                emission_wavelength=em)
        # two channels on one light source (one laser, two filters): add the filter
        lights_used = [light for light, _ in fallback.values() if light]
        for ch, (light, band) in fallback.items():
            if light and band and lights_used.count(light) > 1:
                out[ch] = replace(out[ch], name=f"{light} {band}")
        return out

    def close(self) -> None:
        super().close()
        self._members.close()


def open(path: str, tiles: str = "separate", resolutions: str = "base") -> _CVFile:  # noqa: A001 - backend protocol
    """A CV7000 measurement from its ``.wpi``, ``.mrf`` or ``.mlf`` file
    (*tiles* and *resolutions* change nothing: fields are not tiled, and the
    TIFFs hold no pyramids)."""
    return _CVFile(path)
