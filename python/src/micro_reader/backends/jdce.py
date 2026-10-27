"""Molecular Devices ImageXpress HCS.ai exports (``.jdce``): docs/HCS.md.

MetaXpress Acquire (2025) writes each image set of an acquisition into its
own folder (``experiment`` for projections, ``experiment_z_stack``,
``experiment_montage`` for stitched montages), each with

- ``<name>.jdce``: JSON (``ImageStack``): the plate format (``Plate`` Rows /
  Columns), the objective calibration (pixel size), the wavelengths (filter
  names and wavelengths, z step), the CSV files listing the planes;
- ``image_metadata_<n>.csv``: one row per plane -- well (1-based ``Row`` /
  ``Column``), ``Field``, ``Wavelength``, ``Timepoint``, ``ZIndex`` (0-based),
  its file (``ImageSubFolderPath`` / ``ImageFileName``), stage position and
  the field's offset from the well's centre (``FieldOffsetPointX/Y``);
- the planes: one TIFF each, in ``timepoint<n>`` folders.

Open the ``.jdce`` file (or its folder, or a CSV): micro-reader reads the plate.

- One image per field, axes ``t c z y x``; indices ``plate`` 0,
  ``acquisition`` 0, ``well`` (wells with images, row then column),
  ``field``.  Missing planes as for every file set (``keep_fields``).
- Metadata: pixel sizes (``ObjectiveCalibration``; z the wavelength's
  ``ZStep``, else the planes' ``PositionZUm``; t the ``TimeStampSec`` step),
  channel names (the emission filter) and wavelengths, and the field's first
  pixel relative to its WELL'S CENTRE from ``FieldOffsetPointX/Y`` (the
  field's centre), y pointing down: checked on the montage sample, whose six
  overlapping tiles sit in MetaXpress' own stitched montage at these
  positions (correlation 0.91-1.00; with y up four of them do not).
  Coordinate systems: one per well (plan 4.9).
- The plate's name: the specimen holder's barcode, else the ``PlateId``.
"""
from __future__ import annotations

import builtins
import csv
import glob
import json
import os

from .. import _parts
from .._base import File
from .._errors import CorruptFileError, UnsupportedFormatError
from .._fileset import Members, PlaneImage, keep_fields
from .._metadata import Channel, Metadata, quantity
from .._plate import Plate, Well, layout, row_label, well_system, well_systems

NAME = "jdce"
#: the index file of a set: a folder holding it opens it (docs/COMPOSITE.md)
SET_INDEX = True
SUFFIX = ".jdce"


def may_index(name: str) -> bool:
    """``group_files``' cheap test (reads nothing): a ``.jdce`` file."""
    return name.lower().endswith(SUFFIX)


def may_own(folder: str, name: str, folders) -> bool:
    """``group_files``' cheap test: the plane list and plane statistics may be parts."""
    name = name.lower()
    return (name.startswith("image_metadata") and name.endswith(".csv")) or \
        name.endswith(".statistics.json")


def sniff(head: bytes, path: str) -> bool:
    return path.lower().endswith(SUFFIX) and b'"ImageStack"' in head


def _index_in(folder: str):
    found = sorted(glob.glob(os.path.join(glob.escape(folder), "*" + SUFFIX)))
    return found[0] if len(found) == 1 else None


def owner(path: str, head):
    """The CSV plane lists, the planes' ``.statistics.json`` files and the
    ``timepoint<n>`` folders are parts of the ``.jdce`` of their set."""
    path = os.path.abspath(path)
    name = os.path.basename(path).lower()
    folder = os.path.dirname(path)
    if os.path.isdir(path) and name.startswith("timepoint"):
        index = _index_in(folder)
    elif name.startswith("image_metadata") and name.endswith(".csv"):
        index = _index_in(folder)
    elif name.endswith(".statistics.json"):
        index = _index_in(os.path.dirname(folder))
    else:
        return None
    return _parts.Part(index, index) if index is not None else None


def _unit(text) -> str:
    """A unit as written (MetaXpress writes U+FFFD where 'µ' belongs)."""
    return str(text or "").replace("�", "u").replace("µ", "u")


def _float(row: dict, key: str):
    try:
        return float(row.get(key, ""))
    except ValueError:
        return None


class _JdceImage(PlaneImage):
    """One field; metadata from the index and its planes' CSV rows."""

    def __init__(self, *args, rows: dict, info: dict, system: str, **kwargs):
        super().__init__(*args, **kwargs)
        #: (t, c, z) -> its CSV row
        self._rows, self._info, self._system = rows, info, system

    def _read_metadata(self):
        info = self._info
        sizes = dict(info["pixel_sizes"])
        nt, nc, nz = self.shape[:3]
        first = self._rows.get((0, 0, 0)) or next(iter(self._rows.values()))
        if nz > 1:
            step = info["z_step"]
            if step is None:
                a, b = (self._rows.get((0, 0, z)) for z in (0, 1))
                if a and b and _float(a, "PositionZUm") is not None:
                    step = abs(_float(b, "PositionZUm") - _float(a, "PositionZUm"))
            if step:
                sizes["z"] = quantity(step, "um")
        if nt > 1:
            a, b = (self._rows.get((t, 0, 0)) for t in (0, 1))
            if a and b and _float(a, "TimeStampSec") is not None:
                sizes["t"] = quantity(_float(b, "TimeStampSec") - _float(a, "TimeStampSec"), "s")
        sizes = {a: q for a, q in sizes.items() if q}
        position = {}
        h, w = self.shape[3:5]
        for axis, key, n in (("x", "FieldOffsetPointX", w), ("y", "FieldOffsetPointY", h)):
            value, px = _float(first, key), sizes.get(axis)
            if value is not None and px is not None:
                position[axis] = quantity(value - (n - 1) / 2 * px.to("micrometer").value, "um",
                                          positive=False)
        z = _float(first, "PositionZUm")
        if z is not None and position:
            position["z"] = quantity(z, "um", positive=False)
        position = {a: q for a, q in position.items() if q is not None}
        return Metadata(pixel_sizes=sizes, position=position,
                        coordinate_system=self._system if position else None,
                        channels=tuple(info["channels"].get(c, Channel())
                                       for c in info["channel_ids"]))


class _JdceFile(File):
    backend = NAME
    format = "imagexpress-jdce"

    def __init__(self, path: str):
        folder = os.path.dirname(os.path.abspath(path))
        try:
            with builtins.open(path, "rb") as fh:
                stack = json.loads(fh.read().decode("utf-8-sig", "replace"))["ImageStack"]
        except (ValueError, KeyError) as exc:
            raise CorruptFileError(f"{path!r}: not an ImageXpress .jdce index ({exc})") from None
        protocol = stack.get("AutoLeadAcquisitionProtocol") or {}
        if stack.get("LargeImage"):
            raise UnsupportedFormatError(f"{path!r}: LargeImage sets are not read yet")
        rows = []
        for name in stack.get("ImageMetadataFiles") or sorted(
                os.path.basename(p) for p in glob.glob(os.path.join(glob.escape(folder),
                                                                  "image_metadata_*.csv"))):
            listed = os.path.join(folder, name)
            if not os.path.isfile(listed):
                raise CorruptFileError(f"{path!r}: its plane list {name!r} is missing")
            with builtins.open(listed, encoding="utf-8-sig", errors="replace", newline="") as fh:
                rows.extend(csv.DictReader(fh))
        if not rows:
            raise CorruptFileError(f"{path!r}: its plane lists name no planes")

        def whole(row, key):
            try:
                return int(row[key])
            except (KeyError, ValueError):
                raise CorruptFileError(f"{path!r}: a plane without a whole-number {key}") from None

        listed: dict = {}                      # (row, col, field) -> {(t, c, z): (row, file)}
        for r in rows:
            key = (whole(r, "Row") - 1, whole(r, "Column") - 1, whole(r, "Field"))
            plane = (whole(r, "Timepoint"), whole(r, "Wavelength"), whole(r, "ZIndex"))
            file = os.path.join(folder, r.get("ImageSubFolderPath") or "", r["ImageFileName"])
            if plane in listed.setdefault(key, {}):
                raise CorruptFileError(f"{path!r}: well row {key[0] + 1} column {key[1] + 1} "
                                       f"field {key[2]}: plane {plane} is listed twice")
            listed[key][plane] = (r, file)
        ts = sorted({p[0] for field in listed.values() for p in field})
        cs = sorted({p[1] for field in listed.values() for p in field})
        zs = sorted({p[2] for field in listed.values() for p in field})
        wanted = [(t, c, z) for t in ts for c in cs for z in zs]
        kept = keep_fields({k: {p: f for p, (_r, f) in v.items()} for k, v in listed.items()},
                           wanted, path)
        calibration = protocol.get("ObjectiveCalibration") or {}
        unit = _unit(calibration.get("Unit") or "um")
        channels, z_step = {}, None
        for wl in protocol.get("Wavelengths") or []:
            em, ex = wl.get("EmissionFilter") or {}, wl.get("ExcitationFilter") or {}
            channels[wl.get("Index")] = Channel(
                name=em.get("Name") or ex.get("Name") or None,
                emission_wavelength=quantity(em.get("Wavelength"), em.get("Unit") or "nm"),
                excitation_wavelength=quantity(ex.get("Wavelength"), ex.get("Unit") or "nm"))
            z_step = z_step or wl.get("ZStep")
        for c in cs:                               # a channel the protocol does not list
            if c not in channels:
                name = next((r.get("ExcitationEmissionFilter") for field in listed.values()
                             for (t, cc, z), (r, _f) in field.items() if cc == c), None)
                channels[c] = Channel(name=name or None)
        info = {"pixel_sizes": {a: quantity(calibration.get(k), unit) for a, k in
                                (("x", "PixelWidth"), ("y", "PixelHeight"))},
                "z_step": z_step, "channels": channels, "channel_ids": cs}
        plate_e = protocol.get("Plate") or {}
        holder = stack.get("SpecimenHolder") or {}
        name = holder.get("Barcode") or stack.get("PlateId") or os.path.basename(folder)
        wells = sorted({(r, c) for r, c, _f in kept})
        number = {rc: n for n, rc in enumerate(wells)}
        row_names, column_names = layout(int(plate_e.get("Rows") or 0) or max(r for r, _ in wells) + 1,
                                         int(plate_e.get("Columns") or 0) or max(c for _, c in wells) + 1,
                                         wells, path)
        plates = [Plate(name=name, row_names=row_names, column_names=column_names,
                        wells=tuple(Well(r, c, f"{row_label(r)}{c + 1}") for r, c in wells))]
        self._members = Members()
        position = {(t, c, z): (ts.index(t), cs.index(c), zs.index(z)) for t, c, z in wanted}
        images = []
        try:
            for (r, c, f) in sorted(kept, key=lambda k: (number[k[:2]], k[2])):
                planes = {position[p]: member for p, member in kept[(r, c, f)].items()}
                dtype, read_unit, shape_yx = self._members.plane_info(next(iter(planes.values())))
                images.append(_JdceImage(
                    self._members, len(images), planes, (len(ts), len(cs), len(zs)) + tuple(shape_yx),
                    dtype, f"{name} well {row_label(r)}{c + 1} field {f}", read_unit,
                    {"plate": 0, "acquisition": 0, "well": number[(r, c)], "field": f},
                    rows={position[p]: row for p, (row, _f) in listed[(r, c, f)].items()},
                    info=info, system=well_system(plates, 0, number[(r, c)])))
        except BaseException:
            self._members.close()
            raise
        super().__init__(path, images, plates=plates)

    def _declare_coordinate_systems(self):
        """One system per well (plan 4.9): x / y relative to the well's centre."""
        return well_systems(self.plates)

    def close(self) -> None:
        super().close()
        self._members.close()


def open(path: str, tiles: str = "separate", resolutions: str = "base") -> _JdceFile:  # noqa: A001 - backend protocol
    """The plate of an ImageXpress ``.jdce`` index (*tiles* and *resolutions*
    change nothing)."""
    return _JdceFile(path)
