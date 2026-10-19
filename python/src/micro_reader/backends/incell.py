"""GE / Cytiva IN Cell Analyzer plates (InCell 2000 / 6000 ...): docs/HCS.md.

A plate is a folder of TIFFs, one per plane, listed by an XML index file
``<name>.xdce`` (root ``ImageStack``).  Open the ``.xdce`` file.

- Each ``<Image filename=...>`` is one plane: its ``Well`` (1-based ``Row`` /
  ``Column`` numbers) and an ``Identifier`` with 0-based ``field_index``,
  ``wave_index`` (channel), ``z_index`` and ``time_index``.
- The plate: ``AutoLeadAcquisitionProtocol/Plate`` (rows, columns, name);
  channels: ``Wavelengths/Wavelength`` (name, excitation / emission filter
  wavelengths); pixel size: ``ObjectiveCalibration`` pixel_width / height.
- One micro-reader image per field, axes ``t c z y x``.  Indices: ``plate``
  0, ``acquisition`` 0, ``well`` (row then column), ``field`` = field_index.

**Channels with different z** (a 3-D channel next to 2-D ones, set per
wavelength): the field is a RAGGED image (docs/AXES.md,
``_fileset.ragged_field``), placed by each plane's ``FocusPosition``.
Bio-Formats 8.0.0 does not open such plates ("Invalid Z index").

Missing files: as for every file set (``_fileset.keep_fields``).  Not yet:
stage positions (``OffsetFromWellCenter``: no reference to check them
against), ragged timepoints.
"""
from __future__ import annotations

import os
import xml.etree.ElementTree as ET

from .._base import File
from .._errors import CorruptFileError
from .._fileset import (Members, PlaneImage, check_times, keep_fields, ragged_field,
                        z_parts)
from .._metadata import Channel, Metadata, quantity
from .._plate import Plate, Well, layout, row_label

NAME = "incell"
#: the index file of a set: a folder holding it opens it (docs/COMPOSITE.md)
SET_INDEX = True


def may_index(name: str) -> bool:
    """``group_files``' cheap test (reads nothing): the plate is an ``.xdce`` file."""
    return name.lower().endswith(('.xdce',))


def sniff(head: bytes, path: str) -> bool:
    return path.lower().endswith(".xdce") and b"<ImageStack" in head


def _int(value, what):
    try:
        return int(value)
    except (TypeError, ValueError):
        raise CorruptFileError(f"InCell index: {what} is not a whole number ({value!r})")


def _float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


class _InCellImage(PlaneImage):
    """A field (the whole) or one of its parts; *channels*: the channel
    numbers it holds, *dz*: its z step in micrometres (None: one plane)."""

    def __init__(self, *args, info: dict, channels: list, dz=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._info, self._channels, self._dz = info, channels, dz

    def _read_metadata(self):
        sizes = dict(self._info["pixel_sizes"])
        if self._dz:
            sizes["z"] = quantity(self._dz, "micrometer")
        names = self._info["channels"]
        return Metadata(pixel_sizes={a: q for a, q in sizes.items() if q},
                        channels=tuple(names.get(c, Channel()) for c in self._channels))


class _InCellFile(File):
    backend = NAME
    format = "incell"

    def __init__(self, path: str):
        self.path = path
        self._members = Members()
        try:
            root = ET.parse(path).getroot()
        except ET.ParseError as exc:
            raise CorruptFileError(f"{path!r}: not readable XML ({exc})") from None
        proto = root.find("AutoLeadAcquisitionProtocol")
        plate_e = proto.find("Plate") if proto is not None else None
        if plate_e is None:
            raise CorruptFileError(f"{path!r}: no plate (AutoLeadAcquisitionProtocol/Plate)")
        rows = _int(plate_e.get("rows"), "the plate's rows")
        columns = _int(plate_e.get("columns"), "the plate's columns")
        folder = os.path.dirname(os.path.abspath(path))
        info = {"pixel_sizes": self._pixel_sizes(proto), "channels": self._channels(proto)}

        listed: dict = {}     # (row, col, field) -> {(t, c, z): path}
        focus: dict = {}      # (row, col, field) -> {(t, c, z): focus z in um}
        for e in root.findall("Images/Image"):
            ident = e.find("Identifier")
            r = _int(e.find("Well/Row").get("number"), "a well row") - 1
            c = _int(e.find("Well/Column").get("number"), "a well column") - 1
            key = (r, c, _int(ident.get("field_index"), "field_index"))
            plane = (_int(ident.get("time_index"), "time_index"),
                     _int(ident.get("wave_index"), "wave_index"),
                     _int(ident.get("z_index"), "z_index"))
            name = e.get("filename")
            listed.setdefault(key, {})[plane] = os.path.join(folder, name) if name else None
            fp = e.find("FocusPosition")
            focus.setdefault(key, {})[plane] = _float(fp.get("z")) if fp is not None else None
        if not listed:
            raise CorruptFileError(f"{path!r}: the InCell index lists no images")

        wanted = sorted({p for field in listed.values() for p in field})
        times = sorted({t for t, _c, _z in wanted})
        check_times(wanted, path)
        kept = keep_fields(listed, wanted, path)
        parts = z_parts(wanted)

        wells_rc = sorted({(r, c) for r, c, _f in kept})
        number = {rc: n for n, rc in enumerate(wells_rc)}
        first = next(iter(next(iter(kept.values())).values()))
        dtype, unit, shape_yx = self._members.plane_info(first)
        # the Plate element's "name" is the plate TYPE ("Greiner 96 uClear"): the
        # file's own name names the plate
        base = os.path.splitext(os.path.basename(path))[0]
        images = []
        for r, c, f in sorted(kept, key=lambda k: (number[k[:2]], k[2])):
            name = f"{base} well {row_label(r)}{c + 1} field {f}"

            def make(planes, shape_tcz, chs, dz, index, readable, name=name):
                label = name if "part" not in index else f"{name} part {index['part']}"
                return _InCellImage(self._members, len(images), planes,
                                    shape_tcz + tuple(shape_yx), dtype, label, unit, index,
                                    readable=readable, info=info, channels=chs, dz=dz)

            images.append(ragged_field(
                make, kept[(r, c, f)], focus[(r, c, f)], times, parts,
                {"plate": 0, "acquisition": 0, "well": number[(r, c)], "field": f}))
        row_names, column_names = layout(rows, columns, wells_rc, path)
        plate = Plate(name=base, rows=row_names, columns=column_names,
                      wells=tuple(Well(r, c, f"{row_label(r)}{c + 1}") for r, c in wells_rc))
        super().__init__(path, images, plates=[plate])

    @staticmethod
    def _pixel_sizes(proto) -> dict:
        cal = proto.find("ObjectiveCalibration") if proto is not None else None
        if cal is None:
            return {}
        unit = cal.get("unit") or "um"
        return {a: quantity(cal.get(k), unit, default_unit="um")
                for a, k in (("x", "pixel_width"), ("y", "pixel_height"))}

    @staticmethod
    def _channels(proto) -> dict:
        out = {}
        waves = proto.find("Wavelengths") if proto is not None else None
        for w in (waves.findall("Wavelength") if waves is not None else []):
            ex, em = w.find("ExcitationFilter"), w.find("EmissionFilter")
            out[_int(w.get("index"), "a wavelength index")] = Channel(
                name=w.get("name") or (ex.get("name") if ex is not None else None),
                excitation_wavelength=(quantity(ex.get("wavelength"), ex.get("unit") or "nm")
                                       if ex is not None else None),
                emission_wavelength=(quantity(em.get("wavelength"), em.get("unit") or "nm")
                                     if em is not None else None))
        return out

    def close(self) -> None:
        super().close()
        self._members.close()


def open(path: str, tiles: str = "separate", resolutions: str = "base") -> _InCellFile:  # noqa: A001 - backend protocol
    """An InCell plate from its ``.xdce`` file (*tiles* and *resolutions*
    change nothing: fields are not tiled, and the TIFFs hold no pyramids)."""
    return _InCellFile(path)
