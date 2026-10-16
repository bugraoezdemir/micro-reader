"""PerkinElmer / Revvity Harmony plates (Operetta, Opera Phenix; also exports
of other instruments in the same format, e.g. Yokogawa CV7000): docs/HCS.md.

A plate is a folder of TIFFs, one per plane, listed by an index file
(``Images/Index.idx.xml``, namespace ``.../PEHH/HarmonyV4`` or ``V5``).  Open
the index file: micro-reader reads the plate it describes.

- Each ``<Image>`` is one plane: its file (``URL``, relative to the index
  file), its well (1-based ``Row`` / ``Col``), ``FieldID``, ``PlaneID``,
  ``TimepointID`` and ``ChannelID``.
- One micro-reader image per field, with axes ``t c z y x``: the timepoints,
  channels and planes of the plate, in the order of their IDs (IDs start at
  0 or 1 depending on the instrument).
- Indices: ``plate`` 0, ``acquisition`` 0 (one measurement per index file),
  ``well`` (the wells with images, row then column), ``field`` = FieldID - 1.
- Missing planes (empty ``URL`` or no file; vendors list planes they did not
  write): a field missing some reads them as zeros, a field missing all is
  left out, with one ``MissingImagesWarning`` (``_fileset.keep_fields``).
- Metadata: pixel sizes (``ImageResolutionX/Y``; z from the planes'
  ``PositionZ``, t from ``MeasurementTimeOffset``), channel names and
  main excitation / emission wavelengths, and the position of the field's
  first pixel RELATIVE TO ITS WELL'S CENTRE (``_HarmonyImage._field_position``).

The measurement folder is also a composite (docs/COMPOSITE.md): next to
``Images/`` it holds ``Assaylayout/`` (the plate map) and ``FFC_Profile/`` (the
flat-field model); their files, those two folders and the measurement folder
itself are parts of ``Images/Index.idx.xml`` and open it.
"""
from __future__ import annotations

import os
import xml.etree.ElementTree as ET

from .. import _parts
from .._base import File
from .._errors import CorruptFileError, UnsupportedFormatError
from .._fileset import Members, PlaneImage, keep_fields
from .._metadata import Channel, Metadata, quantity
from .._plate import Plate, Well, row_label

NAME = "harmony"
#: the index file of a set: a folder holding it opens it (docs/COMPOSITE.md)
SET_INDEX = True


#: where a measurement folder keeps its index, and its side folders (lower case)
INDEX_NAMES = (os.path.join("Images", "Index.idx.xml"), os.path.join("Images", "Index.xml"))
SIDE_FOLDERS = ("assaylayout", "ffc_profile")


def _measurement_index(measurement: str):
    """The index file of the measurement folder *measurement*, or None
    (exact names: no folder listing)."""
    for name in INDEX_NAMES:
        path = os.path.join(measurement, name)
        if os.path.isfile(path):
            return path
    return None


def may_own(folder: str, name: str, folders) -> bool:
    """``group_files``' cheap test (reads nothing): a file in an
    ``Assaylayout`` or ``FFC_Profile`` folder may be a part."""
    return folders.base(folder).lower() in SIDE_FOLDERS


def owner(path: str, head):
    """The side folders of a measurement, their files and the measurement
    folder itself are parts of its index (module docstring)."""
    path = os.path.abspath(path)
    if os.path.isdir(path):
        if os.path.basename(path).lower() in SIDE_FOLDERS:
            path = os.path.dirname(path)
        index = _measurement_index(path)
    elif os.path.basename(os.path.dirname(path)).lower() in SIDE_FOLDERS:
        index = _measurement_index(os.path.dirname(os.path.dirname(path)))
    else:
        return None
    return _parts.Part(index, index) if index is not None else None


def may_index(name: str) -> bool:
    """``group_files``' cheap test (reads nothing): an index is an XML file (``Index.idx.xml``, ``Index.xml``)."""
    return name.lower().endswith(('.xml',))


def sniff(head: bytes, path: str) -> bool:
    return b"EvaluationInputData" in head and b"PEHH/Harmony" in head


def _text(e, name):
    child = e.find(f"{{*}}{name}")
    return None if child is None or child.text is None else child.text.strip()


def _int(e, name):
    value = _text(e, name)
    try:
        return int(value)
    except (TypeError, ValueError):
        raise CorruptFileError(f"Harmony index: an <Image> without a whole-number <{name}>")


def _value(e, name, positive=True):
    """A length / time / wavelength the index gives with its ``Unit``."""
    child = None if e is None else e.find(f"{{*}}{name}")
    if child is None:
        return None
    return quantity(child.text, child.get("Unit"), positive=positive)


class _HarmonyImage(PlaneImage):
    """One field; metadata from its planes' ``<Image>`` entries."""

    def __init__(self, *args, entries: dict, **kwargs):
        super().__init__(*args, **kwargs)
        #: (t, c, z) position -> its <Image> element (also for missing planes)
        self._entries = entries

    def _entry(self, t=None, c=None, z=None):
        """A listed plane's <Image> with these positions (None: any), or None."""
        for (tt, cc, zz), e in sorted(self._entries.items()):
            if (t is None or tt == t) and (c is None or cc == c) and (z is None or zz == z):
                return e
        return None

    def _step(self, name, a, b, positive):
        qa, qb = _value(self._entry(*a), name, False), _value(self._entry(*b), name, False)
        if qa and qb and qa.unit == qb.unit:
            d = qb.value - qa.value
            return quantity(abs(d) if positive else d, qa.unit)
        return None

    def _read_metadata(self):
        first = self._entry()
        sizes = {axis: _value(first, name) for axis, name in
                 (("x", "ImageResolutionX"), ("y", "ImageResolutionY"))}
        nt, nc, nz = self.shape[:3]
        if nz > 1:
            sizes["z"] = self._step("PositionZ", (0, 0, 0), (0, 0, 1), True)
        if nt > 1:
            sizes["t"] = self._step("MeasurementTimeOffset", (0, 0, 0), (1, 0, 0), False)
        channels = []
        for c in range(nc):
            e = self._entry(c=c)
            channels.append(Channel() if e is None else Channel(
                name=_text(e, "ChannelName") or None,
                emission_wavelength=_value(e, "MainEmissionWavelength"),
                excitation_wavelength=_value(e, "MainExcitationWavelength")))
        return Metadata(pixel_sizes={a: q for a, q in sizes.items() if q},
                        position=self._field_position(first, sizes), channels=tuple(channels))

    def _field_position(self, e, sizes) -> dict:
        """The centre of the field's first pixel, relative to the WELL centre
        (Harmony gives no stage coordinates): ``PositionX`` / ``PositionY`` are
        the field's centre with y up (the index's ``OrientationMatrix``
        flips y); Bio-Formats reports x, -y.  So, in image coordinates (y
        down): x0 = X - (W - 1) / 2 * px, y0 = -Y - (H - 1) / 2 * py; z is
        the first plane's ``PositionZ``.  Checked against Bio-Formats 8.0.0
        on idr0034 (docs/HCS.md)."""
        out = {}
        h, w = self.shape[-2:]
        for axis, name, sign, n in (("x", "PositionX", 1, w), ("y", "PositionY", -1, h)):
            q, px = _value(e, name, positive=False), sizes.get(axis)
            if q is None or px is None:
                continue
            px = px.to(q.unit)
            out[axis] = quantity(sign * q.value - (n - 1) / 2 * px.value, q.unit, positive=False)
        plane0 = self._entry(t=0, c=0, z=0)
        z = _value(e if plane0 is None else plane0, "PositionZ", positive=False)
        if z is not None:
            out["z"] = z
        return {a: q for a, q in out.items() if q is not None}


class _HarmonyFile(File):
    backend = NAME
    format = "harmony"

    def __init__(self, path: str):
        self.path = path
        self._members = Members()
        try:
            root = ET.parse(path).getroot()
        except ET.ParseError as exc:
            raise CorruptFileError(f"{path!r}: not readable XML ({exc})") from None
        version = root.tag.split("}", 1)[0].rsplit("/", 1)[-1]           # HarmonyV4 / V5
        self.format = "harmony-" + version.replace("Harmony", "").lower()  # harmony-v4
        plates = root.findall("{*}Plates/{*}Plate")
        if len(plates) != 1:
            raise UnsupportedFormatError(
                f"{path!r}: {len(plates)} plates in one Harmony index; micro-reader reads "
                "one plate per index file for now")
        plate_e = plates[0]
        rows, columns = _int(plate_e, "PlateRows"), _int(plate_e, "PlateColumns")
        folder = os.path.dirname(os.path.abspath(path))

        listed: dict = {}                       # (row, col, field) -> {(tp, ch, pl): element}
        flims = set()
        for e in root.findall("{*}Images/{*}Image"):
            key = (_int(e, "Row") - 1, _int(e, "Col") - 1, _int(e, "FieldID"))
            plane = (_int(e, "TimepointID"), _int(e, "ChannelID"), _int(e, "PlaneID"))
            listed.setdefault(key, {})[plane] = e
            flims.add(_text(e, "FlimID"))
        if len(flims) > 1:
            raise UnsupportedFormatError(f"{path!r}: FLIM plates (several FlimID) are not read yet")
        if not listed:
            raise CorruptFileError(f"{path!r}: the Harmony index lists no images")
        tps = sorted({p[0] for field in listed.values() for p in field})
        chs = sorted({p[1] for field in listed.values() for p in field})
        pls = sorted({p[2] for field in listed.values() for p in field})
        wanted = [(t, c, z) for t in tps for c in chs for z in pls]

        def url(e):
            u = _text(e, "URL")
            return os.path.join(folder, u) if u else None

        kept = keep_fields({key: {plane: url(e) for plane, e in field.items()}
                            for key, field in listed.items()}, wanted, path)

        wells_rc = sorted({(r, c) for r, c, _f in kept})
        number = {rc: n for n, rc in enumerate(wells_rc)}
        first_key = next(iter(kept))
        first_e = listed[first_key][next(iter(kept[first_key]))]
        shape_yx = (_int(first_e, "ImageSizeY"), _int(first_e, "ImageSizeX"))
        dtype, unit, _ = self._members.plane_info(next(iter(kept[first_key].values())), shape_yx)
        shape = (len(tps), len(chs), len(pls)) + shape_yx
        position = {(tp, ch, pl): (tps.index(tp), chs.index(ch), pls.index(pl))
                    for tp, ch, pl in wanted}
        plate_name = _text(plate_e, "Name") or _text(plate_e, "PlateID") or ""
        base = plate_name or os.path.basename(os.path.dirname(folder)) or os.path.basename(path)
        images = []
        for (r, c, f) in sorted(kept, key=lambda k: (number[k[:2]], k[2])):
            images.append(_HarmonyImage(
                self._members, len(images),
                {position[p]: member for p, member in kept[(r, c, f)].items()}, shape, dtype,
                f"{base} well {row_label(r)}{c + 1} field {f - 1}", unit,
                {"plate": 0, "acquisition": 0, "well": number[(r, c)], "field": f - 1},
                # every listed plane's <Image> (metadata), whether its file is there or not
                entries={position[p]: e for p, e in listed[(r, c, f)].items() if p in position}))
        plate = Plate(name=plate_name,
                      rows=tuple(row_label(r) for r in range(rows)),
                      columns=tuple(str(c + 1) for c in range(columns)),
                      wells=tuple(Well(r, c, f"{row_label(r)}{c + 1}") for r, c in wells_rc))
        super().__init__(path, images, plates=[plate])

    def close(self) -> None:
        super().close()
        self._members.close()


def open(path: str, tiles: str = "separate", resolutions: str = "base") -> _HarmonyFile:  # noqa: A001 - backend protocol
    """A Harmony plate from its index file (fields are not tiled, and Harmony
    stores no pyramids: *tiles* and *resolutions* change nothing)."""
    return _HarmonyFile(path)
