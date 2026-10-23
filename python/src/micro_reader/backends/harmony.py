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
- The plate's name: the index's ``Name``, else its ``PlateID``, else the
  measurement folder's name (the folder above ``Images``; an index whose
  ``Name`` and ``PlateID`` are empty is common), as outputs are named.
- Missing planes (empty ``URL`` or no file; vendors list planes they did not
  write): a field missing some reads them as zeros, a field missing all is
  left out, with one ``MissingImagesWarning`` (``_fileset.keep_fields``).
- Metadata: pixel sizes (``ImageResolutionX/Y``; z from the planes'
  ``PositionZ``, t from ``MeasurementTimeOffset``), channel names and
  main excitation / emission wavelengths, and the position of the field's
  first pixel RELATIVE TO ITS WELL'S CENTRE (``_HarmonyImage._field_position``).
- Coordinate systems: one per well (``well-B2``; plan 4.9), since positions
  are relative to the well's centre: ``metadata.coordinate_system`` names
  the field's well.

The measurement folder is also a composite (docs/COMPOSITE.md): next to
``Images/`` it holds ``Assaylayout/`` (the plate map), ``FFC_Profile/`` (the
flat-field model) and, once Harmony's analysis has run, ``Evaluation1/``,
``Evaluation2/``, ... (its results: ``PlateResults.xml``, ``ObjectList_*.xml``,
the ``.aas`` analysis script); their files, those folders, the other files in
the measurement folder and the measurement folder itself are parts of
``Images/Index.idx.xml`` and open it.
"""
from __future__ import annotations

import os
import re
import xml.etree.ElementTree as ET

from .. import _parts
from .._base import File
from .._errors import CorruptFileError, UnsupportedFormatError
from .._fileset import Members, PlaneImage, keep_fields
from .._metadata import Channel, Metadata, quantity
from .._plate import Plate, Well, layout, row_label, well_system, well_systems

NAME = "harmony"
#: the index file of a set: a folder holding it opens it (docs/COMPOSITE.md)
SET_INDEX = True


#: where a measurement folder keeps its index (letter case aside: the "hs"
#: archive of Harmony 5.2 / 6 writes ``images/index.xml``), and its side
#: folders (lower case): classic ``Assaylayout`` / ``FFC_Profile``, hs
#: ``flatfieldcorrection`` / ``index`` (measurement descriptions), and the
#: analysis results ``Evaluation<n>`` (``SIDE_FOLDER_PATTERN``)
IMAGES_FOLDERS = ("Images", "images")
INDEX_FILES = ("Index.idx.xml", "Index.xml", "index.xml", "index.idx.xml")
SIDE_FOLDERS = ("assaylayout", "ffc_profile", "flatfieldcorrection", "index")
SIDE_FOLDER_PATTERN = re.compile(r"evaluation\d+")
#: plane files: members of the index, opened as themselves (never parts)
PLANE_SUFFIXES = (".tif", ".tiff")


def _side_folder(name: str) -> bool:
    """Is a folder called *name* (any letter case) a measurement's side folder?"""
    name = name.lower()
    return name in SIDE_FOLDERS or SIDE_FOLDER_PATTERN.fullmatch(name) is not None


def _measurement_index(measurement: str):
    """The index file of the measurement folder *measurement*, or None (the
    spellings above only: ``owner`` runs on every open, so no folder listing)."""
    for folder in IMAGES_FOLDERS:
        images = os.path.join(measurement, folder)
        if os.path.isdir(images):
            for name in INDEX_FILES:
                path = os.path.join(images, name)
                if os.path.isfile(path):
                    return _spelled(path)
    return None


def _measurement_name(images: str) -> str:
    """The name of the measurement folder of the index in folder *images*: the
    folder above ``Images``, or *images* itself when the index sits among its
    planes (zenodo-15149351); ``""`` at a drive root."""
    folder = os.path.abspath(images)
    if os.path.basename(folder).lower() == "images":
        folder = os.path.dirname(folder)
    return os.path.basename(folder)


def _spelled(path: str) -> str:
    """*path* with its folder's and file's names as stored (a case-insensitive
    file system finds ``Index.xml`` for ``index.xml``)."""
    folder, name = os.path.split(path)
    measurement, images = os.path.split(folder)
    try:
        images = next((e for e in os.listdir(measurement) if e.lower() == images.lower()), images)
        folder = os.path.join(measurement, images)
        with os.scandir(folder) as entries:
            name = next((e.name for e in entries if e.name.lower() == name.lower()), name)
    except OSError:
        pass
    return os.path.join(folder, name)


def may_own(folder: str, name: str, folders) -> bool:
    """``group_files``' cheap test (reads nothing): a file in a side folder,
    in a measurement folder (next to ``Images``) or, unless it is a plane,
    in ``Images`` may be a part."""
    base = folders.base(folder).lower()
    return _side_folder(base) or "images" in folders.names(folder) or \
        (base == "images" and not name.lower().endswith(PLANE_SUFFIXES))


def owner(path: str, head):
    """A measurement folder, its side folders and their files, the other
    files in it (hs: the measurement's own description) and the files in
    ``Images`` that are not planes are parts of its index (module docstring)."""
    path = os.path.abspath(path)
    parent = os.path.dirname(path)
    if os.path.isdir(path):
        if _side_folder(os.path.basename(path)):
            path = os.path.dirname(path)
        index = _measurement_index(path)
    elif _side_folder(os.path.basename(parent)):
        index = _measurement_index(os.path.dirname(parent))
    elif os.path.basename(parent).lower() == "images":
        if path.lower().endswith(PLANE_SUFFIXES):
            return None
        index = _measurement_index(os.path.dirname(parent))
        if index is not None and os.path.normcase(index) == os.path.normcase(path):
            return None                              # the index itself
    else:
        index = _measurement_index(parent)
    return _parts.Part(index, index) if index is not None else None


def may_index(name: str) -> bool:
    """``group_files``' cheap test (reads nothing): an index is an XML file (``Index.idx.xml``, ``Index.xml``)."""
    return name.lower().endswith(('.xml',))


#: the index namespace ends in ``HarmonyV<n>``: ``.../PEHH/HarmonyV4`` .. ``V6``,
#: Harmony 5.2's ``43B2A954-E3C3-47E1-B392-6635266B0DD3/HarmonyV7``
NAMESPACE = re.compile(rb'xmlns="[^"]*/HarmonyV\d+"')


def sniff(head: bytes, path: str) -> bool:
    return b"EvaluationInputData" in head and NAMESPACE.search(head) is not None


def _text(e, name):
    child = e.find(f"{{*}}{name}")
    return None if child is None or child.text is None else child.text.strip()


def _int(e, name):
    value = _text(e, name)
    try:
        return int(value)
    except (TypeError, ValueError):
        raise CorruptFileError(f"Harmony index: an <Image> without a whole-number <{name}>")


def _fill_from_maps(root) -> None:
    """Harmony 5.2 / 6 indexes (V6, V7) keep what every plane of a channel
    shares -- image size, pixel size, channel name, wavelengths -- once, in
    ``Maps/Map/Entry[@ChannelID]``, no longer in each ``<Image>``.  Each
    ``<Image>`` gets the entries of its channel it lacks (its own values
    win; among the Maps, the first that has a value)."""
    per_channel: dict = {}
    for entry in root.findall("{*}Maps/{*}Map/{*}Entry"):
        values = per_channel.setdefault(entry.get("ChannelID"), {})
        for child in entry:
            values.setdefault(child.tag.split("}", 1)[-1], child)
    if not per_channel:
        return
    for e in root.findall("{*}Images/{*}Image"):
        values = per_channel.get(_text(e, "ChannelID"))
        if values:
            have = {child.tag.split("}", 1)[-1] for child in e}
            e.extend(child for name, child in values.items() if name not in have)


def _value(e, name, positive=True):
    """A length / time / wavelength the index gives with its ``Unit``."""
    child = None if e is None else e.find(f"{{*}}{name}")
    if child is None:
        return None
    return quantity(child.text, child.get("Unit"), positive=positive)


class _HarmonyImage(PlaneImage):
    """One field; metadata from its planes' ``<Image>`` entries."""

    def __init__(self, *args, entries: dict, system: str, **kwargs):
        super().__init__(*args, **kwargs)
        #: (t, c, z) position -> its <Image> element (also for missing planes)
        self._entries = entries
        #: the id of the coordinate system the position is in: its well's (4.9)
        self._system = system

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
        position = self._field_position(first, sizes)
        return Metadata(pixel_sizes={a: q for a, q in sizes.items() if q}, position=position,
                        coordinate_system=self._system if position else None,
                        channels=tuple(channels))

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
        _fill_from_maps(root)
        version = root.tag.split("}", 1)[0].rsplit("/", 1)[-1]           # HarmonyV4 .. V7
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
        plate_name = (_text(plate_e, "Name") or _text(plate_e, "PlateID")
                      or _measurement_name(folder) or "")
        base = plate_name or os.path.basename(path)
        row_names, column_names = layout(rows, columns, wells_rc, path)
        plates = [Plate(name=plate_name, rows=row_names, columns=column_names,
                        wells=tuple(Well(r, c, f"{row_label(r)}{c + 1}") for r, c in wells_rc))]
        images = []
        for (r, c, f) in sorted(kept, key=lambda k: (number[k[:2]], k[2])):
            images.append(_HarmonyImage(
                self._members, len(images),
                {position[p]: member for p, member in kept[(r, c, f)].items()}, shape, dtype,
                f"{base} well {row_label(r)}{c + 1} field {f - 1}", unit,
                {"plate": 0, "acquisition": 0, "well": number[(r, c)], "field": f - 1},
                # every listed plane's <Image> (metadata), whether its file is there or not
                entries={position[p]: e for p, e in listed[(r, c, f)].items() if p in position},
                system=well_system(plates, 0, number[(r, c)])))
        super().__init__(path, images, plates=plates)

    def _declare_coordinate_systems(self):
        """One system per well (plan 4.9): x / y relative to the well's
        centre, z as the instrument gives it (``_HarmonyImage._field_position``),
        so wells share no frame and no edge joins them."""
        return well_systems(self.plates)

    def close(self) -> None:
        super().close()
        self._members.close()


def open(path: str, tiles: str = "separate", resolutions: str = "base") -> _HarmonyFile:  # noqa: A001 - backend protocol
    """A Harmony plate from its index file (fields are not tiled, and Harmony
    stores no pyramids: *tiles* and *resolutions* change nothing)."""
    return _HarmonyFile(path)
