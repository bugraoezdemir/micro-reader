"""Metadata of TIFF-family images (docs/METADATA.md), from what tifffile
parses: OME-XML, then Zeiss LSM's CZ_LSMINFO, then the ImageJ description,
then MetaMorph STK's calibration, then MetaMorph / MetaSeries TIFF's
calibration, then the whole-slide formats' own records
(Aperio SVS, Leica SCN, Ventana BIF, PerkinElmer QPTIFF, Philips), then the
plain resolution tags.  The first of them a file has is used.

OME ``Plane`` PositionX / PositionY are taken as the centre of the image (as
Bio-Formats writes them) and moved to the first pixel's centre; without a
pixel size that cannot be done, and the position is left unknown.  They are
stage readings: such a position is in the file's ``stage`` system (plan 4.6,
``STAGE``), which OME-TIFF and companion files declare.  ``PositionZ`` is
no part of it (a companion takes only its z step from it).  The slide
formats record where an image lies on the slide, in the system ``slide``
(plan 4.11, ``SLIDE``): the top-left corner (SVS ``Left`` / ``Top``, SCN
view offsets, a QPTIFF scan's and thumbnail's X/YPosition tags), moved
half a pixel inwards, or the centre (a QPTIFF overview's and label's
X/YPosition tags, check 4.20), moved to the first pixel.  An SVS label and
macro record no position.
"""
from __future__ import annotations

import os
import re
import xml.etree.ElementTree as ET

from .._metadata import Channel, Metadata, Quantity, color_hex, quantity
from .._spatial import system_id

#: OME-XML's default units (schema 2016-06)
OME_DEFAULT_LENGTH = "µm"
OME_DEFAULT_TIME = "s"
OME_DEFAULT_WAVELENGTH = "nm"
#: TIFF ResolutionUnit -> the unit the resolution counts pixels per
RESOLUTION_UNITS = {2: "inch", 3: "centimeter"}
#: the system OME ``Plane`` positions are in (plan 4.6): one stage frame for
#: every image of an OME-TIFF or a companion set
STAGE = system_id("stage")


def tiff_metadata(image, n_channels: int, cache: dict, resolution_tags: bool = True):
    """The ``Metadata`` of TIFF *image* (a _TiffImage of series
    ``image._series_index``); *cache* holds what is parsed once per file.
    *resolution_tags* False: real metadata only, never the bare resolution tags
    (a companion's members fill its gaps this way, docs/COMPANION.md)."""
    tif = image._tif
    if tif.is_ome and tif.ome_metadata:
        meta = _ome(image, n_channels, cache)
        if meta is not None:
            return meta
    if tif.is_lsm and tif.lsm_metadata:
        return _lsm(tif.lsm_metadata, n_channels)
    if tif.is_imagej and tif.imagej_metadata is not None:
        return _imagej(tif, image._key, n_channels)
    if tif.is_stk and tif.stk_metadata:
        meta = _stk(tif.stk_metadata, n_channels)
        if meta is not None:
            return meta
    if getattr(tif, "is_metaseries", False):
        meta = _metaseries(tif, n_channels)
        if meta is not None:
            return meta
    if getattr(tif, "is_ndtiff", False):
        meta = _ndtiff(image, n_channels, cache)
        if meta is not None:
            return meta
    for flag, read in SLIDE_READERS:
        if getattr(tif, flag, False):
            meta = read(image, n_channels, cache)
            if meta is not None:
                return meta
            break
    # a BinaryOnly stub keeps its metadata in another file (an OME companion):
    # its resolution tags are no pixel size (a 96 dpi display default gave 264.6 um)
    stub = bool(tif.is_ome and "BinaryOnly" in (tif.ome_metadata or ""))
    sizes = _resolution_tags(image._key) if resolution_tags and not stub else {}
    if sizes:
        return Metadata(pixel_sizes=sizes, channels=_unknown_channels(n_channels))
    return None


def _stk(md: dict, n_channels: int):
    """MetaMorph STK (also MetaXpress plates): the spatial calibration of its
    UIC tags (``XCalibration`` / ``YCalibration`` in ``CalibrationUnits``).
    Their resolution tags are a meaningless 72 dpi.  None without one."""
    unit = md.get("CalibrationUnits") or None
    sizes = {axis: quantity(md.get(f"{axis.upper()}Calibration"), unit) for axis in "xy"}
    sizes = {a: q for a, q in sizes.items() if q}
    if not sizes:
        return None
    return Metadata(pixel_sizes=sizes, channels=_unknown_channels(n_channels))


def metaseries_plane_info(tif) -> dict:
    """The ``PlaneInfo`` of a MetaMorph / MetaSeries TIFF's first page (the
    ``<MetaData>`` XML of its description, as tifffile parses it); {} when
    there is none or it cannot be parsed."""
    try:
        md = tif.metaseries_metadata
    except Exception:                                   # noqa: BLE001 - a damaged XML
        return {}
    info = (md or {}).get("PlaneInfo")
    return info if isinstance(info, dict) else {}


def _metaseries(tif, n_channels: int):
    """MetaMorph / MetaSeries TIFF (the ``<MetaData>`` XML in the description;
    the members of MetaMorph ``.nd`` sets): the spatial calibration
    (``spatial-calibration-x`` / ``-y`` in ``spatial-calibration-units``),
    only when ``spatial-calibration-state`` is on.  None without one."""
    info = metaseries_plane_info(tif)
    if info.get("spatial-calibration-state") is not True:
        return None
    unit = info.get("spatial-calibration-units") or None
    sizes = {axis: quantity(info.get(f"spatial-calibration-{axis}"), unit) for axis in "xy"}
    sizes = {a: q for a, q in sizes.items() if q}
    if not sizes:
        return None
    return Metadata(pixel_sizes=sizes, channels=_unknown_channels(n_channels))


def _ndtiff(image, n_channels: int, cache: dict):
    """Micro-Manager NDTiff: channel names from ``NDTiff.index`` -- the
    string values of its ``channel`` axis, in order of first appearance, as
    tifffile numbers the channels -- and the pixel size from the first
    frame's own metadata (``PixelSizeUm``, um).  None without either."""
    if "ndtiff" not in cache:
        cache["ndtiff"] = _ndtiff_record(image._tif)
    names, size = cache["ndtiff"]
    sizes = {}
    q = quantity(size, "micrometer") if size else None
    if q:
        sizes = {"x": q, "y": q}
    if not sizes and not names:
        return None
    channels = [Channel(name=n) for n in names] if "c" in image.axes or len(names) == 1 else []
    return Metadata(pixel_sizes=sizes, channels=_channels(channels, n_channels))


def _ndtiff_record(tif) -> tuple:
    """(channel names, PixelSizeUm or None) of the dataset *tif* belongs to;
    ([], None) when its index cannot be read."""
    import json
    import struct

    import tifffile
    folder = tif.filehandle.dirname
    names, first = [], None
    try:
        for (axes, filename, _offset, _width, _height, _type, _compression, meta_at, meta_bytes,
             meta_compression) in tifffile.read_ndtiff_index(os.path.join(folder, "NDTiff.index")):
            value = axes.get("channel")
            if isinstance(value, str) and value not in names:
                names.append(value)
            if first is None and meta_bytes > 0 and meta_compression == 0:
                first = (filename, meta_at, meta_bytes)
    except (OSError, ValueError, struct.error):
        return [], None
    size = None
    if first is not None:
        try:
            with open(os.path.join(folder, first[0]), "rb") as fh:
                fh.seek(first[1])
                frame = json.loads(fh.read(first[2]).rstrip(b"\x00").decode("utf-8"))
            value = frame.get("PixelSizeUm") if isinstance(frame, dict) else None
            size = float(value) if isinstance(value, (int, float)) and value > 0 else None
        except (OSError, ValueError, UnicodeDecodeError):
            size = None
    return names, size


def _unknown_channels(n: int) -> tuple:
    return tuple(Channel() for _ in range(max(1, n)))


def _channels(found: list, n: int) -> tuple:
    """*found* padded with unknown channels / cut to the image's *n*."""
    n = max(1, n)
    found = list(found[:n])
    return tuple(found + [Channel()] * (n - len(found)))


# -- OME-XML ----------------------------------------------------------------

def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _children(element, name: str) -> list:
    return [e for e in element if _local(e.tag) == name]


def _ome_images(tif, cache: dict) -> list:
    """(first IFD of this file, Image element) per OME Image, parsed once."""
    if "ome" not in cache:
        images = []
        try:
            root = ET.fromstring(tif.ome_metadata)
        except ET.ParseError:
            root = None
        here = os.path.basename(tif.filehandle.path)
        for element in (root if root is not None else []):
            if _local(element.tag) != "Image":
                continue
            pixels = _children(element, "Pixels")
            first = None
            for data in (_children(pixels[0], "TiffData") if pixels else []):
                uuid = _children(data, "UUID")
                name = uuid[0].get("FileName") if uuid else None
                if name and os.path.basename(name) != here:
                    continue                           # planes in another file
                ifd = int(data.get("IFD", 0))
                first = ifd if first is None else min(first, ifd)
            images.append((first, element))
        cache["ome"] = images
    return cache["ome"]


def _ome_image(image, cache: dict):
    """The OME Image element of *image*'s series: the one whose planes start
    at the series' first IFD, else the one at the series' position."""
    images = _ome_images(image._tif, cache)
    if getattr(image._series, "kind", "") == "mmstack" and "series" in image.index:
        # Micro-Manager positions are one tifffile series (axis R, read as
        # series); its OME-XML has one Image per position, in position order
        n = image.index["series"]
        return images[n][1] if n < len(images) else None
    first_ifd = getattr(image._series.keyframe, "index", None)
    matches = [e for ifd, e in images if ifd is not None and ifd == first_ifd]
    if len(matches) == 1:
        return matches[0]
    n = getattr(image, "_series_index", image._position)
    return images[n][1] if n < len(images) else None


def _ome(image, n_channels: int, cache: dict):
    element = _ome_image(image, cache)
    if element is None:
        return None

    def annotations():
        if "annotations" not in cache:
            try:
                root = ET.fromstring(image._tif.ome_metadata)
            except ET.ParseError:
                root = None
            cache["annotations"] = ome_annotations(root)
        return cache["annotations"]

    # Micro-Manager writes its Java ARGB colours into OME Color (RGBA by the
    # schema): FITC #33FF33 is stored as -13369549, read as RGBA #FF33FF
    return ome_image_metadata(element, dict(zip(image.axes, image.shape)), n_channels,
                              image.index, annotations,
                              argb=bool(getattr(image._tif, "is_micromanager", False)))


def ome_annotations(root) -> dict:
    """{ID: XMLAnnotation element} of an OME root element (None: {})."""
    return {e.get("ID"): e for e in (root.iter() if root is not None else [])
            if _local(e.tag) == "XMLAnnotation"}


def ome_image_metadata(element, shape: dict, n_channels: int, index: dict, annotations,
                       argb: bool = False):
    """The ``Metadata`` an OME ``Image`` element gives, or None without
    ``Pixels``: pixel sizes, time increment, channels, the first plane's stage
    position (in ``STAGE``), Modulo index values.  Shared by OME-TIFF and
    companion files.

    *shape*: axis -> size of the image it describes (positions are moved from
    the field's centre to the first pixel); *index*: that image's index;
    *annotations*: a callable giving {ID: XMLAnnotation element}, called only
    when the image refers to an annotation.  *argb*: Channel ``Color`` holds
    Java ARGB ints (Micro-Manager's OME writer), not OME's RGBA."""
    pixels = _children(element, "Pixels")
    if not pixels:
        return None
    p = pixels[0]
    sizes = {}
    for axis, attr in (("x", "PhysicalSizeX"), ("y", "PhysicalSizeY"), ("z", "PhysicalSizeZ")):
        q = quantity(p.get(attr), p.get(attr + "Unit") or OME_DEFAULT_LENGTH)
        if q:
            sizes[axis] = q
    q = quantity(p.get("TimeIncrement"), p.get("TimeIncrementUnit") or OME_DEFAULT_TIME)
    if q:
        sizes["t"] = q
    channels = []
    for c in _children(p, "Channel"):
        channels.append(Channel(
            name=c.get("Name") or None,
            color=_ome_color(c.get("Color"), argb),
            emission_wavelength=quantity(c.get("EmissionWavelength"),
                                         c.get("EmissionWavelengthUnit") or OME_DEFAULT_WAVELENGTH),
            excitation_wavelength=quantity(c.get("ExcitationWavelength"),
                                           c.get("ExcitationWavelengthUnit") or OME_DEFAULT_WAVELENGTH)))
    position = {}
    planes = _children(p, "Plane")
    if planes:
        # the first plane (TheZ = TheT = TheC = 0 when listed)
        first = min(planes, key=lambda e: (int(e.get("TheT", 0)), int(e.get("TheZ", 0)),
                                           int(e.get("TheC", 0))))
        position = ome_plane_position(first, sizes, shape)
    return Metadata(pixel_sizes=sizes, position=position,
                    coordinate_system=STAGE if position else None,
                    channels=_channels(channels, n_channels),
                    index_values=_ome_index_values(element, index, annotations))


def ome_plane_position(plane, sizes: dict, shape: dict) -> dict:
    """The stage position of an OME ``Plane`` element's first pixel: x / y
    from its PositionX / Y, which Bio-Formats writes as the field's centre
    (checked: an OME-TIFF it converted from a CZI gives the CZI's first pixel
    this way); unknown without a pixel size or a length unit (OME's
    "reference frame")."""
    position = {}
    for axis, attr in (("x", "PositionX"), ("y", "PositionY")):
        q = quantity(plane.get(attr), plane.get(attr + "Unit"), positive=False)
        size = sizes.get(axis)
        if q and size and axis in shape:
            step = size.to(q.unit).value
            position[axis] = Quantity(q.value - (shape[axis] / 2 - 0.5) * step, q.unit)
    return position


#: OME Modulo types with a physical value per index (D5) -> micro-reader axis
MODULO_INDICES = {"lambda": "emission", "lifetime": "lifetime"}


def _ome_index_values(element, index: dict, annotations) -> dict:
    """This image's value of the indices an OME ``Modulo`` annotation adds
    (D5: only those with a physical meaning: lambda, lifetime), from Start /
    Step and Unit and the image's *index*; labels and unit-less values are not
    physical values."""
    refs = {e.get("ID") for e in element if _local(e.tag) == "AnnotationRef"}
    if not refs:
        return {}
    found = annotations()
    out = {}
    for ref in refs:
        annotation = found.get(ref)
        if annotation is None:
            continue
        for along in annotation.iter():
            if not _local(along.tag).startswith("ModuloAlong"):
                continue
            name = MODULO_INDICES.get(along.get("Type", ""))
            k = index.get(name) if name else None
            if k is None or "Start" not in along.attrib:
                continue
            try:
                start, step = float(along.get("Start")), float(along.get("Step", 1))
            except ValueError:
                continue
            q = quantity(start + k * step, along.get("Unit"), positive=False)
            if q:
                out[name] = q.to("nanometer") if name == "emission" else q
    return out


def _ome_color(value, argb: bool = False):
    """OME Color: a signed 32-bit RGBA integer; opaque white (-1) is
    Bio-Formats' and OME's default, i.e. no colour.  *argb*: a Java ARGB int
    (Micro-Manager), whose -1 is an explicit white."""
    if value in (None, ""):
        return None
    try:
        number = int(value)
    except ValueError:
        return None
    if argb:
        return "#%06X" % (number & 0xFFFFFF)
    if number == -1:
        return None
    return color_hex(number)


# -- Zeiss LSM ----------------------------------------------------------------

def _lsm(lm: dict, n_channels: int):
    sizes = {}
    for axis in "xyz":
        q = quantity(lm.get(f"VoxelSize{axis.upper()}"), "meter")     # CZ_LSMINFO: metres -> µm
        if q:
            sizes[axis] = q
    q = quantity(lm.get("TimeIntervall"), "second")
    if q:
        sizes["t"] = q
    colors = lm.get("ChannelColors") or {}
    rgbs = colors.get("Colors") or []
    names = colors.get("ColorNames") or []
    channels = [Channel(name=names[k] if k < len(names) and names[k] else None,
                        color=color_hex(rgbs[k][:3]) if k < len(rgbs) else None)
                for k in range(max(len(rgbs), len(names)))]
    return Metadata(pixel_sizes=sizes, channels=_channels(channels, n_channels))


# -- ImageJ -------------------------------------------------------------------

def _imagej(tif, key, n_channels: int):
    ij = tif.imagej_metadata or {}
    unit = _imagej_unit(ij.get("unit"))
    sizes = {}
    if unit:
        for axis, tag in (("x", "XResolution"), ("y", "YResolution")):
            per_unit = _rational(key.tags.get(tag))
            unit_here = _imagej_unit(ij.get(f"{axis}unit")) or unit
            if per_unit:
                q = quantity(1.0 / per_unit, unit_here)
                if q:
                    sizes[axis] = q
        q = quantity(ij.get("spacing"), _imagej_unit(ij.get("zunit")) or unit)
        if q:
            sizes["z"] = q
    q = quantity(ij.get("finterval"), ij.get("tunit") or "second")
    if q:
        sizes["t"] = q
    channels = []
    luts = ij.get("LUTs") or []
    for lut in luts:
        try:
            channels.append(Channel(color=color_hex([int(lut[0][-1]), int(lut[1][-1]),
                                                     int(lut[2][-1])])))
        except (IndexError, TypeError, ValueError):
            channels.append(Channel())
    return Metadata(pixel_sizes=sizes, channels=_channels(channels, n_channels))


def _imagej_unit(unit):
    """ImageJ writes µm as 'micron' or escaped as '\\u00B5m'; 'pixel' is no
    unit."""
    if not unit or str(unit).lower() in ("pixel", "pixels"):
        return None
    text = str(unit).replace("\\u00B5", "µ").replace("\\u00b5", "µ")
    return text


# -- whole-slide formats ----------------------------------------------------------
#
# Each reader gives the slide image's metadata, or None for an image it has
# nothing for (associated images: the resolution tags apply, as before).
# Every position they give is on the slide: in ``SLIDE``, which the file
# declares (``_TiffFile._declare_coordinate_systems``).

#: the system the slide formats' positions are in (plan 4.11): one frame for
#: every image of an SVS, SCN or QPTIFF file, the slide's
SLIDE = system_id("slide")


def _corner_to_first_pixel(corner: dict, sizes: dict) -> dict:
    """A recorded top-left corner (axis -> Quantity) -> the first pixel's
    centre; unknown without the pixel size."""
    position = {}
    for axis, q in corner.items():
        size = sizes.get(axis)
        if q is not None and size is not None:
            position[axis] = Quantity(q.value + 0.5 * size.to(q.unit).value, q.unit)
    return position


def _centre_to_first_pixel(centre: dict, sizes: dict, counts: dict) -> dict:
    """A recorded centre of the image (axis -> Quantity) -> the first pixel's
    centre, (n - 1) / 2 pixels back for *counts* n (axis -> pixels); unknown
    without the pixel size."""
    position = {}
    for axis, q in centre.items():
        size, n = sizes.get(axis), counts.get(axis)
        if q is not None and size is not None and n:
            position[axis] = Quantity(q.value - 0.5 * (n - 1) * size.to(q.unit).value, q.unit)
    return position


def _on_slide(position: dict) -> dict:
    """``Metadata`` keywords for a slide position: *position* and, when it is
    not empty, its system ``SLIDE`` (I3)."""
    return {"position": position, "coordinate_system": SLIDE if position else None}


def _xml(text):
    """An XML element from a description, its encoding declaration dropped
    (tifffile has already decoded it); None if it is no XML."""
    if not text:
        return None
    text = re.sub(r"^\s*<\?xml[^>]*\?>", "", text)
    try:
        return ET.fromstring(text)
    except ET.ParseError:
        return None


def _svs(image, n_channels: int, cache: dict):
    """Aperio SVS: ``MPP`` (µm per pixel) and ``Left`` / ``Top`` (mm, the
    corner on the slide) in the slide's description, for the slide.  Its
    label and macro record neither: they keep their resolution tags and stay
    unplaced (plan 4.11)."""
    if image._series.name != "Baseline":
        return None
    fields = {}
    for part in image._tif.pages.first.description.split("|")[1:]:
        if "=" in part:
            k, v = part.split("=", 1)
            fields[k.strip()] = v.strip()
    q = quantity(fields.get("MPP"), "micrometer")
    if not q:
        return None
    sizes = {"x": q, "y": q}
    corner = {"x": quantity(fields.get("Left"), "millimeter", positive=False),
              "y": quantity(fields.get("Top"), "millimeter", positive=False)}
    return Metadata(pixel_sizes=sizes, **_on_slide(_corner_to_first_pixel(corner, sizes)),
                    channels=_unknown_channels(n_channels))


def _scn(image, n_channels: int, cache: dict):
    """Leica SCN: the image's ``view`` (its extent and offset in the
    collection, nm) over its pixels gives the pixel size, and its offset the
    corner on the slide; channel names and colours from ``channelSettings``.
    The macro is placed too: its view is the whole collection, the frame
    every other view's offset is in (plan 4.11)."""
    if "scn" not in cache:
        cache["scn"] = _xml(image._tif.pages.first.description)
    root = cache["scn"]
    if root is None:
        return None
    name = getattr(image._series, "scn_name", image._series.name)   # the macro is renamed
    element = next((e for e in root.iter() if _local(e.tag) == "image"
                    and e.get("name") == name), None)
    if element is None:
        return None
    pixels = next((e for e in element if _local(e.tag) == "pixels"), None)
    view = next((e for e in element if _local(e.tag) == "view"), None)
    if pixels is None or view is None:
        return None
    sizes = {}
    for axis in "xy":
        try:
            sizes[axis] = quantity(float(view.get(f"size{axis.upper()}")) /
                                   float(pixels.get(f"size{axis.upper()}")), "nanometer")
        except (TypeError, ValueError, ZeroDivisionError):
            pass
    sizes = {a: q for a, q in sizes.items() if q}
    corner = {axis: quantity(view.get(f"offset{axis.upper()}"), "nanometer", positive=False)
              for axis in "xy"}
    channels = [Channel(name=c.get("name") or None, color=color_hex(c.get("rgb")))
                for c in element.iter() if _local(c.tag) == "channel"]
    return Metadata(pixel_sizes=sizes, **_on_slide(_corner_to_first_pixel(corner, sizes)),
                    channels=_channels(channels, n_channels))


def _bif(image, n_channels: int, cache: dict):
    """Ventana BIF: ``ScanRes`` (µm per pixel) of the scan's ``iScan``
    element, for the slide (its overview and probability images keep their
    resolution tags)."""
    if image._series.name != "Baseline":
        return None
    from ._tiff_variants import _xmp
    root = _xmp(image._series.keyframe)
    iscan = root.find("./SlideInfo/iScan") if root is not None else None
    q = quantity(iscan.get("ScanRes"), "micrometer") if iscan is not None else None
    if not q:
        return None
    return Metadata(pixel_sizes={"x": q, "y": q}, channels=_unknown_channels(n_channels))


#: a QPTIFF page's ``ImageType`` -> the point of the image its X/YPosition tags
#: record (check 4.20, on two Vectra Polaris scans): the scan and its thumbnail
#: give their top-left corner; the overview (the whole slide) and the label,
#: pictures from the wide-field camera, give their centre -- read as corners,
#: they were 31 mm and 15 mm away from where their pixels show the scan
QPI_POSITION_POINTS = {"FullResolution": "corner", "Thumbnail": "corner",
                       "Overview": "centre", "Label": "centre"}
#: the same by micro-reader's series name, for pages without an ``ImageType``:
#: only the scan and its thumbnail are known to give their corner
QPI_SERIES_POINTS = {"Baseline": "corner", "thumbnail": "corner"}


def _qpi_point(image) -> "str | None":
    """Which point of *image* its X/YPosition tags record ("corner",
    "centre"), or None when that is not known (no position then)."""
    root = _xml(getattr(image._series.keyframe, "description", ""))
    image_type = root.findtext("ImageType") if root is not None else None
    if image_type:
        return QPI_POSITION_POINTS.get(image_type.strip())
    return QPI_SERIES_POINTS.get(image._series.name)


def _qpi(image, n_channels: int, cache: dict):
    """PerkinElmer / Akoya QPTIFF: pixel size from the resolution tags, the
    position from the X/YPosition tags (both in ResolutionUnit, as
    Bio-Formats reads them) -- the scan's and thumbnail's corner, the
    overview's and label's centre (``QPI_POSITION_POINTS``; an image of
    another ``ImageType`` gets none) -- channel ``Name`` / ``Color`` from
    each channel page's XML description."""
    key = image._series.keyframe
    sizes = _resolution_tags(key)
    unit = RESOLUTION_UNITS.get(int(key.tags.valueof("ResolutionUnit", 2)))
    recorded = {}
    for axis, tag in (("x", "XPosition"), ("y", "YPosition")):
        value = _rational(key.tags.get(tag))
        if value is not None and unit:
            recorded[axis] = quantity(value, unit, positive=False)
    point = _qpi_point(image) if recorded else None
    if point == "corner":
        position = _corner_to_first_pixel(recorded, sizes)
    elif point == "centre":
        counts = {a: image.shape[image.axes.index(a)] for a in "xy" if a in image.axes}
        position = _centre_to_first_pixel(recorded, sizes, counts)
    else:
        position = {}
    channels = []
    if "c" in image.axes:
        for page in image._series.pages[:n_channels]:
            root = _xml(getattr(page, "description", "") if page is not None else "")
            name = root.findtext("Name") if root is not None else None
            color = root.findtext("Color") if root is not None else None
            rgb = None
            if color:
                try:
                    rgb = color_hex([int(v) for v in color.split(",")])
                except ValueError:
                    rgb = None
            channels.append(Channel(name=name or None, color=rgb))
    if not sizes and not channels:
        return None
    return Metadata(pixel_sizes=sizes, **_on_slide(position),
                    channels=_channels(channels, n_channels))


def _philips(image, n_channels: int, cache: dict):
    """Philips TIFF: the first ``PixelDataRepresentation``'s
    ``DICOM_PIXEL_SPACING`` (mm, rows then columns), for the slide."""
    if image._series.name != "Baseline":
        return None
    root = _xml(image._tif.philips_metadata)
    if root is None:
        return None
    for element in root.iter("DataObject"):
        if element.get("ObjectType") != "PixelDataRepresentation":
            continue
        for attribute in element.iter("Attribute"):
            if attribute.get("Name") == "DICOM_PIXEL_SPACING" and attribute.text:
                values = attribute.text.replace('"', "").split()
                if len(values) >= 2:
                    y, x = quantity(values[0], "millimeter"), quantity(values[1], "millimeter")
                    sizes = {a: q for a, q in (("x", x), ("y", y)) if q}
                    if sizes:
                        return Metadata(pixel_sizes=sizes, channels=_unknown_channels(n_channels))
        break
    return None


#: tifffile's file flag -> the slide format's metadata reader (first match)
SLIDE_READERS = (("is_svs", _svs), ("is_scn", _scn), ("is_bif", _bif), ("is_qpi", _qpi),
                 ("is_philips", _philips))


# -- plain TIFF -----------------------------------------------------------------

def _rational(tag):
    if tag is None:
        return None
    value = tag.value
    try:
        number = value[0] / value[1] if isinstance(value, tuple) else float(value)
    except (ZeroDivisionError, TypeError, IndexError):
        return None
    return number if number > 0 else None


def _resolution_tags(key) -> dict:
    """X / YResolution with an inch or centimetre ResolutionUnit (as
    Bio-Formats reads plain TIFFs); 'no unit' is not a physical size, and
    neither is 1 pixel per unit: a writer's placeholder (decided 2026-10-18:
    ImageXpress JDCE TIFFs carry 1 / inch, which would be 25 400 µm pixels)."""
    unit_tag = key.tags.get("ResolutionUnit")
    unit = RESOLUTION_UNITS.get(int(unit_tag.value) if unit_tag is not None else 2)
    if unit is None:
        return {}
    sizes = {}
    for axis, tag in (("x", "XResolution"), ("y", "YResolution")):
        per_unit = _rational(key.tags.get(tag))
        if per_unit and per_unit != 1:
            sizes[axis] = quantity(1.0 / per_unit, unit)
    return {a: q for a, q in sizes.items() if q}
