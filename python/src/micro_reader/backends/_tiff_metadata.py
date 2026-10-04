"""Metadata of TIFF-family images (docs/METADATA.md), from what tifffile
parses: OME-XML, then Zeiss LSM's CZ_LSMINFO, then the ImageJ description,
then the plain resolution tags.  The first of them a file has is used.

OME ``Plane`` PositionX / PositionY are taken as the centre of the image (as
Bio-Formats writes them) and moved to the first pixel's centre; without a
pixel size that cannot be done, and the position is left unknown.
"""
from __future__ import annotations

import os
import xml.etree.ElementTree as ET

from .._metadata import Channel, Metadata, Quantity, color_hex, quantity

#: OME-XML's default units (schema 2016-06)
OME_DEFAULT_LENGTH = "µm"
OME_DEFAULT_TIME = "s"
OME_DEFAULT_WAVELENGTH = "nm"
#: TIFF ResolutionUnit -> the unit the resolution counts pixels per
RESOLUTION_UNITS = {2: "inch", 3: "centimeter"}


def tiff_metadata(image, n_channels: int, cache: dict):
    """The ``Metadata`` of TIFF *image* (a _TiffImage of series
    ``image._series_index``); *cache* holds what is parsed once per file."""
    tif = image._tif
    if tif.is_ome and tif.ome_metadata:
        meta = _ome(image, n_channels, cache)
        if meta is not None:
            return meta
    if tif.is_lsm and tif.lsm_metadata:
        return _lsm(tif.lsm_metadata, n_channels)
    if tif.is_imagej and tif.imagej_metadata is not None:
        return _imagej(tif, image._key, n_channels)
    sizes = _resolution_tags(image._key)
    if sizes:
        return Metadata(pixel_sizes=sizes, channels=_unknown_channels(n_channels))
    return None


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
            color=_ome_color(c.get("Color")),
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
        shape = dict(zip(image.axes, image.shape))
        for axis, attr in (("x", "PositionX"), ("y", "PositionY")):
            # no unit: OME's "reference frame", which is not a length
            q = quantity(first.get(attr), first.get(attr + "Unit"), positive=False)
            size = sizes.get(axis)
            if q and size and axis in shape:
                # Bio-Formats writes the field's centre (checked: an OME-TIFF
                # it converted from a CZI gives the CZI's first pixel this way)
                step = size.to(q.unit).value
                position[axis] = Quantity(q.value - (shape[axis] / 2 - 0.5) * step, q.unit)
    return Metadata(pixel_sizes=sizes, position=position,
                    channels=_channels(channels, n_channels),
                    index_values=_ome_index_values(image, element, cache))


#: OME Modulo types with a physical value per index (D5) -> micro-reader axis
MODULO_INDICES = {"lambda": "emission", "lifetime": "lifetime"}


def _ome_index_values(image, element, cache: dict) -> dict:
    """This image's value of the indices an OME ``Modulo`` annotation adds
    (D5: only those with a physical meaning: lambda, lifetime), from Start /
    Step and Unit and the image's index; labels and unit-less values are not
    physical values."""
    refs = {e.get("ID") for e in element if _local(e.tag) == "AnnotationRef"}
    if not refs:
        return {}
    if "annotations" not in cache:
        try:
            root = ET.fromstring(image._tif.ome_metadata)
        except ET.ParseError:
            root = None
        cache["annotations"] = {e.get("ID"): e for e in (root.iter() if root is not None else [])
                                if _local(e.tag) == "XMLAnnotation"}
    out = {}
    for ref in refs:
        annotation = cache["annotations"].get(ref)
        if annotation is None:
            continue
        for along in annotation.iter():
            if not _local(along.tag).startswith("ModuloAlong"):
                continue
            name = MODULO_INDICES.get(along.get("Type", ""))
            k = image.index.get(name) if name else None
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


def _ome_color(value):
    """OME Color: a signed 32-bit RGBA integer; opaque white (-1) is
    Bio-Formats' and OME's default, i.e. no colour."""
    if value in (None, ""):
        return None
    try:
        number = int(value)
    except ValueError:
        return None
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
    Bio-Formats reads plain TIFFs); 'no unit' is not a physical size."""
    unit_tag = key.tags.get("ResolutionUnit")
    unit = RESOLUTION_UNITS.get(int(unit_tag.value) if unit_tag is not None else 2)
    if unit is None:
        return {}
    sizes = {}
    for axis, tag in (("x", "XResolution"), ("y", "YResolution")):
        per_unit = _rational(key.tags.get(tag))
        if per_unit:
            sizes[axis] = quantity(1.0 / per_unit, unit)
    return {a: q for a, q in sizes.items() if q}
