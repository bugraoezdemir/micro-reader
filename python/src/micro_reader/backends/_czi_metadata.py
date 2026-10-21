"""Metadata of CZI images (docs/METADATA.md), from the file's XML metadata
segment (``ZISRAWMETADATA``) and the first subblock's own metadata.

- Pixel sizes: ``Metadata/Scaling/Items/Distance`` X / Y / Z, stored in
  metres; given in the file's display unit (``DefaultUnitFormat``, normally
  µm) when OME-Zarr has a name for it, else micrometres.
- Time step: ``Information/Image/Dimensions/T/Positions/Interval/Increment``
  (seconds).
- Channels: ``Information/Image/Dimensions/Channels/Channel`` (Name, Color,
  Excitation / EmissionWavelength in nm), colours missing there from
  ``DisplaySetting/Channels``.
- Stage position: of the centre of the image's first pixel (docs/METADATA.md),
  from ``StageXPosition`` / ``StageYPosition`` (µm) of the image's first
  subblock, which Zeiss records for the centre of that field of view
  (Bio-Formats reports it as is: the centre); files without it: the
  scene's ``Positions/Position`` in the XML, taken as the image's centre.
"""
from __future__ import annotations

import struct
from dataclasses import replace
import xml.etree.ElementTree as ET

from .._metadata import Channel, Metadata, Quantity, color_hex, quantity, unit_name


def read_xml(fh):
    """The root of the XML metadata segment of the CZI open as *fh*, or
    None (no segment, unreadable XML)."""
    fh.seek(32)
    head = fh.read(80)
    metadata_position = struct.unpack_from("<iiii16s16siqqiq", head)[8]
    if not metadata_position:
        return None
    fh.seek(metadata_position)
    seg = fh.read(32 + 8)
    if seg[:14] != b"ZISRAWMETADATA":
        return None
    size, _attachment = struct.unpack_from("<ii", seg, 32)
    fh.seek(metadata_position + 32 + 256)
    try:
        return ET.fromstring(fh.read(size))
    except ET.ParseError:
        return None


def subblock_tags(fh, position: int) -> dict:
    """The ``Tags`` of the subblock segment at *position*: {tag: text}."""
    fh.seek(position)
    head = fh.read(32 + 48)
    if head[:14] != b"ZISRAWSUBBLOCK":
        return {}
    metadata_size, = struct.unpack_from("<i", head, 32)
    ndims, = struct.unpack_from("<i", head, 32 + 44)
    if metadata_size <= 0:
        return {}
    fh.seek(position + 32 + 16 + max(240, 32 + 20 * ndims))
    try:
        root = ET.fromstring(fh.read(metadata_size).rstrip(b"\x00"))
    except ET.ParseError:
        return {}
    tags = root.find("Tags")
    return {} if tags is None else {e.tag: (e.text or "").strip() for e in tags}


def pixel_sizes(root) -> dict:
    sizes = {}
    if root is None:
        return sizes
    for d in root.findall("Metadata/Scaling/Items/Distance"):
        axis = (d.get("Id") or "").lower()
        if axis not in ("x", "y", "z"):
            continue
        q = quantity(d.findtext("Value"), "meter")
        if q is None:
            continue
        shown = unit_name(d.findtext("DefaultUnitFormat"))
        # the display unit when it is a length OME-Zarr names (and not
        # larger than a millimetre: ``quantity``), else µm
        try:
            shown_q = q.to(shown) if shown else q
        except KeyError:
            shown_q = q
        sizes[axis] = quantity(shown_q.value, shown_q.unit)
    increment = root.findtext("Metadata/Information/Image/Dimensions/T/Positions/Interval/Increment")
    q = quantity(increment, "second")
    if q:
        sizes["t"] = q
    return sizes


def channels(root, c_start: int, n: int) -> tuple:
    """Channels with indices c_start .. c_start + n - 1 of the file."""
    found = []
    display = []
    if root is not None:
        found = root.findall("Metadata/Information/Image/Dimensions/Channels/Channel")
        display = root.findall("Metadata/DisplaySetting/Channels/Channel")
    display_color = {}
    for k, d in enumerate(display):
        color = color_hex(d.findtext("Color"))
        for name in (d.get("Name"), d.get("Id"), k):
            display_color.setdefault(name, color)
    out = []
    for k in range(c_start, c_start + max(1, n)):
        if k >= len(found):
            out.append(Channel())
            continue
        c = found[k]
        name = c.get("Name") or None
        color = color_hex(c.findtext("Color"))
        if color is None:
            color = display_color.get(name) or display_color.get(c.get("Id"))
            if color is None and len(display) == len(found):
                color = display_color.get(k)
        out.append(Channel(
            name=name, color=color,
            emission_wavelength=quantity(c.findtext("EmissionWavelength"), "nanometer"),
            excitation_wavelength=quantity(c.findtext("ExcitationWavelength"), "nanometer")))
    return tuple(out)


def scene_position(root, scene: int) -> dict:
    """The scene's stage position in the XML (µm): ``Positions/Position``."""
    if root is None:
        return {}
    for e in root.findall("Metadata/Information/Image/Dimensions/S/Scenes/Scene"):
        if e.get("Index") == str(scene):
            p = e.find("Positions/Position")
            if p is not None:
                out = {a: quantity(p.get(a.upper()), "micrometer", positive=False)
                       for a in ("x", "y")}
                return {a: q for a, q in out.items() if q}
    return {}


def plate_layout(root):
    """The plate a CZI was acquired on (docs/HCS.md), or None when it is not
    a plate: ``{"rows": n, "columns": m, "scenes": {scene: (row, column,
    name)}}`` with 0-based rows and columns.

    - The plate: the sample-holder template of the acquisition
      (``.../SampleHolder/Template`` with ``ShapeRows`` / ``ShapeColumns``;
      e.g. ``Name="Multichamber 96"``, 8 x 12).
    - Each scene's well: ``Dimensions/S/Scenes/Scene`` has a ``Shape`` with
      1-based ``RowIndex`` / ``ColumnIndex`` and the well's ``Name`` ("B2").
      Several scenes can lie in one well.
    It is a plate only when the template gives the layout and every scene
    names its well inside it."""
    if root is None:
        return None
    template = root.find(".//SampleHolder/Template")
    if template is None:
        return None
    try:
        rows = int(template.findtext("ShapeRows") or 0)
        columns = int(template.findtext("ShapeColumns") or 0)
    except ValueError:
        return None
    if rows < 1 or columns < 1:
        return None
    scenes = {}
    for e in root.findall("Metadata/Information/Image/Dimensions/S/Scenes/Scene"):
        shape = e.find("Shape")
        if shape is None:
            return None
        try:
            index = int(e.get("Index"))
            row = int(shape.findtext("RowIndex")) - 1
            column = int(shape.findtext("ColumnIndex")) - 1
        except (TypeError, ValueError):
            return None
        if not (0 <= row < rows and 0 <= column < columns):
            return None
        name = shape.get("Name") or e.get("Name") or ""
        scenes[index] = (row, column, name)
    if not scenes:
        return None
    return {"rows": rows, "columns": columns, "scenes": scenes}


def position_from_tags(tags: dict) -> dict:
    out = {}
    for axis in ("x", "y"):
        q = quantity(tags.get(f"Stage{axis.upper()}Position"), "micrometer", positive=False)
        if q:
            out[axis] = q
    return out


def first_pixel_position(image, centre: dict, s, sizes: dict) -> dict:
    """Stage position of the centre of *image*'s first pixel, from *centre*:
    the stage position of the centre of subblock *s* (Zeiss records where
    the field of view's centre was).  Pixel coordinates are one frame for
    the whole file, so the offset between the two frames found at *s* holds
    for the image's corner too."""
    out = {}
    origin = dict(zip("xy", image.origin))
    extent = {"x": s.width, "y": s.height}
    start = {"x": s.x, "y": s.y}
    for axis, q in centre.items():
        size = sizes.get(axis)
        if size is None:
            continue
        step = size.to(q.unit).value
        offset = q.value - (start[axis] + extent[axis] / 2) * step
        out[axis] = Quantity(offset + (origin[axis] + 0.5) * step, q.unit)
    return out


def czi_metadata(image, fh, root) -> Metadata:
    """*image*'s Metadata; *fh* an open handle of its file, *root* the
    file's XML metadata (or None)."""
    axes = image.axes
    n_c = image.shape[axes.index("c")] if "c" in axes else 1
    sizes = pixel_sizes(root)
    # the first subblock drawn of the first plane
    first = image._planes[min(image._planes)].subblocks[0] if image._planes else None
    position = {}
    if first is not None and first.position >= 0:
        centre = position_from_tags(subblock_tags(fh, first.position))
        position = first_pixel_position(image, centre, first, sizes)
    if not position and first is not None:
        # no stage position per subblock: the scene's, taken as the centre of
        # the whole scene (all its level-0 tiles; as Bio-Formats places it)
        tiles = [e for e in image._czi._level0 if e.series == first.series]
        x0, y0 = min(e.x for e in tiles), min(e.y for e in tiles)
        x1 = max(e.x + e.width for e in tiles)
        y1 = max(e.y + e.height for e in tiles)
        whole = replace(first, x=x0, y=y0, width=x1 - x0, height=y1 - y0)
        position = first_pixel_position(image, scene_position(root, first.series), whole, sizes)
    ids = getattr(image, "_channel_ids", None)           # a part: the file's channels it holds
    if ids is not None:
        every = channels(root, 0, max(ids) + 1)
        chosen = tuple(every[k] for k in ids)
    else:
        chosen = channels(root, image._czi._dim_starts.get("C", 0), n_c)
    return Metadata(pixel_sizes=sizes, position=position, channels=chosen)
