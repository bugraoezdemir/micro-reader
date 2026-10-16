"""Olympus / Evident cellSens VSI (VS120 / VS200 slide scanners, cellSens).

A composite format (docs/COMPOSITE.md, data store): ``<name>.vsi`` holds the
metadata (a tree right after its TIFF header, ``_vsi_tree``) and a few
small pictures as TIFF pages; the images themselves are tiled pyramids in
``_<name>_/stack<N>/frame_t*.ets`` (``_ets``).  Written after slideio's VSI
driver (Stanislav Melnikov, BSD-3: ``vsifile.cpp``, ``volume.cpp``,
``etsfile.cpp``); the series follow Bio-Formats (checked on the OME
cellSens samples).

Images, in this order (Bio-Formats'):

- one per stack folder with an ETS file, in stack-number order: the
  metadata volume (``MULTIDIM_IMAGE_VOLUME``) whose stack id is the folder's
  number and whose size fits the ETS tile grid (else the first fitting
  volume no other folder names).  Named by the volume's ``STACK_NAME``;
  the slide label (stack type 256) is ``label``, the slide overview (type
  1) ``overview``.  Each image carries its pyramid (``resolutions``);
- the ``.vsi``'s first TIFF page, ``macro`` (the slide photo).

Not images (as in Bio-Formats): focus maps, sample masks, focus points
(volumes without an ETS file), ETS "blob" files, the other TIFF pages.

Axes ``tczyx``, plus ``s`` for RGB.  The ETS tile coordinates are tile
column, tile row, then the volume's dimensions -- each
``DIMENSION_DESCRIPTION_VOLUME`` (second tag ``k``) holds its kind
(``DIMENSION_INDEX``: 1 z, 2 t, 3 lambda, 4 channel, 9 phase) at
coordinate ``k + 2`` -- and the pyramid level last; slideio's reconciliation
with the tiles (``fitDimensionOrderToTiles``) is kept.  Lambda or phase
with more than one index are refused for now.

Metadata: pixel size ``RWC_FRAME_SCALE`` in ``RWC_FRAME_UNIT``
(``10^-6m^1``); z / t steps from the dimension parameters (``VALUE`` in
``UNITS``; a negative z step -- a stack taken downwards -- gives its size);
channel names (``CHANNEL_NAME``), emission wavelengths and display colours
per channel, or the stack's own ``CHANNEL_NAME`` for a single channel.  No
stage position.

A bare ``.ets`` (its ``.vsi`` missing) is read alone when it is a 2-D
pyramid; with more dimensions it raises -- only the ``.vsi`` says which
coordinate is z, t or channel.
"""
from __future__ import annotations

import builtins
import os
import re
import warnings

import numpy as np

from .. import _parts
from .._base import File, Image
from .._errors import CompositeWarning, CorruptFileError, UnsupportedFormatError
from .._metadata import Channel, Metadata, color_hex, quantity
from . import _ets
from ._vsi_tree import read_tree

NAME = "vsi"
#: the first volume header right after the TIFF header: size 24, magic "IS"
TREE_MAGIC = b"\x18\x00IS"
STACK_FOLDER = re.compile(r"^stack(\d+)$", re.I)
#: tags (vsitags.hpp)
COLLECTION, IMAGE_VOLUME, FRAME, STACK_PROPERTIES = 2000, 2001, 2002, 2005
DIMENSION_DESCRIPTION, DIMENSION_PARAMETERS, DIMENSION_INDEX = 2007, 2008, 2023
STEP, EXTERNAL_FILE, IMAGE_BOUNDARY, PIXEL_IFD = 2013, 2018, 2053, 2016
STACK_NAME, STACK_TYPE, FRAME_SCALE, FRAME_UNIT = 2030, 2074, 2019, 2020
CHANNEL_NAME, CHANNEL_WAVELENGTH, DISPLAY_COLOR, DISPLAY_LUT = 2419, 2417, 2008, 2004
UNITS, VALUE = 268435456, 268435458
#: DIMENSION_INDEX -> dimension
DIMENSION_KINDS = {1: "z", 2: "t", 3: "lambda", 4: "c", 9: "phase"}
#: stack types named as Bio-Formats names them
TYPE_NAMES = {256: "label", 1: "overview"}
UNIT = re.compile(r"^(?:10\^([-+]?\d+))?([ms])(?:\^\+?1)?$")


def _unit(text):
    """('meter' | 'second', factor) of a VSI unit string (``10^-6m^1``)."""
    match = UNIT.match(re.sub(r"\s", "", str(text or "")))
    if match is None:
        return None, None
    return ("meter" if match.group(2) == "m" else "second"), 10.0 ** int(match.group(1) or 0)


def _length(value, unit_text):
    kind, factor = _unit(unit_text)
    if kind != "meter" or value is None:
        return None
    return quantity(abs(float(value)) * factor, "meter")


# -- the volumes of the metadata ---------------------------------------------------------

class _Volume:
    """What the reader needs from one ``MULTIDIM_IMAGE_VOLUME``."""

    def __init__(self, node):
        self.node = node
        self.stack = node.second
        props = node.child(STACK_PROPERTIES)
        self.props = props
        value = lambda n: n.value if n is not None else None  # noqa: E731
        self.name = value(props.child(STACK_NAME)) if props else None
        stack_type = value(props.child(STACK_TYPE)) if props else None
        self.type = int(stack_type) if isinstance(stack_type, (int, float)) else 0
        frames = [f for f in node.all(FRAME) if f.child(EXTERNAL_FILE) or f.child(PIXEL_IFD)]
        boundary = frames[0].path(EXTERNAL_FILE, IMAGE_BOUNDARY) if frames else None
        self.size = (boundary.value[3], boundary.value[2]) \
            if boundary is not None and isinstance(boundary.value, list) and \
            len(boundary.value) == 4 else None                 # (rows, cols)
        self.slots = {}                                        # dimension -> coordinate slot
        self.steps = {}                                        # dimension -> Quantity
        self.channels = {}                                     # index -> channel node
        by_kind = {}                                           # dimension -> its description
        for described in node.all(DIMENSION_DESCRIPTION):
            for params in described.all(DIMENSION_PARAMETERS):
                index = params.child(DIMENSION_INDEX)
                kind = DIMENSION_KINDS.get(index.value) if index is not None else None
                if kind is None:
                    continue
                by_kind[kind] = described
                self.slots[kind] = described.second + 2
                step = params.path(STEP, VALUE)
                unit = params.path(STEP, UNITS)
                if step is not None and unit is not None:
                    self.steps[kind] = (step.value, unit.value)
        # per-channel properties: the channel dimension's nodes, second tag =
        # channel index; channel 0's is the node that also holds the dimension
        # index (checked on the OME FL samples)
        if "c" in by_kind:
            for params in by_kind["c"].all(DIMENSION_PARAMETERS):
                if params.second >= 0:
                    self.channels[params.second] = params

    def fit_slots(self, ndim: int, pyramid: bool) -> dict:
        """The dimension slots reconciled with an ETS file of *ndim*
        coordinates (slideio ``Volume::fitDimensionOrderToTiles``)."""
        slots = dict(self.slots)
        level = ndim - 1 if pyramid else -1
        upper = ndim - 1 if pyramid else ndim
        for kind in ("z", "t"):
            if slots.get(kind) == level:
                del slots[kind]
        if slots and all(v >= upper for v in slots.values()):
            slots = {k: v - 1 for k, v in slots.items()}
        if "z" not in slots and "t" not in slots and ndim > 4:
            stated_c = "c" in slots
            wanted = {"c": slots.get("c", 2)}
            wanted["t"] = slots["c"] + 2 if stated_c else 3
            if ndim > 5:
                wanted["z"] = slots["c"] + 1 if stated_c else 4
            for kind, slot in wanted.items():
                if kind not in slots and 2 <= slot < upper and slot not in slots.values():
                    slots[kind] = slot
        return {k: v for k, v in slots.items() if 2 <= v < upper}


def _volumes(tree) -> list:
    collection = tree.child(COLLECTION)
    return [_Volume(n) for n in collection.all(IMAGE_VOLUME)] if collection else []


def _ets_files(folder: str) -> list:
    """(stack number, path) of each stack folder's frame ETS file, by number."""
    found = []
    for entry in os.listdir(folder):
        match = STACK_FOLDER.match(entry)
        stack = os.path.join(folder, entry)
        if match is None or not os.path.isdir(stack):
            continue
        frames = sorted(e for e in os.listdir(stack)
                        if e.lower().startswith("frame_t") and e.lower().endswith(".ets"))
        if frames:
            found.append((int(match.group(1)), os.path.join(stack, frames[0])))
    return sorted(found)


def _pair(ets, volumes: list, claimed: set, stack: int):
    """The volume an ETS file belongs to: its stack id, if its size fits the
    tiles; else the first fitting volume no folder names."""
    th, tw = ets.tile
    top_x, top_y = ets.top[0], ets.top[1]
    fits = [v for v in volumes if v.size is not None
            and top_y * th <= v.size[0] <= (top_y + 1) * th
            and top_x * tw <= v.size[1] <= (top_x + 1) * tw]
    for v in fits:
        if v.stack == stack:
            return v
    return next((v for v in fits if v.stack not in claimed), None)


# -- images ------------------------------------------------------------------------------------

class _VsiImage(Image):
    """One level of an ETS-backed volume: axes ``tczyx`` (+ ``s``)."""

    def __init__(self, position: int, name: str, ets, level: int, size: tuple, slots: dict,
                 extents: dict, volume=None, fill=0):
        rows, cols = _halved(size[0], level), _halved(size[1], level)
        rgb = ets.components > 1
        axes = "tczyx" + ("s" if rgb else "")
        shape = (extents["t"], extents["c"], extents["z"], rows, cols) + \
            ((ets.components,) if rgb else ())
        super().__init__(position, axes, shape, ets.dtype.newbyteorder("="), name=name)
        self._ets, self._level, self._slots, self._volume = ets, level, slots, volume
        self._rgb, self._fill, self._size = rgb, fill, size

    def _read_block(self, block):
        ts, cs, zs, ys, xs = block[:5]
        ss = block[5] if self._rgb else slice(0, 1)
        out = np.empty(tuple(s.stop - s.start for s in block), self.dtype)
        n_extra = self._ets.extra_dims
        read = 0
        for t in range(ts.start, ts.stop):
            for c in range(cs.start, cs.stop):
                for z in range(zs.start, zs.stop):
                    extra = [0] * n_extra
                    for kind, value in (("t", t), ("c", c), ("z", z)):
                        if kind in self._slots:
                            extra[self._slots[kind] - 2] = value
                    plane = np.empty((ys.stop - ys.start, xs.stop - xs.start,
                                      self._ets.components), self.dtype)
                    read += self._ets.read(self._level, tuple(extra), ys, xs, plane, self._fill)
                    dest = out[t - ts.start, c - cs.start, z - zs.start]
                    if self._rgb:
                        dest[...] = plane[..., ss]
                    else:
                        dest[...] = plane[..., 0]
        return out, read

    def _read_metadata(self):
        volume = self._volume
        if volume is None:
            return None
        props = volume.props
        sizes = {}
        scale = props.child(FRAME_SCALE) if props else None
        unit = props.child(FRAME_UNIT) if props else None
        if scale is not None and isinstance(scale.value, list) and len(scale.value) == 2:
            for axis, v in zip("xy", scale.value):
                q = _length(v, unit.value if unit else None)
                if q:
                    sizes[axis] = q
        if "z" in volume.steps and self.shape[2] > 1:
            q = _length(*volume.steps["z"])
            if q:
                sizes["z"] = q
        if "t" in volume.steps and self.shape[0] > 1:
            value, unit_text = volume.steps["t"]
            kind, factor = _unit(unit_text)
            if kind == "second" and value:
                sizes["t"] = quantity(abs(float(value)) * factor, "second")
        channels = []
        for c in range(self.shape[1]):
            node = volume.channels.get(c)
            if self.shape[1] == 1 and props is not None and props.child(CHANNEL_NAME) is not None \
                    and (node is None or node.child(CHANNEL_NAME) is None):
                node = props                                   # a single channel: the stack's
            name = node.child(CHANNEL_NAME).value if node is not None and \
                node.child(CHANNEL_NAME) is not None else None
            wave = node.path(CHANNEL_WAVELENGTH, VALUE) if node is not None else None
            color = None
            if node is not None and node is not props:
                display = node.child(DISPLAY_COLOR)
                lut = node.path(DISPLAY_LUT, VALUE)
                for source in (display, lut):
                    if source is not None and isinstance(source.value, list) and source.value:
                        color = color_hex(source.value[-1])
                        break
            channels.append(Channel(
                name=name or None, color=color,
                emission_wavelength=quantity(wave.value, "nanometer")
                if wave is not None and isinstance(wave.value, (int, float)) and wave.value > 0
                else None))
        return Metadata(pixel_sizes={a: q for a, q in sizes.items() if q}, channels=tuple(channels))


class _VsiFile(File):
    backend = NAME

    def __init__(self, path, images, owned, format_name="vsi"):
        self.format = format_name
        super().__init__(path, images)
        self._owned = owned

    def close(self) -> None:
        super().close()
        for item in self._owned:
            item.close()


def _halved(n: int, level: int) -> int:
    """A level's size: level 0's halved *level* times, rounded up (as
    Bio-Formats; the ETS tile grid covers that many pixels)."""
    return -(-n // (1 << level))


def _pyramid(position, name, ets, size, slots, extents, volume, fill) -> _VsiImage:
    levels = [_VsiImage(position, name, ets, k, size, slots, extents, volume, fill)
              for k in range(ets.levels)]
    for level in levels:
        level._resolutions = levels
    for k, level in enumerate(levels):
        level.index = {"scene": position, "resolution": k}
    return levels[0]


def _extents(ets, slots: dict, where: str) -> dict:
    extents = {"t": 1, "c": 1, "z": 1}
    for kind, slot in slots.items():
        n = ets.top[slot] + 1
        if kind in extents:
            extents[kind] = n
        elif n > 1:
            raise UnsupportedFormatError(f"{where}: a {kind} dimension of {n}: not supported yet")
    unknown = [k for k in range(2, 2 + ets.extra_dims) if k not in slots.values()
               and ets.top[k] > 0]
    if unknown:
        raise UnsupportedFormatError(
            f"{where}: tile coordinates {unknown} vary but the metadata names no dimension for "
            "them")
    return extents


# -- the backend protocol ------------------------------------------------------------------

def sniff(head: bytes, path: str) -> bool:
    if head[:4] == b"II*\x00" and head[8:12] == TREE_MAGIC:
        return True
    return head[:4] == b"SIS\x00" and len(head) >= 8 and head[4:8] == b"\x40\x00\x00\x00"


def may_own(folder: str, name: str, folders) -> bool:
    """``group_files``' cheap test (reads nothing): a file two folders below
    a ``_<name>_`` folder may be a part (*folder*: the file's, absolute)."""
    base = folders.base(folder, 1)
    return len(base) > 2 and base.startswith("_") and base.endswith("_")


def owner(path: str, head):
    """The parts of a VSI (docs/COMPOSITE.md): its ``_<name>_`` folder, the
    ``stack<N>`` folders and the files in them open ``<name>.vsi``.  A
    frame ETS is an image on its own (read alone, with a warning, when the
    ``.vsi`` is missing)."""
    path = os.path.abspath(path)
    parts, standalone = [], False
    if os.path.isdir(path):
        parts = [path, os.path.dirname(path)]
    else:
        parts = [os.path.dirname(os.path.dirname(path))]
        name = os.path.basename(path).lower()
        standalone = name.startswith("frame_t") and name.endswith(".ets")
    for folder in parts:
        base = os.path.basename(folder)
        if len(base) > 2 and base.startswith("_") and base.endswith("_"):
            stem = base[1:-1]
            if folder != parts[0] and not STACK_FOLDER.match(os.path.basename(parts[0])):
                continue
            parent = os.path.dirname(folder)
            return _parts.Part(_parts.find(parent, stem + ".vsi"),
                               os.path.join(parent, stem + ".vsi"), standalone)
    return None


def open(path: str, tiles: str = "separate", resolutions: str = "base") -> _VsiFile:  # noqa: A001 - backend protocol
    head = _parts.read_head(path)
    if head[:3] == b"SIS":
        return _open_bare_ets(path)
    with builtins.open(path, "rb") as fh:
        data = fh.read()
    tree = read_tree(data)
    volumes = _volumes(tree)
    where = repr(path)
    folder = _parts.find(os.path.dirname(path), "_" + os.path.splitext(os.path.basename(path))[0] + "_")
    files = _ets_files(folder) if folder and os.path.isdir(folder) else []
    if not files and any(v.size for v in volumes):
        warnings.warn(CompositeWarning(
            f"{where}: its image folder {'_' + os.path.splitext(os.path.basename(path))[0] + '_'!r} "
            "is missing: only the macro picture is read"), stacklevel=3)
    claimed = {stack for stack, _ in files}
    pool = [v for v in volumes if v.size]
    owned, images = [], []
    try:
        for stack, ets_path in files:
            ets = _ets.EtsFile(ets_path)
            owned.append(ets)
            volume = _pair(ets, pool, claimed, stack)
            if volume is None:
                warnings.warn(CompositeWarning(
                    f"{where}: {os.path.relpath(ets_path, os.path.dirname(path))!r} matches no "
                    "image of the metadata: left out"), stacklevel=3)
                continue
            pool.remove(volume)
            slots = volume.fit_slots(ets.ndim, ets.pyramid)
            extents = _extents(ets, slots, f"{where} stack {stack}")
            if ets.components > 1 and extents["c"] > 1:
                raise UnsupportedFormatError(f"{where} stack {stack}: RGB tiles along a channel "
                                             "dimension")
            name = TYPE_NAMES.get(volume.type) or volume.name or f"stack{stack}"
            images.append(_pyramid(len(images), name, ets, volume.size, slots, extents, volume, 0))
        macro = _macro(path, len(images))
        if macro is not None:
            owned.append(macro._file)
            images.append(macro)
        if not images:
            raise CorruptFileError(f"{where}: no image (no ETS file matches the metadata)")
    except BaseException:
        for item in owned:
            item.close()
        raise
    return _VsiFile(path, images, owned)


def _macro(path: str, position: int):
    """The ``.vsi``'s first TIFF page (the slide photo), as ``macro``."""
    from . import tiff
    try:
        file = tiff.open_pages(path)
    except Exception:                                  # noqa: BLE001 - no picture to offer
        return None
    if not file.images:
        file.close()
        return None
    image = file.images[0]
    image.name = "macro"
    # its resolution tags are a display default (96 dpi -> 264.6 um), not a pixel size
    image._metadata = Metadata.unknown(1)
    image.index = {"scene": position}
    image._position = position
    image._file = file
    return image


def _open_bare_ets(path: str) -> _VsiFile:
    """An ETS without its ``.vsi`` (docs/COMPOSITE.md: a part read alone)."""
    ets = _ets.EtsFile(path)
    try:
        if any(ets.top[k] > 0 for k in range(2, 2 + ets.extra_dims)):
            raise UnsupportedFormatError(
                f"{path!r}: an ETS file with {ets.extra_dims} dimensions besides x / y: only "
                "its .vsi says which is z, t or channel; open the .vsi")
        size = ets.size_from_tiles()
        image = _pyramid(0, os.path.splitext(os.path.basename(path))[0], ets, size, {},
                         {"t": 1, "c": 1, "z": 1}, None, 0)
    except BaseException:
        ets.close()
        raise
    return _VsiFile(path, [image], [ets], "ets")
