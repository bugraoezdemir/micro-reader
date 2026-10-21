"""Legacy ND2 (Nikon NIS-Elements, format versions 1 and 2), numpy only.

A JPEG 2000 box file (it starts with the JP2 signature box): a box map at
the end -- signature ``LABORATORY IMAGING ND BOX MAP 00`` and its offset --
lists every box by type; ``LUNK`` boxes are the frames' JPEG 2000
codestreams, one per channel per frame (an RGB channel is a 3-component
codestream); XML boxes in Nikon's old dialect describe the image (``VIMD``
per frame, ``AIM1`` / ``AIMD`` the experiment loops, ``TINF`` the text
description, ``ARTT`` the significant bits); ``jp2h`` / ``ihdr`` gives
height, width and bits per component.

Frames run over the experiment loops, outermost first (time, XY position,
Z), loops of one element left out; loop counts follow the text description
where it names them (some files store more frames than the loops hold).
Images are the XY positions (index ``series``); axes ``tczyx`` plus ``s`` for RGB, as the modern
ND2 backend has them.  Codestreams are decoded whole (micro-reader's JPEG
2000 decoder, on the decode pool) and kept in a bounded cache.

Structure ported, reduced, from nd2 (https://github.com/tlambert03/nd2),
``_readers/_legacy/legacy_reader.py`` and ``_parse/_legacy_xml.py``:
Copyright (c) 2021 Talley Lambert, BSD-3-Clause (full notice in
``_nd2_structure.py``).
"""
from __future__ import annotations

import builtins
import itertools
import os
import re
import struct
import threading
import xml.etree.ElementTree as ET
from collections import defaultdict
from functools import partial

import numpy as np

from .. import _codecs, _memory, _pool
from .._base import File, Image
from .._errors import CorruptFileError, UnsupportedFormatError
from .._io import DecodeCache, ThreadHandles

JP2_SIGNATURE = b"\x00\x00\x00\x0cjP  \r\n\x87\n"
MAP_SIGNATURE = b"LABORATORY IMAGING ND BOX MAP 00"
BOX = struct.Struct(">I4s")
MAP_ENTRY = struct.Struct("<4s4sI")        # box type, Nikon type, offset
MAP_STRIDE = 16                              # map entries are 16 bytes apart (4 unused)
IHDR = struct.Struct(">iihBB")                  # height, width, components, bits - 1, codec
DECODE_CACHE_BYTES = 128 << 20
#: legacy experiment loop type -> axis
LOOP_AXES = {8: "T", 2: "P", 4: "Z"}


def is_legacy(head: bytes) -> bool:
    return head[:12] == JP2_SIGNATURE


# -- Nikon's old XML dialect (nd2 ``_parse/_legacy_xml.py``) ----------------------------

_LOWER = re.compile("^[a-z_]+")
_TYPES = {"bool": bool, "lx_uint32": int, "lx_uint64": int, "lx_int32": int,
          "lx_int64": int, "double": float, "CLxStringW": str,
          "CLxByteArray": partial(bytes, encoding="utf-8"), "unknown": str, None: str}


def _elem2dict(node):
    if "value" in node.attrib:
        convert = _TYPES.get(node.attrib.get("runtype"), str)
        try:
            return convert(node.attrib["value"])
        except ValueError:
            return node.attrib["value"]
    result = {k: v for k, v in node.attrib.items() if k not in ("runtype", "version")}
    for element in node:
        key = element.tag.split("}")[1] if "}" in element.tag else element.tag
        key = _LOWER.sub("", key) or key
        value = element.text if element.text and element.text.strip() else _elem2dict(element)
        if key in result:
            result[key] = (result[key] + [value]) if isinstance(result[key], list) \
                else [result[key], value]
        else:
            result[key] = value
    if "variant" in result and not isinstance(result["variant"], list):
        result = result["variant"]
    if isinstance(result, dict) and set(result) == {"no_name"} \
            and not isinstance(result["no_name"], list):
        result = result["no_name"]
    return result


def _parse_xml(raw: bytes) -> dict:
    try:
        return _elem2dict(ET.fromstring(raw.split(b"?>", 1)[-1]))
    except ET.ParseError as exc:
        raise CorruptFileError(f"legacy ND2 metadata: {exc}") from exc


_DIMSIZE = re.compile(r"(\w+)'?\((\d+)\)")


def _described_sizes(description: str | None) -> dict:
    """Axis sizes named in the text description ('Dimensions: T(5) x XY(3)
    x λ(2)'), XY as 'P' and λ as 'C'."""
    if not description:
        return {}
    match = re.search(r"Dimensions:\s?([^\r]+)\r?\n", description)
    if not match:
        return {}
    dims = match.group(1).replace("λ", "C").replace("XY", "P")
    return {k: int(v) for k, v in _DIMSIZE.findall(dims)}


# -- the file -----------------------------------------------------------------------------

class LegacyNd2Parser:
    """What the images of one legacy file share: boxes, layout, loops."""

    def __init__(self, path: str):
        self.path = path
        with builtins.open(path, "rb") as fh:
            if not is_legacy(fh.read(12)):
                raise UnsupportedFormatError(f"{path!r}: not a legacy ND2 file")
            self.boxes = self._box_map(fh)
            self._fh = fh
            try:
                self._describe()
            finally:
                self._fh = None
        self.handles = ThreadHandles(path)

    def _box_map(self, fh) -> dict:
        fh.seek(0, 2)
        if fh.tell() < 40:
            raise CorruptFileError(f"{self.path!r}: truncated legacy ND2 file")
        fh.seek(-40, 2)
        signature, map_start = struct.unpack("<32sQ", fh.read(40))
        if signature != MAP_SIGNATURE:
            raise CorruptFileError(f"{self.path!r}: no legacy ND2 box map (truncated file?)")
        fh.seek(-map_start, 2)
        count = int.from_bytes(fh.read(4), "big")
        data = fh.read()
        if len(data) < (count - 1) * MAP_STRIDE + MAP_ENTRY.size:
            raise CorruptFileError(f"{self.path!r}: truncated legacy ND2 box map")
        boxes = defaultdict(list)
        for i in range(count):
            box, kind, offset = MAP_ENTRY.unpack_from(data, i * MAP_STRIDE)
            boxes[box if box in (b"jP  ", b"ftyp", b"jp2h") else kind].append(offset)
        return dict(boxes)

    def _box(self, kind: bytes, index: int = 0, fh=None) -> bytes:
        """Contents of the *index*-th box of *kind* (without its header)."""
        offsets = self.boxes.get(kind, [])
        if index >= len(offsets):
            raise KeyError(kind)
        fh = fh or self._fh or self.handles.get()
        fh.seek(offsets[index])
        length, _ = BOX.unpack(fh.read(BOX.size))
        data = fh.read(length - BOX.size)
        if len(data) != length - BOX.size:
            raise CorruptFileError(f"{self.path!r}: box {kind!r} ends early")
        return data

    def _xml(self, kind: bytes, index: int = 0) -> dict:
        try:
            return _parse_xml(self._box(kind, index))
        except KeyError:
            return {}

    def _describe(self) -> None:
        # frame size and depth: the JP2 header box
        if not self.boxes.get(b"jp2h"):
            raise CorruptFileError(f"{self.path!r}: no JP2 header box")
        fh = self._fh
        fh.seek(self.boxes[b"jp2h"][0] + BOX.size + 4)
        if fh.read(4) != b"ihdr":
            raise CorruptFileError(f"{self.path!r}: no image header in the JP2 header")
        self.height, self.width, _nc, bits_minus_one, _codec = IHDR.unpack(fh.read(IHDR.size))
        bits = bits_minus_one + 1
        if bits > 16:
            raise UnsupportedFormatError(f"{self.path!r}: legacy ND2 with {bits}-bit samples")
        self.dtype = np.dtype(np.uint8 if bits <= 8 else np.uint16)
        # channels and components: frame 0's picture planes
        try:
            planes = self._xml(b"VIMD", 0)["PicturePlanes"]
            self.channels, components = int(planes["Count"]), int(planes["CompCount"])
        except (KeyError, TypeError, ValueError):
            self.channels, components = 1, 1
        if components % self.channels:
            raise UnsupportedFormatError(f"{self.path!r}: {components} components "
                                         f"for {self.channels} channels")
        self.per_channel = components // self.channels
        # RGB channels keep their samples (s); otherwise every component is a channel
        self.samples = 3 if self.per_channel == 3 else 1
        self.c = self.channels if self.samples == 3 else components
        self.loops = self._loops()
        self.stored_frames = len(self.boxes.get(b"LUNK", [])) // max(1, self.channels)

    def _loops(self) -> list:
        """[(axis, count)] outermost first, loops of one element left out."""
        raw = self._xml(b"AIM1") or self._xml(b"AIMD")
        raw.pop("UnknownData", None)
        while len(raw) == 1:
            _, raw = raw.popitem()
            if isinstance(raw, dict):
                raw.pop("UnknownData", None)
        description = ""
        for item in self._xml(b"TINF").get("TextInfoItem", []) or []:
            if isinstance(item, dict) and str(item.get("Text", "")).startswith("Metadata:"):
                description = item["Text"]
        described = _described_sizes(description)
        levels = []
        if "LoopNo00" in raw:                                # old style: numbered loops
            levels = [v for k, v in raw.items() if k != "Version"]
        else:
            level = raw
            while level:
                if len(level) == 1:
                    level = next(iter(level.values()))
                    if level and isinstance(level, list):
                        level = level[0]
                if not isinstance(level, dict):
                    break
                levels.append(level)
                level = level.get("NextLevelEx")
        loops = []
        for level in levels:
            kind = level.get("Type")
            if kind == 6 or kind is None:                    # channels: in the frame
                continue
            if kind not in LOOP_AXES:
                raise UnsupportedFormatError(f"{self.path!r}: legacy ND2 loop type {kind}")
            params = level.get("LoopPars", {})
            axis = LOOP_AXES[kind]
            if axis == "T":
                periods = list((params.get("Period") or {}).values())
                count = described.get("T")
                if count is None or not any(p.get("Count") == count for p in periods):
                    count = periods[0]["Count"] if periods else params.get("Count", 1)
            elif axis == "P":
                count = described.get("P") or params.get("Count", 1)
            else:
                count = params.get("Count", 1)
            if int(count) > 1:
                loops.append((axis, int(count)))
        axes = [a for a, _ in loops]
        if len(set(axes)) != len(axes):
            raise UnsupportedFormatError(f"{self.path!r}: repeated experiment loops {axes}")
        return loops

    def size(self, axis: str) -> int:
        return next((n for a, n in self.loops if a == axis), 1)

    def frame_number(self, t: int, p: int, z: int) -> int:
        index = {"T": t, "P": p, "Z": z}
        number = 0
        for axis, count in self.loops:
            number = number * count + index[axis]
        return number


class _LegacyNd2Image(Image):
    """One XY position: axes 'tczyx' (+ 's' for RGB)."""

    def __init__(self, nd2: LegacyNd2Parser, position: int):
        self._f = nd2
        self._p = position
        self._rgb = nd2.samples > 1
        shape = (nd2.size("T"), nd2.c, nd2.size("Z"), nd2.height, nd2.width)
        axes = "tczyx"
        if self._rgb:
            shape, axes = shape + (nd2.samples,), axes + "s"
        self.decode_cache = DecodeCache(DECODE_CACHE_BYTES)
        self._closed = False
        super().__init__(position, axes, shape, nd2.dtype,
                         name=f"{os.path.basename(nd2.path)} position {position}",
                         read_unit=(1, 1, 1, nd2.height, nd2.width)
                         + ((nd2.samples,) if self._rgb else ()))

    def _read_block(self, block):
        if self._closed:
            raise ValueError("this ND2 image has been closed: its file was closed "
                             "(File.close() or the end of a with block)")
        f = self._f
        ts, cs, zs, ys, xs = block[:5]
        ss = block[5] if self._rgb else slice(0, 1)
        out = np.empty(tuple(s.stop - s.start for s in block), dtype=self.dtype)
        view = out if self._rgb else out[..., None]              # (t, c, z, y, x, s)
        tasks = []
        for t, z in itertools.product(range(ts.start, ts.stop), range(zs.start, zs.stop)):
            number = f.frame_number(t, self._p, z)
            for c in range(cs.start, cs.stop):
                # a channel is a codestream (RGB: its 3 samples), or one component of it
                stream, component = (c, None) if f.per_channel in (1, 3) \
                    else divmod(c, f.per_channel)
                dst = view[t - ts.start, c - cs.start, z - zs.start]
                tasks.append(lambda a=(number, stream, component, ys, xs, ss, dst):
                             self._plane_into(*a))
        return out, _pool.run(tasks)

    def _read_metadata(self):
        from ._nd2_metadata import legacy_metadata
        frame = self._f._xml(b"VIMD", self._f.frame_number(0, self._p, 0)) or             self._f._xml(b"VIMD", 0)
        return legacy_metadata(frame, dict(zip(self.axes, self.shape)))

    def _plane_into(self, number, stream, component, ys, xs, ss, dst) -> int:
        f = self._f
        key = (number, stream)
        plane, read = self.decode_cache.get(key), 0
        if plane is None:
            index = number * f.channels + stream
            if number >= f.stored_frames or index >= len(f.boxes.get(b"LUNK", [])):
                dst[...] = 0                                    # never acquired
                return 0
            nbytes = f.height * f.width * f.per_channel * f.dtype.itemsize
            if _memory.should_stream(nbytes):              # too large to cache: the region alone
                return self._area_into(f._box(b"LUNK", index, f.handles.get()),
                                       component, ys, xs, ss, dst)
            data = f._box(b"LUNK", index, f.handles.get())
            plane = np.empty((f.height, f.width, f.per_channel), f.dtype)
            with _memory.reserve(nbytes, f"{f.path!r}: a {f.height} x {f.width} JPEG 2000 frame"):
                got = _codecs.jpeg2k_decode_into(data, plane)
            if (got[1], got[0], got[2]) != plane.shape:
                raise CorruptFileError(
                    f"{f.path!r}: frame {number} decodes to {got[1]} x {got[0]} x {got[2]}, "
                    f"the header says {plane.shape}")
            self.decode_cache.put(key, plane)
            read = plane.size
        if component is not None:
            dst[..., 0] = plane[ys, xs, component]
        else:
            dst[...] = plane[ys, xs][..., ss] if self._rgb else plane[ys, xs, :1]
        return read

    def _area_into(self, data, component, ys, xs, ss, dst) -> int:
        """Rows *ys* and columns *xs* of a frame too large to cache, decoded
        alone (OpenJPEG decodes only the code-blocks covering them)."""
        f = self._f
        row_bytes = (xs.stop - xs.start) * f.per_channel * f.dtype.itemsize
        per_band = max(1, min(8 << 20, _memory.memory_limit() // 4) // row_bytes)
        for a in range(ys.start, ys.stop, per_band):          # bands within the budget
            b = min(ys.stop, a + per_band)
            area = np.empty((b - a, xs.stop - xs.start, f.per_channel), f.dtype)
            with _memory.reserve(area.nbytes, f"{f.path!r}: part of a JPEG 2000 frame"):
                got = _codecs.jpeg2k_decode_area_into(data, area, xs.start, a, xs.stop, b)
                if (got[1], got[0], got[2]) != area.shape:
                    raise CorruptFileError(f"{f.path!r}: JPEG 2000 area decodes to "
                                           f"{got[1]} x {got[0]} x {got[2]}, expected {area.shape}")
                band = dst[a - ys.start:b - ys.start]
                if component is not None:
                    band[..., 0] = area[..., component]
                else:
                    band[...] = area[..., ss] if self._rgb else area[..., :1]
        return (ys.stop - ys.start) * (xs.stop - xs.start) * f.per_channel

    def close(self) -> None:
        self._closed = True
        self.decode_cache.clear()


class _LegacyNd2File(File):
    backend = "nd2"

    def __init__(self, path: str):
        self._nd2 = nd2 = LegacyNd2Parser(path)
        self.format = "nd2-legacy"           # format versions 1 / 2 (JPEG 2000)
        super().__init__(path, [_LegacyNd2Image(nd2, p) for p in range(nd2.size("P"))])

    def close(self) -> None:
        super().close()
        self._nd2.handles.close()
