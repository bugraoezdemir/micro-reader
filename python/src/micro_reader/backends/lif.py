"""Leica image files: LIF, LOF, LIFEXT, XLIF, XLEF and XLCF, own reader.

The LIF container and its XML image descriptions follow Leica's description
of the format.  The structure of the other flavours follows liffile
(Christoph Gohlke, BSD-3-Clause; notice below).  No code from readlif
(GPL-3) or Bio-Formats (GPL-2).

- **LIF**: a header block -- int32 0x70, int32 length, byte 0x2A, int32
  character count, the XML description in UTF-16 -- then memory blocks:
  int32 0x70, int32 length, byte 0x2A, the data size (int32 in version 1,
  int64 in version 2), byte 0x2A, int32 character count, the block's ID in
  UTF-16, then the data.
- **LOF** (one image): a header block whose text is ``LMS_Object_File``,
  two version fields (byte 0x2A, int32 each), the single memory block (byte
  0x2A, int64 size, data), then the XML header block at the end.
- **LIFEXT** (extra image data of a LIF): a LIF-like file whose XML root is
  ``LMSDataContainerEnhancedHeader``; images sit under ``ChildrenOf``
  elements and are named after that element's ``MemoryBlockID``.
- **XLIF** (LAS X, one image): an XML text file; its memory block is
  assembled from frame files -- ``.lof``, ``.tif`` (also OME / Aivia,
  multi-page), ``.jpg``, ``.png``, ``.bmp`` -- each placed at its byte
  ``Offset`` in the block.
- **XLEF** (experiment) and **XLCF** (collection): XML text files that
  reference other files (``Children/Reference File=...``, relative paths);
  their images are those of the referenced files, recursively.  XLCF image
  names get the collection's name in front, XLEF ones do not (as liffile).

An ``Element`` with ``Data/Image`` is an image: its ``ImageDescription``
lists the channels (data type, bit resolution, byte offset ``BytesInc``)
and the dimensions (``DimID``, ``NumberOfElements``, byte stride
``BytesInc``), which place every sample in the memory block.  Pixels are
little-endian.

Images (docs/AXES.md): the Leica images in document order (nested analysis
results included), as readlif and Bio-Formats list them, are the scenes;
each of their tiles, rotations and emission / excitation wavelengths
(lambda scans) is its own image, indexed by ``{"scene": n, "tile": m,
"rotation": r, "emission": l, "excitation": e}`` (an index only where some
image of the file has more than one; the wavelength of an emission /
excitation index in the image's ``metadata.index_values``).  Axes: ``t c z
y x``, and ``s`` for interleaved RGB -- ordered R, G, B by the
ChannelTags for raw blocks (LAS stores B, G, R), as decoded for image-file
frames.  Every
request reads only the rows it needs; a row range is one contiguous read
unless the bytes it would skip cost more than separate reads.

Data type 1 is floating point: 32-bit float, and 16-bit **half** float
(LAS X FLIM results -- lifetimes, phasor coordinates).  readlif and
Bio-Formats return the half floats' raw bits as uint16; micro-reader (like
liffile) returns float16 values.

liffile notice (structure of LOF / LIFEXT / XLIF / XLEF / XLCF):

    Copyright (c) 2023-2026, Christoph Gohlke.  All rights reserved.
    Redistribution and use in source and binary forms, with or without
    modification, are permitted under the BSD-3-Clause licence: source
    redistributions must retain this notice; binary redistributions must
    reproduce it in the documentation; the copyright holder's name may not
    be used to endorse derived products without permission.  THIS SOFTWARE
    IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND.
"""
from __future__ import annotations

import builtins
import itertools
import os
import warnings
import re
import struct
import threading
import xml.etree.ElementTree as ET
from urllib.parse import unquote

import numpy as np

from .. import _codecs, _pool
from .._base import File, Image
from .._errors import CorruptFileError, UnsupportedFormatError
from .._io import DecodeCache, ThreadHandles
from . import raster

NAME = "lif"
MAGIC, STAR = 0x70, 0x2A
LOF_TEXT = "LMS_Object_File"
EXT_ROOT = "LMSDataContainerEnhancedHeader"
RAW_TASK_BYTES = 8 << 20          # planes are read in row bands of about this size
SEPARATE_READ_BYTES = 32 << 10    # bytes worth skipping rather than issuing one more read
FRAME_CACHE_BYTES = 128 << 20     # decoded XLIF frame files kept

#: DimID -> axis; X and Y are 1 and 2
DIM_AXES = {3: "z", 4: "t"}
#: DimID -> image index
DIM_KEYS = {10: "tile", 6: "rotation", 5: "emission", 9: "excitation"}
DIM_NAMES = {1: "X", 2: "Y", 3: "Z", 4: "T", 5: "lambda", 6: "rotation", 7: "XT slice",
             8: "T slice", 9: "excitation wavelength", 10: "tile", 11: "loop"}
#: axis order of an image (before y, x and s)
PLANE_AXES = "tcz"
#: image indices, in index order
OTHER_INDICES = ("tile", "rotation", "emission", "excitation")
#: first bytes of an XML text file -> its encoding
XML_CODECS = ((b"\xef\xbb\xbf", "utf-8-sig"), (b"\xff\xfe", "utf-16"), (b"\xfe\xff", "utf-16"),
              (b"<\x00", "utf-16-le"), (b"\x00<", "utf-16-be"), (b"<", "utf-8"))
IMAGE_FRAMES = (".jpg", ".jpeg", ".png", ".bmp")
TIFF_FRAMES = (".tif", ".tiff")


def _xml_codec(head: bytes):
    for prefix, codec in XML_CODECS:
        if head.startswith(prefix):
            return codec
    return None


def sniff(head: bytes, path: str) -> bool:
    if len(head) >= 13:
        magic, _length, star = struct.unpack_from("<iiB", head)
        if magic == MAGIC and star == STAR:
            return True                                   # LIF, LOF, LIFEXT
    codec = _xml_codec(head)                              # XLIF, XLEF, XLCF
    return codec is not None and "LMSDataContainerHeader" in head.decode(codec, errors="ignore")


def _dtype(data_type: int, resolution: int, where: str) -> np.dtype:
    """Sample type from a channel's DataType (0 integer, 1 float) and bit
    Resolution (12-bit data sits in 16-bit samples)."""
    if data_type == 0:
        for bits, code in ((8, "u1"), (16, "<u2"), (32, "<u4")):
            if resolution <= bits:
                return np.dtype(code)
    elif data_type == 1:
        code = {16: "<f2", 32: "<f4", 64: "<f8"}.get(resolution)
        if code:
            return np.dtype(code)
    raise UnsupportedFormatError(f"{where}: LIF data type {data_type} at {resolution} bits")


def _memory_blocks(fh, version: int, path: str) -> dict:
    """ID -> (data offset, size) of every memory block, walking the headers."""
    blocks = {}
    size_fmt = "<q" if version >= 2 else "<i"
    size_len = struct.calcsize(size_fmt)
    end = os.fstat(fh.fileno()).st_size
    while True:
        at = fh.tell()
        head = fh.read(9)
        if not head:
            return blocks
        if len(head) < 9:
            raise CorruptFileError(f"{path!r}: truncated memory block header at byte {at}")
        magic, _length, star = struct.unpack("<iiB", head)
        if magic != MAGIC or star != STAR:
            raise CorruptFileError(f"{path!r}: no memory block at byte {at}")
        size, = struct.unpack(size_fmt, fh.read(size_len))
        star, nchars = struct.unpack("<Bi", fh.read(5))
        if star != STAR or not 0 <= nchars < 1 << 16:
            raise CorruptFileError(f"{path!r}: bad memory block header at byte {at}")
        name = fh.read(2 * nchars).decode("utf-16-le")
        offset = fh.tell()
        if offset + size > end:
            raise CorruptFileError(f"{path!r}: memory block {name!r} ends beyond the file "
                                   "(truncated file?)")
        blocks[name] = (offset, size)
        fh.seek(size, 1)


def _resolve(dirname: str, ref: str, where: str) -> str:
    """A file referenced from an XLIF / XLEF / XLCF: relative, URL-quoted,
    possibly with backslashes and another letter case (written on Windows)."""
    rel = os.path.normpath(unquote(ref).replace("\\", "/"))
    path = os.path.join(dirname, rel)
    if os.path.exists(path):
        return path
    current = dirname
    for part in rel.split(os.sep):
        try:
            match = [e for e in os.listdir(current or ".") if e.lower() == part.lower()]
        except OSError:
            match = []
        if not match:
            raise CorruptFileError(f"{where}: referenced file {ref!r} not found")
        current = os.path.join(current, match[0])
    return current


# -- memory blocks ------------------------------------------------------------------

class _FileBlock:
    """A memory block stored in a file (LIF, LOF, LIFEXT): positioned reads."""

    by_tags = True                    # raw samples: colour order from the ChannelTags

    def __init__(self, handles: ThreadHandles, offset: int, size: int):
        self.handles, self.offset, self.size = handles, offset, size

    def readinto(self, at: int, buf: np.ndarray) -> None:
        fh = self.handles.get()
        fh.seek(self.offset + at)
        if fh.readinto(memoryview(buf)) != buf.nbytes:
            raise CorruptFileError(f"{self.handles.path!r}: pixel data ends early")


class _FrameBlock:
    """An XLIF memory block assembled from frame files, each at its byte
    offset: LOF frames are read in place, TIFF frames page by page
    (micro-reader's TIFF backend), JPEG / PNG / BMP frames decoded whole
    (micro-reader's codecs) and cached.  A frame decoded as RGB where the
    block holds one sample per pixel contributes its first sample (as
    liffile does)."""

    def __init__(self, image: "_LifFile", frames: list, size: int, where: str):
        self.image, self.size, self.where = image, size, where
        self.frames = sorted(frames, key=lambda f: f[1])       # (path, offset, size)
        # raw LOF frames store colour B, G, R like LIF; decoded image files R, G, B
        self.by_tags = all(p.lower().endswith(".lof") for p, _, _ in frames)
        self.cache = DecodeCache(FRAME_CACHE_BYTES)
        self._lock = threading.Lock()
        self._sources: dict = {}

    def readinto(self, at: int, buf: np.ndarray) -> None:
        end = at + buf.nbytes
        covered = 0
        for path, offset, size in self.frames:
            lo, hi = max(at, offset), min(end, offset + size)
            if lo < hi:
                self._copy(path, size, lo - offset, buf[lo - at:hi - at])
                covered += hi - lo
        if covered != buf.nbytes:                 # bytes no frame provides: background
            mask = np.ones(buf.nbytes, bool)
            for _, offset, size in self.frames:
                lo, hi = max(at, offset), min(end, offset + size)
                if lo < hi:
                    mask[lo - at:hi - at] = False
            buf[mask] = 0

    def _source(self, path: str):
        """How to get a frame file's bytes, set up once per file."""
        with self._lock:
            if path not in self._sources:
                self._sources[path] = self.image._frame_source(path, self.where)
            return self._sources[path]

    def _copy(self, path: str, frame_size: int, local: int, piece: np.ndarray) -> None:
        kind, source = self._source(path)
        if kind == "lof":
            source.readinto(local, piece)
            return
        if kind == "tiff":
            scene, lead, page_bytes = source
            pages = max(1, int(np.prod(lead, dtype=np.int64)))
            total = pages * page_bytes
            factor = 3 if total == 3 * frame_size and scene.shape[-1] == 3 else 1
            if total != frame_size * factor:
                raise CorruptFileError(f"{self.where}: frame {path!r} holds {total} bytes, "
                                       f"the XLIF says {frame_size}")
            # only the page rows the piece covers, read through the TIFF
            # backend (exact rectangles; compressed strips / tiles cached there)
            per_page = page_bytes // factor
            row_bytes = per_page // scene.shape[len(lead)]
            done = 0
            while done < piece.nbytes:
                k, into = divmod(local + done, per_page)
                n = min(piece.nbytes - done, per_page - into)
                r0, r1 = into // row_bytes, (into + n - 1) // row_bytes + 1
                index = tuple(int(i) for i in np.unravel_index(k, lead)) if lead else ()
                rows = scene[index + (slice(r0, r1),)]
                if factor == 3:
                    rows = rows[..., 0]
                raw = np.ascontiguousarray(rows.astype(rows.dtype.newbyteorder("<"),
                                                       copy=False)).view(np.uint8).reshape(-1)
                start = into - r0 * row_bytes
                piece[done:done + n] = raw[start:start + n]
                done += n
            return
        data = self._page(path, source, None, 0, None, frame_size)
        piece[...] = data[local:local + piece.nbytes]

    def _page(self, path, source, lead, k, factor, frame_size=None) -> np.ndarray:
        """Bytes of page *k* of a TIFF frame, or of a whole image frame."""
        key = (path, k)
        data = self.cache.get(key)
        if data is not None:
            return data
        if lead is not None:                                   # TIFF page
            array = source[np.unravel_index(k, lead)] if lead else source[...]
            if factor == 3:
                array = array[..., 0]
        else:                                                  # JPEG / PNG / BMP
            array = source()
            if array.nbytes != frame_size:
                if array.ndim == 3 and array.shape[2] == 3 and array.nbytes == 3 * frame_size:
                    array = array[..., 0]
                else:
                    raise CorruptFileError(f"{self.where}: frame {path!r} decodes to "
                                           f"{array.nbytes} bytes, the XLIF says {frame_size}")
        data = np.ascontiguousarray(array).astype(array.dtype.newbyteorder("<")).view(np.uint8)
        data = data.reshape(-1)
        self.cache.put(key, data)
        return data


# -- image files in XLIF frames ----------------------------------------------------

def _image_reader(path: str, where: str):
    """A function decoding a JPEG / PNG / BMP frame file into an array
    (``raster``'s decoders; a colour palette is applied: frames are colour)."""

    def read():
        with builtins.open(path, "rb") as fh:
            data = fh.read()
        try:
            info = raster.info_of(data)
            pixels = raster.decode(data, info)
            return pixels if info.palette is None else raster.apply_palette(pixels, info.palette)
        except (CorruptFileError, UnsupportedFormatError) as exc:
            raise type(exc)(f"{where}: frame {path!r}: {exc}") from exc
    return read


# -- images -------------------------------------------------------------------------

class _Layout:
    """Where each sample of one image lies in its memory block."""

    def __init__(self, element, where: str, by_tags: bool = True):
        desc = element.find("Data/Image/ImageDescription")
        channels = desc.findall("Channels/ChannelDescription")
        dims = desc.findall("Dimensions/DimensionDescription")
        if not channels or not dims:
            raise UnsupportedFormatError(f"{where}: image without channels or dimensions")
        # samples that do not start on a byte (bit increments) are not read
        # as if they did: refused by name (liffile does not read them either)
        packed = [e.tag for e in channels + dims if int(e.get("BitInc", 0) or 0)]
        if packed:
            raise UnsupportedFormatError(
                f"{where}: bit-packed samples (non-zero BitInc in {sorted(set(packed))})")
        types = {(int(c.get("DataType", 0)), int(c.get("Resolution", 8))) for c in channels}
        if len(types) != 1:
            raise UnsupportedFormatError(f"{where}: channels of different types {sorted(types)}")
        self.dtype = _dtype(*types.pop(), where)
        item = self.dtype.itemsize

        self.size, self.stride = {}, {}                 # axis -> elements, bytes
        for d in dims:
            dim_id, n = int(d.get("DimID")), int(d.get("NumberOfElements"))
            inc = int(d.get("BytesInc"))
            axis = {1: "x", 2: "y"}.get(dim_id) or DIM_AXES.get(dim_id) or DIM_KEYS.get(dim_id)
            if axis is None:
                if n > 1:
                    raise UnsupportedFormatError(
                        f"{where}: LIF dimension {DIM_NAMES.get(dim_id, dim_id)} "
                        f"(DimID {dim_id}) with {n} elements")
                continue
            self.size[axis], self.stride[axis] = n, inc
        if "x" not in self.size or "y" not in self.size:
            raise UnsupportedFormatError(f"{where}: image without X and Y dimensions")

        offsets = [int(c.get("BytesInc", 0)) for c in channels]
        tags = [int(c.get("ChannelTag", 0)) for c in channels]
        pixel = self.stride["x"]
        # interleaved colour: 3 (4) samples inside one pixel, tagged R, G, B
        self.samples = 1
        if len(channels) in (3, 4) and max(offsets) + item <= pixel and sorted(tags[:3]) == [1, 2, 3]:
            if by_tags:                                              # raw: R, G, B by tag
                order = sorted(range(3), key=lambda k: tags[k]) + list(range(3, len(channels)))
            else:                                                    # decoded frames: as stored
                order = sorted(range(len(channels)), key=lambda k: offsets[k])
            self.sample_offsets = [offsets[k] for k in order]
            self.samples = len(channels)
            self.channel_offsets = [0]
        else:
            if pixel < item:
                raise CorruptFileError(f"{where}: pixel stride {pixel} < sample size {item}")
            self.sample_offsets = [0]
            self.channel_offsets = offsets
        self.size["c"] = len(self.channel_offsets)
        self.axes = list(PLANE_AXES)

    def extent(self) -> int:
        """Bytes from the block start to the end of the last sample."""
        last = max(self.channel_offsets) + max(self.sample_offsets) + self.dtype.itemsize
        return last + sum((self.size[a] - 1) * self.stride[a]
                          for a in self.stride if self.size.get(a, 1) > 1)


class _LifImage(Image):
    """One tile / rotation of one Leica image."""

    def __init__(self, index: int, element, name: str, block, layout: "_Layout", key: dict):
        self._layout = layout
        self._block = block
        self._element = element
        # the tile / rotation of this image: a fixed offset into the block
        self._offset = sum(key.get(level, 0) * layout.stride.get(level, 0)
                           for level in OTHER_INDICES)
        shape = [layout.size.get(a, 1) for a in layout.axes] + [layout.size["y"], layout.size["x"]]
        letters = "".join(layout.axes) + "yx"
        if layout.samples > 1:
            shape.append(layout.samples)
            letters += "s"
        self._closed = False
        super().__init__(index, letters, shape, layout.dtype, name=name, index=key)

    @property
    def tile_position(self):
        """Deprecated: use ``image.metadata.position`` (the tile's position,
        as for every format).  This tile's raw (FieldX, FieldY, PosX, PosY)
        from the TileScanInfo attachment; None for an image without tiles."""
        warnings.warn("LifImage.tile_position is deprecated: use image.metadata.position",
                      DeprecationWarning, stacklevel=2)
        if self._layout.size.get("tile", 1) <= 1:
            return None
        tiles = self._element.findall("Data/Image/Attachment[@Name='TileScanInfo']/Tile")
        m = self.index.get("tile", 0)
        if m >= len(tiles):
            return None
        t = tiles[m]
        return (int(t.get("FieldX", 0)), int(t.get("FieldY", 0)),
                float(t.get("PosX", 0)), float(t.get("PosY", 0)))

    def _read_metadata(self):
        from ._lif_metadata import lif_metadata
        return lif_metadata(self._element, dict(zip(self.axes, self.shape)), self.index,
                            tiled=self._layout.size.get("tile", 1) > 1)

    def _read_block(self, block):
        if self._closed:
            raise ValueError("this LIF image has been closed: its file was closed "
                             "(File.close() or the end of a with block)")
        lay = self._layout
        n = len(lay.axes)
        plane_spans, ys, xs = block[:n], block[n], block[n + 1]
        ss = block[n + 2] if lay.samples > 1 else slice(0, 1)
        out = np.empty(tuple(s.stop - s.start for s in block), dtype=self.dtype)
        view = out if lay.samples > 1 else out[..., None]
        per_task = max(1, RAW_TASK_BYTES // max(1, lay.stride["y"]))
        tasks = []
        for key in itertools.product(*(range(s.start, s.stop) for s in plane_spans)):
            base = self._offset
            for axis, k in zip(lay.axes, key):
                base += lay.channel_offsets[k] if axis == "c" else k * lay.stride.get(axis, 0)
            dst = view[tuple(k - s.start for k, s in zip(key, plane_spans))]
            for y0 in range(ys.start, ys.stop, per_task):
                y1 = min(ys.stop, y0 + per_task)
                tasks.append(lambda a=(base, y0, y1, xs, ss, dst[y0 - ys.start:y1 - ys.start]):
                             self._read_rows(*a))
        return out, _pool.run(tasks)

    def _read_rows(self, base, y0, y1, xs, ss, dst) -> int:
        """Rows y0:y1, columns *xs*, samples *ss* of one plane at block
        offset *base* into *dst* (rows, cols, samples).  One read spanning
        the rows when the bytes skipped between them cost less than
        separate reads."""
        lay = self._layout
        sy, sx, item = lay.stride["y"], lay.stride["x"], lay.dtype.itemsize
        rows, width = y1 - y0, xs.stop - xs.start
        lowest = min(lay.sample_offsets)
        first = base + y0 * sy + xs.start * sx + lowest
        row_span = (width - 1) * sx + (max(lay.sample_offsets) - lowest) + item
        skipped = (sy - row_span) * (rows - 1)
        samples = [lay.sample_offsets[k] - lowest for k in range(ss.start, ss.stop)]
        if rows == 1 or skipped <= (rows - 1) * SEPARATE_READ_BYTES:
            span = (rows - 1) * sy + row_span
            buf = np.empty(span, np.uint8)
            self._block.readinto(first, buf)
            for k, off in enumerate(samples):
                dst[..., k] = np.ndarray((rows, width), lay.dtype, buf, off, (sy, sx))
            return span // item
        buf = np.empty(row_span, np.uint8)
        for r in range(rows):
            self._block.readinto(first + r * sy, buf)
            for k, off in enumerate(samples):
                dst[r, :, k] = np.ndarray((width,), lay.dtype, buf, off, (sx,))
        return rows * row_span // item

    def close(self) -> None:
        self._closed = True


class _LifFile(File):
    backend = NAME

    def __init__(self, path: str, tiles: str = "separate"):
        self.path = path
        self._handles: dict = {}            # binary file -> ThreadHandles
        self._tiffs: list = []              # TIFF frame files opened
        self._lofs: dict = {}               # LOF frame file -> its _FileBlock
        self.format = None                  # of the file opened: lif, lof, lifext, xlif, ...
        try:
            entries = self._collect(path, frozenset())
            if not entries:
                raise CorruptFileError(f"{path!r}: no images")
            layouts = []
            for name, element, block, where in entries:
                layout = _Layout(element, where, block.by_tags)
                if layout.extent() > block.size:
                    raise CorruptFileError(f"{where}: pixels need {layout.extent()} bytes, "
                                           f"its memory block holds {block.size}")
                layouts.append(layout)
            # indices: those some image of the file has more than one of
            levels = [level for level in OTHER_INDICES
                      if any(lay.size.get(level, 1) > 1 for lay in layouts)]
            if tiles == "stitched" and "tile" in levels:
                raise UnsupportedFormatError(
                    f"{path!r}: tiles='stitched': micro-reader does not compose Leica tiles "
                    "(each tile is its own image; image.metadata.position gives its place)")
            images = []
            for n, ((name, element, block, _), layout) in enumerate(zip(entries, layouts)):
                for values in itertools.product(*(range(layout.size.get(level, 1))
                                                  for level in levels)):
                    key = {"scene": n, **dict(zip(levels, values))}
                    images.append(_LifImage(len(images), element, name, block, layout, key))
        except BaseException:
            self._close_files()
            raise
        super().__init__(path, images)

    # -- reading the file tree ----------------------------------------------------

    def _collect(self, path: str, seen: frozenset) -> list:
        """(name, element, block, where) of every image in *path* and the
        files it references."""
        real = os.path.normcase(os.path.abspath(path))
        if real in seen:
            raise CorruptFileError(f"{path!r}: files reference each other in a cycle")
        seen = seen | {real}
        with builtins.open(path, "rb") as fh:
            head = fh.read(13)
            if len(head) == 13 and struct.unpack_from("<iiB", head)[::2] == (MAGIC, STAR):
                return self._binary(path, fh)
            fh.seek(0)
            data = fh.read()
        return self._xml(path, data, seen)

    def _flavour(self, name: str) -> None:
        if self.format is None:
            self.format = name.lower()

    def _handles_for(self, path: str) -> ThreadHandles:
        if path not in self._handles:
            self._handles[path] = ThreadHandles(path)
        return self._handles[path]

    def _binary(self, path: str, fh) -> list:
        """LIF, LOF or LIFEXT."""
        fh.seek(0)
        _magic, _length, _star, nchars = struct.unpack("<iiBi", fh.read(13))
        raw = fh.read(2 * nchars)
        if len(raw) != 2 * nchars:
            raise CorruptFileError(f"{path!r}: truncated XML header")
        text = raw.decode("utf-16-le")
        where = os.path.basename(path)
        if text == LOF_TEXT:
            self._flavour("LOF")
            root, block = self._lof(path, fh)
            top = root.find("Element")
            return list(self._images(top, [top.get("Name", "")], block_of=lambda _e: block,
                                     where=where))
        root = self._parse_xml(text, path)
        if root.tag == EXT_ROOT:
            self._flavour("LIFEXT")
            block_of = self._blocks_by_id(_memory_blocks(fh, 2, path), self._handles_for(path))
            out = []
            for group in root.findall("ChildrenOf"):
                for element in group.findall("Element"):
                    out += self._images(element, [group.get("MemoryBlockID", ""),
                                                  element.get("Name", "")],
                                        block_of=block_of, where=where, drop_root=False)
            return out
        self._flavour("LIF")
        block_of = self._blocks_by_id(_memory_blocks(fh, int(root.get("Version", 1)), path),
                                      self._handles_for(path))
        top = root.find("Element")
        return list(self._images(top, [top.get("Name", "")], block_of=block_of, where=where))

    def _lof(self, path: str, fh):
        """The single memory block of a LOF file and its XML (at the end)."""
        head = fh.read(10 + 9)
        if len(head) != 19:
            raise CorruptFileError(f"{path!r}: truncated LOF header")
        star0, _v0, star1, _v1 = struct.unpack_from("<BIBI", head)
        star2, size = struct.unpack_from("<BQ", head, 10)
        if star0 != STAR or star1 != STAR or star2 != STAR:
            raise CorruptFileError(f"{path!r}: corrupt LOF header")
        offset = fh.tell()
        fh.seek(offset + size)
        tail = fh.read(13)
        if len(tail) != 13:
            raise CorruptFileError(f"{path!r}: LOF data ends early (truncated file?)")
        magic, _length, star, nchars = struct.unpack("<iiBi", tail)
        if magic != MAGIC or star != STAR:
            raise CorruptFileError(f"{path!r}: no XML block after the LOF data")
        text = fh.read(2 * nchars).decode("utf-16-le")
        if text.startswith("<Data>"):
            # some (older) LOF files have no LMSDataContainerHeader, just <Data>
            name = os.path.splitext(os.path.basename(path))[0]
            text = (f'<LMSDataContainerHeader Version="2"><Element Name="{name}">{text}'
                    "</Element></LMSDataContainerHeader>")
        root = self._parse_xml(text, path)
        return root, _FileBlock(self._handles_for(path), offset, size)

    def _xml(self, path: str, data: bytes, seen: frozenset) -> list:
        """XLIF, XLEF or XLCF."""
        codec = _xml_codec(data[:4])
        if codec is None:
            raise UnsupportedFormatError(f"{path!r}: not a Leica image file")
        text = data.decode(codec)
        root = self._parse_xml(text, path)
        element = root.find("Element")
        if element is None:
            raise CorruptFileError(f"{path!r}: no Element in the XML")
        dirname, where = os.path.dirname(path), os.path.basename(path)
        if element.find("Data/Image") is not None:                    # XLIF
            self._flavour("XLIF")
            memory = element.find("Memory")
            if memory is None:
                raise CorruptFileError(f"{where}: image without a Memory element")
            frames = [(_resolve(dirname, f.get("File"), where), int(f.get("Offset", 0)),
                       int(f.get("Size", 0))) for f in memory if "File" in f.attrib]
            size = int(memory.get("Size", 0))
            if not frames or not size:
                return []
            block = _FrameBlock(self, frames, size, where)
            return list(self._images(element, [element.get("Name", "")],
                                     block_of=lambda _e: block, where=where))
        kind = "XLCF" if element.find("Data/Collection") is not None else "XLEF"
        self._flavour(kind)
        prefix = element.get("Name", "") if kind == "XLCF" else None
        out = []
        for ref in element.findall("Children/Reference"):
            if ref.get("File") is None:
                continue
            child = _resolve(dirname, ref.get("File"), where)
            for name, el, block, w in self._collect(child, seen):
                out.append((f"{prefix}/{name}" if prefix else name, el, block, w))
        return out

    @staticmethod
    def _parse_xml(text: str, path: str):
        text = re.sub(r"^\s*<\?xml[^>]*\?>", "", text.lstrip("﻿"))
        try:
            return ET.fromstring(text)
        except ET.ParseError as exc:
            raise CorruptFileError(f"{path!r}: XML: {exc}") from exc

    @staticmethod
    def _blocks_by_id(blocks: dict, handles: ThreadHandles):
        def block_of(element):
            memory = element.find("Memory")
            found = blocks.get(memory.get("MemoryBlockID")) if memory is not None else None
            return _FileBlock(handles, *found) if found and found[1] else None
        return block_of

    def _images(self, element, parts, *, block_of, where, drop_root=True):
        """(name, element, block, where) of every image with pixels at and
        below *element*, in document order.  *parts*: the element's name
        path.  Names join it with '/', leaving out the file's root element
        (LIF: the file; LOF / XLIF: the image itself, whose own name is kept
        when it is the image) unless *drop_root* is false (LIFEXT)."""
        if element.find("Data/Image") is not None:
            block = block_of(element)
            name = "/".join(parts[1:] if drop_root and len(parts) > 1 else parts)
            if block is not None:
                yield name, element, block, f"{where!r} image {name!r}"
        for child in element.findall("Children/Element"):
            yield from self._images(child, parts + [child.get("Name", "")], block_of=block_of,
                                    where=where, drop_root=drop_root)

    # -- XLIF frame files -----------------------------------------------------------

    def _frame_source(self, path: str, where: str):
        """(kind, source) for a frame file: ("lof", its _FileBlock),
        ("tiff", (scene, lead shape, page bytes)) or ("image", decoder)."""
        ext = os.path.splitext(path)[1].lower()
        if ext == ".lof":
            if path not in self._lofs:
                with builtins.open(path, "rb") as fh:
                    magic, _length, star, nchars = struct.unpack("<iiBi", fh.read(13))
                    if (magic, star) != (MAGIC, STAR) or \
                            fh.read(2 * nchars).decode("utf-16-le", "replace") != LOF_TEXT:
                        raise CorruptFileError(f"{where}: frame {path!r} is not a LOF file")
                    _root, block = self._lof(path, fh)
                self._lofs[path] = block
            return "lof", self._lofs[path]
        if ext in TIFF_FRAMES:
            from . import tiff as tiff_backend
            img = tiff_backend.open(path)
            self._tiffs.append(img)
            scene = img.images[0]
            lead = tuple(scene._lead_shape)
            page_shape = scene.shape[len(lead):]
            page_bytes = int(np.prod(page_shape, dtype=np.int64)) * scene.dtype.itemsize
            return "tiff", (scene, lead, page_bytes)
        if ext in IMAGE_FRAMES:
            return "image", _image_reader(path, where)
        raise UnsupportedFormatError(f"{where}: frame file type {ext!r} ({path!r})")

    # -- closing --------------------------------------------------------------------

    def _close_files(self) -> None:
        for handles in self._handles.values():
            handles.close()
        self._handles.clear()
        for img in self._tiffs:
            img.close()
        self._tiffs.clear()

    def close(self) -> None:
        super().close()
        self._close_files()


def open(path: str, tiles: str = "separate", resolutions: str = "base") -> _LifFile:  # noqa: A001 - backend protocol
    return _LifFile(path, tiles)
