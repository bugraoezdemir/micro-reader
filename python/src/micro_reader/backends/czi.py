"""Zeiss CZI (ZISRAW), own reader, numpy and stdlib only.

Structure: segments with a 32-byte header (``ZISRAWFILE``, ``ZISRAWDIRECTORY``,
``ZISRAWSUBBLOCK``, ...).  The file header points to the subblock directory:
one entry per subblock with its pixel type, compression, pyramid type, file
position (64-bit: files beyond 4 GB work) and, per dimension, start index,
size and stored size.  A subblock is a tile of one plane: X / Y give its
rectangle in the image's pixel coordinates, the other dimensions its plane
(T, C, Z, views V, phases H, illuminations I, rotations R, blocks B), S its
scene and M its mosaic index.

Images (docs/AXES.md): one per scene, mosaic tile, rotation, view,
illumination, phase and block, indexed by ``{"scene": S, "tile": M,
"rotation": R, "view": V, "illumination": I, "phase": H, "block": B}`` with
the file's own indices; an index is there only where the file has more than
one (tiles: more than one in some scene).  ``tiles="stitched"`` composes each scene's
tiles instead, the way libCZI / pylibCZIrw compose them: the level-0
subblocks of a plane (not pyramid levels) are drawn in mosaic-index order,
later tiles on top, over a black background, inside the bounding box.
Uncompressed subblocks are read exactly: only the rows and columns of each
tile that fall into the requested rectangle.  ``canvas()`` composes all
scenes.

Axes: ``t c z y x``, and ``s`` for RGB pixels -- samples in R, G, B (A)
order (CZI stores them B, G, R).  Not yet: compressed subblocks (zstd: next step, native;
JPEG-XR / JPEG / LZW), multi-part files, complex pixel types.

The binary layout follows czifile (Christoph Gohlke, BSD-3,
https://github.com/cgohlke/czifile) and the published ZISRAW description;
nothing is taken from libCZI or aicspylibczi.
"""
from __future__ import annotations

import builtins
import itertools
import os
import struct
import sys
import warnings
from dataclasses import dataclass, replace

import numpy as np

from .. import _codecs, _jpeg_restarts, _memory, _pool, _stream
from .._base import File, Image
from .._errors import CorruptFileError, UnsupportedFormatError
from .._io import DecodeCache, ThreadHandles

NAME = "czi"
MAGIC = b"ZISRAWFILE"
RAW_TASK_BYTES = 8 << 20            # scenes are read in row bands of this size
#: bytes worth skipping in one contiguous read rather than issuing one more
#: read (a seek + read from Python costs about as much as copying this much)
SEPARATE_READ_BYTES = 32 << 10
NATIVE_LITTLE = sys.byteorder == "little"

#: pixel type -> (dtype, samples)
PIXEL_TYPES = {0: ("<u1", 1), 1: ("<u2", 1), 2: ("<f4", 1), 3: ("<u1", 3), 4: ("<u2", 3),
               8: ("<f4", 3), 9: ("<u1", 4)}
#: plane dimensions: the axes, in the order they become axes (before y, x,
#: s), and the image indices (in index order)
AXIS_DIMS = "TCZ"
KEY_DIMS = {"rotation": "R", "view": "V", "illumination": "I", "phase": "H", "block": "B"}
PLANE_DIMS = AXIS_DIMS + "".join(KEY_DIMS.values())
TILE_POLICIES = ("separate", "stitched")
UNCOMPRESSED, JPEG, LZW, JPEG_LOSSLESS, JPEG_XR, ZSTD0, ZSTD1, CHUNKED, JPEG_XL = range(9)
COMPRESSIONS = {UNCOMPRESSED: "none", JPEG: "JPEG", LZW: "LZW",
                JPEG_LOSSLESS: "JPEG lossless (undocumented; standard lossless JPEG assumed)",
                JPEG_XR: "JPEG XR", ZSTD0: "zstd", ZSTD1: "zstd with header (hi-lo packing)",
                CHUNKED: "chunked (experimental in libCZI)",
                JPEG_XL: "JPEG XL (undocumented; a JPEG XL codestream assumed)"}
IMAGE_CODECS = {JPEG, JPEG_LOSSLESS, JPEG_XR, JPEG_XL}
DECODE_CACHE_BYTES = 64 << 20       # decoded compressed tiles kept for reuse
STREAM_PIECE_BYTES = 8 << 20        # rows of a streamed subblock decoded at a time


def sniff(head: bytes, path: str) -> bool:
    return head[:10] == MAGIC


@dataclass
class Subblock:
    plane: dict              # dimension -> absolute start index (T, C, Z, ...)
    x: int
    y: int
    width: int
    height: int
    scene: int
    mosaic: int
    order: int               # position in the directory
    position: int            # file position of the subblock segment
    compression: int
    pixel_type: int
    level0: bool = True      # full resolution (not a pyramid level)
    key: tuple = ()          # everything the entry says about the subblock except where it is
    data_offset: int = 0     # pixel data, filled in when first read
    data_size: int = 0       # bytes of (possibly compressed) pixel data
    stored_width: int = 0    # pixels stored: smaller than width / height in a pyramid level
    stored_height: int = 0


def _parse_entry(data, pos: int):
    """A ``DV`` directory entry at *data[pos:]* -> (Subblock, entry size).

    The same entry is stored in the directory and again inside the subblock
    segment itself; ``key`` lets the two be compared.
    """
    (schema, pixel_type, position, part, compression, pyramid_type, _, _,
     ndims) = struct.unpack_from("<2siqiiBB4si", data, pos)
    if schema != b"DV":
        raise UnsupportedFormatError(f"CZI directory entry schema {schema!r}")
    if not 0 < ndims < 64:
        raise CorruptFileError(f"CZI directory entry with {ndims} dimensions")
    dims = {}
    for j in range(ndims):
        name, start, size, _coord, stored = struct.unpack_from("<4siifi", data, pos + 32 + 20 * j)
        dims[name.rstrip(b"\x00").decode("cp1252")] = (start, size, stored or size)
    if part:
        raise UnsupportedFormatError("multi-part CZI (file parts)")
    key = (pixel_type, compression, pyramid_type, tuple(sorted(dims.items())))
    x0, w, sw = dims.pop("X")
    y0, h, sh = dims.pop("Y")
    entry = Subblock(
        plane={d: v[0] for d, v in dims.items() if d not in "SM"},
        x=x0, y=y0, width=w, height=h,
        scene=dims.get("S", (0,))[0], mosaic=dims.get("M", (-1,))[0],
        order=0, position=position, compression=compression, pixel_type=pixel_type,
        level0=pyramid_type == 0 and (sw, sh) == (w, h), key=key,
        stored_width=sw, stored_height=sh)
    return entry, 32 + 20 * ndims


def _directory(fh) -> list:
    """All subblock directory entries -> Subblocks (level 0 and pyramids)."""
    fh.seek(0)
    head = fh.read(32 + 80)
    if head[:10] != MAGIC:
        raise CorruptFileError("not a CZI file")
    (major, minor, _, _, _, _, file_part, directory_position, _, _, _) = \
        struct.unpack_from("<iiii16s16siqqiq", head, 32)
    if not directory_position:
        raise CorruptFileError("CZI without a subblock directory")
    fh.seek(directory_position)
    seg = fh.read(32)
    if not seg[:16].rstrip(b"\x00") == b"ZISRAWDIRECTORY":
        raise CorruptFileError(f"no ZISRAWDIRECTORY at byte {directory_position}")
    allocated, used = struct.unpack_from("<qq", seg, 16)
    data = fh.read(used or allocated)
    count, = struct.unpack_from("<i", data)
    pos, entries = 128, []
    for n in range(count):
        entry, size = _parse_entry(data, pos)
        entry.order = n
        entries.append(entry)
        pos += size
    return entries


SUBBLOCK_MAGIC = b"ZISRAWSUBBLOCK\x00\x00"


def _inline_key(fh, position: int):
    """The key of the subblock segment at *position* (from the entry stored
    inside it), or None if no subblock segment starts there."""
    fh.seek(position)
    head = fh.read(32 + 16 + 32 + 20 * 16)
    if head[:16] != SUBBLOCK_MAGIC:
        return None
    try:
        return _parse_entry(head, 32 + 16)[0].key
    except (struct.error, UnicodeDecodeError, CorruptFileError, UnsupportedFormatError):
        return None


def _ranges(values) -> str:
    """[1, 2, 3, 7] -> '1-3, 7' (scene lists can be long)."""
    out, values = [], sorted(values)
    for _, group in itertools.groupby(enumerate(values), lambda p: p[1] - p[0]):
        run = [v for _, v in group]
        out.append(f"{run[0]}-{run[-1]}" if len(run) > 1 else str(run[0]))
    return ", ".join(out)


class DamagedFileWarning(UserWarning):
    """The file is damaged; micro-reader recovered what it could.  Check or
    re-copy the file: other readers will fail or return wrong pixels."""


def _recover(fh, path: str, entries: list) -> dict:
    """Check that every entry points at its subblock; find the real position
    of those that do not.

    A subblock is accepted only where a subblock segment starts *and* the
    entry stored inside it equals the directory's entry (scene, tile, plane,
    rectangle, pixel type, compression).  Fast path: the damage is usually one
    insertion or removal, so the offset between the first misplaced entries
    and their real subblocks applies to all later ones; whatever that leaves
    is matched by scanning the whole file for subblock headers.
    Returns a damage report ({} when the file is fine).
    """
    bad = [e for e in sorted(entries, key=lambda e: e.position)
           if _inline_key(fh, e.position) != e.key]
    if not bad:
        return {}
    first = bad[0].position
    unresolved = {e.key: e for e in bad}
    if len(unresolved) != len(bad):
        raise CorruptFileError(f"{path!r}: damaged, and its directory lists the same "
                               "subblock twice; cannot recover")
    shifts: list = []

    def settle(shift):
        for key, e in list(unresolved.items()):
            if _inline_key(fh, e.position + shift) == key:
                e.position += shift
                del unresolved[key]

    def scan(start, stop, on_hit):
        chunk = 64 << 20
        pos = start
        while pos < stop:
            fh.seek(pos)
            buf = fh.read(min(chunk, stop - pos) + len(SUBBLOCK_MAGIC))
            i = buf.find(SUBBLOCK_MAGIC)
            while i >= 0:
                if on_hit(pos + i):
                    return
                i = buf.find(SUBBLOCK_MAGIC, i + 1)
            pos += chunk

    size = fh.seek(0, 2)

    # fast path: shifts found near the first misplaced entry
    def near(hit):
        key = _inline_key(fh, hit)
        if key in unresolved:
            shift = hit - unresolved[key].position
            shifts.append(shift)
            settle(shift)
        return not unresolved or len(shifts) >= 8
    scan(max(0, first - (64 << 20)), min(size, first + (256 << 20)), near)

    # fallback: every subblock header in the file
    if unresolved:
        found: dict = {}

        def index(hit):
            key = _inline_key(fh, hit)
            if key in unresolved:
                found.setdefault(key, []).append(hit)
            return False
        scan(0, size, index)
        for key, hits in found.items():
            if len(hits) == 1:
                unresolved.pop(key).position = hits[0]
    return {
        "first_bad_offset": first,
        "misplaced": len(bad),
        "recovered": len(bad) - len(unresolved),
        "unrecoverable": len(unresolved),
        "shifts": sorted(set(shifts)),
        "scenes_affected": sorted({e.scene for e in bad}),
        "scenes_unreadable": sorted({e.scene for e in unresolved.values()}),
        "unresolved_keys": set(unresolved),
    }


def _subtract(a, b) -> list:
    """Rectangle *a* minus rectangle *b*; rectangles are (y0, y1, x0, x1).
    Returns up to four disjoint pieces."""
    ay0, ay1, ax0, ax1 = a
    by0, by1, bx0, bx1 = b
    iy0, iy1, ix0, ix1 = max(ay0, by0), min(ay1, by1), max(ax0, bx0), min(ax1, bx1)
    if iy0 >= iy1 or ix0 >= ix1:
        return [a]
    pieces = []
    if ay0 < iy0:
        pieces.append((ay0, iy0, ax0, ax1))              # above
    if iy1 < ay1:
        pieces.append((iy1, ay1, ax0, ax1))              # below
    if ax0 < ix0:
        pieces.append((iy0, iy1, ax0, ix0))              # left
    if ix1 < ax1:
        pieces.append((iy0, iy1, ix1, ax1))              # right
    return pieces


class _PlaneTiles:
    """The subblocks of one plane in drawing order, with a grid index so a
    region finds its tiles without scanning them all (slide scans can have
    tens of thousands per plane)."""

    def __init__(self, subblocks: list, origin):
        ox, oy = origin
        self.subblocks = subblocks
        self.rects = [(s.y - oy, s.y - oy + s.height, s.x - ox, s.x - ox + s.width)
                      for s in subblocks]
        self.cell = (max(s.height for s in subblocks), max(s.width for s in subblocks))
        self.grid: dict = {}
        for i, (y0, y1, x0, x1) in enumerate(self.rects):
            for cy in range(y0 // self.cell[0], (y1 - 1) // self.cell[0] + 1):
                for cx in range(x0 // self.cell[1], (x1 - 1) // self.cell[1] + 1):
                    self.grid.setdefault((cy, cx), []).append(i)

    def query(self, rect, after: int = -1) -> list:
        """Indices (drawing order) of the tiles after *after* touching *rect*."""
        y0, y1, x0, x1 = rect
        found = set()
        for cy in range(y0 // self.cell[0], (y1 - 1) // self.cell[0] + 1):
            for cx in range(x0 // self.cell[1], (x1 - 1) // self.cell[1] + 1):
                found.update(i for i in self.grid.get((cy, cx), ()) if i > after)
        return sorted(i for i in found if _subtract(rect, self.rects[i]) != [rect])

    def visible(self, i: int, region) -> list:
        """The parts of tile *i* inside *region* that no later tile covers."""
        y0, y1, x0, x1 = self.rects[i]
        ry0, ry1, rx0, rx1 = region
        clip = (max(y0, ry0), min(y1, ry1), max(x0, rx0), min(x1, rx1))
        if clip[0] >= clip[1] or clip[2] >= clip[3]:
            return []
        pieces = [clip]
        for j in self.query(clip, after=i):
            pieces = [p for piece in pieces for p in _subtract(piece, self.rects[j])]
            if not pieces:
                break
        return pieces


class _CziImage(Image):
    """One image of a CZI (a mosaic tile, a stitched scene, or the canvas of
    all scenes), composed from its subblocks."""

    def __init__(self, czi: "_CziFile", index: int, subblocks: list, name: str,
                 key: dict, box: tuple = None):
        if not subblocks:
            raise CorruptFileError(f"{czi.path!r}: scene without subblocks")
        types = {s.pixel_type for s in subblocks}
        if len(types) != 1 or next(iter(types)) not in PIXEL_TYPES:
            raise UnsupportedFormatError(f"{czi.path!r}: CZI pixel types {sorted(types)}")
        unsupported = {s.compression for s in subblocks} - set(COMPRESSIONS)
        if unsupported:
            raise UnsupportedFormatError(
                f"{czi.path!r}: CZI compression {sorted(unsupported)}: codes 100-999 "
                "(camera RAW) and 1000+ (system RAW) are vendor-specific formats "
                "without a public description; other codes are unknown")
        image_key = key
        dtype, self._samples = PIXEL_TYPES[next(iter(types))]
        self._file_dtype = np.dtype(dtype)
        self._czi = czi
        # plane axes: T C Z always; B I H only where the file has several
        axes = [d for d in AXIS_DIMS if d in "TCZ" or czi._dim_sizes.get(d, 1) > 1]
        self._plane_dims = axes
        if box is not None:                        # a resolution level: the scene's box, scaled
            x0, y0, x1, y1 = box
        else:
            x0 = min(s.x for s in subblocks)
            y0 = min(s.y for s in subblocks)
            x1 = max(s.x + s.width for s in subblocks)
            y1 = max(s.y + s.height for s in subblocks)
        self.origin = (x0, y0)                     # scene box in image pixel coordinates
        # plane -> subblocks in drawing order: scene, mosaic index, directory
        # order (libCZI's order; the scene only matters on the whole-file canvas)
        by_plane: dict = {}
        for s in sorted(subblocks, key=lambda s: (s.scene, s.mosaic, s.order)):
            key = tuple(s.plane.get(d, czi._dim_starts.get(d, 0)) - czi._dim_starts.get(d, 0)
                        for d in axes)
            by_plane.setdefault(key, []).append(s)
        self._planes = {key: _PlaneTiles(tiles, self.origin) for key, tiles in by_plane.items()}
        shape = [czi._dim_sizes.get(d, 1) for d in axes] + [y1 - y0, x1 - x0]
        letters = "".join(d.lower() for d in axes) + "yx"
        if self._samples > 1:
            shape.append(self._samples)
            letters += "s"
        self._closed = False
        super().__init__(index, letters, shape, self._file_dtype, name=name, index=image_key)

    # -- reading ---------------------------------------------------------------

    def _data_offset(self, s: Subblock) -> int:
        """File offset of a subblock's pixels (read from its segment header)."""
        if not s.data_offset:
            if s.position < 0:
                raise CorruptFileError(
                    f"{self._czi.path!r} is damaged: the subblock for scene {s.scene}, "
                    f"tile {s.mosaic}, plane {s.plane} could not be found")
            fh = self._czi._handles.get()
            fh.seek(s.position)
            head = fh.read(32 + 48)
            if head[:16].rstrip(b"\x00") != b"ZISRAWSUBBLOCK":
                raise CorruptFileError(f"{self._czi.path!r}: no subblock at byte {s.position}")
            metadata_size, _attachment, data_size = struct.unpack_from("<iiq", head, 32)
            ndims, = struct.unpack_from("<i", head, 32 + 44)
            s.data_size = data_size
            s.data_offset = s.position + 32 + 16 + max(240, 32 + 20 * ndims) + metadata_size
        return s.data_offset

    def _decoded(self, s: Subblock):
        """A compressed subblock as (rows, cols x samples) in the file's
        dtype, from the cache or decoded now; and the pixels decoded."""
        tile = self._czi.decode_cache.get(s.position)
        if tile is not None:
            return tile, 0
        nbytes = s.height * s.width * self._samples * self._file_dtype.itemsize
        what = (f"{self._czi.path!r}: a {s.height} x {s.width} CZI subblock compressed with "
                f"{COMPRESSIONS.get(s.compression, s.compression)}")
        with _memory.reserve(nbytes, what):
            tile = self._decode_subblock(s)
        self._czi.decode_cache.put(s.position, tile)
        return tile, tile.size

    def _decode_subblock(self, s: Subblock) -> np.ndarray:
        base = self._data_offset(s)
        data = self._czi._handles.read(base, s.data_size)
        tile = np.empty((s.height, s.width * self._samples), self._file_dtype)
        if s.data_size == tile.nbytes and s.compression not in IMAGE_CODECS:
            tile[...] = np.frombuffer(data, self._file_dtype).reshape(tile.shape)  # stored raw
        elif s.compression == ZSTD0:
            _codecs.zstd_decompress_into(data, tile)
        elif s.compression == ZSTD1:                     # header, then a zstd frame
            header = data[0]
            hilo = header >= 3 and data[1] == 1 and data[2] & 1
            self._maybe_hilo(tile, hilo, lambda out: _codecs.zstd_decompress_into(
                memoryview(data)[header:], out))
        elif s.compression == LZW:
            _codecs.lzw_decode_into(data, tile)          # stored order, like raw data
        elif s.compression == CHUNKED:
            self._chunked(data, tile)
        else:                                            # image codecs: R, G, B out
            self._image_codec(s, data, tile)
        return tile

    def _maybe_hilo(self, tile, hilo: bool, decode) -> None:
        """Decode into *tile*, undoing hi-lo byte packing if it was applied."""
        if hilo and self._file_dtype.itemsize == 2:
            packed = np.empty(tile.nbytes, np.uint8)
            decode(packed)
            _codecs.hilo_unpack(packed, tile)
        else:
            decode(tile)

    def _chunked(self, data: bytes, tile) -> None:
        """Compression 7 (experimental in libCZI): varint header chunks
        {1: compressed sizes, 2: method 0 zstd / 1 LZ4, 3: decompressed
        sizes, 4: preprocessing 1 = hi-lo packing}, then the chunks back to
        back (Zeiss, "Chunked compression")."""
        def varint(buf, pos):
            value = shift = 0
            while True:
                byte = buf[pos]
                value |= (byte & 0x7F) << shift
                pos += 1
                if not byte & 0x80:
                    return value, pos
                shift += 7

        pos, fields = 0, {}
        while True:
            kind, pos = varint(data, pos)
            if kind == 0:
                break
            length, pos = varint(data, pos)
            fields[kind] = bytes(data[pos:pos + length])
            pos += length
        if 1 not in fields or 3 not in fields:
            raise CorruptFileError(f"{self._czi.path!r}: chunked subblock without chunk sizes")
        compressed, p = [], 0
        while p < len(fields[1]):
            n, p = varint(fields[1], p)
            compressed.append(n)
        sizes, p = [], 0
        while p < len(fields[3]):
            n, p = varint(fields[3], p)
            sizes.append(n)
        # [C]: every chunk C bytes; [C, L]: all C except the last, which is L
        decompressed = [sizes[0]] * len(compressed)
        if len(sizes) == 2:
            decompressed[-1] = sizes[1]
        elif len(sizes) not in (1, len(compressed)):
            raise CorruptFileError(f"{self._czi.path!r}: chunked sizes {sizes}")
        elif len(sizes) == len(compressed):
            decompressed = sizes
        method = fields.get(2, b"\x00")[0]
        if method not in (0, 1):
            raise UnsupportedFormatError(f"{self._czi.path!r}: chunked method {method}")
        hilo = fields.get(4, b"\x00")[0] == 1
        if sum(decompressed) != tile.nbytes:
            raise CorruptFileError(f"{self._czi.path!r}: chunked data decodes to "
                                   f"{sum(decompressed)} bytes, expected {tile.nbytes}")

        def decode(out):
            flat = out.reshape(-1).view(np.uint8)
            src = dst = 0
            view = memoryview(data)[pos:]
            for csize, dsize in zip(compressed, decompressed):
                chunk, target = view[src:src + csize], flat[dst:dst + dsize]
                if method == 0:
                    _codecs.zstd_decompress_into(chunk, target)
                else:
                    _codecs.lz4_block_decompress_into(chunk, target)
                src, dst = src + csize, dst + dsize
        self._maybe_hilo(tile, hilo, decode)

    def _image_codec(self, s: Subblock, data: bytes, tile) -> None:
        """JPEG, JPEG lossless, JPEG XR, JPEG XL: they give R, G, B; stored
        into *tile* in the file's B, G, R order like every other codec."""
        decoded = np.empty((s.height, s.width, self._samples), self._file_dtype)
        if s.compression in (JPEG, JPEG_LOSSLESS):
            width, height, samples, bits = _codecs.jpeg_decode_into(data, decoded)
        elif s.compression == JPEG_XL:
            width, height, samples, bits = _codecs.jxl_decode_into(data, decoded)
        else:                                            # JPEG_XR
            _codecs.jpegxr_decode_into(data, decoded)
            width, height, samples = s.width, s.height, self._samples
        if (width, height, samples) != (s.width, s.height, self._samples):
            raise CorruptFileError(
                f"{self._czi.path!r}: subblock codestream is {width} x {height} x {samples}, "
                f"the directory says {s.width} x {s.height} x {self._samples}")
        if self._samples > 1:
            decoded = decoded[..., [2, 1, 0, 3][:self._samples]]      # R G B -> stored B G R
        tile[...] = decoded.reshape(tile.shape)

    def _draw(self, s: Subblock, piece, band_y0, xs, dst) -> int:
        """Draw *piece* (y0, y1, x0, x1 in scene coordinates, inside subblock
        *s* and inside the band) onto *dst* (the band: rows, cols[, samples])."""
        ox, oy = self.origin
        sy0, sx0 = s.y - oy, s.x - ox                       # tile's place in the scene
        y0, y1, x0, x1 = piece
        item = self._file_dtype.itemsize * self._samples
        target = dst[y0 - band_y0:y1 - band_y0, x0 - xs.start:x1 - xs.start]
        width = x1 - x0
        if s.compression != UNCOMPRESSED:
            nbytes = s.height * s.width * self._samples * self._file_dtype.itemsize
            stream = self._stream_source(s) if _memory.should_stream(nbytes) else None
            if stream is not None:                           # too large to cache: its rows
                return self._draw_streamed(s, stream, piece, band_y0, xs, dst)
            if _memory.should_stream(nbytes) and s.compression in (JPEG, JPEG_XL):
                drawn = self._draw_area(s, piece, band_y0, xs, dst)
                if drawn is not None:                        # the piece alone
                    return drawn
            tile, read = self._decoded(s)                    # decoded whole (cached)
            buf = tile[y0 - sy0:y1 - sy0, (x0 - sx0) * self._samples:(x1 - sx0) * self._samples]
            self._place(buf, target, y1 - y0, width)
            return read
        base = self._data_offset(s)
        fh = self._czi._handles.get()
        if width == s.width:                                 # whole tile rows: one read
            buf = np.empty((y1 - y0, width * self._samples), self._file_dtype)
            fh.seek(base + (y0 - sy0) * s.width * item)
            if fh.readinto(memoryview(buf).cast("B")) != buf.nbytes:
                raise CorruptFileError(f"{self._czi.path!r}: subblock data ends early")
        elif y1 - y0 > 1 and (s.width - width) * item <= SEPARATE_READ_BYTES:
            # a few skipped bytes per row cost less than one more read per row: the span
            # from the first row's columns to the last row's, in one read
            rows, row = y1 - y0, s.width * self._samples
            span = (rows - 1) * row + width * self._samples
            flat = np.empty(span, self._file_dtype)
            fh.seek(base + ((y0 - sy0) * s.width + (x0 - sx0)) * item)
            if fh.readinto(memoryview(flat).cast("B")) != flat.nbytes:
                raise CorruptFileError(f"{self._czi.path!r}: subblock data ends early")
            buf = np.lib.stride_tricks.as_strided(
                flat, shape=(rows, width * self._samples),
                strides=(row * flat.itemsize, flat.itemsize))
            self._place(buf, target, rows, width)
            return span
        else:
            buf = np.empty((y1 - y0, width * self._samples), self._file_dtype)
            for i in range(y1 - y0):
                fh.seek(base + ((y0 - sy0 + i) * s.width + (x0 - sx0)) * item)
                if fh.readinto(memoryview(buf[i]).cast("B")) != buf.shape[1] * buf.itemsize:
                    raise CorruptFileError(f"{self._czi.path!r}: subblock data ends early")
        self._place(buf, target, y1 - y0, width)
        return buf.size

    def _stream_source(self, s: Subblock):
        """(codec, offset, length) of subblock *s*'s compressed rows when they
        can be streamed (``_stream``): zstd (``ZSTD0``; ``ZSTD1`` after its
        header, without hi-lo packing, which splits the whole tile into its
        low and high bytes), LZW, or data stored raw; else None."""
        source = self._stream_codec(s)
        return source if source is not None and _stream.streamable(source[0]) else None

    def _stream_codec(self, s: Subblock):
        base = self._data_offset(s)
        nbytes = s.height * s.width * self._samples * self._file_dtype.itemsize
        if s.data_size == nbytes and s.compression not in IMAGE_CODECS:
            return "none", base, s.data_size                 # stored raw
        if s.compression == ZSTD0:
            return "zstd", base, s.data_size
        if s.compression == LZW:
            return "lzw", base, s.data_size
        if s.compression == ZSTD1:
            head = self._czi._handles.read(base, 3)
            header = head[0]
            hilo = header >= 3 and head[1] == 1 and head[2] & 1
            if hilo and self._file_dtype.itemsize == 2:
                return None
            return "zstd", base + header, s.data_size - header
        return None

    def _draw_streamed(self, s: Subblock, source, piece, band_y0, xs, dst) -> int:
        """*piece* of subblock *s* from its rows alone, decoded as a stream in
        pieces of rows, each held within the memory budget."""
        codec, offset, length = source
        ox, oy = self.origin
        sy0, sx0 = s.y - oy, s.x - ox
        y0, y1, x0, x1 = piece
        row = s.width * self._samples
        row_bytes = row * self._file_dtype.itemsize
        per_piece = max(1, min(STREAM_PIECE_BYTES, _memory.memory_limit() // 4) // row_bytes)
        for a in range(y0, y1, per_piece):
            b = min(y1, a + per_piece)
            with _memory.reserve((b - a) * row_bytes, f"{self._czi.path!r}: rows of a subblock"):
                buf = np.empty((b - a) * row_bytes, np.uint8)
                n = _stream.read(self._czi.path, offset, length, codec, (a - sy0) * row_bytes, buf,
                                 size=s.height * row_bytes)
                if n < buf.nbytes:
                    raise CorruptFileError(f"{self._czi.path!r}: subblock data ends early")
                rows = buf.view(self._file_dtype).reshape(b - a, row)
                self._place(rows[:, (x0 - sx0) * self._samples:(x1 - sx0) * self._samples],
                            dst[a - band_y0:b - band_y0, x0 - xs.start:x1 - xs.start],
                            b - a, x1 - x0)
        return (y1 - y0) * (x1 - x0) * self._samples

    def _draw_area(self, s: Subblock, piece, band_y0, xs, dst):
        """*piece* of a JPEG XL subblock (jxl-oxide renders only the groups
        covering it) or of a JPEG subblock with restart markers (cut out of
        it, ``_jpeg_restarts``), in bands of rows within the budget; None
        for a JPEG that cannot be cut."""
        ox, oy = self.origin
        sy0, sx0 = s.y - oy, s.x - ox
        y0, y1, x0, x1 = piece
        base, size = self._data_offset(s), s.data_size
        what = f"{self._czi.path!r}: rows of a subblock"
        if s.compression == JPEG:
            index = self._restart_index(s, base, size)
            row_bytes = s.width * self._samples * self._file_dtype.itemsize
            if not _jpeg_restarts.usable(index, row_bytes, _memory.fits):
                return None
            for a, b, rows in _jpeg_restarts.read_rows(
                    index, lambda p, q: self._czi._handles.read(base + p, q - p),
                    y0 - sy0, y1 - sy0, _codecs.jpeg_decode_into, (self._samples,),
                    self._file_dtype, min(STREAM_PIECE_BYTES, _memory.memory_limit() // 4),
                    lambda n: _memory.reserve(n, what)):
                self._place_decoded(rows[:, x0 - sx0:x1 - sx0], a + sy0, b + sy0, piece,
                                    band_y0, xs, dst)
            return (y1 - y0) * (x1 - x0) * self._samples
        data = self._czi._handles.read(base, size)
        row_bytes = (x1 - x0) * self._samples * self._file_dtype.itemsize
        per_band = max(1, min(STREAM_PIECE_BYTES, _memory.memory_limit() // 4) // row_bytes)
        for a in range(y0, y1, per_band):
            b = min(y1, a + per_band)
            area = np.empty((b - a, x1 - x0, self._samples), self._file_dtype)
            with _memory.reserve(area.nbytes, what):
                got = _codecs.jxl_decode_area_into(data, area, x0 - sx0, a - sy0,
                                                   x1 - sx0, b - sy0)
                if (got[1], got[0], got[2]) != area.shape:
                    raise CorruptFileError(
                        f"{self._czi.path!r}: JPEG XL area decodes to {got[1]} x {got[0]} x "
                        f"{got[2]}, expected {area.shape}")
                self._place_decoded(area, a, b, piece, band_y0, xs, dst)
        return (y1 - y0) * (x1 - x0) * self._samples

    def _place_decoded(self, rows, a: int, b: int, piece, band_y0, xs, dst) -> None:
        """Decoded rows *a*..*b* (scene coordinates) of *piece*, (rows, cols,
        samples) in R, G, B order, onto *dst* in the file's B, G, R order."""
        _, _, x0, x1 = piece
        if self._samples > 1:
            rows = rows[..., [2, 1, 0, 3][:self._samples]]          # R G B -> stored B G R
        self._place(np.ascontiguousarray(rows).reshape(b - a, (x1 - x0) * self._samples),
                    dst[a - band_y0:b - band_y0, x0 - xs.start:x1 - xs.start], b - a, x1 - x0)

    def _restart_index(self, s: Subblock, base: int, size: int):
        """The ``RestartIndex`` of JPEG subblock *s* (None: it cannot be
        cut), built once from its compressed bytes."""
        restarts = self._czi.restarts
        if s.position not in restarts:
            with _memory.reserve(size, f"{self._czi.path!r}: a compressed JPEG subblock"):
                index = _jpeg_restarts.RestartIndex.build(self._czi._handles.read(base, size))
            restarts[s.position] = index
        return restarts[s.position]

    def _place(self, buf, target, rows: int, width: int) -> None:
        """Pixels (rows, width x samples) as stored -> *target*; samples
        from the stored B, G, R (A) to R, G, B (A)."""
        if self._samples > 1:
            pixels = buf.reshape(rows, width, self._samples)
            target[...] = pixels[..., [2, 1, 0, 3][:self._samples]]
        else:
            target[...] = buf

    def _draw_span(self, s: Subblock, pieces, band_y0, xs, dst) -> int:
        """Uncompressed *pieces* of subblock *s* from ONE contiguous read: from
        the first piece pixel to the last, the bytes between them included."""
        ox, oy = self.origin
        sy0, sx0 = s.y - oy, s.x - ox
        by0, by1 = min(p[0] for p in pieces), max(p[1] for p in pieces)
        bx0, bx1 = min(p[2] for p in pieces), max(p[3] for p in pieces)
        row = s.width * self._samples                        # elements per tile row
        count = (by1 - by0 - 1) * row + (bx1 - bx0) * self._samples
        buf = np.empty((by1 - by0) * row, self._file_dtype)
        fh = self._czi._handles.get()
        fh.seek(self._data_offset(s)
                + ((by0 - sy0) * row + (bx0 - sx0) * self._samples) * self._file_dtype.itemsize)
        if fh.readinto(memoryview(buf[:count]).cast("B")) != count * buf.itemsize:
            raise CorruptFileError(f"{self._czi.path!r}: subblock data ends early")
        rows = buf.reshape(by1 - by0, row)                   # column 0 = pixel bx0
        for y0, y1, x0, x1 in pieces:
            self._place(rows[y0 - by0:y1 - by0,
                             (x0 - bx0) * self._samples:(x1 - bx0) * self._samples],
                        dst[y0 - band_y0:y1 - band_y0, x0 - xs.start:x1 - xs.start],
                        y1 - y0, x1 - x0)
        return count

    def _draw_band(self, tiles: _PlaneTiles, y0, y1, xs, dst) -> int:
        """One plane onto one row band: only the part of each tile no later
        tile covers is drawn, and a tile hidden entirely is never read or
        decoded (bands are disjoint, so they can be drawn in parallel).
        An uncompressed tile's visible pieces come from one contiguous read
        unless the bytes skipped between them cost more than the separate
        reads they save (every piece narrower than the tile is one read per
        row)."""
        region, read = (y0, y1, xs.start, xs.stop), 0
        item = self._file_dtype.itemsize * self._samples
        for i in tiles.query(region):
            s, pieces = tiles.subblocks[i], tiles.visible(i, region)
            if not pieces:
                continue
            if s.compression == UNCOMPRESSED:
                by0, by1 = min(p[0] for p in pieces), max(p[1] for p in pieces)
                bx0, bx1 = min(p[2] for p in pieces), max(p[3] for p in pieces)
                span = (by1 - by0 - 1) * s.width + (bx1 - bx0)
                skipped = (span - sum((p[1] - p[0]) * (p[3] - p[2]) for p in pieces)) * item
                reads = sum(1 if p[3] - p[2] == s.width else p[1] - p[0] for p in pieces)
                if len(pieces) > 1 or reads > 1:
                    if skipped <= (reads - 1) * SEPARATE_READ_BYTES:
                        read += self._draw_span(s, pieces, y0, xs, dst)
                        continue
            for piece in pieces:
                read += self._draw(s, piece, y0, xs, dst)
        return read

    def _read_block(self, block):
        if self._closed:
            raise ValueError("this CZI image has been closed: its file was closed "
                             "(File.close() or the end of a with block)")
        n = len(self._plane_dims)
        plane_spans, ys, xs = block[:n], block[n], block[n + 1]
        ss = block[n + 2] if self._samples > 1 else None
        out = np.zeros(tuple(s.stop - s.start for s in block[:n + 2])
                       + ((self._samples,) if ss is not None else ()), dtype=self.dtype)
        row_bytes = max(1, (xs.stop - xs.start) * self._samples * self._file_dtype.itemsize)
        per_task = max(1, RAW_TASK_BYTES // row_bytes)
        tasks = []
        for key in itertools.product(*(range(s.start, s.stop) for s in plane_spans)):
            tiles = self._planes.get(key)
            if tiles is None:
                continue                                    # nothing acquired: background
            dst = out[tuple(k - s.start for k, s in zip(key, plane_spans))]
            for b0 in range(ys.start, ys.stop, per_task):
                b1 = min(ys.stop, b0 + per_task)
                part = dst[b0 - ys.start:b1 - ys.start]
                tasks.append(lambda a=(tiles, b0, b1, xs, part): self._draw_band(*a))
        read = _pool.run(tasks)
        if ss is not None and (ss.start, ss.stop) != (0, self._samples):
            out = out[..., ss]
        return out, read

    def read_region(self, x: int, y: int, width: int, height: int, **planes) -> np.ndarray:
        """A rectangle in the file's own pixel coordinates (libCZI's ``roi``,
        e.g. from ``origin`` or the XML metadata), composed from this scene's
        tiles; parts outside the scene's box are background (0), as libCZI
        returns them.  *planes* select the other axes like ``read``."""
        if width < 0 or height < 0:
            raise ValueError(f"negative region size {width} x {height}")
        ox, oy = self.origin
        ny, nx = self.shape[self.axes.index("y")], self.shape[self.axes.index("x")]
        y0, y1 = min(max(y - oy, 0), ny), min(max(y + height - oy, 0), ny)
        x0, x1 = min(max(x - ox, 0), nx), min(max(x + width - ox, 0), nx)
        y1, x1 = max(y0, y1), max(x0, x1)
        inside = self.read(**planes, y=slice(y0, y1), x=slice(x0, x1))
        if (y1 - y0, x1 - x0) == (height, width):
            return inside
        # y, x are the last two axes, or before the samples axis if it is kept
        at = -3 if self._samples > 1 and not isinstance(planes.get("s"), (int, np.integer)) else -2
        shape = list(inside.shape)
        shape[at], shape[at + 1] = height, width
        out = np.zeros(shape, inside.dtype)
        index = [slice(None)] * out.ndim
        index[at] = slice(y0 + oy - y, y1 + oy - y)
        index[at + 1] = slice(x0 + ox - x, x1 + ox - x)
        out[tuple(index)] = inside
        return out

    def _read_metadata(self):
        from ._czi_metadata import czi_metadata
        return czi_metadata(self, self._czi._handles.get(), self._czi._xml_root)

    def close(self) -> None:
        self._closed = True


class _CziFile(File):
    backend = NAME

    def __init__(self, path: str, tiles: str = "separate", resolutions: str = "base"):
        if tiles not in TILE_POLICIES:
            raise ValueError(f"tiles={tiles!r}: one of {TILE_POLICIES}")
        self.path = path
        self._handles = ThreadHandles(path)
        #: decoded compressed tiles, shared by all images of the file
        self.decode_cache = DecodeCache(DECODE_CACHE_BYTES)
        #: subblock position -> RestartIndex | None of large JPEG subblocks
        self.restarts: dict = {}
        with builtins.open(path, "rb") as fh:
            entries = _directory(fh)
            level0 = [e for e in entries if e.level0] or entries
            #: pyramid subblocks (lower resolution levels of whole scenes)
            pyramid = [e for e in entries if not e.level0] if level0 is not entries else []
            #: {} for a sound file; otherwise what was damaged and recovered
            self.damage = _recover(fh, path, level0)
        if self.damage:
            for e in level0:
                if e.key in self.damage["unresolved_keys"]:
                    e.position = -1                         # reading it raises
            d = self.damage
            lost = (f"; {d['unrecoverable']} subblocks could NOT be found (scenes "
                    f"{d['scenes_unreadable']}) and cannot be read"
                    if d["unrecoverable"] else "")
            warnings.warn(DamagedFileWarning(
                f"DAMAGED CZI FILE: {path!r}. Its subblock directory does not match the "
                f"data from byte {d['first_bad_offset']:,} on ({d['misplaced']} of "
                f"{len(level0)} subblocks misplaced, scenes {_ranges(d['scenes_affected'])}). "
                f"micro-reader recovered the real position of {d['recovered']} of them "
                f"(shift {d['shifts']} bytes), each verified against the subblock's own "
                f"header{lost}. Other readers (pylibCZIrw, aicspylibczi, Bio-Formats) fail "
                "or return wrong pixels on this file: check the file or copy it again."),
                stacklevel=3)
        self._level0 = level0
        # dimension ranges over the whole file (as bioio / pylibCZIrw count them)
        self._dim_starts, self._dim_sizes = {}, {}
        for d in PLANE_DIMS:
            values = [e.plane[d] for e in level0 if d in e.plane]
            if values:
                self._dim_starts[d] = min(values)
                self._dim_sizes[d] = max(values) - min(values) + 1
        per_scene: dict = {}
        for e in level0:
            per_scene.setdefault(e.scene, set()).add(e.mosaic)
        #: some scene has several mosaic tiles
        self._tiled = any(len(m) > 1 for m in per_scene.values())
        #: indices this file has, besides "scene"
        self._levels = (["tile"] if self._tiled and tiles == "separate" else []) + \
            [level for level, d in KEY_DIMS.items() if self._dim_sizes.get(d, 1) > 1]
        groups: dict = {}
        for e in level0:
            groups.setdefault(self._key_of(e), []).append(e)
        base = os.path.basename(path)
        # resolution levels: stored for whole scenes only (pyramid subblocks
        # carry no tile index), and not trusted in a repaired directory
        with_levels = "tile" not in self._levels and pyramid and not self.damage
        listed = resolutions == "all"
        images = []
        for values in sorted(groups):
            key = dict(zip(["scene"] + self._levels, values))
            name = base + " " + " ".join(f"{k} {v}" for k, v in key.items())
            if listed:
                key["resolution"] = 0
            image = _CziImage(self, len(images), groups[values], name, key)
            images.append(image)
            if with_levels:
                levels = self._resolution_levels(image, pyramid, values, name, key)
                if levels:
                    image._resolutions = [image] + levels
                    for level in levels:
                        level._resolutions = image._resolutions
                    if listed:
                        images.extend(levels)
        for n, image in enumerate(images):
            image._position = n
        super().__init__(path, images)

    def _resolution_levels(self, image: "_CziImage", pyramid: list, values: tuple, name: str,
                           key: dict) -> list:
        """*image*'s lower resolution levels: one per stored zoom of its scene
        (x rotation, view), each composed from that zoom's pyramid subblocks
        at their scaled positions in a grid of the scene's size x zoom (rounded
        down, as libCZI sizes a zoomed read)."""
        mine = [e for e in pyramid if self._key_of(e) == values]
        if not mine:
            return []
        x0, y0 = image.origin
        ny, nx = image.shape[image.axes.index("y")], image.shape[image.axes.index("x")]
        by_zoom: dict = {}
        for e in mine:
            by_zoom.setdefault(round(e.stored_width / e.width, 6), []).append(e)
        levels = []
        for k, zoom in enumerate(sorted(by_zoom, reverse=True), start=1):
            # coordinates in the file's frame at this zoom: the scene's corner
            # (its origin) scaled, plus each subblock's scaled offset from it
            ox, oy = int(np.floor(x0 * zoom)), int(np.floor(y0 * zoom))
            placed = [replace(e, x=ox + int(np.floor((e.x - x0) * zoom)),
                              y=oy + int(np.floor((e.y - y0) * zoom)),
                              width=e.stored_width, height=e.stored_height)
                      for e in by_zoom[zoom]]
            level = _CziImage(self, 0, placed, f"{name} resolution {k}",
                             {**key, "resolution": k},
                             box=(ox, oy, ox + int(nx * zoom), oy + int(ny * zoom)))
            level.zoom = zoom
            levels.append(level)
        return levels

    @property
    def _xml_root(self):
        """The root element of the file's XML metadata (parsed on first
        use), or None."""
        if not hasattr(self, "_xml"):
            from ._czi_metadata import read_xml
            self._xml = read_xml(self._handles.get())
        return self._xml

    def _key_of(self, e: Subblock) -> tuple:
        """The image index values of subblock *e*, in index order."""
        values = [e.scene]
        for level in self._levels:
            if level == "tile":
                values.append(max(e.mosaic, 0))
            else:
                d = KEY_DIMS[level]
                values.append(e.plane.get(d, self._dim_starts[d]))
        return tuple(values)

    def canvas(self, **key) -> _CziImage:
        """All scenes composed on one canvas: the box around every level-0
        tile of the file (libCZI's ``total_bounding_rectangle_no_pyramid``),
        background between the scenes.  Its ``origin`` is the box's corner.
        A file with several rotations / views needs ``rotation=`` /
        ``view=``: they are different frames."""
        unknown = set(key) - set(KEY_DIMS)
        if unknown:
            raise KeyError(f"canvas: unknown indices {sorted(unknown)}; "
                           f"it takes {sorted(KEY_DIMS)}")
        missing = [level for level, d in KEY_DIMS.items()
                   if self._dim_sizes.get(d, 1) > 1 and level not in key]
        if missing:
            raise ValueError(f"{self.path!r} has several {' and '.join(missing)}s: "
                             f"canvas({missing[0]}=...) picks one")
        subblocks = [e for e in self._level0
                     if all(e.plane.get(KEY_DIMS[k], self._dim_starts.get(KEY_DIMS[k], 0)) == v
                            for k, v in key.items())]
        if not subblocks:
            raise ValueError(f"canvas: no subblocks with {key}")
        return _CziImage(self, len(self.images), subblocks,
                        f"{os.path.basename(self.path)} canvas", dict(key))

    def close(self) -> None:
        super().close()
        self._handles.close()
        _stream.forget(self.path)


def open(path: str, tiles: str = "separate", resolutions: str = "base") -> _CziFile:  # noqa: A001 - backend protocol
    return _CziFile(path, tiles, resolutions)
