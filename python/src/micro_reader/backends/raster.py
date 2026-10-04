"""Plain raster images: JPEG, PNG and BMP files, one image each.

Axes ``yx``, or ``yxs`` for colour (R, G, B (A)).  None of these formats can
be read by rectangle, so the first read decodes the whole picture (the
read unit is the whole image) and keeps it while it fits the cache.

- **JPEG**: baseline, progressive and lossless (8 / 16 bit), grey or colour,
  through ``_codecs``.  CMYK / YCCK JPEGs are refused by name.  EXIF
  orientation is not applied (as Bio-Formats).
- **PNG**: every colour type and depth, interlaced or not.  Grey depths
  below 8 bits keep their stored values (0-15 for 4 bits, as TIFF packed
  depths); 16-bit samples are native-endian.
- **BMP**: 1 / 2 / 4 / 8-bit palettes (uncompressed, RLE4, RLE8), 16 / 24 /
  32-bit (uncompressed, bit fields, with alpha when a mask says so), JPEG or
  PNG payloads, top-down or bottom-up, Windows and OS/2 headers.  Colour
  fields narrower than 8 bits (5-5-5, 5-6-5) are scaled to 8 bits.

Palettes, as Bio-Formats handles them: a palette holding only greys is
applied (the pixels are the greys); a colour palette is not -- the pixels
are the indices and ``image.palette`` holds the colours ((n, 3) R G B, or
(n, 4) with PNG transparency), so label images keep their labels.
"""
from __future__ import annotations

import builtins
import os
import struct
import threading
import zlib
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .. import _codecs, _jpeg_restarts, _memory, _stream
from .._base import File, Image
from .._errors import CorruptFileError, UnsupportedFormatError

NAME = "raster"
PNG_MAGIC = b"\x89PNG\r\n\x1a\n"
JPEG_MAGIC = b"\xff\xd8\xff"
#: BMP header sizes: OS/2 1.x (12) and 2.x (16, 64), Windows 3.x (40) to v5 (124)
BMP_HEADERS = (12, 16, 40, 52, 56, 64, 108, 124)
HEAD_BYTES = 1 << 20              # read at open for the header; the whole file if needed
STREAM_PIECE_BYTES = 8 << 20     # rows of a streamed PNG decoded at a time
CACHE_BYTES = 256 << 20           # decoded pictures up to this size are kept


def kind_of(head: bytes) -> Optional[str]:
    """"jpeg", "png", "bmp" by content, or None."""
    if head[:3] == JPEG_MAGIC:
        return "jpeg"
    if head[:8] == PNG_MAGIC:
        return "png"
    if head[:2] == b"BM" and len(head) >= 18 and \
            struct.unpack_from("<I", head, 14)[0] in BMP_HEADERS:
        return "bmp"
    return None


def sniff(head: bytes, path: str) -> bool:
    return kind_of(head) is not None


@dataclass
class Info:
    """What a picture's header says: enough to know its shape and decode it."""
    kind: str
    height: int
    width: int
    samples: int
    dtype: np.dtype
    bits: int = 8
    #: colour palette, (n, 3 or 4) uint8: the pixels are indices into it
    palette: Optional[np.ndarray] = None
    #: grey palette, applied on decoding: pixel = lut[index]
    lut: Optional[np.ndarray] = None
    extra: dict = field(default_factory=dict)


def _set_palette(info: Info, rgb: np.ndarray, alpha: Optional[np.ndarray] = None) -> None:
    """Grey palettes are applied, colour palettes kept (module docstring)."""
    rgb = np.ascontiguousarray(rgb, np.uint8)
    opaque = alpha is None or bool((alpha == 255).all())
    if opaque and (rgb[:, 0] == rgb[:, 1]).all() and (rgb[:, 1] == rgb[:, 2]).all():
        lut = np.zeros(256, np.uint8)                 # indices beyond the palette: 0
        lut[:min(256, len(rgb))] = rgb[:256, 0]
        info.lut = lut
    else:
        info.palette = rgb if opaque else np.ascontiguousarray(
            np.concatenate([rgb, alpha[:, None].astype(np.uint8)], axis=1))


def apply_palette(indices: np.ndarray, palette: np.ndarray) -> np.ndarray:
    """Indices -> their colours (indices beyond the palette: black)."""
    full = np.zeros((256, palette.shape[1]), np.uint8)
    full[:min(256, len(palette))] = palette[:256]
    return full[indices]


def info_of(data: bytes) -> Info:
    """The header of a JPEG / PNG / BMP file (*data*: its start or all of it)."""
    kind = kind_of(data)
    if kind == "jpeg":
        return _jpeg_info(data)
    if kind == "png":
        return _png_info(data)
    if kind == "bmp":
        return _bmp_info(data)
    raise UnsupportedFormatError("not a JPEG, PNG or BMP file")


def decode(data: bytes, info: Info) -> np.ndarray:
    """The whole picture, shaped (height, width[, samples])."""
    if "outer_offset" in info.extra:                         # a JPEG / PNG inside a BMP
        data = data[info.extra["outer_offset"]:]
    out = {"jpeg": _jpeg_decode, "png": _png_decode, "bmp": _bmp_decode}[info.kind](data, info)
    if info.lut is not None:
        out = info.lut[out]
    want = (info.height, info.width) + ((info.samples,) if info.samples > 1 else ())
    if out.shape != want:
        raise CorruptFileError(f"{info.kind.upper()} decodes to {out.shape}, "
                               f"its header says {want}")
    return out


# -- JPEG ---------------------------------------------------------------------------

def jpeg_header(data: bytes):
    """(height, width, components, precision) from the first SOF marker."""
    pos = 2
    while pos + 4 <= len(data):
        if data[pos] != 0xFF:
            pos += 1
            continue
        marker = data[pos + 1]
        if marker in (0xD8, 0x01, 0xFF) or 0xD0 <= marker <= 0xD7:
            pos += 1 if marker == 0xFF else 2
            continue
        length = struct.unpack(">H", data[pos + 2:pos + 4])[0]
        if 0xC0 <= marker <= 0xCF and marker not in (0xC4, 0xC8, 0xCC):
            if pos + 10 > len(data):
                break
            bits, height, width, comps = struct.unpack(">BHHB", data[pos + 4:pos + 10])
            return height, width, comps, bits
        pos += 2 + length
    raise CorruptFileError("JPEG without a frame header")


def _jpeg_info(data: bytes) -> Info:
    height, width, comps, bits = jpeg_header(data)
    if comps == 4:
        raise UnsupportedFormatError("JPEG with 4 components (CMYK / YCCK)")
    if comps not in (1, 3):
        raise UnsupportedFormatError(f"JPEG with {comps} components")
    if height == 0 or width == 0:
        raise UnsupportedFormatError("JPEG whose size is set later (DNL marker)")
    return Info("jpeg", height, width, comps, np.dtype("u1") if bits <= 8 else np.dtype("<u2"),
                bits=bits)


def _jpeg_decode(data: bytes, info: Info) -> np.ndarray:
    shape = (info.height, info.width) + ((info.samples,) if info.samples > 1 else ())
    out = np.empty(shape, info.dtype)
    _codecs.jpeg_decode_into(data, out)
    return out


# -- PNG ----------------------------------------------------------------------------

def _png_chunks(data: bytes):
    """(type, body) of each chunk; a chunk running past *data* is an error
    (at open: read more of the file)."""
    pos = 8
    while pos + 8 <= len(data):
        length, ctype = struct.unpack_from(">I4s", data, pos)
        if pos + 12 + length > len(data):
            raise CorruptFileError("PNG ends inside a chunk")
        yield ctype, data[pos + 8:pos + 8 + length]
        if ctype == b"IEND":
            return
        pos += 12 + length
    raise CorruptFileError("PNG ends before its image data")


def _png_info(data: bytes) -> Info:
    if len(data) < 29 or data[12:16] != b"IHDR":
        raise CorruptFileError("PNG without a header chunk")
    width, height, bits, colour, _, _, interlace = struct.unpack(">IIBBBBB", data[16:29])
    samples = {0: 1, 2: 3, 3: 1, 4: 2, 6: 4}.get(colour)
    if samples is None or bits not in (1, 2, 4, 8, 16):
        raise CorruptFileError(f"PNG colour type {colour} at {bits} bits")
    plte = trns = None
    for ctype, body in _png_chunks(data):
        if ctype == b"PLTE":
            plte = body
        elif ctype == b"tRNS":
            trns = body
        elif ctype == b"IDAT":
            break
    info = Info("png", height, width, samples,
                np.dtype("<u2") if bits == 16 else np.dtype("u1"), bits=bits,
                extra={"colour": colour, "interlace": interlace})
    if colour == 3:
        if plte is None:
            raise CorruptFileError("palette PNG without a palette (PLTE)")
        rgb = np.frombuffer(plte, np.uint8, len(plte) // 3 * 3).reshape(-1, 3)
        alpha = None
        if trns:
            alpha = np.full(len(rgb), 255, np.uint8)
            n = min(len(trns), len(rgb))
            alpha[:n] = np.frombuffer(trns, np.uint8, n)
        _set_palette(info, rgb, alpha)
    return info


def _png_decode(data: bytes, info: Info) -> np.ndarray:
    h, w, s, bits = info.height, info.width, info.samples, info.bits
    shape = (h, w, s) if s > 1 else (h, w)
    if bits >= 8 and info.extra["colour"] != 3:
        out = np.empty(shape, info.dtype)
        _codecs.png_decode_into(data, out)
        return out
    # palette indices and depths below 8 bits: packed rows, unpacked here
    row = (w * s * bits + 7) // 8
    if _codecs._use_native(None, "png_decode_into"):
        packed = np.empty(h * row, np.uint8)
        _codecs.png_decode_into(data, packed)
    else:
        packed = _png_rows(data, info, row)
    out = np.empty((h, w * s), np.uint8)
    if bits == 8:
        out[...] = packed.reshape(h, row)
    else:
        _codecs.unpack_bits_into(packed, out, bits, w * s)
    return out.reshape(shape)


def _png_rows(data: bytes, info: Info, row: int) -> np.ndarray:
    """Reference for packed rows: inflate and unfilter in numpy (samples
    below 8 bits or palette indices: one byte per filter unit)."""
    if info.extra["interlace"]:
        raise UnsupportedFormatError(
            "interlaced palette / low-bit PNG needs the native decoders (micro-reader-codecs)")
    idat = b"".join(body for ctype, body in _png_chunks(data) if ctype == b"IDAT")
    try:
        raw = np.frombuffer(zlib.decompress(idat), np.uint8)
    except zlib.error as exc:
        raise CorruptFileError(f"corrupt PNG data: {exc}") from exc
    h = info.height
    if raw.size < h * (row + 1):
        raise CorruptFileError("PNG data ends early")
    raw = raw[:h * (row + 1)].reshape(h, row + 1)
    out = np.empty((h, row), np.uint8)
    prev = np.zeros(row, np.uint8)
    for y in range(h):
        kind, line = raw[y, 0], raw[y, 1:]
        if kind == 0:
            cur = line.copy()
        elif kind == 1:                                       # sub
            cur = np.cumsum(line, dtype=np.uint8)
        elif kind == 2:                                       # up
            cur = line + prev
        elif kind in (3, 4):                                  # average, Paeth
            cur = np.empty(row, np.uint8)
            left = 0
            for i in range(row):
                up = int(prev[i])
                if kind == 3:
                    pred = (left + up) // 2
                else:
                    corner = int(prev[i - 1]) if i else 0
                    p = left + up - corner
                    pa, pb, pc = abs(p - left), abs(p - up), abs(p - corner)
                    pred = left if pa <= pb and pa <= pc else up if pb <= pc else corner
                left = cur[i] = (int(line[i]) + pred) & 0xFF
        else:
            raise CorruptFileError(f"PNG filter type {kind}")
        out[y] = prev = cur
    return out.reshape(-1)


# -- BMP ----------------------------------------------------------------------------

def _bmp_info(data: bytes) -> Info:
    offset, dib = struct.unpack_from("<II", data, 10)
    if dib == 12:                                            # OS/2 1.x
        width, height, _, bpp = struct.unpack_from("<HHHH", data, 18)
        compression, colours, entry = 0, 0, 3
    else:
        width, height, _, bpp = struct.unpack_from("<iiHH", data, 18)
        compression = struct.unpack_from("<I", data, 30)[0] if dib >= 20 else 0
        colours = struct.unpack_from("<I", data, 46)[0] if dib >= 36 else 0
        entry = 4
    if width <= 0 or height == 0:
        raise CorruptFileError(f"BMP of {width} x {height} pixels")
    info = Info("bmp", abs(height), width, 3, np.dtype("u1"), bits=bpp,
                extra={"offset": offset, "compression": compression, "top_down": height < 0})
    if compression in (4, 5):                                # a JPEG / PNG inside
        inner = info_of(data[offset:])
        inner.extra["outer_offset"] = offset
        return inner
    if bpp in (1, 2, 4, 8):
        if not (compression == 0 or (compression, bpp) in ((1, 8), (2, 4))):
            raise UnsupportedFormatError(
                f"BMP with {bpp} bits per pixel, compression {compression}")
        n = colours or (1 << bpp)
        at = 14 + dib
        if len(data) < at + n * entry:
            raise CorruptFileError("BMP ends inside its palette")
        table = np.frombuffer(data, np.uint8, n * entry, at).reshape(n, entry)
        info.samples = 1
        _set_palette(info, table[:, 2::-1])                  # stored B, G, R (, 0)
        return info
    if bpp == 24 and compression == 0:
        return info
    if bpp in (16, 32) and compression in (0, 3, 6):
        if compression == 0:
            masks = [0x7C00, 0x3E0, 0x1F] if bpp == 16 else [0xFF0000, 0xFF00, 0xFF]
        else:
            masks = list(struct.unpack_from("<III", data, 54))
            if (dib >= 56 or compression == 6) and len(data) >= 70:
                alpha = struct.unpack_from("<I", data, 66)[0]
                if alpha:
                    masks.append(alpha)
        for m in masks:
            if not m or bin(m).count("1") > 8:
                raise UnsupportedFormatError(f"BMP bit field mask {m:#x}")
        info.samples = len(masks)
        info.extra["masks"] = masks
        return info
    raise UnsupportedFormatError(f"BMP with {bpp} bits per pixel, compression {compression}")


def _bmp_decode(data: bytes, info: Info) -> np.ndarray:
    e, h, w, bpp = info.extra, info.height, info.width, info.bits
    off, compression = e["offset"], e["compression"]
    if compression in (1, 2):
        pixels = _bmp_rle(memoryview(data)[off:], w, h, bpp)
    else:
        stride = (w * bpp + 31) // 32 * 4
        if len(data) < off + stride * h:
            raise CorruptFileError("BMP pixel data ends early")
        raw = np.frombuffer(data, np.uint8, stride * h, off).reshape(h, stride)
        if bpp <= 8:
            pixels = np.empty((h, w), np.uint8)
            if bpp == 8:
                pixels[...] = raw[:, :w]
            else:
                _codecs.unpack_bits_into(np.ascontiguousarray(raw[:, :(w * bpp + 7) // 8]),
                                         pixels, bpp, w)
        elif bpp == 24:
            pixels = raw[:, :w * 3].reshape(h, w, 3)[..., ::-1]
        else:
            values = np.ascontiguousarray(raw[:, :w * bpp // 8]).view(
                "<u2" if bpp == 16 else "<u4").astype(np.uint32)
            chans = []
            for m in e["masks"]:
                shift = (m & -m).bit_length() - 1
                top = (1 << bin(m).count("1")) - 1
                c = (values >> shift) & top
                if top < 255:                                # scale to 8 bits
                    c = (c * 255 + top // 2) // top
                chans.append(c.astype(np.uint8))
            pixels = np.stack(chans, axis=-1)
    if not e["top_down"]:                                    # stored bottom-up
        pixels = pixels[::-1]
    return np.ascontiguousarray(pixels)


def _bmp_rle(data, w: int, h: int, bpp: int) -> np.ndarray:
    """RLE8 / RLE4 -> (rows, cols) indices in stored (bottom-up) row order;
    pixels the stream skips stay 0."""
    out = np.zeros((h, w), np.uint8)
    x = y = i = 0
    n = len(data)

    def put(values):
        nonlocal x
        if y < h and x < w:
            k = min(len(values), w - x)
            out[y, x:x + k] = values[:k]
        x += len(values)

    def nibbles(byte_values, count):
        b = np.frombuffer(bytes(byte_values), np.uint8)
        return np.stack([b >> 4, b & 15], axis=1).reshape(-1)[:count]

    while i + 1 < n:
        count, value = data[i], data[i + 1]
        i += 2
        if count:                                            # a run
            if bpp == 8:
                put(np.full(count, value, np.uint8))
            else:
                put(nibbles([value] * ((count + 1) // 2), count))
        elif value == 0:                                     # end of line
            x, y = 0, y + 1
        elif value == 1:                                     # end of bitmap
            break
        elif value == 2:                                     # delta
            if i + 1 >= n:
                break
            x, y = x + data[i], y + data[i + 1]
            i += 2
        else:                                                # absolute run
            size = value if bpp == 8 else (value + 1) // 2
            if i + size > n:
                raise CorruptFileError("BMP RLE data ends early")
            chunk = data[i:i + size]
            put(np.frombuffer(bytes(chunk), np.uint8) if bpp == 8 else nibbles(chunk, value))
            i += size + (size & 1)                           # runs end on a 16-bit word
    return out


# -- the backend --------------------------------------------------------------------

class _RasterImage(Image):
    """The picture of a JPEG / PNG / BMP file."""

    def __init__(self, path: str, info: Info):
        shape = (info.height, info.width) + ((info.samples,) if info.samples > 1 else ())
        self._path, self._info = path, info
        #: the colours of a colour palette ((n, 3) R G B, or (n, 4) with
        #: alpha): the pixels are indices into it.  None otherwise.
        self.palette = info.palette
        self._lock = threading.Lock()
        self._cached = None
        self._restarts = False             # the JPEG's RestartIndex once built (None: none)
        self._closed = False
        super().__init__(0, "yx" + ("s" if info.samples > 1 else ""), shape, info.dtype,
                         name=os.path.basename(path), read_unit=shape)

    def _pixels(self):
        """The decoded picture, and the samples decoded to get it now."""
        with self._lock:
            if self._cached is not None:
                return self._cached, 0
            with builtins.open(self._path, "rb") as fh:
                data = fh.read()
            try:
                info = self._info
                nbytes = int(np.prod(self.shape)) * self.dtype.itemsize
                with _memory.reserve(nbytes, f"{self._path!r}: a {info.kind.upper()} picture "
                                             f"of {self.shape[1]} x {self.shape[0]} pixels"):
                    pixels = decode(data, info)
            except (CorruptFileError, UnsupportedFormatError) as exc:
                raise type(exc)(f"{self._path!r}: {exc}") from exc
            if pixels.nbytes <= CACHE_BYTES:
                self._cached = pixels
            return pixels, pixels.size

    def _read_block(self, block):
        if self._closed:
            raise ValueError("this image has been closed: its file was closed "
                             "(File.close() or the end of a with block)")
        if self._cached is None and self._streams():
            return self._streamed(block)
        pixels, read = self._pixels()
        return np.array(pixels[block], copy=True), read

    def _streams(self) -> bool:
        """A picture too large to keep decoded (``CACHE_BYTES``), read by
        rows: a PNG not interlaced (``_stream``; streamed rows cannot be
        reused, and PNG has no checkpoints), or a JPEG with restart markers
        (``_jpeg_restarts``)."""
        info = self._info
        nbytes = int(np.prod(self.shape)) * self.dtype.itemsize
        if nbytes <= CACHE_BYTES:
            return False
        if info.kind == "jpeg":
            row_bytes = info.width * info.samples * info.dtype.itemsize
            return _jpeg_restarts.usable(self._restart_index(), row_bytes, _memory.fits)
        return info.kind == "png" and not info.extra.get("interlace") and _stream.streamable("png")

    def _restart_index(self):
        """The JPEG's ``RestartIndex`` (None: it cannot be cut), built once."""
        with self._lock:
            if self._restarts is False:
                with builtins.open(self._path, "rb") as fh:
                    fh.seek(self._info.extra.get("outer_offset", 0))
                    self._restarts = _jpeg_restarts.RestartIndex.build(fh.read())
            return self._restarts

    def _jpeg_rows(self, block):
        """Rows ``block[0]`` of a JPEG cut at its restart markers, in bands."""
        info, ys = self._info, block[0]
        offset = info.extra.get("outer_offset", 0)
        out = np.empty(tuple(b.stop - b.start for b in block), self.dtype)

        def read(start, stop):
            with builtins.open(self._path, "rb") as fh:
                fh.seek(offset + start)
                return fh.read(stop - start)

        tail = (info.samples,) if info.samples > 1 else ()
        for a, b, rows in _jpeg_restarts.read_rows(
                self._restart_index(), read, ys.start, ys.stop, _codecs.jpeg_decode_into, tail,
                info.dtype, min(STREAM_PIECE_BYTES, _memory.memory_limit() // 4),
                lambda n: _memory.reserve(n, f"{self._path!r}: rows of a JPEG")):
            out[a - ys.start:b - ys.start] = rows[(slice(None),) + tuple(block[1:])]
        return out, (ys.stop - ys.start) * info.width * info.samples

    def _streamed(self, block):
        """Rows ``block[0]`` of a PNG, decoded as a stream in pieces of rows,
        unpacked like a whole decode (``_png_decode``, ``decode``)."""
        info = self._info
        if info.kind == "jpeg":
            return self._jpeg_rows(block)
        ys = block[0]
        w, s, bits = info.width, info.samples, info.bits
        packed = bits < 8 or info.extra["colour"] == 3
        row = (w * s * bits + 7) // 8 if packed else w * s * info.dtype.itemsize
        offset = info.extra.get("outer_offset", 0)
        out = np.empty(tuple(b.stop - b.start for b in block), self.dtype)
        per_piece = max(1, min(STREAM_PIECE_BYTES, _memory.memory_limit() // 4) // row)
        for a in range(ys.start, ys.stop, per_piece):
            b = min(ys.stop, a + per_piece)
            with _memory.reserve((b - a) * row * 2, f"{self._path!r}: rows of a PNG"):
                buf = np.empty((b - a) * row, np.uint8)
                n = _stream.read(self._path, offset, 0, "png", a * row, buf,
                                 size=info.height * row)
                if n < buf.nbytes:
                    raise CorruptFileError(f"{self._path!r}: PNG data ends early")
                if not packed:
                    rows = buf.view(info.dtype).reshape(b - a, w * s)
                elif bits == 8:
                    rows = buf.reshape(b - a, w * s)
                else:
                    rows = np.empty((b - a, w * s), np.uint8)
                    _codecs.unpack_bits_into(buf, rows, bits, w * s)
                if info.lut is not None:
                    rows = info.lut[rows]
                rows = rows.reshape((b - a, w, s) if s > 1 else (b - a, w))
                out[a - ys.start:b - ys.start] = rows[(slice(None),) + tuple(block[1:])]
        return out, (ys.stop - ys.start) * w * s

    def close(self) -> None:
        self._closed = True
        self._cached = None
        _stream.forget(self._path)


class _RasterFile(File):
    backend = NAME

    def __init__(self, path: str, info: Info):
        #: "jpeg", "png" or "bmp" (also for a BMP holding a JPEG / PNG)
        self.format = "bmp" if "outer_offset" in info.extra else info.kind
        super().__init__(path, [_RasterImage(path, info)])


def open(path: str, tiles: str = "separate", resolutions: str = "base") -> _RasterFile:  # noqa: A001 - backend protocol
    with builtins.open(path, "rb") as fh:
        head = fh.read(HEAD_BYTES)
        try:
            try:
                info = info_of(head)
            except (CorruptFileError, struct.error, IndexError, ValueError):
                if len(head) < HEAD_BYTES:                  # that was the whole file
                    raise
                fh.seek(0)
                info = info_of(fh.read())                   # a header beyond the first MB
        except (CorruptFileError, UnsupportedFormatError) as exc:
            raise type(exc)(f"{path!r}: {exc}") from exc
        except (struct.error, IndexError, ValueError) as exc:
            raise CorruptFileError(f"{path!r}: truncated or damaged header ({exc})") from exc
    return _RasterFile(path, info)
