"""Minimal PNG and BMP writers for tests: the layouts common writers do not
produce (grey below 8 bits, every PNG filter, Adam7 interlacing, BMP RLE4 /
RLE8, bit fields, OS/2 headers, a PNG inside a BMP)."""
from __future__ import annotations

import struct
import zlib

import numpy as np

ADAM7 = [(0, 0, 8, 8), (4, 0, 8, 8), (0, 4, 4, 8), (2, 0, 4, 4), (0, 2, 2, 4), (1, 0, 2, 2),
         (0, 1, 1, 2)]                                    # (x0, y0, dx, dy) per pass


def _chunk(kind: bytes, body: bytes) -> bytes:
    return struct.pack(">I", len(body)) + kind + body + struct.pack(">I", zlib.crc32(kind + body))


def _pack_rows(values: np.ndarray, bits: int) -> list:
    """(h, w[, s]) values -> one packed byte string per row (PNG order)."""
    h = values.shape[0]
    flat = values.reshape(h, -1)
    if bits == 16:
        return [r.astype(">u2").tobytes() for r in flat]
    if bits == 8:
        return [r.astype("u1").tobytes() for r in flat]
    rows = []
    for r in flat:
        b = np.unpackbits(r.astype("u1")[:, None], axis=1)[:, 8 - bits:]   # MSB first
        rows.append(np.packbits(b.reshape(-1)).tobytes())
    return rows


def _filter(rows: list, unit: int, filters) -> bytes:
    """Rows filtered with filter types cycling through *filters*."""
    out, prev = bytearray(), bytes(len(rows[0])) if rows else b""
    for y, row in enumerate(rows):
        kind = filters[y % len(filters)]
        cur = bytearray()
        for i, x in enumerate(row):
            a = row[i - unit] if i >= unit else 0
            b = prev[i]
            c = prev[i - unit] if i >= unit else 0
            if kind == 0:
                pred = 0
            elif kind == 1:
                pred = a
            elif kind == 2:
                pred = b
            elif kind == 3:
                pred = (a + b) // 2
            else:
                p = a + b - c
                pa, pb, pc = abs(p - a), abs(p - b), abs(p - c)
                pred = a if pa <= pb and pa <= pc else b if pb <= pc else c
            cur.append((x - pred) & 0xFF)
        out += bytes([kind]) + cur
        prev = row
    return bytes(out)


def write_png(path, values, *, bits=8, colour=0, plte=None, trns=None, filters=(0,),
              interlace=False, text_before=0):
    """*values*: (h, w[, samples]) as stored (indices for colour type 3).
    *text_before*: bytes of tEXt chunk before the palette (a large header)."""
    values = np.asarray(values)
    h, w = values.shape[:2]
    samples = values.shape[2] if values.ndim == 3 else 1
    unit = max(1, samples * bits // 8)
    if interlace:
        data = b""
        for x0, y0, dx, dy in ADAM7:
            sub = values[y0::dy, x0::dx]
            if sub.shape[0] and sub.shape[1]:
                data += _filter(_pack_rows(sub, bits), unit, filters)
    else:
        data = _filter(_pack_rows(values, bits), unit, filters)
    out = b"\x89PNG\r\n\x1a\n" + _chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, bits, colour, 0, 0,
                                                                  int(interlace)))
    if text_before:
        out += _chunk(b"tEXt", b"Comment\x00" + b"x" * text_before)
    if plte is not None:
        out += _chunk(b"PLTE", np.asarray(plte, "u1").tobytes())
    if trns is not None:
        out += _chunk(b"tRNS", np.asarray(trns, "u1").tobytes())
    out += _chunk(b"IDAT", zlib.compress(data)) + _chunk(b"IEND", b"")
    with open(path, "wb") as fh:
        fh.write(out)


def _rle(rows, bpp: int) -> bytes:
    """RLE8 / RLE4 of index rows (bottom-up order given): runs of equal
    values, an absolute run for each stretch of differing ones."""
    out = bytearray()
    for row in rows:
        row = [int(v) for v in row]
        i = 0
        while i < len(row):
            j = i
            while j < len(row) and row[j] == row[i] and j - i < 255:
                j += 1
            if j - i >= 3 or j == len(row):                       # a run
                out += bytes([j - i, row[i] if bpp == 8 else row[i] << 4 | row[i]])
                i = j
                continue
            k = i                                                  # differing stretch
            while k < len(row) and k - i < 255 and not (k + 2 < len(row) and row[k] == row[k + 1] == row[k + 2]):
                k += 1
            n = k - i
            if n < 3:                                              # absolute needs >= 3
                for v in row[i:k]:
                    out += bytes([1, v if bpp == 8 else v << 4])
                i = k
                continue
            if bpp == 8:
                body = bytes(row[i:k])
            else:
                vals = row[i:k] + [0]
                body = bytes(vals[m] << 4 | vals[m + 1] for m in range(0, n, 2))
            out += bytes([0, n]) + body + (b"\x00" if len(body) % 2 else b"")
            i = k
        out += b"\x00\x00"                                         # end of line
    out += b"\x00\x01"                                             # end of bitmap
    return bytes(out)


def write_bmp(path, pixels, *, bpp, palette=None, compression=0, masks=None, top_down=False,
              header=40, payload=None):
    """*pixels*: (h, w) indices or (h, w, 3 / 4) R G B (A) as they should
    read back (for 16-bit and bit fields: the stored field values).
    *header*: 12 (OS/2 1.x), 40, 108 (v4).  *payload*: bytes of a JPEG /
    PNG stored instead (compression 4 / 5)."""
    pixels = np.asarray(pixels)
    h, w = pixels.shape[:2]
    rows = pixels if top_down else pixels[::-1]
    if payload is not None:
        data = payload
    elif compression in (1, 2):
        data = _rle(rows, bpp)
    else:
        stride = (w * bpp + 31) // 32 * 4
        data = bytearray()
        for r in rows:
            if bpp <= 8:
                if bpp == 8:
                    raw = r.astype("u1").tobytes()
                else:
                    b = np.unpackbits(r.astype("u1")[:, None], axis=1)[:, 8 - bpp:]
                    raw = np.packbits(b.reshape(-1)).tobytes()
            elif bpp == 24:
                raw = r[:, ::-1].astype("u1").tobytes()
            else:
                v = np.zeros(w, "u4")
                for c, m in enumerate(masks):
                    shift = (m & -m).bit_length() - 1
                    v |= (r[:, c].astype("u4") << shift) & m
                raw = v.astype("<u2" if bpp == 16 else "<u4").tobytes()
            data += raw.ljust(stride, b"\x00")
        data = bytes(data)
    if header == 12:
        info = struct.pack("<IHHHH", 12, w, h, 1, bpp)
        table = b"" if palette is None else b"".join(bytes([b, g, r]) for r, g, b in palette)
        extra = b""
    else:
        info = struct.pack("<IiiHHIIiiII", header, w, -h if top_down else h, 1, bpp, compression,
                           len(data), 2835, 2835, 0 if palette is None else len(palette), 0)
        extra = b""
        if header >= 52:
            m = list(masks or [0, 0, 0]) + [0] * (4 - len(masks or [0, 0, 0]))
            info += struct.pack("<IIII", *m[:4])
        elif compression == 3:
            extra = struct.pack("<III", *masks[:3])
        info = info.ljust(header, b"\x00")
        table = b"" if palette is None else b"".join(bytes([b, g, r, 0]) for r, g, b in palette)
    offset = 14 + len(info) + len(extra) + len(table)
    out = b"BM" + struct.pack("<IHHI", offset + len(data), 0, 0, offset) + info + extra + table + data
    with open(path, "wb") as fh:
        fh.write(out)
