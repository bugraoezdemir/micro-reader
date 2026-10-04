"""A minimal classic-TIFF writer for tests: what tifffile cannot write --
packed bit depths (2, 4, 10, 12, 14, 24 ...) and FillOrder 2 -- in strips,
either byte order, uncompressed or compressed.  tifffile (+ imagecodecs,
following libtiff) reads the files as the reference.
"""
from __future__ import annotations

import struct
import zlib

import numpy as np

REVERSED = np.array([int(f"{b:08b}"[::-1], 2) for b in range(256)], np.uint8)


def pack_bits(values: np.ndarray, bits: int) -> bytes:
    """(rows, runlen) unsigned values -> TIFF packing: most significant bit
    first, each row starting on a byte boundary."""
    rows, runlen = values.shape
    shifts = np.arange(bits - 1, -1, -1, dtype=np.uint64)
    b = ((values.astype(np.uint64)[..., None] >> shifts) & 1).astype(np.uint8)
    return np.packbits(b.reshape(rows, runlen * bits), axis=1).tobytes()


def write_tiff(path, data, bits, *, byteorder="<", rows_per_strip=None, compression=1,
               fillorder=1, samples_bytes_order=None):
    """*data*: (rows, cols) or (rows, cols, samples) unsigned integers,
    stored at *bits* per sample (packed as TIFF packs them when *bits* is
    not the numpy item size).  *compression*: 1 none, 8 deflate, 32773
    PackBits (not supported here), 5 LZW (via imagecodecs).  *fillorder* 2
    reverses the bits of every stored byte."""
    import imagecodecs
    data = np.asarray(data)
    rows, cols = data.shape[:2]
    samples = data.shape[2] if data.ndim == 3 else 1
    rps = rows_per_strip or rows
    runlen = cols * samples
    strips = []
    for r0 in range(0, rows, rps):
        part = data[r0:r0 + rps].reshape(-1, runlen)
        if bits == 8 * data.dtype.itemsize:
            raw = part.astype(data.dtype.newbyteorder(byteorder)).tobytes()
        elif bits % 8 == 0:                     # e.g. 24-bit: whole bytes, file byte order
            wide = part.astype(np.dtype("<u8" if byteorder == "<" else ">u8"))
            nb = bits // 8
            b = wide.view(np.uint8).reshape(part.shape + (8,))
            raw = (b[..., :nb] if byteorder == "<" else b[..., 8 - nb:]).tobytes()
        else:
            raw = pack_bits(part, bits)
        if compression == 8:
            raw = zlib.compress(raw)
        elif compression == 5:
            raw = bytes(imagecodecs.lzw_encode(raw))
        if fillorder == 2:
            raw = REVERSED[np.frombuffer(raw, np.uint8)].tobytes()
        strips.append(raw)

    bo = byteorder
    tags = {  # tag: (type, values)   types: 3 SHORT, 4 LONG
        256: (4, [cols]), 257: (4, [rows]), 258: (3, [bits] * samples), 259: (3, [compression]),
        262: (3, [2 if samples == 3 else 1]), 266: (3, [fillorder]), 273: (4, [0] * len(strips)),
        277: (3, [samples]), 278: (4, [rps]), 279: (4, [len(s) for s in strips]),
        284: (3, [1]), 339: (3, [1] * samples),
    }
    n = len(tags)
    ifd_at = 8
    extra_at = ifd_at + 2 + n * 12 + 4
    blobs, entries = bytearray(), []
    for tag in sorted(tags):
        typ, values = tags[tag]
        size = {3: 2, 4: 4}[typ]
        fmt = {3: "H", 4: "I"}[typ]
        if len(values) * size <= 4:
            entries.append((tag, typ, values, None))
        else:
            entries.append((tag, typ, values, extra_at + len(blobs)))
            blobs += struct.pack(f"{bo}{len(values)}{fmt}", *values)
    data_at = extra_at + len(blobs)
    offsets, at = [], data_at
    for s in strips:
        offsets.append(at)
        at += len(s)
    out = bytearray(b"II*\x00" if bo == "<" else b"MM\x00*")
    out += struct.pack(f"{bo}I", ifd_at)
    out += struct.pack(f"{bo}H", n)
    blob = bytearray(blobs)
    for tag, typ, values, pointer in entries:
        if tag == 273:
            values = offsets
        fmt = {3: "H", 4: "I"}[typ]
        if pointer is None:
            field = struct.pack(f"{bo}{len(values)}{fmt}", *values).ljust(4, b"\x00")
            out += struct.pack(f"{bo}HHI", tag, typ, len(values)) + field
        else:
            if tag == 273:
                start = pointer - extra_at
                blob[start:start + 4 * len(values)] = struct.pack(f"{bo}{len(values)}I", *values)
            out += struct.pack(f"{bo}HHII", tag, typ, len(values), pointer)
    out += struct.pack(f"{bo}I", 0)
    out += blob
    for s in strips:
        out += s
    with open(path, "wb") as fh:
        fh.write(out)


def write_pages(path, strips, height, width, bits, compression, extra_tags=None,
                byteorder="<", bigtiff=False):
    """A multi-page TIFF whose pages are pre-encoded single strips (e.g. EER
    frames): *strips* = one bytes object per page.  *extra_tags*: {tag:
    (type, values)} added to every page -- type 3 SHORT / 4 LONG with a
    list of ints, 7 UNDEFINED with bytes.  *bigtiff*: BigTIFF layout (EER
    files are BigTIFF)."""
    bo = byteorder
    fmt = {3: "H", 4: "I", 16: "Q"}
    if bigtiff:
        out = bytearray((b"II" if bo == "<" else b"MM") + struct.pack(f"{bo}HHHQ", 43, 8, 0, 0))
        link_at, count_fmt, entry_value, offset_fmt = 8, "Q", 8, "Q"
    else:
        out = bytearray((b"II*\x00" if bo == "<" else b"MM\x00*") + struct.pack(f"{bo}I", 0))
        link_at, count_fmt, entry_value, offset_fmt = 4, "I", 4, "I"
    for strip in strips:
        data_at = len(out)
        out += strip
        if len(out) % 2:
            out += b"\x00"
        tags = {256: (4, [width]), 257: (4, [height]), 258: (3, [bits]),
                259: (3, [compression]), 262: (3, [1]),
                273: (16 if bigtiff else 4, [data_at]), 277: (3, [1]), 278: (4, [height]),
                279: (16 if bigtiff else 4, [len(strip)])}
        tags.update(extra_tags or {})
        encoded = {}
        for tag, (typ, values) in tags.items():
            if typ == 7:
                encoded[tag] = (typ, len(values), bytes(values))
            else:
                encoded[tag] = (typ, len(values), struct.pack(f"{bo}{len(values)}{fmt[typ]}", *values))
        # values that do not fit in an entry go right before the IFD
        pointers = {}
        for tag in sorted(encoded):
            raw = encoded[tag][2]
            if len(raw) > entry_value:
                pointers[tag] = len(out)
                out += raw
                if len(out) % 2:
                    out += b"\x00"
        ifd_at = len(out)
        struct.pack_into(f"{bo}{offset_fmt}", out, link_at, ifd_at)
        out += struct.pack(f"{bo}{count_fmt}", len(encoded))
        for tag in sorted(encoded):
            typ, count, raw = encoded[tag]
            field = (struct.pack(f"{bo}{offset_fmt}", pointers[tag]) if tag in pointers
                     else raw.ljust(entry_value, b"\x00"))
            out += struct.pack(f"{bo}HH{count_fmt}", tag, typ, count) + field
        link_at = len(out)
        out += struct.pack(f"{bo}{offset_fmt}", 0)       # next IFD: patched by the next page
    with open(path, "wb") as fh:
        fh.write(out)
