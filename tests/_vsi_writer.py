"""Olympus / Evident cellSens VSI sets for tests, laid out as the OME cellSens
samples are (checked field by field): ``<stem>.vsi`` -- a little-endian
TIFF whose metadata tree starts at byte 8 and whose first page is the slide
photo -- and ``_<stem>_/stack<N>/frame_t.ets``, the tiled pyramids.

A tree node is ``value(tag, code, value, second=-1)`` or ``volume(tag,
children, second=-1, code=1)`` (an extended field holding one nested
volume)."""
from __future__ import annotations

import os
import struct
import zlib

import numpy as np

INT32, INT64, DOUBLE, STRING, INT_LIST, DOUBLE_LIST, BGR = 5, 7, 10, 8192, 259, 260, 270
EXTRA_TAG, EXTENDED = 1 << 27, 1 << 28
COMPONENT_TYPES = {"int8": 1, "uint8": 2, "int16": 3, "uint16": 4, "int32": 5, "uint32": 6,
                   "float32": 9, "float64": 10}
#: DIMENSION_INDEX values
Z, T, LAMBDA, C, PHASE = 1, 2, 3, 4, 9


def value(tag, code, data, second=-1):
    return ("v", tag, code, data, second)


def volume(tag, children, second=-1, code=1):
    return ("x", tag, code, children, second)


def _value_bytes(code, data) -> bytes:
    if code == INT32:
        return struct.pack("<i", data)
    if code == INT64:
        return struct.pack("<q", data)
    if code == DOUBLE:
        return struct.pack("<d", data)
    if code == STRING:
        return (data + "\x00").encode("utf-16-le")
    if code == INT_LIST:
        return struct.pack(f"<{len(data)}i", *data)
    if code == DOUBLE_LIST:
        return struct.pack(f"<{len(data)}d", *data)
    if code == BGR:
        return b"".join(bytes(t[::-1]) for t in data)
    raise ValueError(code)


def encode_volume(fields) -> bytes:
    """A volume: its 24-byte header, then its fields one after another."""
    records = []
    for kind, tag, code, data, second in fields:
        payload = encode_volume(data) if kind == "x" else _value_bytes(code, data)
        ftype = code | (EXTENDED if kind == "x" else 0) | (EXTRA_TAG if second >= 0 else 0)
        records.append((ftype, tag, second, payload))
    out, pos = [], 24
    starts = []
    for ftype, tag, second, payload in records:
        starts.append(pos)
        pos += 16 + (4 if second >= 0 else 0) + len(payload)
    for k, (ftype, tag, second, payload) in enumerate(records):
        nxt = starts[k + 1] if k + 1 < len(records) else 0
        out.append(struct.pack("<iiiI", ftype, tag, nxt, len(payload)))
        if second >= 0:
            out.append(struct.pack("<i", second))
        out.append(payload)
    return struct.pack("<HHIqII", 24, 21321, 1, 24, len(records), 0) + b"".join(out)


# -- the metadata of one image ----------------------------------------------------------------

def channel(name=None, wavelength=None, color=None) -> dict:
    return {"name": name, "wavelength": wavelength, "color": color}


def image_volume(stack, size, *, name="image", stack_type=64, pixel=(0.5, 0.25),
                 pixel_unit="10^-6m^1", dims=(), channels=(), stack_channel=None):
    """A ``MULTIDIM_IMAGE_VOLUME``.  *size* (rows, cols) or None (no image: a
    focus map); *dims*: (DIMENSION_INDEX, step, step unit) per tile coordinate
    after the tile row, in order; *channels*: per channel of a C dimension."""
    props = [value(2030, STRING, name), value(2074, INT32, stack_type),
             value(2019, DOUBLE_LIST, list(pixel)), value(2020, STRING, pixel_unit)]
    if stack_channel is not None:
        props.append(value(2419, STRING, stack_channel))
    children = [volume(2005, props)]
    if size is not None:
        boundary = value(2053, INT_LIST, [0, 0, size[1], size[0]])
        children.append(volume(2002, [volume(2018, [boundary])]))
    for k, (index, step, unit) in enumerate(dims):
        first = [value(2023, INT32, index)]
        if step is not None:
            first.append(volume(2013, [value(268435456, STRING, unit),
                                       value(268435458, DOUBLE, step)], code=2))
        if index == C:
            params = [volume(2008, first + _channel_props(channels[0]) if channels else first,
                             second=0)]
            params += [volume(2008, _channel_props(ch), second=c)
                       for c, ch in enumerate(channels[1:], 1)]
        else:
            params = [volume(2008, first)]
        children.append(volume(2007, params, second=k, code=0))
    return volume(2001, children, second=stack, code=2)


def _channel_props(ch: dict) -> list:
    props = []
    if ch.get("name") is not None:
        props.append(value(2419, STRING, ch["name"]))
    if ch.get("wavelength") is not None:
        props.append(volume(2417, [value(268435456, STRING, "10^-9m^1"),
                                   value(268435458, DOUBLE, float(ch["wavelength"]))], code=2))
    if ch.get("color") is not None:
        lut = [(0, 0, 0), tuple(v // 2 for v in ch["color"]), tuple(ch["color"])]
        props.append(volume(2004, [value(268435458, BGR, lut)], code=2))
    return props


# -- files --------------------------------------------------------------------------------

def _tiff(tree: bytes, macro) -> bytes:
    """A little-endian TIFF: the tree at byte 8, then one uncompressed
    grey page (*macro*, uint8) or no page."""
    body = b"II*\x00" + b"\x00" * 4 + tree
    if len(body) % 2:
        body += b"\x00"
    if macro is None:
        return body
    macro = np.ascontiguousarray(macro, np.uint8)
    rows, cols = macro.shape
    ifd_at = len(body)
    entries = [(256, 4, cols), (257, 4, rows), (258, 3, 8), (259, 3, 1), (262, 3, 1),
               (273, 4, 0), (277, 3, 1), (278, 4, rows), (279, 4, macro.size)]
    pixels_at = ifd_at + 2 + 12 * len(entries) + 4
    ifd = struct.pack("<H", len(entries))
    for tag, kind, v in entries:
        v = pixels_at if tag == 273 else v
        ifd += struct.pack("<HHII", tag, kind, 1, v) if kind == 4 else \
            struct.pack("<HHIHxx", tag, kind, 1, v)
    ifd += struct.pack("<I", 0)
    body = body[:4] + struct.pack("<I", ifd_at) + body[8:]
    return body + ifd + macro.tobytes()


def write_vsi(folder, stem, volumes, stacks=None, macro=None) -> str:
    """``<folder>/<stem>.vsi`` with the image *volumes*; *stacks*: folder
    number -> ``write_ets`` keyword arguments (``_<stem>_/stack<N>/frame_t.ets``)."""
    tree = encode_volume([volume(2000, list(volumes))])
    path = os.path.join(str(folder), stem + ".vsi")
    with open(path, "wb") as fh:
        fh.write(_tiff(tree, macro))
    for number, kwargs in (stacks or {}).items():
        stack = os.path.join(str(folder), f"_{stem}_", f"stack{number}")
        os.makedirs(stack, exist_ok=True)
        write_ets(os.path.join(stack, "frame_t.ets"), **kwargs)
    return path


def _png(tile: np.ndarray) -> bytes:
    rows, cols, components = tile.shape
    bits = tile.dtype.itemsize * 8
    color_type = {1: 0, 3: 2}[components]
    raw = tile.astype(tile.dtype.newbyteorder(">")).reshape(rows, cols * components)
    scanlines = b"".join(b"\x00" + raw[r].tobytes() for r in range(rows))

    def chunk(kind, data):
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))
    head = struct.pack(">IIBBBBB", cols, rows, bits, color_type, 0, 0, 0)
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", head) + chunk(b"IDAT", zlib.compress(scanlines))
            + chunk(b"IEND", b""))


def write_ets(path, levels, tile=(16, 16), pyramid=True, compression=0, missing=(),
              color_space=None, rgb=False, background=()) -> str:
    """An ETS file of the pyramid *levels*: arrays shaped ``extra + (rows, cols)``
    (+ ``(3,)`` with *rgb*), ``extra`` the tile coordinates after the tile row.
    *compression* 0 (raw) or 8 (PNG); *missing*: tile keys ``(level, tx, ty,
    *extra)`` left out of the table; *background*: the header's background
    values, one per component (zeros when empty)."""
    extra_ndim = levels[0].ndim - 2 - (1 if rgb else 0)
    components = 3 if rgb else 1
    dtype = levels[0].dtype
    th, tw = tile
    ndim = 2 + extra_ndim + (1 if pyramid else 0)
    blobs, table = [], []
    pos = 64 + 228
    for level, data in enumerate(levels):
        planes = data if rgb else data[..., None]
        rows, cols = planes.shape[extra_ndim:extra_ndim + 2]
        for extra in np.ndindex(*planes.shape[:extra_ndim]):
            plane = planes[extra]
            for ty in range(-(-rows // th)):
                for tx in range(-(-cols // tw)):
                    if (level, tx, ty) + tuple(extra) in missing:
                        continue
                    t = np.zeros((th, tw, components), dtype)
                    part = plane[ty * th:(ty + 1) * th, tx * tw:(tx + 1) * tw]
                    t[:part.shape[0], :part.shape[1]] = part
                    blob = t.astype(dtype.newbyteorder("<")).tobytes() if compression == 0 \
                        else _png(t)
                    coords = (tx, ty) + tuple(extra) + ((level,) if pyramid else ())
                    table.append(struct.pack(f"<4x{ndim}iqI4x", *coords, pos, len(blob)))
                    blobs.append(blob)
                    pos += len(blob)
    if color_space is None:
        color_space = 4 if rgb else 1
    ets = b"ETS\x00" + struct.pack("<9I", 0x30001, COMPONENT_TYPES[dtype.name], components,
                                   color_space, compression, 90, tw, th, 1)
    block = np.asarray(background, dtype.newbyteorder("<")).tobytes()
    ets += b"\x00" * (17 * 4) + block + b"\x00" * (10 * 4 - len(block))
    ets += struct.pack("<II", 0, int(pyramid))
    ets += b"\x00" * (228 - len(ets))
    sis = struct.pack("<4sIIIQII QII QII", b"SIS\x00", 64, 2, ndim, 64, 228, 0, pos, len(table),
                      0, 0, 0, 0)
    with open(path, "wb") as fh:
        fh.write(sis + ets + b"".join(blobs) + b"".join(table))
    return path
