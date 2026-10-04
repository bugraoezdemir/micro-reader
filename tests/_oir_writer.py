"""A minimal OIR writer for tests: header, blocks, block index, UIDs and the
two XML documents the readers need (frame properties, lsmimage), modelled on
real FluoView files.  Optionally splits planes into a companion file.
"""
from __future__ import annotations

import os
import struct
import uuid

import numpy as np

MAGIC = b"OLYMPUSRAWFORMAT"
METADATA, FRAMEPROPERTIES, UID, PIXEL = 0, 1, 3, 4


def _frame_xml(width, height, dtype, frame_id="t001_0_1", positions=None):
    depth = np.dtype(dtype).itemsize
    values = "".join(
        f"<commonframe:axisValue><commonframe:axisType>{k}</commonframe:axisType>"
        f"<commonframe:position>{v}</commonframe:position></commonframe:axisValue>"
        for k, v in (positions or {}).items())
    return (
        '<?xml version="1.0" encoding="ASCII"?>'
        '<lsmframe:frameProperties xmlns:lsmframe="http://www.olympus.co.jp/hpf/model/lsmframe" '
        'xmlns:commonframe="http://www.olympus.co.jp/hpf/model/commonframe" '
        f'xmlns:base="http://www.olympus.co.jp/hpf/model/base" id="{frame_id}">'
        '<commonframe:imageDefinition>'
        '<base:colorType>GlayScale</base:colorType>'
        f'<base:width>{width}</base:width><base:height>{height}</base:height>'
        f'<base:depth>{depth}</base:depth><base:bitCounts>{depth * 8}</base:bitCounts>'
        '</commonframe:imageDefinition>' + values + '</lsmframe:frameProperties>').encode("ascii")


def _lsm_xml(channels, axes):
    """channels: [(guid, order)]; axes: {axis type: max size}."""
    chans = "".join(
        f'<commonphase:channel id="{g}" order="{o}"><commonphase:name>CH{o}</commonphase:name>'
        '</commonphase:channel>' for g, o in channels)
    axs = "".join(
        f'<commonparam:axis enable="true"><commonparam:axis>{kind}</commonparam:axis>'
        f'<commonparam:maxSize>{n}</commonparam:maxSize></commonparam:axis>'
        for kind, n in axes.items())
    return (
        '<?xml version="1.0" encoding="ASCII"?>'
        '<lsmimage:imageProperties xmlns:lsmimage="http://www.olympus.co.jp/hpf/model/lsmimage" '
        'xmlns:commonphase="http://www.olympus.co.jp/hpf/model/commonphase" '
        f'xmlns:commonparam="http://www.olympus.co.jp/hpf/model/commonparam">{chans}{axs}'
        '</lsmimage:imageProperties>').encode("ascii")


class _File:
    def __init__(self):
        self.data = bytearray(MAGIC + struct.pack("<IIII", 12, 0, 1, 2) + b"\0" * 16)
        self.offsets = []

    def block(self, kind, payload: bytes, header_extra: bytes = b""):
        self.offsets.append(len(self.data))
        self.data += struct.pack("<II", len(payload), kind) + header_extra + payload

    def uid_pixel(self, uid: str, pixels: bytes):
        raw = uid.encode("ascii")
        self.block(UID, raw, struct.pack("<III", 0, 0, len(raw)))
        self.block(PIXEL, pixels)

    def save(self, path):
        index_pos = len(self.data)
        self.data += b"\xff\xff\xff\xff" + b"".join(struct.pack("<Q", o) for o in self.offsets)
        self.data += struct.pack("<Q", 0)
        self.data[32:48] = struct.pack("<QQ", len(self.data), index_pos)
        with open(path, "wb") as fh:
            fh.write(self.data)


def write_oir(path, data: np.ndarray, *, block_bytes: int = 4096, channel_order=None,
              axes=None, missing=(), truncate=(), companion_planes=(), lambda_index=None,
              extra_rows=0, frame_axes=None):
    """*data*: (T, C, Z, Y, X), or (L, T, C, Z, Y, X) for a lambda scan.  Each plane is stored as pixel blocks of
    *block_bytes* (not necessarily whole rows).

    channel_order: lsmimage ``order`` per channel (defaults 1..C); channels
    are written to the file in a different order to check that the reader
    follows the metadata.  missing: (t, c, z) planes left out.  truncate:
    planes whose last block is cut in half.  companion_planes: planes stored
    in the companion file ``<name>_00001`` instead.  lambda_index: write
    planes with an ``l<n>`` UID prefix.  extra_rows: append rows beyond the
    frame height (as a concatenated line scan would).  frame_axes: {axis
    type: f(l, t, z)}: frame properties for every frame with its position
    along those axes (as FluoView writes them: TIMELAPSE ms, LAMBDA nm).
    """
    lambdas = data.shape[0] if data.ndim == 6 else None
    if lambdas is None:
        data = data[None]
    nt, nc, nz, height, width = data.shape[1:]
    guids = [str(uuid.UUID(int=0x1000 + c)) for c in range(nc)]
    order = list(channel_order or range(1, nc + 1))
    if axes is None:
        axes = {}
        if nt > 1:
            axes["TIMELAPSE"] = nt
        if nz > 1:
            axes["ZSTACK"] = nz
    main, companion = _File(), _File()
    main.block(METADATA, _lsm_xml(list(zip(guids, order)), axes))
    main.block(FRAMEPROPERTIES, _frame_xml(width, height, data.dtype))
    # a reference image block, which readers must ignore
    main.uid_pixel(f"REF_LSM0_{guids[0]}_0", b"\x07" * 64)
    for lam in range(data.shape[0]):
        index = lam + 1 if lambdas else lambda_index
        for t in range(nt):
            for z in range(nz):
                if frame_axes:
                    fid = (f"l{index:03d}" if index is not None else "") + \
                        (f"z{z + 1:03d}" if nz > 1 else "") + f"t{t + 1:03d}_0_1"
                    main.block(FRAMEPROPERTIES, _frame_xml(
                        width, height, data.dtype, fid,
                        {k: f(lam, t, z) for k, f in frame_axes.items()}))
                for c in reversed(range(nc)):               # file order != channel order
                    _write_plane(main, companion, data[lam, t, c, z], (t, c, z), guids[c],
                                 (f"l{index:03d}" if index is not None else "")
                                 + (f"z{z + 1:03d}" if nz > 1 else "") + f"t{t + 1:03d}",
                                 block_bytes, missing, truncate, companion_planes, extra_rows)
    main.save(path)
    if companion.offsets:
        companion.save(f"{os.path.splitext(path)[0]}_00001")


def _write_plane(main, companion, plane, tcz, guid, prefix, block_bytes, missing, truncate,
                 companion_planes, extra_rows):
    if tcz in missing:
        return
    width = plane.shape[1]
    plane = np.ascontiguousarray(plane).astype(plane.dtype.newbyteorder("<"))
    raw = plane.tobytes() + b"\x05" * (extra_rows * width * plane.dtype.itemsize)
    pieces = [raw[i:i + block_bytes] for i in range(0, len(raw), block_bytes)]
    if tcz in truncate:
        pieces[-1] = pieces[-1][:len(pieces[-1]) // 2]
    target = companion if tcz in companion_planes else main
    for b, piece in enumerate(pieces):
        target.uid_pixel(f"{prefix}_0_1_{guid}_{b}", piece)

