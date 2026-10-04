"""ND2 (Nikon, version 3) structure: chunk map, CLX-lite metadata, loops.

Only what is needed to locate pixels: where each frame is, how a frame is
laid out, and which (time, position, Z) each frame number stands for.

Ported, reduced, from nd2 (https://github.com/tlambert03/nd2), modules
``_parse/_chunk_decode.py``, ``_parse/_clx_lite.py`` and ``_parse/_parse.py``:

    Copyright (c) 2021 Talley Lambert

    Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions are met:

    1. Redistributions of source code must retain the above copyright notice,
       this list of conditions and the following disclaimer.
    2. Redistributions in binary form must reproduce the above copyright
       notice, this list of conditions and the following disclaimer in the
       documentation and/or other materials provided with the distribution.
    3. Neither the name of the copyright holder nor the names of its
       contributors may be used to endorse or promote products derived from
       this software without specific prior written permission.

    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
    IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
    THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
    PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
    CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
    EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
    PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
    PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
    LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
    NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
    SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""
from __future__ import annotations

import io
import struct
import zlib
from dataclasses import dataclass

from .._errors import CorruptFileError, UnsupportedFormatError

CHUNK_MAGIC = 0x0ABECEDA
JP2_MAGIC = 0x0C000000                       # legacy (version 1) ND2: JPEG 2000 based
FILE_SIGNATURE = b"ND2 FILE SIGNATURE CHUNK NAME01!"
CHUNKMAP_SIGNATURE = b"ND2 CHUNK MAP SIGNATURE 0000001!"
FILEMAP_SIGNATURE = b"ND2 FILEMAP SIGNATURE NAME 0001!"
CHUNK_HEADER = struct.Struct("<IIQ")         # magic, name length, data length
START_CHUNK = struct.Struct("<IIQ32s64s")    # header, name, data ("Ver3.0")
CHUNKMAP_LOCATION = struct.Struct("<32sQ")   # the last 40 bytes of the file

# experiment loop types (``eType``)
TIME_LOOP, XY_POS_LOOP, Z_STACK_LOOP, SPECT_LOOP, CUSTOM_LOOP, NE_TIME_LOOP = 1, 2, 4, 6, 7, 8


# -- file level ----------------------------------------------------------------

def version(head: bytes) -> tuple[int, int]:
    """(major, minor) from the first chunk; raises for anything else."""
    if len(head) < START_CHUNK.size:
        raise CorruptFileError("too short for an ND2 file")
    magic, name_len, data_len, name, data = START_CHUNK.unpack_from(head)
    if magic == JP2_MAGIC:
        return (1, 0)
    if magic != CHUNK_MAGIC or name != FILE_SIGNATURE:
        raise CorruptFileError("not an ND2 file")
    return int(chr(data[3])), int(chr(data[5]))


def chunkmap(fh) -> dict[bytes, tuple[int, int]]:
    """Chunk name -> (offset of its header, data size), from the end of file."""
    fh.seek(-CHUNKMAP_LOCATION.size, 2)
    sig, location = CHUNKMAP_LOCATION.unpack(fh.read(CHUNKMAP_LOCATION.size))
    if sig != CHUNKMAP_SIGNATURE:
        raise CorruptFileError("ND2 chunk map signature missing (truncated file?)")
    data = read_chunk(fh, location, expect=FILEMAP_SIGNATURE)
    out, pos = {}, 0
    while True:
        end = data.index(b"!", pos) + 1
        name = data[pos:end]
        if name == CHUNKMAP_SIGNATURE:
            return out
        offset, size = struct.unpack_from("<QQ", data, end)
        out[name] = (offset, size)
        pos = end + 16


def data_offset(fh, offset: int) -> tuple[int, int]:
    """(file offset of a chunk's data, its length) from the chunk header."""
    fh.seek(offset)
    magic, name_len, data_len = CHUNK_HEADER.unpack(fh.read(CHUNK_HEADER.size))
    if magic != CHUNK_MAGIC:
        raise CorruptFileError(f"no ND2 chunk at byte {offset}")
    return offset + CHUNK_HEADER.size + name_len, data_len


def read_chunk(fh, offset: int, expect: bytes | None = None) -> bytes:
    start, length = data_offset(fh, offset)
    if expect is not None:
        fh.seek(offset + CHUNK_HEADER.size)
        if not fh.read(len(expect)) == expect:
            raise CorruptFileError(f"expected ND2 chunk {expect!r} at byte {offset}")
    fh.seek(start)
    return fh.read(length)


# -- CLX-lite: Nikon's binary key/value metadata ---------------------------------

_BOOL, _INT32, _UINT32, _INT64, _UINT64, _DOUBLE, _VOIDPTR, _STRING, _BYTES, _DEPRECATED, \
    _LEVEL = range(1, 12)
_COMPRESS = 76
_SCALARS = {_BOOL: "B", _INT32: "i", _UINT32: "I", _INT64: "q", _UINT64: "Q",
            _DOUBLE: "d", _VOIDPTR: "Q"}


def _string(stream: io.BytesIO) -> str:
    value = stream.read(2)
    while not value.endswith(b"\x00\x00"):
        more = stream.read(2)
        if not more:
            break
        value += more
    try:
        return value.decode("utf16")[:-1]
    except UnicodeDecodeError:
        return value.decode("utf8", "replace")


def decode_clx_lite(data: bytes | io.BytesIO, count: int = 1) -> dict:
    """CLX-lite bytes -> nested dicts.  Keys keep their type prefix
    (``uiCount``); lists (items with empty names) become ``{"i0000000000":
    ...}`` dicts, as in Nikon's own JSON export and in nd2."""
    stream = data if isinstance(data, io.BytesIO) else io.BytesIO(data)
    out: dict = {}
    for _ in range(count):
        start = stream.tell()
        header = stream.read(2)
        if len(header) < 2:
            break
        kind, name_len = header
        if kind in (0, _DEPRECATED):
            raise CorruptFileError(f"unknown ND2 metadata type {kind}")
        if kind == _COMPRESS:
            stream.seek(10, 1)
            return decode_clx_lite(zlib.decompress(stream.read()))
        name = stream.read(name_len * 2).decode("utf16")[:-1]
        if kind == _LEVEL:
            items, length = struct.unpack("<IQ", stream.read(12))
            inner = decode_clx_lite(stream.read(length - (stream.tell() - start)), items)
            stream.seek(items * 8, 1)
            if len(inner) == 1 and "" in inner:
                values = inner[""] if isinstance(inner[""], list) else [inner[""]]
                value = {f"i{n:010}": v for n, v in enumerate(values)}
            else:
                value = inner
        elif kind == _BYTES:
            size, = struct.unpack("<Q", stream.read(8))
            value = stream.read(size)           # opaque; not needed to locate pixels
        elif kind == _STRING:
            value = _string(stream)
        elif kind in _SCALARS:
            fmt = "<" + _SCALARS[kind]
            value, = struct.unpack(fmt, stream.read(struct.calcsize(fmt)))
        else:
            value = None
        if name == "" and name in out:
            if not isinstance(out[name], list):
                out[name] = [out[name]]
            out[name].append(value)
        else:
            out[name] = value
    return out


# -- what the pixels need ----------------------------------------------------------

@dataclass
class Attributes:
    height: int
    width: int
    width_bytes: int
    components: int           # values per pixel: channels x RGB components
    channels: int
    bits_in_memory: int
    compression: str | None   # "lossless", "lossy", or None
    sequence_count: int

    @property
    def dtype(self) -> str:
        item = self.bits_in_memory // 8
        return f"<f{item}" if self.bits_in_memory == 32 else f"<u{item}"


def attributes(raw_attrs: dict, raw_meta: dict) -> Attributes:
    attrs = raw_attrs.get("SLxImageAttributes", raw_attrs)
    meta = raw_meta.get("SLxPictureMetadata", raw_meta)
    comp = attrs.get("eCompression", 2)
    tile_w, tile_h = attrs.get("uiTileWidth", 0), attrs.get("uiTileHeight", 0)
    if (0 < tile_w != attrs["uiWidth"]) or (0 < tile_h != attrs["uiHeight"]):
        raise UnsupportedFormatError(f"tiled ND2 frames ({tile_h} x {tile_w})")
    return Attributes(
        height=attrs["uiHeight"], width=attrs["uiWidth"],
        width_bytes=attrs["uiWidthBytes"], components=attrs["uiComp"],
        channels=meta.get("sPicturePlanes", {}).get("uiCount", 1) or 1,
        bits_in_memory=attrs["uiBpcInMemory"],
        compression=["lossless", "lossy", None][comp] if 0 <= comp < 2 else None,
        sequence_count=attrs["uiSequenceCount"],
    )


@dataclass
class Loop:
    kind: str                 # "T", "P" or "Z"
    count: int
    level: int = 0


def _single_loop(exp: dict) -> Loop | None:
    kind = exp.get("eType", 0)
    pars = exp.get("uLoopPars", {})
    if not pars:
        return None
    if list(pars) == ["i0000000000"]:
        pars = next(iter(pars.values()))
    if kind == TIME_LOOP:
        return Loop("T", pars.get("uiCount", 0)) if pars.get("uiCount", 0) else None
    if kind == XY_POS_LOOP:
        valid = exp.get("pItemValid", [])
        points = pars.get("Points")
        count = len(points) if isinstance(points, dict) else len(pars.get("dPosX", {}))
        if valid:
            flags = [v for _, v in sorted(valid.items())] if isinstance(valid, dict) else list(valid)
            if isinstance(flags, (bytes, bytearray)):
                flags = list(flags)
            count = sum(1 for _, ok in zip(range(count), flags) if ok)
        return Loop("P", count)
    if kind == Z_STACK_LOOP:
        return Loop("Z", pars.get("uiCount", 0))
    if kind == NE_TIME_LOOP:
        valid = pars.get("pPeriodValid", [])
        if isinstance(valid, dict):
            valid = [valid[k] for k in sorted(valid)]
        elif isinstance(valid, (bytes, bytearray)):
            valid = list(valid)
        count = sum(p.get("uiCount", 0) for p, ok in zip(pars["pPeriod"].values(), valid) if ok)
        return Loop("T", count)
    if kind == SPECT_LOOP:
        return Loop("S", 0)       # marker only: spectral planes are channels
    if kind == CUSTOM_LOOP:
        raise UnsupportedFormatError("ND2 with a custom experiment loop")
    raise UnsupportedFormatError(f"ND2 experiment loop type {kind}")


def experiment(raw: dict, level: int = 0, dest: list | None = None) -> list[Loop]:
    """The experiment loops, outermost first (frame numbers run over them
    in this order)."""
    raw = raw.get("SLxExperiment", raw) if level == 0 else raw
    dest = dest if dest is not None else []
    loop = _single_loop(raw)
    if loop is None or (loop.count == 0 and loop.kind != "S"):
        return dest
    if loop.kind == "S":
        level -= 1
    elif not dest or dest[-1].level < level:
        loop.level = level
        dest.append(loop)
    else:
        prev = dest[-1]
        if prev.level == level and prev.kind == loop.kind and prev.count < loop.count:
            loop.level = level
            dest[-1] = loop
    pars = raw.get("uLoopPars", {})
    if isinstance(pars, dict) and "pSubLoops" in pars:   # rare nested NE-time layout (nd2 #190)
        sub = pars["pSubLoops"].get("i0000000000", {}).get("ppNextLevelEx", {})
        if "i0000000000" in sub:
            dest.extend(experiment(sub["i0000000000"].get("SLxExperiment", {}), 0, []))
    for item in (raw.get("ppNextLevelEx") or {}).values():
        dest = experiment(item, level + 1, dest)
    return dest
