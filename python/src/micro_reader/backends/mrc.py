"""MRC (electron / light microscopy and cryo-EM volumes).

Written from the public MRC2014 specification (CCP-EM): a 1024-byte header,
an optional extended header of NSYMBT bytes, then raw pixels with X varying
fastest, then Y, then Z (sections).  Uncompressed, so any rectangle is read
exactly -- only the bytes it covers, straight into the result.  This
replaces a JVM / Bio-Formats read in eubi-bridge.

Supported: modes 0 (int8), 1 (int16), 2 (float32), 4 (complex64),
6 (uint16), 12 (float16); little or big endian; extended headers; the
standard axis order (MAPC, MAPR, MAPS) = (1, 2, 3).

Orientation: MRC stores rows bottom-up (origin at the lower left, as IMOD
displays it).  Images return them top-down, identical to Bio-Formats, so
switching eubi-bridge between the two never flips an image.  (mrcfile and a
raw memmap return file order: row 0 there is our last row.)
Not yet: mode 3 (complex int16), mode 101 (4-bit), permuted axis orders.
"""
from __future__ import annotations

import builtins
import os
import struct

import numpy as np

from .._base import File, Image
from .._errors import CorruptFileError, UnsupportedFormatError
from .._metadata import Channel, Metadata, quantity

NAME = "mrc"
HEADER_BYTES = 1024
CHUNK_BYTES = 4 << 20        # full-width rows are read this many bytes at a time
EXTENSIONS = {".mrc", ".mrcs", ".map", ".st", ".ali", ".rec"}

#: MRC2014 mode -> numpy dtype.  Mode 0 is *signed* int8 in MRC2014; older
#: writers used it unsigned, which cannot be told apart from the header.
MODES = {0: "i1", 1: "i2", 2: "f4", 4: "c8", 6: "u2", 12: "f2"}
UNSUPPORTED_MODES = {3: "complex int16", 101: "4-bit packed"}


def sniff(head: bytes, path: str) -> bool:
    """MRC2000+ files carry 'MAP ' at byte 208; older ones only the extension."""
    if len(head) >= 212 and head[208:212] == b"MAP ":
        return True
    return os.path.splitext(path)[1].lower() in EXTENSIONS


def _byte_order(head: bytes) -> str:
    """'<' or '>' from the machine stamp (byte 212), else by plausibility."""
    stamp = head[212] if len(head) > 212 else 0
    if stamp in (0x44, 0x41):            # 0x44 0x44 (or 0x44 0x41): little endian
        return "<"
    if stamp == 0x11:                    # 0x11 0x11: big endian
        return ">"
    # No stamp (older writers): the order in which the mode is a known one
    # and the dimensions are positive is the right one.
    for order in ("<", ">"):
        nx, ny, nz, mode = struct.unpack_from(order + "4i", head, 0)
        if mode in MODES or mode in UNSUPPORTED_MODES:
            if nx > 0 and ny > 0 and nz > 0:
                return order
    raise CorruptFileError("cannot determine the byte order of this MRC header")


class _MrcImage(Image):
    """The whole MRC volume as one image, axes 'zyx'."""

    def __init__(self, path: str, offset: int, shape, file_dtype: np.dtype, sizes=None):
        super().__init__(0, "zyx", shape, file_dtype, name=os.path.basename(path))
        if sizes is not None:
            self._metadata = Metadata(pixel_sizes=sizes, channels=(Channel(),))
        self._path = path
        self._offset = offset
        self._file_dtype = file_dtype
        self._closed = False

    def _read_block(self, block):
        # Plain reads straight into the result, one per row segment.  Not a
        # memory map: mapped pages count as the process's memory (on Windows
        # the whole file, after a full stream), which defeats the point of a
        # memory-bounded reader.  Each call opens its own handle, so reads
        # are thread-safe and nothing holds the file between reads.
        if self._closed:
            raise ValueError("this MRC image has been closed: its file was closed "
                             "(File.close() or the end of a with block)")
        zs, ys, xs = block
        nz, ny, nx = self.shape
        item = self._file_dtype.itemsize
        out = np.empty((zs.stop - zs.start, ys.stop - ys.start, xs.stop - xs.start),
                       dtype=self._file_dtype)
        # MRC's origin is the lower-left corner: the first row stored is the
        # bottom of the image.  Rows are returned top-down, as Bio-Formats
        # (and so eubi-bridge, Fiji, OME-Zarr viewers) presents them: display
        # row y is file row ny-1-y.
        with builtins.open(self._path, "rb", buffering=0) as fh:
            if xs.stop - xs.start == nx:
                self._read_full_rows(fh, zs, ys, out)
            else:
                length = out.shape[2] * item
                for i, z in enumerate(range(zs.start, zs.stop)):
                    for j, y in enumerate(range(ys.start, ys.stop)):
                        row = (z * ny + (ny - 1 - y)) * nx + xs.start
                        fh.seek(self._offset + row * item)
                        if fh.readinto(memoryview(out[i, j]).cast("B")) != length:
                            raise CorruptFileError(
                                f"{self._path!r}: short read at section {z}")
        if not self._file_dtype.isnative:
            out = out.byteswap(inplace=True).view(self.dtype)
        return out, out.size

    def _read_full_rows(self, fh, zs, ys, out):
        """Full-width rows of a section are contiguous in the file: read them
        in chunks of about CHUNK_BYTES and reverse each chunk into place.
        Still exact; the only extra memory is one chunk."""
        nz, ny, nx = self.shape
        item = self._file_dtype.itemsize
        per_chunk = max(1, CHUNK_BYTES // (nx * item))
        scratch = np.empty((min(per_chunk, ys.stop - ys.start), nx), dtype=self._file_dtype)
        first, stop = ny - ys.stop, ny - ys.start          # file rows, bottom-up
        for i, z in enumerate(range(zs.start, zs.stop)):
            for f in range(first, stop, per_chunk):
                n = min(per_chunk, stop - f)
                fh.seek(self._offset + (z * ny + f) * nx * item)
                if fh.readinto(memoryview(scratch[:n]).cast("B")) != n * nx * item:
                    raise CorruptFileError(f"{self._path!r}: short read at section {z}")
                top = ny - f - n - ys.start                   # display row of file row f+n-1
                out[i, top:top + n] = scratch[:n][::-1]

    def close(self) -> None:
        self._closed = True


class _MrcFile(File):
    backend = NAME


def open(path: str, tiles: str = "separate", resolutions: str = "base") -> _MrcFile:  # noqa: A001 - backend protocol name
    size = os.path.getsize(path)
    if size < HEADER_BYTES:
        raise CorruptFileError(f"{path!r} is smaller than an MRC header")
    with builtins.open(path, "rb") as fh:
        head = fh.read(HEADER_BYTES)

    order = _byte_order(head)
    nx, ny, nz, mode = struct.unpack_from(order + "4i", head, 0)
    mapc, mapr, maps = struct.unpack_from(order + "3i", head, 64)
    nsymbt, = struct.unpack_from(order + "i", head, 92)

    if mode in UNSUPPORTED_MODES:
        raise UnsupportedFormatError(
            f"MRC mode {mode} ({UNSUPPORTED_MODES[mode]}) is not supported yet")
    if mode not in MODES:
        raise CorruptFileError(f"unknown MRC mode {mode} in {path!r}")
    if min(nx, ny, nz) <= 0:
        raise CorruptFileError(f"invalid MRC dimensions {(nx, ny, nz)} in {path!r}")
    if (mapc, mapr, maps) not in ((1, 2, 3), (0, 0, 0)):
        raise UnsupportedFormatError(
            f"MRC axis order (MAPC, MAPR, MAPS) = {(mapc, mapr, maps)} is not "
            "supported yet; only the standard (1, 2, 3)")
    if nsymbt < 0:
        raise CorruptFileError(f"negative extended header size {nsymbt} in {path!r}")

    dtype = np.dtype(MODES[mode]).newbyteorder(order)
    offset = HEADER_BYTES + nsymbt
    needed = offset + nx * ny * nz * dtype.itemsize
    if size < needed:
        raise CorruptFileError(
            f"{path!r} is truncated: the header describes {needed} bytes, "
            f"the file has {size}")
    return _MrcFile(path, [_MrcImage(path, offset, (nz, ny, nx), dtype, _pixel_sizes(head, order))])


def _pixel_sizes(head: bytes, order: str) -> dict:
    """Cell size (CELLA, Å) over the sampling (MX, MY, MZ): Å per pixel,
    as MRC2014 defines it; kept in Å (OME-Zarr's ``angstrom``)."""
    mx, my, mz = struct.unpack_from(order + "3i", head, 28)
    cella = struct.unpack_from(order + "3f", head, 40)
    sizes = {}
    for axis, m, length in zip("xyz", (mx, my, mz), cella):
        q = quantity(length / m, "angstrom") if m > 0 else None
        if q:
            sizes[axis] = q
    return sizes
