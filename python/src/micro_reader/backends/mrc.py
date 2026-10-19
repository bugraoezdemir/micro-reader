"""MRC (electron / light microscopy and cryo-EM volumes).

Written from the public MRC2014 specification (CCP-EM): a 1024-byte header,
an optional extended header of NSYMBT bytes, then raw pixels with X varying
fastest, then Y, then Z (sections).  Uncompressed, so any rectangle is read
exactly -- only the bytes it covers, straight into the result.  This
replaces a JVM / Bio-Formats read in eubi-bridge.

Supported: modes 0 (int8), 1 (int16), 2 (float32), 4 (complex64),
6 (uint16), 12 (float16); little or big endian; extended headers; every
axis order.

Axis order: MAPC, MAPR, MAPS say which axis (1 = X, 2 = Y, 3 = Z) the
columns, rows and sections run along; the standard is (1, 2, 3).  The image
keeps the stored order and names it truthfully (Bugra, 2026-10-19): a map
with (3, 1, 2) -- columns along Z, rows along X, sections along Y -- has
axes ``yxz`` (``as_axes("zyx")`` gives the standard order).  Pixel sizes
belong to X, Y and Z whatever the order (CELLA and MX / MY / MZ are given per
axis).  Bio-Formats ignores MAPC / MAPR / MAPS: it names such a map's axes
wrongly.

Orientation: MRC stores Y bottom-up (origin at the lower left, as IMOD
displays it).  Images return Y top-down, identical to Bio-Formats for the
standard order, so switching eubi-bridge between the two never flips an
image; in a permuted map the Y axis is the one flipped, wherever it is
stored.  (mrcfile and a raw memmap return file order: row 0 there is our
last row.)

A unit cell with angles (CELLB) other than 90 degrees -- crystallographic
maps, e.g. MicroED -- has grid axes that are not perpendicular: read as
stored, with the true sampling distance per axis, and a SkewedGridWarning.
Not yet: mode 3 (complex int16), mode 101 (4-bit).
"""
from __future__ import annotations

import builtins
import os
import struct
import warnings

import numpy as np

from .._base import File, Image
from .._errors import CorruptFileError, SkewedGridWarning, UnsupportedFormatError
from .._metadata import Channel, Metadata, quantity

NAME = "mrc"
HEADER_BYTES = 1024
CHUNK_BYTES = 4 << 20        # full-width rows are read this many bytes at a time
SEPARATE_READ_BYTES = 32 << 10  # bytes worth skipping rather than issuing one more read
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
    """The whole MRC volume as one image, in its stored order (sections,
    rows, columns), each axis named by the file's MAPS, MAPR, MAPC."""

    def __init__(self, path: str, offset: int, shape, file_dtype: np.dtype, sizes=None,
                 axes: str = "zyx"):
        super().__init__(0, axes, shape, file_dtype, name=os.path.basename(path))
        if sizes is not None:
            self._metadata = Metadata(pixel_sizes=sizes, channels=(Channel(),))
        self._path = path
        self._offset = offset
        self._file_dtype = file_dtype
        #: the stored axis (0 sections, 1 rows, 2 columns) that is Y: stored
        #: bottom-up, returned top-down
        self._flip = axes.index("y")
        self._closed = False

    def _file_range(self, rng: slice, n: int, axis: int) -> tuple:
        """The file's indices of a requested range along stored *axis*."""
        return (n - rng.stop, n - rng.start) if axis == self._flip else (rng.start, rng.stop)

    def _sections(self, s0: int, s1: int):
        """(output index, file section) of the file sections s0..s1-1."""
        if self._flip == 0:
            return enumerate(range(s1 - 1, s0 - 1, -1))
        return enumerate(range(s0, s1))

    def _place(self, out_i, f: int, n: int, r0: int, r1: int, rows) -> None:
        """File rows f..f+n-1 of the file range r0..r1 -> their output rows of
        one section: reversed when Y is the rows, each row reversed when Y is
        the columns."""
        if self._flip == 1:
            top = r1 - f - n                                 # output row of file row f+n-1
            out_i[top:top + n] = rows[::-1]
        elif self._flip == 2:
            out_i[f - r0:f - r0 + n] = rows[:, ::-1]
        else:
            out_i[f - r0:f - r0 + n] = rows

    def _read_block(self, block):
        # Plain reads straight into the result, one per row segment.  Not a
        # memory map: mapped pages count as the process's memory (on Windows
        # the whole file, after a full stream), which defeats the point of a
        # memory-bounded reader.  Each call opens its own handle, so reads
        # are thread-safe and nothing holds the file between reads.
        if self._closed:
            raise ValueError("this MRC image has been closed: its file was closed "
                             "(File.close() or the end of a with block)")
        ss, rs, cs = block
        ns, nr, nc = self.shape
        item = self._file_dtype.itemsize
        out = np.empty((ss.stop - ss.start, rs.stop - rs.start, cs.stop - cs.start),
                       dtype=self._file_dtype)
        # MRC's origin is the lower-left corner: Y is stored bottom-up.  Y is
        # returned top-down, as Bio-Formats (and so eubi-bridge, Fiji, OME-Zarr
        # viewers) presents it: output index y is file index ny-1-y.
        files = [self._file_range(r, n, a) for a, (r, n) in enumerate(zip(block, self.shape))]
        read = out.size
        with builtins.open(self._path, "rb", buffering=0) as fh:
            if cs.stop - cs.start == nc:
                self._read_full_rows(fh, files, out)
            elif (rs.stop - rs.start > 1
                  and (nc - (cs.stop - cs.start)) * item <= SEPARATE_READ_BYTES):
                read = self._read_row_spans(fh, files, out)
            else:
                (s0, s1), (r0, r1), (c0, _) = files
                length = out.shape[2] * item
                for i, s in self._sections(s0, s1):
                    for f in range(r0, r1):
                        j = r1 - 1 - f if self._flip == 1 else f - r0
                        fh.seek(self._offset + ((s * nr + f) * nc + c0) * item)
                        if fh.readinto(memoryview(out[i, j]).cast("B")) != length:
                            raise CorruptFileError(
                                f"{self._path!r}: short read at section {s}")
                        if self._flip == 2:
                            out[i, j] = out[i, j][::-1]
        if not self._file_dtype.isnative:
            out = out.byteswap(inplace=True).view(self.dtype)
        return out, read

    def _read_row_spans(self, fh, files, out) -> int:
        """Part of each row, when the columns skipped per row are few: they cost less
        than one more read per row (as in LIF). Each chunk of rows is one read, from the
        first row's columns to the last row's, put into place (``_place``).
        Returns the values read."""
        ns, nr, nc = self.shape
        (s0, s1), (r0, r1), (c0, c1) = files
        item = self._file_dtype.itemsize
        width = c1 - c0
        per_chunk = max(1, CHUNK_BYTES // (nc * item))
        flat = np.empty(min(per_chunk, r1 - r0) * nc, dtype=self._file_dtype)
        read = 0
        for i, s in self._sections(s0, s1):
            for f in range(r0, r1, per_chunk):
                n = min(per_chunk, r1 - f)
                span = (n - 1) * nc + width
                fh.seek(self._offset + ((s * nr + f) * nc + c0) * item)
                if fh.readinto(memoryview(flat[:span]).cast("B")) != span * item:
                    raise CorruptFileError(f"{self._path!r}: short read at section {s}")
                rows = np.lib.stride_tricks.as_strided(flat, shape=(n, width),
                                                       strides=(nc * item, item))
                self._place(out[i], f, n, r0, r1, rows)
                read += span
        return read

    def _read_full_rows(self, fh, files, out):
        """Full-width rows of a section are contiguous in the file: read them
        in chunks of about CHUNK_BYTES and put each chunk into place.
        Still exact; the only extra memory is one chunk."""
        ns, nr, nc = self.shape
        (s0, s1), (r0, r1), _ = files
        item = self._file_dtype.itemsize
        per_chunk = max(1, CHUNK_BYTES // (nc * item))
        scratch = np.empty((min(per_chunk, r1 - r0), nc), dtype=self._file_dtype)
        for i, s in self._sections(s0, s1):
            for f in range(r0, r1, per_chunk):
                n = min(per_chunk, r1 - f)
                fh.seek(self._offset + (s * nr + f) * nc * item)
                if fh.readinto(memoryview(scratch[:n]).cast("B")) != n * nc * item:
                    raise CorruptFileError(f"{self._path!r}: short read at section {s}")
                self._place(out[i], f, n, r0, r1, scratch[:n])

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
    if (mapc, mapr, maps) == (0, 0, 0):                  # older writers leave them unset
        mapc, mapr, maps = 1, 2, 3
    if sorted((mapc, mapr, maps)) != [1, 2, 3]:
        raise CorruptFileError(
            f"MRC axis order (MAPC, MAPR, MAPS) = {(mapc, mapr, maps)} in {path!r} is not "
            "an order of the axes 1, 2, 3 (X, Y, Z)")
    axes = "".join("xyz"[m - 1] for m in (maps, mapr, mapc))   # sections, rows, columns
    if nsymbt < 0:
        raise CorruptFileError(f"negative extended header size {nsymbt} in {path!r}")

    dtype = np.dtype(MODES[mode]).newbyteorder(order)
    offset = HEADER_BYTES + nsymbt
    needed = offset + nx * ny * nz * dtype.itemsize
    if size < needed:
        raise CorruptFileError(
            f"{path!r} is truncated: the header describes {needed} bytes, "
            f"the file has {size}")
    angles = struct.unpack_from(order + "3f", head, 52)        # CELLB: alpha, beta, gamma
    if all(a > 0 for a in angles) and any(abs(a - 90.0) > 1e-3 for a in angles):
        warnings.warn(SkewedGridWarning(
            f"{path!r}: the unit cell's angles are {tuple(round(a, 3) for a in angles)} "
            "degrees, so the grid axes are not perpendicular; read as stored, with the "
            "true sampling distance along each axis"), stacklevel=3)
    # N(X, Y, Z) of the header count the columns, rows and sections
    return _MrcFile(path, [_MrcImage(path, offset, (nz, ny, nx), dtype, _pixel_sizes(head, order),
                                     axes)])


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
