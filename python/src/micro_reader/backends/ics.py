"""ICS / IDS: the Image Cytometry Standard (Dean et al. 1990; libics, BSD /
LGPL, svi-opensource/libics), written by SVI Huygens and others.

A text header (``.ics``): its first two characters are the field and line
separators, then one keyword line each, up to ``end``::

    ics_version     1.0 | 2.0
    layout          order   bits x y z t ch ...     (x varies fastest)
    layout          sizes   16 512 512 30 1 2
    layout          coordinates  video | cartesian
    representation  format  integer | real | complex
    representation  sign    signed | unsigned
    representation  compression  uncompressed | gzip
    representation  byte_order   1 2 (little endian) | 2 1 (big endian)
    parameter       scale / units   per dimension (bits first)
    sensor          s_params / s_states   microscope parameters (Huygens)
    history         ...

Pixels: ICS 1.0 keeps them in ``<name>.ids`` beside the header (a composite,
metadata + pixels: the ``.ids`` opens the ``.ics``; docs/COMPOSITE.md);
ICS 2.0 after the header's ``end`` line, or where ``source file`` says.
Huygens Remote Manager leaves ``<name>.parameters.txt``, ``<name>.log.txt``
and ``<name>.hgsb`` beside its results (side metadata: they open the image;
read for processing history later, docs/PROCESSED.md).

- Axes: x y z t and the channel names ``c`` / ``ch`` / ``channel(s)``, in the
  file's own order (fastest last); other dimensions of size 1 (Huygens' ``p``,
  the detector) are dropped, larger ones refused.
- ``coordinates cartesian``: rows are stored bottom-up (y upwards); they are
  returned top-down, as Bio-Formats does.  Checked 2026-10-19 on a Huygens
  result against its raw Leica input (flipped rows align at no shift) and on
  Bio-Formats 8 (it flips the two Huygens files, not libics' ``video`` one).
- Reads are exact (only the bytes a rectangle covers).  Gzip pixels are one
  deflate stream, read through ``_stream`` (micro-reader-codecs' streaming
  inflater): decoders keep their place, so forward reads continue, and a
  checkpoint index lets a jump resume near its target (measured 2026-10-19,
  1.27 GB: a random plane ~7x faster than restarting the stream, a whole
  volume in one pass).  Not read yet: complex pixels, Unix ``compress``
  (``.ids.Z``), gzip files of several members.
- Metadata: pixel sizes and the time step from ``parameter scale`` /
  ``units`` (``undefined`` / ``relative`` units: none); channel wavelengths
  from Huygens' ``LambdaEx`` / ``LambdaEm`` only when ``s_states`` says
  ``verified`` or ``reported`` (an ``estimated`` or ``default`` value is
  Huygens' guess).  No position (``parameter origin``'s convention is not
  verified).
"""
from __future__ import annotations

import builtins
import os
from typing import Optional

import numpy as np

from .. import _parts, _stream
from .._base import File, Image
from .._errors import CorruptFileError, UnsupportedFormatError
from .._metadata import Channel, Metadata, quantity

NAME = "ics"
MAGIC = b"ics_version"
#: the header is read up to its ``end`` line, at most this much
MAX_HEADER_BYTES = 4 << 20
#: ICS 1.0 pixel files beside the header (``.ids.Z``: Unix compress, refused)
PIXEL_SUFFIXES = (".ids", ".ids.gz", ".ids.z")
#: Huygens Remote Manager's files beside a result
SIDE_SUFFIXES = (".parameters.txt", ".log.txt", ".hgsb")
CHANNEL_NAMES = ("c", "ch", "channel", "channels")
AXES = {"x": "x", "y": "y", "z": "z", "t": "t", **{n: "c" for n in CHANNEL_NAMES}}
#: s_states that are the file's own values (not Huygens' estimate or default)
TRUSTED_STATES = ("verified", "reported")
#: bytes worth reading past rather than issuing one more read
SEPARATE_READ_BYTES = 64 << 10
#: whole planes are read this many bytes at a time
CHUNK_BYTES = 4 << 20


#: keywords a header line starts with: a ``.ics`` file without the separator
#: line and ``ics_version`` (seen: OME ICS/gh-4197, Bio-Formats reads it) is
#: recognised by them, read with tab / newline separators as version 1.0
TAB, NEWLINE = bytes([9]), bytes([10])
KEYWORDS = (b"ics_version", b"filename", b"layout", b"representation", b"parameter", b"sensor",
            b"history", b"source")


def _bare(head: bytes) -> bool:
    """A header without its separator line: it starts with a keyword and a tab."""
    return any(head.startswith(k + TAB) for k in KEYWORDS)


def sniff(head: bytes, path: str) -> bool:
    return MAGIC in head[:256] or (path.lower().endswith(".ics") and _bare(head))


def _stem(name: str, suffixes) -> Optional[str]:
    lower = name.lower()
    for suffix in suffixes:
        if lower.endswith(suffix):
            return name[:-len(suffix)]
    return None


def may_own(folder: str, name: str, folders) -> bool:
    """``group_files``' cheap test: a pixel file or a Huygens Remote Manager
    side file with an ``.ics`` of its name beside it."""
    stem = _stem(name, PIXEL_SUFFIXES + SIDE_SUFFIXES)
    return stem is not None and stem.lower() + ".ics" in folders.names(folder)


def owner(path: str, head):
    """``<name>.ids`` (ICS 1.0 pixels) and HRM's ``<name>.parameters.txt`` /
    ``.log.txt`` / ``.hgsb`` are parts of ``<name>.ics``.  A pixel file whose
    header is missing cannot be read (it says nothing about its pixels)."""
    name, folder = os.path.basename(path), os.path.dirname(path)
    pixels = _stem(name, PIXEL_SUFFIXES)
    stem = pixels if pixels is not None else _stem(name, SIDE_SUFFIXES)
    if stem is None:
        return None
    main = _parts.find(folder, stem + ".ics")
    if main is not None and sniff(_parts.read_head(main), main):
        return _parts.Part(main, main)
    if pixels is not None:
        return _parts.Part(None, os.path.join(folder, stem + ".ics"))
    return None


# -- the header -------------------------------------------------------------------

def read_header(path: str) -> tuple:
    """(entries, end): *entries* maps (category, keyword) -- for ``sensor``
    lines (sensor, keyword, name) -- to the list of values; ``history`` lines
    are kept in order under ("history",).  *end*: the byte after the ``end``
    line (ICS 2.0 pixels start there), None when there is none."""
    with builtins.open(path, "rb") as fh:
        raw = fh.read(MAX_HEADER_BYTES)
    if _bare(raw):
        field, line, pos = TAB, NEWLINE, 0
    elif len(raw) >= 3 and MAGIC in raw[:256]:
        field, line, pos = raw[0:1], raw[1:2], 2
    else:
        raise CorruptFileError(f"{path!r}: not an ICS header (no 'ics_version' line)")
    entries: dict = {("history",): []}
    end = None
    while pos < len(raw):
        stop = raw.find(line, pos)
        if stop < 0:
            stop = len(raw)
        text = raw[pos:stop].decode("latin-1").strip("\r")
        pos = stop + 1
        tokens = [t for t in text.split(field.decode("latin-1"))]
        tokens = [t.strip() for t in tokens]
        while tokens and not tokens[-1]:
            tokens.pop()
        if not tokens or not tokens[0]:
            continue
        category = tokens[0].lower()
        if category == "end":
            end = pos
            break
        if category == "history":
            entries[("history",)].append(tokens[1:])
            continue
        if category == "sensor" and len(tokens) > 2 and tokens[1].lower() in ("s_params", "s_states"):
            key, values = (category, tokens[1].lower(), tokens[2]), tokens[3:]
        elif len(tokens) > 1 and category in ("ics_version", "filename"):
            key, values = (category,), tokens[1:]
        elif len(tokens) > 1:
            key, values = (category, tokens[1].lower()), tokens[2:]
        else:
            continue                   # a line that is no keyword line (seen: a bare path)
        if key in entries and entries[key] != values:
            raise CorruptFileError(f"{path!r}: {' '.join(key)!r} given twice, differently "
                                   f"({entries[key]} and {values})")
        entries[key] = values
    return entries, end


def _one(entries: dict, key: tuple, path: str, default=None) -> Optional[str]:
    values = entries.get(key)
    if not values:
        if default is not None:
            return default
        raise CorruptFileError(f"{path!r}: no {' '.join(key)!r} line")
    return values[0].lower()


class _Layout:
    """What the header says about the pixels."""

    def __init__(self, entries: dict, path: str):
        order = [v.lower() for v in entries.get(("layout", "order"), [])]
        try:
            sizes = [int(v) for v in entries.get(("layout", "sizes"), [])]
        except ValueError as exc:
            raise CorruptFileError(f"{path!r}: layout sizes are not whole numbers") from exc
        if not order or order[0] != "bits" or len(order) != len(sizes) or min(sizes) < 1:
            raise CorruptFileError(f"{path!r}: layout order {order} / sizes {sizes} do not "
                                   "describe the pixels ('bits' first, one size each)")
        self.bits = sizes[0]
        kind = _one(entries, ("representation", "format"), path, "integer")
        sign = _one(entries, ("representation", "sign"), path,
                    "signed" if kind == "real" else "unsigned")
        if kind == "complex":
            raise UnsupportedFormatError(f"{path!r}: complex ICS pixels are not supported yet")
        codes = {("integer", "unsigned"): "u", ("integer", "signed"): "i",
                 ("real", "signed"): "f", ("real", "unsigned"): "f"}
        code = codes.get((kind, sign))
        if code is None or self.bits not in ((32, 64) if code == "f" else (8, 16, 32, 64)):
            raise UnsupportedFormatError(f"{path!r}: {sign} {kind} pixels of {self.bits} bits "
                                         "are not supported")
        item = self.bits // 8
        order_bytes = [int(v) for v in entries.get(("representation", "byte_order"), [])
                       if v.strip().isdigit()]
        if item == 1:
            endian = "|"
        elif order_bytes == list(range(1, item + 1)):
            endian = "<"
        elif order_bytes == list(range(item, 0, -1)):
            endian = ">"
        else:
            raise CorruptFileError(f"{path!r}: byte order {order_bytes or 'missing'} does not "
                                   f"fit {item}-byte pixels")
        self.dtype = np.dtype(f"{endian}{code}{item}")
        self.compression = _one(entries, ("representation", "compression"), path, "uncompressed")
        if self.compression not in ("uncompressed", "gzip"):
            raise UnsupportedFormatError(f"{path!r}: ICS compression {self.compression!r} is not "
                                         "supported (uncompressed and gzip are)")
        self.coordinates = _one(entries, ("layout", "coordinates"), path, "video")
        # dimensions, fastest first; dropped: other names of size 1
        self.dims = []                 # (axis letter, size, index in the order)
        for k, (name, n) in enumerate(zip(order[1:], sizes[1:]), 1):
            axis = AXES.get(name)
            if axis is None:
                if n == 1:
                    continue
                raise UnsupportedFormatError(f"{path!r}: dimension {name!r} of size {n} is not "
                                             "supported (x, y, z, t and channels are)")
            if any(a == axis for a, _, _ in self.dims):
                raise CorruptFileError(f"{path!r}: two dimensions are {axis!r} ({order})")
            self.dims.append((axis, n, k))
        if [a for a, _, _ in self.dims[:2]] != ["x", "y"]:
            raise UnsupportedFormatError(f"{path!r}: dimensions {order[1:]}: x and y must vary "
                                         "fastest")
        self.axes = "".join(a for a, _, _ in reversed(self.dims))
        self.shape = tuple(n for _, n, _ in reversed(self.dims))
        self.nbytes = int(np.prod(self.shape)) * item
        self.order = order


def _pixel_source(path: str, entries: dict, end: Optional[int], layout: _Layout) -> tuple:
    """(file, byte offset) of the pixels."""
    version = _one(entries, ("ics_version",), path, "1.0")
    folder, stem = os.path.dirname(path), os.path.splitext(os.path.basename(path))[0]
    source = entries.get(("source", "file"))
    if version.startswith("2"):
        if source:
            data = _parts.resolve(folder, source[0], path)
            offset = int(source[1]) if len(source) > 1 and source[1].isdigit() else 0
            return data, offset
        if end is None:
            raise CorruptFileError(f"{path!r}: an ICS 2.0 header without its 'end' line")
        return path, end
    for suffix in (".ids", ".ids.gz"):
        found = _parts.find(folder, stem + suffix)
        if found is not None:
            return found, 0
    if _parts.find(folder, stem + ".ids.Z") is not None:
        raise UnsupportedFormatError(f"{path!r}: its pixels are in {stem + '.ids.Z'!r}, Unix "
                                     "compress, which micro-reader does not read yet")
    raise CorruptFileError(f"{path!r}: its pixel file {stem + '.ids'!r} is missing")


# -- gzip pixels -------------------------------------------------------------------

def _deflate_span(path: str, offset: int, layout: "_Layout") -> tuple:
    """(offset, length) of the deflate stream of the gzip member at *offset*
    (RFC 1952: a 10-byte header, optional extra / name / comment / header
    CRC, the deflate data, an 8-byte trailer).  The trailer's size (ISIZE,
    the decoded bytes modulo 2**32) must be the pixels': several members, or
    a short stream, are refused."""
    size = os.path.getsize(path)
    with builtins.open(path, "rb") as fh:
        fh.seek(offset)
        head = fh.read(min(size - offset, 64 << 10))
        fh.seek(size - 4)
        isize = int.from_bytes(fh.read(4), "little")
    if len(head) < 18 or head[:2] != b"\x1f\x8b" or head[2] != 8:
        raise CorruptFileError(f"{path!r}: the pixels are not a gzip (deflate) stream")
    flags, at = head[3], 10
    try:
        if flags & 4:                                   # FEXTRA
            at += 2 + int.from_bytes(head[at:at + 2], "little")
        for bit in (8, 16):                             # FNAME, FCOMMENT: zero-terminated
            if flags & bit:
                at = head.index(b"\x00", at) + 1
        if flags & 2:                                   # FHCRC
            at += 2
    except ValueError as exc:
        raise CorruptFileError(f"{path!r}: a damaged gzip header") from exc
    if isize != layout.nbytes % (1 << 32):
        raise UnsupportedFormatError(
            f"{path!r}: the gzip stream decodes to {isize} bytes (mod 2**32), the pixels are "
            f"{layout.nbytes}: several gzip members or a cut stream are not read")
    return offset + at, size - 8 - (offset + at)


class _Inflated:
    """The decoded pixels of a gzip stream, file-like (seek / readinto), read
    through ``_stream``: decoders keep their place between reads and share a
    checkpoint index, so forward reads continue and jumps resume from the
    nearest checkpoint instead of the stream's start."""

    def __init__(self, path: str, offset: int, length: int, size: int):
        self._path, self._offset, self._length, self._size = path, offset, length, size
        self._at = 0

    def seek(self, at: int) -> None:
        self._at = int(at)

    def readinto(self, buf) -> int:
        out = np.frombuffer(buf, dtype=np.uint8)
        n = _stream.read(self._path, self._offset, self._length, "deflate", self._at, out,
                         size=self._size)
        self._at += n
        return n

    def close(self) -> None:
        pass


# -- the image -------------------------------------------------------------------

class _IcsImage(Image):
    """The whole ICS image: its dimensions in the file's order, fastest last."""

    def __init__(self, path: str, data: str, offset: int, layout: _Layout, entries: dict):
        super().__init__(0, layout.axes, layout.shape, layout.dtype,
                         name=os.path.splitext(os.path.basename(path))[0])
        self._data, self._offset, self._layout, self._entries = data, offset, layout, entries
        self._path = path
        self._flip = layout.coordinates == "cartesian"
        self._closed = False
        #: gzip: the deflate stream (offset, length) inside the gzip member
        self._deflate = _deflate_span(data, offset, layout) if layout.compression == "gzip" else None

    def _open(self):
        if self._deflate is not None:
            reader = _Inflated(self._data, *self._deflate, self._layout.nbytes)
            return reader, reader, 0
        fh = builtins.open(self._data, "rb", buffering=0)
        return fh, fh, self._offset

    def _read_block(self, block):
        if self._closed:
            raise ValueError("this ICS image has been closed: its file was closed "
                             "(File.close() or the end of a with block)")
        dtype = self._layout.dtype
        item = dtype.itemsize
        out = np.empty(tuple(s.stop - s.start for s in block), dtype=dtype)
        ny, nx = self.shape[-2:]
        plane = ny * nx * item
        lead = block[:-2]
        ys, xs = block[-2], block[-1]
        # stored rows of the block: bottom-up rows are flipped back on the way out
        fy0, fy1 = (ny - ys.stop, ny - ys.start) if self._flip else (ys.start, ys.stop)
        strides = [int(np.prod(self.shape[k + 1:-2])) for k in range(len(lead))]   # in planes
        reader, raw, base = self._open()
        read = 0
        try:
            whole = (fy0, fy1, xs.start, xs.stop) == (0, ny, 0, nx)
            if whole and lead:                          # whole planes: runs of the last lead axis
                last = lead[-1]
                per = max(1, CHUNK_BYTES // max(plane, 1))
                for index in np.ndindex(*(s.stop - s.start for s in lead[:-1])):
                    first = sum((i + s.start) * st for i, s, st in zip(index, lead[:-1], strides))
                    for k0 in range(last.start, last.stop, per):
                        k1 = min(last.stop, k0 + per)
                        buf = np.empty((k1 - k0, ny, nx), dtype=dtype)
                        self._fill(reader, base + (first + k0) * plane, buf)
                        out[index + (slice(k0 - last.start, k1 - last.start),)] = \
                            buf[:, ::-1] if self._flip else buf
                        read += buf.size
            else:
                for index in np.ndindex(*(s.stop - s.start for s in lead)):
                    p = sum((i + s.start) * st for i, s, st in zip(index, lead, strides))
                    read += self._read_rect(reader, base + p * plane, fy0, fy1, xs,
                                            out[index] if lead else out)
        finally:
            reader.close()
            if raw is not reader:
                raw.close()
        if not dtype.isnative:
            out = out.byteswap(inplace=True).view(self.dtype)
        return out, read

    def _read_rect(self, reader, plane_at: int, fy0: int, fy1: int, xs, out) -> int:
        """Stored rows fy0..fy1-1, columns xs, of the plane at *plane_at*, into
        *out* (display order).  One read from the first row's columns to the
        last's when the columns skipped are few, else one read per row."""
        nx = self.shape[-1]
        item = self._layout.dtype.itemsize
        width, n = xs.stop - xs.start, fy1 - fy0
        rows = out[::-1] if self._flip else out
        if (nx - width) * item <= SEPARATE_READ_BYTES or n == 1:
            span = (n - 1) * nx + width
            flat = np.empty(span, dtype=self._layout.dtype)
            self._fill(reader, plane_at + (fy0 * nx + xs.start) * item, flat)
            rows[...] = np.lib.stride_tricks.as_strided(flat, (n, width), (nx * item, item),
                                                        writeable=False)
            return span
        for j in range(n):
            self._fill(reader, plane_at + ((fy0 + j) * nx + xs.start) * item, rows[j])
        return n * width

    def _fill(self, reader, at: int, out: np.ndarray) -> None:
        buf = memoryview(np.ascontiguousarray(out) if not out.flags.c_contiguous else out).cast("B")
        reader.seek(at)
        done = 0
        while done < len(buf):
            got = reader.readinto(buf[done:])
            if not got:
                raise CorruptFileError(f"{self._data!r}: the pixels end early, at byte {at + done} "
                                       "(truncated file?)")
            done += got
        if not out.flags.c_contiguous:
            out[...] = np.frombuffer(buf, dtype=out.dtype).reshape(out.shape)

    def _read_metadata(self):
        lay, entries = self._layout, self._entries
        scale = entries.get(("parameter", "scale"), [])
        units = entries.get(("parameter", "units"), [])
        sizes = {}
        for axis, _, k in lay.dims:
            if axis not in "xyzt" or k >= len(scale) or k >= len(units):
                continue
            q = quantity(scale[k], units[k].strip() or None)
            if q is not None and (axis != "t" or self.shape[self.axes.index("t")] > 1):
                sizes[axis] = q
        n_c = self.shape[self.axes.index("c")] if "c" in self.axes else 1
        channels = []
        for c in range(n_c):
            values = {}
            for key, attr in (("LambdaEx", "excitation_wavelength"), ("LambdaEm", "emission_wavelength")):
                params = entries.get(("sensor", "s_params", key), [])
                states = [s.lower() for s in entries.get(("sensor", "s_states", key), [])]
                if len(params) == n_c and len(states) == n_c and states[c] in TRUSTED_STATES:
                    values[attr] = quantity(params[c], "nanometer")
            channels.append(Channel(**values))
        return Metadata(pixel_sizes=sizes, channels=tuple(channels))

    def close(self) -> None:
        self._closed = True
        if self._deflate is not None:
            _stream.forget(self._data)


class _IcsFile(File):
    backend = NAME

    def _side_files(self) -> set:
        image = self.images[0]
        folder, stem = os.path.dirname(self.path), os.path.splitext(os.path.basename(self.path))[0]
        out = {image._data} if image._data != self.path else set()
        for suffix in SIDE_SUFFIXES:
            found = _parts.find(folder, stem + suffix)
            if found is not None:
                out.add(found)
        return out


def open(path: str, tiles: str = "separate", resolutions: str = "base") -> _IcsFile:  # noqa: A001 - backend protocol
    entries, end = read_header(path)
    layout = _Layout(entries, path)
    data, offset = _pixel_source(path, entries, end, layout)
    if layout.compression == "uncompressed":
        have = os.path.getsize(data) - offset
        if have < layout.nbytes:
            raise CorruptFileError(f"{data!r} is truncated: the header {os.path.basename(path)!r} "
                                   f"describes {layout.nbytes} bytes of pixels, the file holds {have}")
    return _IcsFile(path, [_IcsImage(path, data, offset, layout, entries)])
