"""Olympus FluoView OIF and OIB (FV1000 / FV1200 / FV3000 confocal).

Structure (as oiffile by Christoph Gohlke, BSD-3, reads it; "no
specification document is available"):

- **OIF** -- a composite format (docs/COMPOSITE.md, decomposition): the main
  settings file ``<name>.oif`` (an INI text, UTF-16 or UTF-8) and its storage
  folder ``<name>.oif.files/`` holding one TIFF per plane, named by its axes
  (``s_C001Z002T003.tif``: channel 1, z 2, time 3, 1-based), a ``.pty``
  settings file per plane, ``s_LUT<n>.lut`` colour tables, ``.roi`` and a
  ``.bmp`` thumbnail.  Opening any file of the folder, or the folder,
  opens the image (``owner``).
- **OIB** -- the same files inside one compound (OLE) file (``_cfb``):
  ``OibInfo.txt`` names each stream (``Stream00001`` ->
  ``Storage00001/s_C001Z001.tif``) and the main settings stream.

One image ``tczyx``, sizes from the main file's ``Axis <i> Parameters
Common`` (``AxisCode`` / ``MaxSize``), planes from their file names; a lambda
(spectral) scan (axis ``L``, planes ``s_C001L001.tif`` ...) is one image per
emission band (index ``emission``, its wavelength in ``metadata.index_values``
from the axis' start and step), as for OIR.  Reads
are exact: a plane's TIFF layout is parsed once (tifffile), then only the
strips holding the requested rows are read -- for OIB at their places in the
``.oib`` file (a stream's sectors mapped to file offsets).  A plane the
settings expect but the folder lacks reads as zeros, with a
``MissingImagesWarning``.

Metadata, as Bio-Formats reads it (checked on the OME FV1000 sample): pixel
size from ``Reference Image Parameter`` ``Width`` / ``HeightConvertValue``,
z from the Z axis' ``Interval``, t from the T axis' ``Interval`` (when
there is more than one time point); channel name ``CH Name``, emission and
excitation wavelengths, colour from the channel's LUT (its last entry, stored
B G R A).  Numbers are written in the acquiring PC's locale: ``0,207.0`` is
0.207 (decimal comma, a ``.0`` appended), so the comma becomes a point and
the leading number is taken.  No stage position.
"""
from __future__ import annotations

import builtins
import io
import os
import re
import threading
import warnings

import numpy as np

from .. import _cfb, _parts, _pool
from .._base import File, Image
from .._errors import CorruptFileError, MissingImagesWarning, UnsupportedFormatError
from .._io import ThreadHandles
from .._metadata import Quantity, Channel, Metadata, color_hex, quantity

NAME = "oif"
OIB_INFO = "OibInfo.txt"
STORAGE_SUFFIX = ".files"
#: a plane file: s_ then one or more <axis letter><3 digits>, .tif
PLANE = re.compile(r"^s_((?:[A-Za-z]\d{3})+)\.tif$", re.I)
PLANE_AXIS = re.compile(r"([A-Za-z])(\d{3})")
#: the axes micro-reader's image has, by their Olympus axis code
AXES = {"C": "c", "Z": "z", "T": "t"}
#: the lambda axis of a spectral scan: one image per band (index ``emission``)
LAMBDA = "L"
LUT_MARKER = "[ColorLUTData]"
LEADING_NUMBER = re.compile(r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?")


# -- settings files ----------------------------------------------------------------

def settings(data: bytes) -> dict:
    """An Olympus settings file (INI; UTF-16 with BOM, or UTF-8) as
    {section: {key: value}}; values are strings, quotes removed.  A
    ``[ColorLUTData]`` section (binary) ends the text."""
    if data[:2] == b"\xff\xfe":
        cut = data.find(LUT_MARKER.encode("utf-16-le"))
        text = data[2:cut if cut >= 0 else None].decode("utf-16-le", "replace")
    else:
        cut = data.find(LUT_MARKER.encode())
        text = data[:cut if cut >= 0 else None].decode("utf-8", "replace")
    out, section = {}, None
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("[") and line.endswith("]"):
            section = out.setdefault(line[1:-1], {})
        elif "=" in line and section is not None:
            key, value = line.split("=", 1)
            section[key.strip()] = value.strip().strip('"')
    return out


def number(value):
    """A number as the acquiring PC's locale wrote it (``0,207.0`` -> 0.207,
    ``510`` -> 510.0); None if there is none."""
    if value is None:
        return None
    match = LEADING_NUMBER.search(str(value).replace(",", "."))
    return float(match.group(0)) if match else None


def lut_color(data: bytes):
    """'#RRGGBB' of a LUT file: its last entry, stored B G R A."""
    for marker in (LUT_MARKER.encode("utf-16-le") + "\r\n".encode("utf-16-le"),
                   (LUT_MARKER + "\r\n").encode()):
        at = data.rfind(marker)
        if at >= 0:
            table = data[at + len(marker):]
            if len(table) >= 4 and len(table) % 4 == 0:
                b, g, r, _ = table[-4:]
                return color_hex((r, g, b))
    return None


# -- where the files are ---------------------------------------------------------------

class _OibFiles:
    """The files of an OIB: streams of its compound file, by their OIF names."""

    format = "oib"

    def __init__(self, path: str):
        self.path = path
        self._cfb = _cfb.CompoundFile(path)
        info_stream = self._cfb.streams.get(OIB_INFO)
        if info_stream is None:
            raise UnsupportedFormatError(f"{path!r}: a compound file without {OIB_INFO} (no OIB)")
        info = settings(info_stream.read_all()).get("OibSaveInfo", {})
        self._streams = {}                       # OIF name (``s_C001Z001.tif``) -> Stream
        main = info.get("MainFileName")
        for key, value in info.items():
            if not key.startswith("Stream"):
                continue
            storage, _, name = value.rpartition("/")
            stream = self._cfb.streams.get(f"{storage}/{key}" if storage else key)
            if stream is not None:
                self._streams[name if storage else value] = stream
                if key == main:
                    self.main_name = value
        if main is None or not hasattr(self, "main_name"):
            raise CorruptFileError(f"{path!r}: {OIB_INFO} names no main settings stream")
        self._handles = ThreadHandles(path)

    def names(self) -> list:
        return list(self._streams)

    def read(self, name: str) -> bytes:
        """A whole (small) file: settings, LUTs."""
        stream = self._streams[name]
        if stream.data is not None:
            return stream.data
        return b"".join(self._handles.read(at, n) for at, n in stream.pieces(0, stream.size))

    def size(self, name: str) -> int:
        return self._streams[name].size

    def pieces(self, name: str, offset: int, size: int):
        """(handles, file offset, length) holding bytes offset:offset+size of *name*."""
        stream = self._streams[name]
        if stream.data is not None:                  # a small plane, held in the mini stream
            return [(_InMemory(stream.data), offset, size)]
        return [(self._handles, at, n) for at, n in stream.pieces(offset, size)]

    def main(self) -> bytes:
        return self.read(self.main_name)

    def close(self) -> None:
        self._handles.close()


class _InMemory:
    """Positioned reads of bytes already in memory (as ``ThreadHandles.read``)."""

    def __init__(self, data: bytes):
        self._data = data

    def read(self, offset: int, size: int) -> bytes:
        if offset + size > len(self._data):
            raise CorruptFileError(f"bytes {offset}:{offset + size} of a {len(self._data)}-byte stream")
        return self._data[offset:offset + size]


class _OifFiles:
    """The files of an OIF: its storage folder."""

    format = "oif"

    def __init__(self, path: str):
        self.path = path
        folder = _parts.find(os.path.dirname(path), os.path.basename(path) + STORAGE_SUFFIX)
        if folder is None or not os.path.isdir(folder):
            raise CorruptFileError(f"{path!r}: its storage folder "
                                   f"{os.path.basename(path) + STORAGE_SUFFIX!r} is missing")
        self.folder = folder
        self._files = {e: os.path.join(folder, e) for e in sorted(os.listdir(folder))}
        self._handles = _parts.Handles()

    def names(self) -> list:
        return list(self._files)

    def read(self, name: str) -> bytes:
        with builtins.open(self._files[name], "rb") as fh:
            return fh.read()

    def size(self, name: str) -> int:
        return os.path.getsize(self._files[name])

    def pieces(self, name: str, offset: int, size: int):
        return [(self._handles.get(self._files[name]), offset, size)]

    def main(self) -> bytes:
        with builtins.open(self.path, "rb") as fh:
            return fh.read()

    def close(self) -> None:
        self._handles.close()


class _PlaneFile(io.RawIOBase):
    """A plane TIFF of a file set, as a file object (for tifffile to parse
    its header and IFD; the pixels are read by ``_OifImage``)."""

    def __init__(self, files, name: str):
        self._files, self._name = files, name
        self._size, self._at = files.size(name), 0

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self._at

    def seek(self, offset: int, whence: int = 0) -> int:
        self._at = {0: offset, 1: self._at + offset, 2: self._size + offset}[whence]
        return self._at

    def readinto(self, buf) -> int:
        n = max(0, min(len(buf), self._size - self._at))
        view, done = memoryview(buf).cast("B"), 0
        for handles, at, length in (self._files.pieces(self._name, self._at, n) if n else []):
            view[done:done + length] = handles.read(at, length)
            done += length
        self._at += n
        return n


class _Layout:
    """How one plane TIFF stores its pixels (parsed once)."""

    def __init__(self, files, name: str):
        import tifffile
        with tifffile.TiffFile(_PlaneFile(files, name), name=name) as tif:
            page = tif.pages.first
            if page.shaped[0] != 1 or page.shaped[1] != 1 or page.shaped[-1] != 1:
                raise UnsupportedFormatError(f"{name!r}: plane TIFF with samples or depth "
                                             f"{page.shaped}")
            self.rows, self.cols = page.imagelength, page.imagewidth
            self.dtype = np.dtype(page.dtype).newbyteorder(tif.byteorder)
            self.raw = (page.compression == 1 and page.predictor == 1 and page.fillorder == 1
                        and page.bitspersample == 8 * self.dtype.itemsize and not page.is_tiled)
            self.offsets, self.counts = page.dataoffsets, page.databytecounts
            self.rows_per_strip = page.rowsperstrip or self.rows


# -- the image ----------------------------------------------------------------------------

class _OifImage(Image):

    def __init__(self, files, path: str, head: dict, sizes: dict, planes: dict, first: _Layout,
                 number: int = 0, emission=None):
        shape = (sizes["t"], sizes["c"], sizes["z"], first.rows, first.cols)
        super().__init__(number, "tczyx", shape, first.dtype.newbyteorder("="),
                         name=os.path.splitext(os.path.basename(path))[0])
        self._files, self._path, self._head = files, path, head
        self._planes = planes                    # (t, c, z) -> plane file name
        self._emission = emission                # the band of a lambda scan (Quantity)
        self._layouts = {planes[min(planes)]: first} if planes else {}
        self._lock = threading.Lock()
        self._first = first
        self._closed = False

    def _layout(self, name: str) -> _Layout:
        layout = self._layouts.get(name)
        if layout is None:
            layout = _Layout(self._files, name)
            if (layout.rows, layout.cols, layout.dtype) != (self._first.rows, self._first.cols,
                                                             self._first.dtype):
                raise CorruptFileError(f"{self._path!r}: plane {name!r} is "
                                       f"{layout.rows} x {layout.cols} {layout.dtype}, the "
                                       f"first {self._first.rows} x {self._first.cols} "
                                       f"{self._first.dtype}")
            with self._lock:
                self._layouts[name] = layout
        return layout

    def _read_block(self, block):
        if self._closed:
            raise ValueError("this image has been closed: its file was closed "
                             "(File.close() or the end of a with block)")
        ts, cs, zs, ys, xs = block
        out = np.empty(tuple(s.stop - s.start for s in block), self.dtype)

        def tasks():
            for t in range(ts.start, ts.stop):
                for c in range(cs.start, cs.stop):
                    for z in range(zs.start, zs.stop):
                        dest = out[t - ts.start, c - cs.start, z - zs.start]
                        name = self._planes.get((t, c, z))
                        if name is None:
                            dest[...] = 0                # missing plane (warned at open)
                            continue
                        yield lambda name=name, dest=dest: self._read_plane(name, ys, xs, dest)
        return out, _pool.run(tasks())

    def _read_plane(self, name: str, ys: slice, xs: slice, dest: np.ndarray) -> int:
        layout = self._layout(name)
        if not layout.raw:
            import tifffile
            with tifffile.TiffFile(_PlaneFile(self._files, name), name=name) as tif:
                dest[...] = tif.pages.first.asarray()[ys, xs]
            return layout.rows * layout.cols
        item = layout.dtype.itemsize
        row_bytes = layout.cols * item
        rps = layout.rows_per_strip
        read = 0
        for strip in range(ys.start // rps, (ys.stop - 1) // rps + 1):
            r0, r1 = max(ys.start, strip * rps), min(ys.stop, (strip + 1) * rps)
            offset = layout.offsets[strip] + (r0 - strip * rps) * row_bytes
            size = (r1 - r0) * row_bytes
            if offset + size > layout.offsets[strip] + layout.counts[strip]:
                raise CorruptFileError(f"{self._path!r}: plane {name!r} strip {strip} is short")
            buf = bytearray(size)
            view, done = memoryview(buf), 0
            for handles, at, length in self._files.pieces(name, offset, size):
                view[done:done + length] = handles.read(at, length)
                done += length
            rows = np.frombuffer(buf, layout.dtype).reshape(r1 - r0, layout.cols)
            dest[r0 - ys.start:r1 - ys.start] = rows[:, xs]
            read += size // item
        return read

    def _read_metadata(self):
        head = self._head
        ref = head.get("Reference Image Parameter", {})
        sizes = {}
        for axis, key, unit in (("x", "WidthConvertValue", "WidthUnit"),
                                ("y", "HeightConvertValue", "HeightUnit")):
            q = quantity(number(ref.get(key)), ref.get(unit))
            if q:
                sizes[axis] = q
        for code, axis, n in (("Z", "z", self.shape[2]), ("T", "t", self.shape[0])):
            info = _axis(head, code)
            if info is None or n < 2:
                continue
            q = _axis_step(info, n)
            if q:
                sizes[axis] = q
        channels = []
        names = set(self._files.names())
        for c in range(self.shape[1]):
            params = head.get(f"Channel {c + 1} Parameters", {})
            lut = f"s_LUT{c + 1}.lut"
            channels.append(Channel(
                name=params.get("CH Name") or None,
                color=lut_color(self._files.read(lut)) if lut in names else None,
                emission_wavelength=_wavelength(params.get("EmissionWavelength")),
                excitation_wavelength=_wavelength(params.get("ExcitationWavelength"))))
        index_values = {"emission": self._emission} if self._emission else {}
        return Metadata(pixel_sizes=sizes, channels=tuple(channels), index_values=index_values)

    def close(self) -> None:
        self._closed = True


def _axis_step(info: dict, n: int):
    """The step of an axis of *n* elements: its ``Interval`` in ``PixUnit``, or --
    when that is 0 or missing (a time series whose frames follow each other as
    fast as the scan allows: FV writes the total in ``EndPosition``) -- the span
    from ``StartPosition`` to ``EndPosition`` over ``n - 1`` steps, in ``UnitName``
    (``PixUnit``), as Bio-Formats computes it."""
    step = number(info.get("Interval"))
    if step:
        return quantity(step, info.get("PixUnit"))
    start, end = number(info.get("StartPosition")), number(info.get("EndPosition"))
    if start is None or end is None or n < 2:
        return None
    return quantity((end - start) / (n - 1), info.get("UnitName") or info.get("PixUnit"))


def _axis(head: dict, code: str):
    for k in range(16):
        info = head.get(f"Axis {k} Parameters Common")
        if info is None:
            break
        if info.get("AxisCode") == code:
            return info
    return None


def _wavelength(value):
    v = number(value)
    return quantity(v, "nanometer") if v and v > 0 else None


class _OifFile(File):
    backend = NAME

    def __init__(self, path: str, files, images):
        self.format = files.format
        super().__init__(path, images)
        self._files = files

    def close(self) -> None:
        super().close()
        self._files.close()


# -- the backend protocol -----------------------------------------------------------------

def sniff(head: bytes, path: str) -> bool:
    lower = path.lower()
    if head[:8] == _cfb.MAGIC:
        if lower.endswith(".oib"):
            return True
        try:                                     # an OIB by another name: its OibInfo.txt
            return OIB_INFO in _cfb.CompoundFile(path).streams
        except Exception:                        # noqa: BLE001 - any other OLE file
            return False
    if lower.endswith(".oif"):
        text = head[2:] if head[:2] == b"\xff\xfe" else head
        return text[:1] == b"[" or text[:2] == "[".encode("utf-16-le")
    return False


def may_own(folder: str, name: str, folders) -> bool:
    """``group_files``' cheap test (reads nothing): a file in a
    ``<name>.oif.files`` folder may be a part."""
    return folders.base(folder).lower().endswith(".oif" + STORAGE_SUFFIX)


def owner(path: str, head):
    """The files of an OIF storage folder, and the folder, open the OIF
    (docs/COMPOSITE.md); a plane TIFF is an image on its own (read alone,
    with a warning, when the ``.oif`` is missing)."""
    if os.path.isdir(path):
        folder, part = os.path.dirname(os.path.abspath(path)), os.path.basename(os.path.abspath(path))
        standalone = False
    else:
        storage = os.path.dirname(os.path.abspath(path))
        folder, part = os.path.dirname(storage), os.path.basename(storage)
        standalone = os.path.basename(path).lower().endswith((".tif", ".tiff"))
    if not part.lower().endswith(".oif" + STORAGE_SUFFIX):
        return None
    main = part[:-len(STORAGE_SUFFIX)]
    return _parts.Part(_parts.find(folder, main), os.path.join(folder, main), standalone)


def open(path: str, tiles: str = "separate", resolutions: str = "base") -> _OifFile:  # noqa: A001 - backend protocol
    files = _OibFiles(path) if path.lower().endswith(".oib") or \
        _parts.read_head(path)[:8] == _cfb.MAGIC else _OifFiles(path)
    try:
        head = settings(files.main())
        sizes = {"t": 1, "c": 1, "z": 1, "l": 1}
        bands = None                             # the lambda axis' settings
        for k in range(16):
            info = head.get(f"Axis {k} Parameters Common")
            if info is None:
                break
            code, size = info.get("AxisCode"), int(number(info.get("MaxSize")) or 0)
            if code in AXES:
                sizes[AXES[code]] = max(1, size)
            elif code == LAMBDA and size > 1:
                sizes["l"], bands = size, info
            elif code not in ("X", "Y") and size > 1:
                raise UnsupportedFormatError(
                    f"{path!r}: an axis {code!r} ({info.get('AxisName')}) of {size}: not supported yet")
        planes = {}
        for name in files.names():
            match = PLANE.match(name)
            if match is None:
                continue
            index = {"t": 0, "c": 0, "z": 0, "l": 0}
            for letter, value in PLANE_AXIS.findall(match.group(1)):
                axis = "l" if letter.upper() == LAMBDA else AXES.get(letter.upper())
                if axis is None:
                    raise UnsupportedFormatError(f"{path!r}: plane {name!r}: axis {letter!r}")
                index[axis] = int(value) - 1
            if any(index[a] >= sizes[a] for a in index):
                raise CorruptFileError(f"{path!r}: plane {name!r} lies beyond the sizes the "
                                       f"settings give ({sizes})")
            planes[(index["t"], index["c"], index["z"], index["l"])] = name
        if not planes:
            raise CorruptFileError(f"{path!r}: no plane TIFF (s_C001Z001.tif ...)")
        expected = sizes["t"] * sizes["c"] * sizes["z"] * sizes["l"]
        if len(planes) < expected:
            warnings.warn(MissingImagesWarning(
                f"{path!r}: {expected - len(planes)} of {expected} planes are missing: they "
                "read as zeros"), stacklevel=3)
        first = _Layout(files, planes[min(planes)])
        images = []
        for band in range(sizes["l"]):
            own = {key[:3]: name for key, name in planes.items() if key[3] == band}
            if not own:
                continue                         # a band without any plane: left out
            emission = None
            if bands is not None:
                # the band's wavelength: the axis' start plus its step per band
                first_band = quantity(number(bands.get("StartPosition")),
                                      bands.get("UnitName") or bands.get("PixUnit"), positive=False)
                step = _axis_step(bands, sizes["l"])
                if first_band and step:
                    emission = Quantity(first_band.value + band * step.to(first_band.unit).value,
                                        first_band.unit)
            image = _OifImage(files, path, head, sizes, own, first, number=len(images),
                              emission=emission)
            if sizes["l"] > 1:
                image.index = {"series": 0, "emission": band}
            images.append(image)
    except BaseException:
        files.close()
        raise
    return _OifFile(path, files, images)
