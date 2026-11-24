"""Nikon ND2 (version 3 and later; legacy versions 1 / 2 in ``_nd2_legacy``), own reader.

A frame is one (time, position, Z) plane of all channels, stored in a chunk
``ImageDataSeq|<n>!`` after an 8-byte timestamp: ``height`` rows of
``width_bytes`` bytes (rows may be padded), each row ``width`` pixels of all
channels and RGB components interleaved.  Frame numbers run over the
experiment loops, outermost first.  Structure parsing is ported from nd2
(Talley Lambert, BSD-3), see ``_nd2_structure``.

- Uncompressed frames: only the requested rows are read (all channels
  come along: they are interleaved per pixel).
- "Lossless" frames are one zlib stream each: decoded whole, cached.

Images are the XY positions, indexed by ``series`` (bioio-nd2 calls them scenes); axes ``tczyx``, plus ``s``
for RGB components, read in R, G, B(, A) order from the order the camera block declares
(``stored_sample_order``: Nikon colour cameras store B, G, R; a file that does not state it
is read as stored, with a warning).  With ``plates=True``, a file whose every XY point is
named after a well as NIS-Elements names them (``A10_0000``, ``B02_0003``,
``E8.1``: JOBS and ND acquisitions over a well plate; a counter after the well
or a zero-padded column, ``NIS_WELL_POINT``, so that ``P1``, ``S2`` stay
positions) reads as a plate: images indexed ``plate`` / ``acquisition`` / ``well`` / ``field``, a
field being the point's place among its well's points, in acquisition order.
The layout is the well plate of the NIS-Elements job that acquired the file
(its ``JobDefinitionV1_0`` chunk), else the smallest standard plate holding
the wells (``_plate.standard_layout``), with a PlateLayoutWarning; the plate's
name the file's.  A folder of files of one well-plate run, one per well or per
point (``WellB05_ChannelDIA_Seq0000.nd2``), reads as one plate with
``plates=True``: ``_nd2_run``.  Legacy JPEG 2000 ND2 (versions 1 / 2): ``_nd2_legacy``.
Not supported:
lossy compression, tiled frames, custom loops.
"""
from __future__ import annotations

import builtins
import itertools
import os
import re
import struct
import threading
import warnings
import zlib

import numpy as np

from .. import _memory, _pool, _stream
from .._base import File, Image
from .._errors import CorruptFileError, UnsupportedFormatError
from .._io import DecodeCache, ThreadHandles
from .._spatial import CoordinateSystem
from . import _nd2_legacy as legacy
from . import _nd2_structure as structure

NAME = "nd2"
#: a point NIS-Elements names after its well: a counter follows (``A10_0000``,
#: ``E8.1``) or the column is zero-padded (``B02``)
NIS_WELL_POINT = re.compile(r"^(?:well[\s_-]*)?[A-Za-z]{1,2}(?:\d{1,2}[\s_.#-]+\d+|0\d)$",
                            re.IGNORECASE)
#: takes plates=True: a file whose XY points are named after wells reads as a plate,
#: and a file of a NIS-Elements well-plate run (backends/_nd2_run.py) its run
PLATES = True
#: takes multifile=False: with plates=True, a run file is read alone
MULTIFILE = True
#: group_files opens a run file as an index only when asked for plates
INDEX_NEEDS_PLATES = True
DECODE_CACHE_BYTES = 64 << 20
RAW_TASK_BYTES = 8 << 20          # uncompressed frames are read in row pieces of this size
SEPARATE_READ_BYTES = 32 << 10    # bytes worth skipping rather than issuing one more read
STREAM_PIECE_BYTES = 8 << 20      # rows of a streamed "lossless" frame decoded at a time
TIMESTAMP_BYTES = 8               # each frame chunk starts with a float64 timestamp
#: a camera component's name -> (the sample it holds, its colour as NIS writes it,
#: 0x00BBGGRR; None: not checked)
SAMPLE_NAMES = {"red": ("R", 0x0000FF), "green": ("G", 0x00FF00), "blue": ("B", 0xFF0000),
                "alpha": ("A", None)}
#: the order samples are read in, by their count
READ_ORDER = {3: ("R", "G", "B"), 4: ("R", "G", "B", "A")}


def stored_sample_order(picture: dict, samples: int):
    """The order the file stores each pixel's RGB samples in, as letters (Nikon
    colour cameras: ``("B", "G", "R")``), or None when the file does not settle it.

    The order comes from the camera block of each channel's sample setting
    (``sPicturePlanes/sSampleSetting/a<k>/pCameraSetting/Metadata/Channels/Channel_<i>``):
    each component's ``Name`` (Blue, Green, Red) gives its sample, and its ``Color``
    must agree. Not settled: no such block, a name or colour that does not fit, not
    one of each sample, a count other than 3 or 4, or channels that disagree."""
    if samples not in READ_ORDER:
        return None
    settings = ((picture.get("sPicturePlanes") or {}).get("sSampleSetting") or {})
    orders = set()
    for setting in (settings.values() if isinstance(settings, dict) else ()):
        meta = ((setting or {}).get("pCameraSetting") or {}).get("Metadata") or {}
        components = meta.get("Channels") or {}
        if not isinstance(components, dict) or len(components) != samples:
            return None
        order = []
        for i in range(samples):
            component = components.get(f"Channel_{i}") or {}
            known = SAMPLE_NAMES.get(str(component.get("Name", "")).strip().lower())
            if known is None:
                return None
            letter, colour = known
            if colour is not None and component.get("Color") != colour:
                return None
            order.append(letter)
        if sorted(order) != sorted(READ_ORDER[samples]):
            return None
        orders.add(tuple(order))
    return orders.pop() if len(orders) == 1 else None


def _sample_index(stored: np.ndarray):
    """Stored sample positions as a slice when they are evenly spaced (a view, no
    copy: B, G, R reversed is ``slice(2, None, -1)``), else as the array itself."""
    if len(stored) == 0:
        return slice(0, 0)
    if len(stored) == 1:
        return slice(int(stored[0]), int(stored[0]) + 1)
    steps = np.diff(stored)
    if steps[0] != 0 and (steps == steps[0]).all():
        stop = int(stored[-1] + steps[0])
        return slice(int(stored[0]), None if stop < 0 else stop, int(steps[0]))
    return stored


def sniff(head: bytes, path: str) -> bool:
    if head[:4] == structure.CHUNK_MAGIC.to_bytes(4, "little"):
        return True
    # legacy ND2 is a JPEG 2000 box file: the JP2 signature alone would also
    # claim plain .jp2 images, so the extension decides (the box map at the
    # end is checked on opening)
    return legacy.is_legacy(head) and path.lower().endswith(".nd2")


class Nd2Parser:
    """What the images of one file share: layout, frame offsets, handles."""

    def __init__(self, path: str):
        self.path = path
        with builtins.open(path, "rb") as fh:
            ver = structure.version(fh.read(structure.START_CHUNK.size))
            cmap = structure.chunkmap(fh)
            # version 2: the same chunks, in XML, their names without "LV"
            two = ver < (3, 0)
            decode = structure.decode_xml if two else structure.decode_clx_lite

            def decoded(name):
                if two:
                    name = name.replace(b"LV", b"")
                if name not in cmap:
                    return {}
                return decode(structure.read_chunk(fh, cmap[name][0]))

            picture = decoded(b"ImageMetadataSeqLV|0!")
            experiment = decoded(b"ImageMetadataLV!")
            if two:
                calibration = decoded(b"ImageCalibrationLV|0!")
                picture.update(dCalibration=calibration.get("dCalibration"),
                               bCalibrated=calibration.get("bValid", False))
                planes = picture.get("sPicturePlanes") or {}
                planes.setdefault("sPlaneNew", planes.get("sPlane") or {})
            self.attrs = structure.attributes(decoded(b"ImageAttributesLV!"), picture)
            self.loops = structure.experiment(experiment)
            #: raw picture metadata and experiment, for ``_nd2_metadata``
            self.picture = picture.get("SLxPictureMetadata", picture)
            self.experiment = experiment
        if self.attrs.compression == "lossy":
            raise UnsupportedFormatError(f"{path!r}: lossy ND2 compression")
        a = self.attrs
        if a.components % a.channels:
            raise UnsupportedFormatError(f"{path!r}: {a.components} components "
                                         f"for {a.channels} channels")
        self.samples = a.components // a.channels
        #: the order the file stores RGB samples in, e.g. ("B", "G", "R"); None when
        #: the file does not state it (or for one sample). Reads give R, G, B(, A).
        self.sample_order = None
        #: read sample i -> its stored position, or None: read as stored
        self._stored = None
        if self.samples > 1:
            self.sample_order = stored_sample_order(self.picture, self.samples)
            if self.sample_order is None:
                warnings.warn(f"{path!r}: the file does not state the order of its "
                              f"{self.samples} colour samples; they are read as stored and "
                              "named R, G, B", stacklevel=4)
            else:
                stored = [self.sample_order.index(s) for s in READ_ORDER[self.samples]]
                if stored != list(range(self.samples)):
                    self._stored = np.array(stored)
        self.dtype = np.dtype(a.dtype)
        self.pixel_bytes = a.components * self.dtype.itemsize
        if a.width_bytes < a.width * self.pixel_bytes:
            raise CorruptFileError(f"{path!r}: rows of {a.width_bytes} bytes cannot hold "
                                   f"{a.width} pixels of {self.pixel_bytes} bytes")
        kinds = [loop.kind for loop in self.loops]
        if len(set(kinds)) != len(kinds):
            raise UnsupportedFormatError(f"{path!r}: repeated experiment loops {kinds}")
        # frame chunk -> offset of its header; data offsets are read lazily
        self.frames = {int(name[13:-1]): offset for name, (offset, _) in cmap.items()
                       if name.startswith(b"ImageDataSeq|")}
        self._data_at: dict = {}
        self._lock = threading.Lock()
        self.handles = ThreadHandles(path)

    def size(self, kind: str) -> int:
        return next((loop.count for loop in self.loops if loop.kind == kind), 1)

    def stored_samples(self, ss: slice):
        """The stored positions of read samples *ss* (R, G, B order): *ss* itself
        when the file stores them in that order, else a slice or index array into
        the stored samples."""
        if self._stored is None:
            return ss
        return _sample_index(self._stored[ss])

    def frame_number(self, t: int, p: int, z: int) -> int:
        index = {"T": t, "P": p, "Z": z}
        number = 0
        for loop in self.loops:
            number = number * loop.count + index[loop.kind]
        return number

    def frame_data(self, number: int):
        """(offset of the frame's pixels or compressed stream, its length),
        or None for a frame that was never written."""
        where = self._data_at.get(number)
        if where is None:
            header = self.frames.get(number)
            if header is None:
                return None
            fh = self.handles.get()
            start, length = structure.data_offset(fh, header)
            where = self._data_at[number] = (start + TIMESTAMP_BYTES, length - TIMESTAMP_BYTES)
        return where

    def _stamp(self, fh, number: int) -> float:
        """The timestamp (ms) a frame chunk starts with."""
        start, _ = structure.data_offset(fh, self.frames[number])
        fh.seek(start)
        return struct.unpack("<d", fh.read(TIMESTAMP_BYTES))[0]

    def time_step(self, position: int):
        """The mean interval (ms) between the frames of *position*, from the stamps of its
        first and last frame written; None for one frame or a file without stamps.  These
        are two small reads, on a handle of their own: metadata is also asked for after
        the file was closed."""
        written = [t for t in range(self.size("T")) if self.frame_number(t, position, 0)
                   in self.frames]
        if len(written) < 2:
            return None
        first, last = written[0], written[-1]
        with builtins.open(self.path, "rb") as fh:
            step = (self._stamp(fh, self.frame_number(last, position, 0)) -
                    self._stamp(fh, self.frame_number(first, position, 0))) / (last - first)
        return step if step > 0 else None


class _Nd2Image(Image):
    """One XY position: axes 'tczyx' (+ 's' for RGB components)."""

    def __init__(self, nd2: Nd2Parser, position: int, index: dict = None):
        a = nd2.attrs
        self._f = nd2
        self._p = position
        self._rgb = nd2.samples > 1
        shape = (nd2.size("T"), a.channels, nd2.size("Z"), a.height, a.width)
        axes = "tczyx"
        if self._rgb:
            shape, axes = shape + (nd2.samples,), axes + "s"
        #: decoded "lossless" frames kept for reuse
        self.decode_cache = DecodeCache(DECODE_CACHE_BYTES)
        self._closed = False
        unit = (1, a.channels, 1, a.height, a.width) + ((nd2.samples,) if self._rgb else ()) \
            if a.compression == "lossless" else None
        super().__init__(position, axes, shape, nd2.dtype,
                         name=f"{os.path.basename(nd2.path)} position {position}",
                         read_unit=unit, index=index)

    @property
    def sample_order(self):
        """The order the file stores the RGB samples in, e.g. ``("B", "G", "R")``
        (Nikon colour cameras); None when the file does not state it or the image
        has one sample. Reads always give R, G, B(, A) on the ``s`` axis."""
        return self._f.sample_order

    def _read_block(self, block):
        if self._closed:
            raise ValueError("this ND2 image has been closed: its file was closed "
                             "(File.close() or the end of a with block)")
        ts, cs, zs, ys, xs = block[:5]
        ss = block[5] if self._rgb else slice(0, 1)
        out_shape = tuple(s.stop - s.start for s in block)
        ss = self._f.stored_samples(ss)                      # read R, G, B from the stored order
        out = np.empty(out_shape, dtype=self.dtype)
        view = out if self._rgb else out[..., None]          # always (..., s)
        tasks = []
        for t, z in itertools.product(range(ts.start, ts.stop), range(zs.start, zs.stop)):
            dst = view[t - ts.start, :, z - zs.start]         # (c, y, x, s)
            where = self._f.frame_data(self._f.frame_number(t, self._p, z))
            if where is None:
                dst[...] = 0
            elif self._f.attrs.compression == "lossless":
                tasks.append(lambda a=(t, z, where, dst, cs, ys, xs, ss): self._decoded_into(*a))
            else:
                tasks.extend(self._raw_tasks(where[0], dst, cs, ys, xs, ss))
        return out, _pool.run(tasks)

    def _frame_view(self, buf: np.ndarray, rows: int) -> np.ndarray:
        """Bytes of *rows* whole rows -> (rows, x, channel, sample)."""
        f, a = self._f, self._f.attrs
        pixels = buf.reshape(rows, a.width_bytes)[:, :a.width * f.pixel_bytes]
        return pixels.view(f.dtype).reshape(rows, a.width, a.channels, f.samples)

    def _raw_tasks(self, start, dst, cs, ys, xs, ss):
        f, a = self._f, self._f.attrs
        per_task = max(1, RAW_TASK_BYTES // a.width_bytes)
        # whole rows when the columns skipped per row are few: fewer bytes than another read
        # call costs (as in LIF), or at most half the row; else only the needed bytes per row
        skipped = a.width_bytes - (xs.stop - xs.start) * f.pixel_bytes
        whole_rows = (xs.stop - xs.start) * 2 >= a.width or skipped <= SEPARATE_READ_BYTES
        for y0 in range(ys.start, ys.stop, per_task):
            y1 = min(ys.stop, y0 + per_task)
            part = dst[:, y0 - ys.start:y1 - ys.start]
            yield lambda args=(start, y0, y1, part, cs, xs, ss, whole_rows): self._read_rows(*args)

    def _read_rows(self, start, y0, y1, dst, cs, xs, ss, whole_rows) -> int:
        """Rows y0:y1 of an uncompressed frame into *dst* (c, y, x, s)."""
        f, a = self._f, self._f.attrs
        fh = f.handles.get()
        n = y1 - y0
        if whole_rows:
            buf = np.empty(n * a.width_bytes, np.uint8)
            fh.seek(start + y0 * a.width_bytes)
            if fh.readinto(memoryview(buf)) != buf.nbytes:
                raise CorruptFileError(f"{f.path!r}: frame data ends early")
            pixels = self._frame_view(buf, n)[:, xs]
            read = n * a.width * a.components
        else:
            width = xs.stop - xs.start
            buf = np.empty((n, width * f.pixel_bytes), np.uint8)
            for i in range(n):
                fh.seek(start + (y0 + i) * a.width_bytes + xs.start * f.pixel_bytes)
                if fh.readinto(memoryview(buf[i])) != buf.shape[1]:
                    raise CorruptFileError(f"{f.path!r}: frame data ends early")
            pixels = buf.view(f.dtype).reshape(n, width, a.channels, f.samples)
            read = n * width * a.components
        dst[...] = pixels[:, :, cs, ss].transpose(2, 0, 1, 3)
        return read

    def _decoded_into(self, t, z, where, dst, cs, ys, xs, ss) -> int:
        frame, read = self.decode_cache.get((t, z)), 0
        a = self._f.attrs
        if frame is None and _memory.should_stream(a.height * a.width_bytes):
            return self._streamed_into(where, dst, cs, ys, xs, ss)
        if frame is None:
            f, a = self._f, self._f.attrs
            with _memory.reserve(a.height * a.width_bytes, f"{f.path!r}: an ND2 frame"):
                data = zlib.decompress(f.handles.read(*where), 15, a.height * a.width_bytes)
            if len(data) < a.height * a.width_bytes:
                raise CorruptFileError(f"{f.path!r}: frame decodes to {len(data)} bytes")
            frame = self._frame_view(np.frombuffer(data, np.uint8)[:a.height * a.width_bytes],
                                     a.height)
            self.decode_cache.put((t, z), frame)
            read = frame.size
        dst[...] = frame[ys, xs][:, :, cs, ss].transpose(2, 0, 1, 3)
        return read

    def _read_metadata(self):
        from ._nd2_metadata import nd2_metadata
        return nd2_metadata(self._f.picture, self._f.experiment, self._p,
                            dict(zip(self.axes, self.shape)), self._f.time_step(self._p),
                            self._f.path)

    def _streamed_into(self, where, dst, cs, ys, xs, ss) -> int:
        """Rows *ys* of a "lossless" frame too large to decode whole: its zlib
        stream decoded as a stream, in pieces of rows (``_stream``)."""
        f, a = self._f, self._f.attrs
        per_piece = max(1, min(STREAM_PIECE_BYTES, _memory.memory_limit() // 4) // a.width_bytes)
        for y0 in range(ys.start, ys.stop, per_piece):
            y1 = min(ys.stop, y0 + per_piece)
            with _memory.reserve((y1 - y0) * a.width_bytes, f"{f.path!r}: rows of a frame"):
                buf = np.empty((y1 - y0) * a.width_bytes, np.uint8)
                n = _stream.read(f.path, where[0], where[1], "zlib", y0 * a.width_bytes, buf,
                                 size=a.height * a.width_bytes)
                if n < buf.nbytes:
                    raise CorruptFileError(f"{f.path!r}: frame ends at {n} of {buf.nbytes} bytes")
                rows = self._frame_view(buf, y1 - y0)
                dst[:, y0 - ys.start:y1 - ys.start] =                     rows[:, xs][:, :, cs, ss].transpose(2, 0, 1, 3)
        return (ys.stop - ys.start) * a.width * a.components

    def close(self) -> None:
        self._closed = True
        self.decode_cache.clear()
        _stream.forget(self._f.path)


class _Nd2File(File):
    backend = NAME

    def __init__(self, path, images, nd2: Nd2Parser, plates=()):
        super().__init__(path, images, plates=plates)
        self._nd2 = nd2

    def _declare_coordinate_systems(self):
        # the positions' one stage frame (plan 4.5), when they can have a
        # position at all: none without the camera-to-stage matrix
        from ._nd2_metadata import STAGE, placed
        return (CoordinateSystem(STAGE),) if placed(self._nd2.picture) else ()

    def close(self) -> None:
        super().close()
        self._nd2.handles.close()


def _point_plate(nd2: Nd2Parser):
    """(per position its plate index, [Plate]) when every XY point is named
    after a well (module docstring), else None."""
    from ._nd2_metadata import _xy_points
    from .._plate import Plate, Well, layout, row_label, standard_layout, well_of
    names = [p.get("dPosName") for p in _xy_points(nd2.experiment)]
    if not names or len(names) != nd2.size("P"):
        return None
    # a bare "P1", "S2" names a position, not a well: every name must carry what
    # NIS-Elements writes, a counter after the well or a zero-padded column
    if not all(NIS_WELL_POINT.match(n or "") for n in names):
        return None
    at = [well_of(n) for n in names]
    if None in at or standard_layout(at) is None:
        return None
    wells = sorted(set(at))
    from ._nd2_run import plate_size
    size = plate_size(nd2.path, wells, nd2.path)        # the job's well plate, else a guess
    number = {rc: w for w, rc in enumerate(wells)}
    seen: dict = {}
    index = []
    for rc in at:                                      # fields in acquisition order
        index.append({"plate": 0, "acquisition": 0, "well": number[rc], "field": seen.get(rc, 0)})
        seen[rc] = seen.get(rc, 0) + 1
    row_names, column_names = layout(*size, wells, nd2.path)
    name = os.path.splitext(os.path.basename(nd2.path))[0]
    return index, [Plate(name=name, row_names=row_names, column_names=column_names,
                         wells=tuple(Well(r, c, f"{row_label(r)}{c + 1}") for r, c in wells))]


def may_index(name: str) -> bool:
    """``group_files``' cheap test (reads nothing): a file of a NIS-Elements
    well-plate run (``Well<well>..._Seq<n>.nd2``) opens its run as a plate."""
    from ._nd2_run import RUN_FILE
    return bool(RUN_FILE.match(name))


def open(path: str, tiles: str = "separate", resolutions: str = "base",  # noqa: A001 - backend protocol name
         plates: bool = False, multifile: bool = True):
    with builtins.open(path, "rb") as fh:
        if legacy.is_legacy(fh.read(12)):
            return legacy._LegacyNd2File(path)
    if plates and multifile:
        # a file of a NIS-Elements well-plate run: its folder's run (backends/_nd2_run.py)
        from ._nd2_run import RUN_FILE, _Nd2RunFile
        if RUN_FILE.match(os.path.basename(path)):
            return _Nd2RunFile(path)
    nd2 = Nd2Parser(path)
    plate = _point_plate(nd2) if plates else None
    index, found = plate if plate is not None else ([None] * nd2.size("P"), [])
    images = [_Nd2Image(nd2, p, index[p]) for p in range(nd2.size("P"))]
    if len(images) > 1:                     # _naming: the XY points' names tell them apart
        from ._nd2_metadata import _xy_points
        points = _xy_points(nd2.experiment)
        if len(points) == len(images):
            for image, point in zip(images, points):
                image._label = (point.get("dPosName") or "").strip() or None
    return _Nd2File(path, images, nd2, plates=found)
