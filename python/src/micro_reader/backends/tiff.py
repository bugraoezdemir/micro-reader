"""TIFF family: TIFF, BigTIFF, OME-TIFF, ImageJ hyperstacks (tifffile, a dependency).

tifffile (Gohlke, BSD-3) parses the structure -- pages, series, axes -- and
decodes compressed segments.  Pixels are read here rather than through
tifffile's ``aszarr``: zarr reads whole chunks, and a TIFF chunk is a stored
strip or tile, which for the common one-strip-per-plane files is the whole
plane (PLAN.md, "M1 design").

- Uncompressed pages: positioned reads of exactly the requested rows and
  columns, whatever the layout; file-adjacent strips are read together.
- Compressed pages: only the strips / tiles touching the rectangle are read
  and decoded, in parallel on the shared decode pool (``_pool``), and a
  bounded cache keeps a segment shared by neighbouring reads from being
  decoded twice.
- Decoding: micro-reader's own codecs (``_codecs``: micro-reader-codecs,
  else a reference decoder) for the compressions microscopy TIFFs use --
  LZW, Deflate, PackBits, LZMA, zstd, JPEG, JPEG 2000, WebP, JPEG XL, PNG,
  EER (Falcon cryo-EM cameras) -- with the horizontal and floating-point
  predictors, packed unsigned depths and FillOrder 2.  Anything else
  (CCITT, JPEG XR, LERC, float24, ...) goes through tifffile's decoder, which
  needs imagecodecs for most of them.

Images and axes (docs/AXES.md): the array axes ``T C Z Y X S`` keep the
series' own order (lower case; EER frames ``I`` are ``t``).  Every other
tifffile axis makes separate images, indexed by: ``A`` (angle) view, ``R`` /
``M`` / ``J K`` tile, LSM positions ``P`` scene, ``P`` (phase) phase, ``H``
lifetime, ``E`` (lambda) emission, and ``I`` / ``Q`` / ``L`` / ``V`` (sequence,
other, exposure, event) sequence; several letters of one index are counted
as one (raveled).  ``open(path, rename={"sequence": "z"})`` reads such a
dimension as an axis instead (plain multi-page TIFFs).  Each series is a
scene otherwise.  Images of one series share its file handles and
decoded-segment cache.

Resolution levels (OME-TIFF SubIFDs, SVS / NDPI-style pyramids: tifffile's
``series.levels``) are each image's ``resolutions``; with
``resolutions="all"`` they are images of their own, indexed by ``resolution``.

Not yet: volumetric tiles (ImageDepth), OME series spread over several files.
"""
from __future__ import annotations

import itertools
import os
import threading

import numpy as np

from .. import _codecs, _jpeg_restarts, _memory, _pool, _stream
from .._base import INDEX_NAMES, File, Image
from .._errors import CorruptFileError, MicroReaderError, UnsupportedFormatError
from .._io import DecodeCache, ThreadHandles

NAME = "tiff"
MAGIC = (b"II*\x00", b"MM\x00*", b"II+\x00", b"MM\x00+")   # classic and BigTIFF
CHUNK_BYTES = 4 << 20            # buffered raw rows are read this many bytes at a time
RAW_TASK_BYTES = 8 << 20         # uncompressed strip runs are split into pool tasks of this size
SEPARATE_READ_BYTES = 32 << 10   # bytes worth skipping rather than issuing one more read
DECODE_CACHE_BYTES = 64 << 20    # decoded segments kept per scene
STREAM_PIECE_BYTES = 8 << 20     # rows of a streamed segment decoded at a time (reserved)
#: image codecs that decode a rectangle alone (``_area_into``)
AREA_DECODERS = {"jpeg2k": _codecs.jpeg2k_decode_area_into, "jxl": _codecs.jxl_decode_area_into}
JPEG_COMPRESSIONS = {6, 7, 33007, 34892}

#: TIFF Compression -> micro-reader's own decoder.  Byte codecs give the
#: stored bytes (then the predictor applies); image codecs give pixels.
BYTE_CODECS = {1: "none", 5: "lzw", 8: "zlib", 32946: "zlib", 32773: "packbits",
               34925: "lzma", 34926: "zstd", 50000: "zstd"}
IMAGE_CODECS = {7: "jpeg", 33007: "jpeg", 34892: "jpeg",
                33003: "jpeg2k", 33004: "jpeg2k", 33005: "jpeg2k", 34712: "jpeg2k",
                34927: "webp", 50001: "webp", 50002: "jxl", 52546: "jxl", 34933: "png"}
#: EER (Falcon cryo-EM cameras): compression -> run-length bits; 65002
#: ("EER v2") takes them and the sub-pixel bits from tags 65007-65009
EER_CODES = {65000: 8, 65001: 7, 65002: None}


def eer_parameters(key) -> tuple:
    """(skip, horizontal, vertical) bits of *key*'s EER stream (as tifffile
    reads them; micro-reader decodes at the camera's pixel size, tifffile's
    default, superres 0)."""
    if key.compression == 65002:
        return (int(key.tags.valueof(65007, 7)), int(key.tags.valueof(65008, 2)),
                int(key.tags.valueof(65009, 2)))
    return EER_CODES[key.compression], 2, 2


_BYTE_DECODERS = {"lzw": _codecs.lzw_decode_into, "zlib": _codecs.zlib_decompress_into,
                  "packbits": _codecs.packbits_decode_into,
                  "lzma": _codecs.lzma_decompress_into}
_IMAGE_DECODERS = {"jpeg2k": _codecs.jpeg2k_decode_into, "webp": _codecs.webp_decode_into,
                   "jxl": _codecs.jxl_decode_into, "png": _codecs.png_decode_into}


def own_codec(key):
    """micro-reader's decoder for tifffile keyframe *key*'s segments, or
    None when tifffile's decoder must do it."""
    dtype = np.dtype(key.dtype)
    if dtype.kind == "c" or not isinstance(key.bitspersample, int):
        return None                                      # complex; RGB565-style depths
    c, predictor = key.compression, key.predictor
    if c in BYTE_CODECS:
        if key.is_subsampled or predictor not in (1, 2, 3):
            return None
        if key.bitspersample != 8 * dtype.itemsize:
            # packed unsigned depths (1-bit masks, 2 / 4 / 10 / 12 / 14 / 24 bit):
            # unpacked here; signed and float24 stay tifffile's, and TIFF
            # predictors are only defined for whole-byte samples
            if dtype.kind not in "ub" or predictor != 1 or key.bitspersample > 32:
                return None
        if (predictor == 2 and dtype.kind not in "iu") or (predictor == 3 and dtype.kind != "f"):
            return None
        return BYTE_CODECS[c]
    if key.fillorder != 1:
        return None                                      # bit order of image codestreams
    if c in EER_CODES:
        # only files that are EER (BigTIFF with the acquisition metadata tag,
        # as tifffile checks): codes from 65000 are a private range
        if getattr(key, "is_eer", False) and key.planarconfig == 1 \
                and key.samplesperpixel == 1 and dtype.kind in "bu" and dtype.itemsize in (1, 2, 4):
            return "eer"
        return None
    if c in IMAGE_CODECS and predictor == 1 and key.planarconfig == 1:
        # the codestream carries its own precision (e.g. 12-bit JPEG in uint16);
        # the decoders give 8- or 16-bit samples
        if dtype.kind != "u" or dtype.itemsize > 2:
            return None
        codec = IMAGE_CODECS[c]
        if codec == "jpeg" and getattr(key, "jpegheader", None):
            return None                                  # NDPI: header outside the segments
        if codec != "jpeg" and key.is_subsampled:
            return None
        return codec
    return None


def jpeg_colour(key) -> str:
    """How the JPEG segments of *key* store colour (as tifffile decides):
    RGB without JFIF marker is stored untransformed (e.g. Aperio SVS)."""
    if key.extrasamples:
        return "auto"
    if key.photometric == 6:
        return "ycbcr"
    if key.photometric == 2 and not key.is_jfif:
        return "rgb"
    return "auto"


def unpack(stored: np.ndarray, shape: tuple, bits: int, dtype: np.dtype,
           byteorder: str) -> np.ndarray:
    """Packed samples of one segment -> (rows, cols, samples) of *dtype*.
    Below a byte, and at 10 / 12 / 14 bits, TIFF packs samples as one bit
    stream, most significant bit first, each row starting on a byte
    boundary.  Whole-byte depths that are not a numpy size (24-bit) are
    stored in the file's byte order (as libtiff does; tifffile returns
    little-endian 24-bit samples shifted left by 8 bits)."""
    rows, cols, samples = shape
    runlen = cols * samples
    out = np.empty(shape, dtype.newbyteorder("=") if dtype.kind != "b" else np.uint8)
    if bits % 8 == 0 and byteorder == "<":
        nb = bits // 8
        b = stored.reshape(rows, -1)[:, :runlen * nb].reshape(rows, runlen, nb).astype(np.uint32)
        value = sum(b[..., k] << np.uint32(8 * k) for k in range(nb))
        out[...] = value.reshape(shape)
    else:
        _codecs.unpack_bits_into(stored, out, bits, runlen)
    return out.view(np.bool_) if dtype.kind == "b" else out


def unpredict(seg: np.ndarray, predictor: int) -> np.ndarray:
    """Undo TIFF's horizontal (2) or floating-point (3) predictor on a
    segment (rows, cols, samples) in the file's dtype; returns it in native
    byte order.  Floating point (libtiff ``fpAcc``): per row, bytes summed
    with a stride of one pixel, then byte planes, most significant first."""
    rows, cols, nc = seg.shape
    if predictor == 2:
        out = seg.astype(seg.dtype.newbyteorder("="), copy=False)
        np.cumsum(out, axis=1, dtype=out.dtype, out=out)
        return out
    item = seg.dtype.itemsize
    b = np.ascontiguousarray(seg).view(np.uint8).reshape(rows, cols * item, nc)
    np.cumsum(b, axis=1, dtype=np.uint8, out=b)
    planes = b.reshape(rows, item, cols * nc).transpose(0, 2, 1)
    values = np.ascontiguousarray(planes).view(f">{seg.dtype.kind}{item}")
    return values.reshape(rows, cols, nc).astype(seg.dtype.newbyteorder("="))

#: tifffile axis letter -> array axis (docs/AXES.md)
AXIS_LETTERS = {"X": "x", "Y": "y", "Z": "z", "T": "t", "C": "c", "S": "s"}
#: tifffile axis letter -> image index (LSM ``P`` is a position: "scene");
#: letters tifffile does not name are "sequence" too
INDEX_LETTERS = {"A": "view", "R": "tile", "M": "tile", "J": "tile", "K": "tile",
                 "P": "phase", "H": "lifetime", "E": "emission",
                 "I": "sequence", "Q": "sequence", "L": "sequence", "V": "sequence"}
#: indices besides "scene", in index order
OTHER_INDICES = tuple(n for n in INDEX_NAMES if n not in ("scene", "resolution"))
#: this backend can make an index an axis (open(..., rename=...))
RENAMES = True


def translate_axes(axes: str, *, lsm: bool = False, eer: bool = False, where: str = "",
                   rename: dict = None):
    """tifffile series axes -> (array axis letter or None per axis, image
    index or None per axis); *rename* makes an index an axis instead."""
    rename = rename or {}
    letters, levels = [], []
    for a in axes.upper():
        if a in AXIS_LETTERS:
            level, letter = None, AXIS_LETTERS[a]
        elif a == "I" and eer:
            level, letter = None, "t"
        else:
            level = "scene" if (a == "P" and lsm) else INDEX_LETTERS.get(a, "sequence")
            letter = None
            if level in rename:
                level, letter = None, rename[level]
        levels.append(level)
        letters.append(letter)
    used = [x for x in letters if x]
    doubled = sorted({x for x in used if used.count(x) > 1})
    if doubled:
        raise UnsupportedFormatError(
            f"{where}: series axes {axes!r} give axis {doubled[0]!r} twice"
            + (f" (with rename={rename})" if rename else ""))
    return letters, levels


#: tifffile page axes -> their positions in the normalised page shape
#: (separate sample, depth, length, width, contig sample)
_PAGE_AXES = {"YX": (2, 3), "YXS": (2, 3, 4), "SYX": (0, 2, 3)}
#: view of one page's part of the result as (separate sample, y, x, contig sample)
_AS_NORMALISED = {"YX": lambda a: a[None, :, :, None],
                  "YXS": lambda a: a[None],
                  "SYX": lambda a: a[..., None]}


def sniff(head: bytes, path: str) -> bool:
    return head[:4] in MAGIC


def _readinto(fh, buf: np.ndarray, path: str) -> None:
    if fh.readinto(memoryview(buf).cast("B")) != buf.nbytes:
        raise CorruptFileError(f"{path!r}: pixel data ends early (truncated file?)")


class _TiffImage(Image):
    """One image of a tifffile series (base level): the whole series, or one
    view / tile / LSM position of it (*fixed*: series axis -> index)."""

    def __init__(self, tif, index: int, series, letters: list, *, fixed: dict = None,
                 key: dict = None, share: "_TiffImage" = None):
        image_key = key
        key = series.keyframe
        where = f"{os.path.basename(tif.filehandle.path)!r} series {index}"
        if series.is_multifile:
            raise UnsupportedFormatError(f"{where}: series spread over several files")
        if key.dtype is None:
            raise UnsupportedFormatError(
                f"{where}: data type not supported (SampleFormat {key.sampleformat}, "
                f"{key.bitspersample}-bit)")
        if key.shaped[1] != 1:
            raise UnsupportedFormatError(f"{where}: volumetric pages (ImageDepth)")
        if key.axes not in _PAGE_AXES:
            raise UnsupportedFormatError(f"{where}: page axes {key.axes!r}")
        # tifffile stacks the pages and reshapes them to the series shape, so
        # the pages must be its trailing dimensions.  Only the shapes must
        # match: series metadata may name the page axes differently (e.g. an
        # RGB-planar page 'SYX' labelled 'ZYX').
        shape, axes, k = tuple(series.shape), series.axes, len(key.shape)
        if shape[len(shape) - k:] != tuple(key.shape):
            raise UnsupportedFormatError(
                f"{where}: pages ({key.axes} {key.shape}) do not map onto the "
                f"trailing axes of the series ({axes} {shape})")

        self._tif = tif
        self._series = series
        self._key = key
        self._path = tif.filehandle.path
        self._lead_shape = shape[:len(shape) - k]
        self._page_pos = _PAGE_AXES[key.axes]
        self._sep, _, self._length, self._width, self._contig = key.shaped
        self._file_dtype = np.dtype(key.dtype).newbyteorder(tif.byteorder)
        self._raw = (key.compression == 1 and key.predictor == 1 and key.fillorder == 1
                     and not key.is_subsampled and key.sampleformat != 5
                     and key.bitspersample == 8 * self._file_dtype.itemsize)
        self._as_normalised = _AS_NORMALISED[key.axes]
        self._tiled = bool(key.is_tiled)
        if self._tiled:
            self._seg_rows, self._seg_cols = key.tilelength, key.tilewidth
        else:
            self._seg_rows, self._seg_cols = (key.rowsperstrip or self._length), self._width
        self._segs_down = -(-self._length // self._seg_rows)
        self._segs_across = -(-self._width // self._seg_cols)

        # Series that list only their first IFD although they hold many
        # pages (ImageJ hyperstacks > 4 GB, MetaMorph STK): the pages follow
        # each other as one contiguous block.
        npages = int(np.prod(self._lead_shape, dtype=np.int64))
        self._page_bytes = int(np.prod(key.shaped, dtype=np.int64)) * self._file_dtype.itemsize
        self._contiguous = None
        if npages > len(series.pages):
            if series.dataoffset is None or not self._raw:
                raise UnsupportedFormatError(f"{where}: truncated series that is not contiguous")
            if os.path.getsize(self._path) < series.dataoffset + npages * self._page_bytes:
                raise CorruptFileError(f"{where}: truncated file")
            self._contiguous = series.dataoffset

        if share is None:                      # the first image of this series
            self._pages: dict = {}             # page -> (offsets, bytecounts, jpegtables) | None
            self._lock = threading.Lock()      # tifffile parses IFDs through one file handle
            self._handles = ThreadHandles(self._path)
            #: decoded segments kept for reuse (``decode_cache.max_bytes`` to
            #: resize); shared by the images of one series
            self.decode_cache = DecodeCache(DECODE_CACHE_BYTES)
            #: (page, segment) -> RestartIndex | None, of JPEG segments too
            #: large to decode whole (``_jpeg_rows_into``)
            self._restarts: dict = {}
        else:
            self._pages, self._lock = share._pages, share._lock
            self._handles, self.decode_cache = share._handles, share.decode_cache
            self._restarts = share._restarts
        self._closed = False
        #: micro-reader's own decoder for the segments; None: tifffile's
        self._codec = None if self._raw else own_codec(key)
        self._predictor = key.predictor
        self._jpeg_colour = jpeg_colour(key)
        if not self._raw and self._codec is None:
            self._decode = key.decode          # tifffile builds its decode function once

        # view / tile / position axes fixed at one index: not axes of the image
        self._fixed = dict(fixed or {})
        if any(pos >= len(self._lead_shape) for pos in self._fixed):
            raise UnsupportedFormatError(f"{where}: a view / tile axis inside the pages ({axes})")
        self._full_shape = shape
        keep = [i for i in range(len(shape)) if i not in self._fixed]
        unit = None
        if not self._raw:
            per_pos = {0: 1, 2: self._seg_rows, 3: self._seg_cols, 4: self._contig}
            unit = (1,) * len(self._lead_shape) + tuple(per_pos[p] for p in self._page_pos)
            unit = [unit[i] for i in keep]
        super().__init__(index, "".join(letters[i] for i in keep), [shape[i] for i in keep],
                         series.dtype, name=series.name or f"series{index}", read_unit=unit,
                         index=image_key)

    # -- page index ---------------------------------------------------------

    def _page(self, p: int):
        """(offsets, bytecounts, jpegtables) of page *p*, None if it is missing."""
        try:
            return self._pages[p]
        except KeyError:
            pass
        with self._lock:
            if p not in self._pages:
                key = self._key
                if self._contiguous is not None:
                    shift = self._contiguous + p * self._page_bytes - key.dataoffsets[0]
                    entry = ([o + shift for o in key.dataoffsets], key.databytecounts, None)
                else:
                    page = self._series.pages[p]       # parses the IFD on first access
                    entry = None if page is None else (
                        page.dataoffsets, page.databytecounts,
                        getattr(page, "jpegtables", None) or key.jpegtables)
                self._pages[p] = entry
            return self._pages[p]

    # -- reading --------------------------------------------------------------

    def _read_block(self, block):
        if self._closed:
            raise ValueError("this TIFF image has been closed: its file was closed "
                             "(File.close() or the end of a with block)")
        if not self._fixed:
            return self._read_series(block)
        it = iter(block)
        full = tuple(slice(self._fixed[i], self._fixed[i] + 1) if i in self._fixed else next(it)
                     for i in range(len(self._full_shape)))
        out, read = self._read_series(full)
        return out.reshape(tuple(s.stop - s.start for s in block)), read

    def _read_series(self, block):
        """Read *block* over all axes of the series."""
        out = np.empty(tuple(s.stop - s.start for s in block), dtype=self.dtype)
        # every page's uncompressed pieces and compressed segments, on all cores
        return out, _pool.run(self._tasks(*self._plan_series(block, out)))

    def _plan_series(self, block, out) -> tuple:
        """(raws, decodes) that fill *out* (shaped like *block*, over all axes
        of the series) once their ``_tasks`` run; parts with nothing to read
        are filled now. Planned apart from running so that a file set can run
        the plans of many member pages in one batch (``_fileset``)."""
        if self._closed:
            raise ValueError("this TIFF image has been closed: its file was closed "
                             "(File.close() or the end of a with block)")
        nlead = len(self._lead_shape)
        lead, inner = block[:nlead], block[nlead:]
        # the page part of the request as (separate sample, y, x, contig sample)
        norm = {0: slice(0, self._sep), 4: slice(0, self._contig)}
        norm.update(zip(self._page_pos, inner))
        ss, ys, xs, cs = norm[0], norm[2], norm[3], norm[4]
        raws, decodes = [], []
        for idx in itertools.product(*(range(s.start, s.stop) for s in lead)):
            p = int(np.ravel_multi_index(idx, self._lead_shape)) if nlead else 0
            dest = self._as_normalised(out[tuple(i - s.start for i, s in zip(idx, lead))])
            self._plan_page(p, ss, ys, xs, cs, dest, raws, decodes)
        return raws, decodes

    def _tasks(self, raws, decodes):
        """The pool tasks of a plan (``_plan_series``)."""
        return itertools.chain(
            (lambda job=job: self._read_rect(self._handles.get(), *job) for job in raws),
            self._decode_tasks(decodes))

    def _plan_page(self, p, ss, ys, xs, cs, dest, raws, decodes) -> None:
        """What page *p* contributes: empty parts are filled here; uncompressed
        pieces go to *raws* and compressed segments to *decodes*, to run
        together on the pool."""
        entry = self._page(p)
        if entry is None:
            dest[...] = self._key.nodata
            return
        offsets, counts, jpegtables = entry
        rows, cols = self._seg_rows, self._seg_cols
        r0, r1 = ys.start // rows, (ys.stop - 1) // rows + 1
        c0, c1 = xs.start // cols, (xs.stop - 1) // cols + 1
        per_sample = self._segs_down * self._segs_across
        for d, s in zip(dest, range(ss.start, ss.stop)):
            if self._raw and not self._tiled:
                self._plan_raw_strips(offsets, counts, s * per_sample, r0, r1, ys, xs, cs, d, raws)
                continue
            for tr in range(r0, r1):
                for tc in range(c0, c1):
                    i = s * per_sample + tr * self._segs_across + tc
                    sy, sx = tr * rows, tc * cols
                    y0, y1 = max(ys.start, sy), min(ys.stop, sy + rows)
                    x0, x1 = max(xs.start, sx), min(xs.stop, sx + cols)
                    dst = d[y0 - ys.start:y1 - ys.start, x0 - xs.start:x1 - xs.start]
                    if not counts[i] or not offsets[i]:           # sparse segment
                        dst[...] = self._key.nodata
                    elif self._raw:                                # uncompressed tile
                        raws.append((offsets[i], cols, (y0 - sy, y1 - sy), (x0 - sx, x1 - sx),
                                     cs, dst))
                    else:
                        decodes.append((p, i, offsets[i], counts[i], jpegtables, dst,
                                        (0, slice(y0 - sy, y1 - sy), slice(x0 - sx, x1 - sx), cs)))

    def _plan_raw_strips(self, offsets, counts, base, r0, r1, ys, xs, cs, d, raws) -> None:
        """Uncompressed strips: strips that follow each other in the file are
        one run, and runs are cut into pieces of about RAW_TASK_BYTES so a
        single large read still spreads over the pool."""
        rows, width = self._seg_rows, self._width
        row_bytes = width * self._contig * self._file_dtype.itemsize
        strip_bytes = rows * row_bytes
        per_task = max(1, RAW_TASK_BYTES // row_bytes)
        tr = r0
        while tr < r1:
            i = base + tr
            end = tr + 1
            while (end < r1 and offsets[base + end] == offsets[base + end - 1] + strip_bytes
                   and counts[base + end - 1] >= strip_bytes):
                end += 1
            sy = tr * rows
            y0, y1 = max(ys.start, sy), min(ys.stop, end * rows)
            if not counts[i] or not offsets[i]:
                d[y0 - ys.start:y1 - ys.start] = self._key.nodata
            else:
                for a in range(y0, y1, per_task):
                    b = min(y1, a + per_task)
                    raws.append((offsets[i], width, (a - sy, b - sy), (xs.start, xs.stop),
                                 cs, d[a - ys.start:b - ys.start]))
            tr = end

    def _read_rect(self, fh, offset, stride, rows, cols, cs, dst) -> int:
        """Rows ``rows`` and columns ``cols`` of an uncompressed segment of
        width *stride* stored at *offset*, into *dst* (rows, cols, samples).
        Straight into *dst* when its memory has the file's layout (native
        byte order, all samples, contiguous rows); else through a buffer."""
        item, nc = self._file_dtype.itemsize, self._contig
        row_len = stride * nc
        direct = ((item == 1 or self._file_dtype.isnative)
                  and (cs.start, cs.stop) == (0, nc))
        if cols == (0, stride) and direct and dst.flags.c_contiguous:
            fh.seek(offset + rows[0] * row_len * item)
            _readinto(fh, dst, self._path)
            return dst.size
        width = cols[1] - cols[0]
        if (cols != (0, stride) and rows[1] - rows[0] > 1
                and (stride - width) * nc * item <= SEPARATE_READ_BYTES):
            # a few skipped bytes per row cost less than one more read per row (as in
            # LIF): read the span from the first row's columns to the last row's, in
            # chunks, and keep the columns asked for
            per_chunk = max(1, CHUNK_BYTES // (row_len * item))
            flat = np.empty(min(per_chunk, rows[1] - rows[0]) * row_len, self._file_dtype)
            read = 0
            for a in range(rows[0], rows[1], per_chunk):
                n = min(per_chunk, rows[1] - a)
                span = (n - 1) * row_len + width * nc
                fh.seek(offset + (a * stride + cols[0]) * nc * item)
                _readinto(fh, flat[:span], self._path)
                rows_view = np.lib.stride_tricks.as_strided(
                    flat, shape=(n, width, nc), strides=(row_len * item, nc * item, item))
                dst[a - rows[0]:a - rows[0] + n] = rows_view[..., cs]
                read += span
            return read
        if cols != (0, stride) and direct and dst[0].flags.c_contiguous:
            width = cols[1] - cols[0]
            for r in range(rows[0], rows[1]):
                fh.seek(offset + (r * stride + cols[0]) * nc * item)
                _readinto(fh, dst[r - rows[0]], self._path)
            return (rows[1] - rows[0]) * width * nc
        if cols == (0, stride):
            # whole rows are contiguous: read them in chunks
            per_chunk = max(1, CHUNK_BYTES // (row_len * item))
            buf = np.empty((min(per_chunk, rows[1] - rows[0]), stride, nc), self._file_dtype)
            for a in range(rows[0], rows[1], per_chunk):
                n = min(per_chunk, rows[1] - a)
                fh.seek(offset + a * row_len * item)
                _readinto(fh, buf[:n], self._path)
                dst[a - rows[0]:a - rows[0] + n] = buf[:n, :, cs]
            return (rows[1] - rows[0]) * row_len
        width = cols[1] - cols[0]
        buf = np.empty((width, nc), self._file_dtype)
        for r in range(rows[0], rows[1]):
            fh.seek(offset + (r * stride + cols[0]) * nc * item)
            _readinto(fh, buf, self._path)
            dst[r - rows[0]] = buf[:, cs]
        return (rows[1] - rows[0]) * width * nc

    def _decode_tasks(self, decodes):
        """For each collected segment: copy it from the cache, or yield a
        task that reads, decodes and copies it on a pool thread -- the whole
        segment when it fits the memory budget, else only the rows asked
        for, streamed (``_stream_into``)."""
        for p, i, offset, count, jpegtables, dst, part in decodes:
            seg = self.decode_cache.get((p, i))
            if seg is not None:
                dst[...] = seg[part]
                continue
            rows, cols, samples = self._segment_shape(i)
            nbytes = rows * cols * samples * self._file_dtype.itemsize
            if _memory.should_stream(nbytes) and self._streamable():
                yield lambda job=(i, offset, count, dst, part): self._stream_into(*job)
                continue
            if _memory.should_stream(nbytes) and self._codec in AREA_DECODERS:
                yield lambda job=(offset, count, dst, part): self._area_into(*job)
                continue
            what = (f"{self._path!r}: a {rows} x {cols} TIFF segment compressed with "
                    f"{getattr(self._key.compression, 'name', self._key.compression)}")
            if _memory.should_stream(nbytes) and self._codec == "jpeg":
                yield lambda job=(p, i, offset, count, jpegtables, dst, part), n=nbytes, w=what: \
                    self._jpeg_rows_into(*job, n, w)
                continue
            if not _memory.fits(nbytes):
                raise _memory.too_large(what, nbytes)
            yield lambda job=(p, i, offset, count, jpegtables, dst, part), n=nbytes, w=what: \
                self._reserved(n, w, self._decode_into, *job)

    def _area_into(self, offset, count, dst, part) -> int:
        """Rows ``part[1]`` and columns ``part[2]`` of a JPEG 2000 or JPEG XL
        segment too large to cache, decoded alone (OpenJPEG decodes only the
        code-blocks covering them, jxl-oxide the groups), in bands of rows
        within the budget.  Runs on a pool thread."""
        decode = AREA_DECODERS[self._codec]
        r0, r1, c0, c1 = part[1].start, part[1].stop, part[2].start, part[2].stop
        data = self._handles.read(offset, count)
        row_bytes = (c1 - c0) * self._contig * self.dtype.itemsize
        per_band = max(1, min(STREAM_PIECE_BYTES, _memory.memory_limit() // 4) // row_bytes)
        for a in range(r0, r1, per_band):                   # bands within the budget
            b = min(r1, a + per_band)
            area = np.empty((b - a, c1 - c0, self._contig), self.dtype)
            with _memory.reserve(area.nbytes, f"{self._path!r}: part of a segment"):
                got = decode(data, area, c0, a, c1, b)
                if (got[1], got[0], got[2]) != area.shape:
                    raise CorruptFileError(
                        f"{self._path!r}: {self._codec} area decodes to {got[1]} x {got[0]} x "
                        f"{got[2]}, expected {area.shape}")
                dst[a - r0:b - r0] = area[..., part[3]]
        return (r1 - r0) * (c1 - c0) * self._contig

    def _jpeg_rows_into(self, p, i, offset, count, jpegtables, dst, part, nbytes, what) -> int:
        """Rows ``part[1]`` of a JPEG segment too large to cache: cut out at
        its restart markers and decoded in bands of rows (``_jpeg_restarts``);
        without restart markers, or progressive, decoded whole within the
        memory limit.  Runs on a pool thread."""
        index = self._restart_index(p, i, offset, count, jpegtables)
        row_bytes = self._segment_shape(i)[1] * self._contig * self.dtype.itemsize
        if not _jpeg_restarts.usable(index, row_bytes, _memory.fits):
            if not _memory.fits(nbytes):
                raise _memory.too_large(what + " that cannot be cut at restart markers", nbytes)
            return self._reserved(nbytes, what, self._decode_into,
                                  p, i, offset, count, jpegtables, dst, part)
        shift = len(jpegtables) - 4 if jpegtables else 0   # tables before the segment's header
        r0, r1 = part[1].start, part[1].stop
        piece = min(STREAM_PIECE_BYTES, _memory.memory_limit() // 4)
        for a, b, rows in _jpeg_restarts.read_rows(
                index, lambda s, e: self._handles.read(offset + s - shift, e - s), r0, r1,
                lambda data, out: _codecs.jpeg_decode_into(data, out, colour=self._jpeg_colour),
                (self._contig,), self.dtype, piece,
                lambda n: _memory.reserve(n, f"{self._path!r}: rows of a JPEG segment")):
            dst[a - r0:b - r0] = rows[:, part[2], part[3]]
        return (r1 - r0) * index.width * self._contig

    def _restart_index(self, p, i, offset, count, jpegtables):
        """The ``RestartIndex`` of JPEG segment *i* of page *p* (None: it
        cannot be cut), built once from the whole compressed segment."""
        key = (p, i)
        if key not in self._restarts:
            with _memory.reserve(count, f"{self._path!r}: a compressed JPEG segment"):
                data = self._handles.read(offset, count)
                if jpegtables:                    # abbreviated stream: tables from JPEGTables
                    data = bytes(jpegtables[:-2]) + bytes(data[2:])
                index = _jpeg_restarts.RestartIndex.build(data)
            with self._lock:
                self._restarts[key] = index
        return self._restarts[key]

    @staticmethod
    def _reserved(nbytes, what, decode, *job) -> int:
        """*decode(*job)* holding *nbytes* of the memory budget."""
        with _memory.reserve(nbytes, what):
            return decode(*job)

    def _streamable(self) -> bool:
        """Whether this page's segments can be decoded row by row: a byte
        codec micro-reader streams, no bit reversal (FillOrder 2)."""
        codec = self._codec
        return (codec in _stream.STREAMABLE and _stream.streamable(codec)
                and self._key.fillorder == 1 and codec in BYTE_CODECS.values())

    def _stream_into(self, i, offset, count, dst, part) -> int:
        """Rows ``part[1]`` of segment *i*, decoded as a stream, into *dst*
        -- in pieces of rows, so a tall region of a wide segment never
        holds more than a piece.  Runs on a pool thread."""
        key = self._key
        codec = self._codec
        _, cols, samples = self._segment_shape(i)
        bits = key.bitspersample
        packed = bits != 8 * self._file_dtype.itemsize
        row_bytes = (cols * samples * bits + 7) // 8 if packed else             cols * samples * self._file_dtype.itemsize
        r0, r1 = part[1].start, part[1].stop
        per_piece = max(1, min(STREAM_PIECE_BYTES, _memory.memory_limit() // 4) // row_bytes)
        for a in range(r0, r1, per_piece):
            b = min(r1, a + per_piece)
            with _memory.reserve((b - a) * row_bytes, f"{self._path!r}: rows of a segment"):
                buf = np.empty((b - a) * row_bytes, np.uint8)
                n = _stream.read(self._path, offset, count, codec, a * row_bytes, buf,
                                 size=self._segment_shape(i)[0] * row_bytes)
                if n < buf.nbytes:                       # short segment: zeros, as decoded whole
                    buf[n:] = 0
                if packed:
                    rows_ = unpack(buf, (b - a, cols, samples), bits, self._file_dtype,
                                   key.parent.byteorder)
                else:
                    rows_ = buf.view(self._file_dtype).reshape(b - a, cols, samples)
                if self._predictor != 1:
                    rows_ = unpredict(rows_, self._predictor)
                dst[a - r0:b - r0] = rows_[:, part[2], part[3]]
        return (r1 - r0) * cols * samples

    def _decode_into(self, p, i, offset, count, jpegtables, dst, part) -> int:
        """Read and decode segment *i* of page *p*, copy ``seg[part]`` into
        *dst*, keep it in the cache if it is small enough; return the number
        of pixels decoded.  Runs on a pool thread."""
        data = self._handles.read(offset, count)
        if self._codec is not None:
            seg = self._decode_segment(data, i, jpegtables)
        else:
            kwargs = {"_fullsize": False}
            if self._key.compression in JPEG_COMPRESSIONS:
                kwargs.update(jpegtables=jpegtables, jpegheader=self._key.jpegheader)
            try:
                seg, _, _ = self._decode(data, i, **kwargs)
            except Exception as exc:                        # noqa: BLE001 - codec specific
                raise self._decode_error(exc) from exc
        dst[...] = seg[part]
        self.decode_cache.put((p, i), seg)
        return seg.size

    def _segment_shape(self, i: int) -> tuple:
        """(rows, cols, samples) of segment *i*: tiles are always whole;
        the last strip of a plane holds only the rows that are left."""
        rows = self._seg_rows
        if not self._tiled:
            strip = i % self._segs_down
            rows = min(rows, self._length - strip * rows)
        return rows, self._seg_cols, self._contig

    def _decode_segment(self, data, i: int, jpegtables) -> np.ndarray:
        """Segment *i* with micro-reader's own codecs -> (1, rows, cols,
        samples), the shape tifffile's decoder gives."""
        shape = self._segment_shape(i)
        codec = self._codec
        if codec in BYTE_CODECS.values():
            if self._key.fillorder == 2:          # bits of every stored byte reversed
                data = _codecs.reverse_bits(data)
            bits = self._key.bitspersample
            packed = bits != 8 * self._file_dtype.itemsize
            rows, cols, samples = shape
            run_bytes = (cols * samples * bits + 7) // 8     # packed rows start on a byte
            # the stored bytes: the samples themselves, or packed ones
            stored = np.empty(rows * run_bytes, np.uint8) if packed else \
                np.empty(shape, self._file_dtype)
            if codec == "none":
                n = min(len(data), stored.nbytes)
                memoryview(stored).cast("B")[:n] = memoryview(data)[:n]
            elif codec == "zstd":
                _codecs.zstd_decompress_into(data, stored)
                n = stored.nbytes
            else:
                n = _BYTE_DECODERS[codec](data, stored, partial=True)
            if n < stored.nbytes:                 # short segment (e.g. LZW without EOI): zeros
                memoryview(stored).cast("B")[n:] = bytes(stored.nbytes - n)
            seg = unpack(stored, shape, bits, self._file_dtype, self._key.parent.byteorder) \
                if packed else stored
            if self._predictor != 1:
                seg = unpredict(seg, self._predictor)
            return seg[None]
        seg = np.empty(shape, self.dtype)
        if codec == "eer":
            _codecs.eer_decode_into(data, seg.reshape(shape[:2]), *eer_parameters(self._key))
            return seg[None]
        if codec == "jpeg":
            if jpegtables:                        # abbreviated stream: tables from JPEGTables
                data = bytes(jpegtables[:-2]) + bytes(data[2:])
            got = _codecs.jpeg_decode_into(data, seg, colour=self._jpeg_colour)
        else:
            got = _IMAGE_DECODERS[codec](data, seg)
        if (got[1], got[0], got[2]) != shape:
            raise CorruptFileError(
                f"{self._path!r}: segment {i} decodes to {got[1]} x {got[0]} x {got[2]}, "
                f"the TIFF tags say {shape[0]} x {shape[1]} x {shape[2]}")
        return seg[None]

    def _decode_error(self, exc: Exception) -> Exception:
        """A tifffile decode failure, explained when imagecodecs is missing."""
        try:
            import imagecodecs  # noqa: F401
        except ImportError:
            key = self._key
            return UnsupportedFormatError(
                f"{self._path!r}: TIFF compression {key.compression} "
                f"({getattr(key.compression, 'name', '?')}, predictor {key.predictor}, "
                f"{key.bitspersample}-bit, fill order {key.fillorder}) is decoded by "
                "imagecodecs, which is not installed: pip install imagecodecs")
        return CorruptFileError(f"{self._path!r}: {exc}")

    def _read_metadata(self):
        from ._tiff_metadata import tiff_metadata
        n = self.shape[self.axes.index("c")] if "c" in self.axes else 1
        return tiff_metadata(self, n, self._meta_cache)

    def close(self) -> None:
        """This image is unusable afterwards; the file handles it shares with
        the other images of its series close with the file."""
        self._closed = True

    def _release(self) -> None:
        self._handles.close()
        self.decode_cache.clear()
        _stream.forget(self._path)


class _TiffFile(File):
    backend = NAME

    def __init__(self, path, images, tif):
        self.format = tiff_format(tif)
        super().__init__(path, images)
        self._tif = tif

    def close(self) -> None:
        super().close()
        for image in self.images:
            getattr(image, "_release", lambda: None)()
        self._tif.close()


#: tifffile's file flags -> File.format, first match wins
TIFF_FORMATS = (("is_ome", "ome-tiff"), ("is_imagej", "imagej"), ("is_lsm", "lsm"),
                ("is_stk", "stk"), ("is_eer", "eer"), ("is_svs", "svs"), ("is_ndpi", "ndpi"))


def tiff_format(tif) -> str:
    """The TIFF variant: a microscopy flavour tifffile recognises, else
    "bigtiff" or "tiff"."""
    for flag, name in TIFF_FORMATS:
        if getattr(tif, flag, False):
            return name
    return "bigtiff" if tif.is_bigtiff else "tiff"


def _images(tif, series: list, path: str, tiles: str, resolutions: str = "base",
            rename: dict = None) -> list:
    """The images of *series*: one per series and view / tile / position;
    each with its pyramid (tifffile's ``series.levels``) as ``resolutions``,
    listed as images of their own with *resolutions* = "all"."""
    plans, levels = [], set()
    for i, s in enumerate(series):
        where = f"{os.path.basename(path)!r} series {i}"
        letters, frame = translate_axes(s.axes, lsm=bool(tif.is_lsm),
                                        eer=bool(getattr(s.keyframe, "is_eer", False)),
                                        where=where, rename=rename)
        # frame axes of more than one index make images; size 1 is just fixed
        by_level: dict = {}
        for pos, level in enumerate(frame):
            if level and s.shape[pos] > 1:
                by_level.setdefault(level, []).append(pos)
        levels.update(level for level in by_level if level != "scene")
        plans.append((s, letters, frame, by_level))
    levels = [level for level in OTHER_INDICES if level in levels]
    if tiles == "stitched" and "tile" in levels:
        raise UnsupportedFormatError(
            f"{path!r}: tiles='stitched': micro-reader does not compose TIFF tiles "
            "(each tile is its own image)")
    listed = resolutions == "all"
    images, scene = [], 0
    for n, (s, letters, frame, by_level) in enumerate(plans):
        positions = [pos for pos, level in enumerate(frame) if level]
        varying = [pos for pos in positions if s.shape[pos] > 1]
        first = None
        # lower resolution levels with the same axes (a pyramid shrinks y / x)
        lower = [lv for lv in list(getattr(s, "levels", None) or [s])[1:]
                 if len(lv.shape) == len(s.shape) and lv.axes == s.axes]
        level_first: dict = {}
        scenes_here = int(np.prod([s.shape[p] for p in by_level.get("scene", [])]))
        for values in itertools.product(*(range(s.shape[p]) for p in varying)):
            fixed = {pos: 0 for pos in positions}
            fixed.update(zip(varying, values))
            key = {"scene": scene + _ravel(fixed, by_level.get("scene", []), s.shape)}
            for level in levels:
                key[level] = _ravel(fixed, by_level.get(level, []), s.shape)
            if listed:
                key["resolution"] = 0
            image = _TiffImage(tif, len(images), s, letters, fixed=fixed, key=key, share=first)
            image._series_index = n                 # position in the listed series
            first = first or image
            images.append(image)
            pyramid = [image]
            for k, level_series in enumerate(lower, start=1):
                level_image = _TiffImage(tif, len(images), level_series, letters, fixed=fixed,
                                        key={**key, "resolution": k},
                                        share=level_first.get(k))
                level_image._series_index = n
                level_first.setdefault(k, level_image)
                pyramid.append(level_image)
                if listed:
                    images.append(level_image)
            if len(pyramid) > 1:
                for member in pyramid:
                    member._resolutions = pyramid
        scene += scenes_here
    images.sort(key=lambda im: tuple(im.index.values()))
    meta_cache: dict = {}                      # metadata parsed once per file
    for n, image in enumerate(images):
        image._position = n
        for member in image.resolutions:
            member._meta_cache = meta_cache
    return images


def _ravel(fixed: dict, positions: list, shape) -> int:
    """The flat index of *fixed*'s values at *positions* (0 without any)."""
    if not positions:
        return 0
    return int(np.ravel_multi_index([fixed[p] for p in positions],
                                    [shape[p] for p in positions]))


#: tifffile's own readings of a file's metadata, switched off to read it as plain
#: pages (its OME-XML may even list planes in other files: tifffile would open them)
PLAIN_PAGES = {"is_ome": False, "is_imagej": False, "is_shaped": False}


def open_pages(path: str) -> _TiffFile:
    """*path* as plain pages, whatever its own metadata says: for the members of
    a file set whose index file says what each page is (an OME companion,
    docs/COMPANION.md). One image per run of same-shaped pages, axes ``zyx``
    (``yx`` for a single page); ``File.pages``: IFD -> (image, page in it)."""
    try:
        import tifffile
    except ImportError as exc:
        raise MicroReaderError(
            "reading TIFF needs tifffile, a dependency of micro-reader: pip install tifffile") from exc
    try:
        tif = tifffile.TiffFile(path, **PLAIN_PAGES)
    except tifffile.TiffFileError as exc:
        raise CorruptFileError(f"{path!r}: {exc}") from exc
    return pages_file(tif, path)


def plain_pages(tif) -> bool:
    """Whether tifffile already reads *tif* as plain pages (an OME ``BinaryOnly``
    stub has no image of its own, so tifffile lists its pages as they are)."""
    return all(getattr(s, "kind", "") in ("generic", "uniform") for s in tif.series)


def pages_file(tif, path: str) -> _TiffFile:
    """``open_pages`` of an already open *tif* whose series are plain pages
    (``plain_pages``); *tif* is closed on error."""
    try:
        series = list(tif.series)
        if not series:
            raise CorruptFileError(f"{path!r} contains no image series")
        images = _images(tif, series, path, "separate", "base", {"sequence": "z"})
        pages = {}
        for image in images:
            for k, page in enumerate(image._series.pages):
                if page is not None:
                    pages[page.index] = (image, k)
    except BaseException:
        tif.close()
        raise
    file = _TiffFile(path, images, tif)
    file.pages = pages
    return file


def open(path: str, tiles: str = "separate", resolutions: str = "base",  # noqa: A001 - backend protocol
         rename: dict = None, companion: bool = True) -> _TiffFile:
    try:
        import tifffile
    except ImportError as exc:
        raise MicroReaderError(
            "reading TIFF needs tifffile, a dependency of micro-reader: pip install tifffile") from exc
    try:
        tif = tifffile.TiffFile(path)
    except tifffile.TiffFileError as exc:
        raise CorruptFileError(f"{path!r}: {exc}") from exc
    try:
        if companion and tif.is_ome:
            # a BinaryOnly stub: its metadata is in an OME companion (docs/COMPANION.md,
            # CO-2); companion=False (internal: file sets, whose index file decides) skips it
            from . import companion
            view = companion.member_view(path, tif)    # takes over *tif* when it is one
            if view is not None:
                return view
        series = list(tif.series)
        if tif.is_lsm:
            # Zeiss LSM stores a small RGB preview of every plane (reduced
            # pages); Bio-Formats does not list them, and neither does
            # micro-reader: they are not data
            series = [s for s in series if not s.keyframe.is_reduced]
        if not series:
            raise CorruptFileError(f"{path!r} contains no image series")
        images = _images(tif, series, path, tiles, resolutions, rename)
    except BaseException:
        tif.close()
        raise
    return _TiffFile(path, images, tif)
