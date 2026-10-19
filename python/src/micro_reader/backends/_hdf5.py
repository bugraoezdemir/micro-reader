"""What every HDF5 backend shares (Imaris, ilastik, BigDataViewer): telling the
layouts apart, and reading chunked datasets in parallel.

**Layouts.**  An HDF5 file says nothing about its layout in its first bytes, so
``kind(path)`` opens it with h5py once and looks at its structure (cached per
file): Imaris (``DataSet/ResolutionLevel n``), BigDataViewer (``s00/resolutions``
next to ``t00000/s00``), an ilastik project (``workflowName``: not an image),
ilastik data (datasets with an ``axistags`` attribute), or none of these.

**Reading.**  h5py runs every HDF5 call -- decompression included -- under one
global lock, so threads do not speed it up.  ``ChunkReader`` asks h5py only
where each chunk is (``chunk_iter``), reads the chunks with per-thread file
handles and decompresses them on the shared decode pool.  Filters handled:
deflate (gzip), shuffle, Fletcher-32 (checked), LZ4 (ImarisWriter), zstd,
scaleoffset (integers).  A dataset with any other filter (szip, Blosc, ...) or
without chunks is read through h5py instead: correct, just not parallel.
Chunks of datasets in other files (external links: BigDataViewer's partitioned
files) are read from those files.  A read h5py fails on raises
CorruptFileError, but for one known refusal (``ChunkReader._refused``).
"""
from __future__ import annotations

import builtins
import itertools
import os
import re
import threading
from typing import Optional

import numpy as np

from .. import _codecs, _memory, _pool
from .._errors import CorruptFileError, MicroReaderError, UnsupportedFormatError
from .._io import DecodeCache
from .._parts import Handles

HDF5_MAGIC = b"\x89HDF\r\n\x1a\n"
DECODE_CACHE_BYTES = 64 << 20
FILTER_DEFLATE, FILTER_SHUFFLE, FILTER_FLETCHER32 = 1, 2, 3
FILTER_LZ4 = 32004                   # HDF Group registered filter; ImarisWriter writes it
FILTER_ZSTD = 32015                  # registered zstd filter: one zstd frame per chunk
FILTER_SCALEOFFSET = 6               # built into HDF5; BigDataViewer writes it (integers only here)
SUPPORTED_FILTERS = {FILTER_DEFLATE, FILTER_SHUFFLE, FILTER_FLETCHER32, FILTER_LZ4, FILTER_ZSTD,
                     FILTER_SCALEOFFSET}
#: HDF5 filters by number (errors name them)
FILTER_NAMES = {1: "deflate", 2: "shuffle", 3: "Fletcher-32", 4: "szip", 6: "scaleoffset",
                307: "bzip2", 32000: "LZF", 32001: "Blosc", 32004: "LZ4", 32008: "bitshuffle", 32013: "zfp", 32015: "zstd", 32017: "SZ",
                32026: "Blosc2"}
#: filters h5py applies itself (a failure with them is the data, not a missing plugin)
H5PY_FILTERS = {FILTER_DEFLATE, FILTER_SHUFFLE, FILTER_FLETCHER32, 4, FILTER_SCALEOFFSET, 32000}
#: chunks at least this large are decoded here, in parallel; smaller ones through
#: h5py (Layout.prefer_native)
NATIVE_MIN_CHUNK = 32 << 10
NATIVE_MIN_CHUNK_SCALEOFFSET = 1 << 20
#: a read needing fewer chunks than this share of a dataset's chunks looks them
#: up one by one instead of mapping every chunk (measured 2026-10-18, a Luxendo
#: file of 2048 chunks: the whole map 1.4 ms = 0.7 us a chunk, one lookup 3.7 us)
LOOKUP_SHARE = 0.2
#: one read call costs about as much as reading this many more bytes (Python
#: call + seek; page cache ~5 GB/s): an uncompressed chunk part is read as one
#: span from its first value to its last when the span is not larger than the
#: part plus this much per read it saves
READ_CALL_BYTES = 64 << 10
#: scaleoffset: parameters (H5Zscaleoffset.c) and the bytes before the packed values
SO_PARAMETERS, SO_HEADER, SO_INTEGER, SO_INT_SCALE = 20, 21, 0, 2
#: datasets ``classify`` looks at for an ``axistags`` attribute before giving up
#: (a file with millions of datasets is not walked whole to classify it; the
#: listing of a file's images walks it whole)
MAX_VISITED = 2000
#: bytes a decoding stage's output may exceed the decoded chunk by: room for
#: scaleoffset's header (a full-precision chunk is header + values) and
#: Fletcher-32 checksums; a stage claiming more is damage
STAGE_SLACK = 64


def is_hdf5(head: bytes) -> bool:
    """The HDF5 signature: at byte 0, or after a user block of 512 * 2**n bytes."""
    return head[:8] == HDF5_MAGIC or head[512:520] == HDF5_MAGIC


def h5py_module(what: str = "reading HDF5 files"):
    """h5py, or the error naming the extra to install."""
    try:
        import h5py
    except ImportError as exc:
        raise MicroReaderError(
            f"{what} needs h5py: pip install 'micro-reader[hdf5]'") from exc
    return h5py


def attr_text(attrs, key) -> Optional[str]:
    """An attribute as text: str, bytes, or an array of 1-byte strings (Imaris)."""
    if key not in attrs:
        return None
    value = attrs[key]
    if isinstance(value, np.ndarray):
        if value.dtype.kind in "SU" and value.ndim and value.size > 1:
            return b"".join(v if isinstance(v, bytes) else str(v).encode()
                            for v in value.ravel()).decode(errors="replace")
        value = value.ravel()[0] if value.size else ""
    return value.decode(errors="replace") if isinstance(value, bytes) else str(value)


# -- which layout --------------------------------------------------------------

_kinds: dict = {}
_kinds_lock = threading.Lock()
SETUP = re.compile(r"^s\d{2,}$")
TIMEPOINT = re.compile(r"^t\d{5,}$")


def kind(path: str) -> Optional[str]:
    """"ims", "bdv", "ilastik-project", "ilastik", "luxendo", "fiji-hdf5", or
    None (an HDF5 file of another layout, or h5py missing / the file
    unreadable).  Cached per file (path, size, modification time)."""
    try:
        st = os.stat(path)
    except OSError:
        return None
    key = (os.path.normcase(os.path.abspath(path)), st.st_size, st.st_mtime_ns)
    with _kinds_lock:
        if key in _kinds:
            return _kinds[key]
    try:
        import h5py
        with h5py.File(path, "r") as h5:
            found = classify(h5)
    except (ImportError, OSError, CorruptFileError):
        # not readable here: the ``ims`` backend, which takes every HDF5 file
        # the others do not, opens it and raises the typed error
        found = None
    with _kinds_lock:
        if len(_kinds) > 256:
            _kinds.clear()
        _kinds[key] = found
    return found


def classify(h5) -> Optional[str]:
    """The layout of an open h5py File (see ``kind``).  A member of an
    unexpected kind (a dataset where a layout has a group) or a broken link
    only means "not this layout"; the file failing to read raises
    CorruptFileError."""
    import h5py
    try:
        names = set(h5.keys())
        imaris = _member(h5, "DataSet", h5py.Group)
        if imaris is not None and any(n.startswith("ResolutionLevel ") for n in imaris):
            return "ims"
        setups = [_member(h5, n, h5py.Group) for n in names if SETUP.match(n)]
        if any(TIMEPOINT.match(n) for n in names) and \
                any(s is not None and "resolutions" in s for s in setups):
            return "bdv"
        if "workflowName" in names or "ilastikVersion" in names:
            return "ilastik-project"
        if axistags_datasets(h5, limit=1, cap=MAX_VISITED):
            return "ilastik"
        metadata = _member(h5, "metadata", h5py.Dataset)
        if _member(h5, "Data", h5py.Dataset) is not None and metadata is not None \
                and b"processingInformation" in _as_bytes(metadata):
            return "luxendo"
        if element_size_datasets(h5, limit=1, cap=MAX_VISITED):
            return "fiji-hdf5"
        return None
    except OSError as exc:
        raise CorruptFileError(f"{h5.filename!r}: {exc}") from exc


def _member(group, name: str, kind):
    """``group[name]`` when it is a *kind* (h5py Group or Dataset), else None
    -- also for a missing member or a broken link (h5py: KeyError)."""
    obj = group.get(name)
    return obj if isinstance(obj, kind) else None


def _as_bytes(ds) -> bytes:
    """A small dataset's value as bytes (Luxendo's JSON metadata), or b''."""
    try:
        if not hasattr(ds, "shape") or ds.shape not in ((), (1,)):
            return b""
        value = ds[()]
        value = value.ravel()[0] if isinstance(value, np.ndarray) else value
        return value if isinstance(value, bytes) else str(value).encode()
    except (OSError, TypeError, ValueError):
        return b""


def element_size_datasets(h5, limit: Optional[int] = None, cap: Optional[int] = None) -> list:
    """Paths of the datasets with an ``element_size_um`` attribute (Fiji HDF5),
    as ``axistags_datasets`` looks for ``axistags``."""
    return _datasets_with(h5, "element_size_um", limit, cap)


def axistags_datasets(h5, limit: Optional[int] = None, cap: Optional[int] = None) -> list:
    """Paths of the datasets that carry an ``axistags`` attribute, in name
    order: at most *limit*, among the first *cap* datasets (``classify``
    passes ``MAX_VISITED``; a listing of the file's images looks at all)."""
    return _datasets_with(h5, "axistags", limit, cap)


def _datasets_with(h5, attribute: str, limit: Optional[int], cap: Optional[int]) -> list:
    import h5py
    found, seen = [], [0]

    def visit(name, obj):
        if not isinstance(obj, h5py.Dataset):
            return None
        seen[0] += 1
        if attribute in obj.attrs:
            found.append(name)
        if (limit is not None and len(found) >= limit) or (cap is not None and seen[0] >= cap):
            return True                             # stop walking
        return None
    h5.visititems(visit)
    return sorted(found)


def datasets(h5, limit: int = 20) -> list:
    """'name shape dtype' of the first datasets of a file, for error messages."""
    import h5py
    out = []

    def visit(name, obj):
        if isinstance(obj, h5py.Dataset):
            out.append(f"{name} {obj.shape} {obj.dtype}")
        return True if len(out) >= limit else None
    h5.visititems(visit)
    return out


# -- reading chunks ------------------------------------------------------------

def _lz4_unframe(data, where: str, limit: int) -> np.ndarray:
    """HDF5's LZ4 filter (H5Zlz4.c): big-endian original size (8 bytes) and
    block size (4), then per block its compressed size (4) and the LZ4
    block -- stored as is when the compressed size equals the block size.
    An original size above *limit* (what the chunk can decode to) is damage."""
    if len(data) < 12:
        raise CorruptFileError(f"{where}: an LZ4 chunk of {len(data)} bytes")
    total = int.from_bytes(data[:8], "big")
    if total > limit:
        raise CorruptFileError(f"{where}: an LZ4 chunk claims {total} bytes, "
                               f"the chunk holds at most {limit}")
    block = min(int.from_bytes(data[8:12], "big"), total) or total
    out = np.empty(total, np.uint8)
    pos, done = 12, 0
    while done < total:
        size = min(block, total - done)
        stored = int.from_bytes(data[pos:pos + 4], "big")
        pos += 4
        if pos + stored > len(data):
            raise CorruptFileError(f"{where}: LZ4 chunk ends early")
        if stored == size:
            out[done:done + size] = np.frombuffer(data, np.uint8, size, pos)
        else:
            try:
                _codecs.lz4_block_decompress_into(memoryview(data)[pos:pos + stored],
                                                  out[done:done + size])
            except CorruptFileError as exc:
                raise CorruptFileError(f"{where}: {exc}") from exc
        pos += stored
        done += size
    return out


#: Fletcher-32 sums are taken modulo this; ``_FLETCHER_INDEX``: positions in a
#: period of it
_FLETCHER_MOD = 65535
_FLETCHER_INDEX = np.arange(_FLETCHER_MOD, dtype=np.uint64)


def fletcher32(data) -> int:
    """HDF5's Fletcher-32 checksum of *data* (H5_checksum_fletcher32,
    H5checksum.c): the bytes as big-endian 16-bit words (an odd last byte is
    the high byte of one more), ``sum1`` the sum of the words and ``sum2``
    the sum of the running ``sum1``s, both folded to 16 bits -> ``sum2 << 16
    | sum1``.

    HDF5 folds every 360 words (``x & 0xffff`` + ``x >> 16``), which keeps
    the value modulo 65535 and is 0 only for 0: the result is the exact sums
    modulo 65535, written 65535 where that is 0 for a non-zero sum.  The
    word at index i of n counts n - i times in ``sum2``; that weight is the
    same modulo 65535 for every index in the same column of rows of 65535
    words, so the words are summed per column (numpy) and the columns
    weighted once."""
    b = np.frombuffer(data, np.uint8)
    n = (b.size + 1) // 2                                   # words, an odd byte included
    words = b[:b.size // 2 * 2].view(">u2")
    rows = words.size // _FLETCHER_MOD
    if rows:
        columns = words[:rows * _FLETCHER_MOD].reshape(rows, _FLETCHER_MOD).sum(
            axis=0, dtype=np.uint64)
        columns[:words.size - rows * _FLETCHER_MOD] += words[rows * _FLETCHER_MOD:]
    else:
        columns = words.astype(np.uint64)
    if b.size % 2:
        last = np.uint64(int(b[-1]) << 8)
        if not rows:                                        # n <= 65535 columns
            columns = np.append(columns, last)
        else:
            columns[(n - 1) % _FLETCHER_MOD] += last
    # no overflow: a column holds at most 65535 * 32769 (a 4 GiB chunk)
    sum1 = int(columns.sum(dtype=np.uint64))
    if sum1 == 0:
        return 0
    weighted = int(np.dot(columns, _FLETCHER_INDEX[:columns.size]))
    sum2 = (n % _FLETCHER_MOD * sum1 - weighted) % _FLETCHER_MOD or _FLETCHER_MOD
    sum1 = sum1 % _FLETCHER_MOD or _FLETCHER_MOD
    return (sum2 << 16) | sum1


def _fletcher_strip(data, where: str):
    """A chunk stage with HDF5's Fletcher-32 checksum (H5Zfletcher32.c: the
    last 4 bytes, little endian) checked -> the bytes before it.  HDF5 also
    accepts the checksum of its old, byte-swapped version."""
    if len(data) < 4:
        raise CorruptFileError(f"{where}: a Fletcher-32 chunk of {len(data)} bytes")
    view = memoryview(data).cast("B")
    stored = int.from_bytes(view[-4:], "little")
    body = view[:-4]
    found = fletcher32(body)
    swapped = ((found & 0x00FF00FF) << 8) | ((found >> 8) & 0x00FF00FF)
    if stored not in (found, swapped):
        raise CorruptFileError(f"{where}: the Fletcher-32 checksum does not match "
                               f"(stored {stored:#010x}, data {found:#010x}): damaged data")
    return body


def _scaleoffset_native(cd: tuple) -> bool:
    """Whether a scaleoffset filter with parameters *cd* is decoded here:
    integer data (BigDataViewer's); floating-point scaling goes through h5py."""
    return len(cd) == SO_PARAMETERS and cd[3] == SO_INTEGER and cd[0] == SO_INT_SCALE \
        and cd[4] in (1, 2, 4, 8)


def scaleoffset_full_precision(data, cd: tuple) -> bool:
    """Whether a scaleoffset chunk (filter parameters *cd*) is stored at full
    precision: ``minbits`` equal to the type's width, the values as they are
    after the header.  HDF5 writes such chunks; HDF5 1.14's reader refuses
    them (h5py 3.15: "filter returned failure"), HDF5 2.0's reads them."""
    width = 8 * cd[4]
    return cd[1] != width and len(data) >= 4 and \
        int.from_bytes(memoryview(data).cast("B")[:4], "little") == width


def scaleoffset_decode(data, cd: tuple, where: str):
    """Undo HDF5's scaleoffset filter for integers (H5Zscaleoffset.c).

    Parameters *cd*: scale type, scale factor, elements per chunk, class,
    size, sign, byte order, fill value defined, the fill value (bytes, least
    significant first, 4 per parameter).  The chunk: ``minbits`` (4 bytes,
    little endian), the size of ``minval`` (1 byte) and ``minval`` (little
    endian; 21 bytes in all), then every element as ``minbits`` bits, most
    significant first, packed without gaps.  An element is its bits plus
    ``minval`` (wrapping at the type's width); with a fill value defined, all
    bits set means the fill value.  A scale factor equal to the type's width
    stores the chunk unchanged.  -> the chunk's bytes (uint8 array, or *data*).

    Values of 1, 2 or 4 bytes are unpacked by the native module straight into
    an array of their type (``_unpack_bits``, numpy, for 8-byte values or
    without it)."""
    _scale_type, factor, n, _cls, size, _sign, order, fill_defined = cd[:8]
    width = 8 * size
    if factor == width:
        return data
    b = np.frombuffer(data, np.uint8)
    if b.size < SO_HEADER:
        raise CorruptFileError(f"{where}: a scaleoffset chunk of {b.size} bytes")
    minbits = int.from_bytes(b[:4].tobytes(), "little")
    unsigned_type = np.dtype(f"<u{size}")
    if minbits == width:
        # full precision: the writer stores the values as they are (in its own
        # byte order, little endian here); HDF5 1.14's reader refuses such
        # chunks (pybdv's example.h5; HDF5 2.0 reads them): read as written
        raw = b[SO_HEADER:SO_HEADER + n * size]
        if raw.size < n * size:
            raise CorruptFileError(f"{where}: a scaleoffset chunk ends early")
        out = raw.view(unsigned_type)
        return (out.astype(unsigned_type.newbyteorder(">")) if order == 1 else out).view(np.uint8)
    if minbits > width:
        raise CorruptFileError(f"{where}: scaleoffset chunk with {minbits} bits per value "
                               f"for a {width}-bit type")
    minval = int.from_bytes(b[5:5 + min(8, int(b[4]))].tobytes(), "little")
    if minbits == 0:
        values = np.zeros(n, unsigned_type)
    else:
        packed = b[SO_HEADER:]
        need = (n * minbits + 7) // 8
        if packed.size < need:
            raise CorruptFileError(f"{where}: a scaleoffset chunk ends early")
        if size <= 4 and _codecs.native_available():
            values = np.empty(n, unsigned_type)
            if _codecs.unpack_bits_into(packed[:need], values, minbits, n, native=True) != n:
                raise CorruptFileError(f"{where}: a scaleoffset chunk ends early")
        else:
            values = _unpack_bits(packed[:need], n, minbits)            # uint64
    fills = values == values.dtype.type((1 << minbits) - 1) if fill_defined else None
    values += values.dtype.type(minval & ((1 << 8 * values.dtype.itemsize) - 1))  # wraps
    out = values if values.dtype == unsigned_type else values.astype(unsigned_type)
    if fills is not None:
        fill_bytes = b"".join(int(v & 0xFFFFFFFF).to_bytes(4, "little") for v in cd[8:])
        out[fills] = int.from_bytes(fill_bytes[:size], "little")
    if order == 1:                                              # the dataset is big-endian
        out = out.byteswap(inplace=True).view(unsigned_type.newbyteorder(">"))
    return out.view(np.uint8)


def _unpack_bits(packed: np.ndarray, n: int, bits: int) -> np.ndarray:
    """*n* unsigned values of *bits* bits each, most significant bit first,
    from the byte array *packed* -> uint64."""
    if bits <= 57:
        # 8 values fill exactly *bits* bytes, so value j of every group of 8
        # starts at the same byte and bit of its group: one strided 8-byte
        # read per j over all groups instead of one per value
        groups = -(-n // 8)
        whole = np.zeros(groups * bits, np.uint8)
        whole[:min(packed.size, whole.size)] = packed[:whole.size]
        rows = np.zeros((groups, bits + 8), np.uint8)          # 8 spare bytes per group
        rows[:, :bits] = whole.reshape(groups, bits)
        out = np.empty(groups * 8, np.uint64)
        mask = np.uint64((1 << bits) - 1)
        for j in range(8):
            start, skip = divmod(j * bits, 8)
            words = np.ascontiguousarray(rows[:, start:start + 8]).view(">u8").ravel()
            out[j::8] = (words >> np.uint64(64 - skip - bits)) & mask
        return out[:n]
    # bit by bit (8 bytes a bit), 4096 values (a whole number of bytes) at a time
    weights = np.uint64(1) << np.arange(bits - 1, -1, -1, dtype=np.uint64)
    out = np.empty(n, np.uint64)
    for first in range(0, n, 4096):
        m = min(4096, n - first)
        part = packed[first * bits // 8:((first + m) * bits + 7) // 8]
        stream = np.unpackbits(part)[:m * bits].reshape(m, bits).astype(np.uint64)
        out[first:first + m] = (stream * weights).sum(axis=1, dtype=np.uint64)
    return out


def unsigned(dtype: np.dtype) -> np.dtype:
    """The unsigned integer type of *dtype*'s size and byte order."""
    return np.dtype(f"u{dtype.itemsize}").newbyteorder(dtype.byteorder)


class Layout:
    """How one dataset is stored: where each chunk is and how to decode it;
    or ``native = False`` to read it through h5py.  *view*: a dtype of the
    same size the stored values are reinterpreted as (BigDataViewer stores
    unsigned 16-bit pixels as signed)."""

    def __init__(self, ds, view: Optional[np.dtype] = None):
        self.path = ds.file.filename                # an external link's own file
        self.dtype = ds.dtype                       # file byte order
        self.view = view
        self.fill = ds.fillvalue
        self.chunks = ds.chunks
        self.shape = ds.shape
        dcpl = ds.id.get_create_plist()
        pipeline = [dcpl.get_filter(i) for i in range(dcpl.get_nfilters())]
        self.filters = [f[0] for f in pipeline]
        #: each filter's parameters (cd_values), as stored
        self.parameters = [tuple(int(v) for v in f[2]) for f in pipeline]
        self.native = bool(self.chunks) and set(self.filters) <= SUPPORTED_FILTERS and all(
            _scaleoffset_native(p) for f, p in zip(self.filters, self.parameters)
            if f == FILTER_SCALEOFFSET)
        #: a contiguous (unchunked) dataset: the byte offset of its data, read
        #: directly; None: chunked, or not stored in the file (never written,
        #: or in an external raw file)
        self.offset = None
        if not self.chunks and not self.filters and ds.id.get_storage_size() and \
                dcpl.get_external_count() == 0 and ds.id.get_offset() is not None:
            self.offset = self._contiguous_offset(ds)
        self.name = ds.name
        self.chunk_nbytes = int(np.prod(self.chunks)) * self.dtype.itemsize if self.chunks else 0
        #: the most bytes a decoding stage gives out (more is damage)
        self.stage_nbytes = self.chunk_nbytes + STAGE_SLACK
        #: decode here rather than through h5py: when h5py cannot apply a filter,
        #: or the chunks are large enough for parallel decoding to beat h5py's C
        #: path (measured 2026-10-17: deflate wins from 32 KiB chunks, up to 8x at
        #: 1 MiB; scaleoffset only from 1 MiB -- h5py's is fast C, ours numpy)
        threshold = (NATIVE_MIN_CHUNK_SCALEOFFSET if FILTER_SCALEOFFSET in self.filters
                     else NATIVE_MIN_CHUNK)
        self.prefer_native = self.native and (
            any(f not in H5PY_FILTERS for f in self.filters) or self.chunk_nbytes >= threshold)
        self.n_chunks = int(np.prod([-(-n // c) for n, c in zip(self.shape, self.chunks)])) \
            if self.chunks else 0
        self._ds = ds
        self._where: Optional[dict] = None
        self._where_lock = threading.Lock()
        #: chunks looked up one by one (``locate``) before any full map; the
        #: user-block shift found on the first of them (None: not yet)
        self._found: dict = {}
        self._shift: Optional[int] = None

    @property
    def where(self) -> dict:
        """Chunk origin -> (byte offset, size, filter mask), mapped on first
        use (a dataset of many chunks costs milliseconds to map, wasted when
        every read goes through h5py).  A chunk position that cannot be
        trusted (``_userblock_shift``) turns ``native`` off: {} then.  The
        map is complete when it is published (other threads read it unlocked);
        ``native`` is off before an empty one is."""
        if self._where is None:
            with self._where_lock:
                if self._where is None:
                    self._where = self._map_chunks()
        return self._where

    def locate(self, origins: list) -> dict:
        """Chunk origin -> (byte offset, size, filter mask), or None for a chunk
        never written, for the chunks one read needs.  A few of a dataset's
        chunks are looked up one by one; many (``LOOKUP_SHARE``) map them all
        (``where``).  Like ``where``, may turn ``native`` off (chunk positions
        that cannot be trusted): {} then."""
        if self._where is None and \
                len(origins) + len(self._found) < LOOKUP_SHARE * self.n_chunks:
            with self._where_lock:
                if self._where is None:
                    return self._look_up(origins)
        where = self.where
        return {o: where.get(o) for o in origins} if self.native else {}

    def _look_up(self, origins: list) -> dict:
        out = {}
        for origin in origins:
            if origin not in self._found:
                info = self._ds.id.get_chunk_info_by_coord(origin)
                found = None if info.byte_offset is None else \
                    (info.byte_offset, info.size, info.filter_mask)
                if found is not None and self._shift is None:
                    self._shift = self._userblock_shift(self._ds, {origin: found})
                    if self._shift is None:             # positions unknown: read through h5py
                        self.native = self.prefer_native = False
                        return {}
                if found is not None and self._shift:
                    found = (found[0] + self._shift,) + found[1:]
                self._found[origin] = found
            out[origin] = self._found[origin]
        return out

    def _map_chunks(self) -> dict:
        where: dict = {}
        if not self.native:
            return where
        self._ds.id.chunk_iter(lambda info: where.__setitem__(
            tuple(info.chunk_offset), (info.byte_offset, info.size, info.filter_mask)))
        shift = self._userblock_shift(self._ds, where)
        if shift is None:                               # positions unknown: read through h5py
            self.native = self.prefer_native = False
            return {}
        if shift:
            where = {k: (o + shift, n, m) for k, (o, n, m) in where.items()}
        return where

    def _contiguous_offset(self, ds):
        """Where a contiguous dataset's data starts in the file, checked
        against h5py's first values (the user-block question of
        ``_userblock_shift``); None: read it through h5py."""
        offset = ds.id.get_offset()
        userblock = ds.file.id.get_create_plist().get_userblock()
        if not userblock:
            return offset
        n = min(int(np.prod(self.shape)) if self.shape else 1, 4096)
        first = np.ascontiguousarray(ds[()] if not self.shape else
                                     ds[tuple(slice(0, 1) for _ in self.shape[:-1])
                                        + (slice(0, min(n, self.shape[-1])),)]).tobytes()
        with builtins.open(self.path, "rb") as fh:
            for shift in (0, userblock):
                fh.seek(offset + shift)
                if fh.read(len(first)) == first:
                    return offset + shift
        return None

    def _userblock_shift(self, ds, where: dict):
        """Bytes to add to the chunk offsets HDF5 reports (*where*, as
        mapped).  Before HDF5 1.14.6 (seen with 1.14.2) they leave out a user
        block at the start of the file; later versions include it.  One
        chunk's stored bytes (read_direct_chunk) are compared with the file
        at both positions; None when neither matches."""
        userblock = ds.file.id.get_create_plist().get_userblock()
        if not userblock or not where:
            return 0
        origin, (offset, size, _) = next(iter(where.items()))
        _, stored = ds.id.read_direct_chunk(origin)
        stored = bytes(stored)
        with builtins.open(self.path, "rb") as fh:
            for shift in (0, userblock):
                fh.seek(offset + shift)
                if fh.read(len(stored)) == stored:
                    return shift
        return None

    def decode_nbytes(self, size: int) -> int:
        """Bytes decoding a chunk stored in *size* bytes holds at most: the
        stored bytes, a stage's input and output, deflate's second copy without
        the native module, and scaleoffset's temporaries (a fill mask; numpy's
        unpacking -- 8-byte values, or without the native module -- works in
        uint64, ~40 bytes a value)."""
        peak = size + 2 * self.stage_nbytes
        if FILTER_DEFLATE in self.filters and not _codecs.native_available():
            # the standard library's zlib builds its result from blocks it joins
            # at the end: the decoded stage twice for a moment, besides the output
            peak += self.stage_nbytes
        if FILTER_SCALEOFFSET in self.filters:
            n = self.chunk_nbytes // self.dtype.itemsize
            peak += n if self.dtype.itemsize <= 4 and _codecs.native_available() else 40 * n
        return peak

    def where_text(self, origin: tuple = ()) -> str:
        """The file, dataset and chunk, for errors."""
        return f"{self.path!r}, dataset {self.name!r}" + (f", chunk at {origin}" if origin else "")

    def decode(self, data, mask: int, origin: tuple = (), full: Optional[list] = None) -> np.ndarray:
        """Undo the filter pipeline (in reverse; bit i of *mask* set means
        filter i was skipped for this chunk).  Every stage's output is bounded
        by ``stage_nbytes``; Fletcher-32 checksums are checked.  *full*: a
        list the chunk's *origin* is added to when it is a full-precision
        scaleoffset chunk (``ChunkReader._refused``)."""
        where = None
        for i in reversed(range(len(self.filters))):
            if mask & (1 << i):
                continue
            fid = self.filters[i]
            where = where or self.where_text(origin)
            if fid == FILTER_DEFLATE:
                out = np.empty(self.stage_nbytes, np.uint8)
                try:
                    written = _codecs.zlib_decompress_into(data, out, partial=True)
                except CorruptFileError as exc:
                    raise CorruptFileError(f"{where}: {exc}") from exc
                data = out[:written]
            elif fid == FILTER_LZ4:
                data = _lz4_unframe(data, where, self.stage_nbytes)
            elif fid == FILTER_ZSTD:
                out = np.empty(self.chunk_nbytes, np.uint8)
                try:
                    _codecs.zstd_decompress_into(data, out)
                except CorruptFileError as exc:
                    raise CorruptFileError(f"{where}: {exc}") from exc
                data = out
            elif fid == FILTER_SHUFFLE:
                item = self.dtype.itemsize
                if item > 1:
                    b = np.frombuffer(data, np.uint8)
                    n = b.size // item
                    out = np.empty(b.size, np.uint8)
                    out[:n * item].reshape(n, item)[...] = b[:n * item].reshape(item, n).T
                    out[n * item:] = b[n * item:]
                    data = out
            elif fid == FILTER_SCALEOFFSET:
                if full is not None and scaleoffset_full_precision(data, self.parameters[i]):
                    full.append(origin)
                data = scaleoffset_decode(data, self.parameters[i], where)
            elif fid == FILTER_FLETCHER32:
                data = _fletcher_strip(data, where)
        if len(data) != self.chunk_nbytes:
            raise CorruptFileError(f"{where or self.where_text(origin)}: chunk decodes to "
                                   f"{len(data)} bytes, expected {self.chunk_nbytes}")
        chunk = np.frombuffer(data, self.dtype).reshape(self.chunks)
        return chunk.view(self.view) if self.view is not None else chunk


def _row_reads(part: tuple, shape: tuple) -> int:
    """Reads ``ChunkReader._plan_contiguous`` makes for *part* of an array of
    *shape*: the axes read whole at the end merge with the last partial one
    into rows, and rows close together of the axis before it merge too -- so
    one read per index of the axes before those."""
    k = len(shape)
    while k > 1 and (part[k - 1].start, part[k - 1].stop) == (0, shape[k - 1]):
        k -= 1
    return int(np.prod([p.stop - p.start for p in part[:max(k - 2, 0)]]))


class _ChunkSpan:
    """One uncompressed chunk seen as a contiguous dataset of the chunk's shape
    (stored whole, also at the edges) at its byte offset, for
    ``ChunkReader._plan_contiguous``."""

    def __init__(self, layout: Layout, offset: int):
        self.path, self.dtype, self.view = layout.path, layout.dtype, layout.view
        self.shape, self.offset = layout.chunks, offset


class ChunkReader:
    """Reads rectangles of the chunked datasets of one HDF5 file (and the files
    it links to): ``plan`` collects the chunks a rectangle needs, ``run``
    decodes them on the decode pool.  One per open file; every h5py call goes
    through ``lock``."""

    def __init__(self, path: str, lock: Optional[threading.Lock] = None):
        self.path = path
        self.lock = lock or threading.Lock()
        self.handles = Handles()
        #: decoded chunks kept for reuse (``cache.max_bytes`` to resize)
        self.cache = DecodeCache(DECODE_CACHE_BYTES)
        self._layouts: dict = {}

    def layout(self, key, ds, view: Optional[np.dtype] = None) -> Layout:
        found = self._layouts.get(key)
        if found is None:
            with self.lock:
                found = self._layouts.get(key)
                if found is None:
                    found = self._layouts[key] = Layout(ds, view)
        return found

    def plan(self, key, ds, slices: tuple, dest: np.ndarray, jobs: list,
             view: Optional[np.dtype] = None) -> int:
        """Fill *dest* with ``ds[slices]`` (contiguous unit-step slices, one per
        dataset axis): chunks never written take the fill value at once,
        the others are added to *jobs* for ``run``.  A dataset that cannot be
        read natively is read through h5py here.  -> pixels read here."""
        layout = self.layout(key, ds, view)
        # uncompressed data, small reads too, is read here: since chunks are looked
        # up one by one and small planes batched, it is at par with h5py or faster
        # (measured 2026-10-18; until then reads under 1 MiB went through h5py)
        if layout.offset is not None:
            return self._plan_contiguous(layout, slices, dest, jobs)
        located = None
        if layout.prefer_native:
            # looked up now: chunk positions not trusted turn native off
            located = layout.locate(self._origins(layout, slices))
        if not layout.prefer_native:
            with self.lock:
                try:
                    data = ds[slices]
                except OSError as exc:
                    data = exc
            if isinstance(data, OSError):
                if layout.offset is not None:        # h5py failed; the bytes are there
                    return self._plan_contiguous(layout, slices, dest, jobs)
                return self._refused(key, layout, slices, dest, data)
            dest[...] = data.view(view) if view is not None else data
            return dest.size
        self._plan_chunks(key, layout, slices, dest, jobs, located)
        return 0

    @staticmethod
    def _origins(layout: Layout, slices: tuple) -> list:
        return list(itertools.product(*(range(s.start // c * c, s.stop, c)
                                        for s, c in zip(slices, layout.chunks))))

    def _refused(self, key, layout: Layout, slices: tuple, dest: np.ndarray,
                 exc: OSError) -> int:
        """h5py's read of *slices* failed.  The one failure that is not the
        data's: HDF5 1.14 refuses scaleoffset chunks stored at full precision
        (``scaleoffset_full_precision``), which HDF5 writes.  For a dataset
        decoded here those chunks are decoded here, and the read is accepted
        only when it holds one (every chunk's Fletcher-32 checksum, deflate's
        checksum and its size checked); any other failure raises
        CorruptFileError (UnsupportedFormatError for a filter missing).
        -> pixels read."""
        if FILTER_SCALEOFFSET not in layout.filters or not layout.native:
            raise self.filter_error(layout, exc, slices) from exc
        layout.where                # mapped now: may turn native off
        if not layout.native:
            raise self.filter_error(layout, exc, slices) from exc
        jobs: list = []
        self._plan_chunks(key, layout, slices, dest, jobs)
        full: list = []             # full-precision chunks seen
        read = _pool.run(lambda job=job: self._decode_into(*job, full=full) for job in jobs)
        if not full:
            raise self.filter_error(layout, exc, slices) from exc
        return read

    def _plan_chunks(self, key, layout: Layout, slices: tuple, dest: np.ndarray,
                     jobs: list, located: Optional[dict] = None) -> None:
        """The jobs decoding (or reading) the chunks *slices* covers; chunks
        never written take the fill value at once.  *located*: the chunks'
        positions (``Layout.locate``), when already looked up."""
        origins = self._origins(layout, slices)
        if located is None:
            located = layout.locate(origins)
        if not layout.native:       # never here: plan() and _refused() check
            raise MicroReaderError(f"{layout.where_text()}: chunks read natively, "
                                   "but their positions are not known")
        for origin in origins:
            lo = [max(s.start, o) for s, o in zip(slices, origin)]
            hi = [min(s.stop, o + c) for s, o, c in zip(slices, origin, layout.chunks)]
            dst = dest[tuple(slice(a - s.start, b - s.start) for a, b, s in zip(lo, hi, slices))]
            where = located.get(origin)
            if where is None:
                dst[...] = layout.fill
                continue
            part = tuple(slice(a - o, b - o) for a, b, o in zip(lo, hi, origin))
            if not layout.filters:
                # an uncompressed chunk (Luxendo, some BDV): the byte span from the
                # part's first to its last value in one read, unless the rows alone
                # are much fewer bytes than the span, counting each read they need
                item = layout.dtype.itemsize
                strides = [int(np.prod(layout.chunks[i + 1:])) for i in range(len(layout.chunks))]
                first = sum(p.start * s for p, s in zip(part, strides))
                last = sum((p.stop - 1) * s for p, s in zip(part, strides)) + 1
                span = (last - first) * item
                if span <= dst.nbytes + _row_reads(part, layout.chunks) * READ_CALL_BYTES:
                    jobs.append((None, layout, ("span", where[0] + first * item, span,
                                                tuple(s * item for s in strides), dst), None, None))
                else:
                    self._plan_contiguous(_ChunkSpan(layout, where[0]), part, dst, jobs)
                continue
            jobs.append(((key, origin), layout, where, dst, part))

    #: bytes one contiguous-read job covers at most (rows are batched into jobs)
    CONTIGUOUS_TASK = 4 << 20
    #: rows this close (bytes between them) are read as one span and copied out
    #: in one strided copy: one read of a little more beats a read per row
    CONTIGUOUS_GAP = 256 << 10

    def _plan_contiguous(self, layout: Layout, slices: tuple, dest: np.ndarray,
                         jobs: list) -> int:
        """Jobs reading ``slices`` of a contiguous dataset.  The trailing axes
        read whole merge with the last partial axis into one run of bytes per
        *row* (an index of the axis before it); each job is a batch of rows at
        a fixed stride, read in one span or row by row (``_read_rows``).  When
        one index of the axis before the rows needs only one small span, a job
        reads the spans of many such indices into one buffer and copies them
        out at once (``_read_planes``): a Python-level job per plane cost twice
        h5py on a 1 MB region of a Fiji HDF5 stack (50 planes, measured
        2026-10-18)."""
        shape, item = layout.shape, layout.dtype.itemsize
        if not shape:                                   # a scalar dataset
            jobs.append((None, layout, (layout.offset, 1, item, item, dest.reshape(1)), None, None))
            return 0
        k = len(shape)
        while k > 1 and (slices[k - 1].start, slices[k - 1].stop) == (0, shape[k - 1]):
            k -= 1
        strides = [int(np.prod(shape[i + 1:])) * item for i in range(len(shape))]   # bytes
        partial = k - 1                                 # the last axis not read whole
        if partial == 0:                                # rows = indices of axis 0, back to back
            rows_axis, run, skip = 0, strides[0], 0
        else:
            rows_axis = partial - 1
            run = (slices[partial].stop - slices[partial].start) * strides[partial]
            skip = slices[partial].start * strides[partial]
        stride = strides[rows_axis]
        per_job = max(1, self.CONTIGUOUS_TASK // max(
            stride if stride - run <= self.CONTIGUOUS_GAP else run, 1))
        rows = slices[rows_axis]
        n_rows = rows.stop - rows.start
        plane_span = (n_rows - 1) * stride + run
        if rows_axis >= 1 and n_rows <= per_job and stride - run <= self.CONTIGUOUS_GAP and \
                slices[rows_axis - 1].stop - slices[rows_axis - 1].start > 1:
            outer, planes = slices[:rows_axis - 1], slices[rows_axis - 1]
            step = strides[rows_axis - 1]
            per_group = max(1, self.CONTIGUOUS_TASK // max(plane_span, 1))
            for index in itertools.product(*(range(s.start, s.stop) for s in outer)):
                block = dest[tuple(i - s.start for i, s in zip(index, slices))] if index else dest
                base = layout.offset + sum(i * st for i, st in zip(index, strides)) + skip
                for p0 in range(planes.start, planes.stop, per_group):
                    p1 = min(planes.stop, p0 + per_group)
                    jobs.append((None, layout, ("planes", base + p0 * step + rows.start * stride,
                                                step, p1 - p0, n_rows, stride, run,
                                                block[p0 - planes.start:p1 - planes.start]),
                                 None, None))
            return 0
        for index in itertools.product(*(range(s.start, s.stop) for s in slices[:rows_axis])):
            block = dest[tuple(i - s.start for i, s in zip(index, slices))] if index else dest
            base = layout.offset + sum(i * st for i, st in zip(index, strides)) + skip
            for r0 in range(rows.start, rows.stop, per_job):
                r1 = min(rows.stop, r0 + per_job)
                jobs.append((None, layout, (base + r0 * stride, r1 - r0, stride, run,
                                            block[r0 - rows.start:r1 - rows.start]), None, None))
        return 0

    def _read_span(self, layout: Layout, spec: tuple) -> int:
        """One read of the bytes from a chunk part's first value to its last,
        then a strided copy of the part (the chunk's strides) into *out*."""
        _tag, offset, nbytes, strides, out = spec
        span = np.empty(nbytes, np.uint8)
        self._readinto(self.handles.get(layout.path).get(), offset, span, layout.path)
        stored = layout.view if layout.view is not None else layout.dtype
        values = np.lib.stride_tricks.as_strided(span.view(stored), out.shape, strides,
                                                 writeable=False)
        out[...] = values
        return nbytes // layout.dtype.itemsize

    def _read_planes(self, layout: Layout, spec: tuple) -> int:
        """*count* planes, *step* bytes apart, each *n* rows of *run* bytes
        *stride* apart: each plane's span read into one buffer, then one
        strided copy into *out* (count, n, ...).  -> pixels read."""
        _tag, offset, step, count, n, stride, run, out = spec
        span = (n - 1) * stride + run
        buf = np.empty((count, span), np.uint8)
        fh = self.handles.get(layout.path).get()
        for i in range(count):
            self._readinto(fh, offset + i * step, buf[i], layout.path)
        stored = layout.view if layout.view is not None else layout.dtype
        rows = np.lib.stride_tricks.as_strided(buf, (count, n, run), (span, stride, 1),
                                               writeable=False)
        out[...] = rows.view(stored).reshape(out.shape)
        return buf.size // layout.dtype.itemsize

    def _read_rows(self, layout: Layout, spec: tuple) -> int:
        """One job of ``_plan_contiguous``: *n* rows of *run* bytes, *stride*
        bytes apart, into the contiguous array *out*.  -> pixels read."""
        offset, n, stride, run, out = spec
        fh = self.handles.get(layout.path).get()
        stored = layout.view if layout.view is not None else layout.dtype
        item = layout.dtype.itemsize
        if not out.flags.c_contiguous:                  # a part of a chunk: via a buffer
            buffer = np.empty(out.shape, out.dtype)
            read = self._read_rows(layout, (offset, n, stride, run, buffer))
            out[...] = buffer
            return read
        if n == 1 or stride == run:                     # back to back: straight into out
            self._readinto(fh, offset, out, layout.path)
            if not stored.isnative:
                out.byteswap(inplace=True)
            return out.size
        if stride - run <= self.CONTIGUOUS_GAP:         # one span, one strided copy
            span = np.empty((n - 1) * stride + run, np.uint8)
            self._readinto(fh, offset, span, layout.path)
            rows = np.lib.stride_tricks.as_strided(span, (n, run), (stride, 1))
            out.reshape(n, -1)[...] = rows.view(stored)
            return span.size // item
        flat = out.reshape(n, -1)
        for i in range(n):                              # far apart: a read per row
            self._readinto(fh, offset + i * stride, flat[i], layout.path)
        if not stored.isnative:
            out.byteswap(inplace=True)
        return n * run // item

    @staticmethod
    def _readinto(fh, offset: int, out: np.ndarray, path: str) -> None:
        """Fill the contiguous array *out* from the file at *offset*."""
        buf = memoryview(out).cast("B")
        fh.seek(offset)
        done = 0
        while done < len(buf):
            n = fh.readinto(buf[done:])
            if not n:
                raise CorruptFileError(f"{path!r}: data at byte {offset} ends early "
                                       "(truncated file?)")
            done += n

    #: contiguous reads are batched into tasks of about this many bytes: the
    #: hand-off to the pool costs more than a small read (measured: a region
    #: read as 382 small jobs took 18 ms on 16 threads, 8 ms on one)
    BATCH_BYTES = 1 << 20

    def run(self, jobs: list) -> int:
        """Decode the planned chunks (cached ones are copied at once) on the
        decode pool.  -> pixels decoded."""
        return _pool.run(self._tasks(self._batched(jobs)))

    def _batched(self, jobs):
        """Contiguous-read jobs (key None) grouped into batches of about
        ``BATCH_BYTES``; chunk jobs as they are."""
        group, size = [], 0
        for job in jobs:
            if job[0] is not None:
                yield job
                continue
            where = job[2]
            if where[0] == "span":
                size += where[2]
            elif where[0] == "planes":
                size += where[3] * where[4] * where[6]
            else:
                size += where[1] * where[3]
            group.append((job[1], where))
            if size >= self.BATCH_BYTES:
                yield (None, None, ("group", group), None, None)
                group, size = [], 0
        if group:
            yield (None, None, ("group", group), None, None)

    def _read_group(self, group: list) -> int:
        readers = {"span": self._read_span, "planes": self._read_planes}
        return sum(readers.get(where[0], self._read_rows)(layout, where) for layout, where in group)

    def _tasks(self, jobs):
        for key, layout, where, dst, part in jobs:
            if key is None:                             # contiguous data: read, no cache
                yield lambda group=where[1]: self._read_group(group)
                continue
            chunk = self.cache.get(key)
            if chunk is not None:
                dst[...] = chunk[part]
                continue
            yield lambda job=(key, layout, where, dst, part): self._decode_into(*job)

    def _decode_into(self, key, layout, where, dst, part, full: Optional[list] = None) -> int:
        offset, size, mask = where
        with _memory.reserve(layout.decode_nbytes(size),
                             f"{layout.path!r}: an HDF5 chunk {layout.chunks}"):
            chunk = layout.decode(self.handles.get(layout.path).read(offset, size), mask,
                                  key[-1], full)
        dst[...] = chunk[part]
        self.cache.put(key, chunk)
        return chunk.size

    def filter_error(self, layout: Layout, exc: Exception, slices: tuple = ()) -> Exception:
        """h5py's read failed: name the HDF5 filters it could not apply."""
        where = layout.where_text()
        if slices:
            where += ", [" + ", ".join(f"{s.start}:{s.stop}" for s in slices) + "]"
        missing = [f for f in layout.filters if f not in SUPPORTED_FILTERS | H5PY_FILTERS]
        if not missing:
            names = ", ".join(FILTER_NAMES.get(f, str(f)) for f in layout.filters)
            return CorruptFileError(f"{where}: the data cannot be decoded (filters: "
                                    f"{names or 'none'}); h5py said: {exc}")
        names = ", ".join(f"{FILTER_NAMES.get(f, 'unknown')} ({f})" for f in missing)
        return UnsupportedFormatError(
            f"{where}: HDF5 filter {names} is not built into micro-reader; h5py "
            f"applies it with the hdf5plugin package (pip install hdf5plugin). h5py said: {exc}")

    def close(self) -> None:
        self.handles.close()
        self.cache.clear()
