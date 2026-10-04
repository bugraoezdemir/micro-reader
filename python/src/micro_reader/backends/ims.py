"""Imaris (.ims): HDF5 with a fixed layout (``[ims]`` extra, h5py).

    /DataSet/ResolutionLevel r/TimePoint t/Channel c/Data      3D (Z, Y, X), chunked
    /DataSet/ResolutionLevel r/TimePoint t/Channel c  .attrs   ImageSizeZ/Y/X

Two things differ from reading the datasets through h5py:

- **Padding is cropped.**  Imaris pads every ``Data`` dataset to whole HDF5
  chunks (a 424 x 424 image stored as 512 x 512, zeros beyond); the real size
  is in the ``ImageSize`` attributes.  Images have the real size, as
  Bio-Formats and Imaris show it.
- **Chunks are read and decoded here, in parallel.**  h5py runs every HDF5
  call -- decompression included -- under one global lock, so threads do not
  speed it up.  h5py only tells us where each chunk is (``chunk_iter``); the
  chunks are then read with per-thread file handles and decompressed on the
  shared decode pool.  Filters handled: deflate (gzip), shuffle, Fletcher-32,
  LZ4 (ImarisWriter), zstd.  A dataset with any other filter (szip, ...) or no chunking is read
  through h5py instead: correct, just not parallel.

One image, full resolution, axes ``tczyx``; every ``ResolutionLevel r`` is in
``image.resolutions``, and an image of its own with ``resolutions="all"``
(index ``{"scene": 0, "resolution": r}``).
"""
from __future__ import annotations

import builtins
import os
import threading
import zlib

import numpy as np

from .. import _codecs, _memory, _pool
from .._base import File, Image
from .._errors import CorruptFileError, MicroReaderError, UnsupportedFormatError
from .._io import DecodeCache, ThreadHandles
from .._metadata import Channel, Metadata, Quantity, color_hex, quantity

NAME = "ims"
HDF5_MAGIC = b"\x89HDF\r\n\x1a\n"
DECODE_CACHE_BYTES = 64 << 20
FILTER_DEFLATE, FILTER_SHUFFLE, FILTER_FLETCHER32 = 1, 2, 3
FILTER_LZ4 = 32004                   # HDF Group registered filter; ImarisWriter writes it
FILTER_ZSTD = 32015                  # registered zstd filter: one zstd frame per chunk
SUPPORTED_FILTERS = {FILTER_DEFLATE, FILTER_SHUFFLE, FILTER_FLETCHER32, FILTER_LZ4, FILTER_ZSTD}
#: registered HDF5 filters h5py cannot apply without plugins (named in errors)
FILTER_NAMES = {4: "szip", 307: "bzip2", 32001: "Blosc", 32004: "LZ4", 32008: "bitshuffle",
                32013: "zfp", 32015: "zstd", 32017: "SZ", 32026: "Blosc2"}


def _lz4_unframe(data: bytes, path: str) -> bytes:
    """HDF5's LZ4 filter (H5Zlz4.c): big-endian original size (8 bytes) and
    block size (4), then per block its compressed size (4) and the LZ4
    block -- stored as is when the compressed size equals the block size."""
    total = int.from_bytes(data[:8], "big")
    block = min(int.from_bytes(data[8:12], "big"), total) or total
    out = np.empty(total, np.uint8)
    pos, done = 12, 0
    while done < total:
        size = min(block, total - done)
        stored = int.from_bytes(data[pos:pos + 4], "big")
        pos += 4
        if pos + stored > len(data):
            raise CorruptFileError(f"{path!r}: LZ4 chunk ends early")
        if stored == size:
            out[done:done + size] = np.frombuffer(data, np.uint8, size, pos)
        else:
            _codecs.lz4_block_decompress_into(memoryview(data)[pos:pos + stored],
                                              out[done:done + size])
        pos += stored
        done += size
    return out.tobytes()


def sniff(head: bytes, path: str) -> bool:
    # the HDF5 superblock sits at 0, or after a user block of 512 * 2**n bytes
    return head[:8] == HDF5_MAGIC or head[512:520] == HDF5_MAGIC


def _attr_str(attrs, key):
    """Imaris attributes are arrays of 1-byte strings."""
    if key not in attrs:
        return None
    value = attrs[key]
    if isinstance(value, np.ndarray):
        return b"".join(v if isinstance(v, bytes) else str(v).encode() for v in value).decode()
    return value.decode() if isinstance(value, bytes) else str(value)


def _image_size(ds) -> tuple:
    """Real (Z, Y, X) of a ``Data`` dataset: its Channel group's ImageSize
    attributes, else the dataset shape."""
    sizes = []
    for axis, stored in zip("ZYX", ds.shape):
        try:
            size = int(_attr_str(ds.parent.attrs, f"ImageSize{axis}"))
        except (TypeError, ValueError):
            size = stored
        sizes.append(size if 0 < size <= stored else stored)
    return tuple(sizes)


def _numbered(group, prefix) -> list:
    """Indices of ``'{prefix} {n}'`` members, which must be 0..n-1."""
    found = sorted(int(k[len(prefix) + 1:]) for k in group
                   if k.startswith(prefix + " ") and k[len(prefix) + 1:].isdigit())
    if found != list(range(len(found))) or not found:
        raise UnsupportedFormatError(f"Imaris file with {prefix} groups {found}")
    return found


class _Layout:
    """How one ``Data`` dataset is stored: where each chunk is and how to
    decode it; or ``native = False`` to read it through h5py."""

    def __init__(self, ds):
        self.path = ds.file.filename
        self.dtype = ds.dtype                       # file byte order
        self.fill = ds.fillvalue
        self.chunks = ds.chunks
        dcpl = ds.id.get_create_plist()
        self.filters = [dcpl.get_filter(i)[0] for i in range(dcpl.get_nfilters())]
        self.native = bool(self.chunks) and set(self.filters) <= SUPPORTED_FILTERS
        self.chunk_nbytes = int(np.prod(self.chunks)) * self.dtype.itemsize if self.chunks else 0
        self.where: dict = {}                       # chunk origin -> (byte offset, size, filter mask)
        if self.native:
            ds.id.chunk_iter(lambda info: self.where.__setitem__(
                tuple(info.chunk_offset), (info.byte_offset, info.size, info.filter_mask)))
            shift = self._userblock_shift(ds)
            if shift is None:
                self.native, self.where = False, {}  # positions unknown: read through h5py
            elif shift:
                self.where = {k: (o + shift, n, m) for k, (o, n, m) in self.where.items()}

    def _userblock_shift(self, ds):
        """Bytes to add to the chunk offsets HDF5 reports.  Before HDF5
        1.14.6 (seen with 1.14.2) they leave out a user block at the start of
        the file; later versions include it.  One chunk's stored bytes
        (read_direct_chunk) are compared with the file at both positions;
        None when neither matches."""
        userblock = ds.file.id.get_create_plist().get_userblock()
        if not userblock or not self.where:
            return 0
        origin, (offset, size, _) = next(iter(self.where.items()))
        _, stored = ds.id.read_direct_chunk(origin)
        stored = bytes(stored)
        with builtins.open(self.path, "rb") as fh:
            for shift in (0, userblock):
                fh.seek(offset + shift)
                if fh.read(len(stored)) == stored:
                    return shift
        return None

    def decode(self, data: bytes, mask: int) -> np.ndarray:
        """Undo the filter pipeline (in reverse; bit i of *mask* set means
        filter i was skipped for this chunk)."""
        for i in reversed(range(len(self.filters))):
            if mask & (1 << i):
                continue
            fid = self.filters[i]
            if fid == FILTER_DEFLATE:
                data = zlib.decompress(data, 15, self.chunk_nbytes)
            elif fid == FILTER_LZ4:
                data = _lz4_unframe(data, self.path)
            elif fid == FILTER_ZSTD:
                out = np.empty(self.chunk_nbytes, np.uint8)
                _codecs.zstd_decompress_into(data, out)
                data = out.tobytes()
            elif fid == FILTER_SHUFFLE:
                item = self.dtype.itemsize
                if item > 1:
                    b = np.frombuffer(data, np.uint8)
                    n = len(b) // item
                    data = b[:n * item].reshape(item, n).T.tobytes() + bytes(b[n * item:])
            elif fid == FILTER_FLETCHER32:
                data = data[:-4]                    # checksum; not verified
        if len(data) != self.chunk_nbytes:
            raise CorruptFileError(f"chunk decodes to {len(data)} bytes, "
                                   f"expected {self.chunk_nbytes}")
        return np.frombuffer(data, self.dtype).reshape(self.chunks)


class _ImsImage(Image):
    """One resolution level: all time points and channels, axes 'tczyx'."""

    def __init__(self, h5, path: str, level: int = 0, key: dict = None):
        base = f"DataSet/ResolutionLevel {level}"
        if base not in h5:
            raise UnsupportedFormatError(f"{path!r}: HDF5 file without Imaris {base!r}")
        group = h5[base]
        times = _numbered(group, "TimePoint")
        channels = _numbered(group[f"TimePoint {times[0]}"], "Channel")
        try:
            self._datasets = [[group[f"TimePoint {t}/Channel {c}/Data"] for c in channels]
                              for t in times]
        except KeyError as exc:
            raise UnsupportedFormatError(f"{path!r}: incomplete Imaris data ({exc})") from exc
        all_ds = [ds for row in self._datasets for ds in row]
        sizes = {_image_size(ds) for ds in all_ds}
        dtypes = {ds.dtype.newbyteorder("=") for ds in all_ds}
        if len(sizes) != 1 or len(dtypes) != 1:
            raise UnsupportedFormatError(
                f"{path!r}: channels / time points differ in size {sizes} or type {dtypes}")
        self._path = path
        self._h5 = h5
        self._channel_ids = channels
        self._lock = threading.Lock()               # h5py objects: one caller at a time
        self._layouts: dict = {}
        self._handles = ThreadHandles(path)
        #: decoded chunks kept for reuse (``decode_cache.max_bytes`` to resize)
        self.decode_cache = DecodeCache(DECODE_CACHE_BYTES)
        self._closed = False
        chunks = all_ds[0].chunks
        super().__init__(level, "tczyx", (len(times), len(channels), *sizes.pop()),
                         dtypes.pop(), name=os.path.basename(path),
                         read_unit=(1, 1, *chunks) if chunks else None,
                         index=key if key is not None else {"scene": 0})

    def _layout(self, t: int, c: int) -> _Layout:
        layout = self._layouts.get((t, c))
        if layout is None:
            with self._lock:
                layout = self._layouts.get((t, c))
                if layout is None:
                    layout = self._layouts[(t, c)] = _Layout(self._datasets[t][c])
        return layout

    def _read_block(self, block):
        if self._closed:
            raise ValueError("this Imaris image has been closed: its file was closed "
                             "(File.close() or the end of a with block)")
        ts, cs, zs, ys, xs = block
        out = np.empty(tuple(s.stop - s.start for s in block), dtype=self.dtype)
        read, jobs = 0, []
        for t in range(ts.start, ts.stop):
            for c in range(cs.start, cs.stop):
                dest = out[t - ts.start, c - cs.start]
                layout = self._layout(t, c)
                if not layout.native:               # other filters / not chunked: h5py
                    with self._lock:
                        try:
                            dest[...] = self._datasets[t][c][zs, ys, xs]
                        except OSError as exc:
                            raise self._filter_error(layout, exc) from exc
                    read += dest.size
                    continue
                self._plan(t, c, layout, zs, ys, xs, dest, jobs)
        return out, read + _pool.run(self._tasks(jobs))

    def _filter_error(self, layout, exc: Exception) -> Exception:
        """h5py's read failed: name the HDF5 filters it could not apply."""
        missing = [f for f in layout.filters if f not in SUPPORTED_FILTERS and f != 32000]
        if not missing:
            return CorruptFileError(f"{self._path!r}: {exc}")
        names = ", ".join(f"{FILTER_NAMES.get(f, 'unknown')} ({f})" for f in missing)
        return UnsupportedFormatError(
            f"{self._path!r}: HDF5 filter {names} is not an Imaris one; h5py applies it "
            f"with the hdf5plugin package (pip install hdf5plugin). h5py said: {exc}")

    def _plan(self, t, c, layout, zs, ys, xs, dest, jobs) -> None:
        """Every chunk the rectangle touches: missing ones (never written)
        take the fill value here, the others become jobs."""
        cz, cy, cx = layout.chunks
        for oz in range(zs.start // cz * cz, zs.stop, cz):
            for oy in range(ys.start // cy * cy, ys.stop, cy):
                for ox in range(xs.start // cx * cx, xs.stop, cx):
                    z0, z1 = max(zs.start, oz), min(zs.stop, oz + cz)
                    y0, y1 = max(ys.start, oy), min(ys.stop, oy + cy)
                    x0, x1 = max(xs.start, ox), min(xs.stop, ox + cx)
                    dst = dest[z0 - zs.start:z1 - zs.start, y0 - ys.start:y1 - ys.start,
                               x0 - xs.start:x1 - xs.start]
                    part = (slice(z0 - oz, z1 - oz), slice(y0 - oy, y1 - oy),
                            slice(x0 - ox, x1 - ox))
                    where = layout.where.get((oz, oy, ox))
                    if where is None:
                        dst[...] = layout.fill
                    else:
                        jobs.append(((t, c, oz, oy, ox), layout, where, dst, part))

    def _tasks(self, jobs):
        for key, layout, where, dst, part in jobs:
            chunk = self.decode_cache.get(key)
            if chunk is not None:
                dst[...] = chunk[part]
                continue
            yield lambda job=(key, layout, where, dst, part): self._decode_into(*job)

    def _decode_into(self, key, layout, where, dst, part) -> int:
        """Read, decode and copy one chunk (on a pool thread)."""
        offset, size, mask = where
        with _memory.reserve(layout.chunk_nbytes, f"{self._path!r}: an HDF5 chunk {layout.chunks}"):
            chunk = layout.decode(self._handles.read(offset, size), mask)
        dst[...] = chunk[part]
        self.decode_cache.put(key, chunk)
        return chunk.size

    def close(self) -> None:
        self._closed = True
        self._handles.close()
        self.decode_cache.clear()


    def _read_metadata(self):
        with self._lock:
            return _metadata(self._h5, self.shape, self._channel_ids)


def _metadata(h5, shape: tuple, channel_ids: list):
    """``DataSetInfo``: the volume's extent (``ExtMin`` / ``ExtMax`` 0-2 = x,
    y, z, edges of the outer pixels, in ``Unit``) over the image size gives
    the pixel sizes and the position of the first pixel; ``TimeInfo`` the
    time points; ``Channel c`` name, colour (``r g b`` in 0-1) and
    wavelengths (nm)."""
    if "DataSetInfo" not in h5:
        return None
    info = h5["DataSetInfo"]
    image = info["Image"].attrs if "Image" in info else {}
    unit = _attr_str(image, "Unit") or "um"
    sizes, position = {}, {}
    nz, ny, nx = shape[2:]
    for k, (axis, n) in enumerate((("x", nx), ("y", ny), ("z", nz))):
        lo, hi = (quantity(_attr_str(image, f"Ext{e}{k}"), unit, positive=False)
                  for e in ("Min", "Max"))
        if lo is None or hi is None:
            continue
        size = quantity((hi.value - lo.value) / n, lo.unit)
        if size:
            sizes[axis] = size
            if axis != "z":
                position[axis] = Quantity(lo.value + size.value / 2, lo.unit)
    times = _time_points(info["TimeInfo"].attrs) if "TimeInfo" in info else []
    if len(times) > 1 and shape[0] > 1:
        step = quantity((times[-1] - times[0]).total_seconds() / (len(times) - 1), "second")
        if step:
            sizes["t"] = step
    channels = []
    for c in channel_ids:
        attrs = info[f"Channel {c}"].attrs if f"Channel {c}" in info else {}
        name = _attr_str(attrs, "Name")
        color = None
        rgb = (_attr_str(attrs, "Color") or "").split()
        if len(rgb) >= 3:
            try:
                color = color_hex(tuple(float(v) for v in rgb[:3]))
            except ValueError:
                pass
        channels.append(Channel(
            name=None if not name or name == "(name not specified)" else name,
            color=color,
            emission_wavelength=quantity(_attr_str(attrs, "LSMEmissionWavelength"), "nanometer"),
            excitation_wavelength=quantity(_attr_str(attrs, "LSMExcitationWavelength"),
                                           "nanometer")))
    return Metadata(pixel_sizes=sizes, position=position, channels=tuple(channels))


def _time_points(attrs) -> list:
    """``TimePoint1`` .. ``TimePointN`` ('2020-02-29 00:07:15.969') as datetimes."""
    import datetime
    out = []
    for n in range(1, len(attrs) + 1):
        text = _attr_str(attrs, f"TimePoint{n}")
        if text is None:
            break
        try:
            out.append(datetime.datetime.fromisoformat(text.strip()))
        except ValueError:
            return []
    return out


class _ImsFile(File):
    backend = NAME

    def __init__(self, path, images, h5):
        super().__init__(path, images)
        self._h5 = h5

    def close(self) -> None:
        super().close()
        self._h5.close()


def open(path: str, tiles: str = "separate", resolutions: str = "base") -> _ImsFile:  # noqa: A001 - backend protocol name
    try:
        import h5py
    except ImportError as exc:
        raise MicroReaderError(
            "reading Imaris needs the [ims] extra: pip install 'micro-reader[ims]'") from exc
    try:
        h5 = h5py.File(path, "r")
    except OSError as exc:
        raise CorruptFileError(f"{path!r}: {exc}") from exc
    try:
        levels = sorted(int(name.rsplit(" ", 1)[1]) for name in h5.get("DataSet", {})
                        if name.startswith("ResolutionLevel "))
        if not levels:
            raise UnsupportedFormatError(f"{path!r}: HDF5 file without Imaris resolution levels")
        listed = resolutions == "all"
        pyramid = [_ImsImage(h5, path, r, key={"scene": 0, "resolution": r}
                            if (listed or r) else {"scene": 0}) for r in levels]
        for image in pyramid:
            image._resolutions = pyramid
        images = pyramid if listed else pyramid[:1]
        for n, image in enumerate(images):
            image._position = n
    except BaseException:
        h5.close()
        raise
    return _ImsFile(path, images, h5)
