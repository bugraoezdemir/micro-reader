"""Damaged and unusual HDF5 files in the shared chunk reader (backends/_hdf5.py):
typed errors instead of wrong pixels, zeros or untyped exceptions; checksums
checked; memory reserved for what decoding holds."""
from __future__ import annotations

import threading
import tracemalloc
import zlib

import numpy as np
import pytest

h5py = pytest.importorskip("h5py")

import micro_reader  # noqa: E402
from micro_reader import CorruptFileError, UnsupportedFormatError  # noqa: E402
from micro_reader.backends import _hdf5  # noqa: E402
from _hdf5_writer import axistags  # noqa: E402


@pytest.fixture(params=["native", "h5py"])
def chunk_path(request, monkeypatch):
    """Chunks decoded here (thresholds 0), or through h5py (the tests' small
    chunks, as real files' small chunks are: _hdf5.Layout.prefer_native)."""
    if request.param == "native":
        monkeypatch.setattr(_hdf5, "NATIVE_MIN_CHUNK", 0)
        monkeypatch.setattr(_hdf5, "NATIVE_MIN_CHUNK_SCALEOFFSET", 0)
        monkeypatch.setattr(_hdf5, "NATIVE_MIN_CHUNK_LZF", 0)
    return request.param


def _ilastik(path, data, keys="zyx", **options):
    with h5py.File(path, "w") as f:
        ds = f.create_dataset("x", data=data, **options)
        ds.attrs["axistags"] = axistags(keys)
        return ds.id.get_chunk_info(1) if ds.chunks and ds.id.get_num_chunks() > 1 else None


def _flip(path, offset):
    raw = bytearray(path.read_bytes())
    raw[offset] ^= 0xFF
    path.write_bytes(bytes(raw))


def _read(path):
    with micro_reader.open(path) as f:
        return f.images[0][...]


def _data(shape=(4, 64, 64), seed=0):
    return np.random.default_rng(seed).integers(0, 60000, shape).astype(np.uint16)


# -- Fletcher-32 ---------------------------------------------------------------

@pytest.mark.parametrize("options", [{}, {"compression": "gzip"},
                                     {"compression": "gzip", "shuffle": True}],
                         ids=["plain", "gzip", "gzip-shuffle"])
def test_fletcher32_good_chunks_read(tmp_path, chunk_path, options):
    data = _data()
    _ilastik(tmp_path / "f.h5", data, chunks=(1, 64, 64), fletcher32=True, **options)
    np.testing.assert_array_equal(_read(tmp_path / "f.h5"), data)


@pytest.mark.parametrize("options", [{}, {"compression": "gzip"}], ids=["plain", "gzip"])
def test_fletcher32_damaged_chunk_raises(tmp_path, chunk_path, options):
    """A flipped byte: h5py's checksum check fails; the read used to fall back
    to the native decoder, which did not check it, and returned the damage."""
    data = _data()
    info = _ilastik(tmp_path / "f.h5", data, chunks=(1, 64, 64), fletcher32=True, **options)
    _flip(tmp_path / "f.h5", info.byte_offset + info.size // 2)
    with pytest.raises(CorruptFileError, match="'/x'"):
        _read(tmp_path / "f.h5")
    with micro_reader.open(tmp_path / "f.h5") as f:          # other chunks still read
        np.testing.assert_array_equal(f.images[0][0], data[0])


def test_fletcher32_matches_hdf5(tmp_path):
    """The checksum HDF5 stores (last 4 bytes, little endian) for chunks of
    odd and even lengths, short and over 65535 words, compressed or not."""
    rng = np.random.default_rng(3)
    cases = [((7, 13), "u1", {}), ((5, 1001), "u1", {}), ((64, 64), "u2", {}),
             ((300, 300), "u2", {}), ((3, 333), "u1", {"compression": "gzip"}),
             ((257, 511), "u1", {}), ((400, 400), "f4", {"shuffle": True, "compression": "gzip"})]
    with h5py.File(tmp_path / "f.h5", "w") as f:
        for i, (shape, dtype, options) in enumerate(cases):
            f.create_dataset(f"d{i}", data=rng.integers(0, 250, shape).astype(dtype),
                             chunks=shape, fletcher32=True, **options)
        f.create_dataset("ff", data=np.full((131071,), 255, np.uint8), chunks=(131071,),
                         fletcher32=True)
        f.create_dataset("zeros", data=np.zeros((5,), np.uint8), chunks=(5,), fletcher32=True)
    checked = 0
    with h5py.File(tmp_path / "f.h5", "r") as f:
        for name in f:
            ds = f[name]
            _, raw = ds.id.read_direct_chunk(ds.id.get_chunk_info(0).chunk_offset)
            raw = bytes(raw)
            assert _hdf5.fletcher32(raw[:-4]) == int.from_bytes(raw[-4:], "little"), name
            assert bytes(_hdf5._fletcher_strip(raw, name)) == raw[:-4]
            damaged = bytearray(raw)
            damaged[0] ^= 1
            with pytest.raises(CorruptFileError, match="Fletcher-32"):
                _hdf5._fletcher_strip(bytes(damaged), name)
            checked += 1
    assert checked == len(cases) + 2


# -- deflate -------------------------------------------------------------------

def test_damaged_deflate_chunk_raises(tmp_path, chunk_path):
    """Was zlib.error from the native path (and wrong pixels after h5py's
    failure, through the fallback)."""
    data = _data()
    info = _ilastik(tmp_path / "f.h5", data, chunks=(1, 64, 64), compression="gzip")
    _flip(tmp_path / "f.h5", info.byte_offset + info.size // 2)
    with pytest.raises(CorruptFileError):
        _read(tmp_path / "f.h5")


def test_deflate_chunk_larger_than_the_chunk_raises(tmp_path, monkeypatch):
    """A deflate stream that inflates past the chunk is damage (and is not
    inflated whole first)."""
    monkeypatch.setattr(_hdf5, "NATIVE_MIN_CHUNK", 0)
    data = _data((2, 64, 64))
    _ilastik(tmp_path / "f.h5", data, chunks=(1, 64, 64), compression="gzip")
    with h5py.File(tmp_path / "f.h5", "r+") as f:
        f["x"].id.write_direct_chunk((1, 0, 0), zlib.compress(bytes(100 * data[1].nbytes)))
    with pytest.raises(CorruptFileError):
        _read(tmp_path / "f.h5")


def test_lz4_size_claim_is_bounded():
    """The LZ4 header's original size is checked against the chunk before
    anything of that size is allocated."""
    header = (1 << 60).to_bytes(8, "big") + (1 << 20).to_bytes(4, "big")
    with pytest.raises(CorruptFileError, match="claims"):
        _hdf5._lz4_unframe(header + bytes(16), "chunk", 4096)


# -- the h5py -> native fallback ------------------------------------------------

class _Refusing:
    """An h5py dataset whose reads fail as HDF5 1.14's do on full-precision
    scaleoffset chunks; everything else is the dataset's."""

    def __init__(self, ds):
        self._ds = ds

    def __getattr__(self, name):
        return getattr(self._ds, name)

    def __getitem__(self, key):
        raise OSError("Can't synchronously read data (filter returned failure during read)")


def _read_refused(path, slices):
    reader = _hdf5.ChunkReader(str(path))
    try:
        with h5py.File(path, "r") as f:
            ds = f["x"]
            dest = np.empty(tuple(s.stop - s.start for s in slices), ds.dtype)
            jobs = []
            read = reader.plan("x", _Refusing(ds), slices, dest, jobs)
            read += reader.run(jobs)
            return dest
    finally:
        reader.close()


def test_h5py_refusal_of_full_precision_chunks_read_here(tmp_path):
    """Values over the whole 16-bit range: HDF5 stores the chunk at full
    precision (HDF5 1.14 then refuses to read it, 2.0 reads it); a read h5py
    fails on is decoded here when it holds such a chunk."""
    data = _data((2, 64, 64))
    data[0, 0, 0], data[0, 0, 1] = 0, 65535
    with h5py.File(tmp_path / "f.h5", "w") as f:
        f.create_dataset("x", data=data, chunks=(1, 64, 64), scaleoffset=0)
    slices = (slice(0, 2), slice(0, 64), slice(0, 64))
    np.testing.assert_array_equal(_read_refused(tmp_path / "f.h5", slices), data)


def test_h5py_failure_without_full_precision_chunk_raises(tmp_path):
    """Any other h5py failure is the data's: no native decoding behind it."""
    data = np.random.default_rng(0).integers(100, 1100, (2, 64, 64)).astype(np.uint16)
    with h5py.File(tmp_path / "f.h5", "w") as f:
        f.create_dataset("x", data=data, chunks=(1, 64, 64), scaleoffset=0)
    slices = (slice(0, 2), slice(0, 64), slice(0, 64))
    with pytest.raises(CorruptFileError, match="h5py said"):
        _read_refused(tmp_path / "f.h5", slices)


def test_h5py_failure_on_deflate_raises(tmp_path):
    data = _data((2, 64, 64))
    with h5py.File(tmp_path / "f.h5", "w") as f:
        f.create_dataset("x", data=data, chunks=(1, 64, 64), compression="gzip")
    with pytest.raises(CorruptFileError, match="h5py said"):
        _read_refused(tmp_path / "f.h5", (slice(0, 2), slice(0, 64), slice(0, 64)))


# -- scaleoffset decoded here ----------------------------------------------------

@pytest.mark.parametrize("dtype,low,high", [
    ("u1", 3, 100), ("u2", 100, 1100), ("i2", -3000, 3000), (">u2", 0, 30000),
    ("u4", 0, 1 << 30), ("i8", -(1 << 40), 1 << 40), ("u8", 0, 1 << 59)])
@pytest.mark.parametrize("fill", [False, True])
def test_scaleoffset_native_matches_h5py(tmp_path, monkeypatch, dtype, low, high, fill):
    """Native bit unpacking (1-4 byte values), numpy's (8 bytes, also past 57
    bits), fill values, big-endian datasets.  Ranges below the full width
    (HDF5 1.14 reads those chunks too)."""
    monkeypatch.setattr(_hdf5, "NATIVE_MIN_CHUNK_SCALEOFFSET", 0)
    rng = np.random.default_rng(1)
    data = rng.integers(low, high, (3, 50, 70), dtype=np.dtype(dtype).newbyteorder("="))
    data = data.astype(dtype)
    with h5py.File(tmp_path / "f.h5", "w") as f:
        ds = f.create_dataset("x", data=data, chunks=(1, 32, 48), scaleoffset=0,
                              fillvalue=7 if fill else None)
        ds.attrs["axistags"] = axistags("zyx")
    with h5py.File(tmp_path / "f.h5", "r") as f:
        expected = f["x"][()]
    np.testing.assert_array_equal(_read(tmp_path / "f.h5"), expected)


# -- chunk positions -----------------------------------------------------------------

def test_unknown_userblock_shift_reads_through_h5py(tmp_path, monkeypatch, chunk_path):
    """When the chunk positions cannot be trusted, native reading turns off;
    the read under way used to give every chunk the fill value."""
    monkeypatch.setattr(_hdf5.Layout, "_userblock_shift", lambda self, *args: None)
    data = _data()
    _ilastik(tmp_path / "f.h5", data, chunks=(1, 64, 64), compression="gzip")
    np.testing.assert_array_equal(_read(tmp_path / "f.h5"), data)


def test_chunk_map_published_complete(tmp_path, monkeypatch):
    """The chunk map is published once complete (shifted): while the user-block
    shift is being found, other threads see no map rather than an unshifted one."""
    monkeypatch.setattr(_hdf5, "NATIVE_MIN_CHUNK", 0)
    seen = []
    original = _hdf5.Layout._userblock_shift

    def spy(self, *args):
        seen.append(self._where)
        return original(self, *args)
    monkeypatch.setattr(_hdf5.Layout, "_userblock_shift", spy)
    data = _data()
    _ilastik(tmp_path / "f.h5", data, chunks=(1, 64, 64), compression="gzip")
    results = []

    def read():
        results.append(_read(tmp_path / "f.h5"))
    threads = [threading.Thread(target=read) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert seen and all(where is None for where in seen)
    for out in results:
        np.testing.assert_array_equal(out, data)


# -- layout detection ------------------------------------------------------------------

def test_dataset_named_dataset_is_not_imaris(tmp_path):
    """A top-level ``DataSet`` that is a dataset: was AttributeError from
    classify, escaping sniffing."""
    with h5py.File(tmp_path / "f.h5", "w") as f:
        f["DataSet"] = np.zeros((3, 4))
    assert _hdf5.kind(str(tmp_path / "f.h5")) is None
    with pytest.raises(UnsupportedFormatError, match="no layout"):
        micro_reader.open(tmp_path / "f.h5")


def test_broken_external_link_is_not_a_layout(tmp_path):
    """A broken top-level external link named like a BigDataViewer setup:
    was KeyError from classify."""
    with h5py.File(tmp_path / "f.h5", "w") as f:
        f["s00"] = h5py.ExternalLink("missing.h5", "/x")
        f["t00000"] = np.zeros(2)
        f["Data"] = h5py.ExternalLink("missing.h5", "/y")
        f["metadata"] = h5py.ExternalLink("missing.h5", "/z")
    assert _hdf5.kind(str(tmp_path / "f.h5")) is None
    with pytest.raises(UnsupportedFormatError):
        micro_reader.open(tmp_path / "f.h5")


def test_broken_link_beside_ilastik_data(tmp_path):
    data = _data((2, 8, 8))
    _ilastik(tmp_path / "f.h5", data)
    with h5py.File(tmp_path / "f.h5", "r+") as f:
        f["s00"] = h5py.ExternalLink("missing.h5", "/x")
        f["DataSet"] = np.zeros(2)
    assert _hdf5.kind(str(tmp_path / "f.h5")) == "ilastik"
    np.testing.assert_array_equal(_read(tmp_path / "f.h5"), data)


def test_ilastik_lists_every_dataset_past_the_classify_cap(tmp_path, monkeypatch):
    """The cap on datasets looked at is for classifying; the listing of the
    images used to stop at it too (2100 datasets opened as 2000 images)."""
    monkeypatch.setattr(_hdf5, "MAX_VISITED", 20)
    with h5py.File(tmp_path / "f.h5", "w") as f:
        for i in range(25):
            f.create_dataset(f"d{i:03d}", data=np.full((2, 3), i, np.uint8)).attrs["axistags"] = \
                axistags("yx")
    with micro_reader.open(tmp_path / "f.h5") as f:
        assert len(f.images) == 25
        assert int(f.images[24][...][0, 0]) == 24
    with h5py.File(tmp_path / "f.h5", "r") as f:
        assert len(_hdf5.axistags_datasets(f)) == 25
        assert len(_hdf5.axistags_datasets(f, cap=20)) == 20


# -- memory -------------------------------------------------------------------------------

@pytest.mark.parametrize("options", [
    {"scaleoffset": 0, "compression": "gzip"},
    {"compression": "gzip", "shuffle": True, "fletcher32": True},
    {"compression": "lzf", "shuffle": True}], ids=["scaleoffset", "shuffle", "lzf"])
@pytest.mark.parametrize("dtype", ["u2", "u8"])
def test_reservation_covers_decoding(tmp_path, monkeypatch, options, dtype):
    """What a chunk's decoding allocates (tracemalloc's peak) stays within the
    memory it reserves; the reservation was the decoded chunk's size, while
    scaleoffset's unpacking holds several uint64 arrays of the chunk."""
    monkeypatch.setattr(_hdf5, "NATIVE_MIN_CHUNK", 0)
    monkeypatch.setattr(_hdf5, "NATIVE_MIN_CHUNK_SCALEOFFSET", 0)
    monkeypatch.setattr(_hdf5, "NATIVE_MIN_CHUNK_LZF", 0)
    data = np.random.default_rng(0).integers(0, 3000, (1, 256, 256)).astype(dtype)
    _ilastik(tmp_path / "f.h5", data, chunks=(1, 256, 256), **options)
    reserved = []
    real = _hdf5._memory.reserve

    class spy(real):
        def __init__(self, nbytes, what="a block"):
            reserved.append(int(nbytes))
            super().__init__(nbytes, what)
    monkeypatch.setattr(_hdf5._memory, "reserve", spy)
    with micro_reader.open(tmp_path / "f.h5") as f:
        image = f.images[0]
        tracemalloc.start()
        try:
            out = image[...]
            _, peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()
    np.testing.assert_array_equal(out, data)
    assert reserved
    # the read's own output (and its bookkeeping) is not the chunk's decoding
    assert peak - out.nbytes - (256 << 10) <= max(reserved)


def test_reference_deflate_is_bounded_by_the_output():
    """Without the native module, deflate inflates at most one byte past what
    the chunk holds: a damaged stream claiming 50 MB is refused, not inflated."""
    from micro_reader import _codecs
    stream = zlib.compress(b"\0" * (50 << 20))
    with pytest.raises(CorruptFileError, match="1025 bytes"):
        _codecs.zlib_decompress_into(stream, np.empty(1024, np.uint8), native=False)


def test_bdv_h5_alone_skips_a_setup_behind_a_broken_link(tmp_path):
    """A BigDataViewer .h5 read without its XML lists its setups from the file;
    a setup group behind a broken external link is none (it raised KeyError)."""
    import warnings
    from _hdf5_writer import write_bdv
    write_bdv(tmp_path, tiles=1, channels=1, timepoints=(0,))
    h5 = next(p for p in tmp_path.iterdir() if p.suffix == ".h5")
    with h5py.File(h5, "a") as f:
        f["s07"] = h5py.ExternalLink("gone.h5", "/s07")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with micro_reader.open(h5) as f:
            assert f.backend == "bdv" and len(f.images) == 1
