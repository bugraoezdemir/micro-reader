"""Imaris backend, against h5py (cropped to the real image size)."""
from __future__ import annotations

import os
import threading

import numpy as np
import pytest

h5py = pytest.importorskip("h5py")

import micro_reader  # noqa: E402
from micro_reader import UnsupportedFormatError  # noqa: E402


def _char(value):
    return np.array(list(str(value)), dtype="S1")


def write_ims(path, real=(5, 21, 27), padded=(8, 32, 32), chunks=(4, 16, 16),
              n_t=1, n_c=2, dtype="u2", userblock=0, skip_chunks=False, **dset_kwargs):
    """An Imaris-shaped file: level 0 only, data padded to *padded* with the
    real size in ImageSize attributes.  Returns the real data (T, C, Z, Y, X)."""
    rng = np.random.default_rng(0)
    dtype = np.dtype(dtype)
    if dtype.kind == "f":
        data = rng.normal(size=(n_t, n_c, *real)).astype(dtype)
    else:
        data = rng.integers(1, np.iinfo(dtype).max, size=(n_t, n_c, *real)).astype(dtype)
    kwargs = {"userblock_size": userblock} if userblock else {}
    with h5py.File(path, "w", **kwargs) as f:
        for t in range(n_t):
            for c in range(n_c):
                grp = f.require_group(f"DataSet/ResolutionLevel 0/TimePoint {t}/Channel {c}")
                ds = grp.create_dataset("Data", shape=padded, dtype=dtype, chunks=chunks,
                                        fillvalue=0, **dset_kwargs)
                if skip_chunks:        # only the first chunk row is written; the rest never is
                    half = chunks[1]
                    ds[:real[0], :half, :real[2]] = data[t, c, :, :half]
                else:
                    ds[:real[0], :real[1], :real[2]] = data[t, c]
                for axis, size in zip("ZYX", real):
                    grp.attrs[f"ImageSize{axis}"] = _char(size)
        img = f.require_group("DataSetInfo/Image")
        for axis, size in zip("ZYX", real):
            img.attrs[axis] = _char(size)
    if skip_chunks:
        data[:, :, :, chunks[1]:] = 0
    return data


VARIANTS = {
    "gzip":            {"compression": "gzip", "compression_opts": 2},
    "gzip_shuffle":    {"compression": "gzip", "shuffle": True},
    "fletcher32":      {"compression": "gzip", "fletcher32": True},
    "uncompressed":    {},
    "float32":         {"compression": "gzip", "dtype": "f4"},
    "uint8":           {"compression": "gzip", "dtype": "u1"},
    "big_endian":      {"compression": "gzip", "shuffle": True, "dtype": ">u2"},
    "userblock":       {"compression": "gzip", "userblock": 512},
    "time_channels":   {"compression": "gzip", "n_t": 3, "n_c": 4},
    "odd_chunks":      {"compression": "gzip", "chunks": (3, 7, 10)},
    "unwritten_chunks": {"compression": "gzip", "skip_chunks": True},
    # not handled natively: must fall back to h5py and still be right
    "lzf_fallback":    {"compression": "lzf"},
    "contiguous_fallback": {"chunks": None},
}


def _lz4_frame(raw: bytes, block: int) -> bytes:
    """HDF5's LZ4 filter framing (H5Zlz4.c), as the plugin writes it: a
    block that does not compress is stored as is."""
    imagecodecs = pytest.importorskip("imagecodecs")
    out = [len(raw).to_bytes(8, "big"), block.to_bytes(4, "big")]
    for at in range(0, len(raw), block):
        part = raw[at:at + block]
        packed = bytes(imagecodecs.lz4_encode(part, header=False))
        if len(packed) >= len(part):
            packed = part
        out += [len(packed).to_bytes(4, "big"), packed]
    return b"".join(out)


def write_ims_lz4(path, shuffle=False, real=(5, 21, 27), padded=(8, 32, 32), chunks=(4, 16, 16),
                  block=1000):
    """An Imaris-shaped file whose chunks carry HDF5's LZ4 filter (32004), as
    ImarisWriter writes them.  No plugin here: chunks are framed by hand and
    written directly; h5py cannot read them back, micro-reader must."""
    rng = np.random.default_rng(1)
    data = rng.integers(0, 50, size=(1, 1, *real)).astype("u2")
    data[..., ::3] = rng.integers(0, 65535, size=data[..., ::3].shape)   # some incompressible
    full = np.zeros(padded, "u2")
    full[:real[0], :real[1], :real[2]] = data[0, 0]
    with h5py.File(path, "w") as f:
        grp = f.require_group("DataSet/ResolutionLevel 0/TimePoint 0/Channel 0")
        ds = grp.create_dataset("Data", shape=padded, dtype="u2", chunks=chunks,
                                compression=32004, compression_opts=(block,),
                                shuffle=shuffle, allow_unknown_filter=True)
        for z in range(0, padded[0], chunks[0]):
            for y in range(0, padded[1], chunks[1]):
                for x in range(0, padded[2], chunks[2]):
                    chunk = np.ascontiguousarray(full[z:z + chunks[0], y:y + chunks[1],
                                                      x:x + chunks[2]])
                    raw = chunk.tobytes()
                    if shuffle:                               # bytes grouped by significance
                        raw = chunk.view(np.uint8).reshape(-1, 2).T.tobytes()
                    ds.id.write_direct_chunk((z, y, x), _lz4_frame(raw, block))
        for axis, size in zip("ZYX", real):
            grp.attrs[f"ImageSize{axis}"] = _char(size)
    return data


@pytest.mark.parametrize("shuffle", [False, True], ids=["lz4", "shuffle_lz4"])
def test_lz4_chunks_decoded_natively(tmp_path, shuffle, decode_threads):
    """ImarisWriter's LZ4 (and shuffle + LZ4) chunks, without hdf5plugin."""
    path = str(tmp_path / "lz4.ims")
    ref = write_ims_lz4(path, shuffle=shuffle)
    with micro_reader.open(path) as img:
        scene = img.images[0]
        layout = scene._layout(0, 0)
        assert layout.native and 32004 in layout.filters
        for key in _keys(ref.shape):
            np.testing.assert_array_equal(scene[key], ref[key], err_msg=str(key))


def _keys(shape, n=25, seed=3):
    rng = np.random.default_rng(seed)
    keys = [(Ellipsis,), (0,), (-1, -1)]
    for _ in range(n):
        key = []
        for size in shape:
            a = int(rng.integers(0, size))
            b = int(rng.integers(a + 1, size + 1))
            choice = rng.integers(0, 4)
            key.append(a if choice == 0 else slice(a, b, 2) if choice == 1 else slice(a, b))
        keys.append(tuple(key))
    return keys


@pytest.fixture(params=[1, 4], ids=["serial", "pool"])
def decode_threads(request):
    old = micro_reader.decode_threads()
    micro_reader.set_decode_threads(request.param)
    yield
    micro_reader.set_decode_threads(old)


@pytest.mark.parametrize("name", list(VARIANTS))
def test_matches_h5py_cropped(tmp_path, name, decode_threads):
    kwargs = dict(VARIANTS[name])
    ref = write_ims(str(tmp_path / "a.ims"), **kwargs)
    with micro_reader.open(str(tmp_path / "a.ims")) as img:
        scene = img.images[0]
        assert img.backend == "ims" and scene.axes == "tczyx"
        assert scene.shape == ref.shape                     # real size, not padded
        assert scene.dtype == ref.dtype.newbyteorder("=")
        assert scene._layout(0, 0).native == (not name.endswith("_fallback"))
        if name == "unwritten_chunks":                     # a mix of written and missing
            layout = scene._layout(0, 0)
            assert (0, 0, 0) in layout.where and (0, 16, 0) not in layout.where
        for key in _keys(ref.shape):
            got = scene[key]
            assert got.dtype.isnative
            np.testing.assert_array_equal(got, ref[key], err_msg=f"{name} {key}")


def test_chunks_decoded_once_and_cached(tmp_path):
    write_ims(str(tmp_path / "a.ims"), compression="gzip")
    with micro_reader.open(str(tmp_path / "a.ims")) as img:
        scene = img.images[0]
        assert scene.read_unit == (1, 1, 4, 16, 16)
        scene[0, 0, 0:2, 0:5, 0:5]
        assert scene.stats.pixels_read == 4 * 16 * 16        # one chunk
        scene[0, 0, 1:3, 6:9, 6:9]
        assert scene.stats.pixels_read == 4 * 16 * 16        # same chunk: cached


def test_padding_never_read(tmp_path):
    """Reading the whole image touches only chunks inside the real size."""
    write_ims(str(tmp_path / "a.ims"), real=(4, 16, 16), padded=(8, 64, 64),
              chunks=(4, 16, 16), n_c=1, compression="gzip")
    with micro_reader.open(str(tmp_path / "a.ims")) as img:
        img.images[0][...]
        assert img.images[0].stats.pixels_read == 4 * 16 * 16


def test_not_imaris_is_refused(tmp_path):
    path = str(tmp_path / "plain.h5")
    with h5py.File(path, "w") as f:
        f.create_dataset("x", data=np.zeros((4, 4)))
    with pytest.raises(UnsupportedFormatError, match="Imaris"):
        micro_reader.open(path)


def test_corrupt_chunk_raises(tmp_path, decode_threads):
    path = str(tmp_path / "a.ims")
    write_ims(path, compression="gzip")
    with h5py.File(path, "r") as f:
        infos = []
        f["DataSet/ResolutionLevel 0/TimePoint 0/Channel 0/Data"].id.chunk_iter(infos.append)
        offset = infos[-1].byte_offset
    with open(path, "r+b") as fh:
        fh.seek(offset)
        fh.write(b"\x00" * 8)
    with micro_reader.open(path) as img:
        with pytest.raises(Exception):
            img.images[0][...]


def test_close_releases_the_file(tmp_path, decode_threads):
    path = str(tmp_path / "a.ims")
    write_ims(path, compression="gzip")
    img = micro_reader.open(path)
    img.images[0][...]
    img.close()
    os.remove(path)
    with pytest.raises(ValueError):
        img.images[0][0]


def test_concurrent_reads(tmp_path, decode_threads):
    path = str(tmp_path / "a.ims")
    ref = write_ims(path, compression="gzip", shuffle=True, n_t=2, n_c=3)
    keys = _keys(ref.shape, n=40, seed=9)
    errors = []
    with micro_reader.open(path) as img:
        scene = img.images[0]

        def worker(i):
            try:
                for key in keys[i::4]:
                    if not np.array_equal(scene[key], ref[key]):
                        errors.append(key)
            except Exception as exc:          # noqa: BLE001
                errors.append(exc)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
    assert errors == []


def _write_filtered(path, filter_id, encode, shape=(4, 16, 16)):
    """One chunk stored with an HDF5 filter h5py lacks (written directly)."""
    data = np.arange(int(np.prod(shape)), dtype="u2").reshape(shape)
    with h5py.File(path, "w") as f:
        grp = f.require_group("DataSet/ResolutionLevel 0/TimePoint 0/Channel 0")
        ds = grp.create_dataset("Data", shape=shape, dtype="u2", chunks=shape,
                                compression=filter_id, allow_unknown_filter=True)
        ds.id.write_direct_chunk((0,) * len(shape), encode(data.tobytes()))
        for axis, size in zip("ZYX", shape):
            grp.attrs[f"ImageSize{axis}"] = _char(size)
    return data


def test_zstd_filter_decoded_natively(tmp_path):
    zstandard = pytest.importorskip("zstandard")
    path = str(tmp_path / "zstd.ims")
    data = _write_filtered(path, 32015, zstandard.ZstdCompressor().compress)
    with micro_reader.open(path) as img:
        scene = img.images[0]
        assert scene._layout(0, 0).native
        np.testing.assert_array_equal(scene[0, 0], data)


def test_missing_hdf5_filter_is_named(tmp_path):
    """A non-Imaris filter h5py cannot apply: the error names it and the fix."""
    path = str(tmp_path / "blosc.ims")
    _write_filtered(path, 32001, lambda raw: raw)
    with micro_reader.open(path) as img:
        with pytest.raises(UnsupportedFormatError, match=r"Blosc \(32001\).*hdf5plugin"):
            img.images[0][...]
