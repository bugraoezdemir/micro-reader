"""Imaris backend, against h5py (cropped to the real image size)."""
from __future__ import annotations

import os
import threading
import warnings

import numpy as np
import pytest

h5py = pytest.importorskip("h5py")

import micro_reader  # noqa: E402
from micro_reader import CorruptFileError, MissingImagesWarning, UnsupportedFormatError  # noqa: E402
from _hdf5_writer import write_ims_luxendo, write_ims_multifile  # noqa: E402


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
    "lzf":             {"compression": "lzf"},
    # not handled natively: must fall back to h5py and still be right
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
        series = img.images[0]
        layout = series._layout(0, 0)
        assert layout.native and 32004 in layout.filters
        for key in _keys(ref.shape):
            np.testing.assert_array_equal(series[key], ref[key], err_msg=str(key))


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


@pytest.fixture(autouse=True)
def native_chunks(monkeypatch):
    """Decode even the tests' tiny chunks here (real files' small chunks go
    through h5py: _hdf5.Layout.prefer_native)."""
    from micro_reader.backends import _hdf5
    monkeypatch.setattr(_hdf5, "NATIVE_MIN_CHUNK", 0)
    monkeypatch.setattr(_hdf5, "NATIVE_MIN_CHUNK_SCALEOFFSET", 0)


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
        series = img.images[0]
        assert img.backend == "ims" and series.axes == "tczyx"
        assert series.shape == ref.shape                     # real size, not padded
        assert series.dtype == ref.dtype.newbyteorder("=")
        assert series._layout(0, 0).native == (not name.endswith("_fallback"))
        if name == "unwritten_chunks":                     # a mix of written and missing
            layout = series._layout(0, 0)
            assert (0, 0, 0) in layout.where and (0, 16, 0) not in layout.where
        for key in _keys(ref.shape):
            got = series[key]
            assert got.dtype.isnative
            np.testing.assert_array_equal(got, ref[key], err_msg=f"{name} {key}")


def test_chunks_decoded_once_and_cached(tmp_path):
    write_ims(str(tmp_path / "a.ims"), compression="gzip")
    with micro_reader.open(str(tmp_path / "a.ims")) as img:
        series = img.images[0]
        assert series.read_unit == (1, 1, 4, 16, 16)
        series[0, 0, 0:2, 0:5, 0:5]
        assert series.stats.pixels_read == 4 * 16 * 16        # one chunk
        series[0, 0, 1:3, 6:9, 6:9]
        assert series.stats.pixels_read == 4 * 16 * 16        # same chunk: cached


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
        series = img.images[0]

        def worker(i):
            try:
                for key in keys[i::4]:
                    if not np.array_equal(series[key], ref[key]):
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
        series = img.images[0]
        assert series._layout(0, 0).native
        np.testing.assert_array_equal(series[0, 0], data)


def test_missing_hdf5_filter_is_named(tmp_path):
    """A non-Imaris filter h5py cannot apply: the error names it and the fix."""
    path = str(tmp_path / "blosc.ims")
    _write_filtered(path, 32001, lambda raw: raw)
    with micro_reader.open(path) as img:
        with pytest.raises(UnsupportedFormatError, match=r"Blosc \(32001\).*hdf5plugin"):
            img.images[0][...]


# -- multi-file Imaris sets (external links) ------------------------------------

def test_multifile_set_reads_its_parts(tmp_path, decode_threads):
    """FileConverter's multi-file Imaris (S-BIAD13 3x3Bin.ims): every
    TimePoint / Channel group is an external link into a part file
    ``<name>--Cxx--Txxxxx.h5``; the master reads as one image, all levels."""
    data = write_ims_multifile(tmp_path)
    with warnings.catch_warnings():
        warnings.simplefilter("error", MissingImagesWarning)       # nothing is missing
        f = micro_reader.open(tmp_path / "set.ims")
    with f:
        image = f.images[0]
        assert (f.backend, image.axes, image.shape) == ("ims", "tczyx", data.shape)
        for key in _keys(data.shape):
            np.testing.assert_array_equal(image[key], data[key], err_msg=str(key))
        level = image.resolutions[1]
        assert level.shape == (2, 2, 3, 10, 12)
        np.testing.assert_array_equal(level[...], data[..., ::2, ::2])
        assert [ch.name for ch in image.metadata.channels] == ["ch0", "ch1"]


def test_multifile_missing_parts_read_as_zeros(tmp_path, decode_threads):
    """A multi-file Imaris set copied without some of its part files (an
    incomplete download): the missing time point / channel parts read as
    zeros, with a MissingImagesWarning naming one, instead of refusing it."""
    data = write_ims_multifile(tmp_path, skip=[(1, 1), (0, 1)])
    with pytest.warns(MissingImagesWarning,
                      match=r"2 of 4 time point / channel parts are missing.*set--C01--T0000"):
        f = micro_reader.open(tmp_path / "set.ims")
    with f:
        image = f.images[0]
        assert image.shape == data.shape
        assert not image.read(t=1, c=1).any() and not image.read(t=0, c=1).any()
        for key in _keys(data.shape):
            np.testing.assert_array_equal(image[key], data[key], err_msg=str(key))
        assert not image.resolutions[1].read(t=0, c=1).any()        # other levels too
        np.testing.assert_array_equal(image.resolutions[1].read(t=1, c=0),
                                      data[1, 0, :, ::2, ::2])


def test_multifile_all_parts_missing_is_refused(tmp_path):
    """Only the master of a multi-file set: nothing to read, the error names a part."""
    write_ims_multifile(tmp_path, skip=[(t, c) for t in range(2) for c in range(2)])
    with pytest.raises(CorruptFileError, match=r"none of its Imaris data.*set--C00--T00000\.h5"):
        micro_reader.open(tmp_path / "set.ims")


def test_luxendo_link_only_ims(tmp_path, decode_threads):
    """Luxendo's ``imaris.ims`` (S-BIAD3686): real Channel groups whose Data is
    an external link to a raw ``.lux.h5`` stack; a missing stack reads as
    zeros, the warning names its file."""
    data = write_ims_luxendo(tmp_path)
    with micro_reader.open(tmp_path / "imaris.ims") as f:
        image = f.images[0]
        assert (f.backend, image.shape) == ("ims", data.shape)
        for key in _keys(data.shape):
            np.testing.assert_array_equal(image[key], data[key], err_msg=str(key))
    lost = tmp_path / "lost"
    lost.mkdir()
    data = write_ims_luxendo(lost, skip=[(1, 0)])
    with pytest.warns(MissingImagesWarning,
                      match=r"1 of 4 .*stack_0-Zstack_channel_0/Cam_long_00001\.lux\.h5"):
        f = micro_reader.open(lost / "imaris.ims")
    with f:
        image = f.images[0]
        assert not image.read(t=1, c=0).any()
        np.testing.assert_array_equal(image[...], data)


def test_multifile_part_opens_its_master(tmp_path):
    """Opening a part file ``<name>--C00--T00000.h5`` opens the whole set:
    ``<name>.ims``, else the ``<name>.h5`` copy FileConverter also writes;
    with neither, the error names the master it expects."""
    data = write_ims_multifile(tmp_path)
    part = tmp_path / "set--C01--T00000.h5"
    with micro_reader.open(part) as f:
        assert f.path.endswith("set.ims") and f.opened.endswith("set--C01--T00000.h5")
        np.testing.assert_array_equal(f.images[0][...], data)
    os.remove(tmp_path / "set.ims")
    with micro_reader.open(part) as f:
        assert f.path.endswith("set.h5")
        np.testing.assert_array_equal(f.images[0][...], data)
    os.remove(tmp_path / "set.h5")
    with pytest.raises(CorruptFileError, match=r"part of .*set\.ims"):
        micro_reader.open(part)


def test_multifile_may_own_reads_nothing(tmp_path, monkeypatch):
    """group_files' cheap test is a name rule over the folder listing: no file
    is opened.  A part name counts only with its master beside it."""
    import builtins
    from micro_reader import _groups, _parts
    from micro_reader.backends import ims
    write_ims_multifile(tmp_path)
    (tmp_path / "other--C00--T00000.h5").write_bytes(b"")        # no other.ims / other.h5

    def refuse(*args, **kwargs):
        raise AssertionError(f"may_own read a file: {args}")
    monkeypatch.setattr(builtins, "open", refuse)
    monkeypatch.setattr(_parts, "read_head", refuse)
    folders = _groups._Folders()
    folder = str(tmp_path)
    owned = {n: ims.may_own(folder, n, folders) for n in sorted(os.listdir(tmp_path))}
    assert owned == {"other--C00--T00000.h5": False, "set--C00--T00000.h5": True,
                     "set--C00--T00001.h5": True, "set--C01--T00000.h5": True,
                     "set--C01--T00001.h5": True, "set.h5": False, "set.ims": False}
    assert ims.may_own(folder, "SET--c00--t00000.H5", folders)              # any letter case


def test_multifile_group_files(tmp_path, monkeypatch):
    """group_files puts the part files in one group with the master; only the
    parts (the files a name rule matched) have their first bytes read."""
    from micro_reader import _parts
    write_ims_multifile(tmp_path)
    reads = []
    real = _parts.read_head
    monkeypatch.setattr(_parts, "read_head", lambda p: reads.append(os.path.basename(p)) or real(p))
    g = micro_reader.group_files(str(tmp_path))
    (group,) = g.groups
    parts = sorted(n for n in os.listdir(tmp_path) if "--C" in n)
    assert (group.kind, group.format) == ("composite", "ims")
    assert os.path.samefile(group.main, tmp_path / "set.ims")
    assert [os.path.basename(p) for p in group.parts] == parts
    assert [os.path.basename(p) for p in g.singles] == ["set.h5"]           # the copy
    assert sorted(reads) == parts


# -- coordinate systems (plan 4.7, check 4.14) ----------------------------------------

@pytest.fixture
def metadata_reads(monkeypatch):
    """How often an Imaris image's ``_read_metadata`` ran."""
    from micro_reader.backends import ims
    count = [0]
    real = ims._ImsImage._read_metadata

    def counting(self):
        count[0] += 1
        return real(self)
    monkeypatch.setattr(ims._ImsImage, "_read_metadata", counting)
    return count


def _set_extent(path, extent, unit="um"):
    """``DataSetInfo/Image`` ``ExtMin`` / ``ExtMax`` k: *extent* maps k (0-2 =
    x, y, z) to (min, max); the ones not given are removed."""
    with h5py.File(path, "a") as f:
        img = f.require_group("DataSetInfo/Image")
        for k in range(3):
            for e in ("Min", "Max"):
                if f"Ext{e}{k}" in img.attrs:
                    del img.attrs[f"Ext{e}{k}"]
        for k, (lo, hi) in extent.items():
            img.attrs[f"ExtMin{k}"], img.attrs[f"ExtMax{k}"] = _char(lo), _char(hi)
        img.attrs["Unit"] = _char(unit)


@pytest.mark.parametrize("resolutions, axes", [("base", None), ("all", None), ("base", "canonical")])
def test_position_is_in_the_stage_system(tmp_path, resolutions, axes, metadata_reads):
    """One ``stage`` per file; the position (ExtMin + half a pixel, x and y)
    names it, at every resolution level (level 0's position, plan 8.22)."""
    from micro_reader._spatial import check_invariants
    write_ims_multifile(tmp_path)                       # 2 levels; 24 x 20 pixels of 0.5 um
    _set_extent(tmp_path / "set.ims", {0: (100.0, 112.0), 1: (-5.0, 5.0), 2: (0.0, 6.0)})
    with micro_reader.open(tmp_path / "set.ims", resolutions=resolutions, axes=axes) as f:
        assert [s.id for s in f.coordinate_systems] == ["stage"]
        assert f.coordinate_systems[0].axes is None and f.coordinate_transformations == ()
        assert metadata_reads[0] == 0                   # declared without metadata
        check_invariants(f)                             # I1-I3, every level too
        levels = f.images[0].resolutions
        assert len(levels) == 2 and len(f.images) == (2 if resolutions == "all" else 1)
        for level in levels:
            meta = level.metadata
            assert meta.coordinate_system == "stage" and set(meta.position) == {"x", "y"}
            assert meta.position["x"].value == pytest.approx(100.25)
            assert meta.position["y"].value == pytest.approx(-4.75)
        assert metadata_reads[0] == 1                   # level 0's, scaled for level 1


@pytest.mark.parametrize("extent, unit", [(None, "um"), ({2: (0.0, 6.0)}, "um"),
                                          ({0: (0.0, 12.0), 1: (0.0, 10.0)}, "furlong"),
                                          ({0: (5.0, 5.0), 1: (3.0, 1.0)}, "um")],
                         ids=["no_extent", "z_only", "unknown_unit", "empty_extent"])
def test_files_without_a_position_declare_no_system(tmp_path, extent, unit, metadata_reads):
    """No position from the extent: no system declared, none named."""
    from micro_reader._spatial import check_invariants
    path = tmp_path / "a.ims"
    write_ims(path)                                     # DataSetInfo without an extent
    if extent is not None:
        _set_extent(path, extent, unit)
    with micro_reader.open(path) as f:
        assert f.coordinate_systems == ()
        assert metadata_reads[0] == 0
        check_invariants(f)
        meta = f.images[0].metadata
        assert meta.position == {} and meta.coordinate_system is None


def test_coordinate_systems_after_close_raise(tmp_path):
    """Declared on first access, which needs the open file: after closing, a
    ValueError says to read them before."""
    from micro_reader.backends import ims
    path = tmp_path / "a.ims"
    write_ims(path)
    f = ims.open(str(path))                             # not the session's open, which reads them
    f.close()
    with pytest.raises(ValueError, match="read coordinate_systems before closing"):
        f.coordinate_systems


def test_ims_is_checked_strictly_by_the_session():
    import conftest
    assert "ims" not in conftest.POSITIONS_WITHOUT_A_SYSTEM
