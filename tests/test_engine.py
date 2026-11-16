"""The native read engine (micro-reader-core, Workplan MR23) against the Python
path, bit for bit, and its own checks: plans it refuses, truncated files, its
cache and file handles.

Skips where micro_reader_core is not installed (CI installs it in the test jobs).
"""
from __future__ import annotations

import numpy as np
import pytest

core = pytest.importorskip("micro_reader_core")
tifffile = pytest.importorskip("tifffile")

import micro_reader  # noqa: E402
from micro_reader import _engine  # noqa: E402
from micro_reader._errors import CorruptFileError  # noqa: E402

from test_tiff import HAVE_CODECS, LAYOUTS, _data, _keys, _native, _reference, _retag  # noqa: E402

if not _engine.enabled():
    pytest.skip("the engine is switched off (MICRO_READER_ENGINE=0)", allow_module_level=True)

# layouts the engine handles differently from plain strips: predictors, byte order
EXTRA = {
    "float_predictor": (_data((300, 500), "f4"), {"compression": "zlib", "predictor": True,
                                                  "rowsperstrip": 37}, True),
    "big_endian_predictor": (_data((300, 500), "i2"), {"byteorder": ">", "compression": "zlib",
                                                       "predictor": True, "tile": (64, 64)}, True),
    "big_endian_zlib": (_data((300, 500), "u2"), {"byteorder": ">", "compression": "zlib",
                                                  "rowsperstrip": 50}, False),
    "rgb_predictor": (_data((200, 300, 3), "u2"), {"photometric": "rgb", "compression": "zlib",
                                                   "predictor": True, "tile": (64, 64)}, True),
    "packbits": (_data((300, 500), "u1"), {"compression": "packbits", "rowsperstrip": 64}, True),
    "rgb_raw_wide": (_data((40, 3000, 3), "u1"), {"photometric": "rgb", "rowsperstrip": 40}, False),
}


@pytest.fixture(scope="module")
def files(tmp_path_factory):
    root = tmp_path_factory.mktemp("engine")
    out = {}
    for name, (data, kwargs, needs_codecs) in {**LAYOUTS, **EXTRA}.items():
        if needs_codecs and not HAVE_CODECS:
            continue
        path = str(root / f"{name}.tif")
        tifffile.imwrite(path, data, **kwargs)
        out[name] = path
    return out


@pytest.fixture
def counted(monkeypatch):
    """The pieces handed to the engine during a test."""
    seen = {"pieces": 0}
    run = _engine.Batch.run

    def counting(self, out, counts=None):
        seen["pieces"] += len(self.rows)
        return run(self, out, counts)
    monkeypatch.setattr(_engine.Batch, "run", counting)
    return seen


@pytest.fixture
def python_path():
    """Run a block on the Python path: ``with python_path(): ...``."""
    class Off:
        def __enter__(self):
            _engine.set_engine(False)

        def __exit__(self, *exc):
            _engine.set_engine(None)
    return Off


@pytest.mark.parametrize("threads", [1, 4])
@pytest.mark.parametrize("name", sorted({**LAYOUTS, **EXTRA}))
def test_engine_matches_the_python_path(files, name, threads, counted, python_path):
    if name not in files:
        pytest.skip("needs imagecodecs to write")
    ref, _ = _reference(files[name])
    old = micro_reader.decode_threads()
    micro_reader.set_decode_threads(threads)
    try:
        keys = _keys(ref.shape, n=15)
        with python_path():
            with _native(files[name]) as img:
                expected = [img.images[0][key] for key in keys]
        assert counted["pieces"] == 0
        with _native(files[name]) as img:
            for key, want in zip(keys, expected):
                got = img.images[0][key]
                assert got.dtype == want.dtype and got.dtype.isnative
                np.testing.assert_array_equal(got, want, err_msg=f"{name} {key}")
                np.testing.assert_array_equal(got, ref[key], err_msg=f"{name} {key}")
        assert counted["pieces"] > 0
    finally:
        micro_reader.set_decode_threads(old)


def test_the_engine_reads_the_same_counts(files, python_path):
    """``stats`` counts the same samples read on both paths."""
    name = "strips_37"
    stats = []
    for engine_on in (False, True):
        _engine.set_engine(engine_on)
        try:
            with _native(files[name]) as img:
                image = img.images[0]
                for key in _keys(image.shape, n=10, seed=11):
                    image[key]
                stats.append((image.stats.pixels_returned, image.stats.pixels_read))
        finally:
            _engine.set_engine(None)
    assert stats[0] == stats[1]


def test_closing_a_file_closes_its_engine_handles(files):
    eng = _engine.engine()
    before = eng.open_files()
    with _native(files["zlib_tiled"]) as img:
        img.images[0][:10, :10]
        assert eng.open_files() == before + 1
    assert eng.open_files() == before


def test_segments_are_cached_and_the_limit_applies(files):
    eng = _engine.engine()
    eng.clear_cache()
    with _native(files["zlib_tiled"]) as img:
        img.images[0][:64, :64]
        assert eng.cached_bytes() == 64 * 64 * 2
        assert micro_reader.cached_bytes() >= eng.cached_bytes()
    assert eng.cached_bytes() == 0                # closing the file forgets its segments
    old = micro_reader.cache_limit()
    micro_reader.set_cache_limit(0)
    try:
        with _native(files["zlib_tiled"]) as img:
            img.images[0][:64, :64]
            assert eng.cached_bytes() == 0
    finally:
        micro_reader.set_cache_limit(old)


# -- the engine's API itself ------------------------------------------------------

def _plan(rows):
    return np.array(rows, dtype=np.int64).reshape(-1, len(_engine.COLUMNS))


def _raw_piece(fid, out, offset=0, rows=(0, 1), cols=(0, 4)):
    s = out.strides
    return [fid, offset, 0, 0, -1, rows[1], out.shape[1], rows[0], rows[1], cols[0], cols[1],
            0, 1, out.__array_interface__["data"][0], s[0], s[1], s[2]]


def test_engine_reads_raw_bytes_in_place(tmp_path):
    path = tmp_path / "raw.bin"
    data = np.arange(32, dtype="<u2").reshape(4, 8, 1)
    path.write_bytes(data.tobytes())
    eng = core.Engine()
    fid = eng.open(str(path))
    out = np.zeros((4, 8, 1), np.uint16)
    codec = core.Codec("raw", 2, 1)
    assert eng.read(_plan([_raw_piece(fid, out, rows=(0, 4), cols=(0, 8))]), [codec], [], out) \
        == 32
    np.testing.assert_array_equal(out, data)
    # big-endian samples come out in native order
    big = core.Codec("raw", 2, 1, big_endian=True)
    out[...] = 0
    eng.read(_plan([_raw_piece(fid, out, rows=(0, 4), cols=(0, 8))]), [big], [], out)
    np.testing.assert_array_equal(out, data.byteswap())
    assert eng.close(fid) and not eng.close(fid)


def test_engine_refuses_destinations_outside_the_output(tmp_path):
    path = tmp_path / "raw.bin"
    path.write_bytes(bytes(64))
    eng = core.Engine()
    fid = eng.open(str(path))
    out = np.zeros((1, 4, 1), np.uint8)
    other = np.zeros((1, 4, 1), np.uint8)
    codec = core.Codec("raw", 1, 1)
    with pytest.raises(ValueError, match="outside the output buffer"):
        eng.read(_plan([_raw_piece(fid, other)]), [codec], [], out)
    piece = _raw_piece(fid, out)
    piece[10] = 9                                   # x1 beyond the segment
    with pytest.raises(ValueError, match="outside its"):
        eng.read(_plan([piece]), [codec], [], out)
    with pytest.raises(ValueError, match="no codec"):
        eng.read(_plan([_raw_piece(fid, out)]), [], [], out)
    with pytest.raises(ValueError, match="not open"):
        eng.read(_plan([_raw_piece(fid + 1, out)]), [codec], [], out)
    with pytest.raises(BufferError, match="read-only"):
        out.setflags(write=False)
        eng.read(_plan([_raw_piece(fid, out)]), [codec], [], out)


def test_engine_reports_truncated_data(tmp_path):
    path = tmp_path / "short.bin"
    path.write_bytes(bytes(10))
    eng = core.Engine()
    fid = eng.open(str(path))
    out = np.zeros((1, 4, 1), np.uint8)
    with pytest.raises(ValueError, match="ends early"):
        eng.read(_plan([_raw_piece(fid, out, offset=8)]), [core.Codec("raw", 1, 1)], [], out)
    with pytest.raises(OSError):
        eng.open(str(tmp_path / "missing.bin"))


def test_truncated_tiff_is_a_corrupt_file_error(tmp_path):
    """A segment past the end of the file: the engine's error is micro-reader's."""
    path = str(tmp_path / "cut.tif")
    tifffile.imwrite(path, _data((256, 256)), compression="zlib", tile=(64, 64))
    with _native(path) as img:
        image = img.images[0]
        # the file is cut after open (open itself checks the file is whole)
        size = __import__("os").path.getsize(path)
        with open(path, "r+b") as fh:
            fh.truncate(size // 2)
        with pytest.raises(CorruptFileError, match="ends early|truncated|corrupt"):
            image[...]


def test_set_engine_true_without_core_is_refused(monkeypatch):
    monkeypatch.setattr(_engine, "_core", None)
    with pytest.raises(RuntimeError, match="not installed"):
        _engine.set_engine(True)


@pytest.mark.parametrize("dtype,predictor,layout", [
    ("u2", 2, {"rowsperstrip": 16}), ("i1", 2, {"tile": (32, 32)}),
    ("f4", 3, {"rowsperstrip": 16}), ("f4", 3, {"tile": (32, 32)})])
def test_a_predictor_on_uncompressed_data_is_ignored_on_both_paths(tmp_path, dtype, predictor,
                                                                   layout):
    """The engine and the Python path both read the stored values, as libtiff
    and tifffile from 2026.11.12 do (test_tiff has the default path)."""
    data = _data((64, 80), dtype)
    path = str(tmp_path / "pred.tif")
    tifffile.imwrite(path, data, extratags=[(32001, 3, 1, predictor, True)], **layout)
    _retag(path, 32001, 317)                                  # Predictor
    for engine in (True, False):
        _engine.set_engine(engine)
        try:
            with _native(path, axes=None) as f:
                np.testing.assert_array_equal(f.images[0][...], data)
        finally:
            _engine.set_engine(None)


# -- CZI ------------------------------------------------------------------------------
# Synthetic files: test_czi's (written with pylibCZIrw: raw and zstd, overlapping
# mosaics, scenes, RGB) and test_czi_codecs' (one subblock per compression code).
# Both modules skip when their writers are missing, so they are imported per test.

CZI_CASES = ["stack", "mosaic_overlap", "scenes", "scene_mosaics", "uint8", "rgb",
             "zstd0_stack", "zstd1_mosaic", "zstd1_hilo_scenes", "zstd1_rgb"]
CZI_CODEC_CASES = ["jpeg_gray", "jpeg_rgb", "lzw_gray16", "lzw_rgb", "jpeg_lossless_16",
                   "jpeg_lossless_8", "jpegxr_gray16", "jpegxr_rgb", "jpegxl_gray16",
                   "jpegxl_rgb", "chunked_zstd_hilo_last"]


def _czi_both_paths(path, threads, keys, **kwargs):
    """Every image of *path* read at *keys(shape)* on the engine and on the
    Python path (files opened afresh: no cache shared), compared bit for bit;
    returns the engine's arrays."""
    old = micro_reader.decode_threads()
    micro_reader.set_decode_threads(threads)
    got = {}
    try:
        for engine in (True, False):
            _engine.set_engine(engine)
            try:
                with micro_reader.open(path, samples="keep", **kwargs) as f:
                    got[engine] = [[image[key] for key in keys(image.shape)]
                                   for image in f.images]
            finally:
                _engine.set_engine(None)
    finally:
        micro_reader.set_decode_threads(old)
    for k, (mine, theirs) in enumerate(zip(got[True], got[False])):
        for a, b in zip(mine, theirs):
            assert a.dtype == b.dtype
            np.testing.assert_array_equal(a, b, err_msg=f"image {k}")
    return got[True]


@pytest.mark.parametrize("threads", [1, 4])
@pytest.mark.parametrize("tiles", ["mosaic", "separate"])
@pytest.mark.parametrize("name", CZI_CASES)
def test_czi_on_the_engine_matches_the_python_path(tmp_path, name, tiles, threads, counted):
    tc = pytest.importorskip("test_czi")
    path, expected, _ = tc._write(tmp_path, name)
    got = _czi_both_paths(path, threads, lambda shape: tc._keys(shape), tiles=tiles)
    assert counted["pieces"] > 0
    if tiles == "mosaic":
        for images, sid in zip(got, sorted(expected)):
            np.testing.assert_array_equal(images[0], expected[sid])     # key 0 is [...]


@pytest.mark.parametrize("separate", [32 << 10, 0], ids=["spans", "separate_reads"])
@pytest.mark.parametrize("compression", [None, "zstd1:ExplicitLevel=1"], ids=["raw", "zstd1"])
def test_czi_overlapping_tiles_on_the_engine(tmp_path, compression, separate, counted,
                                              monkeypatch):
    """Tiles of random sizes overlapping heavily, one hidden entirely: the
    engine gets only the visible pieces (they never overlap), and uncompressed
    pieces in row bands (forced to a few rows here) read the same."""
    tc = pytest.importorskip("test_czi")
    from micro_reader.backends import czi as backend
    monkeypatch.setattr(backend, "SEPARATE_READ_BYTES", separate)
    monkeypatch.setattr(backend, "RAW_TASK_BYTES", 256)
    path = str(tmp_path / "scattered.czi")
    expected, _ = tc.write_czi(path, tc._scattered(), compression=compression)
    got = _czi_both_paths(path, 4, lambda shape: tc._keys(shape, n=40), tiles="mosaic")
    np.testing.assert_array_equal(got[0][0], expected[0])
    assert counted["pieces"] > 0


@pytest.mark.parametrize("name", CZI_CODEC_CASES)
def test_czi_codecs_on_the_engine(tmp_path, name, counted):
    """Every compression the engine decodes, against the Python path; chunked
    subblocks stay on the Python path."""
    tcc = pytest.importorskip("test_czi_codecs")
    if name.startswith("jpegxr") and not _engine.has("HAS_JXRLIB"):
        pytest.skip("micro-reader-core built without jxrlib")
    path, expected, tol, _ = tcc._write(tmp_path, name)
    keys = lambda shape: [(Ellipsis,), (0, 0, 0, slice(10, 30), slice(20, 50)),  # noqa: E731
                          (0, 0, 0, slice(0, 1), slice(None))]
    got = _czi_both_paths(path, 4, keys)
    diff = np.abs(got[0][0][0, 0, 0].astype(np.int64) - expected.astype(np.int64))
    assert diff.max() <= tol
    assert (counted["pieces"] > 0) == (not name.startswith("chunked"))


def test_czi_tiles_too_large_to_decode_whole_stay_on_the_python_path(tmp_path, counted,
                                                                      monkeypatch):
    """A compressed tile the memory budget says to stream is the Python path's
    (decoded in parts); uncompressed tiles still go to the engine."""
    tc = pytest.importorskip("test_czi")
    from micro_reader import _memory
    path, expected, _ = tc._write(tmp_path, "zstd1_mosaic")
    monkeypatch.setattr(_memory, "should_stream", lambda nbytes: True)
    with micro_reader.open(path, tiles="mosaic") as f:
        np.testing.assert_array_equal(f.images[0][...], expected[0])
    assert counted["pieces"] == 0


def test_czi_engine_counts_what_it_reads(tmp_path):
    """stats on the engine: a window narrower than its tile is one span read
    (as on the Python path), a compressed tile counts once, then comes from the
    engine's cache."""
    tc = pytest.importorskip("test_czi")
    path, _, _ = tc._write(tmp_path, "stack")
    with micro_reader.open(path, tiles="mosaic") as f:
        image = f.images[0]
        image[1, 0, 2, 5:15, 10:30]
        assert image.stats.pixels_read == 9 * 56 + 20
    path, _, _ = tc._write(tmp_path, "zstd1_mosaic")
    with micro_reader.open(path, tiles="mosaic") as f:
        image = f.images[0]
        image[0, 0, 0, 0:10, 0:10]
        image[0, 0, 0, 5:15, 5:15]
        assert image.stats.pixels_read == 40 * 56


def test_czi_truncated_subblock_is_a_corrupt_file_error(tmp_path):
    tc = pytest.importorskip("test_czi")
    path, _, _ = tc._write(tmp_path, "zstd1_mosaic")
    with micro_reader.open(path, tiles="mosaic") as f:
        image = f.images[0]
        image.read(c=0, y=slice(0, 1), x=slice(0, 1))          # subblock headers read
        size = __import__("os").path.getsize(path)
        with open(path, "r+b") as fh:
            fh.truncate(size - 4000)
        with pytest.raises(CorruptFileError):
            image[...]
