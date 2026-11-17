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
    """The pieces handed to the engine during a test (a lone uncompressed piece
    among them, which ``run_plans`` then reads on the Python path)."""
    seen = {"pieces": 0}
    add = _engine.Batch.add

    def counting(self, *args, **kwargs):
        n = len(self.rows)
        add(self, *args, **kwargs)
        seen["pieces"] += len(self.rows) - n
    monkeypatch.setattr(_engine.Batch, "add", counting)
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


def test_one_contiguous_uncompressed_read_is_done_on_the_python_path(tmp_path, monkeypatch):
    """A read that is one uncompressed piece of whole segment rows is one read on
    the Python path, cheaper than a call to the engine: run_plans does it there.
    A few columns of a segment, two pieces, or a compressed piece go to the
    engine."""
    data = _data((512, 512), "u2")
    tiles, strips, zpath = (str(tmp_path / f"{n}.tif") for n in ("tiles", "strips", "ztiles"))
    tifffile.imwrite(tiles, data, tile=(256, 256))
    tifffile.imwrite(strips, data, rowsperstrip=1)
    calls = []
    run = _engine.Batch._run

    def counting(self, pieces, out, per_piece=None):
        calls.append(len(pieces))
        return run(self, pieces, out, per_piece)
    monkeypatch.setattr(_engine.Batch, "_run", counting)
    with _native(tiles) as f:
        img = f.images[0]
        np.testing.assert_array_equal(img[..., :256, :256], data[:256, :256])     # one tile
        assert calls == []
        np.testing.assert_array_equal(img[..., 10:20, 30:40], data[10:20, 30:40])
        np.testing.assert_array_equal(img[..., :256, :512], data[:256, :512])     # two tiles
        assert calls == [1, 2]
        assert img.stats.pixels_returned == 256 * 256 + 100 + 256 * 512
    calls.clear()
    with _native(strips) as f:
        img = f.images[0]
        np.testing.assert_array_equal(img[..., 0:64, :], data[0:64, :])          # whole rows
        assert calls == []
        np.testing.assert_array_equal(img[..., 0:64, 100:200], data[0:64, 100:200])
        assert len(calls) == 1
    if HAVE_CODECS:
        tifffile.imwrite(zpath, data, tile=(256, 256), compression="zlib")
        calls.clear()
        with _native(zpath) as f:
            np.testing.assert_array_equal(f.images[0][..., :256, :256], data[:256, :256])
        assert calls == [1]


@pytest.mark.parametrize("cache", [0, 64 << 20], ids=["cache_off", "cache_on"])
def test_a_short_segment_decoded_into_the_output_reads_as_zeros(tmp_path, cache):
    """A whole tile is decoded straight into the output array (np.empty): a
    deflate stream that ends before the tile is full leaves zeros there, as on
    the Python path, not what the memory held."""
    import zlib
    path = str(tmp_path / "short.tif")
    data = np.full((128, 128), 9, np.uint8)
    tifffile.imwrite(path, data, tile=(64, 64), compression="zlib")
    with tifffile.TiffFile(path) as tif:
        offset = tif.pages[0].dataoffsets[0]
        count = tif.pages[0].databytecounts[0]
    short = zlib.compress(bytes([7]) * 2048)               # half of the 64 x 64 tile
    assert len(short) <= count
    with open(path, "r+b") as fh:
        fh.seek(offset)
        fh.write(short + bytes(count - len(short)))
    want = data.copy()
    want[:64, :64] = 0
    want[:32, :64] = 7
    old = micro_reader.cache_limit()
    micro_reader.set_cache_limit(cache)
    try:
        for engine in (True, False):
            _engine.set_engine(engine)
            try:
                for _ in range(2):          # the second read: from the cache, if on
                    with _native(path) as f:
                        np.testing.assert_array_equal(f.images[0][..., :64, :64], want[:64, :64])
                        np.testing.assert_array_equal(f.images[0][...], want)
            finally:
                _engine.set_engine(None)
    finally:
        micro_reader.set_cache_limit(old)


@pytest.mark.parametrize("compression", [None, "zlib"])
def test_a_reordering_view_is_planned_straight_into_its_array(tmp_path, compression,
                                                               monkeypatch, python_path):
    """A view that reorders the file's axes (RGB samples folded into the channels,
    other axes orders) reads on the engine straight into its final array, with no
    copy afterwards; channels that split a pixel's samples are read and copied as
    before.  Pixels and stats as on the Python path."""
    if compression and not HAVE_CODECS:
        pytest.skip("needs imagecodecs to write")
    data = _data((2, 96, 160, 3), "u2")                       # z, y, x, samples
    path = str(tmp_path / "rgb.tif")
    tifffile.imwrite(path, data, photometric="rgb", tile=(32, 32), compression=compression,
                     metadata={"axes": "ZYXS"})
    keys = [(...,), (0, slice(0, 3), 0, slice(5, 70), slice(9, 150)),
            (0, slice(1, 2), slice(None), slice(None), slice(None))]   # one sample: split
    planned = []
    from micro_reader._axes import AxesImage
    plan_into = AxesImage._plan_into

    def logging(self, block, out, stack):
        plans = plan_into(self, block, out, stack)
        planned.append(plans is not None)
        return plans
    monkeypatch.setattr(AxesImage, "_plan_into", logging)
    results = {}
    for engine in (True, False):
        _engine.set_engine(engine)
        try:
            with micro_reader.open(path, axes="tczyx") as f:
                img = f.images[0]
                assert img.axes == "tczyx" and img.shape == (1, 3, 2, 96, 160)
                got = [img[k] for k in keys]
                got.append(img.as_axes("xyzc", samples="channels")[10:50, 20:40, :, :])
                results[engine] = (got, img.stats)
        finally:
            _engine.set_engine(None)
    # the engine's reads only (the Python path does not plan); a view of a view
    # plans through both
    assert planned[:3] == [True, True, False] and all(planned[3:]) and len(planned) > 3
    for a, b in zip(*(results[e][0] for e in (True, False))):
        assert a.flags.c_contiguous
        np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(results[True][0][0][0, :, :, :, :], np.moveaxis(data, 3, 0))
    assert results[True][1] == results[False][1]


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


def test_czi_headers_read_on_the_pool_give_the_same_pixels(tmp_path, counted, monkeypatch):
    """A read needing more than HEADERS_ON_POOL unread subblock headers reads them
    on the pool first (a slide scan's thousands); lowered here so a small mosaic
    takes that way."""
    tc = pytest.importorskip("test_czi")
    from micro_reader import _pool
    from micro_reader.backends import czi
    monkeypatch.setattr(czi, "HEADERS_ON_POOL", 0)
    runs = []
    run = _pool.run

    def logging(tasks):
        tasks = list(tasks)
        runs.append(len(tasks))
        return run(tasks)
    monkeypatch.setattr(czi._pool, "run", logging)
    path, expected, _ = tc._write(tmp_path, "mosaic_overlap")
    got = _czi_both_paths(path, 4, lambda shape: tc._keys(shape), tiles="mosaic")
    assert counted["pieces"] > 0 and any(runs)
    for images, sid in zip(got, sorted(expected)):
        np.testing.assert_array_equal(images[0], expected[sid])


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


# -- one cache limit ------------------------------------------------------------------

def test_the_engine_and_the_python_caches_share_one_limit(tmp_path):
    """The limit covers the engine's cache and the Python caches together, and
    each side can take at least half of it from the other: a process reading
    one format on the Python path and another on the engine stays within it."""
    from micro_reader import _io
    if not HAVE_CODECS:
        pytest.skip("needs imagecodecs to write zlib tiles")
    paths = []
    for name in ("python.tif", "engine.tif"):
        path = str(tmp_path / name)
        tifffile.imwrite(path, _data((512, 512), "u2"), compression="zlib", tile=(64, 64))
        paths.append(path)                                 # 64 tiles of 8 KB decoded
    limit, tile = 256 << 10, 64 * 64 * 2
    old = micro_reader.cache_limit()
    micro_reader.set_cache_limit(limit)
    eng = _engine.engine()

    files = [_native(path) for path in paths]             # open throughout: closing
                                                           # a file empties its caches
    def read(k, engine):
        _engine.set_engine(engine)
        try:
            files[k].images[0][...]
        finally:
            _engine.set_engine(None)

    def held():
        return _io._shared_bytes, eng.cached_bytes()

    try:
        eng.clear_cache()
        read(0, False)                                     # the Python caches fill up
        python, engine = held()
        assert limit - tile < python <= limit and engine == 0
        read(1, True)                                      # they give back half to the engine
        python, engine = held()
        assert python <= limit // 2 and engine >= limit // 2 - tile
        assert python + engine <= limit
        read(0, False)                                     # the engine gives back half
        python, engine = held()
        assert python >= limit // 2 - tile and engine <= limit // 2
        assert python + engine <= limit
        assert micro_reader.cached_bytes() == python + engine
        micro_reader.set_cache_limit(64 << 10)            # lowered: both shrink
        assert micro_reader.cached_bytes() <= 64 << 10
    finally:
        for f in files:
            f.close()
        micro_reader.set_cache_limit(None if old == _io.DEFAULT_CACHE_LIMIT else old)
