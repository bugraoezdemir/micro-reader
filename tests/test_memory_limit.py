"""Memory-bounded reads of compressed blocks (micro_reader._memory, _stream;
docs/PLAN.md).

- A block too large to cache (``STREAM_THRESHOLD``) is streamed where its
  codec allows: decoded from its start, keeping only the rows a read needs.
- Every block decoded whole reserves its size: the blocks in flight never
  exceed ``memory_limit`` together, whichever threads decode them.
- A block that cannot stream and is larger than the limit raises
  ``MemoryLimitError``.
"""
from __future__ import annotations

import importlib.util
import os
import tracemalloc

import numpy as np
import pytest

import micro_reader
from micro_reader import MemoryLimitError, _memory, _stream

tifffile = pytest.importorskip("tifffile")
RNG = np.random.default_rng(12)


@pytest.fixture
def small_budget(monkeypatch):
    """Blocks above 4 MB are streamed; at most 4 MB decoded whole at once."""
    monkeypatch.setattr(_memory, "STREAM_THRESHOLD", 4 << 20)
    monkeypatch.setattr(_memory, "MINIMUM_LIMIT", 1 << 20)
    micro_reader.set_memory_limit("4MB")
    yield _memory.STREAM_THRESHOLD
    micro_reader.set_memory_limit(None)


def test_sizes_and_the_budget(monkeypatch):
    assert _memory.parse_size("8GB") == 8 << 30
    assert _memory.parse_size("1.5 mb") == int(1.5 * 2**20)
    assert _memory.parse_size(1234) == 1234
    with pytest.raises(ValueError):
        _memory.parse_size("lots")
    monkeypatch.setenv("MICRO_READER_MEMORY_LIMIT", "2GB")
    micro_reader.set_memory_limit(None)
    try:
        assert micro_reader.memory_limit() == 2 << 30
        micro_reader.set_memory_limit("1GB")
        assert micro_reader.memory_limit() == 1 << 30
        micro_reader.set_memory_limit(1)                   # never below the floor
        assert micro_reader.memory_limit() == _memory.MINIMUM_LIMIT
    finally:
        micro_reader.set_memory_limit(None)


def _plane(rows=1200, cols=2400, dtype="u2"):   # 5.8 MB: above the 4 MB budget
    data = RNG.integers(0, 40, (rows, cols)) + np.arange(cols)[None, :]
    return data.astype(dtype)


def _imwrite(path, data, **kwargs) -> None:
    """``tifffile.imwrite``, skipping where tifffile needs imagecodecs to
    encode the file (LZW, PackBits, zstd, the floating-point predictor) and
    it is not installed, as in CI's oldest-dependencies job."""
    try:
        tifffile.imwrite(path, data, **kwargs)
    except (KeyError, ImportError) as exc:
        if importlib.util.find_spec("imagecodecs") is not None:
            raise
        pytest.skip(f"tifffile cannot write this test file without imagecodecs: {exc}")


CODECS = [("zlib", None), ("zlib", "horizontal"), ("lzw", None), ("lzw", "horizontal"),
          ("zstd", None), ("packbits", None), ("lzma", None)]


@pytest.mark.parametrize("compression,predictor", CODECS)
def test_one_strip_planes_are_streamed(tmp_path, small_budget, compression, predictor):
    """A plane stored as one compressed strip (bigger than the budget):
    regions and the whole plane equal the data, through the stream."""
    data = _plane()
    path = str(tmp_path / "strip.tif")
    kwargs = {"predictor": predictor} if predictor else {}
    _imwrite(path, data, compression=compression, rowsperstrip=data.shape[0], **kwargs)
    assert data.nbytes > small_budget
    if not _stream.streamable(compression):
        with micro_reader.open(path) as f, pytest.raises(MemoryLimitError, match="decoded whole"):
            f.images[0][500:600, 700:800]
        return
    with micro_reader.open(path) as f:
        image = f.images[0]
        np.testing.assert_array_equal(image[500:600, 700:800], data[500:600, 700:800])
        np.testing.assert_array_equal(image[1100:1200, :], data[1100:1200])
        np.testing.assert_array_equal(image[...], data)


def test_float_predictor_is_streamed(tmp_path, small_budget):
    data = RNG.normal(size=(1100, 1200)).astype("f4")
    path = str(tmp_path / "float.tif")
    _imwrite(path, data, compression="zlib", predictor="floatingpoint",
             rowsperstrip=data.shape[0])
    with micro_reader.open(path) as f:
        np.testing.assert_array_equal(f.images[0][300:310, 50:950], data[300:310, 50:950])


def test_a_region_costs_a_piece_not_the_block(tmp_path, small_budget):
    """Peak memory while reading 100 x 100 pixels of a 9 MB one-strip plane
    stays far below the plane (numpy allocations are traced)."""
    data = _plane(1500, 3000)
    path = str(tmp_path / "strip.tif")
    tifffile.imwrite(path, data, compression="zlib", rowsperstrip=data.shape[0])
    with micro_reader.open(path) as f:
        image = f.images[0]
        tracemalloc.start()
        block = image[700:800, 1400:1500]
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
    np.testing.assert_array_equal(block, data[700:800, 1400:1500])
    assert peak < data.nbytes / 4, f"peak {peak / 2**20:.1f} MB for a {data.nbytes / 2**20:.1f} MB plane"


def test_regions_read_top_to_bottom_continue_the_stream(tmp_path, small_budget):
    """Row bands in order never start the block again (linear, not quadratic)."""
    data = _plane()
    path = str(tmp_path / "strip.tif")
    tifffile.imwrite(path, data, compression="zlib", rowsperstrip=data.shape[0])
    with micro_reader.open(path) as f:
        image = f.images[0]
        for y in range(0, 1200, 200):
            np.testing.assert_array_equal(image[y:y + 200, :], data[y:y + 200])
        decoders = [d for key, ds in _stream._idle.items() if key[0] == path for d in ds]
        assert decoders and all(d.restarts == 0 for d in decoders)
        del decoders
        image[0:10, :]                                       # back to the top: a fresh start
    # the file is released on close (Windows cannot delete an open file)
    os.remove(path)


def test_blocks_within_the_budget_are_decoded_whole(tmp_path, monkeypatch):
    data = _plane(200, 300)
    path = str(tmp_path / "small.tif")
    tifffile.imwrite(path, data, compression="zlib", rowsperstrip=200)

    def no_stream(*args, **kwargs):
        raise AssertionError("a small block was streamed")

    monkeypatch.setattr(_stream, "read", no_stream)
    with micro_reader.open(path) as f:
        np.testing.assert_array_equal(f.images[0][...], data)


def test_nd2_lossless_frames_are_streamed(tmp_path, small_budget):
    from _nd2_writer import write_nd2
    frames = RNG.integers(0, 50, (2, 1200, 1500, 2, 1)).astype(np.uint16)   # 7 MB frames
    path = str(tmp_path / "big.nd2")
    write_nd2(path, frames, [("T", 2)], compression="lossless")
    expected = frames[..., 0].transpose(0, 3, 1, 2)[:, :, None]            # t c z y x
    with micro_reader.open(path) as f:
        image = f.images[0]
        np.testing.assert_array_equal(image[1, :, 0, 500:600, 700:800],
                                      expected[1, :, 0, 500:600, 700:800])
        np.testing.assert_array_equal(image[...], expected)


def test_blocks_that_cannot_stream_are_refused_clearly(tmp_path, small_budget):
    """A JPEG-compressed one-strip plane bigger than the budget: a clear
    error naming the limit and how to raise it, not running out of memory."""
    pytest.importorskip("imagecodecs")
    data = RNG.integers(0, 255, (1500, 3000)).astype("u1")
    path = str(tmp_path / "jpeg.tif")
    tifffile.imwrite(path, data, compression="jpeg", rowsperstrip=data.shape[0])
    with micro_reader.open(path) as f:
        with pytest.raises(MemoryLimitError, match="set_memory_limit"):
            f.images[0][0:10, 0:10]
    micro_reader.set_memory_limit("64MB")
    with micro_reader.open(path) as f:                  # a larger budget reads it
        assert f.images[0][0:10, 0:10].shape == (10, 10)


def test_big_pictures_are_refused_clearly(tmp_path, small_budget):
    PIL = pytest.importorskip("PIL.Image")
    path = str(tmp_path / "big.png")
    PIL.fromarray(RNG.integers(0, 255, (1500, 3000)).astype("u1")).save(path)
    with micro_reader.open(path) as f, pytest.raises(MemoryLimitError, match="PNG picture"):
        f.images[0][0:10, 0:10]


def test_blocks_in_flight_never_exceed_the_limit(tmp_path, monkeypatch):
    """Many threads -- a caller's own, as eubi-bridge's writer reads regions
    -- decoding 1.5 MB blocks whole at once: together they stay within the
    4 MB limit (once, each region's thread decoded its block regardless,
    and seven copies of a 763 MB strip were in memory at once)."""
    import threading
    monkeypatch.setattr(_memory, "MINIMUM_LIMIT", 1 << 20)
    micro_reader.set_memory_limit("4MB")
    try:
        paths = []
        for k in range(8):
            data = _plane(600, 1300)                         # 1.5 MB: decoded whole
            path = str(tmp_path / f"strip{k}.tif")
            tifffile.imwrite(path, data, compression="zlib", rowsperstrip=600)
            paths.append((path, data))
        _memory.peak_reserved(reset=True)
        errors = []

        def work(path, data):
            with micro_reader.open(path) as f:
                if not np.array_equal(f.images[0][100:200, 300:400], data[100:200, 300:400]):
                    errors.append(path)

        threads = [threading.Thread(target=work, args=pd) for pd in paths]
        for th in threads:
            th.start()
        for th in threads:
            th.join()
        assert not errors
        assert 0 < _memory.peak_reserved() <= 4 << 20
    finally:
        micro_reader.set_memory_limit(None)


def test_jumps_back_resume_from_checkpoints(tmp_path, small_budget, monkeypatch):
    """Regions read out of order (a viewer): a read before the decoder's
    place resumes from the nearest checkpoint, not the block's start."""
    monkeypatch.setattr(_stream, "MIN_CHECKPOINT_EVERY", 64 << 10)
    data = _plane(2400, 2400)                                # 11 MB, one strip
    path = str(tmp_path / "strip.tif")
    tifffile.imwrite(path, data, compression="zlib", rowsperstrip=data.shape[0])
    rng = np.random.default_rng(7)
    with micro_reader.open(path) as f:
        image = f.images[0]
        image[2300:2400, :]                                  # one pass: checkpoints laid
        for _ in range(12):
            y, x = (int(v) for v in rng.integers(0, 2300, 2))
            np.testing.assert_array_equal(image[y:y + 100, x:x + 100], data[y:y + 100, x:x + 100])
        decoders = [d for key, ds in _stream._idle.items() if key[0] == path for d in ds]
        assert sum(d.resumes for d in decoders) > 0
        assert sum(d.restarts for d in decoders) == 0
        del decoders


@pytest.mark.parametrize("kind", ["zstd0", "zstd1", "zstd1_hilo"])
def test_large_czi_subblocks_are_streamed(tmp_path, small_budget, kind):
    """A CZI subblock bigger than the cache threshold: zstd (ZSTD0, ZSTD1)
    streamed by rows; hi-lo packed ZSTD1 (low and high bytes of the whole
    tile apart) decoded whole within the budget."""
    zstandard = pytest.importorskip("zstandard")
    from _czi_writer import write_czi
    data = _plane(1200, 2400)                                # 5.8 MB > the 4 MB threshold
    raw = data.astype("<u2").tobytes()
    if kind == "zstd1_hilo":
        packed = np.concatenate([np.frombuffer(raw, np.uint8)[0::2], np.frombuffer(raw, np.uint8)[1::2]])
        payload = bytes([3, 1, 1]) + zstandard.ZstdCompressor().compress(packed.tobytes())
    else:
        frame = zstandard.ZstdCompressor().compress(raw)
        payload = frame if kind == "zstd0" else bytes([3, 1, 0]) + frame
    path = str(tmp_path / "big.czi")
    write_czi(path, [dict(series=0, plane={"C": 0}, x=0, y=0, width=2400, height=1200,
                          dtype="u2", compression=5 if kind == "zstd0" else 6, payload=payload)])
    if kind == "zstd1_hilo":
        micro_reader.set_memory_limit("64MB")            # decoded whole: needs the room
    with micro_reader.open(path) as f:
        image = f.images[0]
        np.testing.assert_array_equal(image[0, 0, 0, 500:600, 700:800], data[500:600, 700:800])
        np.testing.assert_array_equal(image[0, 0, 0], data)
        decoders = [d for key, ds in _stream._idle.items() if key[0] == path for d in ds]
        assert bool(decoders) == (kind != "zstd1_hilo")
        del decoders


@pytest.mark.parametrize("mode", ["L", "I;16", "RGB", "P_grey", "P_colour", "1"])
def test_large_pngs_are_streamed_by_rows(tmp_path, monkeypatch, mode):
    """A PNG too large to keep decoded, not interlaced: its rows streamed,
    unpacked like a whole decode (grey palettes applied, colour palettes as
    indices, 1-bit unpacked, 16-bit little-endian)."""
    PIL = pytest.importorskip("PIL.Image")
    if not _stream.streamable("png"):
        pytest.skip("PNG rows stream with the native codecs only")
    from micro_reader.backends import raster
    path = str(tmp_path / "big.png")
    h, w = 900, 1300
    if mode == "I;16":
        PIL.fromarray(RNG.integers(0, 65535, (h, w)).astype("u2")).save(path)
    elif mode == "RGB":
        PIL.fromarray(RNG.integers(0, 255, (h, w, 3)).astype("u1")).save(path)
    elif mode.startswith("P"):
        img = PIL.fromarray(RNG.integers(0, 16, (h, w)).astype("u1"), mode="P")
        ramp = [v for k in range(256) for v in ((k, k, k) if mode == "P_grey" else (k, 255 - k, 7))]
        img.putpalette(ramp)
        img.save(path)
    elif mode == "1":
        PIL.fromarray(RNG.integers(0, 2, (h, w)).astype(bool)).save(path)
    else:
        PIL.fromarray(RNG.integers(0, 255, (h, w)).astype("u1")).save(path)
    with micro_reader.open(path) as f:                    # whole decode: the reference
        whole = f.images[0][...]
    monkeypatch.setattr(raster, "CACHE_BYTES", 1 << 16)   # now too large to keep: streamed
    with micro_reader.open(path) as f:
        image = f.images[0]
        np.testing.assert_array_equal(image[300:420, 100:900], whole[300:420, 100:900])
        np.testing.assert_array_equal(image[...], whole)
        assert image._cached is None and image._streams()


def test_large_jpeg2000_segments_decode_only_the_region(tmp_path, small_budget, monkeypatch):
    """A JPEG 2000 TIFF segment too large to cache: only the region's
    code-blocks are decoded (OpenJPEG's decode area), never the segment."""
    imagecodecs = pytest.importorskip("imagecodecs")
    from micro_reader import _codecs
    data = _plane(1200, 2400)                                # 5.8 MB > the 4 MB threshold
    path = str(tmp_path / "j2k.tif")
    tifffile.imwrite(path, data, compression="jpeg2000", compressionargs={"level": 0},
                     rowsperstrip=data.shape[0])
    whole = []
    real = _codecs.jpeg2k_decode_into
    monkeypatch.setattr(_codecs, "jpeg2k_decode_into",
                        lambda *a, **k: whole.append(1) or real(*a, **k))
    from micro_reader.backends import tiff
    monkeypatch.setitem(tiff._IMAGE_DECODERS, "jpeg2k", _codecs.jpeg2k_decode_into)
    with micro_reader.open(path) as f:
        image = f.images[0]
        np.testing.assert_array_equal(image[500:600, 700:800], data[500:600, 700:800])
        np.testing.assert_array_equal(image[1100:1200, :], data[1100:1200])
        np.testing.assert_array_equal(image[:, :], data)      # in bands within the budget
    assert not whole                                          # the whole segment never decoded


def test_large_legacy_nd2_frames_decode_only_the_region(tmp_path, small_budget):
    """Legacy ND2 frames are JPEG 2000 codestreams: a frame too large to
    cache is decoded only where a read asks."""
    pytest.importorskip("imagecodecs")
    from _nd2_legacy_writer import write_legacy_nd2
    frames = (RNG.integers(0, 40, (1, 2, 1200, 2400)) + np.arange(2400)).astype("u2")
    path = str(tmp_path / "big.nd2")
    write_legacy_nd2(path, frames, [])
    with micro_reader.open(path) as f:
        image = f.images[0]
        np.testing.assert_array_equal(image[0, :, 0, 500:600, 700:800],
                                      frames[0, :, 500:600, 700:800])
        np.testing.assert_array_equal(image[0, 1, 0], frames[0, 1])
        assert not image.decode_cache.get((0, 0))             # nothing decoded whole


def _jpeg(pixels, **options) -> bytes:
    """*pixels* as a JPEG (Pillow: restart markers by ``restart_marker_rows``)."""
    import io
    PIL = pytest.importorskip("PIL.Image")
    out = io.BytesIO()
    PIL.fromarray(pixels).save(out, "JPEG", quality=90, **options)
    return out.getvalue()


def _jpeg_pixels(data: bytes, shape) -> np.ndarray:
    """A whole decode: what a JPEG cut at its restart markers must equal."""
    from micro_reader import _codecs
    out = np.empty(shape, "u1")
    _codecs.jpeg_decode_into(data, out)
    return out


def _colour(rows, cols):
    return (RNG.integers(0, 60, (rows, cols, 3)) + np.arange(cols)[None, :, None] // 8).astype("u1")


@pytest.mark.parametrize("kind", ["grey", "rgb_420", "rgb_blocks"])
def test_large_jpeg_segments_are_cut_at_restart_markers(tmp_path, small_budget, kind):
    """A JPEG TIFF segment too large to cache, with restart markers: the
    intervals covering the rows asked for are decoded alone (a JPEG of
    their own), never the segment; pixels as a whole decode."""
    if kind == "grey":
        pixels, options = _plane(1500, 3000, "u1"), dict(restart_marker_rows=1)
    elif kind == "rgb_420":
        pixels, options = _colour(1200, 1700), dict(restart_marker_rows=1, subsampling=2)
    else:                    # intervals across MCU rows: 100 MCUs a row, 40 an interval
        pixels, options = _colour(1200, 1600), dict(restart_marker_blocks=40, subsampling=2)
    data = _jpeg(pixels, **options)
    whole = _jpeg_pixels(data, pixels.shape)
    path = str(tmp_path / "jpeg.tif")
    tifffile.imwrite(path, iter([data]), shape=pixels.shape, dtype="u1",
                     photometric="minisblack" if pixels.ndim == 2 else "rgb",
                     compression="jpeg", rowsperstrip=pixels.shape[0])
    with micro_reader.open(path) as f:
        image = f.images[0]
        np.testing.assert_array_equal(image[500:600, 700:800], whole[500:600, 700:800])
        np.testing.assert_array_equal(image[1100:1200, :], whole[1100:1200])
        np.testing.assert_array_equal(image[...], whole)       # in bands within the budget
        assert list(image._restarts.values())[0] is not None
        assert not image.decode_cache.get((0, 0))              # never decoded whole


def test_large_jpegs_without_restart_markers_are_refused_clearly(tmp_path, small_budget):
    """A JPEG segment above the limit without restart markers (or progressive)
    must be decoded whole: a clear MemoryLimitError, not a crash."""
    pixels = _colour(1200, 1700)
    path = str(tmp_path / "jpeg.tif")
    tifffile.imwrite(path, iter([_jpeg(pixels, progressive=True)]), shape=pixels.shape,
                     dtype="u1", photometric="rgb", compression="jpeg",
                     rowsperstrip=pixels.shape[0])
    with micro_reader.open(path) as f, pytest.raises(MemoryLimitError, match="restart markers"):
        f.images[0][0:10, 0:10]


def test_large_jpeg_pictures_are_cut_at_restart_markers(tmp_path, monkeypatch):
    """A .jpg too large to keep decoded, with restart markers: read by bands
    of rows; without them, decoded whole as before."""
    from micro_reader.backends import raster
    pixels = _colour(900, 1300)
    path, plain = str(tmp_path / "big.jpg"), str(tmp_path / "plain.jpg")
    data = _jpeg(pixels, restart_marker_rows=2, subsampling=2)
    with open(path, "wb") as fh:
        fh.write(data)
    with open(plain, "wb") as fh:
        fh.write(_jpeg(pixels, subsampling=2))
    whole = _jpeg_pixels(data, pixels.shape)
    monkeypatch.setattr(raster, "CACHE_BYTES", 1 << 16)   # too large to keep: cut
    with micro_reader.open(path) as f:
        image = f.images[0]
        np.testing.assert_array_equal(image[300:420, 100:900], whole[300:420, 100:900])
        np.testing.assert_array_equal(image[...], whole)
        assert image._cached is None and image._streams()
    with micro_reader.open(plain) as f:
        assert not f.images[0]._streams()
        assert f.images[0][...].shape == pixels.shape


@pytest.mark.parametrize("kind", ["jpeg", "jxl"])
def test_large_czi_image_codec_subblocks_decode_only_the_piece(tmp_path, small_budget,
                                                               monkeypatch, kind):
    """A JPEG (with restart markers) or JPEG XL CZI subblock too large to
    cache: only the piece asked for is decoded, never the subblock."""
    from _czi_writer import write_czi
    from micro_reader.backends import czi
    if kind == "jpeg":
        pixels = _plane(1500, 3000, "u1")
        payload = _jpeg(pixels, restart_marker_rows=1)
        expected, dtype, code = _jpeg_pixels(payload, pixels.shape), "u1", 1
    else:
        imagecodecs = pytest.importorskip("imagecodecs")
        pixels = _plane(1200, 2400)
        payload = imagecodecs.jpegxl_encode(pixels, lossless=True, effort=1)
        expected, dtype, code = pixels, "u2", 8
    monkeypatch.setattr(czi._CziImage, "_decoded",
                        lambda *a: pytest.fail("the subblock was decoded whole"))
    path = str(tmp_path / "big.czi")
    h, w = pixels.shape
    write_czi(path, [dict(series=0, plane={"C": 0}, x=0, y=0, width=w, height=h,
                          dtype=dtype, compression=code, payload=payload)])
    with micro_reader.open(path) as f:
        image = f.images[0]
        np.testing.assert_array_equal(image[0, 0, 0, 500:600, 700:800], expected[500:600, 700:800])
        np.testing.assert_array_equal(image[0, 0, 0], expected)


def test_large_jpegxl_segments_decode_only_the_region(tmp_path, small_budget, monkeypatch):
    """A JPEG XL TIFF segment too large to cache: only the groups covering
    the region are rendered (jxl-oxide), never the segment."""
    pytest.importorskip("imagecodecs")
    from micro_reader import _codecs
    from micro_reader.backends import tiff
    data = _plane(1200, 2400)
    path = str(tmp_path / "jxl.tif")
    tifffile.imwrite(path, data, compression="jpegxl", compressionargs={"lossless": True,
                                                                         "effort": 1},
                     rowsperstrip=data.shape[0])
    monkeypatch.setitem(tiff._IMAGE_DECODERS, "jxl",
                        lambda *a, **k: pytest.fail("the segment was decoded whole"))
    with micro_reader.open(path) as f:
        image = f.images[0]
        assert image._codec == "jxl"
        np.testing.assert_array_equal(image[500:600, 700:800], data[500:600, 700:800])
        np.testing.assert_array_equal(image[:, :], data)


def test_jpegs_whose_intervals_seldom_meet_a_row_are_decoded_whole(tmp_path, small_budget):
    """Restart intervals that start on an MCU row only every 50 rows (107
    MCUs a row, 50 an interval): a band would be larger than the budget, so
    the segment is decoded whole -- refused clearly above the limit."""
    pixels = _colour(1200, 1700)
    path = str(tmp_path / "jpeg.tif")
    tifffile.imwrite(path, iter([_jpeg(pixels, restart_marker_blocks=50, subsampling=2)]),
                     shape=pixels.shape, dtype="u1", photometric="rgb", compression="jpeg",
                     rowsperstrip=pixels.shape[0])
    with micro_reader.open(path) as f, pytest.raises(MemoryLimitError, match="restart markers"):
        f.images[0][0:10, 0:10]
