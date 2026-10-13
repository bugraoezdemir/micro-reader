"""TIFF backend, against tifffile's own ``series.asarray()`` on synthetic files."""
from __future__ import annotations

import os
import threading

import numpy as np
import pytest

tifffile = pytest.importorskip("tifffile")

import micro_reader  # noqa: E402
from micro_reader import UnsupportedFormatError  # noqa: E402
from micro_reader.backends import tiff as tiff_backend  # noqa: E402

try:
    import imagecodecs  # noqa: F401
    HAVE_CODECS = True
except ImportError:
    HAVE_CODECS = False


def _data(shape, dtype="u2", seed=0):
    rng = np.random.default_rng(seed)
    if np.dtype(dtype).kind == "f":
        return rng.normal(size=shape).astype(dtype)
    info = np.iinfo(dtype)
    return rng.integers(info.min, info.max, size=shape, endpoint=True).astype(dtype)


# name -> (data, imwrite kwargs, needs imagecodecs).  Odd sizes so strips and
# tiles end part-way; every layout the reader distinguishes appears.
LAYOUTS = {
    "single_strip":   (_data((300, 500)), {"rowsperstrip": 300}, False),
    "strips_37":      (_data((300, 500)), {"rowsperstrip": 37}, False),
    "strips_1":       (_data((120, 90)), {"rowsperstrip": 1}, False),
    "tiled":          (_data((300, 500)), {"tile": (64, 128)}, False),
    "tiled_narrow":   (_data((300, 100)), {"tile": (64, 128)}, False),  # one padded tile across
    "big_endian":     (_data((300, 500), "i2"), {"byteorder": ">", "rowsperstrip": 37}, False),
    "big_endian_tiled": (_data((300, 500), "f4"), {"byteorder": ">", "tile": (64, 64)}, False),
    "bigtiff":        (_data((300, 500)), {"bigtiff": True, "tile": (64, 64)}, False),
    "uint8":          (_data((300, 500), "u1"), {}, False),
    "float32":        (_data((300, 500), "f4"), {"rowsperstrip": 64}, False),
    "zlib_strips":    (_data((300, 500)), {"compression": "zlib", "rowsperstrip": 37}, False),
    "zlib_single":    (_data((300, 500)), {"compression": "zlib", "rowsperstrip": 300}, False),
    "zlib_tiled":     (_data((300, 500)), {"compression": "zlib", "tile": (64, 64)}, False),
    "zlib_predictor": (_data((300, 500)), {"compression": "zlib", "predictor": True,
                                           "tile": (64, 64)}, True),
    "lzw":            (_data((300, 500)), {"compression": "lzw", "rowsperstrip": 50}, True),
    "zstd_tiled":     (_data((300, 500)), {"compression": "zstd", "tile": (64, 64)}, True),
    "rgb_contig":     (_data((200, 300, 3), "u1"), {"photometric": "rgb"}, False),
    "rgb_contig_tiled": (_data((200, 300, 3), "u1"), {"photometric": "rgb", "tile": (64, 64)}, False),
    "rgb_separate":   (_data((3, 200, 300), "u1"), {"photometric": "rgb",
                                                    "planarconfig": "separate",
                                                    "rowsperstrip": 50}, False),
    "rgb_zlib":       (_data((200, 300, 3), "u1"), {"photometric": "rgb",
                                                    "compression": "zlib", "tile": (64, 64)}, False),
    "jpeg_rgb":       (_data((200, 300, 3), "u1"), {"photometric": "rgb",
                                                    "compression": "jpeg", "tile": (64, 64)}, True),
    "imagej_zyx":     (_data((7, 120, 90)), {"imagej": True, "metadata": {"axes": "ZYX"}}, False),
    "imagej_tzcyx":   (_data((2, 3, 4, 50, 60)), {"imagej": True,
                                                  "metadata": {"axes": "TZCYX"}}, False),
    "ome_czyx_tiled": (_data((2, 5, 128, 96)), {"ome": True, "tile": (32, 32),
                                                "metadata": {"axes": "CZYX"}}, False),
    "ome_zlib":       (_data((3, 4, 100, 70)), {"ome": True, "compression": "zlib",
                                                "metadata": {"axes": "TCYX"}}, False),
}


@pytest.fixture(params=[1, 4], ids=["serial", "pool"])
def decode_threads(request):
    """Decode on the caller's thread, and on the shared pool."""
    old = micro_reader.decode_threads()
    micro_reader.set_decode_threads(request.param)
    yield request.param
    micro_reader.set_decode_threads(old)


@pytest.fixture(scope="module")
def files(tmp_path_factory):
    root = tmp_path_factory.mktemp("tiff")
    out = {}
    for name, (data, kwargs, needs_codecs) in LAYOUTS.items():
        if needs_codecs and not HAVE_CODECS:
            continue
        path = str(root / f"{name}.tif")
        tifffile.imwrite(path, data, **kwargs)
        out[name] = path
    return out


def _reference(path, series=0):
    with tifffile.TiffFile(path) as tf:
        return tf.series[series].asarray(), tf.series[series].axes.lower()


def _keys(shape, n=25, seed=3):
    """Whole array, single planes, edges and random rectangles (with steps)."""
    rng = np.random.default_rng(seed)
    keys = [(Ellipsis,), (0,), (-1,)]
    for _ in range(n):
        key = []
        for size in shape:
            a = int(rng.integers(0, size))
            b = int(rng.integers(a + 1, size + 1))
            choice = rng.integers(0, 5)
            key.append(a if choice == 0 else
                       slice(a, b, int(rng.integers(1, 4))) if choice == 1 else slice(a, b))
        keys.append(tuple(key))
    return keys


@pytest.mark.parametrize("name", list(LAYOUTS))
def test_matches_tifffile(files, name, decode_threads):
    if name not in files:
        pytest.skip("needs imagecodecs")
    ref, axes = _reference(files[name])
    with micro_reader.open(files[name]) as img:
        scene = img.images[0]
        assert img.backend == "tiff"
        assert scene.shape == ref.shape and scene.axes == axes and scene.dtype == ref.dtype
        for key in _keys(ref.shape):
            got = scene[key]
            assert got.dtype.isnative
            if name == "jpeg_rgb":
                # lossy JPEG: our decoder vs libjpeg may differ by a few levels
                # (the standard allows it; measured max 3 on noise)
                diff = np.abs(got.astype(int) - ref[key].astype(int))
                assert diff.max(initial=0) <= 3, f"{name} {key}"
            else:
                np.testing.assert_array_equal(got, ref[key], err_msg=f"{name} {key}")


@pytest.mark.parametrize("name", ["single_strip", "strips_37", "strips_1", "big_endian",
                                  "float32", "imagej_tzcyx"])
def test_uncompressed_strips_read_exactly(files, name, monkeypatch):
    """No over-read, however the rows are split into strips."""
    # the exact path: no skipped columns are read (see test_*_spans for merged rows)
    monkeypatch.setattr(tiff_backend, "SEPARATE_READ_BYTES", 0)
    ref, _ = _reference(files[name])
    with micro_reader.open(files[name]) as img:
        scene = img.images[0]
        assert scene.read_unit is None
        for key in _keys(ref.shape, seed=7):
            scene[key]
        assert scene.stats.over_read == 1.0


def test_uncompressed_single_strip_reads_only_the_rectangle(tmp_path, monkeypatch):
    """The case aszarr handles worst: one strip per plane."""
    # the exact path: no skipped columns are read (see test_*_spans for merged rows)
    monkeypatch.setattr(tiff_backend, "SEPARATE_READ_BYTES", 0)
    data = _data((2000, 3000))
    path = str(tmp_path / "a.tif")
    tifffile.imwrite(path, data)
    with tifffile.TiffFile(path) as tf:
        assert len(tf.pages[0].dataoffsets) == 1           # really one strip
    with micro_reader.open(path) as img:
        scene = img.images[0]
        np.testing.assert_array_equal(scene[1000:1064, 1500:1564], data[1000:1064, 1500:1564])
        assert scene.stats.pixels_read == 64 * 64


@pytest.mark.parametrize("name", ["single_strip", "strips_37", "big_endian", "rgb_contig",
                                  "rgb_separate", "imagej_tzcyx"])
def test_uncompressed_runs_split_into_pool_tasks(files, name, decode_threads, monkeypatch):
    """Strip runs are cut into pool tasks; tiny tasks put a cut in every few
    rows, on direct (native) and buffered (big endian, sample subset) reads."""
    from micro_reader.backends import tiff
    monkeypatch.setattr(tiff, "RAW_TASK_BYTES", 3000)
    monkeypatch.setattr(tiff, "SEPARATE_READ_BYTES", 0)   # the exact path
    ref, _ = _reference(files[name])
    with micro_reader.open(files[name]) as img:
        scene = img.images[0]
        for key in _keys(ref.shape, seed=5):
            np.testing.assert_array_equal(scene[key], ref[key], err_msg=f"{name} {key}")
        if name != "rgb_contig":                  # sample subsets over-read the samples
            assert scene.stats.over_read == 1.0


def test_compressed_read_unit_and_decode_cache(files):
    ref, _ = _reference(files["zlib_tiled"])
    with micro_reader.open(files["zlib_tiled"]) as img:
        scene = img.images[0]
        assert scene.read_unit == (64, 64)
        np.testing.assert_array_equal(scene[10:20, 10:20], ref[10:20, 10:20])
        assert scene.stats.pixels_read == 64 * 64          # one tile decoded
        np.testing.assert_array_equal(scene[30:40, 50:60], ref[30:40, 50:60])
        assert scene.stats.pixels_read == 64 * 64          # same tile: from the cache
        np.testing.assert_array_equal(scene[60:70, 60:70], ref[60:70, 60:70])
        assert scene.stats.pixels_read == 4 * 64 * 64      # touches 4 tiles, 1 cached


def test_rgb_sample_subset(files):
    ref, axes = _reference(files["rgb_contig"])
    assert axes == "yxs"
    with micro_reader.open(files["rgb_contig"]) as img:
        np.testing.assert_array_equal(img.images[0].read(s=1), ref[..., 1])
        np.testing.assert_array_equal(img.images[0].read(s=slice(0, 2), y=slice(5, 9)),
                                      ref[5:9, :, 0:2])


def test_multiple_series(tmp_path):
    a, b, c = _data((40, 50)), _data((3, 20, 30), "u1", seed=1), _data((4, 20, 30), seed=2)
    path = str(tmp_path / "multi.tif")
    with tifffile.TiffWriter(path) as tw:
        tw.write(a)
        # an RGB-planar page that the series metadata labels 'ZYX'
        tw.write(b, compression="zlib", photometric="rgb", planarconfig="separate",
                 metadata={"axes": "ZYX"})
        tw.write(c, photometric="minisblack", tile=(16, 16), metadata={"axes": "ZYX"})
    with micro_reader.open(path) as img:
        assert [s.axes for s in img.images] == ["yx", "zyx", "zyx"]
        for scene, data in zip(img.images, (a, b, c)):
            np.testing.assert_array_equal(scene[...], data)
            np.testing.assert_array_equal(scene[..., 3:17, 5:25], data[..., 3:17, 5:25])


def test_truncated_imagej_hyperstack(tmp_path):
    """ImageJ files > 4 GB list only their first IFD; the pages follow as one
    contiguous block (written here with truncate=True)."""
    data = _data((6, 40, 50))
    path = str(tmp_path / "trunc.tif")
    tifffile.imwrite(path, data, imagej=True, truncate=True, metadata={"axes": "ZYX"})
    with tifffile.TiffFile(path) as tf:
        assert len(tf.pages) == 1 and tf.series[0].is_truncated
    with micro_reader.open(path) as img:
        scene = img.images[0]
        assert scene.shape == data.shape
        np.testing.assert_array_equal(scene[...], data)
        np.testing.assert_array_equal(scene[4, 10:20, 5:45], data[4, 10:20, 5:45])


def test_sparse_tiles_read_as_fill(tmp_path):
    """Tiles with offset and byte count 0 are empty (fill value 0)."""
    path = str(tmp_path / "sparse.tif")
    data = _data((128, 128))
    tifffile.imwrite(path, data, tile=(64, 64))
    with tifffile.TiffFile(path) as tf:
        page = tf.pages[0]
        offsets_tag, counts_tag = page.tags["TileOffsets"], page.tags["TileByteCounts"]
        offsets = list(page.dataoffsets)
        counts = list(page.databytecounts)
    offsets[1] = counts[1] = 0
    tiff_types = {3: "<u2", 4: "<u4", 16: "<u8"}         # SHORT, LONG, LONG8
    with open(path, "r+b") as fh:
        for tag, values in ((offsets_tag, offsets), (counts_tag, counts)):
            fh.seek(tag.valueoffset)
            fh.write(np.asarray(values, tiff_types[int(tag.dtype)]).tobytes())
    expect = data.copy()
    expect[0:64, 64:128] = 0
    np.testing.assert_array_equal(tifffile.imread(path), expect)   # tifffile agrees
    with micro_reader.open(path) as img:
        np.testing.assert_array_equal(img.images[0][...], expect)
        np.testing.assert_array_equal(img.images[0][32:96, 32:96], expect[32:96, 32:96])


def test_volumetric_tiles_are_refused(tmp_path):
    path = str(tmp_path / "vol.tif")
    tifffile.imwrite(path, _data((8, 64, 64)), volumetric=True, tile=(4, 32, 32))
    with pytest.raises(UnsupportedFormatError, match="volumetric"):
        micro_reader.open(path)


def test_truncated_file_is_reported(tmp_path):
    path = str(tmp_path / "a.tif")
    tifffile.imwrite(path, _data((300, 500)), rowsperstrip=37)
    with open(path, "r+b") as fh:
        fh.truncate(os.path.getsize(path) // 2)
    with pytest.raises(micro_reader.MicroReaderError):
        with micro_reader.open(path) as img:
            img.images[0][...]


@pytest.mark.parametrize("compression", [None, "zlib"])
def test_close_releases_the_file(tmp_path, compression, decode_threads):
    """Windows cannot delete open files: closing must release every handle,
    including those the decode threads keep per scene."""
    path = str(tmp_path / "a.tif")
    tifffile.imwrite(path, _data((150, 160)), compression=compression, tile=(32, 32))
    img = micro_reader.open(path)
    img.images[0][0:100, 0:100]
    img.close()
    os.remove(path)
    with pytest.raises(ValueError):
        img.images[0][0]


def test_corrupt_compressed_segment_raises(tmp_path, decode_threads):
    """A decode error on a pool thread reaches the caller."""
    path = str(tmp_path / "bad.tif")
    data = _data((128, 128))
    tifffile.imwrite(path, data, compression="zlib", tile=(64, 64))
    with tifffile.TiffFile(path) as tf:
        offset = tf.pages[0].dataoffsets[3]                 # the last tile
    with open(path, "r+b") as fh:
        fh.seek(offset)
        fh.write(b"\x00" * 16)
    with micro_reader.open(path) as img:
        scene = img.images[0]
        np.testing.assert_array_equal(scene[0:64, 0:64], data[0:64, 0:64])   # intact tile
        with pytest.raises(Exception):
            scene[...]


@pytest.mark.parametrize("name", ["strips_37", "zlib_tiled", "ome_czyx_tiled", "lzw"])
def test_concurrent_reads(files, name, decode_threads):
    if name not in files:
        pytest.skip("needs imagecodecs")
    ref, _ = _reference(files[name])
    with micro_reader.open(files[name]) as img:
        scene, errors = img.images[0], []
        keys = _keys(ref.shape, n=40, seed=11)

        def worker(offset):
            try:
                for key in keys[offset::4]:
                    if not np.array_equal(scene[key], ref[key]):
                        errors.append(key)
            except Exception as exc:        # noqa: BLE001
                errors.append(exc)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert errors == []


@pytest.mark.parametrize("name", ["single_strip", "strips_37", "big_endian", "rgb_contig",
                                  "rgb_separate", "imagej_tzcyx", "float32"])
def test_uncompressed_narrow_reads_merge_rows(files, name):
    """Part of the columns of many rows: the skipped bytes per row cost less than one
    more read per row, so a run of rows is one read (as in LIF); pixels stay exact."""
    ref, _ = _reference(files[name])
    with micro_reader.open(files[name]) as img:
        scene = img.images[0]
        for key in _keys(ref.shape, seed=11):
            np.testing.assert_array_equal(scene[key], ref[key], err_msg=f"{name} {key}")


def test_uncompressed_single_strip_merged_span(tmp_path):
    """One read for the 64 rows: from the first row's columns to the last row's (the
    case that made a 768-column tile of a 2916-wide plane 8x slower, one read per row)."""
    data = _data((2000, 3000))
    path = str(tmp_path / "a.tif")
    tifffile.imwrite(path, data)
    with micro_reader.open(path) as img:
        scene = img.images[0]
        np.testing.assert_array_equal(scene[1000:1064, 1500:1564], data[1000:1064, 1500:1564])
        assert scene.stats.pixels_read == 63 * 3000 + 64
    big = _data((64, 40000))                                    # 80 KB skipped per row
    tifffile.imwrite(path, big)
    with micro_reader.open(path) as img:
        scene = img.images[0]
        np.testing.assert_array_equal(scene[:, 100:164], big[:, 100:164])
        assert scene.stats.pixels_read == 64 * 64              # rows read separately
