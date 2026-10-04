"""LIF backend on synthetic files (``_lif_writer``): every layout the reader
distinguishes, checked against the written arrays, and against readlif where
readlif can read the layout."""
from __future__ import annotations

import os
import threading

import numpy as np
import pytest

import micro_reader
from micro_reader import CorruptFileError, UnsupportedFormatError

from _lif_writer import folder, image, write_lif

RNG = np.random.default_rng(0)


def _ints(shape, dtype="u2", top=None):
    top = top or np.iinfo(dtype).max
    return RNG.integers(0, top, size=shape, endpoint=True).astype(dtype)


# name -> (elements, expected [(image name, axes, array)], write kwargs); each
# tile is its own image (docs/AXES.md)
def _cases():
    stack = _ints((2, 3, 4, 20, 30), top=4095)
    u8 = _ints((1, 2, 5, 17, 23), "u1")
    f32 = RNG.normal(size=(1, 1, 3, 16, 12)).astype("f4")
    f16 = RNG.normal(size=(1, 1, 2, 16, 12)).astype("f2")
    u32 = _ints((1, 1, 2, 9, 11), "u4")
    tl = _ints((3, 2, 2, 4, 10, 12))                                  # t c z l y x
    tiles = _ints((1, 1, 2, 3, 16, 20))                               # t c z m y x
    rgb = _ints((2, 3, 18, 25, 3), "u1")                              # t z y x s (RGB)
    padded = _ints((1, 2, 3, 15, 21))
    order = _ints((2, 3, 4, 12, 14))
    return {
        "stack_12bit": ([image("stack", stack, resolution=12)], [("stack", "tczyx", stack)], {}),
        "uint8": ([image("u8", u8)], [("u8", "tczyx", u8)], {}),
        "float32": ([image("f32", f32)], [("f32", "tczyx", f32)], {}),
        "float16_flim": ([image("Fast Flim", f16)], [("Fast Flim", "tczyx", f16)], {}),
        "uint32": ([image("mask", u32)], [("mask", "tczyx", u32)], {}),
        # a lambda scan: one image per emission band (index "emission")
        "time_lambda": ([image("lambda", tl, "tczlyx")],
                        [("lambda", "tczyx", tl[:, :, :, k]) for k in range(4)], {}),
        "tiles": ([image("scan", tiles, "tczmyx",
                         tiles=[(0, 0, 0.0, 0.0), (1, 0, 1e-4, 0.0), (2, 0, 2e-4, 0.0)])],
                  [("scan", "tczyx", tiles[:, :, :, m]) for m in range(3)], {}),
        "rgb": ([image("colour", rgb, "tzyxs", rgb=True)],
                [("colour", "tczyxs", rgb[:, None])], {}),
        "padded_rows": ([image("pad", padded, row_pad=6)], [("pad", "tczyx", padded)], {}),
        "z_before_c": ([image("zc", order, order="xyzct")], [("zc", "tczyx", order)], {}),
        "version1": ([image("v1", u8)], [("v1", "tczyx", u8)], {"version": 1}),
        "nested": ([image("a", u8), folder("group", [image("b", padded),
                                                     folder("empty"),
                                                     image("c", u8[:, :1])])],
                   [("a", "tczyx", u8), ("group/b", "tczyx", padded),
                    ("group/c", "tczyx", u8[:, :1])], {}),
    }


CASES = _cases()


def _keys(shape, n=20, seed=3):
    rng = np.random.default_rng(seed)
    keys = [(Ellipsis,)]
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


def _write(tmp_path, name):
    elements, expected, kwargs = CASES[name]
    path = str(tmp_path / f"{name}.lif")
    write_lif(path, elements, **kwargs)
    return path, expected


@pytest.mark.parametrize("name", list(CASES))
def test_matches_written_data(tmp_path, name, decode_threads):
    path, expected = _write(tmp_path, name)
    with micro_reader.open(path) as img:
        assert img.backend == "lif"
        assert [(s.name, s.axes) for s in img.images] == [(n, a) for n, a, _ in expected]
        for scene, (_, _, ref) in zip(img.images, expected):
            assert scene.shape == ref.shape and scene.dtype == ref.dtype
            for key in _keys(ref.shape):
                got = scene[key]
                assert got.dtype.isnative
                np.testing.assert_array_equal(got, ref[key], err_msg=f"{name} {key}")


@pytest.mark.parametrize("name", ["stack_12bit", "uint8", "tiles", "z_before_c", "version1",
                                  "nested", "float16_flim"])
def test_agrees_with_readlif(tmp_path, name):
    """readlif (eubi-bridge's LIF path, via bioio-lif) reads the same planes;
    it returns half floats as their uint16 bits.  Not compared: lambda
    (readlif has no lambda axis and returns the wrong planes) and padded
    rows (readlif ignores the row stride) -- both readlif limitations,
    checked against Bio-Formats with harness/conformance_lif.py."""
    readlif = pytest.importorskip("readlif.reader")
    path, expected = _write(tmp_path, name)
    lf = readlif.LifFile(path)
    with micro_reader.open(path) as img:
        assert list(dict.fromkeys(s.name for s in img.images)) == \
            [i["name"] for i in lf.image_list]
        for scene in img.images:
            i, m = scene.index["scene"], scene.index.get("tile")
            for t in range(scene.shape[0]):
                for c in range(scene.shape[1]):
                    for z in range(scene.shape[2]):
                        kw = {"m": m} if m is not None else {}
                        ref = np.asarray(lf.get_image(i).get_frame(z=z, t=t, c=c, **kw))
                        got = scene.read(t=t, c=c, z=z)
                        if got.dtype == np.float16:
                            got = got.view(np.uint16)
                        np.testing.assert_array_equal(got, ref, err_msg=f"{name} {i} {t} {c} {z}")


def test_rgb_samples_follow_the_channel_tags(tmp_path):
    """Colour stored B, G, R (tags 3, 2, 1) comes out R, G, B; stored R, G, B
    (tags 1, 2, 3) too."""
    rgb = _ints((1, 1, 6, 7, 3), "u1")
    for stored_bgr in (True, False):
        path = str(tmp_path / f"rgb_{stored_bgr}.lif")
        el = image("c", rgb, "tzyxs", rgb=stored_bgr,
                   tags=None if stored_bgr else [1, 2, 3])
        write_lif(path, [el])
        with micro_reader.open(path) as img:
            np.testing.assert_array_equal(img.images[0][0, 0, 0], rgb[0, 0])     # rgb: t z y x s
            np.testing.assert_array_equal(img.images[0][0, 0, 0, :, :, 2], rgb[0, 0, :, :, 2])


def test_tiles_are_images_with_positions(tmp_path):
    path, _ = _write(tmp_path, "tiles")
    with micro_reader.open(path) as img:
        assert img.indices == [{"scene": 0, "tile": m} for m in range(3)]
        with pytest.warns(DeprecationWarning, match="metadata.position"):
            assert [s.tile_position for s in img.images] == [(0, 0, 0.0, 0.0), (1, 0, 1e-4, 0.0),
                                                             (2, 0, 2e-4, 0.0)]
        # the replacement: each tile's position, along the image's axes (um)
        assert [round(s.metadata.position["x"].value) for s in img.images] == [0, 100, 200]
    with pytest.raises(UnsupportedFormatError, match="does not compose Leica tiles"):
        micro_reader.open(path, tiles="stitched")


def test_untiled_images_of_a_tiled_file_get_tile_0(tmp_path):
    tiles = _ints((1, 1, 1, 2, 8, 9))
    u8 = _ints((1, 1, 1, 8, 9), "u1")
    path = str(tmp_path / "mixed.lif")
    write_lif(path, [image("plain", u8), image("scan", tiles, "tczmyx")])
    with micro_reader.open(path) as img:
        assert img.indices == [{"scene": 0, "tile": 0}, {"scene": 1, "tile": 0},
                                  {"scene": 1, "tile": 1}]
        assert img.index_range == {"scene": 2, "tile": 2}
        np.testing.assert_array_equal(img.select(scene=1, tile=1)[0][...], tiles[:, :, :, 1])


def _patch(path, old, new):
    data = open(path, "rb").read()
    old, new = old.encode("utf-16-le"), new.encode("utf-16-le")
    assert old in data
    with open(path, "wb") as fh:
        fh.write(data.replace(old, new))


def test_rotations_are_images(tmp_path):
    """DimID 6 (rotation): one image per rotation, like tiles."""
    path, expected = _write(tmp_path, "tiles")
    _patch(path, 'DimID="10"', 'DimID="6" ')                     # same length
    with micro_reader.open(path) as img:
        assert img.indices == [{"scene": 0, "rotation": r} for r in range(3)]
        for r, image_ in enumerate(img.images):
            np.testing.assert_array_equal(image_[...], expected[r][2])


def test_lambda_scans_are_emission_and_excitation_indices(tmp_path):
    path, expected = _write(tmp_path, "time_lambda")
    with micro_reader.open(path) as img:
        assert img.indices == [{"scene": 0, "emission": k} for k in range(4)]
    _patch(path, 'DimID="5"', 'DimID="9"')
    with micro_reader.open(path) as img:
        assert img.indices == [{"scene": 0, "excitation": k} for k in range(4)]
        for k, image_ in enumerate(img.images):
            assert image_.axes == "tczyx"
            np.testing.assert_array_equal(image_[...], expected[k][2])


def test_reads_only_the_rows_asked(tmp_path):
    """A window is one contiguous read from its first to its last pixel."""
    path, expected = _write(tmp_path, "stack_12bit")
    with micro_reader.open(path) as img:
        scene = img.images[0]
        np.testing.assert_array_equal(scene[1, 2, 3, 5:9, 4:10], expected[0][2][1, 2, 3, 5:9, 4:10])
        assert scene.stats.pixels_read == 3 * 30 + 6       # rows 5..8, columns 4..9


def test_unsupported_dimension_is_named(tmp_path):
    path, _ = _write(tmp_path, "uint8")
    data = open(path, "rb").read()
    text = 'DimID="3" NumberOfElements="5"'.encode("utf-16-le")
    assert text in data
    patched = data.replace(text, 'DimID="7" NumberOfElements="5"'.encode("utf-16-le"))
    with open(path, "wb") as fh:
        fh.write(patched)
    with pytest.raises(UnsupportedFormatError, match="XT slice"):
        micro_reader.open(path)


def test_bit_packed_samples_are_refused(tmp_path):
    """A non-zero BitInc means samples not aligned to bytes: refused, never
    read as if aligned."""
    path, _ = _write(tmp_path, "uint8")
    data = open(path, "rb").read()
    text = 'BitInc="0"'.encode("utf-16-le")
    assert text in data
    with open(path, "wb") as fh:
        fh.write(data.replace(text, 'BitInc="4"'.encode("utf-16-le"), 1))
    with pytest.raises(UnsupportedFormatError, match="bit-packed"):
        micro_reader.open(path)


def test_truncated_file_is_reported(tmp_path):
    path, _ = _write(tmp_path, "stack_12bit")
    with open(path, "r+b") as fh:
        fh.truncate(os.path.getsize(path) - 100)
    with pytest.raises(CorruptFileError, match="truncated"):
        micro_reader.open(path)


def test_close_releases_the_file(tmp_path, decode_threads):
    path, _ = _write(tmp_path, "stack_12bit")
    img = micro_reader.open(path)
    img.images[0][...]
    img.close()
    os.remove(path)                                        # Windows: fails if still open
    with pytest.raises(ValueError):
        img.images[0][0]


def test_concurrent_reads(tmp_path, decode_threads):
    path, expected = _write(tmp_path, "time_lambda")
    ref = expected[0][2]
    errors = []
    with micro_reader.open(path) as img:
        scene = img.images[0]

        def work(seed):
            for key in _keys(ref.shape, n=15, seed=seed):
                if not np.array_equal(scene[key], ref[key]):
                    errors.append(key)
        threads = [threading.Thread(target=work, args=(s,)) for s in range(6)]
        for th in threads:
            th.start()
        for th in threads:
            th.join()
    assert not errors
