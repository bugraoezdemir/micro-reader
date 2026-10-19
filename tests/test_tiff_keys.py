"""TIFF axes and image indices (docs/AXES.md): T C Z Y X S are array axes in
the series' own order; every other tifffile letter -- angles, tiles, LSM
positions, phase, lifetime, lambda, unknown -- makes separate images."""
from __future__ import annotations

import numpy as np
import pytest

import micro_reader
from micro_reader import UnsupportedFormatError
from micro_reader.backends.tiff import translate_axes

tifffile = pytest.importorskip("tifffile")
RNG = np.random.default_rng(7)


def _write(path, data, axes):
    tifffile.imwrite(path, data, metadata={"axes": axes}, photometric="minisblack")
    with tifffile.TiffFile(path) as tif:                    # the writer keeps the letters
        # (tifffile 2026.6+ leaves out letters of size 1: AZYX with A = 1 reads as ZYX)
        assert tif.series[0].axes in (axes, "".join(a for a, n in zip(axes, data.shape) if n > 1))


def test_lifetime_and_lambda_are_indices(tmp_path):
    data = RNG.integers(0, 4000, (3, 2, 4, 9, 11)).astype("u2")   # H E Z Y X
    path = str(tmp_path / "flim.tif")
    _write(path, data, "HEZYX")
    with micro_reader.open(path) as f:
        assert f.index_range == {"scene": 1, "emission": 2, "lifetime": 3}
        assert f.indices[:2] == [{"scene": 0, "emission": 0, "lifetime": 0},
                                 {"scene": 0, "emission": 0, "lifetime": 1}]
        for image in f.images:
            assert image.axes == "zyx"
            k, e = image.index["lifetime"], image.index["emission"]
            np.testing.assert_array_equal(image[...], data[k, e])
        assert f.images[0].as_axes("canonical").axes == "tczyx"


def test_angles_are_views(tmp_path):
    data = RNG.integers(0, 4000, (2, 3, 5, 9, 11)).astype("u2")   # A T Z Y X
    path = str(tmp_path / "angles.tif")
    _write(path, data, "ATZYX")
    with micro_reader.open(path) as f:
        assert f.indices == [{"scene": 0, "view": 0}, {"scene": 0, "view": 1}]
        for v, image in enumerate(f.images):
            assert image.axes == "tzyx"
            np.testing.assert_array_equal(image[...], data[v])
            np.testing.assert_array_equal(image[1, 2:4, 3:7, 5:9], data[v, 1, 2:4, 3:7, 5:9])


def test_tile_grid_is_one_tile_level(tmp_path):
    data = RNG.integers(0, 4000, (2, 4, 9, 11)).astype("u2")      # J K Y X (a 3 would be RGB)
    path = str(tmp_path / "grid.tif")
    _write(path, data, "JKYX")
    with micro_reader.open(path) as f:
        assert f.indices == [{"scene": 0, "tile": m} for m in range(8)]
        for image in f.images:
            j, k = divmod(image.index["tile"], 4)
            np.testing.assert_array_equal(image[...], data[j, k])
    with pytest.raises(UnsupportedFormatError, match="does not compose TIFF tiles"):
        micro_reader.open(path, tiles="mosaic")


def test_untiled_series_of_a_tiled_file_get_tile_0(tmp_path):
    tiles = RNG.integers(0, 4000, (2, 9, 11)).astype("u2")
    plain = RNG.integers(0, 4000, (9, 11)).astype("u2")
    path = str(tmp_path / "two.tif")
    with tifffile.TiffWriter(path) as tw:
        tw.write(tiles, metadata={"axes": "RYX"}, photometric="minisblack")
        tw.write(plain, metadata={"axes": "YX"}, photometric="minisblack")
    with micro_reader.open(path) as f:
        assert f.indices == [{"scene": 0, "tile": 0}, {"scene": 0, "tile": 1},
                                {"scene": 1, "tile": 0}]
        np.testing.assert_array_equal(f.select(scene=0, tile=1)[0][...], tiles[1])
        np.testing.assert_array_equal(f.select(scene=1)[0][...], plain)


def test_single_angle_is_not_a_level(tmp_path):
    data = RNG.integers(0, 4000, (1, 4, 9, 11)).astype("u2")
    path = str(tmp_path / "one.tif")
    _write(path, data, "AZYX")
    with micro_reader.open(path) as f:
        assert f.indices == [{"scene": 0}] and f.images[0].axes == "zyx"


def test_two_unknown_axes_are_one_sequence(tmp_path):
    """Several letters of one index are counted as one (raveled)."""
    data = RNG.integers(0, 255, (2, 3, 4, 5)).astype("u1")
    path = str(tmp_path / "iq.tif")
    _write(path, data, "IQYX")
    with micro_reader.open(path) as f:
        assert f.index_range == {"scene": 1, "sequence": 6}
        np.testing.assert_array_equal(f.select(sequence=4)[0][...], data[1, 1])


@pytest.mark.parametrize("axes,kw,letters,levels", [
    ("PTZCYX", dict(lsm=True), [None, "t", "z", "c", "y", "x"], ["scene"] + [None] * 5),
    ("PZYX", {}, [None, "z", "y", "x"], ["phase"] + [None] * 3),        # OME phase
    ("IYX", dict(eer=True), ["t", "y", "x"], [None] * 3),               # EER frames
    ("IYX", {}, [None, "y", "x"], ["sequence", None, None]),
    ("IYX", dict(rename={"sequence": "z"}), ["z", "y", "x"], [None] * 3),
    ("RMYXS", {}, [None, None, "y", "x", "s"], ["tile", "tile", None, None, None]),
    ("IZYX", dict(rename={"sequence": "z"}), None, None),               # z twice
])
def test_translate_axes(axes, kw, letters, levels):
    if letters is None:
        with pytest.raises(UnsupportedFormatError):
            translate_axes(axes, **kw)
        return
    assert translate_axes(axes, **kw) == (letters, levels)


def test_closing_one_image_leaves_its_siblings_readable(tmp_path):
    data = RNG.integers(0, 4000, (2, 9, 11)).astype("u2")
    path = str(tmp_path / "angles.tif")
    _write(path, data, "AYX")
    f = micro_reader.open(path)
    f.images[0].close()
    np.testing.assert_array_equal(f.images[1][...], data[1])
    with pytest.raises(ValueError, match="closed"):
        f.images[0][...]
    f.close()
