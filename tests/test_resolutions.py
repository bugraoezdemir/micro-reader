"""Resolution levels (docs/AXES.md): an index ``resolution``, innermost.

Full resolution only by default; ``open(resolutions="all")`` lists every
level as an image of its own; ``image.resolutions`` gives one image's
pyramid either way, and ``image.is_pyramidal`` says whether it has levels.
Every index has ``resolution`` (0 = full) when the file stores a pyramid,
none when it does not -- in both modes.  Imaris, TIFF and CZI here.
"""
from __future__ import annotations

import numpy as np
import pytest

import micro_reader

RNG = np.random.default_rng(9)


# -- TIFF family: OME-TIFF pyramid in SubIFDs -----------------------------------------

def _pyramid_tiff(path, data, levels=2, tile=(32, 32)):
    tifffile = pytest.importorskip("tifffile")
    with tifffile.TiffWriter(path, ome=True) as tw:
        tw.write(data, subifds=levels, metadata={"axes": "CYX"}, photometric="minisblack",
                 tile=tile)
        for k in range(1, levels + 1):
            tw.write(data[:, ::2 ** k, ::2 ** k], subfiletype=1, photometric="minisblack",
                     tile=tile)


def test_tiff_full_resolution_by_default_with_its_pyramid(tmp_path):
    data = RNG.integers(0, 4000, (2, 96, 80)).astype("u2")
    path = str(tmp_path / "pyr.ome.tif")
    _pyramid_tiff(path, data)
    with micro_reader.open(path) as f:
        (image,) = f.images
        # every level of the pyramid has its resolution, full resolution too
        assert image.is_pyramidal
        assert image.index == {"scene": 0, "resolution": 0}
        assert f.index_range == {"scene": 1, "resolution": 1}      # one level listed
        levels = image.resolutions
        assert [lv.shape for lv in levels] == [(2, 96, 80), (2, 48, 40), (2, 24, 20)]
        assert levels[0] is image
        assert [lv.index for lv in levels] == [{"scene": 0, "resolution": k} for k in range(3)]
        assert all(lv.is_pyramidal for lv in levels)
        for k, lv in enumerate(levels):
            np.testing.assert_array_equal(lv[...], data[:, ::2 ** k, ::2 ** k])
        np.testing.assert_array_equal(levels[2][1, 3:9, 4:15], data[1, ::4, ::4][3:9, 4:15])


def test_tiff_all_levels_listed_as_images(tmp_path):
    data = RNG.integers(0, 4000, (2, 96, 80)).astype("u2")
    path = str(tmp_path / "pyr.ome.tif")
    _pyramid_tiff(path, data)
    with micro_reader.open(path, resolutions="all") as f:
        assert f.indices == [{"scene": 0, "resolution": k} for k in range(3)]
        assert f.index_range == {"scene": 1, "resolution": 3}
        (level1,) = f.select(resolution=1)
        np.testing.assert_array_equal(level1[...], data[:, ::2, ::2])
        assert f.images[0].resolutions == f.images             # one pyramid, the same objects


def test_files_without_levels_are_their_own_pyramid(tmp_path):
    tifffile = pytest.importorskip("tifffile")
    path = str(tmp_path / "flat.tif")
    tifffile.imwrite(path, np.zeros((8, 9), "u1"))
    for policy in ("base", "all"):
        with micro_reader.open(path, resolutions=policy) as f:
            (image,) = f.images
            assert image.resolutions == [image] and not image.is_pyramidal
            assert image.index == {"scene": 0}           # no pyramid in the file: no resolution


def test_axes_images_carry_the_pyramid(tmp_path):
    data = RNG.integers(0, 4000, (2, 96, 80)).astype("u2")
    path = str(tmp_path / "pyr.ome.tif")
    _pyramid_tiff(path, data)
    with micro_reader.open(path, axes="canonical") as f:
        levels = f.images[0].resolutions
        assert [lv.axes for lv in levels] == ["tczyx"] * 3
        assert levels[2].shape == (1, 2, 1, 24, 20)
        np.testing.assert_array_equal(levels[1][0, :, 0], data[:, ::2, ::2])


def test_bad_resolutions_policy(tmp_path):
    tifffile = pytest.importorskip("tifffile")
    path = str(tmp_path / "flat.tif")
    tifffile.imwrite(path, np.zeros((8, 9), "u1"))
    with pytest.raises(ValueError, match="resolutions="):
        micro_reader.open(path, resolutions="levels")


# -- Imaris: ResolutionLevel r ---------------------------------------------------------

def _pyramid_ims(path, data):
    """Level 0 by test_ims.write_ims (padded chunks, real size in attributes),
    then level 1 at half size, also padded."""
    h5py = pytest.importorskip("h5py")
    from test_ims import _char
    n_t, n_c, nz, ny, nx = data.shape
    with h5py.File(path, "w") as f:
        for r, d in enumerate((data, data[:, :, :, ::2, ::2])):
            real = d.shape[2:]
            for t in range(n_t):
                for c in range(n_c):
                    grp = f.require_group(f"DataSet/ResolutionLevel {r}/TimePoint {t}/Channel {c}")
                    padded = tuple(-(-n // 8) * 8 for n in real)
                    ds = grp.create_dataset("Data", shape=padded, dtype=d.dtype,
                                            chunks=(4, 8, 8), fillvalue=0)
                    ds[:real[0], :real[1], :real[2]] = d[t, c]
                    for axis, size in zip("ZYX", real):
                        grp.attrs[f"ImageSize{axis}"] = _char(size)
        img = f.require_group("DataSetInfo/Image")
        for axis, size in zip("ZYX", data.shape[2:]):
            img.attrs[axis] = _char(size)


def test_imaris_levels(tmp_path):
    data = RNG.integers(0, 4000, (1, 2, 5, 21, 27)).astype("u2")
    path = str(tmp_path / "pyr.ims")
    _pyramid_ims(path, data)
    with micro_reader.open(path) as f:
        (image,) = f.images
        assert image.is_pyramidal and image.index == {"scene": 0, "resolution": 0}
        levels = image.resolutions
        assert [lv.shape for lv in levels] == [(1, 2, 5, 21, 27), (1, 2, 5, 11, 14)]
        np.testing.assert_array_equal(levels[1][...], data[:, :, :, ::2, ::2])
    with micro_reader.open(path, resolutions="all") as f:
        assert f.indices == [{"scene": 0, "resolution": 0}, {"scene": 0, "resolution": 1}]
        np.testing.assert_array_equal(f.select(resolution=1)[0][...], data[:, :, :, ::2, ::2])


# -- CZI: pyramid subblocks of whole scenes ----------------------------------------------

def _pyramid_czi(path):
    """One scene of 2 x 2 tiles (32 x 32, at x0=100, y0=40); the pyramid as
    two halves at zoom 0.5 (32 x 64 stored each) and one block at 0.25.
    Returns the full scene and each level's expected pixels."""
    from _czi_writer import raw_payload, write_czi
    full = RNG.integers(0, 4000, (64, 64)).astype("u2")
    half = RNG.integers(0, 4000, (32, 32)).astype("u2")         # the stored pyramid data
    quarter = RNG.integers(0, 4000, (16, 16)).astype("u2")
    tiles = []
    for m, (ty, tx) in enumerate([(0, 0), (0, 32), (32, 0), (32, 32)]):
        tiles.append(dict(scene=0, mosaic=m, plane={"C": 0}, x=100 + tx, y=40 + ty,
                          height=32, width=32, dtype="u2", compression=0,
                          payload=raw_payload(full[ty:ty + 32, tx:tx + 32])))
    for k, x in enumerate((0, 32)):                         # zoom 0.5 in two halves
        tiles.append(dict(scene=0, plane={"C": 0}, x=100 + x, y=40, height=64, width=32,
                          stored=(16, 32), pyramid=2, dtype="u2", compression=0,
                          payload=raw_payload(half[:, 16 * k:16 * k + 16])))
    tiles.append(dict(scene=0, plane={"C": 0}, x=100, y=40, height=64, width=64,
                      stored=(16, 16), pyramid=2, dtype="u2", compression=0,
                      payload=raw_payload(quarter)))
    write_czi(path, tiles)
    return full, half, quarter


def test_czi_mosaic_scene_has_its_stored_pyramid(tmp_path):
    path = str(tmp_path / "pyr.czi")
    full, half, quarter = _pyramid_czi(path)
    with micro_reader.open(path, tiles="mosaic") as f:
        (image,) = f.images
        assert image.is_pyramidal and image.index == {"scene": 0, "resolution": 0}
        levels = image.resolutions
        assert [lv.shape[-2:] for lv in levels] == [(64, 64), (32, 32), (16, 16)]
        assert [lv.zoom for lv in levels[1:]] == [0.5, 0.25]
        np.testing.assert_array_equal(levels[0][0, 0, 0], full)
        np.testing.assert_array_equal(levels[1][0, 0, 0], half)       # both halves in place
        np.testing.assert_array_equal(levels[2][0, 0, 0], quarter)
        np.testing.assert_array_equal(levels[1][0, 0, 0, 5:20, 10:25], half[5:20, 10:25])
        assert levels[1].origin == (50, 20)                      # the scene corner x zoom
    with micro_reader.open(path, tiles="mosaic", resolutions="all") as f:
        assert f.indices == [{"scene": 0, "resolution": k} for k in range(3)]


def test_czi_tiles_have_no_levels(tmp_path):
    """CZI stores pyramids of whole scenes (no tile index on pyramid subblocks)."""
    path = str(tmp_path / "pyr.czi")
    _pyramid_czi(path)
    with micro_reader.open(path, resolutions="all") as f:              # separate tiles
        assert len(f.images) == 4
        # no tile is pyramidal, so no index has a resolution
        assert all("resolution" not in k for k in f.indices)
        assert all(image.resolutions == [image] and not image.is_pyramidal
                   for image in f.images)
