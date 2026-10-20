"""Olympus / Evident VSI backend on synthetic sets (the OME cellSens samples
were compared with Bio-Formats: docs/PLAN.md, "Olympus / Evident VSI")."""
from __future__ import annotations

import os
import types

import numpy as np
import pytest

pytest.importorskip("tifffile")

import micro_reader  # noqa: E402
from micro_reader import (CompositeWarning, CorruptFileError, UnsupportedFormatError)  # noqa: E402
from micro_reader.backends import vsi  # noqa: E402

from _vsi_writer import (C, LAMBDA, T, Z, channel, image_volume, write_ets,  # noqa: E402
                         write_vsi)


def _levels(shape, n, dtype=np.uint16, seed=0, rgb=False):
    """*n* pyramid levels of *shape* (extra..., rows, cols), each halved and
    rounded up, of random values."""
    rng = np.random.default_rng(seed)
    out = []
    *extra, rows, cols = shape
    for k in range(n):
        size = tuple(extra) + (-(-rows // (1 << k)), -(-cols // (1 << k))) + ((3,) if rgb else ())
        out.append(rng.integers(0, 250 if dtype == np.uint8 else 4000, size=size, dtype=dtype))
    return out


def _macro(rows=12, cols=30):
    return np.arange(rows * cols, dtype=np.uint8).reshape(rows, cols)


def _simple(tmp_path, levels, **volume):
    rows, cols = levels[0].shape[-2:]
    return write_vsi(tmp_path, "a", [image_volume(7, (rows, cols), name="scan", **volume)],
                     {7: {"levels": levels}}, macro=_macro())


def test_a_2d_pyramid(tmp_path):
    levels = _levels((45, 70), 3)
    path = _simple(tmp_path, levels)
    with micro_reader.open(path) as f:
        assert f.backend == "vsi" and f.format == "vsi"
        assert [im.name for im in f.images] == ["scan", "macro"]
        im = f.images[0]
        assert im.axes == "tczyx" and im.shape == (1, 1, 1, 45, 70) and im.dtype == np.uint16
        assert [r.shape[-2:] for r in im.resolutions] == [(45, 70), (23, 35), (12, 18)]
        for level, data in zip(im.resolutions, levels):
            np.testing.assert_array_equal(level[0, 0, 0], data)
        np.testing.assert_array_equal(im[0, 0, 0, 7:40, 13:66], levels[0][7:40, 13:66])
        np.testing.assert_array_equal(im.resolutions[2][0, 0, 0, 11:, 17:], levels[2][11:, 17:])
        sizes = im.metadata.pixel_sizes
        assert sizes["x"].value == pytest.approx(0.5) and sizes["y"].value == pytest.approx(0.25)
        assert sizes["x"].unit == "micrometer"
        macro = f.images[1]
        assert macro.index["series"] == 1
        np.testing.assert_array_equal(np.asarray(macro[...]).reshape(12, 30), _macro())
        assert not macro.metadata.pixel_sizes


def test_z_channels_and_time(tmp_path):
    """Tile coordinates (x, y, c, t, z, level), steps and per-channel properties."""
    levels = _levels((3, 2, 4, 21, 37), 2)                 # c, t, z, rows, cols
    chans = [channel("DAPI", 455, (0, 0, 255)), channel("FITC", 518, (0, 255, 0)),
             channel(None, None, None)]
    vol = image_volume(3, (21, 37), name="fl", dims=[(C, None, None), (T, 2.5, "10^-3s^1"),
                                                     (Z, -1.5, "10^-6m^1")], channels=chans)
    path = write_vsi(tmp_path, "a", [vol], {3: {"levels": levels, "tile": (8, 16)}})
    with micro_reader.open(path) as f:
        im = f.images[0]
        assert [i.name for i in f.images] == ["fl"]               # no macro page
        assert im.shape == (2, 3, 4, 21, 37)
        expected = levels[0].transpose(1, 0, 2, 3, 4)            # -> t, c, z
        np.testing.assert_array_equal(im[...], expected)
        np.testing.assert_array_equal(im[1, 2, 1:3, 5:19, 3:30], expected[1, 2, 1:3, 5:19, 3:30])
        np.testing.assert_array_equal(im.resolutions[1][...], levels[1].transpose(1, 0, 2, 3, 4))
        md = im.metadata
        assert md.pixel_sizes["z"].value == pytest.approx(1.5)
        assert md.pixel_sizes["t"].value == pytest.approx(0.0025)
        assert md.pixel_sizes["t"].unit == "second"
        assert [c.name for c in md.channels] == ["DAPI", "FITC", None]
        assert [c.color for c in md.channels] == ["#0000FF", "#00FF00", None]
        assert md.channels[0].emission_wavelength.value == pytest.approx(455)
        assert md.channels[2].emission_wavelength is None


def test_a_single_channel_takes_the_stacks_name(tmp_path):
    levels = _levels((20, 30), 1)
    path = _simple(tmp_path, levels, stack_channel="SR488")
    with micro_reader.open(path) as f:
        assert [c.name for c in f.images[0].metadata.channels] == ["SR488"]


def test_rgb_png_tiles(tmp_path):
    levels = _levels((33, 50), 2, dtype=np.uint8, rgb=True)
    path = write_vsi(tmp_path, "a", [image_volume(1, (33, 50), name="bf")],
                     {1: {"levels": levels, "compression": 8, "rgb": True}})
    with micro_reader.open(path) as f:
        im = f.images[0]
        assert im.axes == "tczyxs" and im.shape == (1, 1, 1, 33, 50, 3)
        np.testing.assert_array_equal(im[0, 0, 0], levels[0])
        np.testing.assert_array_equal(im.resolutions[1][0, 0, 0, :, :, 1:], levels[1][..., 1:])


def test_a_missing_tile_reads_as_zeros(tmp_path):
    levels = _levels((40, 40), 1)
    rows, cols = 40, 40
    path = write_vsi(tmp_path, "a", [image_volume(1, (rows, cols))],
                     {1: {"levels": levels, "missing": {(0, 1, 1)}}})
    expected = levels[0].copy()
    expected[16:32, 16:32] = 0
    with micro_reader.open(path) as f:
        np.testing.assert_array_equal(f.images[0][0, 0, 0], expected)


def test_series_follow_bio_formats(tmp_path):
    """Stack folders in number order, label / overview named by stack type,
    volumes without an ETS file (focus maps) left out, the macro last."""
    lab, ov, scan = _levels((20, 20), 1, seed=1), _levels((30, 50), 2, seed=2), \
        _levels((40, 60), 3, seed=3)
    volumes = [image_volume(1, (20, 20), name="Label", stack_type=256),
               image_volume(10000, (30, 50), name="Overview", stack_type=1),
               image_volume(10001, None, name="Sample Mask", stack_type=8388608),
               image_volume(10002, (40, 60), name="20x_BF_01")]
    path = write_vsi(tmp_path, "a", volumes, {10002: {"levels": scan}, 1: {"levels": lab},
                                               10000: {"levels": ov}}, macro=_macro())
    with micro_reader.open(path) as f:
        assert [im.name for im in f.images] == ["label", "overview", "20x_BF_01", "macro"]
        assert [len(im.resolutions) for im in f.images] == [1, 2, 3, 1]
        for im, data in zip(f.images, (lab, ov, scan)):
            np.testing.assert_array_equal(im[0, 0, 0], data[0])


def test_volumes_pair_by_stack_id_then_by_size(tmp_path):
    a, b = _levels((20, 30), 1, seed=1), _levels((20, 30), 1, seed=2)
    c = _levels((50, 50), 1, seed=3)
    volumes = [image_volume(4, (20, 30), name="four"), image_volume(9, (20, 30), name="nine"),
               image_volume(77, (50, 50), name="big")]
    # folder 9 names its volume; folder 5 takes the first fitting unclaimed one
    path = write_vsi(tmp_path, "a", volumes, {9: {"levels": b}, 5: {"levels": a},
                                               6: {"levels": c}})
    with micro_reader.open(path) as f:
        assert [im.name for im in f.images] == ["four", "big", "nine"]
        np.testing.assert_array_equal(f.images[2][0, 0, 0], b[0])


def test_an_ets_matching_no_volume_is_left_out(tmp_path):
    levels = _levels((20, 30), 1)
    path = write_vsi(tmp_path, "a", [image_volume(1, (20, 30), name="x")],
                     {1: {"levels": levels}, 2: {"levels": _levels((90, 90), 1)}})
    with pytest.warns(CompositeWarning, match="matches no image"):
        f = micro_reader.open(path)
    with f:
        assert [im.name for im in f.images] == ["x"]


def test_unsupported_dimensions_are_refused(tmp_path):
    vol = image_volume(1, (20, 30), dims=[(LAMBDA, None, None)])
    path = write_vsi(tmp_path, "a", [vol], {1: {"levels": _levels((2, 20, 30), 1)}})
    with pytest.raises(UnsupportedFormatError, match="lambda dimension of 2"):
        micro_reader.open(path)
    vol = image_volume(1, (20, 30))
    path = write_vsi(tmp_path, "b", [vol], {1: {"levels": _levels((2, 20, 30), 1)}})
    with pytest.raises(UnsupportedFormatError, match="names no dimension"):
        micro_reader.open(path)


def test_fit_slots_follows_slideio():
    def fit(slots, ndim, pyramid=True):
        return vsi._Volume.fit_slots(types.SimpleNamespace(slots=slots), ndim, pyramid)
    assert fit({"z": 2, "c": 3}, 5) == {"z": 2, "c": 3}
    assert fit({"c": 3}, 4) == {"c": 2}                          # past the last: shifted down
    assert fit({"c": 7}, 4) == {}                                # still outside: dropped
    assert fit({"z": 4}, 5) == {"c": 2, "t": 3}                  # z on the level slot
    assert fit({"c": 2}, 6) == {"c": 2, "t": 4, "z": 3}


# -- the parts of a VSI (docs/COMPOSITE.md) ------------------------------------------------

def test_any_part_opens_the_vsi(tmp_path):
    levels = _levels((20, 30), 2)
    path = _simple(tmp_path, levels)
    folder = os.path.join(str(tmp_path), "_a_")
    ets = os.path.join(folder, "stack7", "frame_t.ets")
    for part in (folder, os.path.join(folder, "stack7"), ets, str(tmp_path)):
        with micro_reader.open(part) as f:
            assert f.backend == "vsi" and os.path.samefile(f.path, path)
            assert [im.name for im in f.images] == ["scan", "macro"]
            np.testing.assert_array_equal(f.images[0][0, 0, 0], levels[0])


def test_an_ets_without_its_vsi_is_read_alone(tmp_path):
    levels = _levels((20, 30), 2)
    path = _simple(tmp_path, levels)
    os.remove(path)
    ets = os.path.join(str(tmp_path), "_a_", "stack7", "frame_t.ets")
    with pytest.warns(CompositeWarning, match="which is missing"):
        f = micro_reader.open(ets)
    with f:
        assert f.backend == "vsi" and f.format == "ets"
        im = f.images[0]
        assert im.shape == (1, 1, 1, 32, 32)                     # whole tiles: no size known
        np.testing.assert_array_equal(im[0, 0, 0, :20, :30], levels[0])
        np.testing.assert_array_equal(im.resolutions[1][0, 0, 0, :10, :15], levels[1])
    with pytest.raises(CorruptFileError, match="which is missing"):
        micro_reader.open(os.path.join(str(tmp_path), "_a_", "stack7"))


def test_an_ets_with_more_dimensions_needs_its_vsi(tmp_path):
    ets = write_ets(os.path.join(str(tmp_path), "frame_t.ets"), _levels((3, 20, 30), 1))
    with pytest.raises(UnsupportedFormatError, match="open the .vsi"):
        micro_reader.open(ets)


def test_a_vsi_without_its_folder_reads_the_macro(tmp_path):
    path = _simple(tmp_path, _levels((20, 30), 1))
    os.rename(os.path.join(str(tmp_path), "_a_"), os.path.join(str(tmp_path), "gone"))
    with pytest.warns(CompositeWarning, match="is missing: only the macro"):
        f = micro_reader.open(path)
    with f:
        assert [im.name for im in f.images] == ["macro"]


def test_other_tiffs_are_not_vsi(tmp_path):
    import tifffile
    path = str(tmp_path / "plain.tif")
    tifffile.imwrite(path, _macro())
    with open(path, "rb") as fh:
        assert not vsi.sniff(fh.read(64), path)
    with micro_reader.open(path) as f:
        assert f.backend == "tiff"


def test_bmp_tiles_are_refused(tmp_path):
    ets = write_ets(os.path.join(str(tmp_path), "frame_t.ets"), _levels((20, 30), 1),
                    compression=0)
    with open(ets, "r+b") as fh:                                 # compression field -> 9 (BMP)
        fh.seek(64 + 4 + 16)
        fh.write((9).to_bytes(4, "little"))
    with pytest.raises(UnsupportedFormatError, match="BMP"):
        micro_reader.open(ets)
