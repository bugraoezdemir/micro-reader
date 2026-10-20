"""CZI images and image keys (docs/AXES.md): scenes, tiles, rotations and
views are separate images; blocks, illuminations and phases are axes in the
canonical order.  Files from the own writer (tests/_czi_writer.py), so any
dimension can be written."""
from __future__ import annotations

import itertools

import numpy as np
import pytest

import micro_reader
from _czi_writer import raw_payload, write_czi

RNG = np.random.default_rng(3)
H, W = 12, 16


def _write(path, tiles):
    """*tiles*: [(series, mosaic or None, plane dict, (x, y))] -> {tile: array}."""
    out, data = [], {}
    for series, mosaic, plane, (x, y) in tiles:
        a = RNG.integers(1, 60000, (H, W)).astype("u2")
        t = dict(series=series, plane=plane, x=x, y=y, height=H, width=W, dtype="u2",
                 compression=0, payload=raw_payload(a))
        if mosaic is not None:
            t["mosaic"] = mosaic
        out.append(t)
        data[(series, mosaic, tuple(sorted(plane.items())))] = a
    write_czi(path, out)
    return data


def test_views_and_rotations_are_images_not_axes(tmp_path):
    path = str(tmp_path / "views.czi")
    tiles = [(0, None, {"C": c, "R": r, "V": v}, (10 * v, 5 * r))
             for r in range(2) for v in range(3) for c in range(2)]
    data = _write(path, tiles)
    with micro_reader.open(path) as f:
        assert f.index_range == {"series": 1, "rotation": 2, "view": 3}
        assert f.indices == [{"series": 0, "rotation": r, "view": v}
                                for r in range(2) for v in range(3)]
        for image in f.images:
            assert image.axes == "tczyx" and image.shape == (1, 2, 1, H, W)
            r, v = image.index["rotation"], image.index["view"]
            assert image.origin == (10 * v, 5 * r)
            for c in range(2):
                np.testing.assert_array_equal(
                    image[0, c, 0], data[(0, None, (("C", c), ("R", r), ("V", v)))])
        assert len(f.select(view=1)) == 2 and f.select(rotation=1, view=2)[0].index["view"] == 2


def test_block_illumination_phase_are_indices(tmp_path):
    """Dimensions sampled at the same points are image indices too: arrays
    have t c z y x only."""
    path = str(tmp_path / "bih.czi")
    tiles = [(0, None, {"B": b, "I": i, "H": h, "C": 0, "Z": z}, (0, 0))
             for b, i, h, z in itertools.product(range(2), range(2), range(3), range(2))]
    data = _write(path, tiles)
    with micro_reader.open(path) as f:
        assert f.index_range == {"series": 1, "illumination": 2, "phase": 3, "block": 2}
        assert len(f.images) == 2 * 2 * 3
        assert f.indices[0] == {"series": 0, "illumination": 0, "phase": 0, "block": 0}
        for image in f.images:
            assert image.axes == "tczyx" and image.shape == (1, 1, 2, H, W)
            i, h, b = (image.index[k] for k in ("illumination", "phase", "block"))
            for z in range(2):
                plane = tuple(sorted({"B": b, "I": i, "H": h, "C": 0, "Z": z}.items()))
                np.testing.assert_array_equal(image[0, 0, z], data[(0, None, plane)])
        (one,) = f.select(illumination=1, phase=2, block=0)
        plane = tuple(sorted({"B": 0, "I": 1, "H": 2, "C": 0, "Z": 1}.items()))
        np.testing.assert_array_equal(one[0, 0, 1], data[(0, None, plane)])


def test_tiles_per_scene_are_ragged_and_stitching_is_opt_in(tmp_path):
    path = str(tmp_path / "tiles.czi")
    tiles = [(0, m, {"C": 0}, (W * m, 0)) for m in range(3)] + \
            [(1, m, {"C": 0}, (500 + W * m, 0)) for m in range(2)]
    data = _write(path, tiles)
    with micro_reader.open(path) as f:
        assert f.indices == [{"series": 0, "tile": m} for m in range(3)] + \
            [{"series": 1, "tile": m} for m in range(2)]
        assert f.index_range == {"series": 2, "tile": 3}
        for image in f.images:
            s, m = image.index["series"], image.index["tile"]
            np.testing.assert_array_equal(image[0, 0, 0], data[(s, m, (("C", 0),))])
    with micro_reader.open(path, tiles="mosaic") as f:
        assert f.indices == [{"series": 0}, {"series": 1}]
        assert f.images[0].shape[-2:] == (H, 3 * W)
        np.testing.assert_array_equal(f.images[1][0, 0, 0, :, W:], data[(1, 1, (("C", 0),))])


def test_single_tile_scenes_have_no_tile_level(tmp_path):
    path = str(tmp_path / "one.czi")
    _write(path, [(0, 0, {"C": 0}, (0, 0)), (1, 0, {"C": 0}, (100, 0))])
    with micro_reader.open(path) as f:
        assert f.indices == [{"series": 0}, {"series": 1}]


def test_canvas_needs_a_view_when_there_are_several(tmp_path):
    path = str(tmp_path / "views.czi")
    data = _write(path, [(s, None, {"C": 0, "V": v}, (100 * s, 0))
                         for s in range(2) for v in range(2)])
    with micro_reader.open(path) as f:
        with pytest.raises(ValueError, match=r"canvas\(view=\.\.\.\)"):
            f.canvas()
        canvas = f.canvas(view=1)
        assert canvas.index == {"view": 1} and canvas.shape[-2:] == (H, 100 + W)
        np.testing.assert_array_equal(canvas[0, 0, 0, :, 100:],
                                      data[(1, None, (("C", 0), ("V", 1)))])


def test_bad_tiles_policy(tmp_path):
    path = str(tmp_path / "one.czi")
    _write(path, [(0, None, {"C": 0}, (0, 0))])
    with pytest.raises(ValueError, match="tiles="):
        micro_reader.open(path, tiles="blended")
