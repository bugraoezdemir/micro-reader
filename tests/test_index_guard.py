"""The image index guard (plan 3.24 / 8.10): the index "scene" was renamed
"series" in micro-reader 0.1.0, and every way of storing or reading the old
name on an image's index raises a KeyError naming the new one -- where a
plain dict would answer quietly (``.get``, ``in``)."""
import concurrent.futures
import copy
import glob
import json
import os
import pickle

import numpy as np
import pytest

import micro_reader
from micro_reader._base import ImageIndex, Image

OLD = "scene"                       # the old name, only ever used to see it refused
DATA = os.environ.get("MICROSCOPY_DATA", os.path.expanduser("~/Desktop/data/microscopy"))


def _index():
    return ImageIndex({"series": 2, "tile": 7})


def _setdefault(idx):
    return idx.setdefault(OLD, 0)


def _ior(idx):
    idx |= {OLD: 0}


def _del(idx):
    del idx[OLD]


REFUSED = {
    "idx[old]": lambda idx: idx[OLD],
    "idx.get(old)": lambda idx: idx.get(OLD),
    "idx.get(old, 0)": lambda idx: idx.get(OLD, 0),
    "old in idx": lambda idx: OLD in idx,
    "idx.pop(old)": lambda idx: idx.pop(OLD),
    "idx.pop(old, None)": lambda idx: idx.pop(OLD, None),
    "del idx[old]": _del,
    "idx.setdefault(old, 0)": _setdefault,
    "idx[old] = 0": lambda idx: idx.__setitem__(OLD, 0),
    "idx.update(old=0)": lambda idx: idx.update(**{OLD: 0}),
    "idx.update({old: 0})": lambda idx: idx.update({OLD: 0}),
    "idx.update([(old, 0)])": lambda idx: idx.update([(OLD, 0)]),
    "idx |= {old: 0}": _ior,
    "idx | {old: 0}": lambda idx: idx | {OLD: 0},
    "{old: 0} | idx": lambda idx: {OLD: 0} | idx,
    "ImageIndex(old=0)": lambda idx: ImageIndex(**{OLD: 0}),
    "ImageIndex({old: 0})": lambda idx: ImageIndex({OLD: 0}),
    "ImageIndex([(old, 0)])": lambda idx: ImageIndex([(OLD, 0)]),
    "ImageIndex.fromkeys([old])": lambda idx: ImageIndex.fromkeys([OLD]),
}


@pytest.mark.parametrize("case", sorted(REFUSED))
def test_the_old_name_raises_naming_series(case):
    idx = _index()
    with pytest.raises(KeyError, match="is now called 'series'"):
        REFUSED[case](idx)
    assert dict(idx) == {"series": 2, "tile": 7}          # nothing was stored


def test_other_keys_work_as_in_a_dict():
    idx = _index()
    assert idx["series"] == 2 and idx.get("view") is None and idx.get("view", 3) == 3
    assert "tile" in idx and "view" not in idx
    idx["view"] = 1
    idx.update(rotation=0)
    idx |= {"phase": 2}
    assert idx.setdefault("view", 9) == 1 and idx.pop("phase") == 2
    idx["absent_soon"] = 5
    del idx["absent_soon"]
    assert idx == {"series": 2, "tile": 7, "view": 1, "rotation": 0}
    with pytest.raises(KeyError):
        idx["absent"]
    with pytest.raises(KeyError, match="absent"):
        del idx["absent"]
    assert ImageIndex.fromkeys(["series", "tile"], 0) == {"series": 0, "tile": 0}
    assert type(ImageIndex.fromkeys(["series"])) is ImageIndex


@pytest.mark.parametrize("make", ["copy", "deepcopy", "copy.copy", "or", "ror", "pickle2",
                                  "pickle5"])
def test_copies_stay_guarded(make):
    idx = _index()
    out = {"copy": lambda: idx.copy(), "deepcopy": lambda: copy.deepcopy(idx),
           "copy.copy": lambda: copy.copy(idx), "or": lambda: idx | {},
           "ror": lambda: {} | idx,
           "pickle2": lambda: pickle.loads(pickle.dumps(idx, 2)),
           "pickle5": lambda: pickle.loads(pickle.dumps(idx, 5))}[make]()
    assert type(out) is ImageIndex and out == idx and out is not idx
    with pytest.raises(KeyError, match="series"):
        out.get(OLD)


def test_merges_keep_the_right_order():
    assert ImageIndex({"series": 0, "tile": 1}) | {"tile": 5} == {"series": 0, "tile": 5}
    merged = {"tile": 5, "view": 0} | ImageIndex({"series": 0, "tile": 1})
    assert merged == {"tile": 1, "view": 0, "series": 0} and type(merged) is ImageIndex


def test_plain_copies_are_silent_by_construction():
    """What leaves the guard as a plain dict says nothing (plan 3.24): that is
    why the packages reading micro-reader rename in the same release."""
    idx = _index()
    assert type(dict(idx)) is dict and dict(idx).get(OLD) is None
    assert {**idx}.get(OLD, 0) == 0
    assert OLD not in idx.keys() and [k for k, _ in idx.items()] == ["series", "tile"]
    assert json.loads(json.dumps(idx)) == {"series": 2, "tile": 7}
    assert idx == {"series": 2, "tile": 7}


def test_image_index_is_wrapped_on_every_assignment():
    plain = {"series": 1, "tile": 0}
    image = Image(0, "yx", (2, 3), "u1", index=plain)
    assert type(image.index) is ImageIndex and image.index == plain
    assert image.index is not plain                       # a copy, as before
    image.index = {"series": 4}
    assert type(image.index) is ImageIndex
    image.index["resolution"] = 1                         # an in-place edit stays guarded
    with pytest.raises(KeyError, match="series"):
        image.index[OLD] = 0
    kept = ImageIndex({"series": 5})
    image.index = kept
    assert image.index is kept
    with pytest.raises(KeyError, match="series"):
        image.index = {OLD: 0}
    assert type(Image(3, "yx", (2, 3), "u1").index) is ImageIndex
    assert Image(3, "yx", (2, 3), "u1").index == {"series": 3}


def _two_series_czi(path):
    from _czi_writer import raw_payload, write_czi
    tiles = []
    for s in (0, 1):
        for m in (0, 1):
            tile = np.full((8, 10), 10 * s + m, np.uint8)
            tiles.append(dict(series=s, mosaic=m, plane={"C": 0}, x=10 * m, y=20 * s,
                              width=10, height=8, dtype="u1", compression=0,
                              payload=raw_payload(tile)))
    write_czi(str(path), tiles)
    return str(path)


def test_file_level_names_say_series(tmp_path):
    path = _two_series_czi(tmp_path / "two.czi")
    with micro_reader.open(path) as f:
        assert f.index_range == {"series": 2, "tile": 2}
        assert [dict(im.index) for im in f.select(series=1)] == \
            [{"series": 1, "tile": 0}, {"series": 1, "tile": 1}]
        assert f.select(series=1, tile=1)[0][0, 0, 0, 0, 0] == 11
        with pytest.raises(KeyError, match="is now called 'series'"):
            f.select(**{OLD: 0})
        with pytest.raises(KeyError, match="unknown indices"):
            f.select(position=0)
        assert "series 0:" in f.outline() and "series 1:" in f.outline()
        assert all(type(im.index) is ImageIndex for im in f.images)
        assert " series 1 tile 0" in f.images[2].name
        assert "series" in repr(f.images[0])
    with micro_reader.open(path, tiles="mosaic") as f:
        assert [dict(im.index) for im in f.images] == [{"series": 0}, {"series": 1}]


def test_views_and_levels_keep_the_guard(tmp_path):
    path = _two_series_czi(tmp_path / "two.czi")
    with micro_reader.open(path, axes="canonical", samples="channels") as f:
        assert all(type(im.index) is ImageIndex for im in f.images)
        assert all(type(im.squeeze().index) is ImageIndex for im in f.images)
        assert all(type(im.as_axes("ctzyx").index) is ImageIndex for im in f.images)
    with micro_reader.open(path, resolutions="all") as f:
        assert all(type(im.index) is ImageIndex for im in f._every_image())


def test_indices_pickle_through_a_process_pool(tmp_path):
    """eubi-bridge sends images' indices to worker processes (spawn)."""
    path = _two_series_czi(tmp_path / "two.czi")
    with micro_reader.open(path) as f:
        indices = [im.index for im in f.images]
    with concurrent.futures.ProcessPoolExecutor(2) as pool:
        types = list(pool.map(type, indices))
    assert types == [ImageIndex] * len(indices)


def _real(pattern):
    hits = sorted(glob.glob(os.path.join(DATA, pattern)))
    if not hits:
        pytest.skip(f"sample not found: {pattern} (set MICROSCOPY_DATA)")
    return hits[0]


def test_mmstack_positions_pick_their_ome_image_by_series():
    """A Micro-Manager MMStack file: its positions are one tifffile series
    (axis R) and its OME-XML has one Image per position, picked by the
    image's series index (_tiff_metadata._ome_image)."""
    from micro_reader.backends import _tiff_metadata
    path = _real("formats/tiff/micromanager/aics-mm2-2pos/*MMStack_2-Pos000_000.ome.tif")
    with micro_reader.open(path) as f:
        assert [dict(im.index) for im in f.images] == [{"series": 0}, {"series": 1}]
        assert all(im._series.kind == "mmstack" for im in f.images)
        cache: dict = {}
        ids = [_tiff_metadata._ome_image(im, cache).get("ID") for im in f.images]
        assert ids == ["Image:0", "Image:1"]
        xs = [im.metadata.position["x"].value for im in f.images]
        assert xs[1] - xs[0] == pytest.approx(512.0)        # StageLabel X -256 / 256


@pytest.mark.parametrize("name", ["formats/czi/challenge/DAPI_PGC.czi",
                                  "formats/czi/os-zeiss/Zeiss-5-JXR.czi"])
def test_czi_label_and_preview_have_no_position(name):
    """Plan 4.18 (interim): the label / slide preview of a slide scan come
    without a stage position until their frame is verified; the slide keeps its."""
    path = _real(name)
    with micro_reader.open(path) as f:
        attached = [im for im in f.images if im.name.endswith((" label", " slide preview"))]
        assert attached
        assert all(im.metadata.position == {} for im in attached)
        assert f.images[0].metadata.position
