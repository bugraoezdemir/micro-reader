"""FluoView tiling sets (``.omp2info`` index + OIR tiles), on synthetic sets
shaped as the real one (formats/oir/ome-zenodo-13680725, docs/COMPOSITE.md)."""
from __future__ import annotations

import os
import warnings

import numpy as np
import pytest

import micro_reader
from micro_reader import CorruptFileError, MissingImagesWarning, UnsupportedFormatError

from _oir_writer import write_oir

H, W, PX = 32, 40, 0.5                  # tile rows, columns, pixel size (Âµm)
LEFT, TOP = -170000.0, -660000.0        # areaLeft / areaTop (nm)
GRID = [(y, x) for y in range(2) for x in range(3)]   # (yIndex, xIndex), index order


def _tile_name(k):
    return f"tiling_A01_G001_{k + 1:04d}.oir"


def _index_xml(names, overlap=10, map_name="Map_A01.oir"):
    areas = "".join(
        f"<matl:area><matl:image>{n}</matl:image><matl:xIndex>{x}</matl:xIndex>"
        f"<matl:yIndex>{y}</matl:yIndex></matl:area>" for n, (y, x) in zip(names, GRID))
    stage = "" if overlap is None else f"<matl:stage><matl:overlap>{overlap}</matl:overlap></matl:stage>"
    return (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<matl:properties xmlns:matl="http://www.olympus.co.jp/hpf/protocol/matl/model/matl">'
        f"<matl:map><matl:image>{map_name}</matl:image></matl:map>{stage}"
        "<matl:group><matl:areaInfo>"
        f"<matl:numOfXAreas>3</matl:numOfXAreas><matl:numOfYAreas>2</matl:numOfYAreas>"
        f"<matl:areaLeft>{LEFT}</matl:areaLeft><matl:areaTop>{TOP}</matl:areaTop>"
        f"<matl:areaWidth>{W * PX * 1000}</matl:areaWidth><matl:areaHeight>{H * PX * 1000}</matl:areaHeight>"
        f"</matl:areaInfo>{areas}</matl:group></matl:properties>")


def _write_set(folder, *, overlap=10, skip=(), pixel_size=(PX, PX), index="matl.omp2info"):
    """A 2 x 3 tiling set; returns (index path, [tile data (1, 2, 1, H, W)])."""
    os.makedirs(folder, exist_ok=True)
    rng = np.random.default_rng(1)
    tiles = [rng.integers(1, 60000, (1, 2, 1, H, W)).astype("u2") for _ in GRID]
    for k, data in enumerate(tiles):
        if k not in skip:
            write_oir(os.path.join(folder, _tile_name(k)), data, pixel_size=pixel_size)
    path = os.path.join(folder, index)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(_index_xml([_tile_name(k) for k in range(len(GRID))], overlap))
    return path, tiles


def test_tiles_are_separate_images_at_the_index_positions(tmp_path):
    path, tiles = _write_set(str(tmp_path))
    with micro_reader.open(path) as f:
        assert (f.backend, f.format, len(f.images)) == ("omp2info", "omp2info", 6)
        for k, (im, (yi, xi)) in enumerate(zip(f.images, GRID)):
            assert im.index == {"series": 0, "tile": k}
            np.testing.assert_array_equal(im[...], tiles[k])
            pos = im.metadata.position                    # the first pixel's centre (Âµm)
            step = 0.95                                   # 10 % overlap: 5 % per side
            assert pos["x"].to("micrometer").value == pytest.approx(
                (LEFT + xi * W * PX * 1000 * step) / 1000 + PX / 2)
            assert pos["y"].to("micrometer").value == pytest.approx(
                (TOP + yi * H * PX * 1000 * step) / 1000 + PX / 2)
            assert im.metadata.pixel_sizes["x"].to("micrometer").value == PX


def test_a_folder_opens_its_index(tmp_path):
    path, _ = _write_set(str(tmp_path))
    with micro_reader.open(str(tmp_path)) as f:
        assert f.backend == "omp2info" and os.path.samefile(f.path, path)


def test_each_tile_still_opens_alone(tmp_path):
    _write_set(str(tmp_path))
    with micro_reader.open(str(tmp_path / _tile_name(4))) as f:
        assert f.backend == "oir" and len(f.images) == 1


@pytest.mark.parametrize("overlap", [10, 0, None])
def test_the_mosaic_places_the_tiles_later_on_top(tmp_path, overlap):
    path, tiles = _write_set(str(tmp_path), overlap=overlap)
    share = 1 - (overlap or 0) / 200
    corners = [(round(yi * H * share), round(xi * W * share)) for yi, xi in GRID]
    expected = np.zeros((1, 2, 1, max(y for y, _ in corners) + H, max(x for _, x in corners) + W), "u2")
    for (y, x), data in zip(corners, tiles):
        expected[..., y:y + H, x:x + W] = data
    with micro_reader.open(path, tiles="mosaic") as f:
        (m,) = f.images
        assert m.index == {"series": 0} and m.shape == expected.shape
        np.testing.assert_array_equal(m[...], expected)
        np.testing.assert_array_equal(m[0, 1, 0, 20:50, 30:70][...], expected[0, 1, 0, 20:50, 30:70])
        assert m.metadata.position["x"].to("micrometer").value == pytest.approx(LEFT / 1000 + PX / 2)
        assert [(y, x) for _t, y, x in m.tiles] == corners


def test_a_missing_tile_warns_and_its_area_reads_zero(tmp_path):
    path, tiles = _write_set(str(tmp_path), skip=(5,))
    with pytest.warns(MissingImagesWarning, match="1 of its tile files"):
        f = micro_reader.open(path)
    with f:
        assert [im.index["tile"] for im in f.images] == [0, 1, 2, 3, 4]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", MissingImagesWarning)
        with micro_reader.open(path, tiles="mosaic") as f:
            canvas = f.images[0][...]
    assert canvas.shape[-2:] == (round(H * 0.95) + H, round(2 * W * 0.95) + W)
    assert not canvas[..., H:, 2 * W:].any()                  # only the missing tile was there


def test_no_tile_present_is_an_error(tmp_path):
    path, _ = _write_set(str(tmp_path), skip=range(len(GRID)))
    with pytest.raises(CorruptFileError, match="none of its 6 tile files"):
        micro_reader.open(path)


def test_a_mosaic_needs_pixel_sizes(tmp_path):
    path, _ = _write_set(str(tmp_path), pixel_size=None)
    with micro_reader.open(path) as f:                         # separate tiles: fine
        assert len(f.images) == 6
    with pytest.raises(UnsupportedFormatError, match="no pixel size"):
        micro_reader.open(path, tiles="mosaic")


def test_a_tile_of_another_shape_is_refused_when_read(tmp_path):
    path, _ = _write_set(str(tmp_path))
    write_oir(str(tmp_path / _tile_name(3)), np.zeros((1, 2, 1, H, W + 1), "u2"), pixel_size=(PX, PX))
    with micro_reader.open(path) as f:
        f.images[0][...]
        with pytest.raises(CorruptFileError, match="a tile of"):
            f.images[3][...]


def test_close_closes_the_tiles(tmp_path):
    path, _ = _write_set(str(tmp_path))
    with micro_reader.open(path, tiles="mosaic") as f:
        m = f.images[0]
        m[...]
    with pytest.raises(ValueError, match="closed"):
        m[...]
    with micro_reader.open(path) as f:
        tile = f.images[2]
    with pytest.raises(ValueError, match="closed"):
        tile[...]


def _write_map(folder):
    """A FluoView overview map as on the sample: an OIR without pixel blocks
    and the same-stem HDF5 that would hold them."""
    write_oir(os.path.join(folder, "Map_A01.oir"), np.zeros((1, 1, 1, 8, 8), "u2"),
              missing=[(0, 0, 0)], reference=False)
    with open(os.path.join(folder, "Map_A01.hdf5"), "wb") as fh:
        fh.write(b"\x89HDF\r\n\x1a\n" + b"\0" * 504)


def test_group_files_tiles_are_members_the_map_side_files(tmp_path):
    path, _ = _write_set(str(tmp_path))
    _write_map(str(tmp_path))
    g = micro_reader.group_files(str(tmp_path))
    (group,) = g.groups
    assert (group.kind, group.format, os.path.basename(group.main)) == ("index", "omp2info", "matl.omp2info")
    assert sorted(os.path.basename(p) for p in group.members) == [_tile_name(k) for k in range(6)]
    assert sorted(os.path.basename(p) for p in group.parts) == ["Map_A01.hdf5", "Map_A01.oir"]
    assert not g.singles


def test_the_map_is_refused_by_name_from_either_file(tmp_path):
    _write_map(str(tmp_path))
    for name in ("Map_A01.oir", "Map_A01.hdf5"):
        with pytest.raises(UnsupportedFormatError, match="FluoView overview map.*Map_A01.hdf5"):
            micro_reader.open(str(tmp_path / name))


# -- coordinate systems (plan 4.10, check 4.19) ---------------------------------------

@pytest.fixture
def metadata_reads(monkeypatch):
    """How often a tile's or a group mosaic's ``_read_metadata`` ran."""
    from micro_reader.backends import omp2info
    count = [0]
    for cls in (omp2info._Tile, omp2info._TilingMosaic):
        real = cls._read_metadata

        def counting(self, real=real):
            count[0] += 1
            return real(self)
        monkeypatch.setattr(cls, "_read_metadata", counting)
    return count


def _write_two_groups(folder):
    """A set of two groups (3 + 2 tiles) at different stage corners."""
    path, _ = _write_set(folder)
    xml = open(path, encoding="utf-8").read()
    second = ("<matl:group><matl:areaInfo><matl:numOfXAreas>2</matl:numOfXAreas>"
              "<matl:numOfYAreas>1</matl:numOfYAreas><matl:areaLeft>50000.0</matl:areaLeft>"
              f"<matl:areaTop>20000.0</matl:areaTop><matl:areaWidth>{W * PX * 1000}</matl:areaWidth>"
              f"<matl:areaHeight>{H * PX * 1000}</matl:areaHeight></matl:areaInfo>"
              + "".join(f"<matl:area><matl:image>{_tile_name(k)}</matl:image><matl:xIndex>{x}"
                        "</matl:xIndex><matl:yIndex>0</matl:yIndex></matl:area>"
                        for k, x in ((4, 0), (5, 1)))
              + "</matl:group>")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(xml.replace("</matl:properties>", second + "</matl:properties>"))
    return path


@pytest.mark.parametrize("tiles", ["separate", "mosaic"])
def test_every_group_is_placed_in_one_stage(tmp_path, tiles, metadata_reads):
    """One ``stage`` for the whole protocol: every tile (or group mosaic) of
    every group names it, at the index positions; declaring reads no metadata."""
    from micro_reader._spatial import check_invariants
    path = _write_two_groups(str(tmp_path))
    with micro_reader.open(path, tiles=tiles) as f:
        assert [s.id for s in f.coordinate_systems] == ["stage"]
        assert f.coordinate_systems[0].axes is None and f.coordinate_transformations == ()
        assert metadata_reads[0] == 0                   # declared without metadata
        check_invariants(f)                             # I1-I3 on every image
        assert [im.index["series"] for im in f.images] == ([0] * 6 + [1] * 2 if tiles == "separate"
                                                          else [0, 1])
        for im in f.images:
            meta = im.metadata
            assert meta.coordinate_system == "stage" and set(meta.position) == {"x", "y"}
        last = f.images[-1].metadata.position                 # group 1's first area is at its corner
        if tiles == "separate":                               # ... and the last one a step along x
            assert last["x"].to("micrometer").value == pytest.approx(50 + W * PX * 0.95 + PX / 2)
        else:
            assert last["x"].to("micrometer").value == pytest.approx(50 + PX / 2)
        assert last["y"].to("micrometer").value == pytest.approx(20 + PX / 2)


def test_tiles_without_a_pixel_size_are_unplaced(tmp_path):
    """No pixel size, no first pixel to place: position ``{}`` and no system."""
    from micro_reader._spatial import check_invariants
    path, _ = _write_set(str(tmp_path), pixel_size=None)
    with micro_reader.open(path) as f:
        assert [s.id for s in f.coordinate_systems] == ["stage"]
        check_invariants(f)
        assert all(im.metadata.position == {} and im.metadata.coordinate_system is None
                   for im in f.images)


def test_a_tile_opened_alone_has_no_position(tmp_path):
    """A plain OIR records no stage position (plan 4.12): only the index places it."""
    _write_set(str(tmp_path))
    with micro_reader.open(str(tmp_path / _tile_name(0))) as f:
        assert f.coordinate_systems == ()
        assert f.images[0].metadata.position == {} and f.images[0].metadata.coordinate_system is None


def test_the_docstring_says_the_step_rests_on_one_sample():
    """Check 4.19 closed as "one sample": the backend says so."""
    from micro_reader.backends import omp2info
    doc = " ".join(omp2info.__doc__.split())
    assert "Derived from one sample only" in doc and "4.19" in doc
