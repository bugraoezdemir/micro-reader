"""Plates (docs/HCS.md): ``File.plates``, the well / field indices, and the
pyramid sub-identity of images.  Zeiss plates from the own CZI writer
(tests/_czi_writer.py), with the plate metadata Zen writes: the sample-holder
template (rows x columns) and a ``Shape`` per series naming its well, 1-based."""
from __future__ import annotations

import numpy as np
import pytest

import micro_reader
from _czi_writer import raw_payload, write_czi
from micro_reader._plate import row_label
from test_resolutions import _pyramid_tiff

RNG = np.random.default_rng(13)
H, W = 10, 12
#: one plate, one acquisition: both indices are there, at 0
ONE = {"plate": 0, "acquisition": 0}


def _plate_xml(scenes, rows=8, columns=12, template=True):
    """*scenes*: {series: (row, column, name) with Zeiss' 1-based row / column,
    or None for a series without a well}."""
    parts = []
    for s, shape in scenes.items():
        inner = ""
        if shape is not None:
            r, c, name = shape
            inner = (f'<Shape Name="{name}" Id="{r}-{c}"><ColumnIndex>{c}</ColumnIndex>'
                     f'<RowIndex>{r}</RowIndex></Shape>')
        parts.append(f'<Scene Index="{s}" Name="{shape[2] if shape else s}">{inner}</Scene>')
    xml = ("<Information><Image><Dimensions><S><Scenes>" + "".join(parts)
           + "</Scenes></S></Dimensions></Image></Information>")
    if template:
        xml += ("<Experiment><ExperimentBlocks><AcquisitionBlock><SubDimensionSetups>"
                "<RegionsSetup><SampleHolder>"
                f'<Template Name="Multichamber {rows * columns}"><ShapeRows>{rows}</ShapeRows>'
                f"<ShapeColumns>{columns}</ShapeColumns></Template>"
                "</SampleHolder></RegionsSetup></SubDimensionSetups></AcquisitionBlock>"
                "</ExperimentBlocks></Experiment>")
    return xml


def _write(path, scenes, metadata, tiles_per_scene=1):
    """One tile (or *tiles_per_scene* mosaic tiles) per series -> {(series, tile): array}."""
    out, data = [], {}
    for s in scenes:
        for m in range(tiles_per_scene):
            a = RNG.integers(1, 60000, (H, W)).astype("u2")
            t = dict(series=s, plane={"C": 0}, x=1000 * s + W * m, y=0, height=H, width=W,
                     dtype="u2", compression=0, payload=raw_payload(a))
            if tiles_per_scene > 1:
                t["mosaic"] = m
            out.append(t)
            data[(s, m)] = a
    write_czi(path, out, metadata)
    return data


# series 0 is in a later well than scenes 1 and 2: images follow the wells
SCENES = {0: (3, 5, "C5"), 1: (2, 2, "B2"), 2: (2, 2, "B2")}


def test_a_zeiss_plate_is_read_as_one_with_plates_true(tmp_path):
    path = str(tmp_path / "plate.czi")
    data = _write(path, SCENES, _plate_xml(SCENES))
    with micro_reader.open(path, plates=True) as f:
        (plate,) = f.plates
        assert plate.rows == tuple("ABCDEFGH") and plate.columns == tuple(str(c) for c in range(1, 13))
        assert plate.wells == (micro_reader.Well(1, 1, "B2"), micro_reader.Well(2, 4, "C5"))
        # wells in row-column order, a well's scenes are its fields in series order
        assert f.indices == [{**ONE, "well": 0, "field": 0}, {**ONE, "well": 0, "field": 1},
                             {**ONE, "well": 1, "field": 0}]
        assert f.index_range == {"plate": 1, "acquisition": 1, "well": 2, "field": 2}
        # plate and acquisition need not be given when there is one of each
        assert f.select(well=0) == f.select(plate=0, acquisition=0, well=0) == f.images[:2]
        assert [image.name.split(".czi ")[1] for image in f.images] == \
            ["well B2 field 0", "well B2 field 1", "well C5 field 0"]
        for image, series in zip(f.images, (1, 2, 0)):
            np.testing.assert_array_equal(image[0, 0, 0], data[(series, 0)])
        (c5,) = f.select(well=plate.well_index("C5"))
        np.testing.assert_array_equal(c5[0, 0, 0], data[(0, 0)])
        with pytest.raises(KeyError, match="no well 'H12'"):
            plate.well_index("H12")


def test_plates_are_opt_in(tmp_path):
    """Without plates=True a plate reads as today: scenes, no File.plates
    (eubi-bridge accepts no well / field indices yet)."""
    path = str(tmp_path / "plate.czi")
    _write(path, SCENES, _plate_xml(SCENES))
    with micro_reader.open(path) as f:
        assert f.plates == []
        assert f.indices == [{"series": s} for s in range(3)]


def test_mosaic_tiles_stay_within_their_field(tmp_path):
    path = str(tmp_path / "plate.czi")
    data = _write(path, SCENES, _plate_xml(SCENES), tiles_per_scene=2)
    with micro_reader.open(path, plates=True) as f:
        assert f.indices == [{**ONE, "well": w, "field": fd, "tile": t}
                             for w, fd in ((0, 0), (0, 1), (1, 0)) for t in range(2)]
        (image,) = f.select(well=0, field=1, tile=1)
        np.testing.assert_array_equal(image[0, 0, 0], data[(2, 1)])
    with micro_reader.open(path, plates=True, tiles="mosaic") as f:
        assert f.indices == [{**ONE, "well": 0, "field": 0}, {**ONE, "well": 0, "field": 1},
                             {**ONE, "well": 1, "field": 0}]
        np.testing.assert_array_equal(f.images[1][0, 0, 0, :, W:], data[(2, 1)])


@pytest.mark.parametrize("xml", [
    _plate_xml(SCENES, template=False),                       # no plate layout
    _plate_xml({0: (3, 5, "C5"), 1: None, 2: (2, 2, "B2")}),  # a series outside any well
    _plate_xml({0: (9, 5, "I5"), 1: (2, 2, "B2"), 2: (2, 2, "B2")}),  # a well off the plate
], ids=["no-template", "series-without-well", "well-off-plate"])
def test_files_that_are_not_plates_keep_their_scenes(tmp_path, xml):
    path = str(tmp_path / "scenes.czi")
    _write(path, SCENES, xml)
    with micro_reader.open(path, plates=True) as f:
        assert f.plates == []
        assert f.indices == [{"series": s} for s in range(3)]


def test_unnamed_wells_are_named_from_their_row_and_column(tmp_path):
    scenes = {0: (2, 2, ""), 1: (11, 30, "")}
    path = str(tmp_path / "plate.czi")
    _write(path, scenes, _plate_xml(scenes, rows=32, columns=48))
    with micro_reader.open(path, plates=True) as f:
        assert [w.name for w in f.plates[0].wells] == ["B2", "K30"]


def test_plates_true_on_a_format_without_plates_changes_nothing(tmp_path):
    tifffile = pytest.importorskip("tifffile")
    path = str(tmp_path / "flat.tif")
    tifffile.imwrite(path, np.zeros((8, 9), "u1"))
    with micro_reader.open(path, plates=True) as f:
        assert f.plates == [] and f.indices == [{"series": 0}]


def test_row_labels():
    assert [row_label(r) for r in (0, 7, 25, 26, 31)] == ["A", "H", "Z", "AA", "AF"]
    with pytest.raises(ValueError):
        row_label(-1)


# -- navigation: plate > well > field > tile (containment only) ---------------------

def test_navigation_reaches_the_same_images_as_select(tmp_path):
    path = str(tmp_path / "plate.czi")
    data = _write(path, SCENES, _plate_xml(SCENES), tiles_per_scene=2)
    with micro_reader.open(path, plates=True) as f:
        plate = f.plates[0]
        well = plate.wells[plate.well_index("B2")]
        assert [fld.index for fld in well.fields] == [0, 1]
        assert well.fields[1].tiles[1] is f.select(well=0, field=1, tile=1)[0]
        np.testing.assert_array_equal(well.fields[1].tiles[1][0, 0, 0], data[(2, 1)])
        # tile 1 of every field of the well
        assert [fld.tiles[1].index["field"] for fld in well.fields] == [0, 1]
        with pytest.raises(ValueError, match="has 2 images"):
            well.fields[0].image                      # tiled: pick a tile
    with micro_reader.open(path, plates=True, tiles="mosaic") as f:
        fld = f.plates[0].wells[1].fields[0]
        assert fld.tiles == [fld.image] and fld.image is f.images[2]


def test_navigation_reaches_the_wrapped_images(tmp_path):
    """open(axes=..., squeeze=...) replaces File.images: the plates follow."""
    path = str(tmp_path / "plate.czi")
    _write(path, SCENES, _plate_xml(SCENES))
    with micro_reader.open(path, plates=True, squeeze=True) as f:
        image = f.plates[0].wells[1].fields[0].image
        assert any(image is i for i in f.images) and image.axes == "yx"


def test_orthogonal_indices_are_picked_with_select(tmp_path):
    """Views are orthogonal (docs/AXES.md): a field with two views has no one
    tile order, so .tiles asks for select(view=...) first."""
    scenes = {0: (1, 1, "A1")}
    out, data = [], {}
    for v in range(2):
        for m in range(2):
            a = RNG.integers(1, 60000, (H, W)).astype("u2")
            out.append(dict(series=0, mosaic=m, plane={"C": 0, "V": v}, x=W * m, y=0,
                            height=H, width=W, dtype="u2", compression=0,
                            payload=raw_payload(a)))
            data[(v, m)] = a
    path = str(tmp_path / "views.czi")
    write_czi(path, out, _plate_xml(scenes))
    with micro_reader.open(path, plates=True) as f:
        fld = f.plates[0].wells[0].fields[0]
        assert len(fld.images) == 4
        with pytest.raises(ValueError, match="several view"):
            fld.tiles
        np.testing.assert_array_equal(fld.select(view=1).tiles[0][0, 0, 0], data[(1, 0)])
        assert "x view 2" in f.outline()


# -- outline: the containment tree, shown by outline() and (capped) repr ------------

def test_the_outline_of_a_plate(tmp_path):
    path = str(tmp_path / "plate.czi")
    _write(path, SCENES, _plate_xml(SCENES), tiles_per_scene=2)
    with micro_reader.open(path, plates=True) as f:
        lines = f.outline().splitlines()
        assert lines[0].startswith("<micro_reader.File ") and "plates=1" in lines[0]
        assert lines[1:] == [
            "  plate 0: 8 x 12, 2 wells with images",
            "    well B2",
            f"      field 0: tiles 0-1, tczyx (1, 1, 1, {H}, {W}) uint16",
            f"      field 1: tiles 0-1, tczyx (1, 1, 1, {H}, {W}) uint16",
            "    well C5",
            f"      field 0: tiles 0-1, tczyx (1, 1, 1, {H}, {W}) uint16",
        ]
        assert repr(f) == f.outline()                 # short enough to show whole
    with micro_reader.open(path) as f:                 # not read as a plate: scenes
        assert f.outline().splitlines()[1:] == [
            f"  series {s}: tiles 0-1, tczyx (1, 1, 1, {H}, {W}) uint16" for s in range(3)]


def test_repr_shows_the_start_of_a_long_outline(tmp_path):
    scenes = {s: (1 + s // 12, 1 + s % 12, "") for s in range(14)}
    path = str(tmp_path / "plate.czi")
    _write(path, scenes, _plate_xml(scenes))
    with micro_reader.open(path, plates=True) as f:
        full = f.outline().splitlines()
        short = repr(f).splitlines()
        assert len(full) == 2 + 14 * 2
        assert short[:1 + micro_reader.File.REPR_LINES] == full[:1 + micro_reader.File.REPR_LINES]
        assert short[-1] == f"  ... {len(full) - 1 - micro_reader.File.REPR_LINES} more lines: .outline()"


def test_the_outline_shows_pyramid_levels_once(tmp_path):
    data = RNG.integers(0, 4000, (2, 96, 80)).astype("u2")
    path = str(tmp_path / "pyr.ome.tif")
    _pyramid_tiff(path, data)
    for policy in ("base", "all"):                     # levels listed as images: still one line
        with micro_reader.open(path, resolutions=policy) as f:
            assert f.outline().splitlines()[1:] == ["  series 0: cyx (2, 96, 80) uint16, 3 levels"]


def test_runs():
    from micro_reader._outline import runs
    assert runs([5, 0, 1, 2, 7, 8]) == "0-2, 5, 7-8" and runs([3]) == "3"


# -- pyramid sub-identity: an image is a pyramid or a level -------------------------

def test_an_image_is_a_pyramid_or_a_level(tmp_path):
    data = RNG.integers(0, 4000, (2, 96, 80)).astype("u2")
    path = str(tmp_path / "pyr.ome.tif")
    _pyramid_tiff(path, data)
    with micro_reader.open(path) as f:
        (image,) = f.images
        assert image.is_pyramid and image.level == 0 and image.pyramid is image
        for k, lv in enumerate(image.resolutions):
            assert lv.level == k and lv.pyramid is image
            assert lv.is_pyramid == (k == 0)
    with micro_reader.open(path, resolutions="all") as f:
        assert [im.level for im in f.images] == [0, 1, 2]
        assert [im.is_pyramid for im in f.images] == [True, False, False]
        assert all(im.pyramid is f.images[0] for im in f.images)


def test_an_image_without_levels_is_its_own_pyramid(tmp_path):
    tifffile = pytest.importorskip("tifffile")
    path = str(tmp_path / "flat.tif")
    tifffile.imwrite(path, np.zeros((8, 9), "u1"))
    with micro_reader.open(path) as f:
        (image,) = f.images
        assert image.is_pyramid and image.level == 0 and image.pyramid is image
        assert not image.is_pyramidal


# -- select by well name, row and column; several values ---------------------------

#: wells B2 (2 fields), C5 and D7 (one field each), in an 8 x 12 plate
WELLS = {0: (3, 5, "C5"), 1: (2, 2, "B2"), 2: (2, 2, "B2"), 3: (4, 7, "D7")}


def test_select_by_well_name_row_and_column(tmp_path):
    path = str(tmp_path / "plate.czi")
    _write(path, WELLS, _plate_xml(WELLS))
    with micro_reader.open(path, plates=True) as f:
        plate = f.plates[0]
        b2 = plate.well_index("B2")
        assert f.select(well="B2") == f.select(well=b2) == f.images[:2]
        assert f.select(well=["B2", "D7"]) == f.select(well=[b2, plate.well_index("D7")])
        assert f.select(row="C") == f.select(row=2) == f.select(well="C5")    # 0-based
        assert f.select(column="5") == f.select(column=4) == f.select(well="C5")
        assert f.select(row=["B", "C"]) == f.images[:3]
        assert f.select(row=range(1, 3), column="2") == f.images[:2]
        assert f.select(row="H") == []                  # a row without images
        assert f.select(field=[0, 1]) == f.images
        assert f.select(well="B2", field=range(1, 5)) == [f.images[1]]


def test_select_refuses_what_is_no_index_value(tmp_path):
    path = str(tmp_path / "plate.czi")
    _write(path, WELLS, _plate_xml(WELLS))
    with micro_reader.open(path, plates=True) as f:
        plate = f.plates[0]
        with pytest.raises(TypeError, match="a Well object is no index value"):
            f.select(well=plate.wells[0])
        with pytest.raises(TypeError, match="not both"):
            f.select(well=0, column=1)
        with pytest.raises(KeyError, match="unknown indices"):
            plate.wells[0].fields[0].select(row=0)
    with micro_reader.open(path) as f:                  # no plates=True: no plates
        with pytest.raises(ValueError, match="plates=True"):
            f.select(row="B")
