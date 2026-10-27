"""ImageXpress HCS.ai exports (backends/jdce.py): a synthetic image set, then the
real samples (plates/imagexpress-jdce)."""
from __future__ import annotations

import csv
import glob
import json
import os

import numpy as np
import pytest

tifffile = pytest.importorskip("tifffile")

import micro_reader  # noqa: E402

RNG = np.random.default_rng(61)
Y, X = 12, 16
COLUMNS = ["FovUuid", "Well", "Row", "Column", "Field", "Wavelength", "Timepoint", "ZIndex",
           "ImageSubFolderPath", "ImageFileName", "TimeStampSec", "ExcitationEmissionFilter",
           "PositionXUm", "PositionYUm", "PositionZUm", "FieldOffsetPointX", "FieldOffsetPointY"]


def _write(folder, *, wells=((3, 5), (3, 6)), fields=2, t=2, c=2, z=3, missing=()):
    """An HCS.ai image set (a .jdce, its CSV, one TIFF per plane in
    timepoint<n> folders); *missing*: (row, col, field, t, c, z) not written.
    -> (index path, {key: plane})."""
    data, rows = {}, []
    for r, col in wells:
        for f in range(fields):
            for tt in range(t):
                sub = os.path.join(folder, f"timepoint{tt}")
                os.makedirs(sub, exist_ok=True)
                for cc in range(c):
                    for zz in range(z):
                        key = (r, col, f, tt, cc, zz)
                        name = f"test_t{tt}_{chr(64 + r)}{col:02d}_s{f}_w{cc}_z{zz}.tif"
                        a = data[key] = RNG.integers(0, 60000, (Y, X)).astype("u2")
                        if key not in missing:
                            tifffile.imwrite(os.path.join(sub, name), a)
                        rows.append({"FovUuid": "x", "Well": f"{chr(64 + r)} - {col}", "Row": r,
                                     "Column": col, "Field": f, "Wavelength": cc, "Timepoint": tt,
                                     "ZIndex": zz, "ImageSubFolderPath": f"timepoint{tt}",
                                     "ImageFileName": name, "TimeStampSec": 1000 + 10 * tt,
                                     "ExcitationEmissionFilter": ("DAPI", "FITC")[cc],
                                     "PositionXUm": 5000, "PositionYUm": 3000,
                                     "PositionZUm": 100 + 4 * zz,
                                     "FieldOffsetPointX": 300 * f, "FieldOffsetPointY": -200})
    with open(os.path.join(folder, "image_metadata_1.csv"), "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, COLUMNS)
        w.writeheader()
        w.writerows(rows)
    index = os.path.join(folder, "screen.jdce")
    stack = {"ImageStack": {
        "PlateId": "screen 2026-03-02", "LargeImage": False,
        "AutoLeadAcquisitionProtocol": {
            "ObjectiveCalibration": {"Unit": "�m", "PixelWidth": 1.294, "PixelHeight": 1.294},
            "Plate": {"Name": "96 Well", "Rows": 8, "Columns": 12},
            "Wavelengths": [{"Index": i, "ZStep": 4.0,
                             "EmissionFilter": {"Name": n, "Wavelength": w, "Unit": "nm"},
                             "ExcitationFilter": {"Name": n, "Wavelength": w, "Unit": "nm"}}
                            for i, (n, w) in enumerate((("DAPI", 440), ("FITC", 520)))]},
        "SpecimenHolder": {"Barcode": ""},
        "ImageMetadataFiles": ["image_metadata_1.csv"]}}
    with open(index, "w", encoding="utf-8") as fh:
        json.dump(stack, fh, indent=4)
    return index, data


def test_an_image_set_is_a_plate(tmp_path):
    index, data = _write(str(tmp_path))
    for path in (index, str(tmp_path), str(tmp_path / "image_metadata_1.csv"), str(tmp_path / "timepoint1")):
        with micro_reader.open(path) as f:
            assert (f.backend, f.format) == ("jdce", "imagexpress-jdce"), path
            (plate,) = f.plates
            assert plate.name == "screen 2026-03-02" and (len(plate.row_names), len(plate.column_names)) == (8, 12)
            assert [(w.name, [x.index for x in w.fields]) for w in plate.wells] == [
                ("C5", [0, 1]), ("C6", [0, 1])]
    with micro_reader.open(index) as f:
        im = f.plates[0].wells[1].fields[1].image
        assert (im.axes, im.shape) == ("tczyx", (2, 2, 3, Y, X))
        np.testing.assert_array_equal(im[1, 1, 2], data[(3, 6, 1, 1, 1, 2)])
        meta = im.metadata
        assert [c.name for c in meta.channels] == ["DAPI", "FITC"]
        assert meta.channels[1].emission_wavelength == micro_reader.Quantity(520, "nanometer")
        assert {a: round(q.value, 3) for a, q in meta.pixel_sizes.items()} == {
            "x": 1.294, "y": 1.294, "z": 4.0, "t": 10.0}
        # the field's centre relative to the well's, y down: the first pixel
        assert meta.position["x"].value == pytest.approx(300 - (X - 1) / 2 * 1.294)
        assert meta.position["y"].value == pytest.approx(-200 - (Y - 1) / 2 * 1.294)
        assert meta.coordinate_system == "well-C6"


def test_missing_planes_read_as_zeros(tmp_path):
    index, _data = _write(str(tmp_path), missing={(3, 5, 0, 0, 1, 2)})
    with pytest.warns(micro_reader.MissingImagesWarning, match="1 image files are missing"):
        with micro_reader.open(index) as f:
            assert not f.images[0][0, 1, 2].any()


def test_an_image_set_is_one_group(tmp_path):
    index, _data = _write(str(tmp_path))
    g = micro_reader.group_files(str(tmp_path))
    assert len(g.groups) == 1 and not g.singles
    assert g.groups[0].format == "jdce" and len(g.groups[0].members) == 2 * 2 * 2 * 2 * 3


DATA = os.environ.get("MICROSCOPY_DATA", os.path.expanduser("~/Desktop/data/microscopy"))
JDCE = os.path.join(DATA, "plates", "imagexpress-jdce", "zenodo-19694932")


def test_zenodo_19694932_every_image_set_reads():
    """Zenodo 19694932 (ZMB, CC BY 4.0): MetaXpress Acquire 2025 HCS.ai sets --
    projections, z-stacks, a stitched montage, 2 wells x 4 fields x 4 t."""
    indexes = sorted(glob.glob(os.path.join(JDCE, "*", "*", "*", "*.jdce")))
    if not indexes:
        pytest.skip(f"sample not found: {JDCE} (set MICROSCOPY_DATA)")
    assert len(indexes) == 9
    for index in indexes:
        with micro_reader.open(index) as f:
            (plate,) = f.plates
            assert (len(plate.row_names), len(plate.column_names)) == (8, 12)
            assert f.images[0].metadata.pixel_sizes["x"].value == pytest.approx(1.294)
    index = os.path.join(JDCE, "2w4s3t2c4z_binning4x4", "test_data_20260302_143452",
                         "experiment_z_stack", "2w4s3t2c4z_binning4x4_z_stack.jdce")
    with micro_reader.open(index) as f:
        assert [(w.name, len(w.fields)) for w in f.plates[0].wells] == [("C5", 4), ("C6", 4)]
        im = f.plates[0].wells[1].fields[3].image
        assert im.shape == (4, 2, 4, 576, 576)
        name = "test_data_t3_C06_s3_w1_z3.tif"
        np.testing.assert_array_equal(im[3, 1, 3], tifffile.imread(os.path.join(
            os.path.dirname(index), "timepoint3", name)))


def test_the_montage_tiles_sit_where_their_positions_say():
    """The y direction of FieldOffsetPointY: the six overlapping tiles of the
    montage sample found in MetaXpress' own stitched montage near the place
    their positions give (y down), a tile per position."""
    folder = os.path.join(JDCE, "1w4s1t1c4z_binning4x4_tiled-L_montage-stitch-zstack",
                          "test_data_20260316_145602")
    if not os.path.exists(folder):
        pytest.skip(f"sample not found: {folder} (set MICROSCOPY_DATA)")
    tiles = micro_reader.open(glob.glob(os.path.join(folder, "experiment_z_stack", "*.jdce"))[0])
    montage = micro_reader.open(glob.glob(os.path.join(folder, "experiment_montage", "*.jdce"))[0])
    with tiles, montage:
        m = np.asarray(montage.images[0][0, 0, 1], float)
        px = 1.294
        fields = tiles.plates[0].wells[0].fields
        x0 = min(f.image.metadata.position["x"].value for f in fields)
        y0 = min(f.image.metadata.position["y"].value for f in fields)
        for field in fields:
            a = np.asarray(field.image[0, 0, 1], float)
            pos = field.image.metadata.position
            r0, c0 = (pos["y"].value - y0) / px, (pos["x"].value - x0) / px
            best = -1.0
            for r in range(max(0, int(r0) - 10), min(m.shape[0] - a.shape[0], int(r0) + 10) + 1):
                for c in range(max(0, int(c0) - 10), min(m.shape[1] - a.shape[1], int(c0) + 10) + 1):
                    b = m[r:r + a.shape[0], c:c + a.shape[1]]
                    best = max(best, float(np.corrcoef(a.ravel(), b.ravel())[0, 1]))
            assert best > 0.85, (field.index, best)
