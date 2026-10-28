"""PerkinElmer Opera Flex plates (backends/flex.py): synthetic .flex files, then
the real samples (plates/flex)."""
from __future__ import annotations

import os

import numpy as np
import pytest

tifffile = pytest.importorskip("tifffile")

import micro_reader  # noqa: E402

RNG = np.random.default_rng(11)
Y, X = 12, 16
PX = 6.5e-7                                          # m


def _write(path, *, row, col, pages, barcode="P1", rows=8, cols=12, factor="1.000000"):
    """A .flex file of well (*row*, *col*) (1-based): *pages* is a list of
    {"field", "z", "exposure", "camera", optional "t", "x", "y", "zpos"}, one
    per page, in page order (BufferNo).  -> {page: data}."""
    arrays, images, data = [], [], {}
    for k, p in enumerate(pages):
        name = f"Exp{p['exposure']}Cam{p['camera'][-1]}"
        arrays.append(f'<Array Type="Image" Name="{name}" Width="{X}" Height="{Y}" '
                      f'BitsPerPixel="16" Factor="1"/>')
        t = f'<Timepoint Type="integer">{p["t"]}</Timepoint>' if "t" in p else ""
        images.append(
            f'<Image BufferNo="{k}"><ExposureNo Type="integer">{p["exposure"]}</ExposureNo>'
            f'<CameraRef>{p["camera"]}</CameraRef><Stack Type="integer">{p["z"]}</Stack>'
            f'<Sublayout Type="integer">{p["field"]}</Sublayout>{t}'
            f'<ImageFactor Type="decimal">{factor}</ImageFactor>'
            f'<ImageResolutionX Type="decimal" Unit="m">{PX}</ImageResolutionX>'
            f'<ImageResolutionY Type="decimal" Unit="m">{PX}</ImageResolutionY>'
            f'<PositionX Type="decimal" Unit="m">{p.get("x", 0.0)}</PositionX>'
            f'<PositionY Type="decimal" Unit="m">{p.get("y", 0.0)}</PositionY>'
            f'<PositionZ Type="decimal" Unit="m">{p.get("zpos", (p["z"] - 1) * 4e-7)}</PositionZ>'
            f'</Image>')
    xml = (f'<Root><Arrays>{"".join(arrays)}</Arrays><FLEX version="1.8.1.0">'
           f'<Plate><PlateName Type="string">96 plate</PlateName><XSize Type="integer">{rows}'
           f'</XSize><YSize Type="integer">{cols}</YSize><Barcode Type="string">{barcode}'
           f'</Barcode></Plate><Well><WellCoordinate Row="{row}" Col="{col}"/><Images>'
           f'{"".join(images)}</Images></Well></FLEX></Root>')
    with tifffile.TiffWriter(path) as w:
        for k in range(len(pages)):
            data[k] = RNG.integers(0, 60000, (Y, X)).astype("u2")
            w.write(data[k], photometric="minisblack", metadata=None,
                    extratags=[(65200, "s", 0, xml, True)] if k == 0 else None)
    return data


def _two_channels_three_z(field=1, x=0.0, y=0.0):
    """Pages as an Opera writes them: the exposures of one plane, plane by plane."""
    return [{"field": field, "z": z, "exposure": e, "camera": "Camera_1", "x": x, "y": y}
            for z in (1, 2, 3) for e in (1, 2)]


def test_a_folder_of_flex_files_is_one_plate(tmp_path):
    a = _write(str(tmp_path / "001002000.flex"), row=1, col=2,
               pages=_two_channels_three_z(x=1e-4, y=-2e-4))
    b = _write(str(tmp_path / "002005000.flex"), row=2, col=5, pages=_two_channels_three_z())
    with micro_reader.open(str(tmp_path / "002005000.flex")) as f:      # any file: the plate
        assert f.backend == "flex"
        (plate,) = f.plates
        assert plate.name == "P1" and (len(plate.row_names), len(plate.column_names)) == (8, 12)
        assert [w.name for w in plate.wells] == ["A2", "B5"]
        im = plate.wells[0].fields[0].image
        assert im.index == {"plate": 0, "acquisition": 0, "well": 0, "field": 0}
        assert (im.axes, im.shape) == ("tczyx", (1, 2, 3, Y, X))
        for k, page in enumerate(_two_channels_three_z()):              # page k: channel, z
            np.testing.assert_array_equal(im[0, page["exposure"] - 1, page["z"] - 1], a[k])
        np.testing.assert_array_equal(plate.wells[1].fields[0].image[0, 1, 2], b[5])
        meta = im.metadata
        assert [c.name for c in meta.channels] == ["Exp1Cam1", "Exp2Cam1"]
        assert meta.pixel_sizes["x"] == micro_reader.Quantity(0.65, "micrometer")
        assert meta.pixel_sizes["z"].value == pytest.approx(0.4)
        # the field's centre relative to the well's centre, y as written: first pixel
        assert meta.position["x"].value == pytest.approx(100 - (X - 1) / 2 * 0.65)
        assert meta.position["y"].value == pytest.approx(-200 - (Y - 1) / 2 * 0.65)
        assert meta.coordinate_system == "well-A2"
    with micro_reader.open(str(tmp_path)) as f:                         # the folder too
        assert [w.name for w in f.plates[0].wells] == ["A2", "B5"]
    with micro_reader.open(str(tmp_path / "002005000.flex"), multifile=False) as f:
        assert [w.name for w in f.plates[0].wells] == ["B5"] and len(f.images) == 1


def test_fields_split_over_files_join_their_well_and_barcodes_make_plates(tmp_path):
    _write(str(tmp_path / "001001003.flex"), row=1, col=1, pages=_two_channels_three_z(field=3))
    _write(str(tmp_path / "001001001.flex"), row=1, col=1, pages=_two_channels_three_z(field=1))
    _write(str(tmp_path / "003016004.flex"), row=3, col=16, barcode="P0", rows=16, cols=24,
           pages=_two_channels_three_z(field=4))
    with micro_reader.open(str(tmp_path)) as f:
        assert [p.name for p in f.plates] == ["P0", "P1"]                # by name
        assert [(w.name, [x.index for x in w.fields]) for w in f.plates[1].wells] == [
            ("A1", [0, 2])]                                             # Sublayout - 1
        assert [im.index["plate"] for im in f.images] == [0, 1, 1]


def test_a_flex_folder_is_one_group(tmp_path):
    for col in (1, 2, 3):
        _write(str(tmp_path / f"00100{col}000.flex"), row=1, col=col, pages=_two_channels_three_z())
    g = micro_reader.group_files(str(tmp_path))
    assert len(g.groups) == 1 and not g.singles
    assert g.groups[0].format == "flex" and len(g.groups[0].members) == 2


@pytest.mark.parametrize("edit, error, match", [
    # channel 2 has no plane 3: ragged
    (dict(pages=_two_channels_three_z()[:-1]), micro_reader.UnsupportedFormatError, "do not fill"),
    (dict(pages=_two_channels_three_z(), factor="2.5"), micro_reader.UnsupportedFormatError,
     "ImageFactor"),
    (dict(pages=_two_channels_three_z(), row=0), micro_reader.CorruptFileError, "count from 1"),
])
def test_what_it_cannot_read_is_refused(tmp_path, edit, error, match):
    _write(str(tmp_path / "001001000.flex"), **{"row": 1, "col": 1, **edit})
    with pytest.raises(error, match=match):
        micro_reader.open(str(tmp_path / "001001000.flex"))


# -- the real samples ---------------------------------------------------------------

DATA = os.environ.get("MICROSCOPY_DATA", os.path.expanduser("~/Desktop/data/microscopy"))
FLEX = os.path.join(DATA, "plates", "flex")


def _sample(*parts):
    path = os.path.join(FLEX, *parts)
    if not os.path.exists(path):
        pytest.skip(f"sample not found: {path} (set MICROSCOPY_DATA)")
    return path


def test_idr0001_measurement_folder_is_a_plate_of_two_wells():
    """idr0001 (OME's Opera Flex sample, CC BY 4.0): a measurement folder of
    two wells, 2 channels x 16 z each, as Bio-Formats 8.0.1 reads it (Well A-2,
    A-3; channels Exp1Cam1, Exp2Cam1; plate X_110331_S13, 8 x 12)."""
    folder = _sample("ome-idr0001", "X_110331_S13", "Meas_01(2011-03-31_11-53-03)")
    with micro_reader.open(os.path.join(folder, "001002000.flex")) as f:
        (plate,) = f.plates
        assert (plate.name, len(plate.row_names), len(plate.column_names)) == ("X_110331_S13", 8, 12)
        assert [w.name for w in plate.wells] == ["A2", "A3"]
        im = plate.wells[0].fields[0].image
        assert (im.axes, im.shape, im.dtype) == ("tczyx", (1, 2, 16, 1040, 1376), np.uint16)
        assert [c.name for c in im.metadata.channels] == ["Exp1Cam1", "Exp2Cam1"]
        assert im.metadata.pixel_sizes["z"].value == pytest.approx(0.4)
        with tifffile.TiffFile(os.path.join(folder, "001002000.flex")) as tif:
            np.testing.assert_array_equal(im[0, 1, 1], tif.pages[3].asarray())


def test_idr0145_three_files_of_two_plates():
    """idr0145 (CC BY 4.0): three .flex files, one field each (Sublayout 3, 4,
    2), of plates rad53-MMS-000 and rad53-MMS-240."""
    with micro_reader.open(_sample("idr0145")) as f:
        assert [(p.name, [(w.name, [x.index for x in w.fields]) for w in p.wells])
                for p in f.plates] == [("rad53-MMS-000", [("A1", [2])]),
                                       ("rad53-MMS-240", [("C16", [3]), ("O7", [1])])]
        assert all(im.shape == (1, 2, 1, 1020, 1370) for im in f.images)


def test_bia_s_biad3_one_well_of_28_fields():
    """BIA S-BIAD3 (CC0): one well, 28 fields x 4 channels; Bio-Formats reads
    the fields as 28 timepoints, but each has its own position."""
    path = _sample("bia-S-BIAD3", "validationscreen3", "Plate1", "C3", "003003.flex")
    with micro_reader.open(path) as f:
        (well,) = f.plates[0].wells
        assert well.name == "C3" and len(well.fields) == 28
        assert {im.shape for im in f.images} == {(1, 4, 1, 499, 667)}
        positions = {(im.metadata.position["x"].value, im.metadata.position["y"].value)
                     for im in f.images}
        assert len(positions) == 28
        with tifffile.TiffFile(path) as tif:
            np.testing.assert_array_equal(well.fields[27].image[0, 3, 0], tif.pages[111].asarray())
