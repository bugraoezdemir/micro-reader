"""PerkinElmer Columbus exports (backends/columbus.py, read by backends/harmony.py):
synthetic exports, then the real samples (plates/columbus)."""
from __future__ import annotations

import os

import numpy as np
import pytest

tifffile = pytest.importorskip("tifffile")

import micro_reader  # noqa: E402

RNG = np.random.default_rng(31)
Y, X = 10, 14
NS = "http://www.perkinelmer.com/Columbus"


def _write(folder, *, wells=((1, 1), (2, 3)), channels=2, planes=3, nested=True,
           page_shape=(Y, X)):
    """A Columbus export of one field per well, one multi-page TIFF per well
    (planes in channel-fastest order, as the OME sample), its image index
    pointing at pages by BufferNo.  *nested*: the measurement index refers to
    ``<measurement>/ImageIndex.ColumbusIDX.xml`` in a subfolder (as Columbus
    writes it); otherwise everything sits in one folder with the reference
    left pointing where it no longer is (an export flattened by hand).
    -> (measurement index, image index, {(row, col, channel, plane): array})."""
    sub = os.path.join(folder, "2024-01-23T120146+1100[7776]") if nested else folder
    os.makedirs(sub, exist_ok=True)
    data, entries = {}, []
    for r, c in wells:
        name = f"{r:03d}{c:03d}-1.tif"
        with tifffile.TiffWriter(os.path.join(sub, name)) as w:
            k = 0
            for p in range(1, planes + 1):
                for ch in range(1, channels + 1):
                    a = data[(r, c, ch, p)] = RNG.integers(0, 4000, page_shape).astype("u2")
                    w.write(a, photometric="minisblack", metadata=None)
                    entries.append(
                        f'<Image Version="1"><id>{r}{c}-{ch}{p}</id><State>Ok</State>'
                        f'<URL BufferNo="{k}">{name}</URL><Row>{r}</Row><Col>{c}</Col>'
                        f"<FieldID>1</FieldID><PlaneID>{p}</PlaneID><TimepointID>1</TimepointID>"
                        f"<ChannelID>{ch}</ChannelID><ChannelName>ch{ch}</ChannelName>"
                        f'<ImageResolutionX Unit="m">9.9e-07</ImageResolutionX>'
                        f'<ImageResolutionY Unit="m">9.9e-07</ImageResolutionY>'
                        f"<ImageSizeX>{X}</ImageSizeX><ImageSizeY>{Y}</ImageSizeY>"
                        f'<PositionX Unit="m">0</PositionX><PositionY Unit="m">0</PositionY>'
                        f'<PositionZ Unit="m">{(p - 1) * 2e-5}</PositionZ>'
                        f'<MeasurementTimeOffset Unit="s">0</MeasurementTimeOffset>'
                        f"<OrientationMatrix>[[1,0,0],[0,-1,0],[0,0,1]]</OrientationMatrix></Image>")
                    k += 1
    image_index = os.path.join(sub, "ImageIndex.ColumbusIDX.xml")
    with open(image_index, "w", encoding="utf-8") as fh:
        fh.write(f'<?xml version="1.0" encoding="UTF-8" ?><EvaluationInputData xmlns="{NS}" '
                 f'Version="1"><User>Undefined</User><Plates><Plate><PlateID>P7</PlateID>'
                 f"<Name>screen 7</Name><PlateTypeName>6 well</PlateTypeName>"
                 f"<PlateRows>2</PlateRows><PlateColumns>3</PlateColumns></Plate></Plates>"
                 f"<Images>{''.join(entries)}</Images></EvaluationInputData>")
    with open(os.path.join(sub, "ImageIndex.ColumbusIDX.csv"), "w", encoding="utf-8") as fh:
        fh.write("Row\tColumn\n")
    measurement_index = os.path.join(folder, "MeasurementIndex.ColumbusIDX.xml")
    with open(measurement_index, "w", encoding="utf-8") as fh:
        fh.write(f'<?xml version="1.0" encoding="UTF-8" ?><ColumbusMeasurementIndex xmlns="{NS}" '
                 f'Version="2"><MeasurementReferences><MeasurementReference>'
                 f'<Reference Class="IMAGEINDEX" Format="Native XML">'
                 f"2024-01-23T120146+1100[7776]/ImageIndex.ColumbusIDX.xml</Reference>"
                 f"</MeasurementReference></MeasurementReferences></ColumbusMeasurementIndex>")
    return measurement_index, image_index, data


@pytest.mark.parametrize("nested", [True, False])
def test_any_index_or_folder_opens_the_plate(tmp_path, nested):
    measurement, image_index, data = _write(str(tmp_path), nested=nested)
    sub = os.path.dirname(image_index)
    for path in (image_index, measurement, str(tmp_path), sub,
                 os.path.join(sub, "ImageIndex.ColumbusIDX.csv")):
        with micro_reader.open(path) as f:
            assert (f.backend, f.format) == ("columbus", "columbus"), path
            assert os.path.normcase(f.path) == os.path.normcase(image_index)
            (plate,) = f.plates
            assert plate.name == "screen 7" and (len(plate.row_names), len(plate.column_names)) == (2, 3)
            assert [w.name for w in plate.wells] == ["A1", "B3"]


def test_pages_are_read_by_buffer_number(tmp_path):
    _measurement, image_index, data = _write(str(tmp_path))
    with micro_reader.open(image_index) as f:
        im = f.plates[0].wells[1].fields[0].image
        assert (im.axes, im.shape) == ("tczyx", (1, 2, 3, Y, X))
        for ch in (1, 2):
            for p in (1, 2, 3):
                np.testing.assert_array_equal(im[0, ch - 1, p - 1], data[(2, 3, ch, p)])
        np.testing.assert_array_equal(im[0, :, 1:, 2:7, 3:9],
                                      np.stack([np.stack([data[(2, 3, ch, p)][2:7, 3:9]
                                                          for p in (2, 3)]) for ch in (1, 2)]))
        meta = im.metadata
        assert [c.name for c in meta.channels] == ["ch1", "ch2"]
        assert meta.pixel_sizes["z"].value == pytest.approx(20.0)
        assert meta.coordinate_system == "well-B3"


def test_a_measurement_index_without_its_image_index_is_an_error(tmp_path):
    measurement, image_index, _data = _write(str(tmp_path))
    os.remove(image_index)
    with pytest.raises(micro_reader.CorruptFileError, match="ImageIndex.ColumbusIDX.xml"):
        micro_reader.open(measurement)


def test_an_export_is_one_group(tmp_path):
    measurement, image_index, _data = _write(str(tmp_path))
    g = micro_reader.group_files(str(tmp_path))
    assert len(g.groups) == 1 and not g.singles
    (group,) = g.groups
    assert group.format == "columbus" and len(group.members) == 2
    assert os.path.normcase(group.main) == os.path.normcase(image_index)


def test_a_page_of_another_size_than_the_index_says_is_an_error(tmp_path):
    _measurement, image_index, _data = _write(str(tmp_path), page_shape=(Y + 2, X))
    with pytest.raises(micro_reader.CorruptFileError, match="the index"):
        micro_reader.open(image_index)


# -- the real samples ---------------------------------------------------------------

DATA = os.environ.get("MICROSCOPY_DATA", os.path.expanduser("~/Desktop/data/microscopy"))
COLUMBUS = os.path.join(DATA, "plates", "columbus")


def _sample(*parts):
    path = os.path.join(COLUMBUS, *parts)
    if not os.path.exists(path):
        pytest.skip(f"sample not found: {path} (set MICROSCOPY_DATA)")
    return path


def test_ome_sample_two_wells_of_six_page_tiffs():
    """OME's Columbus sample (Zenodo 6327496, CC BY 4.0), an export flattened
    into one folder: wells A1, B1 of a 2 x 3 plate, 2 channels x 3 z in one
    6-page TIFF per well, as Bio-Formats 8.0.1 reads it."""
    folder = _sample("ome-zenodo-6327496", "tif")
    with micro_reader.open(os.path.join(folder, "MeasurementIndex.ColumbusIDX.xml")) as f:
        (plate,) = f.plates
        assert (len(plate.row_names), len(plate.column_names)) == (2, 3)
        assert [w.name for w in plate.wells] == ["A1", "B1"]
        im = plate.wells[1].fields[0].image
        assert (im.axes, im.shape, im.dtype) == ("tczyx", (1, 2, 3, 1024, 1360), np.uint16)
        assert [c.name for c in im.metadata.channels] == ["Brightfield", "Alexa 488"]
        assert im.metadata.pixel_sizes["z"].value == pytest.approx(20.0)
        with tifffile.TiffFile(os.path.join(folder, "002001-1.tif")) as tif:
            np.testing.assert_array_equal(im[0, 1, 2], tif.pages[5].asarray())


def test_zenodo_10627948_harmony_archive_imported_into_columbus():
    """Zenodo 10627948 (CC BY 4.0): Harmony archives imported into Columbus and
    exported, one TIFF per plane in the measurement's subfolder: wells A1, B2
    of a 1536-well plate, 4 t x 3 c x 6 z, as Bio-Formats 8.0.1 reads it."""
    folder = _sample("zenodo-10627948", "ImportedHarmonyArchives[3602]",
                     "240113_TRB_w2c3t4f1p6[7271]", "2024-01-23T120146+1100[7776]")
    with micro_reader.open(folder) as f:
        (plate,) = f.plates
        assert plate.name == "240113_TRB_w2c3t4f1p6"
        assert (len(plate.row_names), len(plate.column_names)) == (32, 48)
        assert [w.name for w in plate.wells] == ["A1", "B2"]
        assert {im.shape for im in f.images} == {(4, 3, 6, 1080, 1080)}
        assert [c.name for c in f.images[0].metadata.channels] == ["Alexa 488", "Alexa 568",
                                                                   "Brightfield"]
