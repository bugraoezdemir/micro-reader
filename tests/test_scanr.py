"""Olympus scanR plates (backends/scanr.py, read through backends/companion.py):
a synthetic plate folder, then the real sample (plates/scanr)."""
from __future__ import annotations

import os
import warnings

import numpy as np
import pytest

tifffile = pytest.importorskip("tifffile")

import micro_reader  # noqa: E402

RNG = np.random.default_rng(41)
Y, X = 12, 16
NS = "http://www.openmicroscopy.org/Schemas/OME/2013-06"
CHANNELS = ("TransCon", "488nm")


def _write(folder, *, wells=(("A", 1, 1), ("B", 3, 27)), positions=2, planned=(("C", 5, 53),),
           rows=16, columns=24, companion=True):
    """A scanR plate folder: *wells* (letters, column, well number) have files,
    *planned* wells are in the companion without files.  -> (descriptor,
    {(well name, position, channel): array})."""
    data_dir = os.path.join(folder, "data")
    os.makedirs(data_dir, exist_ok=True)
    data, images = {}, []
    for letters, col, number in tuple(wells) + tuple(planned):
        for p in range(1, positions + 1):
            tiffdata = []
            for c, ch in enumerate(CHANNELS):
                name = f"{letters}{col}--W{number:05d}--P{p:05d}--Z00000--T00000--{ch}.tif"
                if (letters, col, number) in wells:
                    a = data[(f"{letters}{col}", p, ch)] = RNG.integers(0, 4000, (Y, X)).astype("u2")
                    # scanR's members: a BinaryOnly stub pointing to the companion
                    stub = (f'<OME xmlns="{NS}" UUID="urn:uuid:{number}-{p}-{c}"><BinaryOnly '
                            f'MetadataFile="metadata.ome.xml" UUID="urn:uuid:scanr-set"/></OME>')
                    tifffile.imwrite(os.path.join(data_dir, name), a, photometric="minisblack",
                                     metadata=None, description=stub)
                tiffdata.append(f'<TiffData FirstC="{c}" FirstT="0" FirstZ="0">'
                                f'<UUID FileName="{name}">urn:uuid:{number}-{p}-{c}</UUID></TiffData>')
            chans = "".join(f'<Channel ID="Channel:W{number}P{p}:{c}" Name="{ch}"/>'
                            for c, ch in enumerate(CHANNELS))
            images.append(
                f'<Image ID="Image:W{number}P{p}" Name="Well {number}, Position {p} ({letters}{col})">'
                f'<Pixels DimensionOrder="XYCTZ" ID="Pixels:W{number}P{p}" PhysicalSizeX="0.065" '
                f'PhysicalSizeY="0.065" SizeC="{len(CHANNELS)}" SizeT="1" SizeX="{X}" SizeY="{Y}" '
                f'SizeZ="1" Type="uint16">{chans}{"".join(tiffdata)}</Pixels></Image>')
    if companion:
        with open(os.path.join(data_dir, "metadata.ome.xml"), "w", encoding="utf-8") as fh:
            fh.write(f'<?xml version="1.0" encoding="UTF-8"?><OME xmlns="{NS}" '
                     f'UUID="urn:uuid:scanr-set" Creator="Olympus, scanR 3.2.0">{"".join(images)}</OME>')
    descriptor = os.path.join(folder, "experiment_descriptor.xml")
    with open(descriptor, "w", encoding="utf-8") as fh:
        fh.write('<?xml version="1.0" encoding="UTF-8"?><LVData xmlns="http://www.ni.com/LVData">'
                 "<Version>16.0</Version><Cluster><Name>Preferences Out</Name>"
                 "<String><Name>plate name</Name><Val>screen</Val></String>"
                 f"<Cluster><Name>format typedef</Name><U16><Name>Rows</Name><Val>{rows}</Val></U16>"
                 f"<U16><Name>Columns</Name><Val>{columns}</Val></U16></Cluster>"
                 "</Cluster></LVData>")
    for side in ("experiment_descriptor.dat", "AcquisitionLog.dat"):
        with open(os.path.join(folder, side), "w", encoding="utf-8") as fh:
            fh.write("VERSION\tID=0\r\n")
    return descriptor, data


def test_a_scanr_folder_is_a_plate_and_every_part_opens_it(tmp_path):
    descriptor, data = _write(str(tmp_path))
    data_dir = os.path.join(str(tmp_path), "data")
    member = os.path.join(data_dir, "B3--W00027--P00002--Z00000--T00000--488nm.tif")
    for path in (descriptor, str(tmp_path), data_dir, os.path.join(data_dir, "metadata.ome.xml"),
                 os.path.join(str(tmp_path), "AcquisitionLog.dat"), member):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            with micro_reader.open(path) as f:
                assert (f.backend, f.format) == ("scanr", "scanr"), path
                assert os.path.normcase(f.path) == os.path.normcase(descriptor)
                (plate,) = f.plates
                assert plate.name == "screen" and (len(plate.row_names), len(plate.column_names)) == (16, 24)
                assert [(w.name, [x.index for x in w.fields]) for w in plate.wells] == [
                    ("A1", [0, 1]), ("B3", [0, 1])]               # position - 1
        missing = [w for w in caught if issubclass(w.category, micro_reader.MissingImagesWarning)]
        assert len(missing) == 1 and "4 image files are missing" in str(missing[0].message)
    with micro_reader.open(member, multifile=False) as f:              # the one TIFF
        assert f.backend == "tiff" and len(f.images) == 1


def test_fields_read_their_files(tmp_path):
    descriptor, data = _write(str(tmp_path), planned=())
    with micro_reader.open(descriptor) as f:
        im = f.plates[0].wells[1].fields[1].image
        assert im.index == {"plate": 0, "acquisition": 0, "well": 1, "field": 1}
        assert (im.axes, im.shape) == ("tczyx", (1, 2, 1, Y, X))
        assert [c.name for c in im.metadata.channels] == list(CHANNELS)
        for c, ch in enumerate(CHANNELS):
            np.testing.assert_array_equal(im[0, c, 0], data[("B3", 2, ch)])


def test_a_plate_folder_without_the_ome_metadata_is_refused(tmp_path):
    descriptor, _data = _write(str(tmp_path), companion=False)
    with pytest.raises(micro_reader.UnsupportedFormatError, match="metadata.ome.xml"):
        micro_reader.open(descriptor)


def test_a_scanr_folder_is_one_group(tmp_path):
    descriptor, _data = _write(str(tmp_path))
    g = micro_reader.group_files(str(tmp_path))
    assert len(g.groups) == 1 and not g.singles
    (group,) = g.groups
    assert group.format == "scanr" and os.path.normcase(group.main) == os.path.normcase(descriptor)
    assert len(group.members) == 8


# -- the real sample ----------------------------------------------------------------

DATA = os.environ.get("MICROSCOPY_DATA", os.path.expanduser("~/Desktop/data/microscopy"))


def test_bia_s_biad2409_well_a1_of_a_384_well_plate():
    """BIA S-BIAD2409 (CC0): a 384-well scanR plate of which well A1's six
    positions (3 channels each) are here; the companion plans 2304 images.
    Bio-Formats 8.0.1 reads well 1 as 6 fields too (adding a fourth channel,
    'Trans', that has no files)."""
    folder = os.path.join(DATA, "plates", "scanr", "bia-S-BIAD2409", "plate_001")
    if not os.path.exists(folder):
        pytest.skip(f"sample not found: {folder} (set MICROSCOPY_DATA)")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with micro_reader.open(folder) as f:
            (plate,) = f.plates
            assert (len(plate.row_names), len(plate.column_names)) == (16, 24)
            (well,) = plate.wells
            assert well.name == "A1" and [x.index for x in well.fields] == list(range(6))
            im = well.fields[2].image
            assert (im.axes, im.shape, im.dtype) == ("tczyx", (1, 3, 1, 2048, 2048), np.uint16)
            assert [c.name for c in im.metadata.channels] == ["TransCon", "561nm", "488nm"]
            assert im.metadata.pixel_sizes["x"].value == pytest.approx(0.065)
            with tifffile.TiffFile(os.path.join(folder, "data",
                                                "A1--W00001--P00003--Z00000--T00000--561nm.tif")) as tif:
                np.testing.assert_array_equal(im[0, 1, 0], tif.pages[0].asarray())
    missing = [w for w in caught if issubclass(w.category, micro_reader.MissingImagesWarning)]
    assert len(missing) == 1
