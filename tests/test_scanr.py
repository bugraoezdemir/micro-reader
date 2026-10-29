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


# -- older plates: no data/metadata.ome.xml --------------------------------------------

def _write_old(folder, *, wells=(("B", 2, 26), ("C", 5, 53)), positions=2, labelled=True,
               plate_format="384 Well", custom=None, log=True, skip=(), table=None,
               values=(0, 4000)):
    """An older scanR plate folder: plain TIFFs named [<well>]--W<n>--P<n>--Z0--T0--<ch>.tif,
    a LabVIEW descriptor in Latin-1 (no encoding named) with the plate format (or
    only a *custom* (rows, columns) format), channel configs (one idle), and an
    acquisition log with stage positions (*log*).  *skip*: (well, position, channel)
    without a file; *table*: the well table's {number: name} (None: numbers only);
    *values*: the pixels' range.  -> (descriptor, {(well number, position, channel):
    array})."""
    data_dir = os.path.join(folder, "data")
    os.makedirs(data_dir, exist_ok=True)
    data, lines = {}, ["VERSION\tID=0"]
    for letters, col, number in wells:
        for p in range(1, positions + 1):
            for ch in CHANNELS:
                if (number, p, ch) in skip:
                    continue
                prefix = f"{letters}{col}" if labelled else ""
                name = f"{prefix}--W{number:05d}--P{p:05d}--Z00000--T00000--{ch}.tif"
                a = data[(number, p, ch)] = RNG.integers(*values, (Y, X)).astype("u2")
                tifffile.imwrite(os.path.join(data_dir, name), a, photometric="minisblack",
                                 metadata=None, software="National Instruments IMAQ")
            at = (f"\tIMAGEX={1000.0 * number:.6f}\tIMAGEY={500.0 + 10 * p:.6f}\tIMAGEZ=8000.5"
                  if log else "")
            lines.append(f"IMAGEPOS\tW={number}\tP={p}\tT=0\tZ=0{at}\tTIME=1.0")
    channels = "".join(
        f"<Cluster><Name>channel config</Name><NumElts>2</NumElts>"
        f"<String><Name>name</Name><Val>{ch}</Val></String>"
        f"<Boolean><Name>idle</Name><Val>{1 if ch == 'Blank' else 0}</Val></Boolean></Cluster>"
        for ch in ("488nm", "Blank", "TransCon"))            # descriptor order, one idle
    fmt = (f"<Cluster><Name>format typedef</Name><String><Name>Format description</Name>"
           f"<Val>{plate_format}</Val></String><U32><Name>Rows</Name><Val>16</Val></U32>"
           f"<U32><Name>Columns</Name><Val>24</Val></U32></Cluster>") if plate_format else ""
    own = (f"<I32><Name>custom row</Name><Val>{custom[0]}</Val></I32>"
           f"<I32><Name>custom column</Name><Val>{custom[1]}</Val></I32>") if custom else ""
    pairs = "".join(f"<U32><Name>well selection table + cDNA</Name><Val>{n}</Val></U32>"
                    f"<String><Name>well selection table + cDNA</Name><Val>{(table or {}).get(n, '')}</Val></String>"
                    for n in sorted({w for _l, _c, w in wells} | set(table or {})))
    table_xml = (f"<Array><Name>well selection table + cDNA</Name><Dimsize>{len(wells)}</Dimsize>"
                 f"<Dimsize>2</Dimsize>{pairs}</Array>")
    descriptor = os.path.join(folder, "experiment_descriptor.xml")
    with open(descriptor, "wb") as fh:
        fh.write(("<?xml version='1.0' standalone='yes' ?><LVData><Version>8.5.1</Version>"
                  "<Cluster><Name>Preferences Out</Name>"
                  "<String><Name>plate name</Name><Val>old screen</Val></String>"
                  "<U16><Name>interpolation spacing [µm]</Name><Val>1600</Val></U16>"
                  f"<Array><Name>multiple_channel_typedef</Name><Dimsize>3</Dimsize>{channels}</Array>"
                  "<DBL><Name>conversion factor um/pixel</Name><Val>0.64500\t</Val></DBL>"
                  f"{table_xml}{fmt}{own}</Cluster></LVData>").encode("latin-1"))
    with open(os.path.join(folder, "AcquisitionLog.dat"), "w", encoding="latin-1",
              newline="\r\n") as fh:
        fh.write("\n".join(lines) + "\n")
    with open(os.path.join(folder, "experiment_descriptor.dat"), "wb") as fh:
        fh.write(b"\0\0")
    return descriptor, data


def test_an_older_plate_is_read_from_its_file_names(tmp_path):
    """scanR before data/metadata.ome.xml: the plate from the descriptor (Latin-1,
    no encoding named), each field's planes from the file names, its stage
    position from the acquisition log; every part opens it."""
    descriptor, data = _write_old(str(tmp_path))
    data_dir = os.path.join(str(tmp_path), "data")
    member = os.path.join(data_dir, "C5--W00053--P00002--Z00000--T00000--488nm.tif")
    for path in (descriptor, str(tmp_path), data_dir, member,
                 os.path.join(str(tmp_path), "AcquisitionLog.dat")):
        with micro_reader.open(path) as f:
            assert (f.backend, f.format) == ("scanr", "scanr"), path
            assert os.path.normcase(f.path) == os.path.normcase(descriptor)
            (plate,) = f.plates
            assert plate.name == "old screen"
            assert (len(plate.row_names), len(plate.column_names)) == (16, 24)
            assert [(w.name, [x.index for x in w.fields]) for w in plate.wells] == [
                ("B2", [0, 1]), ("C5", [0, 1])]
    with micro_reader.open(descriptor) as f:
        assert f.coordinate_systems[0].id == "stage"
        im = f.plates[0].wells[1].fields[1].image
        assert im.index == {"plate": 0, "acquisition": 0, "well": 1, "field": 1}
        assert (im.axes, im.shape, im.dtype) == ("tczyx", (1, 2, 1, Y, X), np.uint16)
        # the descriptor's order of the channels that are not idle
        assert [c.name for c in im.metadata.channels] == ["488nm", "TransCon"]
        assert im.metadata.pixel_sizes["x"].value == pytest.approx(0.645)
        pos = {a: q.value for a, q in im.metadata.position.items()}
        assert pos == {"x": 53000.0, "y": 520.0, "z": 8000.5}
        assert im.metadata.coordinate_system == "stage"
        for c, ch in enumerate(["488nm", "TransCon"]):
            np.testing.assert_array_equal(im[0, c, 0], data[(53, 2, ch)])
    with micro_reader.open(member, multifile=False) as f:              # the one TIFF
        assert f.backend == "tiff" and len(f.images) == 1


def test_an_older_plate_without_well_names_places_wells_by_number(tmp_path):
    """scanR 2007 names files --W<n>--... only; with no plate format but a custom
    one (32 x 12, a cell array) the numbers count along its rows."""
    descriptor, _data = _write_old(str(tmp_path), wells=(("", 0, 2), ("", 0, 13), ("", 0, 384)),
                                   positions=1, labelled=False, plate_format=None,
                                   custom=(32, 12), log=False)
    with micro_reader.open(descriptor) as f:
        (plate,) = f.plates
        assert (len(plate.row_names), len(plate.column_names)) == (32, 12)
        assert [w.name for w in plate.wells] == ["A2", "B1", "AF12"]
        assert f.coordinate_systems == () and f.images[0].metadata.position == {}


def test_an_older_plate_with_no_format_is_the_smallest_that_holds_its_wells(tmp_path):
    descriptor, _data = _write_old(str(tmp_path), wells=(("", 0, 100),), positions=1,
                                   labelled=False, plate_format=None)
    with pytest.warns(micro_reader.PlateLayoutWarning, match="no plate format.*16 x 24"),             micro_reader.open(descriptor) as f:
        (plate,) = f.plates
        assert (len(plate.row_names), len(plate.column_names)) == (16, 24)
        assert [w.name for w in plate.wells] == ["E4"]              # 100 = 4 x 24 + 4


def test_an_older_plate_whose_names_and_numbers_disagree_is_refused(tmp_path):
    descriptor, _data = _write_old(str(tmp_path), wells=(("B", 2, 27),), positions=1)
    with pytest.raises(micro_reader.CorruptFileError, match="numbered W00027.*B3"):
        micro_reader.open(descriptor)


def test_a_missing_plane_of_an_older_plate_reads_as_zeros(tmp_path):
    descriptor, _data = _write_old(str(tmp_path), skip={(26, 1, "TransCon")})
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with micro_reader.open(descriptor) as f:
            im = f.plates[0].wells[0].fields[0].image
            assert not im[0, 1, 0].any()
    missing = [w for w in caught if issubclass(w.category, micro_reader.MissingImagesWarning)]
    assert len(missing) == 1 and "1 image files are missing" in str(missing[0].message)


def test_opening_an_older_plate_lists_each_folder_once(tmp_path, monkeypatch):
    """The plate folder (data folder, acquisition log) and the data folder (companion,
    plane files) are each listed once (real-data matrix G4: idr0009 listed both twice)."""
    from micro_reader.backends import scanr
    descriptor, _data = _write_old(str(tmp_path))
    listed = []
    real = os.scandir
    monkeypatch.setattr(scanr.os, "scandir", lambda p: listed.append(os.path.normcase(p)) or real(p))
    with micro_reader.open(descriptor, backend="scanr") as f:
        assert len(f.images) == 4
    assert sorted(listed) == sorted({os.path.normcase(str(tmp_path)),
                                     os.path.normcase(os.path.join(str(tmp_path), "data"))})


def test_an_older_plate_field_reads_through_as_axes(tmp_path):
    """Its stage position is not the image's position in the file (``as_axes`` raised
    TypeError: the field kept its stage position where every image keeps its number)."""
    descriptor, data = _write_old(str(tmp_path))
    with micro_reader.open(descriptor) as f:
        im = f.plates[0].wells[0].fields[0].image
        view = im.as_axes("tczyx")
        np.testing.assert_array_equal(np.asarray(view[0, :, 0]), np.asarray(im[0, :, 0]))
        assert im.metadata.position and im.metadata.coordinate_system == "stage"


def test_an_older_scanr_folder_is_one_group(tmp_path):
    descriptor, _data = _write_old(str(tmp_path))
    g = micro_reader.group_files(str(tmp_path))
    assert len(g.groups) == 1 and not g.singles
    (group,) = g.groups
    assert os.path.normcase(group.main) == os.path.normcase(descriptor)


def test_the_well_table_names_check_the_layout(tmp_path):
    """Since about 2015 the well table pairs each number with its name: a name
    that another well's number has is refused; matching names pass."""
    descriptor, _data = _write_old(str(tmp_path / "ok"), table={26: "B2", 53: "C5"})
    with micro_reader.open(descriptor) as f:
        assert [w.name for w in f.plates[0].wells] == ["B2", "C5"]
    descriptor, _data = _write_old(str(tmp_path / "bad"), wells=(("", 0, 26),), positions=1,
                                   labelled=False, table={26: "B3"})
    with pytest.raises(micro_reader.CorruptFileError,
                       match="names well number 26 'B3'.*it is well B2"):
        micro_reader.open(descriptor)


def test_a_format_description_that_disagrees_warns(tmp_path):
    descriptor, _data = _write_old(str(tmp_path), plate_format="96 Well")
    with pytest.warns(micro_reader.PlateLayoutWarning, match="16 x 24.*'96 Well'"), \
            micro_reader.open(descriptor) as f:
        assert (len(f.plates[0].row_names), len(f.plates[0].column_names)) == (16, 24)


def test_twelve_bit_values_stored_with_the_top_bit_are_read_without_it(tmp_path):
    """zenodo 3357662: every pixel is 0x8000 + a 12-bit value (Bio-Formats keeps
    the low 12 bits of every scanR pixel); the top bit is cleared, with one
    StoredValuesWarning."""
    descriptor, data = _write_old(str(tmp_path), values=(0x8000, 0x9000))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with micro_reader.open(descriptor) as f:
            im = f.plates[0].wells[0].fields[1].image
            np.testing.assert_array_equal(im[0, 0, 0], data[(26, 2, "488nm")] - 0x8000)
            assert int(im[:].max()) <= 0x0FFF
    told = [w for w in caught if issubclass(w.category, micro_reader.StoredValuesWarning)]
    assert len(told) == 1 and "top bit" in str(told[0].message)


def test_sixteen_bit_values_are_read_as_stored(tmp_path):
    """A 16-bit camera's plane with a bright background also has the top bit
    set, but uses bits 12-14 too: not a 12-bit plate, read as stored."""
    descriptor, data = _write_old(str(tmp_path), values=(0x8000, 0x10000))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with micro_reader.open(descriptor) as f:
            im = f.plates[0].wells[0].fields[0].image
            np.testing.assert_array_equal(im[0, 0, 0], data[(26, 1, "488nm")])
    assert not [w for w in caught if issubclass(w.category, micro_reader.StoredValuesWarning)]


def test_a_plate_folder_without_a_data_folder_is_refused(tmp_path):
    descriptor, _data = _write(str(tmp_path))
    import shutil
    shutil.rmtree(os.path.join(str(tmp_path), "data"))
    with pytest.raises(micro_reader.CorruptFileError, match="no data folder"):
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


def test_ome_idr0009_an_older_cell_array_plate():
    """OME sample idr0009 (scanR 2007): files --W<n>--P00001--..., a custom 32 x 12
    format (Bio-Formats lays it out as 16 x 24, from the 384 well numbers), three
    channels, no pixel size, no positions in the log."""
    folder = os.path.join(DATA, "plates", "scanr", "ome-idr0009", "0307-10--2007-05-30")
    if not os.path.exists(folder):
        pytest.skip(f"sample not found: {folder} (set MICROSCOPY_DATA)")
    with micro_reader.open(folder) as f:
        (plate,) = f.plates
        assert plate.name == "2007-05-30"
        assert (len(plate.row_names), len(plate.column_names)) == (32, 12)
        assert len(plate.wells) == 357 and plate.wells[0].name == "A2"
        im = plate.wells[0].fields[0].image
        assert (im.shape, im.dtype) == ((1, 3, 1, 1024, 1344), np.uint16)
        assert [c.name for c in im.metadata.channels] == ["nucleus-dapi", "vsvg-cfp", "pm-647"]
        assert im.metadata.pixel_sizes == {} and im.metadata.position == {}
        with tifffile.TiffFile(os.path.join(
                folder, "data", "--W00002--P00001--Z00000--T00000--pm-647.tif")) as tif:
            np.testing.assert_array_equal(im[0, 2, 0], tif.pages[0].asarray())


def test_zenodo_3357662_an_older_384_well_plate_with_positions():
    """Zenodo 3357662 (scanR 2015, a course's siRNA screen): 42 wells x 9 fields,
    one channel, 0.645 um, stage positions from AcquisitionLog.dat; its descriptor
    holds a Latin-1 'µ' without naming an encoding; its well table names 42 wells."""
    folder = os.path.join(DATA, "plates", "scanr", "zenodo-3357662")
    if not os.path.exists(folder):
        pytest.skip(f"sample not found: {folder} (set MICROSCOPY_DATA)")
    with pytest.warns(micro_reader.StoredValuesWarning), micro_reader.open(folder) as f:
        (plate,) = f.plates
        assert (len(plate.row_names), len(plate.column_names)) == (16, 24)
        assert len(plate.wells) == 42 and sum(len(w.fields) for w in plate.wells) == 378
        well = plate.wells[0]
        assert well.name == "B2"
        im = well.fields[1].image
        assert [c.name for c in im.metadata.channels] == ["dapi"]
        assert im.metadata.pixel_sizes["x"].value == pytest.approx(0.645)
        pos = {a: q.value for a, q in im.metadata.position.items()}
        assert pos["x"] == pytest.approx(102879.0) and pos["y"] == pytest.approx(64253.0)
        with tifffile.TiffFile(os.path.join(
                folder, "data", "B2--W00026--P00002--Z00000--T00000--dapi.tif")) as tif:
            stored = tif.pages[0].asarray()
        # 12-bit values stored with the top bit set: read without it (Bio-Formats too)
        assert stored.min() >= 0x8000
        np.testing.assert_array_equal(im[0, 0, 0], stored & 0x7FFF)


def test_a_utf16_descriptor_is_read(tmp_path):
    """The descriptor is read as bytes and decoded by hand (UTF-8, else Latin-1); one
    written as UTF-16, with its byte-order mark, must still parse."""
    from micro_reader.backends.scanr import _root
    path = tmp_path / "experiment_descriptor.xml"
    path.write_bytes('<?xml version="1.0" encoding="UTF-16"?><Data><Name>µm</Name></Data>'
                     .encode("utf-16"))
    assert _root(str(path)).find("Name").text == "µm"