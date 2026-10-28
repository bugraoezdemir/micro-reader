"""Thermo Fisher Cellomics plates (backends/cellomics.py): synthetic .C01 files,
then the real sample (plates/cellomics)."""
from __future__ import annotations

import os
import struct
import warnings
import zlib

import numpy as np
import pytest

import micro_reader

RNG = np.random.default_rng(51)
Y, X = 10, 14
PPM = 1266520                                     # pixels per metre: 0.7896 um


def _c01(path, plane, ppm=PPM, bits=16):
    """A .C01 file: 4 bytes, then a zlib stream of the 52-byte header, the
    pixels (little-endian, top-down) and 64 trailing bytes, as the CX7 writes."""
    y, x = plane.shape
    header = struct.pack("<iiihhiiiiiiiii", 40, x, y, 1, bits, 0, 0, ppm, ppm, 0, 0, 31, 0, 0)
    data = header + plane.astype("<u2" if bits == 16 else "u1").tobytes() + bytes(64)
    with open(path, "wb") as fh:
        fh.write(b"\x00\x00\x00\x10" + zlib.compress(data))


def _plate(folder, wells=("A01", "B03"), fields=2, channels=2, plate="SCREEN_220517060001",
           skip=()):
    """A Cellomics plate folder with its database and logs; *skip*: (well,
    field, channel) not written.  -> {(well, field, channel): plane}."""
    data = {}
    for w in wells:
        for f in range(fields):
            for c in range(channels):
                plane = data[(w, f, c)] = RNG.integers(0, 4000, (Y, X)).astype("u2")
                if (w, f, c) not in skip:
                    _c01(os.path.join(folder, f"{plate}_{w}f{f:02d}d{c}.C01"), plane)
    for side in (f"{plate}.mdb", f"{plate}.spooled", f"{plate}_scan.log"):
        with open(os.path.join(folder, side), "wb") as fh:
            fh.write(b"\x00")
    return data


def test_a_folder_of_c01_files_is_a_plate(tmp_path):
    data = _plate(str(tmp_path))
    for path in (str(tmp_path / "SCREEN_220517060001_B03f01d1.C01"), str(tmp_path),
                 str(tmp_path / "SCREEN_220517060001.mdb"), str(tmp_path / "SCREEN_220517060001_scan.log")):
        with micro_reader.open(path) as f:
            assert f.backend == "cellomics", path
            (plate,) = f.plates
            assert plate.name == "SCREEN_220517060001"
            assert (len(plate.row_names), len(plate.column_names)) == (8, 12)   # at least 96
            assert [(w.name, [x.index for x in w.fields]) for w in plate.wells] == [
                ("A1", [0, 1]), ("B3", [0, 1])]
            im = plate.wells[1].fields[1].image
            assert (im.axes, im.shape) == ("tczyx", (1, 2, 1, Y, X))
            np.testing.assert_array_equal(im[0, 1, 0], data[("B03", 1, 1)])
            np.testing.assert_array_equal(im[0, :, 0, 2:7, 3:9],
                                          np.stack([data[("B03", 1, c)][2:7, 3:9] for c in (0, 1)]))
            assert im.metadata.pixel_sizes["x"].value == pytest.approx(1e6 / PPM)
    with micro_reader.open(str(tmp_path / "SCREEN_220517060001_B03f01d1.C01"), multifile=False) as f:
        assert [x.index for x in f.plates[0].wells[0].fields] == [1] and len(f.images) == 1


def test_a_missing_channel_file_reads_as_zeros_with_one_warning(tmp_path):
    _plate(str(tmp_path), skip={("A01", 1, 1)})
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with micro_reader.open(str(tmp_path)) as f:
            assert not f.plates[0].wells[0].fields[1].image[0, 1, 0].any()
    told = [w for w in caught if issubclass(w.category, micro_reader.MissingImagesWarning)]
    assert len(told) == 1 and "1 channel files are missing" in str(told[0].message)


def test_a_cellomics_folder_is_one_group(tmp_path):
    _plate(str(tmp_path))
    g = micro_reader.group_files(str(tmp_path))
    assert len(g.groups) == 1 and not g.singles
    (group,) = g.groups
    assert group.format == "cellomics" and len(group.members) == 7 and len(group.parts) == 3


def test_a_c01_file_named_otherwise_is_refused(tmp_path):
    _c01(str(tmp_path / "plane.C01"), np.zeros((Y, X), "u2"))
    with pytest.raises(micro_reader.UnsupportedFormatError, match="<plate>_<well>f<field>d<channel>"):
        micro_reader.open(str(tmp_path / "plane.C01"))


DATA = os.environ.get("MICROSCOPY_DATA", os.path.expanduser("~/Desktop/data/microscopy"))


def test_bia_s_biad1069_one_well_of_nine_fields():
    """BIA S-BIAD1069 (CellInsight CX7, CC0): well A01, 9 fields, one channel,
    1104 x 1104 uint16, 0.7896 um, as Bio-Formats 8.0.1 reads it (which names
    the channel DAPI from the Access database and guesses a 96-well layout)."""
    folder = os.path.join(DATA, "plates", "cellomics", "bia-S-BIAD1069", "MFGTMPcx7_220517060001")
    if not os.path.exists(folder):
        pytest.skip(f"sample not found: {folder} (set MICROSCOPY_DATA)")
    with micro_reader.open(folder) as f:
        (plate,) = f.plates
        assert plate.name == "MFGTMPcx7_220517060001"
        (well,) = plate.wells
        assert well.name == "A1" and [x.index for x in well.fields] == list(range(9))
        im = well.fields[0].image
        assert (im.shape, im.dtype) == ((1, 1, 1, 1104, 1104), np.uint16)
        assert im.metadata.pixel_sizes["x"].value == pytest.approx(0.78957, rel=1e-4)
        plane = im[0, 0, 0]
        assert (int(plane.min()), int(plane.max())) == (253, 15970)       # Bio-Formats' plane
