"""Harmony plates (backends/harmony.py, docs/HCS.md): synthetic index files
(``Index.idx.xml``, HarmonyV4 / V5) with one small TIFF per plane."""
from __future__ import annotations

import io
import os
import threading
import warnings
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

import micro_reader
from micro_reader import _fileset

tifffile = pytest.importorskip("tifffile")
RNG = np.random.default_rng(21)
Y, X = 12, 16


def _lzw_or_zlib() -> str:
    """LZW, as Harmony writes its TIFFs, when this tifffile can encode it (its
    oldest supported version needs imagecodecs for that); zlib otherwise."""
    try:
        tifffile.imwrite(io.BytesIO(), np.zeros((2, 2), "u2"), compression="lzw")
        return "lzw"
    except (KeyError, ImportError, AttributeError, ValueError):
        return "zlib"


COMPRESSION = _lzw_or_zlib()


def _write_harmony(folder, wells=((1, 1), (2, 3)), fields=2, channels=2, planes=1,
                   timepoints=1, tp_base=0, version="V4", missing=(), empty=(),
                   plates=1, flim=(0,), dz=2e-6, compress=False, new_style=False):
    """A Harmony plate in *folder*/Images.  *wells*: 1-based (row, col);
    *missing* / *empty*: (row, col, field, tp, ch, pl) planes whose TIFF is
    not written / whose URL is empty.  *new_style*: as Harmony 5.2 / 6 write
    their "hs" archives -- per-channel values in ``Maps`` (the first Map holding
    only a flat-field profile), ``images/index.xml``, one folder per well, and
    Harmony 5.2's namespace.  -> (index path, {plane key: array})."""
    images_dir = os.path.join(folder, "images" if new_style else "Images")
    os.makedirs(images_dir, exist_ok=True)
    channel_tags = ("ChannelName", "ImageResolutionX", "ImageResolutionY", "ImageSizeX",
                    "ImageSizeY", "MainExcitationWavelength", "MainEmissionWavelength")
    maps = {}
    data, entries, well_xml = {}, [], []
    for r, c in wells:
        refs = []
        for f in range(1, fields + 1):
            for t in range(tp_base, tp_base + timepoints):
                for ch in range(1, channels + 1):
                    for p in range(1, planes + 1):
                        for fl in flim:
                            key = (r, c, f, t, ch, p)
                            url = f"r{r:02d}c{c:02d}f{f:02d}p{p:02d}-ch{ch}sk{t + 1}fk1fl{fl}.tiff"
                            if new_style:
                                url = f"r{r:02d}c{c:02d}/{url}"
                                os.makedirs(os.path.join(images_dir, f"r{r:02d}c{c:02d}"),
                                            exist_ok=True)
                            if key not in missing and key not in empty:
                                a = RNG.integers(0, 4000, (Y, X)).astype("u2")
                                tifffile.imwrite(os.path.join(images_dir, url), a,
                                                 compression=COMPRESSION if compress else None)
                                data[key] = a
                            ident = f"{r:02d}{c:02d}K{t + 1}F{f}P{p}R{ch}"
                            refs.append(f'<Image id="{ident}" />')
                            per_channel = (
                                f"<ChannelName>ch{ch}</ChannelName>"
                                f'<ImageResolutionX Unit="m">6.5E-7</ImageResolutionX>'
                                f'<ImageResolutionY Unit="m">6.5E-7</ImageResolutionY>'
                                f"<ImageSizeX>{X}</ImageSizeX><ImageSizeY>{Y}</ImageSizeY>"
                                f'<MainExcitationWavelength Unit="nm">{400 + 100 * ch}</MainExcitationWavelength>'
                                f'<MainEmissionWavelength Unit="nm">{450 + 100 * ch}</MainEmissionWavelength>')
                            maps[ch] = per_channel
                            entries.append(
                                f"<Image Version=\"1\"><id>{ident}</id><State>Ok</State>"
                                f"<URL>{'' if key in empty else url}</URL>"
                                f"<Row>{r}</Row><Col>{c}</Col><FieldID>{f}</FieldID>"
                                f"<PlaneID>{p}</PlaneID><TimepointID>{t}</TimepointID>"
                                f"<ChannelID>{ch}</ChannelID><FlimID>{fl}</FlimID>"
                                + ("" if new_style else per_channel) +
                                f'<PositionX Unit="m">{(f - 1) * 1e-4}</PositionX>'
                                f'<PositionY Unit="m">{(f - 1) * -5e-5}</PositionY>'
                                f'<PositionZ Unit="m">{p * dz}</PositionZ>'
                                f'<MeasurementTimeOffset Unit="s">{(t - tp_base) * 30}</MeasurementTimeOffset>'
                                "</Image>")
        well_xml.append(f"<Well><id>{r:02d}{c:02d}</id><Row>{r}</Row><Col>{c}</Col>"
                        + "".join(refs) + "</Well>")
    plate = ("<Plate><PlateID>P1</PlateID><Name>Screen plate 1</Name>"
             "<PlateRows>8</PlateRows><PlateColumns>12</PlateColumns>"
             + "".join(f'<Well id="{r:02d}{c:02d}" />' for r, c in wells) + "</Plate>")
    namespace = (f"43B2A954-E3C3-47E1-B392-6635266B0DD3/Harmony{version}" if new_style
                 else f"http://www.perkinelmer.com/PEHH/Harmony{version}")
    map_xml = ""
    if new_style:
        flat = "".join(f'<Entry ChannelID="{ch}"><FlatfieldProfile>{{}}</FlatfieldProfile></Entry>'
                       for ch in maps)
        info = "".join(f'<Entry ChannelID="{ch}">{tags}</Entry>' for ch, tags in maps.items())
        map_xml = f"<Maps><Map>{flat}</Map><Map>{info}</Map></Maps>"
    xml = ('<?xml version="1.0" encoding="utf-8"?>\n'
           f'<EvaluationInputData Version="1" xmlns="{namespace}">'
           "<User>test</User><InstrumentType>Operetta</InstrumentType>"
           f"<Plates>{plate * plates}</Plates><Wells>{''.join(well_xml)}</Wells>{map_xml}"
           f"<Images>{''.join(entries)}</Images></EvaluationInputData>")
    index = os.path.join(images_dir, "index.xml" if new_style else "Index.idx.xml")
    with open(index, "w", encoding="utf-8") as fh:
        fh.write(xml)
    return index, data


def test_a_harmony_plate(tmp_path):
    index, data = _write_harmony(str(tmp_path), channels=2, planes=3, compress=True)
    with micro_reader.open(index) as f:
        assert f.backend == "harmony" and f.format == "harmony-v4"
        (plate,) = f.plates
        assert plate.name == "Screen plate 1" and (len(plate.rows), len(plate.columns)) == (8, 12)
        assert [w.name for w in plate.wells] == ["A1", "B3"]
        assert f.indices == [{"plate": 0, "acquisition": 0, "well": w, "field": fd}
                             for w in range(2) for fd in range(2)]
        for image in f.images:
            assert image.axes == "tczyx" and image.shape == (1, 2, 3, Y, X)
            w, fd = image.index["well"], image.index["field"]
            r, c = ((1, 1), (2, 3))[w]
            for ch in range(2):
                for p in range(3):
                    np.testing.assert_array_equal(image[0, ch, p], data[(r, c, fd + 1, 0, ch + 1, p + 1)])
        # a rectangle across channels and planes
        image = f.plates[0].wells[1].fields[1].image
        want = np.stack([np.stack([data[(2, 3, 2, 0, ch, p)][3:9, 4:11] for p in (2, 3)])
                         for ch in (1, 2)])
        np.testing.assert_array_equal(image[0, :, 1:, 3:9, 4:11], want)
        md = image.metadata
        assert md.pixel_sizes["x"] == micro_reader.Quantity(0.65, "micrometer")
        assert md.pixel_sizes["z"].unit == "micrometer"
        assert md.pixel_sizes["z"].value == pytest.approx(2.0)
        assert [c.name for c in md.channels] == ["ch1", "ch2"]
        assert md.channels[1].emission_wavelength == micro_reader.Quantity(650.0, "nanometer")
        assert "plate 0 'Screen plate 1': 8 x 12, 2 wells with images" in f.outline()


@pytest.mark.parametrize("version", ["V6", "V7"])
def test_new_style_hs_archive(tmp_path, version):
    """Harmony 5.2 / 6 "hs" archives (zenodo-10776785, zenodo-20799564): the
    per-channel values in Maps, images/index.xml, a folder per well, a new
    namespace; the measurement folder and its side files open the index."""
    measurement = tmp_path / "hs" / "cf91602d"
    index, data = _write_harmony(str(measurement), channels=2, planes=2, version=version,
                                 new_style=True)
    (measurement / "flatfieldcorrection").mkdir()
    (measurement / "flatfieldcorrection" / "ffc.xml").write_text("<x/>")
    (measurement / "cf91602d.xml").write_text("<x/>")
    (measurement / "images" / "image.index.txt").write_text("Row\tColumn\n")
    with micro_reader.open(index) as f:
        assert f.format == f"harmony-{version.lower()}"
        image = f.plates[0].wells[1].fields[1].image
        assert image.shape == (1, 2, 2, Y, X)
        np.testing.assert_array_equal(image[0, 1, 1], data[(2, 3, 2, 0, 2, 2)])
        md = image.metadata
        assert [c.name for c in md.channels] == ["ch1", "ch2"]
        assert md.pixel_sizes["x"] == micro_reader.Quantity(0.65, "micrometer")
        assert md.channels[1].emission_wavelength == micro_reader.Quantity(650.0, "nanometer")
    for entry in (measurement, measurement / "flatfieldcorrection" / "ffc.xml",
                  measurement / "cf91602d.xml", measurement / "images" / "image.index.txt"):
        with micro_reader.open(entry) as f:
            assert os.path.basename(f.path) == "index.xml"          # as stored
    with micro_reader.open(measurement / "images" / "r01c01" / os.listdir(
            measurement / "images" / "r01c01")[0]) as f:
        assert f.backend == "tiff"                                   # a plane opens as itself
    g = micro_reader.group_files(str(measurement))
    assert len(g.groups) == 1 and g.singles == []
    assert len(g.groups[0].members) == len(data) and len(g.groups[0].parts) == 3


def test_field_positions_are_first_pixels_relative_to_the_well(tmp_path):
    """PositionX / PositionY are the field's centre with y up (as Bio-Formats
    8.0.0 reads them on idr0034: x, -y); micro-reader gives the first pixel's
    centre in image coordinates (y down): x - (W-1)/2 px, -y - (H-1)/2 px."""
    index, _ = _write_harmony(str(tmp_path), wells=((1, 1),), fields=2, channels=1, planes=2)
    with micro_reader.open(index) as f:
        px = 0.65
        for image in f.images:
            fd = image.index["field"]                      # FieldID - 1
            pos = image.metadata.position
            assert pos["x"].unit == pos["y"].unit == "micrometer"
            assert pos["x"].value == pytest.approx(fd * 100.0 - (X - 1) / 2 * px)
            assert pos["y"].value == pytest.approx(fd * 50.0 - (Y - 1) / 2 * px)
            assert pos["z"].value == pytest.approx(2.0)     # the first plane's PositionZ


def test_one_coordinate_system_per_well(tmp_path, monkeypatch):
    """Positions are relative to the well's centre, so each well is its own
    system (plan 4.9): ``well-A1``, ``well-B3``, declared at open without
    reading any metadata; every field names its well's system (I2, I3)."""
    from micro_reader._spatial import check_invariants
    from micro_reader.backends import harmony
    reads = []
    real = harmony._HarmonyImage._read_metadata
    monkeypatch.setattr(harmony._HarmonyImage, "_read_metadata",
                        lambda self: reads.append(self.name) or real(self))
    index, _ = _write_harmony(str(tmp_path), wells=((1, 1), (2, 3)), fields=2)
    with micro_reader.open(index) as f:
        systems = f.coordinate_systems
        assert [(s.id, s.name, s.axes, s.estimated) for s in systems] == [
            ("well-A1", "well A1", None, False), ("well-B3", "well B3", None, False)]
        assert f.coordinate_transformations == ()        # no edge joins two wells
        assert reads == []                               # declared without a metadata read
        for image in f.images:
            well = f.plates[0].wells[image.index["well"]].name
            md = image.metadata
            assert md.coordinate_system == f"well-{well}" and set(md.position) == {"x", "y", "z"}
        check_invariants(f)


def test_a_field_without_positions_is_unplaced(tmp_path):
    """An index without PositionX / Y / Z: no position, so no system (I3)."""
    import re
    index, _ = _write_harmony(str(tmp_path), wells=((1, 1),), fields=1, channels=1)
    with open(index, encoding="utf-8") as fh:
        xml = re.sub(r"<Position[XYZ][^>]*>[^<]*</Position[XYZ]>", "", fh.read())
    with open(index, "w", encoding="utf-8") as fh:
        fh.write(xml)
    with micro_reader.open(index) as f:
        md = f.images[0].metadata
        assert md.position == {} and md.coordinate_system is None
        assert [s.id for s in f.coordinate_systems] == ["well-A1"]


def test_well_system_ids_with_several_plates():
    """With several plates the plate index scopes a well's id (plan 3.23):
    ``well-p0-A1``, ``well-p1-A1``; names say which plate."""
    from micro_reader._plate import Plate, Well, well_system, well_systems
    one = Plate("a", ("A",), ("1",), (Well(0, 0, "A1"),))
    two = Plate("b", ("A", "B"), ("1", "2"), (Well(0, 0, "A1"), Well(1, 1, "B2")))
    assert well_system([two], 0, 1) == "well-B2"
    assert [s.id for s in well_systems([one, two])] == ["well-p0-A1", "well-p1-A1", "well-p1-B2"]
    assert well_systems([one, two])[2].name == "plate 1 well B2"


def test_timepoints_and_ids_counting_from_1(tmp_path):
    """IDs start at 0 (Operetta timepoints) or 1 (CV7000): axes count from 0."""
    index, data = _write_harmony(str(tmp_path), wells=((1, 1),), fields=1, channels=1,
                                 timepoints=3, tp_base=1, version="V5")
    with micro_reader.open(index) as f:
        assert f.format == "harmony-v5"
        (image,) = f.images
        assert image.shape == (3, 1, 1, Y, X)
        for t in range(3):
            np.testing.assert_array_equal(image[t, 0, 0], data[(1, 1, 1, t + 1, 1, 1)])
        assert image.metadata.pixel_sizes["t"] == micro_reader.Quantity(30.0, "second")


def test_missing_planes_read_as_zeros_and_empty_fields_are_left_out(tmp_path):
    """Bugra, 2026-10-13: a field missing some planes is kept (zeros there), a
    field missing all of them is left out; both with a warning."""
    index, data = _write_harmony(
        str(tmp_path),
        missing={(1, 1, 2, 0, 2, 1),                                  # A1 field 2: channel 2 gone
                 (2, 3, 1, 0, 1, 1), (2, 3, 1, 0, 2, 1)},             # B3 field 1: nothing
        empty={(2, 3, 2, 0, 1, 1)})                                   # B3 field 2: channel 1 unlisted
    with pytest.warns(micro_reader.MissingImagesWarning,
                      match="2 fields miss some planes, which read as zeros; "
                            "1 fields have no files and are left out"):
        f = micro_reader.open(index)
    with f:
        assert f.indices == [{"plate": 0, "acquisition": 0, "well": w, "field": fd}
                             for w, fd in ((0, 0), (0, 1), (1, 1))]
        a1f2 = f.plates[0].wells[0].fields[1].image
        np.testing.assert_array_equal(a1f2[0, 0, 0], data[(1, 1, 2, 0, 1, 1)])
        assert not a1f2[0, 1, 0].any()                                # the missing plane: zeros
        b3f2 = f.plates[0].wells[1].fields[0].image
        assert b3f2.index["field"] == 1
        assert not b3f2[0, 0, 0].any()
        np.testing.assert_array_equal(b3f2[0, 1, 0], data[(2, 3, 2, 0, 2, 1)])
        assert [c.name for c in b3f2.metadata.channels] == ["ch1", "ch2"]   # listed, so named


def test_refused(tmp_path):
    index, _ = _write_harmony(str(tmp_path / "two"), plates=2)
    with pytest.raises(micro_reader.UnsupportedFormatError, match="2 plates"):
        micro_reader.open(index)
    index, _ = _write_harmony(str(tmp_path / "flim"), flim=(1, 2))
    with pytest.raises(micro_reader.UnsupportedFormatError, match="FLIM"):
        micro_reader.open(index)
    index, _ = _write_harmony(str(tmp_path / "none"), wells=((1, 1),), fields=1, channels=1,
                              empty={(1, 1, 1, 0, 1, 1)})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", micro_reader.MissingImagesWarning)
        with pytest.raises(micro_reader.CorruptFileError, match="none of the fields it lists has any"):
            micro_reader.open(index)


def test_few_member_files_open_at_once_and_threads_read_safely(tmp_path, monkeypatch):
    monkeypatch.setattr(_fileset, "OPEN_MEMBERS", 2)
    index, data = _write_harmony(str(tmp_path), wells=((1, 1), (1, 2), (2, 1)), fields=3,
                                 channels=3, planes=2)
    with micro_reader.open(index) as f:
        def check(image):
            r, c = [(1, 1), (1, 2), (2, 1)][image.index["well"]]
            fd = image.index["field"] + 1
            got = image[...]
            for ch in range(3):
                for p in range(2):
                    np.testing.assert_array_equal(got[0, ch, p], data[(r, c, fd, 0, ch + 1, p + 1)])
            return True
        with ThreadPoolExecutor(8) as pool:
            assert all(pool.map(check, f.images * 3))
        assert len(f._members._open) <= 2
