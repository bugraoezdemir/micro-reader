"""Yokogawa CV7000 measurements (backends/cv7000.py, docs/HCS.md): synthetic
BTS files with one small TIFF per plane."""
from __future__ import annotations

import os
import warnings

import numpy as np
import pytest

import micro_reader
from micro_reader.backends import cv7000

tifffile = pytest.importorskip("tifffile")
RNG = np.random.default_rng(7000)
Y, X = 6, 10
NS = 'xmlns:bts="http://www.yokogawa.co.jp/BTS/BTSSchema/1.0"'
#: (action, Ch, target, light source nm, filter, focus z of each plane)
TWO_D = [(1, 1, "BFP", 405, "BP445/45", [-3.0]), (2, 2, "GFP", 488, "BP525/50", [-3.0])]


def _write(folder, actions=TWO_D, wells=((2, 2), (3, 5)), fields=2, times=1, skip=(),
           timelines=1, errors=(), pixel=(0.65, 0.65), mes=True, layout=(16, 24)):
    """*wells*: 1-based (row, column); *skip*: (row, col, field, t, action, z)
    not written; *layout*: the (rows, columns) the plate files describe.
    -> {(row, col, field, t, action, z): array}."""
    rows, columns = layout
    os.makedirs(folder, exist_ok=True)
    chans = "".join(
        f'<bts:MeasurementChannel bts:Ch="{ch}" bts:HorizontalPixelDimension="{pixel[0] if ch == 1 else pixel[1]}" '
        f'bts:VerticalPixelDimension="{pixel[0] if ch == 1 else pixel[1]}" bts:HorizontalPixels="{X}" '
        f'bts:VerticalPixels="{Y}" />' for _a, ch, *_ in actions)
    with open(os.path.join(folder, "MeasurementDetail.mrf"), "w", encoding="utf-8") as fh:
        fh.write(f'<?xml version="1.0" encoding="utf-8"?><bts:MeasurementDetail bts:RowCount="{rows}" '
                 f'bts:ColumnCount="{columns}" bts:MeasurementSettingFileName="s.mes" {NS}>'
                 f'<bts:MeasurementSamplePlate bts:Name="P1" bts:WellPlateFileName="P1.wpi" />'
                 f"{chans}</bts:MeasurementDetail>")
    with open(os.path.join(folder, "P1.wpi"), "w", encoding="utf-8") as fh:
        fh.write(f'<?xml version="1.0" encoding="utf-8"?>\n<bts:WellPlate bts:Name="P1" '
                 f'bts:Columns="{columns}" bts:Rows="{rows}" {NS}></bts:WellPlate>')
    with open(os.path.join(folder, "s.wpp"), "w", encoding="utf-8") as fh:
        fh.write(f'<?xml version="1.0" encoding="utf-8"?>\n<bts:WellPlateProduct\n  bts:Rows="{rows}" {NS} />')
    if mes:
        lights = "".join(f'<bts:LightSource bts:Name="{nm}nm" bts:WaveLength="{nm}" />'
                         for _a, _c, _t, nm, _f, _z in actions)
        channels = "".join(
            f'<bts:Channel bts:Ch="{ch}" bts:Target="{target}" bts:Acquisition="{band}" '
            f'bts:Color="#FF00FF00"><bts:LightSourceName>{nm}nm</bts:LightSourceName></bts:Channel>'
            for _a, ch, target, nm, band, _z in actions)
        with open(os.path.join(folder, "s.mes"), "w", encoding="utf-8") as fh:
            fh.write(f'<?xml version="1.0" encoding="utf-8"?><bts:MeasurementSetting {NS}>'
                     f"<bts:LightSourceList>{lights}</bts:LightSourceList>"
                     f"<bts:ChannelList>{channels}</bts:ChannelList></bts:MeasurementSetting>")
    records, data = [], {}
    for r, c in wells:
        for f in range(1, fields + 1):
            for t in range(1, times + 1):
                for action, ch, _t, _nm, _f, zs in actions:
                    for z, focus in enumerate(zs, 1):
                        key = (r, c, f, t, action, z)
                        fn = (f"P1_{chr(64 + r)}{c:02d}_T{t:04d}F{f:03d}L01A{action:02d}"
                              f"Z{z:02d}C{ch:02d}.tif")
                        if key not in skip:
                            a = RNG.integers(1, 60000, (Y, X)).astype("u2")
                            tifffile.imwrite(os.path.join(folder, fn), a)
                            data[key] = a
                        line = (timelines if action == actions[-1][0] else 1)
                        records.append(
                            f'<bts:MeasurementRecord bts:Type="IMG" '
                            f'bts:Time="2017-09-16T15:{t * 10:02d}:00.000+02:00" bts:Column="{c}" '
                            f'bts:Row="{r}" bts:TimePoint="{t}" bts:FieldIndex="{f}" bts:ZIndex="{z}" '
                            f'bts:TimelineIndex="{line}" bts:ActionIndex="{action}" '
                            f'bts:X="{100.0 * f}" bts:Y="{-50.0 * f}" bts:Z="{focus}" bts:Ch="{ch}">'
                            f"{fn}</bts:MeasurementRecord>")
    records += [f'<bts:MeasurementRecord bts:Type="ERR" bts:Row="2" bts:Column="2">{e}'
                "</bts:MeasurementRecord>" for e in errors]
    with open(os.path.join(folder, "MeasurementData.mlf"), "w", encoding="utf-8") as fh:
        fh.write(f'<?xml version="1.0" encoding="utf-8"?>\n<bts:MeasurementData bts:Version="1.0" {NS}>'
                 + "".join(records) + "</bts:MeasurementData>")
    return data


@pytest.mark.parametrize("entry", ["P1.wpi", "MeasurementDetail.mrf", "MeasurementData.mlf"])
def test_a_measurement_opens_from_each_entry_file(tmp_path, entry):
    data = _write(str(tmp_path))
    with micro_reader.open(tmp_path / entry) as f:
        assert f.backend == "cv7000"
        (plate,) = f.plates
        assert plate.name == "P1" and len(plate.rows) == 16 and len(plate.columns) == 24
        assert [w.name for w in plate.wells] == ["B2", "C5"]
        assert f.indices == [{"plate": 0, "acquisition": 0, "well": w, "field": fd}
                             for w in range(2) for fd in range(2)]
        image = plate.wells[1].fields[1].image
        assert image.shape == (1, 2, 1, Y, X) and image.name == "P1 well C5 field 1"
        for a in range(2):
            np.testing.assert_array_equal(image[0, a, 0], data[(3, 5, 2, 1, a + 1, 1)])


def test_wells_outside_the_described_layout_grow_it(tmp_path):
    """A slide holder described as 1 x 4 slides whose records address a well
    of the slide in it (C05, zenodo-12795729): the layout holds the well."""
    _write(str(tmp_path), wells=((3, 5),), fields=1, layout=(1, 4))
    with pytest.warns(micro_reader.PlateLayoutWarning, match=r"1 x 4 layout, but wells C5"):
        f = micro_reader.open(str(tmp_path / "MeasurementData.mlf"))
    with f:
        plate = f.plates[0]
        assert (plate.row_names, plate.column_names) == (("A", "B", "C"), ("1", "2", "3", "4", "5"))
        well = plate.wells[0]
        assert (well.name, plate.row_names[well.row], plate.column_names[well.column]) == ("C5", "C", "5")


def test_metadata_channels_pixel_sizes_and_position(tmp_path):
    _write(str(tmp_path))
    with micro_reader.open(tmp_path / "P1.wpi") as f:
        md = f.images[1].metadata                       # field 2: X = 200, Y = -100
        assert md.pixel_sizes == {"x": micro_reader.Quantity(0.65, "micrometer"),
                                  "y": micro_reader.Quantity(0.65, "micrometer")}
        assert [c.name for c in md.channels] == ["BFP", "GFP"]
        assert md.channels[0].excitation_wavelength == micro_reader.Quantity(405.0, "nanometer")
        assert md.channels[0].emission_wavelength == micro_reader.Quantity(445.0, "nanometer")
        assert md.channels[1].color == "#00FF00"
        # first-pixel centre, image coordinates, relative to the well centre
        assert md.position["x"].value == pytest.approx(200.0 - (X - 1) / 2 * 0.65)
        assert md.position["y"].value == pytest.approx(100.0 - (Y - 1) / 2 * 0.65)
        assert md.position["z"].value == -3.0


def test_one_coordinate_system_per_well(tmp_path, monkeypatch):
    """As for Harmony (plan 4.9): ``well-B2``, ``well-C5``, declared at open
    without reading any metadata; each field names its well's system."""
    from micro_reader._spatial import check_invariants
    reads = []
    real = cv7000._CVImage._read_metadata
    monkeypatch.setattr(cv7000._CVImage, "_read_metadata",
                        lambda self: reads.append(self.name) or real(self))
    _write(str(tmp_path))
    with micro_reader.open(tmp_path / "P1.wpi") as f:
        assert [(s.id, s.name) for s in f.coordinate_systems] == [
            ("well-B2", "well B2"), ("well-C5", "well C5")]
        assert reads == []
        assert [image.metadata.coordinate_system for image in f.images] == [
            "well-B2", "well-B2", "well-C5", "well-C5"]
        check_invariants(f)


def test_a_record_without_stage_coordinates_is_unplaced(tmp_path):
    import re
    _write(str(tmp_path), wells=((2, 2),), fields=1)
    mlf = tmp_path / "MeasurementData.mlf"
    mlf.write_text(re.sub(r' bts:[XYZ]="[^"]*"', "", mlf.read_text(encoding="utf-8")),
                   encoding="utf-8")
    with micro_reader.open(tmp_path / "P1.wpi") as f:
        md = f.images[0].metadata
        assert md.position == {} and md.coordinate_system is None


def test_default_targets_are_named_by_their_light_source(tmp_path):
    """The software's default Target "Microscope Image" names nothing (BIA
    S-BIAD1013 / S-BIAD3934): the light source does, plus the filter when two
    channels share one."""
    default = "Microscope Image"
    actions = [(1, 1, default, 405, "BP445/45", [-3.0]), (2, 2, default, 488, "BP525/50", [-3.0]),
               (3, 3, default, 488, "BP600/37", [-3.0]), (4, 4, "GFP", 561, "BP617/67", [-3.0])]
    _write(str(tmp_path), actions=actions, wells=((2, 2),), fields=1)
    with micro_reader.open(tmp_path / "P1.wpi") as f:
        assert [c.name for c in f.images[0].metadata.channels] == [
            "405nm", "488nm BP525/50", "488nm BP600/37", "GFP"]


def test_timepoints_and_a_settings_file_that_is_missing(tmp_path):
    _write(str(tmp_path), times=3, mes=False)
    with micro_reader.open(tmp_path / "P1.wpi") as f:
        image = f.images[0]
        assert image.shape[0] == 3
        md = image.metadata
        assert md.pixel_sizes["t"] == micro_reader.Quantity(600.0, "second")
        assert [c.name for c in md.channels] == ["Action 1, Ch 1", "Action 2, Ch 2"]


def test_missing_files_and_logged_errors_warn(tmp_path):
    skip = {(2, 2, 1, 1, 2, 1)} | {(3, 5, 2, 1, a, 1) for a in (1, 2)}
    _write(str(tmp_path), skip=skip, errors=["AF Error at B2 No. 1 (P1)"])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        f = micro_reader.open(tmp_path / "P1.wpi")
    told = [str(w.message) for w in caught]
    assert any("logged 1 error" in m and "AF Error" in m for m in told)
    assert any("1 fields miss some planes" in m and "1 fields have no files" in m for m in told)
    with f:
        assert len(f.images) == 3 and not f.images[0][0, 1].any()


def test_a_three_d_action_next_to_a_two_d_one_is_ragged(tmp_path):
    actions = [(1, 1, "BFP", 405, "BP445/45", [3.0]),
               (2, 2, "GFP", 488, "BP525/50", [0.0, 2.0, 4.0, 6.0])]
    data = _write(str(tmp_path), actions=actions, wells=((2, 2),), fields=1)
    with micro_reader.open(tmp_path / "P1.wpi") as f:
        (whole,) = f.images
        assert whole.is_ragged and whole.shape == (1, 2, 4, Y, X)
        flat, stack = whole.parts
        assert flat.shape == (1, 1, 1, Y, X) and stack.shape == (1, 1, 4, Y, X)
        assert stack.metadata.pixel_sizes["z"] == micro_reader.Quantity(2.0, "micrometer")
        assert stack.metadata.position["z"].value == 0.0
        # the whole starts at its grid's slice 0 (the deepest channel's first
        # plane), not at the first channel's plane (JUMP's 2-D fluorescence
        # listed before its 3-D brightfield); the flat part at its own plane
        assert whole.metadata.position["z"].value == 0.0
        assert flat.metadata.position["z"].value == 3.0
        # the whole and each part are in their well's system
        assert {i.metadata.coordinate_system for i in (whole, flat, stack)} == {"well-B2"}
        # BFP at z = 3.0: nearest slice of 0, 2, 4, 6 is 1 (a tie with 2 goes to the first)
        np.testing.assert_array_equal(whole[0, 0, 1], data[(2, 2, 1, 1, 1, 1)])
        assert not np.delete(whole[0, 0], 1, axis=0).any()
        np.testing.assert_array_equal(whole[0, 1, 3], data[(2, 2, 1, 1, 2, 4)])


def test_timelines_are_merged_as_bio_formats_merges_them(tmp_path):
    """The writer puts the last action on timeline 2: the same wells and fields imaged
    by two timelines. One plate without acquisitions; each field holds the channels of
    both timelines (Bio-Formats' expected results for Zenodo 21334895)."""
    data = _write(str(tmp_path), timelines=2)
    with micro_reader.open(tmp_path / "P1.wpi", plates=True) as f:
        plate = f.plates[0]
        assert plate.acquisitions == () and [w.name for w in plate.wells] == ["B2", "C5"]
        assert len(f.images) == 4                                          # 2 wells x 2 fields
        first = f.images[0]
        assert dict(first.index) == {"plate": 0, "acquisition": 0, "well": 0, "field": 0}
        assert first.shape[1] == len(TWO_D)
        last_action = TWO_D[-1][0]
        np.testing.assert_array_equal(first[0, len(TWO_D) - 1, 0], data[(2, 2, 1, 1, last_action, 1)])


def test_a_plane_a_second_timeline_images_again_is_a_channel_of_its_own(tmp_path):
    """Zenodo 21334895 'fallback-reused-channel': timeline 2 images well A1 again with
    timeline 1's first action and channel. Bio-Formats keeps the plane as a third
    channel; so do we, named after its timeline."""
    data = _write(str(tmp_path), timelines=1, wells=((2, 2),), fields=1)
    mlf = tmp_path / "MeasurementData.mlf"
    text = mlf.read_text(encoding="utf-8")
    first = TWO_D[0]
    record = next(r for r in text.split("<bts:MeasurementRecord")[1:]
                  if f'bts:ActionIndex="{first[0]}"' in r and f'bts:Ch="{first[1]}"' in r)
    again = record.replace('bts:TimelineIndex="1"', 'bts:TimelineIndex="2"')
    again = again.replace("L01A", "L02A", 1)                         # another file
    name = again.split(">", 1)[1].split("<", 1)[0]
    import shutil
    shutil.copy(tmp_path / record.split(">", 1)[1].split("<", 1)[0], tmp_path / name)
    text = text.replace("</bts:MeasurementData>", "<bts:MeasurementRecord" + again + "</bts:MeasurementData>")
    mlf.write_text(text, encoding="utf-8")
    with micro_reader.open(tmp_path / "P1.wpi", plates=True) as f:
        (image,) = f.images
        assert image.shape[1] == len(TWO_D) + 1
        names = [c.name for c in image.metadata.channels]
        added = [k for k, n in enumerate(names) if n.endswith("(timeline 2)")]
        assert len(added) == 1 and len(set(names)) == len(names)
        assert names[added[0]] == f"{names[added[0] - 1]} (timeline 2)"    # right after its twin
        np.testing.assert_array_equal(image[0, added[0], 0], image[0, added[0] - 1, 0])


def test_a_timeline_counts_its_timepoints_from_its_own_first(tmp_path):
    """CV7000 numbers TimePoints on from one timeline to the next (the Zenodo 21334895
    packages: timeline 2's first timepoint is TimePoint 2): ranked per timeline, both
    timelines' planes sit at t = 0, as Bio-Formats puts them."""
    _write(str(tmp_path), timelines=2)
    mlf = tmp_path / "MeasurementData.mlf"
    text = mlf.read_text(encoding="utf-8")
    text = text.replace('bts:TimePoint="1" bts:FieldIndex="1" bts:ZIndex="1" bts:TimelineIndex="2"',
                        'bts:TimePoint="2" bts:FieldIndex="1" bts:ZIndex="1" bts:TimelineIndex="2"')
    text = text.replace('bts:TimePoint="1" bts:FieldIndex="2" bts:ZIndex="1" bts:TimelineIndex="2"',
                        'bts:TimePoint="2" bts:FieldIndex="2" bts:ZIndex="1" bts:TimelineIndex="2"')
    assert 'bts:TimePoint="2"' in text
    mlf.write_text(text, encoding="utf-8")
    with micro_reader.open(tmp_path / "P1.wpi", plates=True) as f:
        assert f.images[0].shape[:3] == (1, len(TWO_D), 1)


@pytest.mark.parametrize("kwargs, match", [
    ({"pixel": (0.65, 1.3)}, "different pixel or image sizes"),
])
def test_refused(tmp_path, kwargs, match):
    _write(str(tmp_path), **kwargs)
    with pytest.raises(micro_reader.UnsupportedFormatError, match=match):
        micro_reader.open(tmp_path / "P1.wpi")


def test_only_the_entry_files_are_sniffed(tmp_path):
    _write(str(tmp_path))
    for name, expected in (("P1.wpi", True), ("MeasurementDetail.mrf", True),
                           ("MeasurementData.mlf", True), ("s.wpp", False), ("s.mes", False)):
        with open(tmp_path / name, "rb") as fh:
            assert cv7000.sniff(fh.read(1024), str(tmp_path / name)) is expected, name


def _reprefix(path, prefix, unprefixed=()):
    """Rewrite a BTS file with namespace prefix *prefix* ("" for a default
    namespace) and the attributes *unprefixed* without any prefix."""
    text = path.read_text(encoding="utf-8")
    for name in unprefixed:
        text = text.replace(f" bts:{name}=", f" {name}=")
    text = text.replace(" xmlns:bts=", f" xmlns:{prefix}=" if prefix else " xmlns=")
    text = text.replace("bts:", f"{prefix}:" if prefix else "")
    path.write_text(text, encoding="utf-8")


def test_namespace_prefixes_do_not_matter(tmp_path):
    """Bio-Formats' synthetic packages (zenodo-21334895, centralized-parsing):
    each file with its own prefix, attributes with or without one, a default
    namespace in the .wpi, decimal commas.  Elements and attributes are matched
    by namespace or local name, never by prefix."""
    data = _write(str(tmp_path), wells=((2, 2),), fields=1)
    _reprefix(tmp_path / "MeasurementData.mlf", "record", unprefixed=("X", "Y", "Z"))
    _reprefix(tmp_path / "MeasurementDetail.mrf", "detail", unprefixed=("RowCount",))
    _reprefix(tmp_path / "P1.wpi", "")
    _reprefix(tmp_path / "s.mes", "setting", unprefixed=("Target",))
    mrf = tmp_path / "MeasurementDetail.mrf"
    mrf.write_text(mrf.read_text(encoding="utf-8").replace('PixelDimension="0.65"', 'PixelDimension="0,65"'),
                   encoding="utf-8")
    for entry in ("P1.wpi", "MeasurementDetail.mrf", "MeasurementData.mlf"):
        with open(tmp_path / entry, "rb") as fh:
            assert cv7000.sniff(fh.read(1024), str(tmp_path / entry)), entry
        with micro_reader.open(tmp_path / entry) as f:
            assert f.backend == "cv7000" and f.plates[0].name == "P1"
            assert len(f.plates[0].rows) == 16 and [w.name for w in f.plates[0].wells] == ["B2"]
            (image,) = f.images
            np.testing.assert_array_equal(image[0, 1, 0], data[(2, 2, 1, 1, 2, 1)])
            md = image.metadata
            assert [c.name for c in md.channels] == ["BFP", "GFP"]
            assert md.channels[0].excitation_wavelength == micro_reader.Quantity(405.0, "nanometer")
            assert md.pixel_sizes["x"] == micro_reader.Quantity(0.65, "micrometer")
            assert md.position["x"].value == pytest.approx(100.0 - (X - 1) / 2 * 0.65)
            assert md.position["z"].value == -3.0
    with open(tmp_path / "s.wpp", "rb") as fh:
        assert not cv7000.sniff(fh.read(1024), str(tmp_path / "s.wpp"))


def test_a_plane_listed_twice_reads_the_first_record_with_a_file(tmp_path):
    """Bio-Formats' duplicate-policy package: a metadata-only record (no file)
    before the real one, more records after it (one with a file of its own):
    the first record whose file exists is read; none shadows it."""
    data = _write(str(tmp_path), wells=((2, 2),), fields=1)
    real = "P1_B02_T0001F001L01A01Z01C01.tif"
    other = "P1_B02_T0001F001L01A01Z01C01_ignored_tiff_duplicate.tif"
    tifffile.imwrite(tmp_path / other, np.full((Y, X), 7, "u2"))
    mlf = tmp_path / "MeasurementData.mlf"
    text = mlf.read_text(encoding="utf-8")
    start = text.rfind("<bts:MeasurementRecord", 0, text.index(real))
    record = text[start:text.index("</bts:MeasurementRecord>", start) + len("</bts:MeasurementRecord>")]
    twice = [record.replace(real, "P1_B02_T0001F001L01A01Z01C01_metadata_only_duplicate.tif"), record,
             record.replace(real, other),
             record.replace(real, "P1_B02_T0001F001L01A01Z01C01_metadata_only_duplicate_2.tif")]
    mlf.write_text(text.replace(record, "".join(twice)), encoding="utf-8")
    with warnings.catch_warnings():
        warnings.simplefilter("error", micro_reader.MissingImagesWarning)
        f = micro_reader.open(tmp_path / "P1.wpi")
    with f:
        (image,) = f.images
        np.testing.assert_array_equal(image[0, 0, 0], data[(2, 2, 1, 1, 1, 1)])
        assert image.metadata.position["z"].value == -3.0
    g = micro_reader.group_files(str(tmp_path))
    assert len(g.groups) == 1 and len(g.groups[0].members) == 2
    assert [os.path.basename(s) for s in g.singles] == [other]


CV_PLATES = os.path.join(os.environ.get("MICROSCOPY_DATA",
                                        os.path.expanduser("~/Desktop/data/microscopy")),
                         "plates", "cv7000")


def _real(folder):
    path = os.path.join(CV_PLATES, folder)
    if not os.path.exists(path):
        pytest.skip(f"sample not found: {path} (set MICROSCOPY_DATA)")
    return path


def test_a_real_cv8000_plate_with_a_three_d_brightfield_is_ragged():
    """JUMP cpg0016 source_2 plate 1053601756 (CV8000 AZ01, R2.03.02): five
    2-D fluorescence channels and a 3-plane brightfield (BF3D, z -17 / -9 / -1)
    whose records come after them; the 2-D planes (z -9) sit on slice 1."""
    path = _real("cpg0016-jump-source2-1053601756-cv8000")
    with micro_reader.open(os.path.join(path, "1053601756.wpi")) as f:
        (plate,) = f.plates
        assert (len(plate.row_names), len(plate.column_names), len(plate.wells)) == (16, 24, 384)
        assert len(f.images) == 2304
        whole = f.images[0]
        assert whole.shape == (1, 6, 3, 996, 996) and whole.is_ragged
        flat, stack = whole.parts
        assert flat.shape == (1, 5, 1, 996, 996) and stack.shape == (1, 1, 3, 996, 996)
        md = whole.metadata
        assert [c.name for c in md.channels] == ["Hoechst", "Concanavalin", "SYTO",
                                                 "WGA_Phalloidin", "MitoTracker", "TL"]
        assert md.pixel_sizes["z"] == micro_reader.Quantity(8.0, "micrometer")
        assert md.position["z"].value == -17.0 and flat.metadata.position["z"].value == -9.0
        name = os.path.join(path, "1053601756_A01_T0001F001L01A{a:02d}Z{z:02d}C{a:02d}.tif")
        np.testing.assert_array_equal(whole[0, 0, 1], tifffile.imread(name.format(a=1, z=1)))
        assert not whole[0, 0, 0].any() and not whole[0, 0, 2].any()
        for z in range(3):
            np.testing.assert_array_equal(whole[0, 5, z], tifffile.imread(name.format(a=6, z=z + 1)))


def test_a_real_cv7000_plate_of_three_d_fields():
    """BIA S-BIAD3934 Hit_Validation/CN_1K (CV7000 DT01, R2.03.04): 60 wells of
    a 96-well plate, 16 fields, three 3-D actions of 5 planes."""
    path = _real("bia-S-BIAD3934-cv7000-hitvalidation-cn1k")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        f = micro_reader.open(os.path.join(path, "MeasurementData.mlf"))
    with f:
        (plate,) = f.plates
        assert (len(plate.row_names), len(plate.column_names), len(plate.wells)) == (8, 12, 60)
        image = plate.wells[-1].fields[-1].image
        assert image.shape == (1, 3, 5, 984, 992) and not image.is_ragged
        assert image.metadata.position["z"].value == 4.5
        stem = "20260527_Corticalneurons_MOI_1k_Linus_AAV_Validation_G11_T0001F016L01"
        np.testing.assert_array_equal(image[0, 2, 4],
                                      tifffile.imread(os.path.join(path, stem + "A03Z05C03.tif")))
