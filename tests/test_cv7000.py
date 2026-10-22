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
        assert (plate.rows, plate.columns) == (("A", "B", "C"), ("1", "2", "3", "4", "5"))
        well = plate.wells[0]
        assert (well.name, plate.rows[well.row], plate.columns[well.column]) == ("C5", "C", "5")


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
        # the whole and each part are in their well's system
        assert {i.metadata.coordinate_system for i in (whole, flat, stack)} == {"well-B2"}
        # BFP at z = 3.0: nearest slice of 0, 2, 4, 6 is 1 (a tie with 2 goes to the first)
        np.testing.assert_array_equal(whole[0, 0, 1], data[(2, 2, 1, 1, 1, 1)])
        assert not np.delete(whole[0, 0], 1, axis=0).any()
        np.testing.assert_array_equal(whole[0, 1, 3], data[(2, 2, 1, 1, 2, 4)])


@pytest.mark.parametrize("kwargs, match", [
    ({"timelines": 2}, "2 timelines"),
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
