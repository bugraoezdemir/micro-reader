"""MetaXpress plates (backends/metaxpress.py, docs/HCS.md): synthetic ``.HTD``
files with one small TIFF per plane, named as MetaXpress names them."""
from __future__ import annotations

import os

import numpy as np
import pytest

import micro_reader
from micro_reader.backends.metaxpress import parse_htd

tifffile = pytest.importorskip("tifffile")
RNG = np.random.default_rng(31)
Y, X = 10, 14
STEM = "Plate7"


def _bools(flags):
    return ", ".join("TRUE" if f else "FALSE" for f in flags)


def _write(folder, wells, rows=2, columns=3, sites=None, waves=("DAPI", "FITC"),
           timepoints=1, skip=(), suffix="", extra=""):
    """*wells*: (row, col) 0-based; *sites*: a YSites x XSites grid of bools or
    None (no sites); *waves*: names, or None (no waves).  *skip*: (row, col,
    site, wave, t) not written.  -> (HTD path, {key: array})."""
    os.makedirs(folder, exist_ok=True)
    lines = ['"HTSInfoFile", Version 1.0', '"Description", "test plate"',
             f'"TimePoints", {timepoints}', f'"XWells", {columns}', f'"YWells", {rows}']
    for r in range(rows):
        lines.append(f'"WellsSelection{r + 1}", ' + _bools((r, c) in wells for c in range(columns)))
    n_sites = 1
    if sites is None:
        lines.append('"Sites", FALSE')
    else:
        lines += ['"Sites", TRUE', f'"XSites", {len(sites[0])}', f'"YSites", {len(sites)}']
        lines += [f'"SiteSelection{r + 1}", ' + _bools(row) for r, row in enumerate(sites)]
        n_sites = sum(map(sum, sites))
    if waves is None:
        lines.append('"Waves", FALSE')
    else:
        lines += ['"Waves", TRUE', f'"NWavelengths", {len(waves)}']
        lines += [f'"WaveName{n + 1}", "{w}"' for n, w in enumerate(waves)]
    lines += [extra] if extra else []
    lines.append('"EndFile"')
    htd = os.path.join(folder, f"{STEM}.HTD")
    with open(htd, "w", encoding="latin-1", newline="\r\n") as fh:
        fh.write("\n".join(lines) + "\n")
    data = {}
    for t in range(1, timepoints + 1):
        where = os.path.join(folder, f"TimePoint_{t}") if timepoints > 1 else folder
        os.makedirs(where, exist_ok=True)
        for r, c in wells:
            for s in range(1, n_sites + 1):
                for w in range(1, (len(waves) if waves else 1) + 1):
                    if (r, c, s, w, t) in skip:
                        continue
                    name = (f"{STEM}_{chr(65 + r)}{c + 1:02d}" + (f"_s{s}" if sites else "")
                            + (f"_w{w}" if waves else "") + suffix)
                    a = RNG.integers(0, 4000, (Y, X)).astype("u2")
                    tifffile.imwrite(os.path.join(where, name + ".TIF"), a)
                    tifffile.imwrite(os.path.join(where, name + "_thumb.TIF"), a[::2, ::2])
                    data[(r, c, s, w, t)] = a
    return htd, data


def test_a_metaxpress_plate(tmp_path):
    # a 2 x 2 site grid with 3 sites selected: row by row they are sites 1, 2, 3
    grid = [[True, False], [True, True]]
    htd, data = _write(str(tmp_path), wells=[(0, 1), (1, 0)], sites=grid)
    with micro_reader.open(htd) as f:
        assert f.backend == "metaxpress"
        (plate,) = f.plates
        assert plate.name == STEM and (len(plate.rows), len(plate.columns)) == (2, 3)
        assert [w.name for w in plate.wells] == ["A2", "B1"]
        assert f.indices == [{"plate": 0, "acquisition": 0, "well": w, "field": s}
                             for w in range(2) for s in range(3)]
        for image in f.images:
            assert image.shape == (1, 2, 1, Y, X)
            r, c = ((0, 1), (1, 0))[image.index["well"]]
            for w in range(2):
                np.testing.assert_array_equal(image[0, w, 0], data[(r, c, image.index["field"] + 1, w + 1, 1)])
        assert [ch.name for ch in f.images[0].metadata.channels] == ["DAPI", "FITC"]


def test_without_sites_or_waves(tmp_path):
    htd, data = _write(str(tmp_path), wells=[(0, 0)], sites=None, waves=None)
    with micro_reader.open(htd) as f:
        (image,) = f.images
        assert image.index["field"] == 0 and image.shape == (1, 1, 1, Y, X)
        np.testing.assert_array_equal(image[0, 0, 0], data[(0, 0, 1, 1, 1)])


def test_timepoints_in_folders_and_guid_suffixes(tmp_path):
    htd, data = _write(str(tmp_path), wells=[(1, 2)], sites=[[True]], waves=("DAPI",),
                       timepoints=3, suffix="{4B2C1D9E-0000-4000-8000-1234567890AB}")
    with micro_reader.open(htd) as f:
        (image,) = f.images
        assert image.shape == (3, 1, 1, Y, X)
        for t in range(3):
            np.testing.assert_array_equal(image[t, 0, 0], data[(1, 2, 1, 1, t + 1)])


def test_missing_files_follow_the_file_set_rule(tmp_path):
    grid = [[True, True]]
    htd, data = _write(str(tmp_path), wells=[(0, 0), (0, 1)], sites=grid,
                       skip={(0, 0, 2, 2, 1),                     # A1 site 2: FITC missing
                             (0, 1, 1, 1, 1), (0, 1, 1, 2, 1)})   # A2 site 1: nothing
    with pytest.warns(micro_reader.MissingImagesWarning,
                      match=r"Plate7_A01_s2_w2\.TIF.*1 fields miss some planes.*"
                            "1 fields have no files"):
        f = micro_reader.open(htd)
    with f:
        assert f.indices == [{"plate": 0, "acquisition": 0, "well": w, "field": s}
                             for w, s in ((0, 0), (0, 1), (1, 1))]
        a1s2 = f.plates[0].wells[0].fields[1].image
        np.testing.assert_array_equal(a1s2[0, 0, 0], data[(0, 0, 2, 1, 1)])
        assert not a1s2[0, 1, 0].any()


@pytest.mark.parametrize("wrong", ["larger", "smaller", "float", "rgb"])
def test_a_member_that_does_not_fit_raises(tmp_path, wrong):
    """A plane file of another size or type than the index's planes raises
    (it was cropped or cast without a word); the other planes still read.
    The checks are shared by every file-set backend (_fileset.PlaneImage)."""
    htd, data = _write(str(tmp_path), wells=[(0, 0)], sites=[[True, True]])
    tifffile.imwrite(os.path.join(str(tmp_path), f"{STEM}_A01_s2_w2.TIF"),
                     {"larger": np.zeros((2 * Y, X), "u2"), "smaller": np.zeros((Y // 2, X), "u2"),
                      "float": np.full((Y, X), 70000.5, "f4"),
                      "rgb": np.zeros((Y, X, 3), "u1")}[wrong])
    with micro_reader.open(htd) as f:
        good, bad = f.images
        np.testing.assert_array_equal(good[0, 1, 0], data[(0, 0, 1, 2, 1)])
        np.testing.assert_array_equal(bad[0, 0, 0], data[(0, 0, 2, 1, 1)])
        with pytest.raises(micro_reader.CorruptFileError, match=r"_A01_s2_w2\.TIF.*expects one"):
            bad[0, 1, 0]
        with pytest.raises(micro_reader.CorruptFileError, match=r"1 of 4 member files"):
            f.check_members()


def test_check_members(tmp_path):
    htd, _ = _write(str(tmp_path), wells=[(0, 0), (1, 2)], sites=[[True, True]])
    with micro_reader.open(htd) as f:
        assert f.check_members() == 8
    with micro_reader.open(htd, axes="canonical", squeeze=True) as f:    # wrapped images
        assert f.check_members() == 8
    plain = os.path.join(str(tmp_path), "plain.tif")
    tifffile.imwrite(plain, np.zeros((Y, X), "u2"))
    with micro_reader.open(plain) as f:
        assert f.check_members() == 0                            # not a file set


def test_refused(tmp_path):
    htd, _ = _write(str(tmp_path / "z"), wells=[(0, 0)], extra='"ZSteps", 4')
    with pytest.raises(micro_reader.UnsupportedFormatError, match="ZSteps"):
        micro_reader.open(htd)
    htd, _ = _write(str(tmp_path / "dup"), wells=[(0, 0)], sites=None, waves=None)
    tifffile.imwrite(os.path.join(str(tmp_path / "dup"), f"{STEM}_A01.tiff"),
                     np.zeros((Y, X), "u2"))                     # same plane twice
    with pytest.raises(micro_reader.CorruptFileError, match="same plane"):
        micro_reader.open(htd)


def test_parse_htd():
    htd = parse_htd('"XWells", 24\r\n"WellsSelection1", TRUE, FALSE\r\n"WaveName1", "DAPI"\r\n')
    assert htd == {"XWells": [24], "WellsSelection1": [True, False], "WaveName1": ["DAPI"]}
