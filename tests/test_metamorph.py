"""MetaMorph ``.nd`` sets (backends/metamorph.py, docs/COMPOSITE.md): the
``.nd`` file names its TIFF / STK files by a rule; opening any of them opens
the set.  Synthetic sets from ``_metamorph_writer``."""
from __future__ import annotations

import os
import warnings

import numpy as np
import pytest

import micro_reader
from micro_reader.backends import metamorph

tifffile = pytest.importorskip("tifffile")

from _metamorph_writer import member_name, write_metaseries, write_nd, write_set  # noqa: E402

Y, X = 8, 12
UM = "micrometer"


def _plane(data, s, t, w, z):
    return data[(s, t, w)][z]


def test_a_full_set_reads_every_plane(tmp_path):
    nd, data, _ = write_set(str(tmp_path), times=3, stages=["A1", "B7"],
                            waves=[("DAPI", True), ("GFP", True)], nz=4, z_step=1.5)
    with micro_reader.open(nd) as f:
        assert (f.backend, f.format) == ("metamorph", "metamorph")
        assert [im.index for im in f.images] == [{"scene": 0}, {"scene": 1}]
        assert [im.name for im in f.images] == ["exp A1", "exp B7"]
        for s, im in enumerate(f.images):
            assert im.axes == "tczyx" and im.shape == (3, 2, 4, Y, X) and im.dtype == "uint16"
            assert not im.is_ragged
            whole = im[...]
            for t in range(3):
                for w in range(2):
                    np.testing.assert_array_equal(whole[t, w], data[(s, t, w)])
            np.testing.assert_array_equal(im[2, 1, 3, 2:5, 1:9], data[(s, 2, 1)][3, 2:5, 1:9])
        md = f.images[1].metadata
        assert [c.name for c in md.channels] == ["DAPI", "GFP"]
        assert md.pixel_sizes == {"x": micro_reader.Quantity(0.65, UM),
                                  "y": micro_reader.Quantity(0.65, UM),
                                  "z": micro_reader.Quantity(1.5, UM)}
        assert md.position == {}                        # not read (module docstring)
        assert len(f._member_files()) == 2 * 3 * 2


def test_names_follow_the_rule(tmp_path):
    _nd, _data, paths = write_set(str(tmp_path), times=2, stages=["p"], waves=[("Cy 5", True)])
    assert os.path.basename(paths[(0, 1, 0)]) == "exp_w1Cy 5_s1_t2.TIF"
    lay = metamorph.layout(metamorph.read_nd(_nd), _nd)
    assert metamorph.member_key(lay, "EXP_w1cy 5_s1_t2.tif") == (0, 1, 0)   # any letter case
    assert metamorph.member_key(lay, "exp_w1Cy 5_s1_t3.TIF") is None        # t out of range
    assert metamorph.member_key(lay, "exp_w2Cy 5_s1_t1.TIF") is None        # no wavelength 2
    assert metamorph.member_key(lay, "exp_w1GFP_s1_t1.TIF") is None         # another name


def test_the_nd_file_and_its_description(tmp_path):
    nd = write_nd(str(tmp_path / "d.nd"), times=2, waves=[("FRET", False)],
                  description="40xOil, bins4\ntimelapse: 31 points\n")
    info = metamorph.read_nd(nd)
    assert info["Description"] == "40xOil, bins4\ntimelapse: 31 points"
    assert info["NDInfoFile"] == "Version 2.0" and info["NTimePoints"] == 2
    assert info["WaveName1"] == "FRET" and info["WaveDoZ1"] is False and info["EndFile"] is None


def test_any_member_and_the_folder_open_the_set(tmp_path):
    nd, data, paths = write_set(str(tmp_path), times=2, stages=["a", "b"], waves=[("w", True)])
    for opened in (paths[(1, 1, 0)], str(tmp_path)):
        with micro_reader.open(opened) as f:
            assert f.backend == "metamorph" and os.path.samefile(f.path, nd)
            assert f.opened == opened and len(f.images) == 2
            np.testing.assert_array_equal(f.images[1][1, 0, 0], _plane(data, 1, 1, 0, 0))
    # backend="tiff" reads the member alone
    with micro_reader.open(paths[(1, 1, 0)], backend="tiff") as f:
        assert f.backend == "tiff" and f.images[0].shape == (Y, X)
        # the MetaSeries calibration of a TIFF opened on its own
        assert f.images[0].metadata.pixel_sizes["x"] == micro_reader.Quantity(0.65, UM)


def test_a_file_outside_the_rule_opens_alone(tmp_path):
    nd, _data, _paths = write_set(str(tmp_path), times=2, waves=[("w", True)])
    other = str(tmp_path / "exp_w1w_t9.TIF")                      # timepoint 9 of 2
    tifffile.imwrite(other, np.zeros((Y, X), "u2"))
    thumb = str(tmp_path / "exp_w1w_thumb_t1.TIF")
    tifffile.imwrite(thumb, np.zeros((4, 6), "u2"))
    for path in (other, thumb):
        with micro_reader.open(path) as f:
            assert f.backend == "tiff"


def test_wave_not_in_file_name(tmp_path):
    nd, data, paths = write_set(str(tmp_path), times=2, stages=["B2", "B3"],
                                waves=[("BF", True)], waves_in_name=False)
    assert os.path.basename(paths[(1, 0, 0)]) == "exp_s2_t1.TIF"
    with micro_reader.open(paths[(1, 0, 0)]) as f:
        assert len(f.images) == 2 and f.images[0].shape == (2, 1, 1, Y, X)
        assert f.images[0].metadata.channels[0].name == "BF"
        np.testing.assert_array_equal(f.images[1][1, 0, 0], _plane(data, 1, 1, 0, 0))


def test_several_waves_in_one_file_are_refused(tmp_path):
    nd = write_nd(str(tmp_path / "exp.nd"), times=2, waves=[("a", True), ("b", True)],
                  waves_in_name=False)
    member = write_metaseries(str(tmp_path / "exp_t1.TIF"), np.zeros((2, Y, X), "u2"))
    for path in (nd, member):                         # the member is claimed: it says why
        with pytest.raises(micro_reader.UnsupportedFormatError, match="backend='tiff'"):
            micro_reader.open(path)
    with micro_reader.open(member, backend="tiff") as f:
        assert f.backend == "tiff"


def test_wave_points_collected_is_refused(tmp_path):
    nd = write_nd(str(tmp_path / "exp.nd"), times=3, waves=[("a", True), ("b", True)],
                  extra=[("WavePointsCollected2", "2, 1, 3")])
    with pytest.raises(micro_reader.UnsupportedFormatError, match="WavePointsCollected"):
        micro_reader.open(nd)


def test_no_wavelengths_like_the_zenodo_sample(tmp_path):
    """DoWave FALSE: no _w part; the channel's name comes from the members'
    illumination setting (the .nd file names none: a gap filled)."""
    nd, data, paths = write_set(str(tmp_path), times=3, stages=["B2", "B3", "C4"],
                                waves_in_name=False, illumination="Brightfield")
    assert os.path.basename(paths[(2, 1, 0)]) == "exp_s3_t2.TIF"
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with micro_reader.open(nd) as f:
            assert [im.name for im in f.images] == ["exp B2", "exp B3", "exp C4"]
            assert f.images[2].metadata.channels[0].name == "Brightfield"
            np.testing.assert_array_equal(f.images[2][1, 0, 0], _plane(data, 2, 1, 0, 0))


def test_the_nd_names_win_over_the_members(tmp_path):
    nd, _data, _paths = write_set(str(tmp_path), times=1, waves=[("GFP", True)],
                                  illumination="FITC")
    with micro_reader.open(nd) as f:
        with pytest.warns(micro_reader.CompositeWarning, match="'GFP' in the .nd file, 'FITC'"):
            names = [c.name for c in f.images[0].metadata.channels]
        assert names == ["GFP"]


def test_a_dimension_of_one_may_leave_its_part_out(tmp_path):
    """DoTimelapse TRUE with one timepoint / DoStage TRUE with one position:
    the file is found with or without its _t1 / _s1."""
    write_nd(str(tmp_path / "exp.nd"), times=1, stages=["only"], waves=[("w", True)])
    a = np.full((1, Y, X), 7, "u2")
    write_metaseries(str(tmp_path / "exp_w1w.TIF"), a)
    with micro_reader.open(str(tmp_path / "exp.nd")) as f:
        np.testing.assert_array_equal(f.images[0][0, 0, 0], a[0])
    # both spellings: neither is used
    write_metaseries(str(tmp_path / "exp_w1w_s1_t1.TIF"), a)
    with pytest.warns(micro_reader.CompositeWarning, match="more than one file"), \
            pytest.warns(micro_reader.MissingImagesWarning):
        with pytest.raises(micro_reader.CorruptFileError, match="none of the fields"):
            micro_reader.open(str(tmp_path / "exp.nd"))


def test_missing_files_read_as_zeros(tmp_path):
    with pytest.warns(micro_reader.MissingImagesWarning, match="3 image files are missing") as rec:
        nd, data, _ = write_set(str(tmp_path), times=2, stages=["a", "b", "c"],
                                waves=[("w", True)], skip={(0, 1, 0), (2, 0, 0), (2, 1, 0)})
        f = micro_reader.open(nd)
    message = str(rec[0].message)
    assert "1 fields miss some planes" in message and "1 fields have no files" in message
    with f:
        assert [im.index["scene"] for im in f.images] == [0, 1]       # position c left out
        assert not f.images[0][1].any()
        np.testing.assert_array_equal(f.images[0][0, 0, 0], _plane(data, 0, 0, 0, 0))


def test_a_ragged_set_has_parts(tmp_path):
    """A wavelength without z (WaveDoZ FALSE) next to a z-stack: placed at its
    focus position (the middle slice here), zeros elsewhere."""
    nd, data, _ = write_set(str(tmp_path), times=2, waves=[("DAPI", True), ("BF", False)],
                            nz=5, z_step=2.0)
    with micro_reader.open(nd) as f:
        (whole,) = f.images
        assert whole.is_ragged and whole.shape == (2, 2, 5, Y, X)
        stack, flat = whole.parts
        assert stack.shape == (2, 1, 5, Y, X) and flat.shape == (2, 1, 1, Y, X)
        assert [c.name for c in stack.metadata.channels] == ["DAPI"]
        assert [c.name for c in flat.metadata.channels] == ["BF"]
        assert stack.metadata.pixel_sizes["z"] == micro_reader.Quantity(2.0, UM)
        assert "z" not in flat.metadata.pixel_sizes
        np.testing.assert_array_equal(flat[1, 0, 0], _plane(data, 0, 1, 1, 0))
        np.testing.assert_array_equal(stack[1, 0], data[(0, 1, 0)])
        w = whole[...]
        np.testing.assert_array_equal(w[1, 0], data[(0, 1, 0)])
        np.testing.assert_array_equal(w[1, 1, 2], _plane(data, 0, 1, 1, 0))
        assert not np.delete(w[:, 1], 2, axis=1).any()


def test_stk_members(tmp_path):
    """STK files (one IFD, the planes after each other; tifffile calls them T
    by their times -- the .nd file says they are z)."""
    nd, data, paths = write_set(str(tmp_path), times=2, stages=["a", "b"],
                                waves=[("DAPI", True), ("BF", False)], nz=3, z_step=0.5, stk=True)
    assert paths[(1, 1, 0)].endswith("exp_w1DAPI_s2_t2.STK")
    with micro_reader.open(paths[(1, 1, 1)]) as f:
        assert f.backend == "metamorph" and len(f.images) == 2
        whole = f.images[1]
        assert whole.is_ragged and whole.shape == (2, 2, 3, Y, X)
        w = whole[...]
        np.testing.assert_array_equal(w[1, 0], data[(1, 1, 0)])
        np.testing.assert_array_equal(w[0, 1, 1], _plane(data, 1, 0, 1, 0))   # AbsoluteZ: middle
        assert whole.metadata.pixel_sizes == {"x": micro_reader.Quantity(1.25, UM),
                                              "y": micro_reader.Quantity(1.25, UM),
                                              "z": micro_reader.Quantity(0.5, UM)}


def test_a_member_of_the_wrong_depth_is_an_error(tmp_path):
    nd, _data, paths = write_set(str(tmp_path), times=2, waves=[("w", True)], nz=3)
    write_metaseries(paths[(0, 1, 0)], np.zeros((2, Y, X), "u2"))      # 2 planes, not 3
    with micro_reader.open(nd) as f:
        f.images[0][0]                                                 # timepoint 0 is fine
        with pytest.raises(micro_reader.CorruptFileError, match="expects 3"):
            f.images[0][1]
        with pytest.raises(micro_reader.CorruptFileError, match="1 of 2 member files"):
            f.check_members()


def test_group_files(tmp_path):
    nd, _data, paths = write_set(str(tmp_path), times=2, stages=["a", "b"], waves=[("w", True)])
    thumb = str(tmp_path / "exp_w1w_s1_thumb.TIF")
    tifffile.imwrite(thumb, np.zeros((4, 6), "u2"))
    lone = str(tmp_path / "notes.tif")
    tifffile.imwrite(lone, np.zeros((4, 6), "u2"))
    g = micro_reader.group_files(str(tmp_path))
    (group,) = g.groups
    assert (group.kind, group.format, group.notes) == ("index", "metamorph", ())
    assert os.path.samefile(group.main, nd)
    assert {os.path.normcase(p) for p in group.members} == \
        {os.path.normcase(p) for p in paths.values()}
    assert [os.path.basename(p) for p in group.parts] == [os.path.basename(thumb)]
    assert [os.path.basename(p) for p in g.singles] == ["notes.tif"]
    # members given without their .nd file: one composite group under it
    g = micro_reader.group_files(list(paths.values()))
    (group,) = g.groups
    assert group.kind == "composite" and os.path.samefile(group.expected, nd)
    assert len(group.parts) == len(paths) and not g.singles


def test_member_name_helper():
    assert member_name("x", 2, 3, 1, "GFP") == "x_w1GFP_s2_t3.TIF"
