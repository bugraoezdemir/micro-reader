"""Micro-Manager image-per-file acquisitions (backends/micromanager.py,
docs/COMPOSITE.md): a ``metadata.txt`` per stage position folder and one TIFF
per plane; opening any part opens the acquisition.  Synthetic acquisitions
from ``_micromanager_writer``."""
from __future__ import annotations

import json
import os
import warnings

import numpy as np
import pytest

import micro_reader
from micro_reader import _parts
from micro_reader.backends import micromanager

tifffile = pytest.importorskip("tifffile")

from _micromanager_writer import name_14, name_20, write_acquisition, write_plane  # noqa: E402

Y, X = 8, 12
UM = "micrometer"


def _check_all(f, data):
    """Every plane written equals the image's; every other plane is zeros."""
    by_scene = {im.index["series"]: im[...] for im in f.images}
    for (p, t, c, z), plane in data.items():
        if p in by_scene:
            np.testing.assert_array_equal(by_scene[p][t, c, z], plane)


def test_one_position_1_4(tmp_path):
    root = str(tmp_path / "acq_1")
    data, files = write_acquisition(root, frames=3, channels=("FITC", "Rhodamine"), slices=2,
                                    pixel_size=0.65, z_step=1.5, interval=2000,
                                    colors=[-13369549, -65536], prefix="acq_1")
    assert os.path.basename(os.path.dirname(files[0])) == "Pos0"
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with micro_reader.open(files[0]) as f:
            assert (f.backend, f.format) == ("micromanager", "micromanager-files")
            (im,) = f.images
            assert im.axes == "tczyx" and im.shape == (3, 2, 2, Y, X) and im.dtype == "uint16"
            assert im.index == {"series": 0} and im.name == "acq_1"
            _check_all(f, data)
            np.testing.assert_array_equal(im[2, 1, 1, 2:5, 3:9], data[(0, 2, 1, 1)][2:5, 3:9])
            md = im.metadata
            assert md.pixel_sizes == {"x": micro_reader.Quantity(0.65, UM),
                                      "y": micro_reader.Quantity(0.65, UM),
                                      "z": micro_reader.Quantity(1.5, UM),
                                      "t": micro_reader.Quantity(2000.0, "millisecond")}
            assert [(c.name, c.color) for c in md.channels] == [("FITC", "#33FF33"),
                                                                ("Rhodamine", "#FF0000")]
            assert md.position == {}
            assert len(f._member_files()) == 3 * 2 * 2


def test_the_2_0_beta_naming(tmp_path):
    """``Coords-<file>`` entries and ``img_channel<c>_position<p>_time<t>_z<z>.tif``
    names; the pixel size and stage coordinates only in the TIFFs (gaps filled)."""
    root = str(tmp_path / "beta")
    data, files = write_acquisition(root, style="2.0", frames=2, channels=("FITC", "DAPI"),
                                    positions=2, prefix="", tiff_pixel_size=0.5,
                                    stage=lambda p: (10.0 + p, -20.0, 3.5))
    assert os.path.isfile(os.path.join(root, "Pos1", name_20(1, 1, 1, 0)))
    with micro_reader.open(os.path.join(root, "Pos1", name_20(0, 1, 1, 0))) as f:
        assert f.backend == "micromanager" and os.path.samefile(f.path, files[0])
        assert [im.name for im in f.images] == ["beta Pos0", "beta Pos1"]
        assert [im.shape for im in f.images] == [(2, 2, 1, Y, X)] * 2
        _check_all(f, data)
        md = f.images[1].metadata
        assert md.pixel_sizes == {"x": micro_reader.Quantity(0.5, UM),
                                  "y": micro_reader.Quantity(0.5, UM)}
        assert md.stage_position == {"x": (11.0, UM), "y": (-20.0, UM), "z": (3.5, UM)}
        assert [c.name for c in md.channels] == ["FITC", "DAPI"]
        assert [c.color for c in md.channels] == [None, None]


def test_positions_are_scenes_and_any_part_opens_them(tmp_path):
    root = str(tmp_path / "multi")
    labels = ["A1-Site_0", "A1-Site_1", "B2-Site_0"]
    data, files = write_acquisition(root, frames=2, positions=3, labels=labels,
                                    stage=lambda p: (100.0 * p, -50.0 * p, 2880.25))
    plane = os.path.join(root, "B2-Site_0", name_14(1, "GFP", 0))
    for opened in (plane, files[2], os.path.join(root, "A1-Site_1"), root,
                   os.path.join(root, "display_and_comments.txt")):
        with micro_reader.open(opened) as f:
            assert f.backend == "micromanager" and os.path.samefile(f.path, files[0])
            assert f.opened == opened
            assert [im.index for im in f.images] == [{"series": k} for k in range(3)]
            assert [im.name for im in f.images] == [f"acq {label}" for label in labels]
            np.testing.assert_array_equal(f.images[2][1, 1, 0], data[(2, 1, 1, 0)])
    with micro_reader.open(root) as f:
        _check_all(f, data)
        md = f.images[2].metadata
        assert md.stage_position == {"x": (200.0, UM), "y": (-100.0, UM), "z": (2880.25, UM)}
        assert md.position == {}
    # backend="tiff" reads one plane alone
    with micro_reader.open(plane, backend="tiff") as f:
        assert f.backend == "tiff" and f.images[0].shape == (Y, X)


def test_pos_k_folders_without_a_position_list(tmp_path):
    root = str(tmp_path / "plain")
    data, files = write_acquisition(root, frames=1, positions=2, labels=None)
    assert sorted(os.listdir(root)) == ["Pos0", "Pos1", "display_and_comments.txt"]
    with micro_reader.open(os.path.join(root, "Pos1")) as f:
        assert [im.name for im in f.images] == ["acq Pos0", "acq Pos1"]
        _check_all(f, data)


def test_a_position_folder_that_is_missing(tmp_path):
    """The Summary lists four positions, two folders are there (the OME serge
    sample): those two read, the others are left out with a warning."""
    root = str(tmp_path / "serge")
    data, files = write_acquisition(root, positions=4, labels=["Pos0", "Pos1", "Pos2", "Pos3"],
                                    skip_positions={0, 2})
    with pytest.warns(micro_reader.MissingImagesWarning,
                      match=r"2 of the 4 stage positions .* \(Pos0, Pos2\): left out"):
        f = micro_reader.open(root)
    with f:
        assert os.path.samefile(f.path, files[1])          # the first position there
        assert [im.index["series"] for im in f.images] == [1, 3]
        _check_all(f, data)


def test_missing_and_unlisted_planes_read_as_zeros(tmp_path):
    root = str(tmp_path / "holes")
    with pytest.warns(micro_reader.MissingImagesWarning, match="2 image files are missing"):
        data, files = write_acquisition(root, frames=3, positions=2,
                                        skip={(0, 1, 0, 0)}, unlisted={(1, 2, 1, 0)})
        f = micro_reader.open(files[0])
    with f:
        assert [im.shape for im in f.images] == [(3, 2, 1, Y, X)] * 2
        assert not f.images[0][1, 0, 0].any()
        assert not f.images[1][2, 1, 0].any()
        np.testing.assert_array_equal(f.images[1][2, 0, 0], data[(1, 2, 0, 0)])


def test_pixel_size_0_is_unknown_and_1_is_kept(tmp_path):
    """0 is Micro-Manager's "no calibration" (OME serge sample, where
    Bio-Formats gives none too); 1 is a calibration (the demo configuration's
    10x objective: OME thomas sample, Bio-Formats 1 um)."""
    for size, want in ((0, None), (1, micro_reader.Quantity(1.0, UM))):
        root = str(tmp_path / f"px{size}")
        _data, files = write_acquisition(root, pixel_size=size)
        with micro_reader.open(files[0]) as f:
            assert f.images[0].metadata.pixel_sizes.get("x") == want


def test_metadata_txt_wins_over_the_tiffs(tmp_path):
    root = str(tmp_path / "disagree")
    _data, files = write_acquisition(root, pixel_size=0.65, tiff_pixel_size=0.5)
    with micro_reader.open(files[0]) as f:
        with pytest.warns(micro_reader.CompositeWarning, match="0.65 in the Summary, 0.5 in"):
            sizes = f.images[0].metadata.pixel_sizes
        assert sizes["x"] == micro_reader.Quantity(0.65, UM)


def test_time_step(tmp_path):
    """Interval_ms; 0 with uniform CustomIntervals_ms (the serge sample): those;
    0 without: unknown."""
    cases = (({"interval": 2000}, 2000.0),
             ({"interval": 0, "custom": [0, 300000, 300000]}, 300000.0),
             ({"interval": 0, "custom": [0, 1000, 2000]}, None),
             ({"interval": 0}, None))
    for k, (kw, want) in enumerate(cases):
        _data, files = write_acquisition(str(tmp_path / f"t{k}"), frames=3, **kw)
        with micro_reader.open(files[0]) as f:
            got = f.images[0].metadata.pixel_sizes.get("t")
            assert got == (None if want is None else micro_reader.Quantity(want, "millisecond"))


def test_a_folder_of_another_acquisition_is_not_used(tmp_path):
    root = str(tmp_path / "mixed")
    _data, files = write_acquisition(root, positions=2, labels=["Pos0", "Pos1"])
    md = files[1]
    other = json.load(open(md, encoding="utf-8"))
    other["Summary"]["UUID"] = "another"
    with open(md, "w", encoding="utf-8") as fh:
        json.dump(other, fh)
    with pytest.warns(micro_reader.CompositeWarning, match="not a file of the acquisition"), \
            pytest.warns(micro_reader.MissingImagesWarning, match="1 of the 2 stage positions"):
        f = micro_reader.open(files[0])
    with f:
        assert len(f.images) == 1


def test_a_plane_file_outside_the_rule_opens_alone(tmp_path):
    root = str(tmp_path / "acq")
    _data, files = write_acquisition(root)
    lone = os.path.join(root, "Pos0", "snap.tif")
    tifffile.imwrite(lone, np.zeros((Y, X), "u2"))
    with micro_reader.open(lone) as f:
        assert f.backend == "tiff"
    # a plane-named TIFF without a metadata.txt next to it is a plain TIFF
    other = tmp_path / "elsewhere"
    other.mkdir()
    path = str(other / name_14(0, "DAPI", 0))
    write_plane(path, np.zeros((Y, X), "u2"), {"FileName": "x"})
    with micro_reader.open(path) as f:
        assert f.backend == "tiff"


def test_broken_metadata_txt_is_an_error(tmp_path):
    root = str(tmp_path / "cut")
    _data, files = write_acquisition(root)
    text = open(files[0], encoding="utf-8").read()
    with open(files[0], "w", encoding="utf-8") as fh:
        fh.write(text[: len(text) * 2 // 3])                # an acquisition that did not finish
    with pytest.raises(micro_reader.CorruptFileError, match="not complete JSON"):
        micro_reader.open(files[0])
    plane = os.path.join(root, "Pos0", name_14(0, "DAPI", 0))
    with micro_reader.open(plane, backend="tiff") as f:     # the way out it names
        assert f.images[0].shape == (Y, X)


def test_group_files(tmp_path):
    root = str(tmp_path / "grp")
    _data, files = write_acquisition(root, frames=2, positions=2, labels=["Pos0", "Pos1"])
    (tmp_path / "grp" / "notes.txt").write_text("x")
    g = micro_reader.group_files(root)
    (group,) = g.groups
    assert (group.kind, group.format, group.notes) == ("index", "micromanager", ())
    assert os.path.samefile(group.main, files[0])
    assert len(group.members) == 2 * 2 * 2
    assert sorted(os.path.relpath(p, root) for p in group.parts) == \
        [os.path.join("Pos1", "metadata.txt"), "display_and_comments.txt"]
    assert [os.path.basename(p) for p in g.singles] == ["notes.txt"]
    # planes of one position given alone: one composite group under the main file
    planes = [os.path.join(root, "Pos1", n) for n in os.listdir(os.path.join(root, "Pos1"))
              if n.endswith(".tif")]
    g = micro_reader.group_files(planes)
    (group,) = g.groups
    assert group.kind == "composite" and os.path.samefile(group.expected, files[0])
    assert len(group.parts) == 4 and not g.singles


def test_owner_and_sniff(tmp_path):
    root = str(tmp_path / "o")
    _data, files = write_acquisition(root, positions=2, labels=["Pos0", "Pos1"])
    assert micromanager.sniff(_parts.read_head(files[0]), files[0])
    assert _parts.part_of(files[0], _parts.read_head(files[0]), [micromanager]) is None   # main
    part = _parts.part_of(files[1], _parts.read_head(files[1]), [micromanager])
    assert os.path.samefile(part.main, files[0]) and not part.standalone
    plane = os.path.join(root, "Pos1", name_14(0, "GFP", 0))
    part = _parts.part_of(plane, _parts.read_head(plane), [micromanager])
    assert os.path.samefile(part.main, files[0]) and part.standalone
    assert micromanager.plane_key(name_14(4, "GFP", 2), {"ChNames": ["DAPI", "GFP"]}) == (4, 1, 2)
    assert micromanager.plane_key(name_20(4, 1, 0, 2), {}) == (4, 1, 2)
    assert micromanager.plane_key(name_14(4, "Cy5", 2), {"ChNames": ["DAPI"]}) is None


def test_micro_manager_stacks_still_open_as_stacks(tmp_path):
    """MMStack files and their ``<prefix>_MMStack_Pos0_metadata.txt`` (which
    also starts with a Summary) stay the TIFF backend's, as does a stack
    acquisition's ``display_and_comments.txt``."""
    names = ["run_MMStack_Pos0.ome.tif", "run_MMStack_Pos0_1.ome.tif"]
    for name in names:
        tifffile.imwrite(tmp_path / name, np.zeros((4, 5), np.uint8))
    side = tmp_path / "run_MMStack_Pos0_metadata.txt"
    side.write_text('{\n"Summary": {"Positions": 1, "ChNames": ["a"]}\n}')
    (tmp_path / "display_and_comments.txt").write_text('{"Channels": []}')
    for opened in (str(tmp_path), str(side), str(tmp_path / names[1])):
        with micro_reader.open(opened) as f:
            assert f.backend == "tiff" and os.path.basename(f.path) == names[0]
    assert not micromanager.sniff(_parts.read_head(str(side)), str(side))


@pytest.mark.parametrize("micromanager", [True, False])
def test_mmstack_ome_colours_are_argb(tmp_path, micromanager):
    """Micro-Manager writes Java ARGB ints into OME Channel Color (RGBA by the
    schema): -13369549 is FITC green #33FF33 in a Micro-Manager file, and
    stays OME's RGBA #FF33FF in any other OME-TIFF."""
    import tifffile
    path = str(tmp_path / "s_MMStack_Pos0.ome.tif")
    data = np.zeros((2, 8, 9), np.uint16)
    extra = [(51123, "s", 0, '{"Source": "Micro-Manager"}', True)] if micromanager else []
    tifffile.imwrite(path, data, ome=True, photometric="minisblack", extratags=extra,
                     metadata={"axes": "CYX", "Channel": {"Name": ["FITC", "Rhodamine"],
                                                          "Color": [-13369549, -65536]}})
    with tifffile.TiffFile(path) as t:
        assert t.is_micromanager == micromanager and 'Color="-13369549"' in t.ome_metadata
    with micro_reader.open(path) as f:
        colours = [c.color for c in f.images[0].metadata.channels]
    assert colours == (["#33FF33", "#FF0000"] if micromanager else ["#FF33FF", "#FFFF00"])


def test_the_2_0_0_naming_with_the_position_folder_in_the_file_name(tmp_path):
    """Micro-Manager 2.0.0 (the Zenodo 6659532 sample): ``Coords-Pos0/img_...``
    entries with FrameIndex / ChannelIndex / SliceIndex values and the plane's
    metadata in ``Metadata-Pos0/img_...``.  The name is relative to the
    acquisition folder, not the position folder."""
    root = str(tmp_path / "HJ")
    data, files = write_acquisition(root, style="2.0.0", frames=3, channels=("637", "532"),
                                    positions=2, prefix="HJ",
                                    stage=lambda p: (400.0 + p, 897.75, 99.7))
    listed = json.load(open(files[1], encoding="utf-8"))
    assert f"Coords-Pos1/{name_20(2, 1, 1, 0)}" in listed
    assert listed[f"Coords-Pos1/{name_20(2, 1, 1, 0)}"]["FrameIndex"] == 2
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with micro_reader.open(os.path.join(root, "Pos1", name_20(0, 1, 1, 0))) as f:
            assert f.backend == "micromanager" and os.path.samefile(f.path, files[0])
            assert [im.name for im in f.images] == ["HJ Pos0", "HJ Pos1"]
            assert [im.shape for im in f.images] == [(3, 2, 1, Y, X)] * 2
            _check_all(f, data)
            md = f.images[1].metadata
            assert [c.name for c in md.channels] == ["637", "532"]
            assert md.pixel_sizes == {}                           # PixelSizeUm 0: unknown
            assert md.stage_position == {"x": (401.0, UM), "y": (897.75, UM), "z": (99.7, UM)}
            assert len(f._member_files()) == 2 * 3 * 2
    assert micromanager._coords({"Frame": 3, "FrameIndex": 3, "PositionIndex": 0, "Slice": 0,
                                 "SliceIndex": 0, "ChannelIndex": 1}) == (3, 1, 0)
    assert micromanager._coords({"position": 0, "time": 2, "z": 4, "channel": 1}) == (2, 1, 4)
    with pytest.raises(ValueError):
        micromanager._coords({"time": "two"})


def test_a_renamed_position_folder_keeps_its_2_0_0_planes(tmp_path):
    """``Pos0/img_...`` names a folder that is not there any more: the bare
    name next to ``metadata.txt`` is used."""
    root = str(tmp_path / "HJ")
    data, files = write_acquisition(root, style="2.0.0", frames=2, prefix="HJ")
    os.rename(os.path.join(root, "Pos0"), os.path.join(root, "site"))
    md = os.path.join(root, "site", "metadata.txt")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with micro_reader.open(md) as f:
            (im,) = f.images
            assert im.shape == (2, 2, 1, Y, X)
            _check_all(f, data)
    assert micromanager._plane_path(os.path.join(root, "site"), "Pos0/x.tif") == \
        os.path.join(root, "Pos0", "x.tif")                   # neither there: the listed one


DATA = os.environ.get("MICROSCOPY_DATA", os.path.expanduser("~/Desktop/data/microscopy"))


def _sample(*parts):
    path = os.path.join(DATA, "formats", "tiff", "micromanager", *parts)
    if not os.path.exists(path):
        pytest.skip(f"sample not found: {path} (set MICROSCOPY_DATA)")
    return path


def test_zenodo_6659532_micro_manager_2_0_0_subset():
    """The MARS subset (CC BY 4.0), cut to 2 of 20 positions and t 0-24 of
    1000: two MissingImagesWarnings, and an image per position of the sizes
    the Summary gives, with the planes that are there."""
    root = _sample("zenodo-6659532-mars-subset")
    with pytest.warns(micro_reader.MissingImagesWarning) as rec:
        f = micro_reader.open(root)
    messages = [str(w.message) for w in rec if w.category is micro_reader.MissingImagesWarning]
    assert any("18 of the 20 stage positions" in m for m in messages)
    assert any("3900 image files are missing" in m for m in messages)
    with f:
        assert os.path.samefile(f.path, os.path.join(root, "Pos0", "metadata.txt"))
        assert [im.name for im in f.images] == ["HJ Pos0", "HJ Pos1"]
        assert [im.shape for im in f.images] == [(1000, 2, 1, 1024, 1024)] * 2
        assert len(f._member_files()) == 2 * 25 * 2
        im = f.images[1]
        assert [c.name for c in im.metadata.channels] == ["637", "532"]
        assert im.metadata.pixel_sizes == {}                  # PixelSizeUm 0.0: unknown
        assert im.metadata.stage_position["y"] == (897.85, UM)
        ref = tifffile.imread(os.path.join(root, "Pos1", name_20(24, 1, 1, 0)))
        np.testing.assert_array_equal(im[24, 1, 0], ref)
        assert not im[25, 1, 0].any()                         # cut from the subset


@pytest.mark.parametrize("parts, backend, shapes, names", [
    (("ome-serge",), "micromanager", [(7, 4, 1, 2048, 2048)] * 2,
     ["YAV_1724_1778_test_lectin_1 Pos0", "YAV_1724_1778_test_lectin_1 Pos1"]),
    (("ome-thomas", "test", "test_sep"), "micromanager", [(4, 2, 1, 128, 128)], ["test_sep"]),
    (("ome-thomas", "test", "test2_sep"), "micromanager", [(2, 2, 1, 128, 128)], ["test2_sep"]),
    (("aics-mm2-2pos", "image_stack_tpzc_50tp_2p_5z_3c_512k_1_MMStack_2-Pos000_000.ome.tif"),
     "tiff", [(50, 5, 3, 256, 256)] * 2, ["series0", "series1"]),
], ids=["serge-1.4.16", "thomas-1.4.22", "thomas-2.0-beta3", "aics-2.0-mmstack"])
def test_the_other_micro_manager_samples_still_open(parts, backend, shapes, names):
    """1.4 image-per-file (bare FileName), the 2.0 beta and a 2.0 MMStack
    (the TIFF backend's) as before."""
    path = _sample(*parts)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", micro_reader.MissingImagesWarning)   # serge: 4 of 6 folders
        with micro_reader.open(path) as f:
            assert f.backend == backend
            assert [im.shape for im in f.images] == shapes
            assert [im.name for im in f.images] == names
            assert f.images[-1][0, 0, 0].any()
