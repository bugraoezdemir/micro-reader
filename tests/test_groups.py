"""group_files (docs/COMPOSITE.md, "Grouping files"): which of many files
make one image -- composite parts under their main file, an index file's
members in its group, every other file a plain path that is never read."""
from __future__ import annotations

import os
import shutil

import numpy as np
import pytest

import micro_reader
from micro_reader import FileGroup, _parts

tifffile = pytest.importorskip("tifffile")

import test_companion  # noqa: E402
import test_metaxpress  # noqa: E402
import test_oir  # noqa: E402
from _ndtiff_writer import write_ndtiff  # noqa: E402
from _oif_writer import write_oif  # noqa: E402
from _vsi_writer import image_volume, write_vsi  # noqa: E402
from test_dv import _data as dv_data, write_dv  # noqa: E402

ACQUISITION_LOG = "#\n#Tag 0x00FF0408\n\nResolve3D Image File: /data/{name}\n"


def _names(paths, root):
    return sorted(os.path.relpath(p, root).replace(os.sep, "/") for p in paths)


def _by_main(grouping, root):
    return {os.path.relpath(g.expected, root).replace(os.sep, "/"): g for g in grouping.groups}


@pytest.fixture
def mixed(tmp_path):
    """One folder holding several multi-file images and some plain files."""
    root = tmp_path
    write_vsi(root, "slide", [image_volume(1, (20, 30))],
              {1: {"levels": [np.zeros((20, 30), np.uint16)]}})
    write_oif(str(root), "cells", np.zeros((1, 2, 2, 9, 11), np.uint16))
    write_dv(root / "a_R3D.dv", dv_data((1, 1, 2, 5, 6)))
    (root / "a_R3D.dv.log").write_text(ACQUISITION_LOG.format(name="a_R3D.dv"))
    test_oir._write(root, "companion_file")                         # main + _00001
    for name in ("run_MMStack_Pos0.ome.tif", "run_MMStack_Pos0_1.ome.tif"):
        tifffile.imwrite(root / name, np.zeros((4, 5), np.uint8))
    frames = [({"time": t}, np.zeros((6, 7), np.uint16)) for t in range(4)]
    write_ndtiff(str(root / "nd"), frames, per_file=2)
    test_metaxpress._write(str(root / "plate"), wells=[(0, 0)], sites=[[True, True]])
    tifffile.imwrite(root / "plain.tif", np.zeros((4, 4), np.uint8))
    (root / "notes.txt").write_text("not an image")
    return root


def test_a_folder_groups_into_images(mixed):
    g = micro_reader.group_files(str(mixed))
    groups = _by_main(g, mixed)
    assert sorted(groups) == ["a_R3D.dv", "cells.oif", "companion_file.oir", "nd/data_NDTiffStack.tif",
                              "plate/Plate7.HTD", "run_MMStack_Pos0.ome.tif", "slide.vsi"]
    vsi = groups["slide.vsi"]
    assert (vsi.kind, vsi.format) == ("composite", "vsi")
    assert _names(vsi.parts, mixed) == ["_slide_/stack1/frame_t.ets"]
    assert os.path.samefile(vsi.main, mixed / "slide.vsi")
    assert len(groups["cells.oif"].parts) > 4 and groups["cells.oif"].format == "oif"
    assert _names(groups["a_R3D.dv"].parts, mixed) == ["a_R3D.dv.log"]
    assert _names(groups["companion_file.oir"].parts, mixed) == ["companion_file_00001"]
    assert _names(groups["run_MMStack_Pos0.ome.tif"].parts, mixed) == ["run_MMStack_Pos0_1.ome.tif"]
    assert _names(groups["nd/data_NDTiffStack.tif"].parts, mixed) == \
        ["nd/NDTiff.index", "nd/data_NDTiffStack_1.tif"]
    plate = groups["plate/Plate7.HTD"]
    assert (plate.kind, plate.format, plate.notes) == ("index", "metaxpress", ())
    assert len(plate.members) == 4 and all(os.path.isfile(m) for m in plate.members)
    # the plate's thumbnails are not named by the index, but are its parts
    assert _names(plate.parts, mixed) == sorted(
        f"plate/Plate7_A01_s{s}_w{w}_thumb.TIF" for s in (1, 2) for w in (1, 2))
    assert _names(g.singles, mixed) == ["notes.txt", "plain.tif"]


def test_every_file_is_in_one_place(mixed):
    g = micro_reader.group_files(str(mixed))
    placed = list(g.singles)
    for group in g.groups:
        placed += list(group.parts) + list(group.members) + [group.main]
    keys = [os.path.normcase(os.path.abspath(p)) for p in placed]
    assert len(keys) == len(set(keys))
    walked = {os.path.normcase(os.path.join(r, n)) for r, _, files in os.walk(mixed) for n in files}
    assert set(keys) == walked


def test_a_metaxpress_preview_opens_its_plate(tmp_path):
    htd, _ = test_metaxpress._write(str(tmp_path), wells=[(0, 0)], sites=[[True]])
    thumb = os.path.join(str(tmp_path), "Plate7_A01_s1_w1_thumb.TIF")
    with micro_reader.open(thumb) as f:
        assert f.backend == "metaxpress" and os.path.samefile(f.path, htd)
    os.remove(htd)                                               # no .HTD: just a TIFF
    lone = os.path.join(str(tmp_path), "Plate7_A01_s1_w2_thumb.TIF")
    with micro_reader.open(lone) as f:
        assert f.backend == "tiff"


def test_harmony_side_folders_open_the_measurement(tmp_path):
    import test_harmony
    index = test_harmony._write_harmony(str(tmp_path / "Measurement 1"))[0]
    images = tmp_path / "Measurement 1" / "Images"
    for side, name in (("Assaylayout", "layout.xml"), ("FFC_Profile", "FFC_Profile_Measurement 1.xml")):
        os.makedirs(images.parent / side)
        (images.parent / side / name).write_text("<Results/>")
    for target in (images.parent, images.parent / "Assaylayout", images.parent / "FFC_Profile" /
                   "FFC_Profile_Measurement 1.xml"):
        with micro_reader.open(str(target)) as f:
            assert f.backend == "harmony" and os.path.samefile(f.path, index)
    (group,) = micro_reader.group_files(str(tmp_path)).groups
    assert _names(group.parts, images.parent) == ["Assaylayout/layout.xml",
                                                  "FFC_Profile/FFC_Profile_Measurement 1.xml"]


def test_opening_a_plate_folder_reads_only_its_index(tmp_path, monkeypatch):
    htd, _ = test_metaxpress._write(str(tmp_path), wells=[(0, 0), (0, 1)], sites=[[True, True]])
    reads = []
    real = _parts.read_head
    monkeypatch.setattr(_parts, "read_head", lambda p: reads.append(os.path.basename(p)) or real(p))
    with micro_reader.open(str(tmp_path)) as f:
        assert os.path.samefile(f.path, htd)
    assert reads == ["Plate7.HTD"]


def test_plain_files_are_never_read(tmp_path, monkeypatch):
    for t in range(3):
        for z in range(20):
            (tmp_path / f"img_t{t}_z{z:03d}.tif").write_bytes(b"")
    reads = []
    real = _parts.read_head
    monkeypatch.setattr(_parts, "read_head", lambda p: reads.append(p) or real(p))
    g = micro_reader.group_files(str(tmp_path))
    assert (len(g.singles), g.groups, reads) == (60, [], [])


def test_given_files_and_a_missing_main_file(tmp_path):
    path = write_oif(str(tmp_path), "cells", np.zeros((1, 2, 1, 9, 11), np.uint16))
    planes = sorted(os.path.join(path + ".files", n) for n in os.listdir(path + ".files")
                    if n.endswith(".tif"))
    g = micro_reader.group_files(planes)                         # parts only: the main is found
    (group,) = g.groups
    assert os.path.samefile(group.main, path) and list(group.parts) == planes and not g.singles
    os.remove(path)
    (group,) = micro_reader.group_files(planes).groups           # the main file is missing
    assert group.main is None and group.expected == path and list(group.parts) == planes
    with pytest.raises(FileNotFoundError):
        micro_reader.group_files([path])


def test_index_files_of_the_same_images_are_one_group(tmp_path):
    first, _ = test_companion._write(str(tmp_path), images=1)
    second = os.path.join(str(tmp_path), "copy.companion.ome")
    shutil.copyfile(first, second)
    g = micro_reader.group_files(str(tmp_path))
    (group,) = g.groups
    assert group.kind == "index" and group.format == "companion"
    assert {os.path.basename(group.main), *map(os.path.basename, group.parts)} == \
        {"set.companion.ome", "copy.companion.ome"}
    assert group.members and not g.singles


def test_a_leica_experiment_holds_its_references(tmp_path):
    pytest.importorskip("imagecodecs")
    from _lif_writer import image, write_xlef, write_xlif
    data = np.zeros((2, 3, 20, 30), np.uint16)
    write_xlif(str(tmp_path / "a.xlif"), image("a", data, "czyx"), frames="png")
    write_xlef(str(tmp_path / "exp.xlef"), ["a.xlif"])
    g = micro_reader.group_files(str(tmp_path))
    (group,) = g.groups                                          # a.xlif is a member, not a group
    assert os.path.basename(group.main) == "exp.xlef" and group.format == "lif"
    assert "a.xlif" in {os.path.basename(m) for m in group.members}
    assert not g.singles                                         # the frames are members too


def test_an_index_that_cannot_be_read_is_reported(tmp_path):
    (tmp_path / "bad.HTD").write_text('"HTSInfoFile", Version 1.0\n"XWells", nonsense\n')
    (group,) = micro_reader.group_files(str(tmp_path)).groups
    assert group.kind == "index" and not group.members
    assert any("cannot be read" in note for note in group.notes)


def test_group_is_a_value():
    group = FileGroup("composite", "a.vsi", "a.vsi", "vsi", ("x.ets",))
    assert group == FileGroup("composite", "a.vsi", "a.vsi", "vsi", ("x.ets",))
    with pytest.raises(AttributeError):
        group.main = "b.vsi"


def test_ndtiff_stack_files_open_the_dataset(tmp_path):
    frames = [({"time": t}, np.full((6, 7), t, np.uint16)) for t in range(4)]
    paths = write_ndtiff(str(tmp_path), frames, per_file=2)
    with micro_reader.open(paths[1]) as f:                       # the second stack file
        assert os.path.samefile(f.path, paths[0]) and f.opened == paths[1]


def test_a_cellvoyager_measurement_is_one_group(tmp_path):
    """Its .wpi is the main file; the other entry files, the Harmony-format
    export and the settings files are parts; the planes are members."""
    import test_cv7000
    data = test_cv7000._write(str(tmp_path))
    (tmp_path / "Index.idx.xml").write_text("<EvaluationInputData/>")     # the instrument's export
    g = micro_reader.group_files(str(tmp_path))
    (group,) = g.groups
    assert (group.kind, group.format, os.path.basename(group.main)) == ("index", "cv7000", "P1.wpi")
    assert _names(group.parts, tmp_path) == ["Index.idx.xml", "MeasurementData.mlf",
                                             "MeasurementDetail.mrf", "s.mes", "s.wpp"]
    assert len(group.members) == len(data) and not g.singles
    for target in ("s.mes", "Index.idx.xml", "MeasurementDetail.mrf", ""):    # "" = the folder
        with micro_reader.open(os.path.join(str(tmp_path), target) if target else str(tmp_path)) as f:
            assert f.backend == "cv7000" and os.path.basename(f.path) == "P1.wpi"
    os.remove(tmp_path / "P1.wpi")                               # no .wpi: the .mrf is the main file
    (group,) = micro_reader.group_files(str(tmp_path)).groups
    assert os.path.basename(group.main) == "MeasurementDetail.mrf"
    assert "MeasurementData.mlf" in _names(group.parts, tmp_path)
