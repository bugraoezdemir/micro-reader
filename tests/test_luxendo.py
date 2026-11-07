"""Luxendo experiment folders (backends/luxendo.py): any part opens the
experiment; pixels from raw/, link files say what should exist."""
from __future__ import annotations

import os
import shutil
import warnings

import numpy as np
import pytest

h5py = pytest.importorskip("h5py")

import micro_reader  # noqa: E402
from micro_reader import (CompositeWarning, CorruptFileError, MissingImagesWarning,  # noqa: E402
                          UnsupportedFormatError)
from micro_reader.backends import luxendo  # noqa: E402
from _hdf5_writer import write_ims_luxendo, write_luxendo  # noqa: E402

SHAPE = (3, 20, 24)
#: (folder channel number, description) of the default experiment, out of order on disk
CHANNELS = ((10, "Far-red"), (2, "RFP"))


def _folder(stack, channel, description, objective="bottom"):
    return f"stack_{stack}-Zstack_channel_{channel}-{description}_obj_{objective}"


def experiment(root, n_t=3, channels=CHANNELS, stacks=(0,), cameras=("long",), skip=(),
               levels=(), sidecar=True, seed=0):
    """An experiment folder: raw/<stack folder>/Cam_<camera>_<ttttt>.lux.h5 (+ .json).
    -> {(stack, camera): data (T, C, Z, Y, X)}, channels sorted by number, zeros
    where (t, channel number) is in *skip*."""
    rng = np.random.default_rng(seed)
    order = sorted(channels)
    out = {}
    for stack in stacks:
        for camera in cameras:
            data = rng.integers(1, 4000, (n_t, len(order), *SHAPE)).astype(np.uint16)
            for c, (number, description) in enumerate(order):
                folder = root / "raw" / _folder(stack, number, description)
                folder.mkdir(parents=True, exist_ok=True)
                for t in range(n_t):
                    if (t, number) in skip:
                        data[t, c] = 0
                        continue
                    write_luxendo(folder / f"Cam_{camera}_{t:05d}.lux.h5", data[t, c],
                                  levels=levels, channel=number, time_point=t, stack=stack,
                                  camera=camera, description=description, sidecar=sidecar,
                                  stamp=f"2025-03-17T18:{13 + 30 * t // 60:02d}:{(30 * t) % 60:02d}"
                                        ".000+01:00")
            out[(stack, camera)] = data
    return out


def _keys(shape, n=12, seed=4):
    rng = np.random.default_rng(seed)
    keys = [tuple(slice(None) for _ in shape)]
    for _ in range(n):
        key = []
        for size in shape:
            a = int(rng.integers(0, size))
            b = int(rng.integers(a + 1, size + 1))
            key.append(slice(a, b))
        keys.append(tuple(key))
    return keys


def test_raw_alone_is_one_image(tmp_path):
    root = tmp_path / "2025-03-17_181310"
    data = experiment(root)[(0, "long")]
    with micro_reader.open(root) as f:
        assert (f.backend, f.format, f.path) == ("luxendo", "luxendo", str(root))
        assert f.link_files == []
        image, = f.images
        assert (image.name, image.axes, image.shape) == (root.name, "tczyx", data.shape)
        assert image.index == {"series": 0}
        for key in _keys(data.shape):
            np.testing.assert_array_equal(image[key], data[key], err_msg=str(key))
        meta = image.metadata
        # channels by Luxendo channel number (2 before 10), named by their description
        assert [c.name for c in meta.channels] == ["RFP", "Far-red"]
        assert meta.pixel_sizes["x"] == (0.26, "micrometer")
        assert meta.pixel_sizes["t"] == (30.0, "second")          # first and last time stamps
        assert f.collection.name == root.name


@pytest.mark.parametrize("target", [
    ".", "raw", "raw/stack_0-Zstack_channel_2-RFP_obj_bottom",
    "raw/stack_0-Zstack_channel_2-RFP_obj_bottom/Cam_long_00001.lux.h5",
    "raw/stack_0-Zstack_channel_2-RFP_obj_bottom/Cam_long_00001.json",
    "imaris.ims", "ims/stacks/imaris_0.ims", "bdv.xml"])
def test_every_part_opens_the_experiment(tmp_path, target):
    root = tmp_path / "exp"
    data = experiment(root)[(0, "long")]
    _links(root)
    with micro_reader.open(root / target) as f:
        assert (f.backend, f.path, f.opened) == ("luxendo", str(root), str(root / target))
        np.testing.assert_array_equal(f.images[0][...], data)


def _links(root, n_t=3, channels=CHANNELS):
    """imaris.ims, ims/stacks/imaris_0.ims and a bdv.xml (+ bdv.h5) over the raw files."""
    order = sorted(channels)
    stacks = root / "ims" / "stacks"
    stacks.mkdir(parents=True, exist_ok=True)
    for path, prefix in ((root / "imaris.ims", ""), (stacks / "imaris_0.ims", "../../")):
        with h5py.File(path, "w") as f:
            for t in range(n_t):
                for c, (number, description) in enumerate(order):
                    g = f.require_group(f"DataSet/ResolutionLevel 0/TimePoint {t}/Channel {c}")
                    g["Data"] = h5py.ExternalLink(
                        f"{prefix}raw/{_folder(0, number, description)}/Cam_long_{t:05d}.lux.h5", "Data")
    with h5py.File(root / "bdv.h5", "w") as f:
        for t in range(n_t):
            for s, (number, description) in enumerate(order):
                f[f"t{t:05d}/s{s:02d}/0/cells"] = h5py.ExternalLink(
                    f"raw/{_folder(0, number, description)}/Cam_long_{t:05d}.lux.h5", "Data")
    (root / "bdv.xml").write_text(
        '<?xml version="1.0"?><SpimData version="0.2"><SequenceDescription>'
        '<ImageLoader format="bdv.hdf5"><hdf5 type="relative">bdv.h5</hdf5></ImageLoader>'
        '</SequenceDescription></SpimData>')


def test_multifile_false_reads_the_file_alone(tmp_path):
    root = tmp_path / "exp"
    data = experiment(root)[(0, "long")]
    _links(root)
    stack = root / "raw" / _folder(0, 2, "RFP") / "Cam_long_00001.lux.h5"
    with micro_reader.open(stack, multifile=False) as f:
        assert (f.backend, f.images[0].shape) == ("hdf5", (1, 1) + SHAPE)
        np.testing.assert_array_equal(f.images[0][0, 0], data[1, 0])
    with micro_reader.open(root / "imaris.ims", multifile=False) as f:
        assert f.backend == "ims"
    with micro_reader.open(root, multifile=False) as f:          # a folder has no "alone"
        assert f.backend == "luxendo"


def test_a_file_the_link_file_names_is_missing(tmp_path):
    """The link file says what should exist: a missing stack and a channel
    folder that is not there at all read as zeros, one warning."""
    root = tmp_path / "exp"
    data = experiment(root, skip={(2, 2)})[(0, "long")]
    _links(root, channels=CHANNELS + ((3, "Magenta"),))
    with pytest.warns(MissingImagesWarning, match=r"4 of 9 stack files are missing .*by a link file"):
        f = micro_reader.open(root)
    with f:
        image = f.images[0]
        assert image.shape == (3, 3) + SHAPE
        assert [c.name for c in image.metadata.channels] == ["RFP", "Magenta", "Far-red"]
        np.testing.assert_array_equal(image.read(c=0), data[:, 0])
        np.testing.assert_array_equal(image.read(c=2), data[:, 1])
        assert not image.read(c=1).any()
        assert f.link_files == [str(root / "imaris.ims")]


def test_a_gap_in_the_time_numbering(tmp_path):
    root = tmp_path / "exp"
    data = experiment(root, skip={(1, 10)})[(0, "long")]
    with pytest.warns(MissingImagesWarning,
                      match=r"1 of 6 .*channel_10-Far-red_obj_bottom/Cam_long_00001\.lux\.h5.*time numbering"):
        f = micro_reader.open(root)
    with f:
        np.testing.assert_array_equal(f.images[0][...], data)


def test_bdv_xml_without_its_h5_is_passed_over(tmp_path):
    """S-BIAD3686 deposited every bdv.xml without its bdv.h5."""
    root = tmp_path / "exp"
    data = experiment(root, n_t=2)[(0, "long")]
    _links(root, n_t=2)
    for name in ("imaris.ims", "bdv.h5"):
        os.remove(root / name)
    shutil.rmtree(root / "ims")
    with pytest.warns(CompositeWarning, match=r"bdv\.xml.*names 'bdv\.h5', which is missing"):
        f = micro_reader.open(root / "bdv.xml")
    with f:
        assert f.link_files == []
        np.testing.assert_array_equal(f.images[0][...], data)


def test_a_file_that_does_not_fit_its_place(tmp_path):
    """Each file's JSON is checked when it is first read."""
    root = tmp_path / "exp"
    experiment(root)
    wrong = root / "raw" / _folder(0, 10, "Far-red") / "Cam_long_00002.lux.h5"
    write_luxendo(wrong, np.zeros(SHAPE, np.uint16), levels=(), channel=10, time_point=7,
                  description="Far-red")
    with micro_reader.open(root) as f:
        image = f.images[0]
        image.read(t=1)                                           # the others read
        with pytest.raises(CorruptFileError, match=r"Cam_long_00002.*time_point 7.*2"):
            image.read(t=2, c=1)
    small = root / "raw" / _folder(0, 2, "RFP") / "Cam_long_00001.lux.h5"
    write_luxendo(small, np.zeros((3, 10, 24), np.uint16), levels=(), channel=2, time_point=1,
                  description="RFP")
    with micro_reader.open(root) as f:
        with pytest.raises(CorruptFileError, match=r"Cam_long_00001.*Data is \(3, 10, 24\)"):
            f.images[0].read(t=1, c=0)


def test_stacks_and_cameras_are_tiles_and_views(tmp_path):
    root = tmp_path / "exp"
    data = experiment(root, n_t=1, stacks=(0, 1), cameras=("left", "right"))
    with micro_reader.open(root) as f:
        assert [dict(i.index) for i in f.images] == [
            {"series": 0, "tile": 0, "view": 0}, {"series": 0, "tile": 0, "view": 1},
            {"series": 0, "tile": 1, "view": 0}, {"series": 0, "tile": 1, "view": 1}]
        assert f.images[3].name == "st:1 obj:bottom cam:right"     # labels tell them apart (MR-2)
        for image, key in zip(f.images, [(0, "left"), (0, "right"), (1, "left"), (1, "right")]):
            np.testing.assert_array_equal(image[...], data[key])
    with pytest.raises(UnsupportedFormatError, match="mosaic"):
        micro_reader.open(root, tiles="mosaic")


def test_resolution_levels(tmp_path):
    root = tmp_path / "exp"
    data = experiment(root, n_t=2, levels=(2,))[(0, "long")]
    with micro_reader.open(root) as f:
        image = f.images[0]
        level = image.resolutions[1]
        assert level.shape == (2, 2, 2, 10, 12)
        np.testing.assert_array_equal(level[...], data[:, :, ::2, ::2, ::2])
        assert level.metadata.pixel_sizes["x"] == (0.52, "micrometer")


def test_open_files_are_bounded(tmp_path, monkeypatch):
    monkeypatch.setattr(luxendo, "OPEN_FILES", 2)
    root = tmp_path / "exp"
    data = experiment(root, n_t=4)[(0, "long")]
    with micro_reader.open(root) as f:
        image = f.images[0]
        for t in range(4):
            for c in range(2):
                np.testing.assert_array_equal(image.read(t=t, c=c), data[t, c])
                assert len(f._files._open) <= 2
        np.testing.assert_array_equal(image[...], data)           # all 8 at once, then closed
        assert len(f._files._open) <= 2


def test_reading_after_close(tmp_path):
    root = tmp_path / "exp"
    experiment(root, n_t=1)
    f = micro_reader.open(root)
    image = f.images[0]
    f.close()
    with pytest.raises(ValueError, match="closed"):
        image.read(t=0)


def test_group_files_one_experiment(tmp_path):
    root = tmp_path / "exp"
    experiment(root, n_t=2)
    _links(root, n_t=2)
    (root / "processed").mkdir()
    write_luxendo(root / "processed" / "fused_00000.lux.h5", np.zeros(SHAPE, np.uint16), levels=())
    grouping = micro_reader.group_files(tmp_path)
    group, = grouping.groups
    assert (group.kind, group.main, group.format) == ("composite", str(root), "luxendo")
    assert len(group.parts) == 2 * 2 * 2 + 4                     # stacks + sidecars, 3 link files + .h5
    assert grouping.singles == [str(root / "processed" / "fused_00000.lux.h5")]
    with micro_reader.open(grouping.singles[0]) as f:            # processed/ opens alone
        assert f.backend == "hdf5"


def test_a_stack_outside_an_experiment_reads_alone(tmp_path):
    write_luxendo(tmp_path / "Cam_long_00000.lux.h5", np.zeros(SHAPE, np.uint16), levels=())
    with micro_reader.open(tmp_path / "Cam_long_00000.lux.h5") as f:
        assert f.backend == "hdf5"
    assert luxendo.experiment_folder(tmp_path) is None


def test_folder_names_without_descriptions(tmp_path):
    """The spec's own example names (stack_0-x00-y01_channel_0_obj_left)."""
    root = tmp_path / "exp"
    folder = root / "raw" / "stack_0-x00-y01_channel_0_obj_left"
    folder.mkdir(parents=True)
    write_luxendo(folder / "Cam_left_00000.lux.h5", np.ones(SHAPE, np.uint16), levels=(),
                  objective="left", camera="left")
    with micro_reader.open(folder / "Cam_left_00000.lux.h5") as f:
        assert (f.backend, f.images[0].shape) == ("luxendo", (1, 1) + SHAPE)


def test_folder_whose_chosen_file_fails_names_the_others(tmp_path):
    """A folder opens its index file first (a BigDataViewer XML); when that
    fails, the error names the folder's other images."""
    import tifffile
    (tmp_path / "dataset.xml").write_text(
        '<?xml version="1.0"?><SpimData version="0.2"><SequenceDescription>'
        '<ImageLoader format="bdv.hdf5"><hdf5 type="relative">dataset.h5</hdf5></ImageLoader>'
        '<ViewSetups><ViewSetup><id>0</id><size>4 4 2</size></ViewSetup></ViewSetups>'
        '<Timepoints type="range"><first>0</first><last>0</last></Timepoints>'
        '</SequenceDescription></SpimData>')
    tifffile.imwrite(tmp_path / "other.tif", np.zeros((4, 4), np.uint8))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises((CorruptFileError, UnsupportedFormatError),
                           match=r"the folder also holds other\.tif"):
            micro_reader.open(tmp_path)
