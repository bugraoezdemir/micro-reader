"""Closing HDF5 files (and the shared ``File.close``), the errors of the
``ims`` catch-all, and BigDataViewer partition files opening their dataset.

Reading any image a file gave out after ``File.close()`` raises ValueError
(docs/api/guide/opening.md): its resolution levels and parts too, also those
``resolutions="base"`` leaves out of ``images``."""
from __future__ import annotations

import warnings

import numpy as np
import pytest

h5py = pytest.importorskip("h5py")

import micro_reader  # noqa: E402
from micro_reader import CompositeWarning, CorruptFileError, UnsupportedFormatError  # noqa: E402
from micro_reader._base import File, Image  # noqa: E402
from _hdf5_writer import (axistags, write_bdv, write_fiji, write_ilastik,  # noqa: E402
                          write_ims_multifile, write_luxendo)


@pytest.fixture(autouse=True, params=["native", "h5py"])
def chunk_path(request, monkeypatch):
    """Both read paths: chunks decoded here, and through h5py."""
    from micro_reader.backends import _hdf5
    if request.param == "native":
        monkeypatch.setattr(_hdf5, "NATIVE_MIN_CHUNK", 0)
        monkeypatch.setattr(_hdf5, "NATIVE_MIN_CHUNK_SCALEOFFSET", 0)
    return request.param


def _corner(image):
    return image[tuple(slice(0, 1) for _ in image.shape)]


def _pyramidal(kind, folder):
    """A file of *kind* with lower resolution levels -> path."""
    if kind == "bdv-xml":
        write_bdv(folder, tiles=1, channels=1, timepoints=(0,))
        return folder / "dataset.xml"
    if kind == "bdv-hdf5":
        write_bdv(folder, tiles=1, channels=1, timepoints=(0,), xml=False)
        return folder / "dataset.h5"
    if kind == "luxendo":
        data = np.random.default_rng(0).integers(0, 4000, (8, 40, 48)).astype(np.uint16)
        write_luxendo(folder / "stack.lux.h5", data, levels=(2, 4))
        return folder / "stack.lux.h5"
    if kind == "ims":
        write_ims_multifile(folder, levels=3, copy=False)
        return folder / "set.ims"
    if kind == "tiff":
        tifffile = pytest.importorskip("tifffile")
        data = np.random.default_rng(1).integers(0, 4000, (2, 96, 80)).astype("u2")
        with tifffile.TiffWriter(folder / "pyr.ome.tif", ome=True) as tw:
            tw.write(data, subifds=2, metadata={"axes": "CYX"}, photometric="minisblack",
                     tile=(32, 32))
            for k in (1, 2):
                tw.write(data[:, ::2 ** k, ::2 ** k], subfiletype=1, photometric="minisblack",
                         tile=(32, 32))
        return folder / "pyr.ome.tif"
    raise AssertionError(kind)


@pytest.mark.parametrize("axes", [None, "canonical"])
@pytest.mark.parametrize("kind", ["bdv-xml", "bdv-hdf5", "luxendo", "ims", "tiff"])
def test_every_level_raises_after_close(tmp_path, kind, axes):
    """Levels behind resolutions="base" were never closed: BigDataViewer's read
    on after close (native) or raised RuntimeError (h5py); so did Luxendo's,
    multi-file Imaris' and TIFF's."""
    path = _pyramidal(kind, tmp_path)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", CompositeWarning)
        f = micro_reader.open(path, resolutions="base", axes=axes)
    with f:
        levels = f.images[0].resolutions
        assert len(levels) > 1
        for level in levels:
            _corner(level)                       # handles open, chunks cached
    for level in levels + f.images[0].resolutions:
        with pytest.raises(ValueError, match="closed"):
            _corner(level)


def test_listed_levels_raise_after_close(tmp_path):
    path = _pyramidal("bdv-xml", tmp_path)
    with micro_reader.open(path, resolutions="all") as f:
        images = list(f.images)
    assert len(images) == 2
    for image in images:
        with pytest.raises(ValueError, match="closed"):
            _corner(image)


class _Counted(Image):
    def __init__(self, position, shape=(2, 3)):
        super().__init__(position, "yx", shape, np.uint8)
        self.closes = 0

    def _read_block(self, block):
        if self.closes:
            raise ValueError("closed")
        return np.zeros(tuple(s.stop - s.start for s in block), np.uint8), 0

    def close(self):
        self.closes += 1


def test_file_close_closes_every_image_once():
    """File.close: images, their levels and parts, each once -- also an image
    listed in ``images`` and in a pyramid, and images shown with other axes."""
    base, level, other = _Counted(0, (4, 6)), _Counted(1, (2, 3)), _Counted(2)
    base._resolutions = level._resolutions = [base, level]
    ragged, part0, part1 = _Counted(3), _Counted(4), _Counted(5)
    ragged._parts = [part0, part1]
    part0._whole = part1._whole = ragged
    f = File("x", [base, level, other, ragged])
    f.images = [image.as_axes("canonical") for image in f.images]   # open(axes=...)
    f.close()
    for image in (base, level, other, ragged, part0, part1):
        assert image.closes == 1, image
    with pytest.raises(ValueError):
        _corner(f.images[0].resolutions[1])


# -- the ims catch-all ------------------------------------------------------------

def _other_layout(kind, folder):
    if kind == "bdv":
        write_bdv(folder, tiles=1, channels=1, timepoints=(0,), xml=False)
        return folder / "dataset.h5"
    if kind == "ilastik":
        write_ilastik(folder / "export.h5", np.zeros((4, 5), np.uint8), "yx")
        return folder / "export.h5"
    if kind == "luxendo":
        write_luxendo(folder / "stack.lux.h5", np.zeros((4, 8, 8), np.uint16), levels=())
        return folder / "stack.lux.h5"
    write_fiji(folder / "fiji.h5", {"volume": np.zeros((3, 4, 5), np.uint8)})
    return folder / "fiji.h5"


@pytest.mark.parametrize("kind, what, backend", [
    ("bdv", "a BigDataViewer HDF5 file", "bdv"),
    ("ilastik", "an ilastik HDF5 file", "ilastik"),
    ("luxendo", "a Luxendo HDF5 file", "hdf5"),
    ("fiji", "a Fiji HDF5 file", "hdf5")])
def test_ims_backend_names_the_layout_and_its_backend(tmp_path, kind, what, backend):
    path = _other_layout(kind, tmp_path)
    with pytest.raises(UnsupportedFormatError, match=f"is {what}, not an Imaris file: "
                                                     f"the '{backend}' backend reads it"):
        micro_reader.open(path, backend="ims")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", CompositeWarning)        # the BDV file alone
        f = micro_reader.open(path, backend=backend)              # and that backend reads it
    with f:
        assert f.images


def test_ims_without_numbered_levels_is_corrupt(tmp_path):
    path = tmp_path / "odd.ims"
    with h5py.File(path, "w") as f:
        f.create_dataset("DataSet/ResolutionLevel x/TimePoint 0/Channel 0/Data",
                         data=np.zeros((2, 4, 4), np.uint8))
    with pytest.raises(CorruptFileError, match="without 'ResolutionLevel 0'"):
        micro_reader.open(path)


# -- BigDataViewer partition files --------------------------------------------------

def _partitioned(folder, listed=True):
    """bdv-core's partitioned dataset: the master dataset.h5 links time point 1
    to the partition file dataset-01-00.h5, which holds that view and the
    setup's resolutions (as WriteSequenceToHdf5 writes it)."""
    stored = write_bdv(folder, tiles=1, channels=1, levels=((1, 1, 1),))
    part = folder / "dataset-01-00.h5"
    with h5py.File(folder / "dataset.h5", "a") as master, h5py.File(part, "w") as p:
        p["s00/resolutions"] = master["s00/resolutions"][()]
        p["s00/subdivisions"] = master["s00/subdivisions"][()]
        p.create_dataset("t00001/s00/0/cells", data=master["t00001/s00/0/cells"][()],
                         chunks=(4, 16, 16))
        del master["t00001/s00/0/cells"]
        master["t00001/s00/0/cells"] = h5py.ExternalLink(part.name, "/t00001/s00/0/cells")
    if listed:
        xml = folder / "dataset.xml"
        xml.write_text(xml.read_text(encoding="utf-8").replace(
            "</ImageLoader>", '<partition><path type="relative">dataset-01-00.h5</path>'
            "<timepoints>1</timepoints><setups>0</setups></partition></ImageLoader>"),
            encoding="utf-8")
    return stored, part


def test_bdv_partition_file_opens_the_dataset(tmp_path):
    stored, part = _partitioned(tmp_path)
    with warnings.catch_warnings():
        warnings.simplefilter("error", CompositeWarning)        # not read alone
        f = micro_reader.open(part)
    with f:
        assert f.path.endswith("dataset.xml") and f.opened.endswith("dataset-01-00.h5")
        assert f.format == "bdv-xml"
        image = f.images[0]
        assert image.shape[0] == 2                                  # both time points
        for t in (0, 1):
            np.testing.assert_array_equal(image.read(t=t, c=0),
                                          stored[(t, 0, 0, 0)].view(np.uint16))
    g = micro_reader.group_files(tmp_path)
    assert len(g.groups) == 1 and g.singles == []


def test_bdv_partition_file_the_xml_does_not_list_reads_alone(tmp_path):
    _stored, part = _partitioned(tmp_path, listed=False)
    with pytest.warns(CompositeWarning, match="without the XML"):
        f = micro_reader.open(part)
    with f:
        assert f.format == "bdv-hdf5" and f.path.endswith("dataset-01-00.h5")


def test_bdv_partition_names_read_nothing(tmp_path):
    """``may_own`` (group_files' cheap test) goes by names only."""
    from micro_reader._groups import _Folders
    from micro_reader.backends import bdv
    (tmp_path / "dataset.xml").write_text("not read", encoding="utf-8")
    folders, folder = _Folders(), str(tmp_path)
    assert bdv.may_own(folder, "dataset-01-00.h5", folders)
    assert bdv.may_own(folder, "dataset.h5", folders)
    assert not bdv.may_own(folder, "other-01-00.h5", folders)
    assert not bdv.may_own(folder, "dataset-1.h5", folders)
