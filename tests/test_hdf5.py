"""HDF5 layouts other than Imaris: ilastik and BigDataViewer, against the
arrays written (tests/_hdf5_writer.py)."""
from __future__ import annotations

import warnings

import numpy as np
import pytest

h5py = pytest.importorskip("h5py")

import micro_reader  # noqa: E402
from micro_reader import (CompositeWarning, CorruptFileError, MissingImagesWarning,  # noqa: E402
                          UnsupportedFormatError)
from _hdf5_writer import axistags, write_bdv, write_fiji, write_ilastik, write_luxendo  # noqa: E402


@pytest.fixture(autouse=True, params=["native", "h5py"])
def chunk_path(request, monkeypatch):
    """Every test both ways: chunks decoded here, and small chunks through h5py
    (the choice real files get by chunk size, _hdf5.Layout.prefer_native)."""
    from micro_reader.backends import _hdf5
    if request.param == "native":
        monkeypatch.setattr(_hdf5, "NATIVE_MIN_CHUNK", 0)
        monkeypatch.setattr(_hdf5, "NATIVE_MIN_CHUNK_SCALEOFFSET", 0)
        monkeypatch.setattr(_hdf5, "NATIVE_MIN_CHUNK_LZF", 0)
    return request.param


@pytest.fixture(params=[1, 4], ids=["serial", "pool"])
def decode_threads(request):
    old = micro_reader.decode_threads()
    micro_reader.set_decode_threads(request.param)
    yield request.param
    micro_reader.set_decode_threads(old)


def _keys(shape, n=20, seed=1):
    rng = np.random.default_rng(seed)
    out = [tuple(slice(None) for _ in shape)]
    for _ in range(n):
        key = []
        for size in shape:
            a = int(rng.integers(0, size))
            b = int(rng.integers(a + 1, size + 1))
            key.append(slice(a, b))
        out.append(tuple(key))
    return out


# -- ilastik --------------------------------------------------------------------

@pytest.mark.parametrize("layout", ["contiguous", "chunked", "gzip", "lzf", "lzf_shuffle"])
def test_ilastik_matches_h5py(tmp_path, layout, decode_threads, chunk_path):
    data = np.random.default_rng(0).integers(0, 60000, (3, 7, 33, 41, 2)).astype(np.uint16)
    data[1] = 7                                     # runs: LZF back references
    path = tmp_path / "export.h5"
    opts = {"contiguous": {}, "chunked": {"chunks": (1, 4, 16, 16, 2)},
            "gzip": {"chunks": (1, 4, 16, 16, 2), "compression": "gzip"},
            "lzf": {"chunks": (1, 4, 16, 16, 2), "compression": "lzf"},     # as ilastik exports
            "lzf_shuffle": {"chunks": (1, 4, 16, 16, 2), "compression": "lzf",
                            "shuffle": True}}[layout]
    write_ilastik(path, data, "tzyxc", **opts)
    with micro_reader.open(path) as f:
        assert (f.backend, f.format) == ("ilastik", "ilastik")
        image = f.images[0]
        assert image.axes == "tzyxc" and image.shape == data.shape
        assert image.read_unit == opts.get("chunks")
        for key in _keys(data.shape):
            np.testing.assert_array_equal(image[key], data[key])
        assert image.read(c=1, z=3).shape == (3, 33, 41)
        if layout == "contiguous" and chunk_path == "native":
            image.stats.reset()                     # whole rows: read exactly
            np.testing.assert_array_equal(image[1, 2:5], data[1, 2:5])
            assert image.stats.over_read == 1.0
            image.stats.reset()                     # part of each row: rows read as one span
            np.testing.assert_array_equal(image[1, 2, 5:9, 3:7], data[1, 2, 5:9, 3:7])
            assert 1.0 < image.stats.over_read < 41 / 4 * 1.01


def test_contiguous_far_rows_read_one_by_one(tmp_path, monkeypatch):
    from micro_reader.backends import _hdf5
    monkeypatch.setattr(_hdf5.ChunkReader, "CONTIGUOUS_GAP", 0)
    data = np.random.default_rng(2).random((5, 60, 70)).astype(">f4")    # big-endian too
    path = tmp_path / "be.h5"
    write_ilastik(path, data, "zyx")
    with micro_reader.open(path) as f:
        image = f.images[0]
        assert image.dtype == np.dtype("float32")
        for key in _keys(data.shape):
            np.testing.assert_array_equal(image[key], data[key])
        image.stats.reset()
        image[2, 10:20, 5:9]
        assert image.stats.over_read == 1.0


def test_ilastik_several_datasets_are_scenes(tmp_path):
    raw = np.arange(2 * 20 * 30, dtype=np.uint8).reshape(2, 20, 30)
    probs = np.random.default_rng(1).random((20, 30, 3)).astype(np.float32)
    path = tmp_path / "two.h5"
    write_ilastik(path, raw, "cyx", name="data", extra={"exported_data": (probs, "yxc")})
    with h5py.File(path, "a") as f:
        f["plain"] = np.zeros(5)                    # no axistags: not an image
    with micro_reader.open(path) as f:
        assert [im.axes for im in f.images] == ["cyx", "yxc"]       # name order: data, exported_data
        assert [im.index for im in f.images] == [{"series": 0}, {"series": 1}]
        np.testing.assert_array_equal(f.images[1].read(c=2), probs[..., 2])
        assert f.images[1].as_axes("canonical").shape == (1, 3, 1, 20, 30)
        assert len(f.images[0].metadata.channels) == 2
        assert f.images[0].metadata.pixel_sizes == {}               # resolution has no unit


def _ilastik_with_units(path, keys, resolutions, units):
    """As ilastik >= 2025-09 writes an export (OpH5N5WriterBigDataset): vigra
    axistags with resolutions, and ``axis_units`` as json.dumps of a dict."""
    import json
    data = np.zeros((2,) * len(keys), np.uint8)
    tags = {"axes": [{"key": k, "typeFlags": {"t": 8, "c": 1}.get(k, 2), "resolution": r,
                      "description": ""} for k, r in zip(keys, resolutions)]}
    with h5py.File(path, "w") as f:
        ds = f.create_group("volume").create_dataset("data", data=data)
        ds.attrs["axistags"] = json.dumps(tags)
        if units is not None:
            ds.attrs["axis_units"] = json.dumps(units if isinstance(units, dict)
                                                else dict(zip(keys, units)))


@pytest.mark.parametrize("keys, resolutions, units, expected", [
    # ilastik's own test cases (test_pixel_sizes.py), µ in both spellings, json-escaped
    ("tzcyx", [120, 0.54321, 0.0, 6.000024000096, 13], ["s", "µm", "", "mm", "µm"],
     {"t": (120, "second"), "z": (0.54321, "micrometer"), "y": (6.000024000096, "millimeter"),
      "x": (13, "micrometer")}),
    ("zyx", [0.5, 0.2, 0.2], ["μm", "nm", "cm"],
     {"z": (0.5, "micrometer"), "y": (0.2, "nanometer"), "x": (2000.0, "micrometer")}),
    ("tyx", [3, 1, 1], ["min", "micron", "microns"],
     {"t": (3, "minute"), "y": (1, "micrometer"), "x": (1, "micrometer")}),
    # no unit, "pixel", an unknown unit, a zero resolution: no pixel size
    ("zyx", [2.0, 0.3, 0.3], ["pixel", "", "mistake"], {}),
    ("yx", [0.0, 0.0], ["µm", "µm"], {}),
    ("cyx", [0.0, 0.25, 0.25], None, {}),                        # before 2025-09: no units
    ("yx", [0.25, 0.25], "not json", {}),
])
def test_ilastik_pixel_sizes_from_axis_units(tmp_path, keys, resolutions, units, expected):
    path = tmp_path / "export.h5"
    _ilastik_with_units(path, keys, resolutions, units)
    if units == "not json":
        with h5py.File(path, "a") as f:
            f["volume/data"].attrs["axis_units"] = "{x: um"
    with micro_reader.open(path) as f:
        sizes = f.images[0].metadata.pixel_sizes
        got = {a: (pytest.approx(q.value), q.unit) for a, q in sizes.items()}
        assert got == expected
        if "c" in keys:
            assert len(f.images[0].metadata.channels) == 2


def test_ilastik_bad_axistags(tmp_path):
    path = tmp_path / "bad.h5"
    with h5py.File(path, "w") as f:
        f.create_dataset("data", data=np.zeros((4, 5))).attrs["axistags"] = axistags("zyx")
    with pytest.raises(CorruptFileError, match="3 axes"):
        micro_reader.open(path)


def test_ilastik_project_refused(tmp_path):
    path = tmp_path / "project.ilp"
    with h5py.File(path, "w") as f:
        f["workflowName"] = b"Pixel Classification"
        f.create_dataset("Input Data/local_data/x", data=np.zeros((4, 5))).attrs["axistags"] = \
            axistags("yx")
    with pytest.raises(UnsupportedFormatError, match="ilastik project"):
        micro_reader.open(path)


def test_plain_hdf5_refused_with_its_datasets(tmp_path):
    path = tmp_path / "plain.h5"
    with h5py.File(path, "w") as f:
        f["volume"] = np.zeros((4, 5, 6), np.uint16)
    with pytest.raises(UnsupportedFormatError, match=r"no layout.*volume \(4, 5, 6\) uint16"):
        micro_reader.open(path)


# -- BigDataViewer --------------------------------------------------------------

def test_bdv_tiles_channels_timepoints_levels(tmp_path, decode_threads):
    stored = write_bdv(tmp_path)
    with micro_reader.open(tmp_path / "dataset.xml") as f:
        assert (f.backend, f.format) == ("bdv", "bdv-xml")
        assert f.index_range == {"series": 1, "tile": 2, "resolution": 1}
        assert len(f.images) == 2
        image = f.select(tile=1)[0]
        assert image.axes == "tczyx" and image.shape == (2, 2, 6, 40, 50)
        assert image.dtype == np.uint16                     # stored as int16
        want = np.stack([np.stack([stored[(t, 1, c, 0)].view(np.uint16) for c in range(2)])
                         for t in range(2)])
        for key in _keys(want.shape):
            np.testing.assert_array_equal(image[key], want[key])
        assert image.read_unit == (1, 1, 4, 16, 16)
        level = image.resolutions[1]
        assert level.shape == (2, 2, 6, 20, 25) and level.index["resolution"] == 1
        np.testing.assert_array_equal(level.read(t=1, c=0), stored[(1, 1, 0, 1)].view(np.uint16))
        meta = image.metadata
        assert meta.pixel_sizes == {"x": (0.5, "micrometer"), "y": (0.5, "micrometer"),
                                    "z": (2.0, "micrometer")}
        assert meta.position["x"] == (pytest.approx(45 * 0.5), "micrometer")   # tile 1: 45 px
        assert level.metadata.pixel_sizes["x"] == (1.0, "micrometer")


def test_bdv_scaleoffset_as_bigdataviewer_writes_it(tmp_path, decode_threads):
    """HDF5 scaleoffset then deflate (bdv-core's writer): values packed into a
    few bits each, decoded here or by h5py, both equal to what was written."""
    stored = write_bdv(tmp_path, tiles=1, scaleoffset=True, value_range=(100, 1100))
    with h5py.File(tmp_path / "dataset.h5", "r") as f:
        pipeline = f["t00000/s00/0/cells"].id.get_create_plist()
        assert [pipeline.get_filter(i)[0] for i in range(pipeline.get_nfilters())] == [6, 1]
    with micro_reader.open(tmp_path / "dataset.xml") as f:
        image = f.images[0]
        for t in range(2):
            for c in range(2):
                np.testing.assert_array_equal(image.read(t=t, c=c),
                                              stored[(t, 0, c, 0)].view(np.uint16))
        np.testing.assert_array_equal(image[1, 0, 2:5, 7:30, 3:41],
                                      stored[(1, 0, 0, 0)].view(np.uint16)[2:5, 7:30, 3:41])


def test_bdv_scaleoffset_full_precision_chunks(tmp_path, chunk_path):
    """Values over the whole 16-bit range: HDF5 stores such chunks as they are
    (full precision, as pybdv's example.h5 has them).  HDF5 1.14's reader
    refuses them (h5py 3.15 / HDF5 1.14.6 raises OSError); HDF5 2.0 reads
    them (h5py 3.16 / HDF5 2.0.0).  micro-reader reads them as written on
    both: natively, or through h5py's failure on 1.14."""
    stored = write_bdv(tmp_path, tiles=1, channels=1, timepoints=(0,), levels=((1, 1, 1),),
                       scaleoffset=True)
    want = stored[(0, 0, 0, 0)]
    with h5py.File(tmp_path / "dataset.h5", "r") as f:
        try:
            got = f["t00000/s00/0/cells"][()]
        except OSError:                             # HDF5 < 2.0 refuses them
            got = None
    if got is not None:                             # HDF5 2.0+: read as written
        np.testing.assert_array_equal(got, want)
    with micro_reader.open(tmp_path / "dataset.xml") as f:
        np.testing.assert_array_equal(f.images[0].read(t=0, c=0), want.view(np.uint16))


def test_bdv_channel_names_and_all_levels(tmp_path):
    write_bdv(tmp_path, tiles=1, channel_names={0: "GFP", 1: "RFP"})
    with micro_reader.open(tmp_path / "dataset.xml", resolutions="all") as f:
        assert f.index_range == {"series": 1, "resolution": 2}
        assert [ch.name for ch in f.images[0].metadata.channels] == ["GFP", "RFP"]
        assert f.select(resolution=1)[0].shape[-2:] == (20, 25)


def test_bdv_missing_views_read_as_zeros(tmp_path):
    stored = write_bdv(tmp_path, tiles=1, missing=[(1, 1)])
    with micro_reader.open(tmp_path / "dataset.xml") as f:
        image = f.images[0]
        assert not image.read(t=1, c=1).any()
        np.testing.assert_array_equal(image.read(t=0, c=1), stored[(0, 0, 1, 0)].view(np.uint16))
    # a view neither stored nor listed as missing: zeros, with a warning
    write_bdv(tmp_path, name="gap", tiles=1, missing=[(1, 0)], xml=False)
    write_bdv(tmp_path, name="gap2", tiles=1, xml=False)
    (tmp_path / "gap.xml").write_text(
        (tmp_path / "dataset.xml").read_text(encoding="utf-8")
        .replace("dataset.h5", "gap.h5").replace("<MissingViews>", "<Unused>")
        .replace("</MissingViews>", "</Unused>"), encoding="utf-8")
    with pytest.warns(MissingImagesWarning, match="neither stored"):
        f = micro_reader.open(tmp_path / "gap.xml")
    f.close()


def test_bdv_hdf5_opens_its_xml(tmp_path):
    write_bdv(tmp_path)
    with micro_reader.open(tmp_path / "dataset.h5") as f:
        assert f.path.endswith("dataset.xml") and f.opened.endswith("dataset.h5")
        assert f.images[0].metadata.pixel_sizes["z"] == (2.0, "micrometer")


def test_bdv_hdf5_alone(tmp_path):
    stored = write_bdv(tmp_path, xml=False)
    with pytest.warns(CompositeWarning, match="without the XML"):
        f = micro_reader.open(tmp_path / "dataset.h5")
    with f:
        assert f.format == "bdv-hdf5"
        assert f.index_range == {"series": 1, "sequence": 4, "resolution": 1}
        image = f.select(sequence=3)[0]                     # setup 3 = tile 1, channel 1
        np.testing.assert_array_equal(image.read(t=0, c=0), stored[(0, 1, 1, 0)].view(np.uint16))
        assert image.metadata.pixel_sizes == {}


def test_bdv_data_type_attribute(tmp_path):
    """bdv-core 10.4.6+: a setup's dataType names the type; signed 16-bit data
    declared int16 stays int16 (only undeclared int16 is the old uint16 rule)."""
    stored = write_bdv(tmp_path, tiles=1, channels=1, timepoints=(0,))
    with h5py.File(tmp_path / "dataset.h5", "a") as f:
        f["s00"].attrs["dataType"] = np.bytes_(b"int16")
    with micro_reader.open(tmp_path / "dataset.xml") as f:
        image = f.images[0]
        assert image.dtype == np.int16
        np.testing.assert_array_equal(image.read(t=0, c=0), stored[(0, 0, 0, 0)])


def test_bdv_setups_without_attributes_are_a_sequence(tmp_path):
    """An old XML whose setups carry no attributes (MPI-CBG drosophila.xml):
    nothing says what tells them apart, so each is an image, index sequence."""
    stored = write_bdv(tmp_path)
    xml = tmp_path / "dataset.xml"
    text = xml.read_text(encoding="utf-8")
    import re
    xml.write_text(re.sub(r"<attributes>.*?</attributes>", "", text, flags=re.S),
                   encoding="utf-8")
    with micro_reader.open(xml) as f:
        assert f.index_range == {"series": 1, "sequence": 4, "resolution": 1}
        image = f.select(sequence=3)[0]                     # setup 3: tile 1, channel 1
        assert image.shape[:2] == (2, 1)
        np.testing.assert_array_equal(image.read(t=1, c=0), stored[(1, 1, 1, 0)].view(np.uint16))
        assert image.metadata.channels[0].name is None      # "setup 3" is no channel name


def test_bdv_settings_file_belongs_to_the_dataset(tmp_path):
    write_bdv(tmp_path)
    (tmp_path / "dataset.settings.xml").write_text("<Settings />", encoding="utf-8")
    with micro_reader.open(tmp_path / "dataset.settings.xml") as f:
        assert f.path.endswith("dataset.xml")
    g = micro_reader.group_files(tmp_path)
    assert len(g.groups) == 1 and g.singles == []
    assert [p.rsplit("\\", 1)[-1].rsplit("/", 1)[-1] for p in g.groups[0].parts] == \
        ["dataset.settings.xml"]


def test_bdv_partition_file_missing(tmp_path):
    """Partitioned BigDataViewer: the master file links to one file per part;
    a missing part file reads as zeros, with a warning naming it."""
    stored = write_bdv(tmp_path, tiles=1, channels=1, levels=((1, 1, 1),))
    with h5py.File(tmp_path / "dataset.h5", "a") as f:
        del f["t00001/s00/0/cells"]
        f["t00001/s00/0/cells"] = h5py.ExternalLink("dataset-01-00.h5", "/t00001/s00/0/cells")
    xml = tmp_path / "dataset.xml"
    xml.write_text(xml.read_text(encoding="utf-8").replace(
        "</ImageLoader>", '<partition><path type="relative">dataset-01-00.h5</path>'
        "</partition></ImageLoader>"), encoding="utf-8")
    with pytest.warns(MissingImagesWarning, match="partition files the XML lists are missing"):
        f = micro_reader.open(xml)
    with f:
        image = f.images[0]
        np.testing.assert_array_equal(image.read(t=0, c=0), stored[(0, 0, 0, 0)].view(np.uint16))
        assert not image.read(t=1, c=0).any()


def test_bdv_float_data_kept(tmp_path):
    write_bdv(tmp_path, tiles=1, channels=1, dtype=np.float32)
    with micro_reader.open(tmp_path / "dataset.xml") as f:
        assert f.images[0].dtype == np.float32


def test_bdv_timepoint_pattern(tmp_path):
    xml = '    <Timepoints type="pattern"><integerpattern>3,5</integerpattern></Timepoints>'
    write_bdv(tmp_path, tiles=1, timepoints=(3, 5), timepoint_xml=xml)
    with micro_reader.open(tmp_path / "dataset.xml") as f:
        assert f.images[0].shape[0] == 2


def test_bdv_other_loader_refused(tmp_path):
    write_bdv(tmp_path, loader="bdv.n5")
    with pytest.raises(UnsupportedFormatError, match="bdv.n5"):
        micro_reader.open(tmp_path / "dataset.xml")


def test_bdv_mosaic_refused(tmp_path):
    write_bdv(tmp_path)
    with pytest.raises(UnsupportedFormatError, match="mosaic"):
        micro_reader.open(tmp_path / "dataset.xml", tiles="mosaic")


def test_bdv_group_files(tmp_path):
    write_bdv(tmp_path)
    (tmp_path / "notes.txt").write_text("x")
    g = micro_reader.group_files(tmp_path)
    assert len(g.groups) == 1
    group = g.groups[0]
    assert (group.kind, group.format) == ("index", "bdv")
    assert group.main.endswith("dataset.xml")
    assert {p.rsplit("\\", 1)[-1].rsplit("/", 1)[-1] for p in group.members} == {"dataset.h5"}
    assert [s.rsplit("\\", 1)[-1].rsplit("/", 1)[-1] for s in g.singles] == ["notes.txt"]


def test_bdv_folder_opens_the_xml(tmp_path):
    write_bdv(tmp_path)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with micro_reader.open(tmp_path) as f:
            assert f.path.endswith("dataset.xml")


# -- Luxendo and Fiji HDF5 ------------------------------------------------------

def test_luxendo_stack(tmp_path, decode_threads):
    data = np.random.default_rng(5).integers(0, 4000, (20, 70, 90)).astype(np.uint16)
    stored = write_luxendo(tmp_path / "Cam_long_00000.lux.h5", data)
    with micro_reader.open(tmp_path / "Cam_long_00000.lux.h5") as f:
        assert (f.backend, f.format) == ("hdf5", "luxendo")
        image = f.images[0]
        assert image.axes == "tczyx" and image.shape == (1, 1, 20, 70, 90)
        for key in _keys(data.shape):
            np.testing.assert_array_equal(image[(0, 0) + key], data[key])
        level = image.resolutions[1]
        np.testing.assert_array_equal(level.read(t=0, c=0), stored["Data_2_2_2"])
        meta = image.metadata
        assert meta.pixel_sizes == {"z": (2.0, "micrometer"), "y": (0.26, "micrometer"),
                                    "x": (0.26, "micrometer")}           # float32 attribute, 0.26
        channel = meta.channels[0]
        assert (channel.name, channel.excitation_wavelength, channel.emission_wavelength) == \
            ("Cyan", (445.0, "nanometer"), (479.0, "nanometer"))
        assert meta.position == {}
        assert level.metadata.pixel_sizes["x"] == (0.52, "micrometer")


def test_luxendo_wavelengths_only_when_unambiguous(tmp_path):
    write_luxendo(tmp_path / "a.lux.h5", np.zeros((4, 8, 8), np.uint16), levels=(),
                  lasers_on=(445, 488), band="BP 525/50")
    with micro_reader.open(tmp_path / "a.lux.h5") as f:
        channel = f.images[0].metadata.channels[0]
        assert channel.excitation_wavelength is None                    # two lasers on
        assert channel.emission_wavelength == (525.0, "nanometer")      # centre / width


def test_fiji_hdf5_time_channel_grid(tmp_path):
    rng = np.random.default_rng(6)
    arrays = {f"t{t}/channel{c}": rng.integers(0, 255, (5, 30, 40)).astype(np.uint8)
              for t in range(3) for c in range(2) if (t, c) != (2, 1)}
    write_fiji(tmp_path / "stack.h5", arrays)
    with pytest.warns(MissingImagesWarning, match="1 of 6"):
        f = micro_reader.open(tmp_path / "stack.h5")
    with f:
        assert (f.backend, f.format) == ("hdf5", "fiji-hdf5")
        image = f.images[0]
        assert image.axes == "tczyx" and image.shape == (3, 2, 5, 30, 40)
        np.testing.assert_array_equal(image.read(t=1, c=1), arrays["t1/channel1"])
        np.testing.assert_array_equal(image[2, 0, 1:3, 5:25, 7:33], arrays["t2/channel0"][1:3, 5:25, 7:33])
        assert not image.read(t=2, c=1).any()
        assert image.metadata.pixel_sizes == {"z": (0.5, "micrometer"), "y": (0.2, "micrometer"),
                                              "x": (0.2, "micrometer")}


def test_fiji_hdf5_planes_and_scenes(tmp_path):
    planes = {f"t{t}/channel0": np.full((12, 16), t, np.uint16) for t in range(2)}
    write_fiji(tmp_path / "planes.h5", planes, size_zyx=(1.0, 1.0, 1.0))
    with micro_reader.open(tmp_path / "planes.h5") as f:
        image = f.images[0]
        assert image.shape == (2, 1, 1, 12, 16)
        assert image.read(t=1, c=0, z=0).min() == 1
        assert image.metadata.pixel_sizes == {}                          # 1, 1, 1: uncalibrated
    write_fiji(tmp_path / "named.h5", {"raw": np.zeros((3, 8, 9), np.uint8),
                                       "mask": np.ones((8, 9), np.uint8)})
    with micro_reader.open(tmp_path / "named.h5") as f:
        assert [(im.axes, im.index["series"]) for im in f.images] == [("yx", 0), ("zyx", 1)]
    write_fiji(tmp_path / "four.h5", {"vol": np.zeros((2, 3, 8, 9), np.uint8)})
    with pytest.raises(UnsupportedFormatError, match="no axis order"):
        micro_reader.open(tmp_path / "four.h5")


def test_bdv_position_is_the_nominal_placement(tmp_path):
    """BigStitcher lists stitching / ICP results before the grid translation and
    the calibration: the position is the grid (nominal), not the registered
    shift, and an affine ICP refinement does not take the position away."""
    def chain(tile):
        icp = (1, 0.01, 0, 2.5, 0, 1, 0, -1.0, 0, 0, 1, 0.5) if tile else (1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0)
        return [("Tile ICP Refinement", icp),
                ("Stitching Transform", (1, 0, 0, -3.0, 0, 1, 0, 9.0, 0, 0, 1, 0)),
                ("Translation to Regular Grid", (1, 0, 0, 45.0 * tile, 0, 1, 0, 0, 0, 0, 1, 0)),
                ("calibration", (1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 4, 0))]
    write_bdv(tmp_path, transforms=chain)
    with micro_reader.open(tmp_path / "dataset.xml") as f:
        for tile in (0, 1):
            meta = f.select(tile=tile)[0].metadata
            assert meta.position["x"] == (pytest.approx(45 * tile * 0.5), "micrometer")
            assert meta.position["y"] == (pytest.approx(0.0), "micrometer")


def test_bdv_unknown_transform_name_is_a_registration(tmp_path):
    def chain(tile):
        return [("Translation to Regular Grid", (1, 0, 0, 45.0 * tile, 0, 1, 0, 0, 0, 0, 1, 0)),
                ("My manual alignment", (1, 0, 0, 7.0, 0, 1, 0, 0, 0, 0, 1, 0)),
                ("calibration", (1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 4, 0))]
    write_bdv(tmp_path, transforms=chain)
    with micro_reader.open(tmp_path / "dataset.xml") as f:    # the run stops at the unknown name:
        assert f.select(tile=1)[0].metadata.position == {}     # the bare calibration places nothing


def test_bdv_calibration_alone_places_nothing(tmp_path):
    """Multi-angle views registered by beads: the nominal run is the bare
    calibration (no translation), so no position -- not 0, 0, 0."""
    def chain(tile):
        return [("Fast 3d geometric hashing (rotation invariant), AffineModel3D on [beads (c=0)]",
                 (0.9, 0.1, 0, 4.0, -0.1, 0.9, 0, 80.0, 0, 0, 1, 370.0)),
                ("calibration", (1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 4, 0))]
    write_bdv(tmp_path, transforms=chain)
    with micro_reader.open(tmp_path / "dataset.xml") as f:
        assert all(image.metadata.position == {} for image in f.images)


def test_bdv_hdf5_alone_names_the_xmls_that_name_it(tmp_path):
    """A second XML naming the file (another registration of the same data) is
    named in the warning, never read from: the file still reads alone."""
    write_bdv(tmp_path)
    (tmp_path / "dataset.xml").rename(tmp_path / "dataset_v1.xml")
    (tmp_path / "unrelated.xml").write_text("<SpimData/>", encoding="utf-8")
    with pytest.warns(CompositeWarning, match=r"naming it, 'dataset_v1.xml', for them"):
        f = micro_reader.open(tmp_path / "dataset.h5")
    with f:
        assert f.format == "bdv-hdf5" and "sequence" in f.index_range
        assert f.images[0].metadata.pixel_sizes == {}


@pytest.mark.parametrize("dtype", ["<u2", ">u2", "u1", ">f4"])
@pytest.mark.parametrize("chunks", [None, (8, 16, 16), (5, 7, 33)], ids=["contiguous", "chunked", "odd"])
def test_uncompressed_regions_equal_h5py(tmp_path, dtype, chunks, decode_threads):
    """Random regions of uncompressed data (contiguous: rows, planes batched
    into one job; chunked: spans or rows per chunk part, chunks looked up one by
    one or mapped) against h5py, in either byte order."""
    rng = np.random.default_rng(3)
    data = (rng.random((23, 61, 75)) * 4000).astype(dtype)
    path = tmp_path / "stack.h5"
    with h5py.File(path, "w") as f:
        f.create_dataset("t0/channel0", data=data, chunks=chunks).attrs["element_size_um"] = \
            np.array((0.5, 0.2, 0.2), np.float32)
    with micro_reader.open(path) as fh:
        image = fh.images[0]
        for key in _keys(data.shape, n=60, seed=11) + [
                (slice(2, 20), slice(5, 6), slice(0, 75)), (slice(0, 23), slice(0, 61), slice(3, 4)),
                (slice(7, 8), slice(0, 61), slice(0, 75)), (slice(1, 22), slice(10, 50), slice(0, 75))]:
            got = image[(0, 0) + key]
            np.testing.assert_array_equal(got, data[key], err_msg=str(key))


@pytest.mark.parametrize("compression", [None, "gzip"])
def test_few_chunks_are_looked_up_one_by_one(tmp_path, compression, chunk_path):
    """A small read of a dataset of many chunks looks its chunks up one by one
    (Layout.locate), behind a user block, chunks never written taking the fill
    value; a large read later maps them all.  Both agree with h5py."""
    from micro_reader.backends import _hdf5
    rng = np.random.default_rng(4)
    data = np.full((40, 128, 128), 9, np.uint16)
    path = tmp_path / "sparse.h5"
    with h5py.File(path, "w", userblock_size=512) as f:
        ds = f.create_dataset("t0/channel0", (40, 128, 128), np.uint16, chunks=(4, 16, 16),
                              fillvalue=9, compression=compression)
        ds.attrs["element_size_um"] = np.array((0.5, 0.2, 0.2), np.float32)
        for z, y, x in [(0, 0, 0), (4, 16, 32), (8, 48, 48), (36, 112, 112)]:   # 4 of 640 chunks
            block = rng.integers(0, 4000, (4, 16, 16)).astype(np.uint16)
            ds[z:z + 4, y:y + 16, x:x + 16] = block
            data[z:z + 4, y:y + 16, x:x + 16] = block
    with micro_reader.open(path) as f:
        image = f.images[0]
        for key in [(slice(0, 3), slice(5, 20), slice(0, 16)), (slice(3, 10), slice(10, 60), slice(30, 60)),
                    (slice(20, 30), slice(70, 90), slice(0, 10)), (slice(35, 40), slice(100, 128), slice(100, 128))]:
            np.testing.assert_array_equal(image[(0, 0) + key], data[key], err_msg=str(key))
        layout = next(iter(image._reader._layouts.values())) if hasattr(image, "_reader") else None
        if chunk_path == "native" and layout is not None:
            assert layout._where is None and layout._found        # looked up, not mapped
        np.testing.assert_array_equal(image[...][0, 0], data)       # all: the whole map
        if chunk_path == "native" and layout is not None:
            assert layout._where is not None
