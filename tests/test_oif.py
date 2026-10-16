"""Olympus OIF / OIB backend on synthetic files (the real OME FV1000 sample was
compared with Bio-Formats plane by plane: docs/PLAN.md, "Olympus OIB / OIF")."""
from __future__ import annotations

import os

import numpy as np
import pytest

pytest.importorskip("tifffile")

import micro_reader  # noqa: E402
from micro_reader import (CompositeWarning, CorruptFileError, MissingImagesWarning,  # noqa: E402
                          UnsupportedFormatError, _cfb)
from micro_reader.backends import oif  # noqa: E402

from _cfb_writer import write_cfb  # noqa: E402
from _oif_writer import write_oib, write_oif  # noqa: E402


def _data(shape=(3, 2, 4, 37, 53), seed=0):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 4096, size=shape, dtype=np.uint16)


def _write(tmp_path, kind, data, **kwargs):
    if kind == "oib":
        return write_oib(tmp_path / "a.oib", data, **kwargs)
    return write_oif(str(tmp_path), "a", data, **kwargs)


@pytest.mark.parametrize("kind", ["oib", "oif"])
@pytest.mark.parametrize("shape", [(1, 2, 4, 37, 53), (3, 2, 4, 37, 53), (5, 1, 1, 20, 30)])
def test_planes_and_rectangles(tmp_path, kind, shape):
    data = _data(shape)
    path = _write(tmp_path, kind, data, rows_per_strip=7)
    with micro_reader.open(path) as f:
        assert f.backend == "oif" and f.format == kind
        im = f.images[0]
        assert im.axes == "tczyx" and im.shape == shape and im.dtype == np.uint16
        np.testing.assert_array_equal(im[...], data)
        np.testing.assert_array_equal(im[:, :, :, 5:31, 9:40], data[:, :, :, 5:31, 9:40])
        np.testing.assert_array_equal(im[-1, 0, -1, 36:, :], data[-1, 0, -1, 36:, :])


def test_fragmented_oib_streams(tmp_path):
    """Sectors of the plane streams allocated in turn: every plane is several runs."""
    data = _data((1, 2, 3, 40, 60))
    path = write_oib(tmp_path / "a.oib", data, interleave=True, rows_per_strip=5)
    runs = [len(s.runs) for s in _cfb.CompoundFile(path).streams.values() if s.runs]
    assert max(runs) > 1
    with micro_reader.open(path) as f:
        np.testing.assert_array_equal(f.images[0][...], data)
        np.testing.assert_array_equal(f.images[0][0, 1, 2, 3:38, 7:51], data[0, 1, 2, 3:38, 7:51])


def test_compressed_planes(tmp_path):
    data = _data((1, 2, 2, 20, 30))
    path = _write(tmp_path, "oif", data, compression="zlib")
    with micro_reader.open(path) as f:
        np.testing.assert_array_equal(f.images[0][...], data)


def test_metadata(tmp_path):
    data = _data((3, 2, 4, 9, 11))
    path = _write(tmp_path, "oib", data, pixel="0,207.0", z_interval=510, t_interval=2500)
    with micro_reader.open(path) as f:
        meta = f.images[0].metadata
    assert meta.pixel_sizes["x"].value == pytest.approx(0.207)
    assert meta.pixel_sizes["x"].unit == "micrometer"
    assert meta.pixel_sizes["z"].to("micrometer").value == pytest.approx(0.51)
    assert meta.pixel_sizes["t"].to("second").value == pytest.approx(2.5)
    assert [(c.name, c.color, c.emission_wavelength.value, c.excitation_wavelength.value)
            for c in meta.channels] == [("CH1", "#0000FF", 461.0, 405.0),
                                        ("CH3", "#FF0000", 618.0, 543.0)]


@pytest.mark.parametrize("written, value", [("0,207.0", 0.207), ("1,35.0", 1.35), ("510", 510.0),
                                            ("-1530410.0", -1530410.0), ('"0.5"', 0.5), ("", None)])
def test_numbers_in_the_acquiring_locale(written, value):
    assert oif.number(written.strip('"')) == value


def test_missing_planes_read_as_zeros(tmp_path):
    data = _data((1, 2, 3, 9, 11))
    path = _write(tmp_path, "oif", data, skip={(0, 1, 2)})
    with pytest.warns(MissingImagesWarning, match="1 of 6 planes"):
        f = micro_reader.open(path)
    with f:
        assert not f.images[0][0, 1, 2].any()
        np.testing.assert_array_equal(f.images[0][0, 0], data[0, 0])


def test_an_axis_not_supported_yet_is_refused(tmp_path):
    path = _write(tmp_path, "oif", _data((1, 1, 2, 9, 11)), extra_axis=("L", 4, "nm"))
    with pytest.raises(UnsupportedFormatError, match="axis 'L'"):
        micro_reader.open(path)


def test_the_storage_folder_is_required(tmp_path):
    path = _write(tmp_path, "oif", _data((1, 1, 2, 9, 11)))
    os.rename(path + ".files", path + ".gone")
    with pytest.raises(CorruptFileError, match="storage folder"):
        micro_reader.open(path)


def test_other_ole_files_are_not_oib(tmp_path):
    path = tmp_path / "doc.bin"
    write_cfb(str(path), {"WordDocument": b"x" * 5000, "SummaryInformation": b"y" * 100})
    assert not oif.sniff(_head(path), str(path))


def _head(path):
    with open(path, "rb") as fh:
        return fh.read(1024)


# -- the parts of an OIF (docs/COMPOSITE.md) ------------------------------------------------

def test_any_file_of_the_storage_folder_opens_the_oif(tmp_path):
    data = _data((1, 2, 2, 9, 11))
    path = _write(tmp_path, "oif", data)
    storage = path + ".files"
    for part in ("s_C002Z001.tif", "s_C001Z002.pty", "s_LUT1.lut", "s_Thumb.bmp"):
        with micro_reader.open(os.path.join(storage, part)) as f:
            assert f.backend == "oif" and os.path.samefile(f.path, path)
    for folder in (storage, str(tmp_path)):
        with micro_reader.open(folder) as f:
            assert os.path.samefile(f.path, path)
            np.testing.assert_array_equal(f.images[0][...], data)


def test_a_plane_without_its_oif_is_read_alone(tmp_path):
    data = _data((1, 2, 2, 9, 11))
    path = _write(tmp_path, "oif", data)
    os.remove(path)
    plane = os.path.join(path + ".files", "s_C002Z001.tif")
    with pytest.warns(CompositeWarning, match="which is missing"):
        f = micro_reader.open(plane)
    with f:
        assert f.backend == "tiff"
        np.testing.assert_array_equal(f.images[0][...], data[0, 1, 0])
    with pytest.raises(CorruptFileError, match="which is missing"):
        micro_reader.open(os.path.join(path + ".files", "s_C001Z001.pty"))
