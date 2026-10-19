"""ICS / IDS (backends/ics.py) against the arrays written (tests/_ics_writer.py)."""
from __future__ import annotations

import os

import numpy as np
import pytest

import micro_reader
from micro_reader import CorruptFileError, UnsupportedFormatError

from _ics_writer import write_ics

RNG = np.random.default_rng(11)


def _keys(shape, n=25, seed=3):
    rng = np.random.default_rng(seed)
    out = [tuple(slice(None) for _ in shape)]
    for _ in range(n):
        key = []
        for size in shape:
            a = int(rng.integers(0, size))
            key.append(slice(a, int(rng.integers(a + 1, size + 1))))
        out.append(tuple(key))
    return out


def _check(path, data, axes):
    with micro_reader.open(path) as f:
        assert f.backend == "ics"
        image = f.images[0]
        assert (image.axes, image.shape, image.dtype) == (axes, data.shape, data.dtype.newbyteorder("="))
        for key in _keys(data.shape):
            np.testing.assert_array_equal(image[key], data[key], err_msg=str(key))


@pytest.mark.parametrize("dtype", ["u1", "<u2", ">u2", "<i2", "<u4", ">i4", "<f4", ">f4", "<f8"])
@pytest.mark.parametrize("coordinates", ["video", "cartesian"])
def test_types_byte_orders_and_row_order(tmp_path, dtype, coordinates):
    data = (RNG.random((3, 4, 21, 37)) * 200).astype(dtype)          # c z y x
    _check(write_ics(tmp_path, data, "czyx", coordinates=coordinates), data, "czyx")


@pytest.mark.parametrize("version", ["1.0", "2.0"])
@pytest.mark.parametrize("compression", ["uncompressed", "gzip"])
def test_versions_and_compression(tmp_path, version, compression):
    data = RNG.integers(0, 4000, (2, 3, 2, 17, 29)).astype("<u2")    # c t z y x
    path = write_ics(tmp_path, data, "ctzyx", version=version, compression=compression,
                     coordinates="cartesian")
    _check(path, data, "ctzyx")
    if version == "2.0":
        assert os.listdir(tmp_path) == ["image.ics"]


def test_dimension_names_and_dropped_size_one_dims(tmp_path):
    """Huygens' size-1 'p' (detector) is dropped; 'channel' names the c axis."""
    data = RNG.integers(0, 255, (3, 5, 9, 11)).astype("u1")          # c z y x
    path = write_ics(tmp_path, data, "czyx", extra_dims=[("p", 1)], order_names={"c": "channel"})
    _check(path, data, "czyx")


def test_unknown_dimension_larger_than_one_is_refused(tmp_path):
    data = np.zeros((5, 9, 11), "u1")
    write_ics(tmp_path, data, "zyx", extra_dims=[("p", 2)])
    (tmp_path / "image.ids").write_bytes(np.zeros((2, 5, 9, 11), "u1").tobytes())
    with pytest.raises(UnsupportedFormatError, match="dimension 'p' of size 2"):
        micro_reader.open(tmp_path / "image.ics")


def test_header_without_its_separator_line(tmp_path):
    """OME's ICS/gh-4197: no separator line and no ics_version (read as 1.0)."""
    data = RNG.integers(0, 4000, (2, 13, 15)).astype(">u2")
    _check(write_ics(tmp_path, data, "cyx", bare=True), data, "cyx")


def test_pixel_sizes_time_step_and_wavelengths(tmp_path):
    data = np.zeros((2, 3, 4, 8, 9), "<u2")                         # c t z y x
    sensor = [("s_params", "LambdaEx", "488.0", "561.0"), ("s_states", "LambdaEx", "verified", "estimated"),
              ("s_params", "LambdaEm", "520.0", "600.0"), ("s_states", "LambdaEm", "reported", "default")]
    path = write_ics(tmp_path, data, "ctzyx", scale={"x": 0.1, "y": 0.1, "z": 0.3, "t": 2.5},
                     units={"x": "micrometers", "y": "micrometers", "z": "micrometers", "t": "seconds"},
                     sensor=sensor)
    with micro_reader.open(path) as f:
        meta = f.images[0].metadata
        assert meta.pixel_sizes == {"x": (0.1, "micrometer"), "y": (0.1, "micrometer"),
                                    "z": (0.3, "micrometer"), "t": (2.5, "second")}
        first, second = meta.channels
        assert (first.excitation_wavelength, first.emission_wavelength) == \
            ((488.0, "nanometer"), (520.0, "nanometer"))
        # Huygens' estimate / default is no recorded value
        assert (second.excitation_wavelength, second.emission_wavelength) == (None, None)
        assert meta.position == {}


def test_undefined_units_give_no_pixel_size(tmp_path):
    path = write_ics(tmp_path, np.zeros((2, 8, 9), "<u2"), "zyx")    # libics' test image: undefined
    with micro_reader.open(path) as f:
        assert f.images[0].metadata.pixel_sizes == {}


def test_parts_open_the_header(tmp_path):
    """The .ids and Huygens Remote Manager's side files open the .ics."""
    data = RNG.integers(0, 255, (4, 6, 7)).astype("u1")
    path = write_ics(tmp_path, data, "zyx", side_files=(".parameters.txt", ".log.txt", ".hgsb"))
    for part in ("image.ids", "image.parameters.txt", "image.log.txt", "image.hgsb"):
        with micro_reader.open(tmp_path / part) as f:
            assert os.path.samefile(f.path, path)
            np.testing.assert_array_equal(f.images[0][...], data)


def test_group_files(tmp_path):
    write_ics(tmp_path, np.zeros((2, 6, 7), "u1"), "zyx", side_files=(".parameters.txt", ".hgsb"))
    g = micro_reader.group_files(str(tmp_path))
    (group,) = g.groups
    assert group.format == "ics" and os.path.basename(group.main) == "image.ics"
    names = {os.path.basename(p) for p in list(group.members) + list(group.parts)}
    assert names >= {"image.ids", "image.parameters.txt", "image.hgsb"}
    assert not g.singles


def test_missing_or_damaged_pixels(tmp_path):
    data = np.zeros((3, 8, 9), "<u2")
    path = write_ics(tmp_path, data, "zyx")
    with open(tmp_path / "image.ids", "r+b") as fh:                 # cut short
        fh.truncate(100)
    with pytest.raises(CorruptFileError, match="truncated"):
        micro_reader.open(path)
    os.remove(tmp_path / "image.ids")
    with pytest.raises(CorruptFileError, match="pixel file 'image.ids' is missing"):
        micro_reader.open(path)


def test_ids_without_its_header_cannot_be_read(tmp_path):
    write_ics(tmp_path, np.zeros((2, 8, 9), "u1"), "zyx")
    os.remove(tmp_path / "image.ics")
    with pytest.raises(Exception, match="image.ics"):
        micro_reader.open(tmp_path / "image.ids")


def test_unix_compress_pixels_are_refused(tmp_path):
    write_ics(tmp_path, np.zeros((2, 8, 9), "u1"), "zyx", ids_name="image.ids.Z")
    with pytest.raises(UnsupportedFormatError, match="Unix compress"):
        micro_reader.open(tmp_path / "image.ics")


def test_complex_pixels_are_refused(tmp_path):
    path = write_ics(tmp_path, np.zeros((2, 8, 9), "<f4"), "zyx")
    text = open(path, encoding="latin-1").read().replace("format\treal", "format\tcomplex")
    open(path, "w", encoding="latin-1").write(text)
    with pytest.raises(UnsupportedFormatError, match="complex"):
        micro_reader.open(path)


def test_closed_image(tmp_path):
    path = write_ics(tmp_path, np.zeros((2, 8, 9), "u1"), "zyx")
    f = micro_reader.open(path)
    image = f.images[0]
    f.close()
    with pytest.raises(ValueError, match="closed"):
        image[0]


def test_gzip_header_with_a_name_and_jumps_back_and_forth(tmp_path):
    """A gzip header carrying the file name (FNAME), read in a jumping order:
    the streamed decoder resumes from its checkpoints."""
    import gzip
    data = RNG.integers(0, 60000, (6, 40, 300)).astype("<u2")
    path = write_ics(tmp_path, data, "zyx")
    raw = (tmp_path / "image.ids").read_bytes()
    with gzip.GzipFile(filename="image.ids", mode="wb", fileobj=open(tmp_path / "image.ids", "wb")) as g:
        g.write(raw)
    text = open(path, encoding="latin-1").read().replace("compression	uncompressed", "compression	gzip")
    open(path, "w", encoding="latin-1").write(text)
    with micro_reader.open(path) as f:
        image = f.images[0]
        for z in (5, 0, 3, 3, 1, 4):
            np.testing.assert_array_equal(image[z, 7:33, 100:250], data[z, 7:33, 100:250])


def test_gzip_of_several_members_is_refused(tmp_path):
    import gzip
    data = np.arange(2 * 8 * 9, dtype="<u2").reshape(2, 8, 9)
    path = write_ics(tmp_path, data, "zyx", compression="gzip")
    raw = data.tobytes()
    (tmp_path / "image.ids").write_bytes(gzip.compress(raw[:100]) + gzip.compress(raw[100:]))
    with pytest.raises(UnsupportedFormatError, match="several gzip members"):
        micro_reader.open(path)
