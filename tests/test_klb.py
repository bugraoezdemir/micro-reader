"""KLB: written here from the format's description, then read back by rectangle."""
import bz2
import math
import struct
import zlib

import numpy as np
import pytest

import micro_reader
from micro_reader import CorruptFileError, UnsupportedFormatError

BLOCK = (4, 3, 2, 1, 2)                  # x y z c t


def write_klb(path, data, sampling=(1, 1, 1, 1, 1), compression=1, block=BLOCK, data_type=1,
              version=2):
    """*data*: t c z y x."""
    xyzct = data.shape[::-1]
    pack = {0: bytes, 1: bz2.compress, 2: zlib.compress}[compression]
    counts = [math.ceil(n / b) for n, b in zip(xyzct, block)]
    stored, ends, end = [], [], 0
    for bt in range(counts[4]):
        for bc in range(counts[3]):
            for bz in range(counts[2]):
                for by in range(counts[1]):
                    for bx in range(counts[0]):
                        part = data[bt * block[4]:(bt + 1) * block[4],
                                    bc * block[3]:(bc + 1) * block[3],
                                    bz * block[2]:(bz + 1) * block[2],
                                    by * block[1]:(by + 1) * block[1],
                                    bx * block[0]:(bx + 1) * block[0]]
                        raw = pack(part.tobytes())
                        stored.append(raw)
                        end += len(raw)
                        ends.append(end)
    head = struct.pack("<B5I5fBB256s5I", version, *xyzct, *sampling, data_type, compression,
                       b"a note", *block)
    path.write_bytes(head + struct.pack(f"<{len(ends)}Q", *ends) + b"".join(stored))


@pytest.fixture
def data():
    return np.arange(3 * 2 * 5 * 7 * 9, dtype="<u2").reshape(3, 2, 5, 7, 9)


@pytest.mark.parametrize("compression", [0, 1, 2])
def test_every_time_point_and_channel_reads(tmp_path, data, compression):
    path = tmp_path / "a.klb"
    write_klb(path, data, compression=compression)
    with micro_reader.open(path) as f:
        image = f.images[0]
        assert f.backend == "klb"
        assert image.axes == "tczyx" and image.shape == data.shape and image.dtype == data.dtype
        np.testing.assert_array_equal(image[...], data)
        np.testing.assert_array_equal(image[1:3, 1, 2:5, 1:6, 3:8], data[1:3, 1, 2:5, 1:6, 3:8])


def test_a_read_decodes_only_the_blocks_it_touches(tmp_path, data):
    path = tmp_path / "a.klb"
    write_klb(path, data)
    with micro_reader.open(path, axes="native") as f:
        image = f.images[0]
        assert image.read_unit == BLOCK[::-1]
        image[0, 0, 0, 0, 0]
        assert image.stats.pixels_read == np.prod(BLOCK[::-1])


def test_a_second_read_comes_from_the_cache(tmp_path, data):
    path = tmp_path / "a.klb"
    write_klb(path, data)
    with micro_reader.open(path, axes="native") as f:
        image = f.images[0]
        image[0, 0, 0]
        first = image.stats.pixels_read
        image[0, 0, 0]
        assert image.stats.pixels_read == first


def test_the_header_sampling_gives_the_pixel_sizes(tmp_path, data):
    path = tmp_path / "a.klb"
    write_klb(path, data, sampling=(0.4, 0.4, 2, 1, 30))
    with micro_reader.open(path) as f:
        sizes = f.images[0].metadata.pixel_sizes
    assert sizes["x"].value == pytest.approx(0.4) and sizes["x"].unit == "micrometer"
    assert sizes["z"].value == 2 and sizes["t"].value == 30 and sizes["t"].unit == "second"
    assert "c" not in sizes


@pytest.mark.parametrize("sampling, expected", [
    ((0.4, 0.4, 2, 1, 1), {"x", "y", "z"}),       # no time step set
    ((0.4, 0.4, 1, 1, 1), {"x", "y"}),            # no z step set
    ((1, 1, 2, 1, 1), {"x", "y", "z"}),           # a z of its own: x and y of 1 are real
    ((1, 1, 1, 1, 30), {"t"}),
    ((-1, -1, -1, 1, 1), set()),
])
def test_a_one_in_the_header_is_its_default(tmp_path, data, sampling, expected):
    path = tmp_path / "a.klb"
    write_klb(path, data, sampling=sampling)
    with micro_reader.open(path) as f:
        assert set(f.images[0].metadata.pixel_sizes) == expected


def test_one_time_point_has_no_time_step(tmp_path):
    path = tmp_path / "a.klb"
    write_klb(path, np.zeros((1, 1, 2, 4, 4), "<u2"), sampling=(0.4, 0.4, 2, 1, 30))
    with micro_reader.open(path) as f:
        assert "t" not in f.images[0].metadata.pixel_sizes


def test_a_large_block_is_decoded_once(tmp_path):
    data = np.ones((1, 1, 64, 1024, 1024), "<u2")           # 137 MB, above the default cache's share
    path = tmp_path / "a.klb"
    write_klb(path, data, block=(1024, 1024, 64, 1, 1))
    with micro_reader.open(path, axes="native") as f:
        image = f.images[0]
        image[0, 0, 0]
        first = image.stats.pixels_read
        image[0, 0, 1]
        assert image.stats.pixels_read == first


def test_a_header_of_ones_records_no_size(tmp_path, data):
    path = tmp_path / "a.klb"
    write_klb(path, data)
    with micro_reader.open(path) as f:
        assert not f.images[0].metadata.pixel_sizes


@pytest.mark.parametrize("code, dtype", [(0, "u1"), (2, "u4"), (5, "i2"), (8, "f4"), (9, "f8")])
def test_the_data_types(tmp_path, code, dtype):
    data = np.arange(60, dtype=dtype).reshape(1, 1, 3, 4, 5)
    path = tmp_path / "a.klb"
    write_klb(path, data, data_type=code)
    with micro_reader.open(path) as f:
        np.testing.assert_array_equal(f.images[0][...], data)


def test_another_header_version_is_refused(tmp_path, data):
    path = tmp_path / "a.klb"
    write_klb(path, data, version=3)
    with pytest.raises(UnsupportedFormatError, match="version 3"):
        micro_reader.open(path)


def test_a_truncated_file_is_refused(tmp_path, data):
    path = tmp_path / "a.klb"
    write_klb(path, data)
    path.write_bytes(path.read_bytes()[:-40])
    with pytest.raises(CorruptFileError, match="truncated"):
        micro_reader.open(path)


def test_a_damaged_block_names_the_block(tmp_path, data):
    path = tmp_path / "a.klb"
    write_klb(path, data)
    raw = bytearray(path.read_bytes())
    raw[-20:] = b"\0" * 20
    path.write_bytes(bytes(raw))
    with micro_reader.open(path) as f:
        with pytest.raises(CorruptFileError, match="block"):
            f.images[0][...]
