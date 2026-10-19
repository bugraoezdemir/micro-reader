"""MRC backend, on synthetic files written to the MRC2014 layout."""
from __future__ import annotations

import os
import struct
import threading

import numpy as np
import pytest

import micro_reader
from micro_reader import CorruptFileError, UnsupportedFormatError
from micro_reader.backends import mrc as mrc_backend

MODE_OF = {"i1": 0, "i2": 1, "f4": 2, "c8": 4, "u2": 6, "f2": 12}


def stored_axes(axis_order) -> str:
    """The axes of the stored array (sections, rows, columns) for (MAPC, MAPR, MAPS)."""
    mapc, mapr, maps = axis_order
    return "".join("xyz"[m - 1] for m in (maps, mapr, mapc))


def write_mrc(path, data, order="<", ext_header=0, stamp=True, map_tag=True,
              axis_order=(1, 2, 3), mode=None, cell=None, angles=(90.0, 90.0, 90.0)):
    """Minimal MRC2014 writer: header, optional extended header, raw data.

    ``data`` is (z, y, x) in display orientation (row 0 at the top); Y is
    stored bottom-up, as MRC files are, and the axes in the order *axis_order*
    (MAPC, MAPR, MAPS) names.  *cell*: CELLA (X, Y, Z lengths in Å; default
    one Å per pixel); *angles*: CELLB.
    """
    data = np.asarray(data)
    if data.ndim == 2:
        data = data[None]
    nz, ny, nx = data.shape
    stored = np.transpose(data[:, ::-1], ["zyx".index(a) for a in stored_axes(axis_order)])
    ns, nr, nc = stored.shape
    dtype = data.dtype.newbyteorder(order)
    if mode is None:
        mode = MODE_OF[data.dtype.str[1:]]
    head = bytearray(1024)
    struct.pack_into(order + "4i", head, 0, nc, nr, ns, mode)     # NX NY NZ: columns, rows, sections
    struct.pack_into(order + "3i", head, 28, nx, ny, nz)          # MX MY MZ: per axis
    struct.pack_into(order + "3f", head, 40, *(cell or (nx, ny, nz)))   # CELLA
    struct.pack_into(order + "3f", head, 52, *angles)              # CELLB
    struct.pack_into(order + "3i", head, 64, *axis_order)          # MAPC MAPR MAPS
    struct.pack_into(order + "i", head, 92, ext_header)            # NSYMBT
    if map_tag:
        head[208:212] = b"MAP "
    if stamp:
        head[212:214] = b"\x44\x44" if order == "<" else b"\x11\x11"
    with open(path, "wb") as fh:
        fh.write(head)
        fh.write(b"\xab" * ext_header)
        fh.write(stored.astype(dtype).tobytes())
    return str(path)


def _data(dtype, shape=(4, 37, 53), seed=0):
    rng = np.random.default_rng(seed)
    if np.dtype(dtype).kind == "c":
        return (rng.normal(size=shape) + 1j * rng.normal(size=shape)).astype(dtype)
    if np.dtype(dtype).kind == "f":
        return rng.normal(size=shape).astype(dtype)
    info = np.iinfo(dtype)
    return rng.integers(info.min, info.max, size=shape, endpoint=True).astype(dtype)


@pytest.mark.parametrize("order", ["<", ">"])
@pytest.mark.parametrize("dtype", ["i1", "i2", "f4", "c8", "u2", "f2"])
def test_every_mode_and_byte_order_reads_back(tmp_path, dtype, order):
    data = _data(dtype)
    path = write_mrc(tmp_path / "a.mrc", data, order=order)
    with micro_reader.open(path) as img:
        scene = img.images[0]
        assert img.backend == "mrc"
        assert scene.axes == "zyx" and scene.shape == data.shape
        out = scene[...]
        assert out.dtype == np.dtype(dtype) and out.dtype.isnative
        np.testing.assert_array_equal(out, data)


@pytest.mark.parametrize("key", [
    (0, slice(0, 10), slice(0, 20)), (slice(1, 3), slice(5, 36), slice(7, 8)),
    (3, 36, 52), (slice(None), slice(10, 11)), (Ellipsis, slice(40, 53)),
    (2, slice(0, 37, 5), slice(1, 53, 7)),
])
def test_exact_rectangles_and_no_over_read(tmp_path, key, monkeypatch):
    # the exact path: no skipped columns are read (see test_*_spans for merged rows)
    monkeypatch.setattr(mrc_backend, "SEPARATE_READ_BYTES", 0)
    data = _data("u2")
    with micro_reader.open(write_mrc(tmp_path / "a.mrc", data)) as img:
        scene = img.images[0]
        np.testing.assert_array_equal(scene[key], data[key])
        assert scene.stats.over_read == 1.0


def test_rows_are_returned_top_down_like_bioformats(tmp_path):
    """The first row stored is the bottom of the image (MRC's lower-left
    origin); checked against the raw file, independently of write_mrc."""
    data = _data("u2")
    path = write_mrc(tmp_path / "a.mrc", data)
    stored = np.fromfile(path, dtype="<u2", offset=1024).reshape(data.shape)
    with micro_reader.open(path) as img:
        scene = img.images[0]
        np.testing.assert_array_equal(scene[1, -1], stored[1, 0])
        np.testing.assert_array_equal(scene[1, 0], stored[1, -1])
        np.testing.assert_array_equal(scene[:, 5:9, 3:30], stored[:, ::-1][:, 5:9, 3:30])


@pytest.mark.parametrize("order", ["<", ">"])
def test_full_width_reads_across_chunk_boundaries(tmp_path, monkeypatch, order):
    """Full-width rows are read in chunks; 5-row chunks with odd row ranges
    make every chunk boundary land somewhere different."""
    from micro_reader.backends import mrc
    monkeypatch.setattr(mrc, "CHUNK_BYTES", 5 * 53 * 2)
    data = _data("u2")
    with micro_reader.open(write_mrc(tmp_path / "a.mrc", data, order=order)) as img:
        scene = img.images[0]
        for key in [(slice(None),), (1, slice(3, 36)), (slice(1, 4), slice(0, 7)),
                    (2, slice(30, 37)), (0, slice(12, 13))]:
            np.testing.assert_array_equal(scene[key], data[key])
        assert scene.stats.over_read == 1.0


def test_keyword_form(tmp_path):
    data = _data("u2")
    with micro_reader.open(write_mrc(tmp_path / "a.mrc", data)) as img:
        scene = img.images[0]
        np.testing.assert_array_equal(scene.read(z=2, x=slice(3, 9)), data[2, :, 3:9])
        with pytest.raises(KeyError):
            scene.read(c=0)


def test_extended_header_is_skipped(tmp_path):
    data = _data("f4")
    path = write_mrc(tmp_path / "a.mrc", data, ext_header=4096)
    with micro_reader.open(path) as img:
        np.testing.assert_array_equal(img.images[0][...], data)


@pytest.mark.parametrize("order", ["<", ">"])
def test_byte_order_without_a_machine_stamp(tmp_path, order):
    """Older writers leave the stamp empty; the header is checked for sense."""
    data = _data("i2")
    path = write_mrc(tmp_path / "a.mrc", data, order=order, stamp=False)
    with micro_reader.open(path) as img:
        np.testing.assert_array_equal(img.images[0][...], data)


def test_old_file_without_map_tag_is_found_by_extension(tmp_path):
    data = _data("u2")
    path = write_mrc(tmp_path / "old.mrc", data, map_tag=False)
    with micro_reader.open(path) as img:
        np.testing.assert_array_equal(img.images[0][...], data)
    other = write_mrc(tmp_path / "old.bin", data, map_tag=False)
    with pytest.raises(UnsupportedFormatError):
        micro_reader.open(other)
    with micro_reader.open(other, backend="mrc") as img:   # explicit backend
        assert img.images[0].shape == data.shape


def test_2d_image_is_one_section(tmp_path):
    data = _data("u2", shape=(21, 34))
    with micro_reader.open(write_mrc(tmp_path / "a.mrc", data)) as img:
        scene = img.images[0]
        assert scene.shape == (1, 21, 34)
        np.testing.assert_array_equal(scene[0], data)


def test_truncated_file_is_refused(tmp_path):
    path = write_mrc(tmp_path / "a.mrc", _data("u2"))
    with open(path, "r+b") as fh:
        fh.truncate(os.path.getsize(path) - 10)
    with pytest.raises(CorruptFileError, match="truncated"):
        micro_reader.open(path)


def test_unsupported_mode_and_invalid_axis_order(tmp_path):
    data = _data("i2")
    with pytest.raises(UnsupportedFormatError, match="mode 3"):
        micro_reader.open(write_mrc(tmp_path / "m3.mrc", data, mode=3))
    path = write_mrc(tmp_path / "ax.mrc", data)
    with open(path, "r+b") as fh:                       # MAPC, MAPR, MAPS = 2, 2, 3
        fh.seek(64)
        fh.write(struct.pack("<3i", 2, 2, 3))
    with pytest.raises(CorruptFileError, match="axis order"):
        micro_reader.open(path)


PERMUTATIONS = [(1, 2, 3), (1, 3, 2), (2, 1, 3), (2, 3, 1), (3, 1, 2), (3, 2, 1)]


@pytest.mark.parametrize("paths", ["exact", "spans", "chunks"])
@pytest.mark.parametrize("axis_order", PERMUTATIONS)
def test_every_axis_order_reads_with_true_axes(tmp_path, monkeypatch, axis_order, paths):
    """The stored order, each axis named by MAPC / MAPR / MAPS; Y top-down
    wherever it is stored; as_axes gives z, y, x back.  All three read paths:
    one read per row, merged row spans, full rows in small chunks."""
    if paths == "exact":
        monkeypatch.setattr(mrc_backend, "SEPARATE_READ_BYTES", 0)
    if paths == "chunks":
        monkeypatch.setattr(mrc_backend, "CHUNK_BYTES", 3 * 64 * 2)
    data = _data("u2", shape=(5, 13, 17))
    axes = stored_axes(axis_order)
    expected = np.transpose(data, ["zyx".index(a) for a in axes])
    path = write_mrc(tmp_path / "p.mrc", data, axis_order=axis_order, order=">")
    with micro_reader.open(path) as img:
        image = img.images[0]
        assert image.axes == axes and image.shape == expected.shape
        np.testing.assert_array_equal(image[...], expected)
        np.testing.assert_array_equal(image.as_axes("zyx")[...], data)
        rng = np.random.default_rng(sum(axis_order))
        for _ in range(25):
            key = []
            for n in expected.shape:
                a = int(rng.integers(0, n))
                key.append(slice(a, int(rng.integers(a + 1, n + 1))))
            np.testing.assert_array_equal(image[tuple(key)], expected[tuple(key)],
                                          err_msg=str(key))
        if paths == "exact":
            assert image.stats.over_read == 1.0


def test_permuted_y_is_flipped_against_the_raw_file(tmp_path):
    """(3, 1, 2): columns along Z, rows along X, sections along Y (EMD-3001's
    order).  Checked against the raw file, independently of write_mrc: the
    first section stored is the bottom of Y."""
    data = _data("u2", shape=(6, 5, 7))
    path = write_mrc(tmp_path / "p.mrc", data, axis_order=(3, 1, 2))
    raw = np.fromfile(path, "<u2", offset=1024).reshape(5, 7, 6)   # sections, rows, columns
    with micro_reader.open(path) as img:
        image = img.images[0]
        assert image.axes == "yxz"
        np.testing.assert_array_equal(image[0], raw[-1])
        np.testing.assert_array_equal(image[...], raw[::-1])


def test_pixel_sizes_follow_the_axes_not_the_storage(tmp_path):
    data = _data("u2", shape=(6, 5, 7))
    path = write_mrc(tmp_path / "p.mrc", data, axis_order=(3, 1, 2), cell=(14.0, 5.0, 3.0))
    with micro_reader.open(path) as img:
        sizes = img.images[0].metadata.pixel_sizes
        assert {a: (round(q.value, 6), q.unit) for a, q in sizes.items()} == {
            "x": (2.0, "angstrom"), "y": (1.0, "angstrom"), "z": (0.5, "angstrom")}


def test_skewed_cell_warns_and_reads(tmp_path):
    data = _data("f4", shape=(6, 5, 7))
    path = write_mrc(tmp_path / "p.mrc", data, axis_order=(3, 1, 2), angles=(90.0, 94.326, 90.0))
    with pytest.warns(micro_reader.SkewedGridWarning, match="94.326"):
        img = micro_reader.open(path)
    with img:
        np.testing.assert_array_equal(img.images[0].as_axes("zyx")[...], data)
    import warnings
    for angles in [(90.0, 90.0, 90.0), (0.0, 0.0, 0.0)]:        # rectangular, or unset
        with warnings.catch_warnings():
            warnings.simplefilter("error", micro_reader.SkewedGridWarning)
            micro_reader.open(write_mrc(tmp_path / "q.mrc", data, angles=angles)).close()


def test_close_releases_the_file(tmp_path):
    """Windows keeps mapped files locked; closing must release them."""
    path = write_mrc(tmp_path / "a.mrc", _data("u2"))
    img = micro_reader.open(path)
    img.images[0][0]
    img.close()
    os.remove(path)                       # raises PermissionError if still mapped
    with pytest.raises(ValueError):
        img.images[0][0]


def test_results_are_copies_not_views(tmp_path):
    data = _data("u2")
    with micro_reader.open(write_mrc(tmp_path / "a.mrc", data)) as img:
        out = img.images[0][0]
        assert out.base is None or not isinstance(out.base, np.memmap)
        out[...] = 0                       # writable, and the file is untouched
        np.testing.assert_array_equal(img.images[0][0], data[0])


def test_concurrent_reads(tmp_path, monkeypatch):
    # the exact path: no skipped columns are read (see test_*_spans for merged rows)
    monkeypatch.setattr(mrc_backend, "SEPARATE_READ_BYTES", 0)
    data = _data("u2", shape=(8, 256, 300))
    with micro_reader.open(write_mrc(tmp_path / "a.mrc", data)) as img:
        scene, errors = img.images[0], []

        def worker(z):
            try:
                for y in range(0, 256, 32):
                    block = scene[z, y:y + 32, 50:250]
                    if not np.array_equal(block, data[z, y:y + 32, 50:250]):
                        errors.append((z, y))
            except Exception as exc:        # noqa: BLE001
                errors.append(exc)

        threads = [threading.Thread(target=worker, args=(z,)) for z in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert errors == []
        assert scene.stats.reads == 64 and scene.stats.over_read == 1.0


def test_narrow_reads_merge_rows(tmp_path):
    """Part of the columns of many rows: one read per run of rows, the skipped bytes
    included (they cost less than one more read per row); rows still bottom-up."""
    data = _data("u2", shape=(3, 64, 300))
    with micro_reader.open(write_mrc(tmp_path / "a.mrc", data)) as img:
        scene = img.images[0]
        np.testing.assert_array_equal(scene[1, 10:42, 50:250], data[1, 10:42, 50:250])
        assert scene.stats.pixels_read == 31 * 300 + 200
        np.testing.assert_array_equal(scene[:, 0:1, 7:9], data[:, 0:1, 7:9])     # one row
