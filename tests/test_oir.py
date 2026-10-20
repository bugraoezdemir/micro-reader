"""OIR backend on synthetic files (tests/_oir_writer.py), against the known
data and, where available, oirfile reading the same files (set OIRFILE_SRC
to an unpacked oirfile source tree if it is not installed)."""
from __future__ import annotations

import os
import sys
import threading

import numpy as np
import pytest

import micro_reader
from micro_reader import UnsupportedFormatError

from _oir_writer import write_oir

if os.environ.get("OIRFILE_SRC"):
    sys.path.insert(0, os.environ["OIRFILE_SRC"])
try:
    import oirfile
except ImportError:
    oirfile = None


def _data(shape=(2, 3, 4, 37, 53), dtype="u2", seed=0):
    rng = np.random.default_rng(seed)
    dtype = np.dtype(dtype)
    if dtype.kind == "f":
        return rng.normal(size=shape).astype(dtype)
    return rng.integers(1, np.iinfo(dtype).max, size=shape, endpoint=True).astype(dtype)


def _expected(data, channel_order=None, missing=(), truncate=(), block_bytes=4096):
    """What a reader must return: channels in metadata order, missing planes
    and bytes past a truncated plane's end as zeros."""
    out = data.copy()
    for t, c, z in missing:
        out[t, c, z] = 0
    for t, c, z in truncate:
        raw = bytearray(out[t, c, z].tobytes())
        last = (len(raw) - 1) // block_bytes * block_bytes
        keep = last + (len(raw) - last) // 2
        raw[keep:] = bytes(len(raw) - keep)
        out[t, c, z] = np.frombuffer(bytes(raw), data.dtype).reshape(out.shape[-2:])
    if channel_order:
        out = out[:, np.argsort(channel_order, kind="stable")]
    return out


CASES = {
    "row_aligned_blocks": dict(block_bytes=53 * 2 * 8),
    "unaligned_blocks":   dict(block_bytes=1000),
    "one_block_per_plane": dict(block_bytes=1 << 20),
    "uint8":              dict(dtype="u1", block_bytes=777),
    "float32":            dict(dtype="f4", block_bytes=2048),
    "channel_order":      dict(channel_order=[3, 1, 2]),
    "missing_plane":      dict(missing=((1, 2, 3),)),
    "truncated_plane":    dict(truncate=((0, 1, 2),)),
    "companion_file":     dict(companion_planes=((1, 0, 0), (1, 1, 3), (0, 2, 1))),
    "single_plane":       dict(shape=(1, 1, 1, 64, 80)),
}


def _write(tmp_path, name):
    case = dict(CASES[name])
    data = _data(case.pop("shape", (2, 3, 4, 37, 53)), case.pop("dtype", "u2"))
    path = str(tmp_path / f"{name}.oir")
    write_oir(path, data, **case)
    return path, _expected(data, case.get("channel_order"), case.get("missing", ()),
                           case.get("truncate", ()), case.get("block_bytes", 4096))


def _keys(shape, n=25, seed=3):
    rng = np.random.default_rng(seed)
    keys = [(Ellipsis,), (0,), (-1, 0)]
    for _ in range(n):
        key = []
        for size in shape:
            a = int(rng.integers(0, size))
            b = int(rng.integers(a + 1, size + 1))
            choice = rng.integers(0, 4)
            key.append(a if choice == 0 else slice(a, b, 2) if choice == 1 else slice(a, b))
        keys.append(tuple(key))
    return keys


@pytest.fixture(params=[1, 4], ids=["serial", "pool"])
def decode_threads(request):
    old = micro_reader.decode_threads()
    micro_reader.set_decode_threads(request.param)
    yield
    micro_reader.set_decode_threads(old)


@pytest.mark.parametrize("name", list(CASES))
def test_matches_known_data(tmp_path, name, decode_threads):
    path, expected = _write(tmp_path, name)
    with micro_reader.open(path) as img:
        series = img.images[0]
        assert img.backend == "oir" and series.axes == "tczyx"
        assert series.shape == expected.shape and series.dtype == expected.dtype
        for key in _keys(expected.shape):
            np.testing.assert_array_equal(series[key], expected[key], err_msg=f"{name} {key}")
        # exact reads: never more than asked (missing planes read nothing)
        assert series.stats.over_read <= 1.0


@pytest.mark.skipif(oirfile is None, reason="oirfile not available (set OIRFILE_SRC)")
@pytest.mark.parametrize("name", [n for n in CASES if n != "truncated_plane"])
def test_writer_agrees_with_oirfile(tmp_path, name):
    """oirfile reads the synthetic files as the known data too."""
    path, expected = _write(tmp_path, name)
    with oirfile.OirFile(path, squeeze=False) as f:
        arr, dims = f.asarray(), list(f.sizes)
    for d in "TCZ":
        if d not in dims:
            arr, dims = np.expand_dims(arr, 0), [d] + dims
    np.testing.assert_array_equal(arr.transpose([dims.index(d) for d in "TCZYX"]), expected)


@pytest.mark.parametrize("kwargs, reason", [
    (dict(extra_rows=5), "longer than one frame"),
])
def test_unsupported_layouts_are_refused(tmp_path, kwargs, reason):
    path = str(tmp_path / "x.oir")
    write_oir(path, _data((1, 2, 1, 16, 20)), **kwargs)
    with pytest.raises(UnsupportedFormatError, match=reason):
        micro_reader.open(path)


@pytest.mark.parametrize("kwargs", [dict(lambda_index=1), dict(axes={"LAMBDA": 3})])
def test_one_lambda_index_is_no_axis(tmp_path, kwargs):
    """Planes of one lambda index (or a LAMBDA axis without l planes): the
    plain t c z y x image (once refused as a lambda scan)."""
    path = str(tmp_path / "x.oir")
    data = _data((1, 2, 1, 16, 20))
    write_oir(path, data, **kwargs)
    with micro_reader.open(path) as f:
        assert f.images[0].axes == "tczyx"
        np.testing.assert_array_equal(f.images[0][...], data)


def test_lambda_scan_is_one_image_per_emission_band(tmp_path):
    """A lambda (spectral) scan: one image per emission band (index
    ``emission``), each with its wavelength, and the time step from the
    frames' own positions."""
    path = str(tmp_path / "x.oir")
    data = _data((3, 2, 2, 1, 16, 20))                       # l t c z y x
    write_oir(path, data, frame_axes={"LAMBDA": lambda lam, t, z: 500.0 + 25 * lam,
                                      "TIMELAPSE": lambda lam, t, z: 1500.0 * t})
    with micro_reader.open(path) as f:
        assert f.indices == [{"series": 0, "emission": k} for k in range(3)]
        for k, image in enumerate(f.images):
            assert (image.axes, image.shape) == ("tczyx", data.shape[1:])
            np.testing.assert_array_equal(image[...], data[k])
            np.testing.assert_array_equal(image[1, :, 0, 4:9, 2:7], data[k, 1, :, 0, 4:9, 2:7])
            meta = image.metadata
            assert meta.index_values["emission"] == (500.0 + 25 * k, "nanometer")
            assert meta.pixel_sizes["t"] == (1500.0, "millisecond")


def test_a_missing_continuation_in_the_middle_warns(tmp_path):
    """_00001 missing, _00002 present: a warning names _00001, and the frames
    in _00002 are still read (each frame says where it belongs)."""
    path, expected = _write(tmp_path, "companion_file")
    stem = os.path.splitext(path)[0]
    os.rename(stem + "_00001", stem + "_00002")
    with pytest.warns(micro_reader.MissingImagesWarning, match=r"companion_file_00001 is missing"):
        f = micro_reader.open(path)
    with f:
        np.testing.assert_array_equal(f.images[0][...], expected)


def test_close_releases_every_file(tmp_path, decode_threads):
    path, _ = _write(tmp_path, "companion_file")
    img = micro_reader.open(path)
    img.images[0][...]
    img.close()
    os.remove(path)
    os.remove(os.path.splitext(path)[0] + "_00001")
    with pytest.raises(ValueError):
        img.images[0][0]


def test_concurrent_reads(tmp_path, decode_threads):
    path, expected = _write(tmp_path, "companion_file")
    keys, errors = _keys(expected.shape, n=40, seed=9), []
    with micro_reader.open(path) as img:
        series = img.images[0]

        def worker(i):
            try:
                for key in keys[i::4]:
                    if not np.array_equal(series[key], expected[key]):
                        errors.append(key)
            except Exception as exc:        # noqa: BLE001
                errors.append(exc)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
    assert errors == []
