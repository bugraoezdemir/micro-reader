"""ND2 backend on synthetic files (tests/_nd2_writer.py), checked against the
known data and, where installed, against the nd2 library reading the same
file (which also validates the writer)."""
from __future__ import annotations

import os
import struct
import threading

import numpy as np
import pytest

import micro_reader
from micro_reader import MicroReaderError, UnsupportedFormatError
from micro_reader.backends import nd2 as nd2_backend

from _nd2_writer import write_nd2

try:
    import nd2 as nd2lib
except ImportError:                   # the cross-check is optional
    nd2lib = None


def _frames(n, h=37, w=53, channels=2, samples=1, dtype="u2", seed=0):
    rng = np.random.default_rng(seed)
    dtype = np.dtype(dtype)
    if dtype.kind == "f":
        return rng.normal(size=(n, h, w, channels, samples)).astype(dtype)
    return rng.integers(0, np.iinfo(dtype).max, size=(n, h, w, channels, samples),
                        endpoint=True).astype(dtype)


def _expected(frames, loops, missing=()):
    """Per series (XY position): t c z y x [s] from frames in loop order."""
    frames = frames.copy()
    for i in missing:
        frames[i] = 0
    counts = [c for _, c in loops]
    kinds = ["T" if k == "NE" else k for k, _ in loops]
    arr = frames.reshape(*counts, *frames.shape[1:])       # loops..., y, x, c, s
    for k in "TPZ":
        if k not in kinds:
            arr = np.expand_dims(arr, 0)
            kinds.insert(0, k)
    n = len(kinds)
    order = [kinds.index("P"), kinds.index("T"), n + 2, kinds.index("Z"), n, n + 1, n + 3]
    arr = arr.transpose(order)                             # p, t, c, z, y, x, s
    return [a if a.shape[-1] > 1 else a[..., 0] for a in arr]


CASES = {
    "z_rgb_padded":   dict(frames=_frames(5, channels=1, samples=3, dtype="u1"),
                           loops=[("Z", 5)], pad_bytes=3),
    "t_p_z":          dict(frames=_frames(24), loops=[("T", 3), ("P", 2), ("Z", 4)]),
    "p_outer":        dict(frames=_frames(6), loops=[("P", 3), ("T", 2)]),
    "lossless":       dict(frames=_frames(6), loops=[("T", 2), ("Z", 3)],
                           compression="lossless"),
    "lossless_padded_rgb": dict(frames=_frames(4, channels=2, samples=3, dtype="u1"),
                                loops=[("T", 4)], compression="lossless", pad_bytes=5),
    "padded_multichannel_rgb": dict(frames=_frames(3, channels=2, samples=3, dtype="u1"),
                                    loops=[("Z", 3)], pad_bytes=7),
    "float32":        dict(frames=_frames(3, dtype="f4"), loops=[("Z", 3)]),
    "ne_time":        dict(frames=_frames(5), loops=[("NE", 5)], ne_periods=[2, 3]),
    "invalid_position": dict(frames=_frames(4), loops=[("T", 2), ("P", 2)],
                             valid_positions=[1, 0, 1]),
    "missing_frame":  dict(frames=_frames(4), loops=[("Z", 4)], missing=(2,)),
    "single_frame":   dict(frames=_frames(1, channels=3), loops=[]),
}


def _write(tmp_path, name):
    case = dict(CASES[name])
    path = str(tmp_path / f"{name}.nd2")
    frames, loops = case.pop("frames"), case.pop("loops")
    write_nd2(path, frames, loops, **case)
    return path, _expected(frames, loops, case.get("missing", ()))


def _keys(shape, n=20, seed=3):
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
        assert img.backend == "nd2"
        assert len(img.images) == len(expected)
        for series, ref in zip(img.images, expected):
            assert series.shape == ref.shape and series.dtype == ref.dtype
            assert series.axes == ("tczyxs" if ref.ndim == 6 else "tczyx")
            for key in _keys(ref.shape):
                np.testing.assert_array_equal(series[key], ref[key], err_msg=f"{name} {key}")


#: nd2 0.11 reads padded rows with more than one channel of RGB components
#: wrongly: it steps between channels by one component instead of one pixel's
#: components (strides (widthBytes, n_comp*bpc, bpc, bpc)).  Our reader is
#: checked against the known data above; strict, so a fix upstream shows up.
_ND2_PADDED_MULTICHANNEL_RGB_BUG = pytest.mark.xfail(
    strict=True, reason="nd2 library: wrong channel stride for padded multi-channel RGB")


@pytest.mark.skipif(nd2lib is None, reason="nd2 library not installed")
@pytest.mark.parametrize("name", [
    pytest.param(n, marks=_ND2_PADDED_MULTICHANNEL_RGB_BUG)
    if n in ("lossless_padded_rgb", "padded_multichannel_rgb") else n
    for n in CASES])
def test_writer_agrees_with_nd2_library(tmp_path, name):
    """The nd2 library reads the synthetic files as the known data too, so
    the writer (and with it the tests above) follows the real format."""
    path, expected = _write(tmp_path, name)
    with nd2lib.ND2File(path) as f:
        sizes = dict(f.sizes)
        arr = f.asarray()
    assert sizes.get("P", 1) == len(expected)
    axes = list(sizes)
    for p, ref in enumerate(expected):
        a, ax = (arr[(slice(None),) * axes.index("P") + (p,)], [x for x in axes if x != "P"]) \
            if "P" in axes else (arr, list(axes))
        for k in "TCZ":
            if k not in ax:
                a, ax = np.expand_dims(a, 0), [k] + ax
        order = [ax.index(k) for k in "TCZYX"] + ([ax.index("S")] if "S" in ax else [])
        np.testing.assert_array_equal(a.transpose(order), ref)


def test_uncompressed_reads_only_the_rows_asked_for(tmp_path, monkeypatch):
    # the exact path: no skipped columns are read (see test_*_spans for merged rows)
    monkeypatch.setattr(nd2_backend, "SEPARATE_READ_BYTES", 0)
    path, expected = _write(tmp_path, "z_rgb_padded")
    with micro_reader.open(path) as img:
        series = img.images[0]
        assert series.read_unit is None
        np.testing.assert_array_equal(series[0, 0, 2, 10:14], expected[0][0, 0, 2, 10:14])
        assert series.stats.pixels_read == 4 * 53 * 3                 # 4 whole rows
        series.stats.reset()
        np.testing.assert_array_equal(series[0, 0, 2, 10:14, 5:9], expected[0][0, 0, 2, 10:14, 5:9])
        assert series.stats.pixels_read == 4 * 4 * 3                  # narrow: only its bytes


def test_lossless_frames_decoded_once(tmp_path):
    path, expected = _write(tmp_path, "lossless")
    with micro_reader.open(path) as img:
        series = img.images[0]
        assert series.read_unit == (1, 2, 1, 37, 53)
        series[0, 0, 1, 0:5, 0:5]
        assert series.stats.pixels_read == 37 * 53 * 2                # the whole frame
        series[0, 1, 1, 20:30, 10:40]
        assert series.stats.pixels_read == 37 * 53 * 2                # cached


def test_legacy_nd2_is_refused(tmp_path):
    path = str(tmp_path / "old.nd2")
    with open(path, "wb") as fh:
        fh.write(struct.pack("<I", 0x0C000000) + b"\x00" * 200)
    with pytest.raises((UnsupportedFormatError, MicroReaderError)):
        micro_reader.open(path, backend="nd2")


def test_truncated_file_is_refused(tmp_path):
    path, _ = _write(tmp_path, "t_p_z")
    with open(path, "r+b") as fh:
        fh.truncate(os.path.getsize(path) - 100)
    with pytest.raises(MicroReaderError):
        micro_reader.open(path)


def test_close_releases_the_file(tmp_path, decode_threads):
    path, _ = _write(tmp_path, "lossless")
    img = micro_reader.open(path)
    img.images[0][...]
    img.close()
    os.remove(path)
    with pytest.raises(ValueError):
        img.images[0][0]


def test_concurrent_reads(tmp_path, decode_threads):
    path, expected = _write(tmp_path, "t_p_z")
    errors = []
    with micro_reader.open(path) as img:
        def worker(i):
            try:
                for p, (series, ref) in enumerate(zip(img.images, expected)):
                    for key in _keys(ref.shape, n=10, seed=i * 7 + p):
                        if not np.array_equal(series[key], ref[key]):
                            errors.append((p, key))
            except Exception as exc:        # noqa: BLE001
                errors.append(exc)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
    assert errors == []


def test_uncompressed_narrow_reads_merge_rows(tmp_path):
    """Narrow columns: the few skipped bytes per row cost less than one read per row,
    so whole rows are read (as in LIF); pixels stay exact."""
    path, expected = _write(tmp_path, "z_rgb_padded")
    with micro_reader.open(path) as img:
        series = img.images[0]
        np.testing.assert_array_equal(series[0, 0, 2, 10:14, 5:9], expected[0][0, 0, 2, 10:14, 5:9])
        assert series.stats.pixels_read == 4 * 53 * 3                 # 4 whole rows


# -- coordinate systems (plan 4.5) --------------------------------------------------------

CAMERA_TURNED = {"dStgLgCT11": -1.0, "dStgLgCT12": 0.0, "dStgLgCT21": 0.0, "dStgLgCT22": -1.0}


@pytest.fixture
def metadata_reads(monkeypatch):
    """How often an ND2 image's ``_read_metadata`` ran."""
    count = [0]
    real = nd2_backend._Nd2Image._read_metadata

    def counting(self):
        count[0] += 1
        return real(self)
    monkeypatch.setattr(nd2_backend._Nd2Image, "_read_metadata", counting)
    return count


def test_positions_are_in_the_stage_system(tmp_path, metadata_reads):
    from micro_reader._spatial import check_invariants
    path = str(tmp_path / "a.nd2")
    write_nd2(path, _frames(3, channels=1), [("P", 3)], picture=CAMERA_TURNED)
    with micro_reader.open(path) as f:
        assert [s.id for s in f.coordinate_systems] == ["stage"]
        assert f.coordinate_systems[0].axes is None and f.coordinate_transformations == ()
        assert metadata_reads[0] == 0                       # declared without metadata
        check_invariants(f)                                 # I1-I3 on every image
        for image in f.images:
            assert image.metadata.position and image.metadata.coordinate_system == "stage"
        # every position of the file in one frame: the points 1 um apart in stage x
        # are 1 um apart along the image's x (turned by 180 degrees: -1)
        xs = [image.metadata.position["x"].value for image in f.images]
        assert xs[1] - xs[0] == pytest.approx(-1.0) and xs[2] - xs[1] == pytest.approx(-1.0)


@pytest.mark.parametrize("picture", [None, {"bCalibrated": False}, dict(CAMERA_TURNED, dStgLgCT22=0.0)],
                         ids=["no_camera_matrix", "uncalibrated", "singular_matrix"])
def test_files_without_a_position_declare_no_system(tmp_path, picture, metadata_reads):
    from micro_reader._spatial import check_invariants
    path = str(tmp_path / "a.nd2")
    if picture is not None and "dStgLgCT11" not in picture:
        picture = dict(CAMERA_TURNED, **picture)
    write_nd2(path, _frames(2, channels=1), [("P", 2)], picture=picture)
    with micro_reader.open(path) as f:
        assert f.coordinate_systems == ()
        assert metadata_reads[0] == 0
        check_invariants(f)
        for image in f.images:
            assert image.metadata.position == {} and image.metadata.coordinate_system is None


def test_legacy_nd2_declares_no_system(tmp_path):
    pytest.importorskip("imagecodecs")             # encodes the JPEG 2000 frames
    from _nd2_legacy_writer import write_legacy_nd2
    from micro_reader._spatial import check_invariants
    path = str(tmp_path / "old.nd2")
    write_legacy_nd2(path, np.zeros((2, 1, 8, 10), np.uint16), [("P", 2)])
    with micro_reader.open(path) as f:
        assert f.backend == "nd2" and f.coordinate_systems == ()
        check_invariants(f)
        assert all(image.metadata.position == {} and image.metadata.coordinate_system is None
                   for image in f.images)


def test_nd2_is_checked_strictly_by_the_session():
    import conftest
    assert "nd2" not in conftest.POSITIONS_WITHOUT_A_SYSTEM
