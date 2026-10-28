"""Legacy (JPEG 2000) ND2 on synthetic files (``_nd2_legacy_writer``, modelled
on OME's real samples): against the written frames and the nd2 library.
Real files: ``harness/conformance_nd2.py`` (5 / 5 of OME's aryeh samples
identical to nd2, and to Bio-Formats)."""
from __future__ import annotations

import os
import threading

import numpy as np
import pytest

pytest.importorskip("imagecodecs")             # encodes the JPEG 2000 frames

import micro_reader  # noqa: E402
from micro_reader import CorruptFileError, _codecs  # noqa: E402
from micro_reader.backends import nd2 as nd2_backend  # noqa: E402

from _nd2_legacy_writer import write_legacy_nd2  # noqa: E402


def _native(*args, **kwargs):
    """``micro_reader.open`` with RGB samples kept as an ``s`` axis: these tests
    check the backend's own layout (open's default folds them into channels)."""
    return micro_reader.open(*args, **{"samples": "keep", **kwargs})

RNG = np.random.default_rng(3)

# name -> (loops outermost first, frames (n, c, y, x[, s]), extra stored frames)
CASES = {
    "time_positions_channels": ([("T", 3), ("P", 2)],
                                RNG.integers(0, 4000, (6, 2, 20, 24)).astype("u2"), 0),
    "rgb_time": ([("T", 4)], RNG.integers(0, 255, (4, 1, 16, 20, 3)).astype("u1"), 0),
    "positions_z_extra_frame": ([("P", 2), ("Z", 3)],
                                RNG.integers(0, 255, (6, 1, 12, 14)).astype("u1"), 1),
    "single_frame": ([], RNG.integers(0, 4000, (1, 3, 18, 10)).astype("u2"), 0),
}


def _write(tmp_path, name):
    loops, frames, extra = CASES[name]
    path = str(tmp_path / f"{name}.nd2")
    write_legacy_nd2(path, frames, loops, extra_frames=extra)
    return path, loops, frames


def _expected(loops, frames):
    """Per position: t c z y x [s], from frames in loop order."""
    sizes = dict(loops)
    shape = [sizes.get(a, 1) for a, _ in loops] or []
    t, p, z = sizes.get("T", 1), sizes.get("P", 1), sizes.get("Z", 1)
    out = []
    for pos in range(p):
        series = np.empty((t, frames.shape[1], z) + frames.shape[2:], frames.dtype)
        for ti in range(t):
            for zi in range(z):
                index = {"T": ti, "P": pos, "Z": zi}
                number = 0
                for axis, count in loops:
                    number = number * count + index[axis]
                series[ti, :, zi] = frames[number]
        out.append(series)
    del shape
    return out


def _keys(shape, n=10, seed=5):
    rng = np.random.default_rng(seed)
    keys = [(Ellipsis,)]
    for _ in range(n):
        key = []
        for size in shape:
            a = int(rng.integers(0, size))
            key.append(slice(a, int(rng.integers(a + 1, size + 1))))
        keys.append(tuple(key))
    return keys


@pytest.fixture(params=[1, 4], ids=["serial", "pool"])
def decode_threads(request):
    old = micro_reader.decode_threads()
    micro_reader.set_decode_threads(request.param)
    yield
    micro_reader.set_decode_threads(old)


@pytest.mark.parametrize("name", list(CASES))
def test_matches_written_frames(tmp_path, name, decode_threads):
    path, loops, frames = _write(tmp_path, name)
    expected = _expected(loops, frames)
    with _native(path) as img:
        assert (img.backend, img.format) == ("nd2", "nd2-legacy")
        assert len(img.images) == len(expected)
        for series, ref in zip(img.images, expected):
            if series.axes == "tczyxs":
                assert ref.ndim == 6
            assert series.shape == ref.shape and series.dtype == ref.dtype
            for key in _keys(ref.shape):
                np.testing.assert_array_equal(series[key], ref[key], err_msg=f"{name} {key}")


@pytest.mark.parametrize("name", list(CASES))
def test_agrees_with_nd2(tmp_path, name):
    nd2 = pytest.importorskip("nd2")
    path, _, _ = _write(tmp_path, name)
    with nd2.ND2File(path) as f:
        ref, sizes = f.asarray(), dict(f.sizes)
    dims = list(sizes)
    for axis in "TPCZYXS":
        if axis not in dims:
            ref, dims = ref[..., None], dims + [axis]
    ref = ref.transpose([dims.index(a) for a in "TPCZYXS"])
    with _native(path) as img:
        assert len(img.images) == ref.shape[1]
        for p, series in enumerate(img.images):
            got = series[...]
            np.testing.assert_array_equal(got if got.ndim == 6 else got[..., None], ref[:, p])


def test_reference_decoder_path(tmp_path, monkeypatch):
    """Without the native module, imagecodecs decodes the same frames."""
    path, loops, frames = _write(tmp_path, "time_positions_channels")
    monkeypatch.setattr(_codecs, "_native", None)
    with _native(path) as img:
        for series, ref in zip(img.images, _expected(loops, frames)):
            np.testing.assert_array_equal(series[...], ref)


def test_jp2_image_is_not_claimed(tmp_path):
    """A plain JPEG 2000 image starts like a legacy ND2: not an ND2."""
    import imagecodecs
    path = str(tmp_path / "picture.jp2")
    with open(path, "wb") as fh:
        fh.write(imagecodecs.jpeg2k_encode(np.zeros((8, 8), np.uint8), codecformat="JP2"))
    with open(path, "rb") as fh:
        assert not nd2_backend.sniff(fh.read(1024), path)


def test_truncated_file_is_reported(tmp_path):
    path, _, _ = _write(tmp_path, "rgb_time")
    with open(path, "r+b") as fh:
        fh.truncate(os.path.getsize(path) - 30)
    with pytest.raises(CorruptFileError, match="box map"):
        _native(path)


def test_frames_are_decoded_once(tmp_path):
    path, _, _ = _write(tmp_path, "rgb_time")
    with _native(path) as img:
        series = img.images[0]
        series[1, 0, 0, 0:4, 0:4]
        first = series.stats.pixels_read
        series[1, 0, 0, 5:9, 5:9]                        # same frame: cached
        assert series.stats.pixels_read == first == 16 * 20 * 3


def test_concurrent_reads(tmp_path, decode_threads):
    path, loops, frames = _write(tmp_path, "time_positions_channels")
    expected = _expected(loops, frames)
    errors = []
    with _native(path) as img:
        def work(seed):
            for series, ref in zip(img.images, expected):
                for key in _keys(ref.shape, n=5, seed=seed):
                    if not np.array_equal(series[key], ref[key]):
                        errors.append(key)
        threads = [threading.Thread(target=work, args=(s,)) for s in range(5)]
        for th in threads:
            th.start()
        for th in threads:
            th.join()
    assert not errors
