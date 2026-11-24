"""RGB ND2: samples are read in R, G, B order from the order the file declares
(Workplan MR30).

Nikon colour cameras store each pixel's samples as blue, green, red, and say so in the
camera block of each channel's sample setting (component names Blue, Green, Red, colours
0xFF0000, 0x00FF00, 0x0000FF as 0x00BBGGRR). micro-reader used to pass them on as
stored and name them R, G, B, so red and blue were swapped. The worst outcome is names
that do not match the data, so these tests check that the channel named "<name> R"
holds the stored red sample on every read path.
"""
from __future__ import annotations

import os
import warnings

import numpy as np
import pytest

import micro_reader
from micro_reader import _memory
from micro_reader.backends import nd2 as nd2_backend

from _nd2_writer import write_nd2

BGR = [("Blue", 0xFF0000), ("Green", 0x00FF00), ("Red", 0x0000FF)]
RGB = [("Red", 0x0000FF), ("Green", 0x00FF00), ("Blue", 0xFF0000)]
BGRA = BGR + [("Alpha", 0)]


def _stored(n=3, h=21, w=17, channels=2, samples=3, seed=0):
    """(frames, h, w, channels, samples) uint8, each sample of its own value range so
    a swap cannot pass by chance."""
    rng = np.random.default_rng(seed)
    data = rng.integers(0, 60, size=(n, h, w, channels, samples), dtype=np.uint8)
    return data + (np.arange(samples, dtype=np.uint8) * 64)


def _write(tmp_path, frames, sample_names, **kwargs):
    path = str(tmp_path / "rgb.nd2")
    write_nd2(path, frames, [("Z", frames.shape[0])], sample_names=sample_names, **kwargs)
    return path


def _as_tczyxs(frames):
    """(z, y, x, c, s) frames -> t c z y x s."""
    return frames.transpose(3, 0, 1, 2, 4)[None]


KEYS = [
    np.s_[...],
    np.s_[0, :, 1, 3:15, 2:11, :],           # a region of one plane
    np.s_[..., 0],                           # one sample (R)
    np.s_[..., 1:3],                         # G and B
    np.s_[:, 1, :, ::3, ::2, ::2],           # steps, one channel
    np.s_[0, 0, 2],                          # a whole plane of one channel
]


@pytest.mark.parametrize("compression", [None, "lossless"])
@pytest.mark.parametrize("pad", [0, 5])
def test_bgr_is_read_as_rgb(tmp_path, compression, pad):
    frames = _stored()
    path = _write(tmp_path, frames, BGR, compression=compression, pad_bytes=pad)
    want = _as_tczyxs(frames)[..., ::-1]                   # stored B, G, R -> R, G, B
    with warnings.catch_warnings():
        warnings.simplefilter("error")                     # a declared order: no warning
        f = micro_reader.open(path, samples="keep")
    with f:
        image = f.images[0]
        assert image.sample_order == ("B", "G", "R")
        for key in KEYS:
            np.testing.assert_array_equal(np.asarray(image[key]), want[key], err_msg=str(key))


def test_streamed_lossless_frames_are_reordered_too(tmp_path, monkeypatch):
    frames = _stored()
    path = _write(tmp_path, frames, BGR, compression="lossless")
    monkeypatch.setattr(_memory, "should_stream", lambda nbytes: True)
    want = _as_tczyxs(frames)[..., ::-1]
    with micro_reader.open(path, samples="keep") as f:
        for key in KEYS:
            np.testing.assert_array_equal(np.asarray(f.images[0][key]), want[key])


def test_rgb_declared_stays_as_stored(tmp_path):
    frames = _stored()
    path = _write(tmp_path, frames, RGB)
    with micro_reader.open(path, samples="keep") as f:
        assert f.images[0].sample_order == ("R", "G", "B")
        np.testing.assert_array_equal(np.asarray(f.images[0][...]), _as_tczyxs(frames))


def test_bgra_is_read_as_rgba(tmp_path):
    frames = _stored(samples=4)
    path = _write(tmp_path, frames, BGRA)
    want = _as_tczyxs(frames)[..., [2, 1, 0, 3]]
    with micro_reader.open(path, samples="keep") as f:
        assert f.images[0].sample_order == ("B", "G", "R", "A")
        for key in KEYS:
            np.testing.assert_array_equal(np.asarray(f.images[0][key]), want[key])


@pytest.mark.parametrize("names", [
    None,                                                  # no camera block
    [("Blue", 0xFF0000), ("Green", 0x00FF00), ("Red", 0x00FF00)],   # colour disagrees
    [("Blue", 0xFF0000), ("Blue", 0xFF0000), ("Red", 0x0000FF)],    # not one of each
    [("Cyan", 0xFFFF00), ("Green", 0x00FF00), ("Red", 0x0000FF)],   # unknown name
    BGR[:2],                                               # fewer components than samples
], ids=["none", "colour", "twice", "unknown", "short"])
def test_an_order_the_file_does_not_settle_warns_and_keeps_the_stored_order(tmp_path, names):
    frames = _stored()
    path = _write(tmp_path, frames, names)
    with pytest.warns(UserWarning, match="does not state the order of its 3 colour samples"):
        f = micro_reader.open(path, samples="keep")
    with f:
        assert f.images[0].sample_order is None
        np.testing.assert_array_equal(np.asarray(f.images[0][...]), _as_tczyxs(frames))


def test_one_sample_has_no_order_and_no_warning(tmp_path):
    frames = _stored(samples=1)
    path = _write(tmp_path, frames, None)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with micro_reader.open(path) as f:
            assert f.images[0].sample_order is None


def test_channel_names_match_their_data(tmp_path):
    """samples="channels" (open's default): "<name> R" holds the stored red sample
    (stored position 2), within each channel and never across channels."""
    frames = _stored()
    path = _write(tmp_path, frames, BGR)
    stored = _as_tczyxs(frames)                            # t c z y x s, stored order
    with micro_reader.open(path) as f:
        image = f.images[0]
        names = [c.name for c in image.metadata.channels]
        data = np.asarray(image[...])
    assert names == ["Ch0 R", "Ch0 G", "Ch0 B", "Ch1 R", "Ch1 G", "Ch1 B"]
    for c in range(2):
        for k, letter in enumerate("RGB"):
            np.testing.assert_array_equal(data[:, 3 * c + k], stored[:, c, ..., 2 - k],
                                          err_msg=f"Ch{c} {letter}")


# -- real files: the nd2 package passes the samples through as stored ---------------

DATA = os.path.join(os.environ.get("MICROSCOPY_DATA",
                                   os.path.expanduser("~/Desktop/data/microscopy")),
                    "formats", "nd2", "github-tlambert03-nd2")


@pytest.mark.parametrize("name", ["dims_rgb.nd2", "dims_rgb_c2x64y64.nd2",
                                  "dims_rgb_t3p2c2z3x64y64.nd2"])
def test_real_rgb_nd2_against_the_stored_samples(name):
    """The nd2 package's samples reversed (B, G, R -> R, G, B), on every position."""
    nd2lib = pytest.importorskip("nd2")
    path = os.path.join(DATA, name)
    if not os.path.exists(path):
        pytest.skip(f"sample not found: {path} (set MICROSCOPY_DATA)")
    with nd2lib.ND2File(path) as f:
        sizes = dict(f.sizes)
        raw = np.array(f.asarray(), copy=True)
    axes = list(sizes)
    with micro_reader.open(path, samples="keep") as f:
        for p, image in enumerate(f.images):
            assert image.sample_order == ("B", "G", "R")
            a, ax = (np.take(raw, p, axis=axes.index("P")), [x for x in axes if x != "P"]) \
                if "P" in axes else (raw, list(axes))
            for k in "TCZ":
                if k not in ax:
                    a, ax = a[None], [k] + ax
            stored = a.transpose([ax.index(k) for k in "TCZYXS"])
            for key in (np.s_[...], np.s_[..., 0], np.s_[..., 1:3],
                        np.s_[0, :, 0, 5:40, 3:33, :], np.s_[..., ::3, ::2, :]):
                np.testing.assert_array_equal(np.asarray(image[key]),
                                              stored[..., ::-1][key], err_msg=f"{name} {key}")
