"""Stepped reads (``image[::4, ::4]``) hold one piece of the span at a time, not the span (tenth
review, micro-reader N-07: ``image[::16]`` of a 6.8 GB MRC held 6.8 GB, in every format).

The limit is made small here, so that every way of splitting is taken on small files: runs of
positions, single positions, and down to rows; the values must equal numpy's own stepping."""
import itertools
import random

import numpy as np
import pytest

from micro_reader import _base
from micro_reader._base import Image


class Recording(Image):
    """An in-memory image that remembers the size of each block it is asked for."""

    def __init__(self, data):
        super().__init__(0, "zyx"[-data.ndim:], data.shape, data.dtype)
        self.data, self.blocks = data, []

    def _read_block(self, block):
        part = self.data[block].copy()
        self.blocks.append(part.nbytes)
        return part, part.size


DATA = np.arange(12 * 40 * 30, dtype="u2").reshape(12, 40, 30)
KEYS = [np.s_[::4], np.s_[::3, ::5, ::7], np.s_[1::2, 3:37:4, ::1], np.s_[:, ::2], np.s_[::11],
        np.s_[2, ::3, 1::4], np.s_[..., ::6], np.s_[5:6, ::39, ::29], np.s_[::1, ::1, ::2],
        np.s_[0:12:5, 7, ::3], np.s_[-3::2, -10:, ::-1 + 2]]


@pytest.mark.parametrize("limit", [64, 2400, 30 * 2 * 7, 1 << 30])
@pytest.mark.parametrize("key", KEYS)
def test_a_stepped_read_equals_numpy_and_reads_pieces_within_the_limit(monkeypatch, key, limit):
    monkeypatch.setattr(_base, "STEPPED_READ_BYTES", limit)
    image = Recording(DATA)
    out = image[key]
    np.testing.assert_array_equal(out, DATA[key])
    row = DATA.shape[-1] * DATA.itemsize                # a row is never split
    assert max(image.blocks, default=0) <= max(limit, row) or len(image.blocks) == 1


def test_random_stepped_reads(monkeypatch):
    monkeypatch.setattr(_base, "STEPPED_READ_BYTES", 500)
    rng = random.Random(7)
    image = Recording(DATA)
    for _ in range(300):
        key = tuple(slice(rng.randrange(n), rng.randrange(n + 1) if rng.random() < .5 else None,
                          rng.randint(1, 9)) for n in DATA.shape)
        np.testing.assert_array_equal(image[key], DATA[key])


def test_a_large_step_reads_only_the_selected_planes(monkeypatch):
    monkeypatch.setattr(_base, "STEPPED_READ_BYTES", 40 * 30 * 2)       # one plane fits
    image = Recording(DATA)
    image[::4]
    assert image.blocks == [40 * 30 * 2] * 3                            # planes 0, 4, 8 only


def test_a_small_stepped_read_and_an_unstepped_one_are_one_read():
    image = Recording(DATA)
    image[::2, ::2]
    image[1:5]
    assert len(image.blocks) == 2


@pytest.mark.parametrize("compression", [None, "zlib"])
def test_stepped_reads_of_a_tiff(tmp_path, monkeypatch, compression):
    tifffile = pytest.importorskip("tifffile")
    import micro_reader
    path = tmp_path / "v.tif"
    tifffile.imwrite(path, DATA, compression=compression, **({"tile": (16, 16)} if compression else {}))
    monkeypatch.setattr(_base, "STEPPED_READ_BYTES", 1000)
    with micro_reader.open(str(path), axes=None, rename={"sequence": "z"}) as f:
        image = f.images[0]
        for key in itertools.islice(KEYS, 8):
            np.testing.assert_array_equal(image[key], DATA[key])
