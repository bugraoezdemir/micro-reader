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


# --- reading only the selected positions (micro-reader-napari, 2026-11-20) -----------------
# A step at least the backend's read unit puts every selected position in its own unit, so
# reading the positions alone never decodes more than reading runs of them (before: [::64] of
# a 6.9 GB MRC read the whole file, 640-row runs, to keep 1 row in 64).

PLANE = 40 * 30 * 2


def _planes(image):
    return sum(image.blocks) / PLANE


@pytest.mark.parametrize("exact", [False, True])
@pytest.mark.parametrize("limit", [64, 2400, PLANE * 4, 1 << 30])
@pytest.mark.parametrize("key", KEYS)
def test_values_are_the_same_with_and_without_exact_reads(monkeypatch, key, limit, exact):
    monkeypatch.setattr(_base, "STEPPED_READ_BYTES", limit)
    image = Recording(DATA)
    image.exact_reads = exact
    np.testing.assert_array_equal(image[key], DATA[key])


def test_an_exact_image_reads_only_the_selected_planes(monkeypatch):
    monkeypatch.setattr(_base, "STEPPED_READ_BYTES", PLANE * 4)           # four planes fit
    monkeypatch.setattr(_base, "SKIP_READ_MIN_STEP", 2)
    image = Recording(DATA)
    image.exact_reads = True
    np.testing.assert_array_equal(image[::4], DATA[::4])
    assert _planes(image) == 3                                          # 0, 4, 8: not 0-11


def test_an_image_that_does_not_say_keeps_reading_runs(monkeypatch):
    monkeypatch.setattr(_base, "STEPPED_READ_BYTES", PLANE * 4)
    monkeypatch.setattr(_base, "SKIP_READ_MIN_STEP", 2)
    image = Recording(DATA)                                             # read_unit None, not exact
    image[::4]
    assert _planes(image) == 12                                         # runs 0-3, 4-7, 8-11


@pytest.mark.parametrize("unit, step, selected_only", [
    (3, 6, True), (3, 7, True), (1, 2, True), (1, 4, True),
    (3, 3, False),                     # a step of one unit skips no unit: the runs win (NDPI)
    (3, 5, False), (3, 2, False), (2, 3, False)])
def test_a_declared_read_unit_decides(monkeypatch, unit, step, selected_only):
    monkeypatch.setattr(_base, "STEPPED_READ_BYTES", PLANE * 6)
    monkeypatch.setattr(_base, "SKIP_READ_MIN_STEP", 2)
    image = Recording(DATA)
    image.read_unit = (unit, 40, 30)
    np.testing.assert_array_equal(image[::step], DATA[::step])
    selected = len(range(0, 12, step))
    assert (_planes(image) == selected) is selected_only


@pytest.mark.parametrize("step, selected_only", [(3, False), (4, True), (6, True)])
def test_the_default_smallest_step(monkeypatch, step, selected_only):
    monkeypatch.setattr(_base, "STEPPED_READ_BYTES", PLANE * 6)
    image = Recording(DATA)
    image.exact_reads = True
    np.testing.assert_array_equal(image[::step], DATA[::step])
    assert (_planes(image) == len(range(0, 12, step))) is selected_only


def test_a_small_step_on_an_exact_image_still_reads_runs(monkeypatch):
    """Below SKIP_READ_MIN_STEP one call per position costs more than it saves."""
    monkeypatch.setattr(_base, "STEPPED_READ_BYTES", PLANE * 4)
    monkeypatch.setattr(_base, "SKIP_READ_MIN_STEP", 3)
    image = Recording(DATA)
    image.exact_reads = True
    image[::2]
    assert _planes(image) == 12


@pytest.mark.parametrize("compression, exact", [(None, True), ("zlib", False)])
def test_tiff_says_whether_it_reads_exactly_also_through_axes(tmp_path, compression, exact):
    tifffile = pytest.importorskip("tifffile")
    import micro_reader
    path = tmp_path / "e.tif"
    tifffile.imwrite(path, DATA, compression=compression)
    with micro_reader.open(str(path), rename={"sequence": "z"}) as f:
        native = f.images[0]
        assert native.exact_reads is exact
        image = native.as_axes("canonical")                             # through AxesImage
        assert image.axes == "tczyx" and image.exact_reads is exact
        np.testing.assert_array_equal(image[0, 0, ::4, ::3, ::2], DATA[::4, ::3, ::2])
