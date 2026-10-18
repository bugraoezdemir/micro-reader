"""JPEG 2000 with subsampled chroma (sYCC 4:2:2: Aperio's TIFF compression 33003, real-data round
row C): the native decoder upsamples and converts to RGB as OpenJPEG and imagecodecs do, bit for
bit, for the whole codestream and for areas -- also areas starting on an odd column or row."""
from __future__ import annotations

import os

import numpy as np
import pytest

from micro_reader import _codecs

imagecodecs = pytest.importorskip("imagecodecs")          # the reference path

TILE = os.path.join(os.path.dirname(__file__), "data", "aperio_33003_ycbcr422_tile.j2k")
needs_native = pytest.mark.skipif(not _codecs.native_available(), reason="native module not built")


def _tile():
    with open(TILE, "rb") as fh:
        return fh.read()


@pytest.mark.parametrize("native", [pytest.param(True, marks=needs_native), False],
                         ids=["native", "reference"])
def test_subsampled_ycbcr_decodes_to_rgb(native):
    data = _tile()
    expected = np.asarray(imagecodecs.jpeg2k_decode(data))
    assert expected.shape == (256, 256, 3)
    out = np.empty_like(expected)
    w, h, n, bits = _codecs.jpeg2k_decode_into(data, out, native=native)
    assert (w, h, n, bits) == (256, 256, 3, 8)
    np.testing.assert_array_equal(out, expected)


@needs_native
@pytest.mark.parametrize("area", [(0, 0, 256, 256), (1, 0, 37, 20), (0, 1, 50, 51), (3, 5, 200, 131),
                                  (17, 33, 18, 34), (128, 7, 256, 256), (255, 255, 256, 256)])
def test_subsampled_areas_equal_the_whole(area):
    """An area is the same rectangle of the whole image: an odd first column / row has no
    chroma sample of its own, so the native decoder decodes from even coordinates and crops."""
    data = _tile()
    whole = np.asarray(imagecodecs.jpeg2k_decode(data))
    x0, y0, x1, y1 = area
    out = np.empty((y1 - y0, x1 - x0, 3), whole.dtype)
    _codecs.jpeg2k_decode_area_into(data, out, x0, y0, x1, y1, native=True)
    np.testing.assert_array_equal(out, whole[y0:y1, x0:x1])


@needs_native
@pytest.mark.parametrize("shape,dtype", [((70, 90, 3), "u1"), ((64, 40), "u2")])
def test_full_resolution_components_unchanged(shape, dtype):
    """RGB (with the codestream's own colour transform) and grey stay as they were, also in an
    area with an odd origin."""
    img = np.random.default_rng(1).integers(0, np.iinfo(dtype).max, shape, dtype=dtype)
    data = imagecodecs.jpeg2k_encode(img, level=0)                      # lossless
    out = np.empty_like(img)
    _codecs.jpeg2k_decode_into(data, out, native=True)
    np.testing.assert_array_equal(out, img)
    area = np.empty((20, 23) + img.shape[2:], img.dtype)
    _codecs.jpeg2k_decode_area_into(data, area, 5, 3, 28, 23, native=True)
    np.testing.assert_array_equal(area, img[3:23, 5:28])
