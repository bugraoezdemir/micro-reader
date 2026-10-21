"""Lossy JPEG (baseline and extended sequential, 8-bit) decodes to libjpeg-turbo's pixels, bit
for bit: the native decoder follows libjpeg's islow IDCT, fancy upsampling and YCbCr tables.
The reference is imagecodecs (libjpeg-turbo, as tifffile and Pillow use it)."""
from __future__ import annotations

import io

import numpy as np
import pytest

from micro_reader import _codecs

imagecodecs = pytest.importorskip("imagecodecs")
Image = pytest.importorskip("PIL.Image")

pytestmark = pytest.mark.skipif(not _codecs.native_available(), reason="native module not built")

SUBSAMPLINGS = ["444", "422", "420", "440", "411"]
SIZES = [(1, 1), (1, 17), (9, 1), (8, 8), (15, 33), (16, 16), (37, 61), (64, 130), (131, 77)]


def _image(h, w, c, seed):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w]
    base = 128 + 70 * np.sin(xx / rng.uniform(2, 30)) * np.cos(yy / rng.uniform(2, 30))
    planes = [np.clip(base + rng.normal(0, 25, (h, w)) + rng.uniform(-60, 60), 0, 255)
              for _ in range(c)]
    stack = np.stack(planes, -1).astype(np.uint8)
    return stack[..., 0] if c == 1 else stack


def _decode(data, colour="auto"):
    ref = imagecodecs.jpeg8_decode(data) if colour == "auto" else imagecodecs.jpeg8_decode(
        data, colorspace=colour, outcolorspace="rgb")
    out = np.zeros(ref.shape, np.uint8)
    h, w = ref.shape[:2]
    n = ref.shape[2] if ref.ndim == 3 else 1
    assert _codecs.jpeg_decode_into(data, out, colour=colour, native=True) == (w, h, n, 8)
    return out, ref


@pytest.mark.parametrize("sub", SUBSAMPLINGS)
@pytest.mark.parametrize("h, w", SIZES)
def test_ycbcr_subsampling(sub, h, w):
    img = _image(h, w, 3, h * 1000 + w)
    for level, optimize in [(90, False), (35, True), (100, False)]:
        data = imagecodecs.jpeg8_encode(img, level=level, subsampling=sub, optimize=optimize)
        out, ref = _decode(data)
        np.testing.assert_array_equal(out, ref, err_msg=f"{sub} {h}x{w} q{level}")


@pytest.mark.parametrize("h, w", SIZES)
def test_grey(h, w):
    img = _image(h, w, 1, h + w)
    out, ref = _decode(imagecodecs.jpeg8_encode(img, level=80))
    np.testing.assert_array_equal(out, ref)


@pytest.mark.parametrize("pil_sub", ["4:4:4", "4:2:2", "4:2:0"])
@pytest.mark.parametrize("blocks", [1, 3, 8])
def test_restart_markers(pil_sub, blocks):
    img = _image(75, 133, 3, blocks)
    buf = io.BytesIO()
    Image.fromarray(img, "RGB").save(buf, "JPEG", quality=85, subsampling=pil_sub,
                                     restart_marker_blocks=blocks)
    out, ref = _decode(buf.getvalue())
    np.testing.assert_array_equal(out, ref)


def test_adobe_rgb_and_colour_overrides():
    """Adobe RGB (no transform), and TIFF's photometric overriding the markers."""
    img = _image(40, 56, 3, 7)
    buf = io.BytesIO()
    Image.fromarray(img, "RGB").save(buf, "JPEG", quality=90, subsampling="4:4:4", keep_rgb=True)
    out, ref = _decode(buf.getvalue())
    np.testing.assert_array_equal(out, ref)
    ycc = imagecodecs.jpeg8_encode(img, level=90, subsampling="444")
    for colour in ("ycbcr", "rgb"):
        out, ref = _decode(ycc, colour)
        np.testing.assert_array_equal(out, ref, err_msg=colour)


def test_progressive_still_decodes():
    """Progressive JPEG is decoded by jpeg-decoder (not bit-exact with libjpeg-turbo)."""
    img = _image(48, 64, 3, 3)
    buf = io.BytesIO()
    Image.fromarray(img, "RGB").save(buf, "JPEG", quality=90, progressive=True)
    out, ref = _decode(buf.getvalue())
    assert np.abs(out.astype(int) - ref.astype(int)).max() <= 8
