"""The C decoders micro-reader-codecs 0.0.3 links (codecs/build.rs): libjpeg-turbo for
JPEG and jxrlib for JPEG XR.  Their pixels must be imagecodecs' -- the same libraries,
called the same way -- so imagecodecs stays optional for both.

NDPI's real slides (not writable synthetically: tifffile does not write NDPI's layout)
are compared when the data folder holds them."""
from __future__ import annotations

import functools
import os
from pathlib import Path

import numpy as np
import pytest

from micro_reader import _codecs
from micro_reader._errors import CorruptFileError

imagecodecs = pytest.importorskip("imagecodecs")
native = _codecs._native

needs_jxrlib = pytest.mark.skipif(not getattr(native, "HAS_JXRLIB", False),
                                  reason="micro-reader-codecs without jxrlib")
needs_turbo = pytest.mark.skipif(not getattr(native, "HAS_LIBJPEG_TURBO", False),
                                 reason="micro-reader-codecs without libjpeg-turbo")

DATA = Path(os.environ.get("MICROSCOPY_DATA") or r"C:\Users\oezdemir\Desktop\data\microscopy")


def _image(shape, dtype, seed=0):
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[:shape[0], :shape[1]]
    if np.dtype(dtype).kind == "f":
        base = np.sin(x / 7.0) * np.cos(y / 5.0) + rng.normal(0, 0.05, shape[:2])
        image = base.astype(dtype)
    else:
        top = np.iinfo(dtype).max
        base = (x * 3 + y * 5) % (top // 2 + 1) + rng.integers(0, max(2, top // 50), shape[:2])
        image = base.astype(dtype)
    if len(shape) == 3:
        image = np.stack([np.roll(image, 7 * k, axis=1) for k in range(shape[2])], axis=-1)
    return np.ascontiguousarray(image)


# --- JPEG XR ----------------------------------------------------------------------------

@needs_jxrlib
@pytest.mark.parametrize("shape, dtype", [
    ((67, 129), np.uint8), ((67, 129, 3), np.uint8), ((67, 129, 4), np.uint8),
    ((40, 70), np.uint16), ((40, 70, 3), np.uint16),
    ((33, 65), np.float32), ((33, 65, 3), np.float32), ((33, 65), np.float16),
])
@pytest.mark.parametrize("level", [1.0, 0.8])               # lossless, lossy
def test_jpegxr_is_imagecodecs(shape, dtype, level):
    image = _image(shape, dtype)
    if np.dtype(dtype).kind == "f" and level == 1.0:
        level = 0.95                                        # float: lossy only
    data = imagecodecs.jpegxr_encode(image, level=level)
    want = imagecodecs.jpegxr_decode(data)
    got = np.empty(want.shape, want.dtype)
    _codecs.jpegxr_decode_into(data, got, native=True)
    np.testing.assert_array_equal(got, want)


@needs_jxrlib
def test_jpegxr_bgr_is_returned_rgb():
    """CZI stores JPEG XR as BGR; like imagecodecs, the decode gives R, G, B."""
    image = _image((24, 40, 3), np.uint8)
    data = imagecodecs.jpegxr_encode(image, level=1.0)
    got = np.empty_like(image)
    _codecs.jpegxr_decode_into(data, got, native=True)
    np.testing.assert_array_equal(got, imagecodecs.jpegxr_decode(data))


@needs_jxrlib
def test_jpegxr_wrong_size_and_garbage_are_corrupt():
    data = imagecodecs.jpegxr_encode(_image((16, 16), np.uint8), level=1.0)
    with pytest.raises(CorruptFileError, match="decodes to 256 bytes"):
        _codecs.jpegxr_decode_into(data, np.empty((16, 17), np.uint8), native=True)
    with pytest.raises(CorruptFileError, match="JPEG XR"):
        _codecs.jpegxr_decode_into(b"\x00" * 64, np.empty((16, 16), np.uint8), native=True)


def test_without_any_jpegxr_decoder_the_message_names_the_way_out(monkeypatch):
    import builtins
    real = builtins.__import__

    def no_imagecodecs(name, *args, **kwargs):
        if name == "imagecodecs":
            raise ImportError(name)
        return real(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_imagecodecs)
    with pytest.raises(Exception, match=r'micro-reader-codecs>=0\.0\.3.*micro-reader\[codecs\]'):
        _codecs.jpegxr_decode_into(b"", np.empty(4, np.uint8), native=False)


# --- JPEG -------------------------------------------------------------------------------

@needs_turbo
@pytest.mark.parametrize("shape", [(61, 83), (61, 83, 3)])
@pytest.mark.parametrize("colour", ["auto", "ycbcr", "rgb"])
def test_8bit_jpeg_is_imagecodecs(shape, colour):
    image = _image(shape, np.uint8)
    kwargs = {} if colour == "auto" or len(shape) == 2 else {"colorspace": colour}
    data = imagecodecs.jpeg8_encode(image, level=85, **kwargs)
    want = (imagecodecs.jpeg8_decode(data) if colour == "auto" or len(shape) == 2
            else imagecodecs.jpeg8_decode(data, colorspace=colour, outcolorspace="rgb"))
    got = np.empty(want.shape, np.uint8)
    w, h, s, bits = _codecs.jpeg_decode_into(data, got, colour=colour, native=True)
    assert (w, h, s, bits) == (shape[1], shape[0], 1 if len(shape) == 2 else 3, 8)
    np.testing.assert_array_equal(got, want)


@needs_turbo
def test_12bit_jpeg_is_native_now():
    image = (_image((50, 70), np.uint16) & 0x0FFF).astype(np.uint16)
    data = imagecodecs.jpeg8_encode(image, level=90, bitspersample=12)
    want = imagecodecs.jpeg8_decode(data)
    got = np.empty(want.shape, np.uint16)
    assert _codecs.jpeg_decode_into(data, got, native=True)[3] == 16
    np.testing.assert_array_equal(got, want)


@needs_turbo
@pytest.mark.parametrize("bits", [8, 12, 16])
def test_lossless_jpeg_is_imagecodecs(bits):
    dtype = np.uint8 if bits == 8 else np.uint16
    image = _image((40, 52), dtype)
    if bits == 12:
        image &= 0x0FFF
    data = imagecodecs.ljpeg_encode(image, bitspersample=bits)
    got = np.empty(image.shape, dtype)
    _codecs.jpeg_decode_into(data, got, native=True)
    np.testing.assert_array_equal(got, image)


@needs_turbo
def test_a_truncated_jpeg_decodes_as_before():
    """libjpeg-turbo would repair it (grey rows, a warning): such data goes on to
    micro-reader's own decoder, as before 0.0.3 -- which decodes what is there
    without an error (review point R2-09, open: libjpeg-turbo can now report the
    premature end)."""
    image = (np.arange(64 * 64) % 251).reshape(64, 64).astype(np.uint8)
    data = imagecodecs.jpeg8_encode(image, level=90)
    got = np.empty((64, 64), np.uint8)
    _codecs.jpeg_decode_into(data[: len(data) // 2], got, native=True)
    assert not (got[-1] == 128).all()                       # not libjpeg's grey repair


# --- NDPI on real slides ----------------------------------------------------------------

NDPI = [p for p in [
    DATA / "formats/tiff/slides/openslide/CMU-1.ndpi",
    DATA / "formats/tiff/slides/openslide/Hamamatsu-2.ndpi",
    DATA / "formats/tiff/slides/ome-ndpi-manuel/test3-DAPI 2 (387) .ndpi",
] if p.exists()]


@needs_turbo
@pytest.mark.skipif(not NDPI, reason="no NDPI slides in the data folder")
@pytest.mark.parametrize("path", NDPI, ids=lambda p: p.name)
def test_ndpi_native_route_is_tifffiles(path, monkeypatch):
    """NDPI stripes decode natively (libjpeg-turbo) with tifffile + imagecodecs' pixels."""
    import micro_reader as mr
    from micro_reader.backends import tiff

    def regions(image):
        h, w = image.shape[-2:]
        return [(slice(h * k // 4, h * k // 4 + 700), slice(w * (3 - k) // 4, w * (3 - k) // 4 + 900))
                for k in (1, 2)]

    def read(with_turbo):
        monkeypatch.setattr(_codecs, "has_libjpeg_turbo", lambda: with_turbo)
        with mr.open(str(path)) as f:
            image = f.images[0]
            backend = image
            while getattr(backend, "source", None) is not None:   # axes / RGB wrappers
                backend = backend.source
            assert (tiff.own_codec(backend._key) == "jpeg") is with_turbo
            fixed = {a: 0 for a in image.axes if a not in "yxc"}
            return [image.read(y=ys, x=xs, **fixed) for ys, xs in regions(image)]

    for got, want in zip(read(True), read(False)):
        np.testing.assert_array_equal(got, want)
