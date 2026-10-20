"""Plain JPEG / PNG / BMP files (backend ``raster``): every PNG colour type,
depth, filter and interlacing; BMP palettes, RLE, bit fields and headers;
JPEG against Pillow.  PNG and BMP are checked against the written values,
with the native decoders and with the reference ones."""
from __future__ import annotations

import io

import numpy as np
import pytest

import micro_reader
from micro_reader import CorruptFileError, UnsupportedFormatError, _codecs
from micro_reader.backends import raster

from _raster_writer import write_bmp, write_png

RNG = np.random.default_rng(11)
H, W = 13, 21                                    # odd sizes: partial bytes, padding


@pytest.fixture(params=["native", "reference"])
def decoders(request, monkeypatch):
    if request.param == "reference":
        pytest.importorskip("imagecodecs")
        monkeypatch.setattr(_codecs, "_use_native", lambda native, name: False)
    elif not _codecs._use_native(None, "png_decode_into"):
        pytest.skip("native decoders not installed")
    return request.param


def _read(path):
    with micro_reader.open(path) as f:
        assert f.backend == "raster" and f.indices == [{"series": 0}]
        image = f.images[0]
        whole = image[...]
        part = image[2:9, 3:17]
        np.testing.assert_array_equal(part, whole[2:9, 3:17])
        return image, whole, f.format


def _colour_palette(n):
    pal = RNG.integers(0, 256, (n, 3)).astype("u1")
    pal[:, 0] = np.arange(n) * 7 % 256                     # some channels differ: colour
    pal[:, 1] = 255 - pal[:, 0]
    return pal


# -- PNG ------------------------------------------------------------------------------

@pytest.mark.parametrize("bits,colour,samples", [
    (1, 0, 1), (2, 0, 1), (4, 0, 1), (8, 0, 1), (16, 0, 1),
    (8, 2, 3), (16, 2, 3), (8, 4, 2), (16, 4, 2), (8, 6, 4), (16, 6, 4),
])
@pytest.mark.parametrize("filters", [(0,), (1, 2, 3, 4)], ids=["none", "all"])
def test_png_values_as_stored(tmp_path, decoders, bits, colour, samples, filters):
    shape = (H, W) if samples == 1 else (H, W, samples)
    values = RNG.integers(0, 1 << bits, shape).astype("u2" if bits == 16 else "u1")
    path = str(tmp_path / "v.png")
    write_png(path, values, bits=bits, colour=colour, filters=filters)
    image, whole, fmt = _read(path)
    assert fmt == "png" and image.axes == ("yx" if samples == 1 else "yxs")
    assert image.palette is None
    np.testing.assert_array_equal(whole, values)


@pytest.mark.parametrize("bits", [1, 2, 4, 8])
@pytest.mark.parametrize("filters", [(0,), (1, 2, 3, 4)], ids=["none", "all"])
def test_png_colour_palette_gives_indices(tmp_path, decoders, bits, filters):
    n = 1 << bits
    pal = _colour_palette(n)
    idx = RNG.integers(0, n, (H, W)).astype("u1")
    path = str(tmp_path / "p.png")
    write_png(path, idx, bits=bits, colour=3, plte=pal, filters=filters)
    image, whole, _ = _read(path)
    np.testing.assert_array_equal(whole, idx)
    np.testing.assert_array_equal(image.palette, pal)


def test_png_grey_palette_is_applied(tmp_path, decoders):
    grey = np.arange(16, dtype="u1")[::-1] * 17                  # an inverted ramp
    pal = np.stack([grey] * 3, axis=1)
    idx = RNG.integers(0, 16, (H, W)).astype("u1")
    path = str(tmp_path / "g.png")
    write_png(path, idx, bits=4, colour=3, plte=pal)
    image, whole, _ = _read(path)
    assert image.palette is None
    np.testing.assert_array_equal(whole, grey[idx])


def test_png_transparency_goes_into_the_palette(tmp_path, decoders):
    pal = _colour_palette(4)
    path = str(tmp_path / "t.png")
    write_png(path, np.zeros((H, W), "u1"), bits=2, colour=3, plte=pal, trns=[0, 128])
    image, _, _ = _read(path)
    assert image.palette.shape == (4, 4)
    np.testing.assert_array_equal(image.palette[:, 3], [0, 128, 255, 255])


@pytest.mark.parametrize("bits,colour,samples", [(4, 0, 1), (2, 3, 1), (8, 2, 3), (16, 0, 1)])
def test_png_interlaced(tmp_path, decoders, bits, colour, samples):
    shape = (H, W) if samples == 1 else (H, W, samples)
    values = RNG.integers(0, 1 << bits, shape).astype("u2" if bits == 16 else "u1")
    pal = _colour_palette(1 << bits) if colour == 3 else None
    path = str(tmp_path / "i.png")
    write_png(path, values, bits=bits, colour=colour, plte=pal, filters=(1, 4), interlace=True)
    if decoders == "reference" and (bits < 8 or colour == 3):
        with pytest.raises(UnsupportedFormatError, match="interlaced"):
            micro_reader.open(path).images[0][...]
        return
    _, whole, _ = _read(path)
    np.testing.assert_array_equal(whole, values)


def test_png_header_beyond_the_first_megabyte(tmp_path):
    pal = _colour_palette(4)
    idx = RNG.integers(0, 4, (H, W)).astype("u1")
    path = str(tmp_path / "big.png")
    write_png(path, idx, bits=2, colour=3, plte=pal, text_before=raster.HEAD_BYTES + 10)
    image, whole, _ = _read(path)
    np.testing.assert_array_equal(image.palette, pal)
    np.testing.assert_array_equal(whole, idx)


# -- BMP ------------------------------------------------------------------------------

@pytest.mark.parametrize("bpp", [1, 2, 4, 8])
@pytest.mark.parametrize("top_down", [False, True])
def test_bmp_palette_indices(tmp_path, decoders, bpp, top_down):
    pal = _colour_palette(1 << bpp)
    idx = RNG.integers(0, 1 << bpp, (H, W)).astype("u1")
    path = str(tmp_path / "p.bmp")
    write_bmp(path, idx, bpp=bpp, palette=pal, top_down=top_down)
    image, whole, fmt = _read(path)
    assert fmt == "bmp" and image.axes == "yx"
    np.testing.assert_array_equal(whole, idx)
    np.testing.assert_array_equal(image.palette, pal)


@pytest.mark.parametrize("bpp,compression", [(8, 1), (4, 2)])
def test_bmp_rle(tmp_path, bpp, compression):
    n = 1 << bpp
    pal = _colour_palette(n)
    idx = RNG.integers(0, n, (H, W)).astype("u1")
    idx[3:6, 2:15] = 5                                        # runs as well as noise
    path = str(tmp_path / "rle.bmp")
    write_bmp(path, idx, bpp=bpp, palette=pal, compression=compression)
    _, whole, _ = _read(path)
    np.testing.assert_array_equal(whole, idx)


def test_bmp_grey_palette_and_os2_header(tmp_path):
    ramp = np.arange(256, dtype="u1")
    pal = np.stack([ramp] * 3, axis=1)
    values = RNG.integers(0, 256, (H, W)).astype("u1")
    path = str(tmp_path / "os2.bmp")
    write_bmp(path, values, bpp=8, palette=pal, header=12)
    image, whole, _ = _read(path)
    assert image.palette is None
    np.testing.assert_array_equal(whole, values)


def test_bmp_24_and_32_bit(tmp_path):
    rgb = RNG.integers(0, 256, (H, W, 3)).astype("u1")
    path = str(tmp_path / "c24.bmp")
    write_bmp(path, rgb, bpp=24, top_down=True)
    _, whole, _ = _read(path)
    np.testing.assert_array_equal(whole, rgb)
    path = str(tmp_path / "c32.bmp")                          # BI_RGB 32: the 4th byte is unused
    write_bmp(path, rgb, bpp=32, masks=[0xFF0000, 0xFF00, 0xFF])
    image, whole, _ = _read(path)
    assert image.axes == "yxs" and whole.shape == (H, W, 3)
    np.testing.assert_array_equal(whole, rgb)


@pytest.mark.parametrize("bpp,masks,compression,header", [
    (16, [0x7C00, 0x3E0, 0x1F], 0, 40),                        # 5-5-5
    (16, [0xF800, 0x7E0, 0x1F], 3, 40),                        # 5-6-5 bit fields
    (32, [0xFF0000, 0xFF00, 0xFF, 0xFF000000], 3, 108),        # v4 with alpha
])
def test_bmp_bit_fields_scale_to_8_bits(tmp_path, bpp, masks, compression, header):
    widths = [bin(m).count("1") for m in masks]
    fields = np.stack([RNG.integers(0, 1 << n, (H, W)) for n in widths], axis=-1)
    path = str(tmp_path / "bf.bmp")
    write_bmp(path, fields, bpp=bpp, masks=masks, compression=compression, header=header)
    _, whole, _ = _read(path)
    top = np.array([(1 << n) - 1 for n in widths])
    np.testing.assert_array_equal(whole, (fields * 255 + top // 2) // top)


def test_bmp_holding_a_png(tmp_path):
    values = RNG.integers(0, 256, (H, W, 3)).astype("u1")
    png = tmp_path / "inner.png"
    write_png(str(png), values, colour=2)
    path = str(tmp_path / "wrap.bmp")
    write_bmp(path, values, bpp=0, compression=5, payload=png.read_bytes())
    _, whole, fmt = _read(path)
    assert fmt == "bmp"
    np.testing.assert_array_equal(whole, values)


# -- JPEG -----------------------------------------------------------------------------

@pytest.mark.parametrize("mode", ["L", "RGB"])
@pytest.mark.parametrize("progressive", [False, True])
def test_jpeg_matches_pillow(tmp_path, decoders, mode, progressive):
    Image = pytest.importorskip("PIL.Image")
    shape = (40, 56) if mode == "L" else (40, 56, 3)
    smooth = np.add.outer(np.arange(40), np.arange(56)) * 2
    data = (smooth[..., None] if mode == "RGB" else smooth) + RNG.integers(0, 20, shape)
    path = str(tmp_path / "p.jpg")
    Image.fromarray(data.astype("u1"), mode).save(path, quality=90, progressive=progressive)
    _, whole, fmt = _read(path)
    assert fmt == "jpeg"
    ref = np.asarray(Image.open(path))
    assert whole.shape == ref.shape
    assert np.abs(whole.astype(int) - ref).max() <= 3       # lossy decoders may differ slightly


def test_cmyk_jpeg_is_refused_by_name(tmp_path):
    Image = pytest.importorskip("PIL.Image")
    path = str(tmp_path / "cmyk.jpg")
    Image.new("CMYK", (8, 8)).save(path)
    with pytest.raises(UnsupportedFormatError, match="CMYK"):
        micro_reader.open(path)


# -- the backend ----------------------------------------------------------------------

def test_decoded_once_and_read_unit_is_the_image(tmp_path):
    values = RNG.integers(0, 256, (H, W)).astype("u1")
    path = str(tmp_path / "v.png")
    write_png(path, values)
    with micro_reader.open(path) as f:
        image = f.images[0]
        assert image.read_unit == (H, W)
        image[0:2, 0:2]
        assert image.stats.pixels_read == H * W
        image[5:7, 5:9]
        assert image.stats.pixels_read == H * W               # cached
        assert image.as_axes("canonical").axes == "tczyx"


@pytest.mark.parametrize("cut", [30])     # inside the pixel data (a missing IEND alone:
def test_truncated_png_is_reported(tmp_path, cut):  # libpng tolerates it, png-rs does not)
    path = tmp_path / "t.png"
    write_png(str(path), RNG.integers(0, 256, (H, W)).astype("u1"))
    path.write_bytes(path.read_bytes()[:-cut])
    with pytest.raises(CorruptFileError, match="t.png"):
        with micro_reader.open(str(path)) as f:             # at open or at the first read
            f.images[0][...]


def test_other_files_are_not_claimed():
    assert not raster.sniff(b"BM\x00\x00" + b"\x00" * 20, "x.bmp")   # "BM" alone is not enough
    assert not raster.sniff(b"GIF89a" + b"\x00" * 20, "x.gif")
