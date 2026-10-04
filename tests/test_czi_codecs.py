"""Every CZI compression code, on synthetic files (tests/_czi_writer.py) with
payloads encoded by imagecodecs (test-only) -- rule: support every codec an
accepted format can contain, whether or not our real data has it."""
from __future__ import annotations

import numpy as np
import pytest

import micro_reader
from micro_reader import UnsupportedFormatError, _codecs

from _czi_writer import raw_payload, write_czi

imagecodecs = pytest.importorskip("imagecodecs")
zstandard = pytest.importorskip("zstandard")

H, W = 56, 72
RNG = np.random.default_rng(0)


def _gray(dtype):
    ramp = np.add.outer(np.arange(H), np.arange(W))
    if np.dtype(dtype) == np.uint8:
        return ((ramp * 3 + RNG.integers(0, 8, (H, W))) % 256).astype("u1")
    return ((ramp * 431 + RNG.integers(0, 300, (H, W))) % 65000).astype("<u2")


def _rgb():
    g = _gray("u1")
    return np.stack([g, g[::-1], 255 - g], -1)


def _varint(n: int) -> bytes:
    out = bytearray()
    while True:
        byte = n & 0x7F
        n >>= 7
        out.append(byte | (0x80 if n else 0))
        if not n:
            return bytes(out)


def chunked_payload(stored: bytes, chunk: int, method: int, hilo: bool, sizes_form: str) -> bytes:
    """Encode *stored* (raw subblock bytes) as CZI compression 7, per Zeiss'
    "Chunked compression" description."""
    data = stored
    if hilo:
        a = np.frombuffer(stored, np.uint8)
        data = (a[0::2].tobytes() + a[1::2].tobytes())            # low bytes, then high bytes
    pieces = [data[i:i + chunk] for i in range(0, len(data), chunk)]
    if method == 0:
        comp = [zstandard.ZstdCompressor().compress(p) for p in pieces]
    else:
        comp = [imagecodecs.lz4_encode(p, header=False) for p in pieces]
    sizes = [len(p) for p in pieces]
    dec = {"single": [sizes[0]] if len(set(sizes)) == 1 else None,
           "last": [sizes[0], sizes[-1]], "all": sizes}[sizes_form]
    header = b""
    for kind, payload in ((1, b"".join(_varint(len(c)) for c in comp)),
                          (2, bytes([method])), (3, b"".join(_varint(s) for s in dec)),
                          (4, bytes([1 if hilo else 0]))):
        header += _varint(kind) + _varint(len(payload)) + payload
    return header + _varint(0) + b"".join(comp)


def _case(name):
    """-> (tiles for write_czi, expected pixels as micro-reader returns them,
    tolerance vs expected)."""
    gray8, gray16, rgb = _gray("u1"), _gray("u2"), _rgb()
    if name == "jpeg_gray":
        enc = imagecodecs.jpeg8_encode(gray8, level=92)
        return enc, "u1", 1, 1, imagecodecs.jpeg8_decode(enc), 3
    if name == "jpeg_rgb":
        enc = imagecodecs.jpeg8_encode(rgb, level=92)
        return enc, "u1", 3, 1, imagecodecs.jpeg8_decode(enc), 3
    if name == "lzw_gray16":
        return imagecodecs.lzw_encode(raw_payload(gray16)), "u2", 1, 2, gray16, 0
    if name == "lzw_rgb":
        return imagecodecs.lzw_encode(raw_payload(rgb)), "u1", 3, 2, rgb, 0
    if name == "jpeg_lossless_16":
        return imagecodecs.ljpeg_encode(gray16), "u2", 1, 3, gray16, 0
    if name == "jpeg_lossless_8":
        return imagecodecs.ljpeg_encode(gray8), "u1", 1, 3, gray8, 0
    if name == "jpegxr_gray16":
        return imagecodecs.jpegxr_encode(gray16, level=1.0), "u2", 1, 4, gray16, 0
    if name == "jpegxr_rgb":
        return imagecodecs.jpegxr_encode(rgb, level=1.0), "u1", 3, 4, rgb, 0
    if name == "jpegxl_gray16":
        return imagecodecs.jpegxl_encode(gray16, lossless=True), "u2", 1, 8, gray16, 0
    if name == "jpegxl_rgb":
        return imagecodecs.jpegxl_encode(rgb, lossless=True), "u1", 3, 8, rgb, 0
    if name.startswith("chunked"):
        _, method, hilo, form = name.split("_")
        stored = raw_payload(gray16)
        payload = chunked_payload(stored, 1000, 0 if method == "zstd" else 1, hilo == "hilo", form)
        return payload, "u2", 1, 7, gray16, 0
    raise KeyError(name)


CASES = ["jpeg_gray", "jpeg_rgb", "lzw_gray16", "lzw_rgb", "jpeg_lossless_16",
         "jpeg_lossless_8", "jpegxr_gray16", "jpegxr_rgb", "jpegxl_gray16", "jpegxl_rgb",
         "chunked_zstd_plain_last", "chunked_zstd_hilo_last", "chunked_lz4_plain_all",
         "chunked_lz4_hilo_last"]


def _write(tmp_path, name):
    payload, dtype, samples, code, expected, tol = _case(name)
    path = str(tmp_path / f"{name}.czi")
    write_czi(path, [dict(plane={"C": 0}, x=5, y=7, height=H, width=W, dtype=dtype,
                          samples=samples, compression=code, payload=payload)])
    return path, expected, tol, code


@pytest.fixture(params=["native", "reference"])
def decoders(request, monkeypatch):
    if request.param == "native" and not _codecs.native_available():
        pytest.skip("native module not built")
    if request.param == "reference":
        monkeypatch.setattr(_codecs, "_native", None)
    return request.param


@pytest.mark.parametrize("name", CASES)
def test_every_compression_decodes(tmp_path, name, decoders):
    path, expected, tol, code = _write(tmp_path, name)
    with micro_reader.open(path) as img:
        scene = img.images[0]
        assert {s.compression for s in img._level0} == {code}
        got = scene[0, 0, 0]
        assert got.shape == expected.shape and got.dtype == expected.dtype
        diff = np.abs(got.astype(np.int64) - expected.astype(np.int64))
        assert diff.max() <= tol, f"{name}: max difference {diff.max()} > {tol}"
        # a region inside the tile (drawn from the cached decoded tile)
        np.testing.assert_allclose(scene[0, 0, 0, 10:30, 20:50], got[10:30, 20:50])


@pytest.mark.parametrize("name", ["jpegxr_gray16", "jpegxr_rgb"])
def test_jpegxr_agrees_with_libczi(tmp_path, name):
    """pylibCZIrw (libCZI, Zeiss) decodes JPEG XR itself: an independent check
    of the codec and of the colour order."""
    pyczi = pytest.importorskip("pylibCZIrw.czi")
    path, expected, _, _ = _write(tmp_path, name)
    with pyczi.open_czi(path) as czi:
        ref = czi.read(plane={"C": 0})
    ref = ref[..., 0] if ref.shape[-1] == 1 else ref[..., ::-1]
    with micro_reader.open(path) as img:
        np.testing.assert_array_equal(img.images[0][0, 0, 0], ref)
    np.testing.assert_array_equal(ref, expected)


def test_vendor_raw_is_refused_clearly(tmp_path):
    path = str(tmp_path / "raw150.czi")
    write_czi(path, [dict(plane={"C": 0}, x=0, y=0, height=H, width=W, dtype="u2",
                          compression=150, payload=raw_payload(_gray("u2")))])
    with pytest.raises(UnsupportedFormatError, match="camera RAW"):
        micro_reader.open(path)
