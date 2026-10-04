"""Native decoders against their reference path, bit for bit (PLAN.md,
"Native rule"), plus the error handling both must share."""
from __future__ import annotations

import numpy as np
import pytest

from micro_reader import CorruptFileError, _codecs

zstandard = pytest.importorskip("zstandard")          # test-only: makes the frames

needs_native = pytest.mark.skipif(not _codecs.native_available(),
                                  reason="native module not built")


def _payloads():
    rng = np.random.default_rng(0)
    ramp = (np.add.outer(np.arange(300), np.arange(400)) * 37 % 60000).astype("<u2")
    return {
        "noise_u8": rng.integers(0, 256, 100_000, dtype=np.uint8),
        "image_u16": ramp + rng.integers(0, 200, ramp.shape, dtype="<u2"),
        "constant": np.full(1 << 20, 7, np.uint8),
        "one_byte": np.array([42], np.uint8),
        "large_f32": rng.normal(size=(1024, 1024)).astype("<f4"),
    }


@pytest.mark.parametrize("level", [1, 3, 19])
@pytest.mark.parametrize("name", list(_payloads()))
@pytest.mark.parametrize("native", [pytest.param(True, marks=needs_native), False],
                         ids=["native", "reference"])
def test_zstd_decodes_exactly(name, level, native):
    data = _payloads()[name]
    frame = zstandard.ZstdCompressor(level=level).compress(data.tobytes())
    out = np.empty_like(data)
    _codecs.zstd_decompress_into(frame, out, native=native)
    np.testing.assert_array_equal(out, data)


@needs_native
def test_native_equals_reference_bit_for_bit():
    for data in _payloads().values():
        frame = zstandard.ZstdCompressor(level=5).compress(data.tobytes())
        a, b = np.empty_like(data), np.empty_like(data)
        _codecs.zstd_decompress_into(frame, a, native=True)
        _codecs.zstd_decompress_into(frame, b, native=False)
        assert a.tobytes() == b.tobytes()


@needs_native
def test_native_accepts_any_byte_buffer():
    data = _payloads()["image_u16"]
    frame = zstandard.ZstdCompressor().compress(data.tobytes())
    for wrapped in (frame, bytearray(frame), memoryview(frame), np.frombuffer(frame, np.uint8)):
        out = np.empty_like(data)
        _codecs.zstd_decompress_into(wrapped, out, native=True)
        np.testing.assert_array_equal(out, data)


@pytest.mark.parametrize("native", [pytest.param(True, marks=needs_native), False],
                         ids=["native", "reference"])
def test_corrupt_or_mis_sized_frames_raise(native):
    """What zstd can detect: a checksum mismatch (only for frames that carry
    one -- a flipped literal byte in a frame without a checksum just decodes
    to wrong values, for every decoder) and a truncated frame."""
    data = _payloads()["image_u16"]
    frame = bytearray(zstandard.ZstdCompressor(write_checksum=True).compress(data.tobytes()))
    frame[len(frame) // 2] ^= 0xFF
    with pytest.raises(CorruptFileError):
        _codecs.zstd_decompress_into(bytes(frame), np.empty_like(data), native=native)
    good = zstandard.ZstdCompressor().compress(data.tobytes())
    with pytest.raises(CorruptFileError):                       # truncated
        _codecs.zstd_decompress_into(good[:len(good) // 2], np.empty_like(data), native=native)
    with pytest.raises(CorruptFileError):                       # output too small
        _codecs.zstd_decompress_into(good, np.empty(data.size - 1, data.dtype), native=native)
    with pytest.raises(CorruptFileError):                       # output too large
        _codecs.zstd_decompress_into(good, np.empty(data.size + 1, data.dtype), native=native)


@needs_native
def test_native_refuses_read_only_output():
    data = _payloads()["noise_u8"]
    frame = zstandard.ZstdCompressor().compress(data.tobytes())
    out = np.empty_like(data)
    out.setflags(write=False)
    with pytest.raises((BufferError, ValueError)):
        _codecs.zstd_decompress_into(frame, out, native=True)


def test_hilo_unpack_known_answer():
    values = np.array([0x0102, 0xA0B0, 0xFFEE], "<u2")
    packed = np.array([0x02, 0xB0, 0xEE, 0x01, 0xA0, 0xFF], np.uint8)   # low bytes, high bytes
    out = np.empty_like(values)
    _codecs.hilo_unpack(packed, out)
    np.testing.assert_array_equal(out, values)


def test_pure_python_packbits_matches_native():
    """The last-resort PackBits decoder (no native module, no imagecodecs)
    against the native one, on runs, literals and the -128 no-op."""
    from micro_reader._codecs import _packbits_decode
    rng = np.random.default_rng(11)
    data = np.repeat(rng.integers(0, 256, 400), rng.integers(1, 9, 400)).astype(np.uint8)
    imagecodecs = pytest.importorskip("imagecodecs")
    packed = bytes(imagecodecs.packbits_encode(data.tobytes())) + b"\x80"   # trailing no-op
    assert _packbits_decode(packed, data.size) == data.tobytes()
    if _codecs.native_available():
        out = np.empty_like(data)
        assert _codecs.packbits_decode_into(packed, out, native=True) == data.size
        np.testing.assert_array_equal(out, data)
