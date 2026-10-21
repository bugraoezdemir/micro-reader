"""Decoders: the native package when it is installed, a reference path otherwise.

Native: ``micro_reader_codecs`` -- the separate package micro-reader-codecs
(Rust), a direct dependency -- decoding straight into the caller's buffer
without the GIL.  Reference: an established independent decoder -- Python
3.14's ``compression.zstd``, else ``zstandard``, else ``imagecodecs``.  It is
used only where the native package is missing (e.g. a development checkout)
or lacks a codec (JPEG XR), and by the tests, which compare native and
reference bit for bit.  ``MICRO_READER_NO_NATIVE=1`` forces the reference path.
"""
from __future__ import annotations

import os

import numpy as np

from ._errors import CorruptFileError, MicroReaderError

if os.environ.get("MICRO_READER_NO_NATIVE"):
    _native = None
else:
    try:
        import micro_reader_codecs as _native
    except ImportError:              # not installed (e.g. a source checkout)
        _native = None


def native_available() -> bool:
    return _native is not None


def _reference_zstd(data) -> bytes:
    try:
        from compression import zstd                  # Python >= 3.14
        return zstd.decompress(bytes(data))
    except ImportError:
        pass
    try:
        import zstandard
        return zstandard.ZstdDecompressor().decompress(bytes(data))
    except ImportError:
        pass
    try:
        import imagecodecs
        return bytes(imagecodecs.zstd_decode(bytes(data)))
    except ImportError:
        pass
    raise MicroReaderError(
        "zstd-compressed data needs micro-reader's native module (install a wheel), "
        "Python >= 3.14, or the 'zstandard' package")


def zstd_decompress_into(data, out: np.ndarray, *, native: bool | None = None) -> None:
    """Decompress one zstd frame into the C-contiguous array *out*, which
    must be exactly the decompressed size."""
    view = memoryview(out).cast("B")
    use_native = _native is not None if native is None else native
    if use_native:
        if _native is None:
            raise MicroReaderError("micro-reader's native module is not available")
        try:
            written = _native.zstd_decompress_into(data, view)
        except ValueError as exc:
            raise CorruptFileError(f"corrupt zstd data: {exc}") from exc
    else:
        try:
            raw = _reference_zstd(data)
        except Exception as exc:                       # noqa: BLE001 - decoder specific
            if isinstance(exc, MicroReaderError):
                raise
            raise CorruptFileError(f"corrupt zstd data: {exc}") from exc
        written = len(raw)
        if written == len(view):
            view[:] = raw
    if written != len(view):
        raise CorruptFileError(f"zstd data decodes to {written} bytes, expected {len(view)}")


def _imagecodecs(what: str):
    try:
        import imagecodecs
    except ImportError as exc:
        raise MicroReaderError(
            f"{what} needs micro-reader's native module or the [codecs] extra (imagecodecs)"
        ) from exc
    return imagecodecs


def _use_native(native, name: str) -> bool:
    if native is None:
        return _native is not None and hasattr(_native, name)
    if native and (_native is None or not hasattr(_native, name)):
        raise MicroReaderError(f"micro-reader's native module has no {name}")
    return native


def _fill(out: np.ndarray, decoded, what: str) -> None:
    decoded = np.asarray(decoded)
    view = memoryview(out).cast("B")
    raw = decoded.astype(decoded.dtype.newbyteorder("<"), copy=False).tobytes()
    if len(raw) != len(view):
        raise CorruptFileError(f"{what} data decodes to {len(raw)} bytes, expected {len(view)}")
    view[:] = raw


def lz4_block_decompress_into(data, out: np.ndarray, *, native: bool | None = None) -> None:
    """One LZ4 block (no frame header) -> *out*, exactly its size."""
    if _use_native(native, "lz4_block_decompress_into"):
        try:
            written = _native.lz4_block_decompress_into(data, memoryview(out).cast("B"))
        except ValueError as exc:
            raise CorruptFileError(f"corrupt LZ4 data: {exc}") from exc
    else:
        codecs = _imagecodecs("LZ4 data")
        try:
            raw = codecs.lz4_decode(bytes(data), header=False, out=out.nbytes)
        except Exception as exc:                        # noqa: BLE001
            raise CorruptFileError(f"corrupt LZ4 data: {exc}") from exc
        _fill(out.reshape(-1).view(np.uint8), np.frombuffer(raw, np.uint8), "LZ4")
        return
    if written != out.nbytes:
        raise CorruptFileError(f"LZ4 data decodes to {written} bytes, expected {out.nbytes}")


def lzf_available() -> bool:
    """Whether LZF can be decoded here: the native module, or imagecodecs."""
    if _use_native(None, "lzf_decompress_into"):
        return True
    try:
        import imagecodecs
    except ImportError:
        return False
    return bool(getattr(getattr(imagecodecs, "LZF", None), "available", False))


def lzf_decompress_into(data, out: np.ndarray, *, native: bool | None = None) -> None:
    """LZF data (liblzf's format, no header; h5py's HDF5 filter 32000) -> *out*,
    exactly its size."""
    if _use_native(native, "lzf_decompress_into"):
        try:
            written = _native.lzf_decompress_into(data, memoryview(out).cast("B"))
        except ValueError as exc:
            raise CorruptFileError(f"corrupt LZF data: {exc}") from exc
    else:
        codecs = _imagecodecs("LZF data")
        try:                                            # straight into *out*: no copies
            written = len(codecs.lzf_decode(data, header=False, out=out.reshape(-1).view(np.uint8)))
        except Exception as exc:                        # noqa: BLE001
            raise CorruptFileError(f"corrupt LZF data: {exc}") from exc
    if written != out.nbytes:
        raise CorruptFileError(f"LZF data decodes to {written} bytes, expected {out.nbytes}")


def _check_written(written: int, out: np.ndarray, what: str, partial: bool) -> int:
    if written > out.nbytes or (written < out.nbytes and not partial):
        raise CorruptFileError(f"{what} data decodes to {written} bytes, expected {out.nbytes}")
    return written


def _fill_partial(out: np.ndarray, raw, what: str, partial: bool) -> int:
    """A reference decoder's bytes -> *out*.  With *partial*, a shorter
    result fills the start of *out* (TIFF readers accept short segments)
    and a longer one is cut; returns the bytes written."""
    try:
        # the decoder's own buffer, not a copy of it: a copy is one more decoded
        # block alive at once, beyond the memory the caller reserved
        raw = memoryview(raw).cast("B")
    except (TypeError, ValueError):                    # not a contiguous buffer
        raw = memoryview(bytes(raw)).cast("B")
    view = memoryview(out).cast("B")
    if not partial:
        _check_written(len(raw), out, what, partial)
    n = min(len(raw), len(view))
    view[:n] = raw[:n]
    return n


def lzw_decode_into(data, out: np.ndarray, *, partial: bool = False,
                    native: bool | None = None) -> int:
    """TIFF-style LZW -> *out*, which it must fill unless *partial*;
    returns the bytes written."""
    if _use_native(native, "lzw_decode_into"):
        try:
            written = _native.lzw_decode_into(data, memoryview(out).cast("B"))
        except ValueError as exc:
            raise CorruptFileError(f"corrupt LZW data: {exc}") from exc
        return _check_written(written, out, "LZW", partial)
    codecs = _imagecodecs("LZW data")
    try:
        raw = codecs.lzw_decode(bytes(data), out=out.nbytes)
    except Exception as exc:                            # noqa: BLE001
        raise CorruptFileError(f"corrupt LZW data: {exc}") from exc
    return _fill_partial(out, raw, "LZW", partial)


def zlib_decompress_into(data, out: np.ndarray, *, partial: bool = False,
                         native: bool | None = None) -> int:
    """A zlib stream (TIFF / HDF5 deflate) -> *out*; returns the bytes
    written.  Reference: the standard library."""
    if _use_native(native, "zlib_decompress_into"):
        try:
            written = _native.zlib_decompress_into(data, memoryview(out).cast("B"))
        except ValueError as exc:
            raise CorruptFileError(f"corrupt deflate data: {exc}") from exc
        return _check_written(written, out, "deflate", partial)
    import zlib
    try:
        # at most one byte more than *out* holds: a stream claiming more (damaged
        # data) is caught without inflating all of it into memory
        # zlib reads any bytes-like input in place: no copy of the stored bytes
        raw = zlib.decompressobj().decompress(data, out.nbytes + 1)
    except zlib.error as exc:
        raise CorruptFileError(f"corrupt deflate data: {exc}") from exc
    return _fill_partial(out, raw, "deflate", partial)


def packbits_decode_into(data, out: np.ndarray, *, partial: bool = False,
                         native: bool | None = None) -> int:
    """PackBits -> *out*; returns the bytes written.  Reference: imagecodecs,
    else a pure-Python decoder (slow; only without the native module)."""
    if _use_native(native, "packbits_decode_into"):
        written = _native.packbits_decode_into(data, memoryview(out).cast("B"))
        return _check_written(written, out, "PackBits", partial)
    try:
        import imagecodecs
        decoded = imagecodecs.packbits_decode(bytes(data))
    except ImportError:
        decoded = _packbits_decode(bytes(data), out.nbytes)
    return _fill_partial(out, decoded, "PackBits", partial)


def _packbits_decode(data: bytes, limit: int) -> bytes:
    """PackBits in pure Python: a header byte n, then n + 1 literal bytes
    (n >= 0) or the next byte 1 - n times (n < 0; -128 is a no-op); stops
    at *limit* bytes, as TIFF readers do."""
    out = bytearray()
    i = 0
    while i < len(data) and len(out) < limit:
        n = data[i] - 256 if data[i] > 127 else data[i]
        i += 1
        if n >= 0:
            out += data[i:i + n + 1]
            i += n + 1
        elif n != -128 and i < len(data):
            out += bytes([data[i]]) * (1 - n)
            i += 1
    return bytes(out[:limit])


def lzma_decompress_into(data, out: np.ndarray, *, partial: bool = False) -> int:
    """An LZMA / xz stream -> *out*, with the standard library (rare in
    TIFF; not worth a native decoder)."""
    import lzma
    try:
        raw = lzma.decompress(bytes(data))
    except lzma.LZMAError as exc:
        raise CorruptFileError(f"corrupt LZMA data: {exc}") from exc
    return _fill_partial(out, raw, "LZMA", partial)


def _image_codec(name: str, what: str, data, out, native, reference: str):
    """The image codecs' common shape: native straight into *out*, else the
    imagecodecs *reference* function's array copied in."""
    if _use_native(native, name):
        try:
            return getattr(_native, name)(data, memoryview(out).cast("B"))
        except ValueError as exc:
            raise CorruptFileError(f"corrupt {what} data: {exc}") from exc
    codecs = _imagecodecs(f"{what} data")
    try:
        image = np.asarray(getattr(codecs, reference)(bytes(data)))
    except Exception as exc:                            # noqa: BLE001
        raise CorruptFileError(f"corrupt {what} data: {exc}") from exc
    _fill(out, image, what)
    return (image.shape[1], image.shape[0], image.shape[2] if image.ndim == 3 else 1,
            image.dtype.itemsize * 8)


def png_decode_into(data, out: np.ndarray, *, native: bool | None = None):
    """A PNG (8 or 16 bit, no palette) -> *out*.  Returns (width, height,
    samples, bits)."""
    return _image_codec("png_decode_into", "PNG", data, out, native, "png_decode")


def webp_decode_into(data, out: np.ndarray, *, native: bool | None = None):
    """A WebP -> *out* as 8-bit R, G, B (A).  Returns (width, height, samples, 8)."""
    return _image_codec("webp_decode_into", "WebP", data, out, native, "webp_decode")


def jpeg2k_decode_into(data, out: np.ndarray, *, native: bool | None = None):
    """A JPEG 2000 codestream or JP2 -> *out*, components interleaved.
    Returns (width, height, components, bits).  The native decoder is
    OpenJPEG, like imagecodecs', so lossy data decodes identically."""
    return _image_codec("jpeg2k_decode_into", "JPEG 2000", data, out, native,
                        "jpeg2k_decode")[:4]


def jpeg2k_decode_area_into(data, out: np.ndarray, x0: int, y0: int, x1: int, y1: int, *,
                            native: bool | None = None):
    """The rectangle x0 <= x < x1, y0 <= y < y1 of a JPEG 2000 codestream or
    JP2 -> *out*, components interleaved.  Native: OpenJPEG decodes only the
    code-blocks covering it.  Reference: the whole image decoded, then cut
    (memory of the whole image).  Returns (width, height, components, bits)
    of the rectangle."""
    if _use_native(native, "jpeg2k_decode_area_into"):
        try:
            return _native.jpeg2k_decode_area_into(data, memoryview(out).cast("B"),
                                                   x0, y0, x1, y1)[:4]
        except ValueError as exc:
            raise CorruptFileError(f"corrupt JPEG 2000 data: {exc}") from exc
    codecs = _imagecodecs("JPEG 2000 data")
    try:
        image = np.asarray(codecs.jpeg2k_decode(bytes(data)))[y0:y1, x0:x1]
    except Exception as exc:                            # noqa: BLE001
        raise CorruptFileError(f"corrupt JPEG 2000 data: {exc}") from exc
    _fill(out, image, "JPEG 2000")
    return (image.shape[1], image.shape[0], image.shape[2] if image.ndim == 3 else 1,
            image.dtype.itemsize * 8)


def jxl_decode_area_into(data, out: np.ndarray, x0: int, y0: int, x1: int, y1: int, *,
                         native: bool | None = None):
    """The rectangle x0 <= x < x1, y0 <= y < y1 of a JPEG XL codestream ->
    *out*, like ``jxl_decode_into``.  Native: jxl-oxide renders only the
    groups covering it.  Reference: the whole image decoded, then cut.
    Returns (width, height, channels, bits) of the rectangle."""
    if _use_native(native, "jxl_decode_area_into"):
        try:
            return _native.jxl_decode_area_into(data, memoryview(out).cast("B"), x0, y0, x1, y1)
        except ValueError as exc:
            raise CorruptFileError(f"corrupt JPEG XL data: {exc}") from exc
    codecs = _imagecodecs("JPEG XL data")
    try:
        image = np.asarray(codecs.jpegxl_decode(bytes(data)))[y0:y1, x0:x1]
    except Exception as exc:                            # noqa: BLE001
        raise CorruptFileError(f"corrupt JPEG XL data: {exc}") from exc
    _fill(out, image, "JPEG XL")
    return (image.shape[1], image.shape[0], image.shape[2] if image.ndim == 3 else 1,
            image.dtype.itemsize * 8)


def _is_lossless_jpeg(data) -> bool:
    return b"\xff\xc3" in bytes(data[:4096])            # SOF3 marker


def _is_12bit_lossy_jpeg(data) -> bool:
    """A lossy JPEG with 12-bit samples (SOF0/1/2 precision 12), which the
    native decoder does not handle."""
    head = bytes(data[:4096])
    for marker in (b"\xff\xc0", b"\xff\xc1", b"\xff\xc2"):
        at = head.find(marker)
        if 0 <= at and at + 4 < len(head):
            return head[at + 4] == 12
    return False


def jpeg_decode_into(data, out: np.ndarray, *, colour: str = "auto",
                     native: bool | None = None):
    """A JPEG (baseline, progressive or lossless) -> *out* as (rows, cols[,
    samples]) little-endian, colour in R, G, B order.  Returns (width,
    height, samples, bits).  *colour*: "auto" (the JPEG's own markers),
    "ycbcr", or "rgb" (stored without a colour transform: TIFF Photometric
    RGB).  Baseline and extended 8-bit JPEG decode to libjpeg-turbo's pixels
    (imagecodecs, tifffile, Pillow), bit for bit; progressive JPEG may differ
    from them by a few grey levels (allowed by the JPEG standard); lossless is
    exact.  12-bit lossy JPEG goes to imagecodecs."""
    if _use_native(native, "jpeg_decode_into") and not _is_12bit_lossy_jpeg(data):
        try:
            return _native.jpeg_decode_into(data, memoryview(out).cast("B"), colour)
        except ValueError as exc:
            raise CorruptFileError(f"corrupt JPEG data: {exc}") from exc
    codecs = _imagecodecs("JPEG data")
    try:
        if _is_lossless_jpeg(data):
            image = codecs.ljpeg_decode(bytes(data))
        elif colour == "auto":
            image = codecs.jpeg8_decode(bytes(data))
        else:
            image = codecs.jpeg8_decode(bytes(data), colorspace=colour, outcolorspace="rgb")
    except Exception as exc:                            # noqa: BLE001
        raise CorruptFileError(f"corrupt JPEG data: {exc}") from exc
    _fill(out, image, "JPEG")
    return (image.shape[1], image.shape[0], image.shape[2] if image.ndim == 3 else 1,
            image.dtype.itemsize * 8)


def jxl_decode_into(data, out: np.ndarray, *, native: bool | None = None):
    """A JPEG XL codestream -> *out* (colour R, G, B).  Returns (width,
    height, channels, bits)."""
    if _use_native(native, "jxl_decode_into"):
        try:
            return _native.jxl_decode_into(data, memoryview(out).cast("B"))
        except ValueError as exc:
            raise CorruptFileError(f"corrupt JPEG XL data: {exc}") from exc
    codecs = _imagecodecs("JPEG XL data")
    try:
        image = codecs.jpegxl_decode(bytes(data))
    except Exception as exc:                            # noqa: BLE001
        raise CorruptFileError(f"corrupt JPEG XL data: {exc}") from exc
    _fill(out, image, "JPEG XL")
    return (image.shape[1], image.shape[0], image.shape[2] if image.ndim == 3 else 1,
            image.dtype.itemsize * 8)


def jpegxr_decode_into(data, out: np.ndarray):
    """A JPEG XR codestream -> *out* (colour R, G, B).  Through imagecodecs
    until the native jxrlib binding is decided (it needs libclang to build)."""
    codecs = _imagecodecs("JPEG XR data")
    try:
        image = codecs.jpegxr_decode(bytes(data))
    except Exception as exc:                            # noqa: BLE001
        raise CorruptFileError(f"corrupt JPEG XR data: {exc}") from exc
    _fill(out, image, "JPEG XR")


def eer_decode_into(data, out: np.ndarray, skipbits: int, horzbits: int, vertbits: int,
                    superres: int = 0, *, native: bool | None = None) -> int:
    """An EER frame (Electron Event Representation, Falcon cameras) ->
    electron counts in *out* (rows, cols; bool, uint8, uint16 or uint32),
    super-resolved when *superres* > 0.  Returns the number of electrons.
    Reference: imagecodecs."""
    rows, cols = out.shape[:2]
    if _use_native(native, "eer_decode_into"):
        try:
            return _native.eer_decode_into(data, memoryview(out).cast("B"), rows, cols, skipbits,
                                           horzbits, vertbits, superres, out.dtype.itemsize)
        except ValueError as exc:
            raise CorruptFileError(f"corrupt EER data: {exc}") from exc
    codecs = _imagecodecs("EER data")
    try:
        image = codecs.eer_decode(bytes(data), (rows, cols), skipbits, horzbits, vertbits,
                                  superres=superres)
    except Exception as exc:                            # noqa: BLE001
        raise CorruptFileError(f"corrupt EER data: {exc}") from exc
    out.reshape(rows, cols)[...] = image
    return int(np.count_nonzero(image)) if image.dtype == bool else int(image.sum())


#: every byte with its bits in reverse order (TIFF FillOrder 2)
REVERSED_BITS = np.array([int(f"{b:08b}"[::-1], 2) for b in range(256)], np.uint8)


def reverse_bits(data) -> bytes:
    """Bytes with the bit order of each reversed (TIFF FillOrder 2)."""
    return REVERSED_BITS[np.frombuffer(data, np.uint8)].tobytes()


def unpack_bits_into(packed, out: np.ndarray, bits: int, runlen: int, *,
                     native: bool | None = None) -> int:
    """Unsigned samples packed at *bits* bits (most significant bit first,
    every *runlen* samples -- a TIFF row -- starting on a byte boundary) ->
    *out* (C-contiguous, 1-, 2- or 4-byte unsigned or bool).  Returns the
    samples written (whole runs).  Reference: numpy bit arithmetic."""
    item = out.dtype.itemsize
    if _use_native(native, "unpack_bits_into"):
        try:
            return _native.unpack_bits_into(packed, memoryview(out).cast("B"), bits, runlen, item)
        except ValueError as exc:
            raise CorruptFileError(f"packed samples: {exc}") from exc
    run_bytes = (runlen * bits + 7) // 8
    data = np.frombuffer(packed, np.uint8)
    runs = min(data.size // run_bytes, out.size // runlen)
    b = np.unpackbits(data[:runs * run_bytes].reshape(runs, run_bytes), axis=1)
    b = b[:, :runlen * bits].reshape(runs, runlen, bits)
    weights = np.left_shift(np.uint64(1), np.arange(bits - 1, -1, -1, dtype=np.uint64))
    values = (b.astype(np.uint64) * weights).sum(axis=-1, dtype=np.uint64)
    out.reshape(-1)[:runs * runlen] = values.reshape(-1).astype(out.dtype)
    return runs * runlen


def hilo_unpack(packed: np.ndarray, out: np.ndarray) -> None:
    """Undo CZI's hi-lo byte packing of 16-bit data: all low bytes first,
    then all high bytes -> interleaved (little-endian) samples in *out*."""
    n = packed.size // 2
    raw = out.view(np.uint8).reshape(-1)
    raw[0::2] = packed[:n]
    raw[1::2] = packed[n:2 * n]
