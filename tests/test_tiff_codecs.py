"""TIFF compressions decoded by micro-reader's own codecs (compression audit,
2026-10-01): every codec and predictor, in strips, tiles, colour and big-endian
files written by tifffile + imagecodecs, compared with tifffile's reading.
Also the reference path (no native module) and the fallback to tifffile."""
from __future__ import annotations

import sys

import numpy as np
import pytest

tifffile = pytest.importorskip("tifffile")
pytest.importorskip("imagecodecs")             # writes the files and is the reference

import micro_reader  # noqa: E402
from micro_reader import UnsupportedFormatError, _codecs  # noqa: E402


def _data(shape, dtype="u2", seed=0):
    """Smooth-ish data with noise: compresses, and predictors matter."""
    rng = np.random.default_rng(seed)
    grid = np.indices(shape[:2]).sum(axis=0)
    base = grid[..., None] if len(shape) == 3 else grid
    if np.dtype(dtype).kind == "f":
        return (np.sin(base / 17.0) * 100 + rng.normal(size=shape)).astype(dtype)
    top = 250 if np.dtype(dtype).itemsize == 1 else 60000
    return ((base * 7 % top) + rng.integers(0, 5, size=shape)).astype(dtype)


GREY, RGB = (150, 173), (120, 141, 3)
Q90 = {"level": 90}
STRIPS, TILES = {"rowsperstrip": 37}, {"tile": (64, 48)}

# name -> (data, imwrite kwargs, expected own codec, tolerance vs tifffile)
CASES = {
    "lzw_strips":         (_data(GREY), dict(compression="lzw", **STRIPS), "lzw", 0),
    "lzw_tiles_pred":     (_data(GREY), dict(compression="lzw", predictor=True, **TILES), "lzw", 0),
    "lzw_rgb_pred":       (_data(RGB, "u1"), dict(compression="lzw", predictor=True,
                                                  photometric="rgb", **STRIPS), "lzw", 0),
    "lzw_float_pred3":    (_data(GREY, "f4"), dict(compression="lzw", predictor=3, **STRIPS),
                           "lzw", 0),
    "lzw_float64_pred3_be": (_data(GREY, "f8"), dict(compression="lzw", predictor=3,
                                                     byteorder=">", **TILES), "lzw", 0),
    "lzw_big_endian_pred": (_data(GREY, "i2"), dict(compression="lzw", predictor=True,
                                                    byteorder=">", **STRIPS), "lzw", 0),
    "adobe_deflate":      (_data(GREY), dict(compression="adobe_deflate", **TILES), "zlib", 0),
    "deflate_pred_u32":   (_data(GREY, "u4"), dict(compression="zlib", predictor=True, **STRIPS),
                           "zlib", 0),
    "deflate_float_pred3_rgb": (_data(RGB, "f4"), dict(compression="zlib", predictor=3,
                                                       photometric="rgb", **TILES), "zlib", 0),
    "packbits_u8":        (_data(GREY, "u1"), dict(compression="packbits", **STRIPS),
                           "packbits", 0),
    "packbits_u16":       (_data(GREY), dict(compression="packbits", **TILES), "packbits", 0),
    "lzma":               (_data(GREY), dict(compression="lzma", **TILES), "lzma", 0),
    "zstd_pred":          (_data(GREY), dict(compression="zstd", predictor=True, **STRIPS),
                           "zstd", 0),
    "rgb_separate_lzw":   (_data((3,) + GREY, "u1"), dict(compression="lzw",
                                                          photometric="rgb",
                                                          planarconfig="separate",
                                                          **STRIPS), "lzw", 0),
    # image codecs
    "jpeg_grey":          (_data(GREY, "u1"), dict(compression="jpeg", compressionargs=Q90,
                                                   **TILES), "jpeg", 0),
    "jpeg_ycbcr_strips":  (_data(RGB, "u1"), dict(compression="jpeg", compressionargs=Q90,
                                                  photometric="ycbcr", **STRIPS), "jpeg", 0),
    "jpeg_rgb_tiles":     (_data(RGB, "u1"), dict(compression="jpeg", compressionargs=Q90,
                                                  photometric="rgb", **TILES), "jpeg", 0),
    "jpeg_lossless_12bit": (_data(GREY) % 4096, dict(compression="jpeg", compressionargs=dict(
        lossless=True, bitspersample=12), **TILES), "jpeg", 0),
    "jpeg_lossless_16bit": (_data(GREY), dict(compression="jpeg", compressionargs=dict(
        lossless=True, bitspersample=16), **TILES), "jpeg", 0),
    "jpeg2000_u16":       (_data(GREY), dict(compression="jpeg2000", **TILES), "jpeg2k", 0),
    "jpeg2000_rgb_lossy": (_data(RGB, "u1"), dict(compression="jpeg2000",
                                                  compressionargs={"level": 40},
                                                  photometric="rgb", **TILES), "jpeg2k", 0),
    "webp_lossless":      (_data(RGB, "u1"), dict(compression="webp",
                                                  compressionargs={"lossless": True},
                                                  photometric="rgb", **TILES), "webp", 0),
    "webp_lossy":         (_data(RGB, "u1"), dict(compression="webp", compressionargs={"level": 70},
                                                  photometric="rgb", **TILES), "webp", 0),
    "jpegxl_u16":         (_data(GREY), dict(compression="jpegxl", **TILES), "jxl", 0),
    "jpegxl_rgb":         (_data(RGB, "u1"), dict(compression="jpegxl", photometric="rgb",
                                                  **TILES), "jxl", 0),
    "png_u16":            (_data(GREY), dict(compression="png", **TILES), "png", 0),
    "png_rgb":            (_data(RGB, "u1"), dict(compression="png", photometric="rgb",
                                                  **STRIPS), "png", 0),
}


@pytest.fixture(scope="module")
def files(tmp_path_factory):
    root = tmp_path_factory.mktemp("tiffcodecs")
    out = {}
    for name, (data, kwargs, _, _) in CASES.items():
        path = str(root / f"{name}.tif")
        try:
            tifffile.imwrite(path, data, **kwargs)
        except Exception as exc:                    # noqa: BLE001 - encoder not in this build
            out[name] = exc
            continue
        out[name] = path
    return out


def _path(files, name):
    if isinstance(files[name], Exception):
        pytest.skip(f"imagecodecs cannot write {name}: {files[name]}")
    return files[name]


def _keys(shape, n=12, seed=4):
    rng = np.random.default_rng(seed)
    keys = [(Ellipsis,)]
    for _ in range(n):
        key = []
        for size in shape:
            a = int(rng.integers(0, size))
            key.append(slice(a, int(rng.integers(a + 1, size + 1))))
        keys.append(tuple(key))
    return keys


def _compare(got, ref, tolerance, msg):
    if tolerance:
        diff = np.abs(got.astype(np.int64) - ref.astype(np.int64)).max() if got.size else 0
        assert diff <= tolerance, f"{msg}: differs by {diff}"
    else:
        np.testing.assert_array_equal(got, ref, err_msg=msg)


@pytest.fixture(params=[1, 4], ids=["serial", "pool"])
def decode_threads(request):
    old = micro_reader.decode_threads()
    micro_reader.set_decode_threads(request.param)
    yield
    micro_reader.set_decode_threads(old)


@pytest.mark.parametrize("name", list(CASES))
def test_own_codec_matches_tifffile(files, name, decode_threads):
    path = _path(files, name)
    _, _, codec, tolerance = CASES[name]
    with tifffile.TiffFile(path) as tf:
        ref = tf.series[0].asarray()
    with micro_reader.open(path) as img:
        scene = img.images[0]
        assert scene._codec == codec                    # decoded by micro-reader, not tifffile
        assert scene.shape == ref.shape and scene.dtype == ref.dtype
        for key in _keys(ref.shape):
            got = scene[key]
            assert got.dtype.isnative
            _compare(got, ref[key], tolerance, f"{name} {key}")


@pytest.mark.parametrize("name", list(CASES))
def test_reference_path_agrees(files, name, monkeypatch):
    """Without the native module, the reference decoders give the same
    pixels."""
    if not _codecs.native_available():
        pytest.skip("native module not installed")
    path = _path(files, name)
    tolerance = CASES[name][3]
    with micro_reader.open(path) as img:
        native = img.images[0][...]
    monkeypatch.setattr(_codecs, "_native", None)
    with micro_reader.open(path) as img:
        _compare(img.images[0][...], native, tolerance, name)


def _split_jpeg(stream: bytes):
    """A complete JPEG -> (tables-only stream, abbreviated image stream), as
    libtiff writes JPEG TIFFs: DQT / DHT in JPEGTables, not in each tile."""
    assert stream[:2] == b"\xff\xd8"
    tables, image, pos = [b"\xff\xd8"], [b"\xff\xd8"], 2
    while True:
        marker = stream[pos:pos + 2]
        length = int.from_bytes(stream[pos + 2:pos + 4], "big")
        segment = stream[pos:pos + 2 + length]
        if marker == b"\xff\xda":                   # SOS: the rest is scan data
            image.append(stream[pos:])
            break
        (tables if marker in (b"\xff\xdb", b"\xff\xc4") else image).append(segment)
        pos += 2 + length
    return b"".join(tables) + b"\xff\xd9", b"".join(image)


@pytest.mark.parametrize("photometric,shape", [("minisblack", GREY), ("rgb", RGB)])
def test_jpeg_tables_are_joined(tmp_path, photometric, shape):
    """Tiles as abbreviated JPEG streams + one JPEGTables tag (libtiff,
    GDAL, Aperio SVS): micro-reader joins them before decoding."""
    import imagecodecs
    data = _data(shape, "u1")
    th, tw = 64, 48
    tiles, tables = [], None
    for y in range(0, shape[0], th):
        for x in range(0, shape[1], tw):
            tile = np.zeros((th, tw) + shape[2:], "u1")
            part = data[y:y + th, x:x + tw]
            tile[:part.shape[0], :part.shape[1]] = part
            full = imagecodecs.jpeg8_encode(tile, level=90, colorspace="rgb" if len(shape) == 3
                                            else None)
            tables, abbreviated = _split_jpeg(bytes(full))
            tiles.append(abbreviated)
    path = str(tmp_path / f"tables_{photometric}.tif")
    with tifffile.TiffWriter(path) as tif:
        tif.write(iter(tiles), shape=shape, dtype="u1", tile=(th, tw), compression="jpeg",
                  photometric=photometric, extratags=[(347, 7, len(tables), tables, True)])
    with tifffile.TiffFile(path) as tf:
        assert tf.pages[0].jpegtables == tables        # the case under test
        ref = tf.asarray()
    with micro_reader.open(path) as img:
        scene = img.images[0]
        assert scene._codec == "jpeg"
        _compare(scene[...], ref, 3, photometric)
        assert np.abs(scene[...].astype(int) - data.astype(int)).mean() < 3   # really the image


def test_other_compressions_go_to_tifffile(tmp_path):
    """LERC is not one of micro-reader's codecs: tifffile (imagecodecs) decodes it."""
    data = _data(GREY, "f4")
    path = str(tmp_path / "lerc.tif")
    try:
        tifffile.imwrite(path, data, compression="lerc", **TILES)
    except Exception as exc:                            # noqa: BLE001
        pytest.skip(f"no LERC encoder: {exc}")
    with micro_reader.open(path) as img:
        scene = img.images[0]
        assert scene._codec is None
        np.testing.assert_array_equal(scene[...], tifffile.imread(path))


def test_missing_imagecodecs_is_explained(tmp_path, monkeypatch):
    """When tifffile's decoder fails and imagecodecs is not installed, the
    error names the compression and the package."""
    data = _data(GREY, "f4")
    path = str(tmp_path / "lerc.tif")
    try:
        tifffile.imwrite(path, data, compression="lerc", **TILES)
    except Exception as exc:                            # noqa: BLE001
        pytest.skip(f"no LERC encoder: {exc}")
    with micro_reader.open(path) as img:
        scene = img.images[0]

        def fail(*args, **kwargs):
            raise ValueError("requires the imagecodecs package")
        monkeypatch.setattr(scene, "_decode", fail)
        monkeypatch.setitem(sys.modules, "imagecodecs", None)   # import fails
        with pytest.raises(UnsupportedFormatError, match="imagecodecs"):
            scene[...]


# -- packed bit depths and FillOrder 2 (codec audit follow-up #2) ----------------------

from _tiff_writer import write_tiff  # noqa: E402

PACKED = [(1, "u1"), (2, "u1"), (4, "u1"), (10, "u2"), (12, "u2"), (14, "u2"), (24, "u4"),
          (8, "u1"), (16, "u2")]                       # last two: FillOrder 2 on whole bytes


def _packed_case(tmp_path, bits, dtype, byteorder, compression, fillorder, samples):
    rng = np.random.default_rng(bits * 7 + samples)
    shape = (23, 17) + ((samples,) if samples > 1 else ())
    data = rng.integers(0, 1 << bits, shape, dtype=np.uint64).astype(dtype)
    path = str(tmp_path / f"p{bits}{byteorder == '<'}{compression}{fillorder}{samples}.tif")
    write_tiff(path, data, bits, byteorder=byteorder, rows_per_strip=5,
               compression=compression, fillorder=fillorder)
    return path, data.astype(bool) if bits == 1 else data


@pytest.mark.parametrize("fillorder", [1, 2])
@pytest.mark.parametrize("compression", [1, 8, 5], ids=["none", "deflate", "lzw"])
@pytest.mark.parametrize("byteorder", ["<", ">"], ids=["II", "MM"])
@pytest.mark.parametrize("bits,dtype", PACKED, ids=[f"{b}bit" for b, _ in PACKED])
def test_packed_depths_and_fill_order(tmp_path, bits, dtype, byteorder, compression, fillorder):
    if bits % 8 == 0 and bits == 8 * np.dtype(dtype).itemsize and fillorder == 1:
        pytest.skip("plain whole-byte samples: covered elsewhere")
    path, data = _packed_case(tmp_path, bits, dtype, byteorder, compression, fillorder, 1)
    with micro_reader.open(path) as img:
        scene = img.images[0]
        assert scene._codec is not None                  # decoded by micro-reader
        for key in _keys(data.shape):
            np.testing.assert_array_equal(scene[key], data[key], err_msg=str(key))
    if bits == 24:
        # tifffile is no reference for 24-bit samples, and its behaviour
        # changed between versions: 2025.12.20 shifts little-endian samples
        # left by 8 bits, 2025.5.21 garbles big-endian ones (both noted for
        # the upstream reports).  The written data above is the reference.
        return
    np.testing.assert_array_equal(tifffile.imread(path), data)


@pytest.mark.parametrize("bits,dtype", [(4, "u1"), (12, "u2")])
def test_packed_rgb(tmp_path, bits, dtype):
    """Colour samples are packed together along the row."""
    path, data = _packed_case(tmp_path, bits, dtype, "<", 8, 1, 3)
    with micro_reader.open(path) as img:
        np.testing.assert_array_equal(img.images[0][...], data)
    np.testing.assert_array_equal(tifffile.imread(path), data)


@pytest.mark.parametrize("bits,dtype", [(1, "u1"), (12, "u2"), (24, "u4")])
def test_packed_reference_path(tmp_path, bits, dtype, monkeypatch):
    """Without the native module, numpy unpacks the same samples."""
    path, data = _packed_case(tmp_path, bits, dtype, ">", 5, 2, 1)
    monkeypatch.setattr(_codecs, "_native", None)
    with micro_reader.open(path) as img:
        np.testing.assert_array_equal(img.images[0][...], data)


def test_unpack_bits_native_matches_reference():
    if not _codecs.native_available():
        pytest.skip("native module not installed")
    from _tiff_writer import pack_bits
    rng = np.random.default_rng(3)
    for bits in (1, 2, 3, 5, 7, 9, 11, 12, 13, 17, 20, 24, 31, 32):
        dtype = np.uint8 if bits <= 8 else np.uint16 if bits <= 16 else np.uint32
        values = rng.integers(0, 1 << bits, (6, 11), dtype=np.uint64).astype(dtype)
        packed = pack_bits(values, bits)
        for native in (True, False):
            out = np.empty_like(values)
            assert _codecs.unpack_bits_into(packed, out, bits, 11, native=native) == values.size
            np.testing.assert_array_equal(out, values, err_msg=f"{bits} native={native}")


# -- EER (Falcon cryo-EM movies; codec audit follow-up #3) -----------------------------

from _eer import eer_encode, random_events  # noqa: E402
from _tiff_writer import write_pages  # noqa: E402

EER_META = b"<metadata><item key='sensorPixelSize.width'>1.0</item></metadata>"
# name -> (compression, skip, horizontal, vertical bits, extra tags)
EER_CASES = {
    "eer_65000_skip8": (65000, 8, 2, 2, {}),
    "eer_65001_skip7": (65001, 7, 2, 2, {}),
    "eer_65002_v2_tags": (65002, 8, 1, 1, {65007: (3, [8]), 65008: (3, [1]), 65009: (3, [1])}),
}


def _eer_movie(tmp_path, name, bits=1, frames=5, shape=(40, 56), metadata=True, seed=1):
    code, skip, hb, vb, tags = EER_CASES[name]
    rng = np.random.default_rng(seed)
    strips, truth = [], []
    for _ in range(frames):
        events, counts = random_events(rng, *shape, 0.08, hb, vb)
        strips.append(eer_encode(events, shape[0] * shape[1], skip, hb, vb))
        truth.append(counts)
    extra = dict(tags)
    if metadata:
        extra[65001] = (7, EER_META)
    path = str(tmp_path / f"{name}_{bits}.eer")
    write_pages(path, strips, *shape, bits, code, extra, bigtiff=True)
    return path, np.stack(truth)


@pytest.mark.parametrize("bits", [1, 8], ids=["bool", "uint8"])
@pytest.mark.parametrize("name", list(EER_CASES))
def test_eer_movies(tmp_path, name, bits, decode_threads):
    path, truth = _eer_movie(tmp_path, name, bits)
    # tifffile gives structure and type; pixels are checked against the
    # written counts (tifffile's own EER decoding depends on its imagecodecs
    # version: 2025.5.21 calls eer_decode with a since-renamed argument)
    with tifffile.TiffFile(path) as tf:
        dtype, shape = tf.series[0].dtype, tf.series[0].shape
    ref = truth.astype(dtype)
    with micro_reader.open(path) as img:
        scene = img.images[0]
        assert scene._codec == "eer" and scene.dtype == dtype and scene.shape == shape
        for key in _keys(ref.shape):
            np.testing.assert_array_equal(scene[key], ref[key], err_msg=str(key))


def test_eer_without_imagecodecs(tmp_path, monkeypatch):
    """The point of the native decoder: EER needs no imagecodecs."""
    if not _codecs.native_available():
        pytest.skip("native module not installed")
    path, truth = _eer_movie(tmp_path, "eer_65001_skip7")
    monkeypatch.setitem(sys.modules, "imagecodecs", None)
    with micro_reader.open(path) as img:
        np.testing.assert_array_equal(img.images[0][...].astype(np.uint8), truth)


def test_eer_reference_path(tmp_path, monkeypatch):
    path, truth = _eer_movie(tmp_path, "eer_65002_v2_tags", bits=8)
    monkeypatch.setattr(_codecs, "_native", None)
    with micro_reader.open(path) as img:
        np.testing.assert_array_equal(img.images[0][...], truth)


def test_eer_codes_without_eer_metadata_are_not_claimed(tmp_path):
    """Compression 65000-65002 without Falcon's metadata tag is not EER
    (a private code range): left to tifffile, as tifffile leaves it."""
    path, _ = _eer_movie(tmp_path, "eer_65001_skip7", metadata=False)
    with micro_reader.open(path) as img:
        assert img.images[0]._codec is None
        with pytest.raises(Exception):
            img.images[0][0]


def test_corrupt_eer_frame_raises(tmp_path):
    from micro_reader import CorruptFileError
    path, _ = _eer_movie(tmp_path, "eer_65001_skip7", frames=2)
    with tifffile.TiffFile(path) as tf:
        offset, count = tf.pages[1].dataoffsets[0], tf.pages[1].databytecounts[0]
    with open(path, "r+b") as fh:                   # a huge skip: events beyond the image
        fh.seek(offset)
        fh.write(b"\x7e" * count)
    with micro_reader.open(path) as img:
        img.images[0][0]                              # frame 0 intact
        with pytest.raises(CorruptFileError, match="EER"):
            img.images[0][1]


@pytest.mark.parametrize("superres", [0, 1, 2])
@pytest.mark.parametrize("skip,hb,vb", [(8, 2, 2), (7, 2, 2), (7, 1, 1), (6, 2, 2), (10, 3, 3)])
def test_eer_decoder_matches_imagecodecs(skip, hb, vb, superres):
    """The native decoder, bit for bit against imagecodecs' (the reference,
    Gohlke, BSD-3; ours is ported from it), super-resolution included."""
    if not _codecs.native_available():
        pytest.skip("native module not installed")
    import imagecodecs
    rng = np.random.default_rng(skip * 100 + hb * 10 + superres)
    h, w = 32, 48
    events, counts = random_events(rng, h, w, 0.1, hb, vb)
    data = eer_encode(events, h * w, skip, hb, vb)
    shape = (h * 2 ** min(superres, vb), w * 2 ** min(superres, hb))
    ref = imagecodecs.eer_decode(data, shape, skip, hb, vb, superres=superres)
    for dtype in (np.bool_, np.uint8, np.uint16):
        out = np.empty(shape, dtype)
        assert _codecs.eer_decode_into(data, out, skip, hb, vb, superres, native=True) == len(events)
        np.testing.assert_array_equal(out.astype(np.uint8), ref.astype(np.uint8))
