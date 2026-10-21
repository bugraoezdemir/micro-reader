# micro-reader-codecs

**Native image decoders for [micro-reader](https://github.com/bugraoezdemir/micro-reader).**

`micro-reader-codecs` holds the Rust decoders that micro-reader uses to read compressed microscopy data. It is a separate package so that micro-reader itself stays pure Python.

You normally don't need to install it yourself. It is installed automatically with:

```bash
pip install micro-reader
```

It has no Python dependencies.

## How it works

* Every decoder writes straight into a buffer you provide, such as a NumPy array.
* Decoders run without holding Python's lock, so micro-reader's thread pool can decode on all cores.
* Baseline JPEG, JPEG 2000 and JPEG XL decode on one thread each. micro-reader already decodes many tiles at once, and threads inside the codec would compete for the same cores.

## Decoders

| Function                    | Codec                                     | Rust crate                            |
| --------------------------- | ----------------------------------------- | ------------------------------------- |
| `zstd_decompress_into`      | zstd                                      | `zstd` (libzstd)                      |
| `lz4_block_decompress_into` | LZ4 block                                 | `lz4_flex`                            |
| `lzf_decompress_into`       | LZF (h5py's HDF5 filter)                  | own                                   |
| `lzw_decode_into`           | LZW, TIFF variant                         | `weezl`                               |
| `zlib_decompress_into`      | zlib / deflate                            | `libdeflater` (libdeflate)            |
| `packbits_decode_into`      | PackBits                                  | own                                   |
| `jpeg_decode_into`          | JPEG: baseline and extended (same pixels as libjpeg-turbo) | own, following libjpeg-turbo |
|                             | JPEG: progressive and lossless            | `jpeg-decoder`                        |
| `jxl_decode_into`           | JPEG XL                                   | `jxl-oxide`                           |
| `jpeg2k_decode_into`        | JPEG 2000 (J2K, JP2), also YCbCr with subsampled colour (e.g. Aperio slides) | `jpeg2k` (OpenJPEG) |
| `png_decode_into`           | PNG                                       | `png`                                 |
| `webp_decode_into`          | WebP, lossless and lossy                  | `image-webp`                          |
| `unpack_bits_into`          | packed samples, 1 to 32 bit               | own                                   |
| `eer_decode_into`           | EER (Falcon cryo-EM movies)               | own, ported from imagecodecs (BSD-3)  |

## Reading large blocks in parts

Some compressed blocks are too large to decode as a whole. These decoders read only part of a block:

* `jpeg2k_decode_area_into` and `jxl_decode_area_into` decode only the rectangle you ask for.
* `StreamDecoder` decodes one compressed block of a file as a stream and keeps only the bytes you ask for. It supports zlib, deflate, LZW, zstd, PackBits, PNG rows and uncompressed data. Each read continues where the last one stopped.
* `BlockIndex` keeps checkpoints of a deflate block. Several `StreamDecoder`s can share one, so jumping back in the block resumes from the nearest checkpoint instead of the start.

## Not included yet

JPEG XR is not included, because the `jpegxr` crate needs libclang to build. Until then, micro-reader decodes JPEG XR through imagecodecs:

```bash
pip install "micro-reader[codecs]"
```

## Building

The Rust toolchain is pinned in `rust-toolchain.toml`. To build and install into the active virtual environment:

```bash
maturin develop --release
```

## Tests

The tests live in the repository's `tests/` folder:

* `test_codecs.py`
* `test_czi_codecs.py`
* `test_tiff_codecs.py`
* `test_jpeg_exact.py`
* `test_memory_limit.py`

They compare these decoders with independent reference decoders, bit for bit.

## License

MIT

The bundled crates and C libraries (libzstd, libdeflate, OpenJPEG) are MIT, Apache-2.0, BSD or Zlib licensed. The JPEG decoder follows libjpeg-turbo (IJG and BSD licences): this software is based in part on the work of the Independent JPEG Group. All notices ship in every wheel as `THIRD_PARTY_LICENSES.txt`.

Whenever `Cargo.lock` changes, regenerate it with:

```bash
python scripts/third_party_licenses.py
```
