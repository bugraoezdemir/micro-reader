# micro-reader

**Fast, memory-bounded reading of image regions from large microscopy files.**

`micro-reader` is a Python library for reading regions of interest from microscopy images. It lets you access a small part of a large image without loading the entire image into memory, enabling streaming of image regions from arbitrary microscopy datasets. 

The library supports a wide range of microscopy formats through a common interface. It reads only the data needed for each request, works with the file's stored resolution levels, and uses parallel decoding to make region access fast.

[Documentation](https://bugraoezdemir.github.io/micro-reader/) · [PyPI](https://pypi.org/project/micro-reader/) · [Changelog](https://github.com/bugraoezdemir/micro-reader/blob/main/CHANGELOG.md)

> **Alpha release (0.x)**
>
> The API may change between releases. Pin the version you use in your projects.

## Installation

```bash
pip install micro-reader
```

Requires Python 3.11 or newer. The native image decoders are provided by `micro-reader-codecs`, which is installed automatically. Prebuilt wheels are available for Linux (x86-64 and ARM), macOS (Intel and Apple silicon) and Windows (x86-64).

Some formats and less common TIFF compression methods require optional dependencies:

```bash
pip install "micro-reader[hdf5]"     # HDF5-based formats
pip install "micro-reader[codecs]"   # additional TIFF codecs
pip install "micro-reader[all]"      # both
```

The HDF5 extra supports Imaris (`.ims`), ilastik, BigDataViewer, Luxendo and Fiji HDF5 files. The codecs extra adds support for less common TIFF compression methods, including LERC, CCITT fax and float24.

## Quick start

Open a microscopy file, choose an image and read the region you need:

```python
import micro_reader

with micro_reader.open("image.czi") as f:
    image = f.images[0]

    print(image.axes, image.shape, image.dtype)
    # tczyx (1, 3, 40, 5120, 6144) uint16

    # Read a 2000 × 2048 region from channel 1, plane 10
    region = image.read(
        c=1, z=10,
        y=slice(0, 2048),
        x=slice(1000, 3000),
    )

    print(region.shape)
    # (1, 2048, 2000): t is kept, c and z are dropped
```

You can also use NumPy-style indexing:

```python
region = image[0, 1, 10, 0:2048, 1000:3000]
```

The returned array contains only the requested region. The memory required for the read therefore depends on the region and the underlying storage blocks, and is independent of the total size of the image.

Images use the axes `t c z y x`. Other dimensions, such as series, tiles, views, wells and fields, are represented as separate images and identified by their indices. Use `f.select()` to find images by these indices:

```python
image = f.select(series=1, tile=3)[0]
```

## Reading regions from large images

Microscopy files often contain images that are too large to load into memory comfortably. Whole-slide scans, large mosaics and high-resolution volumetric images are common examples.

`micro-reader` is designed to access these datasets by region. When you request a rectangular area, it reads and decodes the data needed to produce that area, rather than loading the complete image first.

For compressed files, the amount of data that must be decoded depends on how the image is stored. Some formats allow individual regions or rows to be decoded independently. Others store pixels in larger blocks that must be decoded together. `micro-reader` handles these differences and reports how much data was actually read through `image.stats`.

### Example: a whole-slide CZI

This 3.6 GB Zeiss CZI slide scan contains 6,093 JPEG XR-compressed tiles. Open it as a mosaic to access each scan region as one image:

```python
f = micro_reader.open("22_0670_6D_11_MUS_3.czi", tiles="mosaic")

for image in f.images:
    print(image.name, image.shape)

# ScanRegion0    (1, 3, 1, 79995, 60633)
# ScanRegion1    (1, 3, 1, 76768, 56314)
# ScanRegion2    (1, 3, 1, 71377, 56337)
# label          (1, 3, 1, 545, 689)
# slide preview  (1, 3, 1, 717, 1568)
```

Read a small RGB region from one of the scan areas:

```python
image = f.images[0]

region = image[0, :, 0, 40000:42048, 30000:32048]
region.shape
# (3, 2048, 2048)
```

The file also contains a resolution pyramid, which can be used to read smaller versions of the image without calculating them from the full-resolution data:

```python
len(image.resolutions)
# 8

overview = image.resolutions[-1].read()
```

With `tiles="mosaic"`, tiles are placed at their recorded positions on a common canvas. They are not registered or blended. By default, tiles remain separate images with their own stage positions, which is useful when you want to perform registration or stitching yourself.

### Example: a high-content screening plate

The same interface can be used with plate-based acquisitions:

```python
f = micro_reader.open("1053601756.wpi")  # Yokogawa CV8000 plate

plate = f.plates[0]
plate.name
# '1053601756'

len(plate.wells), len(f.images)
# (384, 2304)

fields = f.select(plate_name=plate.name, well_name="B2")
# The six fields acquired in well B2 of this plate
# Each field is an Image object and has its API

[ch.name for ch in fields[0].metadata.channels]
# ['Hoechst', 'Concanavalin', 'SYTO',
#  'WGA_Phalloidin', 'MitoTracker', 'TL']

# Read a field like any other Image object
nuclei = fields[0].read(c=0)
```

> **Important!**
>
> `select()` matches every index in the File object by default. Without `plate_name` (or the plate's position, `plate`), a well name selects that well in every plate the file holds. It is therefore good practice to name every index you want to search within.
>
> Each plate's name is in `f.plates[i].name`, and `print(f)` lists the plates with their names. A plate whose file records no name has the name `""`; select it by position instead, for example `plate=0`.

See [Getting started](https://bugraoezdemir.github.io/micro-reader/getting-started/) for a step-by-step introduction.

## Memory-bounded image reading

A region read does not require the complete image to fit in memory. `micro-reader` reads the strips, tiles, subblocks or chunks that overlap the requested region.

It can also decode large compressed blocks in smaller parts when the compression method allows it. This includes Deflate, LZW, zstd, JPEG 2000 and JPEG with restart markers. For compression methods that require a complete block to be decoded, the block size determines the minimum amount of data that must be processed.

All decoding is subject to a process-wide memory limit. This helps keep memory use under control when reading large images or processing several regions concurrently:

```python
micro_reader.set_memory_limit("2GB")
```

The library also provides read statistics, including the number of pixels returned and the number of pixels that had to be read or decoded. These help identify cases where the file's storage layout makes a region read more expensive than expected.

See [Memory and threads](https://bugraoezdemir.github.io/micro-reader/guide/performance/) for details.

## Fast image decoding

`micro-reader` uses native decoding libraries provided by `micro-reader-codecs`. These include libjpeg-turbo, jxrlib, OpenJPEG, libdeflate and zstd.

Decoding is parallelised across CPU cores, and native decoding releases Python's global interpreter lock. A single read can therefore make use of multiple cores. Decoded blocks are also cached, so nearby region requests can reuse data that has already been decoded.

The benchmarks below show typical reading times for several microscopy formats.

## Supported formats

| Format | Supported variants |
| --- | --- |
| **TIFF / BigTIFF** | TIFF, OME-TIFF, ImageJ, Zeiss LSM, MetaMorph STK, Micro-Manager, ScanImage and whole-slide TIFFs (Aperio SVS, Hamamatsu NDPI, Leica SCN, Ventana BIF, Akoya QPTIFF, Philips) |
| **OME companion files** | `.companion.ome` sets of OME-TIFF files |
| **Zeiss CZI** | Mosaic tiles, stored resolution pyramids and JPEG XR slide scans |
| **Nikon ND2** | Including legacy JPEG 2000 files |
| **Leica** | LIF, LOF, LIFEXT, XLIF, XLEF, XLCF |
| **Olympus / Evident** | OIR, OIB, OIF and VSI slide scans |
| **Hamamatsu NDPIS** | Fluorescence sets with one NDPI file per channel |
| **Imaris** | `.ims`, requires the `[hdf5]` extra |
| **Other HDF5** | ilastik, BigDataViewer / BigStitcher, Luxendo and Fiji HDF5, requires the `[hdf5]` extra |
| **ICS / IDS** | ICS 1.0 and 2.0 |
| **MetaMorph** | `.nd` acquisitions |
| **Micro-Manager 1.4** | Image-per-file acquisitions |
| **DeltaVision** | `.dv`, `.r3d` |
| **MRC** | Common MRC modes |
| **KLB** | Keller Lab Block files |
| **JPEG / PNG / BMP** | Common raster image formats |
| **Screening plates** | Harmony, Columbus, MetaXpress, ImageXpress HCS.ai, IN Cell, CellVoyager, Opera Flex, scanR and Cellomics; also CZI, ND2, OME companion and OME-TIFF files recorded on Yokogawa CQ1 plates |

The [Formats](https://bugraoezdemir.github.io/micro-reader/formats/) page describes format-specific features and metadata support.

The following formats are not yet supported: DICOM whole-slide images, 3DHISTECH MRXS, Hamamatsu VMS, Zeiss ZVI, Becker & Hickl SDT, PicoQuant PTU, Imaris 3 files stored as TIFF, and ICS 1.0 files with Unix-compressed pixels.

## Benchmarks

The following measurements were made with micro-reader 0.4.4 on a laptop with an Intel Core Ultra 7 255H (16 cores), 31 GB RAM and an NVMe SSD, running Windows 11.

The default settings were used (16 decoding threads). Each measurement is the median of three runs in fresh processes. The 512 × 512 measurements are the median over 100 randomly selected regions at full resolution.

All times are for reading from disk. The operating system's file cache, the copy of recently read file data that the operating system keeps in memory, was emptied before every run and again before every large read. Reading the same data a second time is faster.

Peak memory recorded during the read includes the returned array. "≈ read region's size" marks reads whose peak memory is almost entirely the returned array itself.

| File | File size | Read | Time | Peak memory |
| --- | ---: | --- | ---: | ---: |
| Hamamatsu NDPI slide (JPEG, 101,376 × 188,160) | 6.9 GB | 512 × 512 RGB region | 4.6 ms | |
| | | 8192 × 8192 RGB region (192 MB) | 0.33 s | 378 MB |
| Ventana BIF slide (JPEG, 93,951 × 105,813) | 3.9 GB | 512 × 512 RGB region | 8.5 ms | |
| | | 8192 × 8192 RGB region (192 MB) | 0.21 s | 384 MB |
| Zeiss CZI slide scan (JPEG XR, 6,093 tiles) | 3.6 GB | 512 × 512 RGB region | 31 ms | |
| | | 8192 × 8192 RGB region (192 MB) | 0.52 s | 383 MB |
| Aperio SVS slide (JPEG) | 596 MB | 512 × 512 RGB region | 1.5 ms | |
| | | 8192 × 8192 RGB region (192 MB) | 0.24 s | 379 MB |
| MRC image (uncompressed, 67,320 × 50,960) | 6.9 GB | 512 × 512 region | 29 ms | |
| | | 8192 × 8192 region (128 MB) | 0.85 s | 121 MB (≈ read region's size) |
| ICS / IDS, deconvolved (float32, 4 c × 226 z of 1352 × 1352) | 6.6 GB | One channel: 226 z of 1352 × 1352 (1.5 GB) | 0.92 s | 1.5 GB (≈ read region's size) |
| Micro-Manager OME-TIFF, two files (140 t × 3 c × 21 z) | 4.7 GB | One channel: 140 t × 21 z of 512 × 512 (1.4 GB) | 0.26 s | 1.4 GB (≈ read region's size) |
| Imaris IMS (gzip chunks, float32, 9 c × 569 z) | 3.1 GB | One channel: 569 z of 512 × 512 (569 MB) | 0.17 s | 636 MB |
| Nikon ND2, 46 XY points | 395 MB | Whole file: 46 points × 31 t × 2 c (362 MB) | 0.24 s | 356 MB (≈ read region's size) |
| Zeiss CZI plate: 20 wells, 500 fields | 27.5 GB | Open | 1.8 s | |
| | | One well: 25 fields × 7 c × 3 z of 1144 × 1144 (1.3 GB) | 0.29 s | 1.3 GB (≈ read region's size) |
| Zeiss CZI fluorescence mosaic (5 × 9 tiles per well) | 738 MB | One channel of a well: 2 t × 4 z of 3444 × 2944 (77 MB) | 0.04 s | 74 MB (≈ read region's size) |
| CellVoyager CV8000 plate: 384 wells, 2,304 fields, 18,452 files | 35 GB | Open | 1.9 s | |
| | | One well: 6 fields × 6 c × 3 z of 996 × 996, from 108 files (204 MB) | 0.21 s | 205 MB (≈ read region's size) |

Actual performance depends on hardware and storage.

## Learn more

- **[Getting started](https://bugraoezdemir.github.io/micro-reader/getting-started/):** Install the package, open an image and read a region.
- **[Concepts](https://bugraoezdemir.github.io/micro-reader/concepts/):** Files, images, indices, axes and coordinate systems.
- **[User guide](https://bugraoezdemir.github.io/micro-reader/guide/opening/):** Opening files, finding images, reading regions, resolution levels, metadata and plates.
- **[Formats](https://bugraoezdemir.github.io/micro-reader/formats/):** Supported formats, format-specific features and optional dependencies.
- **[API reference](https://bugraoezdemir.github.io/micro-reader/reference/):** Public functions, classes, attributes and errors.
- **[Examples](https://bugraoezdemir.github.io/micro-reader/examples/):** Recipes for image processing, Zarr export, Dask, plates and parallel workers.

## Used by

- **[EuBI-Bridge](https://github.com/Euro-BioImaging/EuBI-Bridge)** converts microscopy files to OME-Zarr and uses micro-reader as its default reader for supported formats.
- **[dyna-zarr](https://pypi.org/project/dyna-zarr/)** uses `micro_reader.Image` as a lazy data source for writing Zarr arrays.

## Feedback

Bug reports, requests for additional formats and sample files are welcome on [GitHub](https://github.com/bugraoezdemir/micro-reader/issues).

### Development note

The project has been developed with substantial AI assistance. Its functionality is continuously tested against real datasets, but some issues or edge cases may remain undiscovered. 

## License

`micro-reader` and `micro-reader-codecs` are MIT licensed. The codec wheels include third-party libraries. Their notices are available in [`codecs/THIRD_PARTY_LICENSES.txt`](https://github.com/bugraoezdemir/micro-reader/blob/main/codecs/THIRD_PARTY_LICENSES.txt).
