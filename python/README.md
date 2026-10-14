# micro-reader

**Memory-bounded pixel reader for microscopy images.**

`micro-reader` reads regions of large microscopy files without loading the entire image into memory. It provides a common interface across TIFF, CZI, ND2, Leica, OIR, Imaris, MRC, and common raster formats.

> **Status: pre-alpha (0.0.x)**
>
> The API may still change.

## Installation

For the main formats:

```bash
pip install micro-reader
```

Optional support:

```bash
pip install "micro-reader[ims]"      # Imaris (.ims)
pip install "micro-reader[codecs]"   # additional codecs
```

## Quick start

```python
import micro_reader

with micro_reader.open("image.czi") as f:
    image = f.images[0]

    print(image.axes)
    print(image.shape)
    print(image.dtype)

    # Read only the region you need
    block = image.read(
        c=1,
        z=10,
        y=slice(0, 2048),
        x=slice(1000, 3000),
    )
```

You can also use NumPy-style indexing:

```python
block = image[0, 1, 10, 0:2048, 1000:3000]
```

Reads return native-endian NumPy arrays.

For each image, you can inspect how much data had to be read internally:

```python
print(image.stats.over_read)
```

## How it works

`micro-reader` uses three concepts:

* **File**: the opened microscopy file
* **Image**: one image or frame that can be read as an array
* **Index**: identifies images that are separated by dimensions such as scene, tile, view, or emission wavelength

The key distinction is:

> **`t`, `c`, `z`, `y`, and `x` are array axes. Other dimensions are represented as image indices.**

This lets different microscopy formats use the same interface even when their internal structures are very different.

The basic model is:

```text
File
└── Image
    ├── axes
    ├── shape
    ├── dtype
    ├── index
    ├── metadata
    └── read(...)
```

## Files and images

Every file is opened as a `micro_reader.File`:

```python
f = micro_reader.open("image.czi")
```

You can inspect which reader and format variant were detected:

```python
f.backend
f.format
```

For example:

```text
f.backend  -> "czi"
f.format   -> "czi"
```

A file contains one or more `Image` objects:

```python
image = f.images[0]
```

Each image has an index describing where it came from:

```python
image.index
# {"scene": 0, "tile": 3, "view": 0}
```

You can inspect all indices in the file:

```python
f.indices
f.index_range
```

And select images by index:

```python
f.select(scene=1, view=0)
```

This returns the images matching the requested indices. For example, if a file contains several scenes, tiles, and views, selecting `scene=1, view=0` returns the tiles belonging to that scene and view.

### Available index dimensions

`micro-reader` uses these index names when a format provides the corresponding dimensions:

```text
plate
acquisition
well
field
scene
tile
rotation
view
illumination
phase
block
emission
excitation
lifetime
sequence
part
resolution
```

Only dimensions actually present in a file appear in an image's index.

## Axes

An image array uses these axes:

```text
t c z y x
```

and optionally:

```text
s
```

for RGB samples.

For example:

```python
image.axes
# "tczyx"

image.shape
# (1, 3, 40, 5120, 6144)
```

Other dimensions, such as illumination, emission, tile, etc., are represented as image indices rather than array axes, as discussed above.

### Changing axis order

You can request a specific axis order:

```python
image = f.images[0].as_axes("tczyx")
```

or when opening the file:

```python
f = micro_reader.open("image.tif", axes="tczyx")
```

You can also request the canonical order:

```python
f = micro_reader.open("image.tif", axes="canonical")
```

Canonical order is:

```text
t c z y x
```

followed by `s` when RGB samples are present.

Missing axes are added with size 1 and existing axes are reordered.

### Removing singleton axes

You can remove singleton axes with:

```python
plane = image.squeeze()
```

or:

```python
plane = image.squeeze("tz")
```

Only axes with size 1 can be removed. `y` and `x` are never removed.

### RGB samples

By default, RGB samples are represented by the `s` axis.

You can instead represent RGB as channels:

```python
f = micro_reader.open(
    "image.tif",
    samples="channels",
)
```

If there are existing channels already, RGB samples will fold with them, leading to a total of c × s channels.


## Ambiguous dimensions

Some formats do not say what a dimension represents.

For example, a plain multi-page TIFF may contain 80 pages without saying whether those pages are `z`, `t`, or something else.

`micro-reader` represents such pages using the `sequence` index:

```python
f = micro_reader.open("stack.tif")

print(f.index_range)
# {"scene": 1, "sequence": 80}

print(len(f.images))
# 80

print(f.images[0].index)
# {"scene": 0, "sequence": 0}

print(f.images[0].axes)
# yx
```

If you know that the pages represent `z`, you can tell the reader:

```python
f = micro_reader.open(
    "stack.tif",
    rename={"sequence": "z"},
)
```

The result is a single image with `zyx` axes.

```python
print(f.images[0].axes)
# zyx
```


## Resolution levels

Pyramidal files can contain multiple resolution levels:

```text
resolution 0  → full resolution
resolution 1  → lower resolution
resolution 2  → even lower resolution
```

By default, `f.images` contains only the full-resolution images:

```python
f = micro_reader.open(
    path,
    resolutions="base",  # the default, can be left out
)
```

Each image of a pyramidal file carries its resolution level in its index. In this mode, every image in `f.images` is level 0:

```python
print(f.index_range)
# {"scene": 4, "resolution": 1}

print(f.images[0].index)
# {"scene": 0, "resolution": 0}
```

`index_range` shows `"resolution": 1` because only the full-resolution level is in `f.images`.

Lower-resolution versions are available through the image via `image.resolutions`:

```python
image_scene0 = f.images[0]

image_scene0.is_pyramidal
# True

img_scene0_level2 = image_scene0.resolutions[2]  # a lower-resolution image

img_scene0_level2.index
# {"scene": 0, "resolution": 2}
```

If you prefer to expose every resolution level as an image directly under the `File` object:

```python
f = micro_reader.open(
    path,
    resolutions="all",  # instead of "base"
)

print(f.index_range)
# {"scene": 4, "resolution": 6}

img_scene0_level2 = f.select(scene=0, resolution=2)[0]
```

Supported pyramids include:

* **Imaris**: all stored levels
* **TIFF**: OME-TIFF and other TIFF pyramids such as SVS and NDPI
* **CZI**: stored pyramids of whole scenes

Other formats expose a single resolution level.

CZI pyramids belong to whole scenes. Separately opened CZI mosaic tiles therefore do not have their own resolution levels.

## Mosaic tiles

Mosaic tiles are separate images by default.

For CZI files, you can instead request stitched scenes:

```python
f = micro_reader.open(
    path,
    tiles="stitched",
)
```

The tiles are composed according to the format's stitching information.

Currently, stitching is supported for **CZI only**.

## Plates

High-content screening plates are opened through their index file, for example `Index.idx.xml` for Harmony. Each field is one image, indexed by `plate`, `acquisition`, `well` and `field`.

The plate layout is available through:

```python
plate = f.plates[0]
plate.wells[0].name
# "B2"
```

Images can be reached through the wells, or selected by well:

```python
image = plate.wells[0].fields[0].image
f.select(well=plate.well_index("B2"))
```

`f.outline()` shows the plates, wells and fields of a file.

Zeiss CZI plates are read as plates with `plates=True`, and as scenes otherwise:

```python
f = micro_reader.open("plate.czi", plates=True)
```

Missing plane files are read as zeros, with a `MissingImagesWarning`. A field without any of its files is left out.

When the channels of a field have different z sampling, `image.parts` gives one image per group of channels that share it.

## Metadata

Pixel and channel metadata are available through:

```python
meta = image.metadata
```

For example:

```python
meta.pixel_sizes
meta.position
meta.channels
meta.index_values
```

A typical result might look like:

```python
meta.pixel_sizes
# {
#     "x": Quantity(0.065, "micrometer"),
#     "z": Quantity(...),
#     "t": Quantity(2.5, "second"),
# }

meta.position
# {
#     "x": Quantity(1250.0, "micrometer"),
#     "y": ...
# }

meta.channels[0]
# Channel(
#     name="DAPI",
#     color="#00A0FF",
#     emission_wavelength=Quantity(461.0, "nanometer"),
# )
```

### Metadata principles

`micro-reader` follows a few important rules:

* Units use OME-Zarr-compatible names such as `micrometer`, `nanometer`, and `millisecond`.
* Missing metadata is left as `None` or absent.
* Pixel sizes are never invented as `1.0`.
* Colours are not invented when the file does not provide them.
* Each resolution level has its own pixel sizes.

### Image position

`image.metadata.position` gives the position of the centre of the first pixel in stage coordinates.

Where possible, format-specific coordinate conventions are converted to this common representation.

The position is omitted when it cannot be determined reliably.

## Supported formats

| Format               | Notes                                                 |
| -------------------- | ----------------------------------------------------- |
| **TIFF / BigTIFF**   | TIFF, OME-TIFF, ImageJ, LSM, STK and TIFF pyramids    |
| **OME companion**    | `.companion.ome` sets of OME-TIFF files               |
| **CZI**              | Zeiss CZI, including mosaic tiles and stitched scenes |
| **ND2**              | Nikon ND2, including legacy JPEG 2000 ND2             |
| **Leica**            | LIF, LOF, LIFEXT, XLIF, XLEF, XLCF                    |
| **OIR**              | Olympus / Evident FluoView                            |
| **Imaris**           | `.ims`, with the `[ims]` extra                        |
| **MRC**              | Common MRC modes and extended headers                 |
| **Plates (HCS)**     | Harmony, MetaXpress, IN Cell and CellVoyager plates   |
| **JPEG / PNG / BMP** | Common raster image formats                           |

All formats use the same `File`, `Image`, axes, indexing, metadata, and reading interface wherever possible.

Format-specific behaviour is described below.

### TIFF

Supported TIFF formats include:

* TIFF
* BigTIFF
* OME-TIFF
* ImageJ
* Zeiss LSM
* MetaMorph STK

Compressed TIFF data is decoded only for the strips or tiles needed by the requested region.

Supported codecs include:

* LZW
* Deflate
* PackBits
* LZMA
* zstd
* JPEG
* JPEG 2000
* WebP
* JPEG XL
* PNG
* EER

Additional codecs such as JPEG XR and LERC are available with:

```bash
pip install "micro-reader[codecs]"
```

### OME companion files

An OME companion file (`.companion.ome`) describes images stored in separate TIFF files. Opening the companion reads all those images and groups them together as one `t c z y x` image:

```python
f = micro_reader.open("data.companion.ome")
```

Opening one of the TIFF files on its own reads only that file, with its pixel sizes, channels and position taken from the companion. If the companion is missing, the file is read without pixel sizes, with a `CompanionWarning`.

### CZI

CZI files support:

* mosaic tiles
* scenes
* views
* rotations
* RGB
* large files
* stored scene pyramids

CZI-specific operations include:

```python
image.read_region(x, y, w, h)
```

and:

```python
f.canvas()
```

`image.origin` gives the image corner in the file's pixel coordinates.

Damaged subblock directories can be recovered with a `DamagedFileWarning`. Details are available through:

```python
f.damage
```

### ND2

Nikon ND2 support includes:

* ND2 version 3+
* legacy JPEG 2000 ND2 versions 1 and 2
* one image per XY position
* `t c z y x` axes
* RGB samples

### Leica

Supported Leica formats include:

* LIF
* LOF
* LIFEXT
* XLIF
* XLEF
* XLCF

Leica files can contain nested analysis results, tiles, rotations, and emission/excitation scans.

These are represented using the common image/index model.

### OIR

Olympus / Evident FluoView OIR files are supported, including companion files and spectral scans.

### Imaris

Install Imaris support with:

```bash
pip install "micro-reader[ims]"
```

Imaris resolution level 0 is exposed at the actual image size, with Imaris chunk padding removed.

Chunked compression is decoded in parallel.

### MRC

Supported MRC modes include:

```text
0, 1, 2, 4, 6, 12
```

Both byte orders and extended headers are supported.

### JPEG, PNG and BMP

Supported features include:

* PNG colour types and bit depths
* interlaced PNG
* BMP palettes, RLE and bit fields
* OS/2 BMP headers
* baseline, progressive and lossless JPEG

Raster images are normally decoded as a whole on the first read.

For images larger than 256 MB:

* non-interlaced PNGs are read by rows
* JPEGs with restart markers are read in bands of rows

For paletted images, pixel values remain indices and the palette is available through:

```python
image.palette
```

## Performance

Compressed data is decoded using a shared thread pool.

By default, the decoder uses one thread per CPU.

You can change this with:

```python
micro_reader.set_decode_threads(4)
```

or:

```bash
export MICRO_READER_DECODE_THREADS=4
```

The native codec implementations are provided by the separate Rust package `micro-reader-codecs`, which is installed automatically.

## Memory usage

Memory usage is controlled by a configurable limit:

```python
micro_reader.set_memory_limit("8GB")
micro_reader.memory_limit()
```

The default limit is one quarter of the computer's memory.

You can also set it through the environment:

```bash
export MICRO_READER_MEMORY_LIMIT=8GB
```

The limit covers the blocks being decoded at once. Decoded blocks are also kept for reuse,
so a block shared by neighbouring reads is not decoded twice. All open files share one
cache limit, 256 MB by default:

```python
micro_reader.set_cache_limit("512MB")
micro_reader.cache_limit()
micro_reader.cached_bytes()     # what the caches hold now
```

or `export MICRO_READER_CACHE_LIMIT=512MB`. The process then uses at most the memory
limit plus the cache limit for decoding, however many files are open. One exception: a
single picture (PNG, JPEG, ... up to 256 MB) is kept decoded by its own open file.

For uncompressed data, only the requested rows are read. When you ask for part of
each row, the rest of the row is read along with it if that is faster than reading the
rows one by one.

Compressed data is stored in blocks such as TIFF strips or tiles, CZI subblocks, and ND2 frames. Small blocks are decoded as a whole. Blocks larger than 16 MB are read in parts where the codec allows it.

### Streamed decoding

Deflate, LZW, zstd, PackBits, and LZMA blocks are decoded from their start while keeping only the rows needed by the requested region.

This covers, among others:

* TIFF strips and tiles
* ND2 lossless frames
* CZI subblocks
* non-interlaced PNG

Reading a block from top to bottom stays a single pass: each read continues where the last one stopped.

Jumping back in a large block, as a viewer does, resumes from a checkpoint for deflate. For LZW and zstd, the block is decoded again from its start.

### Region decoding

JPEG 2000 and JPEG XL can decode only the requested region.

This is used for TIFF, CZI, and legacy ND2 where supported by the format.

### JPEG restart markers

JPEG blocks with restart markers can be decoded only for the part covering the requested rows.

### Codecs that require whole-block decoding

Some blocks can only be decoded as a whole, including:

* progressive and lossless JPEG
* JPEG without restart markers
* interlaced PNG
* JPEG XR
* CZI zstd with hi-lo packing

### How the limit is enforced

Every decode, whole or in parts, counts against the memory limit.

If a block that must be decoded whole is larger than the configured limit, `MemoryLimitError` is raised instead of allowing the process to run out of memory.

Blocks decoded concurrently from multiple threads are accounted for together. If a decode does not fit within the current memory budget, it waits until sufficient memory is available.

## Design principles

`micro-reader` aims to keep the application-facing interface small even when the underlying file formats are complex.

Applications work with the same `File` and `Image` model described in [How it works](#how-it-works). Format-specific complexity stays inside the reader.

The goal is that an application can open a microscopy file, identify the image it needs, read only the required pixels, and obtain standardized metadata without needing to understand the vendor-specific file structure.

## License

MIT

Source code, tests, and development documentation:

https://github.com/bugraoezdemir/micro-reader
