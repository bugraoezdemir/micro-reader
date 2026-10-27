# micro-reader

**Memory-bounded pixel reader for microscopy images.**

`micro-reader` reads regions of large microscopy files without loading the entire image into memory. It provides a common interface across TIFF, CZI, ND2, Leica, OIR, OIB / OIF, Olympus VSI, Imaris, DeltaVision, MRC, common raster formats, OME companion file sets, and high-content screening plates (Harmony, Columbus, MetaXpress, ImageXpress HCS.ai, IN Cell, CellVoyager, CQ1, Opera Flex, scanR, Cellomics).

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

## Compatibility with eubi-bridge

micro-reader 0.1.0 renamed the image index `scene` to `series`. eubi-bridge 0.1.3 to 0.2.x and eubi-bridge-lite below 0.3.0 still read `scene`, so `import micro_reader` refuses to load next to them. It raises an `ImportError` that names the installed eubi version and the two ways out: upgrade eubi (`pip install -U 'eubi-bridge>=0.3.0'`, or the same for `eubi-bridge-lite`), or keep that eubi with the micro-reader it was built for (`pip install 'micro-reader<0.1'`). The check reads the versions of the installed packages; it never imports eubi. An editable install of eubi is checked like any other: its recorded version can be out of date, so the message then says to reinstall it.

To skip the check, set the environment variable `MICRO_READER_SKIP_EUBI_CHECK=1`. Only the value `1` skips it; any other value keeps the check. micro-reader then warns once that the check was skipped. This is meant for work on an eubi checkout, not for normal use.

The check will be removed when every eubi-bridge / eubi-bridge-lite line below 0.3.0 has left the support window, checked by the release version matrix; not before.

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
* **Index**: identifies images that are separated by dimensions such as series, tile, view, or emission wavelength

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
# {"series": 0, "tile": 3, "view": 0}
```

You can inspect all indices in the file:

```python
f.indices
f.index_range
```

And select images by index:

```python
f.select(series=1, view=0)
```

This returns the images matching the requested indices. For example, if a file contains several series, tiles, and views, selecting `series=1, view=0` returns the tiles belonging to that series and view.

### Available index dimensions

`micro-reader` uses these index names when a format provides the corresponding dimensions:

```text
plate
acquisition
well
field
series
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

### Files made of several files

Some formats always come as several files: an OIR file with its continuation files (`name_00001`), a Micro-Manager stack split over `..._MMStack_Pos0.ome.tif`, `..._MMStack_Pos0_1.ome.tif`, an NDTiff folder, an Olympus OIF with its `.oif.files` folder, an Olympus VSI with its `_name_` folder, a DeltaVision image with its softWoRx `.log`, a plate's side files (Harmony `Assaylayout/` and `FFC_Profile/`, CellVoyager settings files, MetaXpress `_thumb` previews). Opening any of these files opens the whole image:

```python
f = micro_reader.open("cells_R3D.dv.log")
f.path     # the main file: "cells_R3D.dv"
f.opened   # what was opened: "cells_R3D.dv.log"
```

You can also open the folder. A folder opens the image it belongs to (an NDTiff dataset folder, a Harmony or CellVoyager measurement folder, a folder of Opera `.flex` files), or the one image it holds. An Opera Flex plate has no index file: any of its `.flex` files opens the whole folder as the plate (`multifile=False` reads only that file). A folder with several images raises an error that lists them.

To sort many files at once, for example everything under a folder you want to convert, `group_files` says which files make one image:

```python
g = micro_reader.group_files("experiment/")
for group in g.groups:
    group.main      # the file to open: "slide.vsi", "plate/Index.idx.xml"
    group.parts     # the other files of a multi-file image
    group.members   # for a plate or companion index: the files its images come from
g.singles           # every other file, as given
```

A file that a plate or companion index names is listed only in that index's group. When the main file of a multi-file image is missing, its group has `main=None`.

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

`micro_reader.open` turns RGB samples into channels: an RGB image has 3 channels, named "<name> R", "G" and "B" and coloured red, green and blue (a file without a channel name gives three unnamed channels in those colours). An image that has channels already gets c × s of them: "<name> R" ... for each.

To keep RGB samples as their own `s` axis, the way the file stores them:

```python
f = micro_reader.open(
    "image.tif",
    samples="keep",
)
```

`image.source` is then not needed: with the default, it is the image as stored, with its `s` axis. `image.as_axes(...)` keeps `s` unless asked for `samples="channels"`.


## Ambiguous dimensions

Some formats do not say what a dimension represents.

For example, a plain multi-page TIFF may contain 80 pages without saying whether those pages are `z`, `t`, or something else.

`micro-reader` represents such pages using the `sequence` index:

```python
f = micro_reader.open("stack.tif")

print(f.index_range)
# {"series": 1, "sequence": 80}

print(len(f.images))
# 80

print(f.images[0].index)
# {"series": 0, "sequence": 0}

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
# {"series": 4, "resolution": 1}

print(f.images[0].index)
# {"series": 0, "resolution": 0}
```

`index_range` shows `"resolution": 1` because only the full-resolution level is in `f.images`.

Lower-resolution versions are available through the image via `image.resolutions`:

```python
image_series0 = f.images[0]

image_series0.is_pyramidal
# True

img_series0_level2 = image_series0.resolutions[2]  # a lower-resolution image

img_series0_level2.index
# {"series": 0, "resolution": 2}
```

If you prefer to expose every resolution level as an image directly under the `File` object:

```python
f = micro_reader.open(
    path,
    resolutions="all",  # instead of "base"
)

print(f.index_range)
# {"series": 4, "resolution": 6}

img_series0_level2 = f.select(series=0, resolution=2)[0]
```

Supported pyramids include:

* **Imaris**: all stored levels
* **TIFF**: OME-TIFF and whole-slide pyramids (SVS, NDPI, SCN, BIF, QPTIFF, Philips)
* **CZI**: stored pyramids of whole series

Other formats expose a single resolution level.

CZI pyramids belong to whole series. Separately opened CZI mosaic tiles therefore do not have their own resolution levels.

## Mosaic tiles

Mosaic tiles are separate images by default, each with its own position.

For CZI files and FluoView tiling sets (`.omp2info`), you can instead get one image per series, its tiles placed on one canvas:

```python
f = micro_reader.open(
    path,
    tiles="mosaic",
)
```

CZI tiles are placed by the file's own stitching information, FluoView tiles by the tiling set's grid (where tiles overlap, the later tile is on top; areas without a tile read as 0). Other formats with tiles raise an error for `tiles="mosaic"`.

`tiles="stitched"`, the name before 0.1.0, still works and warns.

## Plates

High-content screening plates are opened through their index file, for example `Index.idx.xml` for Harmony. Each field is one image, indexed by `plate`, `acquisition`, `well` and `field`.

The plate layout is available through:

```python
plate = f.plates[0]
plate.wells[0].name
# "B2"
plate.row_names, plate.column_names     # ('A', 'B', ...), ('1', '2', ...)
part = plate.rows[2:5].columns[3:7]     # a part of the plate: still a plate
```

Every level knows what it is and what holds it: `.kind` is `"file"`, `"plate"`, `"well"`
or `"field"`, and `.parent` is the level above. Accessors go down only: a file holds
`.plates`, a plate `.wells`, a well `.fields`, a field `.images`. `plate.rows` and
`plate.columns` slice a plate by position, name or a list of them, as `f.select(row=...,
column=...)` does; rows and columns are a well's address, not a level.

Images can be reached through the wells, or selected by well:

```python
image = plate.wells[0].fields[0].image
f.select(well=plate.well_index("B2"))       # by its place in plate.wells
f.select(well_name="B2")                    # by name
f.select(row_name="B", column=range(1, 4))  # rows and columns: names, or 0-based numbers
f.select(well=["B2", "C5"], field=[0, 1])   # a list, tuple, range or set: any of them
f.select(row=slice(2, None))                # a slice of positions: row C onward
```

A plate's places have a position and a name, a keyword each: `plate` / `plate_name`,
`acquisition` / `acquisition_name`, `well` / `well_name`, `row` / `row_name`, `column` /
`column_name` (the columns of ome-zarr-pyramid's table). A name given to a position keyword
is its name (`well="B2"` is `well_name="B2"`).

`f.outline()` shows the plates, wells and fields of a file.

Zeiss CZI plates are read as plates with `plates=True`, and as series otherwise:

```python
f = micro_reader.open("plate.czi", plates=True)
```

Missing plane files are read as zeros, with a `MissingImagesWarning`. A field without any of its files is left out.

A plane file whose size or pixel type differs from what the index describes raises an error when it is read. To find such files before reading anything, check them all at once:

```python
f.check_members()   # opens every plane file once; returns how many were checked
```

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
| **TIFF / BigTIFF**   | TIFF, OME-TIFF, ImageJ, LSM, STK, Micro-Manager, whole-slide TIFFs |
| **OME companion**    | `.companion.ome` sets of OME-TIFF files               |
| **CZI**              | Zeiss CZI, including mosaic tiles (separate or on one canvas) |
| **ND2**              | Nikon ND2, including legacy JPEG 2000 ND2             |
| **Leica**            | LIF, LOF, LIFEXT, XLIF, XLEF, XLCF                    |
| **OIR**              | Olympus / Evident FluoView, including tiling sets (`.omp2info`) |
| **OIB / OIF**        | Olympus FluoView FV1000 / FV1200 / FV3000             |
| **VSI**              | Olympus / Evident cellSens and VS120 / VS200 slide scanners |
| **NDPIS**            | Hamamatsu NanoZoomer fluorescence sets (one NDPI per channel) |
| **Imaris**           | `.ims`, with the `[hdf5]` extra                       |
| **HDF5**             | ilastik, BigDataViewer / BigStitcher, Luxendo, Fiji HDF5, with the `[hdf5]` extra |
| **ICS / IDS**        | Image Cytometry Standard 1.0 and 2.0 (SVI Huygens and others) |
| **MetaMorph**        | `.nd` acquisitions and their TIFF / STK files         |
| **Micro-Manager 1.4**| image-per-file acquisitions (`metadata.txt` + one TIFF per plane) |
| **DeltaVision**      | `.dv`, `.r3d` (softWoRx, OMX, Priism)                 |
| **MRC**              | Common MRC modes and extended headers                 |
| **Plates (HCS)**     | Harmony, Columbus exports, MetaXpress, ImageXpress HCS.ai (`.jdce`), IN Cell, CellVoyager, Opera Flex, scanR and Cellomics (`.C01`) plates; with `plates=True`, CZI plates, OME companion and OME-TIFF plates (Yokogawa CQ1) and ND2 files whose XY points are named after wells |
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
* Micro-Manager stacks (MMStack and NDTiff), also when one acquisition is spread over several files
* Whole-slide TIFFs: Aperio SVS, Hamamatsu NDPI, Leica SCN, Ventana / Roche BIF (VENTANA DP 200, iScan Coreo), PerkinElmer / Akoya QPTIFF and Philips

Micro-Manager stage positions are series. NDTiff channel names and pixel sizes come from the dataset's index and frame metadata.

A whole-slide file gives the slide first, then the other pictures it stores as further series, as Bio-Formats lists them: `label` and `macro` (overview), and for some formats `thumbnail` (QPTIFF) or `probability` (BIF tissue map). Pixel sizes, positions and channel names come from each vendor's own metadata.

Ventana BIF scans store overlapping tiles. VENTANA DP 200 scans are composed as Roche's description of the format places them. iScan Coreo scans, which that description does not cover, are composed as OpenSlide shows them, with one difference: each tile starts at the nearest whole pixel and keeps its stored pixels, where OpenSlide interpolates to a sub-pixel position. Areas outside the scanned regions read as 0. Layouts that were not checked on real files (other EncodeInfo versions, several focal planes, overlapping scan areas) raise an error.

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

Opening one of the TIFF files of the set also reads the whole set, as if the companion had been opened: `f.path` is then the companion, and the companion's metadata is used. To read only that file, with its pixel sizes, channels and position taken from the companion:

```python
f = micro_reader.open("data_t001.ome.tif", multifile=False)
```

If the companion is missing or does not list the file, opening the whole set raises an error that names `multifile=False`. With `multifile=False` the file is then read without pixel sizes, with a `CompanionWarning`. A TIFF whose metadata lives in another OME-TIFF (instead of a `.companion.ome`) works the same way, with that OME-TIFF in the companion's place.

A companion that describes a plate (`Plate` / `Well` / `WellSample`) reads as one with `plates=True`, as a CZI plate does: `f.plates` lists its plates and wells, and its images are indexed by plate, acquisition, well and field. RGB images in a companion have a sample axis (`t c z y x s`), as RGB OME-TIFFs do.

Some OME-TIFF sets keep the full OME-XML in their TIFF files instead of a companion. Opening one of those files also reads the whole set, and `multifile=False` reads only the planes stored in that file, with the set's metadata.

Micro-Manager stacks always open whole, also with `multifile=False`: they split their files by size, so one file does not hold a complete part of the image.

### CZI

CZI files support:

* mosaic tiles
* series
* views
* rotations
* RGB
* large files
* stored series pyramids

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

A multi-area tiling set is opened through its index, `.omp2info`, or its folder: one image per tile, each at the stage position the index gives it (the tiles' own files record none), or one canvas per group with `tiles="mosaic"`. Positions follow the protocol's grid, not a registration. The grid step was derived from one sample, as the backend documents.

### Olympus OIB and OIF

Older FluoView files come in two forms with the same content: an OIB is one file, an OIF is a `.oif` settings file plus a `.oif.files` folder with one TIFF per plane. Both open as one `t c z y x` image, with pixel sizes, z and time steps, channel names, colours and wavelengths. Opening any file in the `.oif.files` folder opens the OIF.

### Olympus VSI

A cellSens VSI is a `.vsi` file plus a `_<name>_` folder that holds the pixels (`stack1/frame_t.ets`, `stack10000/frame_t.ets`, ...). The images are listed as Bio-Formats lists them: one per scanned area, named as in cellSens, with the slide label (`label`) and overview (`overview`) when the scanner took them, and the slide photo (`macro`) last. Each scanned area keeps its resolution levels.

Images are `t c z y x` (`t c z y x s` for brightfield RGB scans), with pixel sizes, z and time steps, and each channel's name, colour and emission wavelength. Stage positions are not read.

Opening the `_<name>_` folder, a `stack` folder or a file in it opens the `.vsi`. An `.ets` file whose `.vsi` is missing is read alone, with a warning, when it is a plain 2-D image; its size is then rounded up to whole tiles.

### Imaris

Install Imaris support with:

```bash
pip install "micro-reader[hdf5]"
```

(`[ims]` is the extra's former name and still works.)

Imaris resolution level 0 is exposed at the actual image size, with Imaris chunk padding removed.

Chunked compression is decoded in parallel.

### HDF5: ilastik, BigDataViewer, Luxendo, Fiji

These need the `[hdf5]` extra as well.

* **ilastik**: every dataset with ilastik's `axistags` (its exports, its input files, steinbock's `img`) is one image, in the dataset's own axis order. Pixel sizes are read where ilastik recorded units (`axis_units`, since 2025-09).
* **BigDataViewer / BigStitcher**: the XML with its HDF5 file (also partitioned). One image per tile, angle and illumination, channels on `c` and time points on `t`, with the stored resolution levels. Opening the HDF5 file opens its XML.
* **Luxendo / Bruker light-sheet** (`.lux.h5`): one stack per file. A whole acquisition opens through the Imaris file Luxendo writes next to it.
* **Fiji HDF5** (the Vibez plugin): datasets with `element_size_um`; `t<n>/channel<n>` datasets make one `t c z y x` image.

### ICS / IDS

ICS 1.0 (an `.ics` header with an `.ids` pixel file; opening either opens the image) and ICS 2.0 (one file), as written by SVI Huygens and others. Gzip-compressed pixels are read without decompressing the whole file. Images that store rows bottom-up (`coordinates cartesian`) are returned top-down, as Bio-Formats does.

### MetaMorph .nd

A MetaMorph multi-dimensional acquisition: an `.nd` file and the TIFF / STK files named after it. One image per stage position (`series`), `t c z y x`, channels named by the wavelengths. Opening the `.nd` file, any of its files or its folder opens the whole set. The stage coordinates the files record are kept as `Metadata.stage_position`, not as a position: which point of the field they mark is not recorded.

### Micro-Manager 1.4

Image-per-file acquisitions (Micro-Manager 1.4 and the 2.0 betas): a folder per stage position with a `metadata.txt` and one TIFF per plane. One image per stage position, `t c z y x`, with channel names and colours, pixel sizes and the z step. Opening any file or folder of the acquisition opens the whole acquisition.

### Hamamatsu NDPIS

An `.ndpis` set lists one NDPI file per fluorescence channel. Opened, it is one slide image with one channel per file, at every stored resolution level; channels are named by their filter set. Each NDPI file still opens alone as an ordinary slide.

### DeltaVision

DeltaVision files (`.dv`, `.r3d`, `.r3d_d3d`, also deconvolved and projected ones) are recognised by their content, whatever their extension. The image keeps the file's own order of time, waves and z (for example `ctzyx`).

Pixel sizes, and each channel's emission and excitation wavelengths, are read from the file. Channel names come from the softWoRx acquisition log next to it (`<file>.log`); a deconvolved or projected file takes the names from the log of the raw file it was made from. Stage positions are not read.

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
