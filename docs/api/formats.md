# Formats

All supported formats use the same `File` and `Image` model, with consistent access to axes, indices, metadata and pixel data. This page summarises what each format supports and describes format-specific behaviour.

## Overview

| Format | Extensions | `backend` | `format` values | Indices you may see |
| --- | --- | --- | --- | --- |
| TIFF / BigTIFF | `.tif`, `.tiff`, `.btf`, `.lsm`, `.stk`, `.svs`, `.ndpi`, `.scn`, `.bif`, `.qptiff`, ... | `tiff` | `tiff`, `bigtiff`, `ome-tiff`, `imagej`, `lsm`, `stk`, `eer`, `svs`, `ndpi`, `scn`, `bif`, `qptiff`, `philips`, `ndtiff`, `micromanager`, `scanimage` | series, tile, view, phase, lifetime, sequence, resolution |
| Hamamatsu NDPIS | `.ndpis` | `ndpis` | `ndpis` | series, resolution |
| OME companion | `.companion.ome`, `.ome`, `.ome.xml` | `companion` | `ome-companion` | Plate indices for plates; `series` with `plates=False` |
| Zeiss CZI | `.czi` | `czi` | `czi` | series, tile, rotation, view, illumination, phase, block, resolution; plate indices by default, or `series` with `plates=False` |
| Nikon ND2 | `.nd2` | `nd2` | `nd2`, `nd2-legacy`, `nd2-run` | Plate indices for plates; `series` with `plates=False` |
| Leica | `.lif`, `.lof`, `.lifext`, `.xlif`, `.xlef`, `.xlcf` | `lif` | `lif`, `lof`, `lifext`, `xlif`, `xlef`, `xlcf` | series, tile, rotation, excitation |
| Olympus / Evident OIR | `.oir` | `oir` | `oir` | series |
| FluoView tiling sets | `.omp2info` | `omp2info` | `omp2info` | series, tile |
| Olympus OIB / OIF | `.oib`, `.oif` | `oif` | `oib`, `oif` | series |
| Olympus / Evident VSI | `.vsi` (+ `.ets`) | `vsi` | `vsi` | series, phase, emission, resolution |
| Imaris | `.ims` | `ims` | `ims` | series, resolution |
| HDF5: BigDataViewer / BigStitcher | `.xml` + `.h5` | `bdv` | `bdv-xml` | series, tile, view, illumination, resolution |
| HDF5: ilastik | `.h5` | `ilastik` | `ilastik` | series |
| Luxendo experiments | A folder containing `raw/` (or any part of it) | `luxendo` | `luxendo` | series, tile, view |
| HDF5: one Luxendo stack, Fiji | `.h5` | `hdf5` | `luxendo`, `fiji-hdf5` | series |
| ICS / IDS | `.ics`, `.ids` | `ics` | `ics` | series |
| MetaMorph | `.nd` | `metamorph` | `metamorph` | series |
| Micro-Manager 1.4 | `metadata.txt` | `micromanager` | `micromanager-files` | series |
| DeltaVision | `.dv`, `.r3d`, `.r3d_d3d`, any extension | `dv` | `dv` | series |
| MRC | `.mrc`, `.mrcs`, `.map`, `.st`, `.ali`, `.rec` | `mrc` | `mrc` | series |
| KLB | `.klb` | `klb` | `klb` | series |
| PNG / JPEG / BMP | `.png`, `.jpg`, `.jpeg`, `.bmp` | `raster` | `png`, `jpeg`, `bmp` | series |
| Harmony | `Index.idx.xml` | `harmony` | `harmony-v4`, `harmony-v5`, ... | plate, acquisition, well, field |
| Columbus | `ImageIndex.ColumbusIDX.xml` | `columbus` | `columbus` | plate, acquisition, well, field |
| MetaXpress | `.HTD` | `metaxpress` | `metaxpress` | plate, acquisition, well, field |
| ImageXpress HCS.ai | `.jdce` | `jdce` | `imagexpress-jdce` | plate, acquisition, well, field |
| IN Cell | `.xdce` | `incell` | `incell` | plate, acquisition, well, field, part |
| CellVoyager | `.mlf`, `.mrf`, `.wpi` | `cv7000` | `cv7000` | plate, acquisition, well, field, part |
| Opera Flex | `.flex` | `flex` | `opera-flex` | plate, acquisition, well, field |
| scanR | `experiment_descriptor.xml` | `scanr` | `scanr` | plate, acquisition, well, field |
| Cellomics | `.C01` | `cellomics` | `cellomics` | plate, acquisition, well, field |

A lambda scan (Leica, Olympus, OME-TIFF) is one image with its bands along `c`, so it adds no index (see [Lambda scans](guide/axes.md#lambda-scans)). Its bands remain an `emission` index only where they cannot be joined into one image, for example bands that store their own lower resolutions.

Format detection is based on file contents, so extensions are only a guide. Imaris and the other HDF5-based formats require the `[hdf5]` extra, which installs `h5py`. The older `[ims]` extra name is also still supported.

Plates opened as OME-TIFF (Yokogawa CQ1) or OME companion files report the TIFF or companion `backend`, respectively.

## Metadata by format

The table summarises which metadata fields are available through `image.metadata`, when the file records them.

| Format | Pixel sizes | Time step | Channel names / colours | Wavelengths | Position |
| --- | --- | --- | --- | --- | --- |
| OME-TIFF, companion | ✓ | ✓ | ✓ / ✓ | ✓ | ✓ |
| ImageJ TIFF | ✓ | ✓ | – / LUT | – | – |
| Zeiss LSM | ✓ | ✓ | ✓ / ✓ | Lambda scans | – |
| Whole-slide TIFF (SVS, SCN, BIF, QPTIFF, Philips) | ✓ | – | Some | – | SVS, SCN, QPTIFF |
| CZI | ✓ | ✓ | ✓ / ✓ | ✓ | ✓ |
| ND2 | ✓ | ✓ | ✓ / ✓ | ✓ | ✓ (only when a camera-to-stage matrix is available) |
| Leica | ✓ | ✓ | Dye names / LUT | Widefield emission, lambda scans | ✓ |
| OIR | ✓ | ✓ | ✓ / LUT | ✓ | Tiling sets (`.omp2info`) |
| OIB / OIF | ✓ | ✓ | ✓ / LUT | ✓ | – |
| VSI | ✓ | ✓ | ✓ / LUT | Emission | – |
| Imaris | ✓ | ✓ | ✓ / ✓ | ✓ | ✓ |
| DeltaVision | ✓ | ✓ | From the `.log` file / – | ✓ | – |
| MRC | ✓ (Å) | – | – | – | – |
| KLB | ✓ | ✓ | – | – | – |
| PNG / JPEG / BMP | – | – | – | – | – |
| Harmony | ✓ | ✓ | ✓ / – | ✓ | Relative to the well centre |
| MetaXpress | ✓ | – | ✓ / – | – | – |
| IN Cell | ✓ | – | ✓ / – | ✓ | – |
| CellVoyager | ✓ | ✓ | ✓ / ✓ | ✓ | Relative to the well centre |
| Columbus | ✓ | ✓ | ✓ / – | ✓ | Relative to the well centre |
| Opera Flex | ✓ | ✓ | ✓ / – | – | Relative to the well centre |
| ImageXpress HCS.ai | ✓ | ✓ | ✓ / – | – | Relative to the well centre |
| BigDataViewer / BigStitcher | ✓ | ✓ | – | – | ✓ (when views share one placement) |
| MetaMorph, Micro-Manager 1.4, ICS, NDPIS | ✓ | ✓ | ✓ | Some | – |

A dash (`–`) means either that the format does not store the value or that micro-reader does not currently read it. In either case, the value is left absent.

**Position** is a kind of metadata that describes where an image is located within a particular coordinate system, such as `stage`, `slide`, `well-B2` or `global`. To learn more about positions, see [Coordinate systems](reference/coordinates.md).

Positions are available in CZI, ND2, OME-TIFF and OME companion sets, Imaris, Leica, FluoView tiling sets, whole-slide TIFFs (SVS, SCN and QPTIFF), BigDataViewer, scanR, and the plate formats Harmony, Columbus, CellVoyager, Opera Flex and ImageXpress HCS.ai. The other formats do not provide positions or coordinate systems.

## TIFF

micro-reader supports the following TIFF variants:

- TIFF and BigTIFF.
- OME-TIFF, stored as a single file or as multiple files. See [OME companion sets](guide/multi-file.md#ome-companion-sets).
- ImageJ hyperstacks, Zeiss LSM, MetaMorph STK, ScanImage and EER.
- Micro-Manager stacks (MMStack and NDTiff), including acquisitions distributed across multiple files.
- Whole-slide TIFFs: Aperio SVS, Hamamatsu NDPI, Leica SCN, Ventana / Roche BIF (VENTANA DP 200 and iScan Coreo), PerkinElmer / Akoya QPTIFF and Philips.

With `axes=None`, TIFF images keep the axis order stored in the file (by default, every image is `t c z y x`). A dimension that the file does not name becomes the `sequence` index. Use `rename={"sequence": "z"}` (or `"t"` or `"c"`) to interpret it as an axis. See [Opening files](guide/opening.md#rename-assign-a-meaning-to-an-unnamed-dimension).

The following files are not yet supported and raise `UnsupportedFormatError`:

- Imaris 3 files (`.ims` files stored as TIFF).
- PicoQuant `.ptu`, `.phu` and `.sptw` files.
- FluoView overview maps.
- ICS 1.0 files whose pixels are Unix-compressed (`.ids.Z`). Uncompressed and gzip-compressed ICS files are supported.

Imaris 5 files use HDF5 and are supported. PerkinElmer Opera Flex files are read as plates; see [Plates](#plates).

**Micro-Manager.** Stage positions are represented as `series`. For NDTiff, channel names and pixel sizes come from the dataset index and frame metadata.

**Acquisitions stopped early.** A ScanImage or NDTiff acquisition may contain fewer planes than its metadata specifies if acquisition was stopped early. micro-reader reads the planes present in the file and returns zeros for the missing planes. It also raises a `MissingImagesWarning`.

**Pixel sizes.** For plain TIFF files, pixel sizes are derived from the resolution tags only when the `ResolutionUnit` is explicitly centimetres. Resolutions in inches, TIFF's default when the tag is absent, are a screen or print density on real files (72 or 96 dpi), not a microscope calibration, so they give no pixel size. A resolution of one pixel per unit, which some software writes as a placeholder, gives none either.

**Whole-slide files.** The order of images in a file varies by format.

- In VSI, the images follow the stack folders, as Bio-Formats lists them, and the slide photo (`macro`) comes last. See [VSI](#vsi).
- In SVS, NDPI, BIF and QPTIFF, the main scan comes first.
- Other images in the file appear as additional series, in the order reported by Bio-Formats. These can include `label`, `macro` (the overview), and, for some formats, `thumbnail` (QPTIFF) or `probability` (the BIF tissue map).
- In Leica SCN, the whole-slide overview is called `macro`. A fluorescence overview alongside it is called `macro fluorescence`.

Regions outside the scanned areas read as 0.

**Whole-slide tile placement.** Ventana BIF tiles overlap. VENTANA DP 200 scans are assembled according to Roche's description of the format. iScan Coreo scans follow the placement shown by OpenSlide, except that each tile starts at the nearest whole pixel.

Where tiles overlap, pixels come from the later tile: the tile to the right, or the tile below in an iScan scan. Pixel values are used unchanged; tiles are neither blended nor resampled.

Bio-Formats distributes the overlap of a DP 200 scan evenly across the tiles instead of using the overlap values stored in the file. As a result, its tile positions can differ from micro-reader's by several pixels. Both readers decode the same pixel values; only the tile placement differs.

**Codecs.** The following codecs are built in: LZW, Deflate, PackBits, LZMA, zstd, JPEG (8-bit and 12-bit, lossless, and NDPI variants through libjpeg-turbo), JPEG XR (jxrlib), JPEG 2000, WebP, JPEG XL, PNG and EER.

Install the optional codecs with:

```bash
pip install "micro-reader[codecs]"
```

This adds LERC, CCITT fax and other less common codecs. For region reads, only the strips or tiles overlapping the requested region are decoded.

## CZI

CZI support includes mosaic tiles, multiple series, views, rotations, illumination sides, phases, blocks, RGB images, files larger than 4 GB and stored series pyramids. RGB pixel types such as `Bgr24` are returned as RGB.

Opening a CZI file with `tiles="mosaic"` returns one image per series. Its tiles are placed on a shared canvas using the stitching information stored in the file. Tiles are not registered or blended, and gaps between them are filled with 0.

The older option `tiles="stitched"` still works but raises a warning. CZI files containing plates are opened as plates by default. Pass `plates=False` to read them as series instead.

CZI images provide two additional features.

### `image.origin` and `image.read_region()`

`image.origin` gives the `(y, x)` coordinates of the image corner in the file's pixel coordinate system. libCZI calls this the bounding box, and both libCZI and ZEN store its coordinates in `(x, y)` order.

Use `origin` with coordinates from the CZI XML or other CZI tools:

```python
f = micro_reader.open("S=2_2x2_CH=1.czi", tiles="mosaic")

im = f.images[0]

im.origin  # (174757, 134757): y, x

# A 100 x 50 rectangle in file coordinates,
# starting 10 pixels outside the series
r = im.read_region(y=174747, x=134747, height=50, width=100, c=0)

r.shape  # (1, 1, 50, 100): t z y x; c is dropped as in read()
```

The method signature is:

```python
image.read_region(*, y, x, height, width, **planes) -> numpy.ndarray
```

`planes` selects the other axes in the same way as `read()`. Parts of the requested rectangle outside the series are filled with 0, matching libCZI behaviour.

### `f.canvas()`

`f.canvas(**index)` returns one image spanning all series in the file. The series are placed on a shared canvas, with background values of 0 between them. The canvas covers the bounding box of all full-resolution tiles:

```python
cv = f.canvas()

cv.shape   # (1, 1, 1, 486, 90486): two series far apart on the slide
cv.origin  # (174757, 134757)

cv.read(c=0, x=slice(0, 486)).shape
```

If the file contains multiple rotations or views, specify `rotation=` or `view=` because these represent different frames:

```python
f.canvas(view=0)
```

### Damaged files

A CZI subblock directory may no longer match the data, for example after an interrupted or corrupted copy. In this case, micro-reader searches for the actual position of each subblock and checks it against the subblock's own header before reading it.

The reader raises a `DamagedFileWarning`. The `f.damage` attribute describes the problems found:

```python
f.damage

# {} for a sound file; otherwise a dictionary that may include:
# 'first_bad_offset', 'misplaced', 'recovered', 'shifts',
# 'unrecoverable', 'series_affected', 'series_unreadable'
```

Subblocks that cannot be located raise an error when read. Other readers may fail to open such files or return incorrect pixels, so check the file or copy it again. A damaged file is read without its stored pyramid.

## ND2

- ND2 version 3 and later are supported. Legacy JPEG 2000 ND2 files (versions 1 and 2) are also supported with `format="nd2-legacy"`.
- Each XY position is represented as a `series` and exposed as one image with `t c z y x` axes. RGB images are returned as three channels.
- RGB samples are returned in red, green, blue order. Nikon colour cameras store them as blue, green, red, and the file says so in its camera metadata. `image.sample_order` gives the order the file stores. A file that does not state the order, such as a legacy ND2 file, is read as stored, with a warning.
- An ND2 file whose XY points are named after wells (for example, `A10_0000` and `B02_0003`) is opened as a plate.
- A folder containing ND2 files from one NIS-Elements well-plate run is also opened as a plate, whether there is one file per well or one per point. Opening any file in the folder opens the plate. Pass `multifile=False` to read only the specified file.

## Leica

Supported Leica formats are LIF, LOF, LIFEXT, XLIF, XLEF and XLCF.

The `.xlef`, `.xlcf` and `.xlif` files can refer to other files using relative paths. These paths are resolved even when the path case differs from the original Windows paths.

Leica projects can contain nested analysis results, tiles, rotations, and emission or excitation scans. These are represented using the usual images and indices.

Tile scans can be placed on one canvas per scan with `tiles="mosaic"`. Each tile is placed at its stage position, without registration.

Channel names and widefield emission wavelengths are read from the image's hardware settings. A lambda scan is one image with its bands along `c`, each channel named with its wavelength, for example `"HyD S 1 420 nm"` (see [Lambda scans](guide/axes.md#lambda-scans)). Excitation scans remain an `excitation` index.

## OIR

Olympus / Evident FluoView OIR files are read together with their continuation files (`name_00001`, ...).

A spectral (lambda) scan is one image with its bands along `c`. Each channel has its band's `emission_wavelength`, and its name ends with the wavelength, for example `"CH1 420 nm"`. With `axes=None`, the bands are an `l` axis instead. See [Lambda scans](guide/axes.md#lambda-scans).

**Tiling sets.** A multi-area tiling set can be opened using its index file (`.omp2info`) or its folder. Each tile is exposed as a separate image at the stage position recorded in the index. Alternatively, use `tiles="mosaic"` to get one canvas per group.

Tile positions follow the acquisition protocol's grid. They are not the result of image registration.

## OIB and OIF

Older FluoView files come in two forms with the same image content:

- An OIB is a single file.
- An OIF consists of a `.oif` settings file and an `.oif.files` folder containing one TIFF file per plane.

Both formats open as one `t c z y x` image and provide pixel sizes, z and time steps, channel names, colours and wavelengths.

Opening any file inside an `.oif.files` folder opens the corresponding OIF. A lambda (spectral) scan is one image with its bands along `c`, as for OIR.

## VSI

A cellSens VSI dataset consists of a `.vsi` file and a `_<name>_` folder containing the image data (for example, `stack1/frame_t.ets`).

Images are listed in the same order as Bio-Formats:

1. One image per stack folder, in stack order: the scanned areas, named as in cellSens, and the slide label (`label`) and overview (`overview`) when the scanner took them.
2. The slide photograph (`macro`), last, if present.

Each scanned area retains its resolution levels.

Brightfield RGB scans are returned as three channels. Pass `samples="keep"` to retain the `s` axis instead.

You can open the `.vsi` file, its `_<name>_` folder, a `stack` folder or a file inside one of these folders. All resolve to the same dataset.

Tiles outside the scanned area are not stored. Reads in these regions return the slide's background value.

An `.ets` file whose corresponding `.vsi` file is missing can be opened on its own, with a warning, when it contains a plain 2-D image. Its dimensions are then rounded up to whole tiles.

## Imaris

Imaris support requires the HDF5 extra:

```bash
pip install "micro-reader[hdf5]"
```

All stored resolution levels are available. Level 0 is returned at the actual image dimensions, without Imaris chunk padding. Chunks are decoded in parallel.

## HDF5: ilastik, BigDataViewer, Luxendo and Fiji

These formats also require the `[hdf5]` extra.

**ilastik.** Each dataset containing ilastik `axistags` is exposed as one image in the dataset's own axis order. This includes ilastik exports and input files. Pixel sizes are read when ilastik has recorded units.

**BigDataViewer / BigStitcher.** The XML file is read together with its HDF5 data file, which may itself be partitioned. Each tile, view and illumination is exposed as a separate image. Channels use the `c` axis, time points use `t`, and stored resolution levels are available.

Opening the HDF5 file also opens its XML file. Positions and registrations are described in [Coordinate systems](reference/coordinates.md).

With `tiles="mosaic"`, the tiles of each angle and illumination are placed on one canvas at their nominal positions (the acquisition grid or tile configuration, not a stitching result), rounded to whole pixels, with every stored level. Nothing is blended, and the later tile supplies the pixel where tiles overlap. Tiles that cannot be placed this way, such as turned, sheared or mirrored views, tiles apart along `z` or tiles of different pixel sizes, are refused with a message that says why.

**Luxendo / Bruker light-sheet.** A Luxendo experiment is represented by a folder. Its `raw/` directory contains one folder per stack, channel and objective. Each folder contains one `.lux.h5` stack per camera and time point.

The experiment is exposed as one image per stack, objective and camera. Time points use the `t` axis and channels use `c`. Channels are ordered by Luxendo channel number and named using their descriptions.

Opening any part of the experiment opens the entire dataset. You can open the experiment folder, `raw/`, a channel folder, a `.lux.h5` file or its `.json` file. You can also open a link file alongside `raw/`, such as `main_raw.lux.h5`, `imaris.ims`, `ims/stacks/*.ims` or `bdv.xml`.

Additional details:

- **Where pixels come from.** Pixel data is always read from `raw/`. Link files only describe which files should exist. If a file listed in a link file is missing from `raw/`, its pixels are returned as zeros and a `MissingImagesWarning` is raised. Without a link file, only gaps in the time-point numbering can be detected.
- **File validation.** When first read, each `.lux.h5` file is checked against its expected position in the experiment. This includes its time point, channel, size and data type. A mismatch raises `CorruptFileError` and identifies the file.
- **Indices.** When an experiment contains multiple stacks or cameras, `tile` identifies the stack and `view` identifies the objective and camera.
- **Reading individual files.** Pass `multifile=False` to open one `.lux.h5` file on its own. It is then exposed as one image with `t` and `c` sizes of 1. The same option opens an `imaris.ims` file as a regular Imaris file. A `.lux.h5` file outside an experiment folder is also read on its own. The `processed/` folder is not part of the experiment, so its files are opened individually.

**Fiji HDF5.** Datasets written by the Vibez plugin are supported when they contain `element_size_um`. Datasets named `t<n>/channel<n>` are combined into one `t c z y x` image.

## ICS and IDS

ICS 1.0 and ICS 2.0 are supported, including files written by SVI Huygens and other software.

- ICS 1.0 consists of an `.ics` header and an `.ids` pixel-data file. Opening either file opens the image.
- ICS 2.0 stores the data in a single file.

## MetaMorph

A MetaMorph multidimensional acquisition consists of an `.nd` file and the TIFF / STK files it references.

Each stage position is exposed as one `series`, with a `t c z y x` image. Channel names are derived from their wavelengths.

Opening the `.nd` file, any of its associated files or the containing folder opens the entire acquisition.

## Micro-Manager 1.4

These acquisitions store each image in a separate file. Each stage position has its own folder, containing a `metadata.txt` file and one TIFF file per plane.

Each stage position is exposed as one image, with channel names, colours, pixel sizes and z-step information.

Opening any file or folder within the acquisition opens the complete acquisition. Micro-Manager stacks stored as TIFF are covered under [TIFF](#tiff).

## Hamamatsu NDPIS

An `.ndpis` set references one NDPI file per fluorescence channel.

The set opens as one slide image, with one channel per file and all stored resolution levels available. Each NDPI file can also be opened on its own as a regular slide image.

## DeltaVision

DeltaVision files (`.dv`, `.r3d`, `.r3d_d3d`), including deconvolved and projected images, are recognised from their contents regardless of their extension.

The image retains the order of time points, waves and z-planes stored in the file.

Channel names are read from the softWoRx log file alongside the image (`<file>.log`). For deconvolved or projected files, channel names are taken from the log of the original raw file.

## KLB

Keller Lab Block files support bzip2, zlib and uncompressed blocks, including all time points and channels.

A read decodes only the blocks it overlaps, and decoding is parallelised.

Pixel sizes and time steps are read from the header, in micrometres and seconds. A header value of 1 on every axis is the default and does not represent recorded physical sizes.

KLB files have no magic number, so the `.klb` extension is used to identify them.

## MRC

MRC modes 0, 1, 2, 4, 6 and 12 are supported, in both byte orders and with extended headers.

Pixel sizes are expressed in ångström.

MRC stores the Y axis bottom-up. micro-reader flips the rows so that the image is returned top-down, matching the orientation used by other readers.

## JPEG, PNG and BMP

Supported raster features include:

- **PNG:** All colour types and bit depths, including interlaced images.
- **BMP:** Palettes, RLE compression, bit fields and OS/2 headers.
- **JPEG:** Baseline, progressive and lossless images.

Raster images are decoded in full on the first read and kept in memory until the file is closed. For images larger than 256 MB, non-interlaced PNG files are read by rows, and JPEG files with restart markers are read in row bands.

**Palettes.** Greyscale palettes are applied so that the returned pixels contain the corresponding grey values. Colour palettes are not applied. For these images, the pixel values are palette indices, and `image.palette` contains the colours in an `(n, 3)` or `(n, 4)` `uint8` array.

Images without a colour palette have `image.palette = None`.

You can apply a colour palette to the returned indices as follows:

```python
im = f.images[0]

idx = im.read()
rgb = im.palette[idx] if im.palette is not None else idx
```

## Plates

Plate formats open as a `File` with `f.plates`. The [Plates](guide/plates.md) guide describes the plate hierarchy, selection by well name, row and column, and format-specific behaviour.

Supported formats include:

- **Harmony and Columbus** (PerkinElmer). Open `Index.idx.xml` or `ImageIndex.ColumbusIDX.xml`. Other index files and the export folders can also be opened.
- **MetaXpress / ImageXpress.** Open the `.HTD` file. Z-series and time points are included.
- **ImageXpress HCS.ai.** Open the `.jdce` file together with its `image_metadata_*.csv` files. Projections, z-stacks and stitched montages are supported.
- **IN Cell** (`.xdce`) and **CellVoyager** (CV7000 / CV8000: `.wpi`, `.mlf`, `.mrf`).
- **Opera Flex** (`.flex`). Open any `.flex` file or its containing folder. Each barcode represents one plate.
- **scanR** (Olympus / Evident). Open `experiment_descriptor.xml`, one of its folders or a data file. Older scanR plate formats are also supported.
- **Cellomics** (`.C01`). Open any `.C01` file, its folder or the database. The plate layout is inferred, assuming at least 96 wells.
- **Plates in generic formats.** These include Zeiss CZI plates, OME companion plates, OME-TIFF plates (Yokogawa CQ1), ND2 files whose XY positions are named after wells, and folders of ND2 files from one NIS-Elements well-plate run. Pass `plates=False` to read these by series instead.