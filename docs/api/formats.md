# Formats

Every format opens as a `File` of `Image`s, with the same axes, indices, metadata and reading. This page lists what each format gives you and the few extras that some formats add.

## Overview

| Format | Extensions | `backend` | `format` values | Indices you may see |
| --- | --- | --- | --- | --- |
| TIFF / BigTIFF | `.tif`, `.tiff`, `.btf`, `.lsm`, `.stk`, `.svs`, `.ndpi`, `.scn`, `.bif`, `.qptiff`, ... | `tiff` | `tiff`, `bigtiff`, `ome-tiff`, `imagej`, `lsm`, `stk`, `eer`, `svs`, `ndpi`, `scn`, `bif`, `qptiff`, `philips`, `ndtiff`, `micromanager`, `scanimage` | series, tile, view, phase, emission, lifetime, sequence, resolution |
| Hamamatsu NDPIS | `.ndpis` | `ndpis` | `ndpis` | series, resolution |
| OME companion | `.companion.ome`, `.ome`, `.ome.xml` | `companion` | `ome-companion` | plate indices when it is a plate (`plates=False`: series) |
| Zeiss CZI | `.czi` | `czi` | `czi` | series, tile, rotation, view, illumination, phase, block, resolution; plate indices (by default; `plates=False`: by series) |
| Nikon ND2 | `.nd2` | `nd2` | `nd2`, `nd2-legacy`, `nd2-run` | plate indices when it is a plate (`plates=False`: series) |
| Leica | `.lif`, `.lof`, `.lifext`, `.xlif`, `.xlef`, `.xlcf` | `lif` | `lif`, `lof`, `lifext`, `xlif`, `xlef`, `xlcf` | series, tile, rotation, emission, excitation |
| Olympus / Evident OIR | `.oir` | `oir` | `oir` | series, emission |
| FluoView tiling sets | `.omp2info` | `omp2info` | `omp2info` | series, tile |
| Olympus OIB / OIF | `.oib`, `.oif` | `oif` | `oib`, `oif` | series, emission |
| Olympus / Evident VSI | `.vsi` (+ `.ets`) | `vsi` | `vsi` | series, phase, emission, resolution |
| Imaris | `.ims` | `ims` | `ims` | series, resolution |
| HDF5: BigDataViewer / BigStitcher | `.xml` + `.h5` | `bdv` | `bdv-xml` | series, tile, view, illumination, resolution |
| HDF5: ilastik | `.h5` | `ilastik` | `ilastik` | series |
| Luxendo experiments | a folder with `raw/` (any part) | `luxendo` | `luxendo` | series, tile, view |
| HDF5: one Luxendo stack, Fiji | `.h5` | `hdf5` | `luxendo`, `fiji-hdf5` | series |
| ICS / IDS | `.ics`, `.ids` | `ics` | `ics` | series |
| MetaMorph | `.nd` | `metamorph` | `metamorph` | series |
| Micro-Manager 1.4 | `metadata.txt` | `micromanager` | `micromanager-files` | series |
| DeltaVision | `.dv`, `.r3d`, `.r3d_d3d`, any | `dv` | `dv` | series |
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

Detection is by content, so the extensions are only a guide. Imaris and the other HDF5 formats need the `[hdf5]` extra (`h5py`); `[ims]` is its older name and still works. Plates opened as OME-TIFF (Yokogawa CQ1) or OME companion files report the TIFF or companion `backend`.

## Metadata per format

What `image.metadata` holds for each format, when the file records it:

| Format | Pixel sizes | Time step | Channel names / colours | Wavelengths | Position |
| --- | --- | --- | --- | --- | --- |
| OME-TIFF, companion | ✓ | ✓ | ✓ / ✓ | ✓ | ✓ |
| ImageJ TIFF | ✓ | ✓ | – / LUT | – | – |
| Zeiss LSM | ✓ | ✓ | ✓ / ✓ | – | – |
| Whole-slide TIFF (SVS, SCN, BIF, QPTIFF, Philips) | ✓ | – | some | – | SVS, SCN, QPTIFF |
| CZI | ✓ | ✓ | ✓ / ✓ | ✓ | ✓ |
| ND2 | ✓ | ✓ | ✓ / ✓ | ✓ | ✓ (only for files with a camera-to-stage matrix) |
| Leica | ✓ | ✓ | dye names / LUT | – | ✓ (not for tiles) |
| OIR | ✓ | ✓ | ✓ / LUT | ✓ | tiling sets (`.omp2info`) |
| OIB / OIF | ✓ | ✓ | ✓ / LUT | ✓ | – |
| VSI | ✓ | ✓ | ✓ / LUT | emission | – |
| Imaris | ✓ | ✓ | ✓ / ✓ | ✓ | ✓ |
| DeltaVision | ✓ | ✓ | from the `.log` / – | ✓ | – |
| MRC | ✓ (Å) | – | – | – | – |
| KLB | ✓ | ✓ | – | – | – |
| PNG / JPEG / BMP | – | – | – | – | – |
| Harmony | ✓ | ✓ | ✓ / – | ✓ | relative to the well centre |
| MetaXpress | ✓ | – | ✓ / – | – | – |
| IN Cell | ✓ | – | ✓ / – | ✓ | – |
| CellVoyager | ✓ | ✓ | ✓ / ✓ | ✓ | relative to the well centre |
| Columbus | ✓ | ✓ | ✓ / – | ✓ | relative to the well centre |
| Opera Flex | ✓ | ✓ | ✓ / – | – | relative to the well centre |
| ImageXpress HCS.ai | ✓ | ✓ | ✓ / – | – | relative to the well centre |
| BigDataViewer / BigStitcher | ✓ | ✓ | – | – | ✓ (where the views share one placement) |
| MetaMorph, Micro-Manager 1.4, ICS, NDPIS | ✓ | ✓ | ✓ | some | – |

A "–" means micro-reader does not read it for that format, or the format does not store it. In both cases the value is absent, never invented.

A position names the coordinate system it is in (`stage`, `slide`, `well-B2`, `global`; see [Coordinate systems](reference/coordinates.md)). The formats that give one are CZI, ND2, OME-TIFF and OME companion sets, Imaris, Leica, FluoView tiling sets, whole-slide TIFFs (SVS, SCN, QPTIFF), BigDataViewer and the plate formats. The other formats give no position and no coordinate system.

## TIFF

Supported TIFF variants:

- TIFF and BigTIFF;
- OME-TIFF, single-file and multi-file (see [Multi-file images](guide/multi-file.md#ome-companion-sets));
- ImageJ hyperstacks, Zeiss LSM, MetaMorph STK, ScanImage, EER;
- Micro-Manager stacks (MMStack and NDTiff), also when one acquisition is spread over several files;
- whole-slide TIFFs: Aperio SVS, Hamamatsu NDPI, Leica SCN, Ventana / Roche BIF (VENTANA DP 200 and iScan Coreo), PerkinElmer / Akoya QPTIFF and Philips.

TIFF images keep the file's axis order. A dimension the file does not name becomes the `sequence` index. Use `rename={"sequence": "z"}` (or `"t"`, `"c"`) to make it an axis (see [Opening files](guide/opening.md#rename-say-what-an-unnamed-dimension-is)).

Not supported yet, and refused with `UnsupportedFormatError`: Imaris 3 files (`.ims` stored as a TIFF; Imaris 5 files, which are HDF5, are supported), PicoQuant `.ptu`, FluoView overview maps and compressed ICS. The full eubi-bridge converts Imaris 3 through Bio-Formats. PerkinElmer Opera Flex files are read as plates, see [Plates](#plates) below.

**Micro-Manager.** Stage positions are series. NDTiff channel names and pixel sizes come from the dataset's index and frame metadata.

**Acquisitions stopped early.** When a file holds fewer planes than its metadata announces (a ScanImage or NDTiff acquisition that was stopped), the planes it holds are read and the others read as zeros, with a `MissingImagesWarning`.

**Pixel sizes.** A plain TIFF's pixel size comes from its resolution tags in inches or centimetres. A resolution of 1 pixel per unit is a placeholder some programs write, so it gives no pixel size.

**Whole-slide files.** Images are in the file's order, which is not the same in every format: in a VSI the label and overview come first, then the scans. The main scan of an SVS, NDPI, BIF or QPTIFF comes first, then the other pictures the file stores, as further series and in the order Bio-Formats lists them: `label`, `macro` (overview), and for some formats `thumbnail` (QPTIFF) or `probability` (BIF tissue map). In a Leica SCN, the overview of the whole slide is called `macro` (a fluorescence overview next to it, `macro fluorescence`). Ventana BIF tiles overlap. VENTANA DP 200 scans are composed the way Roche's description of the format places them; iScan Coreo scans as OpenSlide shows them, except that each tile starts at the nearest whole pixel. Areas outside the scanned regions read as 0.

**Codecs.** Built in: LZW, Deflate, PackBits, LZMA, zstd, JPEG, JPEG 2000, WebP, JPEG XL, PNG and EER. With `pip install "micro-reader[codecs]"`: JPEG XR, LERC and others. Only the strips or tiles under a region are decoded.

## CZI

CZI support covers mosaic tiles, series, views, rotations, illumination sides, phases, blocks, RGB (`Bgr24` and similar, given as RGB), files over 4 GB and stored series pyramids.

Opening with `tiles="mosaic"` gives one image per series, with its tiles placed on one canvas by the file's stitching information. Nothing is registered or blended, and the gaps between tiles are 0. (`tiles="stitched"`, the older name, still works but warns.) CZI files that are plates read as plates (the default; `plates=False` reads them by series).

CZI images have two extras.

### `image.origin` and `image.read_region()`

`image.origin` is the `(x, y)` of the image's corner in the file's own pixel coordinates (what libCZI calls the bounding box). Use it with coordinates from the CZI's XML or from other CZI tools:

```python
f = micro_reader.open("S=2_2x2_CH=1.czi", tiles="mosaic")
im = f.images[0]
im.origin                    # (134757, 174757)

# a 100 x 50 rectangle in file coordinates, starting 10 px outside the series
r = im.read_region(134747, 174747, 100, 50, c=0)
r.shape                      # (1, 1, 50, 100): t z y x, c dropped as in read()
```

```python
image.read_region(x, y, width, height, **planes) -> numpy.ndarray
```

`planes` selects the other axes as in `read()`. Parts of the rectangle outside the series are 0, as libCZI returns them.

### `f.canvas()`

`f.canvas(**index)` returns one image covering every series of the file, placed on a shared canvas with background (0) between the series. It is the box around every full-resolution tile:

```python
cv = f.canvas()
cv.shape     # (1, 1, 1, 486, 90486): two series far apart on the slide
cv.origin    # (134757, 174757)
cv.read(c=0, x=slice(0, 486)).shape
```

A file with several rotations or views needs `rotation=` or `view=`, because those are different frames:

```python
f.canvas(view=0)
```

### Damaged files

If a CZI's subblock directory does not match its data (for example after a broken copy), micro-reader finds the real position of each subblock, checks it against the subblock's own header, and reads the file anyway. You get a `DamagedFileWarning`, and `f.damage` describes what was found:

```python
f.damage
# {} for a sound file; otherwise a dict with, among others:
# 'first_bad_offset', 'misplaced', 'recovered', 'shifts',
# 'unrecoverable', 'series_affected', 'series_unreadable'
```

Subblocks that cannot be found raise an error when read. Other readers fail on such files or return wrong pixels, so check or copy the file again. A damaged file is read without its stored pyramid.

## ND2

- ND2 version 3 and later, and legacy JPEG 2000 ND2 (versions 1 and 2, `format="nd2-legacy"`).
- One image per XY position (`series`), with `t c z y x` axes. RGB images read as three channels.
- A file whose XY points are named after wells (`A10_0000`, `B02_0003`) reads as a plate. So does a folder of ND2 files from one NIS-Elements well-plate run (one file per well or per point): open any of its files. `multifile=False` reads the one file.

## Leica

LIF, LOF, LIFEXT, XLIF, XLEF and XLCF. `.xlef` / `.xlcf` / `.xlif` files refer to other files by relative paths, which are resolved even when they were written on Windows with different letter case. Leica projects can contain nested analysis results, tiles, rotations, and emission or excitation scans. These all map to the usual images and indices. Tile scans can be placed on one canvas per scan with `tiles="mosaic"`, each tile at its stage position and nothing registered. Channel names and widefield emission wavelengths come from the image's hardware setting.

## OIR

Olympus / Evident FluoView OIR files with their continuation files (`name_00001`, ...). Spectral (lambda) scans give one image per band, indexed by `emission`, with the band's wavelength in `metadata.index_values["emission"]`.

**Tiling sets.** A multi-area tiling set opens through its index (`.omp2info`) or its folder: one image per tile, each at the stage position the index gives it, or one canvas per group with `tiles="mosaic"`. Positions follow the protocol's grid, not a registration.

## OIB and OIF

Older FluoView files come in two forms with the same content. An OIB is one file. An OIF is a `.oif` settings file plus a `.oif.files` folder with one TIFF per plane. Both open as one `t c z y x` image with pixel sizes, z and time steps, channel names, colours and wavelengths. Opening any file in the `.oif.files` folder opens the OIF. Lambda (spectral) scans give one image per band, indexed by `emission`.

## VSI

A cellSens VSI is a `.vsi` file plus a `_<name>_` folder holding the pixels (`stack1/frame_t.ets`, ...). Images are listed as Bio-Formats lists them: one per scanned area, named as in cellSens, then the slide label (`label`) and overview (`overview`) when the scanner took them, and the slide photo (`macro`) last. Each scanned area keeps its resolution levels. Brightfield RGB scans have three channels (or the `s` axis with `samples="keep"`). Opening the `_<name>_` folder, a `stack` folder or a file in it opens the `.vsi`. Tiles that the scan does not store, outside the scanned area, read as the slide's background value.

An `.ets` file whose `.vsi` is missing is read alone, with a warning, when it is a plain 2-D image. Its size is then rounded up to whole tiles.

## Imaris

Install with `pip install "micro-reader[hdf5]"`. All stored resolution levels are available. Level 0 is given at the real image size, without Imaris chunk padding. Chunks are decoded in parallel.

## HDF5: ilastik, BigDataViewer, Luxendo, Fiji

These need the `[hdf5]` extra too.

- **ilastik:** every dataset with ilastik's `axistags` (its exports and input files) is one image, in the dataset's own axis order. Pixel sizes are read where ilastik recorded units.
- **BigDataViewer / BigStitcher:** the XML with its HDF5 file (also partitioned). One image per tile, view and illumination, channels on `c`, time points on `t`, with the stored resolution levels. Opening the HDF5 file opens its XML. Positions and registrations are described in [Coordinate systems](reference/coordinates.md).
- **Luxendo / Bruker light-sheet:** an experiment is a folder. Its `raw/` holds one folder per stack, channel and objective, each with one `.lux.h5` stack per camera and time point. The experiment opens as one image per stack, objective and camera, with the time points on `t` and the channels on `c`, ordered by Luxendo channel number and named by their description. Opening any part opens the whole experiment: the folder, `raw/`, a channel folder, a `.lux.h5` or its `.json`, or a link file next to `raw/` (`main_raw.lux.h5`, `imaris.ims`, `ims/stacks/*.ims`, `bdv.xml`).
    - The pixels always come from `raw/`. The link files only say which files should exist; one they name that `raw/` lacks reads as zeros, with a `MissingImagesWarning`. Without a link file, only gaps in the time numbering are found.
    - Each `.lux.h5` is checked against its place when it is first read (its time point, channel, size and type); one that does not fit raises `CorruptFileError` naming it.
    - Several stacks or cameras: the index `tile` (stack) and `view` (objective and camera).
    - `multifile=False` reads one `.lux.h5` alone (one image, `t` and `c` of 1), or an `imaris.ims` as the Imaris file it is. A `.lux.h5` outside an experiment folder reads alone too. `processed/` is not part of the experiment: its files open alone.
- **Fiji HDF5** (the Vibez plugin): datasets with `element_size_um`; `t<n>/channel<n>` datasets make one `t c z y x` image.

## ICS and IDS

ICS 1.0 (an `.ics` header with an `.ids` pixel file; opening either opens the image) and ICS 2.0 (one file), as written by SVI Huygens and others.

## MetaMorph

A MetaMorph multi-dimensional acquisition: an `.nd` file and the TIFF / STK files named after it. One image per stage position (`series`), `t c z y x`, channels named by their wavelengths. Opening the `.nd` file, any of its files or its folder opens the whole set.

## Micro-Manager 1.4

Image-per-file acquisitions: a folder per stage position with a `metadata.txt` and one TIFF per plane. One image per stage position, with channel names, colours, pixel sizes and the z step. Opening any file or folder of the acquisition opens the whole acquisition. (Micro-Manager stacks stored as TIFF are under [TIFF](#tiff).)

## Hamamatsu NDPIS

An `.ndpis` set lists one NDPI file per fluorescence channel. Opened, it is one slide image with one channel per file, at every stored resolution level. Each NDPI file still opens alone as an ordinary slide.

## DeltaVision

`.dv`, `.r3d`, `.r3d_d3d`, including deconvolved and projected files, recognised by content whatever the extension. The image keeps the file's order of time, waves and z. Channel names come from the softWoRx log next to the file (`<file>.log`). A deconvolved or projected file takes them from the log of the raw file it was made from.

## KLB

Keller Lab Block files, bzip2, zlib or stored, with every time point and channel. A read decodes only the blocks it touches, in parallel. Sizes come from the header, in µm and seconds; a header holding 1 on every axis, its default, records none. The extension identifies the format, since a KLB has no magic number.

## MRC

MRC modes 0, 1, 2, 4, 6 and 12, both byte orders, with extended headers. Pixel sizes are in ångström. MRC stores Y bottom-up, so rows come top-down, flipped from their order in the file, as other readers give them.

## JPEG, PNG and BMP

- PNG: all colour types and bit depths, interlaced too.
- BMP: palettes, RLE, bit fields, OS/2 headers.
- JPEG: baseline, progressive and lossless.

Raster images are decoded whole on the first read and kept until the file is closed. Above 256 MB, non-interlaced PNGs are read by rows and JPEGs with restart markers by bands of rows.

**Palettes.** A palette that holds only greys is applied, so the pixels are the grey values. A colour palette is not applied: the pixels are indices and `image.palette` holds the colours as an `(n, 3)` or `(n, 4)` `uint8` array. Images without a colour palette have `image.palette = None`.

```python
im = f.images[0]
idx = im.read()
rgb = im.palette[idx] if im.palette is not None else idx
```

## Plates

Plate formats open as a `File` with `f.plates`; see [Plates](guide/plates.md) for the layout, selecting by well name, row and column, and each format's details.

- **Harmony and Columbus** (PerkinElmer): open `Index.idx.xml` or `ImageIndex.ColumbusIDX.xml` (the other index files and the export's folders work too).
- **MetaXpress / ImageXpress:** open the `.HTD`, including z-series and time points.
- **ImageXpress HCS.ai** (`.jdce` with `image_metadata_*.csv`): projections, z-stacks and stitched montages.
- **IN Cell** (`.xdce`) and **CellVoyager** (CV7000 / CV8000: `.wpi`, `.mlf`, `.mrf`).
- **Opera Flex** (`.flex`): any `.flex` file, or its folder; one plate per barcode.
- **scanR** (Olympus / Evident): the `experiment_descriptor.xml`, the folders or a data file; older scanR plates too.
- **Cellomics** (`.C01`): any `.C01` file, the folder or the database. The layout is guessed, at least 96 wells.
- **Plates in generic formats** (`plates=False` reads them by series): Zeiss CZI plates, OME companion plates, OME-TIFF plates (Yokogawa CQ1), ND2 files whose XY points are named after wells, and folders of ND2 files from one NIS-Elements well-plate run.
