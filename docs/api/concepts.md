# Concepts

Microscopy formats organise image data in different ways. A Zeiss CZI file may contain several series, each made of mosaic tiles. A light-sheet file may record each tile from two illumination sides. A screening plate may contain hundreds of wells, each imaged at several sites.

micro-reader represents all these structures using three concepts: a **file**, the **images** it contains, and an **index** that identifies each image. Once you understand these concepts, you can work with different formats in the same way.

```text
File                       one opened file (or set of files)
├── images[0]  Image       one array you can read
│   ├── index              where it comes from: {"series": 0, "tile": 3, ...}
│   ├── axes, shape, dtype the array: "tczyx", (1, 3, 41, 1000, 1500), uint16
│   ├── metadata           pixel sizes, position and coordinate system, channels
│   ├── resolutions        lower-resolution versions, if available
│   └── read(...) / [...]  read a region
├── images[1]  Image
├── ...
├── plates                 plate layouts, for screening files
├── coordinate_systems     spaces in which image positions are defined
└── collection             the file's hierarchy: plate > well > field, or series
```

## Files

Open a file with `micro_reader.open()`. micro-reader identifies the format primarily from the file's contents, using the filename when the contents alone are not sufficient. This means files can often be read even if they have unusual extensions. For example, an Olympus VSI file saved with a `.tif` extension is still recognised as VSI.

The result is a `File` object with the same interface regardless of the format:

```python
>>> f = micro_reader.open("MouseBrain_41Slices_2x2Tiles_3Channels_2Illuminations_1Angle.czi")
>>> f.backend          # the reader handling the file
'czi'

>>> len(f.images)      # number of images in the file
8
```

Some images are stored across multiple files. Examples include an Olympus OIF with its folder of TIFF planes, a DeltaVision image with its log file, and a plate consisting of thousands of plane files. Opening any file in such a set opens the complete image. You do not need to work out which file contains which data.

See [Multi-file images](guide/multi-file.md).

## Images and their axes

An image behaves much like a NumPy array, but its pixels are read from the file when requested. Its dimensions are described by **axes**. The five standard axes are:

- `t`: time
- `c`: channel
- `z`: focal plane
- `y`: row
- `x`: column

RGB images, such as photographs and brightfield slides, are represented as three channels (red, green and blue), so they use the same five axes. If you open an image with `samples="keep"`, the colour samples are instead represented by a sixth axis, `s`, matching the way they are stored in the file.

```python
>>> image = f.images[0]
>>> image.axes
'tczyx'

>>> image.shape
(1, 3, 41, 1000, 1500)
```

Read pixels using NumPy-style indexing or by specifying axes in `read()`:

```python
>>> image[0, :, 5, 100:200, 100:200].shape
(3, 100, 100)

>>> image.read(t=0, z=5, y=slice(100, 200), x=slice(100, 200)).shape
(3, 100, 100)
```

Unlike a NumPy array, an image does not keep its pixels in memory. Each read accesses the file and decodes only the requested region. This allows you to work with images that are much larger than your available memory.

By default, every image has the five axes `t c z y x`, whatever order the file stores. Axes that the file lacks are added with size 1, so a single plane is `(1, 1, 1, Y, X)`. To keep the order the file stores instead, such as `zyx` for a plain TIFF z-stack, pass `axes=None` to `open()`.

See [Axes](guide/axes.md).

## Telling images apart: the index

!!! info "Key concept: the index"

    Every image has an **index**: a small dictionary that identifies where the image comes from within the file. For example, `{'series': 0, 'tile': 3}`. Use it to find images and to identify their outputs.

Many files contain multiple images, such as one per series, mosaic tile or well. micro-reader keeps these as separate images rather than adding more axes to a single array. The number and meaning of these images vary between formats, so the index identifies each one:

```python
>>> image.index
{'series': 0, 'tile': 3, 'illumination': 1}
```

This image is tile 3 of series 0, recorded with illumination from side 1. A light-sheet file with two illumination sides therefore provides two images per tile, each with the usual `tczyx` axes, rather than one array with an additional format-specific dimension.

A spectral (lambda) scan, by contrast, is one image: its bands lie along `c`, and each channel's metadata records the wavelength of its band. See [Lambda scans](guide/axes.md#lambda-scans).

Use `index_range` to see which indices a file contains and how many values each index has. Use `select()` to retrieve images by their index:

```python
>>> f.index_range
{'series': 1, 'tile': 4, 'illumination': 2}

>>> f.select(illumination=0)  # every tile imaged from side 0
```

Index values are non-negative integers starting at 0. In `select()`, you can also pass a list, tuple, range or set to select multiple values:

```python
f.select(illumination=[0, 1])
```

For plate formats, you can also select by name:

```python
f.select(plate=0, well_name="B2")
f.select(plate=0, row="C", column=range(2, 5))
```

A level you do not name matches every value, so name the plate too when a file can hold several plates.

See [Plates](guide/plates.md) for plate selection.

Physical values, such as emission wavelengths, are stored in the [Metadata](guide/metadata.md), not in the index. Series numbers come from the file itself and do not necessarily form a continuous sequence. For example, a CZI file may contain series 0, 2 and 5.

See [Finding images](guide/finding-images.md).

### All index names

An image only carries the indices that are present in its file. micro-reader uses the following index names, from outermost to innermost:

| Index | What it identifies |
| --- | --- |
| `plate` | Plates in a screening file |
| `acquisition` | Imaging runs of a plate |
| `well` | Wells in a plate (a position in `plate.wells`) |
| `field` | Fields or sites imaged within a well |
| `series` | Series (Bio-Formats), the CZI `S` dimension, XY positions or regions |
| `tile` | Tiles of a mosaic |
| `rotation` | Sample rotations (CZI) |
| `view` | Views or angles of a light-sheet acquisition |
| `illumination` | Illumination sides of a light-sheet acquisition |
| `phase` | Phases (CZI) |
| `block` | Acquisition blocks (CZI) |
| `emission` | Emission bands of a spectral (lambda) scan. By default the bands are channels of one image instead (see [Lambda scans](guide/axes.md#lambda-scans)) |
| `excitation` | Excitation bands of an excitation scan |
| `lifetime` | Time bins of a FLIM measurement |
| `sequence` | A dimension that is not named by the file |
| `part` | Rectangular parts of a ragged image (see below) |
| `resolution` | Pyramid levels, with 0 representing full resolution |

### Two kinds of index

Indices relate to one another in two different ways. Understanding the distinction makes a file's structure easier to interpret.

Some indices **nest** like folders. A plate contains wells, a well contains fields, and a series contains its mosaic tiles. Each level belongs to the one above it, forming a tree. These are called *containment indices*:

`plate`, `well`, `field`, `series`, `tile`, `part` and `resolution`.

Other indices **cross** this hierarchy rather than nesting within it. In a light-sheet acquisition, for example, every tile is recorded from both illumination sides. Illumination is therefore not part of a tile. It is an independent choice that applies across all tiles.

These are called *orthogonal indices* because they vary independently of the containment hierarchy:

`acquisition`, `rotation`, `view`, `illumination`, `phase`, `block`, `emission`, `excitation`, `lifetime` and `sequence`.

The file outline shows both types. Containment indices appear as levels in the tree. Orthogonal indices appear after an `x`, followed by the number of values they have:

```text
<micro_reader.File 'MouseBrain_41Slices_2x2Tiles_3Channels_2Illuminations_1Angle.czi' backend='czi' format='czi' images=8>
  series 0: tiles 0-3, x illumination 2, tczyx (1, 3, 41, 1000, 1500) uint16
```

This describes one series with four tiles, each recorded from two illumination sides. The result is 4 × 2 = 8 images.

See [the outline](guide/finding-images.md#the-outline).

## Where things are: coordinate systems

An image's `metadata.position` describes the location of its first pixel. A position is meaningful only in relation to other positions in the same coordinate system. Each image therefore also identifies the coordinate system in which its position is defined.

The file lists its coordinate systems in `f.coordinate_systems`, and each image identifies its own in `metadata.coordinate_system`:

```python
>>> f.coordinate_systems
(CoordinateSystem(id='stage', name='stage', axes=None, estimated=False, extra={}),)

>>> image.metadata.position
{'x': Quantity(value=-3946.82..., unit='micrometer'),
 'y': Quantity(value=15846.91..., unit='micrometer')}

>>> image.metadata.coordinate_system
'stage'
```

Images with positions in the same coordinate system can be placed relative to one another.

Common coordinate systems include:

- `stage`: the microscope stage
- `slide`: a whole-slide scan
- `well-B2`, for example: positions relative to the centre of a particular screening-plate well
- `global`: the global coordinate system used by BigDataViewer

An image without a position has no coordinate system. If a file stores registration information, such as a BigStitcher transformation, micro-reader exposes it through `f.coordinate_transformations`. These transformations describe the relationships between coordinate systems.

See [Coordinate systems](reference/coordinates.md).

## The file's tree

The `f.collection` property describes how images are organised, without reading any pixel data.

For screening plates, the hierarchy is plate → well → field → image. Outside plate formats, the tree is organised by series. A field or series containing just one image is represented by that image directly.

```python
>>> c = f.collection  # a Harmony plate
>>> c.children[0]
<micro_reader.Collection 'HipSci Experiment 10 Plate 1' plate 0: 3 children>

>>> c.children[0].names  # its wells
('A1', 'A2', 'B1')

>>> len(c.images)  # all images contained in the plate
6
```

Each level of a plate hierarchy identifies its type and its parent. The `kind` property is one of `"file"`, `"plate"`, `"well"` or `"field"`, while `parent` refers to the level above it.

The hierarchy exposes the following properties:

- A file has `.plates`.
- A plate has `.wells`.
- A well has `.fields`.
- A field has `.images`.

Plate row and column names are available through `plate.row_names` and `plate.column_names`. You can also select a region of a plate using slices. For example, `plate.rows[2:5]` and `plate.columns[3:7]` return parts of the plate, which are themselves plates.

## Images that belong together

In addition to the index, micro-reader provides three relationships between images that belong together:

| Relationship | Attribute | Example |
| --- | --- | --- |
| Lower-resolution versions of an image | `image.resolutions` | The levels of a whole-slide pyramid |
| Rectangular parts of a ragged image | `image.parts` | One channel is a z-stack, while the others are single planes |
| The whole image to which a part belongs | `image.whole` | `part.whole` |

A *ragged image* is one whose channels or time points do not all have the same shape. Since it cannot be represented as a single regular array, micro-reader exposes it as several rectangular parts that you can read individually.

See [Resolution levels](guide/pyramids.md) and [Ragged images](guide/plates.md#ragged-fields).

## Metadata

The `image.metadata` property provides information needed to interpret and display an image, including pixel sizes, stage position, and channel names, colours and wavelengths.

Physical values are represented as `Quantity` objects, which pair a value with its unit. The unit names follow OME-Zarr conventions:

```python
>>> image.metadata.pixel_sizes["x"]
Quantity(value=2.5663047987549903, unit='micrometer')
```

!!! info "Key concept: missing means unknown"

    micro-reader reports only values recorded in the file. If a value is missing, such as the z-step size of a single-plane image, it remains absent. micro-reader does not fill missing values with defaults such as `1.0`. This lets you distinguish measured values from unknown ones.

See [Metadata](guide/metadata.md).

## Process-wide settings

Three settings control how micro-reader operates across a process:

- The memory limit for decoding
- The amount of decoded data kept in the cache
- The number of threads used for parallel decoding

The defaults are suitable for most use cases. You may want to change them when micro-reader runs alongside other workloads, such as a converter that manages its own threads.

| Setting | Default | Function | Environment variable |
| --- | --- | --- | --- |
| Memory for decoding | A quarter of the RAM | `set_memory_limit()` | `MICRO_READER_MEMORY_LIMIT` |
| Cache of decoded blocks | 256 MB | `set_cache_limit()` | `MICRO_READER_CACHE_LIMIT` |
| Decode threads | Number of CPUs | `set_decode_threads()` | `MICRO_READER_DECODE_THREADS` |

See [Memory and threads](guide/performance.md).