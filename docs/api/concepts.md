# Concepts

Microscopy files come in many formats, and each format organises its data in its own
way. A Zeiss CZI file may hold several series, each made of mosaic tiles. A light-sheet
file may record every tile from two sides. A plate from a screening microscope holds
dozens or hundreds of wells, each imaged at several sites. micro-reader describes all of
these with the same three ideas: a **file**, the **images** it contains, and the
**index** that tells those images apart. Once you know these three ideas, you can work
with every supported format in the same way.

```text
File                       one opened file (or set of files)
├── images[0]  Image       one array you can read
│   ├── index              where it comes from: {"series": 0, "tile": 3, ...}
│   ├── axes, shape, dtype the array: "tczyx", (1, 3, 41, 1000, 1500), uint16
│   ├── metadata           pixel sizes, position (and its coordinate system), channels
│   ├── resolutions        its lower-resolution versions, if the file has them
│   └── read(...) / [...]  read a region
├── images[1]  Image
├── ...
├── plates                 the plate layout, for screening files
├── coordinate_systems     the spaces the positions are in
└── collection             the file's tree: plate > well > field, or series
```

## Files

You open a file with `micro_reader.open()`. micro-reader looks at the first bytes of the
file to recognise its format, and uses the file name only when the content alone is not
enough. This means that a file with an unusual extension is still read correctly: an
Olympus VSI saved as `.tif`, for example, is read as VSI. The result is a `File`, which
looks the same whatever the format:

```python
>>> f = micro_reader.open("MouseBrain_2x2Tiles_3Channels_2Illuminations.czi")
>>> f.backend          # which reader handles it
'czi'
>>> len(f.images)      # how many images it contains
8
```

Some images are stored as more than one file on disk: an Olympus OIF and its folder of
TIFF planes, a DeltaVision image and its log file, or a plate made of thousands of plane
files. Opening any one of these files opens the whole image, so you do not need to know
which file holds what. See [Multi-file images](guide/multi-file.md).

## Images and their axes

An image is an array that you can read from, much like a NumPy array. Its dimensions
are called **axes**, and there are only five of them: time (`t`), channel (`c`), focal
plane (`z`), row (`y`) and column (`x`). An RGB image (a photo, a brightfield slide) is opened as three
channels, red, green and blue, so it has the same five axes. Open with `samples="keep"` to get
the colour samples as a sixth axis, `s`, the way the file stores them.

```python
>>> image = f.images[0]
>>> image.axes
'tczyx'
>>> image.shape
(1, 3, 41, 1000, 1500)
```

You read pixels with NumPy-style indexing, or by naming the axes in `read()`:

```python
>>> image[0, :, 5, 100:200, 100:200].shape
(3, 100, 100)
>>> image.read(t=0, z=5, y=slice(100, 200), x=slice(100, 200)).shape
(3, 100, 100)
```

Unlike a NumPy array, an image does not keep its pixels in memory. Every read goes back
to the file and decodes only the part you asked for. This is what lets you work with
images that are far larger than your computer's memory.

Most formats give their images in the order `t c z y x`. A few formats have an axis order
of their own, and micro-reader keeps it: a plain TIFF z-stack is `zyx`, for example. If
you would rather have the same layout for every file, pass `axes="canonical"` to `open()`.
See [Axes](guide/axes.md).

## Telling images apart: the index

!!! info "Key concept: the index"
    Every image has an **index**: a small dictionary that says where in the file the
    image comes from, such as `{'series': 0, 'tile': 3}`. You use it to find the images
    you want, and to name your outputs.

Many files contain more than one image, for example one per series, mosaic tile or well.
micro-reader keeps these as separate images rather than adding more axes to one large
array, because their number and meaning vary so much from format to format. Each image's
`index` says which one it is:

```python
>>> image.index
{'series': 0, 'tile': 3, 'illumination': 1}
```

Here the image is tile 3 of series 0, recorded with illumination from side 1. A
light-sheet file with two illumination sides therefore gives you two images per tile,
each with the usual `tczyx` axes, rather than one array with an extra, format-specific
dimension. In the same way, a spectral scan gives you one image per emission band, and
each image's metadata records the wavelength of its band.

To see which indices a file has and how many values each takes, use `index_range`; to
pick images by their index, use `select()`:

```python
>>> f.index_range
{'series': 1, 'tile': 4, 'illumination': 2}
>>> f.select(illumination=0)         # every tile, imaged from side 0
```

Index values are always whole numbers counting from 0. A value in `select()` may also be a
list, tuple, range or set, meaning any of them: `f.select(illumination=[0, 1])`. On plates you
can select by name as well, `f.select(well="B2")` or `f.select(row="C", column=range(2, 5))`;
see [Plates](guide/plates.md). Physical values, such as an emission wavelength, are in
[Metadata](guide/metadata.md). Series numbers
are the file's own, so they can skip values: a CZI file may have series 0, 2 and 5.
See [Finding images](guide/finding-images.md).

### All index names

A file's images carry only the indices the file actually has. These are all the names
micro-reader uses, from the outermost to the innermost:

| Index | What it tells apart |
| --- | --- |
| `plate` | the plates of a screening file |
| `acquisition` | the imaging runs of a plate |
| `well` | the wells of a plate (a position in `plate.wells`) |
| `field` | the fields (sites) imaged inside a well |
| `series` | series (Bio-Formats), the CZI `S` dimension, XY positions or regions |
| `tile` | the tiles of a mosaic |
| `rotation` | sample rotations (CZI) |
| `view` | views or angles of a light-sheet acquisition |
| `illumination` | illumination sides of a light-sheet acquisition |
| `phase` | phases (CZI) |
| `block` | acquisition blocks (CZI) |
| `emission` | emission bands of a spectral (lambda) scan |
| `excitation` | excitation bands of a spectral scan |
| `lifetime` | time bins of a FLIM measurement |
| `sequence` | a dimension the file does not name |
| `part` | the rectangular parts of a ragged image (see below) |
| `resolution` | levels of a pyramid, 0 being the full resolution |

### Two kinds of index

Indices relate to each other in two different ways, and knowing which is which makes a
file's structure easier to read.

Some indices **nest**, like folders. A plate contains wells, a well contains fields, and
a series contains its mosaic tiles. Each of these is part of the one above it, so they
form a tree. We call these *containment* indices: `plate`, `well`, `field`, `series`,
`tile`, `part` and `resolution`.

Other indices **cross** the tree instead of nesting in it. In a light-sheet acquisition,
every tile is recorded from both illumination sides, so the illumination side is not a
part of a tile: it is a second, independent choice that applies to every tile alike. We
call these *orthogonal* indices, because they vary independently of the tree, like the
columns of a table: `acquisition`, `rotation`, `view`, `illumination`, `phase`, `block`,
`emission`, `excitation`, `lifetime` and `sequence`.

The file's outline shows both kinds. Nesting indices appear as lines of the tree;
crossing indices appear after an `x`, with the number of values they take:

```text
<micro_reader.File 'MouseBrain_2x2Tiles_3Channels_2Illuminations.czi' backend='czi' format='czi' images=8>
  series 0: tiles 0-3, x illumination 2, tczyx (1, 3, 41, 1000, 1500) uint16
```

This reads as: one series with four tiles, each of which was recorded with two
illumination sides, giving 4 × 2 = 8 images. See
[the outline](guide/finding-images.md#the-outline).

## Where things are: coordinate systems

An image's `metadata.position` says where its first pixel is. Positions only mean something
next to each other when they are in the same space, so the position also names that space.
A file lists its spaces in `f.coordinate_systems`, and each image says which one it is in:

```python
>>> f.coordinate_systems
(CoordinateSystem(id='stage', name='stage', axes=None, estimated=False, extra={}),)
>>> image.metadata.position
{'x': Quantity(value=-3946.82..., unit='micrometer'), 'y': Quantity(value=15846.91..., unit='micrometer')}
>>> image.metadata.coordinate_system
'stage'
```

Two images with the same coordinate system can be placed against each other. Typical systems
are `stage` (a microscope stage), `slide` (a whole-slide scan), one per well on screening
plates (`well-B2`, positions relative to the well centre) and `global` (BigDataViewer). An
image without a position has no coordinate system. A registration that a file stores, for
example in BigStitcher, is a transformation between systems and is listed in
`f.coordinate_transformations`. See [Coordinate systems](reference/coordinates.md).

## The file's tree

`f.collection` shows how the images are nested, without reading any pixels: plates hold wells,
wells hold fields, and a field holds its images. Outside plates the tree is series. A field or
series with a single image is that image.

```python
>>> c = f.collection                 # a Harmony plate
>>> c.children[0]
<micro_reader.Collection 'HipSci Experiment 10 Plate 1' plate 0: 3 children>
>>> c.children[0].names              # its wells
('A1', 'A2', 'B1')
>>> len(c.images)                    # every image under it
6
```

On plates every level knows what it is and what holds it: `.kind` is `"file"`, `"plate"`,
`"well"` or `"field"`, and `.parent` is the level above. A file holds `.plates`, a plate
`.wells`, a well `.fields` and a field `.images`. The names of the rows and columns are
`plate.row_names` and `plate.column_names`; `plate.rows[2:5]` and `plate.columns[3:7]` give
a part of the plate, which is still a plate.

## Images that belong together

Besides the index, four relations link images that belong together:

| Relation | Attribute | Example |
| --- | --- | --- |
| Lower-resolution versions of an image | `image.resolutions` | the levels of a whole-slide pyramid |
| The full-resolution image of a level | `image.pyramid` | `level2.pyramid is f.images[0]` |
| Rectangular parts of a ragged image | `image.parts` | one channel is a z-stack, the others are single planes |
| The whole that a part belongs to | `image.whole` | `part.whole` |

A *ragged* image is one whose channels or time points do not all have the same shape, so
it cannot be read as a single array. micro-reader offers it as a few rectangular parts
that you read one at a time. See [Resolution levels](guide/pyramids.md) and
[ragged images](guide/plates.md#ragged-fields).

## Metadata

`image.metadata` holds what you need to place and display an image: pixel sizes, the
stage position, and each channel's name, colour and wavelengths. Physical values are
given as a `Quantity`, a value with its unit, using the unit names of OME-Zarr:

```python
>>> image.metadata.pixel_sizes["x"]
Quantity(value=2.5663047987549903, unit='micrometer')
```

!!! info "Key concept: missing means unknown"
    micro-reader reports only what the file says. When a file does not record a value,
    such as the z step of a single plane, the value is simply absent. It is never filled
    in with a default such as `1.0`, so you can tell a real value from a guess.

See [Metadata](guide/metadata.md).

## Process-wide settings

Three settings apply to everything you read in a process: how much memory decoding may
use, how much decoded data is kept for reuse, and how many threads decode in parallel.
The defaults suit most uses; you will mainly change them when micro-reader runs alongside
other work, such as a converter with its own threads:

| Setting | Default | Function | Environment variable |
| --- | --- | --- | --- |
| Memory for decoding | a quarter of the RAM | `set_memory_limit()` | `MICRO_READER_MEMORY_LIMIT` |
| Cache of decoded blocks | 256 MB | `set_cache_limit()` | `MICRO_READER_CACHE_LIMIT` |
| Decode threads | the number of CPUs | `set_decode_threads()` | `MICRO_READER_DECODE_THREADS` |

See [Memory and threads](guide/performance.md).
