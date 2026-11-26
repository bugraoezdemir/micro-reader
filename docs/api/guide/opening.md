# Opening files

The main entry point for reading microscopy data is `micro_reader.open()`:

```python
micro_reader.open(
    path,
    backend=None,
    *,
    tiles="separate",
    resolutions="base",
    rename=None,
    plates=True,
    multifile=True,
    axes="canonical",
    samples="channels",
    squeeze=False,
) -> File
```

This page explains each option and when to use it. For a compact overview, see the [API reference](../reference/functions.md#open).

## The basics

Open a file and access its images:

```python
import micro_reader

f = micro_reader.open("cells.nd2")

try:
    image = f.images[0]
    ...
finally:
    f.close()
```

The recommended approach is to use a context manager:

```python
with micro_reader.open("cells.nd2") as f:
    image = f.images[0]
    ...
```

The `path` argument accepts a string or any `os.PathLike` object, such as `pathlib.Path`.

!!! warning "Keep the file open while reading"

    Images read their pixels through the handles owned by their parent file. Once `f.close()` is called, or the context manager exits, further reads from its images raise an error:

    ```text
    ValueError: this ND2 image has been closed: its file was closed (File.close() or the end of a with block)
    ```

    Keep the file open for as long as you need to read its images.

## How the format is detected

micro-reader reads the first kilobyte of a file and asks its backends, in turn, whether they recognise the contents. The file extension is used only when the contents are not sufficient to identify the format.

For example, a TIFF extension does not necessarily mean the file is a TIFF. An Olympus VSI file saved as `.tif` is still recognised as VSI. DeltaVision files are recognised by their contents regardless of their extension.

```python
f = micro_reader.open("scan.tif")

f.backend  # 'tiff'
f.format   # 'ome-tiff', 'imagej', 'svs', 'micromanager', ... or 'tiff'
```

If no backend recognises the file, `open()` raises `UnsupportedFormatError`:

```python
try:
    f = micro_reader.open("notes.txt")
except micro_reader.UnsupportedFormatError as e:
    print(e)  # no micro-reader backend recognises 'notes.txt'
```

### Forcing a backend

You can specify `backend=` to bypass automatic detection:

```python
f = micro_reader.open("odd_extension.dat", backend="mrc")
```

The available backend names are listed in `micro_reader.BACKENDS`:

```python
[b.NAME for b in micro_reader.BACKENDS]
# ['vsi', 'flex', 'tiff', 'czi', 'luxendo', 'bdv', 'ilastik', 'hdf5', 'ims', 'nd2', 'oir',
#  'omp2info', 'oif', 'lif', 'harmony', 'columbus', 'scanr', 'cellomics', 'jdce',
#  'metaxpress', 'metamorph', 'micromanager', 'incell', 'cv7000', 'companion',
#  'ndpis', 'ics', 'raster', 'klb', 'dv', 'mrc']
```

!!! note

    A forced backend reads the specified `path` directly. It does not redirect a file that belongs to a multi-file dataset to the dataset's main file. Automatic detection handles this redirection, as described below.

## Opening part of a dataset or a folder

Some microscopy datasets are spread across multiple files. In these cases, opening any one of the associated files can open the complete dataset.

For example, opening a DeltaVision log file resolves to its main image file:

```python
f = micro_reader.open("cells_R3D.dv.log")

f.path    # 'cells_R3D.dv': the main file that was read
f.opened  # 'cells_R3D.dv.log': the path passed to open()
```

The same applies to an OIF plane inside `name.oif.files/`, an OIR continuation file, a Micro-Manager stack file, an `.ets` file belonging to a VSI dataset, and side files belonging to a plate.

You can also pass a folder to `open()`. Depending on its contents, micro-reader opens the dataset the folder belongs to, such as an NDTiff dataset or a Harmony or CellVoyager measurement, or the single image contained in the folder.

A folder containing Opera Flex `.flex` files opens as a plate. A folder that holds no image itself, such as a downloaded sample folder with the data one or two levels down, opens the one dataset found in its subfolders. If a folder contains several unrelated images or datasets, `open()` raises an error listing them.

To organise a folder containing many images into groups, use [`group_files`](multi-file.md#grouping-many-files).

## Options

### `tiles`: separate tiles or combine them into a mosaic

By default, every tile is exposed as a separate image with a `tile` index. Separate tiles are the default because they are what the file stores: a mosaic places them by their recorded positions, without registration, and registration or stitching downstream needs the tiles themselves. The default is also the same for every format, since only some formats can place their tiles.

Pass `tiles="mosaic"` to combine the tiles into one image per series, placed on a shared canvas. Slide scans usually want this: a Zeiss slide scan opens as thousands of camera tiles, and its stored pyramid is reached only through the mosaic. The file summary (`print(f)`) says so on every series whose tiles `tiles="mosaic"` can place, and names the stored levels when the mosaic has more of them than a tile.

**Separate tiles (default):**

```python
f = micro_reader.open("mosaic.czi")
print(f)
```

```text
<micro_reader.File 'mosaic.czi' backend='czi' format='czi' images=8>
  series 0: tiles 0-3, x illumination 2, tczyx (1, 3, 41, 1000, 1500) uint16 (tiles="mosaic": its tiles as one image)
```

**Mosaic:**

```python
f = micro_reader.open("mosaic.czi", tiles="mosaic")
print(f)
```

```text
<micro_reader.File 'mosaic.czi' backend='czi' format='czi' images=2>
  series 0: x illumination 2, tczyx (1, 3, 41, 1850, 2774) uint16
```

Tiles are placed according to the information stored in the file:

- Zeiss CZI uses its stitching information.
- Leica LIF / XLEF tile scans use their stage positions.
- FluoView tiling sets (`.omp2info`) use their acquisition grid.
- BigDataViewer / BigStitcher tiles use their nominal positions (the acquisition grid or tile configuration, not a registration), rounded to whole pixels, one canvas per angle and illumination. Views that are turned, sheared or mirrored, tiles apart along `z` and tiles of different pixel sizes cannot be placed this way and are refused, naming why.

Tiles are not registered or blended. Where they overlap, the later tile is placed on top, and areas without a tile read as 0.

Mosaic assembly is currently supported for these tiled formats. Passing `tiles="mosaic"` for a tiled format without mosaic support raises `UnsupportedFormatError`.

The older option `tiles="stitched"` is an alias for `tiles="mosaic"`. It still works but raises a warning.

### `resolutions`: full resolution or every pyramid level

Pyramidal formats, including whole-slide TIFF, Imaris and CZI files with stored pyramids, contain lower-resolution versions of their images.

There are two ways to expose these levels:

- **`resolutions="base"` (default):** `f.images` contains the full-resolution images. Access lower-resolution levels through `image.resolutions`.
- **`resolutions="all"`:** Every resolution level is exposed as a separate image in `f.images`, identified by the `resolution` index.

For example:

```python
f = micro_reader.open("cells.ims", resolutions="all")

f.index_range  # {'series': 1, 'resolution': 4}

level2 = f.select(series=0, resolution=2)[0]
```

See [Resolution levels](pyramids.md).

### `rename`: assign a meaning to an unnamed dimension

Some formats contain dimensions whose meaning is not recorded in the file. For example, a plain multi-page TIFF with 80 pages could represent a z-stack, a time series or something else.

micro-reader does not guess. By default, an unnamed dimension is represented by an index called `sequence`, with one image per value:

```python
f = micro_reader.open("stack.tif")

f.index_range     # {'series': 1, 'sequence': 80}
f.images[0].axes  # 'tczyx' (with axes=None: 'yx')
```

When you know what the pages represent, use `rename` to convert the index into an array axis:

```python
f = micro_reader.open("stack.tif", rename={"sequence": "z"})

len(f.images)        # 1
f.images[0].axes     # 'tczyx' (with axes=None: 'zyx')
f.images[0].shape    # (1, 1, 80, 256, 300)
                     # with axes=None: (80, 256, 300)
```

A file can have more than one unnamed dimension. For example, tifffile writes a 4-D array without axis names as `QQYX`, two unnamed dimensions before y and x. They could be t and z, c and z, t and c, or the other way round, and the file does not say which. Give one letter per dimension, in the order the file stores them (the outer one first):

```python
f = micro_reader.open("unnamed.tif", rename={"sequence": "tz"})

f.images[0].shape    # with axes=None: (2, 3, 32, 48), t then z
```

A single letter for two unnamed dimensions is refused, since reading them as one axis would be a guess. The error message lists the dimensions and their sizes.

The following rules apply:

- **Keys:** Only indices sampled at the same points can be renamed: `illumination`, `phase`, `block`, `emission`, `excitation`, `lifetime` and `sequence`.
- **Values:** Each key maps to `"t"`, `"c"` or `"z"`, or to one of these letters per dimension of that index (`"tz"`), outer first. A dimension of size 1 needs no letter. No axis can be given twice, within one key or across keys.
- **Backend support:** Currently, only the TIFF backend supports `rename`. For other backends, it raises `UnsupportedFormatError` if the file contains the specified index. If the index is absent, the option has no effect.

You can pass the same `rename` mapping to a batch of files. Files without the specified index are left unchanged:

```python
RENAME = {"sequence": "z"}

for path in paths:
    with micro_reader.open(path, rename=RENAME) as f:
        ...
```

### `plates`: interpret plate datasets as plates

Dedicated plate formats, including Harmony, Columbus, MetaXpress, IN Cell, CellVoyager, Opera Flex, scanR, Cellomics and ImageXpress HCS.ai, open as plates.

Other formats may also contain plate data. With the default `plates=True`, micro-reader recognises the following as plates:

- Zeiss CZI files acquired on a plate.
- ND2 files whose XY positions are named after wells, such as `A10_0000`, and folders of NIS-Elements ND2 files from a well-plate run.
- OME companion files and OME-TIFF files describing a plate, such as a Yokogawa CQ1 `MeasurementResult.ome.tif`.

For example:

```python
f = micro_reader.open("W96_B2+B4.czi")

f.plates[0].wells[0].name  # 'B2'
```

For formats that support switching between plate and series representations, pass `plates=False` to read the data by series instead. In that case, `f.plates` is empty.

For files that do not contain plate data, this option has no effect.

See [Plates](plates.md).

### `multifile`: open a complete file set or just one file

Some OME-TIFF datasets consist of multiple TIFF files, each containing the complete OME-XML metadata. By default, opening any one of these files opens the entire set.

Pass `multifile=False` to read only the planes stored in the specified file, while retaining the set's metadata:

```python
f = micro_reader.open("set_t001.ome.tif", multifile=False)
```

The same option applies to other sets in which one file opens the whole set. With `multifile=False`, only the file you opened is read:

- A TIFF of an OME companion set, which is then read with the companion's metadata.
- A plane file of a plate (Harmony, CellVoyager, MetaXpress, IN Cell, ...), an Opera Flex file, a Cellomics `.C01` file, or one ND2 file of a NIS-Elements well-plate run.
- A Leica `.lof`, `.xlif` or `.lifext` that belongs to a larger experiment.
- A `.lux.h5` stack or an `imaris.ims` of a Luxendo experiment.

`multifile` does not apply to:

- Other multi-file formats such as OIR, OIF and VSI.
- Micro-Manager stacks, where files are split by size and an individual file is not a complete part of the image.
- Single-file OME-TIFF datasets.

### `axes`, `samples` and `squeeze`: choose the array layout

These three options control the layout of every image in the file. They apply the same transformations as [`Image.as_axes`](axes.md) does to an individual image.

```python
f = micro_reader.open("image.tif", axes="canonical", squeeze=False)

f.images[0].axes   # 'tczyx'
f.images[0].source # the image in its native layout
```

| Option | Values | Effect |
| --- | --- | --- |
| `axes` | `None` (keep native order), `"canonical"` (`tczyx` plus `s`), or axis strings such as `"zyx"` | Sets the axis order. Missing axes are added with size 1. |
| `samples` | `"channels"` (default), `"keep"` | Converts RGB samples into channels, or retains them as an `s` axis. |
| `squeeze` | `False` (default), `True` | Removes every size-1 axis except `y` and `x`. |

By default, RGB images are represented as three channels (red, green and blue). Pass `samples="keep"` to retain the colour samples as an `s` axis.

See [Axes](axes.md#rgb-samples).

## Batch processing many files

When converting or inspecting many files, a few practices make processing more reliable:

- **Use a context manager for each file.** File handles are released even if a read fails.
- **Catch `UnsupportedFormatError`.** This lets you skip unsupported files or route them to another reader.
- **Treat warnings as data.** Missing planes, damaged CZI files and missing companion files may be reported as warnings rather than errors. See [Errors and warnings](../reference/errors.md).

For example, the following function inventories a collection of files and records any warnings raised while opening and inspecting them:

```python
import warnings
import micro_reader


def inventory(paths):
    for path in paths:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")

            try:
                with micro_reader.open(path) as f:
                    yield (
                        path,
                        f.format,
                        len(f.images),
                        f.index_range,
                        [str(w.message) for w in caught],
                    )
            except micro_reader.UnsupportedFormatError:
                yield path, None, 0, {}, ["not supported"]
```