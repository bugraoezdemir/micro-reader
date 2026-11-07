# Opening files

Everything starts with `micro_reader.open()`:

```python
micro_reader.open(path, backend=None, *, tiles="separate", resolutions="base",
                  rename=None, plates=True, multifile=True,
                  axes="canonical", samples="channels", squeeze=False) -> File
```

This page shows what each option is for. The [reference](../reference/functions.md#open) lists them in one table.

## The basics

```python
import micro_reader

f = micro_reader.open("cells.nd2")
try:
    image = f.images[0]
    ...
finally:
    f.close()
```

Or, the same with a context manager:

```python
with micro_reader.open("cells.nd2") as f:
    image = f.images[0]
    ...
```

`path` can be a `str` or any `os.PathLike` such as `pathlib.Path`.

!!! warning "Reading from a closed file raises"
    Images read through their file's handles. After `f.close()`, or after the `with` block, every read raises:

    ```text
    ValueError: this ND2 image has been closed: its file was closed (File.close() or the end of a with block)
    ```

    Keep the file open for as long as you read from its images.

## How the format is found

micro-reader reads the first kilobyte of the file and asks each backend in turn whether it recognises the content. The extension is used only when the content is not enough, for example for MRC. A `.tif` that is really an Olympus VSI is read as VSI, and a DeltaVision file with any extension is read as DeltaVision.

```python
f = micro_reader.open("scan.tif")
f.backend    # 'tiff'
f.format     # 'ome-tiff', 'imagej', 'svs', 'micromanager', ... or plain 'tiff'
```

If no backend recognises the file, you get `UnsupportedFormatError`:

```python
try:
    f = micro_reader.open("notes.txt")
except micro_reader.UnsupportedFormatError as e:
    print(e)    # no micro-reader backend recognises 'notes.txt'
```

### Forcing a backend

Pass `backend=` to skip detection:

```python
f = micro_reader.open("odd_extension.dat", backend="mrc")
```

The names are those in `micro_reader.BACKENDS`:

```python
[b.NAME for b in micro_reader.BACKENDS]
# ['vsi', 'flex', 'tiff', 'czi', 'luxendo', 'bdv', 'ilastik', 'hdf5', 'ims', 'nd2', 'oir',
#  'omp2info', 'oif', 'lif', 'harmony', 'columbus', 'scanr', 'cellomics', 'jdce',
#  'metaxpress', 'metamorph', 'micromanager', 'incell', 'cv7000', 'companion',
#  'ndpis', 'ics', 'raster', 'dv', 'mrc']
```

!!! note
    A named backend reads `path` itself. It does not redirect a part of a multi-file image to its main file, as automatic detection does (see below).

## Opening a part or a folder

Some images are spread over several files. Opening **any** of them opens the whole image:

```python
f = micro_reader.open("cells_R3D.dv.log")   # the softWoRx log of a DeltaVision file
f.path      # 'cells_R3D.dv'       the main file that was read
f.opened    # 'cells_R3D.dv.log'   what you passed
```

The same works for an OIF plane in `name.oif.files/`, an OIR continuation file, a Micro-Manager stack file, an `.ets` file of a VSI, and the side files of a plate.

You can also open a folder. It opens the image the folder belongs to (an NDTiff dataset, a Harmony or CellVoyager measurement), or the one image it holds. A folder of Opera Flex `.flex` files opens as a plate. A folder with several images raises an error that lists them. To sort a folder of many images, use [`group_files`](multi-file.md#grouping-many-files).

## Options

### `tiles`: separate tiles or one mosaic

By default, each tile is its own image with a `tile` index. With `tiles="mosaic"` you get one image per series instead, with its tiles placed on one canvas:

=== "separate (default)"

    ```python
    f = micro_reader.open("mosaic.czi")
    print(f)
    ```
    ```text
    <micro_reader.File 'mosaic.czi' backend='czi' format='czi' images=8>
      series 0: tiles 0-3, x illumination 2, tczyx (1, 3, 41, 1000, 1500) uint16
    ```

=== "mosaic"

    ```python
    f = micro_reader.open("mosaic.czi", tiles="mosaic")
    print(f)
    ```
    ```text
    <micro_reader.File 'mosaic.czi' backend='czi' format='czi' images=2>
      series 0: x illumination 2, tczyx (1, 3, 41, 1850, 2774) uint16
    ```

Tiles are placed where the file says they are: Zeiss CZI by its stitching information, Leica LIF / XLEF tile scans by their stage positions, FluoView tiling sets (`.omp2info`) by their grid. Nothing is registered, and where tiles overlap the later tile is on top. For a tiled file of another format, `tiles="mosaic"` raises `UnsupportedFormatError`.

`tiles="stitched"` is the old name for `tiles="mosaic"`. It still works and warns.

### `resolutions`: base level only, or every level

Pyramidal files (whole-slide TIFFs, Imaris, CZI with stored pyramids) store lower resolution levels.

- `resolutions="base"` (default): `f.images` holds the full-resolution images. You reach the lower levels with `image.resolutions`.
- `resolutions="all"`: every level is an image of its own in `f.images`, indexed by `resolution`.

```python
f = micro_reader.open("cells.ims", resolutions="all")
f.index_range                     # {'series': 1, 'resolution': 4}
level2 = f.select(series=0, resolution=2)[0]
```

See [Resolution levels](pyramids.md).

### `rename`: say what an unnamed dimension is

Some files do not say what a dimension means. A plain multi-page TIFF with 80 pages can be a z-stack, a time series or anything else. micro-reader does not guess. It gives one image per page, indexed by `sequence`:

```python
f = micro_reader.open("stack.tif")
f.index_range         # {'series': 1, 'sequence': 80}
f.images[0].axes      # 'tczyx' (axes=None: 'yx')
```

If you know what the pages are, use `rename` to make the index an array axis:

```python
f = micro_reader.open("stack.tif", rename={"sequence": "z"})
len(f.images)         # 1
f.images[0].axes      # 'tczyx' (axes=None: 'zyx')
f.images[0].shape     # (1, 1, 80, 256, 300) (axes=None: (80, 256, 300))
```

Rules:

- The keys must be indices that are sampled at the same points: `illumination`, `phase`, `block`, `emission`, `excitation`, `lifetime` or `sequence`.
- The values must be `"t"`, `"c"` or `"z"`, and two keys cannot map to the same axis.
- Only the TIFF backend can rename for now. For other backends, `rename` raises `UnsupportedFormatError` when the file has that index, and changes nothing when it does not.

Because a file without the index is not affected, you can pass the same `rename` to every file of a batch:

```python
RENAME = {"sequence": "z"}
for path in paths:
    with micro_reader.open(path, rename=RENAME) as f:
        ...
```

### `plates`: read more files as plates

Plate formats (Harmony, Columbus, MetaXpress, IN Cell, CellVoyager, Opera Flex, scanR, Cellomics, ImageXpress HCS.ai) always open as plates. Some files can be a plate or not. By default (`plates=True`) they open as a plate:

- a Zeiss CZI recorded on a plate;
- an ND2 whose positions are named after wells (`A10_0000`), or a folder of NIS-Elements ND2 files from a well-plate run;
- an OME companion file or an OME-TIFF that describes a plate, such as a Yokogawa CQ1 `MeasurementResult.ome.tif`.

```python
f = micro_reader.open("W96_B2+B4.czi")
f.plates[0].wells[0].name     # 'B2'
```

`plates=True` is the default, so a plate in any format reads as a plate; `plates=False` reads such a file by its series instead (`f.plates` is then empty). For a file that is not a plate, it changes nothing. See [Plates](plates.md).

### `multifile`: a whole OME-TIFF set or one file

Some OME-TIFF sets keep the full OME-XML in each of their TIFF files. Opening one of those files reads the whole set by default. With `multifile=False`, you get only the planes stored in the file you opened, with the set's metadata:

```python
f = micro_reader.open("set_t001.ome.tif", multifile=False)
```

For a plate made of many files (an Opera Flex folder, a NIS-Elements run of ND2 files), `multifile=False` reads only the file you opened.

`multifile` does not apply to:

- parts of other multi-file formats (OIR, OIF, VSI);
- Micro-Manager stacks: they split their files by size, so one file is not a complete part of the image;
- single-file OME-TIFFs.

### `axes`, `samples`, `squeeze`: present every image in one layout

These three options reshape every image of the file, as [`Image.as_axes`](axes.md) does for one image:

```python
f = micro_reader.open("image.tif", axes="canonical", squeeze=False)
f.images[0].axes      # 'tczyx'
f.images[0].source    # the image in its native layout
```

| Option | Values | Effect |
| --- | --- | --- |
| `axes` | `None` (keep), `"canonical"` (`tczyx` + `s`), or letters such as `"zyx"` | the order of the axes; missing axes are added with size 1 |
| `samples` | `"channels"` (default), `"keep"` | fold RGB samples into the channels, or keep them as an `s` axis |
| `squeeze` | `False`, `True` | drop every size-1 axis except `y` and `x` |

An RGB image opens with 3 channels (red, green, blue) by default. Pass `samples="keep"` to get the `s` axis. See [Axes](axes.md#rgb-samples).

## Batch processing many files

When you convert or scan many files, keep these in mind:

- Wrap each file in a `with` block, so that handles are released even when a read fails.
- Catch `micro_reader.UnsupportedFormatError` to skip or route files that micro-reader does not read.
- Treat warnings as data. Missing plane files, damaged CZI files and missing companion files are reported as warnings, not errors (see [Errors and warnings](../reference/errors.md)).

```python
import warnings
import micro_reader

def inventory(paths):
    for path in paths:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            try:
                with micro_reader.open(path) as f:
                    yield path, f.format, len(f.images), f.index_range, \
                          [str(w.message) for w in caught]
            except micro_reader.UnsupportedFormatError:
                yield path, None, 0, {}, ["not supported"]
```
