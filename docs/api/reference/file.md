# File

```python
class micro_reader.File
```

An opened file with its images, the plates they belong to, and the backend that reads them. You get a `File` from [`open()`](functions.md#open) and do not call its constructor. Every format returns this class, and the backend subclasses are internal.

A `File` is a context manager. Leaving the `with` block closes it.

```python
with micro_reader.open("cells.czi") as f:
    image = f.images[0]
```

## Attributes

| Attribute | Type | Description |
| --- | --- | --- |
| `path` | `str` | The file that was read. For a multi-file image, this is its main file. |
| `opened` | `str` | The path given to `open()`. It is a part, a folder, or the same as `path`. |
| `backend` | `str` | The backend that reads the file. It is one of the names in [`BACKENDS`](constants.md#backends). |
| `format` | `str` | The format variant, for example `"ome-tiff"`, `"imagej"`, `"svs"`, `"nd2-legacy"`, `"lof"`, `"oib"`, `"png"`, `"harmony-v4"` or `"opera-flex"`. When the backend reads only one variant, this is the backend's name. See [Formats](../formats.md#overview). |
| `images` | `list[Image]` | The images of the file. They are full resolution only, unless the file was opened with `resolutions="all"`. |
| `plates` | `list[Plate]` | The plates the images belong to. This is empty for a file that is not a plate, and for a file opened with `plates=False`. |
| `indices` | `list[dict]` | The index of every image, in the order of `images`. |
| `index_range` | `dict[str, int]` | The number of distinct values that each index name takes. |
| `coordinate_systems` | `tuple[CoordinateSystem, ...]` | The shared coordinate systems that the positions of the images are in. This is empty when the format gives no position. The systems are declared when the file opens, without reading the metadata of any image. See [Coordinates](coordinates.md). |
| `coordinate_transformations` | `tuple[Transformation, ...]` | The edges of the transformation graph of the file that are not plain positions. This is empty except for BigDataViewer / BigStitcher files. See [Coordinates](coordinates.md#transformation). |
| `collection` | [`Collection`](coordinates.md#collection) | The tree of the file: plate > well > field inside plates, and series outside them. It is built from the indices and plates alone, so no metadata is read. |
| `kind` | `str` | Always `"file"`. It is the level of the object in the tree of [plate classes](plates.md#containment). |
| `parent` | `None` | Always `None`, because nothing holds a file. The `parent` of a plate is its file. |
| `damage` | `dict` | **CZI only.** `{}` for a sound file. Otherwise, it says what was damaged and recovered. See [Formats › CZI](../formats.md#damaged-files). |

### `index_range`

The widest range of each index in the file, as a count of distinct values:

```python
f.index_range      # {'series': 2, 'tile': 5}
```

It is not a total and not a shape. With 3 tiles in series 0 and 5 tiles in series 1, the range of `tile` is 5. Tile, rotation, view, well and field values count from 0, so their values are `range(n)`. Series values are the file's own and may skip numbers.

## Methods

### `select`

```python
File.select(**index) -> list[Image]
```

Returns the images whose index matches every given name and value.

| Parameter | Description |
| --- | --- |
| `**index` | Index names (from [`INDEX_NAMES`](constants.md#index_names)) and integer values. |

The result is a list, which can be empty. Images without one of the given names do not match. A known index that no image has selects nothing. An index that is not given matches every value, so `select(well_name="B2")` selects well B2 in every plate of the file. Give `plate` or `plate_name` as well to select one plate's well.

**Values.** An index value is an `int`, or a `list`, `tuple`, `range` or `set` of them, which means any of them, as in `select(illumination=[2, 3])`. A `slice` of positions works as a range. Examples are `well=slice(0, 12)`, `row=slice(2, None)` for row C onward, and `column=slice(0, None, 2)`. A slice has no negative bound and a step of at least 1, otherwise `ValueError` is raised. Which position is last depends on what is selected.

**The places of a plate.** Each place of a plate is addressed by position and by name, with one keyword for each:

| Position (0-based) | Name | Place |
| --- | --- | --- |
| `plate` | `plate_name` | The plate. |
| `acquisition` | `acquisition_name` | The imaging run of the plate. |
| `well` | `well_name` | the well. `well` is its place in `plates[p].wells`, and `well_name` is its name (`"C3"`). |
| `row` | `row_name` | The row of the well. `row=2` is the row named `"C"`. |
| `column` | `column_name` | The column of the well. `column=2` is the column named `"3"`. |

A string given to a position keyword is taken as a name, so `select(well="C3")` is the same as `select(well_name="C3")`. A number given to a name keyword raises `TypeError`, and the message names the position keyword. `row` and `column` are another address of the same well, so combining them with `well` or `well_name` raises `TypeError`. They also need a plate, and a file opened with `plates=False` has none, which raises `ValueError`.

A `Plate`, `Well`, `Field` or `Acquisition` object is not a valid value and raises `TypeError`. The message gives the index of the object, as in `plate.well_index("C3")`, or its name.

**Raises.** `KeyError` for a name that is neither an index name nor one of the keywords above. The old name `scene` raises a `KeyError` that points to `series`. Changed in 0.1.0: the index `scene` was renamed to `series`.

```python
f.select(series=1, view=0)
f.select(plate=0, well=f.plates[0].well_index("B2"), field=0)
f.select(plate=0, well="B2", field=0)                      # the same, by name
f.select(plate=0, row=["B", "C"], column=range(2, 5))      # rows B and C, columns 3 to 5
f.select(plate=0, row_name="C", column=slice(0, None, 2))  # row C, every second column
f.select(series=0, resolution=2)                   # with resolutions="all"
```

[`Field.select`](plates.md#field) takes the same values for the image indices, but not for the places of a plate. To pick a part of a plate as a plate, slice it with [`Plate.rows` and `Plate.columns`](plates.md#plate).

### `outline`

```python
File.outline() -> str
```

Returns the structure of the file as text, with its plates, wells and fields, or its series. Each line shows the tiles, the orthogonal indices that vary (such as `x view 2`), the axes, the shape and dtype, the pyramid levels and any ragged parts. `repr(f)` shows the first 6 lines of it.

```text
<micro_reader.File 'mosaic.czi' backend='czi' format='czi' images=8>
  series 0: tiles 0-3, x illumination 2, tczyx (1, 3, 41, 1000, 1500) uint16
```

### `check_members`

```python
File.check_members() -> int
```

For a file set, such as a plate or an OME companion set, this opens every member file once. It checks that each member is the plane its index describes, which means the right size, the right pixel type and a single plane. It returns the number of members checked, which is 0 for a file that is not a file set. If members do not fit, it raises [`CorruptFileError`](errors.md#corruptfileerror), which names up to five of them and gives the total count. The members are checked in parallel on the decode pool.

Reading makes the same check plane by plane at no extra cost. Use `check_members()` to find bad files before a long job.

### `canvas`

```python
File.canvas(**index) -> Image          # CZI only
```

Only a file opened from a CZI has this method.

It composes all series of a CZI on one canvas. The canvas is the box around every full-resolution tile, with background (0) between the series. The `origin` of the image is the corner of the box. A file with several rotations or views needs `rotation=` or `view=`. Other names raise `KeyError`, and a missing required name raises `ValueError`.

### `close`

```python
File.close() -> None
```

Closes the file handles of every image. Reads afterwards raise an error. Calling it twice is harmless.

### Context manager

`__enter__` returns the file. `__exit__` calls `close()` and does not suppress exceptions.

## Pickling

`File` objects cannot be pickled. To read in other processes, pass the path and the `index` of the image, and open the file in the worker. See [Reading from several processes](../guide/reading.md#reading-from-several-processes).
