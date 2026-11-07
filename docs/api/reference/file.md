# File

```python
class micro_reader.File
```

An opened file: its images, the plates they belong to, and the backend that reads them. You get a `File` from [`open()`](functions.md#open), not from its constructor. Every format returns this class. Backend subclasses are internal.

A `File` is a context manager. Leaving the `with` block closes it.

```python
with micro_reader.open("cells.czi") as f:
    image = f.images[0]
```

## Attributes

| Attribute | Type | Description |
| --- | --- | --- |
| `path` | `str` | The file that was read. For a multi-file image, its main file. |
| `opened` | `str` | The path given to `open()`: a part, a folder, or the same as `path`. |
| `backend` | `str` | The backend that reads the file: one of the names in [`BACKENDS`](constants.md#backends). |
| `format` | `str` | The format variant, for example `"ome-tiff"`, `"imagej"`, `"svs"`, `"nd2-legacy"`, `"lof"`, `"oib"`, `"png"`, `"harmony-v4"`, `"opera-flex"`. The backend's name when it reads only one variant. See [Formats](../formats.md#overview). |
| `images` | `list[Image]` | The file's images, full resolution only unless opened with `resolutions="all"`. |
| `plates` | `list[Plate]` | The plates the images belong to. Empty for a file that is not a plate, or opened with `plates=False`. |
| `indices` | `list[dict]` | Every image's index, in the order of `images`. |
| `index_range` | `dict[str, int]` | For each index name, the number of distinct values it takes. |
| `coordinate_systems` | `tuple[CoordinateSystem, ...]` | The shared coordinate systems the images' positions are in. Empty when the format gives no position. Declared when the file opens, without reading any image's metadata. See [Coordinates](coordinates.md). |
| `coordinate_transformations` | `tuple[Transformation, ...]` | The edges of the file's transformation graph that are not plain positions. Empty except for BigDataViewer / BigStitcher. See [Coordinates](coordinates.md#transformation). |
| `collection` | [`Collection`](coordinates.md#collection) | The file's tree: plate > well > field inside plates, series outside them. Built from the indices and plates alone; no metadata is read. |
| `kind` | `str` | `"file"`. The level of the object in the tree of [plate classes](plates.md#containment). |
| `parent` | `None` | A file is held by nothing. A plate's `parent` is its file. |
| `damage` | `dict` | **CZI only.** `{}` for a sound file. Otherwise, what was damaged and recovered. See [Formats › CZI](../formats.md#damaged-files). |

### `index_range`

The widest range of each index in the file, as a count of distinct values:

```python
f.index_range      # {'series': 2, 'tile': 5}
```

It is neither a total nor a shape. With 3 tiles in series 0 and 5 in series 1, the range of `tile` is 5. Tile, rotation, view, well and field values count from 0, so their values are `range(n)`. Series values are the file's own and may skip numbers.

## Methods

### `select`

```python
File.select(**index) -> list[Image]
```

The images whose index matches every given name and value.

| Parameter | Description |
| --- | --- |
| `**index` | Index names (from [`INDEX_NAMES`](constants.md#index_names)) and integer values. |

Returns a list, possibly empty. Images without one of the given names do not match. A known index that no image has selects nothing.

**Values.** An index value is an `int`, or a `list`, `tuple`, `range` or `set` of them, meaning any of them: `select(illumination=[2, 3])`. A `slice` of positions works as a range: `well=slice(0, 12)`, `row=slice(2, None)` (row C onward), `column=slice(0, None, 2)`. A slice has no negative bound and a step of at least 1 (`ValueError`); which position is last depends on what is selected.

**A plate's places.** Each place of a plate is addressed by position and by name, one keyword each:

| Position (0-based) | Name | Place |
| --- | --- | --- |
| `plate` | `plate_name` | the plate |
| `acquisition` | `acquisition_name` | the plate's imaging run |
| `well` | `well_name` | the well: `well` is its place in `plates[p].wells`, `well_name` its name (`"C3"`) |
| `row` | `row_name` | the well's row (`row=2` is the row named `"C"`) |
| `column` | `column_name` | the well's column (`column=2` is the column named `"3"`) |

A string given to a position keyword is its name: `select(well="C3")` is `select(well_name="C3")`. A number given to a name keyword raises `TypeError` that names the position keyword. `row` and `column` are another address of the same well, so they do not go with `well` / `well_name` (`TypeError`). They also need a plate: a file opened with `plates=False` has none (`ValueError`).

A `Plate`, `Well`, `Field` or `Acquisition` object is no value (`TypeError`); the message gives its index (`plate.well_index("C3")`) or name.

**Raises.** `KeyError` for a name that is neither an index name nor one of the keywords above; the old name of `series` (before micro-reader 0.1.0) raises a `KeyError` that names `series`.

```python
f.select(series=1, view=0)
f.select(well=f.plates[0].well_index("B2"), field=0)
f.select(well="B2")                                # the same, by name
f.select(row=["B", "C"], column=range(2, 5))       # rows B and C, columns 3 to 5
f.select(row_name="C", column=slice(0, None, 2))   # row C, every second column
f.select(series=0, resolution=2)                   # with resolutions="all"
```

[`Field.select`](plates.md#field) takes the same values for the image indices (not the plate's places). To pick a part of a plate as a plate, slice it with [`Plate.rows` and `Plate.columns`](plates.md#plate).

### `outline`

```python
File.outline() -> str
```

The file's structure as text: its plates, wells and fields, or its series. Each line shows tiles, the orthogonal indices that vary (`x view 2`), axes, shape and dtype, pyramid levels, and ragged parts. `repr(f)` shows the first 6 lines of it.

```text
<micro_reader.File 'mosaic.czi' backend='czi' format='czi' images=8>
  series 0: tiles 0-3, x illumination 2, tczyx (1, 3, 41, 1000, 1500) uint16
```

### `check_members`

```python
File.check_members() -> int
```

For a file set (a plate, an OME companion set): open every member file once and check that it is the plane its index describes (size, pixel type, a single plane). Returns the number of members checked, which is 0 for a file that is not a file set. Raises [`CorruptFileError`](errors.md#corruptfileerror) naming up to five members that do not fit, with the total count. Members are checked in parallel on the decode pool.

Reading makes the same check plane by plane at no extra cost. Use `check_members()` to find bad files before a long job.

### `canvas`

```python
File.canvas(**index) -> Image          # CZI only
```

(Only a file opened from a CZI has this method.)

All series of a CZI composed on one canvas: the box around every full-resolution tile, with background (0) between the series. The image's `origin` is the box's corner. A file with several rotations or views needs `rotation=` or `view=`. Other names raise `KeyError`. A missing required name raises `ValueError`.

### `close`

```python
File.close() -> None
```

Close every image's file handles. Reads afterwards raise. Calling it twice is harmless.

### Context manager

`__enter__` returns the file. `__exit__` calls `close()` and does not suppress exceptions.

## Pickling

`File` objects cannot be pickled. To read in other processes, pass the path and the image's `index`, and open the file in the worker. See [Reading from several processes](../guide/reading.md#reading-from-several-processes).
