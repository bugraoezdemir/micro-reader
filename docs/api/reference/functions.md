# Functions

## `open`

```python
micro_reader.open(path, backend=None, *, tiles="separate", resolutions="base",
                  rename=None, plates=True, multifile=True,
                  axes="canonical", samples="channels", squeeze=False) -> File
```

Open `path` with the backend that recognises its content, or with the one named.

A part of a multi-file image (an OIR continuation, a DeltaVision log, an OIF plane, a plate side file) opens the whole image. A folder opens the image it belongs to, or the one image it holds. In both cases `File.path` is the main file and `File.opened` is the `path` you gave.

#### Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `path` | `str` or `os.PathLike` | required | A file, a part of a multi-file image, or a folder. |
| `backend` | `str` or `None` | `None` | Force a backend by name (see [`BACKENDS`](constants.md#backends)). The named backend reads `path` itself, without redirecting parts to their main file. |
| `tiles` | `"separate"` or `"mosaic"` | `"separate"` | `"separate"`: each mosaic tile is its own image with a `tile` index. `"mosaic"`: a series' tiles placed at their stored positions on one canvas, one image per series. Nothing is registered or blended; where tiles overlap, the later tile supplies the pixel. Formats that define tile positions: CZI, Leica LIF / XLEF, FluoView `.omp2info`. A tiled file of another format, or tiles without a position, raises `UnsupportedFormatError`. `"stitched"` is the old name of `"mosaic"`: it still works and issues a `DeprecationWarning`. |
| `resolutions` | `"base"` or `"all"` | `"base"` | `"base"`: `File.images` holds the full-resolution images. `"all"`: every stored level is also an image, indexed by `resolution`. `Image.resolutions` gives the pyramid either way. |
| `rename` | `dict` or `None` | `None` | Make an index an array axis, e.g. `{"sequence": "z"}`. Keys: [`SAMPLED_INDICES`](constants.md#sampled_indices). Values: `"t"`, `"c"` or `"z"`, each at most once. TIFF only; ignored for files without that index. |
| `plates` | `bool` | `True` | Read a file that is a plate as a plate (`False`: by its series): `File.plates` describes it, and its images are indexed by `plate`, `acquisition`, `well` and `field` instead of `series`. Files that can be a plate: CZI; ND2 (XY points named after wells, or a NIS-Elements run folder of one ND2 file per well or point); OME companion files with `Plate` / `Well` / `WellSample`; an OME-TIFF whose OME-XML describes a plate (Yokogawa CQ1). No effect on other files. Formats that exist only as plates (Harmony, Columbus, MetaXpress, IN Cell, CellVoyager, scanR, Cellomics, ImageXpress HCS.ai, Opera Flex) are always plates. |
| `multifile` | `bool` | `True` | For a file that is one of several making a set: `True` opens the whole set, `False` only this file. Applies to an OME-TIFF whose XML lists planes in other files (`False`: the planes in this file, with the set's metadata), an OME-TIFF member that points to a companion file, a NIS-Elements ND2 well-plate run (read as a plate, the default), Opera Flex and Cellomics plates. |
| `axes` | `str` or `None` | `"canonical"` | Present every image with these axes: `"canonical"` gives `tczyx` (+ `s`), `None` keeps each image's stored order, letters such as `"zyx"` give that exact order. See [`Image.as_axes`](image.md#as_axes). |
| `samples` | `"channels"` or `"keep"` | `"channels"` | `"channels"`: RGB samples become channels. An RGB image has `c` 3 instead of an `s` axis, and its metadata lists one channel per sample, named `"<name> R"`, `"G"`, `"B"` (`"A"`), coloured red, green, blue (white for `A`). A single unnamed channel gives three unnamed, coloured ones; other sample counts give numbered, uncoloured channels. A file that already lists one channel per sample keeps its own. Only images with samples are changed (their `.source` is the stored image). `"keep"`: the samples stay a trailing `s` axis. The default was `"keep"` before 0.3.0. |
| `squeeze` | `bool` | `False` | Drop every size-1 axis except `y` and `x`. |

[`Image.as_axes`](image.md#as_axes) has its own `samples` parameter, which still defaults to `"keep"`.

#### Returns

A [`File`](file.md). When `axes` or `squeeze` is set, or `samples="channels"` folds an image's samples, that image's native form is its `.source`.

#### Raises

| Exception | When |
| --- | --- |
| `ValueError` | `tiles`, `resolutions`, `rename`, `axes` or `samples` has an invalid value; an image has an axis that `axes` cannot hold |
| `UnsupportedFormatError` | no backend recognises the file; `backend` names an unknown backend; `tiles="mosaic"` on a tiled file whose format cannot place its tiles; the backend cannot `rename` an index the file has; the file uses something not supported (the message says what) |
| `FileNotFoundError` | `path` does not exist |
| `CorruptFileError` | the file is recognised but inconsistent (e.g. truncated); `path` is a part whose main file is missing and that is not an image on its own; a file referenced by the format is missing |
| `UnsupportedFormatError` (folder) | no file in the folder is an image |
| `MicroReaderError` (folder) | the folder holds several images (the message lists them) |

#### Warns

`MissingImagesWarning`, `CompanionWarning`, `CompositeWarning`, `DamagedFileWarning`, `PlateLayoutWarning`, `StoredValuesWarning`, `SkewedGridWarning`; a `DeprecationWarning` for `tiles="stitched"`. See [Errors and warnings](errors.md).

#### Examples

```python
import micro_reader

# auto-detect, defaults
with micro_reader.open("cells.czi") as f:
    print(f)

# CZI tiles placed on one canvas per series, every image tczyx
f = micro_reader.open("slide.czi", tiles="mosaic", axes="canonical")

# RGB samples kept as a trailing s axis (the behaviour before 0.3.0)
f = micro_reader.open("rgb.tif", samples="keep")

# a plain multi-page TIFF that you know is a z-stack
f = micro_reader.open("stack.tif", rename={"sequence": "z"})

# every pyramid level as its own image
f = micro_reader.open("cells.ims", resolutions="all")

# a CZI recorded on a plate, as a plate (also ND2, OME companion, CQ1)
f = micro_reader.open("W96.czi", plates=True)

# only the planes of one file in a multi-file OME-TIFF set
f = micro_reader.open("set_t001.ome.tif", multifile=False)

# force the MRC backend for a file with an unusual extension
f = micro_reader.open("volume.dat", backend="mrc")
```

---

## `group_files`

```python
micro_reader.group_files(paths, plates=True) -> Grouping
```

Sort files into the images they make, without opening every file.

- Files that are parts of a **composite** image (an OIF folder, a VSI `_name_` folder, a DeltaVision log, OIR continuations, Micro-Manager and NDTiff stack files) are grouped under their main file.
- **Index** files (a plate index such as Harmony's or Columbus's, an Opera Flex folder, an OME companion, a Leica XLEF / XLCF / XLIF) are opened, and get the files their images are read from as `members`. A file that an index names is listed only in that index's group.
- Every other file is returned as a single, unread.

#### Parameters

| Parameter | Type | Description |
| --- | --- | --- |
| `paths` | `str`, `os.PathLike`, or an iterable of them | Files and folders. Folders are walked down with all their files. |
| `plates` | `bool` | Group as `open(plates=True)` reads: a set that is a plate only then (a NIS-Elements run folder of ND2 files) is one group. Default `True`. |

#### Returns

A [`Grouping`](grouping.md#grouping) with `groups` (a list of [`FileGroup`](grouping.md#filegroup), sorted by expected main file) and `singles` (a sorted list of paths as given).

#### Raises

`FileNotFoundError` if a given path does not exist. Errors from opening an index file are not raised. They are recorded in that group's `notes`.

#### Example

```python
g = micro_reader.group_files(["run1/", "run2/extra.tif"])
for grp in g.groups:
    print(grp.kind, grp.format, grp.main, len(grp.parts), len(grp.members))
print(len(g.singles), "single files")
```

See [Grouping many files](../guide/multi-file.md#grouping-many-files).

---

## `memory_limit`

```python
micro_reader.memory_limit() -> int
```

The decoding memory budget of this process, in bytes. If it has not been set, it is taken from `MICRO_READER_MEMORY_LIMIT`, or else a quarter of the computer's memory (4 GB if that cannot be found). It is never below 64 MB.

## `set_memory_limit`

```python
micro_reader.set_memory_limit(limit) -> None
```

| Parameter | Type | Description |
| --- | --- | --- |
| `limit` | `int`, `float`, `str` or `None` | Bytes, or a size such as `"8GB"`. Values below 64 MB are raised to 64 MB. `None` restores the default. |

Raises `ValueError` for a string that is not a size.

The limit covers blocks being decoded at the same time, across all threads. A block that must be decoded whole and is larger than the limit raises [`MemoryLimitError`](errors.md#memorylimiterror) when read. See [Memory and threads](../guide/performance.md#the-memory-limit).

## `cache_limit`

```python
micro_reader.cache_limit() -> int
```

What all decoded-block caches of this process may hold together, in bytes. Default: `MICRO_READER_CACHE_LIMIT`, else 256 MB.

## `set_cache_limit`

```python
micro_reader.set_cache_limit(limit) -> None
```

| Parameter | Type | Description |
| --- | --- | --- |
| `limit` | `int`, `float`, `str` or `None` | Bytes, or a size such as `"512MB"`. `0` keeps nothing. `None` restores the default. |

If the caches hold more than the new limit, the least recently used blocks are dropped at once.

## `cached_bytes`

```python
micro_reader.cached_bytes() -> int
```

What the decoded-block caches of this process hold now, in bytes.

## `decode_threads`

```python
micro_reader.decode_threads() -> int
```

The number of decoding threads. Default: `MICRO_READER_DECODE_THREADS`, else the number of CPUs.

## `set_decode_threads`

```python
micro_reader.set_decode_threads(n) -> None
```

| Parameter | Type | Description |
| --- | --- | --- |
| `n` | `int` | Number of threads. A value below 1 is raised to 1. With `1`, decoding runs on the calling thread, without a pool. |

Unlike the other setters, `None` is not accepted (`TypeError`); to restore the default, pass the number of CPUs.

The old pool is shut down without waiting. Reads in progress finish on it, and new reads use the new pool.
