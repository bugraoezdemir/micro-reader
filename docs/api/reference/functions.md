# Functions

## `open`

```python
micro_reader.open(path, backend=None, *, tiles="separate", resolutions="base",
                  rename=None, plates=True, multifile=True,
                  axes="canonical", samples="channels", squeeze=False) -> File
```

Opens `path` with the backend that recognises its content, or with the backend you name.

If `path` is one part of a multi-file image, the whole image is opened. Examples are an OIR continuation, a DeltaVision log, an OIF plane and a plate side file. A folder opens the image it belongs to, or the one image it holds. In both cases `File.path` is the main file and `File.opened` is the `path` you gave.

#### Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `path` | `str` or `os.PathLike` | required | A file, a part of a multi-file image, or a folder. |
| `backend` | `str` or `None` | `None` | Forces a backend by name (see [`BACKENDS`](constants.md#backends)). The named backend reads `path` itself and does not redirect parts to their main file. |
| `tiles` | `"separate"` or `"mosaic"` | `"separate"` | With `"separate"`, each mosaic tile is its own image with a `tile` index. With `"mosaic"`, the tiles of a series are placed at their stored positions on one canvas, which gives one image per series. Nothing is registered or blended, and where tiles overlap, the later tile supplies the pixel. The formats that define tile positions are CZI, Leica LIF / XLEF, FluoView `.omp2info` and BigDataViewer / BigStitcher (nominal positions; turned, sheared or mirrored views are refused). A tiled file of another format, or tiles without a position, raises `UnsupportedFormatError`. A file without tiles opens as usual. `"stitched"` is the old name of `"mosaic"`. It still works and issues a `DeprecationWarning`. |
| `resolutions` | `"base"` or `"all"` | `"base"` | With `"base"`, `File.images` holds the full-resolution images. With `"all"`, every stored level is also an image, indexed by `resolution`. `Image.resolutions` gives the pyramid either way. |
| `rename` | `dict` or `None` | `None` | Turns an index into an array axis, for example `{"sequence": "z"}`. The keys come from [`SAMPLED_INDICES`](constants.md#sampled_indices). The values are `"t"`, `"c"` or `"z"`, each at most once. This works for TIFF only, and it is ignored for files without that index. |
| `plates` | `bool` | `True` | Reads a file that is a plate as a plate. With `False`, it is read by its series instead. When it is read as a plate, `File.plates` describes it and its images are indexed by `plate`, `acquisition`, `well` and `field` instead of `series`. The files that can be a plate are CZI, ND2, OME companion files with `Plate` / `Well` / `WellSample`, and an OME-TIFF whose OME-XML describes a plate (Yokogawa CQ1). For ND2, this means XY points named after wells, or a NIS-Elements run folder with one ND2 file per well or point. The option has no effect on other files. Formats that exist only as plates are always plates. These are Harmony, Columbus, MetaXpress, IN Cell, CellVoyager, scanR, Cellomics, ImageXpress HCS.ai and Opera Flex. |
| `multifile` | `bool` | `True` | For a file that is one of several making a set, `True` opens the whole set and `False` opens only this file. It applies to an OME-TIFF whose XML lists planes in other files. With `False`, you get the planes in this file together with the set's metadata. It also applies to an OME-TIFF member that points to a companion file, to a NIS-Elements ND2 well-plate run (read as a plate by default), and to Opera Flex and Cellomics plates. |
| `axes` | `str` or `None` | `"canonical"` | Presents every image with these axes. `"canonical"` gives `tczyx`, plus `s` if the image has samples. `None` keeps each image's stored order. Letters such as `"zyx"` give that exact order. See [`Image.as_axes`](image.md#as_axes). |
| `samples` | `"channels"` or `"keep"` | `"channels"` | With `"channels"`, RGB samples become channels. An RGB image then has `c` of 3 instead of an `s` axis. Its metadata lists one channel per sample, named `"<name> R"`, `"G"`, `"B"` and `"A"`, and coloured red, green, blue and white. A single unnamed channel gives three unnamed, coloured channels, and other sample counts give numbered, uncoloured channels. A file that already lists one channel per sample keeps its own. Only images with samples are changed, and their `.source` is the stored image. With `"keep"`, the samples stay a trailing `s` axis. Changed in 0.3.0: the default was `"keep"` before. |
| `squeeze` | `bool` | `False` | Drops every size-1 axis except `y` and `x`. |

[`Image.as_axes`](image.md#as_axes) has its own `samples` parameter, and that one still defaults to `"keep"`.

#### Returns

A [`File`](file.md). When `axes` or `squeeze` is set, or `samples="channels"` folds the samples of an image, the native form of that image is its `.source`.

#### Raises

| Exception | When |
| --- | --- |
| `ValueError` | `tiles`, `resolutions`, `rename`, `axes` or `samples` has an invalid value, or an image has an axis that `axes` cannot hold. |
| `UnsupportedFormatError` | No backend recognises the file. Or `backend` names an unknown backend. Or `tiles="mosaic"` is used on a tiled file whose format cannot place its tiles. Or the backend cannot `rename` an index the file has. Or the file uses something that is not supported, and the message says what. |
| `FileNotFoundError` | `path` does not exist. |
| `CorruptFileError` | The file is recognised but inconsistent, for example truncated. Or `path` is a part whose main file is missing and that is not an image on its own. Or a file that the format refers to is missing. |
| `UnsupportedFormatError` (folder) | No file in the folder is an image. |
| `MicroReaderError` (folder) | The folder holds several images, and the message lists them. |

#### Warns

`MissingImagesWarning`, `CompanionWarning`, `CompositeWarning`, `DamagedFileWarning`, `PlateLayoutWarning`, `StoredValuesWarning` and `SkewedGridWarning`. `tiles="stitched"` issues a `DeprecationWarning`. See [Errors and warnings](errors.md).

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

Sorts files into the images they make, without opening every file.

- **Composite images.** Files that are parts of a composite image are grouped under their main file. Examples are an OIF folder, a VSI `_name_` folder, a DeltaVision log, OIR continuations, and Micro-Manager and NDTiff stack files.
- **Index files.** An index file is opened, and the files its images are read from become its `members`. Examples are a plate index such as Harmony's or Columbus's, an Opera Flex folder, an OME companion and a Leica XLEF / XLCF / XLIF. A file that an index names is listed only in the group of that index.
- **Other files.** Every other file is returned as a single and is not read.

#### Parameters

| Parameter | Type | Description |
| --- | --- | --- |
| `paths` | `str`, `os.PathLike`, or an iterable of them | Files and folders. Folders are walked down, and all their files are included. |
| `plates` | `bool` | Groups the files the way `open(plates=True)` reads them. A set that is a plate only in that case, such as a NIS-Elements run folder of ND2 files, is one group. The default is `True`. |

#### Returns

A [`Grouping`](grouping.md#grouping). Its `groups` is a list of [`FileGroup`](grouping.md#filegroup), sorted by expected main file. Its `singles` is a sorted list of paths as given.

#### Raises

`FileNotFoundError` if a given path does not exist. Errors from opening an index file are not raised. They are recorded in the `notes` of that group.

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

The process-wide memory limit for decoding, in bytes. If it has not been set, it is taken from `MICRO_READER_MEMORY_LIMIT`. Without that variable, it is a quarter of the computer's memory, or 4 GB if the memory size cannot be found. It is never below 64 MB.

## `set_memory_limit`

```python
micro_reader.set_memory_limit(limit) -> None
```

| Parameter | Type | Description |
| --- | --- | --- |
| `limit` | `int`, `float`, `str` or `None` | Bytes, or a size such as `"8GB"`. Values below 64 MB are raised to 64 MB. `None` restores the default. |

Raises `ValueError` for a string that is not a size.

The limit is a safeguard for decoding. It covers the blocks that are being decoded at the same time, across all threads. A decode that does not fit waits until other decodes finish. Only a single block that must be decoded whole and is larger than the whole limit raises [`MemoryLimitError`](errors.md#memorylimiterror), when it is read. The array that a read returns is not counted against the limit. The memory a read needs grows with the region you ask for, not with the size of the file. See [Memory and threads](../guide/performance.md#the-memory-limit).

## `cache_limit`

```python
micro_reader.cache_limit() -> int
```

The amount that all decoded-block caches of this process may hold together, in bytes. The default is `MICRO_READER_CACHE_LIMIT`, or 256 MB without that variable.

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

The amount that the decoded-block caches of this process hold now, in bytes.

## `decode_threads`

```python
micro_reader.decode_threads() -> int
```

The number of decoding threads. The default is `MICRO_READER_DECODE_THREADS`, or the number of CPUs without that variable.

## `set_decode_threads`

```python
micro_reader.set_decode_threads(n) -> None
```

| Parameter | Type | Description |
| --- | --- | --- |
| `n` | `int` | The number of threads. A value below 1 is raised to 1. With `1`, decoding runs on the calling thread and no pool is used. |

Unlike the other setters, this one does not accept `None` and raises `TypeError`. To restore the default, pass the number of CPUs.

The old pool is shut down without waiting. Reads in progress finish on it, and new reads use the new pool.
