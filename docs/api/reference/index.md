# API reference

Everything listed here is importable from the top-level `micro_reader` package. Anything else, including modules whose names start with `_` and the backend classes, is internal and may change without notice.

```python
import micro_reader
```

## Functions

| Name | Purpose |
| --- | --- |
| [`open(path, ...)`](functions.md#open) | Opens a file, folder or file set and returns a [`File`](file.md). |
| [`group_files(paths)`](functions.md#group_files) | Sorts many files into the images they make and returns a [`Grouping`](grouping.md). |
| [`memory_limit()`](functions.md#memory_limit) / [`set_memory_limit(x)`](functions.md#set_memory_limit) | The process-wide memory limit for decoding. |
| [`cache_limit()`](functions.md#cache_limit) / [`set_cache_limit(x)`](functions.md#set_cache_limit) / [`cached_bytes()`](functions.md#cached_bytes) | The decoded-block cache. |
| [`decode_threads()`](functions.md#decode_threads) / [`set_decode_threads(n)`](functions.md#set_decode_threads) | The decode thread pool. |

## Classes

| Name | What it is |
| --- | --- |
| [`File`](file.md) | An opened file, with its images, plates, backend and format. |
| [`Image`](image.md) | One readable array, with its index, axes, metadata and pyramid. |
| [`ReadStats`](image.md#readstats) | The counts of what an image has read so far. |
| [`Metadata`](metadata.md#metadata) | Pixel sizes, position, channels and index values. |
| [`Channel`](metadata.md#channel) | The name, colour and wavelengths of one channel. |
| [`Quantity`](metadata.md#quantity) | A value with an OME-Zarr unit. |
| [`Plate`](plates.md#plate), [`Well`](plates.md#well), [`Field`](plates.md#field), [`Acquisition`](plates.md#acquisition) | The plate layout and how to navigate it. |
| [`CoordinateSystem`](coordinates.md#coordinatesystem), [`Axis`](coordinates.md#axis), [`Transformation`](coordinates.md#transformation), [`Collection`](coordinates.md#collection) | Where the images sit, and the file's tree. |
| [`Grouping`](grouping.md#grouping), [`FileGroup`](grouping.md#filegroup) | The result of `group_files`. |

## Errors and warnings

| Name | Kind |
| --- | --- |
| [`MicroReaderError`](errors.md#microreadererror) | The base class of the errors below. |
| [`UnsupportedFormatError`](errors.md#unsupportedformaterror) | Error. |
| [`CorruptFileError`](errors.md#corruptfileerror) | Error. |
| [`MemoryLimitError`](errors.md#memorylimiterror) | Error. |
| [`InvariantError`](errors.md#invarianterror) | Error. |
| [`DamagedFileWarning`](errors.md#damagedfilewarning) | Warning. |
| [`MissingImagesWarning`](errors.md#missingimageswarning) | Warning. |
| [`CompanionWarning`](errors.md#companionwarning) | Warning. |
| [`CompositeWarning`](errors.md#compositewarning) | Warning. |
| [`PlateLayoutWarning`](errors.md#platelayoutwarning) | Warning. |
| [`SkewedGridWarning`](errors.md#skewedgridwarning) | Warning. |
| [`StoredValuesWarning`](errors.md#storedvalueswarning) | Warning. |

## Constants

| Name | Value |
| --- | --- |
| [`INDEX_NAMES`](constants.md#index_names) | Every index name, outermost first. |
| [`PLATE_INDICES`](constants.md#plate_indices) | `("plate", "acquisition", "well", "field")` |
| [`SAMPLED_INDICES`](constants.md#sampled_indices) | The indices that `rename` can turn into axes. |
| [`BACKENDS`](constants.md#backends) | The backend modules, in the order they are tried. |
| `__version__` | The package version, for example `"0.4.4"`. |

## Conventions on these pages

- Signatures are given as Python. A `*` marks the start of the keyword-only parameters.
- A *size* argument accepts an `int` in bytes, or a `str` with binary units such as `"8GB"` or `"512 MB"`.
- An *index* is an image's `dict` of index names to integers, for example `{"series": 0, "tile": 3}`.
- All attributes are read-only unless stated otherwise. Changing them is not supported.
- Plate rows and columns count from 0. The row and column *names* of a plate, such as `"C"` and `"3"`, are in `Plate.row_names` and `Plate.column_names`.
