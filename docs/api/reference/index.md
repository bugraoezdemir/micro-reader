# API reference

Everything listed here is importable from the top-level `micro_reader` package. Anything else, including modules whose names start with `_` and the backend classes, is internal and may change without notice.

```python
import micro_reader
```

## Functions

| Name | Purpose |
| --- | --- |
| [`open(path, ...)`](functions.md#open) | Open a file, folder or file set → [`File`](file.md) |
| [`group_files(paths)`](functions.md#group_files) | Sort many files into the images they make → [`Grouping`](grouping.md) |
| [`memory_limit()`](functions.md#memory_limit) / [`set_memory_limit(x)`](functions.md#set_memory_limit) | Decoding memory budget |
| [`cache_limit()`](functions.md#cache_limit) / [`set_cache_limit(x)`](functions.md#set_cache_limit) / [`cached_bytes()`](functions.md#cached_bytes) | Decoded-block cache |
| [`decode_threads()`](functions.md#decode_threads) / [`set_decode_threads(n)`](functions.md#set_decode_threads) | Decode thread pool |

## Classes

| Name | What it is |
| --- | --- |
| [`File`](file.md) | An opened file: its images, plates, backend and format |
| [`Image`](image.md) | One readable array, with index, axes, metadata and pyramid |
| [`ReadStats`](image.md#readstats) | What an image has read so far |
| [`Metadata`](metadata.md#metadata) | Pixel sizes, position, channels, index values |
| [`Channel`](metadata.md#channel) | One channel's name, colour and wavelengths |
| [`Quantity`](metadata.md#quantity) | A value with an OME-Zarr unit |
| [`Plate`](plates.md#plate), [`Well`](plates.md#well), [`Field`](plates.md#field), [`Acquisition`](plates.md#acquisition) | Plate layout and navigation |
| [`CoordinateSystem`](coordinates.md#coordinatesystem), [`Axis`](coordinates.md#axis), [`Transformation`](coordinates.md#transformation), [`Collection`](coordinates.md#collection) | Where images sit, and the file's tree |
| [`Grouping`](grouping.md#grouping), [`FileGroup`](grouping.md#filegroup) | Result of `group_files` |

## Errors and warnings

| Name | Kind |
| --- | --- |
| [`MicroReaderError`](errors.md#microreadererror) | base class of the errors below |
| [`UnsupportedFormatError`](errors.md#unsupportedformaterror) | error |
| [`CorruptFileError`](errors.md#corruptfileerror) | error |
| [`MemoryLimitError`](errors.md#memorylimiterror) | error |
| [`InvariantError`](errors.md#invarianterror) | error |
| [`DamagedFileWarning`](errors.md#damagedfilewarning) | warning |
| [`MissingImagesWarning`](errors.md#missingimageswarning) | warning |
| [`CompanionWarning`](errors.md#companionwarning) | warning |
| [`CompositeWarning`](errors.md#compositewarning) | warning |
| [`PlateLayoutWarning`](errors.md#platelayoutwarning) | warning |
| [`SkewedGridWarning`](errors.md#skewedgridwarning) | warning |
| [`StoredValuesWarning`](errors.md#storedvalueswarning) | warning |

## Constants

| Name | Value |
| --- | --- |
| [`INDEX_NAMES`](constants.md#index_names) | every index name, outermost first |
| [`PLATE_INDICES`](constants.md#plate_indices) | `("plate", "acquisition", "well", "field")` |
| [`SAMPLED_INDICES`](constants.md#sampled_indices) | indices `rename` can turn into axes |
| [`BACKENDS`](constants.md#backends) | the backend modules, in detection order |
| `__version__` | the package version, e.g. `"0.3.0"` |

## Conventions on these pages

- Signatures are given as Python. `*` marks the start of keyword-only parameters.
- *Size* arguments accept an `int` (bytes) or a `str` with binary units such as `"8GB"` or `"512 MB"`.
- *Index* means an image's `dict` of index names to integers, for example `{"series": 0, "tile": 3}`.
- All attributes are read-only unless stated otherwise. Changing them is not supported.
- Plate rows and columns count from 0. A plate's row and column *names* (`"C"`, `"3"`) are in `Plate.row_names` and `Plate.column_names`.
