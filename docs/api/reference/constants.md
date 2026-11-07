# Constants

## INDEX_NAMES

```python
micro_reader.INDEX_NAMES
# ('plate', 'acquisition', 'well', 'field',
#  'series', 'tile', 'rotation', 'view', 'illumination', 'phase', 'block',
#  'emission', 'excitation', 'lifetime', 'sequence', 'part', 'resolution')
```

Every index name, outermost first. `File.select()` and `Field.select()` accept these names. `File.select()` also takes a plate's `row`, `column` and the name keywords (`plate_name`, `acquisition_name`, `well_name`, `row_name`, `column_name`), which are not index names: they are not on the images. The [Concepts](../concepts.md#all-index-names) page explains each one.

## PLATE_INDICES

```python
micro_reader.PLATE_INDICES
# ('plate', 'acquisition', 'well', 'field')
```

The indices on every image of a plate. `plate` and `acquisition` are 0 when a file holds one of each.

## SAMPLED_INDICES

```python
micro_reader.SAMPLED_INDICES
# ('illumination', 'phase', 'block', 'emission', 'excitation', 'lifetime', 'sequence')
```

The indices that are not separate frames but dimensions sampled at the same points. Other readers often keep these as array axes. These are the keys that `open(rename=...)` accepts.

## BACKENDS

```python
[b.NAME for b in micro_reader.BACKENDS]
# ['vsi', 'flex', 'tiff', 'czi', 'luxendo', 'bdv', 'ilastik', 'hdf5', 'ims', 'nd2', 'oir',
#  'omp2info', 'oif', 'lif', 'harmony', 'columbus', 'scanr', 'cellomics', 'jdce',
#  'metaxpress', 'metamorph', 'micromanager', 'incell', 'cv7000', 'companion',
#  'ndpis', 'ics', 'raster', 'dv', 'mrc']
```

The backend modules, in the order they are tried when detecting a format. Their `NAME` is what `open(backend=...)` takes and what `File.backend` reports. The modules themselves are internal. Use only `NAME`.

## `__version__`

```python
micro_reader.__version__      # '0.3.0'
```

The installed package version.

## Environment variables

| Variable | Effect | Default |
| --- | --- | --- |
| `MICRO_READER_MEMORY_LIMIT` | default of [`memory_limit()`](functions.md#memory_limit) | ¼ of RAM |
| `MICRO_READER_CACHE_LIMIT` | default of [`cache_limit()`](functions.md#cache_limit) | `256MB` |
| `MICRO_READER_DECODE_THREADS` | default of [`decode_threads()`](functions.md#decode_threads) | number of CPUs |
| `MICRO_READER_SKIP_EUBI_CHECK` | `1` skips the import check that refuses to load next to an eubi-bridge older than 0.3.0 (which reads the old index name `scene`), with one warning. Any other value keeps the check. For work on an eubi checkout only. | unset |

The first three are read the first time the setting is needed, so set them before the first read. A value that is not a size (or, for threads, not a number) raises `ValueError` then. `set_memory_limit(None)` and `set_cache_limit(None)` read their variable again. `set_decode_threads` takes a number only (`None` raises `TypeError`).
