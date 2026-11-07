# Image

```python
class micro_reader.Image
```

One readable array of a file. You get images from `File.images`, `File.select()`, plate navigation, `Image.resolutions` and `Image.parts`. Images do not hold pixels. Every read goes to the file and returns a new NumPy array.

```python
image = f.images[0]
image
# <micro_reader.Image 'cells.czi series 0' index={'series': 0} axes='tczyx'
#  shape=(1, 3, 41, 1000, 1500) dtype=uint16 (any rectangle)>
```

The end of the printout shows the read unit (`any rectangle`, or `unit (1, 1, 64, 64, 64)`). An image made by `as_axes` adds `of 'zyx'`, its native axes.

## Attributes

| Attribute | Type | Description |
| --- | --- | --- |
| `axes` | `str` | Lower-case axis letters from `tczyxs`, e.g. `"tczyx"`, `"zyx"`, `"yxs"`. |
| `shape` | `tuple[int, ...]` | Size of each axis, in the order of `axes`. |
| `dtype` | `numpy.dtype` | Pixel type, always in native byte order. |
| `ndim` | `int` | `len(shape)`. |
| `nbytes` | `int` | Size of the whole image in bytes. |
| `name` | `str` | A readable name, by one rule for every format: a plate's image is `"well B2 field 0"`; otherwise the format's own name (Leica element, OME `Image Name`, Imaris, slide roles), else a label the file records (CZI scene, ND2 point, Micro-Manager position), else the index values that tell the file's images apart (`"series 2"`), else, for one image, the main file's name without its extension. Duplicates get what differs appended. |
| `index` | `ImageIndex` (a `dict[str, int]`) | Which image of the file this is, e.g. `{"series": 0, "tile": 3}`; in a plate, `{"plate": 0, "acquisition": 0, "well": 3, "field": 0}`. Only the names the file has. Storing or reading the old name of `series` (before micro-reader 0.1.0) raises a `KeyError` that names `series`; `dict(image.index)` is a plain copy. |
| `read_unit` | `tuple[int, ...]` or `None` | `None`: any rectangle is read exactly. Otherwise, the smallest block read at once, per axis. |
| `stats` | [`ReadStats`](#readstats) | What this image has read so far. |
| `metadata` | [`Metadata`](metadata.md#metadata) | Pixel sizes, position, coordinate system, channels and index values, read on first access. |
| `source` | `Image` | **Images made by `as_axes` / `squeeze` / `open(axes=...)` only, and RGB images of `open` (its default `samples="channels"`).** The native image. |
| `origin` | `tuple[int, int]` | **CZI only:** `(x, y)` of the image's corner in the file's pixel coordinates. **Images made with `tiles="mosaic"` from other formats:** `(y, x)`, the canvas corner in the tiles' pixel grid. Note the order. |
| `tiles` | `list[tuple[Image, int, int]]` | **Images made with `tiles="mosaic"` from other formats only.** `(tile, y, x)` for each tile and its corner on the canvas, in drawing order (later on top). |
| `palette` | `numpy.ndarray` or `None` | **PNG / BMP only.** Colour palette `(n, 3 or 4)` `uint8`, when pixels are palette indices. |

### Pyramid attributes

| Attribute | Type | Description |
| --- | --- | --- |
| `resolutions` | `list[Image]` | The pyramid, full resolution first. `[self]` when the file stores no lower levels. |
| `is_pyramidal` | `bool` | `len(resolutions) > 1`. |
| `level` | `int` | This image's level: 0 for full resolution. |
| `pyramid` | `Image` | The level-0 image of this pyramid (itself at level 0). |
| `is_pyramid` | `bool` | Deprecated since 0.4.0: use `image.level == 0`. |

### Ragged-image attributes

| Attribute | Type | Description |
| --- | --- | --- |
| `parts` | `list[Image]` | The rectangular parts: channel groups that share one sampling. `[self]` for an image that is not ragged. |
| `is_ragged` | `bool` | `len(parts) > 1`. |
| `whole` | `Image` | The image a part belongs to. Itself for a whole. |

## Reading

### `__getitem__`

```python
Image[key] -> numpy.ndarray
```

NumPy basic indexing over `axes`, in order.

| Key element | Effect |
| --- | --- |
| `int` (negative allowed) | that position; the axis is removed |
| `slice` with step ≥ 1 | that span; the axis is kept |
| `...` (once) | all axes in between |
| missing trailing keys | whole axes |

Raises `IndexError` for `None` / `np.newaxis`, negative steps, lists, arrays, booleans, more than one `...`, too many keys, and out-of-bounds integers. Slices are clipped to the shape. Steps are applied after reading: a span up to 64 MB (`STEPPED_READ_BYTES`) is read whole and thinned; a larger one a piece of at most that size at a time (runs of planes or rows, or only the selected planes when one step spans more), so the memory a stepped read takes is its result and one piece.

### `read`

```python
Image.read(**selection) -> numpy.ndarray
```

The keyword form: `image.read(z=3, y=slice(0, 512))`. Each keyword is an axis letter of this image, with an `int` or a `slice`. Omitted axes are read whole. Raises `KeyError` for an axis the image does not have.

```python
image.read()                                  # everything
image.read(t=0, c=slice(0, 2), z=10)          # -> (2, ny, nx)
image.read(y=slice(0, None, 2), x=slice(0, None, 2))  # every second pixel
```

### `read_region`

```python
Image.read_region(x, y, width, height, **planes) -> numpy.ndarray     # CZI only
```

A rectangle in the file's own pixel coordinates (see `origin`), composed from the series' tiles. Parts outside the image are 0. `planes` selects the other axes like `read()`. Raises `ValueError` for a negative size.

## Layout

### `as_axes`

```python
Image.as_axes(axes=None, *, samples="keep", squeeze=False) -> Image
```

This image presented with other axes. Reads stay exact. Calling it does not read anything.

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `axes` | `str` or `None` | `None` | `None`: keep the order. `"canonical"`: `tczyx`, then `s` if present (and not folded). Letters: that exact order. Missing axes are inserted with size 1. |
| `samples` | `"keep"` or `"channels"` | `"keep"` | `"channels"` folds RGB samples into `c`: channel `k` is sample `k % S` of channel `k // S`. Needs `c` in `axes` when `axes` is given. Unlike [`open`](functions.md#open), whose `samples` defaults to `"channels"`, `as_axes` keeps `s` by default. |
| `squeeze` | `bool` or `str` | `False` | `True`: drop every size-1 axis except `y` and `x`. A string such as `"tz"`: drop exactly those, each of which must have size 1. |

Raises `ValueError` when an axis of the image is left out and is not of size 1, when a letter is not one of `tczyxs` or is repeated, or when a squeezed axis is not of size 1.

The result's `source` is the native image. Its `metadata` (with the channels of the folded samples) and `index` follow the native image. Its `resolutions` are the native levels, each with the same layout. Calling `as_axes` on it starts again from the native image.

### `squeeze`

```python
Image.squeeze(axes=None) -> Image
```

This image without size-1 axes: all except `y` and `x`, or only those named in `axes` (each must have size 1, else `ValueError`).

## Other

### `close`

```python
Image.close() -> None
```

Release the image's handles. Usually you close the `File` instead, which closes all its images. Reads afterwards raise `ValueError`.

### Thread safety

Reads from several threads, on one image or on several images of a file, are safe and are the intended way to read in parallel. Each thread gets its own file handle. `stats` is updated under a lock.

---

## ReadStats

```python
class micro_reader.ReadStats
```

What an image has read since it was opened, or since `reset()`.

| Attribute | Type | Description |
| --- | --- | --- |
| `reads` | `int` | Number of reads. |
| `pixels_returned` | `int` | Pixels in the requested spans, before steps. |
| `pixels_read` | `int` | Pixels the backend actually read or decoded to produce them. |
| `over_read` | `float` | `pixels_read / pixels_returned`; 1.0 before the first read. |

| Method | Description |
| --- | --- |
| `reset()` | Set all counts to 0. |

```python
image.stats.reset()
image.read(z=0)
print(f"{image.stats.over_read:.2f}x")
```
