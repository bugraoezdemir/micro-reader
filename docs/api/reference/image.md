# Image

```python
class micro_reader.Image
```

One readable array of a file. You get images from `File.images`, `File.select()`, plate navigation, `Image.resolutions` and `Image.parts`. Images do not hold pixels. Every read goes to the file and returns a new NumPy array. Only the strips, tiles, subblocks or chunks that overlap the region you ask for are read and decoded, so the memory a read needs grows with that region and not with the size of the file.

```python
image = f.images[0]
image
# <micro_reader.Image 'cells' index={'series': 0} axes='tczyx'
#  shape=(1, 3, 41, 1000, 1500) dtype=uint16 (any rectangle)>
```

The end of the printout shows the read unit, which is `any rectangle` or a unit such as `unit (1, 1, 64, 64, 64)`. An image made by `as_axes` adds its native axes, for example `of 'zyx'`.

## Attributes

| Attribute | Type | Description |
| --- | --- | --- |
| `axes` | `str` | Lower-case axis letters from `tczyxs`, for example `"tczyx"`, `"zyx"` or `"yxs"`. |
| `shape` | `tuple[int, ...]` | The size of each axis, in the order of `axes`. |
| `dtype` | `numpy.dtype` | The pixel type, always in native byte order. |
| `ndim` | `int` | `len(shape)`. |
| `nbytes` | `int` | The size of the whole image in bytes. |
| `name` | `str` | A readable name, chosen by one rule for every format. The image of a plate is named like `"well B2 field 0"`. Otherwise the name is the format's own name, such as a Leica element, an OME `Image Name`, an Imaris name or a slide role. If there is none, it is a label the file records, such as a CZI scene, an ND2 point or a Micro-Manager position. If there is none, it is the index values that tell the images of the file apart, such as `"series 2"`. For a file with one image, it is the name of the main file without its extension. Duplicate names get what differs appended. |
| `index` | `ImageIndex` (a `dict[str, int]`) | Which image of the file this is, for example `{"series": 0, "tile": 3}`. In a plate it is like `{"plate": 0, "acquisition": 0, "well": 3, "field": 0}`. It holds only the names the file has. Storing or reading the old name `scene` raises a `KeyError` that points to `series`. `dict(image.index)` is a plain copy. Changed in 0.1.0: the index `scene` was renamed to `series`. |
| `read_unit` | `tuple[int, ...]` or `None` | `None` means that any rectangle is read exactly. Otherwise, it is the smallest block that is read at once, per axis. |
| `stats` | [`ReadStats`](#readstats) | The counts of what this image has read so far. |
| `metadata` | [`Metadata`](metadata.md#metadata) | Pixel sizes, position, coordinate system, channels and index values. They are read on first access. |
| `source` | `Image` | **Only for images made by `as_axes`, `squeeze` or `open(axes=...)`, and for RGB images of `open` with its default `samples="channels"`.** The native image. |
| `origin` | `tuple[int, int]` | **Only for CZI images and images made with `tiles="mosaic"`.** The `(y, x)` of the corner of the image, in the order of the axes of the image. On CZI, it is in the pixel coordinates of the file itself, which libCZI and ZEN write with x first. On mosaics from other formats, it is in the pixel grid of the tiles. The tiles are placed from the corner of the canvas, so it is `(0, 0)`, and the stage position of the corner is `metadata.position`. Changed in 0.4.2: CZI images gave `(x, y)` before. |
| `tiles` | `list[tuple[Image, int, int]]` | **Only for images made with `tiles="mosaic"` from other formats.** A `(tile, y, x)` entry for each tile and its corner on the canvas, in drawing order. Later tiles are drawn on top. |
| `palette` | `numpy.ndarray` or `None` | **Only for PNG and BMP.** The colour palette, of shape `(n, 3 or 4)` and type `uint8`, when the pixels are palette indices. |

### Pyramid attributes

| Attribute | Type | Description |
| --- | --- | --- |
| `resolutions` | `list[Image]` | The pyramid, with full resolution first. It is `[self]` when the file stores no lower levels. |
| `is_pyramidal` | `bool` | `len(resolutions) > 1`. |
| `level` | `int` | The level of this image. Level 0 is full resolution. |
| `pyramid` | `Image` | Deprecated since 0.5.1. Use `image.resolutions[0]` for the level-0 image, or `image.level == 0` to check for level 0. |
| `is_pyramid` | `bool` | Deprecated since 0.4.0. Use `image.level == 0` instead. |

### Ragged-image attributes

| Attribute | Type | Description |
| --- | --- | --- |
| `parts` | `list[Image]` | The rectangular parts, which are channel groups that share one sampling. It is `[self]` for an image that is not ragged. |
| `is_ragged` | `bool` | `len(parts) > 1`. |
| `whole` | `Image` | The image that a part belongs to. For a whole image, it is the image itself. |

## Reading

### `__getitem__`

```python
Image[key] -> numpy.ndarray
```

NumPy basic indexing over `axes`, in order. Each element of the key has the following effect.

| Key element | Effect |
| --- | --- |
| `int` (negative allowed) | Selects that position and removes the axis. |
| `slice` with step ≥ 1 | Selects that span and keeps the axis. |
| `...` (once) | Stands for all axes in between. |
| missing trailing keys | Read the whole axes. |

`IndexError` is raised for `None` and `np.newaxis`, negative steps, lists, arrays, booleans, more than one `...`, too many keys, and out-of-bounds integers. Slices are clipped to the shape. Steps are applied after reading. A span of up to 64 MB (`STEPPED_READ_BYTES`) is read whole and then thinned. A larger span is read one piece at a time, where each piece is at most that size. A piece is a run of planes or rows, or only the selected planes when one step spans more. So a stepped read needs memory for its result and one piece.

### `read`

```python
Image.read(**selection) -> numpy.ndarray
```

The keyword form, for example `image.read(z=3, y=slice(0, 512))`. Each keyword is an axis letter of this image, with an `int` or a `slice`. Omitted axes are read whole. It raises `KeyError` for an axis the image does not have.

```python
image.read()                                  # everything
image.read(t=0, c=slice(0, 2), z=10)          # -> (2, ny, nx)
image.read(y=slice(0, None, 2), x=slice(0, None, 2))  # every second pixel
```

### `read_region`

```python
Image.read_region(*, y, x, height, width, **planes) -> numpy.ndarray     # CZI only
```

Reads a rectangle in the pixel coordinates of the file itself (see `origin`), composed from the tiles of the series. Parts outside the image are 0. `planes` selects the other axes like `read()`. It raises `ValueError` for a negative size. All arguments are keywords. A positional call raises `TypeError`, so that it cannot read with y and x swapped. Changed in 0.4.2: it took `(x, y, width, height)` by position before.

## Layout

### `as_axes`

```python
Image.as_axes(axes=None, *, samples="keep", squeeze=False) -> Image
```

Returns this image presented with other axes. Reads stay exact, and calling it does not read anything.

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `axes` | `str` or `None` | `None` | `None` keeps the order. `"canonical"` gives `tczyx`, then `s` if it is present and not folded. Letters give that exact order. Missing axes are inserted with size 1. |
| `samples` | `"keep"` or `"channels"` | `"keep"` | `"channels"` folds RGB samples into `c`, so that channel `k` is sample `k % S` of channel `k // S`. It needs `c` in `axes` when `axes` is given. [`open`](functions.md#open) defaults to `"channels"`, but `as_axes` keeps `s` by default. |
| `squeeze` | `bool` or `str` | `False` | `True` drops every size-1 axis except `y` and `x`. A string such as `"tz"` drops exactly those axes, and each of them must have size 1. |

It raises `ValueError` when an axis of the image is left out and is not of size 1, when a letter is not one of `tczyxs` or is repeated, or when a squeezed axis is not of size 1.

The `source` of the result is the native image. Its `metadata` and `index` follow the native image, and the metadata has the channels of the folded samples. Its `resolutions` are the native levels, each with the same layout. Calling `as_axes` on the result starts again from the native image.

### `squeeze`

```python
Image.squeeze(axes=None) -> Image
```

Returns this image without its size-1 axes. By default it drops all of them except `y` and `x`. With `axes`, it drops only the axes named there, and each of them must have size 1, otherwise `ValueError` is raised.

## Other

### `close`

```python
Image.close() -> None
```

Releases the handles of the image. Usually you close the `File` instead, which closes all its images. Reads afterwards raise `ValueError`.

### Thread safety

Reads from several threads are safe, on one image or on several images of a file, and this is the intended way to read in parallel. Each thread gets its own file handle. `stats` is updated under a lock.

---

## ReadStats

```python
class micro_reader.ReadStats
```

The counts of what an image has read since it was opened, or since `reset()`.

| Attribute | Type | Description |
| --- | --- | --- |
| `reads` | `int` | The number of reads. |
| `pixels_returned` | `int` | The pixels in the requested spans, before steps. |
| `pixels_read` | `int` | The pixels that the backend actually read or decoded to produce them. |
| `over_read` | `float` | The ratio `pixels_read / pixels_returned`. It is 1.0 before the first read. |

| Method | Description |
| --- | --- |
| `reset()` | Sets all counts to 0. |

```python
image.stats.reset()
image.read(z=0)
print(f"{image.stats.over_read:.2f}x")
```
