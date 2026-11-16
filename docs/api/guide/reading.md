# Reading pixels

An `Image` can be read like a read-only NumPy array. Each read returns the requested region as a NumPy array, and its memory requirements depend on the size of that region, not the size of the file.

You can use NumPy-style indexing or the keyword-based `read()` method.

## Two ways to request a region

=== "NumPy indexing"

    ```python
    image.axes                       # 'tczyx'
    image[0, 1, 10, 0:2048, 1000:3000]
    ```

    The key follows the image's axes, in order.

=== "Keyword form"

    ```python
    image.read(t=0, c=1, z=10, y=slice(0, 2048), x=slice(1000, 3000))
    ```

    Specify axes by name, in any order. Axes you omit are read in full.

Both forms return the same array. The keyword form is easier to read and does not depend on axis order. It therefore continues to work when an image is opened with `axes=None` and has `zyx` instead of `tczyx`. Prefer it in code that handles multiple formats.

```python
image.read(z=3)                       # the whole z=3 plane, for every t and c
image.read()                          # the whole image (beware of large images)
image.read(c=0, x=slice(0, 100))
```

Requesting an axis that the image does not have, for example `c` on a `zyx` image opened with `axes=None`, raises a `KeyError` with a hint:

```text
KeyError: "unknown axes ['c']; this image has 'zyx' (open(path, axes='canonical')
or image.as_axes('canonical') adds the missing standard axes with size 1)"
```

## Supported indexing

| Key | Example | Result |
| --- | --- | --- |
| Integer | `image[0]`, `z=5` | Selects that position and drops the axis, as in NumPy |
| Negative integer | `image[-1]` | Selects a position counted from the end |
| Slice | `image[:, 2:5]`, `y=slice(0, 512)` | Selects a range and keeps the axis |
| Slice with a step | `image[..., ::4]`, `x=slice(0, None, 2)` | Selects every nth pixel |
| Ellipsis | `image[0, ..., 100:200]` | Selects all axes in between |
| Fewer keys than axes | `image[0]` | Reads the remaining axes in full |

The following keys raise `IndexError`: `None` / `np.newaxis`, negative steps, lists or arrays of indices (fancy indexing), boolean masks, and more keys than there are axes.

Out-of-range integer indices also raise `IndexError`, as in NumPy. Out-of-range slice bounds are clipped, also as in NumPy.

For a `zyx` image, such as a TIFF z-stack opened with `axes=None`:

```python
image.shape                       # (80, 256, 300)
image[5].shape                    # (256, 300)
image[..., ::4].shape             # (80, 256, 75)
image[-1, 10:20, 0:30:2].shape    # (10, 15)
image[0:0].shape                  # (0, 256, 300): nothing is read
```

!!! warning "A step reduces memory use, but not necessarily read time"
    For `image[..., ::4]`, micro-reader reads the requested span in pieces of at most 64 MB and keeps every fourth pixel from each piece. Memory use is limited to the result plus one piece.

    Within each piece, however, all rows and columns are still read. A step therefore makes the result smaller but does not necessarily reduce the amount of data read. The exception is when a step skips entire pieces. For example, `image[::16]` on large planes reads only the planes selected by the step.

    To obtain a smaller image with less reading, use a lower [resolution level](pyramids.md) when the file provides one.

## What you get back

- **A new array:** The result is a `numpy.ndarray`, never a view into the file or micro-reader's cache.
- **Native byte order:** The result uses native byte order, even when the file stores big-endian data. `image.dtype` is native as well.
- **The image's dtype:** The dtype is preserved, with no scaling or conversion.
- **Palette indices:** For paletted PNG and BMP images, the array contains palette indices. The corresponding colours are available through `image.palette` (see [Formats](../formats.md#jpeg-png-and-bmp)).
- **Zeros where data is absent:** For example, between CZI tiles or where a plane file is missing from a plate.

## How much data was actually read

Each image tracks its reads in `image.stats`:

```python
image.stats
# ReadStats(reads=4, pixels_returned=6413100, pixels_read=6415530)

image.stats.over_read              # pixels_read / pixels_returned
# 1.0004

image.stats.reset()
```

`pixels_returned` counts the requested regions before applying any step. `pixels_read` counts the pixels that had to be read or decoded to produce them. An over-read of 1.0 means micro-reader read exactly the requested region.

Over-read increases when the file stores pixels in blocks that cannot be split. In that case, `image.read_unit` reports the smallest block read at once, expressed as a size along each axis:

```python
image.read_unit
# None: any rectangle can be read exactly (uncompressed data, streamable codecs)
# (1, 1, 64, 64, 64): the smallest block read at once, per axis
```

Imaris uses 64 × 64 × 64 chunks, which illustrates the effect:

```python
level = f.images[0].resolutions[2]  # shape (1, 10, 91, 128, 128)
a = level.read(c=0, z=slice(0, 4))  # request 4 planes
level.stats.over_read               # 16.0: a whole 64-plane chunk was decoded
```

## Reading efficiently

**Align reads with `read_unit`.** When `read_unit` is not `None`, reads that cover whole blocks avoid unnecessary decoding. A read that intersects only part of a block still decodes the whole block.

Decoded blocks are kept in a [cache](performance.md#the-decoded-block-cache), so a later read of a neighbouring region does not decode them again. Each block still has to be decoded at least once.

```python
import itertools

def blocks(image, plane_tile=1024):
    """Yield (key, array) for the whole image, one read per storage block.

    If there is no read unit, read one plane at a time in
    plane_tile x plane_tile pieces.
    """
    unit = image.read_unit or tuple(
        plane_tile if axis in "yx" else 1 for axis in image.axes
    )
    starts = [range(0, n, u) for n, u in zip(image.shape, unit)]

    for corner in itertools.product(*starts):
        key = tuple(
            slice(s, min(s + u, n))
            for s, u, n in zip(corner, unit, image.shape)
        )
        yield key, image[key]
```

**Read from top to bottom.** Some large compressed blocks, including Deflate, LZW, and zstd, can be decoded in parts as a stream. Reading rows in order continues from where the previous read stopped. Jumping back requires a restart for LZW and zstd, while Deflate can resume from a checkpoint. See [Memory and threads](performance.md#streamed-decoding).

**Prefer wide regions for uncompressed data.** With uncompressed data, only the requested rows need to be read. However, when you request part of each row, micro-reader may read the rest of each row too if that is faster than reading the rows individually. Full-width regions avoid this trade-off.

**Prefer one large read to many small ones.** Every read has some fixed overhead, and a single read can distribute its decoding work across all [decode threads](performance.md#decode-threads). A few large reads are usually faster than many small ones.

## Reading from several threads

You can read one image, or multiple images from the same file, in several threads at once. Each thread gets its own file handle, while decoding uses a shared thread pool:

```python
from concurrent.futures import ThreadPoolExecutor

with micro_reader.open("cells.nd2") as f:
    image = f.images[0]
    with ThreadPoolExecutor(8) as ex:
        planes = list(ex.map(
            lambda t: image.read(t=t, c=0),
            range(image.shape[0])
        ))
```

If your application already runs many reads in parallel, reduce the [decode thread](performance.md#decode-threads) count to avoid oversubscribing the CPU.

## Reading from several processes

`File` and `Image` objects cannot be pickled. Pass the **path** and **index** to each worker, and open the file in the worker process:

```python
from concurrent.futures import ProcessPoolExecutor
import micro_reader

def plane_max(path, index, z):
    with micro_reader.open(path) as f:
        image = f.select(**index)[0]
        return int(image.read(z=z).max())

with micro_reader.open(path) as f:
    jobs = [
        (path, im.index, z)
        for im in f.images
        for z in range(im.shape[im.axes.index("z")])
    ]

with ProcessPoolExecutor() as ex:
    maxima = list(ex.map(plane_max, *zip(*jobs)))
```

Opening a file is inexpensive for most formats. It can be slow for plates containing thousands of plane files, however. For those datasets, open the file once per worker rather than once per task. The [examples](../examples.md#parallel-processing-with-worker-processes) show how to keep a file open in a worker.

The memory limit, cache limit, and decode thread count apply separately to each process. Configure them in every worker.