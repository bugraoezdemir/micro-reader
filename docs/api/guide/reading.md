# Reading pixels

An `Image` reads like a read-only NumPy array that never loads more than you ask for. You can use NumPy indexing or the keyword method `read()`.

## Two ways to ask for a region

=== "NumPy indexing"

    ```python
    image.axes                    # 'tczyx'
    image[0, 1, 10, 0:2048, 1000:3000]
    ```

    The key follows the image's `axes`, in order.

=== "Keyword form"

    ```python
    image.read(t=0, c=1, z=10, y=slice(0, 2048), x=slice(1000, 3000))
    ```

    You name the axes, in any order. Axes you leave out are read whole.

Both return the same array. The keyword form is easier to read and does not depend on the axis order, so it keeps working when a file has `zyx` instead of `tczyx`. Prefer it in code that handles many formats.

```python
image.read(z=3)            # the whole of plane z=3, every t and c
image.read()               # the whole image (careful with big ones)
image.read(c=0, x=slice(0, 100))
```

Asking for an axis the image does not have raises `KeyError`, with a hint:

```text
KeyError: "unknown axes ['c']; this image has 'zyx' (open(path, axes='canonical')
or image.as_axes('canonical') adds the missing standard axes with size 1)"
```

## What the indexing supports

| Key | Example | Result |
| --- | --- | --- |
| integer | `image[0]`, `z=5` | that position; the axis is dropped, as in NumPy |
| negative integer | `image[-1]` | counted from the end |
| slice | `image[:, 2:5]`, `y=slice(0, 512)` | that span; the axis is kept |
| slice with a step | `image[..., ::4]`, `x=slice(0, None, 2)` | every n-th pixel |
| Ellipsis | `image[0, ..., 100:200]` | all the axes in between |
| fewer keys than axes | `image[0]` | the remaining axes are read whole |

These raise `IndexError`: `None` / `np.newaxis`, negative steps, lists or arrays of indices (fancy indexing), boolean masks, and more keys than axes. Out-of-range integers raise `IndexError` as in NumPy. Out-of-range slices are clipped as in NumPy.

```python
image.shape                    # (80, 256, 300)
image[5].shape                 # (256, 300)
image[..., ::4].shape          # (80, 256, 75)
image[-1, 10:20, 0:30:2].shape # (10, 15)
image[0:0].shape               # (0, 256, 300): nothing is read
```

!!! warning "A step saves memory, not much reading"
    For `image[..., ::4]`, micro-reader reads the span a piece of at most 64 MB at a time and keeps every fourth pixel of each, so the memory it takes is the result and one piece (before 0.4.0, the whole span). Within a piece it still reads every row and column: a step makes the result smaller, not the reading much shorter, except where one step skips more than a piece (`image[::16]` of large planes reads only the planes kept). To get a smaller image cheaply, read a lower [resolution level](pyramids.md) if the file has one.

## What you get back

- A new `numpy.ndarray`. It is never a view into the file or into micro-reader's cache.
- Always in native byte order, even when the file is big-endian, so `image.dtype` is native too.
- The image's dtype, unchanged: no scaling and no conversion.
- For paletted PNG and BMP images, the palette indices. The colours are in `image.palette` (see [Formats](../formats.md#jpeg-png-and-bmp)).
- Zeros where the file has no data: background between CZI tiles, a plane file missing from a plate.

## How much was actually read

Every image counts its reads in `image.stats`:

```python
image.stats
# ReadStats(reads=4, pixels_returned=6413100, pixels_read=6415530)

image.stats.over_read     # pixels_read / pixels_returned
# 1.0004

image.stats.reset()
```

`pixels_returned` counts the regions you asked for, before any step. `pixels_read` counts what had to be read or decoded to produce them. An over-read of 1.0 means micro-reader read exactly your region.

The over-read goes above 1 when the file stores pixels in blocks that cannot be split. Then `image.read_unit` tells you the block size:

```python
image.read_unit
# None: any rectangle can be read exactly (uncompressed data, streamable codecs)
# (1, 1, 64, 64, 64): the smallest block read at once, per axis
```

An example with Imaris, which stores 64 × 64 × 64 chunks:

```python
level = f.images[0].resolutions[2]     # shape (1, 10, 91, 128, 128)
a = level.read(c=0, z=slice(0, 4))     # 4 planes asked
level.stats.over_read                  # 16.0: a whole 64-plane chunk was decoded
```

## Reading efficiently

**Align your reads with `read_unit`.**
When `read_unit` is not `None`, reads that cover whole blocks waste nothing. Reads that cut a block decode the whole block. Decoded blocks are kept in a [cache](performance.md#the-decoded-block-cache), so the next read of a neighbouring region does not decode them again, but they cost the time to decode them once.

```python
import itertools

def blocks(image, plane_tile=1024):
    """Yield (key, array) over the whole image, one read per storage block.

    Without a read unit, read one plane at a time in plane_tile x plane_tile pieces."""
    unit = image.read_unit or tuple(plane_tile if a in "yx" else 1 for a in image.axes)
    starts = [range(0, n, u) for n, u in zip(image.shape, unit)]
    for corner in itertools.product(*starts):
        key = tuple(slice(s, min(s + u, n)) for s, u, n in zip(corner, unit, image.shape))
        yield key, image[key]
```

**Read top to bottom.**
Some large compressed blocks (deflate, LZW, zstd and others) are decoded as a stream. Reading rows in order continues where the last read stopped. Jumping back costs a restart for LZW and zstd. Deflate resumes from a checkpoint. See [Memory and threads](performance.md#streamed-decoding).

**Prefer wide regions for uncompressed data.**
For uncompressed data only the requested rows are read. When you ask for part of each row, micro-reader may read the rest of the row too if that is faster than reading the rows one by one. Full-width regions avoid that choice.

**Prefer one large read to many tiny ones.**
Each read has a fixed overhead, and one read spreads its decoding over all [decode threads](performance.md#decode-threads). A few large reads are usually faster than many small ones.

## Reading from several threads

You can read one image, or several images of one file, from several threads at once. Each thread gets its own file handle, and decoding uses one shared pool:

```python
from concurrent.futures import ThreadPoolExecutor

with micro_reader.open("cells.nd2") as f:
    image = f.images[0]
    with ThreadPoolExecutor(8) as ex:
        planes = list(ex.map(lambda t: image.read(t=t, c=0), range(image.shape[0])))
```

If you run many reads in parallel yourself, lower the [decode threads](performance.md#decode-threads) so the two pools do not oversubscribe the CPU.

## Reading from several processes

`File` and `Image` objects cannot be pickled. Pass the **path** and the **index** to the worker, and open the file there:

```python
from concurrent.futures import ProcessPoolExecutor
import micro_reader

def plane_max(path, index, z):
    with micro_reader.open(path) as f:
        image = f.select(**index)[0]
        return int(image.read(z=z).max())

with micro_reader.open(path) as f:
    jobs = [(path, im.index, z) for im in f.images for z in range(im.shape[im.axes.index("z")])]

with ProcessPoolExecutor() as ex:
    maxima = list(ex.map(plane_max, *zip(*jobs)))
```

Opening a file is cheap for most formats, but it can be slow for plates with many thousands of plane files. In that case, open the file once per worker, not once per task. The [examples](../examples.md#parallel-processing-with-worker-processes) show a worker that keeps its file open.

The memory limit, the cache limit and the decode threads apply per process. Set them in each worker.
