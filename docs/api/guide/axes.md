# Axes

Every image's array uses the axes `t c z y x`, plus `s` for RGB samples. `open()` gives every image all five, `tczyx`, by default (`axes="canonical"`); with `axes=None` (also spelled `"native"`) each image keeps only the axes its file has, in the file's own order for formats that have one. This page shows how to get the layout you want.

## The native layout

What `open(..., axes=None)` gives:

| Format | Typical native axes |
| --- | --- |
| CZI, ND2, LIF, OIR, OIB / OIF, VSI, Imaris, plates, companion sets | `tczyx` (RGB: `tczyxs` as stored, see [RGB samples](#rgb-samples)) |
| TIFF (plain, ImageJ, OME-TIFF, ...) | the file's own order, for example `yx`, `zyx`, `tzcyx`, `yxs` |
| DeltaVision | the file's own order of time, waves and z, for example `ctzyx` |
| MRC | `zyx` or `yx` |
| PNG, JPEG, BMP | `yx` or `yxs` as stored |

When code handles many formats, it is easier to ask for one layout.

## `as_axes()`

```python
image.as_axes(axes=None, *, samples="keep", squeeze=False) -> Image
```

`as_axes` returns a new image that shows the same pixels in another layout. Nothing is read when you call it. Reads on the new image are turned into reads in the native layout, so they stay exact.

=== "canonical"

    ```python
    im = image.as_axes("canonical")      # "tczyx", then "s" if the image is RGB
    ```

    ```text
    <micro_reader.Image 'series0' index={'series': 0} axes='tczyx' shape=(1, 1, 80, 256, 300) dtype=uint16 (any rectangle) of 'zyx'>
    ```

    The `of 'zyx'` at the end of the printout shows the native axes.

=== "explicit letters"

    ```python
    im = image.as_axes("zyx")    # only z, y, x: t and c must have size 1
    im = image.as_axes("ctzyx")  # any order you need
    im = image.as_axes("yxz")    # z last, for tools that want it
    ```

=== "keep the order"

    ```python
    im = image.as_axes(None, samples="channels")   # same order, RGB folded into c
    ```

Rules:

- Axes that the image lacks are added with size 1.
- An axis that the image has but you leave out must have size 1. Otherwise you get an error that names it:

    ```text
    ValueError: 'series0' has axes 'z' that 'cyx' cannot hold
    ```

- Each letter must be one of `tczyxs` and appear only once. Index names such as `view` cannot be axes. Use [`rename`](opening.md#rename-say-what-an-unnamed-dimension-is) for a TIFF `sequence`.
- The native image stays available as `.source`:

    ```python
    im.source.axes    # 'zyx'
    ```

- Calling `as_axes` on the result starts again from the native image. Layouts do not stack.

## `squeeze()`

```python
image.squeeze(axes=None) -> Image
```

`squeeze()` drops every size-1 axis except `y` and `x`. A plane stays a plane:

```python
image.axes, image.shape              # ('tczyx', (1, 1, 80, 256, 300))
image.squeeze().axes                 # 'zyx'
image.squeeze("t").axes              # 'czyx': only t
image.squeeze("c").shape             # (1, 80, 256, 300)
```

Naming an axis that is not size 1 raises `ValueError`. You can also squeeze while changing the layout: `image.as_axes("canonical", squeeze=True)`.

## RGB samples

Colour images store three (RGB) or four (RGBA) samples per pixel. `open()` turns them into channels by default, so an RGB image has `c` of 3 and no `s` axis:

```python
f = micro_reader.open("photo.jpg")
image = f.images[0]
image.axes, image.shape           # ('tczyx', (1, 3, 1, 1080, 1920)); axes=None: ('cyx', (3, 1080, 1920))
[ch.color for ch in image.metadata.channels]
# ['#FF0000', '#00FF00', '#0000FF']
```

The metadata lists one channel per sample: red, green and blue (and white for alpha). A channel named "Stain" becomes "Stain R", "Stain G" and "Stain B"; an unnamed channel stays unnamed. Wavelengths are kept. A file that already lists one channel per sample keeps its own channels.

Images without samples are not changed. The image as the file stores it, with its `s` axis, is `image.source`:

```python
image.source.axes                 # 'yxs'
```

To keep the `s` axis, open with `samples="keep"`:

```python
f = micro_reader.open("photo.jpg", samples="keep")
f.images[0].axes, f.images[0].shape    # ('tczyxs', (1, 1, 1, 1080, 1920, 3)); axes=None: ('yxs', (1080, 1920, 3))
```

When the image already has channels, the samples are folded into them: channel `k` of the result is sample `k % S` of channel `k // S`. A 2-channel RGB image gives 6 channels, in the order R0 G0 B0 R1 G1 B1.

`as_axes()` is different: it keeps `s` unless you ask for `samples="channels"`, and it starts from the stored image (`.source`), not from the image `open()` gave you:

```python
image.as_axes("canonical").axes                       # 'tczyxs'
image.as_axes("canonical", samples="channels").axes   # 'tczyx'
```

The channel metadata of a folded image has one entry per output channel, so `len(im.metadata.channels)` equals `im.shape[1]`.

## Setting the layout for a whole file

`open()` takes the same three options and applies them to every image:

```python
f = micro_reader.open(path, axes="canonical")
all(im.axes == "tczyx" for im in f.images)    # True
```

This is the usual setup for a converter that writes 5-D OME-Zarr. Each image's native form is `im.source`. Plate navigation (`f.plates[...]`) reaches the reshaped images.

## Turning an unnamed dimension into an axis

When a TIFF does not say what its pages are, they become the `sequence` index rather than an axis. `as_axes` cannot change that, because an index is not an axis. Tell `open()` instead:

```python
f = micro_reader.open("stack.tif", rename={"sequence": "z"})
```

See [Opening files](opening.md#rename-say-what-an-unnamed-dimension-is).
