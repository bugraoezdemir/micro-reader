# Axes

Images use five standard axes: `t c z y x`, with an optional `s` axis for colour samples.

By default, `open()` uses `axes="canonical"` and gives every image the five axes `tczyx`. With `axes=None` (also accepted as `"native"`), each image retains the axes present in its original layout and their original order.

RGB samples are represented as channels by default. To retain the stored `s` axis, pass `samples="keep"`.

This page explains how to choose the array layout you need.

## Native layout

With `open(..., axes=None)`, images retain their native axis order. Typical layouts include:

| Format | Typical native axes |
| --- | --- |
| CZI, ND2, LIF, OIR, OIB / OIF, VSI, Imaris, plates, companion sets | `tczyx` (RGB samples can be retained as `tczyxs`; see [RGB samples](#rgb-samples)) |
| TIFF (plain, ImageJ, OME-TIFF, ...) | The order stored in the file, for example `yx`, `zyx`, `tzcyx`, `yxs` |
| DeltaVision | The file's own order of time, waves and z, for example `ctzyx` |
| MRC | `zyx` or `yx` |
| PNG, JPEG, BMP | `yx` or `yxs`, as stored |
| Lambda scans (LIF, OIR, OIB / OIF, OME-TIFF) | The format's axes with the bands along an added `l`, for example `tlczyx` (see [Lambda scans](#lambda-scans)) |

When working with multiple formats, using a consistent layout is usually more convenient. You can do this with `axes="canonical"` or by reshaping individual images with `as_axes()`.

## `as_axes()`

```python
image.as_axes(axes=None, *, samples="keep", squeeze=False) -> Image
```

`as_axes()` returns a new image with the same pixels presented in a different axis layout. It does not read any pixel data when called. Reads from the new image are translated into reads in the original layout, so the pixel values are unchanged.

### Examples

=== "Canonical layout"

    ```python
    im = image.as_axes("canonical")
    # "tczyx", followed by "s" if the image has RGB samples
    ```

    ```text
    <micro_reader.Image 'stack' index={'series': 0} axes='tczyx' shape=(1, 1, 80, 256, 300) dtype=uint16 (any rectangle) of 'zyx'>
    ```

    The `of 'zyx'` at the end shows the image's native axes.

=== "Explicit axis order"

    ```python
    im = image.as_axes("zyx")   # only z, y and x; t and c must have size 1
    im = image.as_axes("ctzyx") # any order you need
    im = image.as_axes("yxz")   # z last, for tools that require it
    ```

=== "Keep the native order"

    ```python
    im = image.as_axes(None, samples="channels")
    # Same axis order, with RGB samples folded into c
    ```

### Rules

- **Missing axes:** Axes absent from the image are added with size 1.
- **Omitted axes:** An axis present in the image can be left out only if its size is 1. Otherwise, `as_axes()` raises an error identifying the axis that cannot be omitted:

    ```text
    ValueError: 'stack' has axes 'z' that 'cyx' cannot hold
    ```

- **Valid letters:** Axis names must be single letters from `tczyxs`, or an axis the native image has, such as the `l` of a [lambda scan](#lambda-scans), with no duplicates. Index names such as `view` cannot be used as axes. To turn a TIFF `sequence` index into an axis, use [`rename`](opening.md#rename-assign-a-meaning-to-an-unnamed-dimension) when opening the file.
- **Native image:** The original image remains available through `.source`:

    ```python
    im.source.axes  # 'zyx'
    ```

- **No stacking:** Calling `as_axes()` on an already reshaped image starts from its native source. Layout transformations therefore do not accumulate.

## `squeeze()`

```python
image.squeeze(axes=None) -> Image
```

`squeeze()` removes size-1 axes. With no argument, it removes every size-1 axis except `y` and `x`, so a 2-D plane remains a 2-D array.

```python
image.axes, image.shape   # ('tczyx', (1, 1, 80, 256, 300))

image.squeeze().axes      # 'zyx'
image.squeeze("t").axes   # 'czyx': remove only t
image.squeeze("c").shape  # (1, 80, 256, 300)
```

You can specify which axes to remove. An axis must have size 1; otherwise, `squeeze()` raises `ValueError`.

You can also squeeze while changing the layout:

```python
image.as_axes("canonical", squeeze=True)
```

## RGB samples

Colour images store three samples per pixel for RGB or four for RGBA. By default, `open()` converts these samples into channels. An RGB image therefore has three channels and no `s` axis:

```python
f = micro_reader.open("photo.jpg")
image = f.images[0]

image.axes, image.shape
# ('tczyx', (1, 3, 1, 1080, 1920))
# With axes=None: ('cyx', (3, 1080, 1920))

[ch.color for ch in image.metadata.channels]
# ['#FF0000', '#00FF00', '#0000FF']
```

The channel metadata contains one entry per sample: red, green and blue, and white for alpha.

Channel names are expanded to match the samples. For example, a channel named `"Stain"` becomes `"Stain R"`, `"Stain G"` and `"Stain B"`. An unnamed channel remains unnamed. Wavelengths are preserved. Files that already represent each sample as a separate channel retain their existing channels.

Images without colour samples are unchanged.

The original image layout, including its `s` axis when present, remains available through `image.source`:

```python
image.source.axes  # 'yxs'
```

To retain the `s` axis in the image returned by `open()`, pass `samples="keep"`:

```python
f = micro_reader.open("photo.jpg", samples="keep")

f.images[0].axes, f.images[0].shape
# ('tczyxs', (1, 1, 1, 1080, 1920, 3))
# With axes=None: ('yxs', (1080, 1920, 3))
```

If the image already has multiple channels, samples are folded into those channels. Output channel `k` corresponds to sample `k % S` of input channel `k // S`, where `S` is the number of samples per pixel.

For example, an image with two RGB channels becomes six channels, ordered as:

`R0 G0 B0 R1 G1 B1`

### RGB behaviour in `as_axes()`

`as_axes()` has a different default from `open()`. It starts from the stored image (`.source`) and keeps the `s` axis unless you explicitly request `samples="channels"`.

```python
image.as_axes("canonical").axes
# 'tczyxs'

image.as_axes("canonical", samples="channels").axes
# 'tczyx'
```

When samples are folded into channels, the resulting channel metadata contains one entry per output channel. Therefore:

```python
len(im.metadata.channels) == im.shape[1]
```

## Setting the layout for a whole file

`open()` accepts the same three layout options and applies them to every image:

```python
f = micro_reader.open(path, axes="canonical")

all(im.axes == "tczyx" for im in f.images)
# True
```

This is useful when converting data into a common format, such as 5-D OME-Zarr.

Each image's original layout remains accessible through `im.source`. Plate navigation through `f.plates[...]` also returns the reshaped images.

## Turning an unnamed dimension into an axis

Some TIFF files contain pages without identifying what those pages represent. micro-reader exposes these pages through the `sequence` index rather than guessing that they represent z-planes or time points.

`as_axes()` cannot turn an index into an axis. Use `rename` when opening the file to specify what the dimension represents:

```python
f = micro_reader.open("stack.tif", rename={"sequence": "z"})
```

See [Opening files](opening.md#rename-assign-a-meaning-to-an-unnamed-dimension).

## Lambda scans

A spectral (lambda) scan is one image in every format. Its bands lie along `c`, with the bands of one detector next to each other in wavelength order. Each channel has its `emission_wavelength`, and its name ends with the wavelength:

```python
f = micro_reader.open("Convalaria_LambdaScan.lif")
image = f.images[0]

image.axes, image.shape
# ('tczyx', (1, 29, 1, 512, 512))

[ch.name for ch in image.metadata.channels][:3]
# ['HyD S 1 420 nm', 'HyD S 1 430 nm', 'HyD S 1 440 nm']

image.metadata.channels[0].emission_wavelength
# Quantity(value=420.0, unit='nanometer')
```

With `axes=None`, or with axes that name `l` such as `axes="tlczyx"`, the bands are a real `l` axis instead, and `metadata.index_values["emission"]` holds the wavelength of each band, in order:

```python
f = micro_reader.open("Convalaria_LambdaScan.lif", axes=None)
f.images[0].axes                                   # 'tlczyx'
f.images[0].metadata.index_values["emission"][:3]
# (Quantity(value=420.0, unit='nanometer'), Quantity(value=430.0, unit='nanometer'), ...)
```

A Zeiss LSM stores its bands as channels, so it has no `l` axis. Its channels carry their wavelengths. Excitation scans remain an `excitation` index.