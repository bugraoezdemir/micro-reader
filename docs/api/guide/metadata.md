# Metadata

`image.metadata` provides the metadata that downstream tools need to place, scale and display an image. It exposes a small, format-independent model rather than reproducing the vendor's original XML.

```python
meta = image.metadata

meta.pixel_sizes       # {axis: Quantity}: pixel sizes along x, y, z and time step along t
meta.position          # {axis: Quantity}: position of the first pixel's centre
meta.coordinate_system # str or None: coordinate system of the position
meta.channels          # (Channel, ...): one entry per channel of the native c axis
meta.index_values      # {index: Quantity}: physical values of indices, such as FLIM time bins
```

Metadata is read the first time you access `image.metadata` and cached for subsequent access.

## Example

Here is the metadata from an image in a Zeiss CZI plate:

```python
meta = f.images[0].metadata

meta.pixel_sizes
# {
#     'x': Quantity(value=0.4, unit='micrometer'),
#     'y': Quantity(value=0.4, unit='micrometer'),
#     'z': Quantity(value=1.0, unit='micrometer'),
#     't': Quantity(value=99.25, unit='second')
# }

meta.position
# {
#     'x': Quantity(value=12911.4, unit='micrometer'),
#     'y': Quantity(value=16811.8, unit='micrometer')
# }

meta.coordinate_system
# 'stage'

meta.channels[0]
# Channel(
#     name='DAPI',
#     color='#00A1FF',
#     emission_wavelength=Quantity(value=465.0, unit='nanometer'),
#     excitation_wavelength=Quantity(value=353.0, unit='nanometer')
# )
```

## General rules

### Missing values are not invented

If a value is not present in the file, it is left absent from the relevant dictionary or set to `None` in a `Channel`.

micro-reader does not assign default pixel sizes of `1.0`, invent channel colours or set missing positions to zero. Check for missing values before using them:

```python
size_z = meta.pixel_sizes.get("z")  # None if the file does not provide it

if size_z is None:
    ...  # Choose your own default, if appropriate
```

### Units follow OME-Zarr conventions

Physical quantities use UDUNITS-2 unit names accepted by OME-Zarr, such as `micrometer`, `nanometer`, `millisecond` and `second`.

Values retain the units provided by the file. For example, an OIF z-step may be reported as `510.0 nanometer`. Lengths expressed in units larger than a millimetre are converted to micrometres.

Use `Quantity.to()` to convert between units:

```python
meta.pixel_sizes["z"]
# Quantity(value=510.0, unit='nanometer')

meta.pixel_sizes["z"].to("micrometer")
# Quantity(value=0.51, unit='micrometer')
```

### Each resolution level has its own pixel sizes

Pixel sizes can differ between pyramid levels. See [Resolution levels](pyramids.md#pixel-sizes-per-level).

### Metadata follows the image layout

An image created with `as_axes()` exposes the metadata of its source image. When RGB samples are converted into channels, `channels` contains one entry per output channel, with names expanded using the R, G and B suffixes where applicable.

See [RGB samples](axes.md#rgb-samples).

## Pixel sizes

`pixel_sizes` can contain the keys `x`, `y`, `z` and `t`. Spatial axes describe pixel dimensions; `t` describes the time interval between frames.

For example, this helper returns spatial pixel sizes in micrometres, using `None` when a size is unknown:

```python
def scale_um(meta, axes="zyx"):
    """Return pixel sizes in micrometres for the requested axes."""
    out = []

    for a in axes:
        q = meta.pixel_sizes.get(a)
        out.append(q.to("micrometer").value if q is not None else None)

    return out


scale_um(image.metadata)
# [0.5, 0.314..., 0.314...] for a DeltaVision zyx stack
```

## Position and coordinate systems

`position` describes the position of the **centre of the first pixel**. The accompanying `coordinate_system` identifies the space in which that position is defined, such as the microscope's `stage`, a `slide` or a plate `well`.

Positions can only be compared directly when they belong to the same coordinate system. See [Coordinate systems](coordinates.md).

micro-reader converts each format's position convention into this common representation. Source formats may describe the image corner, the centre of the field or use inverted stage axes.

If a position cannot be converted reliably, `position` is empty and `coordinate_system` is `None`. This currently applies to:

- OIR files on their own (the tiles of a FluoView tiling set, `.omp2info`, do have positions)
- OIB / OIF
- VSI
- DeltaVision
- MetaXpress
- IN Cell
- Cellomics
- Legacy ND2 files without a camera matrix
- MetaMorph and Micro-Manager image-per-file acquisitions (see below)

MetaMorph and Micro-Manager's separate files record where the stage was, but not which point of the field that is. Their values are kept, as recorded, in `metadata.stage_position`. They have no coordinate system and should not be used to place images against each other.

For plate acquisitions, positions are relative to the centre of the well.

The [format table](../formats.md#metadata-by-format) lists the position information available from each format.

### Positions in tiled acquisitions

For tiled acquisitions, positions describe where each tile is located:

```python
for im in f.select(illumination=0):
    p = im.metadata.position
    print(
        im.index["tile"],
        round(p["x"].value, 1),
        round(p["y"].value, 1)
    )

# 0 -3946.8 13665.6
# 1 -677.3  13665.6
# 2 -3946.8 15846.9
# 3 -677.3  15846.9
```

To convert a physical position difference into pixels, first convert the position and pixel size to the same unit, then divide the difference by the pixel size.

## Channels

`channels` is a tuple of `Channel` objects, one for each channel along the native `c` axis. Entries follow the channel order:

```python
Channel(
    name=None,
    color=None,
    emission_wavelength=None,
    excitation_wavelength=None
)
```

| Field | Type | Example |
| --- | --- | --- |
| `name` | `str` or `None` | `'DAPI'` |
| `color` | `'#RRGGBB'` or `None` | `'#00A1FF'` |
| `emission_wavelength` | `Quantity` or `None` | `Quantity(465.0, 'nanometer')` |
| `excitation_wavelength` | `Quantity` or `None` | `Quantity(353.0, 'nanometer')` |

An image without a `c` axis still has one channel entry. If the file provides no channel metadata, micro-reader returns `Channel()` entries with all fields set to `None`.

You can supply your own fallbacks when using channel metadata:

```python
names = [
    ch.name or f"channel {i}"
    for i, ch in enumerate(meta.channels)
]

colors = [
    ch.color or "#FFFFFF"
    for ch in meta.channels
]  # Your fallback, not a value assigned by micro-reader
```

## Index values

Some image indices have an associated physical value, such as the wavelength of an excitation band or the time of a FLIM bin. `index_values` provides that value for the current image, for example `{"lifetime": Quantity(2.5, "nanosecond")}`.

A lambda scan opened with `axes=None` has its bands along an `l` axis. Its `index_values["emission"]` is then a tuple with the wavelength of each band, in the order of the axis:

```python
f = micro_reader.open("DAPI-mCherry_3T_4Z_5Lambda.oir", axes=None)
image = f.images[0]

image.axes                                         # 'tlczyx'
[q.value for q in image.metadata.index_values["emission"]]
# [420.0, 470.0, 520.0, 570.0, 620.0]
```

With the default layout, the bands are channels instead, and each channel's `emission_wavelength` gives its band (see [Lambda scans](axes.md#lambda-scans)).

Indices without a physical value, such as `tile` or `view`, are not included in `index_values`.

## Quantity

A `Quantity` represents a numeric value and its unit:

```python
Quantity(value: float, unit: str)
```

It is a named tuple, so you can unpack its fields:

```python
value, unit = meta.pixel_sizes["x"]
```

Use `Quantity.to(unit)` to convert between units of the same kind, either length or time. It accepts OME-Zarr's unit names and the short spellings that files use, such as `"um"`, `"nm"`, `"ms"` and `"s"`:

```python
micro_reader.Quantity(0.5, "micrometer").to("nanometer")
# Quantity(value=499.99999999999994, unit='nanometer')

micro_reader.Quantity(60000, "ms").to("s")
# Quantity(value=60.0, unit='second')
```

Converting a length to a time, or to a unit name it does not know, raises `ValueError`:

```text
ValueError: Quantity.to: 'nanometer' cannot be converted to 'second': a length and a time
```

The supported unit names follow the OME-Zarr space and time units.

- **Space:** From `yoctometer` to `yottameter`, including `angstrom`, `inch`, `foot`, `yard`, `mile` and `parsec`.
- **Time:** From `yoctosecond` to `yottasecond`, including `minute`, `hour` and `day`.

## Using metadata when writing OME-Zarr

The following example sketches how to build the `multiscales` scale and axis metadata, as well as the `omero` channel metadata, for one image in canonical `tczyx` layout.

```python
def ngff_scale(image):
    """Return scale coordinate transformations for a canonical tczyx image."""
    meta = image.metadata
    scale = []

    for a in "tczyx":
        q = meta.pixel_sizes.get(a)
        scale.append(q.value if q is not None and a != "c" else 1.0)

    return [{"type": "scale", "scale": scale}]


def ngff_axes(image):
    meta = image.metadata
    out = []

    for a in "tczyx":
        kind = {"t": "time", "c": "channel"}.get(a, "space")
        ax = {"name": a, "type": kind}

        q = meta.pixel_sizes.get(a)
        if q is not None and a != "c":
            ax["unit"] = q.unit

        out.append(ax)

    return out


def omero_channels(image):
    return [
        {
            "label": ch.name or f"c{i}",
            "color": (ch.color or "#FFFFFF").lstrip("#"),
            "active": True,
        }
        for i, ch in enumerate(image.metadata.channels)
    ]
```

The `1.0` values in `ngff_scale()` are placeholders required by this example's OME-Zarr output. They are assigned by your code, not by micro-reader. If a pixel or time size is unknown, consider recording that fact separately rather than presenting the placeholder as a measured value.

## Plates

Plate-level information, such as the plate name, number of rows and columns, wells and acquisitions, is available through `f.plates`, not `image.metadata`.

See [Plates](plates.md).