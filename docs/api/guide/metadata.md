# Metadata

`image.metadata` holds the metadata that downstream tools need to place and display an image. It is not a dump of the vendor's XML. It is a small model that is the same for every format:

```python
meta = image.metadata

meta.pixel_sizes     # {axis: Quantity}        size of a pixel along x, y, z; time step along t
meta.position        # {axis: Quantity}        position of the first pixel's centre
meta.coordinate_system  # str or None          the coordinate system the position is in
meta.channels        # (Channel, ...)          one per channel of the native c axis
meta.index_values    # {index: Quantity}       physical value of this image's index, e.g. its emission band
```

The metadata is read the first time you access it, and kept.

## A real example

From a Zeiss CZI plate:

```python
meta = f.images[0].metadata

meta.pixel_sizes
# {'x': Quantity(value=0.4, unit='micrometer'),
#  'y': Quantity(value=0.4, unit='micrometer'),
#  'z': Quantity(value=1.0, unit='micrometer'),
#  't': Quantity(value=99.25, unit='second')}

meta.position
# {'x': Quantity(value=12911.4, unit='micrometer'),
#  'y': Quantity(value=16811.8, unit='micrometer')}

meta.coordinate_system
# 'stage'

meta.channels[0]
# Channel(name='DAPI', color='#00A1FF',
#         emission_wavelength=Quantity(value=465.0, unit='nanometer'),
#         excitation_wavelength=Quantity(value=353.0, unit='nanometer'))
```

## The rules

**Nothing is invented.** A value the file does not give is absent from the dict, or `None` in a `Channel`. Pixel sizes are never set to `1.0`, colours are never assigned and positions are never set to zero. Always check before you use a value:

```python
size_z = meta.pixel_sizes.get("z")         # None when the file does not say
if size_z is None:
    ...                                    # your own default, your own decision
```

**Units are OME-Zarr names.** Units use the UDUNITS-2 names that OME-Zarr accepts: `micrometer`, `nanometer`, `millisecond`, `second`, and so on. Values keep the file's own unit (an OIF z step may be `510.0 nanometer`), except lengths in units larger than a millimetre, which are converted to micrometres. Convert with `Quantity.to()` when you need one unit:

```python
meta.pixel_sizes["z"]                       # Quantity(value=510.0, unit='nanometer')
meta.pixel_sizes["z"].to("micrometer")      # Quantity(value=0.51, unit='micrometer')
```

**Each resolution level has its own pixel sizes.** See [Resolution levels](pyramids.md#pixel-sizes-per-level).

**Metadata follows the layout.** An image made with `as_axes()` shows its source's metadata. When RGB samples are turned into channels, `channels` has one entry per output channel, "<name> R", "G" and "B" (see [Axes](axes.md#rgb-samples)).

## Pixel sizes

`pixel_sizes` can hold the keys `x`, `y`, `z` and `t`. For `t` it is the time between frames.

```python
def scale_um(meta, axes="zyx"):
    """Pixel sizes in micrometres for the given spatial axes; None where unknown."""
    out = []
    for a in axes:
        q = meta.pixel_sizes.get(a)
        out.append(q.to("micrometer").value if q is not None else None)
    return out

scale_um(image.metadata)     # [0.5, 0.314..., 0.314...] for a DeltaVision zyx stack
```

## Position

`position` is the position of the **centre of the first pixel**. `coordinate_system` says which space it is in: the microscope's `stage`, a `slide`, or a `well` for plates. Two positions can be compared only when they have the same coordinate system. See [Coordinate systems](coordinates.md).

micro-reader converts each format's own convention (corner of the image, centre of the field, inverted stage axes) to this one. When a file's position cannot be turned into this reliably, `position` is empty and `coordinate_system` is `None`. Today that includes OIR, OIB / OIF, VSI, DeltaVision, MetaXpress, IN Cell, Cellomics and legacy ND2 files without a camera matrix. On plates, the position is relative to the well centre. The [format table](../formats.md#metadata-per-format) lists what each format gives.

For tiled acquisitions, the positions place each tile:

```python
for im in f.select(illumination=0):
    p = im.metadata.position
    print(im.index["tile"], round(p["x"].value, 1), round(p["y"].value, 1))
# 0 -3946.8 13665.6
# 1 -677.3  13665.6
# 2 -3946.8 15846.9
# 3 -677.3  15846.9
```

To convert positions into pixel offsets, divide by the pixel size, after converting both to the same unit.

## Channels

`channels` is a tuple of `Channel` objects, one per channel of the native `c` axis, in order:

```python
Channel(name=None, color=None, emission_wavelength=None, excitation_wavelength=None)
```

| Field | Type | Example |
| --- | --- | --- |
| `name` | `str` or `None` | `'DAPI'` |
| `color` | `'#RRGGBB'` or `None` | `'#00A1FF'` |
| `emission_wavelength` | `Quantity` or `None` | `Quantity(465.0, 'nanometer')` |
| `excitation_wavelength` | `Quantity` or `None` | `Quantity(353.0, 'nanometer')` |

An image without a `c` axis has one channel entry. A file without any channel metadata gives `Channel()` entries with every field `None`.

```python
names = [ch.name or f"channel {i}" for i, ch in enumerate(meta.channels)]
colors = [ch.color or "#FFFFFF" for ch in meta.channels]     # your fallback, not micro-reader's
```

## Index values

Some indices have a physical value: the wavelength of an emission band, the time of a FLIM bin. `index_values` gives this image's value:

```python
f = micro_reader.open("DAPI-mCherry_5Lambda.oir")
[(im.index["emission"], im.metadata.index_values["emission"].value) for im in f.images]
# [(0, 420.0), (1, 470.0), (2, 520.0), (3, 570.0), (4, 620.0)]
```

`index_values` is empty for indices without a physical value, such as `tile` or `view`.

## Quantity

```python
Quantity(value: float, unit: str)
```

A `Quantity` is a named tuple, so you can unpack it:

```python
value, unit = meta.pixel_sizes["x"]
```

`Quantity.to(unit)` converts between units of the same kind, length or time:

```python
micro_reader.Quantity(0.5, "micrometer").to("nanometer")
# Quantity(value=499.99999999999994, unit='nanometer')
```

!!! warning
    `to()` raises `KeyError` for a unit name it does not know. It does not check that both units are of the same kind, so converting a length to a time gives a meaningless number. Convert lengths only to length units and times only to time units.

The known unit names are all OME-Zarr space and time units, from `yoctometer` to `yottameter` (plus `angstrom`, `inch`, `foot`, `yard`, `mile`, `parsec`) and from `yoctosecond` to `yottasecond` (plus `minute`, `hour`, `day`).

## From micro-reader metadata to OME-Zarr

A sketch of the `multiscales` scale and the `omero` channels for one image in canonical axes:

```python
def ngff_scale(image):
    """coordinateTransformations scale for a canonical tczyx image (one level)."""
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
    return [{"label": ch.name or f"c{i}", "color": (ch.color or "#FFFFFF").lstrip("#"),
             "active": True}
            for i, ch in enumerate(image.metadata.channels)]
```

Here `1.0` is a placeholder that OME-Zarr requires, written by your code, not by micro-reader. Record somewhere that the size was unknown if that matters to you.

## Plates

For plates, the plate layout (plate name, rows, columns, wells, acquisitions) is on `f.plates`, not in `image.metadata`. See [Plates](plates.md).
