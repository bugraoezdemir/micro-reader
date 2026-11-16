# Metadata classes

All three classes are immutable. See the [Metadata guide](../guide/metadata.md) for how to use them.

## Metadata

```python
@dataclass(frozen=True)
class micro_reader.Metadata:
    pixel_sizes: dict = {}       # axis -> Quantity
    position: dict = {}          # axis -> Quantity
    coordinate_system: str | None = None   # id of the system position is in
    channels: tuple = ()         # Channel per native channel
    index_values: dict = {}      # index name -> Quantity
    stage_position: dict = {}    # axis -> Quantity, unverified
```

The metadata of one image, from `Image.metadata`. Anything the file does not say is absent from the dicts, or `None` in a `Channel`.

| Field | Type | Description |
| --- | --- | --- |
| `pixel_sizes` | `dict[str, Quantity]` | The keys come from `x`, `y`, `z` and `t`. For the space axes, the value is the size of a pixel. For `t`, it is the time between frames. The values are scaled per [resolution level](../guide/pyramids.md#pixel-sizes-per-level). |
| `position` | `dict[str, Quantity]` | The position of the centre of the first pixel, in the coordinate system `coordinate_system`. The keys come from `x`, `y` and `z`. It is empty when the file gives none. |
| `coordinate_system` | `str` or `None` | The `id` of the [coordinate system](coordinates.md#coordinatesystem) that `position` is in. It is one of `File.coordinate_systems`. It is set exactly when `position` is not empty, and it is `None` for an image without a position. Constructing a `Metadata` with a coordinate system and an empty position raises `ValueError`. |
| `channels` | `tuple[Channel, ...]` | One channel for each index of the native `c` axis, or one entry when there is no `c` axis. |
| `index_values` | `dict[str, Quantity]` or `dict[str, tuple[Quantity, ...]]` | The physical value of those indices of this image that have one, for example `{"excitation": Quantity(488.0, "nanometer")}` or `{"lifetime": Quantity(2.5, "nanosecond")}`. For a lambda scan opened with an `l` axis (`axes=None`), `"emission"` holds a tuple with the wavelength of each band along `l`. |
| `stage_position` | `dict[str, Quantity]` | The stage coordinates that the file records for this image, as recorded. The convention of the file is not verified, which means it is unknown which point of the field they refer to and how the directions of the stage relate to the image axes. They are not a `position`. They have no coordinate system, and they should not be used to place images against each other. The keys come from `x`, `y` and `z`. MetaMorph and Micro-Manager files fill them. |

| Method | Description |
| --- | --- |
| `scaled(factors) -> Metadata` | Returns a copy with each pixel size multiplied by `factors[axis]`, where `factors` is a `dict` of axis to `float`. It is used for resolution levels. |
| `unplaced() -> Metadata` | Returns a copy without a position. `position` is empty and `coordinate_system` is `None`, and the rest is kept. |
| `Metadata.unknown(n_channels=1) -> Metadata` | Class method that returns empty metadata with `n_channels` empty channels. |

## Channel

```python
@dataclass(frozen=True)
class micro_reader.Channel:
    name: str | None = None
    color: str | None = None                     # '#RRGGBB'
    emission_wavelength: Quantity | None = None
    excitation_wavelength: Quantity | None = None
```

| Field | Type | Description |
| --- | --- | --- |
| `name` | `str` or `None` | The channel name as the file gives it, which is a dye, a filter or a channel label. For RGB samples folded into channels, which is the default of [`open`](functions.md#open), the names are `"<name> R"`, `"G"`, `"B"` and `"A"`. |
| `color` | `str` or `None` | The display colour as upper-case `'#RRGGBB'`. For folded RGB samples it is red, green and blue, and white for `A`. |
| `emission_wavelength` | `Quantity` or `None` | The emission wavelength, or the centre of the emission band. |
| `excitation_wavelength` | `Quantity` or `None` | The excitation wavelength of the laser or filter. |

## Quantity

```python
class micro_reader.Quantity(NamedTuple):
    value: float
    unit: str
```

A number with an OME-Zarr (UDUNITS-2) unit name such as `"micrometer"`, `"nanometer"`, `"second"` or `"millisecond"`. Because it is a named tuple, it compares, hashes and unpacks like a tuple:

```python
q = micro_reader.Quantity(0.5, "micrometer")
value, unit = q
q == (0.5, "micrometer")      # True
```

### `to`

```python
Quantity.to(unit) -> Quantity
```

Returns this quantity in another unit. It returns `self` when `unit` is already the unit.

| Parameter | Type | Description |
| --- | --- | --- |
| `unit` | `str` | An OME-Zarr space or time unit name, or a short spelling that files use, such as `"um"`, `"nm"`, `"ms"` or `"s"`. |

It raises `ValueError` for a unit name it does not know, and for a conversion between a length and a time, such as `Quantity(1, "nanometer").to("second")`. The message says which.

```python
micro_reader.Quantity(510.0, "nanometer").to("micrometer")    # Quantity(0.51, 'micrometer')
micro_reader.Quantity(40715.0, "millisecond").to("second")    # Quantity(40.715, 'second')
micro_reader.Quantity(60000, "ms").to("s")                    # Quantity(60.0, 'second')
```

### Unit names

| Kind | Names |
| --- | --- |
| Space | `angstrom`, `attometer`, `centimeter`, `decimeter`, `exameter`, `femtometer`, `foot`, `gigameter`, `hectometer`, `inch`, `kilometer`, `megameter`, `meter`, `micrometer`, `mile`, `millimeter`, `nanometer`, `parsec`, `petameter`, `picometer`, `terameter`, `yard`, `yoctometer`, `yottameter`, `zeptometer`, `zettameter` |
| Time | `attosecond`, `centisecond`, `day`, `decisecond`, `exasecond`, `femtosecond`, `gigasecond`, `hectosecond`, `hour`, `kilosecond`, `megasecond`, `microsecond`, `millisecond`, `minute`, `nanosecond`, `petasecond`, `picosecond`, `second`, `terasecond`, `yoctosecond`, `yottasecond`, `zeptosecond`, `zettasecond` |

Values from files keep the unit of the file. The exception is lengths in units larger than a millimetre, which are given in micrometres.
