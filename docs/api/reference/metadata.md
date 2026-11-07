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

One image's metadata, from `Image.metadata`. Anything the file does not say is absent from the dicts, or `None` in a `Channel`.

| Field | Type | Description |
| --- | --- | --- |
| `pixel_sizes` | `dict[str, Quantity]` | Keys from `x`, `y`, `z`, `t`. Space: the size of a pixel. `t`: the time between frames. Scaled per [resolution level](../guide/pyramids.md#pixel-sizes-per-level). |
| `position` | `dict[str, Quantity]` | Position of the centre of the first pixel, in the coordinate system `coordinate_system`. Keys from `x`, `y`, `z`. Empty when the file gives none. |
| `coordinate_system` | `str` or `None` | The `id` of the [coordinate system](coordinates.md#coordinatesystem) `position` is in: one of `File.coordinate_systems`. Set exactly when `position` is not empty; `None` for an image without a position. Constructing a `Metadata` with a coordinate system and an empty position raises `ValueError`. |
| `channels` | `tuple[Channel, ...]` | One per index of the native `c` axis; one entry when there is no `c` axis. |
| `index_values` | `dict[str, Quantity]` | The physical value of this image's indices that have one: `{"emission": Quantity(510.0, "nanometer")}`, `{"lifetime": Quantity(2.5, "nanosecond")}`. |
| `stage_position` | `dict[str, Quantity]` | Stage coordinates the file records for this image, as recorded, where the file's convention (which point of the field, the stage's directions against the image axes) is not verified. They are not a `position`: no coordinate system, and not to be used to place images against each other. Keys from `x`, `y`, `z`. Filled by MetaMorph and Micro-Manager files. |

| Method | Description |
| --- | --- |
| `scaled(factors) -> Metadata` | A copy with each pixel size multiplied by `factors[axis]` (a `dict` of axis to `float`). Used for resolution levels. |
| `unplaced() -> Metadata` | A copy without a position: `position` empty and `coordinate_system` `None`; the rest kept. |
| `Metadata.unknown(n_channels=1) -> Metadata` | Class method: empty metadata with `n_channels` empty channels. |

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
| `name` | `str` or `None` | Channel name as the file gives it: dye, filter or channel label. For RGB samples folded into channels (the default of [`open`](functions.md#open)): `"<name> R"`, `"G"`, `"B"` (`"A"`). |
| `color` | `str` or `None` | Display colour, upper-case `'#RRGGBB'`. Red, green, blue for folded RGB samples (white for `A`). |
| `emission_wavelength` | `Quantity` or `None` | Emission wavelength, or the centre of the emission band. |
| `excitation_wavelength` | `Quantity` or `None` | Excitation (laser or filter) wavelength. |

## Quantity

```python
class micro_reader.Quantity(NamedTuple):
    value: float
    unit: str
```

A number with an OME-Zarr (UDUNITS-2) unit name such as `"micrometer"`, `"nanometer"`, `"second"` or `"millisecond"`. As a named tuple it compares, hashes and unpacks like a tuple:

```python
q = micro_reader.Quantity(0.5, "micrometer")
value, unit = q
q == (0.5, "micrometer")      # True
```

### `to`

```python
Quantity.to(unit) -> Quantity
```

This quantity in another unit. Returns `self` when `unit` is already the unit.

| Parameter | Type | Description |
| --- | --- | --- |
| `unit` | `str` | An OME-Zarr space or time unit name. |

Raises `KeyError` for an unknown unit name. It does not check that both units are of the same kind: `Quantity(1, "nanometer").to("second")` returns a number without error, and the number is meaningless. Convert lengths to lengths and times to times.

```python
micro_reader.Quantity(510.0, "nanometer").to("micrometer")    # Quantity(0.51, 'micrometer')
micro_reader.Quantity(40715.0, "millisecond").to("second")    # Quantity(40.715, 'second')
```

### Unit names

| Kind | Names |
| --- | --- |
| Space | `angstrom`, `attometer`, `centimeter`, `decimeter`, `exameter`, `femtometer`, `foot`, `gigameter`, `hectometer`, `inch`, `kilometer`, `megameter`, `meter`, `micrometer`, `mile`, `millimeter`, `nanometer`, `parsec`, `petameter`, `picometer`, `terameter`, `yard`, `yoctometer`, `yottameter`, `zeptometer`, `zettameter` |
| Time | `attosecond`, `centisecond`, `day`, `decisecond`, `exasecond`, `femtosecond`, `gigasecond`, `hectosecond`, `hour`, `kilosecond`, `megasecond`, `microsecond`, `millisecond`, `minute`, `nanosecond`, `petasecond`, `picosecond`, `second`, `terasecond`, `yoctosecond`, `yottasecond`, `zeptosecond`, `zettasecond` |

Values from files keep the file's unit, except lengths in units larger than a millimetre, which are given in micrometres.
