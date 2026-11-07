# Coordinates and the file's tree

Where images sit, and what contains what (new in 0.2.0). Four small immutable classes, exported from `micro_reader`, and three `File` attributes that hold them:

| `File` attribute | Holds |
| --- | --- |
| [`coordinate_systems`](#coordinatesystem) | `tuple[CoordinateSystem, ...]`: the shared systems the images' positions are in |
| [`coordinate_transformations`](#transformation) | `tuple[Transformation, ...]`: the edges that are not plain positions |
| [`collection`](#collection) | a `Collection`: the tree of plate > well > field, or series |

All three come from what opening the file already read. None of them reads an image's metadata, so they cost nothing to ask for. Code that only reads `metadata.position` keeps working.

## How a position names its system

An image's `metadata.position` is in one coordinate system, and `metadata.coordinate_system` is its `id`. Both are set together or not at all: an image without a position has `coordinate_system` `None`. Look the system up in `File.coordinate_systems`:

```python
systems = {s.id: s for s in f.coordinate_systems}
md = f.images[0].metadata
if md.coordinate_system is not None:
    system = systems[md.coordinate_system]
```

Two images are in the same space exactly when their `coordinate_system` ids are equal. Images in different systems cannot be compared by position.

Ids use only `a-z A-Z 0-9 _ -`: other characters become `_`, and an id never holds `.` or `--`.

### Systems per format

| Format | System |
| --- | --- |
| CZI | `stage`, one per file. Label and slide-preview pictures have no position. |
| ND2 | `stage`, for files calibrated with a camera-to-stage matrix |
| OME-TIFF, OME companion sets | `stage` |
| Imaris | `stage` |
| Leica LIF | `stage` when all placed elements come from one session (one microscope, one LAS X run, no break over 2 hours). A file of several sessions gets one system per session, `stage-r0`, `stage-r1`, ... in series order. An element that records too little to tie it to a session gets `stage-s<series>`. |
| Harmony, Columbus, CellVoyager, Opera Flex, ImageXpress HCS.ai plates | one per well: `well-B2`; `well-p<plate>-B2` with several plates. Positions are relative to the well centre. |
| scanR plates, NIS-Elements well-plate runs | `stage`, when the files place their fields |
| SVS, SCN, QPTIFF slides | `slide` |
| FluoView tiling sets (`.omp2info`) | `stage`, all groups |
| BigDataViewer / BigStitcher | `global`, for images whose view transforms are plain (diagonal, positive); `global-registered` for registrations |

Every other format gives no position and declares no system. See [Formats](../formats.md) for each format's position rules.

## CoordinateSystem

```python
@dataclass(frozen=True)
class micro_reader.CoordinateSystem:
    id: str
    name: str | None = None
    axes: tuple[Axis, ...] | None = None
    estimated: bool = False
    extra: dict = {}
```

| Field | Description |
| --- | --- |
| `id` | The system's identifier, as `Metadata.coordinate_system` names it. Letters, digits, `_`, `.`, `-` only (`ValueError` otherwise). |
| `name` | A readable name. The `id` when none is given. |
| `axes` | A tuple of [`Axis`](#axis), or `None` for a system the backend declares without listing axes (the writer fills them from the images in it). Names must not repeat. |
| `estimated` | `False` for a nominal system, the only kind a position goes into. `True` for a system a registration estimated (BigStitcher's `global-registered`). Must be a `bool` (`TypeError` otherwise). |
| `extra` | The other keys of a system read from a file, as read. |

```python
f.coordinate_systems
# (CoordinateSystem(id='stage', name='stage', axes=None, estimated=False, extra={}),)
```

## Axis

```python
@dataclass(frozen=True)
class micro_reader.Axis:
    name: str
    type: str | None = None
    unit: str | None = None
    discrete: bool | None = None
    long_name: str | None = None
    extra: dict = {}
```

One axis of a coordinate system, with the fields OME-Zarr (NGFF 0.6) gives an axis.

| Field | Description |
| --- | --- |
| `name` | Required, non-empty. |
| `type` | `"space"`, `"time"`, `"channel"` or another NGFF 0.6 type. Left `None`, it is inferred from the name for `t`, `c`, `x`, `y`, `z`; any other name without a type raises `ValueError`. |
| `unit` | An OME-Zarr unit name (see [Unit names](metadata.md#unit-names)), or `None`. |
| `discrete`, `long_name` | As NGFF 0.6 allows. |
| `extra` | The other keys of an axis read from a file, as read. |

## Transformation

```python
@dataclass(frozen=True)
class micro_reader.Transformation:
    type: str
    input: object
    output: object
    params: dict = {}
    name: str | None = None
```

An edge of the file's transformation graph that is not a plain position. Only BigDataViewer / BigStitcher files have any: an image whose nominal view transform turns, shears or mirrors it gets an `affine` edge into `global` instead of a position, and each registration (stitching, ICP, bead registration) is an edge into `global-registered`.

| Field | Description |
| --- | --- |
| `type` | `"identity"`, `"translation"`, `"scale"`, `"affine"` or `"sequence"` are the types modelled; any other NGFF 0.6 type is carried as read. |
| `input`, `output` | A coordinate system id (`str`), or an image key: a tuple of `(index name, value)` pairs without `resolution`, such as `(("series", 0), ("view", 1))`, meaning that image's own system. `None` only inside a sequence. |
| `params` | The transformation's other JSON fields as read: `{"affine": rows}` (each row `n + 1` numbers: the matrix and the translation), `{"translation": [...]}`, `{"transformations": [...]}` of a sequence. |
| `name` | Optional. |

```python
for edge in f.coordinate_transformations:
    print(edge.type, edge.input, "->", edge.output)
# e.g.: affine (('series', 0), ('view', 1)) -> global
```

## Collection

```python
@dataclass(frozen=True)
class micro_reader.Collection:
    name: str
    kind: str | None = None
    index: object | None = None
    children: tuple = ()
```

A group in the file's tree. `File.collection` is the root: plate > well > field inside plates, and series outside them.

| Field / property | Description |
| --- | --- |
| `name` | The root is named after the file without its extension (`cells.ome.tif` and `cells.ome.zarr` give `cells`; a folder keeps its name). A plate is its `Plate.name`, `plate<n>` when it has none; a well its name (`"B2"`); a field `field<n>`; a series `series<n>`. |
| `kind` | `"plate"`, `"well"`, `"field"`, `"series"`; `None` for the root. |
| `index` | This collection's value of that index (the plate's number, the well's index, ...); `None` for the root. |
| `children` | The collections and full-resolution [`Image`](image.md) objects in it, in `File.images` order. |
| `images` | Every image under this collection, depth first (the leaves). |
| `names` | Each child's name in the tree, in `children` order. A collection's own `name`; for an image, its index values that no ancestor holds, as `tile3` or `view1_illumination0` (without `resolution`), or `image` when every value is held. |

A field or series that holds one image collapses into that image: the image sits in its parent and keeps its index. A plate and a well never collapse. Tiles and the orthogonal indices (view, illumination, acquisition, ...) make no collection; the images that differ in them sit side by side. The leaves are the objects of `File.images` (also after `open(axes=...)`), and the tree follows them.

```python
c = f.collection
c.name              # 'hcs.companion'
c.names             # ('Sample HCS OME-TIFF',)
plate = c.children[0]
plate.kind, plate.names     # ('plate', ('A2', 'B1', 'B3', 'C2'))
```

ome_zarr_pyramid builds the same tree from the same rules.
