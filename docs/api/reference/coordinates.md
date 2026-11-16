# Coordinates and the file's tree

These classes describe where the images sit and what contains what. They are new in 0.2.0. There are four small immutable classes, which are exported from `micro_reader`, and three `File` attributes that hold them:

| `File` attribute | Holds |
| --- | --- |
| [`coordinate_systems`](#coordinatesystem) | A `tuple[CoordinateSystem, ...]` of the shared systems that the positions of the images are in. |
| [`coordinate_transformations`](#transformation) | A `tuple[Transformation, ...]` of the edges that are not plain positions. |
| [`collection`](#collection) | A `Collection`, which is the tree of plate > well > field, or of series. |

All three come from what opening the file already read. None of them reads the metadata of an image, so asking for them costs nothing. Code that only reads `metadata.position` keeps working.

## How a position names its system

The `metadata.position` of an image is in one coordinate system, and `metadata.coordinate_system` is the `id` of that system. Both are set together or not at all, so an image without a position has `coordinate_system` set to `None`. Look the system up in `File.coordinate_systems`:

```python
systems = {s.id: s for s in f.coordinate_systems}
md = f.images[0].metadata
if md.coordinate_system is not None:
    system = systems[md.coordinate_system]
```

Two images are in the same space exactly when their `coordinate_system` ids are equal. Images in different systems cannot be compared by position.

Ids use only the characters `a-z A-Z 0-9 _ -`. Other characters become `_`, and an id never holds `.` or `--`.

### Systems per format

| Format | System |
| --- | --- |
| CZI | `stage`, one per file. The label and slide-preview pictures have no position. |
| ND2 | `stage`, for files calibrated with a camera-to-stage matrix. |
| OME-TIFF, OME companion sets | `stage`. |
| Imaris | `stage`. |
| Leica LIF | `stage` when all placed elements come from one session, which means one microscope, one LAS X run and no break of more than 2 hours. A file of several sessions gets one system per session, named `stage-r0`, `stage-r1` and so on, in series order. An element that records too little to tie it to a session gets `stage-s<series>`. |
| Harmony, Columbus, CellVoyager, Opera Flex, ImageXpress HCS.ai plates | One system per well, named like `well-B2`, or `well-p<plate>-B2` with several plates. Positions are relative to the well centre. |
| scanR plates, NIS-Elements well-plate runs | `stage`, when the files place their fields. |
| SVS, SCN, QPTIFF slides | `slide`. |
| FluoView tiling sets (`.omp2info`) | `stage`, for all groups. |
| BigDataViewer / BigStitcher | `global`, for images whose view transforms are plain, which means diagonal and positive. `global-registered` is for registrations. |

Every other format gives no position and declares no system. See [Formats](../formats.md) for the position rules of each format.

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
| `id` | The identifier of the system, as `Metadata.coordinate_system` names it. It may hold only letters, digits, `_`, `.` and `-`, otherwise `ValueError` is raised. |
| `name` | A readable name. It is the `id` when none is given. |
| `axes` | A tuple of [`Axis`](#axis), or `None` for a system that the backend declares without listing axes. The writer fills the axes from the images in the system. Names must not repeat. |
| `estimated` | `False` for a nominal system, which is the only kind that a position goes into. `True` for a system that a registration estimated, such as `global-registered` of BigStitcher. It must be a `bool`, otherwise `TypeError` is raised. |
| `extra` | The other keys of a system that was read from a file, as read. |

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

One axis of a coordinate system, with the fields that OME-Zarr (NGFF 0.6) gives an axis.

| Field | Description |
| --- | --- |
| `name` | Required, and it must not be empty. |
| `type` | `"space"`, `"time"`, `"channel"` or another NGFF 0.6 type. If it is left `None`, it is inferred from the name for `t`, `c`, `x`, `y` and `z`. Any other name without a type raises `ValueError`. |
| `unit` | An OME-Zarr unit name (see [Unit names](metadata.md#unit-names)), or `None`. |
| `discrete`, `long_name` | Both are as NGFF 0.6 allows. |
| `extra` | The other keys of an axis that was read from a file, as read. |

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

An edge of the transformation graph of the file that is not a plain position. Only BigDataViewer / BigStitcher files have any. An image whose nominal view transform turns, shears or mirrors it gets an `affine` edge into `global` instead of a position. Each registration, such as stitching, ICP or bead registration, is an edge into `global-registered`.

| Field | Description |
| --- | --- |
| `type` | The modelled types are `"identity"`, `"translation"`, `"scale"`, `"affine"` and `"sequence"`. Any other NGFF 0.6 type is carried as read. |
| `input`, `output` | A coordinate system id (`str`), or an image key. An image key is a tuple of `(index name, value)` pairs without `resolution`, such as `(("series", 0), ("view", 1))`, and it means the own system of that image. `None` is allowed only inside a sequence. |
| `params` | The other JSON fields of the transformation, as read. They are `{"affine": rows}`, where each row has `n + 1` numbers for the matrix and the translation, `{"translation": [...]}`, and `{"transformations": [...]}` for a sequence. |
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

A group in the tree of the file. `File.collection` is the root. The tree is plate > well > field inside plates, and series outside them.

| Field / property | Description |
| --- | --- |
| `name` | The root is named after the file without its extension, so `cells.ome.tif` and `cells.ome.zarr` give `cells`, and a folder keeps its name. A plate is named by its `Plate.name`, or `plate<n>` when it has none. A well is named by its name, such as `"B2"`. A field is named `field<n>` and a series `series<n>`. |
| `kind` | `"plate"`, `"well"`, `"field"` or `"series"`. It is `None` for the root. |
| `index` | The value of that index for this collection, such as the number of the plate or the index of the well. It is `None` for the root. |
| `children` | The collections and full-resolution [`Image`](image.md) objects in it, in `File.images` order. |
| `images` | Every image under this collection, depth first. These are the leaves. |
| `names` | The name of each child in the tree, in `children` order. For a collection, it is its own `name`. For an image, it is the index values that no ancestor holds, written like `tile3` or `view1_illumination0` and without `resolution`. It is `image` when an ancestor holds every value. |

A field or series that holds one image collapses into that image. The image sits in its parent and keeps its index. A plate and a well never collapse. Tiles and the orthogonal indices, such as view, illumination and acquisition, make no collection, and the images that differ in them sit side by side. The leaves are the objects of `File.images`, also after `open(axes=...)`, and the tree follows them.

```python
c = f.collection
c.name              # 'hcs.companion'
c.names             # ('Sample HCS OME-TIFF',)
plate = c.children[0]
plate.kind, plate.names     # ('plate', ('A2', 'B1', 'B3', 'C2'))
```

`ome_zarr_pyramid` builds the same tree from the same rules.
