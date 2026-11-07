# Coordinate systems

An image's `metadata.position` is only useful when you know what it is measured against. Each position names its coordinate system, so you can tell which images can be placed next to each other.

```python
f = micro_reader.open("slide_scan.czi")
f.coordinate_systems
# (CoordinateSystem(id='stage', name='stage', axes=None, estimated=False, extra={}),)

meta = f.images[0].metadata
meta.position             # {'x': Quantity(...), 'y': Quantity(...)}
meta.coordinate_system    # 'stage'
```

`f.coordinate_systems` lists the systems of the whole file. It is known as soon as the file is open and reads no image.

## Which images can be compared

Two images are in the same space when their `coordinate_system` is equal. Positions in different systems cannot be compared.

| Files | Coordinate systems |
| --- | --- |
| CZI, ND2 with a camera matrix, OME-TIFF, Imaris, FluoView tiling sets, scanR | `stage` |
| Leica LIF | `stage`, or one system per acquisition session (`stage-r0`, `stage-r1`, ...) |
| SVS, SCN, QPTIFF slides | `slide` |
| Plates (Harmony, Columbus, CellVoyager, Opera Flex, ImageXpress HCS.ai) | one per well: `well-B2` |
| BigDataViewer / BigStitcher | `global`, or `global-registered` |

The [reference](../reference/coordinates.md#how-a-position-names-its-system) lists the rules per format.

On a plate, positions are relative to the centre of their well, so fields of different wells are never in the same system:

```python
f = micro_reader.open("Measurement1/Images/Index.idx.xml")
[s.id for s in f.coordinate_systems]
# ['well-A1', 'well-A2', 'well-B1']

f.images[0].metadata.position
# {'x': Quantity(-674.9, 'micrometer'), 'y': Quantity(-508.0, 'micrometer'), 'z': Quantity(11.0, 'micrometer')}
```

An image without a position has `coordinate_system` `None`. `metadata.unplaced()` gives a copy of the metadata without position and coordinate system.

## Images that are turned or registered

BigDataViewer and BigStitcher files can hold views that are rotated, sheared or mirrored, or that a registration moved. These have no plain position. `f.coordinate_transformations` holds an edge for each of them:

```python
with micro_reader.open("HisYFP-SPIM.xml") as f:
    [s.id for s in f.coordinate_systems]        # ['global-registered']
    t = f.coordinate_transformations[1]
    t.type                                      # 'affine'
    t.input                                     # (('series', 0), ('view', 1)): the image
    t.output                                    # 'global-registered'
    t.params["affine"]                          # the matrix
```

A system with `estimated=True`, like `global-registered`, comes from a registration and not from the microscope's own record. Other formats have no transformations: `f.coordinate_transformations` is empty.

See [Coordinates and the file's tree](../reference/coordinates.md#transformation) for the fields of `Transformation`.

## The file's tree

`f.collection` lists what contains what: plates, wells, fields, and series outside plates. See [Finding images](finding-images.md#the-files-tree).
