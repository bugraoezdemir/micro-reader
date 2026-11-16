# Coordinate systems

An image's `metadata.position` describes where it is located, but a position is meaningful only in relation to a coordinate system. Each image therefore identifies the coordinate system in which its position is defined. This tells you which images can be placed or compared directly.

```python
f = micro_reader.open("slide_scan.czi")

f.coordinate_systems
# (CoordinateSystem(id='stage', name='stage', axes=None, estimated=False, extra={}),)

meta = f.images[0].metadata

meta.position             # {'x': Quantity(...), 'y': Quantity(...)}
meta.coordinate_system    # 'stage'
```

`f.coordinate_systems` lists the coordinate systems used by the entire file. It is available as soon as the file is opened and does not require reading any image pixels.

## Which images can be compared?

Two images are in the same coordinate system when their `metadata.coordinate_system` values are equal. Their positions can then be compared directly. Positions expressed in different coordinate systems cannot be compared without a transformation between those systems.

| Files | Coordinate systems |
| --- | --- |
| CZI, ND2 with a camera matrix, OME-TIFF, Imaris, FluoView tiling sets, scanR | `stage` |
| Leica LIF | `stage`, or one system per acquisition session (`stage-r0`, `stage-r1`, ...), or `stage-s<series>` for an element that records too little to tie it to a session |
| SVS, SCN, QPTIFF slides | `slide` |
| Plates (Harmony, Columbus, CellVoyager, Opera Flex, ImageXpress HCS.ai) | One system per well, such as `well-B2` |
| BigDataViewer / BigStitcher | `global` or `global-registered` |

The [reference](../reference/coordinates.md#how-a-position-names-its-system) describes how each format assigns coordinate systems.

For plate acquisitions, positions are relative to the centre of each well. Consequently, fields from different wells belong to different coordinate systems and their positions cannot be compared directly.

```python
f = micro_reader.open("Measurement1/Images/Index.idx.xml")

[s.id for s in f.coordinate_systems]
# ['well-A1', 'well-A2', 'well-B1']

f.images[0].metadata.position
# {'x': Quantity(-674.9, 'micrometer'),
#  'y': Quantity(-508.0, 'micrometer'),
#  'z': Quantity(11.0, 'micrometer')}
```

An image without a position has `None` as its `coordinate_system`. To obtain a copy of its metadata without either the position or coordinate system, use `metadata.unplaced()`.

## Images that are rotated or registered

BigDataViewer and BigStitcher files can contain views that are rotated, sheared or mirrored, or that have been moved through registration. These views do not have a simple position in the output coordinate system. Instead, `f.coordinate_transformations` describes how each view maps into that system.

```python
with micro_reader.open("HisYFP-SPIM.xml") as f:
    [s.id for s in f.coordinate_systems]
    # ['global-registered']

    t = f.coordinate_transformations[1]

    t.type       # 'affine'
    t.input      # (('series', 0), ('view', 1)): the image
    t.output     # 'global-registered'
    t.params["affine"]  # the transformation matrix
```

A coordinate system with `estimated=True`, such as `global-registered`, is derived from registration rather than directly recorded by the microscope.

Other formats do not currently provide coordinate transformations, so `f.coordinate_transformations` is empty for them.

See [Coordinates and the file's tree](../reference/coordinates.md#transformation) for the fields of `Transformation`.

## The file's tree

`f.collection` describes how images are organised within a file: plates contain wells, wells contain fields, and files outside plate formats are organised by series.

See [Finding images](finding-images.md#the-files-tree).