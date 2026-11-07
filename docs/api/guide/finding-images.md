# Finding images

A file holds a list of images. This page shows how to see what is in a file and how to pick the images you want.

## The outline

`print(f)` shows the start of a file's structure. `f.outline()` shows all of it:

```python
print(f.outline())
```

```text
<micro_reader.File 'MouseBrain_41Slices_1Tile_3Channel_2Illuminations_2Angles.czi' backend='czi' format='czi' images=4>
  series 0: x view 2, illumination 2, tczyx (1, 3, 41, 1000, 1500) uint16
```

How to read it:

- Each line is a *containment* node: a plate, a well, a field or a series.
- `tiles 0-3` lists the tiles in that node.
- `x view 2, illumination 2` lists the *orthogonal* indices that vary in it, with their counts. Here 2 views × 2 sides = 4 images.
- Then come the axes, shape and dtype of its images (`shapes vary` when they differ), the number of pyramid levels (`4 levels`), and the parts of a ragged image.

More outlines from real files:

=== "ND2, 4 positions"

    ```text
    <micro_reader.File 'multipoint.nd2' backend='nd2' format='nd2-legacy' images=4>
      series 0: tczyx (11, 1, 5, 1040, 1392) uint16
      series 1: tczyx (11, 1, 5, 1040, 1392) uint16
      series 2: tczyx (11, 1, 5, 1040, 1392) uint16
      series 3: tczyx (11, 1, 5, 1040, 1392) uint16
    ```

=== "OIR lambda scan"

    ```text
    <micro_reader.File 'DAPI-mCherry_3T_4Z_5Lambda.oir' backend='oir' format='oir' images=5>
      series 0: x emission 5, tczyx (3, 1, 4, 512, 512) uint16
    ```

=== "Imaris pyramid"

    ```text
    <micro_reader.File '362.ims' backend='ims' format='ims' images=1>
      series 0: tczyx (1, 10, 364, 512, 512) float32, 4 levels
    ```

=== "Harmony plate"

    ```text
    <micro_reader.File 'Index.idx.xml' backend='harmony' format='harmony-v4' images=6 plates=1>
      plate 0 'HipSci Experiment 10 Plate 1': 8 x 12, 3 wells with images
        well A1
          field 0: tczyx (1, 4, 1, 1024, 1360) uint16
          field 1: tczyx (1, 4, 1, 1024, 1360) uint16
        well A2
          ...
    ```

=== "CZI plate, tiled fields"

    ```text
    <micro_reader.File 'W96_B2+B4.czi' backend='czi' format='czi' images=90 plates=1>
      plate 0: 8 x 12, 2 wells with images
        well B2
          field 0: tiles 0-44, tczyx (2, 3, 4, 420, 640) uint8
        well B4
          field 0: tiles 0-44, tczyx (2, 3, 4, 420, 640) uint8
    ```

## `indices` and `index_range`

`f.indices` lists every image's index, in the order of `f.images`:

```python
f.indices
# [{'series': 0, 'view': 0, 'illumination': 0},
#  {'series': 0, 'view': 0, 'illumination': 1},
#  {'series': 0, 'view': 1, 'illumination': 0},
#  {'series': 0, 'view': 1, 'illumination': 1}]
```

`f.index_range` gives the number of distinct values of each index:

```python
f.index_range
# {'series': 1, 'view': 2, 'illumination': 2}
```

!!! note "`index_range` is not a shape"
    It is the *widest* range of each index. If series 0 has 3 tiles and series 1 has 5, the range of `tile` is 5: it is not the total (8), and not every series has 5 tiles. Do not use it to compute the number of images. Use `len(f.images)`, or `select()` per series.

## `select()`

`f.select(**index)` returns the images whose index matches every name you give:

```python
f.select(view=1)                      # both illumination sides of view 1
f.select(view=1, illumination=0)[0]   # one image
f.select(series=3)                     # everything in series 3 (tiles, views, ...)
```

It always returns a list, which is empty when nothing matches. An unknown index name raises `KeyError` with the list of valid names:

```python
f.select(position=1)
# KeyError: "unknown indices ['position']; the indices are ('plate', 'acquisition', 'well', ...)"
```

### Several values at once

An index value can be a list, tuple, range or set, meaning any of them:

```python
f.select(illumination=[0, 1], view=range(2))   # any of these illuminations and views
f.select(series={0, 2})
```

On a plate you can also select by well name, by row and column, and with slices. See [Plates](plates.md#selecting-images).

!!! tip "Missing names do not match"
    `select(tile=0)` on a file without tiles returns `[]`, because none of the images has a `tile` index. To write code that works for files with and without tiles, check `f.index_range` first:

    ```python
    want = {"series": 0}
    if "tile" in f.index_range:
        want["tile"] = 0
    image = f.select(**want)[0]
    ```

## The file's tree

`f.collection` is the file's contents as a tree: plates hold wells, wells hold fields, and outside plates the series hold their images. It is built from the indices alone, so it reads no metadata:

```python
tree = f.collection
tree.name                         # the file name without its extension
[c.name for c in tree.children]   # ['series0']
tree.images                       # every image below it
```

The reference describes it in full: [Collection](../reference/coordinates.md#collection).

## Image names

Each image has a readable `name`, made from the file name and its index where the format gives no name of its own:

```python
f.images[0].name
# 'view 0, illumination 0'
```

Use names for display and logs. To identify an image, use its `index`.

## Common patterns

### Loop over series, then over the images in each

```python
from collections import defaultdict

by_series = defaultdict(list)
for image in f.images:
    by_series[image.index["series"]].append(image)

for series, images in sorted(by_series.items()):
    print(series, len(images), "images")
```

### Pair the illumination sides of each tile

```python
for tile in range(f.index_range.get("tile", 1)):
    sides = f.select(tile=tile) if "tile" in f.index_range else f.images
    left, right = sorted(sides, key=lambda im: im.index["illumination"])
    fused = (left.read(c=0, z=20).astype("float32") + right.read(c=0, z=20)) / 2
```

### Where is image *i* in the list?

```python
position = f.images.index(image)
```

## Plates

On a plate, images are indexed by `plate`, `acquisition`, `well` and `field` instead of `series`, and you can walk the plate layout. See [Plates](plates.md).
