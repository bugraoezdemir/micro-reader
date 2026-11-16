# Finding images

A file contains a list of images. This page shows how to inspect that list, understand the file's structure and select the images you need.

## The outline

`print(f)` shows a summary of the file's structure. Use `f.outline()` to inspect the full structure:

```python
print(f.outline())
```

```text
<micro_reader.File 'MouseBrain_41Slices_1Tile_3Channel_2Illuminations_2Angles.czi' backend='czi' format='czi' images=4>
  series 0: x view 2, illumination 2, tczyx (1, 3, 41, 1000, 1500) uint16
```

Each line describes a **containment node**: a plate, well, field or series. It can include the following information:

- **Tiles:** `tiles 0-3` lists the tiles belonging to that node.
- **Orthogonal indices:** `x view 2, illumination 2` lists the orthogonal indices and the number of values for each. Here, two views and two illumination sides give 2 × 2 = 4 images.
- **Image layout:** The axes, shape and data type of the images. If image shapes differ, the outline reports `shapes vary`.
- **Pyramids and ragged parts:** The number of resolution levels (for example, `4 levels`) and the parts of a ragged image appear at the end.

Here are examples from other file types.

=== "ND2: four positions"

    ```text
    <micro_reader.File 'multipoint.nd2' backend='nd2' format='nd2-legacy' images=4>
      series 0: tczyx (11, 1, 5, 1040, 1392) uint16
      series 1: tczyx (11, 1, 5, 1040, 1392) uint16
      series 2: tczyx (11, 1, 5, 1040, 1392) uint16
      series 3: tczyx (11, 1, 5, 1040, 1392) uint16
    ```

=== "OIR: lambda scan"

    ```text
    <micro_reader.File 'DAPI-mCherry_3T_4Z_5Lambda.oir' backend='oir' format='oir' images=1>
      series 0: tczyx (3, 5, 4, 512, 512) uint16
    ```

    The five bands are channels of one image (see [Lambda scans](axes.md#lambda-scans)).

=== "Imaris: pyramid"

    ```text
    <micro_reader.File '362.ims' backend='ims' format='ims' images=1>
      series 0: tczyx (1, 10, 364, 512, 512) float32, 4 levels
    ```

=== "Harmony: plate"

    ```text
    <micro_reader.File 'Index.idx.xml' backend='harmony' format='harmony-v4' images=6 plates=1>
      plate 0 'HipSci Experiment 10 Plate 1': 8 x 12, 3 wells with images
        well A1
          field 0: tczyx (1, 4, 1, 1024, 1360) uint16
          field 1: tczyx (1, 4, 1, 1024, 1360) uint16
        well A2
          ...
    ```

=== "CZI: plate with tiled fields"

    ```text
    <micro_reader.File 'W96_B2+B4.czi' backend='czi' format='czi' images=90 plates=1>
      plate 0: 8 x 12, 2 wells with images
        well B2
          field 0: tiles 0-44, tczyx (2, 3, 4, 420, 640) uint8
        well B4
          field 0: tiles 0-44, tczyx (2, 3, 4, 420, 640) uint8
    ```

## `indices` and `index_range`

`f.indices` lists the index of every image, in the same order as `f.images`:

```python
f.indices
# [
#     {'series': 0, 'view': 0, 'illumination': 0},
#     {'series': 0, 'view': 0, 'illumination': 1},
#     {'series': 0, 'view': 1, 'illumination': 0},
#     {'series': 0, 'view': 1, 'illumination': 1}
# ]
```

`f.index_range` reports the number of distinct values for each index:

```python
f.index_range
# {'series': 1, 'view': 2, 'illumination': 2}
```

!!! note "`index_range` is not an image count or array shape"

    `index_range` reports the widest range of each index. For example, if series 0 has 3 tiles and series 1 has 5, the `tile` range is 5. This does not mean there are 5 tiles in every series, or 8 tiles in total.

    Do not use `index_range` to calculate the number of images. Use `len(f.images)` instead, or select images within each series.

## `select()`

Use `f.select(**index)` to retrieve images matching all the index values you specify:

```python
f.select(view=1)                     # both illumination sides of view 1
f.select(view=1, illumination=0)[0]  # one image
f.select(series=3)                   # all images in series 3, including its tiles and views
```

`select()` always returns a list. It returns an empty list when no images match.

An unknown index name raises `KeyError` and reports the valid index names:

```python
f.select(position=1)
# KeyError: "unknown indices ['position']; the indices are ('plate', 'acquisition', 'well', ...)"
```

### Selecting several values

An index value can also be a list, tuple, range or set. An image matches if its index is any of the specified values:

```python
f.select(illumination=[0, 1], view=range(2))
f.select(series={0, 2})
```

For plates, you can also select by well name, row, column or slices. See [Plates](plates.md#selecting-images).

!!! tip "An absent index does not match"

    Calling `select(tile=0)` on a file without a `tile` index returns `[]`. No image in that file has a tile index.

    To write code that works with files both with and without tiles, check `f.index_range` first:

    ```python
    want = {"series": 0}

    if "tile" in f.index_range:
        want["tile"] = 0

    image = f.select(**want)[0]
    ```

## The file's tree

`f.collection` represents the file's contents as a hierarchy. Plates contain wells, wells contain fields, and outside plate formats, series contain their images.

The tree is built from the image indices and does not require reading image pixels or metadata:

```python
tree = f.collection

tree.name                         # file name without its extension
[c.name for c in tree.children]   # ['series0']
tree.images                       # every image below this node
```

See the [Collection reference](../reference/coordinates.md#collection) for the full API.

## Image names

Each image has a readable `name`. When the format does not provide its own name, micro-reader generates one using the image's identifying indices:

```python
f.images[0].name
# 'view 0, illumination 0'
```

Use names for display and logging. For unambiguous identification, use the image's `index`.

## Common patterns

### Group images by series

You can group images by their `series` index and process each series separately:

```python
from collections import defaultdict

by_series = defaultdict(list)

for image in f.images:
    by_series[image.index["series"]].append(image)

for series, images in sorted(by_series.items()):
    print(series, len(images), "images")
```

### Pair illumination sides of each tile

For acquisitions with two illumination sides, you can select the images for each tile and pair them by their `illumination` index:

```python
for tile in range(f.index_range.get("tile", 1)):
    sides = f.select(tile=tile) if "tile" in f.index_range else f.images

    left, right = sorted(
        sides,
        key=lambda im: im.index["illumination"]
    )

    fused = (
        left.read(c=0, z=20).astype("float32")
        + right.read(c=0, z=20)
    ) / 2
```

This example assumes each tile has exactly two illumination sides.

### Find an image's position in `f.images`

To get the index of an image in the file's image list:

```python
position = f.images.index(image)
```

## Plates

In plate datasets, images are organised by `plate`, `acquisition`, `well` and `field` indices rather than `series`. You can also navigate the plate layout directly through `f.plates`.

See [Plates](plates.md) for details.