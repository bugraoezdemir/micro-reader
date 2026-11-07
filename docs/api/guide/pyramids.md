# Resolution levels

Some files store each image at several resolutions: level 0 at full size, then smaller copies. micro-reader reads these stored levels. It never computes new ones.

| Format | Levels |
| --- | --- |
| Imaris (`.ims`) | every stored level |
| TIFF | OME-TIFF sub-resolutions, whole-slide pyramids (SVS, NDPI, SCN, BIF, QPTIFF, Philips) |
| CZI | stored pyramids of whole series |
| VSI | each scanned area keeps its levels |
| others | one level |

## Level 0 and its pyramid

By default, `f.images` holds the full-resolution images. Each one gives its pyramid in `resolutions`, full resolution first:

```python
f = micro_reader.open("362.ims")
image = f.images[0]

image.is_pyramidal                       # True
[lv.shape for lv in image.resolutions]
# [(1, 10, 364, 512, 512),
#  (1, 10, 182, 256, 256),
#  (1, 10, 91, 128, 128),
#  (1, 10, 45, 64, 64)]

thumb = image.resolutions[-1].read(c=0, z=20)   # a 64 x 64 preview
```

A level is an ordinary `Image`. You read it, slice it and ask for its metadata like any other image.

When a file stores a pyramid, every index carries `resolution`. Level 0 has `resolution: 0`, even in files where only some images have lower levels:

```python
image.index                      # {'series': 0, 'resolution': 0}
image.resolutions[2].index       # {'series': 0, 'resolution': 2}
```

Files without stored levels have no `resolution` in their indices, and `image.resolutions == [image]`.

## Every level as an image

With `resolutions="all"`, each level is also listed in `f.images`, so you can `select` it:

```python
f = micro_reader.open("362.ims", resolutions="all")
f.index_range               # {'series': 1, 'resolution': 4}
len(f.images)               # 4

level2 = f.select(series=0, resolution=2)[0]
level2.shape                # (1, 10, 91, 128, 128)
```

Use this when your code treats every level the same, for example when copying a pyramid level by level. The [outline](finding-images.md#the-outline) shows only level 0 either way, with `4 levels` at its end.

## Moving between levels

Every image knows its place in its pyramid:

| Attribute | Level 0 | A lower level |
| --- | --- | --- |
| `image.level` | `0` | `1`, `2`, ... |
| `image.pyramid` | the image itself | the level-0 image |
| `image.resolutions` | all levels | all levels (the same list) |
| `image.is_pyramidal` | `True` if lower levels exist | `True` |

```python
level2.level                       # 2
level2.pyramid is f.images[0]      # True
```

## Pixel sizes per level

Each level has its own pixel sizes. They are level 0's pixel sizes scaled by the shape ratio:

```python
image.metadata.pixel_sizes["x"]                  # Quantity(0.2965, 'micrometer')
image.resolutions[2].metadata.pixel_sizes["x"]   # Quantity(1.1861, 'micrometer'): 512 / 128 = 4x
image.resolutions[2].metadata.pixel_sizes["z"]   # Quantity(1.2, 'micrometer'):    364 / 91 = 4x
```

Channels and positions are the same as level 0's.

## Pick a level for a target size

```python
def level_for(image, max_side=2048):
    """The largest level whose y and x both fit in max_side."""
    y, x = image.axes.index("y"), image.axes.index("x")
    for level in image.resolutions:
        if level.shape[y] <= max_side and level.shape[x] <= max_side:
            return level
    return image.resolutions[-1]

preview = level_for(f.images[0], 1024).read(c=0, z=0)
```

## Notes per format

- **CZI.** Pyramids belong to whole series. With `tiles="separate"`, the tiles of a series have no levels of their own. Open with `tiles="mosaic"` to get each series with its levels. A [damaged CZI](../formats.md#czi) is read without its pyramid.
- **Imaris.** Level 0 has the real image size. The padding that Imaris adds to fill its chunks is removed on every level.
- **Whole-slide TIFF.** The slide comes first with its levels. Label, macro and thumbnail pictures are further series (see [Formats](../formats.md#tiff)).
