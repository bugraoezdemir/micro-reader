# Resolution levels

Some microscopy formats store each image at multiple resolutions, with level 0 at full size and progressively smaller copies at lower levels. micro-reader reads these stored levels directly. It does not compute additional resolution levels.

| Format | Levels |
| --- | --- |
| Imaris (`.ims`) | Every stored level |
| TIFF | OME-TIFF sub-resolutions and whole-slide pyramids (SVS, NDPI, SCN, BIF, QPTIFF, Philips) |
| CZI | Stored pyramids of whole series |
| VSI | Each scanned area retains its own levels |
| Hamamatsu NDPIS | Every stored level of the set's NDPI files |
| BigDataViewer / BigStitcher | Every stored level of each view |
| Other formats | One level |

## Level 0 and its pyramid

By default, `f.images` contains the full-resolution images. Each image exposes its pyramid through `resolutions`, with level 0 first:

```python
f = micro_reader.open("362.ims")
image = f.images[0]

image.is_pyramidal                       # True
[lv.shape for lv in image.resolutions]
# [(1, 10, 364, 512, 512),
#  (1, 10, 182, 256, 256),
#  (1, 10,  91, 128, 128),
#  (1, 10,  45,  64,  64)]

thumb = image.resolutions[-1].read(c=0, z=20)  # a 64 x 64 preview
```

Each level is an ordinary `Image`. You can read it, slice it, and access its metadata just as you would for any other image.

For files with stored pyramids, every image index includes a `resolution` entry. Level 0 has `resolution: 0`, even in files where only some images have lower-resolution levels.

```python
image.index                       # {'series': 0, 'resolution': 0}
image.resolutions[2].index        # {'series': 0, 'resolution': 2}
```

Images without stored resolution levels have no `resolution` entry in their indices, and `image.resolutions == [image]`.

## Every level as an image

Pass `resolutions="all"` to include every level in `f.images`. You can then select a particular level directly:

```python
f = micro_reader.open("362.ims", resolutions="all")
f.index_range                     # {'series': 1, 'resolution': 4}
len(f.images)                     # 4

level2 = f.select(series=0, resolution=2)[0]
level2.shape                      # (1, 10, 91, 128, 128)
```

This is useful when your code processes every level in the same way, for example when copying a pyramid level by level.

The [outline](finding-images.md#the-outline) shows only level 0, regardless of this setting. It indicates the number of stored levels at the end, for example `4 levels`.

## Moving between levels

Every image knows its position in the pyramid:

| Attribute | Level 0 | Lower level |
| --- | --- | --- |
| `image.level` | `0` | `1`, `2`, ... |
| `image.pyramid` | The image itself | The level-0 image |
| `image.resolutions` | All levels | All levels (the same list) |
| `image.is_pyramidal` | `True` if lower levels exist | `True` |

```python
level2.level                      # 2
level2.pyramid is f.images[0]     # True
```

## Pixel sizes per level

Each level has its own pixel sizes, scaled from level 0 according to the change in shape:

```python
image.metadata.pixel_sizes["x"]                  # Quantity(0.2965, 'micrometer')
image.resolutions[2].metadata.pixel_sizes["x"]   # Quantity(1.1861, 'micrometer'): 512 / 128 = 4x
image.resolutions[2].metadata.pixel_sizes["z"]   # Quantity(1.2, 'micrometer'):   364 / 91 = 4x
```

Channel and position metadata are the same as for level 0.

## Choosing a level for a target size

The following helper returns the largest level whose `y` and `x` dimensions both fit within `max_side`:

```python
def level_for(image, max_side=2048):
    """Return the largest level whose y and x dimensions fit within max_side."""
    y, x = image.axes.index("y"), image.axes.index("x")
    for level in image.resolutions:
        if level.shape[y] <= max_side and level.shape[x] <= max_side:
            return level
    return image.resolutions[-1]

preview = level_for(f.images[0], 1024).read(c=0, z=0)
```

## Notes per format

- **CZI:** Pyramids belong to whole series. With `tiles="separate"`, individual tiles do not have their own resolution levels. Open with `tiles="mosaic"` to get each series with its pyramid. A [damaged CZI](../formats.md#czi) is read without its pyramid.
- **Imaris:** Level 0 has the actual image size. Imaris adds padding to fill its chunks, and micro-reader removes this padding at every level.
- **Whole-slide TIFF:** The slide image comes first, followed by its resolution levels. Label, macro, and thumbnail images appear as additional series (see [Formats](../formats.md#tiff)).