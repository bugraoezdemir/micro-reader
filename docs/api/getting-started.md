# Getting started

This page takes you from installing micro-reader to reading pixels and metadata. Each step links to the guide page that covers it in full.

## 1. Install

```bash
pip install micro-reader
```

Add `[hdf5]` for Imaris, ilastik, BigDataViewer and other HDF5 files: `pip install "micro-reader[hdf5]"`.

## 2. Open a file and look inside

```python
import micro_reader

f = micro_reader.open("MouseBrain_2x2Tiles_3Channels_2Illuminations.czi")
print(f)
```

```text
<micro_reader.File 'MouseBrain_2x2Tiles_3Channels_2Illuminations.czi' backend='czi' format='czi' images=8>
  series 0: tiles 0-3, x illumination 2, tczyx (1, 3, 41, 1000, 1500) uint16
```

The printout shows the file's structure:

- one series, made of four mosaic tiles (`tiles 0-3`);
- each tile was imaged from two sides (`x illumination 2`);
- so the file holds 4 × 2 = 8 images;
- each image is a `t c z y x` array of `uint16`.

Use the file as a context manager, so that its file handles are closed when you are done:

```python
with micro_reader.open("image.czi") as f:
    ...
```

→ [Opening files](guide/opening.md)

## 3. Pick an image

Every image has an `index` that says which one it is:

```python
f.images[0].index
# {'series': 0, 'tile': 0, 'illumination': 0}

f.index_range
# {'series': 1, 'tile': 4, 'illumination': 2}

left_side = f.select(illumination=0)     # the 4 tiles imaged from side 0
image = f.select(tile=2, illumination=1)[0]
```

A value may also be a list, tuple, range or set, meaning any of them: `f.select(tile=[0, 1])`. For plates you can select by well name: `f.select(well="B2")`, `f.select(row="C", column=range(2, 5))` (see [Plates](guide/plates.md)).

→ [Finding images](guide/finding-images.md)

## 4. Read pixels

An image behaves like a read-only NumPy array that is never fully loaded:

```python
image.axes     # 'tczyx'
image.shape    # (1, 3, 41, 1000, 1500)
image.dtype    # dtype('uint16')

plane = image[0, 1, 20]                   # t=0, channel 1, z=20 -> (1000, 1500)
block = image.read(c=1, z=slice(10, 20), y=slice(0, 512))   # keyword form
block.shape    # (1, 10, 512, 1500): an integer drops its axis, as in NumPy
```

Only the bytes under the requested region are read.

→ [Reading pixels](guide/reading.md)

## 5. Get the metadata

```python
meta = image.metadata

meta.pixel_sizes
# {'x': Quantity(value=2.5663047987549903, unit='micrometer'),
#  'y': Quantity(value=2.5663047987549903, unit='micrometer'),
#  'z': Quantity(value=7.0, unit='micrometer')}

[ch.name for ch in meta.channels]
# ['Cam1-T1', 'Cam2-T1', 'Cam2-T2']

meta.channels[0].emission_wavelength.unit
# 'nanometer'

meta.position                # where the image is, in the stage's coordinates
# {'x': Quantity(value=-3946.82..., unit='micrometer'), 'y': Quantity(value=15846.91..., unit='micrometer')}
meta.coordinate_system       # the system the position is in
# 'stage'
```

→ [Metadata](guide/metadata.md)

## 6. Choose the array layout you want

Different formats store their axes in different orders. To get the same layout everywhere, ask for the canonical one:

```python
f = micro_reader.open("stack.tif", axes="canonical")   # every image t c z y x (+ s)
```

Or reshape a single image:

```python
image.as_axes("zyx")          # drop size-1 t and c, put z first
image.squeeze()               # drop every size-1 axis except y and x
```

RGB images (photos, brightfield slides) open as three channels, `c` = 3: red, green and blue. Pass `samples="keep"` to `open` to get the stored `s` axis instead.

→ [Axes](guide/axes.md)

## 7. Where next

| You want to… | Read |
| --- | --- |
| Understand files, images and indices | [Concepts](concepts.md) |
| Read big files without running out of memory | [Memory and threads](guide/performance.md) |
| Work with pyramids (whole-slide, Imaris, CZI) | [Resolution levels](guide/pyramids.md) |
| Read plates (Harmony, Columbus, MetaXpress, IN Cell, CellVoyager, Opera Flex, scanR, ...) | [Plates](guide/plates.md) |
| Sort a folder into images before converting | [Multi-file images](guide/multi-file.md#grouping-many-files) |
| See complete programs | [Examples](examples.md) |
