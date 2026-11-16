# Getting started

This page takes you from installation to reading image pixels and metadata. Each step links to a guide with more detail.

## 1. Install

```bash
pip install micro-reader
```

Imaris, ilastik, BigDataViewer and other HDF5-based formats require the `hdf5` extra:

```bash
pip install "micro-reader[hdf5]"
```

## 2. Open a file and inspect its contents

```python
import micro_reader

f = micro_reader.open("MouseBrain_41Slices_2x2Tiles_3Channels_2Illuminations_1Angle.czi")
print(f)
```

```text
<micro_reader.File 'MouseBrain_41Slices_2x2Tiles_3Channels_2Illuminations_1Angle.czi' backend='czi' format='czi' images=8>
  series 0: tiles 0-3, x illumination 2, tczyx (1, 3, 41, 1000, 1500) uint16
```

The output describes the file's structure. It contains one series with four mosaic tiles (`tiles 0-3`), each imaged from two illumination sides (`illumination 2`). This gives 4 × 2 = 8 images. Each image is a `t c z y x` array of `uint16`.

Use a context manager to ensure that file handles are closed when you are done:

```python
with micro_reader.open("image.czi") as f:
    ...
```

→ [Opening files](guide/opening.md)

## 3. Select an image

Each image has an `index` that identifies it within the file:

```python
f.images[0].index
# {'series': 0, 'tile': 0, 'illumination': 0}

f.index_range
# {'series': 1, 'tile': 4, 'illumination': 2}

left_side = f.select(illumination=0)  # the four tiles imaged from side 0
image = f.select(tile=2, illumination=1)[0]
```

You can also pass a list, tuple, range or set to select multiple values. For example:

```python
f.select(tile=[0, 1])
```

For plate formats, you can select images by name, too:

```python
f.select(plate=0, well_name="B2")
f.select(plate=0, row="C", column=range(2, 5))
```

Name the plate as well, by position (`plate=0`) or by name (`plate_name=`). A level you do not name matches every value, so `well_name="B2"` alone selects well B2 in every plate the file holds.

See [Plates](guide/plates.md) for the available plate-selection keywords.

→ [Finding images](guide/finding-images.md)

## 4. Read pixels

An image behaves like a read-only NumPy array. Its pixels remain in the file until you request them:

```python
image.axes   # 'tczyx'
image.shape  # (1, 3, 41, 1000, 1500)
image.dtype  # dtype('uint16')

# NumPy style indexing
plane = image[0, 1, 20]  # t=0, channel=1, z=20 -> (1000, 1500)

# Or use .read() method for axis-aware indexing
block = image.read(c=1, z=slice(10, 20), y=slice(0, 512))
block.shape  # (1, 10, 512, 1500)
```

As in NumPy, integer indexing removes the corresponding axis. Only the parts of the file overlapping the requested region are read and decoded.

→ [Reading pixels](guide/reading.md)

## 5. Read metadata

Metadata is available through `image.metadata`:

```python
meta = image.metadata

meta.pixel_sizes
# {
#     'x': Quantity(value=2.5663047987549903, unit='micrometer'),
#     'y': Quantity(value=2.5663047987549903, unit='micrometer'),
#     'z': Quantity(value=7.0, unit='micrometer')
# }

[ch.name for ch in meta.channels]
# ['Cam1-T1', 'Cam2-T1', 'Cam2-T2']

meta.channels[0].emission_wavelength.unit
# 'nanometer'

meta.position  # image position in the stage coordinate system
# {'x': Quantity(value=-3946.82..., unit='micrometer'),
#  'y': Quantity(value=15846.91..., unit='micrometer')}

meta.coordinate_system
# 'stage'
```

→ [Metadata](guide/metadata.md)

## 6. Choose the array layout

Different formats store image axes in different orders. By default, `open()` gives every image the same layout, `t c z y x`, adding missing axes with size 1. To keep the order the file stores instead, pass `axes=None`:

```python
f = micro_reader.open("cells.ics")             # every image is t c z y x
f = micro_reader.open("cells.ics", axes=None)  # the file's own order, for example c t z y x
```

You can also change the layout of an individual image:

```python
image.as_axes("zyx")  # drop size-1 t and c axes, and put z first
image.squeeze()       # drop size-1 axes except y and x
```

RGB images, such as photographs and brightfield slides, are represented as three channels (red, green and blue), so `c` has a size of 3. To preserve the stored `s` axis instead, pass `samples="keep"` to `open`.

→ [Axes](guide/axes.md)

## 7. Where to go next

| If you want to… | Read |
| --- | --- |
| Understand files, images and indices | [Concepts](concepts.md) |
| Read large files without running out of memory | [Memory and threads](guide/performance.md) |
| Work with pyramidal images (whole-slide, Imaris, CZI) | [Resolution levels](guide/pyramids.md) |
| Read high-content screening plates (Harmony, Columbus, MetaXpress, IN Cell, CellVoyager, Opera Flex, scanR, etc.) | [Plates](guide/plates.md) |
| Group files into images before converting them | [Multi-file images](guide/multi-file.md#grouping-many-files) |
| See complete working programs | [Examples](examples.md) |