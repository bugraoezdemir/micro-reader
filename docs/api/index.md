# micro-reader

**Read regions of microscopy images with bounded memory.**

`micro-reader` is a Python library for reading microscopy files. Open a file, select an image, and read your region of interest as a NumPy array. Memory usage is therefore determined by the region being read, regardless of the image size.

```python
import micro_reader

with micro_reader.open("brain.czi") as f:
    image = f.images[0]
    block = image.read(
        c=1,
        z=10,
        y=slice(0, 2048),
        x=slice(1000, 3000),
    )
```

The same interface works across a wide range of formats:

- **Microscopy files:** TIFF and OME-TIFF (including whole-slide images), Zeiss CZI, Nikon ND2, Leica LIF, Olympus OIR / OIB / OIF / VSI, Imaris, ICS, MetaMorph, Micro-Manager, DeltaVision, MRC and KLB.
- **HDF5-based formats:** ilastik, BigDataViewer and Luxendo.
- **Image formats:** PNG, JPEG and BMP.
- **File collections:** OME companion sets and high-content screening plates from Harmony, Columbus, MetaXpress, ImageXpress, IN Cell, CellVoyager, CQ1, Opera Flex, scanR and Cellomics.

> **Alpha (0.x)**
>
> The API may change between releases. Pin the version you use in your projects.

## Where to start

- **[Getting started](getting-started.md)**  
  Install the package, open a file and read a region.

- **[Concepts](concepts.md)**  
  Understand files, images, indices, axes and coordinate systems.

- **[User guide](guide/opening.md)**  
  Practical guides to opening files, finding images, reading regions, working with axes and pyramids, accessing metadata, and handling plates and file sets.

- **[API reference](reference/index.md)**  
  Public functions, classes, attributes and errors, with signatures and parameter descriptions.

- **[Formats](formats.md)**  
  Supported formats, format-specific features and optional dependencies.

- **[Examples](examples.md)**  
  Recipes for tile-wise processing, Zarr export, Dask, plates, spectral scans and parallel workers.

## What it offers

### Region-based reading with bounded memory

A read returns the requested region as a NumPy array. Only the strips, tiles, subblocks or chunks that overlap the region are read and decoded.

For large compressed blocks, supported codecs can decode data in parts rather than loading the entire block. This includes Deflate, LZW, zstd, JPEG 2000 and JPEG with restart markers.

Decoding is also managed under a process-wide memory limit. See [Memory and threads](guide/performance.md).

### One model across formats

Image arrays use the axes `t c z y x`. Other dimensions, such as series, tiles, views, illumination sides and wells, are represented as image indices. The bands of a spectral (lambda) scan are channels.

RGB images are exposed as three channels.

Once you understand this model, you can work with different microscopy formats through the same interface.

### Metadata from the file

Pixel sizes, positions, channel names and colours are read from the file and use OME-Zarr-compatible unit names.

Positions also identify their coordinate system. If a value is not recorded in the file, it is left empty rather than filled with an assumed default such as `1.0`.

### A library you can build on

`micro-reader` has only three process-wide settings: memory, cache and threads. It does not configure logging or depend on Java or Dask behind the scenes.

You can use it as the reading layer in a converter, viewer, processing pipeline or notebook.

## Installation

```bash
pip install micro-reader
```

Optional dependencies:

```bash
pip install "micro-reader[hdf5]"     # HDF5-based formats, including Imaris (.ims)
pip install "micro-reader[codecs]"   # additional TIFF codecs, including LERC and CCITT fax
pip install "micro-reader[all]"      # both extras
```

`micro-reader` requires Python 3.11 or newer.

Native decoders are provided by the separate compiled package `micro-reader-codecs`, which is installed automatically.