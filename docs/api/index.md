# micro-reader

**Read exactly the rectangle you ask for, from any microscopy file.**

`micro-reader` is a Python library that reads pixels from microscopy files without loading the whole image. You open a file, pick the image you want, and read the region you need as a NumPy array. Large compressed blocks are decoded only as far as your region needs. Memory stays within a limit you set.

```python
import micro_reader

with micro_reader.open("brain.czi") as f:
    image = f.images[0]
    block = image.read(c=1, z=10, y=slice(0, 2048), x=slice(1000, 3000))
```

The same few objects work for every format: TIFF and OME-TIFF (including whole-slide files), Zeiss CZI, Nikon ND2, Leica LIF, Olympus OIR / OIB / OIF / VSI, Imaris, HDF5 (ilastik, BigDataViewer, Luxendo), ICS, MetaMorph, Micro-Manager, DeltaVision, MRC, KLB, PNG / JPEG / BMP, OME companion sets and high-content screening plates (Harmony, Columbus, MetaXpress, ImageXpress, IN Cell, CellVoyager, CQ1, Opera Flex, scanR, Cellomics).

!!! warning "Alpha (0.x)"
    The API may still change between releases. Pin the version you build on.

## Where to start

<div class="grid cards" markdown>

-   **[Getting started](getting-started.md)**

    Install it, open a file, read a region. Five minutes.

-   **[Concepts](concepts.md)**

    Files, images, indices, axes and coordinate systems: the model every format maps onto.

-   **[User guide](guide/opening.md)**

    One page per task: opening, finding images, reading, axes, pyramids, metadata, plates, file sets, memory.

-   **[API reference](reference/index.md)**

    Every public function, class, attribute and error, with signatures and parameter tables.

-   **[Formats](formats.md)**

    What each format gives you, and the extras some formats add.

-   **[Examples](examples.md)**

    Longer recipes: tile-wise processing, Zarr export, dask, plates, spectral scans, parallel workers.

</div>

## What makes it different

**It is memory-bounded.**
Reads touch only the blocks under your region. A block that must be decoded whole counts against a memory limit. If it cannot fit, you get a `MemoryLimitError`, not an out-of-memory crash.

**It has one model for every format.**
Arrays always have the axes `t c z y x`. Everything else, such as series, tiles, views, illumination sides, wells or spectral bands, is an *index* that tells the images apart. RGB images read as three channels. You learn this once and it works for every format.

**It never makes up metadata.**
Pixel sizes, positions, channel names and colours come from the file, with OME-Zarr unit names. A position also names the coordinate system it is in. When the file does not say, the value is absent. It is never set to `1.0`.

**It is a library, not an application.**
It has no global state beyond three process-wide settings (memory, cache, threads), no logging setup and no dependency on Java, dask or bioio. You can build it into a converter, a viewer, a pipeline or a notebook.

## Install

```bash
pip install micro-reader
pip install "micro-reader[hdf5]"     # Imaris (.ims), ilastik, BigDataViewer, Luxendo, Fiji HDF5: adds h5py
pip install "micro-reader[codecs]"   # JPEG XR, LERC and other rare codecs
pip install "micro-reader[all]"      # both
```

micro-reader needs Python 3.11 or later. The native decoders come in the separate `micro-reader-codecs` package, which is installed automatically.
