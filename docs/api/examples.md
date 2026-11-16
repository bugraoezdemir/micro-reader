# Examples

These recipes combine features from the user guide into practical workflows. Each example is self-contained and can be adapted to your data. Unless noted otherwise, they use micro-reader, NumPy, and Python's standard library.

[TOC]

---

## Inventory a folder

This recipe inventories the images under a folder, recording their format, shape, pixel size, and channel names. Grouping files first ensures that plate members and multi-file images are handled as one image set rather than as separate images.

```python
import csv
import warnings
import micro_reader


def inventory(folder, out_csv):
    g = micro_reader.group_files(folder)
    paths = [grp.main for grp in g.groups if grp.main] + g.singles
    rows = []

    for path in paths:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            try:
                with micro_reader.open(path) as f:
                    for im in f.images:
                        meta = im.metadata
                        px = meta.pixel_sizes.get("x")
                        rows.append({
                            "file": f.path,
                            "format": f.format,
                            "index": im.index,
                            "axes": im.axes,
                            "shape": im.shape,
                            "dtype": str(im.dtype),
                            "pixel_size_um": (
                                round(px.to("micrometer").value, 4) if px else ""
                            ),
                            "channels": "; ".join(
                                ch.name or "?" for ch in meta.channels
                            ),
                            "levels": len(im.resolutions),
                            "warnings": " | ".join(
                                str(w.message) for w in caught
                            ),
                        })
            except micro_reader.UnsupportedFormatError:
                continue  # not an image

    with open(out_csv, "w", newline="") as fh:
        writer = csv.DictWriter(
            fh, fieldnames=list(rows[0]) if rows else ["file"]
        )
        writer.writeheader()
        writer.writerows(rows)

    return rows
```

Incomplete image sets, whose main file is missing, appear in `g.groups` with `main=None`. Report these separately if you need to track missing files.

---

## Maximum projection of a large stack

This computes a maximum-intensity projection along `z`. It reads one band of rows from one plane at a time, so it works with images of any size without loading the entire stack into memory.

```python
import numpy as np


def max_projection(image, c=0, t=0, band_rows=1024):
    im = image.as_axes("canonical", samples="channels")
    _, _, nz, ny, nx = im.shape
    out = np.empty((ny, nx), dtype=im.dtype)

    # Align bands with storage blocks along y when their size is known.
    if im.read_unit is not None:
        band_rows = max(
            band_rows // im.read_unit[3], 1
        ) * im.read_unit[3]

    for y0 in range(0, ny, band_rows):
        rows = slice(y0, min(y0 + band_rows, ny))

        band = im.read(t=t, c=c, z=0, y=rows)
        for z in range(1, nz):
            np.maximum(
                band, im.read(t=t, c=c, z=z, y=rows), out=band
            )

        out[rows] = band

    return out
```

Reading `z` in the inner loop and `y` bands in the outer loop keeps each read small. For formats that stream large compressed blocks from top to bottom, this order follows the direction in which the data is decoded.

---

## Export to Zarr, block by block

*Requires `zarr` (v3).*

This copies one image into a Zarr array. Reads are aligned to both the file's storage blocks and the output chunks, so each storage block can be decoded once.

```python
import itertools
import numpy as np
import zarr


def to_zarr(image, store, chunks=None, plane_tile=1024):
    im = image.as_axes("canonical", samples="channels")

    if chunks is None:
        chunks = tuple(
            min(n, plane_tile) if a in "yx" else 1
            for a, n in zip(im.axes, im.shape)
        )

    arr = zarr.create_array(
        store,
        shape=im.shape,
        chunks=chunks,
        dtype=im.dtype,
        dimension_names=list(im.axes),
        overwrite=True,
    )

    # Read regions aligned to both storage blocks and output chunks.
    unit = im.read_unit or (1,) * im.ndim
    step = tuple(
        int(np.lcm(u, c)) for u, c in zip(unit, chunks)
    )

    for corner in itertools.product(
        *(range(0, n, s) for n, s in zip(im.shape, step))
    ):
        key = tuple(
            slice(o, min(o + s, n))
            for o, s, n in zip(corner, step, im.shape)
        )
        arr[key] = im[key]

    meta = im.metadata
    arr.attrs["pixel_sizes"] = {
        a: [q.value, q.unit] for a, q in meta.pixel_sizes.items()
    }
    arr.attrs["channels"] = [ch.name for ch in meta.channels]

    return arr
```

For a complete OME-Zarr writer with pyramids, NGFF metadata, and parallel writes, use [EuBI-Bridge](https://github.com/Euro-BioImaging/EuBI-Bridge), which reads images through micro-reader.

---

## Use micro-reader with Dask

*Requires `dask`.*

An `Image` exposes `shape`, `dtype`, `ndim`, and `__getitem__`, so you can wrap it directly with `dask.array.from_array`. When the read unit is known, it is used to choose the chunk size:

```python
import dask.array as da


def as_dask(image, max_plane=2048):
    im = image.as_axes("canonical", samples="channels")

    if im.read_unit is not None:
        chunks = tuple(
            min(u, n) for u, n in zip(im.read_unit, im.shape)
        )
    else:
        chunks = tuple(
            min(n, max_plane) if a in "yx" else 1
            for a, n in zip(im.axes, im.shape)
        )

    return da.from_array(
        im, chunks=chunks, asarray=False, fancy=False, name=False
    )
```

```python
with micro_reader.open("cells.nd2") as f:
    x = as_dask(f.images[0])
    mip = x[0, 0].max(axis=0).compute()  # keep the file open until compute() finishes
```

!!! warning
    A Dask graph holds the `Image`, so compute it while the `File` is still open. Use the threaded scheduler.

    Process-based schedulers, such as `distributed` with processes or `multiprocessing`, need to pickle the image, which is not supported. For these cases, see [Parallel processing with worker processes](#parallel-processing-with-worker-processes).

---

## Use micro-reader with dyna-zarr

*Requires `dyna-zarr`.*

[dyna-zarr](https://pypi.org/project/dyna-zarr/) accepts a `micro_reader.Image` as a source and uses its `read_unit` to determine the chunk grid. Data is read lazily:

```python
from dyna_zarr import from_array, io

with micro_reader.open("cells.czi", tiles="mosaic") as f:
    lazy = from_array(
        f.images[0].as_axes("canonical", samples="channels")
    )
    io.write(lazy, "series0.zarr", zarr_format=3)
```

---

## Stitch tiles using their positions

`open(..., tiles="mosaic")` places the tiles of CZI, Leica, FluoView tiling sets and BigDataViewer datasets on a common canvas, without registration or blending.

For other tiled formats, or when you want to control how overlaps are handled, you can place tiles yourself using `metadata.position`. The position gives the centre of each tile's first pixel. The offset between two tiles is therefore the difference between their positions divided by the pixel size.

The tiles must have the same pixel size and use a common coordinate system.

```python
import numpy as np


def stitch(tiles, c=0, z=0, t=0):
    """Paste tiles (one tczyx image each) onto a canvas. The last tile wins."""
    px = tiles[0].metadata.pixel_sizes["x"].to("micrometer").value
    py = tiles[0].metadata.pixel_sizes["y"].to("micrometer").value

    # Use the first tile as the position reference.
    ref = tiles[0].metadata.position
    ref_y = ref["y"].to("micrometer").value
    ref_x = ref["x"].to("micrometer").value

    offsets = []
    for im in tiles:
        pos = im.metadata.position
        if "x" not in pos or "y" not in pos:
            raise ValueError(f"{im.name}: no stage position")

        offsets.append((
            round((pos["y"].to("micrometer").value - ref_y) / py),
            round((pos["x"].to("micrometer").value - ref_x) / px),
        ))

    y0 = min(o[0] for o in offsets)
    x0 = min(o[1] for o in offsets)

    height = max(
        o[0] - y0 + im.shape[im.axes.index("y")]
        for o, im in zip(offsets, tiles)
    )
    width = max(
        o[1] - x0 + im.shape[im.axes.index("x")]
        for o, im in zip(offsets, tiles)
    )

    canvas = np.zeros((height, width), dtype=tiles[0].dtype)

    for (oy, ox), im in zip(offsets, tiles):
        plane = im.read(t=t, c=c, z=z)
        canvas[
            oy - y0: oy - y0 + plane.shape[0],
            ox - x0: ox - x0 + plane.shape[1],
        ] = plane

    return canvas
```

```python
with micro_reader.open(
    "MouseBrain_41Slices_2x2Tiles_3Channels_2Illuminations_1Angle.czi"
) as f:
    side0 = sorted(
        f.select(illumination=0),
        key=lambda im: im.index["tile"],
    )
    mosaic = stitch(side0, c=0, z=20)
    mosaic.shape  # (1850, 2774): the same bounding box as tiles="mosaic"
```

Stage axes can be inverted relative to image axes. micro-reader normalises each format's convention so that positions increase along image `x` and `y`. This is why the position differences can be used directly.

---

## Fuse the illumination sides of a light-sheet scan

Light-sheet files can store each illumination side as a separate image. This function averages the sides plane by plane:

```python
import numpy as np


def fuse_sides(f, **where):
    sides = sorted(
        f.select(**where),
        key=lambda im: im.index["illumination"],
    )
    ref = sides[0]
    nz = ref.shape[ref.axes.index("z")]
    nc = ref.shape[ref.axes.index("c")]

    out = np.empty((nc, nz) + ref.shape[-2:], dtype=np.float32)

    for c in range(nc):
        for z in range(nz):
            planes = [
                s.read(t=0, c=c, z=z).astype(np.float32)
                for s in sides
            ]
            out[c, z] = np.mean(planes, axis=0)

    return out


with micro_reader.open(
    "MouseBrain_41Slices_1Tile_3Channel_2Illuminations_2Angles.czi"
) as f:
    view0 = fuse_sides(f, view=0)  # (3, 41, 1000, 1500), float32
```

---

## Emission spectrum of a lambda scan

A spectral (lambda) scan is one image with its bands along `c`. Each channel's `emission_wavelength` gives the wavelength of its band (see [Lambda scans](guide/axes.md#lambda-scans)).

This function returns the mean spectrum for a selected region:

```python
import numpy as np


def spectrum(image, y=slice(None), x=slice(None), t=0, z=0):
    wavelengths = [
        ch.emission_wavelength.to("nanometer").value
        for ch in image.metadata.channels
    ]
    means = [
        float(image.read(t=t, c=c, z=z, y=y, x=x).mean())
        for c in range(len(wavelengths))
    ]

    return np.array(wavelengths), np.array(means)


with micro_reader.open("DAPI-mCherry_3T_4Z_5Lambda.oir") as f:
    nm, intensity = spectrum(
        f.images[0], y=slice(200, 300), x=slice(200, 300)
    )
    # nm -> [420. 470. 520. 570. 620.]
```

---

## Per-well measurements on a plate

This function builds a table of mean intensity per channel for every field on a plate:

```python
import numpy as np


def plate_table(f):
    rows = []

    for plate in f.plates:
        for well in plate.wells:
            for field in well.fields:
                for im in field.images:  # all tiles and acquisitions
                    im = im.as_axes("canonical")
                    names = [ch.name for ch in im.metadata.channels]

                    for c in range(im.shape[1]):
                        data = im.read(t=0, c=c)
                        rows.append({
                            "plate": plate.name,
                            "well": well.name,
                            "field": field.index,
                            "index": im.index,
                            "channel": names[c] if c < len(names) else c,
                            "mean": float(data.mean()),
                            "max": int(data.max()),
                        })

    return rows


with micro_reader.open("Measurement1/Images/Index.idx.xml") as f:
    f.check_members()  # fail early if a plane file does not match the index
    table = plate_table(f)
```

With pandas, `pd.DataFrame(table).pivot_table(index="well", columns="channel", values="mean")` gives a well-by-channel overview.

---

## Select wells by name and read them

On a plate, `select()` accepts well names, rows, columns, and lists or ranges of positions.

This example reads the first field of each well in rows B and C, columns 2 to 4, and stores the results in a dictionary:

```python
with micro_reader.open("Measurement1/Images/Index.idx.xml") as f:
    images = f.select(
        plate=0, acquisition=0,
        row=["B", "C"], column=range(1, 4), field=0,
    )  # columns 2, 3, and 4 of the first plate

    plate = f.plates[0]
    by_well = {}
    for im in images:
        well = plate.wells[im.index["well"]]
        by_well[well.name] = im.read(t=0, c=0)

    # {'B2': array(...), 'B3': ..., 'C2': ...}
```

`column=range(1, 4)` uses zero-based positions, so it selects columns `"2"`, `"3"`, and `"4"`. You can also specify these by name with `column=["2", "3", "4"]`.

`plate=0` and `acquisition=0` keep the selection to one plate and one imaging run. Without them, a file with several plates or acquisitions would give several images per well name, and the dictionary would keep only the last of them.

Use one addressing scheme at a time. For example, `well="B2"` cannot be combined with `row=` or `column=`.

---

## A tile server for a viewer

A viewer can request 256 × 256 tiles at different zoom levels. This class uses the stored pyramid when available and increases the cache limit so that revisiting previously viewed regions can reuse decoded blocks.

```python
import numpy as np
import micro_reader

micro_reader.set_cache_limit("2GB")


class TileSource:
    def __init__(self, path, tile=256, **open_options):
        self.f = micro_reader.open(
            path, axes="canonical", samples="channels", **open_options
        )
        self.tile = tile

    def levels(self, image=0):
        return [
            lv.shape[-2:] for lv in self.f.images[image].resolutions
        ]

    def get(self, level, ty, tx, c=0, z=0, t=0, image=0):
        im = self.f.images[image].resolutions[level]
        ny, nx = im.shape[-2:]

        y0, x0 = ty * self.tile, tx * self.tile
        if y0 >= ny or x0 >= nx:
            return np.zeros((self.tile, self.tile), im.dtype)

        block = im.read(
            t=t, c=c, z=z,
            y=slice(y0, y0 + self.tile),
            x=slice(x0, x0 + self.tile),
        )

        if block.shape != (self.tile, self.tile):
            block = np.pad(
                block,
                [
                    (0, self.tile - block.shape[0]),
                    (0, self.tile - block.shape[1]),
                ],
            )

        return block

    def close(self):
        self.f.close()
```

---

## Create a thumbnail from any file

*Requires Pillow.*

This function selects the smallest stored resolution level that is still large enough for the requested thumbnail size. It then scales the intensities to 8 bits:

```python
import numpy as np
from PIL import Image as PILImage


def thumbnail(path, size=512, c=0):
    with micro_reader.open(
        path, axes="canonical", samples="channels"
    ) as f:
        im = f.images[0]
        level = next(
            (
                lv for lv in reversed(im.resolutions)
                if max(lv.shape[-2:]) >= size
            ),
            im,
        )

        nz = level.shape[2]
        plane = level.read(
            t=0, c=c, z=nz // 2
        ).astype(np.float32)

    lo, hi = np.percentile(plane, (0.5, 99.5))
    plane = np.clip((plane - lo) / max(hi - lo, 1e-6), 0, 1)

    pic = PILImage.fromarray((plane * 255).astype(np.uint8))
    pic.thumbnail((size, size))

    return pic
```

---

## Parallel processing with worker processes

`File` and `Image` objects cannot be pickled, so each worker must open the file itself. An initializer opens it once per worker. Tasks then need to pass only an image index and a plane number.

```python
from concurrent.futures import ProcessPoolExecutor
import os
import micro_reader

_file = None


def _init(path, memory):
    global _file
    micro_reader.set_memory_limit(memory)  # settings are per process
    micro_reader.set_decode_threads(1)     # the process pool handles parallelism
    _file = micro_reader.open(path)


def _plane_stats(index, z):
    im = _file.select(**index)[0].as_axes("canonical")
    plane = im.read(t=0, c=0, z=z)
    return index, z, float(plane.mean())


def parallel_stats(path, workers=None):
    workers = workers or os.cpu_count()

    with micro_reader.open(path) as f:
        jobs = [
            (im.index, z)
            for im in f.images
            for z in range(im.as_axes("canonical").shape[2])
        ]

    if not jobs:
        return []

    with ProcessPoolExecutor(
        workers,
        initializer=_init,
        initargs=(path, "1GB"),
    ) as ex:
        return list(ex.map(_plane_stats, *zip(*jobs)))


if __name__ == "__main__":
    print(parallel_stats("multipoint.nd2")[:3])
```

On Windows and macOS, worker processes are started using *spawn*. Keep the call under `if __name__ == "__main__":` to avoid starting new workers when the module is imported.

---

## Read a plain TIFF stack whose structure you know

A multi-page TIFF without metadata is exposed as 80 separate single-plane images, indexed by `sequence` from 0 to 79.

Suppose you know that the pages represent 20 z-planes and 4 channels, with the channels stored consecutively for each z-plane. You can rename `sequence` to `z` and reshape the result:

```python
with micro_reader.open(
    "stack.tif", rename={"sequence": "z"}
) as f:
    im = f.images[0]          # t c z y x: (1, 1, 80, ny, nx)
    data = im.read(t=0, c=0)  # z y x: (80, ny, nx)

    stack = data.reshape(
        20, 4, *data.shape[1:]
    ).transpose(1, 0, 2, 3)  # c z y x: (4, 20, ny, nx)
```

For files too large to read at once, read a range of `z` at a time. Map page number `k` to `(z, c)` using `divmod(k, 4)`.

---

## Read a CZI region using file coordinates

Coordinates from the CZI XML or ZEN are expressed in the file's own pixel coordinate system. `read_region()` accepts these coordinates directly:

```python
with micro_reader.open("slide.czi", tiles="mosaic") as f:
    for im in f.images:
        y0, x0 = im.origin
        ny, nx = im.shape[-2:]
        print(
            im.index,
            "covers x", x0, "-", x0 + nx,
            "y", y0, "-", y0 + ny,
        )

    roi = f.images[0].read_region(
        y=174800, x=134800, height=512, width=512, c=0
    )
```

`f.canvas()` covers all series at once, using the same coordinate system.