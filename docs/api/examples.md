# Examples

Complete recipes that combine the pieces from the user guide. Each one is a function you can copy. They use only micro-reader and NumPy unless the heading says otherwise.

[TOC]

---

## Inventory a folder

List every image under a folder, with its format, shape, pixel size and channel names, without opening any file twice. Plates and multi-file images count once.

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
                            "pixel_size_um": round(px.to("micrometer").value, 4) if px else "",
                            "channels": "; ".join(ch.name or "?" for ch in meta.channels),
                            "levels": len(im.resolutions),
                            "warnings": " | ".join(str(w.message) for w in caught),
                        })
            except micro_reader.UnsupportedFormatError:
                continue                                  # not an image
    with open(out_csv, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]) if rows else ["file"])
        writer.writeheader()
        writer.writerows(rows)
    return rows
```

Incomplete images, whose main file is missing, are in `g.groups` with `main=None`. Report them separately if you need to.

---

## Maximum projection of a large stack

A maximum-intensity projection over `z`, with flat memory use: only one band of rows of one plane is in memory at a time. It works for any image size.

```python
import numpy as np


def max_projection(image, c=0, t=0, band_rows=1024):
    im = image.as_axes("canonical", samples="channels")
    _, _, nz, ny, nx = im.shape
    out = np.zeros((ny, nx), dtype=im.dtype)
    # follow the storage blocks along y when they are known
    if im.read_unit is not None:
        band_rows = max(band_rows // im.read_unit[3], 1) * im.read_unit[3]
    for y0 in range(0, ny, band_rows):
        rows = slice(y0, min(y0 + band_rows, ny))
        for z in range(nz):
            np.maximum(out[rows], im.read(t=t, c=c, z=z, y=rows), out=out[rows])
    return out
```

Reading `z` in the inner loop and `y` bands in the outer loop keeps each read small. For formats that stream large blocks top to bottom, the `y` order matches the order in which the data is stored.

---

## Export to Zarr, block by block

*Needs `zarr` (v3).*

Copy one image into a Zarr array, reading in pieces aligned to the file's storage blocks, so every block is decoded once.

```python
import itertools
import numpy as np
import zarr


def to_zarr(image, store, chunks=None, plane_tile=1024):
    im = image.as_axes("canonical", samples="channels")
    if chunks is None:
        chunks = tuple(min(n, plane_tile) if a in "yx" else 1 for a, n in zip(im.axes, im.shape))
    arr = zarr.create_array(store, shape=im.shape, chunks=chunks, dtype=im.dtype,
                            dimension_names=list(im.axes), overwrite=True)
    # read in steps that are whole storage blocks and whole output chunks
    unit = im.read_unit or (1,) * im.ndim
    step = tuple(int(np.lcm(u, c)) for u, c in zip(unit, chunks))
    for corner in itertools.product(*(range(0, n, s) for n, s in zip(im.shape, step))):
        key = tuple(slice(o, min(o + s, n)) for o, s, n in zip(corner, step, im.shape))
        arr[key] = im[key]
    meta = im.metadata
    arr.attrs["pixel_sizes"] = {a: [q.value, q.unit] for a, q in meta.pixel_sizes.items()}
    arr.attrs["channels"] = [ch.name for ch in meta.channels]
    return arr
```

For a complete OME-Zarr writer, with pyramids, NGFF metadata and parallel writes, use [eubi-bridge](https://github.com/Euro-BioImaging/EuBI-Bridge), which reads through micro-reader.

---

## Use micro-reader with dask

*Needs `dask`.*

An `Image` has `shape`, `dtype`, `ndim` and `__getitem__`, so `dask.array.from_array` can wrap it directly. Use the read unit for chunks where it is small enough:

```python
import dask.array as da


def as_dask(image, max_plane=2048):
    im = image.as_axes("canonical", samples="channels")
    if im.read_unit is not None:
        chunks = tuple(min(u, n) for u, n in zip(im.read_unit, im.shape))
    else:
        chunks = tuple(min(n, max_plane) if a in "yx" else 1 for a, n in zip(im.axes, im.shape))
    return da.from_array(im, chunks=chunks, asarray=False, fancy=False, name=False)
```

```python
with micro_reader.open("cells.nd2") as f:
    x = as_dask(f.images[0])
    mip = x[0, 0].max(axis=0).compute()     # keep the file open until compute() ends
```

!!! warning
    A dask graph holds the `Image`. Compute it while the `File` is open, and use the threaded scheduler. Process-based schedulers (`distributed` with processes, `multiprocessing`) need to pickle the image, which is not possible. For those, see [worker processes](#parallel-processing-with-worker-processes).

---

## Use micro-reader with dyna-zarr

*Needs `dyna-zarr`.*

[dyna-zarr](https://pypi.org/project/dyna-zarr/) accepts a `micro_reader.Image` as a source and takes its chunk grid from `read_unit`. Reads are pulled lazily and stay memory-bounded:

```python
from dyna_zarr import from_array, io

with micro_reader.open("cells.czi", tiles="mosaic") as f:
    lazy = from_array(f.images[0].as_axes("canonical", samples="channels"))
    io.write(lazy, "series0.zarr", zarr_format=3)
```

---

## Stitch tiles yourself from their positions

`open(..., tiles="mosaic")` places the tiles of a CZI, Leica or FluoView scan on one canvas, with nothing registered or blended. For other tiled files, or to control how overlaps are blended, place the tiles yourself with `metadata.position`. The position is the centre of each tile's first pixel, so the offset between two tiles is the difference of their positions divided by the pixel size.

```python
import numpy as np


def stitch(tiles, c=0, z=0, t=0):
    """Paste tiles (one tczyx image each, same pixel size) onto one canvas, last one wins."""
    px = tiles[0].metadata.pixel_sizes["x"].to("micrometer").value
    py = tiles[0].metadata.pixel_sizes["y"].to("micrometer").value
    offsets = []
    for im in tiles:
        pos = im.metadata.position
        if "x" not in pos or "y" not in pos:
            raise ValueError(f"{im.name}: no stage position")
        offsets.append((round(pos["y"].to("micrometer").value / py),
                        round(pos["x"].to("micrometer").value / px)))
    y0 = min(o[0] for o in offsets)
    x0 = min(o[1] for o in offsets)
    height = max(o[0] - y0 + im.shape[im.axes.index("y")] for o, im in zip(offsets, tiles))
    width = max(o[1] - x0 + im.shape[im.axes.index("x")] for o, im in zip(offsets, tiles))
    canvas = np.zeros((height, width), dtype=tiles[0].dtype)
    for (oy, ox), im in zip(offsets, tiles):
        plane = im.read(t=t, c=c, z=z)
        canvas[oy - y0: oy - y0 + plane.shape[0], ox - x0: ox - x0 + plane.shape[1]] = plane
    return canvas
```

```python
with micro_reader.open("MouseBrain_41Slices_2x2Tiles_3Channels_2Illuminations_1Angle.czi") as f:
    side0 = sorted(f.select(illumination=0), key=lambda im: im.index["tile"])
    mosaic = stitch(side0, c=0, z=20)
    mosaic.shape          # (1850, 2774): the same box as tiles="mosaic"
```

Stage axes can be inverted relative to the image. micro-reader already converts every format's convention so that positions grow with `x` and `y` in the image, which is why plain subtraction works.

---

## Fuse the illumination sides of a light-sheet scan

Light-sheet files store each side as its own image. Averaging the sides, plane by plane:

```python
import numpy as np


def fuse_sides(f, **where):
    sides = sorted(f.select(**where), key=lambda im: im.index["illumination"])
    ref = sides[0]
    nz = ref.shape[ref.axes.index("z")]
    nc = ref.shape[ref.axes.index("c")]
    out = np.empty((nc, nz) + ref.shape[-2:], dtype=np.float32)
    for c in range(nc):
        for z in range(nz):
            planes = [s.read(t=0, c=c, z=z).astype(np.float32) for s in sides]
            out[c, z] = np.mean(planes, axis=0)
    return out


with micro_reader.open("MouseBrain_41Slices_1Tile_3Channel_2Illuminations_2Angles.czi") as f:
    view0 = fuse_sides(f, view=0)      # (3, 41, 1000, 1500) float32
```

---

## Emission spectrum of a lambda scan

A spectral (lambda) scan gives one image per band, indexed by `emission`, with the band's wavelength in `metadata.index_values`. Here is the mean spectrum of a region:

```python
import numpy as np


def spectrum(f, y=slice(None), x=slice(None), t=0, z=0):
    bands = sorted(f.select(series=0), key=lambda im: im.index["emission"])
    wavelengths = [im.metadata.index_values["emission"].to("nanometer").value for im in bands]
    means = [float(im.read(t=t, c=0, z=z, y=y, x=x).mean()) for im in bands]
    return np.array(wavelengths), np.array(means)


with micro_reader.open("DAPI-mCherry_3T_4Z_5Lambda.oir") as f:
    nm, intensity = spectrum(f, y=slice(200, 300), x=slice(200, 300))
    # nm -> [420. 470. 520. 570. 620.]
```

---

## Per-well measurements on a plate

Mean intensity per channel for every field of a plate, as a table:

```python
import numpy as np


def plate_table(f):
    rows = []
    for plate in f.plates:
        for well in plate.wells:
            for field in well.fields:
                for im in field.images:                 # all tiles / acquisitions
                    im = im.as_axes("canonical")
                    names = [ch.name for ch in im.metadata.channels]
                    for c in range(im.shape[1]):
                        data = im.read(t=0, c=c)
                        rows.append({
                            "plate": plate.name, "well": well.name, "field": field.index,
                            "index": im.index, "channel": names[c] if c < len(names) else c,
                            "mean": float(data.mean()), "max": int(data.max()),
                        })
    return rows


with micro_reader.open("Measurement1/Images/Index.idx.xml") as f:
    f.check_members()                     # fail early on a bad plane file
    table = plate_table(f)
```

With pandas, `pd.DataFrame(table).pivot_table(index="well", columns="channel", values="mean")` gives a well × channel overview.

---

## Pick wells by name and read them

On a plate, `select` takes well names, rows and columns, and lists or ranges of them. Here, the first field of every well in rows B and C, columns 2 to 4, as a dictionary:

```python
with micro_reader.open("Measurement1/Images/Index.idx.xml") as f:
    images = f.select(row=["B", "C"], column=range(1, 4), field=0)     # columns 2, 3, 4
    by_well = {}
    for im in images:
        plate = f.plates[im.index["plate"]]
        by_well[plate.wells[im.index["well"]].name] = im.read(t=0, c=0)
    # {'B2': array(...), 'B3': ..., 'C2': ...}
```

`column=range(1, 4)` counts from 0, so it is columns "2", "3" and "4"; `column=["2", "3", "4"]` names them. Give one address or the other: `well="B2"` does not go with `row=` or `column=`.

---

## A tile server for a viewer

A viewer asks for 256 × 256 tiles at a zoom level. Use the stored pyramid when there is one, and raise the cache so that panning back costs nothing:

```python
import numpy as np
import micro_reader

micro_reader.set_cache_limit("2GB")


class TileSource:
    def __init__(self, path, tile=256, **open_options):
        self.f = micro_reader.open(path, axes="canonical", samples="channels", **open_options)
        self.tile = tile

    def levels(self, image=0):
        return [lv.shape[-2:] for lv in self.f.images[image].resolutions]

    def get(self, level, ty, tx, c=0, z=0, t=0, image=0):
        im = self.f.images[image].resolutions[level]
        ny, nx = im.shape[-2:]
        y0, x0 = ty * self.tile, tx * self.tile
        if y0 >= ny or x0 >= nx:
            return np.zeros((self.tile, self.tile), im.dtype)
        block = im.read(t=t, c=c, z=z, y=slice(y0, y0 + self.tile), x=slice(x0, x0 + self.tile))
        if block.shape != (self.tile, self.tile):                 # pad edge tiles
            block = np.pad(block, [(0, self.tile - block.shape[0]), (0, self.tile - block.shape[1])])
        return block

    def close(self):
        self.f.close()
```

---

## A thumbnail from any file

*Needs Pillow.*

Pick the smallest level that is still large enough, then scale the intensities to 8 bits:

```python
import numpy as np
from PIL import Image as PILImage


def thumbnail(path, size=512, c=0):
    with micro_reader.open(path, axes="canonical", samples="channels") as f:
        im = f.images[0]
        level = next((lv for lv in reversed(im.resolutions)
                      if max(lv.shape[-2:]) >= size), im)
        nz = level.shape[2]
        plane = level.read(t=0, c=c, z=nz // 2).astype(np.float32)
    lo, hi = np.percentile(plane, (0.5, 99.5))
    plane = np.clip((plane - lo) / max(hi - lo, 1e-6), 0, 1)
    pic = PILImage.fromarray((plane * 255).astype(np.uint8))
    pic.thumbnail((size, size))
    return pic
```

---

## Parallel processing with worker processes

Files cannot be pickled, so each worker opens the file itself. An initializer opens it once per worker. The tasks then only carry an index and a plane number.

```python
from concurrent.futures import ProcessPoolExecutor
import os
import micro_reader

_file = None


def _init(path, memory):
    global _file
    micro_reader.set_memory_limit(memory)      # settings are per process
    micro_reader.set_decode_threads(1)         # the pool already uses every core
    _file = micro_reader.open(path)


def _plane_stats(index, z):
    im = _file.select(**index)[0].as_axes("canonical")
    plane = im.read(t=0, c=0, z=z)
    return index, z, float(plane.mean())


def parallel_stats(path, workers=None):
    workers = workers or os.cpu_count()
    with micro_reader.open(path) as f:
        jobs = [(im.index, z) for im in f.images
                for z in range(im.as_axes("canonical").shape[2])]
    with ProcessPoolExecutor(workers, initializer=_init, initargs=(path, "1GB")) as ex:
        return list(ex.map(_plane_stats, *zip(*jobs)))


if __name__ == "__main__":
    print(parallel_stats("multipoint.nd2")[:3])
```

On Windows and macOS, worker processes are started with *spawn*: keep the call under `if __name__ == "__main__":`.

---

## Handle a plain TIFF stack whose meaning you know

A multi-page TIFF without metadata gives 80 `yx` images (`sequence` 0 to 79). If you know the pages are, for example, 20 z-planes × 4 channels in that order, read it as a z-stack and reshape:

```python
with micro_reader.open("stack.tif", rename={"sequence": "z"}) as f:
    im = f.images[0]                         # zyx (80, ny, nx)
    data = im.read()
    stack = data.reshape(20, 4, *data.shape[1:]).transpose(1, 0, 2, 3)    # c z y x
```

For files too large to read whole, read a z range at a time and map plane `k` to `(z, c) = divmod(k, 4)`.

---

## Read a CZI region by its file coordinates

Coordinates from the CZI's XML, or from ZEN, are in the file's own pixel coordinates. `read_region` accepts them directly:

```python
with micro_reader.open("slide.czi", tiles="mosaic") as f:
    for im in f.images:
        y0, x0 = im.origin
        ny, nx = im.shape[-2:]
        print(im.index, "covers x", x0, "-", x0 + nx, "y", y0, "-", y0 + ny)

    roi = f.images[0].read_region(y=174800, x=134800, height=512, width=512, c=0)
```

`f.canvas()` covers all series at once, with the same coordinates.
