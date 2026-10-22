# Changelog

## Unreleased: coordinate systems and the file's tree

### Added

- **Coordinate systems.** A position now says which space it is in. `File.coordinate_systems` lists the shared coordinate systems of a file (`CoordinateSystem(id, name, axes, estimated)`), declared when the file opens, without reading any image's metadata. `Metadata.coordinate_system` is the id of the system `metadata.position` is in: set exactly when the position is not empty, None for an image without a position. `Metadata.unplaced()` drops both. Ids are made of `[a-zA-Z0-9_-]` only: other characters become `_`, and an id never holds `.` or `--`.
- **Systems per format.**

  | Format | System |
  | --- | --- |
  | CZI | `stage`, one per file. Label and slide-preview pictures have no position. |
  | ND2 | `stage`, for files calibrated with a camera-to-stage matrix |
  | OME-TIFF, OME companion sets | `stage` |
  | Imaris | `stage` |
  | Leica LIF | `stage` when all placed elements come from one session (one microscope, one LAS X run, no break over 2 hours); a file of several sessions gets one system per session, `stage-r0`, `stage-r1`, ... in series order; an element that records too little to tie it to a session gets `stage-s<series>` |
  | Harmony, CellVoyager plates | one per well: `well-B2`; `well-p<plate>-B2` with several plates. Positions are relative to the well centre. |
  | SVS, SCN, QPTIFF slides | `slide` |
  | FluoView tiling sets (`.omp2info`) | `stage`, all groups |
  | BigDataViewer / BigStitcher | `global`, for images whose view transforms are plain (diagonal, positive) |

  Every other format gives no position and declares no system.
- **Transformations.** `File.coordinate_transformations` holds the edges that are not plain positions (`Transformation(type, input, output, params, name)`). BigDataViewer fills it: an image whose nominal view transform turns, shears or mirrors it gets an affine edge into `global` instead of a position, and a registration (stitching, ICP, bead registration) becomes an edge into `global-registered`, a system marked `estimated=True`.
- **The file's tree.** `File.collection` is a `Collection(name, kind, index, children)`: plate > well > field, and series outside plates. A field or series holding one image is that image; tiles, views and resolution levels make no collection of their own. It is built from the current images, so it follows `open(axes=...)`, and it reads no image metadata. The root is named after the file without its extension (`cells.ome.tif` and `cells.ome.zarr` give `cells`; a folder keeps its name), a plate without a name is `plate<n>`; ome_zarr_pyramid builds the same tree.
- `Axis`, `CoordinateSystem`, `Transformation`, `Collection` and `InvariantError` are exported from `micro_reader`.

### Fixes

- QPTIFF overview and label pictures were placed by their corner, about 31 mm off: their position tags are centres, and are now read as such.
- A BigDataViewer image is placed only when all its views (every time point and channel) share one placement; before, only the first view was read.
- A Leica SCN macro picture is placed on the slide, in the same frame as the scan.
- An ND2 position steps along y by the y pixel size (it used x's; the same value in every file seen so far).

### Known limits

- Leica positions are unchanged and can be tens of micrometres off (field centre versus first pixel); a check is planned.
- The FluoView tiling grid step comes from one public sample and is 2-6 px short per tile on it.

## 0.1.0 (2026-10-21)

### Breaking changes

- **The image index `scene` is now called `series`.** Every image index key, `File.select(series=...)`, `File.index_range`, `File.outline()` (`series 0: ...`), image names (`file.czi series 0 tile 3`), `Image.__repr__`, `as_axes` and every backend that builds an index (BDV, OME companion, CZI, HDF5, ilastik, Imaris, Leica, MetaMorph, Micro-Manager, ND2, NDPIS, OIR, omp2info, TIFF, VSI) use the new name. Plates are unchanged: their images are indexed by `plate`, `acquisition`, `well` and `field`, with no `series`. The old name is never reused: storing it, or reading it from an image's index, raises (see "Guards").
- `File.select(scene=...)` raises a `KeyError` that names `series`.
- A damaged CZI's `File.damage` keys `scenes_affected` / `scenes_unreadable` are now `series_affected` / `series_unreadable`, and the damaged-file warning says "series".
- `micro_reader.INDEX_NAMES` and the plate containment order (`_plate.CONTAINMENT`) say `series`.
- File.collection (the coordinate systems of plan 9.7) is not in this release.

### Fixes

- The CZI label and slide preview pictures (attachments listed as further series) no longer get a stage position (`metadata.position` is `{}`). Their inner CZIs' stage tags are in a frame not verified against the slide's: some were 46-105 mm away from the slide. They get one back once that frame is verified (plan 4.18).

### Guards

- **The index guard.** `Image.index` is an `ImageIndex`, a `dict` subclass. Storing the key `scene` raises a `KeyError` (`index["scene"] = ...`, `update`, `|=`, `setdefault`, `ImageIndex(...)`, `ImageIndex.fromkeys`), and so do `index["scene"]`, `index.get("scene")`, `"scene" in index`, `index.pop("scene")` and `del index["scene"]`, which on a plain dict would answer quietly. The message is "the image index 'scene' is now called 'series' (micro-reader 0.1.0); use 'series'". `copy()`, `|` (either side), `copy.deepcopy` and pickling keep the type, so indices sent to worker processes stay guarded. Copies made with `dict(index)` or `{**index}`, `keys()`, `items()`, JSON and `==` are not guarded: packages that read micro-reader rename in the same release instead.
  - Cost: none measurable. `File.index_range` and `File.select` read the indices through `dict`'s own methods, and are faster than in 0.0.4 (100,000 images: `index_range` 20 ms against 64 ms, `select` 9 ms against 25 ms); building an `ImageIndex` costs about 0.3 microseconds per image.
  - Removal: when no supported eubi-bridge release needs it and at least three micro-reader minor releases have passed (release policy rule 4), checked by the release version matrix; not before.

- **The import guard against old eubi** (plan 8.28). eubi-bridge 0.1.3 to 0.2.x and eubi-bridge-lite below 0.3.0 still read `scene`: beside micro-reader 0.1.0, full eubi-bridge would switch silently to its older readers, and eubi-bridge-lite would fail every file with a wrong message. So `import micro_reader` refuses to load next to them. It raises an `ImportError` that names the installed eubi version and the two ways out: upgrade eubi (`pip install -U 'eubi-bridge>=0.3.0'`, or the same for `eubi-bridge-lite`), or keep that eubi with the micro-reader it was built for (`pip install 'micro-reader<0.1'`). eubi-bridge 0.1.2 and older never use micro-reader and are not refused; development builds of 0.3.0 (`0.3.0.dev0`, `0.3.0rc1`) are not refused either. The check reads the versions of the installed packages (`importlib.metadata`); it never imports eubi, and a refused import loads nothing else. An editable install of eubi is checked like any other: its recorded version can be out of date, so the message then says to reinstall it. An eubi whose version cannot be read (not a PEP 440 version, or none recorded) is not checked: the import goes on with one warning that names the version, or says that none is recorded.
  - Escape: the environment variable `MICRO_READER_SKIP_EUBI_CHECK=1` skips the check, with one warning per process. Only the value `1` skips it; any other value keeps the check. It is meant for work on an eubi checkout; the release version matrix strips it from every run.
  - Cost: about 1 ms per import when no eubi is installed (a listing of the folders on `sys.path`); with an eubi installed, the version lookup adds about 35 ms (measured: 140 ms -> 175 ms for the whole `import micro_reader`), next to eubi's own much longer import.
  - Removal: when every eubi-bridge / eubi-bridge-lite line below 0.3.0 has left the support window, checked by the release version matrix; not before.

### Supported consumers

- eubi-bridge and eubi-bridge-lite 0.3.0 (the first releases that read `series`).
- Older eubi-bridge / eubi-bridge-lite releases read `scene`: they keep micro-reader below 0.1 (`pip install 'micro-reader<0.1'`; the pin releases 0.1.7 and 0.2.1 require it).
