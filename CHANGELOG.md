# Changelog

## 0.3.0 (unreleased): select by well name, row and column

A minor release: `select` takes new arguments, and refuses a plate, well, field or
acquisition object it used to match nothing with. Ships together with ome_zarr_pyramid
0.5.0, whose `PyramidCollection.select` takes the same arguments and gives the same
images. eubi-bridge 0.3.2 and earlier bound micro-reader below 0.3 and keep 0.2.x;
eubi-bridge calls `select` with single index values only. Two changes break: `Plate.rows` /
`Plate.columns` (the names) are `Plate.row_names` / `Plate.column_names` now, and `open`
turns RGB samples into channels by default (see Changed); eubi-bridge 0.3.3 adapts to the
second before widening its bound to `<0.4`.

### New

- `File.select(well="C3")`: a well by name, in each plate (`well=` still takes a
  well's place in `plates[p].wells`).
- `File.select(row=..., column=...)`: a plate's wells by row and column, as 0-based
  numbers (as `Well.row` / `Well.column`: `column=2` is the column named "3") or names
  (`row="C"`, `column="3"`). They are another address of the well, so they do not go
  with `well=` (TypeError).
- Any index value may be a list, tuple, range or set, meaning any of them:
  `select(illumination=[2, 3])`, `select(row=["B", "C"], column=range(2, 5))`;
  `Field.select` too.
- `File.select` takes each place of a plate by position and by name, a keyword each:
  `plate` / `plate_name`, `acquisition` / `acquisition_name`, `well` / `well_name`, `row` /
  `row_name`, `column` / `column_name` (the columns of ome-zarr-pyramid's table). A name
  given to a position keyword is its name (`well="C3"`, `plate="Plate 1"`); a number given
  to a name keyword raises `TypeError` naming the position keyword.
- `select` and `Field.select` take slices of positions, as ranges: `well=slice(0, 12)`,
  `row=slice(2, None)` (open-ended: row C onward), `column=slice(0, None, 2)`. A slice
  selected nothing, silently; a negative bound or a step below 1 raises `ValueError`, a
  slice on a name keyword `TypeError`.

### New (CQ1, Cellomics, ImageXpress HCS.ai)

- An OME-TIFF whose OME-XML describes a plate and whose planes name their files (Yokogawa
  CQ1's `MeasurementResult.ome.tif`, and its member TIFFs) reads as a plate with
  `plates=True`, through the companion reader.
- Thermo Fisher Cellomics `.C01` plates (backends/cellomics.py): any `.C01` file, the
  folder, the database or a log opens the plate; the layout is guessed (the format is only
  in the Access database), at least 96 wells.
- Molecular Devices ImageXpress HCS.ai exports (`.jdce` + `image_metadata_*.csv`,
  backends/jdce.py): projections, z-stacks and stitched montages, each image set a plate;
  field positions relative to the well's centre.

### Fixed

- A multi-file OME-TIFF whose image starts at IFD 0 of a member file took the metadata of
  the image at IFD 0 of the opened file (CQ1: every field the title image's -- no channel
  names, no pixel sizes): images are matched by file and IFD.
- `group_files` counted a set's opened file among its own members when its path was spelled
  differently (relative): the set then had no group of its own.

### New (scanR, ND2 plates)

- Olympus / Evident scanR plates (`experiment_descriptor.xml` with `data/metadata.ome.xml`),
  which opened as an OME companion without a plate: wells and fields from the data files'
  names, the layout from the descriptor. The descriptor, the folders, the `.dat` files,
  the companion and (with `multifile=True`) any data TIFF open the plate.
- ND2 with `plates=True`: a file whose XY points are named after wells as NIS-Elements
  names them (`A10_0000`, `B02_0003`, `E8.1`) reads as a plate (layout: the smallest
  standard plate holding the wells; fields in acquisition order). Other point names
  (`P1`, `A1`) keep the positions as they are.
- A member that points to a metadata file which is itself a part of a composite (scanR's
  companion) opens the composite.

### New (Columbus)

- PerkinElmer Columbus exports, which were not recognised: `ImageIndex.ColumbusIDX.xml`
  is read as a Harmony index (`File.backend` / `format` `"columbus"`), including planes
  stored as pages of multi-page TIFFs (`URL BufferNo`). `MeasurementIndex.ColumbusIDX.xml`,
  the CSV index and the export's folders open the same plate.

### New (Opera Flex)

- PerkinElmer Opera Flex plates (`.flex`), which were refused: any `.flex` file, or its
  folder, opens the folder's `.flex` files as plates (one per barcode; `multifile=False`:
  the one file). Wells from `WellCoordinate`, fields from `Sublayout`, axes `t c z y x`
  from `Timepoint`, exposure and camera, and `Stack`; pixel sizes, z step, channel names
  and positions relative to the well's centre. A field whose planes do not fill its grid,
  and scaled intensities (`ImageFactor`), are refused. `group_files` gives a Flex folder
  as one group (an index file's members are no longer opened again).

### New (OME companion files)

- OME companion plates: `open(path, plates=True)` reads a companion with `Plate` /
  `Well` / `WellSample` as a plate. `File.plates` describes it and the images in its
  wells are indexed `plate` / `acquisition` / `well` / `field` instead of `series`. A
  well's fields are its WellSamples in `Index` order (per `PlateAcquisition`). The layout
  is the Plate's `Rows` x `Columns`, else the smallest that holds every well listed. Plates
  and wells are those that hold images, and an image in no well keeps `series`. Without
  `plates=True` nothing changes.
- RGB images in a companion (`SamplesPerPixel` 3: one Channel standing for SizeC 3, as
  OME-TIFF writes it) read as `t c z y x s`, c = SizeC / SamplesPerPixel, as an RGB
  OME-TIFF opens; they were refused. A member opened alone (`multifile=False`) too.
  Channels of different SamplesPerPixel in one image stay refused
  (`UnsupportedFormatError`); a member whose pages hold other samples than listed is a
  `CorruptFileError`.
- A companion whose images miss member files gives one `MissingImagesWarning` for the
  set (was one per image), counting files, not planes.

### Changed

- **`open` turns RGB samples into channels by default** (`samples="channels"`, was
  `"keep"`): an RGB image has `c` 3 instead of an `s` axis, and its metadata lists one
  channel per sample, "<name> R" / "G" / "B" ("A"), coloured red / green / blue (white for
  A), wavelengths kept -- the names eubi-bridge gives; a single unnamed channel gives three
  unnamed, coloured ones; other sample counts are numbered and uncoloured. A file listing
  one channel per sample already keeps its own. Only images with samples are wrapped (their
  `.source` is the stored image); others are unchanged. `samples="keep"` gives the old
  behaviour; `Image.as_axes` still keeps `s` by default. eubi-bridge 0.3.3 must read the
  channel names of the stored image (`image.source.metadata`) where it folds samples
  itself, before widening its bound to `<0.4`.

- A `Plate`, `Well`, `Field` or `Acquisition` object given as an index value raises
  `TypeError` (it selected nothing): `select(well=plate.wells[2])` names
  `well=plate.well_index(...)` and `well="C3"`. Row, column and well names on a file
  opened without `plates=True` raise `ValueError` naming it.
- **Breaking:** `Plate.rows` and `Plate.columns` slice the plate (below); the row and
  column names they held are `Plate.row_names` and `Plate.column_names`. Update
  `plate.rows[well.row]` to `plate.row_names[well.row]`; `len(plate.rows)` still works.
  The `Plate(...)` fields are named `row_names` / `column_names` too.

### New (containment, decision C-19)

- Every level knows what it is and what holds it: `.kind` (`"file"`, `"plate"`,
  `"well"`, `"field"`) and `.parent` (a field's well, a well's plate, a plate's file).
- Accessors go down only, by level: a file holds `.plates`, a plate `.wells`, a well
  `.fields`, a field `.images`; another level's accessor (`well.plates`, `f.wells`) raises
  `AttributeError` naming what the object holds and `.parent`.
- `plate.rows[...]` / `plate.columns[...]` give a part of the plate, still a plate: an
  index, a slice (negative bounds from the layout's end), a name or a list of them, in
  either order (`plate.rows[2:5].columns[3:7]`), as `select(row=..., column=...)`.
  Positions are the whole layout's; `well_index` answers with the index the images carry;
  `is_part` says whether a plate is a part.

## 0.2.1 (2026-10-23): plates and composites from more real files

A patch release: what 0.2.0 read keeps its pixels and positions; files it refused or
misread now open. Found on the 2026-10-22 round of new samples (three full plates and
30 new sample sets; `C:\Users\oezdemir\Desktop\data\microscopy\catalogue.csv`).

### Fixed

- CZI scenes of large compressed tiles (JPEG XR slide scans, JPEG, zstd) are read much faster:
  a scene was read in row bands of about 8 MB, shorter than one tile, so every band decoded the
  tiles it crossed again (237 decodes for 25 tiles in one 10240 x 10240 read; now 43). A band is
  now at least one tile tall. A 50821 x 56323 RGB slide scan (iCAIRD, 8.6 GB) converts to
  OME-Zarr in 96 s instead of 588 s, with identical pixels.
- MetaMorph `.nd` sets whose wave names hold a `_`: MetaMorph writes it as `-` in the
  file names (`WaveName` `iSIM1_DAPI` -> `_w1iSIM1-DAPI`); both spellings are looked
  for. Such a set stopped with "none of the fields it lists has any of its files"
  (Zenodo 7963332).
- Micro-Manager 2.0.0 image-per-file acquisitions, whose `metadata.txt` names each plane
  with its position folder (`Pos0/img_...`) and keys its coordinates `FrameIndex` /
  `ChannelIndex` / `SliceIndex`: the folder was joined twice and every plane was
  "missing" (Zenodo 6659532). The channel names of a set read through an axes wrapper
  (`open(..., axes=...)`) no longer fail.
- OME companion files whose bytes are not in the encoding their XML declaration says
  (Prairie View / Bruker: Windows `µ` bytes under UTF-8) are read as cp1252 (else
  latin-1) with a `CompanionWarning` instead of failing; the Bruker T-series of Zenodo
  15224926 opens as one 30000-frame image, and its members open the whole set.
- CellVoyager CV7000 / CV8000 measurements whose files use another XML namespace prefix
  than `bts:` (or none) were not recognised; elements and attributes are matched by
  namespace, numbers with a decimal comma are read. A plane listed twice is read from the
  first record whose file exists (a metadata-only record never shadows a real one). A
  measurement of several timelines (`TimelineIndex`) is read, the timelines merged as
  Bio-Formats merges them: each timeline's timepoints counted from its own first, a
  plane a second timeline images again with the same channel and timepoint a channel of
  its own, named after its timeline (checked against Bio-Formats' expected results for
  all 14 synthetic packages of Zenodo 21334895); it was refused.
- Harmony: the analysis folders `Evaluation1`, `Evaluation2`, ... of a measurement are
  parts of its index (`group_files` lists them with the measurement; a converter given the
  folder no longer takes their XML files as inputs), and a plate whose index gives no
  `Name` nor `PlateID` is named after its measurement folder.
- Olympus OIB / OIF: a time series whose T axis has `Interval` 0 (frames as fast as the
  scan allows: a RICS series, Zenodo 11075037) gets its time step from the axis span
  (`EndPosition` - `StartPosition` over n - 1), as Bio-Formats computes it; a Z or T axis
  of one element gives no step.

### Added

- MetaXpress / ImageXpress z-series: a plate whose HTD gives `ZSteps` is read with its
  z planes from the `ZStep_1` .. `ZStep_<n>` folders (per time point when there are
  several); `ZStep_0`, the projection written when `ZProjection` is TRUE, is left out.
  The copies of the HTD in those folders are parts of the plate's: `group_files` lists
  one plate, and opening a copy opens the plate. The z step itself is not read (the HTD
  gives none). Such plates were refused (Zenodo 23468516, 12724928).
- Leica LIF / XLEF tile scans read with `tiles="mosaic"`: one image per scan, each tile
  at its stage position (`TileScanInfo` PosX / PosY, along the image's axes) on one
  canvas, nominal like the CZI and Olympus mosaics -- nothing registered, the later
  tile supplying an overlap's pixels. LAS's own merged images are registered and differ
  (checked on Intro.lif: same corner and orientation). They were refused; a scan whose
  tiles have no position (SwapXY / FlipX / FlipY set) still is.
- Olympus OIB / OIF lambda (spectral) scans: axis `L` gives one image per emission
  band (index `emission`, its wavelength in `metadata.index_values` from the axis' start
  and step), as for OIR. They were refused (Zenodo 14925254).
- Leica LIF / XLEF channel names and widefield emission wavelengths, from the image's
  hardware setting: confocal, the active detectors' dyes (`DyeName`, LAS's `Leica/`
  prefix dropped) else the detector's name (`HyD S 2`); widefield, the user's channel
  name (`UserDefName`) else the filter cube, with `EmissionWavelength`. Only when the
  setting describes exactly the image's channels, as before.

## 0.2.0 (2026-10-22): coordinate systems, transformations and the file's tree

A minor release under the release policy: positions some users already have change (the
QPTIFF overview and label pictures, the ND2 y step, a Leica SCN macro picture), so it is
not a patch. eubi-bridge 0.3.0 and 0.3.1 bound micro-reader below 0.2 and keep 0.1.0;
eubi-bridge 0.3.2 is the first that reads 0.2.0 (it writes each image's coordinate system).
The new attributes are additive: code that reads only `metadata.position` keeps working.

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
