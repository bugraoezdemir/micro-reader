# Changelog

## 0.4.1 (unreleased)

### New

- BigDataViewer / BigStitcher tiles open as one image with `tiles="mosaic"` (they were
  refused): every level of each angle's and illumination's tiles on one canvas, at their
  nominal positions (the acquisition grid or tile configuration, not a stitching result),
  rounded to whole pixels; the later tile supplies the pixel where tiles overlap, nothing is
  blended. The image is placed at the canvas corner in `global`. Tiles that cannot be placed
  so (a turned, sheared or mirrored view, tiles apart along z, different pixel sizes) are
  still refused, naming why. On BigStitcher's grid-3d sample, 6 tiles of 512 x 512 make one
  1434 x 973 image.

## 0.4.0 (2026-11-07)

### New

- KLB (Keller Lab Block) files open (backends/klb.py), with no Java and no dependency: every
  time point and channel, bzip2, zlib or stored, decoding only the blocks a read touches, in
  parallel. On the three samples held the pixels equal a separate decode of the file's blocks,
  and all 10 time points of `test.klb` read (Bio-Formats reports one). A 1 in the header is
  its default, no measurement: on t, on z beside a set x or y, and on x, y and z together.
- ND2 version 2 files open: the chunked layout with its metadata in XML, which
  `Ver2.x` files have (13 of the 83 ND2 samples we hold). Pixels equal the `nd2` package's,
  and sizes, on all but one, which that package misreads.
- Luxendo / Bruker light-sheet experiments (backends/luxendo.py): the experiment folder
  (`raw/` with one folder of `.lux.h5` stacks per stack, channel and objective) opens as
  one image per stack, objective and camera, `t` the time points, `c` the channels by
  Luxendo channel number, named by their description; the time step from the stacks'
  time stamps. Any part opens it: the folder, `raw/`, a channel folder, a `.lux.h5` or
  its `.json`, and the link files next to `raw/` (`main_raw.lux.h5`, `imaris.ims`,
  `ims/stacks/*.ims`, `bdv.xml` + `bdv.h5`). The pixels always come from `raw/`; the
  link files say which files should exist, and one they name that `raw/` lacks reads as
  zeros with one `MissingImagesWarning` (BioImage Archive S-BIAD3686: every `bdv.xml`
  without its `bdv.h5`, passed over with a `CompositeWarning`). Each stack is checked
  against its place when first read (`CorruptFileError` naming it). Several stacks or
  cameras: index `tile` / `view`. At most 64 stack files are kept open.
- A folder whose chosen file fails to open (an index file is chosen over images) names
  the folder's other images in the error.
- A folder that holds no image itself opens the one dataset in its sub-folders
  (decision MR-6): a downloaded sample folder over its export, down to 6 levels, each
  sub-folder taken as `open` takes it (a format's root, else its index file, else its
  images). On the real-data collection 32 sample folders open that did not (38 of 54
  plate samples failed), none differently, none newly failing. Several datasets: the
  error lists them (relative paths). With `plates=True` a folder of one NIS-Elements
  well-plate run is one dataset, and without it the error says that `plates=True`
  reads the run as a plate.
- A folder warns about `.ptu`, `.phu` and `.sptw` files beside the images it opens
  (a `CompositeWarning` naming them). An orphan softWoRx log names the deconvolved image
  beside it, and a companion with none of its files names a companion in the folder above.
- Clearer refusals: an OME-Zarr store (or a file in one) names ome_zarr_pyramid, which
  reads it; `.ptu`, `.phu` and `.sptw` files name their format (PicoQuant) instead of
  "no micro-reader backend recognises".
- `Quantity.to(unit)` takes the spellings files use (`"um"`, `"ms"`, `"nm"`) as well as
  OME-Zarr's names, so values that formats record in different units compare
  (`pixel_sizes["t"].to("s")`; decision MR-3: each value keeps the file's unit). A
  length converted to a time, or an unknown unit, raises `ValueError`; before, a length
  "converted" to seconds returned a number and `"um"` raised a bare `KeyError`.

### Changed

- `Image.name` follows one rule for every format and entry point (decision MR-2;
  README "Image names"): a plate's image is `"well B2 field 0"`; otherwise the
  format's own name (Leica element, OME `Image Name`, Imaris, slide roles), else a label
  the file records (CZI scene, ND2 point, Micro-Manager / MetaMorph position), else the
  index values that tell the file's images apart (`"series 2"`), else, for one image,
  the dataset's main file without its extension; duplicates get what differs appended.
  Before, each backend had its own rule (`"file.czi series 0"`, `"file.nd2 position 0"`,
  `"series0"`, the file name with its extension), a LIF lambda scan gave 29 images
  named `Series001`, and the same dataset was named differently through different
  files. Checked on the real-data collection: every entry point of a dataset gives the
  same names. eubi-bridge writes the name as the OME `Image Name` and ome_zarr_pyramid
  as the pyramid's name; output paths do not depend on it.
- `open` gives every image in the canonical order `t c z y x` (then `s`) by default
  (decision MR-4): axes the file lacks are added with size 1, the others reordered. Before,
  each image came in its backend's order: ICS `ctzyx`, Micro-Manager OME-TIFF `tzcyx`,
  ilastik `txyc` (x before y), a 2-D TIFF `yx`. `axes="native"` (or `axes=None`) keeps
  that order. An image already in the order asked for stays the backend's own object (no
  wrapper); reads take the same time (a 100 MB ImageJ movie: 14 ms either way).
  eubi-bridge asks for the order it reads in itself and sees no change.
- A `.lux.h5` or an `imaris.ims` in a Luxendo experiment folder opens the experiment
  (`backend` `"luxendo"`), not the one stack (`"hdf5"`) or the Imaris file (`"ims"`);
  `open(..., multifile=False)` reads them alone as before.
- A plate's plane file opens the plate in every format (CellVoyager, Harmony, Columbus,
  JDCE, IN Cell and IN Carta, MetaXpress did not; Cellomics, Flex, scanR and companions
  did), and `multifile=False` reads that file alone. A file named like a plane with no
  index near it is still a TIFF. 91 planes of the real-data plates checked.
- A slide's main scan is named after its file, not `Baseline` (tifffile's word for it),
  and its associated images are lower case everywhere (`macro`, `map`; NDPI gave `Macro`,
  `Map`). `images` stays in file order: which image is the scan is for image roles.
- Leica: `FrameProperties` and `IOManagerConfiguation` files are parts of their
  experiment, and a `.lifext` and an `.xml` of a `.lif` are parts of the `.lif`
  (`multifile=False` reads a `.lifext` itself). `group_files` over the Leica samples gives
  22 groups that all open, against 57 of which 39 failed; no dataset is lost.
- A file its set does not list (a plate's calibration image) raises
  `UnsupportedFormatError`, not `CorruptFileError`: nothing is wrong with the file.
- `File.path` and messages spell a path one way (`os.path.normpath` of what was opened).
- A lambda scan is one image in every format (decision MR-5): its bands along `c`, a
  detector's bands together, each channel with its `emission_wavelength` and its wavelength
  in its name; with `axes="native"`, or axes naming `l`, a real axis `l` and the wavelengths
  in `metadata.index_values["emission"]`. Before, LIF, OIR, OIB and OME-TIFF gave one image
  per band (29 images named `Series001`, index `emission`) and an LSM 30 channels without
  wavelengths; LSM channels now carry theirs. Excitation scans stay an index. Reads go band
  by band through the backends' own reads.
- `open(plates=True)` and `group_files(plates=True)` are the default: a CZI, ND2, OME-TIFF
  or companion acquired on a plate reads as a plate (`well` and `field` indices,
  `"well B2 field 0"` names, the plate's image order), and one `.nd2` of a NIS-Elements run
  opens the run. `plates=False` gives the old reading by series. A file that is no plate
  does not change.
- A Leica element's `.lof` and `.xlif` open their experiment, as a plate's plane opens its
  plate: the same image, name, coordinate system and time step from each (a `.lof` beside its
  `.xlef` gave `Series001` and another system). `multifile=False` reads the file alone.
- ScanImage: the frames of a recording of one slice are `t`, with the time step from the
  frame rate (tifffile calls them `z` when fast z is on: 1000 frames of a plane read as a
  z stack).
- Channels with nothing to say say nothing: several channels all `#FFFFFF` (Imaris,
  Micro-Manager defaults) have no colour, BigDataViewer channels whose name is their id and
  VSI channels named like their image have no name. One white channel, or white among other
  colours, stays.
- `Image.is_pyramid` is deprecated (a `DeprecationWarning`): it means level 0, also for an
  image with no lower levels, and is too close to `is_pyramidal`, which says whether lower
  levels are stored. Use `image.level == 0`.
- `select` raises `ValueError` for a plate, acquisition, well, row or column name no plate
  has, naming the names there are (`select(well="Z99")` returned `[]`, decided
  2026-10-30); a name the layout has whose wells hold no images still selects nothing.
  ome_zarr_pyramid 0.6.0's `select` raises the same.

### Fixed

- A stepped read (`image[::4, ::4]`, `image[::16]`) holds its result and one piece of at most
  64 MB (`STEPPED_READ_BYTES`), not the whole span it covers, in every format: the span was read
  whole and then thinned, so `image[:, ::16, ::16]` of a 6.8 GB MRC held 6.6 GB (also in 0.3.x);
  it now peaks at 98 MB, the result 27 MB. Large spans are read in runs of planes or rows that
  fit, each thinned, so a compressed tile is decoded once as before; where one step spans more
  than a piece (`[::16]` of large planes), only the planes kept are read. Small stepped reads
  stay one read. Against the old single read (which held the whole span), a 302 MB plane read
  with steps took 1.2 to 1.7 times as long compressed (0.18 s against 0.11 s) and up to 2.7
  times uncompressed (0.08 s against 0.03 s); `[:, ::4, ::4]` of the 6.8 GB MRC takes 4 s, the
  first version of this fix 7 s. Reading only the selected rows of an uncompressed file is a
  faster way still, for a later release.
- A CZI cut short before its subblock directory says it is truncated and to copy it again,
  not only "no ZISRAWDIRECTORY at byte N". `set_memory_limit` refuses a negative budget and
  `set_decode_threads` fewer than 1 thread (they were taken in silence). A Harmony or
  Columbus plate refused while it opens (a page of another size than the index says) closes
  the member TIFFs it had opened (they stayed open, with ResourceWarnings).
- An ND2 with damaged metadata raises `CorruptFileError` ("damaged ND2 metadata ..."), not the
  decoder's own `UnicodeDecodeError`. The reference pages give `open()`'s and
  `group_files()`'s defaults as they are (`plates=True`, `axes="canonical"`), the image name
  rule of this release, and the README's "Ambiguous dimensions" outputs as 0.4.0 prints them.
- A plate in one OME-TIFF no longer takes minutes to describe: each image's metadata parsed
  the whole file's OME-XML (0.65 s per image, 25 minutes for 2304 images; now 0.4 s for all).
- An OME-TIFF a plate's OME-XML cannot be read as a plate from (CQ1's `MeasurementResultMIP`)
  opens by its series, also now that plates are the default.
- An OME companion's time step comes from its planes' `DeltaT` where `TimeIncrement` is
  absent (Bruker time series).
- ND2 time steps are the mean interval of the frames' own stamps, read from the first and last
  frame; without stamps, the file's measured `dAvgPeriodDiff`, then the requested period. They
  agree with the `nd2` package's frame times on all 42 time series compared. A camera slower than
  the loop gave rates up to twice too fast, old NIS files record a wrong `dAvgPeriodDiff` (14 s
  for frames 30 s apart), and two files gave no step at all. Time loops whose valid periods
  differ warn: one step does not describe them.
- `is_ragged` is False for images whose parts differ only in pixel type or wavelength.
- BigDataViewer voxel sizes in the unit `pixel` are kept, as relative sizes in that unit.
- Imaris TIFF exports give their z step (`Dz`).
- FileConverter's `.h5` copy of an Imaris `.ims` belongs to its set: one group, not two.
- `rename=` of the wrong type raises a `TypeError` naming the form.
- Legacy ND2 files give the z step and time interval of their experiment loops.
- `np.asarray(image)` gives the pixels (it wrapped the Image in a 0-d object array), and
  `len(image)` works.
- ScanImage: pages that do not fill the file's shape warn, instead of being dropped in silence.
- Leica settings files in nested collections belong to their experiment.
- CellVoyager: one `Ch` used by several actions with other filters gives each its own
  name and wavelengths.
- Leica histogram series no longer report their bins as pixel sizes (3921 um per pixel:
  1 m over the axis is a normalised axis, as 1 m per pixel already was for phasor plots).
- Two channels of one name folded with their RGB samples stay apart (`"BF 0 R"`, `"BF 1 R"`;
  an RGB ND2 of two `Brightfield` channels gave the same names twice).
- HDF5 without `axistags`: the refusal says why (a dataset does not say what its axes are,
  and micro-reader does not guess them) and that h5py reads it.
- Imaris files whose text attributes are arrays of 1-byte integers (Luxendo writes
  them so) lost their metadata: channel names read as the first character's code
  (`"99"`) and the pixel sizes were missing (the unit read as `"117"`).
- Plain TIFFs: the resolution tags give a pixel size only with an explicit centimetre
  `ResolutionUnit`. Inches (TIFF's default when the tag is absent) were a screen or print
  density on every real file: a CellVoyager plane opened alone gave 903 111 µm per pixel
  (0.028 / inch), IDR CellVoyager and Prairie View planes 265 µm and 353 µm (96 and
  72 dpi). Every microscope calibration in the real-data collection uses centimetres.
- Zeiss LSM files with one z plane gave a z pixel size of 1e-13 µm (`VoxelSizeZ` is
  1e-19 m on all of them); an Imaris file with one plane gave 0.001 µm (its made-up z
  extent). Neither reports a z step now.
- ImageJ TIFFs reported a frame interval for stacks without frames (`finterval=1` on a
  z stack); the interval is read only with `frames` > 1.
- A BigDataViewer dataset opened through its `.h5` was named in lower case
  (`hisyfp-spim.xml angle 0`, also `File.path` and the tree's root): on Windows the XML
  found from the file's lower-cased stem kept those letters. Files a format's rule
  finds are now spelled as the folder spells them.
- `group_files` gave each file of a multi-file OME-TIFF set as an image of its own (a
  4-D set of 86 files: 86 singles), although opening any of them opens the whole set:
  the set is one `index` group now, its first file the main. Only an OME-TIFF whose
  OME-XML names another file is opened for this (its first page read first), after
  every other index file, so one file alone stays a plain path and grouping a folder
  costs one open per set. A `BinaryOnly` stub belongs to the file it points to.
- `select`, `Plate.well_index` and `Plate.rows` / `.columns` compare well, row and column
  names as vendors spell them (MetaXpress and Cellomics `A01`, Harmony and NGFF `A1`):
  without letter case and without the leading zeros of the number, so `well="B02"` is
  `"B2"` and `column_name="05"` is `"5"`; before, they selected nothing. `rows["B":"D"]`
  takes a slice of names, both ends included (it raised a bare `TypeError`).
  ome_zarr_pyramid 0.6.0 compares them the same way.
- A truncated OIR file raised `UnsupportedFormatError` ("OIR file without image data");
  it is a `CorruptFileError` naming the size its header gives, as for the other formats.
- A Micro-Manager 2.0 set names its positions by their labels (`2-Pos000_000`): 2.0 writes
  `Label`, micro-reader looked for `label` and fell back to `Pos0`, `Pos1`; the positions of
  an MMStack OME-TIFF were `series 0`, `series 1`.
- Time steps come from the planes' own time stamps where a format records them, as for ND2:
  a MetaMorph `.nd` set from its planes' `acquisition-time-local` (it had no time step; the
  `.nd` says "1 min", the planes 60.4 s), Micro-Manager from the frames' `ElapsedTime-ms`
  first and the Summary's `Interval_ms` second (a set without the interval had none). When
  the stamps and the declared interval differ by more than 5%, the stamps are used and one
  warning names both. Imaris warns when its time points are unevenly spaced (the largest gap
  over 1.1 times the smallest); the step stays their mean.
- A truncated JPEG (one cut short by an interrupted copy) is refused with `CorruptFileError`;
  before, its lost rows were read as fill with no error. Data after the end-of-image marker
  still reads.
- A TIFF named like a plane of an index-based plate (Harmony, CellVoyager, ...) that the index
  does not list opens as itself; before, `open()` opened the whole plate from it and the file
  itself was never read.
- A squeezed view lists the pixel sizes of its own axes only: `axes="yx"` of a z-stack gave a
  z size the view has no axis for. A view of all five axes keeps the sizes the file gives,
  size-1 axes included, and invents none.
- BigDataViewer images take the name their view setups give in the XML (`Angle 45`,
  `Fused`), when the setups of an image all give the same one and it is not just the setup's
  id; they were named `view 0`, `view 1`. BigStitcher datasets, which name setups by id, keep
  `tile 0`, `tile 1`.
- An Imaris image whose stored name is a path (the acquisition machine's file,
  `D:\FusionImages\...\x.ims`) takes its file's name instead.
- OME-TIFF and OME companion files take their time step from the planes' own times
  (`Plane@DeltaT`), as the other formats do: a Micro-Manager MMStack file that declares
  1 ms ("as fast as possible") for frames 1.12 s apart reads 1118 ms, with one warning
  naming both; files with times but no `TimeIncrement` had no time step at all.
- Olympus OIR lambda scans: each band's excitation is the laser the acquisition let through
  (405 nm), not the last laser its channel lists (640 nm, longer than the emission).
- `select(well=...)` with a well of the plate's layout that holds no images selects
  nothing, as rows and columns do; it raised `ValueError`. A well outside the layout is
  still refused.
- `Quantity.to` takes the quantity's own unit by the same spellings as the target
  (`Quantity(60000, "ms").to("s")` failed), and its error names the real reason (an
  unknown unit, or a time and a length).
- The README and the guides described `plates=True` as an opt-in: it is the default since
  0.4.0, and `plates=False` reads a plate by its series. The opening guide's CZI plate example
  opened with `plates=False` and then read `f.plates`. The axes guides printed the axes of
  0.3.x (`axes="canonical"` is the default; `axes=None` gives the file's own).

## 0.3.0 (2026-10-28): select by well name, row and column

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
- VSI: a tile an ETS file does not store (outside a slide's scanned area) read as 0, black
  around the tissue of a VS200 H&E slide. It reads as the ETS header's background value
  now, one per component, as Bio-Formats fills it (`0xDCDCDC`: light grey). Checked on the
  9 VSI samples held: every image and pyramid level gives Bio-Formats' pixels, except lossy
  JPEG 2000 tiles, where Bio-Formats' decoder rounds up to 1 apart from OpenJPEG's
  (micro-reader decodes as OpenJPEG).
- CellVoyager: a ragged field (a 3-D action next to 2-D ones) took its z position from the
  first channel's plane, not from its slice 0. On JUMP's CV8000 Cell Painting plates (five
  2-D fluorescence channels at z -9 um, then a 3-plane brightfield at -17 / -9 / -1 um) the
  whole field sat one z step (8 um) too high. z is now the first plane of the channel with
  the most planes, and each part keeps its own. Checked on JUMP plate 1053601756 (CV8000,
  2304 fields) and BIA S-BIAD3934 (CV7000, 960 fields): every plane of 40 random fields
  equals its TIFF, and every position equals its record.
- NIS-Elements runs: a field whose channels come from files of their own said it was ragged
  (`is_ragged` True, `parts` the per-file images), because it kept its files where ragged
  images keep their parts. Its parts are now just the field.
- Older scanR plates: opening one listed the plate folder and the data folder twice each.
  Each is listed once now (real-data matrix, G4).
- MetaXpress / ImageXpress exports: the copies of a plate's `.HTD` in its `TimePoint_` folders
  were plates of their own for `group_files`, and opening one failed (a time series: none of
  its fields' files are beside it); a single-timepoint export with a `TimePoint_1` folder gave
  the plate twice. Every copy is now a part of the outermost `.HTD`, which it opens. The
  projection TIFFs in `ZStep_0` (not read) are parts of the plate too, so they are no longer
  images of their own (a folder conversion converted each one). Checked on zenodo 12724928
  (21 plates in 3 exports) and 23468516.

### New (scanR, ND2 plates)

- Olympus / Evident scanR plates (`experiment_descriptor.xml` with `data/metadata.ome.xml`),
  which opened as an OME companion without a plate: wells and fields from the data files'
  names, the layout from the descriptor. The descriptor, the folders, the `.dat` files,
  the companion and (with `multifile=True`) any data TIFF open the plate.
- Older scanR plates (2007 to about 2015, no `data/metadata.ome.xml`), which were refused:
  the plate from the descriptor and the data files' names (`--W00002--P00001--...` without
  a well name in scanR 2007), the layout from the plate format or the custom format,
  channel names from the descriptor, the pixel size, and stage positions from
  `AcquisitionLog.dat` when it has them. The layout is checked against the well table's
  names (refused when they disagree) and the format's description (a
  `PlateLayoutWarning`); a guessed layout warns too. 12-bit values stored with the top
  bit set are read without it, with the new `StoredValuesWarning` (Bio-Formats keeps the
  low 12 bits of every scanR pixel, which would cut 16-bit data); other plates as stored.
  A descriptor with Latin-1 bytes and no encoding named (`[µm]`) failed as not readable
  XML; it is read as Latin-1.
- ND2 with `plates=True`: a file whose XY points are named after wells as NIS-Elements
  names them (`A10_0000`, `B02_0003`, `E8.1`) reads as a plate (fields in acquisition
  order). Other point names (`P1`, `A1`) keep the positions as they are. The layout is the
  well plate of the NIS-Elements job that acquired the file (its job definition chunk),
  else the smallest standard plate holding the wells, with a `PlateLayoutWarning`.
- NIS-Elements well-plate runs with `plates=True`: a folder of ND2 files, one per well
  (`WellB05_ChannelDIA_Seq0000.nd2`) or per point
  (`WellA01_PointA01_0003_ChannelFITC BP_Seq0003.nd2`), reads as one plate from any of its
  files; each well's points are its fields, channels in files of their own are stacked.
  `multifile=False` or `plates=False` read the one file, as before. A run micro-reader
  cannot place (two files of one channel in a field, a file of several XY positions, RGB
  files) is refused, naming `multifile=False`. At most 32 of its files stay open; the
  others open again when read.
- `group_files(paths, plates=False)`: with `plates=True`, sets that are plates only then (a
  NIS-Elements run folder) are one group.
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
