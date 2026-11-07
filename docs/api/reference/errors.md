# Errors and warnings

micro-reader raises plain Python exceptions, so they can be pickled across processes. Its errors derive from `MicroReaderError`. Problems that still leave the file readable are reported as warnings, which are subclasses of `UserWarning`.

```text
Exception
└── MicroReaderError
    ├── UnsupportedFormatError
    ├── CorruptFileError
    ├── MemoryLimitError
    └── InvariantError

UserWarning
├── DamagedFileWarning
├── MissingImagesWarning
├── CompanionWarning
├── CompositeWarning
├── PlateLayoutWarning
├── SkewedGridWarning
└── StoredValuesWarning
```

Standard exceptions are used where they fit: `ValueError` for invalid arguments, `TypeError` for a value of the wrong kind (a `Well` object given to `select`, a number given to a `*_name` keyword), `KeyError` for unknown axis, index or well names, `IndexError` for unsupported or out-of-range indexing, `FileNotFoundError` for missing paths, `AttributeError` for an accessor of another level of a [plate](plates.md#containment). `tiles="stitched"` also issues a `DeprecationWarning`.

## Errors

### MicroReaderError

Base class of all micro-reader errors. Catch it to handle any failure of the library. It is also raised directly when a folder holds several images.

### UnsupportedFormatError

No backend recognises the file, or the file uses a feature that micro-reader does not support yet. Examples:

- the content matches no format;
- `backend=` names an unknown backend;
- `tiles="mosaic"` for a tiled file of a format that cannot place its tiles, or whose tiles have no position;
- `rename=` for an index that the backend cannot turn into an axis;
- a folder in which no file is an image.

Typical handling: skip the file, or hand it to another reader.

```python
try:
    f = micro_reader.open(path)
except micro_reader.UnsupportedFormatError:
    f = other_reader(path)
```

### CorruptFileError

The file is recognised but inconsistent. Examples:

- the data ends early (a truncated file);
- a part of a multi-file image whose main file is missing, when the part is not an image on its own;
- a file that the format refers to is missing (Leica XLEF / XLIF);
- a plate plane file whose size or pixel type does not match the index (on read, or from `check_members()`);
- reading a CZI subblock that could not be found in a damaged file;
- an OME-TIFF member whose metadata file cannot be followed (missing, unreadable, not listing the file), or an OME companion whose pages hold other samples than it lists.

### InvariantError

Raised while a file's coordinate systems, transformations or tree are declared, when the spatial model's rules do not hold: a coordinate system id that is not valid or not unique, a position in a system the file does not declare (or in an estimated one), a position and a coordinate system given without each other, an edge whose end is neither a system nor an image, or images of a plate or well that `File.plates` does not hold. It means a bug in a backend. `File.images` and `File.plates` still reach the images. Report it with the file's format.

### MemoryLimitError

A block that must be decoded whole does not fit the [memory limit](../guide/performance.md#the-memory-limit). It is raised by the read, not by `open()`. The message gives both sizes and how to raise the limit:

```python
try:
    plane = image.read(z=0)
except micro_reader.MemoryLimitError:
    micro_reader.set_memory_limit("32GB")
    plane = image.read(z=0)
```

## Warnings

Warnings are issued once, when the file is opened (or when the problem is first found). To turn them into errors in a pipeline that must not accept incomplete data:

```python
import warnings
warnings.simplefilter("error", micro_reader.MissingImagesWarning)
```

To collect them per file instead, see the [batch example](../guide/opening.md#batch-processing-many-files).

### DamagedFileWarning

A CZI's subblock directory does not match its data. micro-reader recovered what it could, checking each subblock against its own header, and reads the file. `File.damage` has the details. Other readers fail on such files or return wrong pixels. Check the file or copy it again.

### MissingImagesWarning

A file set (a plate, a companion set) lists plane files that are missing. Missing planes are read as zeros. Fields or images with none of their files are left out. The message gives counts and an example file name. A companion whose images miss member files gives one warning for the set, counting files.

### CompanionWarning

An OME companion set's metadata came from somewhere other than the file opened, or could not be found. For example, a TIFF of a set was opened alone and its companion is missing, so it is read without pixel sizes. Also issued when a member path had to be resolved differently from how it was written, or when the companion's bytes are not in the encoding its XML declares (Prairie View / Bruker files): they are read as cp1252, else latin-1.

### CompositeWarning

A part of a multi-file format was read on its own because its main file is missing, or a part could not be told apart from another file (for example, two files whose names differ only in letter case).

### PlateLayoutWarning

A plate's wells lie outside the layout its file describes, or the layout had to be guessed. For example a CellVoyager slide holder described as 1 x 4 slides whose records address the wells of the slide in it, or a scanR or ND2 plate whose layout was guessed from the wells' names. The layout is given so that it holds every well, and the message says what was done.

### SkewedGridWarning

The file's sampling grid has axes that are not perpendicular (a crystallographic unit cell with angles other than 90 degrees, e.g. an MRC map). The image is read as stored, with the true sampling distance along each grid axis. OME-Zarr before 0.6 cannot hold the skew.

### StoredValuesWarning

The file stores its pixels in a form that is not the measured values, and micro-reader reads the measured ones. The message says what was changed. Example: an older scanR plate's 12-bit values stored with the top bit set are read without it.
