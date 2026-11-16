# Errors and warnings

micro-reader raises plain Python exceptions, so they can be pickled across processes. Its errors derive from `MicroReaderError`. Problems that leave the file readable are reported as warnings, which are subclasses of `UserWarning`.

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

Standard exceptions are used where they fit.

- **`ValueError`** is raised for invalid arguments.
- **`TypeError`** is raised for a value of the wrong kind, such as a `Well` object given to `select` or a number given to a `*_name` keyword.
- **`KeyError`** is raised for unknown axis, index or well names.
- **`IndexError`** is raised for unsupported or out-of-range indexing.
- **`FileNotFoundError`** is raised for missing paths.
- **`AttributeError`** is raised for an accessor of another level of a [plate](plates.md#containment).

`tiles="stitched"` also issues a `DeprecationWarning`.

## Errors

### MicroReaderError

The base class of all micro-reader errors. Catch it to handle any failure of the library. It is also raised directly when a folder holds several images.

### UnsupportedFormatError

No backend recognises the file, or the file uses a feature that micro-reader does not support yet. It is raised in these cases:

- The content matches no format.
- `backend=` names an unknown backend.
- `tiles="mosaic"` is used for a tiled file of a format that cannot place its tiles, or whose tiles have no position.
- `rename=` is used for an index that the backend cannot turn into an axis.
- A folder has no file that is an image.

To handle it, skip the file or hand it to another reader:

```python
try:
    f = micro_reader.open(path)
except micro_reader.UnsupportedFormatError:
    f = other_reader(path)
```

### CorruptFileError

The file is recognised but inconsistent. It is raised in these cases:

- The data ends early, which means the file is truncated.
- A part of a multi-file image has a missing main file, and the part is not an image on its own.
- A file that the format refers to is missing, as in Leica XLEF / XLIF.
- A plate plane file has a size or pixel type that does not match the index. This is found on read or by `check_members()`.
- A CZI subblock that could not be found in a damaged file is read.
- An OME-TIFF member has a metadata file that cannot be followed, because it is missing, unreadable or does not list the file.
- An OME companion has pages that hold other samples than it lists.

### InvariantError

Raised while the coordinate systems, transformations or tree of a file are declared, when the rules of the spatial model do not hold. This happens in these cases:

- A coordinate system id is not valid or not unique.
- A position is in a system that the file does not declare, or in an estimated one.
- A position and a coordinate system are given without each other.
- An edge has an end that is neither a system nor an image.
- Images of a plate or well are not held by `File.plates`.

It means there is a bug in a backend. `File.images` and `File.plates` still reach the images. Report it together with the format of the file.

### MemoryLimitError

A single block that must be decoded whole is larger than the whole [memory limit](../guide/performance.md#the-memory-limit). It is raised by the read and not by `open()`. A decode that merely does not fit next to other running decodes does not raise this error. It waits until they finish. The message gives both sizes and says how to raise the limit:

```python
try:
    plane = image.read(z=0)
except micro_reader.MemoryLimitError:
    micro_reader.set_memory_limit("32GB")
    plane = image.read(z=0)
```

## Warnings

Warnings are issued once, when the file is opened or when the problem is first found. To turn a warning into an error in a pipeline that must not accept incomplete data, use a filter:

```python
import warnings
warnings.simplefilter("error", micro_reader.MissingImagesWarning)
```

To collect the warnings per file instead, see the [batch example](../guide/opening.md#batch-processing-many-files).

### DamagedFileWarning

The subblock directory of a CZI does not match its data. micro-reader recovered what it could by checking each subblock against its own header, and it reads the file. `File.damage` has the details. Other readers fail on such files or return wrong pixels. Check the file or copy it again.

### MissingImagesWarning

A file set, such as a plate or a companion set, lists plane files that are missing. Missing planes are read as zeros. Fields or images with none of their files are left out. The message gives counts and an example file name. A companion whose images miss member files gives one warning for the set, which counts files.

### CompanionWarning

The metadata of an OME companion set came from somewhere other than the file that was opened, or could not be found. For example, a TIFF of a set was opened alone and its companion is missing, so it is read without pixel sizes. The warning is also issued when a member path had to be resolved differently from how it was written. It is also issued when the bytes of the companion are not in the encoding that its XML declares, as in Prairie View / Bruker files. These bytes are read as cp1252, or as latin-1 if that fails.

### CompositeWarning

A part of a multi-file format was read on its own because its main file is missing, or a part could not be told apart from another file. An example is two files whose names differ only in letter case.

### PlateLayoutWarning

The wells of a plate lie outside the layout that its file describes, or the layout had to be guessed. One example is a CellVoyager slide holder that is described as 1 x 4 slides, but whose records address the wells of the slide in it. Another example is a scanR or ND2 plate whose layout was guessed from the names of the wells. The layout is chosen so that it holds every well, and the message says what was done.

### SkewedGridWarning

The sampling grid of the file has axes that are not perpendicular. An example is a crystallographic unit cell with angles other than 90 degrees, such as an MRC map. The image is read as stored, with the true sampling distance along each grid axis. OME-Zarr before 0.6 cannot hold the skew.

### StoredValuesWarning

The file stores its pixels in a form that is not the measured values, and micro-reader reads the measured ones. The message says what was changed. An example is an older scanR plate whose 12-bit values are stored with the top bit set. They are read without it.
