"""Exceptions.  Plain Python exceptions only: they must survive pickling across
processes, which native exception objects (e.g. JPype's Java errors) do not."""


class MicroReaderError(Exception):
    """Base class for all micro-reader errors."""


class UnsupportedFormatError(MicroReaderError):
    """No backend recognises the file, or it uses a feature not supported yet."""


class CorruptFileError(MicroReaderError):
    """The file is recognised but inconsistent, e.g. truncated."""


class InvariantError(MicroReaderError):
    """The spatial model's invariants (plan 3.22) do not hold for a file: a
    coordinate system id that is not one micro-reader makes or not unique
    (I1), a position in a system the file does not declare or an estimated
    one (I2), a position and a coordinate system given without each other
    (I3), or an edge whose end is neither (I5).  A backend bug."""


class OpenInstead(Exception):
    """Internal, not an error: a backend found that *path* is a member of a set
    and that the set's metadata file (``.path``) is what opens it (``open``'s
    ``multifile=True``); the registry opens that file with the same options."""

    def __init__(self, path: str):
        super().__init__(path)
        self.path = path


class MissingImagesWarning(UserWarning):
    """A file set lists images whose files are missing; they are left out
    (docs/HCS.md)."""


class CompositeWarning(UserWarning):
    """A part of a composite format was read on its own because its main file
    is missing, or a part could not be told apart from another file
    (docs/COMPOSITE.md)."""


class CompanionWarning(UserWarning):
    """An OME companion set's metadata came from somewhere other than the file
    opened, a member path had to be resolved differently than written, or the
    companion's bytes had to be read in another encoding than its XML
    declares (docs/COMPANION.md)."""


class SkewedGridWarning(UserWarning):
    """The file's sampling grid has axes that are not perpendicular (a
    crystallographic unit cell with angles other than 90 degrees, e.g. an MRC
    map).  The image is read as stored, with the true sampling distance along
    each grid axis; an axis-aligned image (OME-Zarr before 0.6) cannot hold
    the skew."""


class StoredValuesWarning(UserWarning):
    """The file stores its pixels in a form that is not the measured values, and
    micro-reader reads the measured ones; the warning says what was changed
    (e.g. an older scanR plate's 12-bit values stored with the top bit set)."""


class PlateLayoutWarning(UserWarning):
    """A plate's wells lie outside the layout its file describes (e.g. a slide
    holder whose records address the wells of the slide in it); the layout
    is grown to hold them (docs/HCS.md)."""
