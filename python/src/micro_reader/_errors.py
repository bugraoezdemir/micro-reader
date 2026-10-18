"""Exceptions.  Plain Python exceptions only: they must survive pickling across
processes, which native exception objects (e.g. JPype's Java errors) do not."""


class MicroReaderError(Exception):
    """Base class for all micro-reader errors."""


class UnsupportedFormatError(MicroReaderError):
    """No backend recognises the file, or it uses a feature not supported yet."""


class CorruptFileError(MicroReaderError):
    """The file is recognised but inconsistent, e.g. truncated."""


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
    opened, or a member path had to be resolved differently than written
    (docs/COMPANION.md)."""


class PlateLayoutWarning(UserWarning):
    """A plate's wells lie outside the layout its file describes (e.g. a slide
    holder whose records address the wells of the slide in it); the layout
    is grown to hold them (docs/HCS.md)."""
