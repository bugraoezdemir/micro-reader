"""Exceptions.  Plain Python exceptions only: they must survive pickling across
processes, which native exception objects (e.g. JPype's Java errors) do not."""


class MicroReaderError(Exception):
    """Base class for all micro-reader errors."""


class UnsupportedFormatError(MicroReaderError):
    """No backend recognises the file, or it uses a feature not supported yet."""


class CorruptFileError(MicroReaderError):
    """The file is recognised but inconsistent, e.g. truncated."""


class MissingImagesWarning(UserWarning):
    """A file set lists images whose files are missing; they are left out
    (docs/HCS.md)."""


class CompanionWarning(UserWarning):
    """An OME companion set's metadata came from somewhere other than the file
    opened, or a member path had to be resolved differently than written
    (docs/COMPANION.md)."""
