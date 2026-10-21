"""micro-reader: read exactly the rectangle you ask for.

>>> import micro_reader
>>> with micro_reader.open("image.mrc") as f:
...     block = f.images[0].read(z=0, y=slice(0, 512))
"""
from . import _eubi_guard  # first: refuses old eubi before any backend loads (plan 8.28)
_eubi_guard.check()
from ._version import __version__
from ._base import INDEX_NAMES, PLATE_INDICES, SAMPLED_INDICES, File, Image, ReadStats
from ._metadata import Channel, Metadata, Quantity
from ._plate import Acquisition, Field, Plate, Well
from ._errors import (CompanionWarning, CompositeWarning, CorruptFileError, MicroReaderError,
                      MissingImagesWarning, PlateLayoutWarning, SkewedGridWarning,
                      UnsupportedFormatError)
from ._memory import MemoryLimitError, memory_limit, set_memory_limit
from ._io import cache_limit, cached_bytes, set_cache_limit
from ._pool import decode_threads, set_decode_threads
from ._registry import BACKENDS, open  # noqa: A004 - the public entry point
from ._groups import FileGroup, Grouping, group_files
from .backends.czi import DamagedFileWarning

__all__ = [
    "open", "group_files", "FileGroup", "Grouping", "File", "Image", "INDEX_NAMES", "PLATE_INDICES", "SAMPLED_INDICES", "ReadStats",
    "BACKENDS", "Metadata", "Channel", "Quantity", "Plate", "Well", "Field", "Acquisition",
    "decode_threads", "set_decode_threads", "memory_limit", "set_memory_limit",
    "cache_limit", "set_cache_limit", "cached_bytes",
    "MicroReaderError", "UnsupportedFormatError", "CorruptFileError", "MemoryLimitError", "DamagedFileWarning",
    "MissingImagesWarning", "CompanionWarning", "CompositeWarning", "PlateLayoutWarning",
    "SkewedGridWarning", "__version__",
]
