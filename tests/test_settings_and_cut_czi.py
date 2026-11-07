"""Settings that make no sense are refused, and a cut CZI says it is cut (tenth review,
micro-reader N-09 and C-21)."""
import numpy as np
import pytest

import micro_reader
from micro_reader._errors import CorruptFileError


def test_a_negative_memory_limit_and_no_threads_are_refused():
    with pytest.raises(ValueError, match="cannot be negative"):
        micro_reader.set_memory_limit(-1)
    old = micro_reader.decode_threads()
    for n in (0, -2):
        with pytest.raises(ValueError, match="at least 1 thread"):
            micro_reader.set_decode_threads(n)
    assert micro_reader.decode_threads() == old            # left as it was
    micro_reader.set_memory_limit(None)


def test_a_cut_czi_says_it_is_truncated(tmp_path):
    from tests._czi_writer import raw_payload, write_czi
    path = tmp_path / "cut.czi"
    write_czi(str(path), [dict(series=0, plane={"C": 0}, x=0, y=0, height=8, width=8, dtype="u2",
                               compression=0, payload=raw_payload(np.ones((8, 8), "u2")))])
    data = path.read_bytes()
    import struct
    directory = struct.unpack_from("<iiii16s16siqqiq", data, 32)[7]   # as backends/czi.py reads it
    path.write_bytes(data[:directory - 1])                  # stops before the directory
    with pytest.raises(CorruptFileError, match="it is truncated.*Copy the file again"):
        micro_reader.open(str(path))
