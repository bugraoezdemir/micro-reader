"""A damaged ND2 metadata block is a CorruptFileError, not the decoder's own error (tenth review,
micro-reader N-02: a raw UnicodeDecodeError from decode_clx_lite)."""
import struct

import pytest

from micro_reader._errors import CorruptFileError
from micro_reader.backends._nd2_structure import decode_clx_lite


@pytest.mark.parametrize("data", [
    bytes([8, 2]) + b"\x00\xd8\x00\xd8",               # a name of lone surrogates: not UTF-16
    bytes([3, 1]) + "a\x00".encode("utf-16-le")[:2],   # a uint with no bytes left
    bytes([76, 0]) + b"\x00" * 10 + b"not zlib",       # compressed, but not zlib
])
def test_damaged_metadata_is_a_corrupt_file(data):
    with pytest.raises(CorruptFileError, match="damaged ND2 metadata"):
        decode_clx_lite(data)


def test_good_metadata_still_decodes():
    name = "uiCount\x00".encode("utf-16-le")
    data = bytes([3, len(name) // 2]) + name + struct.pack("<I", 7)
    assert decode_clx_lite(data) == {"uiCount": 7}
