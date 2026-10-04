"""numpy basic indexing, reproduced through contiguous spans."""
import numpy as np
import pytest

from micro_reader._index import normalise_index

SHAPE = (3, 5, 7)
DATA = np.arange(np.prod(SHAPE)).reshape(SHAPE)


def _via_spans(key):
    spans, squeeze, steps = normalise_index(key, SHAPE)
    out = DATA[tuple(slice(a, b) for a, b in spans)]
    if any(s != 1 for s in steps):
        out = out[tuple(slice(None, None, s) for s in steps)]
    return out.squeeze(axis=squeeze) if squeeze else out


@pytest.mark.parametrize("key", [
    0, -1, (1, 2), (1, 2, 3), (-1, -2, -3), ..., (Ellipsis, 3), (0, Ellipsis),
    slice(None), (slice(1, 3), slice(None, None, 2)), (slice(0, 5, 3), 4, slice(2, 7)),
    (slice(5, 2),), (slice(None), slice(10, 20)), (np.int64(2), slice(1, 4)),
])
def test_matches_numpy(key):
    np.testing.assert_array_equal(_via_spans(key), DATA[key])


@pytest.mark.parametrize("key", [
    (0, 0, 0, 0),                 # too many indices
    3,                            # out of bounds
    np.newaxis,                   # new axes have no meaning in storage
    slice(None, None, -1),        # negative steps
    [0, 1],                       # fancy indexing
    (Ellipsis, Ellipsis),         # two ellipses
    True,                         # booleans are not integers here
])
def test_rejects_what_storage_cannot_serve(key):
    with pytest.raises(IndexError):
        normalise_index(key, SHAPE)
