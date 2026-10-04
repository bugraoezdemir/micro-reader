"""numpy basic indexing, reduced to what a storage backend can serve.

Backends only ever read contiguous, unit-step spans.  Integer indices and
steps are applied to the result in memory afterwards, so every backend gets
numpy semantics for free.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np

Spans = Tuple[Tuple[int, int], ...]


def normalise_index(key, shape) -> tuple[Spans, tuple[int, ...], tuple[int, ...]]:
    """numpy basic-index *key* -> (per-axis (start, stop), axes to squeeze, steps).

    Supports ints (negative too), slices with positive steps, and a single
    Ellipsis.  ``None`` / newaxis, negative steps and fancy indexing raise:
    they have no meaning at the storage layer.
    """
    if not isinstance(key, tuple):
        key = (key,)
    if sum(k is Ellipsis for k in key) > 1:
        raise IndexError("an index can only have a single ellipsis ('...')")
    if any(k is Ellipsis for k in key):
        at = next(i for i, k in enumerate(key) if k is Ellipsis)
        key = key[:at] + (slice(None),) * (len(shape) - len(key) + 1) + key[at + 1:]
    if len(key) > len(shape):
        raise IndexError(f"too many indices: array is {len(shape)}-D, got {len(key)}")
    key = key + (slice(None),) * (len(shape) - len(key))

    spans, squeeze, steps = [], [], []
    for axis, (k, size) in enumerate(zip(key, shape)):
        if isinstance(k, (int, np.integer)) and not isinstance(k, bool):
            index = int(k) + size if k < 0 else int(k)
            if not 0 <= index < size:
                raise IndexError(
                    f"index {k} is out of bounds for axis {axis} with size {size}")
            spans.append((index, index + 1))
            squeeze.append(axis)
            steps.append(1)
        elif isinstance(k, slice):
            start, stop, step = k.indices(size)
            if step < 0:
                raise IndexError("negative steps are not supported")
            spans.append((start, max(start, stop)))
            steps.append(step)
        else:
            raise IndexError(f"unsupported index type {type(k).__name__}; "
                             "use integers, slices or '...'")
    return tuple(spans), tuple(squeeze), tuple(steps)
