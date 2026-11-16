"""``read_many``: the regions of many images read in one go, each into an array
the caller gives (Workplan MR23, joins).

A join of many files (eubi's z stack of 768 one-plane TIFFs, through dyna's
concatenation) used to read a region file by file: one Python read per file,
each with its own small engine call, then a copy of all parts into one array.
Here the reads of all files are planned first, straight into their places in the
caller's array, and run together: one engine call for every file's strips and
tiles, decoded on all threads with the GIL released, and no copy afterwards.
"""
from __future__ import annotations

import contextlib

import numpy as np

from ._index import normalise_index


def _root(array: np.ndarray) -> np.ndarray:
    """The array whose memory *array* is a view of."""
    while isinstance(array.base, np.ndarray):
        array = array.base
    return array


def read_many(requests) -> None:
    """Read many images' regions in one go, each into its own array.

    *requests*: ``(image, key, out)`` triples. *key* is any index ``image[key]``
    takes, and *out* a writable NumPy array of the shape and dtype ``image[key]``
    returns; ``out`` receives what ``image[key]`` would return.

    The images may belong to many files. All their strips and tiles are read and
    decoded together, on all decode threads, and written straight into the
    arrays: for a region of a stack of one-plane files, give one array for the
    region and, as each request's *out*, its slice for that file::

        region = np.empty((len(images), 1024, 1024), dtype)
        micro_reader.read_many([(image, (0, 0, 0, ys, xs), region[k])
                                for k, image in enumerate(images)])

    Images whose reads cannot be planned this way (formats the read engine does
    not handle yet, stepped keys) are read one by one into their arrays, so the
    result is always the same as ``out[...] = image[key]`` for each request.
    Each image's ``stats`` count its read as for ``image[key]``.
    """
    from . import _engine
    groups: dict = {}                        # id(root array) -> (root, [(image, n, plans)])
    one_by_one = []
    with contextlib.ExitStack() as stack:
        for image, key, out in requests:
            if not isinstance(out, np.ndarray):
                raise TypeError(f"read_many: out must be a NumPy array, not "
                                f"{type(out).__name__}")
            if not out.flags.writeable:
                raise ValueError("read_many: out is read-only")
            if out.dtype != image.dtype:
                raise ValueError(f"read_many: out has dtype {out.dtype}; {image.name!r} "
                                 f"reads as {image.dtype}")
            spans, squeeze, steps = normalise_index(key, image.shape)
            shape = tuple(len(range(a, b, step)) for axis, ((a, b), step)
                          in enumerate(zip(spans, steps)) if axis not in squeeze)
            if out.shape != shape:
                raise ValueError(f"read_many: out has shape {out.shape}; {image.name!r}"
                                 f"[{key!r}] reads as {shape}")
            requested = int(np.prod([b - a for a, b in spans], dtype=np.int64))
            root = _root(out)
            if requested == 0:
                image.stats.add(0, 0)
                continue
            if any(step != 1 for step in steps) or not (
                    root.flags.c_contiguous or root.flags.f_contiguous):
                one_by_one.append((image, key, out))
                continue
            block = tuple(slice(a, b) for a, b in spans)
            plans = image._plan_into(block, np.expand_dims(out, squeeze) if squeeze else out,
                                     stack)
            if plans is None:
                one_by_one.append((image, key, out))
                continue
            groups.setdefault(id(root), (root, []))[1].append((image, requested, plans))
        for root, items in groups.values():
            flat = [plan for _, _, plans in items for plan in plans]
            counts = [0] * len(flat)
            _engine.run_plans(flat, root, counts)
            at = 0
            for image, requested, plans in items:
                image.stats.add(requested, sum(counts[at:at + len(plans)]))
                at += len(plans)
    for image, key, out in one_by_one:
        out[...] = image[key]
