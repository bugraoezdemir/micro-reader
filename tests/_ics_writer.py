"""Synthetic ICS / IDS files (tests/test_ics.py), written as libics and SVI
Huygens write them: a tab / newline header, x fastest."""
from __future__ import annotations

import gzip
import os

import numpy as np

#: numpy axis letter -> the ICS order name
ICS_NAMES = {"x": "x", "y": "y", "z": "z", "t": "t", "c": "ch"}


def header(data: np.ndarray, axes: str, *, version="1.0", coordinates="video", compression="uncompressed",
           scale=None, units=None, extra_dims=(), sensor=(), bare=False, filename="image",
           order_names=None) -> str:
    """The header text for *data* with numpy *axes* (slowest first).  *extra_dims*:
    (name, 1) dimensions appended after the others (Huygens' ``p``); *scale* /
    *units*: per numpy axis letter; *sensor*: extra ``sensor`` lines (lists of
    fields); *bare*: no separator line and no ``ics_version`` (OME's gh-4197)."""
    dims = [(order_names or {}).get(a, ICS_NAMES[a]) for a in reversed(axes)] + [n for n, _ in extra_dims]
    sizes = list(reversed(data.shape)) + [s for _, s in extra_dims]
    item = data.dtype.itemsize
    kind = "real" if data.dtype.kind == "f" else "integer"
    sign = "signed" if data.dtype.kind in "if" else "unsigned"
    little = data.dtype.byteorder in "<=|" if item > 1 else True
    byte_order = list(range(1, item + 1)) if little else list(range(item, 0, -1))
    sc = [1.0] + [(scale or {}).get(a, 1.0) for a in reversed(axes)] + [1.0] * len(extra_dims)
    un = ["bits"] + [(units or {}).get(a, "undefined") for a in reversed(axes)] + [""] * len(extra_dims)
    lines = [] if bare else [f"ics_version\t{version}", f"filename\t{filename}"]
    lines += [f"layout\tparameters\t{len(dims) + 1}",
              "layout\torder\tbits\t" + "\t".join(dims),
              f"layout\tsizes\t{item * 8}\t" + "\t".join(str(s) for s in sizes),
              f"layout\tcoordinates\t{coordinates}",
              f"representation\tformat\t{kind}",
              f"representation\tsign\t{sign}",
              f"representation\tcompression\t{compression}",
              "representation\tbyte_order\t" + "\t".join(str(b) for b in byte_order),
              "parameter\tscale\t" + "\t".join(f"{v:.6f}" for v in sc),
              "parameter\tunits\t" + "\t".join(un)]
    lines += ["\t".join(["sensor"] + list(s)) for s in sensor]
    lines += ["history\tsoftware\tSVI-Huygens", "end"]
    return ("" if bare else "\t\n") + "\n".join(lines) + "\n"


def stored(data: np.ndarray, coordinates: str) -> bytes:
    """The pixel bytes: rows bottom-up for ``cartesian``."""
    if coordinates == "cartesian":
        data = data[..., ::-1, :]
    return np.ascontiguousarray(data).tobytes()


def write_ics(folder, data: np.ndarray, axes: str, name="image", *, version="1.0", coordinates="video",
              compression="uncompressed", side_files=(), ids_name=None, **kw) -> str:
    """Write ``<name>.ics`` (+ ``<name>.ids`` for version 1.0); *data* in display
    orientation.  -> the .ics path."""
    os.makedirs(folder, exist_ok=True)
    ics = os.path.join(folder, name + ".ics")
    head = header(data, axes, version=version, coordinates=coordinates, compression=compression,
                  filename=name, **kw).encode("latin-1")
    pixels = stored(data, coordinates)
    if compression == "gzip":
        pixels = gzip.compress(pixels)
    with open(ics, "wb") as fh:
        fh.write(head)
        if version.startswith("2"):
            fh.write(pixels)
    if version.startswith("1"):
        with open(os.path.join(folder, ids_name or name + ".ids"), "wb") as fh:
            fh.write(pixels)
    for suffix in side_files:
        with open(os.path.join(folder, name + suffix), "w", encoding="utf-8") as fh:
            fh.write("Huygens Remote Manager side file\n")
    return ics
