"""Synthetic Hamamatsu NDPIS sets (tests/test_ndpis.py): an ``.ndpis`` INI and
one member per channel.  A member is a pyramidal RGB TIFF carrying NDPI's
channel tags (65434 filter set, 65451 emission wavelength), the signal in the
band of its wavelength -- what the NDPIS backend reads.  (Real ``.ndpi``
files use NDPI's 64-bit offset layout, which tifffile does not write; the
members are named ``.tif``.)"""
from __future__ import annotations

import os

import numpy as np
import tifffile


def write_member(path, band, name, wavelength, shape=(96, 128), seed=0, extra=True):
    """-> the RGB level 0 written (signal in *band*, a little in the others)."""
    rng = np.random.default_rng(seed)
    base = rng.integers(0, 6, shape + (3,)).astype(np.uint8)          # JPEG-like bleed
    base[..., band] = rng.integers(0, 255, shape)
    tags = [(65434, "s", 0, name, True), (65451, "f", 1, float(wavelength), True)]
    with tifffile.TiffWriter(path) as tw:
        tw.write(base, photometric="rgb", tile=(32, 32), subifds=1, extratags=tags)
        tw.write(base[::2, ::2].copy(), photometric="rgb", tile=(32, 32), subfiletype=1)
        if extra:                                                       # a "macro"
            tw.write(np.full((20, 60, 3), 200, np.uint8), photometric="rgb")
    return base


def write_set(folder, channels=(("DAPI 2 (387)", 455.0, 2), ("FITC 2 (485)", 519.0, 1),
                                ("TRITC 2 (560)", 622.0, 0)), name="test3", skip=(), shapes=None):
    """``<name>.ndpis`` + one member per (channel name, wavelength, band).
    -> (ndpis path, [level-0 RGB per channel], [member paths])."""
    os.makedirs(folder, exist_ok=True)
    bases, paths = [], []
    for k, (cname, wave, band) in enumerate(channels):
        path = os.path.join(folder, f"{name}-{cname}.tif")
        paths.append(path)
        shape = shapes[k] if shapes else (96, 128)
        bases.append(write_member(path, band, cname, wave, shape=shape, seed=k))
        if k in skip:
            os.remove(path)
    lines = ["[NanoZoomer Digital Pathology Image Set]", f"NoImages={len(channels)}"]
    lines += [f"Image{k}={os.path.basename(p)}" for k, p in enumerate(paths)]
    ndpis = os.path.join(folder, name + ".ndpis")
    with open(ndpis, "w", encoding="utf-8", newline="") as fh:
        fh.write("\r\n".join(lines) + "\r\n")
    return ndpis, bases, paths
