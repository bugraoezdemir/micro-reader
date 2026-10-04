"""EER (Electron Event Representation) encoding for tests.

A stream of little-endian bit codes: ``skipbits`` counting pixels without an
electron (all ones: skip that many, no electron, go on), then -- unless all
ones -- ``horzbits`` + ``vertbits`` sub-pixel bits of the electron at the
next pixel.  imagecodecs' ``eer_decode`` reads what this writes (checked in
the tests), which validates the encoder.
"""
from __future__ import annotations

import numpy as np


def eer_encode(events, pixelsize: int, skipbits: int = 7, horzbits: int = 2,
               vertbits: int = 2) -> bytes:
    """*events*: sorted (pixel index, horizontal code, vertical code), at
    most one per pixel; *pixelsize*: pixels of the (base resolution) frame."""
    codes = []                                           # (value, bits)
    skipmask = (1 << skipbits) - 1
    current = 0
    for pixel, h, v in events:
        skip = pixel - current
        if skip < 0:
            raise ValueError("events must be sorted, one per pixel")
        while skip >= skipmask:
            codes.append((skipmask, skipbits))
            skip -= skipmask
        codes += [(skip, skipbits), (h, horzbits), (v, vertbits)]
        current = pixel + 1
    skip = pixelsize - current                           # run to the end: stops the decoder
    while skip >= skipmask:
        codes.append((skipmask, skipbits))
        skip -= skipmask
    codes.append((skip, skipbits))
    nbits = sum(b for _, b in codes)
    # the decoder reads a code only while nbits of room remain: pad, even length
    nbytes = (nbits + 7) // 8 + (skipbits + horzbits + vertbits + 7) // 8 + 1
    nbytes += nbytes % 2
    acc, pos = 0, 0
    for value, bits in codes:
        acc |= (value & ((1 << bits) - 1)) << pos
        pos += bits
    return acc.to_bytes(nbytes, "little")


def random_events(rng, height, width, density=0.05, horzbits=2, vertbits=2):
    """Random electrons (one per pixel at most) with random sub-pixel codes,
    and their counts per pixel."""
    pixels = np.flatnonzero(rng.random(height * width) < density)
    h = rng.integers(0, 1 << horzbits, pixels.size)
    v = rng.integers(0, 1 << vertbits, pixels.size)
    counts = np.zeros(height * width, np.uint8)
    counts[pixels] = 1
    return list(zip(pixels.tolist(), h.tolist(), v.tolist())), counts.reshape(height, width)
