"""A minimal Micro-Manager NDTiff (v3) writer for tests: uncompressed 16-bit
frames in one or more classic TIFF files plus ``NDTiff.index``, laid out as
tifffile reads them (``read_micromanager_metadata``, ``read_ndtiff_index``).
"""
from __future__ import annotations

import json
import os
import struct

import numpy as np

NDTIFF_HEADER = 483729
SUMMARY_HEADER = 2355492
MICROMANAGER_TAG = 51123


def _tiff_file(path, frames, summary: dict, metadata=None) -> list:
    """*frames*: (height, width) uint16 arrays -> one TIFF; *metadata*: one
    JSON dict per frame (default {}).  Returns each frame's (data offset,
    metadata offset, metadata length)."""
    summary_json = json.dumps(summary).encode()
    out = bytearray(b"II*\x00" + struct.pack("<I", 0))
    # NDTiff v3 header at offset 8: major version, minor version + summary
    out += struct.pack("<IIIII", NDTIFF_HEADER, 3, 0, SUMMARY_HEADER, len(summary_json))
    out += summary_json
    if len(out) % 2:
        out += b"\x00"
    link_at, offsets = 4, []
    for k, frame in enumerate(frames):
        height, width = frame.shape
        data_at = len(out)
        out += frame.astype("<u2").tobytes()
        md = json.dumps(metadata[k] if metadata else {}).encode() + b"\x00"
        md_at = len(out)
        offsets.append((data_at, md_at, len(md)))
        out += md
        if len(out) % 2:
            out += b"\x00"
        tags = [(256, 4, 1, width), (257, 4, 1, height), (258, 3, 1, 16), (259, 3, 1, 1),
                (262, 3, 1, 1), (273, 4, 1, data_at), (277, 3, 1, 1), (278, 4, 1, height),
                (279, 4, 1, frame.size * 2), (MICROMANAGER_TAG, 2, len(md), md_at)]
        ifd_at = len(out)
        struct.pack_into("<I", out, link_at, ifd_at)
        out += struct.pack("<H", len(tags))
        for tag, typ, count, value in tags:
            if typ == 3:
                out += struct.pack("<HHIHH", tag, typ, count, value, 0)
            else:
                out += struct.pack("<HHII", tag, typ, count, value)
        link_at = len(out)
        out += struct.pack("<I", 0)
    with open(path, "wb") as fh:
        fh.write(out)
    return offsets


def write_ndtiff(directory, frames: list, *, per_file: int = None, prefix="data_NDTiffStack",
                 frame_metadata=None):
    """*frames*: (axes dict, e.g. {"position": 0, "time": 1, "channel": "DAPI",
    "z": 2}; (height, width) uint16) pairs -> TIFF files of *per_file* frames
    each (all in one by default) and ``NDTiff.index``; *frame_metadata*: the
    JSON dict written for every frame (e.g. {"PixelSizeUm": 0.65}), its offset
    and length in the index.  Returns the TIFF paths."""
    os.makedirs(directory, exist_ok=True)
    items = list(frames)
    per_file = per_file or len(items)
    summary = {"PixelType": "GRAY16"}
    index, paths = bytearray(), []
    for n, start in enumerate(range(0, len(items), per_file)):
        chunk = items[start:start + per_file]
        name = f"{prefix}.tif" if n == 0 else f"{prefix}_{n}.tif"
        path = os.path.join(directory, name)
        metadata = [frame_metadata or {} for _ in chunk]
        offsets = _tiff_file(path, [np.asarray(f) for _, f in chunk], summary, metadata)
        paths.append(path)
        for (axes, frame), (offset, md_at, md_len) in zip(chunk, offsets):
            axes_json = json.dumps(axes).encode()
            filename = name.encode()
            height, width = np.asarray(frame).shape
            index += struct.pack("<i", len(axes_json)) + axes_json
            index += struct.pack("<i", len(filename)) + filename
            index += struct.pack("<IiiiiIii", offset, width, height, 1, 0, md_at, md_len, 0)
    with open(os.path.join(directory, "NDTiff.index"), "wb") as fh:
        fh.write(index)
    return paths
