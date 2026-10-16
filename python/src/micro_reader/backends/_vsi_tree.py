"""The metadata tree of an Olympus / Evident cellSens ``.vsi`` file.

Layout as slideio reads it (Stanislav Melnikov, BSD-3: ``vsifile.cpp``
``readMetadata``, ``vsistruct.hpp``, ``vsitools.cpp``): the ``.vsi`` is a
little-endian classic TIFF, and right after its 8-byte TIFF header, at byte
8, starts a tree of *volumes*:

- a volume header (24 bytes): header size 24 (uint16), magic 21321
  (uint16), version (uint32), offset of the first field from the header
  (int64), flags (uint32; the low 28 bits count the fields), 4 unused;
- each field (16 bytes, + 4 for a second tag): field type (int32: bits 0-23
  the value type, bit 27 a second tag follows, bit 28 an extended field
  holding nested volumes, bit 30 the value is the size field itself), tag
  (int32; negative ends the volume), the next field's offset from the
  volume header (int32; 0: the last field), the data size (uint32);
- extended types: 0 one or more nested volumes in the data size, 1 / 2 one
  nested volume, 0xA a TIFF IFD link (its second tag is the IFD index),
  others nothing to read.

Nodes keep typed values (slideio turns them into 6-decimal strings; here
doubles keep their precision).  Strings are UTF-16LE, cut at the first NUL.
"""
from __future__ import annotations

import struct
from dataclasses import dataclass, field
from typing import Optional

VOLUME = struct.Struct("<HHIqII")
FIELD = struct.Struct("<iiiI")
VOLUME_MAGIC = 21321
EXTRA_TAG, EXTENDED, INLINE = 1 << 27, 1 << 28, 1 << 30
NEW_VOLUME_HEADER, PROPERTY_SET_VOLUME, NEW_MDIM_VOLUME_HEADER, TIFF_ID = 0, 1, 2, 0xA

#: scalar value types -> struct code
SCALARS = {1: "b", 2: "B", 3: "h", 4: "H", 5: "i", 6: "I", 7: "q", 8: "Q", 9: "f", 10: "d",
           12: "?", 14: "I", 17: "Q", 18: "d", 271: "I", 272: "I", 273: "I"}
#: value types holding lists of int32 / doubles
INT_LISTS = {256, 257, 258, 259, 267, 274, 275, 276, 277, 8195, 8199, 8200, 8470}
DOUBLE_LISTS = {11, 260, 261, 262, 263, 264, 265, 266, 268, 279, 280}
STRINGS = {13, 8192}
RGB_TYPES = {269: "rgb", 270: "bgr"}
#: the tree may not go deeper than this (a corrupt file must not recurse forever)
MAX_DEPTH = 64


@dataclass
class Node:
    tag: int
    code: int = 0                 # value type, or extended type for an extended field
    extended: bool = False
    second: int = -1              # the second tag (-1: none)
    value: object = None
    children: list = field(default_factory=list)

    def child(self, tag: int, second: Optional[int] = None) -> Optional["Node"]:
        for c in self.children:
            if c.tag == tag and (second is None or c.second == second):
                return c
        return None

    def all(self, tag: int) -> list:
        return [c for c in self.children if c.tag == tag]

    def path(self, *tags) -> Optional["Node"]:
        node = self
        for tag in tags:
            node = node.child(tag) if node is not None else None
        return node

    def find(self, tag: int) -> Optional["Node"]:
        """The first node with *tag* anywhere below, depth first."""
        for c in self.children:
            if c.tag == tag:
                return c
            found = c.find(tag)
            if found is not None:
                return found
        return None


def _decode(data: bytes, code: int):
    if code in SCALARS:
        fmt = "<" + SCALARS[code]
        return struct.unpack_from(fmt, data, 0)[0] if len(data) >= struct.calcsize(fmt) else None
    if code in STRINGS:
        data = data[:len(data) // 2 * 2]
        return data.decode("utf-16-le", "replace").split("\x00", 1)[0]
    if code in INT_LISTS:
        return list(struct.unpack_from(f"<{len(data) // 4}i", data, 0))
    if code in DOUBLE_LISTS:
        return list(struct.unpack_from(f"<{len(data) // 8}d", data, 0))
    if code in RGB_TYPES:
        triples = [tuple(data[k:k + 3]) for k in range(0, len(data) - 2, 3)]
        if code == 270:
            triples = [t[::-1] for t in triples]
        return triples
    return None


def read_tree(data: bytes) -> Node:
    """The tree of a ``.vsi`` file whose bytes are *data* (the root: tag -2;
    no children when there is no valid volume at byte 8)."""
    root = Node(-2)
    if data[:4] != b"II*\x00":
        return root
    _read_volume(data, 8, root, 0)
    return root


def _read_volume(data: bytes, start: int, parent: Node, depth: int):
    """Parse the volume at *start*; its fields become *parent*'s children.
    Returns the position after it, or None when the header is not a volume."""
    size = len(data)
    if depth > MAX_DEPTH or start + VOLUME.size > size:
        return None
    header_size, magic, _version, first, flags, _ = VOLUME.unpack_from(data, start)
    if header_size != 24 or magic != VOLUME_MAGIC or first < 0 or start + first >= size:
        return None
    at = start + first
    for _ in range(flags & 0x0FFFFFFF):
        if at + FIELD.size > size:
            return None
        ftype, tag, nxt, dsize = FIELD.unpack_from(data, at)
        at += FIELD.size
        second = -1
        if ftype & EXTRA_TAG:
            second = struct.unpack_from("<i", data, at)[0]
            at += 4
        if tag < 0:                                   # the end of this volume
            return None
        code = ftype & 0xFFFFFF
        node = Node(tag, code, bool(ftype & EXTENDED), second)
        if node.extended:
            if code == NEW_VOLUME_HEADER:
                end, pos = min(at + dsize, size), at
                while pos < end:
                    after = _read_volume(data, pos, node, depth + 1)
                    if after is None or after <= pos:
                        break
                    pos = after
            elif code in (PROPERTY_SET_VOLUME, NEW_MDIM_VOLUME_HEADER):
                _read_volume(data, at, node, depth + 1)
        elif ftype & INLINE:
            node.value = dsize
        elif dsize:
            node.value = _decode(data[at:at + dsize], code)
        parent.children.append(node)
        if nxt == 0:                                  # the last field
            return start + dsize + 32 if 0 <= start + dsize + 32 < size else None
        if not 0 <= start + nxt < size:
            return None
        at = start + nxt
    return at
