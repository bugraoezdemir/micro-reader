"""Rows of a JPEG too large to decode whole, decoded alone through its
restart markers (docs/PLAN.md, "memory-bounded reads of huge compressed
blocks", step 4).

A sequential JPEG with a restart interval (DRI) resets its entropy coder
every *Ri* MCUs and marks the place (RST0..RST7).  The intervals covering
some MCU rows, put behind the JPEG's own header with the frame height
changed, make a JPEG of those rows alone: any decoder decodes it, and only
it is held in memory.  Pixels are the same as a whole decode: blocks decode
alone, and one MCU row more above and below is decoded and dropped where
chroma is subsampled (upsampling looks at the neighbouring rows).

Cut only:
- baseline / extended sequential JPEG (SOF0, SOF1), one scan of all
  components;
- with intervals that start on MCU rows often enough: every
  lcm(Ri, MCUs per row) / MCUs per row MCU rows (``group``).

Progressive JPEG (its refinement scans hold the whole image), lossless and
JPEG without restart markers return None from ``RestartIndex.build`` and are
decoded whole, within the memory limit.
"""

from __future__ import annotations

import math
import re
import struct

import numpy as np

#: RST0..RST7 and EOI in entropy-coded data (0xFF 0x00 is a stuffed 0xFF)
_MARKERS = re.compile(rb"\xff[\xd0-\xd7\xd9]")


class RestartIndex:
    """Where the restart intervals of one sequential JPEG start and end, in
    bytes of its codestream; ``build`` it from the codestream."""

    def __init__(self, header: bytes, sof: int, height: int, width: int,
                 mcu_rows: int, mcu_height: int, group: int, starts, ends, subsampled: bool,
                 intervals_per_group: int):
        self.header = header                    # SOI .. the scan header, kept
        self._sof = sof                         # offset of the frame height in *header*
        self.height, self.width = height, width
        self.mcu_rows, self.mcu_height = mcu_rows, mcu_height
        #: MCU rows between interval starts that fall on an MCU row
        self.group = group
        self._per_group = intervals_per_group
        self._starts, self._ends = starts, ends
        self._overlap = 1 if subsampled else 0  # groups decoded above / below and dropped

    @classmethod
    def build(cls, data) -> "RestartIndex | None":
        """The index of the JPEG codestream *data*, or None when it cannot
        be cut (see the module docstring)."""
        data = memoryview(data).cast("B")
        if bytes(data[:2]) != b"\xff\xd8":
            return None
        pos, sof, frame, interval, scan = 2, None, None, 0, None
        while pos + 4 <= len(data):
            if data[pos] != 0xFF:
                return None
            marker = data[pos + 1]
            if marker == 0xFF:                  # fill byte
                pos += 1
                continue
            length = struct.unpack(">H", data[pos + 2:pos + 4])[0]
            body = bytes(data[pos + 4:pos + 2 + length])
            if marker in (0xC0, 0xC1):
                if frame is not None:
                    return None
                sof = pos + 5
                _bits, height, width, n = struct.unpack(">BHHB", body[:6])
                frame = (height, width, [(body[7 + 3 * k] >> 4, body[7 + 3 * k] & 15)
                                         for k in range(n)])
            elif 0xC2 <= marker <= 0xCF and marker not in (0xC4, 0xC8, 0xCC):
                return None                     # progressive, lossless, arithmetic
            elif marker == 0xDD:
                interval = struct.unpack(">H", body[:2])[0]
            elif marker == 0xDA:
                if frame is None or body[0] != len(frame[2]):
                    return None                 # a scan of only some components
                scan = pos + 2 + length
                break
            pos += 2 + length
        if scan is None or interval == 0 or frame[0] == 0:
            return None
        height, width, sampling = frame
        if len(sampling) == 1:                  # one component: 8 x 8 MCUs
            mcu_w = mcu_h = 8
        else:
            mcu_w, mcu_h = 8 * max(h for h, _ in sampling), 8 * max(v for _, v in sampling)
        across = -(-width // mcu_w)
        mcu_rows = -(-height // mcu_h)
        group = math.lcm(interval, across) // across
        markers = [(m.start(), data[m.start() + 1]) for m in _MARKERS.finditer(data, scan)]
        end = next((p for p, kind in markers if kind == 0xD9), len(data))
        rst = [p for p, kind in markers if kind != 0xD9 and p < end]
        starts = np.array([scan] + [p + 2 for p in rst], np.int64)
        ends = np.array(rst + [end], np.int64)
        if len(starts) != -(-across * mcu_rows // interval):
            return None                         # markers missing or extra: decode whole
        subsampled = len({s for s in sampling}) > 1
        return cls(bytes(data[:scan]), sof, height, width, mcu_rows, mcu_h, group,
                   starts, ends, subsampled, group * across // interval)

    def group_bytes(self, row_bytes: int) -> int:
        """Decoded bytes of one group of MCU rows, *row_bytes* per image row."""
        return self.group * self.mcu_height * row_bytes

    def smallest_band(self, row_bytes: int) -> int:
        """Memory of the smallest band ``read_rows`` decodes: one group, the
        overlap above and below, and the decoder's own planes."""
        return 2 * (1 + 2 * self._overlap) * self.group_bytes(row_bytes)


    def bands(self, y0: int, y1: int, groups: int):
        """(g0, g1): the groups of MCU rows covering image rows *y0*..*y1*,
        *groups* at a time (``codestream`` adds the overlap)."""
        rows_per_group = self.group * self.mcu_height
        g0, g1 = y0 // rows_per_group, -(-y1 // rows_per_group)
        for g in range(g0, g1, groups):
            yield g, min(g1, g + groups)

    def codestream(self, read, g0: int, g1: int):
        """The JPEG of groups *g0*..*g1* (plus the overlap) and the image row
        it starts at.  *read(start, stop)*: bytes start..stop of the
        original codestream."""
        groups = -(-self.mcu_rows // self.group)
        g0, g1 = max(0, g0 - self._overlap), min(groups, g1 + self._overlap)
        j0, j1 = g0 * self._per_group, min(len(self._starts), g1 * self._per_group)
        first = g0 * self.group * self.mcu_height
        stop = min(self.height, g1 * self.group * self.mcu_height)
        header = bytearray(self.header)
        header[self._sof:self._sof + 2] = struct.pack(">H", stop - first)
        raw = read(int(self._starts[j0]), int(self._ends[j1 - 1]))
        base = int(self._starts[j0])
        parts = [bytes(header)]
        for j in range(j0, j1):
            parts.append(raw[self._starts[j] - base:self._ends[j] - base])
            if j < j1 - 1:
                parts.append(bytes((0xFF, 0xD0 + (j - j0) % 8)))
        parts.append(b"\xff\xd9")
        return b"".join(parts), first, stop


def usable(index: "RestartIndex | None", row_bytes: int, fits) -> bool:
    """Whether *index* cuts its JPEG in bands that *fits(nbytes)* allows:
    intervals that seldom start on an MCU row make every band large."""
    return index is not None and fits(index.smallest_band(row_bytes))


def read_rows(index: RestartIndex, read, y0: int, y1: int, decode, shape_tail, dtype,
              piece_bytes: int, reserve):
    """Image rows *y0*..*y1* of a cut JPEG, in bands: yields (a, b, rows),
    rows *a*..*b* of the image as (b - a, width, *shape_tail*).  *decode(data,
    out)* decodes a JPEG into *out*; *reserve(nbytes)* holds the memory
    budget while a band is decoded (the band and the decoder's own planes)."""
    row_bytes = index.width * int(np.prod(shape_tail, dtype=np.int64)) * np.dtype(dtype).itemsize
    groups = max(1, piece_bytes // max(1, index.group_bytes(row_bytes)))
    for g0, g1 in index.bands(y0, y1, groups):
        data, first, stop = index.codestream(read, g0, g1)
        band = np.empty((stop - first, index.width) + tuple(shape_tail), dtype)
        with reserve(2 * band.nbytes):                          # the band and the decoder's
            got = decode(data, band)
            if (got[1], got[0]) != band.shape[:2]:
                raise ValueError(f"a cut JPEG decodes to {got[1]} x {got[0]}, "
                                 f"expected {band.shape[0]} x {band.shape[1]}")
            rows_per_group = index.group * index.mcu_height       # the band's own rows,
            a = max(y0, g0 * rows_per_group)                      # not the overlap
            b = min(y1, g1 * rows_per_group, index.height)
            if a < b:
                yield a, b, band[a - first:b - first]
