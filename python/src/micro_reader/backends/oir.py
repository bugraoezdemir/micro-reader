"""Olympus / Evident OIR (FluoView), own reader, numpy and stdlib only.

Structure: an ``OLYMPUSRAWFORMAT`` header points to a block index at the end
of the file.  Blocks: a UID block names a plane --
``[l<n>][z<n>][t<n>][_<a>_<b>]_<channel guid>_<block>`` -- and is followed by
its PIXEL block (raw rows, little endian); FRAMEPROPERTIES XML gives width,
height and bytes per sample; METADATA XML (``lsmimage``) the channel order
and the acquisition axes.  A plane is the concatenation of its pixel blocks
in index order, possibly continued in companion files ``<name>_00001``,
``<name>_00002``, ...

Reading is exact: any rows / columns of a plane are read straight from the
blocks holding those bytes.  One image, axes ``tczyx``; planar RGB appears as
channels; a lambda (spectral) scan is one image per emission band (index
``emission``, its wavelength in ``metadata.index_values``).
Not yet: line scans, POIR archives.

The block-index parsing is ported, reduced, from oirfile
(https://github.com/cgohlke/oirfile):

    Copyright (c) 2025-2026, Christoph Gohlke
    All rights reserved.

    Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions are met:

    1. Redistributions of source code must retain the above copyright notice,
       this list of conditions and the following disclaimer.
    2. Redistributions in binary form must reproduce the above copyright
       notice, this list of conditions and the following disclaimer in the
       documentation and/or other materials provided with the distribution.
    3. Neither the name of the copyright holder nor the names of its
       contributors may be used to endorse or promote products derived from
       this software without specific prior written permission.

    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
    IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
    THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
    PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
    CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
    EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
    PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
    PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
    LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
    NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
    SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""
from __future__ import annotations

import bisect
import copy
import builtins
import itertools
import os
import re
import struct
import sys
import warnings
from xml.etree import ElementTree

import numpy as np

from .. import _parts, _pool
from .._base import File, Image
from .._errors import CorruptFileError, MissingImagesWarning, UnsupportedFormatError
from .._metadata import Channel, Metadata, color_hex, quantity
from .._io import ThreadHandles

NAME = "oir"
MAGIC = b"OLYMPUSRAWFORMAT"
RAW_TASK_BYTES = 8 << 20          # planes are read in row pieces of this size
BLOCK_METADATA, BLOCK_FRAMEPROPERTIES, BLOCK_UID, BLOCK_PIXEL = 0, 1, 3, 4
UID_PATTERN = re.compile(
    r"(?:l(\d+))?(?:z(\d+))?(?:t(\d+))?(?:_(\d+)_(\d+))?_([0-9a-f-]+)_(\d+)$")
NATIVE_LITTLE = sys.byteorder == "little"


#: a frame's id: l<lambda>z<z>t<t> (each part where the image has the axis)
FRAME_ID = re.compile(r"(?:l(\d+))?(?:z(\d+))?(?:t(\d+))?(?:_|$)")
FRAME_AXIS = re.compile(r"<commonframe:axisType>(\w+)</commonframe:axisType>\s*"
                        r"<commonframe:position>([^<]*)</commonframe:position>")


def _frame_positions(frames: list) -> dict:
    """frame id -> {axis type: position} from the frame-properties XMLs."""
    out = {}
    for text in frames:
        match = re.search(r'frameProperties[^>]*\sid="([^"]+)"', text)
        if not match:
            continue
        values = {}
        for axis, value in FRAME_AXIS.findall(text):
            try:
                values[axis] = float(value)
            except ValueError:
                pass
        out.setdefault(match.group(1), values)
    return out


#: a channel guid, then its lookup table
LUT_PATTERN = re.compile(rb"([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})"
                         rb"[^<]{0,64}<\?xml[^>]*>\s*(<lut:LUT.*?</lut:LUT>)", re.S)
LUT_CONTRAST = re.compile(rb"<lut:contrast>([^<]*)</lut:contrast>")
#: OIR pixelUnit -> unit
OIR_UNITS = {"MICRO_METER": "micrometer", "NANO_METER": "nanometer", "MILLI_METER": "millimeter",
             "METER": "meter", "PICO_METER": "picometer"}


def sniff(head: bytes, path: str) -> bool:
    return head[:16] == MAGIC


#: a continuation file: <stem>_00001, <stem>_00002, ... (five digits, no extension)
CONTINUATION = re.compile(r"^(?P<stem>.+)_(?P<n>\d{5})$")


def may_own(folder: str, name: str, folders) -> bool:
    """``group_files``' cheap test (reads nothing): ``<stem>_NNNNN`` may be a
    continuation."""
    return CONTINUATION.match(name) is not None


def owner(path: str, head):
    """A continuation file opens its main file ``<stem>.oir``
    (docs/COMPOSITE.md); it is no image on its own."""
    if head is None or head[:16] != MAGIC:
        return None
    match = CONTINUATION.match(os.path.basename(path))
    if match is None:
        return None
    name = match.group("stem") + ".oir"
    folder = os.path.dirname(path)
    return _parts.Part(_parts.find(folder, name), os.path.join(folder, name))


def _continuations(path: str) -> list:
    """The continuation files of main file *path* (``<stem>_00001``, ...,
    any letter case), in number order.  A gap -- ``_00002`` missing while
    ``_00003`` exists -- warns (``MissingImagesWarning``): the frames the
    missing file held are not read; the files after it still are (every
    frame says where it belongs)."""
    folder, stem = os.path.dirname(path), os.path.splitext(os.path.basename(path))[0]
    try:
        names = os.listdir(folder or ".")
    except OSError:
        return []
    numbered = {}
    for name in sorted(names):
        match = CONTINUATION.match(name)
        if match is not None and match["stem"].lower() == stem.lower() and int(match["n"]) > 0:
            numbered.setdefault(int(match["n"]), os.path.join(folder, name))
    missing = [n for n in range(1, max(numbered, default=0) + 1) if n not in numbered]
    if missing:
        shown = ", ".join(f"{stem}_{n:05d}" for n in missing[:5]) + \
            (f", ... ({len(missing)} in all)" if len(missing) > 5 else "")
        warnings.warn(MissingImagesWarning(
            f"{path!r}: continuation file{'s' if len(missing) > 1 else ''} {shown} "
            f"{'are' if len(missing) > 1 else 'is'} missing: the frames {'they' if len(missing) > 1 else 'it'} "
            f"held are not read; the {len(numbered)} other continuation files are"), stacklevel=4)
    return [numbered[n] for n in sorted(numbered)]


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _scan(path: str, file_index: int):
    """Blocks of one OIR file -> ([(uid, file_index, pixel offset, length)],
    [frame-properties XML], [lsmimage XML], {channel guid: colour}).  A
    channel's colour is its lookup table's (``lut:LUT``, after the guid):
    the contrast of its red, green and blue components."""
    pairs, frames, lsm, luts = [], [], [], {}
    with builtins.open(path, "rb") as fh:
        head = fh.read(48)
        if len(head) < 48 or head[:16] != MAGIC:
            raise CorruptFileError(f"{path!r} is not an OIR file")
        file_size, index_pos = struct.unpack("<QQ", head[32:48])
        fh.seek(index_pos)
        index = fh.read(file_size - index_pos)
        offsets = []
        for (offset,) in struct.iter_unpack("<Q", index[4:4 + (len(index) - 4) // 8 * 8]):
            if offset == 0 or offset > file_size:
                break
            offsets.append(offset)

        def read_at(offset, size):
            fh.seek(offset)
            return fh.read(size)

        i = 0
        while i < len(offsets):
            header = read_at(offsets[i], 20)
            if len(header) < 8:
                break
            length, kind = struct.unpack("<II", header[:8])
            if kind == BLOCK_UID and len(header) == 20 and i + 1 < len(offsets):
                uid_len, = struct.unpack("<I", header[16:20])
                uid = read_at(offsets[i] + 20, uid_len).decode("ascii")
                plen, ptype = struct.unpack("<II", read_at(offsets[i + 1], 8))
                if ptype == BLOCK_PIXEL:
                    pairs.append((uid, file_index, offsets[i + 1] + 8, plen))
                    i += 2
                    continue
            elif kind == BLOCK_FRAMEPROPERTIES:
                content = read_at(offsets[i] + 8, length)
                start = content.find(b"<?xml")
                if start >= 0:
                    frames.append(content[start:].decode("utf-8"))
            elif kind == BLOCK_METADATA:
                content = read_at(offsets[i] + 8, length)
                for piece in content.split(b"<?xml")[1:]:
                    xml = b"<?xml" + piece[:piece.rfind(b">") + 1]
                    if b"lsmimage:" in xml[:200]:
                        lsm.append(xml.decode("utf-8", "replace"))
                for match in LUT_PATTERN.finditer(content):
                    guid, body = match.group(1).decode("ascii"), match.group(2)
                    rgb = [LUT_CONTRAST.search(body, body.find(b"<lut:" + c + b">"))
                           for c in (b"red", b"green", b"blue")]
                    if all(rgb):
                        luts.setdefault(guid, tuple(min(1.0, max(0.0, float(m.group(1))))
                                                    for m in rgb))
            i += 1
    return pairs, frames, lsm, luts


class _OirImage(Image):
    """An image of one OIR file (with its companion files): axes 'tczyx'; a
    lambda (spectral) scan has one image per emission band (``at_lambda``)."""

    def __init__(self, path: str):
        pairs, frames, lsm, self._luts = _scan(path, 0)
        if not frames or not pairs:
            raise UnsupportedFormatError(f"{path!r}: OIR file without image data")
        # continuation files: <name>_00001, <name>_00002, ... (no extension)
        self._paths = [path]
        for more_path in _continuations(path):
            more, _, _, _ = _scan(more_path, len(self._paths))
            self._paths.append(more_path)
            pairs.extend(more)

        frame = ElementTree.fromstring(frames[0])
        found = {}
        for elem in frame.iter():
            if elem.text and _local(elem.tag) in ("width", "height", "depth", "colorType"):
                found.setdefault(_local(elem.tag), elem.text.strip())
        width, height = int(found.get("width", 0)), int(found.get("height", 0))
        depth = int(found.get("depth", 0))
        if not (width and height and depth in (1, 2, 4)):
            raise UnsupportedFormatError(f"{path!r}: OIR frame {found}")
        self._file_dtype = np.dtype({1: "<u1", 2: "<u2", 4: "<f4"}[depth])

        channel_order, axes = self._lsm_info(lsm)
        self._lsm = lsm[0] if lsm else None
        #: frame id (``l001z001t002``) -> {axis type: position}: TIMELAPSE in
        #: ms since the start, LAMBDA in nm, ZSTACK in um
        self._frame_positions = _frame_positions(frames)

        planes: dict = {}
        for uid, file_index, offset, length in pairs:
            match = UID_PATTERN.match(uid)
            if match is None:                     # e.g. REF_ reference images
                continue
            l_str, z_str, t_str, _a, _b, guid, _blk = match.groups()
            key = (int(l_str or 0), int(t_str or 0), int(z_str or 0), guid)
            planes.setdefault(key, []).append((file_index, offset, length))

        lambdas = sorted({lam for lam, _, _, _ in planes})
        times = sorted({t for _, t, _, _ in planes})
        zs = sorted({z for _, _, z, _ in planes})
        guids = {g for _, _, _, g in planes}
        channels = [g for g in channel_order if g in guids] + sorted(guids - set(channel_order))
        self._row_bytes = width * depth
        plane_bytes = height * self._row_bytes
        if any(sum(n for _, _, n in blocks) > plane_bytes for blocks in planes.values()):
            raise UnsupportedFormatError(
                f"{path!r}: OIR planes longer than one frame (line scan / concatenated scan)")
        if "TIMELAPSE" in axes and len(times) == 1 and \
                int(axes["TIMELAPSE"].get("maxSize", -1)) == height:
            raise UnsupportedFormatError(f"{path!r}: OIR line scan (kymograph)")

        # plane (l, t, c, z) -> blocks and their start within the plane's bytes
        self._planes = {}
        for (lam, t, z, guid), blocks in planes.items():
            starts = list(itertools.accumulate([0] + [n for _, _, n in blocks]))
            self._planes[(lambdas.index(lam), times.index(t), channels.index(guid),
                          zs.index(z))] = (blocks, starts)
        self._handles = [ThreadHandles(p) for p in self._paths]
        self._channel_ids = channels
        self._ids = (lambdas, times, zs)                # file indices per axis
        self._closed = False
        #: which emission band of a lambda scan this image is (0: the only one)
        self._lam = 0
        shape = (len(times), len(channels), len(zs), height, width)
        super().__init__(0, "tczyx", shape, self._file_dtype, name=os.path.basename(path))

    def at_lambda(self, k: int) -> "_OirImage":
        """This file's image of emission band *k* (index ``emission``), sharing
        the parsed structure and file handles."""
        image = copy.copy(self)
        image._lam, image._position = k, k
        image.index = {"scene": 0, "emission": k}
        image._metadata = None
        image.stats = type(self.stats)()
        return image

    def _read_metadata(self):
        """From the lsmimage XML: per channel (``imageInfo``) the pixel
        ``length`` x / y in its ``pixelUnit``, the name, the detection band
        (emission: its middle) and the laser (excitation: the wavelength in
        its name, e.g. LD405), the LUT colour; z: the step of the enabled Z
        stack.  From the frames' own axis positions: the time step (ms) and
        the wavelength of each lambda index (D5).  Stage x / y: not in OIR
        files (a tiling's layout is in the experiment's .omp2info)."""
        if not self._lsm:
            return None
        root = ElementTree.fromstring(self._lsm)
        by_id = {}
        for elem in root.iter():
            if _local(elem.tag) == "channel" and elem.get("id") and elem.get("id") not in by_id:
                if any(_local(c.tag) == "length" for c in elem):
                    by_id[elem.get("id")] = elem
        sizes, channels = {}, []
        for guid in self._channel_ids:
            elem = by_id.get(guid)
            if elem is None:
                channels.append(Channel())
                continue
            parts = {_local(c.tag): c for c in elem}
            if not sizes and "length" in parts:
                units = {_local(u.tag): (u.text or "").strip() for u in parts.get("pixelUnit", [])}
                for v in parts["length"]:
                    axis = _local(v.tag)
                    if axis in ("x", "y"):
                        q = quantity(v.text, OIR_UNITS.get(units.get(axis), units.get(axis)))
                        if q:
                            sizes[axis] = q
            band = [float(w.text) for w in elem.iter()
                    if _local(w.tag) in ("startWavelength", "endWavelength") and w.text]
            laser = parts.get("laserDataId")
            match = re.match(r"\D*(\d{3,4})", laser.text or "") if laser is not None else None
            name = parts.get("name")
            rgb = self._luts.get(guid)
            channels.append(Channel(
                name=(name.text or "").strip() or None if name is not None else None,
                color=color_hex(rgb) if rgb and rgb != (1.0, 1.0, 1.0) else None,
                emission_wavelength=quantity(sum(band[:2]) / 2, "nanometer") if len(band) >= 2 else None,
                excitation_wavelength=quantity(match.group(1), "nanometer") if match else None))
        lambdas, times, zs = self._ids
        lam = lambdas[self._lam]
        index_values = {}
        if len(lambdas) > 1:
            value = self._frame_position(lam, times[0], zs[0], "LAMBDA")
            q = quantity(value, "nanometer") if value is not None else None
            if q:
                index_values["emission"] = q
        if len(times) > 1:
            # the frames' own times (ms since the start; their timestamps agree),
            # not the time-lapse step (the interval asked for, in s, 0 for
            # free run) nor seriesInterval (the time one series takes)
            t0 = self._frame_position(lam, times[0], zs[0], "TIMELAPSE")
            t1 = self._frame_position(lam, times[1], zs[0], "TIMELAPSE")
            if t0 is not None and t1 is not None:
                q = quantity(t1 - t0, "millisecond")
                if q:
                    sizes["t"] = q
        if self.shape[self.axes.index("z")] > 1:
            for axis in root.iter():
                if _local(axis.tag) == "axis" and axis.get("enable") == "true"                         and axis.get("paramEnable", "true") == "true":
                    kids = {_local(c.tag): (c.text or "").strip() for c in axis}
                    if kids.get("axis") == "ZSTACK":
                        q = quantity(abs(float(kids.get("step") or 0)), "micrometer")
                        if q:
                            sizes["z"] = q
                            break
        return Metadata(pixel_sizes=sizes, channels=tuple(channels), index_values=index_values)

    def _frame_position(self, lam: int, t: int, z: int, axis: str):
        """Position of frame (lambda, t, z) along *axis*, from its frame
        properties; None if not recorded."""
        for fid, values in self._frame_positions.items():
            m = FRAME_ID.match(fid)
            if m and (int(m.group(1) or 0), int(m.group(3) or 0), int(m.group(2) or 0))                     == (lam, t, z) and axis in values:
                return values[axis]
        return None

    @staticmethod
    def _lsm_info(lsm):
        """(channel guids in acquisition order, {axis type: info}) from lsmimage."""
        if not lsm:
            return [], {}
        root = ElementTree.fromstring(lsm[0])
        channels, axes = {}, {}
        for elem in root.iter():
            tag = _local(elem.tag)
            if tag == "channel" and elem.get("id"):
                channels.setdefault(elem.get("id"), int(elem.get("order") or 0))
            elif tag == "axis" and elem.get("enable") == "true":
                info, kind = {}, ""
                for child in elem:
                    ctag = _local(child.tag)
                    if ctag == "axis" and child.text:
                        kind = child.text.strip()
                    elif ctag == "maxSize" and child.text:
                        info["maxSize"] = int(child.text.strip())
                if kind:
                    axes[kind] = info
        order = sorted(channels, key=lambda g: channels[g])   # stable: ties keep file order
        return order, axes

    # -- reading ---------------------------------------------------------------

    def _read_range(self, blocks, starts, start: int, dest: memoryview) -> int:
        """Bytes [start, start + len(dest)) of a plane into *dest*; bytes past
        the plane's data (an unfinished acquisition) become zeros."""
        pos, end = start, start + len(dest)
        i = bisect.bisect_right(starts, pos) - 1
        while pos < end and i < len(blocks):
            file_index, offset, length = blocks[i]
            take = min(end, starts[i] + length) - pos
            fh = self._handles[file_index].get()
            fh.seek(offset + pos - starts[i])
            if fh.readinto(dest[pos - start:pos - start + take]) != take:
                raise CorruptFileError(f"{self._paths[file_index]!r}: pixel block ends early")
            pos += take
            i += 1
        if pos < end:
            dest[pos - start:] = bytes(end - pos)
        return end - start

    def _read_rows(self, plane, y0, y1, xs, dst) -> int:
        """Rows y0:y1, columns xs of a plane into *dst* (rows, columns)."""
        blocks, starts = plane
        item, row = self._file_dtype.itemsize, self._row_bytes
        direct = NATIVE_LITTLE or item == 1
        if xs.start == 0 and xs.stop * item == row:
            if direct and dst.flags.c_contiguous:
                self._read_range(blocks, starts, y0 * row, memoryview(dst).cast("B"))
            else:
                buf = np.empty((y1 - y0, xs.stop), self._file_dtype)
                self._read_range(blocks, starts, y0 * row, memoryview(buf).cast("B"))
                dst[...] = buf
            return dst.size
        buf = None if direct else np.empty(xs.stop - xs.start, self._file_dtype)
        for i, y in enumerate(range(y0, y1)):
            target = dst[i] if direct else buf
            self._read_range(blocks, starts, y * row + xs.start * item,
                             memoryview(target).cast("B"))
            if not direct:
                dst[i] = buf
        return dst.size

    def _read_block(self, block):
        if self._closed:
            raise ValueError("this OIR image has been closed: its file was closed "
                             "(File.close() or the end of a with block)")
        ts, cs, zs, ys, xs = block
        out = np.empty(tuple(s.stop - s.start for s in block), dtype=self.dtype)
        per_task = max(1, RAW_TASK_BYTES // self._row_bytes)
        tasks = []
        for t, c, z in itertools.product(range(ts.start, ts.stop), range(cs.start, cs.stop),
                                         range(zs.start, zs.stop)):
            dst = out[t - ts.start, c - cs.start, z - zs.start]
            plane = self._planes.get((self._lam, t, c, z))
            if plane is None:
                dst[...] = 0                        # never acquired
                continue
            for y0 in range(ys.start, ys.stop, per_task):
                y1 = min(ys.stop, y0 + per_task)
                part = dst[y0 - ys.start:y1 - ys.start]
                tasks.append(lambda a=(plane, y0, y1, xs, part): self._read_rows(*a))
        return out, _pool.run(tasks)

    def close(self) -> None:
        self._closed = True
        for handles in self._handles:
            handles.close()


class _OirFile(File):
    backend = NAME


def open(path: str, tiles: str = "separate", resolutions: str = "base") -> _OirFile:  # noqa: A001 - backend protocol name
    image = _OirImage(path)
    lambdas = image._ids[0]
    if len(lambdas) > 1:                    # a lambda scan: one image per emission band
        return _OirFile(path, [image.at_lambda(k) for k in range(len(lambdas))])
    return _OirFile(path, [image])
