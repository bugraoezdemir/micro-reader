"""DeltaVision / Priism (IVE) files: ``.dv``, ``.r3d``, ``.r3d_d3d`` (GE /
Cytiva DeltaVision, OMX; softWoRx, Priism).

An MRC-like layout (UCSF IVE header; field layout as the ``mrc`` package by
Sebastian Haase and Talley Lambert reads it, BSD-3): a 1024-byte header with
the DeltaVision ID -16224 at bytes 96-97 (it also gives the byte order),
an extended header of ``next`` bytes, then raw planes ("sections"), rows
stored bottom-up as in MRC.  Uncompressed, so any rectangle is read exactly,
through the MRC backend's row reader.

- Sections are interleaved by ``imgSequence``: 0 = ZTW (z fastest, then
  time, then wave), 1 = WZT, 2 = ZWT.  The image keeps that order: axes
  ``ctzyx``, ``tzcyx`` or ``tczyx`` (slowest first).
- Pixel types: mode 0 uint8 (unsigned in DeltaVision files, unlike
  MRC2014), 1 int16, 2 float32, 4 complex64, 5 int16, 6 uint16, 7 int32.
- Rows are returned top-down, as Bio-Formats does (checked on the OME
  DeltaVision samples), so switching eubi-bridge's reader never flips an
  image.

Metadata (docs/METADATA.md): pixel sizes ``dx dy dz`` (um), the time step
from the extended header's time stamps, each wave's emission (header) and
excitation (extended header) wavelength -- and the channel names from the
softWoRx acquisition log, a metadata side file at a place the format fixes
(docs/COMPOSITE.md, "DeltaVision"):

1. ``<file>.log``, when it is an acquisition log (``Resolve3D Image File:``)
   naming this file: the name of channel ``c`` is the ``EM filter`` of the
   log's ``Image N.`` for this file's section of (t 0, c, z 0) -- the log
   lists the images in the file's section order;
2. else the file's processing log (``<file>.log``: FRAP results, ...;
   ``<stem>_log.txt``: deconvolution) names its input image; that image's
   acquisition log, up to three steps back;
3. else processing marks stripped from the name (``X_R3D_D3D.dv`` ->
   ``X_R3D.dv``, ``IN_01.r3d_D3D_PRJ.dv`` -> ``IN_01.r3d``; ``_R3D`` is the
   raw file's own mark and stays).

In 2 and 3 the names come from the log's ``#KEY CHANNEL`` lines (wave order;
their EM field), else from its ``EM filter`` values in order of first
appearance.  A log is used only when its ``Image File`` names the image it is
taken for, and only when it lists as many channels as the image has (else a
``CompositeWarning``).  Not read: the stage position (the extended header and
the log disagree by half a field on some files; which point either records
is unknown).
"""
from __future__ import annotations

import builtins
import itertools
import math
import os
import re
import struct
import warnings

import numpy as np

from .. import _parts
from .._base import File, Image
from .._errors import CompositeWarning, CorruptFileError, UnsupportedFormatError
from .._metadata import Channel, Metadata, quantity
from .mrc import HEADER_BYTES, _MrcImage

NAME = "dv"
#: the DeltaVision ID (int16 -16224) at bytes 96-97, in either byte order
DVID = {b"\xa0\xc0": "<", b"\xc0\xa0": ">"}
#: DeltaVision mode -> numpy dtype
MODES = {0: "u1", 1: "i2", 2: "f4", 4: "c8", 5: "i2", 6: "u2", 7: "i4"}
UNSUPPORTED_MODES = {3: "complex int16"}
#: imgSequence -> section order, slowest first (W is the wave: axis c)
SEQUENCES = {0: "ctz", 1: "tzc", 2: "tcz"}
#: header fields read here: name -> (offset, struct code)
FIELDS = {"nx": (0, "i"), "ny": (4, "i"), "sections": (8, "i"), "mode": (12, "i"),
          "dx": (40, "f"), "dy": (44, "f"), "dz": (48, "f"), "next": (92, "i"),
          "n_ints": (128, "h"), "n_floats": (130, "h"), "n_times": (180, "h"),
          "sequence": (182, "h"), "n_waves": (196, "h")}
#: extended header floats per section (after its integers), as written by softWoRx
EXT_TIME, EXT_EXCITATION, EXT_EMISSION = 1, 10, 11


def sniff(head: bytes, path: str) -> bool:
    order = DVID.get(head[96:98]) if len(head) >= 224 else None
    if order is None:
        return False
    nx, ny, sections, mode = struct.unpack_from(order + "4i", head, 0)
    return min(nx, ny, sections) > 0 and (mode in MODES or mode in UNSUPPORTED_MODES)


#: what softWoRx writes at the start of its logs: the acquisition log
#: (``<file>.log``), processing logs (``<file>.log``: FRAP results) and the
#: deconvolution log (``<name>_log.txt``)
LOG_SIGNATURES = (b"#Tag 0x00FF0408", b"Resolve3D", b"DECON3D", b"FRAP Results")


def may_own(folder: str, name: str, folders) -> bool:
    """``group_files``' cheap test (reads nothing): a ``.log`` or
    ``_log.txt`` may be a softWoRx log."""
    return name.lower().endswith((".log", "_log.txt"))


def owner(path: str, head):
    """softWoRx's logs are parts of their image (docs/COMPOSITE.md):
    ``<file>.log`` (the acquisition log, e.g. ``X_R3D.dv.log``) and
    ``<name>_log.txt`` (the deconvolution log of ``<name>.dv``) open it.
    A log whose image is missing is a part only if it is a softWoRx log."""
    if head is None:
        return None
    name, folder = os.path.basename(path), os.path.dirname(path)
    lower = name.lower()
    if lower.endswith("_log.txt"):
        image = name[:-len("_log.txt")] + ".dv"
    elif lower.endswith(".log"):
        image = name[:-len(".log")]
    else:
        return None
    main = _parts.find(folder, image)
    if main is not None and sniff(_parts.read_head(main), main):
        return _parts.Part(main, main)
    if any(sign in head for sign in LOG_SIGNATURES):
        return _parts.Part(None, os.path.join(folder, image))
    return None


# -- the softWoRx logs -------------------------------------------------------------

#: an acquisition log names its image
LOG_IMAGE_FILE = re.compile(r"Resolve3D Image File:[ \t]*(\S[^\r\n]*?)[ \t]*$", re.M)
#: a processing log names its input image (FRAP results; deconvolution)
LOG_INPUT = re.compile(r"^[ \t]*(?:Input Image:|Image Name\.+)[ \t]*(\S[^\r\n]*?)[ \t]*$", re.M)
LOG_CHANNEL_KEY = re.compile(r"^#KEY CHANNEL[ \t]+([^\r\n]*)$", re.M)
LOG_IMAGE = re.compile(r"^Image (\d+)\.[ \t]*$", re.M)
LOG_EM = re.compile(r"EM filter:[ \t]*([^\r\n]*?)[ \t]*$", re.M)
#: a processing step's mark at the end of a name (_D3D, _PRJ, _VOL, _FRAP, ...)
PROCESSING_MARK = re.compile(r"_([A-Z][A-Z0-9]{2,3})$")
RAW_MARK = "R3D"
MAX_LOG_BYTES = 16 << 20
MAX_LOG_STEPS = 3


def _read_log(path):
    """A log's text (latin-1, at most MAX_LOG_BYTES, line ends made
    ``\\n``: logs copied through Windows end lines in ``\\r\\n``), None if
    unreadable."""
    try:
        with builtins.open(path, "rb") as fh:
            text = fh.read(MAX_LOG_BYTES).decode("latin-1")
    except OSError:
        return None
    return re.sub(r"\r+\n?", "\n", text)


def _name(written: str) -> str:
    """The file name of a path a log wrote (Unix or Windows separators)."""
    return written.replace("\\", "/").rstrip("/").rsplit("/", 1)[-1]


def _acquisition_log_of(folder: str, image: str):
    """(path, text) of ``<image>.log`` when it is the acquisition log of
    *image* (its ``Image File`` names it), else None."""
    path = _parts.find(folder, image + ".log")
    text = _read_log(path) if path else None
    match = LOG_IMAGE_FILE.search(text) if text else None
    if match and _name(match.group(1)).lower() == image.lower():
        return path, text
    return None


def acquisition_log(path: str):
    """(log path, text, own) of the acquisition log of DeltaVision file
    *path* by the rules of the module docstring; *own*: the log is the
    file's own (its images are this file's sections).  None without one."""
    folder, name = os.path.dirname(path), os.path.basename(path)
    target, seen = name, {name.lower()}
    for _ in range(MAX_LOG_STEPS + 1):
        found = _acquisition_log_of(folder, target)
        if found:
            return found + (target == name,)
        following = None
        for processing in (target + ".log", os.path.splitext(target)[0] + "_log.txt"):
            log = _parts.find(folder, processing)
            text = _read_log(log) if log else None
            if text and not LOG_IMAGE_FILE.search(text):
                match = LOG_INPUT.search(text)
                if match:
                    following = _name(match.group(1))
                    break
        if following is None or following.lower() in seen:
            break
        seen.add(following.lower())
        target = following
    stem = os.path.splitext(name)[0]
    for _ in range(MAX_LOG_STEPS + 1):
        match = PROCESSING_MARK.search(stem)
        if match is None or match.group(1).upper() == RAW_MARK:
            break
        stem = stem[:match.start()]
        for raw in (stem, stem + ".dv", stem + ".r3d"):
            found = _acquisition_log_of(folder, raw)
            if found:
                return found + (False,)
    return None


def log_channel_names(text: str, own_sections=None) -> list:
    """Channel names an acquisition log gives.  *own_sections*: the file's
    section of each channel's first plane, when the log is the file's own --
    each name is then that image's ``EM filter``; else the ``#KEY CHANNEL``
    lines (their EM field), else the ``EM filter`` values in order of first
    appearance."""
    if own_sections is not None:
        starts = [(int(m.group(1)), m.end()) for m in LOG_IMAGE.finditer(text)]
        em = {}
        for k, (number, start) in enumerate(starts):
            end = starts[k + 1][1] if k + 1 < len(starts) else len(text)
            match = LOG_EM.search(text, start, end)
            if match:
                em[number] = match.group(1)
        names = [em.get(section + 1) for section in own_sections]
        if names and all(names):
            return names
    keys = [line.split(",") for line in LOG_CHANNEL_KEY.findall(text)]
    if keys:
        return [(k[2] if len(k) > 2 else k[-1]).strip() for k in keys]
    names = []
    for value in LOG_EM.findall(text):
        if value and value not in names:
            names.append(value)
    return names


def _header(head: bytes, order: str) -> dict:
    return {name: struct.unpack_from(order + code, head, offset)[0]
            for name, (offset, code) in FIELDS.items()}


class _DvImage(Image):
    """All sections of a DeltaVision file as one image: the section axes in
    the file's interleave order, then ``y x``."""

    def __init__(self, path: str, raw: bytes, head: dict, order: str, sizes: dict, ext: bytes):
        self._order = SEQUENCES[head["sequence"]]
        lead = tuple(sizes[a] for a in self._order)
        dtype = np.dtype(MODES[head["mode"]]).newbyteorder(order)
        #: every section, stacked: the MRC reader does the positioned, bottom-up reads
        self._sections = _MrcImage(path, HEADER_BYTES + head["next"],
                                   (head["sections"], head["ny"], head["nx"]), dtype)
        super().__init__(0, self._order + "yx", lead + (head["ny"], head["nx"]), dtype,
                         name=os.path.basename(path))
        self._lead = lead
        self._head, self._byte_order, self._ext, self._sizes = head, order, ext, sizes
        self._raw_head = raw
        self._path = path

    def _read_block(self, block):
        a, b, c, ys, xs = block
        out = np.empty(tuple(s.stop - s.start for s in block), dtype=self.dtype)
        read = 0
        n1, n2 = self._lead[1], self._lead[2]
        # the innermost section axis is contiguous in the file: one read per (i, j)
        for i, j in itertools.product(range(a.start, a.stop), range(b.start, b.stop)):
            first = (i * n1 + j) * n2
            part, n = self._sections._read_block(
                (slice(first + c.start, first + c.stop), ys, xs))
            out[i - a.start, j - b.start] = part
            read += n
        return out, read

    def _section(self, t: int, c: int, z: int) -> int:
        """The file's section number of plane (t, c, z)."""
        index = {"t": t, "c": c, "z": z}
        i, j, k = (index[a] for a in self._order)
        return (i * self._lead[1] + j) * self._lead[2] + k

    def _ext_float(self, section: int, k: int):
        """Float *k* of *section*'s extended header, None if there is none."""
        n_ints, n_floats = self._head["n_ints"], self._head["n_floats"]
        if k >= n_floats or n_ints < 0:
            return None
        at = (section * (n_ints + n_floats) + n_ints + k) * 4
        if at + 4 > len(self._ext):
            return None
        value = struct.unpack_from(self._byte_order + "f", self._ext, at)[0]
        return value if math.isfinite(value) else None

    def _read_metadata(self):
        head = self._head
        sizes = {}
        for axis in "xyz":
            q = quantity(head["d" + axis], "micrometer")
            if q:
                sizes[axis] = q
        if self._sizes["t"] > 1:
            t0 = self._ext_float(self._section(0, 0, 0), EXT_TIME)
            t1 = self._ext_float(self._section(1, 0, 0), EXT_TIME)
            # softWoRx leaves junk stamps in some processed files: a step only
            # when both are plausible
            if t0 is not None and t1 is not None and t0 >= 0 and t1 > t0:
                sizes["t"] = quantity(t1 - t0, "second")
        waves = struct.unpack_from(self._byte_order + "5h", self._raw_head, 198)
        names = self._log_names()
        channels = []
        for c in range(self._sizes["c"]):
            emission = quantity(waves[c], "nanometer") if c < 5 and waves[c] > 0 else None
            ex = self._ext_float(self._section(0, c, 0), EXT_EXCITATION)
            channels.append(Channel(
                name=names[c] if names else None,
                emission_wavelength=emission,
                excitation_wavelength=quantity(ex, "nanometer") if ex and ex > 0 else None))
        return Metadata(pixel_sizes={a: q for a, q in sizes.items() if q},
                        channels=tuple(channels))

    def _log_names(self):
        """The channel names of the acquisition log (module docstring), or
        None: no log, or one that lists another number of channels."""
        found = acquisition_log(self._path)
        if found is None:
            return None
        log, text, own = found
        n = self._sizes["c"]
        sections = [self._section(0, c, 0) for c in range(n)] if own else None
        names = log_channel_names(text, sections)
        if len(names) != n:
            warnings.warn(CompositeWarning(
                f"{self._path!r}: the softWoRx log {log!r} lists {len(names)} channel(s) "
                f"({names}), the image has {n}: its channel names are not used"), stacklevel=4)
            return None
        return names

    def close(self) -> None:
        self._sections.close()
        self._closed = True


class _DvFile(File):
    backend = NAME


def open(path: str, tiles: str = "separate", resolutions: str = "base") -> _DvFile:  # noqa: A001 - backend protocol name
    size = os.path.getsize(path)
    if size < HEADER_BYTES:
        raise CorruptFileError(f"{path!r} is smaller than a DeltaVision header")
    with builtins.open(path, "rb") as fh:
        raw = fh.read(HEADER_BYTES)
        order = DVID.get(raw[96:98])
        if order is None:
            raise CorruptFileError(f"{path!r}: no DeltaVision ID at bytes 96-97")
        head = _header(raw, order)
        if head["next"] < 0:
            raise CorruptFileError(f"{path!r}: negative extended header size {head['next']}")
        ext = fh.read(head["next"])
    mode = head["mode"]
    if mode in UNSUPPORTED_MODES:
        raise UnsupportedFormatError(
            f"{path!r}: DeltaVision mode {mode} ({UNSUPPORTED_MODES[mode]}) is not supported yet")
    if mode not in MODES:
        raise CorruptFileError(f"{path!r}: unknown DeltaVision mode {mode}")
    if head["sequence"] not in SEQUENCES:
        raise UnsupportedFormatError(
            f"{path!r}: DeltaVision interleave (imgSequence) {head['sequence']} is not one of "
            f"{sorted(SEQUENCES)}")
    nx, ny, sections = head["nx"], head["ny"], head["sections"]
    if min(nx, ny, sections) <= 0:
        raise CorruptFileError(f"{path!r}: invalid dimensions {(nx, ny, sections)}")
    waves, times = max(1, head["n_waves"]), max(1, head["n_times"])
    if sections % (waves * times):
        raise CorruptFileError(
            f"{path!r}: {sections} sections are not {waves} waves x {times} time points x z")
    sizes = {"c": waves, "t": times, "z": sections // (waves * times)}
    needed = HEADER_BYTES + head["next"] + nx * ny * sections * np.dtype(MODES[mode]).itemsize
    if size < needed:
        raise CorruptFileError(
            f"{path!r} is truncated: the header describes {needed} bytes, the file has {size}")
    return _DvFile(path, [_DvImage(path, raw, head, order, sizes, ext)])
