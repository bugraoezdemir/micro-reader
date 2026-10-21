"""MetaMorph multi-dimensional acquisitions: an ``.nd`` file and its TIFF / STK
files (docs/COMPOSITE.md, "MetaMorph .nd sets").

The ``.nd`` file (``"NDInfoFile"``) is a text file of ``"Key", value``
lines (a ``Description`` may run over several lines).  It names no file:
MetaMorph names the image files after it, by a rule -- a **composite**
(docs/COMPOSITE.md, "metadata + pixels"):

    <base>[_w<k><WaveName k>][_s<p>][_t<t>].TIF  (or .STK)

``<base>`` is the ``.nd`` file's name without its extension; ``_w`` is there
when the set has wavelengths (``DoWave``) and ``WaveInFileName`` is not
FALSE, ``_s`` when it has stage positions (``DoStage``), ``_t`` when it is a
time-lapse (``DoTimelapse``); numbers start at 1, without zero padding.  A
dimension of one value may also be written without its part (both spellings
are tried; two files for one name: a warning, neither is used).  Each file
holds the z planes of one wavelength, position and timepoint: ``NZSteps``
planes (``DoZSeries``), one for a wavelength with ``WaveDoZ<k>`` FALSE.

- One micro-reader image per stage position (index ``series``), axes
  ``t c z y x``; channels are the wavelengths, named by ``WaveName<k>``.
- Wavelengths with and without z: the image is ragged (docs/AXES.md,
  ``_fileset.ragged_field``), its planes placed by the members' own focus
  positions (MetaSeries ``z-position``, STK ``AbsoluteZ``).
- Opening the ``.nd`` file, any of its TIFF / STK files, or a folder holding
  one ``.nd`` file opens the whole set.  ``backend="tiff"`` reads a member
  file alone.  ``_thumb`` files are previews: they open alone and are listed
  among the set's parts by ``group_files``.
- Missing files: as for every file set (``_fileset.keep_fields``): zeros,
  one ``MissingImagesWarning``; a position without any of its files is left
  out.
- Metadata (the ``.nd`` file wins, the members fill its gaps, a disagreement
  warns): channel names (``WaveName``; else the members' illumination
  setting), x / y pixel sizes from the members' MetaMorph calibration, z
  step from ``ZStepSize`` (micrometres, MetaMorph's focus unit).  The
  members' stage coordinates (MetaSeries ``stage-position-x`` / ``-y``) are
  ``Metadata.stage_position``, as recorded, not ``position``: which point of
  the field they mark and which way the stage runs against the camera are not
  recorded (Bugra, 2026-10-19).  Not read: the time step, wavelengths.
- Refused (``UnsupportedFormatError``): several wavelengths in one file
  (``WaveInFileName`` FALSE with more than one wavelength) and wavelengths
  imaged at some timepoints only (``WavePointsCollected``) -- no sample yet.
"""
from __future__ import annotations

import io
import itertools
import os
import re
import warnings
from typing import NamedTuple, Optional

import numpy as np

from .. import _parts
from .._base import File
from .._errors import (CompositeWarning, CorruptFileError, MicroReaderError,
                       UnsupportedFormatError)
from .._fileset import Members, PlaneImage, keep_fields, ragged_field, z_parts
from .._metadata import Channel, Metadata, quantity

NAME = "metamorph"
#: the index file of a set: a folder holding it opens it (docs/COMPOSITE.md)
SET_INDEX = True
MAGIC = b'"NDInfoFile"'
#: member file extensions MetaMorph writes (compared in lower case)
MEMBER_EXTENSIONS = (".tif", ".stk")
#: a preview, not a member: opens alone
THUMB = "_thumb"
#: relative tolerance when the .nd file and a member file give one value
TOLERANCE = 1e-3
#: the rest of a member file name after the base: _w<k><name>, _s<p>, _t<t>
_REST = re.compile(r"(?:_w(?P<w>\d+).*?)?(?:_s(?P<s>\d+))?(?:_t(?P<t>\d+))?",
                   re.IGNORECASE | re.DOTALL)
_KEY_LINE = re.compile(r'^\s*"(?P<key>[^"]*)"\s*(?:,\s*(?P<value>.*?))?\s*$')


# -- the .nd file -------------------------------------------------------------

def may_index(name: str) -> bool:
    """``group_files``' cheap test (reads nothing): the set is an ``.nd`` file."""
    return name.lower().endswith(".nd")


def sniff(head: bytes, path: str) -> bool:
    return head.lstrip(b"\xef\xbb\xbf \t\r\n")[:len(MAGIC)] == MAGIC


def _value(raw: Optional[str]):
    """A value as the ``.nd`` file writes it: a quoted string stays a string;
    TRUE / FALSE -> bool, numbers -> int / float, the rest a string."""
    if raw is None:
        return None
    raw = raw.strip()
    if len(raw) >= 2 and raw[0] == raw[-1] == '"':
        return raw[1:-1]
    if raw.upper() in ("TRUE", "FALSE"):
        return raw.upper() == "TRUE"
    for kind in (int, float):
        try:
            return kind(raw)
        except ValueError:
            pass
    return raw


def parse_nd(text: str) -> dict:
    """``{key: value}`` of an ``.nd`` file; a line that does not start a key
    continues the value before it (the ``Description``)."""
    out, last = {}, None
    for line in re.split(r"\r\n|\r|\n", text):
        match = _KEY_LINE.match(line)
        if match:
            last = match["key"]
            out[last] = _value(match["value"])
        elif last is not None and line.strip():
            before = out[last]
            out[last] = f"{before}\n{line.strip()}" if before not in (None, "") else line.strip()
    return out


def read_nd(path: str) -> dict:
    """The ``.nd`` file's keys (Latin-1: every byte decodes)."""
    with io.open(path, "r", encoding="latin-1", newline="") as fh:
        return parse_nd(fh.read())


class Layout(NamedTuple):
    """What an ``.nd`` file says about its set (module docstring)."""
    base: str
    times: int
    timelapse: bool
    stages: list            # stage names (None: not named); [None] without DoStage
    stage_on: bool
    waves: list             # [(name or None, has z)]; [(None, True)] without DoWave
    wave_on: bool
    waves_in_name: Optional[bool]   # None: the file does not say (WaveInFileName absent)
    nz: int
    z_step: Optional[float]

    def depth(self, w: int) -> int:
        """Planes in a file of wavelength *w* (0-based)."""
        return self.nz if self.waves[w][1] else 1


def _flag(info: dict, key: str, path: str) -> bool:
    value = info.get(key, False)
    if not isinstance(value, bool):
        raise CorruptFileError(f"{path!r}: {key} is {value!r}, not TRUE or FALSE")
    return value


def _count(info: dict, key: str, path: str) -> int:
    value = info.get(key)
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise CorruptFileError(f"{path!r}: {key} is {value!r}, not a count of 1 or more")
    return value


def layout(info: dict, path: str) -> Layout:
    """The set an ``.nd`` file's keys describe; refuses what is not read yet."""
    base = os.path.splitext(os.path.basename(path))[0]
    timelapse = _flag(info, "DoTimelapse", path)
    times = _count(info, "NTimePoints", path) if timelapse else 1
    stage_on = _flag(info, "DoStage", path)
    if stage_on:
        n = _count(info, "NStagePositions", path)
        stages = [_name(info.get(f"Stage{k}")) for k in range(1, n + 1)]
    else:
        stages = [None]
    zseries = _flag(info, "DoZSeries", path)
    nz = _count(info, "NZSteps", path) if zseries else 1
    z_step = info.get("ZStepSize") if zseries else None
    z_step = abs(float(z_step)) if isinstance(z_step, (int, float)) and not isinstance(
        z_step, bool) and z_step else None
    wave_on = _flag(info, "DoWave", path)
    if wave_on:
        n = _count(info, "NWavelengths", path)
        waves = []
        for k in range(1, n + 1):
            doz = info.get(f"WaveDoZ{k}", True)
            if not isinstance(doz, bool):
                raise CorruptFileError(f"{path!r}: WaveDoZ{k} is {doz!r}, not TRUE or FALSE")
            waves.append((_name(info.get(f"WaveName{k}")), doz))
    else:
        waves = [(None, True)]
    in_name = info.get("WaveInFileName")
    if in_name is not None and not isinstance(in_name, bool):
        raise CorruptFileError(f"{path!r}: WaveInFileName is {in_name!r}, not TRUE or FALSE")
    if in_name is False and len(waves) > 1:
        raise UnsupportedFormatError(
            f"{path!r}: {len(waves)} wavelengths with WaveInFileName FALSE (several wavelengths "
            "in each file) are not read yet; backend='tiff' reads one file alone")
    if any(key.lower().startswith("wavepointscollected") for key in info):
        raise UnsupportedFormatError(
            f"{path!r}: wavelengths imaged at some timepoints only (WavePointsCollected) are not "
            "read yet; backend='tiff' reads one file alone")
    return Layout(base, times, timelapse, stages, stage_on, waves, wave_on, in_name, nz, z_step)


def _name(value) -> Optional[str]:
    if value is None or value == "":
        return None
    return str(value)


# -- the naming rule ------------------------------------------------------------

def _wave_parts(lay: Layout, w: int) -> list:
    if not lay.wave_on or lay.waves_in_name is False:
        return [""]
    part = f"_w{w + 1}{lay.waves[w][0] or ''}"
    return [part, ""] if lay.waves_in_name is None and len(lay.waves) == 1 else [part]


def _optional(on: bool, letter: str, value: int, count: int) -> list:
    if not on:
        return [""]
    part = f"_{letter}{value + 1}"
    return [part] if count > 1 else [part, ""]


def candidates(lay: Layout, s: int, t: int, w: int) -> list:
    """The names the file of stage *s*, timepoint *t*, wavelength *w*
    (0-based) may have, the rule's first (``.TIF``); compare in lower case."""
    stems = [lay.base + a + b + c
             for a in _wave_parts(lay, w)
             for b in _optional(lay.stage_on, "s", s, len(lay.stages))
             for c in _optional(lay.timelapse, "t", t, lay.times)]
    seen, out = set(), []
    for stem in stems:
        for ext in MEMBER_EXTENSIONS:
            name = stem + ext.upper()
            if name.lower() not in seen:
                seen.add(name.lower())
                out.append(name)
    return out


def member_key(lay: Layout, name: str) -> Optional[tuple]:
    """(s, t, w), 0-based, of the member file called *name*, or None when the
    rule does not give that name."""
    stem, ext = os.path.splitext(name)
    if ext.lower() not in MEMBER_EXTENSIONS or not stem.lower().startswith(lay.base.lower()):
        return None
    match = _REST.fullmatch(stem[len(lay.base):])
    if match is None:
        return None
    s, t, w = (int(match[k]) - 1 if match[k] else 0 for k in ("s", "t", "w"))
    if not (0 <= s < len(lay.stages) and 0 <= t < lay.times and 0 <= w < len(lay.waves)):
        return None
    return (s, t, w) if name.lower() in {n.lower() for n in candidates(lay, s, t, w)} else None


def _bases(stem: str) -> list:
    """The ``.nd`` base names a member called *stem* (no extension) may have:
    every prefix ending before an underscore, longest first, and *stem*."""
    cuts = [m.start() for m in re.finditer("_", stem) if m.start() > 0]
    return [stem] + [stem[:k] for k in reversed(cuts)]


def _nd_next_to(folder: str, base: str) -> Optional[str]:
    """``<base>.nd`` in *folder* (exact names: no folder listing)."""
    for ext in (".nd", ".ND"):
        path = os.path.join(folder, base + ext)
        if os.path.isfile(path):
            return path
    return None


def _claims(nd: str, name: str) -> bool:
    """Whether the ``.nd`` file *nd* names a member *name* by its rule.  An
    ``.nd`` file that is not read yet (or is damaged) claims every name of
    the rule's shape, so that opening the member says why."""
    try:
        if not sniff(_parts.read_head(nd), nd):
            return False
        info = read_nd(nd)
    except OSError:
        return False
    try:
        return member_key(layout(info, nd), name) is not None
    except MicroReaderError:
        base = os.path.splitext(os.path.basename(nd))[0]
        stem = os.path.splitext(name)[0]
        return stem.lower().startswith(base.lower()) and \
            _REST.fullmatch(stem[len(base):]) is not None


def may_own(folder: str, name: str, folders) -> bool:
    """``group_files``' cheap test (reads nothing but the folder listing
    *folders* caches): a TIFF / STK file next to an ``.nd`` file its name
    may come from."""
    lower = name.lower()
    if not lower.endswith(MEMBER_EXTENSIONS) or THUMB in lower:
        return False
    names = folders.names(folder)
    return any(base + ".nd" in names for base in _bases(os.path.splitext(lower)[0]))


def owner(path: str, head):
    """A TIFF / STK file the naming rule derives from an ``.nd`` file next to
    it is a part of that set (opening it opens the set); ``_thumb`` previews
    are not.  Looks only for ``<base>.nd`` of the bases the name allows."""
    from . import tiff
    if head is None or head[:4] not in tiff.MAGIC:
        return None
    name = os.path.basename(path)
    lower = name.lower()
    if not lower.endswith(MEMBER_EXTENSIONS) or THUMB in lower:
        return None
    folder = os.path.dirname(os.path.abspath(path))
    found = []
    for base in _bases(os.path.splitext(name)[0]):
        nd = _nd_next_to(folder, base)
        if nd is not None and _claims(nd, name):
            found.append(nd)
    if not found:
        return None
    if len(found) > 1:
        warnings.warn(CompositeWarning(
            f"{path!r} fits the naming rule of {len(found)} .nd files "
            f"({sorted(os.path.basename(p) for p in found)}): none is used"), stacklevel=4)
        return None
    return _parts.Part(found[0], found[0], standalone=True)


def _stage_position(tif) -> dict:
    """The MetaSeries ``stage-position-x`` / ``-y`` of a member's first page,
    as recorded (micrometres, MetaMorph's stage unit): ``Metadata.stage_position``,
    not ``position`` -- which point of the field they mark and how the stage
    runs against the camera are not recorded (Bugra, 2026-10-19)."""
    from ._tiff_metadata import metaseries_plane_info
    if tif.is_stk:
        return {}
    info = metaseries_plane_info(tif)
    out = {a: quantity(info.get(f"stage-position-{a}"), "micrometer", positive=False) for a in "xy"}
    return out if all(out.values()) else {}


# -- the images -----------------------------------------------------------------

def _plane_image(member, path: str):
    """(image, planes) of a member opened as pages: one stack of planes."""
    images = member.images
    if len(images) != 1:
        raise CorruptFileError(f"{path!r}: {len(images)} runs of pages of different sizes; "
                               "a MetaMorph .nd member holds one stack of planes")
    image = images[0]
    if image.axes == "yx":
        return image, 1
    if len(image.axes) == 3 and image.axes[1:] == "yx" and not getattr(image, "_fixed", None):
        return image, image.shape[0]
    raise UnsupportedFormatError(f"{path!r}: a {image.axes} image; MetaMorph .nd members are "
                                 "read as stacks of single-sample planes (RGB not read yet)")


class _NDImage(PlaneImage):
    """One stage position (the whole) or one of its parts.  *planes*:
    (t, c, z) -> (member path, plane in it, planes the member must hold);
    *channels*: the wavelengths (0-based) it holds; *dz*: its z step from the
    members' focus positions (None: unknown or one plane)."""

    def __init__(self, *args, owner, channels: list, dz=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._owner, self._channels, self._dz = owner, channels, dz

    def _read_block(self, block):
        if self._closed:
            raise ValueError(f"{self.name!r} has been closed: its file was closed "
                             "(File.close() or the end of a with block)")
        if not self._readable:
            raise UnsupportedFormatError(
                f"{self.name!r} is ragged and its parts do not share a z grid: read its "
                f"parts (image.parts)")
        from .. import _pool
        (t0, t1), (c0, c1), (z0, z1) = ((s.start, s.stop) for s in block[:3])
        ys, xs = block[3], block[4]
        out = np.empty((t1 - t0, c1 - c0, z1 - z0, ys.stop - ys.start, xs.stop - xs.start),
                       dtype=self.dtype)
        # every plane's reads planned straight into *out*, then run in one batch on
        # the decode pool (as ``PlaneImage._read_pages``)
        entries, plans = [], []
        try:
            for t in range(t0, t1):
                for c in range(c0, c1):
                    for z in range(z0, z1):
                        dest = out[t - t0, c - c0, z - z0]
                        spec = self._planes.get((t, c, z))
                        if spec is None:                # a missing plane reads as zeros
                            dest[...] = 0
                            continue
                        path, k, n = spec
                        entry = self._members.acquire(path)
                        entries.append(entry)
                        image = self._fits_member(path, entry[0], n)
                        if image.ndim == 2:
                            plan = image._plan_series((ys, xs), dest)
                        else:
                            plan = image._plan_series((slice(k, k + 1), ys, xs), dest[None])
                        plans.append((image, plan))
            return out, _pool.run(itertools.chain.from_iterable(
                image._tasks(*plan) for image, plan in plans))
        finally:
            for entry in entries:
                self._members.release(entry)

    def _fits_member(self, path: str, member, n: int):
        """The member's image, after checking it holds *n* planes of this
        image's size and type (the ``.nd`` file decides what a member is)."""
        image, planes = _plane_image(member, path)
        y, x = self.shape[-2:]
        if planes == n and tuple(image.shape[-2:]) == (y, x) and image.dtype == self.dtype:
            return image
        raise CorruptFileError(
            f"{path!r}: {planes} {tuple(image.shape[-2:])} {image.dtype} plane(s), "
            f"{self.name!r} expects {n} {y} x {x} {self.dtype} plane(s) per file")

    def _check_members(self, seen: set):
        for path, _k, n in self._planes.values():
            if path in seen:
                continue
            seen.add(path)
            yield lambda path=path, n=n: self._check_one(path, n)

    def _check_one(self, path: str, n):
        entry = self._members.acquire(path)
        try:
            self._fits_member(path, entry[0], n)
        except (CorruptFileError, UnsupportedFormatError) as exc:
            return str(exc)
        finally:
            self._members.release(entry)
        return 1

    def _read_metadata(self):
        first = self._planes[min(self._planes)][0] if self._planes else None
        sizes, stage = {}, {}
        if first is not None:
            entry = self._members.acquire(first)
            try:
                sizes = {a: q for a, q in entry[0].images[0].metadata.pixel_sizes.items()
                         if a in "xy"}
                stage = _stage_position(entry[0]._tif)
            finally:
                self._members.release(entry)
        if self.shape[2] > 1:
            z = self._owner._z_size(self._dz, self.name)
            if z is not None:
                sizes["z"] = z
        names = self._owner._channel_names()
        return Metadata(pixel_sizes=sizes, stage_position=stage,
                        channels=tuple(Channel(name=names[c]) for c in self._channels))


class _NDFile(File):
    backend = NAME
    format = "metamorph"

    def __init__(self, path: str):
        self.path = path
        self._members = Members(pages=True)
        self._names = None
        self._z_told = False
        try:
            images = self._images(path)
        except BaseException:
            self._members.close()
            raise
        super().__init__(path, images)

    def _images(self, path: str) -> list:
        self._lay = lay = layout(read_nd(path), path)
        self._folder = os.path.dirname(os.path.abspath(path))
        try:
            entries = os.listdir(self._folder)
        except OSError as exc:
            raise CorruptFileError(f"{path!r}: its folder cannot be listed ({exc})") from None
        self._entries = entries
        by_lower: dict = {}
        for e in entries:
            by_lower.setdefault(e.lower(), []).append(e)

        wanted = [(t, w, z) for t in range(lay.times) for w in range(len(lay.waves))
                  for z in range(lay.depth(w))]
        listed, ambiguous = {}, []
        for s in range(len(lay.stages)):
            field = {}
            for t in range(lay.times):
                for w in range(len(lay.waves)):
                    names = candidates(lay, s, t, w)
                    hits = [by_lower[n.lower()] for n in names if n.lower() in by_lower]
                    if len(hits) == 1 and len(hits[0]) == 1:
                        file = os.path.join(self._folder, hits[0][0])
                        n = lay.depth(w)
                        for z in range(n):
                            field[(t, w, z)] = (file, z, n)
                        continue
                    # missing: its expected name (for the warning); ambiguous: None
                    # (named by the warning below) -- both read as zeros
                    expected = os.path.join(self._folder, names[0])
                    if hits:
                        ambiguous.append(sorted(itertools.chain.from_iterable(hits)))
                        expected = None
                    for z in range(lay.depth(w)):
                        field[(t, w, z)] = expected
            listed[s] = field
        if ambiguous:
            warnings.warn(CompositeWarning(
                f"{path!r}: {len(ambiguous)} planes have more than one file fitting the naming "
                f"rule (e.g. {ambiguous[0]}): none of them is used"), stacklevel=3)
        kept = keep_fields(listed, wanted, path)
        parts = z_parts(wanted)

        first = next(iter(next(iter(kept.values())).values()))
        entry = self._members.acquire(first[0])
        try:
            image, _n = _plane_image(entry[0], first[0])
            dtype, shape_yx = image.dtype, tuple(image.shape[-2:])
            unit = tuple(image.read_unit[-2:]) if image.read_unit is not None else None
        finally:
            self._members.release(entry)

        times = list(range(lay.times))
        images = []
        for s in sorted(kept):
            name = self._image_name(s)
            index = {"series": s}
            z_at = self._focus(kept[s]) if len(parts) > 1 else {}

            def make(planes, shape_tcz, chs, dz, index, readable, name=name):
                label = name if "part" not in index else f"{name} part {index['part']}"
                return _NDImage(self._members, len(images), planes, shape_tcz + shape_yx, dtype,
                                label, unit, index, readable=readable, owner=self,
                                channels=chs, dz=dz)

            images.append(ragged_field(make, kept[s], z_at, times, parts, index))
        return images

    def _image_name(self, s: int) -> str:
        lay = self._lay
        if not lay.stage_on:
            return lay.base
        label = lay.stages[s]
        return f"{lay.base} {label}" if label else f"{lay.base} stage {s + 1}"

    def _focus(self, field: dict) -> dict:
        """(t, c, z) -> the member's focus position of that plane, at the first
        timepoint (ragged sets only: one file per wavelength opened)."""
        out = {}
        files = {}
        for (t, c, z), (path, k, _n) in field.items():
            if t == 0:
                files.setdefault((c, path), []).append((z, k))
        for (c, path), planes in files.items():
            zs = self._member_focus(path)
            for z, k in planes:
                out[(0, c, z)] = zs[k] if zs is not None and k < len(zs) else None
        return out

    def _member_focus(self, path: str):
        """The focus position of each plane of member *path* (MetaSeries
        ``z-position`` per page, STK ``AbsoluteZ`` when valid), or None."""
        import tifffile
        entry = self._members.acquire(path)
        try:
            tif = entry[0]._tif
            if tif.is_stk:
                md = tif.stk_metadata or {}
                absolute, valid = md.get("AbsoluteZ"), md.get("AbsoluteZValid")
                if absolute is None or valid is None or not np.all(np.asarray(valid)):
                    return None
                return [float(v) for v in np.atleast_1d(absolute)]
            if getattr(tif, "is_metaseries", False):
                # tifffile parses the first page only (``metaseries_metadata``)
                parse = getattr(tifffile, "metaseries_description_metadata", None) or \
                    tifffile.tifffile.metaseries_description_metadata
                out = []
                for page in tif.pages:
                    try:
                        out.append(float(parse(page.description)["PlaneInfo"]["z-position"]))
                    except Exception:                   # noqa: BLE001 - no position on this page
                        out.append(None)
                return out
            return None
        finally:
            self._members.release(entry)

    def _z_size(self, dz, name: str):
        """The z step: ``ZStepSize`` (the ``.nd`` file wins), else the members'
        focus step; both in micrometres.  A disagreement warns once."""
        lay = self._lay
        if lay.z_step is not None:
            if dz and abs(dz - lay.z_step) > TOLERANCE * lay.z_step and not self._z_told:
                self._z_told = True
                warnings.warn(CompositeWarning(
                    f"{self.path!r}: ZStepSize is {lay.z_step} um, the member files' focus "
                    f"positions step by {dz} ({name!r}): using ZStepSize"), stacklevel=4)
            return quantity(lay.z_step, "micrometer")
        return quantity(dz, "micrometer") if dz else None

    def _channel_names(self) -> list:
        """Each wavelength's name: ``WaveName`` (the ``.nd`` file wins), else
        the illumination setting its member files record; read once."""
        if self._names is not None:
            return self._names
        from ._tiff_metadata import metaseries_plane_info
        lay = self._lay
        names = [w[0] for w in lay.waves]
        firsts: dict = {}                       # wavelength -> a member file of it
        for part in (p for image in self.images for p in image.parts):
            for (_t, c, _z), spec in part._planes.items():
                firsts.setdefault(part._channels[c], spec[0])
                if len(firsts) == len(names):
                    break
            if len(firsts) == len(names):
                break
        told = []
        for w, path in sorted(firsts.items()):
            entry = self._members.acquire(path)
            try:
                tif = entry[0]._tif
                setting = metaseries_plane_info(tif).get("_IllumSetting_") if not tif.is_stk else None
            finally:
                self._members.release(entry)
            setting = _name(setting)
            if setting is None:
                continue
            if names[w] is None:
                names[w] = setting
            elif setting != names[w]:
                told.append(f"wavelength {w + 1}: {names[w]!r} in the .nd file, {setting!r} in "
                            f"{os.path.basename(path)!r}")
        if told:
            warnings.warn(CompositeWarning(
                f"{self.path!r}: channel names differ ({'; '.join(told)}): using the .nd file's"),
                stacklevel=4)
        self._names = names
        return names

    def _side_files(self) -> set:
        """The set's ``_thumb`` previews (associated images, docs/COMPOSITE.md):
        parts of the set for ``group_files``, images on their own when opened."""
        prefix = self._lay.base.lower() + "_"
        return {os.path.join(self._folder, e) for e in self._entries
                if e.lower().startswith(prefix) and THUMB in e.lower()
                and e.lower().endswith(MEMBER_EXTENSIONS + (".tiff",))}

    def close(self) -> None:
        super().close()
        self._members.close()


def open(path: str, tiles: str = "separate", resolutions: str = "base") -> _NDFile:  # noqa: A001 - backend protocol
    """A MetaMorph set from its ``.nd`` file (*tiles* and *resolutions* change
    nothing: stage positions are separate images, and the files hold no
    pyramids)."""
    return _NDFile(path)
