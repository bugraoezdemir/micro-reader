"""Micro-Manager image-per-file acquisitions (Micro-Manager 1.4 and the 2.0 betas):
a ``metadata.txt`` per stage position and one TIFF per plane (docs/COMPOSITE.md,
"Micro-Manager 1.4 image-per-file acquisitions").

Layout (checked on the OME samples 1.4.16 serge, 1.4.22 and 2.0.0-beta3 thomas)::

    <acquisition>/                      Summary "Prefix"; the folder may be renamed
        display_and_comments.txt        display settings (JSON), optional
        <position>/                     one folder per stage position: its label
            metadata.txt                (InitialPositionList "Label"), else Pos<k>
            img_<t:9>_<channel>_<z:3>.tif                    (1.4)
            img_channel<c:3>_position<p:3>_time<t:9>_z<z:3>.tif  (2.0 betas)

``metadata.txt`` is JSON: ``"Summary"`` (the whole acquisition: Frames,
Channels, Slices, Positions, ChNames, ChColors, z-step_um, Interval_ms,
PixelSize_um, InitialPositionList, ...; the same in every position's file but
for PositionIndex and Time), then one entry per plane of this position:
``"FrameKey-<t>-<c>-<z>"`` (1.4; ``FileName``, stage, pixel size, ...) or
``"Coords-<file name>"`` (2.0 betas: ``{"time", "channel", "z", "position"}``,
the file's own metadata in its TIFF).  Each TIFF is a plain ImageJ TIFF of
one plane, with the plane's Micro-Manager metadata as JSON in its ImageJ
``Info``.

- A **composite** (metadata + pixels): ``metadata.txt`` says what each plane
  is, the TIFFs hold them; neither names the other folders' files.  The main
  file is the ``metadata.txt`` of the first position (Summary order) whose
  folder is there.  Opening any part -- a plane TIFF, any ``metadata.txt``,
  ``display_and_comments.txt``, a position folder or the acquisition folder
  -- opens the whole acquisition.  ``backend="tiff"`` reads a plane TIFF
  alone.
- One image per stage position (index ``scene``, the Summary's order), axes
  ``t c z y x``, sizes from the Summary (Frames, Channels, Slices).
- Missing: a position the Summary lists without its folder is left out, a
  plane whose file is missing or not listed reads as zeros; one
  ``MissingImagesWarning`` each.  A folder named like a position that holds
  another acquisition is not used (``CompositeWarning``).
- Metadata (``metadata.txt`` wins, the TIFFs' own fill gaps, a disagreement
  warns): channel names ``ChNames``, colours ``ChColors`` (Java ARGB ints);
  x / y ``PixelSize_um`` -- 0 is Micro-Manager's "no pixel size calibration",
  so unknown -- else the planes' ``PixelSizeUm``; z ``z-step_um`` (0:
  unknown); t ``Interval_ms``, else ``CustomIntervals_ms`` when all its
  intervals after the first are one value (0: unknown); the first plane's
  ``XPositionUm`` / ``YPositionUm`` / ``ZPositionUm`` as
  ``Metadata.stage_position`` (as recorded: which point of the field they
  mark and how the stage runs against the camera are not recorded).
"""
from __future__ import annotations

import builtins
import json
import os
import re
import warnings
from typing import Optional

from .. import _parts
from .._base import File
from .._errors import CompositeWarning, CorruptFileError, MicroReaderError, MissingImagesWarning
from .._fileset import Members, PlaneImage, keep_fields
from .._metadata import Channel, Metadata, color_hex, quantity

NAME = "micromanager"
#: the index file of a set: a folder holding it opens it (docs/COMPOSITE.md)
SET_INDEX = True
METADATA = "metadata.txt"
DISPLAY = "display_and_comments.txt"
#: relative tolerance when metadata.txt and a plane's TIFF give one value
TOLERANCE = 1e-6
#: the two plane file names (Micro-Manager 1.4; the 2.0 betas)
NAME_14 = re.compile(r"img_(?P<t>\d+)_(?P<channel>.+)_(?P<z>\d+)\.tiff?", re.IGNORECASE)
NAME_20 = re.compile(r"img_channel(?P<c>\d+)_position(?P<p>\d+)_time(?P<t>\d+)_z(?P<z>\d+)\.tiff?",
                     re.IGNORECASE)
_SUMMARY = re.compile(r'\s*\{\s*"Summary"\s*:\s*')
#: Summary keys compared when the files carry no acquisition UUID
_SAME = ("Prefix", "ChNames", "Frames", "Channels", "Slices", "Positions", "ComputerName",
         "UserName", "MicroManagerVersion", "Directory")


# -- metadata.txt -----------------------------------------------------------------

def may_index(name: str) -> bool:
    """``group_files``' cheap test (reads nothing)."""
    return name.lower() == METADATA


def sniff(head: bytes, path: str) -> bool:
    return (os.path.basename(path).lower() == METADATA and _SUMMARY.match(
        head.lstrip(b"\xef\xbb\xbf").decode("latin-1")) is not None)


def _text(path: str, size: int = -1) -> str:
    with builtins.open(path, "rb") as fh:
        raw = fh.read(size)
    return raw.decode("utf-8", errors="replace").lstrip("﻿")


def read_metadata(path: str) -> dict:
    """The whole ``metadata.txt``."""
    try:
        data = json.loads(_text(path))
    except ValueError as exc:
        raise CorruptFileError(
            f"{path!r} is not complete JSON ({exc}): an acquisition that did not finish "
            "writing it? backend='tiff' reads a plane file alone") from None
    if not isinstance(data, dict) or not isinstance(data.get("Summary"), dict):
        raise CorruptFileError(f"{path!r}: no Micro-Manager \"Summary\"")
    return data


def read_summary(path: str) -> Optional[dict]:
    """``metadata.txt``'s Summary, read from the start of the file only (the
    planes after it may run to megabytes); None when it is no Micro-Manager
    metadata file."""
    size = 1 << 16
    while True:
        try:
            text = _text(path, size)
        except OSError:
            return None
        match = _SUMMARY.match(text)
        if match is None:
            return None
        try:
            summary, _end = json.JSONDecoder().raw_decode(text, match.end())
            return summary if isinstance(summary, dict) else None
        except ValueError:
            if len(text) < size // 2 or size > 1 << 26:     # the whole file read: broken
                return None
            size *= 4


def _count(summary: dict, key: str) -> int:
    value = summary.get(key)
    return value if isinstance(value, int) and not isinstance(value, bool) and value > 0 else 0


def _labels(summary: dict) -> list:
    """The stage positions' folder names, in the Summary's order: the position
    list's labels (1.4 ``InitialPositionList``, 2.0 ``StagePositions``) when it
    lists them all, else ``Pos<k>`` (Micro-Manager's default label)."""
    n = max(1, _count(summary, "Positions"))
    for key, field in (("InitialPositionList", "Label"), ("StagePositions", "label")):
        listed = summary.get(key)
        if isinstance(listed, list) and len(listed) == n:
            labels = [p.get(field) if isinstance(p, dict) else None for p in listed]
            if all(isinstance(x, str) and x for x in labels):
                return labels
    return [f"Pos{k}" for k in range(n)]


def _same(a: dict, b: dict) -> bool:
    """Whether two Summaries are one acquisition's (its UUID when both have one)."""
    if a.get("UUID") and b.get("UUID"):
        return a["UUID"] == b["UUID"]
    return all(a.get(k) == b.get(k) for k in _SAME)


def _position_index(summary: dict, folder: str, labels: list) -> int:
    """Which position a ``metadata.txt`` is: its folder's label, else the
    Summary's ``PositionIndex``."""
    name = os.path.basename(folder).lower()
    for k, label in enumerate(labels):
        if label.lower() == name:
            return k
    k = summary.get("PositionIndex")
    return k if isinstance(k, int) and 0 <= k < len(labels) else 0


def positions(path: str, summary: dict) -> list:
    """[(label, its metadata.txt or None)] of the acquisition whose position file
    *path* (with *summary*) is, in the Summary's order: the other positions are
    the folders next to *path*'s named by their labels (a folder holding
    another acquisition is not used, with a warning)."""
    labels = _labels(summary)
    folder = os.path.dirname(os.path.abspath(path))
    own = _position_index(summary, folder, labels)
    out = []
    for k, label in enumerate(labels):
        if k == own:
            out.append((label, path))
            continue
        found = _parts.find(os.path.dirname(folder), label)
        md = _parts.find(found, METADATA) if found and os.path.isdir(found) else None
        if md is not None:
            other = read_summary(md)
            if other is None or not _same(summary, other):
                warnings.warn(CompositeWarning(
                    f"{md!r} is not a file of the acquisition {path!r} belongs to "
                    f"(position {label!r}): not used"), stacklevel=4)
                md = None
        out.append((label, md))
    return out


#: _main's answers by (metadata.txt, its size and time): every plane file of a
#: position asks for the same one (``group_files`` given the planes alone)
_MAINS: dict = {}
_MAINS_KEPT = 256


def _main(md: str) -> Optional[str]:
    """The main file of the acquisition position file *md* belongs to (the
    first position's ``metadata.txt`` that is there), or None when *md* is no
    Micro-Manager metadata file.  Cached on the file's size and time."""
    try:
        stat = os.stat(md)
    except OSError:
        return None
    key = (os.path.normcase(os.path.abspath(md)), stat.st_size, stat.st_mtime_ns)
    if key not in _MAINS:
        if len(_MAINS) >= _MAINS_KEPT:
            _MAINS.clear()
        _MAINS[key] = _find_main(md)
    main = _MAINS[key]
    return main if main is None or os.path.isfile(main) else _find_main(md)


def _find_main(md: str) -> Optional[str]:
    summary = read_summary(md)
    if summary is None:
        return None
    if _count(summary, "Positions") <= 1:
        return md
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", CompositeWarning)   # said again on opening
        return next(p for _label, p in positions(md, summary) if p is not None)


def _acquisition_files(folder: str) -> list:
    """The Micro-Manager position files in *folder*'s direct subfolders."""
    out = []
    try:
        with os.scandir(folder) as entries:
            subs = sorted(e.path for e in entries if e.is_dir())
    except OSError:
        return []
    for sub in subs:
        md = os.path.join(sub, METADATA)
        if os.path.isfile(md) and sniff(_parts.read_head(md), md):
            out.append(md)
    return out


def _folder_main(folder: str) -> Optional[str]:
    """The main file of the acquisition *folder* holds (its position folders),
    or None; several acquisitions there are an error naming them."""
    files = _acquisition_files(folder)
    if not files:
        return None
    mains = []
    for md in files:
        main = _main(md)
        if main is not None and not any(os.path.samefile(main, m) for m in mains):
            mains.append(main)
    if len(mains) > 1:
        raise MicroReaderError(
            f"{folder!r} holds {len(mains)} Micro-Manager acquisitions "
            f"({[os.path.relpath(m, folder) for m in mains]}): open one of their metadata.txt")
    return mains[0] if mains else None


# -- parts ------------------------------------------------------------------------

def plane_key(name: str, summary: dict) -> Optional[tuple]:
    """(t, c, z) of a plane file called *name* by the naming rule (1.4: the
    channel by its name in ``ChNames``), or None."""
    match = NAME_20.fullmatch(name)
    if match:
        return int(match["t"]), int(match["c"]), int(match["z"])
    match = NAME_14.fullmatch(name)
    if match:
        names = summary.get("ChNames") or []
        if match["channel"] in names:
            return int(match["t"]), names.index(match["channel"]), int(match["z"])
    return None


def _plane_name(name: str) -> bool:
    return NAME_20.fullmatch(name) is not None or NAME_14.fullmatch(name) is not None


def may_own(folder: str, name: str, folders) -> bool:
    """``group_files``' cheap test (reads nothing but the folder listing
    *folders* caches)."""
    lower = name.lower()
    if lower in (METADATA, DISPLAY):
        return True
    return _plane_name(name) and METADATA in folders.names(folder)


def owner(path: str, head):
    """The parts of an acquisition (module docstring): a plane TIFF named by
    the rule next to a Micro-Manager ``metadata.txt``, another position's
    ``metadata.txt``, ``display_and_comments.txt``, a position folder or the
    acquisition folder -> the main ``metadata.txt``."""
    from . import tiff
    if head is None:                                     # a folder
        if not os.path.isdir(path):
            return None
        md = os.path.join(path, METADATA)
        if os.path.isfile(md) and sniff(_parts.read_head(md), md):
            main = _main(md)
        else:
            main = _folder_main(path)
        return _parts.Part(main, main) if main is not None else None
    name = os.path.basename(path)
    lower = name.lower()
    folder = os.path.dirname(os.path.abspath(path))
    if lower == METADATA:
        if not sniff(head, path):
            return None
        main = _main(path)
        return None if main is None or os.path.samefile(main, path) else _parts.Part(main, main)
    if lower == DISPLAY:
        main = _folder_main(folder)
        return _parts.Part(main, main) if main is not None else None
    if head[:4] not in tiff.MAGIC or not _plane_name(name):
        return None
    md = _parts.find(folder, METADATA)
    if md is None or not sniff(_parts.read_head(md), md):
        return None
    main = _main(md)
    return _parts.Part(main, main, standalone=True) if main is not None else None


# -- the planes of one position ------------------------------------------------------

def _unwrap(value):
    """A 2.0-beta property value (``{"PropVal": v, "PropType": ...}``) -> v."""
    return value.get("PropVal") if isinstance(value, dict) and "PropVal" in value else value


def _planes(data: dict, folder: str, summary: dict, path: str) -> tuple:
    """({(t, c, z): file path}, {(t, c, z): the plane's metadata}) that a
    position's ``metadata.txt`` lists; without any entry, the folder's files
    named by the rule."""
    files, meta = {}, {}
    names = summary.get("ChNames") or []
    for key, value in data.items():
        if key.startswith("FrameKey-"):
            try:
                t, c, z = (int(v) for v in key[len("FrameKey-"):].split("-"))
            except ValueError:
                raise CorruptFileError(f"{path!r}: entry {key!r} is not FrameKey-<t>-<c>-<z>") from None
            value = value if isinstance(value, dict) else {}
            name = value.get("FileName")
            if not isinstance(name, str) or not name:
                channel = value.get("Channel") or (names[c] if c < len(names) else None)
                name = f"img_{t:09d}_{channel}_{z:03d}.tif" if channel else None
            files[(t, c, z)] = os.path.join(folder, name) if name else None
            meta[(t, c, z)] = value
        elif key.startswith("Coords-"):
            name = key[len("Coords-"):]
            value = value if isinstance(value, dict) else {}
            try:
                t, c, z = (int(value.get(k, 0)) for k in ("time", "channel", "z"))
            except (TypeError, ValueError):
                raise CorruptFileError(f"{path!r}: entry {key!r} has no time / channel / z") from None
            files[(t, c, z)] = os.path.join(folder, name)
            extra = data.get("Metadata-" + name)
            meta[(t, c, z)] = extra if isinstance(extra, dict) else {}
    if not files:
        try:
            listing = os.listdir(folder)
        except OSError:
            listing = []
        for name in listing:
            key = plane_key(name, summary)
            if key is not None:
                files[key] = os.path.join(folder, name)
    return files, meta


# -- the images --------------------------------------------------------------------

def _tiff_info(tif) -> dict:
    """A plane TIFF's own Micro-Manager metadata: the ImageJ ``Info`` JSON
    (1.4, 2.0 betas), else the Micro-Manager tag (51123)."""
    page = tif.pages[0]
    for code in (50839, 51123):
        tag = page.tags.get(code)
        if tag is None:
            continue
        value = tag.value
        if code == 50839:
            value = value.get("Info") if isinstance(value, dict) else None
        if isinstance(value, (bytes, str)):
            try:
                value = json.loads(value)
            except ValueError:
                continue
        if isinstance(value, dict):
            return {k: _unwrap(v) for k, v in value.items()}
    return {}


def _argb(value) -> Optional[str]:
    """A Java ARGB colour int (Micro-Manager ``ChColors``) -> '#RRGGBB'."""
    if not isinstance(value, int) or isinstance(value, bool):
        return None
    v = value & 0xFFFFFFFF
    return color_hex(((v >> 16) & 255, (v >> 8) & 255, v & 255))


def _positive(value) -> Optional[float]:
    value = _unwrap(value)
    return float(value) if isinstance(value, (int, float)) and not isinstance(
        value, bool) and value > 0 else None


def _differ(a: float, b: float) -> bool:
    return abs(a - b) > TOLERANCE * max(abs(a), abs(b), 1.0)


class _MMImage(PlaneImage):
    """One stage position: *planes* (t, c, z) -> (plane file, page 0);
    *meta*: the planes' entries in its ``metadata.txt``."""

    def __init__(self, *args, owner, meta: dict, **kwargs):
        super().__init__(*args, **kwargs)
        self._owner, self._meta = owner, meta

    def _read_metadata(self):
        f, summary = self._owner, self._owner._summary
        first = min(self._planes) if self._planes else None
        listed = self._meta.get(first, {}) if first is not None else {}
        info = {}
        if first is not None:                 # the first plane's TIFF fills gaps
            entry = self._members.acquire(self._planes[first][0])
            try:
                info = _tiff_info(entry[0]._tif)
            finally:
                self._members.release(entry)
        told = []

        def pick(what, *values):
            """The first value given, with a warning when a later one differs."""
            found = [(source, v) for source, v in values if v is not None]
            if not found:
                return None
            for source, v in found[1:]:
                if _differ(found[0][1], v):
                    told.append(f"{what}: {found[0][1]} in {found[0][0]}, {v} in {source}")
            return found[0][1]

        sizes = {}
        tif_name = os.path.basename(self._planes[first][0]) if first is not None else ""
        xy = pick("pixel size (um)", ("the Summary", _positive(summary.get("PixelSize_um"))),
                  ("metadata.txt", _positive(listed.get("PixelSizeUm"))),
                  (tif_name, _positive(info.get("PixelSizeUm", info.get("PixelSize_um")))))
        if xy is not None:
            sizes["x"] = sizes["y"] = quantity(xy, "micrometer")
        dz = _positive(summary.get("z-step_um"))
        if dz is not None and self.shape[2] > 1:
            sizes["z"] = quantity(dz, "micrometer")
        dt = f._time_step()
        if dt is not None and self.shape[0] > 1:
            sizes["t"] = quantity(dt, "millisecond")
        stage = {}
        for axis in "xyz":
            key = f"{axis.upper()}PositionUm"
            value = pick(f"stage {axis} (um)", ("metadata.txt", _number(listed.get(key))),
                         (tif_name, _number(info.get(key))))
            if value is not None:
                stage[axis] = quantity(value, "micrometer", positive=False)
        if told:
            warnings.warn(CompositeWarning(
                f"{f.path!r} ({self.name!r}): {'; '.join(told)}: using the first"), stacklevel=4)
        return Metadata(pixel_sizes=sizes, stage_position=stage, channels=f._channels())


def _number(value) -> Optional[float]:
    value = _unwrap(value)
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None


class _MMFile(File):
    backend = NAME
    format = "micromanager-files"   # image-per-file; MMStack stacks are the TIFF backend's "micromanager"

    def __init__(self, path: str):
        self.path = path
        self._members = Members(pages=True)
        try:
            images = self._images(path)
        except BaseException:
            self._members.close()
            raise
        super().__init__(path, images)

    def _images(self, path: str) -> list:
        data = read_metadata(path)
        self._summary = summary = data["Summary"]
        listed = positions(path, summary)
        self._acquisition = os.path.dirname(os.path.dirname(os.path.abspath(path)))
        self._position_files = [p for _label, p in listed if p is not None]
        absent = [label for label, p in listed if p is None]
        if absent:
            warnings.warn(MissingImagesWarning(
                f"{path!r}: {len(absent)} of the {len(listed)} stage positions its Summary lists "
                f"have no folder ({', '.join(absent[:10])}{', ...' if len(absent) > 10 else ''}): "
                "left out"), stacklevel=3)

        fields, metas = {}, {}
        for k, (label, md) in enumerate(listed):
            if md is None:
                continue
            own = data if md == path else read_metadata(md)
            if own is not data:
                other = own["Summary"]
                if any(other.get(key) != summary.get(key) for key in ("Frames", "Channels", "Slices")):
                    warnings.warn(CompositeWarning(
                        f"{md!r}: its Summary's sizes differ from {path!r}'s: using {path!r}'s"),
                        stacklevel=3)
            fields[k], metas[k] = _planes(own, os.path.dirname(md), summary, md)
        sizes = [max(1, _count(summary, key)) for key in ("Frames", "Channels", "Slices")]
        for planes in fields.values():
            for key in planes:
                sizes = [max(n, i + 1) for n, i in zip(sizes, key)]
        nt, nc, nz = sizes
        self._nc = nc
        wanted = [(t, c, z) for t in range(nt) for c in range(nc) for z in range(nz)]
        kept = keep_fields({k: {key: (p, 0) if p else None for key, p in planes.items()}
                            for k, planes in fields.items()}, wanted, path)

        first = next(iter(next(iter(kept.values())).values()))
        dtype, unit, shape_yx = self._members.page_info(first[0], 0)
        base = summary.get("Prefix") or os.path.basename(self._acquisition) or "micromanager"
        images = []
        for k in sorted(kept):
            label = listed[k][0]
            name = base if len(listed) == 1 else f"{base} {label}"
            images.append(_MMImage(self._members, len(images), kept[k], (nt, nc, nz) + tuple(shape_yx),
                                   dtype, name, unit, {"scene": k}, owner=self, meta=metas[k]))
        return images

    def _time_step(self) -> Optional[float]:
        """ms: ``Interval_ms``, else ``CustomIntervals_ms`` when its intervals
        after the first are all one value; 0 is unknown."""
        summary = self._summary
        step = _positive(summary.get("Interval_ms"))
        if step is not None:
            return step
        custom = summary.get("CustomIntervals_ms")
        if isinstance(custom, list) and len(custom) > 1:
            rest = {_positive(v) for v in custom[1:]}
            if len(rest) == 1 and None not in rest:
                return rest.pop()
        return None

    def _channels(self) -> tuple:
        names = self._summary.get("ChNames")
        colors = self._summary.get("ChColors")
        names = names if isinstance(names, list) else []
        colors = colors if isinstance(colors, list) and len(colors) == len(names) else []
        return tuple(Channel(name=str(names[c]) if c < len(names) and names[c] not in (None, "") else None,
                             color=_argb(colors[c]) if c < len(colors) else None)
                     for c in range(self._nc))

    def _side_files(self) -> set:
        """The other positions' ``metadata.txt`` and ``display_and_comments.txt``:
        parts of the acquisition for ``group_files``."""
        out = {p for p in self._position_files if p != self.path}
        display = _parts.find(self._acquisition, DISPLAY)
        if display is not None:
            out.add(display)
        return out

    def close(self) -> None:
        super().close()
        self._members.close()


def open(path: str, tiles: str = "separate", resolutions: str = "base") -> _MMFile:  # noqa: A001 - backend protocol
    """A Micro-Manager image-per-file acquisition from a position's
    ``metadata.txt`` (*tiles* and *resolutions* change nothing: positions are
    separate images, and the files hold no pyramids)."""
    return _MMFile(path)
