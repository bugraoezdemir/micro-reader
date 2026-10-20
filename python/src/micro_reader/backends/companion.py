"""OME companion files (``*.companion.ome``): docs/COMPANION.md, CO-1.

A companion is an OME-XML file whose images keep their pixels in other files,
OME-TIFFs: each ``TiffData`` says which file (``UUID FileName``), which pages
(``IFD``, ``PlaneCount``) and from which plane on (``FirstZ`` / ``FirstC`` /
``FirstT``, counted in ``DimensionOrder``). Open the companion file.

- One micro-reader image per OME ``Image``, axes ``t c z y x``, indexed by
  ``series`` (its place among the companion's images), stacked lazily from the
  member files (the file-set layer, ``_fileset``). The members are read as
  plain pages: the companion alone says what each page is.
- Metadata: the companion's, through the OME-XML parser OME-TIFF uses. It
  wins; the members' own metadata only fills its gaps, with a
  ``CompanionWarning`` (never their bare TIFF resolution tags). The z step
  comes from ``PhysicalSizeZ``, else the planes' even ``PositionZ`` steps,
  else the members.
- Matching: a member belongs to the set when its own OME UUID is the one the
  companion lists for it, or its ``BinaryOnly`` stub points to this
  companion's UUID; it is an error when a check fails and none passes; without
  UUIDs to compare the file name decides. Every member must fit its entry
  (size, pixel type). Members are checked when first read, not all at open.
- Paths: as written, relative to the companion; an absolute path that does
  not exist falls back to the same name in the companion's folder, with a
  ``CompanionWarning``.
- Missing members: as for every file set (``_fileset.keep_fields``).
- The parsed companion is cached per process (``CORE_CACHE`` companions, by
  path, modification time and size), apart from the pixel caches.

Not yet: companion plates (``Plate`` / ``Well`` / ``WellSample``), RGB
(``SamplesPerPixel`` > 1), pixels inside the XML (``BinData``), the members as
separate images (docs/COMPANION.md, "A").
"""
from __future__ import annotations

import builtins
import math
import os
import threading
import warnings
import xml.etree.ElementTree as ET
from collections import OrderedDict
from dataclasses import replace

import numpy as np

from .._base import File
from .._errors import CompanionWarning, CorruptFileError, UnsupportedFormatError
from .._fileset import Members, PlaneImage, keep_fields
from .._metadata import Metadata, quantity
from .tiff import PLAIN_PAGES
from ._tiff_metadata import (OME_DEFAULT_LENGTH, _children, _local, ome_annotations,
                             ome_image_metadata, ome_plane_position, tiff_metadata)

NAME = "companion"
#: the index file of a set: a folder holding it opens it (docs/COMPOSITE.md)
SET_INDEX = True

#: member files kept open at once: a read region often spans a hundred planes
#: (one file each), and reopening a member costs more than keeping its handle
OPEN_MEMBERS = 128
#: parsed companions kept per process
CORE_CACHE = 8
#: how unevenly PositionZ may step (as a fraction of the step) and still give z
Z_STEP_TOLERANCE = 0.01

#: OME Pixels Type -> numpy dtype
OME_TYPES = {"int8": "i1", "int16": "i2", "int32": "i4", "uint8": "u1", "uint16": "u2",
             "uint32": "u4", "float": "f4", "double": "f8", "complex": "c8",
             "double-complex": "c16", "bit": "?"}


def may_index(name: str) -> bool:
    """``group_files``' cheap test (reads nothing): an OME companion file (``.companion.ome``)."""
    return name.lower().endswith(('.ome', '.ome.xml'))


def sniff(head: bytes, path: str) -> bool:
    name = path.lower()
    return name.endswith((".companion.ome", ".ome", ".ome.xml")) and b"<OME" in head


# -- the parsed companion ------------------------------------------------------

class _Series:
    """One OME ``Image`` of a companion: its sizes, plane order and planes."""

    def __init__(self, number: int, element, name: str, sizes: dict, order: str, dtype):
        self.number, self.element, self.name = number, element, name
        #: {"t", "c", "z", "y", "x"} -> size
        self.sizes = sizes
        #: the plane axes, fastest first (DimensionOrder without X Y), e.g. "zct"
        self.order = order
        self.dtype = dtype
        #: (t, c, z) -> (member path, IFD)
        self.planes: dict = {}
        self._meta = None
        self._plane_at = None
        self._lock = threading.Lock()

    def companion_metadata(self, core: "Core"):
        """What the companion says about this image (pixel sizes with the
        PositionZ z step, channels, the first plane's position), worked out once:
        member files opened on their own (CO-2) all share it."""
        with self._lock:
            if self._meta is None:
                meta = ome_image_metadata(self.element, dict(self.sizes), self.sizes["c"],
                                          {"series": self.number},
                                          lambda: ome_annotations(core.root))
                meta = meta or Metadata.unknown(self.sizes["c"])
                if "z" not in meta.pixel_sizes and self.sizes["z"] > 1:
                    step = _position_z_step(self)
                    if step is not None:
                        meta = replace(meta, pixel_sizes={**meta.pixel_sizes, "z": step})
                        warnings.warn(CompanionWarning(
                            f"{core.path!r} image {self.number}: no PhysicalSizeZ; the z step "
                            f"{step.value:g} {step.unit} is taken from the planes' PositionZ"),
                            stacklevel=4)
                self._meta = meta
            return self._meta

    def plane_at(self, plane: tuple):
        """The OME ``Plane`` element of (t, c, z), or None."""
        with self._lock:
            if self._plane_at is None:
                pixels = _children(self.element, "Pixels")[0]
                self._plane_at = {(int(e.get("TheT", 0)), int(e.get("TheC", 0)),
                                   int(e.get("TheZ", 0))): e
                                  for e in _children(pixels, "Plane")}
            return self._plane_at.get(tuple(plane))

    def flat(self, t: int, c: int, z: int) -> int:
        at, i, step = {"t": t, "c": c, "z": z}, 0, 1
        for axis in self.order:
            i += at[axis] * step
            step *= self.sizes[axis]
        return i

    def place(self, i: int) -> tuple:
        at = {}
        for axis in self.order:
            at[axis] = i % self.sizes[axis]
            i //= self.sizes[axis]
        return at["t"], at["c"], at["z"]

    @property
    def n_planes(self) -> int:
        return self.sizes["t"] * self.sizes["c"] * self.sizes["z"]


class Core:
    """A companion file, parsed: its images (``series``), the UUID it lists
    for each member and the IFDs it uses of it, the path notes."""

    def __init__(self, path: str):
        self.path = path
        try:
            self.root = _ome_root(path)
        except ET.ParseError as exc:
            raise CorruptFileError(f"{path!r}: not readable XML ({exc})") from None
        except OSError as exc:
            raise CorruptFileError(f"{path!r}: {exc}") from None
        if _local(self.root.tag) != "OME":
            raise CorruptFileError(f"{path!r}: not OME-XML (root {_local(self.root.tag)!r})")
        self.uuid = self.root.get("UUID")
        self.folder = os.path.dirname(os.path.abspath(path))
        #: member path -> the UUID the companion gives it (None: not given)
        self.member_uuid: dict = {}
        #: member path -> [(series, (t, c, z), IFD), ...] the companion reads from it
        self.member_pages: dict = {}
        #: (as written, as resolved) for absolute paths that did not exist
        self.moved: list = []
        self.series = [self._series(n, e) for n, e in enumerate(
            e for e in self.root if _local(e.tag) == "Image")]
        if not self.series:
            raise CorruptFileError(f"{path!r}: the companion lists no images")
        #: lookups for a member file opened on its own (CO-2)
        self.by_uuid = {u: p for p, u in self.member_uuid.items() if u}
        self.by_path = {os.path.normcase(os.path.abspath(p)): p for p in self.member_pages}

    def _series(self, number: int, element) -> _Series:
        where = f"{self.path!r} image {number}"
        pixels = _children(element, "Pixels")
        if not pixels:
            raise CorruptFileError(f"{where}: no Pixels")
        p = pixels[0]
        if _children(p, "BinData"):
            raise UnsupportedFormatError(f"{where}: pixels inside the XML (BinData) are not read")
        try:
            sizes = {a: int(p.get(f"Size{a.upper()}", 1)) for a in "tczyx"}
        except ValueError:
            raise CorruptFileError(f"{where}: a Pixels size is not a whole number") from None
        if min(sizes.values()) < 1:
            raise CorruptFileError(f"{where}: a Pixels size below 1 ({sizes})")
        samples = max([int(c.get("SamplesPerPixel", 1) or 1) for c in _children(p, "Channel")]
                      or [1])
        if samples > 1:
            raise UnsupportedFormatError(f"{where}: RGB images (SamplesPerPixel {samples}) in "
                                         "a companion are not read yet")
        order = (p.get("DimensionOrder") or "XYZCT").upper()
        if order[:2] != "XY" or sorted(order[2:]) != ["C", "T", "Z"]:
            raise CorruptFileError(f"{where}: DimensionOrder {order!r}")
        kind = (p.get("Type") or "").lower()
        dtype = np.dtype(OME_TYPES[kind]) if kind in OME_TYPES else None
        series = _Series(number, element, element.get("Name") or f"image{number}", sizes,
                         order[2:].lower(), dtype)
        for data in _children(p, "TiffData"):
            self._tiffdata(series, data, where)
        return series

    def _tiffdata(self, series: _Series, data, where: str) -> None:
        uuid = _children(data, "UUID")
        name = uuid[0].get("FileName") if uuid else None
        if not name:
            raise CorruptFileError(f"{where}: a TiffData names no file (a companion holds no "
                                   "pixels of its own)")
        listed = (uuid[0].text or "").strip() or None
        member = self._resolve(name)
        try:
            first = series.flat(int(data.get("FirstT", 0)), int(data.get("FirstC", 0)),
                                 int(data.get("FirstZ", 0)))
            ifd = int(data.get("IFD", 0))
            if "PlaneCount" in data.attrib:
                count = int(data.get("PlaneCount"))
            elif "IFD" in data.attrib:
                count = 1                      # OME: one IFD when an IFD is given
            else:
                count = _page_count(member)    # OME: every IFD of the file
        except ValueError:
            raise CorruptFileError(f"{where}: a TiffData attribute is not a whole number") from None
        if first + count > series.n_planes:
            raise CorruptFileError(f"{where}: TiffData for {name!r} runs past the image's "
                                   f"{series.n_planes} planes")
        known = self.member_uuid.setdefault(member, listed)
        if listed and known and listed != known:
            raise CorruptFileError(f"{where}: {name!r} is listed with two UUIDs")
        for k in range(count):
            plane = series.place(first + k)
            if plane in series.planes:
                raise CorruptFileError(f"{where}: plane (t, c, z) {plane} is listed twice")
            series.planes[plane] = (member, ifd + k)
            self.member_pages.setdefault(member, []).append((series, plane, ifd + k))

    def _resolve(self, name: str) -> str:
        """A member path: as written, relative to the companion; an absolute
        path that does not exist falls back to its name in this folder."""
        path = name if os.path.isabs(name) else os.path.join(self.folder, name)
        path = os.path.normpath(path)
        if os.path.isabs(name) and not os.path.isfile(path):
            here = os.path.join(self.folder, os.path.basename(name.replace("\\", "/")))
            self.moved.append((name, here))
            return here
        return path

    def check_member(self, path: str, member) -> None:
        """Raise when member file *path* (opened as pages) is not part of this
        set, or has pages of another size or type.

        It belongs when either check passes (Bugra, 2026-10-14): its own UUID
        is the one the companion lists for it, or its ``BinaryOnly`` stub points
        to this companion's UUID. Viventis writes a member UUID that differs
        from the companion's entry, with a stub that points to the companion.
        It is refused only when a check fails and none passes; without the
        UUIDs to compare, its file name decides."""
        listed = self.member_uuid.get(path)
        own, points_to = _own_uuids(member)
        checks = [None if not (a and b) else a == b
                  for a, b in ((own, listed), (points_to, self.uuid))]
        if False in checks and True not in checks:
            told = []
            if checks[0] is False:
                told.append(f"its UUID is {own}, the companion lists {listed}")
            if checks[1] is False:
                told.append(f"it points to the companion {points_to}, this one is {self.uuid}")
            raise CorruptFileError(f"{path!r} is not part of the set {self.path!r} "
                                   f"describes: {'; '.join(told)}")
        from .._fileset import _page
        for series, _plane, ifd in self.member_pages.get(path, ()):
            image, _k = _page(member, ifd, path)
            want = (series.sizes["y"], series.sizes["x"])
            if tuple(image.shape[-2:]) != want or (series.dtype is not None
                                                   and image.dtype != series.dtype):
                raise CorruptFileError(
                    f"{path!r} page {ifd}: {image.shape[-2]} x {image.shape[-1]} "
                    f"{image.dtype}, the companion {self.path!r} expects {want[0]} x {want[1]} "
                    f"{series.dtype}")


#: the first bytes of a TIFF / BigTIFF, either byte order
TIFF_MAGIC = (b"II*\x00", b"MM\x00*", b"II+\x00", b"MM\x00+")


def _ome_root(path: str):
    """The OME-XML root of *path*: a companion file's own XML, or -- when a
    ``BinaryOnly`` stub's ``MetadataFile`` is an OME-TIFF holding the full
    XML -- the XML of that TIFF's first page."""
    with builtins.open(path, "rb") as fh:
        magic = fh.read(4)
    if magic not in TIFF_MAGIC:
        return ET.parse(path).getroot()
    import tifffile
    try:
        with tifffile.TiffFile(path, **PLAIN_PAGES) as tif:
            text = tif.pages.first.description or ""
    except tifffile.TiffFileError as exc:
        raise CorruptFileError(f"{path!r}: {exc}") from None
    if "<OME" not in text:
        raise CorruptFileError(f"{path!r}: a TIFF without OME-XML in its first page")
    return ET.fromstring(text)


def _page_count(path: str) -> int:
    """Pages of a member file (an OME TiffData without IFD or PlaneCount covers
    them all); 1 when the file is missing (it reads as zeros anyway)."""
    if not os.path.isfile(path):
        return 1
    import tifffile
    with tifffile.TiffFile(path, **PLAIN_PAGES) as tif:
        return len(tif.pages)


def _own_uuids(member) -> tuple:
    """(the UUID of a member's own OME-XML, the companion UUID its
    ``BinaryOnly`` stub points to), each None when absent; from its first
    page's description."""
    try:
        text = member._tif.pages.first.description or ""
    except Exception:                                    # noqa: BLE001 - no description
        return None, None
    if "<OME" not in text:
        return None, None
    try:
        root = ET.fromstring(text)
    except ET.ParseError:
        return None, None
    stub = next((e for e in root if _local(e.tag) == "BinaryOnly"), None)
    return root.get("UUID"), (stub.get("UUID") if stub is not None else None)


_cores: "OrderedDict[tuple, Core]" = OrderedDict()
_cores_lock = threading.Lock()


def load(path: str) -> Core:
    """The parsed companion at *path*, parsed once per process while it does not
    change (by path, modification time and size); the CORE_CACHE most recently
    used are kept."""
    path = os.path.abspath(path)
    try:
        stat = os.stat(path)
    except OSError as exc:
        raise CorruptFileError(f"{path!r}: {exc}") from None
    key = (os.path.normcase(path), stat.st_mtime_ns, stat.st_size)
    with _cores_lock:
        core = _cores.get(key)
        if core is not None:
            _cores.move_to_end(key)
            return core
    core = Core(path)
    with _cores_lock:
        _cores[key] = core
        while len(_cores) > CORE_CACHE:
            _cores.popitem(last=False)
    return core


# -- the images ----------------------------------------------------------------

class _CompanionImage(PlaneImage):
    """One OME ``Image`` of a companion, stacked from its member files."""

    def __init__(self, *args, core: Core, series: _Series, **kwargs):
        super().__init__(*args, **kwargs)
        self._core, self._series_info = core, series

    def _read_metadata(self):
        return _series_metadata(self._core, self._series_info, self.index,
                                members=self._members_sizes)

    def _members_sizes(self, axes: list) -> tuple:
        """({axis: size} the first readable member's own metadata gives for
        *axes*, that member's path); never the bare resolution tags."""
        from . import tiff
        for path, _ifd in sorted(self._planes.values(), key=lambda p: p[1]):
            if not os.path.isfile(path):
                continue
            try:
                member = tiff.open(path, "separate", "base", companion=False)
            except Exception:                            # noqa: BLE001 - e.g. multi-file OME
                continue
            try:
                image = member.images[0]
                n = image.shape[image.axes.index("c")] if "c" in image.axes else 1
                own = tiff_metadata(image, n, {}, resolution_tags=False)
            finally:
                member.close()
            if own is None:
                return {}, path
            return {a: own.pixel_sizes[a] for a in axes if a in own.pixel_sizes}, path
        return {}, None


def _series_metadata(core: Core, s: _Series, index: dict, *, members=None, start=(0, 0, 0),
                     n_channels=None):
    """The metadata of series *s* as an image beginning at *start* (t, c, z)
    with *n_channels* channels (default: the whole series): the companion's
    (worked out once per series, ``_Series.companion_metadata``), the position
    of its own first plane, and (CO-1) the members' own metadata for what is
    still missing (*members(axes)* -> ({axis: size}, member path)), with a
    CompanionWarning."""
    meta = s.companion_metadata(core)
    n = s.sizes["c"] if n_channels is None else n_channels
    changes = {"channels": meta.channels[start[1]:start[1] + n]}
    if tuple(start) != (0, 0, 0):
        plane = s.plane_at(start)
        if plane is not None:
            changes["position"] = ome_plane_position(plane, meta.pixel_sizes, s.sizes)
    sizes = dict(meta.pixel_sizes)
    missing = [a for a in "xyzt" if a not in sizes and (a in "xy" or s.sizes[a] > 1)]
    if missing and members is not None:
        found, source = members(missing)
        sizes.update(found)
        if found:
            changes["pixel_sizes"] = sizes
            warnings.warn(CompanionWarning(
                f"{core.path!r} image {s.number}: the companion gives no "
                f"{', '.join(sorted(found))} size; taken from the member file "
                f"{os.path.basename(source)!r}"), stacklevel=3)
    return replace(meta, **changes)


def _position_z_step(s: _Series):
    """The z step from the planes' PositionZ at the first t and c, when they
    step evenly; None otherwise."""
    pixels = _children(s.element, "Pixels")[0]
    at = {}
    unit = None
    for plane in _children(pixels, "Plane"):
        if int(plane.get("TheT", 0)) or int(plane.get("TheC", 0)):
            continue
        try:
            at[int(plane.get("TheZ", 0))] = float(plane.get("PositionZ"))
        except (TypeError, ValueError):
            continue
        unit = plane.get("PositionZUnit") or OME_DEFAULT_LENGTH
    if len(at) != s.sizes["z"] or len(at) < 2:
        return None
    zs = [at[k] for k in sorted(at)]
    steps = [b - a for a, b in zip(zs, zs[1:])]
    step = steps[0]
    if not step or not math.isfinite(step) or any(
            abs(d - step) > Z_STEP_TOLERANCE * abs(step) for d in steps):
        return None
    return quantity(abs(step), unit)


class _CompanionFile(File):
    backend = NAME
    format = "ome-companion"

    def __init__(self, path: str):
        core = load(path)
        if core.moved:
            written, used = core.moved[0]
            warnings.warn(CompanionWarning(
                f"{path!r}: {len(core.moved)} member paths are absolute and do not exist; read "
                f"from the companion's folder instead (e.g. {written!r} -> {used!r})"),
                stacklevel=3)
        self._members = Members(limit=OPEN_MEMBERS, pages=True, check=core.check_member)
        images = []
        try:
            for s in core.series:
                wanted = [s.place(i) for i in range(s.n_planes)]
                try:
                    kept = keep_fields({s.number: s.planes}, wanted, f"{path} image {s.number}")
                except CorruptFileError:
                    continue                    # none of its files: left out (warned)
                planes = kept[s.number]
                member, ifd = next(iter(planes.values()))
                dtype, unit, shape_yx = self._members.page_info(member, ifd)
                shape = (s.sizes["t"], s.sizes["c"], s.sizes["z"]) + tuple(shape_yx)
                images.append(_CompanionImage(
                    self._members, len(images), planes, shape, dtype, s.name, unit,
                    {"series": s.number}, core=core, series=s))
            if not images:
                raise CorruptFileError(f"{path!r}: none of its images has any of its files")
        except BaseException:
            self._members.close()
            raise
        super().__init__(path, images)

    def close(self) -> None:
        super().close()
        self._members.close()


# -- CO-2: one member file, its metadata from its companion ----------------------

class _MemberImage(_CompanionImage):
    """The planes one member file holds, as a block of its companion image
    beginning at *start* (t, c, z)."""

    def __init__(self, *args, start: tuple, **kwargs):
        super().__init__(*args, **kwargs)
        self._start = start

    def _read_metadata(self):
        # the file opened wins; a BinaryOnly file has no metadata of its own, so
        # all of it comes from the companion (never from other members)
        return _series_metadata(self._core, self._series_info, self.index, start=self._start,
                                n_channels=self.shape[1])


class _MemberFile(File):
    """A member file opened on its own, its metadata from its companion."""
    backend = "tiff"
    format = "ome-tiff"

    def __init__(self, path: str, images, members: Members, companion: str):
        super().__init__(path, images)
        self._members = members
        #: the companion file the metadata came from
        self.companion = companion

    def close(self) -> None:
        super().close()
        self._members.close()


def _stub(tif):
    """(own UUID, companion as written, the companion UUID it points to) of a
    TIFF whose OME-XML is a ``BinaryOnly`` stub; None otherwise."""
    text = tif.ome_metadata or ""
    if "BinaryOnly" not in text:
        return None
    try:
        root = ET.fromstring(text)
    except ET.ParseError:
        return None
    stub = next((e for e in root if _local(e.tag) == "BinaryOnly"), None)
    if stub is None or not stub.get("MetadataFile"):
        return None
    return root.get("UUID"), stub.get("MetadataFile"), stub.get("UUID")


def _has_channel_info(channel) -> bool:
    return any(v is not None for v in (channel.name, channel.color, channel.emission_wavelength,
                                       channel.excitation_wavelength))


class _NotFollowed(Exception):
    """A ``BinaryOnly`` stub's metadata file that cannot be used: *why*, and
    the file as the stub writes it (*written*)."""

    def __init__(self, why: str, written: str):
        super().__init__(why)
        self.why, self.written = why, written


def _follow(path: str, tif):
    """The metadata file TIFF *path*'s ``BinaryOnly`` stub points to, checked:
    (where, parsed core, the file's key in it); None when the file is not a
    stub.  ``_NotFollowed`` when the metadata file is missing, unreadable, does
    not list the file, describes another set or another size / type.

    - The stub's ``MetadataFile`` is followed as written, relative to the file;
      an absolute path that does not exist falls back to its name in the
      file's folder (with a CompanionWarning).
    - The file is found in the metadata file by its UUID, else by its path; it
      is recognised when either check passes (its UUID is the entry's, or its
      stub points to this metadata file's UUID), and its pages must fit their
      entries (size, type, one sample)."""
    found = _stub(tif)
    if found is None:
        return None
    own, written, points_to = found
    folder = os.path.dirname(os.path.abspath(path))
    where = os.path.normpath(written if os.path.isabs(written) else os.path.join(folder, written))
    if os.path.isabs(written) and not os.path.isfile(where):
        here = os.path.join(folder, os.path.basename(written.replace("\\", "/")))
        if os.path.isfile(here):
            warnings.warn(CompanionWarning(
                f"{path!r}: its companion {written!r} does not exist; {here!r} is used"),
                stacklevel=4)
            where = here
    if not os.path.isfile(where):
        raise _NotFollowed(f"is not there ({where!r})", written)
    try:
        core = load(where)
    except (CorruptFileError, UnsupportedFormatError) as exc:
        raise _NotFollowed(f"cannot be read ({exc})", written) from None
    key = core.by_uuid.get(own) if own else None
    if key is None:
        key = core.by_path.get(os.path.normcase(os.path.abspath(path)))
    if key is None:
        raise _NotFollowed("does not list it", written)
    listed = core.member_uuid.get(key)
    checks = [None if not (a and b) else a == b for a, b in ((own, listed), (points_to, core.uuid))]
    if False in checks and True not in checks:
        raise _NotFollowed("describes another set (its UUIDs do not match)", written)
    misfit = _misfit(tif, core, key)
    if misfit:
        raise _NotFollowed(misfit, written)
    return where, core, key


def metadata_file_of(path: str, tif):
    """``open(..., multifile=True)`` (the default) on a member of a set: the
    metadata file its ``BinaryOnly`` stub points to, checked as ``_follow``
    does -- the set opens as if that file were opened (docs/COMPANION.md,
    "multifile chooses between CO-1 and CO-2").  None when the file is not a
    stub.  A metadata file that cannot be followed is an error: the set cannot
    be opened, and ``multifile=False`` opens the file alone."""
    try:
        followed = _follow(path, tif)
    except _NotFollowed as exc:
        raise CorruptFileError(
            f"{path!r} keeps its metadata in {exc.written!r}, which {exc.why}: the set it "
            f"belongs to cannot be opened. multifile=False opens this file alone.") from None
    return None if followed is None else followed[0]


def member_view(path: str, tif):
    """CO-2, ``open(..., multifile=False)`` (docs/COMPANION.md): the images of
    TIFF *path* (opened as *tif*) with their metadata from the metadata file
    its ``BinaryOnly`` stub points to (``_follow``).  Its planes must form one
    block of an image there; it opens as that block, axes ``t c z y x``
    without the size-1 axes the metadata says nothing about (no z / t size,
    no channel information).

    None when the file is not a stub (it opens as it is); None with a loud
    CompanionWarning when the metadata file cannot be followed (it opens with
    default pixel sizes). When it returns the images, it has taken over *tif*
    (kept as the member, or closed)."""
    try:
        followed = _follow(path, tif)
    except _NotFollowed as exc:
        warnings.warn(CompanionWarning(
            f"{path!r} keeps its metadata in the companion {exc.written!r}, which {exc.why}: it "
            f"opens with default pixel sizes. Open the companion file to read the whole set."),
            stacklevel=3)
        return None
    if followed is None:
        return None
    where, core, key = followed
    return _member_file(path, tif, core, key, where)


def _misfit(tif, core: Core, key: str):
    """Why member *key*'s pages (in the open *tif*) do not fit what *core*
    says of them, or None."""
    for s, _plane, ifd in core.member_pages[key]:
        page = tif.pages[ifd] if ifd < len(tif.pages) else None
        if page is None:
            return f"lists its page {ifd}, which it does not have"
        if ((page.imagelength, page.imagewidth) != (s.sizes["y"], s.sizes["x"])
                or page.samplesperpixel != 1
                or (s.dtype is not None and np.dtype(page.dtype) != s.dtype)):
            return "describes it with another size or pixel type"
    return None


def single_file_view(path: str, tif):
    """``open(..., multifile=False)``: an OME-TIFF whose own OME-XML lists
    planes in other files, opened with only the planes it holds -- as the
    block of its image they form, with the metadata of that XML (as a
    companion's member opens, CO-2).  None when its XML lists no other file
    (it opens as it is).  Its pages must fit the XML and form one block, else
    ``CorruptFileError`` / ``UnsupportedFormatError``."""
    try:
        core = load(path)
    except (CorruptFileError, UnsupportedFormatError):
        return None                                   # no usable XML: it opens as it is
    me = os.path.normcase(os.path.abspath(path))
    if all(os.path.normcase(p) == me for p in core.member_pages):
        return None                                   # a single-file OME-TIFF
    key = core.by_path.get(me)
    if key is None:
        root_uuid = core.root.get("UUID")
        key = core.by_uuid.get(root_uuid) if root_uuid else None
    if key is None:
        raise CorruptFileError(f"{path!r}: its OME-XML lists planes in other files but none "
                               "in this one")
    misfit = _misfit(tif, core, key)
    if misfit:
        raise CorruptFileError(f"{path!r}: its own OME-XML {misfit}")
    return _member_file(path, tif, core, key, path)


def _member_file(path: str, tif, core: Core, key: str, where: str):
    """Member *key* of *core* (the file *path*, open as *tif*) as its own
    file: the block of its image its planes form, metadata from *core*
    (*where*: the file the metadata came from).  Takes over *tif*."""
    from . import tiff
    members = Members(limit=4, pages=True)
    if tiff.plain_pages(tif):
        # the file is open already and tifffile lists its pages as they are: no
        # second open (a member read again later, after closing, opens as pages)
        members.adopt(path, tiff.pages_file(tif, path))
    else:
        tif.close()
    images = []
    by_series: dict = {}
    for s, plane, ifd in core.member_pages[key]:
        by_series.setdefault(s, {})[plane] = ifd
    try:
        for s, mine in by_series.items():
            ts, cs, zs = (sorted({plane[k] for plane in mine}) for k in range(3))
            spans = [range(v[0], v[-1] + 1) for v in (ts, cs, zs)]
            if len(mine) != len(spans[0]) * len(spans[1]) * len(spans[2]):
                raise UnsupportedFormatError(
                    f"{path!r}: its pages are not one block of {s.name!r} in the OME-XML of "
                    f"{where!r}; open that file to read the set")
            start = (ts[0], cs[0], zs[0])
            planes = {(t - ts[0], c - cs[0], z - zs[0]): (path, ifd)
                      for (t, c, z), ifd in mine.items()}
            dtype, unit, shape_yx = members.page_info(path, next(iter(planes.values()))[1])
            image = _MemberImage(members, len(images), planes,
                                 tuple(len(r) for r in spans) + tuple(shape_yx), dtype, s.name,
                                 unit, {"series": len(images)}, core=core, series=s, start=start)
            meta = image.metadata
            drop = "".join(a for a, n, explicit in (
                ("t", len(spans[0]), "t" in meta.pixel_sizes),
                ("c", len(spans[1]), any(_has_channel_info(ch) for ch in meta.channels)),
                ("z", len(spans[2]), "z" in meta.pixel_sizes)) if n == 1 and not explicit)
            images.append(image.as_axes(squeeze=drop) if drop else image)
    except BaseException:
        members.close()
        raise
    return _MemberFile(path, images, members, where)


def open(path: str, tiles: str = "separate", resolutions: str = "base") -> _CompanionFile:  # noqa: A001 - backend protocol
    """The images of an OME companion file (*tiles* and *resolutions* change
    nothing: members are plain pages, without tiles or pyramids)."""
    return _CompanionFile(path)
