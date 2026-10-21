"""Olympus / Evident FluoView multi-area tiling sets: the ``.omp2info`` index
(docs/COMPOSITE.md, OIR; relation "pointed, index only").

FluoView's multi-area time-lapse protocol (XML root ``matl:properties``) names
the files of a tiling acquisition: the overview map (``matl:map/matl:image``)
and, per group (``matl:group``, a mosaic region), every tile
(``matl:area/matl:image``) with its place ``xIndex`` / ``yIndex`` on a grid of
``numOfXAreas`` x ``numOfYAreas`` areas of ``areaWidth`` x ``areaHeight``
from ``areaLeft`` / ``areaTop`` (nm, stage coordinates).  The tiles are OIR
files that store no x / y stage position themselves: the index is the only
record of where they go.

- One image per tile (``tiles="separate"``), index ``series`` (the group) and
  ``tile`` (its place in the index), read from its own file (opened when first
  read); its metadata is the tile file's, with the position from the index
  (the first pixel's centre: ``areaLeft + xIndex * step`` plus half a pixel;
  y likewise).  Positions are nominal (the protocol's grid), not registered.
- The grid step is ``areaWidth * (1 - overlap / 200)``: the protocol's
  ``overlap`` (``matl:stage/matl:overlap``, %) shared by the two tiles of a
  seam.  Derived from the one sample (2026-10-18): ``areaWidth`` is exactly a
  tile's field (512 px); FluoView's own stitched image holds all 25 tiles
  pixel for pixel, at a fitted step of 487.2 px in y and 490.7 in x (+-9 px
  registration corrections).  This rule's 486.4 px matches y and is 4.3 px
  per column short in x; a step of the whole field (512) or of the full 10 %
  overlap (460.8) is off by 21-30 px per tile.  To confirm on a second sample.
- ``tiles="mosaic"``: one image per group, the tiles placed at those positions
  on one canvas (``_mosaic.Mosaic``: fusion "overwrite", later tile on top;
  areas without a tile read as 0).
- The instrument's stitched result (``Stitch_<area>_<group>``) is not named by
  the index; it is an image of its own (derived, registered).
- Missing tile files: a ``MissingImagesWarning``, the others read; none: an
  error.  ``group_files``: the tiles are the members, the map (its ``.oir`` and
  ``.hdf5``) a side file.
"""
from __future__ import annotations

import os
import threading
import warnings
import xml.etree.ElementTree as ET
from dataclasses import replace

from .. import _parts
from .._base import File, Image
from .._errors import CorruptFileError, MissingImagesWarning, UnsupportedFormatError
from .._metadata import quantity
from .._mosaic import Mosaic

NAME = "omp2info"
#: the index file of a set: a folder holding it opens it (docs/COMPOSITE.md)
SET_INDEX = True
MATL = "{http://www.olympus.co.jp/hpf/protocol/matl/model/matl}"


def sniff(head: bytes, path: str) -> bool:
    return b"<matl:properties" in head


def may_index(name: str) -> bool:
    """``group_files``' cheap test (reads nothing)."""
    return name.lower().endswith(".omp2info")


def _number(element, tag: str, where: str) -> float:
    text = element.findtext(MATL + tag)
    try:
        return float(text)
    except (TypeError, ValueError):
        raise CorruptFileError(f"{where}: no number in matl:{tag} ({text!r})") from None


class _Members:
    """The tile files, each opened once when first read, closed with the set."""

    def __init__(self):
        self._open: dict = {}
        self._lock = threading.Lock()

    def get(self, path: str) -> File:
        with self._lock:
            f = self._open.get(path)
            if f is None:
                from .._registry import open as open_file
                f = self._open[path] = open_file(path)
            return f

    def close(self) -> None:
        with self._lock:
            files, self._open = list(self._open.values()), {}
        for f in files:
            f.close()


class _Tile(Image):
    """One tile of a tiling set, read from its own file; *corner*: the nominal
    position of its first pixel's corner (x, y, nm)."""

    def __init__(self, position: int, members: _Members, path: str, like: Image, name: str,
                 index: dict, corner: tuple):
        super().__init__(position, like.axes, like.shape, like.dtype, name=name,
                         read_unit=like.read_unit, index=index)
        self._members, self._path, self._corner = members, path, corner
        self._closed = False

    def _image(self) -> Image:
        if self._closed:
            raise ValueError(f"{self.name!r} has been closed: its file was closed "
                             "(File.close() or the end of a with block)")
        f = self._members.get(self._path)
        if len(f.images) != 1 or f.images[0].axes != self.axes or \
                f.images[0].shape != self.shape or f.images[0].dtype != self.dtype:
            got = [(im.axes, im.shape, str(im.dtype)) for im in f.images]
            raise CorruptFileError(f"{self._path!r}: a tile of {self.name!r} holds {got}, the set's "
                                   f"tiles are {self.axes} {self.shape} {self.dtype}")
        return f.images[0]

    def _read_block(self, block):
        return self._image()._read_block(block)

    def _read_metadata(self):
        md = self._image().metadata
        return replace(md, position=_first_pixel(md, self._corner))

    def close(self) -> None:
        self._closed = True


def _first_pixel(md, corner: tuple) -> dict:
    """*md*'s position with x / y the centre of the first pixel of a tile whose
    corner is *corner* (nm); z as the tile file has it."""
    position = dict(md.position)
    for axis, value in zip("xy", corner):
        px = md.pixel_sizes.get(axis)
        if px is not None:
            position[axis] = quantity(value / 1000 + px.to("micrometer").value / 2,
                                      "micrometer", positive=False)
    return position


class _TilingMosaic(Mosaic):
    """A group's tiles on one canvas; its metadata is the first tile's, placed
    at the canvas corner."""

    def __init__(self, position, tiles, name, index, corner):
        super().__init__(position, tiles, name, index)
        self._corner = corner

    def _read_metadata(self):
        md = self._tiles[0][0].metadata
        return replace(md, position=_first_pixel(md, self._corner))


class _TilingFile(File):
    backend = NAME
    format = "omp2info"

    def __init__(self, path: str, tiles: str = "separate"):
        self._members = _Members()
        try:
            self._build(path, tiles)
        except BaseException:
            self._members.close()
            raise

    def _build(self, path: str, tiles: str) -> None:
        where = os.path.basename(path)
        try:
            root = ET.parse(path).getroot()
        except ET.ParseError as exc:
            raise CorruptFileError(f"{path!r}: not readable XML ({exc})") from None
        folder = os.path.dirname(os.path.abspath(path))
        self._folder = folder
        self._map = (root.findtext(f"{MATL}map/{MATL}image") or "").strip() or None
        groups = []                              # (areas [(file, x, y)], info) per mosaic group
        for group in root.iter(MATL + "group"):
            info = group.find(MATL + "areaInfo")
            areas = [((a.findtext(MATL + "image") or "").strip(),
                      int(_number(a, "xIndex", where)), int(_number(a, "yIndex", where)))
                     for a in group.findall(MATL + "area")]
            if info is not None and areas:
                groups.append((areas, {k: _number(info, k, where) for k in
                                       ("areaLeft", "areaTop", "areaWidth", "areaHeight")}))
        if not groups:
            raise CorruptFileError(f"{path!r}: the protocol lists no tiled area")
        overlap = root.findtext(f"{MATL}stage/{MATL}overlap")
        try:
            share = 1 - float(overlap) / 200 if overlap else 1.0    # half the overlap per seam
        except ValueError:
            raise CorruptFileError(f"{path!r}: no number in matl:overlap ({overlap!r})") from None

        found, missing = {}, []
        for areas, _info in groups:
            for name, _x, _y in areas:
                hit = _parts.find(folder, name) if name else None
                if hit is None:
                    missing.append(name)
                else:
                    found[name] = hit
        if not found:
            raise CorruptFileError(f"{path!r}: none of its {len(missing)} tile files is there "
                                   f"(e.g. {missing[0]!r})")
        if missing:
            warnings.warn(MissingImagesWarning(
                f"{path!r}: {len(missing)} of its tile files are missing (e.g. {missing[0]!r}); "
                "the others are read"), stacklevel=4)
        # every tile is shaped as the first one found (checked when each is read)
        first = self._members.get(next(iter(found.values())))
        if len(first.images) != 1:
            raise UnsupportedFormatError(
                f"{path!r}: its tiles hold {len(first.images)} images each (lambda scans?): "
                "a tiling set of such tiles is not read yet")
        like = first.images[0]
        px = like.metadata.pixel_sizes
        if tiles == "mosaic" and not ("x" in px and "y" in px):
            raise UnsupportedFormatError(f"{path!r}: tiles='mosaic': its tiles have no pixel size")

        images = []
        for g, (areas, info) in enumerate(groups):
            placed = []
            for k, (name, xi, yi) in enumerate(areas):
                if name not in found:
                    continue
                corner = (info["areaLeft"] + xi * info["areaWidth"] * share,
                          info["areaTop"] + yi * info["areaHeight"] * share)
                tile = _Tile(len(images) if tiles == "separate" else 0, self._members,
                             found[name], like, f"{where} group {g} tile {k} "
                             f"({os.path.splitext(name)[0]})", {"series": g, "tile": k}, corner)
                placed.append((tile, xi, yi, corner))
                if tiles == "separate":
                    images.append(tile)
            if tiles == "mosaic" and placed:
                step_x = info["areaWidth"] * share / px["x"].to("nanometer").value
                step_y = info["areaHeight"] * share / px["y"].to("nanometer").value
                corner = (min(c[0] for *_r, c in placed), min(c[1] for *_r, c in placed))
                images.append(_TilingMosaic(
                    len(images), [(t, round(yi * step_y), round(xi * step_x)) for t, xi, yi, _c in placed],
                    f"{where} group {g}", {"series": g}, corner))
        super().__init__(path, images)

    def _member_files(self) -> set:
        return {tile._path for image in self.images
                for tile in ([image] if isinstance(image, _Tile) else [t for t, _y, _x in image.tiles])}

    def _side_files(self) -> set:
        """The overview map the index names: its ``.oir`` and the ``.hdf5`` beside it."""
        if not self._map:
            return set()
        out = set()
        hit = _parts.find(self._folder, self._map)
        if hit is not None:
            out.add(hit)
            pixels = _parts.find(self._folder, os.path.splitext(self._map)[0] + ".hdf5")
            if pixels is not None:
                out.add(pixels)
        return out

    def close(self) -> None:
        super().close()
        self._members.close()


def open(path: str, tiles: str = "separate", resolutions: str = "base") -> _TilingFile:  # noqa: A001 - backend protocol
    """A FluoView tiling set from its ``.omp2info`` (*resolutions* changes
    nothing: OIR tiles hold no pyramids)."""
    return _TilingFile(path, tiles)
