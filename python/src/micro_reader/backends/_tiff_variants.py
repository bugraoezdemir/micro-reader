"""TIFF variants of whole-slide scanners and Micro-Manager: which of
tifffile's series are images, what they are called, and how the tiles of a
Ventana BIF scan are placed.

Series kept follow Bio-Formats per format, so that eubi-bridge gives the
same scenes whichever reader it uses (Bugra, 2026-10-15):

- Aperio SVS: the thumbnail is left out (a small copy of the pyramid);
  label and macro stay, after the slide.
- Ventana BIF: the overview ("macro") and IFD 1 stay -- a DP 200's tissue
  probability image, an iScan's thumbnail (described ``Thumbnail``).
- PerkinElmer / Akoya QPTIFF: thumbnail, macro and label all stay.
- Leica SCN, Philips: every series stays (Bio-Formats reads Philips as a
  plain TIFF; its label / macro are kept here as for the other slides).

Associated images are named in lower case: ``thumbnail``, ``label``,
``macro``, ``probability``.  A Leica SCN names its images ``image_0000001819``;
the one that spans the whole slide is called ``macro`` (``_scn_macros``).

Ventana BIF (Roche, "BIF image file format for digital pathology", v1.0,
2020): level 0 of a VENTANA DP 200 scan is a grid of tiles that overlap
horizontally.  Its tiles are numbered along a serpentine path from the
lower left (the slide's physical coordinates); for each pair of
neighbours, ``TileJointInfo`` gives the overlap, and the right tile
(``Tile2``) replaces the left one's pixels in it -- no blending.  So each
tile column ``c`` of a tile row shows the tile's columns from its start up
to where the next tile starts.  Lower levels do not overlap; they are cut
to the level-0 width halved per level (as Bio-Formats does).  Unscanned
tiles (no bytes) are filled with ``ScanWhitePoint``.  Files the paper does
not cover (other scanners, EncodeInfo before version 2, vertical overlap,
several overlapping areas of interest) raise ``UnsupportedFormatError``:
the paper says they cannot be reconstructed from it, and a wrong placement
must not pass silently.  Bio-Formats does not follow the joints: it spreads
the total overlap evenly over the columns, so its tiles sit up to ~half an
overlap away from where the joints place them.

iScan BIF (iScan Coreo / HT, before the DP 200; EncodeInfo with no
``ScannerModel``): the paper does not cover them, so they are placed as
OpenSlide 4.0.1 shows them -- worked out from its images and the files'
XMP, checked on OS-1.bif and OS-2.bif (``iscan_layouts``).  Tiles overlap
both ways (~110 px), and the joints, with confidences below 100 and pairs
that did not join, do not fit one placement together.  So each area of
interest is a regular grid: a tile is ``tile size - mean overlap`` from
its neighbours, the means weighted by the joints' ``Confidence`` over the
whole scan; the areas sit on one canvas by their ``Pos-X`` / ``Pos-Y``;
a later tile (right, lower) replaces an earlier one's pixels (fusion
"overwrite", tile order).  OpenSlide draws each tile at its fractional
position and interpolates; micro-reader starts it at the nearest whole
pixel and keeps the stored pixels, so the two differ by that sub-pixel
shift (OpenSlide's image is micro-reader's interpolated by it).  Outside
the areas: ``ScanWhitePoint`` if the scan names one (iScan scans do not),
else 0 -- what OpenSlide's transparent pixels hold.  Refused, as not
checked: EncodeInfo other than version 2, joints other than RIGHT / UP,
several focal planes, areas overlapping on the canvas, areas not starting
on a tile, no confident joint for an overlap that is needed.
"""
from __future__ import annotations

import math
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Optional

from .._errors import UnsupportedFormatError

#: tifffile's associated-image series names -> micro-reader's (lower case)
ASSOCIATED = {"Thumbnail": "thumbnail", "Label": "label", "Macro": "macro"}
#: formats whose thumbnail Bio-Formats leaves out
DROP_THUMBNAIL = ("svs",)
#: the scanner the BIF paper covers
BIF_SCANNER = "VENTANA DP 200"
BIF_DIRECTIONS = ("LEFT", "RIGHT", "UP", "DOWN")


#: iScan BIF (Coreo, HT): the EncodeInfo version and joint directions both samples have
ISCAN_ENCODE_VERSION = "2"
ISCAN_DIRECTIONS = ("RIGHT", "UP")


@dataclass(frozen=True)
class Area:
    """One area of interest of a composed level (iScan BIF): its tile cells
    along y (*rows*) and along x (*cols*), each ``(composed start, composed
    end, stored start)`` -- the composed pixels ``start:end`` show the stored
    ones from *stored start* on.  Cells do not overlap: each runs to where
    the next starts (the later tile wins its overlap)."""
    rows: tuple
    cols: tuple


@dataclass(frozen=True)
class TileLayout:
    """How one level is composed from the stored image (BIF): *width* of the
    composed level; *band* rows at a time, each band made of *pieces*
    ``(composed x, stored x)`` -- a piece runs to the next one's composed x
    (the last to *width*) and shows the stored columns from its stored x on;
    no pieces: the stored image, cut to *width*.  *fill*: the value of
    unscanned tiles.  iScan BIF: *height* of the composed level and its
    *areas* (``Area``) instead of pieces; pixels outside them are *fill*
    (the TIFF's no-data value when None)."""
    width: int
    band: int = 0
    pieces: tuple = ()
    fill: Optional[int] = None
    height: Optional[int] = None
    areas: tuple = ()


def kind(tif) -> str:
    """The variant whose series rules apply ('' for none)."""
    for flag in ("svs", "scn", "bif", "qpi", "philips"):
        if getattr(tif, "is_" + flag, False):
            return flag
    return ""


def select(tif, series: list) -> list:
    """*series* as micro-reader lists them (module docstring); renames the
    associated images and attaches the BIF tile layout (``series.layout``)."""
    variant = kind(tif)
    if not variant:
        return series
    kept = []
    scn_macros = _scn_macros(tif) if variant == "scn" else {}
    for s in series:
        name = s.name or ""
        if variant == "scn":
            s.scn_name = name                         # the SCN XML's image name (metadata)
            if name in scn_macros:
                s.name = scn_macros[name]
        elif variant == "bif":
            # tifffile's names: up to 2025.12 Label / Thumbnail, from 2026.10.15
            # Overview / Probability.  An iScan's IFD 1 is a thumbnail of the
            # slide, described "Thumbnail" (a DP 200's: "Probability_Image")
            if _description(s) == "Thumbnail":
                s.name = "thumbnail"
            elif name in ("Label", "Overview"):         # IFD 0: the whole slide, label included
                s.name = "macro"
            elif name in ("Thumbnail", "Probability"):  # IFD 1: tissue probability
                s.name = "probability"
        elif name in ASSOCIATED:
            if name == "Thumbnail" and variant in DROP_THUMBNAIL:
                continue
            s.name = ASSOCIATED[name]
        kept.append(s)
    if variant == "bif":
        _attach_bif_layout(kept)
    return kept


# -- Leica SCN -------------------------------------------------------------------

def _scn_macros(tif) -> dict:
    """SCN image name -> micro-reader's name for the images whose ``view`` spans
    the whole ``collection`` (the slide): the first brightfield one is ``macro``
    (OpenSlide's rule), others ``macro <illumination>`` (Leica-Fluorescence-1
    has a fluorescence one too).  Decided 2026-10-18: the SCN names
    (``image_0000001819``) say nothing."""
    try:
        root = ET.fromstring(tif.pages.first.description)
    except (ET.ParseError, TypeError):
        return {}
    local = lambda e: e.tag.rsplit("}", 1)[-1]   # noqa: E731
    out = {}
    for collection in (e for e in root.iter() if local(e) == "collection"):
        size = (collection.get("sizeX"), collection.get("sizeY"))
        for image in (e for e in collection if local(e) == "image"):
            view = next((e for e in image if local(e) == "view"), None)
            if view is None or (view.get("sizeX"), view.get("sizeY")) != size:
                continue
            light = next((e.text for e in image.iter() if local(e) == "illuminationSource"), None)
            light = (light or "").strip().lower()
            name = "macro" if light in ("", "brightfield") and "macro" not in out.values() else \
                f"macro {light or 'other'}"
            out[image.get("name")] = name
    return out


# -- Ventana BIF -----------------------------------------------------------------

def _xmp(page):
    """The XMP tag of *page* as an XML element (trailing NULs stripped), or None."""
    tag = page.tags.get(700)
    if tag is None:
        return None
    value = tag.value
    text = value if isinstance(value, str) else bytes(value).decode("utf-8", "replace")
    try:
        return ET.fromstring(text.rstrip("\x00").strip())
    except ET.ParseError:
        return None


def _description(series) -> str:
    """The ImageDescription of *series*' first page ('' if none)."""
    try:
        return (series.keyframe.description or "").strip()
    except (AttributeError, IndexError):
        return ""


def _attach_bif_layout(series: list) -> None:
    base = next((s for s in series if s.name == "Baseline"), None)
    if base is None:
        return
    key = base.keyframe
    where = f"{key.parent.filehandle.name!r} (Ventana BIF)"
    root = _xmp(key)
    if root is None or root.tag != "EncodeInfo":
        raise UnsupportedFormatError(f"{where}: no EncodeInfo in the scan's XMP: tile placement unknown")
    iscan = root.find("./SlideInfo/iScan")
    if iscan is not None and iscan.get("ScannerModel") is None:
        # an iScan (Coreo / HT) scan: its EncodeInfo names no scanner model
        levels = [(level.keyframe.description or "") for level in base.levels]
        for level, layout in zip(base.levels, iscan_layouts(root, key, levels, where)):
            level.layout = layout
        return
    shifts, width, fill = bif_shifts(root, key, where)
    if shifts is None:                       # no overlap: the stored grid as it is
        if fill is not None:
            for level in base.levels:
                level.layout = TileLayout(width=level.keyframe.imagewidth, fill=fill)
        return
    for k, level in enumerate(base.levels):
        level.layout = level_layout(shifts, width, k, key.tilewidth, key.tilelength, fill)


def level_layout(shifts: tuple, width: int, k: int, tw: int, th: int, fill) -> TileLayout:
    """The ``TileLayout`` of pyramid level *k* (factor 2**k) from level 0's
    *shifts* (per tile row, the pixels each tile column moves left).  Lower
    levels are the stored tile grid downsampled -- overlaps included
    (checked on Ventana-1.bif: features of level 1 sit where the unstitched
    grid puts them) -- so each level drops the same strips, scaled; at
    levels where a strip is not a whole number of pixels its ends are
    rounded."""
    f = 1 << k
    band = max(1, th // f)
    rows = []
    for b in range(len(shifts) * th // band):
        r = min(len(shifts) - 1, (b * band * f) // th)
        pieces = []
        for c, shift in enumerate(shifts[r]):
            composed = int(round((c * tw - shift) / f))
            stored = int(round(c * tw / f))
            if pieces and composed <= pieces[-1][0]:
                pieces.pop()                  # a piece narrowed to nothing at this level
            pieces.append((composed, stored))
        rows.append(tuple(pieces))
    return TileLayout(width=width >> k, band=band, pieces=tuple(rows), fill=fill)


def bif_shifts(root, key, where: str) -> tuple:
    """(shifts, composed width, fill) of a BIF scan from its ``EncodeInfo``
    XML (*key*: the scan's TIFF page): *shifts* per tile row, the pixels each
    tile column moves left (None: no overlap anywhere)."""
    if int(root.get("Ver", "0")) < 2:
        raise UnsupportedFormatError(
            f"{where}: EncodeInfo version {root.get('Ver')!r}: tile placement is defined "
            "from version 2 (Roche BIF paper)")
    iscan = root.find("./SlideInfo/iScan")
    model = iscan.get("ScannerModel", "") if iscan is not None else ""
    if model != BIF_SCANNER:
        raise UnsupportedFormatError(
            f"{where}: scanner {model!r}: the BIF paper defines tile placement for "
            f"{BIF_SCANNER} files only")
    fill = None
    if iscan is not None and iscan.get("ScanWhitePoint"):
        fill = int(float(iscan.get("ScanWhitePoint")))
    tw, th = key.tilewidth, key.tilelength
    across = -(-key.imagewidth // tw)
    down = -(-key.imagelength // th)
    origins = {}
    origin_node = root.find("./AoiOrigin")
    for aoi in (origin_node if origin_node is not None else []):
        if aoi.tag.startswith("AOI"):
            origins[int(aoi.tag[3:])] = (int(aoi.get("OriginX", 0)) // tw,
                                         int(aoi.get("OriginY", 0)) // th)
    infos = [e for e in root.iter("ImageInfo") if e.get("AOIScanned", "1") == "1"]
    shifts = [[0] * across for _ in range(down)]   # pixels a tile moves left
    any_overlap = False
    for info in infos:
        rows, cols = int(info.get("NumRows")), int(info.get("NumCols"))
        ox, oy = origins.get(int(info.get("AOIIndex", 0)), (0, 0))
        if (int(info.get("Width", tw)), int(info.get("Height", th))) != (tw, th):
            raise UnsupportedFormatError(f"{where}: AOI tiles of another size than the TIFF tiles")
        if ox + cols > across or oy + rows > down:
            raise UnsupportedFormatError(f"{where}: an AOI outside the tile grid")

        def place(n, rows=rows, cols=cols):
            """serpentine tile number (1 = lower left) -> (row, col) in the AOI"""
            from_bottom, k = divmod(n - 1, cols)
            return rows - 1 - from_bottom, (k if from_bottom % 2 == 0 else cols - 1 - k)

        overlaps = [[0] * cols for _ in range(rows)]   # [r][c]: with the tile left of c
        for joint in info.iter("TileJointInfo"):
            direction = joint.get("Direction")
            if (joint.get("FlagJoined") != "1" or joint.get("Confidence") != "100"
                    or direction not in BIF_DIRECTIONS):
                raise UnsupportedFormatError(
                    f"{where}: a tile joint the BIF paper does not define ({dict(joint.attrib)})")
            (r1, c1), (r2, c2) = place(int(joint.get("Tile1"))), place(int(joint.get("Tile2")))
            if direction in ("UP", "DOWN"):
                if int(joint.get("OverlapY", 0)):
                    raise UnsupportedFormatError(f"{where}: vertical tile overlap")
                continue
            if r1 != r2 or abs(c1 - c2) != 1 or int(joint.get("OverlapY", 0)):
                raise UnsupportedFormatError(f"{where}: a horizontal joint of tiles that are not neighbours")
            if c2 < c1:
                # Tile2 is the one drawn on top: on the left it would cover the
                # left tile's right edge -- not in any file the paper describes
                if int(joint.get("OverlapX", 0)):
                    raise UnsupportedFormatError(f"{where}: an overlap won by the left tile")
                continue
            overlaps[r1][c2] = int(joint.get("OverlapX", 0))
        for r in range(rows):
            total = 0
            for c in range(cols):
                total += overlaps[r][c]
                shifts[oy + r][ox + c] = total
            any_overlap |= bool(total)
    if any_overlap and len(infos) > 1:
        raise UnsupportedFormatError(f"{where}: several areas of interest with overlapping tiles")
    if not any_overlap:
        return None, key.imagewidth, fill
    # the composed width: where the AOI's last tile ends (every row ends there
    # in the files the paper describes); columns past it are padding
    info = infos[0]
    ox, oy = origins.get(int(info.get("AOIIndex", 0)), (0, 0))
    last = ox + int(info.get("NumCols")) - 1
    ends = {(last * tw - shifts[r][last]) + tw for r in range(oy, oy + int(info.get("NumRows")))}
    if len(ends) != 1:
        raise UnsupportedFormatError(f"{where}: tile rows of different widths after the overlaps")
    return tuple(tuple(row[:last + 1]) for row in shifts), ends.pop(), fill


# -- iScan BIF (Coreo, HT) -------------------------------------------------------

def iscan_layouts(root, key, levels: list, where: str) -> list:
    """The ``TileLayout`` of each pyramid level of an iScan BIF scan from its
    ``EncodeInfo`` XML (*key*: the scan's level-0 TIFF page; *levels*: each
    level's ImageDescription, ``level=k ...``).

    Placed as OpenSlide (4.0.1) shows these files -- worked out from its
    images, not its code: every area of interest is a regular grid of its
    tiles, each tile ``tile size - mean overlap`` from its neighbours, the
    mean overlaps taken over all the scan's joints weighted by their
    ``Confidence`` (``OverlapX`` of RIGHT joints, ``OverlapY`` of UP ones;
    the per-joint overlaps are not used one by one).  An area's bottom left
    corner is at its ``Pos-X`` / ``Pos-Y`` (+y up), its size rounded up to
    whole pixels; its tiles are the stored grid's from ``AoiOrigin`` on.  A
    later tile (right, lower) replaces an earlier one's pixels.  OpenSlide
    draws each tile at its fractional position, interpolated; here each
    starts at the nearest whole pixel, its stored pixels unchanged.  Lower
    levels: the same cells of the stored level (``tile size / 2**k``) at
    the positions divided by ``2**k``, the level ``ceil(size / 2**k)``."""
    if root.get("Ver") != ISCAN_ENCODE_VERSION:
        raise UnsupportedFormatError(
            f"{where}: iScan EncodeInfo version {root.get('Ver')!r}: tile placement is "
            f"checked on version {ISCAN_ENCODE_VERSION} only")
    iscan = root.find("./SlideInfo/iScan")
    if iscan.get("Z-layers", "1") not in ("1", "1.0"):
        raise UnsupportedFormatError(f"{where}: iScan scan of {iscan.get('Z-layers')} focal planes: "
                                     "checked on single-plane scans only")
    fill = int(float(iscan.get("ScanWhitePoint"))) if iscan.get("ScanWhitePoint") else None
    tw, th = key.tilewidth, key.tilelength
    across, down = -(-key.imagewidth // tw), -(-key.imagelength // th)
    origins = {}
    origin_node = root.find("./AoiOrigin")
    for aoi in (origin_node if origin_node is not None else []):
        if aoi.tag.startswith("AOI") and aoi.tag[3:].isdigit():
            ox, oy = int(aoi.get("OriginX", 0)), int(aoi.get("OriginY", 0))
            if ox % tw or oy % th:
                raise UnsupportedFormatError(
                    f"{where}: an area of interest starts inside a tile ({aoi.tag} at {ox}, {oy})")
            origins[int(aoi.tag[3:])] = (ox // tw, oy // th)
    # mean overlaps, weighted by the joints' confidence, over every area
    weight = {"RIGHT": 0.0, "UP": 0.0}
    overlap = {"RIGHT": 0.0, "UP": 0.0}
    areas = []
    for n, info in enumerate(root.iter("ImageInfo")):
        for joint in info.iter("TileJointInfo"):
            direction = joint.get("Direction")
            if direction not in ISCAN_DIRECTIONS:
                raise UnsupportedFormatError(
                    f"{where}: an iScan tile joint of direction {direction!r}: tile placement "
                    f"is checked on {' / '.join(ISCAN_DIRECTIONS)} joints only")
            try:
                w = float(joint.get("Confidence", 0))
                o = float(joint.get("OverlapX" if direction == "RIGHT" else "OverlapY", 0))
            except ValueError:
                raise UnsupportedFormatError(f"{where}: a tile joint that does not parse "
                                             f"({dict(joint.attrib)})") from None
            weight[direction] += w
            overlap[direction] += w * o
        if info.get("AOIScanned", "1") == "1":
            areas.append((int(info.get("AOIIndex", n)), info))
    if not areas:
        raise UnsupportedFormatError(f"{where}: no scanned area of interest")
    placed = []
    for index, info in areas:
        rows, cols = int(info.get("NumRows", 0)), int(info.get("NumCols", 0))
        if rows < 1 or cols < 1:
            raise UnsupportedFormatError(f"{where}: an area of interest of {rows} x {cols} tiles")
        if (int(info.get("Width", tw)), int(info.get("Height", th))) != (tw, th):
            raise UnsupportedFormatError(f"{where}: AOI tiles of another size than the TIFF tiles")
        if index not in origins:
            raise UnsupportedFormatError(f"{where}: no AoiOrigin for area of interest {index}")
        oc, orow = origins[index]
        if oc + cols > across or orow + rows > down:
            raise UnsupportedFormatError(f"{where}: an AOI outside the tile grid")
        if info.get("Pos-X") is None or info.get("Pos-Y") is None:
            raise UnsupportedFormatError(f"{where}: an area of interest without Pos-X / Pos-Y")
        placed.append((int(info.get("Pos-X")), int(info.get("Pos-Y")), rows, cols, oc, orow))
    mean = {}
    for direction, size, many in (("RIGHT", tw, any(p[3] > 1 for p in placed)),
                                  ("UP", th, any(p[2] > 1 for p in placed))):
        if not weight[direction]:
            if many:
                raise UnsupportedFormatError(
                    f"{where}: no {direction} tile joint with a confidence: the tile overlap is unknown")
            mean[direction] = 0.0
            continue
        mean[direction] = overlap[direction] / weight[direction]
        if not 0 <= mean[direction] < size:
            raise UnsupportedFormatError(
                f"{where}: mean {direction} tile overlap {mean[direction]:.3f} px: not within a tile")
    mx, my = mean["RIGHT"], mean["UP"]
    # the canvas: +y up in the file, top-down here; areas sized in whole pixels
    sized = [(px, py, math.ceil(cols * tw - (cols - 1) * mx), math.ceil(rows * th - (rows - 1) * my))
             for px, py, rows, cols, _, _ in placed]
    left = min(s[0] for s in sized)
    top = max(s[1] + s[3] for s in sized)
    width = max(s[0] + s[2] for s in sized) - left
    height = top - min(s[1] for s in sized)
    boxes = [(px - left, top - py - h, w, h) for px, py, w, h in sized]
    for a in range(len(boxes)):
        for b in range(a):
            (xa, ya, wa, ha), (xb, yb, wb, hb) = boxes[a], boxes[b]
            if xa < xb + wb and xb < xa + wa and ya < yb + hb and yb < ya + ha:
                raise UnsupportedFormatError(
                    f"{where}: areas of interest that overlap: which one shows is unknown")
    layouts = []
    for k, description in enumerate(levels):
        stated = re.match(r"level=(\d+)", description)
        if stated and int(stated.group(1)) != k:
            raise UnsupportedFormatError(f"{where}: pyramid level {k} described as {description!r}")
        f = 1 << k
        level_areas = tuple(Area(rows=_cells(y, rows, th, my, orow, f),
                                 cols=_cells(x, cols, tw, mx, oc, f))
                            for (x, y, _, _), (_, _, rows, cols, oc, orow) in zip(boxes, placed))
        layouts.append(TileLayout(width=-(-width // f), height=-(-height // f), fill=fill,
                                  areas=level_areas))
    return layouts


def _nearest(v: float) -> int:
    """*v* rounded to the nearest integer, halves up."""
    return math.floor(v + 0.5)


def _cells(offset: int, n: int, size: int, overlap: float, first: int, f: int) -> tuple:
    """Along one axis of an area of *n* tiles of *size* whose first tile is
    at *offset* of the canvas (level 0) and is stored tile *first*: the
    ``(composed start, composed end, stored start)`` of each tile's cell at
    the level of factor *f* -- a cell runs to where the next starts, the
    last one a stored cell further; cells narrowed to nothing are left out."""
    starts = [_nearest((offset + i * (size - overlap)) / f) for i in range(n)]
    stored = [_nearest((first + i) * size / f) for i in range(n + 1)]
    out = []
    for i in range(n):
        end = starts[i + 1] if i + 1 < n else starts[i] + stored[i + 1] - stored[i]
        if end > starts[i]:
            out.append((starts[i], end, stored[i]))
    return tuple(out)
