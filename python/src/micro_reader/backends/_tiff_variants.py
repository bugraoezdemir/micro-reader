"""TIFF variants of whole-slide scanners and Micro-Manager: which of
tifffile's series are images, what they are called, and how the tiles of a
Ventana BIF scan are placed.

Series kept follow Bio-Formats per format, so that eubi-bridge gives the
same scenes whichever reader it uses (Bugra, 2026-10-15):

- Aperio SVS: the thumbnail is left out (a small copy of the pyramid);
  label and macro stay, after the slide.
- Ventana BIF: the overview ("macro") and the tissue probability image stay.
- PerkinElmer / Akoya QPTIFF: thumbnail, macro and label all stay.
- Leica SCN, Philips: every series stays (Bio-Formats reads Philips as a
  plain TIFF; its label / macro are kept here as for the other slides).

Associated images are named in lower case: ``thumbnail``, ``label``,
``macro``, ``probability``.

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
the paper says they cannot be reconstructed from it, and a wrong stitch
must not pass silently.  Bio-Formats does not follow the joints: it spreads
the total overlap evenly over the columns, so its tiles sit up to ~half an
overlap away from where the joints place them.
"""
from __future__ import annotations

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


@dataclass(frozen=True)
class TileLayout:
    """How one level is composed from the stored image (BIF): *width* of the
    composed level; *band* rows at a time, each band made of *pieces*
    ``(composed x, stored x)`` -- a piece runs to the next one's composed x
    (the last to *width*) and shows the stored columns from its stored x on;
    no pieces: the stored image, cut to *width*.  *fill*: the value of
    unscanned tiles."""
    width: int
    band: int = 0
    pieces: tuple = ()
    fill: Optional[int] = None


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
    for s in series:
        name = s.name or ""
        if variant == "bif":
            if name == "Label":                 # IFD 0: the whole slide, label included
                s.name = "macro"
            elif name == "Thumbnail":           # IFD 1: tissue probability (tifffile: Thumbnail)
                s.name = "probability"
        elif name in ASSOCIATED:
            if name == "Thumbnail" and variant in DROP_THUMBNAIL:
                continue
            s.name = ASSOCIATED[name]
        kept.append(s)
    if variant == "bif":
        _attach_bif_layout(kept)
    return kept


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


def _attach_bif_layout(series: list) -> None:
    base = next((s for s in series if s.name == "Baseline"), None)
    if base is None:
        return
    key = base.keyframe
    where = f"{key.parent.filehandle.name!r} (Ventana BIF)"
    root = _xmp(key)
    if root is None or root.tag != "EncodeInfo":
        raise UnsupportedFormatError(f"{where}: no EncodeInfo in the scan's XMP: tile placement unknown")
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
