"""TIFF variants on synthetic files: whole-slide formats (Aperio SVS, Leica
SCN, Ventana BIF, PerkinElmer QPTIFF, Philips) and Micro-Manager NDTiff --
the series kept (Bio-Formats' choice per format), their names, metadata,
BIF tile placement (DP 200 by its joints, iScan as OpenSlide places it), series
spread over several files."""
from __future__ import annotations

import re

import numpy as np
import pytest

tifffile = pytest.importorskip("tifffile")

import micro_reader  # noqa: E402
from micro_reader import UnsupportedFormatError  # noqa: E402
from micro_reader._metadata import Quantity  # noqa: E402

from _ndtiff_writer import write_ndtiff  # noqa: E402

RNG = np.random.default_rng(7)


def _rgb(shape):
    return RNG.integers(0, 256, size=(*shape, 3), dtype=np.uint8)


# -- Aperio SVS -------------------------------------------------------------------

SVS_DESCRIPTION = ("Aperio Image Library v12.0.0\r\n96x80 [0,0 96x80] (32x32) RAW|AppMag = 20"
                   "|MPP = 0.4990|Left = 25.5|Top = 23.25")


def _svs(path):
    base, thumb, label, macro = _rgb((80, 96)), _rgb((20, 24)), _rgb((30, 20)), _rgb((25, 60))
    with tifffile.TiffWriter(path) as tw:
        tw.write(base, metadata=None, tile=(32, 32), photometric="rgb", description=SVS_DESCRIPTION)
        tw.write(thumb, metadata=None, photometric="rgb", description="Aperio Image Library v12.0.0\r\nthumb")
        tw.write(label, metadata=None, photometric="rgb", subfiletype=1,
                 description="Aperio Image Library v12.0.0\r\nlabel 20x30")
        tw.write(macro, metadata=None, photometric="rgb", subfiletype=9,
                 description="Aperio Image Library v12.0.0\r\nmacro 60x25")
    return base, label, macro


def test_svs_drops_thumbnail_names_associated(tmp_path):
    path = str(tmp_path / "slide.svs")
    base, label, macro = _svs(path)
    with micro_reader.open(path) as f:
        assert f.format == "svs"
        assert [im.name for im in f.images] == ["Baseline", "label", "macro"]
        for im, ref in zip(f.images, (base, label, macro)):
            np.testing.assert_array_equal(im[...], ref)


def test_svs_metadata(tmp_path):
    path = str(tmp_path / "slide.svs")
    _svs(path)
    with micro_reader.open(path) as f:
        meta = f.images[0].metadata
        assert meta.pixel_sizes == {"x": Quantity(0.499, "micrometer"),
                                    "y": Quantity(0.499, "micrometer")}
        # Left / Top are the image's corner: the first pixel's centre is half a pixel in
        assert meta.position["x"].unit == "millimeter"
        assert meta.position["x"].value == pytest.approx(25.5 + 0.499e-3 / 2)
        assert meta.position["y"].value == pytest.approx(23.25 + 0.499e-3 / 2)
        assert f.images[1].metadata is None or not f.images[1].metadata.position


# -- PerkinElmer / Akoya QPTIFF ------------------------------------------------------

def _qpi_description(image_type, name=None, color=None):
    parts = [f"<ImageType>{image_type}</ImageType>"]
    if name:
        parts.append(f"<Name>{name}</Name>")
    if color:
        parts.append(f"<Color>{color}</Color>")
    return ('<?xml version="1.0" encoding="utf-16"?>\r\n<PerkinElmer-QPI-ImageDescription>'
            + "".join(parts) + "</PerkinElmer-QPI-ImageDescription>")


def _qptiff(path):
    channels = RNG.integers(0, 255, size=(3, 40, 56), dtype=np.uint8)
    thumb = _rgb((10, 14))
    # 20000 px/cm = 0.5 um; X/YPosition 0.5 cm / 4.25 cm (in ResolutionUnit, cm)
    position = [(286, 5, 1, (1, 2), True), (287, 5, 1, (17, 4), True)]
    with tifffile.TiffWriter(path) as tw:
        for k, (name, color) in enumerate((("DAPI", "0,0,255"), ("CD8 (Opal 540)", "255,255,0"),
                                           ("Autofluorescence", "0,0,0"))):
            tw.write(channels[k], metadata=None, software="PerkinElmer-QPI", resolution=(20000, 20000),
                     resolutionunit="CENTIMETER", extratags=position,
                     description=_qpi_description("FullResolution", name, color))
        tw.write(thumb, metadata=None, photometric="rgb", software="PerkinElmer-QPI",
                 description=_qpi_description("Thumbnail"))
    return channels, thumb


def test_qptiff_channels_and_thumbnail_kept(tmp_path):
    path = str(tmp_path / "scan.qptiff")
    channels, thumb = _qptiff(path)
    with micro_reader.open(path) as f:
        assert f.format == "qptiff"
        assert [im.name for im in f.images] == ["Baseline", "thumbnail"]   # Bio-Formats keeps it
        np.testing.assert_array_equal(f.images[0][...], channels)
        np.testing.assert_array_equal(f.images[1][...], thumb)
        meta = f.images[0].metadata
        assert [c.name for c in meta.channels] == ["DAPI", "CD8 (Opal 540)", "Autofluorescence"]
        assert [c.color for c in meta.channels] == ["#0000FF", "#FFFF00", "#000000"]
        assert meta.pixel_sizes["x"].to("micrometer").value == pytest.approx(0.5)
        assert meta.position["x"].to("micrometer").value == pytest.approx(5000 + 0.25)
        assert meta.position["y"].to("micrometer").value == pytest.approx(42500 + 0.25)


# -- Leica SCN -------------------------------------------------------------------------

SCN_XML = """<?xml version="1.0"?>
<scn xmlns="http://www.leica-microsystems.com/scn/2010/10/01">
  <collection name="slide" sizeX="1000000" sizeY="3000000">
    <image name="overview">
      <pixels sizeX="40" sizeY="120">
        <dimension sizeX="40" sizeY="120" r="0" ifd="0" />
      </pixels>
      <view sizeX="1000000" sizeY="3000000" offsetX="0" offsetY="0" spacingZ="0" />
    </image>
    <image name="tissue">
      <pixels sizeX="64" sizeY="48">
        <dimension sizeX="64" sizeY="48" r="0" c="0" ifd="1" />
        <dimension sizeX="64" sizeY="48" r="0" c="1" ifd="2" />
        <dimension sizeX="16" sizeY="12" r="1" c="0" ifd="3" />
        <dimension sizeX="16" sizeY="12" r="1" c="1" ifd="4" />
      </pixels>
      <view sizeX="32000" sizeY="24000" offsetX="200000" offsetY="500000" spacingZ="400" />
      <scanSettings><channelSettings>
        <channel index="0" name="405|Empty" rgb="#0000ff" />
        <channel index="1" name="L5|Empty" rgb="#00ff00" />
      </channelSettings></scanSettings>
    </image>
  </collection>
</scn>"""


def test_scn_series_metadata(tmp_path):
    path = str(tmp_path / "slide.scn")
    overview = _rgb((120, 40))
    tissue = RNG.integers(0, 255, size=(2, 48, 64), dtype=np.uint8)
    small = RNG.integers(0, 255, size=(2, 12, 16), dtype=np.uint8)
    with tifffile.TiffWriter(path) as tw:
        tw.write(overview, metadata=None, photometric="rgb", tile=(16, 16), description=SCN_XML)
        for plane in tissue:
            tw.write(plane, metadata=None, tile=(16, 16))
        for plane in small:
            tw.write(plane, metadata=None, tile=(16, 16))
    with micro_reader.open(path) as f:
        assert f.format == "scn"
        # the image whose view spans the whole slide is the macro (its SCN name says nothing)
        assert [im.name for im in f.images] == ["macro", "tissue"]
        assert f.images[0].metadata.pixel_sizes["x"] == Quantity(25000.0, "nanometer")
        np.testing.assert_array_equal(f.images[1][...], tissue)
        np.testing.assert_array_equal(f.images[1].resolutions[1][...], small)
        meta = f.images[1].metadata
        assert meta.pixel_sizes == {"x": Quantity(500.0, "nanometer"),
                                    "y": Quantity(500.0, "nanometer")}
        assert meta.position == {"x": Quantity(200250.0, "nanometer"),
                                 "y": Quantity(500250.0, "nanometer")}
        assert [(c.name, c.color) for c in meta.channels] == [("405|Empty", "#0000FF"),
                                                              ("L5|Empty", "#00FF00")]


# -- Philips ---------------------------------------------------------------------------

PHILIPS_XML = """<DataObject ObjectType="DPUfsImport">
<Attribute Name="PIM_DP_SCANNED_IMAGES" PMSVR="IDataObjectArray"><Array>
<DataObject ObjectType="DPScannedImage">
<Attribute Name="PIM_DP_IMAGE_TYPE" PMSVR="IString">WSI</Attribute>
<Attribute Name="PIIM_PIXEL_DATA_REPRESENTATION_SEQUENCE" PMSVR="IDataObjectArray"><Array>
<DataObject ObjectType="PixelDataRepresentation">
<Attribute Name="DICOM_PIXEL_SPACING" PMSVR="IDoubleArray">"0.00025" "0.00025"</Attribute>
</DataObject>
<DataObject ObjectType="PixelDataRepresentation">
<Attribute Name="DICOM_PIXEL_SPACING" PMSVR="IDoubleArray">"0.0005" "0.0005"</Attribute>
</DataObject>
</Array></Attribute>
</DataObject>
</Array></Attribute>
</DataObject>"""


def test_philips_levels_macro_metadata(tmp_path):
    path = str(tmp_path / "slide.tiff")
    base, level1, macro = _rgb((64, 96)), _rgb((32, 48)), _rgb((20, 40))
    with tifffile.TiffWriter(path) as tw:
        tw.write(base, metadata=None, photometric="rgb", tile=(32, 32), software="Philips DP v1.0",
                 description=PHILIPS_XML)
        tw.write(level1, metadata=None, photometric="rgb", tile=(32, 32), software="Philips DP v1.0",
                 description="level=1 mag=20 quality=80")
        tw.write(macro, metadata=None, photometric="rgb", software="Philips DP v1.0",
                 description="Macro -offset=(0,0)-pixelsize=(0.0315,0.0315)")
    with micro_reader.open(path) as f:
        assert f.format == "philips"
        assert [im.name for im in f.images] == ["Baseline", "macro"]
        np.testing.assert_array_equal(f.images[0][...], base)
        np.testing.assert_array_equal(f.images[0].resolutions[1][...], level1)
        np.testing.assert_array_equal(f.images[1][...], macro)
        sizes = f.images[0].metadata.pixel_sizes
        assert sizes["x"].to("micrometer").value == pytest.approx(0.25)
        assert sizes["y"].to("micrometer").value == pytest.approx(0.25)


# -- Ventana BIF -------------------------------------------------------------------------

TILE = 16
ROWS, COLS = 3, 4                    # the area of interest, in tiles
STORED_COLS = COLS + 1               # the TIFF grid has a column of padding
OVERLAPS = {1: 0, 2: 4, 3: 2}        # tile column -> overlap with the tile on its left


def _serpentine(row, col):
    """(row from the top, col) -> the BIF tile number (1 = lower left)."""
    from_bottom = ROWS - 1 - row
    k = col if from_bottom % 2 == 0 else COLS - 1 - col
    return from_bottom * COLS + k + 1


def _encode_info(model="VENTANA DP 200", version=2, overlap_y=0, white=235):
    joints = []
    for r in range(ROWS):
        for c in range(1, COLS):
            joints.append(f'<TileJointInfo FlagJoined="1" Confidence="100" Direction="LEFT" '
                          f'Tile1="{_serpentine(r, c - 1)}" Tile2="{_serpentine(r, c)}" '
                          f'OverlapX="{OVERLAPS[c]}" OverlapY="0"/>')
    for r in range(1, ROWS):
        for c in range(COLS):
            joints.append(f'<TileJointInfo FlagJoined="1" Confidence="100" Direction="UP" '
                          f'Tile1="{_serpentine(r, c)}" Tile2="{_serpentine(r - 1, c)}" '
                          f'OverlapX="0" OverlapY="{overlap_y}"/>')
    frames = "".join(f'<Frame XY="{c},{r}" Z="0"/>' for r in range(ROWS) for c in range(COLS))
    return (f'<?xml version="1.0" encoding="utf-8"?><EncodeInfo Ver="{version}"><SlideInfo>'
            f'<iScan Mode="brightfield" Magnification="40" ScanRes="0.25" ScannerModel="{model}" '
            f'ScanWhitePoint="{white}" Z-layers="1"/>'
            f'<AoiInfo XIMAGESIZE="{TILE}" YIMAGESIZE="{TILE}" NumRows="{ROWS}" NumCols="{COLS}"/>'
            f'</SlideInfo><SlideStitchInfo><ImageInfo AOIScanned="1" AOIIndex="0" '
            f'NumRows="{ROWS}" NumCols="{COLS}" Width="{TILE}" Height="{TILE}">'
            + "".join(joints) + f'</ImageInfo><FrameInfo AOIScanned="1" AOIIndex="0">{frames}'
            '</FrameInfo></SlideStitchInfo><AoiOrigin><AOI0 OriginX="0" OriginY="0"/>'
            '</AoiOrigin></EncodeInfo>\x00').encode()


def _bif(path, empty_tile=None, **xml):
    """A BIF of uncompressed tiles: returns (stored level-0 grid, the overview)."""
    grid = _rgb((ROWS * TILE, STORED_COLS * TILE))
    overview, probability = _rgb((60, 20)), RNG.integers(0, 2, size=(60, 20), dtype=np.uint8) * 255
    xmp_meta = b'<Metadata><iScan ScannerModel="VENTANA DP 200"/></Metadata>'

    def tiles(image):
        for r in range(0, image.shape[0], TILE):
            for c in range(0, image.shape[1], TILE):
                if empty_tile == (r // TILE, c // TILE):
                    yield None
                else:
                    yield image[r:r + TILE, c:c + TILE]

    with tifffile.TiffWriter(path, bigtiff=True) as tw:
        tw.write(overview, metadata=None, photometric="rgb", description="Label_Image",
                 software="ScanOutputManager 1.1", extratags=[(700, 1, len(xmp_meta), xmp_meta, True)])
        tw.write(probability, metadata=None, description="Probability_Image", software="ScanOutputManager 1.1",
                 extratags=[(700, 1, len(xmp_meta), xmp_meta, True)])
        info = _encode_info(**xml)
        tw.write(tiles(grid), metadata=None, shape=grid.shape, dtype=grid.dtype, tile=(TILE, TILE),
                 photometric="rgb", description="level=0 mag=40 quality=95",
                 software="ScanOutputManager 1.1", extratags=[(700, 1, len(info), info, True)])
        level1 = grid[::2, ::2]
        tw.write(level1, metadata=None, tile=(TILE, TILE), photometric="rgb",
                 description="level=1 mag=20 quality=95", software="ScanOutputManager 1.1")
    return grid, overview


def _compose(grid):
    """Level 0 as the BIF paper places it: tile column c starts OVERLAPS
    pixels further left and covers its left neighbour's last columns."""
    starts, x = [0], 0
    for c in range(1, COLS):
        x += TILE - OVERLAPS[c]
        starts.append(x)
    width = starts[-1] + TILE
    out = np.empty((grid.shape[0], width, 3), grid.dtype)
    for c in range(COLS):                      # left to right: the right tile wins
        out[:, starts[c]:starts[c] + TILE] = grid[:, c * TILE:(c + 1) * TILE]
    return out


def test_bif_tiles_placed_by_their_joints(tmp_path):
    path = str(tmp_path / "slide.bif")
    grid, overview = _bif(path)
    expected = _compose(grid)
    with micro_reader.open(path) as f:
        assert f.format == "bif"
        assert [im.name for im in f.images] == ["Baseline", "macro", "probability"]
        im = f.images[0]
        assert im.shape == (ROWS * TILE, COLS * TILE - 6, 3)
        np.testing.assert_array_equal(im[...], expected)
        for y0, y1, x0, x1 in [(0, 48, 13, 37), (5, 30, 26, 30), (17, 18, 0, 58), (40, 48, 41, 58)]:
            np.testing.assert_array_equal(im[y0:y1, x0:x1], expected[y0:y1, x0:x1])
        np.testing.assert_array_equal(f.images[1][...], overview)
        assert im.metadata.pixel_sizes["x"] == Quantity(0.25, "micrometer")


def test_bif_lower_level_drops_the_same_strips(tmp_path):
    """Lower levels are the stored grid downsampled, overlaps included: each
    drops the overlap strips scaled, and so is level 0 downsampled."""
    path = str(tmp_path / "slide.bif")
    grid, _ = _bif(path)
    with micro_reader.open(path) as f:
        level1 = f.images[0].resolutions[1]
        assert level1.shape == (ROWS * TILE // 2, (COLS * TILE - 6) // 2, 3)
        np.testing.assert_array_equal(level1[...], _compose(grid)[::2, ::2])


def test_bif_unscanned_tile_is_white_point(tmp_path):
    path = str(tmp_path / "slide.bif")
    grid, _ = _bif(path, empty_tile=(1, 0))
    with micro_reader.open(path) as f:
        block = f.images[0][TILE:2 * TILE, 0:TILE - 1]
        assert (block == 235).all()


@pytest.mark.parametrize("xml, message", [
    ({"model": "iScan HT"}, "scanner"),
    ({"version": 1}, "version"),
    ({"overlap_y": 3}, "vertical"),
])
def test_bif_outside_the_paper_refused(tmp_path, xml, message):
    path = str(tmp_path / "slide.bif")
    _bif(path, **xml)
    with pytest.raises(UnsupportedFormatError, match=message):
        micro_reader.open(path)


# -- iScan BIF (Coreo / HT): areas of interest on a canvas, as OpenSlide places them ----

IS_TW, IS_TH = 32, 48                         # tiles not square, as the iScan's 1024 x 1360
IS_GRID = (6, 9)                              # the stored tile grid (rows, cols)
#: (scanned, rows, cols, origin col, origin row, Pos-X, Pos-Y): Pos-* is the
#: area's lower left corner, +y up; area 1 was not scanned
IS_AREAS = [(1, 3, 3, 0, 3, 0, 30), (0, 2, 2, 0, 0, 20, 190), (1, 4, 5, 4, 0, 120, 5)]
#: RIGHT joints cycle through (OverlapX, Confidence); one unjoined (confidence 0)
IS_RIGHT = [(3, 100), (4, 90), (0, 0), (5, 95)]
IS_UP = [(5, 100), (6, 80), (4, 99)]          # (OverlapY, Confidence)


def _iscan_joints():
    """Per area, its TileJointInfo elements (the cross components, OverlapY of
    RIGHT and OverlapX of UP joints, are set to show they are not used)."""
    out, k = [], 0
    for scanned, rows, cols, *_ in IS_AREAS:
        joints = []
        for r in range(rows):
            for c in range(1, cols):
                ox, conf = IS_RIGHT[k % len(IS_RIGHT)] if scanned else (0, 0)
                k += 1
                joints.append(f"<TileJointInfo FlagJoined='{int(conf > 0)}' Confidence='{conf}' "
                              f"Direction='RIGHT' Tile1='{r * cols + c}' Tile2='{r * cols + c + 1}' "
                              f"OverlapX='{ox}' OverlapY='{2 if conf else 0}'/>")
        for r in range(1, rows):
            for c in range(cols):
                oy, conf = IS_UP[(r * cols + c) % len(IS_UP)] if scanned else (0, 0)
                joints.append(f"<TileJointInfo FlagJoined='{int(conf > 0)}' Confidence='{conf}' "
                              f"Direction='UP' Tile1='{(r - 1) * cols + c + 1}' Tile2='{r * cols + c + 1}' "
                              f"OverlapX='{-2 if conf else 0}' OverlapY='{oy}'/>")
        out.append(joints)
    return out


def _iscan_mean(direction):
    """The confidence-weighted mean overlap, counted from the joints written."""
    total = weight = 0.0
    for joints in _iscan_joints():
        for j in joints:
            if f"Direction='{direction}'" in j:
                attrs = dict(a.split("='") for a in j[len("<TileJointInfo "):-3].split("' "))
                w = float(attrs["Confidence"])
                total += w * float(attrs["OverlapX" if direction == "RIGHT" else "OverlapY"])
                weight += w
    return total / weight


def _iscan_info(version="2", direction="RIGHT", z_layers="1", origin_x=None, pos_x_last=None,
                confidence=True):
    areas = [list(a) for a in IS_AREAS]
    if pos_x_last is not None:
        areas[-1][5] = pos_x_last
    infos, origins = [], []
    for n, ((scanned, rows, cols, oc, orow, px, py), joints) in enumerate(zip(areas, _iscan_joints())):
        joints = [j.replace("Direction='RIGHT'", f"Direction='{direction}'") for j in joints]
        if not confidence:                     # no RIGHT joint with a confidence
            joints = [re.sub(r"Confidence='\d+'", "Confidence='0'", j) if "'RIGHT'" in j else j
                      for j in joints]
        infos.append(f'<ImageInfo AOIScanned="{scanned}" Width="{IS_TW}" Height="{IS_TH}" '
                     f'NumRows="{rows}" NumCols="{cols}" Pos-X="{px}" Pos-Y="{py}">'
                     + "".join(joints) + "</ImageInfo>")
        ox = oc * IS_TW if origin_x is None or n else origin_x
        origins.append(f'<AOI{n} OriginX="{ox}" OriginY="{orow * IS_TH}"/>')
    return (f"<?xml version='1.0'?><EncodeInfo Ver='{version}'><SlideInfo>"
            f"<iScan Magnification='40' ScanRes='0.23250' Z-layers='{z_layers}' Z-spacing='1'/>"
            "</SlideInfo><SlideStitchInfo>" + "".join(infos) + "</SlideStitchInfo><AoiOrigin>"
            + "".join(origins) + "</AoiOrigin></EncodeInfo>")


def _nearest(v):
    return int(np.floor(v + 0.5))


def _iscan_compose(stored, k):
    """Level *k* as OpenSlide places the tiles, to the nearest pixel: each
    scanned area's tile cells painted in order (a later tile wins), the rest 0."""
    import math
    f = 1 << k
    mx, my = _iscan_mean("RIGHT"), _iscan_mean("UP")
    scanned = [a for a in IS_AREAS if a[0]]
    sizes = [(math.ceil(c * IS_TW - (c - 1) * mx), math.ceil(r * IS_TH - (r - 1) * my))
             for _, r, c, *_ in scanned]
    top = max(a[6] + h for a, (_, h) in zip(scanned, sizes))
    width = max(a[5] + w for a, (w, _) in zip(scanned, sizes))
    height = top - min(a[6] for a in scanned)
    out = np.zeros((-(-height // f), -(-width // f), 3), np.uint8)
    for (_, rows, cols, oc, orow, px, py), (_, h) in zip(scanned, sizes):
        ax, ay = px, top - py - h
        for r in range(rows):
            for c in range(cols):
                y, x = _nearest((ay + r * (IS_TH - my)) / f), _nearest((ax + c * (IS_TW - mx)) / f)
                cell = stored[_nearest((orow + r) * IS_TH / f):_nearest((orow + r + 1) * IS_TH / f),
                              _nearest((oc + c) * IS_TW / f):_nearest((oc + c + 1) * IS_TW / f)]
                h_, w_ = min(cell.shape[0], out.shape[0] - y), min(cell.shape[1], out.shape[1] - x)
                out[y:y + h_, x:x + w_] = cell[:h_, :w_]
    return out


def _iscan_bif(path, **xml):
    """An iScan BIF: level 0 uncompressed, levels 1 and 2 Deflate (one stored
    tile holds several cells there); returns the stored levels and the macro."""
    grid = _rgb((IS_GRID[0] * IS_TH, IS_GRID[1] * IS_TW))
    levels = [grid, grid[::2, ::2].copy(), grid[::4, ::4].copy()]
    macro, thumb = _rgb((60, 20)), _rgb((12, 18))
    label_xmp = b'<iScan Magnification="40" ScanRes="0.232500" Z-layers="1"/>'
    info = _iscan_info(**xml).encode() + b"\x00"
    software = "C:\\Program Files\\Ventana Medical Systems Inc\\iScan\\StitchEncode.exe  v0.1"
    with tifffile.TiffWriter(path, bigtiff=True) as tw:
        tw.write(macro, metadata=None, photometric="rgb", description="Label Image",
                 extratags=[(700, 1, len(label_xmp), label_xmp, True)])
        tw.write(thumb, metadata=None, photometric="rgb", description="Thumbnail", software=software)
        for k, level in enumerate(levels):
            extratags = [(700, 1, len(info), info, True)] if k == 0 else []
            tw.write(level, metadata=None, tile=(IS_TH, IS_TW), photometric="rgb",
                     compression=None if k == 0 else "zlib", software=software,
                     description=f"level={k} mag={40 >> k} quality=90", extratags=extratags)
    return levels, macro, thumb


def test_iscan_bif_areas_placed_as_openslide_does(tmp_path):
    path = str(tmp_path / "iscan.bif")
    levels, macro, thumb = _iscan_bif(path)
    with micro_reader.open(path) as f:
        assert f.format == "bif"
        assert [im.name for im in f.images] == ["Baseline", "macro", "thumbnail"]
        im = f.images[0]
        expected = _iscan_compose(levels[0], 0)
        assert im.shape == expected.shape
        np.testing.assert_array_equal(im[...], expected)
        h, w = expected.shape[:2]
        for y0, y1, x0, x1 in [(0, h, 0, 30), (5, 40, 55, 90), (h - 7, h, w - 33, w), (17, 18, 0, w)]:
            np.testing.assert_array_equal(im[y0:y1, x0:x1], expected[y0:y1, x0:x1])
        for k in (1, 2):                       # cells of a stored tile, decoded once
            level = im.resolutions[k]
            expected = _iscan_compose(levels[k], k)
            assert level.shape == expected.shape
            np.testing.assert_array_equal(level[...], expected)
            np.testing.assert_array_equal(level[3:11, 9:30], expected[3:11, 9:30])
        np.testing.assert_array_equal(f.images[1][...], macro)
        np.testing.assert_array_equal(f.images[2][...], thumb)
        assert im.metadata.pixel_sizes["x"] == Quantity(0.2325, "micrometer")


def test_iscan_layout_from_encode_info():
    """The layout itself: weighted mean overlaps, areas placed +y up, cells
    at the nearest pixel, each running to where the next starts."""
    import math
    import types
    import xml.etree.ElementTree as ET
    from micro_reader.backends import _tiff_variants as tv
    key = types.SimpleNamespace(tilewidth=IS_TW, tilelength=IS_TH, imagewidth=IS_GRID[1] * IS_TW,
                                imagelength=IS_GRID[0] * IS_TH)
    layouts = tv.iscan_layouts(ET.fromstring(_iscan_info()), key, ["level=0", "level=1"], "test")
    mx, my = _iscan_mean("RIGHT"), _iscan_mean("UP")
    assert mx != int(mx) and my != int(my)             # fractional: rounding is exercised
    first, second = layouts[0].areas
    w0 = math.ceil(3 * IS_TW - 2 * mx)
    h0, h2 = math.ceil(3 * IS_TH - 2 * my), math.ceil(4 * IS_TH - 3 * my)
    top = max(30 + h0, 5 + h2)
    assert layouts[0].height == top - 5 and layouts[0].width == 120 + math.ceil(5 * IS_TW - 4 * mx)
    assert first.cols[0] == (0, _nearest(IS_TW - mx), 0)
    assert first.rows[0] == (top - 30 - h0, top - 30 - h0 + _nearest(IS_TH - my), 3 * IS_TH)
    assert first.cols[-1][1] - first.cols[-1][0] == IS_TW            # the last cell is whole
    assert second.cols[0][0] == 120 and second.rows[0][0] == top - 5 - h2
    assert layouts[1].width == -(-layouts[0].width // 2) and layouts[1].fill is None
    # (composed start, end, stored start): halves round up; at a lower level the
    # positions and stored cells are divided; cells narrowed to nothing are left
    # out; a cell runs to where the next starts, even past its stored cell
    assert tv._cells(0, 2, 4, 3.5, 0, 1) == ((0, 1, 0), (1, 5, 4))
    assert tv._cells(10, 3, 16, 3.4, 2, 2) == ((5, 11, 16), (11, 18, 24), (18, 26, 32))
    assert tv._cells(0, 3, 4, 3.75, 0, 2) == ((0, 2, 4),)
    assert tv._cells(0, 2, 4, -1.0, 0, 1) == ((0, 5, 0), (5, 9, 4))


@pytest.mark.parametrize("xml, message", [
    ({"version": "1"}, "version '1'"),
    ({"version": "3"}, "version '3'"),
    ({"direction": "LEFT"}, "direction 'LEFT'"),
    ({"z_layers": "3"}, "focal planes"),
    ({"origin_x": 8}, "inside a tile"),
    ({"pos_x_last": 10}, "overlap"),
    ({"confidence": False}, "no RIGHT tile joint with a confidence"),
])
def test_iscan_bif_unchecked_files_refused(tmp_path, xml, message):
    path = str(tmp_path / "iscan.bif")
    _iscan_bif(path, **xml)
    with pytest.raises(UnsupportedFormatError, match=message):
        micro_reader.open(path)


# -- Micro-Manager NDTiff ---------------------------------------------------------------

def _plane(p, t, c, z, shape=(24, 40)):
    y, x = np.mgrid[:shape[0], :shape[1]]
    return ((p * 10000 + t * 1000 + c * 100 + z * 10) + (y * shape[1] + x) % 7).astype(np.uint16)


def _ndtiff(directory, per_file=None):
    frames = [({"position": p, "time": t, "channel": name, "z": z}, _plane(p, t, c, z))
              for p in range(2) for t in range(3)
              for c, name in enumerate(("DAPI", "GFP")) for z in range(2)]
    return write_ndtiff(directory, frames, per_file=per_file)


@pytest.mark.parametrize("per_file", [None, 5])
def test_ndtiff_positions_are_scenes(tmp_path, per_file):
    paths = _ndtiff(str(tmp_path / "acq"), per_file)
    assert len(paths) == (1 if per_file is None else 5)
    with micro_reader.open(paths[0]) as f:
        assert f.format == "ndtiff"
        assert [im.index["series"] for im in f.images] == [0, 1]
        for im in f.images:
            p = im.index["series"]
            a = im.as_axes("tczyx")
            assert a.shape == (3, 2, 2, 24, 40)
            whole = a[...]
            for t in range(3):
                for c in range(2):
                    for z in range(2):
                        np.testing.assert_array_equal(whole[t, c, z], _plane(p, t, c, z))
            np.testing.assert_array_equal(a[1:3, 1:, :, 5:20, 7:33], whole[1:3, 1:, :, 5:20, 7:33])


def test_ndtiff_channel_names_and_pixel_size(tmp_path):
    """Channel names are the index's channel values (as tifffile numbers
    them: in order of first appearance); the pixel size is the frames'
    PixelSizeUm."""
    frames = [({"time": t, "channel": name, "z": 0}, _plane(0, t, c, 0))
              for t in range(2) for c, name in enumerate(("Cy5", "DAPI", "GFP"))]
    paths = write_ndtiff(str(tmp_path / "acq"), frames, frame_metadata={"PixelSizeUm": 0.65})
    with micro_reader.open(paths[0]) as f:
        im = f.images[0]
        md = im.metadata
        assert [c.name for c in md.channels] == ["Cy5", "DAPI", "GFP"]
        assert md.pixel_sizes["x"].value == pytest.approx(0.65)
        assert md.pixel_sizes["x"].unit == "micrometer"
        np.testing.assert_array_equal(im.as_axes("tczyx")[1, 2, 0], _plane(0, 1, 2, 0))
    plain = write_ndtiff(str(tmp_path / "plain"), [({"time": 0}, _plane(0, 0, 0, 0))])
    with micro_reader.open(plain[0]) as f:                      # nothing recorded: unknown
        assert not f.images[0].metadata.pixel_sizes
        assert [c.name for c in f.images[0].metadata.channels] == [None]


def test_ndtiff_missing_plane_is_zeros_with_a_warning(tmp_path):
    frames = [({"position": 0, "time": t, "z": z}, _plane(0, t, 0, z))
              for t in range(2) for z in range(3) if (t, z) != (1, 2)]
    paths = write_ndtiff(str(tmp_path / "acq"), frames)
    with pytest.warns(micro_reader.MissingImagesWarning, match="1 missing plane"):
        f = micro_reader.open(paths[0])
    with f:
        a = f.images[0].as_axes("tczyx")
        assert not a[1, 0, 2].any()
        np.testing.assert_array_equal(a[1, 0, 1], _plane(0, 1, 0, 1))
