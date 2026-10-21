"""Pixel and channel metadata (docs/METADATA.md) on synthetic files, per
backend: values in the file's own units (D1), None for what the file does
not say (D2), positions of the first pixel's centre along the image's axes."""
from __future__ import annotations

import struct

import numpy as np
import pytest

import micro_reader
from micro_reader import Quantity
from micro_reader._metadata import color_hex, quantity

tifffile = pytest.importorskip("tifffile")


def _close(q, value, unit):
    assert q is not None, f"expected {value} {unit}, got None"
    assert q.unit == unit
    assert q.value == pytest.approx(value, rel=1e-6, abs=1e-9)


# -- the model ----------------------------------------------------------------

def test_quantities_keep_the_files_unit_under_ome_zarr_names():
    assert quantity("0.5", "µm") == Quantity(0.5, "micrometer")
    assert quantity(0.5, "�m") == Quantity(0.5, "micrometer")      # µ in a wrong charset
    assert quantity(2, "ms") == Quantity(2.0, "millisecond")
    assert quantity(3, "Å") == Quantity(3.0, "angstrom")
    assert quantity(1, "furlong") is None                                # unknown unit
    assert quantity(0, "um") is None and quantity(-1, "um") is None      # sizes are positive
    assert quantity(-1, "um", positive=False) == Quantity(-1.0, "micrometer")
    assert quantity(float("nan"), "um") is None
    assert Quantity(1.5, "micrometer").to("nanometer").value == pytest.approx(1500)


def test_lengths_above_a_millimetre_become_micrometres():
    # Bugra: "metre is absurd"; mm and smaller stay in the file's unit (D1)
    _close(quantity(2.1e-8, "m"), 0.021, "micrometer")
    _close(quantity(0.01, "cm"), 100.0, "micrometer")
    _close(quantity(0.01, "inch"), 254.0, "micrometer")
    assert quantity(1.5, "mm") == Quantity(1.5, "millimeter")
    assert quantity(500, "nm") == Quantity(500.0, "nanometer")
    assert quantity(2, "s") == Quantity(2.0, "second")                   # time: kept


def test_colours_from_the_ways_files_store_them():
    assert color_hex("#FF00A1FF") == "#00A1FF"          # AARRGGBB (Zeiss)
    assert color_hex("00ff5b") == "#00FF5B"
    assert color_hex(10616831) == "#00A1FF"             # OME: signed RGBA int
    assert color_hex((0.0, 1.0, 0.0)) == "#00FF00"      # Imaris: floats 0-1
    assert color_hex((255, 0, 255, 0)) == "#FF00FF"     # LSM: RGBA bytes
    assert color_hex("not a colour") is None


def test_nothing_is_invented_for_a_file_without_metadata(tmp_path):
    path = tmp_path / "plain.tif"
    tifffile.imwrite(path, np.zeros((2, 8, 8), np.uint8), photometric="minisblack",
                     resolutionunit="NONE", metadata=None)
    with micro_reader.open(path) as f:
        meta = f.images[0].metadata
    assert meta.pixel_sizes == {} and meta.position == {}
    assert all(c.name is None and c.color is None for c in meta.channels)


# -- TIFF family ----------------------------------------------------------------

def test_ome_tiff(tmp_path):
    path = tmp_path / "a.ome.tif"
    data = np.zeros((2, 3, 8, 10), np.uint16)                            # C Z Y X
    tifffile.imwrite(path, data, photometric="minisblack", metadata={
        "axes": "CZYX", "PhysicalSizeX": 0.25, "PhysicalSizeXUnit": "µm",
        "PhysicalSizeY": 0.25, "PhysicalSizeYUnit": "µm",
        "PhysicalSizeZ": 500.0, "PhysicalSizeZUnit": "nm",
        "Channel": {"Name": ["DAPI", "GFP"], "Color": [10616831, -1],
                    "EmissionWavelength": [461.0, 510.0], "ExcitationWavelength": [405.0, 488.0]},
        "Plane": {"PositionX": [12.5] * 6, "PositionXUnit": ["mm"] * 6,
                  "PositionY": [-3.0] * 6, "PositionYUnit": ["mm"] * 6}})
    with micro_reader.open(path) as f:
        meta = f.images[0].metadata
    _close(meta.pixel_sizes["x"], 0.25, "micrometer")
    _close(meta.pixel_sizes["z"], 500.0, "nanometer")                    # the file's unit (D1)
    assert [c.name for c in meta.channels] == ["DAPI", "GFP"]
    assert [c.color for c in meta.channels] == ["#00A1FF", None]         # -1: OME's default
    _close(meta.channels[1].emission_wavelength, 510.0, "nanometer")
    _close(meta.channels[0].excitation_wavelength, 405.0, "nanometer")
    # Plane positions are the image's centre (as Bio-Formats writes them):
    # the first pixel's centre is 4.5 (x) / 3.5 (y) pixels of 0.25 um before it
    _close(meta.position["x"], 12.5 - 4.5 * 0.25e-3, "millimeter")
    _close(meta.position["y"], -3.0 - 3.5 * 0.25e-3, "millimeter")


def test_imagej_tiff(tmp_path):
    path = tmp_path / "ij.tif"
    tifffile.imwrite(path, np.zeros((3, 8, 10), np.uint8), imagej=True,
                     resolution=(4.0, 4.0),
                     metadata={"axes": "ZYX", "spacing": 2.0, "unit": "um"})
    with micro_reader.open(path) as f:
        meta = f.images[0].metadata
    _close(meta.pixel_sizes["x"], 0.25, "micrometer")
    _close(meta.pixel_sizes["z"], 2.0, "micrometer")


@pytest.mark.parametrize("unit, expected", [("CENTIMETER", ("micrometer", 100.0)),
                                            ("INCH", ("micrometer", 254.0)), ("NONE", None)])
def test_plain_tiff_resolution_tags(tmp_path, unit, expected):
    path = tmp_path / "plain.tif"
    tifffile.imwrite(path, np.zeros((8, 8), np.uint8), resolution=(100.0, 100.0),
                     resolutionunit=unit, metadata=None)
    with micro_reader.open(path) as f:
        meta = f.images[0].metadata
    if expected is None:                     # pixels per "no unit": not a size
        assert meta.pixel_sizes == {}
    else:
        _close(meta.pixel_sizes["x"], expected[1], expected[0])


def test_resolution_levels_have_their_own_pixel_sizes(tmp_path):
    path = tmp_path / "pyramid.ome.tif"
    data = np.zeros((64, 48), np.uint8)
    with tifffile.TiffWriter(path) as tw:
        tw.write(data, subifds=1, photometric="minisblack",
                 metadata={"axes": "YX", "PhysicalSizeX": 0.5, "PhysicalSizeY": 0.5})
        tw.write(data[::2, ::2], subfiletype=1, photometric="minisblack")
    with micro_reader.open(path) as f:
        base, level = f.images[0].resolutions
        _close(base.metadata.pixel_sizes["x"], 0.5, "micrometer")
        _close(level.metadata.pixel_sizes["x"], 1.0, "micrometer")
        _close(level.metadata.pixel_sizes["y"], 1.0, "micrometer")


def test_axes_views_pass_metadata_through(tmp_path):
    path = tmp_path / "a.ome.tif"
    tifffile.imwrite(path, np.zeros((2, 8, 8), np.uint8), photometric="minisblack",
                     metadata={"axes": "CYX", "PhysicalSizeX": 0.3, "Channel": {"Name": ["a", "b"]}})
    with micro_reader.open(path, axes="canonical", squeeze=True) as f:
        meta = f.images[0].metadata
    _close(meta.pixel_sizes["x"], 0.3, "micrometer")
    assert [c.name for c in meta.channels] == ["a", "b"]


# -- CZI ----------------------------------------------------------------------------

CZI_XML = (
    "<Scaling><Items>"
    '<Distance Id="X"><Value>1E-07</Value><DefaultUnitFormat>µm</DefaultUnitFormat></Distance>'
    '<Distance Id="Y"><Value>1E-07</Value><DefaultUnitFormat>µm</DefaultUnitFormat></Distance>'
    "</Items></Scaling>"
    "<Information><Image><Dimensions>"
    "<T><Positions><Interval><Start>0</Start><Increment>2.5</Increment></Interval></Positions></T>"
    '<Channels><Channel Id="Channel:0" Name="EGFP"><Color>#FF00FF5B</Color>'
    "<ExcitationWavelength>488</ExcitationWavelength><EmissionWavelength>509</EmissionWavelength>"
    "</Channel></Channels>"
    "</Dimensions></Image></Information>")


def _czi_tile(x, mosaic, stage_x):
    from _czi_writer import raw_payload
    tile = np.zeros((8, 10), np.uint8)
    return dict(series=0, mosaic=mosaic, plane={"C": 0, "T": 0}, x=x, y=0, width=10, height=8,
                dtype="u1", compression=0, payload=raw_payload(tile),
                tags={"StageXPosition": stage_x, "StageYPosition": 200.4})


def test_czi(tmp_path):
    from _czi_writer import write_czi
    path = tmp_path / "a.czi"
    # two tiles side by side; Zeiss records the stage position of each
    # tile's centre: tile 0 centred at 100.5 um, 10 pixels of 0.1 um further 101.5
    write_czi(path, [_czi_tile(0, 0, 100.5), _czi_tile(10, 1, 101.5)], metadata=CZI_XML)
    with micro_reader.open(path, tiles="mosaic") as f:
        meta = f.images[0].metadata
    _close(meta.pixel_sizes["x"], 0.1, "micrometer")
    _close(meta.pixel_sizes["t"], 2.5, "second")
    channel = meta.channels[0]
    assert (channel.name, channel.color) == ("EGFP", "#00FF5B")
    _close(channel.excitation_wavelength, 488, "nanometer")
    _close(channel.emission_wavelength, 509, "nanometer")
    # the first pixel's centre: half a tile (5 px) left of tile 0's centre, plus half a pixel
    _close(meta.position["x"], 100.05, "micrometer")
    _close(meta.position["y"], 200.05, "micrometer")
    with micro_reader.open(path) as f:                    # each tile its own image
        second = f.select(tile=1)[0].metadata
    _close(second.position["x"], 101.05, "micrometer")


# -- Imaris ---------------------------------------------------------------------------

def test_imaris(tmp_path):
    h5py = pytest.importorskip("h5py")
    from test_ims import _char, write_ims
    path = tmp_path / "a.ims"
    write_ims(path, real=(4, 10, 20), padded=(4, 16, 32), chunks=(4, 16, 16), n_t=3, n_c=2)
    with h5py.File(path, "a") as f:
        img = f["DataSetInfo/Image"]
        for k, (lo, hi) in enumerate([(100.0, 110.0), (-5.0, 0.0), (0.0, 8.0)]):
            img.attrs[f"ExtMin{k}"], img.attrs[f"ExtMax{k}"] = _char(lo), _char(hi)
        img.attrs["Unit"] = _char("um")
        ch = f.require_group("DataSetInfo/Channel 0")
        ch.attrs["Name"], ch.attrs["Color"] = _char("GFP"), _char("0.000 1.000 0.000")
        ch.attrs["LSMEmissionWavelength"] = _char("510")
        f.require_group("DataSetInfo/Channel 1").attrs["Name"] = _char("(name not specified)")
        times = f.require_group("DataSetInfo/TimeInfo")
        for n, stamp in enumerate(["2020-01-01 10:00:00.000", "2020-01-01 10:00:02.000",
                                   "2020-01-01 10:00:04.000"], start=1):
            times.attrs[f"TimePoint{n}"] = _char(stamp)
    with micro_reader.open(path) as f:
        meta = f.images[0].metadata
    _close(meta.pixel_sizes["x"], 0.5, "micrometer")          # 10 um over 20 pixels
    _close(meta.pixel_sizes["y"], 0.5, "micrometer")
    _close(meta.pixel_sizes["z"], 2.0, "micrometer")
    _close(meta.pixel_sizes["t"], 2.0, "second")
    _close(meta.position["x"], 100.25, "micrometer")           # ExtMin: edge of the first pixel
    assert (meta.channels[0].name, meta.channels[0].color) == ("GFP", "#00FF00")
    _close(meta.channels[0].emission_wavelength, 510, "nanometer")
    assert meta.channels[1].name is None                       # Imaris's placeholder


# -- ND2 ----------------------------------------------------------------------------------

def test_nd2(tmp_path):
    from _nd2_writer import write_nd2
    path = tmp_path / "a.nd2"
    frames = np.zeros((2 * 2 * 3, 8, 10, 1, 1), np.uint16)
    band = {"m_pFilter": {"i0000000000": {"m_EmissionSpectrum": {"pPoint": {
        "Point0": {"eType": 2, "dWavelength": 500.0, "dTValue": 0.0},
        "Point1": {"eType": 3, "dWavelength": 550.0, "dTValue": 0.0}}}}}}
    write_nd2(path, frames, [("T", 2), ("P", 2), ("Z", 3)],
              picture={"dStgLgCT11": -1.0, "dStgLgCT12": 0.0, "dStgLgCT21": 0.0,
                       "dStgLgCT22": -1.0},
              plane_fields={0: {"sDescription": "GFP", "uiColor": 0x00FF00,
                                "pFilterPath": band}})
    with micro_reader.open(path) as f:
        meta = f.select(series=1)[0].metadata
    _close(meta.pixel_sizes["x"], 0.5, "micrometer")
    _close(meta.pixel_sizes["z"], 1.0, "micrometer")
    _close(meta.pixel_sizes["t"], 100.0, "millisecond")       # the file's unit (D1)
    channel = meta.channels[0]
    assert (channel.name, channel.color) == ("GFP", "#00FF00")
    _close(channel.emission_wavelength, 525.0, "nanometer")   # a band: its middle
    # position 1 is at stage (1, 0); the camera is turned by 180 degrees,
    # so along the image's axes the centre is at (-1, 0), and the first
    # pixel 4.5 (x) / 3.5 (y) pixels of 0.5 um before it
    _close(meta.position["x"], -1.0 - 4.5 * 0.5, "micrometer")
    _close(meta.position["y"], 0.0 - 3.5 * 0.5, "micrometer")


# -- Leica ----------------------------------------------------------------------------------

def test_lif_sizes_are_length_over_intervals(tmp_path):
    from _lif_writer import image, write_lif
    path = tmp_path / "a.lif"
    write_lif(path, [image("img", np.zeros((3, 8, 11), np.uint8), axes="zyx")])
    with micro_reader.open(path) as f:
        meta = f.images[0].metadata
    _close(meta.pixel_sizes["x"], 100.0 / 10, "micrometer")    # first to last of 11 samples
    _close(meta.pixel_sizes["z"], 100.0 / 2, "micrometer")     # (stored in metres)
    assert meta.channels[0].color is None                      # LUT "Gray": no colour


# -- MRC ------------------------------------------------------------------------------------

def test_mrc_pixel_size_in_angstrom(tmp_path):
    from test_mrc import write_mrc
    path = tmp_path / "a.mrc"
    write_mrc(path, np.zeros((4, 8, 10), np.float32))
    raw = bytearray(path.read_bytes())
    struct.pack_into("<3f", raw, 40, 10 * 1.5, 8 * 1.5, 4 * 3.0)     # CELLA, Å
    path.write_bytes(bytes(raw))
    with micro_reader.open(path) as f:
        meta = f.images[0].metadata
    _close(meta.pixel_sizes["x"], 1.5, "angstrom")
    _close(meta.pixel_sizes["z"], 3.0, "angstrom")


# -- values of the non-frame indices (D5: physical meaning only) -----------------------

OME_MODULO = (
    '<?xml version="1.0" encoding="UTF-8"?>'
    '<OME xmlns="http://www.openmicroscopy.org/Schemas/OME/2016-06" '
    'xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">'
    '<Image ID="Image:0" Name="spectral"><Pixels ID="Pixels:0" DimensionOrder="XYCZT" '
    'Type="uint8" SizeX="6" SizeY="4" SizeC="3" SizeZ="1" SizeT="1" BigEndian="false">'
    '<Channel ID="Channel:0:0" SamplesPerPixel="1"/><Channel ID="Channel:0:1" SamplesPerPixel="1"/>'
    '<Channel ID="Channel:0:2" SamplesPerPixel="1"/>'
    '<TiffData IFD="0" PlaneCount="3"/></Pixels>'
    '<AnnotationRef ID="Annotation:Modulo:0"/></Image>'
    '<StructuredAnnotations><XMLAnnotation ID="Annotation:Modulo:0" '
    'Namespace="openmicroscopy.org/omero/dimension/modulo"><Value>'
    '<Modulo namespace="http://www.openmicroscopy.org/Schemas/Additions/2011-09">'
    '<ModuloAlongC Type="lambda" Start="500" Step="10" End="520" Unit="nm"/>'
    "</Modulo></Value></XMLAnnotation></StructuredAnnotations></OME>")


def test_ome_modulo_lambda_values(tmp_path):
    path = tmp_path / "spectral.ome.tif"
    tifffile.imwrite(path, np.zeros((3, 4, 6), np.uint8), photometric="minisblack",
                     description=OME_MODULO, metadata=None)
    with micro_reader.open(path) as f:
        assert f.indices == [{"series": 0, "emission": k} for k in range(3)]
        values = [image.metadata.index_values["emission"] for image in f.images]
    assert [q.value for q in values] == [500.0, 510.0, 520.0]
    assert {q.unit for q in values} == {"nanometer"}


def test_lif_lambda_values(tmp_path):
    from _lif_writer import image, write_lif
    path = tmp_path / "lambda.lif"
    # the writer's dimensions start at 0 and span 1e-4 m
    write_lif(path, [image("img", np.zeros((3, 8, 10), np.uint8), axes="lyx")])
    with micro_reader.open(path) as f:
        values = [image.metadata.index_values["emission"] for image in f.images]
    assert [round(q.value) for q in values] == [0, 50000, 100000]
    assert {q.unit for q in values} == {"nanometer"}


def test_czi_scene_position_places_tiles_by_their_pixels(tmp_path):
    """Without stage positions per subblock, the series's XML position is the
    centre of the whole series; each tile is placed by its own pixels (once
    every tile got the series's centre)."""
    from _czi_writer import raw_payload, write_czi
    xml = CZI_XML.replace(
        "</Dimensions>",
        '<S><Scenes><Scene Index="0"><Positions><Position X="50" Y="20" Z="0"/>'
        "</Positions></Scene></Scenes></S></Dimensions>")
    tiles = [dict(series=0, mosaic=m, plane={"C": 0, "T": 0}, x=x, y=0, width=10, height=8,
                  dtype="u1", compression=0, payload=raw_payload(np.zeros((8, 10), np.uint8)))
             for m, x in ((0, 0), (1, 10))]
    path = tmp_path / "a.czi"
    write_czi(path, tiles, metadata=xml)
    with micro_reader.open(path) as f:
        first, second = (f.select(tile=k)[0].metadata.position for k in (0, 1))
    # the series spans 20 x 8 pixels of 0.1 um centred at (50, 20)
    _close(first["x"], 50 - 9.5 * 0.1, "micrometer")
    _close(second["x"], 50 + 0.5 * 0.1, "micrometer")
    _close(first["y"], 20 - 3.5 * 0.1, "micrometer")


def test_nd2_without_camera_matrix_has_no_position(tmp_path):
    """Without dStgLgCT the camera's orientation on the stage is unknown (a
    real multipoint file without it is turned by 180 degrees): position
    unknown rather than possibly mirrored (once identity was assumed)."""
    from _nd2_writer import write_nd2
    path = tmp_path / "a.nd2"
    write_nd2(path, np.zeros((2, 8, 10, 1, 1), np.uint16), [("P", 2)])
    with micro_reader.open(path) as f:
        meta = f.select(series=1)[0].metadata
    assert meta.position == {}
    _close(meta.pixel_sizes["x"], 0.5, "micrometer")
