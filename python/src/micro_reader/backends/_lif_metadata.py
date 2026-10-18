"""Metadata of Leica images (docs/METADATA.md), from the image's XML
element.

- Pixel sizes: ``DimensionDescription`` ``Length`` (first to last sample,
  in ``Unit``; negative for a z stack taken downwards) over
  ``NumberOfElements - 1``, for x, y, z and t (DimID 1, 2, 3, 4) -- as
  Bio-Formats computes them.  A spatial step of exactly 1 m (``Length`` =
  ``NumberOfElements - 1`` in m) is LAS's index space -- FLIM "Phasor
  Plot", "Phasor Mask" and "Pattern Matching Scatter Plot", the only such
  images in every local Leica file -- and gives no pixel size.
- Channels: colour from ``ChannelDescription`` ``LUTName``; names: the dye
  names (``MultiBand DyeName``) of the active detectors of the image's
  confocal setting, only when there is one per channel (sequential scans
  spread channels over several settings: names unknown).  Channel
  wavelengths: out of scope for now.
- Index values (D5): the image's wavelength along a lambda (``emission``,
  DimID 5) or excitation (``excitation``, DimID 9) dimension, ``Origin`` +
  k x ``Length`` / (N - 1), in nm.
- Position (the first pixel along the image's axes): the setting's
  ``StagePosX`` / ``StagePosY`` -- for a tile, its ``TileScanInfo`` PosX /
  PosY -- plus the x / y dimensions' ``Origin``.  Verified on a LAS tile
  scan and its merged image (harness/verify_positions.py): the positions
  are along the image's axes (the scanner's SwapXY / FlipX / FlipY do not
  apply to them), and the merged image's first pixel is its first tile's.
"""
from __future__ import annotations

from .._metadata import Channel, Metadata, Quantity, color_hex, quantity

#: LAS lookup table names -> colour; "Gray" is no colour
LUT_COLORS = {"red": "#FF0000", "green": "#00FF00", "blue": "#0000FF", "cyan": "#00FFFF",
              "magenta": "#FF00FF", "yellow": "#FFFF00"}
DIM_SIZE_AXES = {1: "x", 2: "y", 3: "z", 4: "t"}
#: dimensions whose samples are wavelengths (D5): emission lambda, excitation
WAVELENGTH_INDICES = {5: "emission", 9: "excitation"}


def _setting(element):
    for tag in ("ATLConfocalSettingDefinition", "ATLCameraSettingDefinition"):
        found = element.find(f"Data/Image/Attachment[@Name='HardwareSetting']/{tag}")
        if found is None:
            found = next(element.iter(tag), None)
        if found is not None:
            return found
    return None


def _stage_position(element, setting, tile) -> dict:
    """The stage position recorded for the image (*tile*: None) or for its
    tile (``TileScanInfo`` PosX / PosY): {axis: Quantity}.  Both are along
    the image's own axes -- checked on a LAS tile scan against LAS's merged
    image: a tile with +PosX sits further along x, whatever the scanner's
    SwapXY / FlipX / FlipY (which describe the scan, not the stage)."""
    if tile is not None:
        info = element.find("Data/Image/Attachment[@Name='TileScanInfo']")
        tiles = info.findall("Tile") if info is not None else []
        if tile >= len(tiles) or any(info.get(k, "0") != "0" for k in ("SwapXY", "FlipX", "FlipY")):
            return {}               # tile positions in another frame: not verified
        source, names = tiles[tile], ("PosX", "PosY")
    elif setting is not None:
        source, names = setting, ("StagePosX", "StagePosY")
    else:
        return {}
    out = {a: quantity(source.get(n), "m", positive=False) for a, n in zip("xy", names)}
    return out if all(out.values()) else {}


def lif_metadata(element, shape: dict, index: dict, tiled: bool = False) -> Metadata:
    """*index*: the image's index; *tiled*: whether it is a tile of a tile scan."""
    tile = index.get("tile", 0) if tiled else None
    desc = element.find("Data/Image/ImageDescription")
    sizes = {}
    for d in desc.findall("Dimensions/DimensionDescription"):
        axis = DIM_SIZE_AXES.get(int(d.get("DimID", 0)))
        n = int(d.get("NumberOfElements", 1))
        if axis is None or n < 2:
            continue
        try:
            length = abs(float(d.get("Length")))
        except (TypeError, ValueError):
            continue
        unit = d.get("Unit") or ("s" if axis == "t" else "m")
        if axis != "t" and unit == "m" and length == n - 1:
            continue                # 1 m per sample: index space (FLIM phasor plots), no size
        q = quantity(length / (n - 1), unit)
        if q:
            sizes[axis] = q
    index_values = {}
    for d in desc.findall("Dimensions/DimensionDescription"):
        name = WAVELENGTH_INDICES.get(int(d.get("DimID", 0)))
        n = int(d.get("NumberOfElements", 1))
        k = index.get(name) if name else None
        if k is None or n < 2:
            continue
        try:
            origin, length = float(d.get("Origin", 0)), float(d.get("Length"))
        except (TypeError, ValueError):
            continue
        q = quantity(origin + k * length / (n - 1), d.get("Unit") or "m", positive=False)
        if q:
            index_values[name] = q.to("nanometer")
    descriptions = desc.findall("Channels/ChannelDescription")
    n_c = max(1, shape.get("c", 1))
    colors = [LUT_COLORS.get((c.get("LUTName") or "").lower()) for c in descriptions]
    if shape.get("s", 1) > 1:                 # interleaved RGB: one channel, no LUT colour
        colors = []
    setting = _setting(element)
    names = []
    if setting is not None:
        bands = {b.get("Channel"): b.get("DyeName") or None for b in setting.iter("MultiBand")}
        active = [d.get("Channel") for d in setting.iter("Detector") if d.get("IsActive") == "1"]
        if len(active) == n_c:
            names = [bands.get(ch) for ch in active]
    channels = []
    for k in range(n_c):
        channels.append(Channel(name=names[k] if k < len(names) else None,
                                color=colors[k] if k < len(colors) else None))
    position = {}
    stage = _stage_position(element, setting, tile)
    if stage:
        # Origin: the first sample's place relative to the stage position (a
        # panned scan's offset; 0 when not panned), along the image's axes
        for d in desc.findall("Dimensions/DimensionDescription"):
            axis = {1: "x", 2: "y"}.get(int(d.get("DimID", 0)))
            if axis in stage:
                origin = quantity(d.get("Origin", 0), d.get("Unit") or "m", positive=False)
                q = stage[axis]
                position[axis] = Quantity(q.value + (origin.to(q.unit).value if origin else 0.0),
                                          q.unit)
    return Metadata(pixel_sizes=sizes, position=position, channels=tuple(channels),
                    index_values=index_values)
