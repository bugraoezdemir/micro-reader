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
  The tiles of a scan with SwapXY, FlipX or FlipY set have no position (not
  verified on any file).
- Coordinate system (plan 4.4, 4.14, 4.15): a file has no session id, so
  ``stage_systems`` decides from the placed images' XML elements whether
  their positions share one stage frame.  They do -- system ``stage`` --
  when every placed element records one ``SystemSerialNumber``, one LAS X
  run (the node and clock sequence of its version-1 ``UniqueID``), and a
  ``TimeStampList`` whose acquisitions follow each other with no break of
  more than ``SESSION_GAP_HOURS``.  A file holding several such sessions
  gives each its own frame, ``stage-r<k>`` (the k-th session in series
  order; decision K1-2), so the series of one session stay in one frame.  A
  placed element without a time stamp, a serial number or a version-1 UUID
  cannot be tied to a session: its series is its own frame,
  ``stage-s<series>`` (8.17).  The time guard is needed because a saved or
  exported project can renumber the UUIDs.  Checked on every local LIF
  (check 4.14): one 2018 file holds a run from 6 days later.
"""
from __future__ import annotations

import uuid

from .._metadata import Channel, Metadata, Quantity, color_hex, quantity
from .._spatial import system_id

#: LAS lookup table names -> colour; "Gray" is no colour
LUT_COLORS = {"red": "#FF0000", "green": "#00FF00", "blue": "#0000FF", "cyan": "#00FFFF",
              "magenta": "#FF00FF", "yellow": "#FFFF00"}
DIM_SIZE_AXES = {1: "x", 2: "y", 3: "z", 4: "t"}
#: dimensions whose samples are wavelengths (D5): emission lambda, excitation
WAVELENGTH_INDICES = {5: "emission", 9: "excitation"}
#: TileScanInfo flags: a scan with any of them set leaves its tiles unplaced
TILE_FLAGS = ("SwapXY", "FlipX", "FlipY")
#: the system of a file whose placed images share one stage frame (4.4)
STAGE = system_id("stage")
#: the longest break between two acquisitions of one session (check 4.14:
#: the longest break inside a run on the samples is 0.95 h, the one split 138 h)
SESSION_GAP_HOURS = 2.0
#: FILETIME ticks (100 ns) per hour
_TICKS_PER_HOUR = 3600 * 10 ** 7


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
        tiles = _placed_tiles(element)
        if tile >= len(tiles):
            return {}
        source, names = tiles[tile], ("PosX", "PosY")
    elif setting is not None:
        source, names = setting, ("StagePosX", "StagePosY")
    else:
        return {}
    return _xy(source, names)


def _xy(source, names) -> dict:
    out = {a: quantity(source.get(n), "m", positive=False) for a, n in zip("xy", names)}
    return out if all(out.values()) else {}


def _placed_tiles(element) -> list:
    """The ``Tile`` entries of a tile scan, or [] when the scan has none or
    has SwapXY / FlipX / FlipY set (tile positions in another frame: not
    verified)."""
    info = element.find("Data/Image/Attachment[@Name='TileScanInfo']")
    if info is None or any(info.get(k, "0") != "0" for k in TILE_FLAGS):
        return []
    return info.findall("Tile")


def has_position(element, tiles: int) -> bool:
    """Whether some image of the element gets a position from
    ``lif_metadata`` (*tiles*: the element's number of tiles; above 1 each
    tile is its own image), asking the same XML ``lif_metadata`` does."""
    if tiles > 1:
        return any(_xy(t, ("PosX", "PosY")) for t in _placed_tiles(element)[:tiles])
    return bool(_stage_position(element, _setting(element), None))


def _acquisition(element) -> tuple:
    """(serial number, LAS X run, first and last time stamp) of an image
    element, each None when the XML does not record it.  The run is the
    (node, clock sequence) of a version-1 ``UniqueID``; the time stamps are
    the first and last entries of ``TimeStampList`` (FILETIME ticks)."""
    setting = _setting(element)
    serial = (setting.get("SystemSerialNumber") or None) if setting is not None else None
    try:
        u = uuid.UUID(element.get("UniqueID") or "")
        run = (u.node, u.clock_seq) if u.version == 1 else None
    except ValueError:
        run = None
    first = last = None
    text = (element.findtext("Data/Image/TimeStampList") or "").strip()
    if text:
        try:
            first, last = int(text.split(None, 1)[0], 16), int(text.rsplit(None, 1)[-1], 16)
        except ValueError:
            first = last = None
    return serial, run, first, last


def stage_systems(elements: list) -> list:
    """The coordinate system id of each series' positions (plan 4.4, 4.14,
    4.15, decision K1-2): *elements* lists each series' (XML element, number
    of tiles).  The placed series are grouped into sessions (one microscope,
    one LAS X run, no break of more than ``SESSION_GAP_HOURS``: module
    docstring).  ``STAGE`` for every placed series when they are all one
    session; else ``stage-r<k>`` for the k-th session (numbered in series
    order), and ``stage-s<series>`` for a series whose XML records too little
    to tell (8.17).  None for a series no image of which is placed."""
    placed = [has_position(element, tiles) for element, tiles in elements]
    records = {n: _acquisition(element) for n, ((element, _), p) in enumerate(zip(elements, placed))
               if p}
    if not records:
        return [None] * len(elements)
    known = {n: r for n, r in records.items() if r[0] is not None and r[1] is not None
             and r[2] is not None}
    sessions = _sessions(known)
    if len(sessions) == 1 and len(known) == len(records):
        return [STAGE if p else None for p in placed]
    session_of = {}
    for k, members in enumerate(sorted(sessions, key=min)):
        for n in members:
            session_of[n] = system_id("stage", f"r{k}")
    return [None if not p else session_of.get(n) or system_id("stage", f"s{n}")
            for n, p in enumerate(placed)]


def _sessions(records: dict) -> list:
    """The sessions of *records* ({series: (serial, run, first, last)}, every
    field known): series of one microscope and one LAS X run whose time
    spans follow each other with no break of more than ``SESSION_GAP_HOURS``,
    as lists of series numbers."""
    gap = SESSION_GAP_HOURS * _TICKS_PER_HOUR
    groups: dict = {}
    for n, (serial, run, first, last) in records.items():
        groups.setdefault((serial, run), []).append((first, last, n))
    out = []
    for spans in groups.values():
        current, end = [], None
        for first, last, n in sorted(spans):
            if end is not None and first - end > gap:
                out.append(current)
                current, end = [], None
            current.append(n)
            end = last if end is None else max(end, last)
        out.append(current)
    return out


def lif_metadata(element, shape: dict, index: dict, tiled: bool = False,
                 system=None) -> Metadata:
    """*index*: the image's index; *tiled*: whether it is a tile of a tile
    scan; *system*: the id of the coordinate system its series' positions
    are in (``stage_systems``), named when the image has a position."""
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
                    index_values=index_values,
                    coordinate_system=system if position else None)
