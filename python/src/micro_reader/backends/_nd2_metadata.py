"""Metadata of ND2 images (version 3; docs/METADATA.md), from the picture
metadata (``ImageMetadataSeqLV|0``) and the experiment loops
(``ImageMetadataLV``).  Field meanings follow nd2 (Talley Lambert, BSD-3).

- Pixel sizes: ``dCalibration`` (µm per pixel, x and y) when
  ``bCalibrated``; z: the Z loop's ``dZStep`` (µm); t: the time loop's
  ``dPeriod`` (the interval asked for, ms), else its measured
  ``dAvgPeriodDiff``.
- Channels: ``sPicturePlanes/sPlaneNew/a<k>``: ``sDescription``, ``uiColor``
  (ABGR), and for the wavelengths the fluorescent probe's spectrum, else
  the first filter's (a line, or a band's middle: ``_wavelength_of``).
- Position (docs/METADATA.md: the first pixel along the image's axes): the
  XY loop's point for the image's position (else ``dXPos`` / ``dYPos``),
  which NIS records for the centre of the field of view in stage axes,
  brought into the image's axes through the inverse of the camera-to-stage
  matrix ``dStgLgCT``, then moved to the first pixel; unknown without the
  matrix (old files: a multipoint file without one is turned by 180
  degrees, measured on its overlapping fields).
"""
from __future__ import annotations

from .._metadata import Channel, Metadata, Quantity, color_hex, quantity
from ._nd2_structure import NE_TIME_LOOP, TIME_LOOP, XY_POS_LOOP, Z_STACK_LOOP


def _loops(raw: dict) -> list:
    """Every loop of the experiment tree: (eType, parameters), outermost first."""
    out = []

    def walk(node):
        node = node.get("SLxExperiment", node)
        pars = node.get("uLoopPars") or {}
        if list(pars) == ["i0000000000"]:
            pars = pars["i0000000000"]
        out.append((node.get("eType", 0), pars))
        for item in (node.get("ppNextLevelEx") or {}).values():
            walk(item)
    if raw:
        walk(raw)
    return out


def _valid_points(node_pars: dict, valid) -> list:
    points = node_pars.get("Points")
    if isinstance(points, dict):
        points = [points[k] for k in sorted(points)]
    elif "dPosX" in node_pars:
        xs, ys = node_pars["dPosX"], node_pars["dPosY"]
        points = [{"dPosX": xs[k], "dPosY": ys[k]} for k in sorted(xs)]
    else:
        return []
    if valid:
        flags = [v for _, v in sorted(valid.items())] if isinstance(valid, dict) else list(valid)
        points = [p for p, ok in zip(points, flags) if ok]
    return points


def _xy_points(raw: dict) -> list:
    node = (raw or {}).get("SLxExperiment", raw or {})
    stack = [node]
    while stack:
        n = stack.pop(0)
        n = n.get("SLxExperiment", n)
        if n.get("eType") == XY_POS_LOOP:
            pars = n.get("uLoopPars") or {}
            if list(pars) == ["i0000000000"]:
                pars = pars["i0000000000"]
            return _valid_points(pars, n.get("pItemValid"))
        stack.extend((n.get("ppNextLevelEx") or {}).values())
    return []


#: spectrum point types: a band's cut-on / cut-off edge, a single line
BAND_START, BAND_END, LINE = 2, 3, 4


def _wavelength_of(points: list) -> float:
    """One wavelength for a spectrum given as (type, wavelength, transmission)
    points: the line with the highest transmission; else a band's middle;
    else its one edge (a long- / short-pass filter) -- as Bio-Formats
    reports filters (500-550 nm -> 525)."""
    points = [p for p in points if p[1] and p[1] > 0]
    lines = [p for p in points if p[0] == LINE]
    if lines:
        return max(lines, key=lambda p: p[2])[1]
    starts = [p[1] for p in points if p[0] == BAND_START]
    ends = [p[1] for p in points if p[0] == BAND_END]
    if starts and ends:
        return (starts[0] + ends[0]) / 2
    return points[0][1] if points else 0.0


def _spectrum_peak(spectrum) -> float:
    points = (spectrum or {}).get("pPoint") or {}
    return _wavelength_of([(p.get("eType", 0), p.get("dWavelength", 0.0), p.get("dTValue", 0.0))
                           for p in points.values() if isinstance(p, dict)])


def _wavelengths(plane: dict) -> tuple:
    """(excitation, emission) nm, 0 when unknown -- nd2's rules."""
    probe = plane.get("pFluorescentProbe") or {}
    filters = (plane.get("pFilterPath") or {}).get("m_pFilter") or {}
    filt = next(iter(filters.values()), {}) if isinstance(filters, dict) else {}
    ex = _spectrum_peak(probe.get("m_ExcitationSpectrum"))
    if not ex and filt:
        ex = _spectrum_peak(filt.get("m_ExcitationSpectrum"))
    em = _spectrum_peak(probe.get("m_EmissionSpectrum"))
    if not em and filt:
        em = _spectrum_peak(filt.get("m_EmissionSpectrum"))
    return ex, em


def _abgr(value):
    if not isinstance(value, int):
        return None
    return color_hex((value & 255, value >> 8 & 255, value >> 16 & 255))


def nd2_metadata(picture: dict, experiment: dict, position: int, shape: dict) -> Metadata:
    """*picture*: SLxPictureMetadata; *experiment*: the raw experiment;
    *position*: the image's XY position; *shape*: axis -> size."""
    sizes = {}
    if picture.get("bCalibrated"):
        q = quantity(picture.get("dCalibration"), "micrometer")
        if q:
            sizes["x"] = sizes["y"] = q
    for kind, pars in _loops(experiment):
        if kind == Z_STACK_LOOP and "z" not in sizes:
            q = quantity(abs(pars.get("dZStep", 0.0) or 0.0), "micrometer")
            if q:
                sizes["z"] = q
        elif kind == TIME_LOOP and "t" not in sizes:
            q = quantity(pars.get("dPeriod"), "millisecond") or \
                quantity(pars.get("dAvgPeriodDiff"), "millisecond")
            if q:
                sizes["t"] = q
        elif kind == NE_TIME_LOOP and "t" not in sizes:
            periods = pars.get("pPeriod") or {}
            first = periods[min(periods)] if periods else {}
            q = quantity(first.get("dPeriod"), "millisecond") or \
                quantity(first.get("dAvgPeriodDiff"), "millisecond")
            if q:
                sizes["t"] = q

    planes = ((picture.get("sPicturePlanes") or {}).get("sPlaneNew") or {})
    channels = []
    for key in sorted(planes, key=lambda k: int(k[1:]) if k[1:].isdigit() else 0):
        plane = planes[key]
        ex, em = _wavelengths(plane)
        channels.append(Channel(
            name=plane.get("sDescription") or None,
            color=_abgr(plane.get("uiColor")),
            emission_wavelength=quantity(em, "nanometer"),
            excitation_wavelength=quantity(ex, "nanometer")))
    n_c = max(1, shape.get("c", 1))
    channels = (channels + [Channel()] * n_c)[:n_c]

    points = _xy_points(experiment)
    point = points[position] if position < len(points) else \
        {"dPosX": picture.get("dXPos"), "dPosY": picture.get("dYPos")}
    position_out = {}
    centre = (quantity(point.get("dPosX"), "micrometer", positive=False),
              quantity(point.get("dPosY"), "micrometer", positive=False))
    if all(centre) and "x" in sizes:
        # the field's centre along the image's axes: the stage position
        # through the inverse of the camera-to-stage matrix (NIS: about -1
        # on the diagonal for a camera turned by 180 degrees)
        # without the matrix the camera's orientation is unknown (an old
        # multipoint file without one is turned by 180 degrees): no position
        keys = ("dStgLgCT11", "dStgLgCT12", "dStgLgCT21", "dStgLgCT22")
        m11, m12, m21, m22 = (picture.get(k, 0.0) for k in keys)
        det = m11 * m22 - m12 * m21
        if all(k in picture for k in keys) and abs(det) > 1e-6:
            sx, sy = centre[0].value, centre[1].value
            cx = (m22 * sx - m12 * sy) / det
            cy = (-m21 * sx + m11 * sy) / det
            step = sizes["x"].value
            position_out = {
                "x": Quantity(cx + (0.5 - shape.get("x", 1) / 2) * step, "micrometer"),
                "y": Quantity(cy + (0.5 - shape.get("y", 1) / 2) * step, "micrometer")}
    return Metadata(pixel_sizes=sizes, position=position_out, channels=tuple(channels))


# -- legacy ND2 (versions 1 / 2): the per-frame ``VIMD`` XML ----------------------

def _legacy_peak(spectrum) -> float:
    points = (spectrum or {}).get("Point") or {}
    return _wavelength_of([(p.get("Type", 0), p.get("Wavelength", 0.0), p.get("TValue", 0.0))
                           for p in points.values() if isinstance(p, dict)])


def legacy_metadata(frame: dict, shape: dict) -> Metadata:
    """*frame*: the ``VIMD`` of the image's first frame (as nd2 reads it:
    ``Calibration`` µm per pixel, plane ``OpticalConfigName`` and ``Color``
    (ABGR); emission from the probe's spectrum, else the filter's, as
    Bio-Formats has it).  No position: see below."""
    sizes = {}
    q = quantity(frame.get("Calibration"), "micrometer")
    if q:
        sizes["x"] = sizes["y"] = q
    planes = ((frame.get("PicturePlanes") or {}).get("Plane") or {})
    channels = []
    for key in sorted(planes, key=lambda k: int(k) if str(k).isdigit() else 0):
        plane = planes[key]
        filters = (plane.get("FilterPath") or {}).get("Filter") or {}
        filt = filters if "Name" in filters else next(iter(filters.values()), {})
        filt = filt if isinstance(filt, dict) else {}
        em = _legacy_peak((plane.get("FluorescentProbe") or {}).get("EmissionSpectrum")) or \
            _legacy_peak(filt.get("EmissionSpectrum"))
        ex = _legacy_peak((plane.get("FluorescentProbe") or {}).get("ExcitationSpectrum")) or \
            _legacy_peak(filt.get("ExcitationSpectrum"))
        channels.append(Channel(
            name=plane.get("OpticalConfigName") or None,
            color=_abgr(plane.get("Color")),
            emission_wavelength=quantity(em, "nanometer"),
            excitation_wavelength=quantity(ex, "nanometer")))
    n_c = max(1, shape.get("c", 1))
    if len(channels) != n_c:            # channels are components of one plane
        channels = []
    channels = (channels + [Channel()] * n_c)[:n_c]
    # XPos / YPos are stage coordinates of the field's centre, but old files
    # record no camera-to-stage matrix: their direction along the image's
    # axes is unknown, so no position (docs/METADATA.md)
    return Metadata(pixel_sizes=sizes, channels=tuple(channels))
