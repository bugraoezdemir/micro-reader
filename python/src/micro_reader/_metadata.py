"""Image metadata: the minimal model of docs/METADATA.md.

Pixel sizes per axis, stage position, channel names / colours /
wavelengths, and the physical value of the image's non-frame indices (the
wavelength of an emission / excitation index, the time of a lifetime bin) --
each
physical value a ``Quantity(value, unit)`` in the file's own unit when
OME-Zarr has a name for it (D1), ``None`` / absent when the file does not
say (D2).  Backends fill one ``Metadata`` per image; ``image.metadata``.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field, replace
from typing import NamedTuple, Optional


class Quantity(NamedTuple):
    value: float
    unit: str

    def to(self, unit: str) -> "Quantity":
        """This quantity in another unit of the same kind."""
        if unit == self.unit:
            return self
        return Quantity(self.value * _FACTORS[self.unit] / _FACTORS[unit], unit)


#: OME-Zarr's units (UDUNITS-2 names), with each one's size in metres or
#: seconds -- the factors convert between units of one kind.
SPACE_UNITS = {
    "angstrom": 1e-10, "attometer": 1e-18, "centimeter": 1e-2, "decimeter": 1e-1,
    "exameter": 1e18, "femtometer": 1e-15, "foot": 0.3048, "gigameter": 1e9,
    "hectometer": 1e2, "inch": 0.0254, "kilometer": 1e3, "megameter": 1e6, "meter": 1.0,
    "micrometer": 1e-6, "mile": 1609.344, "millimeter": 1e-3, "nanometer": 1e-9,
    "parsec": 3.0857e16, "petameter": 1e15, "picometer": 1e-12, "terameter": 1e12,
    "yard": 0.9144, "yoctometer": 1e-24, "yottameter": 1e24, "zeptometer": 1e-21,
    "zettameter": 1e21,
}
TIME_UNITS = {
    "attosecond": 1e-18, "centisecond": 1e-2, "day": 86400.0, "decisecond": 1e-1,
    "exasecond": 1e18, "femtosecond": 1e-15, "gigasecond": 1e9, "hectosecond": 1e2,
    "hour": 3600.0, "kilosecond": 1e3, "megasecond": 1e6, "microsecond": 1e-6,
    "millisecond": 1e-3, "minute": 60.0, "nanosecond": 1e-9, "petasecond": 1e15,
    "picosecond": 1e-12, "second": 1.0, "terasecond": 1e12, "yoctosecond": 1e-24,
    "yottasecond": 1e24, "zeptosecond": 1e-21, "zettasecond": 1e21,
}
_FACTORS = {**SPACE_UNITS, **TIME_UNITS}

#: how files spell units -> OME-Zarr's name (renaming, not converting)
_ALIASES = {
    "m": "meter", "metre": "meter", "meters": "meter", "metres": "meter",
    "cm": "centimeter", "mm": "millimeter", "um": "micrometer", "µm": "micrometer",
    "μm": "micrometer", "micron": "micrometer", "microns": "micrometer",
    "micrometre": "micrometer", "micrometers": "micrometer",
    "�m": "micrometer", "Âµm": "micrometer",   # µ decoded with the wrong charset
    "nm": "nanometer",
    "nanometre": "nanometer", "pm": "picometer", "a": "angstrom", "å": "angstrom",
    "Å": "angstrom", "ang": "angstrom", "angstroms": "angstrom", "in": "inch",
    "inches": "inch", "ft": "foot",
    "s": "second", "sec": "second", "secs": "second", "seconds": "second",
    "ms": "millisecond", "msec": "millisecond", "milliseconds": "millisecond",
    "us": "microsecond", "µs": "microsecond", "μs": "microsecond",
    "microseconds": "microsecond", "ns": "nanosecond", "ps": "picosecond",
    "fs": "femtosecond", "min": "minute", "mins": "minute", "minutes": "minute",
    "h": "hour", "hr": "hour", "hrs": "hour", "hours": "hour", "d": "day", "days": "day",
}


def unit_name(unit: Optional[str]) -> Optional[str]:
    """OME-Zarr's name for *unit* as a file spells it, or None (unknown)."""
    if unit is None:
        return None
    text = str(unit).strip()
    if text in _FACTORS:
        return text
    if text in _ALIASES:                                  # case matters: "Å", "ms" vs "Ms"
        return _ALIASES[text]
    lowered = text.lower()
    if lowered in _FACTORS:
        return lowered
    return _ALIASES.get(lowered)


#: lengths in units above this many metres (cm, m, inch, ...) are given in
#: LARGE_LENGTH_UNIT: microscopy has no use for a pixel of 2.1e-8 meter
#: (Bugra, 2026-10-02); millimetres and smaller keep the file's unit (D1)
LARGEST_KEPT_LENGTH = 1e-3
LARGE_LENGTH_UNIT = "micrometer"


def quantity(value, unit: Optional[str], *, positive: bool = True,
             default_unit: Optional[str] = None) -> Optional[Quantity]:
    """A Quantity from a file's value and unit spelling; None when the value
    is missing or not a finite number (or not positive, for sizes), or the
    unit is unknown.  *default_unit* is used when the file gives none.  The
    file's unit is kept, except lengths in units larger than a millimetre,
    which are converted to micrometres."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number) or (positive and number <= 0):
        return None
    name = unit_name(unit if unit not in (None, "") else default_unit)
    if name is None:
        return None
    if SPACE_UNITS.get(name, 0.0) > LARGEST_KEPT_LENGTH:
        return Quantity(number, name).to(LARGE_LENGTH_UNIT)
    return Quantity(number, name)


def color_hex(value) -> Optional[str]:
    """'#RRGGBB' from the ways files store a colour: '#RRGGBB', '#AARRGGBB',
    'RRGGBB', a signed 32-bit RGBA int (OME), or an (r, g, b) tuple."""
    if value is None:
        return None
    if isinstance(value, (tuple, list)) and len(value) >= 3:
        r, g, b = (int(round(float(v))) for v in value[:3])
        if max(r, g, b) <= 1 and any(isinstance(v, float) for v in value[:3]):
            r, g, b = (int(round(float(v) * 255)) for v in value[:3])
        return "#{:02X}{:02X}{:02X}".format(r & 255, g & 255, b & 255)
    if isinstance(value, int):                             # OME: RGBA, signed
        v = value & 0xFFFFFFFF
        return "#{:02X}{:02X}{:02X}".format((v >> 24) & 255, (v >> 16) & 255, (v >> 8) & 255)
    text = str(value).strip().lstrip("#")
    if re.fullmatch(r"[0-9A-Fa-f]{8}", text):              # AARRGGBB (Zeiss, Imaris)
        return "#" + text[2:].upper()
    if re.fullmatch(r"[0-9A-Fa-f]{6}", text):
        return "#" + text.upper()
    return None


@dataclass(frozen=True)
class Channel:
    name: Optional[str] = None
    color: Optional[str] = None                            # '#RRGGBB'
    emission_wavelength: Optional[Quantity] = None
    excitation_wavelength: Optional[Quantity] = None


@dataclass(frozen=True)
class Metadata:
    """One image's metadata (docs/METADATA.md); what the file does not say
    is absent / None."""
    pixel_sizes: dict = field(default_factory=dict)       # axis -> Quantity (x y z t)
    position: dict = field(default_factory=dict)          # axis -> Quantity
    channels: tuple = ()                                   # Channel per index of c
    #: this image's value of its indices that have one (D5): {"emission":
    #: Quantity(510.0, "nanometer"), "lifetime": Quantity(2.5, "nanosecond")}
    index_values: dict = field(default_factory=dict)
    #: the stage coordinates the file records for this image, as recorded,
    #: where the file's convention (which point of the field, the stage's
    #: directions against the image axes) is NOT verified -- so they are not
    #: a ``position``: axis -> Quantity (Bugra, 2026-10-19; MetaMorph)
    stage_position: dict = field(default_factory=dict)

    def scaled(self, factors: dict) -> "Metadata":
        """This metadata for a resolution level whose pixels are *factors*
        (axis -> float) times larger than level 0's."""
        sizes = {a: Quantity(q.value * factors.get(a, 1.0), q.unit)
                 for a, q in self.pixel_sizes.items()}
        return replace(self, pixel_sizes=sizes)

    @classmethod
    def unknown(cls, n_channels: int = 1) -> "Metadata":
        return cls(channels=tuple(Channel() for _ in range(max(1, n_channels))))
