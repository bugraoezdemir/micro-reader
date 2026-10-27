"""Plates (high-content screening): the layout of a plate whose images a File
holds, and navigation into them (docs/HCS.md).

``File.plates`` lists them, like ``File.images`` lists the images; an image
of a plate says which plate, well and field it is in its index
(``{"plate": 0, "acquisition": 0, "well": 3, "field": 0}``), with integers:
names such as "B2" come from the plate.  Rows and columns count from 0, as in
OME-Zarr (NGFF).

Navigation follows containment only (docs/AXES.md, index architecture):
``plate.wells[i].fields[j].tiles[k]`` reaches the same ``Image`` objects as
``File.images``; orthogonal indices (view, illumination, acquisition, ...)
are picked with ``select``."""
from __future__ import annotations

import numbers
import re
from dataclasses import dataclass, field as _field, replace
from typing import Any, ClassVar, Tuple

#: indices that nest (docs/AXES.md); every other index is orthogonal
CONTAINMENT = ("plate", "well", "field", "series", "tile", "part", "resolution")


def row_label(row: int) -> str:
    """The letters of a 0-based plate row: A, B, ..., Z, AA, AB, ... (1536-well
    plates have 32 rows: A to AF)."""
    if row < 0:
        raise ValueError(f"row {row}: rows count from 0")
    label = ""
    n = row + 1
    while n:
        n, r = divmod(n - 1, 26)
        label = chr(ord("A") + r) + label
    return label


def layout(rows: int, columns: int, wells, where: str) -> tuple:
    """(row names, column names) of a plate layout of *rows* x *columns* as
    the file describes it, grown to hold every well in *wells* ((row,
    column), 0-based) when it is smaller, with a ``PlateLayoutWarning``.  A
    plate's wells always lie inside its layout (consumers index ``rows`` /
    ``columns`` with them).  Seen: a CellVoyager slide holder described as
    1 x 4 slides, whose records address the wells of the 18-well slide in it
    (C05)."""
    wells = list(wells)
    need_rows = max([r + 1 for r, _ in wells] + [0])
    need_columns = max([c + 1 for _, c in wells] + [0])
    if need_rows > rows or need_columns > columns:
        import warnings
        from ._errors import PlateLayoutWarning
        outside = sorted(f"{row_label(r)}{c + 1}" for r, c in wells if r >= rows or c >= columns)
        warnings.warn(PlateLayoutWarning(
            f"{where}: the file describes a {rows} x {columns} layout, but wells "
            f"{', '.join(outside[:5])}{' ...' if len(outside) > 5 else ''} hold images outside "
            f"it; the layout is given as {max(rows, need_rows)} x {max(columns, need_columns)} "
            "to hold them"), stacklevel=3)
        rows, columns = max(rows, need_rows), max(columns, need_columns)
    return tuple(row_label(r) for r in range(rows)), tuple(str(c + 1) for c in range(columns))


#: the standard plate formats (rows, columns), smallest first: 6 to 1536 wells
STANDARD_LAYOUTS = ((2, 3), (3, 4), (4, 6), (6, 8), (8, 12), (16, 24), (32, 48))
#: a well's name in a free-form label: letters, then the column (zero-padded or
#: not), then optionally a separator and a number (a field or point counter):
#: ``A10_0000``, ``B02_0003``, ``E8.1``, ``Well C5``
WELL_LABEL = re.compile(r"^(?:well[\s_-]*)?([A-Za-z]{1,2})0*(\d{1,2})(?:[\s_.#-]+\d+)?$",
                        re.IGNORECASE)


def well_of(label: str):
    """(row, column), 0-based, of a label that names a well (``WELL_LABEL``),
    else None."""
    found = WELL_LABEL.match((label or "").strip())
    if found is None or int(found.group(2)) < 1:
        return None
    letters = found.group(1).upper()
    row = 0
    for ch in letters:
        row = row * 26 + ord(ch) - ord("A") + 1
    return row - 1, int(found.group(2)) - 1


def standard_layout(wells, minimum: int = 0) -> tuple:
    """(rows, columns) of the smallest standard plate (``STANDARD_LAYOUTS``)
    of at least *minimum* wells holding every well in *wells* ((row, column),
    0-based), for files that name their wells but not their plate; None when
    none holds them."""
    rows = max(r for r, _ in wells) + 1
    columns = max(c for _, c in wells) + 1
    return next(((r, c) for r, c in STANDARD_LAYOUTS
                 if rows <= r and columns <= c and r * c >= minimum), None)


def well_system(plates, plate: int, well: int) -> str:
    """The id of the coordinate system of well *well* of plate *plate*
    (indices into *plates* and its ``wells``), for backends whose positions
    are relative to the well's centre (plan 4.9; ids 3.23): ``well-B2``, or
    ``well-p1-B2`` when *plates* holds several (``p`` + the plate index)."""
    from ._spatial import system_id
    name = plates[plate].wells[well].name
    return system_id("well", *((f"p{plate}", name) if len(plates) > 1 else (name,)))


def well_systems(plates) -> tuple:
    """One ``CoordinateSystem`` per well with images of *plates*, plate by
    plate in well order (``well_system``'s ids; names "well B2", or "plate 1
    well B2" with several plates): what a backend of plan 4.9 declares."""
    from ._spatial import CoordinateSystem
    several = len(plates) > 1
    return tuple(CoordinateSystem(well_system(plates, p, w),
                                  f"{f'plate {p} ' if several else ''}well {well.name}")
                 for p, plate in enumerate(plates) for w, well in enumerate(plate.wells))


@dataclass(frozen=True, eq=False)
class Field:
    """One field of a well: its field index and its images (full resolution,
    in ``File.images`` order).  ``tiles`` and ``image`` reach them; when an
    orthogonal index (view, illumination, acquisition, ...) also varies in
    the field, ``select`` picks one first:
    ``field.select(view=1).tiles[7]``.  ``parent`` is its well."""
    index: int
    images: Tuple = ()
    parent: Any = _field(default=None, compare=False, repr=False)
    kind: ClassVar[str] = "field"

    def _orthogonal(self) -> list:
        """The orthogonal indices that take several values in this field."""
        names = {name for image in self.images for name in image.index
                 if name not in CONTAINMENT}
        return sorted(name for name in names
                      if len({image.index.get(name) for image in self.images}) > 1)

    @property
    def tiles(self) -> list:
        """The field's images in tile order (one image when it is not tiled,
        or placed as one mosaic).  ``tiles[k]`` is the k-th tile present; ``.index["tile"]``
        is its tile index."""
        varying = self._orthogonal()
        if varying:
            raise ValueError(
                f"field {self.index} also has several {', '.join(varying)}: pick one with "
                f"select({varying[0]}=...) first, or use .images")
        return sorted(self.images, key=lambda image: image.index.get("tile", 0))

    @property
    def image(self):
        """The field's one image (a field that is not tiled, or placed as one mosaic)."""
        if len(self.images) != 1:
            raise ValueError(f"field {self.index} has {len(self.images)} images: use .tiles "
                             "(or .select(...)) to pick one")
        return self.images[0]

    def select(self, **index) -> "Field":
        """This field with only the images whose index matches, e.g.
        ``select(view=1)`` or ``select(view=[0, 1])``.  A field sits in one
        well: ``row`` / ``column`` are ``File.select``'s."""
        from ._base import INDEX_NAMES
        unknown = set(index) - set(INDEX_NAMES)
        if unknown:
            raise KeyError(f"unknown indices {sorted(unknown)}; the indices are {INDEX_NAMES}")
        wanted = {name: index_values(name, value, "Field.select")
                  for name, value in index.items()}
        return replace(self, images=tuple(
            image for image in self.images
            if all(image.index.get(name) in allowed for name, allowed in wanted.items())))

    def __getattr__(self, name):
        return _wrong_level(self, name)

    def __repr__(self) -> str:
        return f"<micro_reader.Field {self.index}: {len(self.images)} images>"


@dataclass(frozen=True)
class Well:
    """A well that holds images: its 0-based row and column, its name (e.g.
    "B2": row 1, column 1) and its fields (``fields[j]`` is the j-th field
    present; ``.index`` is its field index).  ``parent`` is its plate."""
    row: int
    column: int
    name: str
    fields: Tuple[Field, ...] = _field(default=(), compare=False, repr=False)
    parent: Any = _field(default=None, compare=False, repr=False)
    kind: ClassVar[str] = "well"

    def __getattr__(self, name):
        return _wrong_level(self, name)


@dataclass(frozen=True)
class Acquisition:
    """One imaging run of a plate."""
    id: int
    name: str = ""


@dataclass(frozen=True)
class Plate:
    """A plate: its name (``""`` when the file has none), the row and column
    names of its layout (``row_names[i]`` is the name of row ``i``), the wells
    that hold images (index ``"well": i`` is ``wells[i]``) and its imaging
    runs (``acquisitions``; empty when the file does not record them).
    ``parent`` is its file.

    ``rows`` and ``columns`` slice it (decision C-19): an index, a slice, a
    name or a list of them gives the part of the plate in those rows (or
    columns), still a plate - ``plate.rows[2:5].columns[3:7]``, as
    ``File.select(row=slice(2, 5), column=slice(3, 7))``.  Positions are the
    whole layout's, so a part's ``rows[3]`` is still row D; rows and columns
    are a well's address, not levels of containment."""
    name: str
    row_names: Tuple[str, ...]
    column_names: Tuple[str, ...]
    wells: Tuple[Well, ...]
    acquisitions: Tuple[Acquisition, ...] = ()
    parent: Any = _field(default=None, compare=False, repr=False)
    #: the whole plate's wells, for a part of it (``rows`` / ``columns``):
    #: ``well_index`` answers with the index the images carry
    whole: Tuple[Well, ...] = _field(default=(), compare=False, repr=False)
    kind: ClassVar[str] = "plate"

    @property
    def rows(self) -> "Axis":
        """The plate's rows as a slicer: ``rows[2:5]``, ``rows["B"]``,
        ``rows[["B", "D"]]`` give the part of the plate in those rows."""
        return Axis(self, "row")

    @property
    def columns(self) -> "Axis":
        """The plate's columns as a slicer (see ``rows``)."""
        return Axis(self, "column")

    @property
    def is_part(self) -> bool:
        """Whether this is a part of its plate (``rows`` / ``columns``)."""
        return bool(self.whole) and len(self.wells) != len(self.whole)

    def _cut(self, axis: str, keep) -> "Plate":
        keep = set(keep)
        wells = tuple(w for w in self.wells if getattr(w, axis) in keep)
        return replace(self, wells=wells, whole=self.whole or self.wells)

    def well_index(self, name: str) -> int:
        """The index of the well called *name* (e.g. ``"B2"``), for
        ``File.select(well=...)``: its place in the whole plate's wells (the
        index its images carry), also for a part of the plate."""
        for i, well in enumerate(self.whole or self.wells):
            if well.name == name:
                return i
        raise KeyError(f"no well {name!r} with images in plate {self.name!r}; "
                       f"wells: {[w.name for w in self.whole or self.wells]}")

    def __getattr__(self, name):
        return _wrong_level(self, name)

    def __repr__(self) -> str:
        n = len(self.wells)
        part = " (part)" if self.is_part else ""
        return (f"<micro_reader.Plate {self.name!r}{part} {len(self.row_names)} x "
                f"{len(self.column_names)}, {n} well{'' if n == 1 else 's'} with images>")


class Axis:
    """A plate's rows or columns as a slicer (``Plate.rows``, ``Plate.columns``;
    decision C-19): ``axis[i]``, ``axis[a:b]``, ``axis["C"]`` and
    ``axis[["B", "D"]]`` give the part of the plate in those rows or columns.
    Positions count from 0 in the whole layout; a slice's negative bounds count
    from its end.  ``len`` is the layout's row (column) count; the names are
    ``plate.row_names`` (``column_names``)."""

    def __init__(self, plate: Plate, axis: str):
        self.plate, self.axis = plate, axis
        self.names = plate.row_names if axis == "row" else plate.column_names

    def __len__(self) -> int:
        return len(self.names)

    def __iter__(self):
        raise TypeError(f"plate.{self.axis}s is a slicer, not a sequence: its names are "
                        f"plate.{self.axis}_names; plate.{self.axis}s[...] gives a part of "
                        f"the plate")

    def _positions(self, value) -> list:
        n = len(self.names)
        if isinstance(value, slice):
            return list(range(*value.indices(n)))
        if isinstance(value, str):
            if value not in self.names:
                raise KeyError(f"plate {self.plate.name!r} has no {self.axis} {value!r}; its "
                               f"{self.axis}s are {list(self.names)}")
            return [self.names.index(value)]
        if isinstance(value, _SEQUENCES):
            return [p for v in value for p in self._positions(v)]
        if isinstance(value, numbers.Integral) and not isinstance(value, bool):
            i = int(value) + n if value < 0 else int(value)
            if not 0 <= i < n:
                raise IndexError(f"{self.axis} {value} of a plate with {n} {self.axis}s")
            return [i]
        raise TypeError(f"plate.{self.axis}s[...]: a position, slice, name or a list of "
                        f"them, not {value!r}")

    def __getitem__(self, value) -> Plate:
        return self.plate._cut(self.axis, self._positions(value))

    def __repr__(self) -> str:
        shown = ", ".join(self.names[:4]) + (", ..." if len(self.names) > 4 else "")
        return f"<{self.axis}s of plate {self.plate.name!r}: {shown} ({len(self.names)})>"


#: what each level holds (decision C-19): accessors go down only
_BELOW = {"file": ("plates",), "plate": ("wells", "rows", "columns"), "well": ("fields",),
          "field": ("images", "tiles", "image")}
_ABOVE = {"wells": "plate", "fields": "well", "images": "field", "tiles": "field",
          "plates": "file", "rows": "plate", "columns": "plate"}


def _wrong_level(obj, name: str):
    """``AttributeError`` for an accessor of another level (``well.plates``),
    naming the way: what the object holds, and its ``parent`` going up."""
    if name.startswith("__") or name not in _ABOVE:
        raise AttributeError(f"{type(obj).__name__!r} object has no attribute {name!r}")
    holds = ", ".join(f".{a}" for a in _BELOW.get(obj.kind, ()))
    raise AttributeError(
        f"a {obj.kind} holds no {name} ({name} sit in a {_ABOVE[name]}); a {obj.kind} "
        f"holds {holds or 'images'}, and .parent is what holds it")


#: ``select``'s other address of a plate's wells (0-based numbers or names);
#: not index names, so not on the images
WELL_ADDRESS = ("row", "column")
#: ``select``'s name keywords, beside the positions (``well`` is a place in
#: ``plates[p].wells``, ``well_name`` its name), as the columns of ozp's
#: table.  A string given to a position keyword is its name too
#: (``well="B2"``).
PLACE_NAMES = ("plate_name", "acquisition_name", "well_name", "row_name", "column_name")
#: the positions whose names ``select`` matches: plate, acquisition and well
#: are indices of the images, row and column the well's address
_PLACES = ("plate", "acquisition", "well", "row", "column")
#: a value of one of these types is "any of" its items, in ``select``
_SEQUENCES = (list, tuple, range, set, frozenset)


class Span:
    """A slice as a ``select`` value: every position from ``start`` (0 when None) up to
    ``stop`` (no end when None) in steps of ``step``, as a ``range`` (``row=slice(2,
    None)``: row C onward). Equal to each position it covers, so it matches as any other
    value. Positions count from 0; a negative bound is refused (which position is last
    depends on what is selected)."""

    __slots__ = ("start", "stop", "step")

    def __init__(self, value: slice, where: str, name: str):
        def whole(x):
            return isinstance(x, numbers.Integral) and not isinstance(x, bool)
        for bound in (value.start, value.stop):
            if bound is not None and (not whole(bound) or bound < 0):
                raise ValueError(f"{where}({name}={value!r}): a slice's bounds are 0-based "
                                 f"positions, not {bound!r} (no negative bounds: which "
                                 f"position is last depends on what is selected)")
        if value.step is not None and (not whole(value.step) or value.step < 1):
            raise ValueError(f"{where}({name}={value!r}): a slice's step is a whole number "
                             f"from 1, not {value.step!r}")
        self.start = int(value.start or 0)
        self.stop = None if value.stop is None else int(value.stop)
        self.step = int(value.step or 1)

    def __eq__(self, other):
        if isinstance(other, Span):
            return (self.start, self.stop, self.step) == (other.start, other.stop, other.step)
        if not isinstance(other, numbers.Integral) or isinstance(other, bool):
            return False
        return other >= self.start and (self.stop is None or other < self.stop) \
            and (other - self.start) % self.step == 0

    def __hash__(self):
        return hash((Span, self.start, self.stop, self.step))

    def __repr__(self):
        return f"slice({self.start}, {self.stop}, {self.step})"


def index_values(name: str, value, where: str = "select") -> tuple:
    """``select``'s values for *name*: one value, or a list, tuple, range or
    set of them (any of); a slice of positions is a :class:`Span`.  A plate,
    well, field or acquisition object is refused: it is no index value."""
    values = tuple(value) if isinstance(value, _SEQUENCES) else (value,)
    values = tuple(Span(v, where, name) if isinstance(v, slice) else v for v in values)
    for v in values:
        kind = type(v)
        if kind.__name__ in ("Plate", "Well", "Field", "Acquisition") and \
                kind.__module__.split(".")[0] in ("micro_reader", "ome_zarr_pyramid"):
            raise TypeError(
                f"{where}({name}=...): a {kind.__name__} object is no index value; give "
                f"its index (well=plate.well_index('C3')) or its name (well='C3')")
    return values


def place_conditions(wanted: dict, where: str = "select") -> list:
    """``select``'s conditions on a plate's places, taken out of *wanted*:
    ``(place, positions, names)`` for each keyword that names a place - a
    ``*_name`` keyword, ``row`` or ``column``, or a position keyword given a
    name (``well="B2"``).  A position keyword given numbers only stays in
    *wanted*, matched as any other index.  Every condition must hold; within
    one, any of its positions or names."""
    out = []
    for place in _PLACES:
        named = f"{place}_name"
        if named in wanted:
            values = wanted.pop(named)
            wrong = [v for v in values if not isinstance(v, str)]
            if wrong:
                raise TypeError(f"{where}({named}=...): a name, not {wrong[0]!r}; for a "
                                f"0-based position, {place}={wrong[0]!r}")
            out.append((place, (), frozenset(values)))
        values = wanted.get(place)
        if values is not None and (place in WELL_ADDRESS
                                   or any(isinstance(v, str) for v in values)):
            del wanted[place]
            out.append((place, tuple(v for v in values if not isinstance(v, str)),
                        frozenset(v for v in values if isinstance(v, str))))
    given = {place for place, _, _ in out}
    if "well" in given or "well" in wanted:
        if given & {"row", "column"}:
            raise TypeError(f"{where}: give well= / well_name= or row= / column= (and "
                            f"their _name), not both (row and column are another "
                            f"address of the well)")
    return out


def place_matches(conditions, positions: dict, names: dict) -> bool:
    """Whether an image at *positions* (0-based plate, acquisition, well, row,
    column) with *names* (their names) meets every place condition
    (``place_conditions``)."""
    for place, numbers, labels in conditions:
        if positions.get(place) not in numbers and names.get(place) not in labels:
            return False
    return True


def places_of(plates, index: dict):
    """(positions, names) of an image's places, from its index and the plates
    (``place_matches``); None for an image in no plate."""
    p, w = index.get("plate", 0), index.get("well")
    if w is None or not 0 <= p < len(plates) or not 0 <= w < len(plates[p].wells):
        return None
    plate = plates[p]
    well = plate.wells[w]
    a = index.get("acquisition")
    acq = plate.acquisitions[a].name if a is not None and 0 <= a < len(plate.acquisitions) \
        else None
    positions = {"plate": p, "acquisition": a, "well": w, "row": well.row,
                 "column": well.column}
    names = {"plate": plate.name, "acquisition": acq, "well": well.name,
             "row": plate.row_names[well.row] if 0 <= well.row < len(plate.row_names)
             else None,
             "column": plate.column_names[well.column]
             if 0 <= well.column < len(plate.column_names) else None}
    return positions, names


def navigate(plates, images) -> list:
    """*plates* with their wells' fields filled in from *images* (the images
    at full resolution; lower levels are reached through ``resolutions``)."""
    groups: dict = {}
    for image in images:
        index = image.index
        if "well" in index and image.is_pyramid:
            key = (index.get("plate", 0), index["well"], index.get("field", 0))
            groups.setdefault(key, []).append(image)
    out = []
    for p, plate in enumerate(plates):
        wells = []
        for w, well in enumerate(plate.wells):
            numbers = sorted(f for (pp, ww, f) in groups if pp == p and ww == w)
            fields = tuple(Field(f, tuple(groups[(p, w, f)])) for f in numbers)
            wells.append(replace(well, fields=fields))
        new = replace(plate, wells=tuple(wells))
        for well in new.wells:                 # each level knows what holds it
            object.__setattr__(well, "parent", new)
            for f in well.fields:
                object.__setattr__(f, "parent", well)
        out.append(new)
    return out
