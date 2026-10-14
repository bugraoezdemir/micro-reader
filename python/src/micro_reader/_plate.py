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

from dataclasses import dataclass, field as _field, replace
from typing import Tuple

#: indices that nest (docs/AXES.md); every other index is orthogonal
CONTAINMENT = ("plate", "well", "field", "scene", "tile", "part", "resolution")


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


@dataclass(frozen=True, eq=False)
class Field:
    """One field of a well: its field index and its images (full resolution,
    in ``File.images`` order).  ``tiles`` and ``image`` reach them; when an
    orthogonal index (view, illumination, acquisition, ...) also varies in
    the field, ``select`` picks one first:
    ``field.select(view=1).tiles[7]``."""
    index: int
    images: Tuple = ()

    def _orthogonal(self) -> list:
        """The orthogonal indices that take several values in this field."""
        names = {name for image in self.images for name in image.index
                 if name not in CONTAINMENT}
        return sorted(name for name in names
                      if len({image.index.get(name) for image in self.images}) > 1)

    @property
    def tiles(self) -> list:
        """The field's images in tile order (one image when it is not tiled,
        or stitched).  ``tiles[k]`` is the k-th tile present; ``.index["tile"]``
        is its tile index."""
        varying = self._orthogonal()
        if varying:
            raise ValueError(
                f"field {self.index} also has several {', '.join(varying)}: pick one with "
                f"select({varying[0]}=...) first, or use .images")
        return sorted(self.images, key=lambda image: image.index.get("tile", 0))

    @property
    def image(self):
        """The field's one image (a field that is not tiled, or stitched)."""
        if len(self.images) != 1:
            raise ValueError(f"field {self.index} has {len(self.images)} images: use .tiles "
                             "(or .select(...)) to pick one")
        return self.images[0]

    def select(self, **index) -> "Field":
        """This field with only the images whose index matches, e.g.
        ``select(view=1)``."""
        from ._base import INDEX_NAMES
        unknown = set(index) - set(INDEX_NAMES)
        if unknown:
            raise KeyError(f"unknown indices {sorted(unknown)}; the indices are {INDEX_NAMES}")
        return replace(self, images=tuple(
            image for image in self.images
            if all(image.index.get(name) == value for name, value in index.items())))

    def __repr__(self) -> str:
        return f"<micro_reader.Field {self.index}: {len(self.images)} images>"


@dataclass(frozen=True)
class Well:
    """A well that holds images: its 0-based row and column, its name (e.g.
    "B2": row 1, column 1) and its fields (``fields[j]`` is the j-th field
    present; ``.index`` is its field index)."""
    row: int
    column: int
    name: str
    fields: Tuple[Field, ...] = _field(default=(), compare=False, repr=False)


@dataclass(frozen=True)
class Acquisition:
    """One imaging run of a plate."""
    id: int
    name: str = ""


@dataclass(frozen=True)
class Plate:
    """A plate: its name (``""`` when the file has none), the row and column
    names of its layout (``rows[i]`` is the name of row ``i``), the wells
    that hold images (index ``"well": i`` is ``wells[i]``) and its imaging
    runs (``acquisitions``; empty when the file does not record them)."""
    name: str
    rows: Tuple[str, ...]
    columns: Tuple[str, ...]
    wells: Tuple[Well, ...]
    acquisitions: Tuple[Acquisition, ...] = ()

    def well_index(self, name: str) -> int:
        """The index of the well called *name* (e.g. ``"B2"``), for
        ``File.select(well=...)`` and ``wells[...]``."""
        for i, well in enumerate(self.wells):
            if well.name == name:
                return i
        raise KeyError(f"no well {name!r} with images in plate {self.name!r}; "
                       f"wells: {[w.name for w in self.wells]}")

    def __repr__(self) -> str:
        n = len(self.wells)
        return (f"<micro_reader.Plate {self.name!r} {len(self.rows)} x {len(self.columns)}, "
                f"{n} well{'' if n == 1 else 's'} with images>")


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
        out.append(replace(plate, wells=tuple(wells)))
    return out
