"""Image names: one rule for every format and entry point (decision MR-2,
"micro-reader: specifications of the supported formats", 2026-10-29).

``open`` names every full-resolution image of a file, after the backend has
opened it and before axes are presented:

1. A plate's image: ``"well B2 field 0"``, the field counted from 0 as its
   index (``select(field=0)``); with several plates in the file the plate's
   name in front (``"P1 well B2 field 0"``).  No plate format names its
   fields, so plates always build their names.
2. Otherwise the format's own name, where its specification gives images
   one: a Leica element's Name, an OME ``Image@Name``, an Imaris
   ``Image/Name``, a BigDataViewer setup's name, a slide's associated image by
   its role in lower case (backend: ``_own_name``).  A slide's main scan has no
   name in its format: it takes the main file's stem (``_main_scan``).
3. Otherwise, in a file of several images, a label the format records for
   the image -- a CZI scene name, an ND2 XY point name, a Micro-Manager or
   MetaMorph position label (``_label``; not in a specification, so only to
   tell images of one file apart) -- or else the image's index values in
   index order, those that differ between the file's images: ``"series 2"``,
   ``"series 1, tile 3"``, a BigDataViewer set's ``"view 1"``.
4. Otherwise -- one image in the file -- the main file's stem (``File.path``:
   a set's index file, a composite's main file, a Luxendo experiment's
   folder), never the member that was opened, so every entry point of a
   dataset gives the same name.

Two images left with one name get the index values that differ between
them appended: ``"Series001, emission 3"``.  A resolution level carries its
image's name.  Nothing here reads metadata: the names come from what the
backend read to open the file.
"""
from __future__ import annotations

from collections import defaultdict

#: index names that are no part of an image's identity in a name
_NOT_NAMED = ("resolution",)


def _values(index) -> dict:
    return {k: v for k, v in dict.items(index) if k not in _NOT_NAMED}


def _plate_name(image, index: dict, plates) -> str:
    p, w = index.get("plate", 0), index.get("well")
    plate = plates[p] if isinstance(p, int) and 0 <= p < len(plates) else None
    if plate is not None and isinstance(w, int) and 0 <= w < len(plate.wells):
        well = plate.wells[w].name
    else:
        well = str(w)
    name = f"well {well} field {index.get('field', 0)}"
    if len(plates) > 1:
        name = f"{(plate.name if plate is not None else '') or f'plate {p}'} {name}"
    return name


def _joined(values: dict, keys) -> str:
    return ", ".join(f"{k} {values[k]}" for k in keys)


def assign_names(file) -> None:
    """Name every full-resolution image of *file* (and its levels) by the
    rule above (module docstring)."""
    from ._spatial import file_stem
    images = [image for image in file.images if image.level == 0]
    if not images:
        return
    plates = file.plates
    several = len(images) > 1
    stem = file_stem(file.path)
    # built names show the index values that tell the file's images apart: a value
    # every image shares ("series 0" of a BigDataViewer set's views) is left out
    every = [_values(image.index) for image in images]
    varying = [k for k in every[0] if len({repr(v.get(k)) for v in every}) > 1] if several else []
    named = []                                      # (image, name, its index values, keys in name)
    for image in images:
        values = _values(image.index)
        if "well" in values and plates:
            name, used = _plate_name(image, values, plates), {"plate", "well", "field"}
        else:
            own = getattr(image, "_own_name", None)
            label = getattr(image, "_label", None) if several else None
            if getattr(image, "_main_scan", False):
                name, used = stem, set()
            elif own:
                name, used = own, set()
            elif label:
                name, used = label, set()
            elif several:
                keys = [k for k in values if k in varying] or list(values)
                name, used = _joined(values, keys), set(values)
            else:
                name, used = stem, set(values)
        named.append((image, name, values, used))
    by_name = defaultdict(list)
    for entry in named:
        by_name[entry[1]].append(entry)
    for name, entries in by_name.items():
        if len(entries) > 1:
            keys = [k for k in entries[0][2]
                    if k not in entries[0][3]
                    and len({repr(e[2].get(k)) for e in entries}) > 1]
            for image, _name, values, _used in entries:
                extra = _joined(values, [k for k in keys if k in values])
                _set(image, f"{name}, {extra}" if extra else name)
        else:
            _set(entries[0][0], name)


def _set(image, name: str) -> None:
    image.name = name
    for level in image.resolutions:
        level.name = name
