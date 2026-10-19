"""A File's structure as text (docs/AXES.md, index architecture): a tree of
its containment indices (plate > well > field, or scene), each leaf with
its tiles, the orthogonal indices that vary inside it, the image shape and
the pyramid levels."""
from __future__ import annotations

from ._plate import CONTAINMENT

INDENT = "  "


def runs(values) -> str:
    """Sorted integers as runs: [0, 1, 2, 5, 7, 8] -> "0-2, 5, 7-8"."""
    values = sorted(set(values))
    parts, start = [], None
    for n, v in enumerate(values):
        if start is None:
            start = v
        if n + 1 == len(values) or values[n + 1] != v + 1:
            parts.append(str(start) if start == v else f"{start}-{v}")
            start = None
    return ", ".join(parts)


def _leaf(images) -> str:
    """What one leaf (a field, or a scene) holds."""
    parts = []
    tiles = [image.index["tile"] for image in images if "tile" in image.index]
    if tiles:
        parts.append(f"tiles {runs(tiles)}" if len(set(tiles)) > 1 else f"tile {tiles[0]}")
    names = []
    for image in images:
        names += [n for n in image.index if n not in CONTAINMENT and n not in names]
    varying = [(n, len({image.index.get(n) for image in images})) for n in names]
    varying = [f"{n} {count}" for n, count in varying if count > 1]
    if varying:
        parts.append("x " + ", ".join(varying))
    first = images[0]
    same = all(image.shape == first.shape and image.axes == first.axes for image in images)
    parts.append(f"{first.axes} {first.shape} {first.dtype}" if same else "shapes vary")
    levels = len(first.resolutions)
    if levels > 1:
        parts.append(f"{levels} levels")
    if first.is_ragged:
        typed = len({p.dtype for p in first.parts}) > 1      # parts of different pixel types
        parts.append(("parts: " if typed else "ragged: ") + " + ".join(
            f"{p.shape} {p.dtype}" if typed else str(p.shape) for p in first.parts))
    return ", ".join(parts)


def outline_lines(f) -> list:
    """The outline of File *f*, one string per line (without the header)."""
    images = [image for image in f.images if image.is_pyramid]
    lines = []
    if f.plates:
        for p, plate in enumerate(f.plates):
            name = f" {plate.name!r}" if plate.name else ""
            n = len(plate.wells)
            lines.append(f"plate {p}{name}: {len(plate.rows)} x {len(plate.columns)}, "
                         f"{n} well{'' if n == 1 else 's'} with images")
            for well in plate.wells:
                lines.append(INDENT + f"well {well.name}")
                for fld in well.fields:
                    lines.append(INDENT * 2 + f"field {fld.index}: {_leaf(fld.images)}")
        rest = [image for image in images if "well" not in image.index]
    else:
        rest = images
    by_scene: dict = {}
    for image in rest:
        by_scene.setdefault(image.index.get("scene", 0), []).append(image)
    for scene, members in sorted(by_scene.items()):
        lines.append(f"scene {scene}: {_leaf(members)}")
    return lines
