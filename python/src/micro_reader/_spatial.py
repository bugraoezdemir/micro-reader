"""The spatial model (plan "NGFF 0.6 and RFC 8: one spatial model", section 3):
the value classes of 3.16 and 3.1, the ids micro-reader makes (3.23) and the
invariants a file keeps (3.22).

A ``File`` declares the shared coordinate systems its positions are in when it
opens, from what opening already read (``File._declare_coordinate_systems``),
never from an image's metadata.  An image's ``metadata.position`` names the
system it is in through ``metadata.coordinate_system``, set exactly when the
position is not empty (I3).  Edges other than positions are
``File.coordinate_transformations`` (only BDV's affine views and
registrations, 9.16),
and the tree of what contains what is ``File.collection`` (3.17).

Frozen dataclasses.  ozp defines its own and reads these by their fields, as
it reads ``image.index``; it never imports them."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional, Sequence

from ._errors import InvariantError

#: the NGFF type of an axis, inferred from its name when none is given
AXIS_TYPES = {"t": "time", "c": "channel", "x": "space", "y": "space", "z": "space"}

#: the transformation types micro-reader and ozp model (parse, compose, invert
#: where possible; 8.20); every other type is carried as raw JSON in ``params``
MODELLED_TYPES = ("identity", "translation", "scale", "affine", "sequence")

#: *pattern-safe* (3.23): what RFC 8 allows in an id
PATTERN_SAFE = re.compile(r"[a-zA-Z0-9_.-]+")
#: an id micro-reader makes: pattern-safe without "." (kept for the ids of an
#: image's own system) and without "--" (kept for the prefixes of merged sets)
MADE_ID = re.compile(r"(?!.*--)[a-zA-Z0-9_-]+")

#: the index that never enters an image key (1.14)
_RESOLUTION = "resolution"


# -- value classes (3.16, 3.1) --------------------------------------------------

@dataclass(frozen=True)
class Axis:
    """One axis of a coordinate system, with every field NGFF 0.6 gives an
    axis.  *type* is "space", "time", "channel" or another 0.6 type; left
    None, it is inferred from the name for ``t``, ``c``, ``x``, ``y`` and
    ``z``, and any other name raises.  *unit*: a UDUNITS-2 name
    (``_metadata.SPACE_UNITS`` / ``TIME_UNITS``) or None; *discrete* and
    *long_name* as 0.6 allows; *extra*: the other keys of an axis read from a
    file, as read."""
    name: str
    type: Optional[str] = None
    unit: Optional[str] = None
    discrete: Optional[bool] = None
    long_name: Optional[str] = None
    extra: dict = field(default_factory=dict, hash=False)

    def __post_init__(self):
        if not isinstance(self.name, str) or not self.name:
            raise ValueError(f"Axis: the name must be a non-empty string, not {self.name!r}")
        if self.type is None:
            inferred = AXIS_TYPES.get(self.name)
            if inferred is None:
                raise ValueError(
                    f"Axis {self.name!r}: no type can be inferred from the name (only "
                    f"{', '.join(AXIS_TYPES)} have one): give type=, e.g. "
                    f"Axis({self.name!r}, 'channel')")
            object.__setattr__(self, "type", inferred)
        elif not isinstance(self.type, str) or not self.type:
            raise ValueError(f"Axis {self.name!r}: type must be a non-empty string or None "
                             f"(inferred), not {self.type!r}")
        if self.unit is not None and not isinstance(self.unit, str):
            raise ValueError(f"Axis {self.name!r}: unit must be a UDUNITS-2 name or None, "
                             f"not {self.unit!r}")
        object.__setattr__(self, "extra", dict(self.extra))


@dataclass(frozen=True)
class CoordinateSystem:
    """A shared coordinate system of an opened file (1.4): a pattern-safe
    *id* (3.23; ``system_id`` makes one), a readable *name* (the id when
    none is given), *axes* as a tuple of ``Axis`` -- or None for a system a
    backend declares: the writer fills them from the images in it (3.12) --
    and *estimated*: False for a nominal system (the only kind a position
    goes into, 3.7), True for one a registration estimated.  *extra*: the
    other keys of a system read from a file, as read."""
    id: str
    name: Optional[str] = None
    axes: Optional[tuple] = None
    estimated: bool = False
    extra: dict = field(default_factory=dict, hash=False)

    def __post_init__(self):
        if not isinstance(self.id, str) or not PATTERN_SAFE.fullmatch(self.id):
            raise ValueError(f"CoordinateSystem: id {self.id!r} is not pattern-safe "
                             f"(plan 3.23: {PATTERN_SAFE.pattern}); system_id() makes an id "
                             "from any text")
        if self.name is None:
            object.__setattr__(self, "name", self.id)
        elif not isinstance(self.name, str):
            raise ValueError(f"CoordinateSystem {self.id!r}: name must be a string or None, "
                             f"not {self.name!r}")
        if self.axes is not None:
            axes = tuple(self.axes)
            bad = [a for a in axes if not isinstance(a, Axis)]
            if bad:
                raise TypeError(f"CoordinateSystem {self.id!r}: axes must be Axis objects, "
                                f"not {bad[0]!r}")
            names = [a.name for a in axes]
            if len(set(names)) != len(names):
                raise ValueError(f"CoordinateSystem {self.id!r}: axis names repeat: {names}")
            object.__setattr__(self, "axes", axes)
        if not isinstance(self.estimated, bool):       # "no" or 0 would read as a choice
            raise TypeError(f"CoordinateSystem {self.id!r}: estimated must be True or False, "
                            f"not {self.estimated!r} (pass bool(...) of what the file says)")
        object.__setattr__(self, "extra", dict(self.extra))


def _check_endpoint(value, what: str, where: str):
    """*value* as an edge endpoint (I5): a system id, an image key (a tuple
    of (name, value) pairs, 1.14) or None (only inside a sequence)."""
    if value is None or (isinstance(value, str) and PATTERN_SAFE.fullmatch(value)):
        return value
    if isinstance(value, tuple) and all(isinstance(p, tuple) and len(p) == 2
                                        and isinstance(p[0], str) for p in value):
        return value
    raise TypeError(f"Transformation {where}: {what} must be a coordinate system id, an "
                    f"image key (a tuple of (index name, value) pairs; image_key() makes "
                    f"one) or None, not {value!r}")


@dataclass(frozen=True)
class Transformation:
    """An edge of the transformation graph (1.5, 3.9): *type* (one of
    ``MODELLED_TYPES``, or any other 0.6 type carried as read), *input* and
    *output* -- a shared system id, an image key (1.14: the image's own
    system; ``image_key``), or None inside a sequence; nothing else --
    *params*: the transformation's other JSON fields as read (``"affine"``,
    ``"translation"``, ``"transformations"`` of a sequence, ...), and an
    optional *name*, as 0.6 allows."""
    type: str
    input: object
    output: object
    params: dict = field(default_factory=dict, hash=False)
    name: Optional[str] = None

    def __post_init__(self):
        if not isinstance(self.type, str) or not self.type:
            raise ValueError(f"Transformation: type must be a non-empty string "
                             f"(e.g. one of {MODELLED_TYPES}), not {self.type!r}")
        where = f"{self.type!r}" + (f" {self.name!r}" if self.name else "")
        _check_endpoint(self.input, "input", where)
        _check_endpoint(self.output, "output", where)
        object.__setattr__(self, "params", dict(self.params))


#: the indices a collection of each kind holds, with its ancestors' (3.17):
#: plate > well > field nest inside a plate, a series sits under the root
_HELD = {None: (), "plate": ("plate",), "well": ("plate", "well"),
         "field": ("plate", "well", "field"), "series": ("series",)}
#: the kinds of collection that collapse into their image when they hold only
#: one; a plate and a well never collapse (3.17)
_COLLAPSING = ("field", "series")


@dataclass(frozen=True)
class Collection:
    """A group in a file's tree (1.2, 3.1, 3.17): what contains what.  *kind*
    says what it is in micro-reader's words ("plate", "well", "field",
    "series"; None for the root), *index* its value of that index (None for
    the root), *children* the collections and level-0 images in it, in
    ``File.images`` order.  ``images`` lists the leaves, depth first;
    ``names`` the children's names in the tree (an image's is not its
    ``Image.name``).  ``File.collection`` builds one (``file_collection``)."""
    name: str
    kind: Optional[str] = None
    index: Optional[object] = None
    children: tuple = ()

    def __post_init__(self):
        from ._base import Image
        if not isinstance(self.name, str):
            raise ValueError(f"Collection: name must be a string, not {self.name!r}")
        children = tuple(self.children)
        bad = [c for c in children if not isinstance(c, (Collection, Image))]
        if bad:
            raise TypeError(f"Collection {self.name!r}: children must be Collections or "
                            f"Images, not {bad[0]!r}")
        object.__setattr__(self, "children", children)

    @property
    def images(self) -> list:
        """Every image under this collection, depth first (the leaves)."""
        out = []
        for child in self.children:
            if isinstance(child, Collection):
                out.extend(child.images)
            else:
                out.append(child)
        return out

    @property
    def names(self) -> tuple:
        """Each child's name in the tree, in ``children`` order (3.17): a
        collection's own ``name``; for an image, its index values that no
        ancestor holds, as ``tile3`` or ``view1_illumination0`` (in the
        index's own order, without ``resolution``), or ``image`` when every
        value is held.  What the ancestors hold follows from this
        collection's kind (plate > well > field, or series)."""
        held = _HELD.get(self.kind, (self.kind,))
        return tuple(child.name if isinstance(child, Collection) else image_name(child, held)
                     for child in self.children)

    def __repr__(self) -> str:
        kind = f" {self.kind} {self.index}" if self.kind is not None else ""
        return f"<micro_reader.Collection {self.name!r}{kind}: {len(self.children)} children>"


# -- the tree (3.17, 3.8) ---------------------------------------------------------

def image_name(image, held=()) -> str:
    """*image*'s name in a tree (3.17): its index values outside *held* (the
    index names its ancestors hold), each as name + value, joined with "_",
    in the index's own order and without ``resolution``; ``image`` when none
    is left."""
    parts = [f"{name}{value}" for name, value in dict.items(image.index)
             if name != _RESOLUTION and name not in held]
    return "_".join(parts) or "image"


def file_stem(path) -> str:
    """A file's name without its extension, the root name of ``File.collection``
    (3.17; the rule ozp's ``tree_from_file`` uses, decided 2026-10-21, M1-2):
    ``xyz.czi`` -> ``xyz``, ``cells.ome.tif`` / ``cells.ome.zarr`` -> ``cells``,
    ``Index.idx.xml`` -> ``Index.idx``; a folder keeps its whole name, except
    ``.zarr``."""
    import os
    path = str(path).rstrip("/\\")
    base = os.path.basename(path)
    for pattern in (r"\.ome\.[A-Za-z0-9]+$", r"\.zarr$"):
        stem = re.sub(pattern, "", base, flags=re.IGNORECASE)
        if stem != base:
            return stem or base
    if os.path.isdir(path):
        return base
    return os.path.splitext(base)[0] or base


def file_collection(file) -> Collection:
    """The tree of an opened *file* (plan 3.17), from its current ``images``
    and ``plates`` -- never from an image's metadata.

    The root has kind None and the file's stem as name.  Inside a plate,
    plate > well > field nest, each from the image's index ("plate", "well",
    "field") and named from ``File.plates`` (``Plate.name``, ``plate<n>`` for
    a plate without one, ``Well.name``, ``field<n>``); outside plates, an image with a "series" index sits in
    ``series<n>``, any other directly under the root (a CZI label is a
    series of its own).  A field or series that holds one image collapses:
    the image sits in its parent and its index keeps the value.  A plate
    and a well never collapse.  Tile, part, resolution and the orthogonal
    indices (view, illumination, acquisition, ...) make no collection: the
    images differing in them sit side by side.  Only full-resolution images
    are leaves (``resolutions="all"`` lists the levels in ``images`` too),
    in ``images`` order; every leaf is an object of ``images``.

    Every collection's (kind, index) pair is in the index of each image
    under it (3.8).  A plate image whose index names no plate, or a plate or
    well that ``File.plates`` does not hold, raises ``InvariantError``."""
    contains, get = dict.__contains__, dict.get    # not ImageIndex's checks (100,000 images)
    plates = file.plates
    # a node: [kind, value, members (nodes and images, in images order), {(kind, value): node}]
    top = [None, None, [], {}]
    for image in file.images:
        if not image.is_pyramid:
            continue
        index = image.index
        if contains(index, "plate") or contains(index, "well"):
            if not contains(index, "plate"):
                raise InvariantError(
                    f"{file.path}: {image.name!r} has a well but no plate in its index "
                    f"{dict(index)} (a plate image names its plate, 0 when there is one); "
                    "File.images and File.plates still reach it")
            path = [(kind, get(index, kind)) for kind in ("plate", "well", "field")
                    if contains(index, kind)]
        elif contains(index, "series"):
            path = [("series", get(index, "series"))]
        else:
            path = []
        node = top
        for step in path:
            child = node[3].get(step)
            if child is None:
                child = node[3][step] = [step[0], step[1], [], {}]
                node[2].append(child)
            node = child
        node[2].append(image)

    def named(kind, value, plate):
        if kind not in ("plate", "well"):
            return f"{kind}{value}"
        p = value if kind == "plate" else plate
        if not isinstance(p, int) or not 0 <= p < len(plates):
            raise InvariantError(
                f"{file.path}: images of plate {p!r} are in no plate of File.plates "
                f"({len(plates)} plate(s)): the backend's index and plates disagree")
        if kind == "plate":
            # a plate without a name (Plate.name ""): a collection needs one
            return plates[p].name or f"plate{p}"
        wells = plates[p].wells
        if not isinstance(value, int) or not 0 <= value < len(wells):
            raise InvariantError(
                f"{file.path}: images of well {value!r} are in no well of plate {p} "
                f"({len(wells)} well(s) with images): the backend's index and plates "
                "disagree")
        return wells[value].name

    def build(node, plate=None):
        kind, value, members, _ = node
        if kind == "plate":
            plate = value
        children = []
        for member in members:
            if not isinstance(member, list):
                children.append(member)
            elif member[0] in _COLLAPSING and len(member[2]) == 1 \
                    and not isinstance(member[2][0], list):
                children.append(member[2][0])          # one image: the field / series collapses
            else:
                children.append(build(member, plate))
        name = file_stem(file.path) if kind is None else named(kind, value, plate)
        return Collection(name, kind, value, children)

    return build(top)


def check_tree(collection: Collection, images: Sequence, *, where: str = "") -> None:
    """I7 for a file's tree (3.8, 3.11): every leaf is an object of *images*
    (the file's current ``images``, C2-07) and appears once, every
    full-resolution image of *images* is a leaf, and every collection's
    (kind, index) pair is in the index of each image under it.  Raises
    ``InvariantError``."""
    here = f"{where}: " if where else ""
    listed = {id(image) for image in images}
    seen = set()

    def walk(node, pairs):
        if node.kind is not None:
            pairs = pairs + ((node.kind, node.index),)
        for child in node.children:
            if isinstance(child, Collection):
                walk(child, pairs)
                continue
            if id(child) not in listed:
                raise InvariantError(f"{here}the tree holds {child.name!r}, which is not "
                                     "in File.images (C2-07: built from the current images)")
            if id(child) in seen:
                raise InvariantError(f"{here}{child.name!r} is in the tree twice (I7: one "
                                     "entry per image)")
            seen.add(id(child))
            for kind, value in pairs:
                if not (dict.__contains__(child.index, kind)
                        and dict.get(child.index, kind) == value):
                    raise InvariantError(
                        f"{here}{child.name!r} sits in {kind} {value!r}, which its index "
                        f"{dict(child.index)} does not say (3.8: tree and index agree)")

    walk(collection, ())
    missing = [image.name for image in images if image.is_pyramid and id(image) not in seen]
    if missing:
        raise InvariantError(f"{here}images missing from the tree: {missing[:5]}"
                             f"{' ...' if len(missing) > 5 else ''} (I7)")


# -- ids (3.23, 8.14) and keys (1.14) ---------------------------------------------

def clean_id_part(text) -> str:
    """*text* as one part of an id micro-reader makes: every character
    outside ``[a-zA-Z0-9_-]`` becomes "_", and so does a "-" that starts or
    ends the part or follows another "-".  So a part is pattern-safe, holds
    no "." and, joined to others with "-", never makes "--" (both are
    reserved, 3.23).  An empty text or None raises: there is nothing to name."""
    if text is None or not str(text):
        raise ValueError(f"an empty id part ({text!r}): nothing to name (give the plate, well "
                         "or series it stands for)")
    text = str(text)
    out = []
    for i, ch in enumerate(text):
        if ch == "-":
            out.append("-" if 0 < i < len(text) - 1 and text[i - 1] != "-" else "_")
        elif ch.isascii() and (ch.isalnum() or ch == "_"):
            out.append(ch)
        else:
            out.append("_")
    return "".join(out)


def system_id(what: str, *scope) -> str:
    """The id of a system micro-reader makes (3.23, 8.14): *what* it is
    ("stage", "slide", "global", "well"), then the tokens that scope it,
    each cleaned (``clean_id_part``) and joined with "-".

    >>> system_id("stage"), system_id("stage", "s3"), system_id("well", "B2")
    ('stage', 'stage-s3', 'well-B2')
    >>> system_id("well", "p1", "B2"), system_id("global", "registered")
    ('well-p1-B2', 'global-registered')

    The last is the estimated twin of ``global`` (3.7)."""
    return "-".join(clean_id_part(part) for part in (what, *scope))


def image_key(image_or_index) -> tuple:
    """An image's key (1.14): its index without ``resolution``, as a tuple of
    (name, value) pairs in the index's own order -- the tuple ozp keys its
    entries with, and the name of the image's own system in an edge."""
    index = getattr(image_or_index, "index", image_or_index)
    return tuple((name, value) for name, value in dict.items(index) if name != _RESOLUTION)


# -- invariants (3.22) --------------------------------------------------------------

def _declared_sequence(values, hook: str, kind: str, here: str) -> tuple:
    """What a ``File._declare_*`` hook returned, as a tuple: a tuple or list
    (ordered; ``()`` for none), never None, a single object or a set."""
    if not isinstance(values, (tuple, list)):
        raise InvariantError(f"{here}File.{hook}() returned {values!r}: return a tuple of "
                             f"{kind} (() for none)")
    return tuple(values)


def check_declared_systems(systems: Sequence, *, where: str = "") -> tuple:
    """I1 for the systems a file declares: a tuple (or list) of
    ``CoordinateSystem``, each with an id micro-reader makes (``MADE_ID``: no
    "." and no "--"), every id unique.  Returns them as a tuple; raises
    ``InvariantError``."""
    here = f"{where}: " if where else ""
    systems = _declared_sequence(systems, "_declare_coordinate_systems", "CoordinateSystem", here)
    seen = set()
    for system in systems:
        if not isinstance(system, CoordinateSystem):
            raise InvariantError(f"{here}a declared coordinate system must be a "
                                 f"CoordinateSystem, not {system!r}")
        if not MADE_ID.fullmatch(system.id):
            raise InvariantError(f"{here}coordinate system id {system.id!r} is not one "
                                 f"micro-reader makes (I1: {MADE_ID.pattern}; system_id() "
                                 "makes one)")
        if system.id in seen:
            raise InvariantError(f"{here}coordinate system id {system.id!r} is declared "
                                 "twice (I1: ids are unique in a file)")
        seen.add(system.id)
    return systems


def check_edges(edges: Sequence, systems: Sequence, keys, *, where: str = "") -> tuple:
    """I5 for a file's ``coordinate_transformations``: a tuple (or list) of
    ``Transformation``, each with an input and output that are a declared
    system id or the key of an image in the file (*keys*).  Returns them as a
    tuple; raises ``InvariantError``."""
    here = f"{where}: " if where else ""
    edges = _declared_sequence(edges, "_declare_coordinate_transformations", "Transformation",
                               here)
    ids = {system.id for system in systems}
    keys = set(keys)
    for edge in edges:
        if not isinstance(edge, Transformation):
            raise InvariantError(f"{here}a coordinate transformation must be a "
                                 f"Transformation, not {edge!r}")
        for what in ("input", "output"):
            end = getattr(edge, what)
            if not (end in ids if isinstance(end, str) else end in keys):
                raise InvariantError(
                    f"{here}transformation {edge.type!r}: {what} {end!r} is neither a "
                    f"declared coordinate system ({sorted(ids)}) nor the key of an image "
                    "in the file (I5)")
    return edges


def check_placement(metadata, systems: Sequence, *, require_system: bool = True,
                    where: str = "") -> None:
    """I2 and I3 for one image's metadata: ``coordinate_system`` is None or
    the id of one of *systems* with ``estimated=False``, and it is set exactly
    when ``position`` is not empty.  *require_system=False* tolerates a
    position without a system (a backend that declares no systems yet); a
    system without a position never passes.  Raises ``InvariantError``."""
    here = f"{where}: " if where else ""
    system, position = metadata.coordinate_system, metadata.position
    if system is None:
        if position and require_system:
            raise InvariantError(f"{here}position {sorted(position)} without a coordinate "
                                 "system (I3: a placed image names the system it is in)")
        return
    if not position:
        raise InvariantError(f"{here}coordinate_system {system!r} with an empty position "
                             "(I3: an image is placed exactly when both are given)")
    found = {s.id: s for s in systems}.get(system)
    if found is None:
        raise InvariantError(f"{here}coordinate_system {system!r} is not one the file "
                             f"declares ({sorted(s.id for s in systems)}) (I2)")
    if found.estimated:
        raise InvariantError(f"{here}coordinate_system {system!r} is an estimated system "
                             "(I2: only nominal positions are positions, 3.7)")


def check_invariants(file, *, require_systems: bool = True) -> None:
    """Every invariant of 3.22 micro-reader keeps, on an opened *file*: I1
    and I5 for what it declares (``coordinate_systems``,
    ``coordinate_transformations``), I7 for its tree (``collection``,
    ``check_tree``), I2 and I3 for every image it gave out
    (levels and parts included: ``File._every_image``).  Reads every image's
    metadata.  *require_systems=False*: as ``check_placement``.  Raises
    ``InvariantError`` naming the file and the image."""
    systems = file.coordinate_systems         # I1 and I5 run inside the properties
    file.coordinate_transformations
    check_tree(file.collection, file.images, where=file.path)      # I7
    for image in file._every_image():
        check_placement(image.metadata, systems, require_system=require_systems,
                        where=f"{file.path}: {image.name!r}")
