"""The spatial model's contract (plan section 3, ``micro_reader._spatial``):
the value classes of 3.16 and 3.1, the ids micro-reader makes (3.23), the
``File`` / ``Metadata`` API (3.3, 3.4, 4.2) and the invariants (3.22).  A
toy backend stands in for the real ones, which declare nothing yet."""
from __future__ import annotations

import copy
import pickle
import types

import numpy as np
import pytest

import micro_reader
from micro_reader import (Axis, Collection, CoordinateSystem, InvariantError, Metadata,
                          Quantity, Transformation)
from micro_reader._base import File, Image
from micro_reader._spatial import (MADE_ID, PATTERN_SAFE, check_declared_systems, check_edges,
                                   check_invariants, check_placement, clean_id_part,
                                   image_key, system_id)
from micro_reader import _registry

UM = "micrometer"


# -- a toy backend ----------------------------------------------------------------

class _ToyImage(Image):
    """A 4 x 4 image placed at (y 2, x 1) um in *system*, or unplaced."""

    def __init__(self, n, index, system=None, placed=True):
        super().__init__(n, "yx", (4, 4), np.uint8, index=index)
        self._system, self._placed = system, placed
        self.metadata_reads = 0

    def _read_block(self, block):
        out = np.zeros(tuple(s.stop - s.start for s in block), self.dtype)
        return out, out.size

    def _read_metadata(self):
        self.metadata_reads += 1
        position = {"y": Quantity(2.0, UM), "x": Quantity(1.0, UM)} if self._placed else {}
        return Metadata(pixel_sizes={"y": Quantity(0.5, UM), "x": Quantity(0.5, UM)},
                        position=position, coordinate_system=self._system if position else None)


class _ToyFile(File):
    backend = "toy"

    def __init__(self, path, images, systems=(), edges=(), read_metadata=False):
        self._systems, self._edges, self._read = systems, edges, read_metadata
        super().__init__(path, images)

    def _declare_coordinate_systems(self):
        if self._read:                                  # what a backend must NOT do
            for image in self.images:
                image.metadata
        return self._systems

    def _declare_coordinate_transformations(self):
        return self._edges


def _file(path="toy.toy", *images, **options):
    return _ToyFile(path, images or (_ToyImage(0, {"series": 0}, "stage"),), **options)


def _module(make):
    """A backend module for ``_registry.BACKENDS``: ``make(path)`` opens it."""
    return types.SimpleNamespace(NAME="toy", sniff=lambda head, path: False,
                                 open=lambda path, tiles, resolutions: make(path))


# -- Axis -------------------------------------------------------------------------

def test_axis_infers_the_type_of_the_five_known_names():
    assert [Axis(n).type for n in "tczyx"] == ["time", "channel", "space", "space", "space"]
    assert Axis("x", unit=UM) == Axis("x", "space", UM)
    assert Axis("x").unit is None and Axis("x").discrete is None and Axis("x").extra == {}


def test_axis_with_an_unknown_name_needs_a_type():
    with pytest.raises(ValueError, match="no type can be inferred.*give type="):
        Axis("lambda")
    assert Axis("lambda", "channel", discrete=True, long_name="wavelength").type == "channel"
    with pytest.raises(ValueError):
        Axis("")
    with pytest.raises(ValueError):
        Axis("x", "")
    with pytest.raises(ValueError):
        Axis("x", unit=5)


def test_axis_keeps_unknown_keys_as_read_and_is_frozen():
    extra = {"vendor": "zeiss"}
    axis = Axis("y", extra=extra)
    extra["vendor"] = "changed"
    assert axis.extra == {"vendor": "zeiss"}            # a copy
    with pytest.raises(AttributeError):
        axis.name = "z"
    assert pickle.loads(pickle.dumps(axis)) == axis and copy.deepcopy(axis) == axis


# -- CoordinateSystem -------------------------------------------------------------

def test_coordinate_system_defaults():
    stage = CoordinateSystem("stage")
    assert (stage.name, stage.axes, stage.estimated, stage.extra) == ("stage", None, False, {})
    named = CoordinateSystem("well-B2", "well B2", axes=[Axis("y"), Axis("x")], estimated=True)
    assert named.name == "well B2" and named.axes == (Axis("y"), Axis("x"))
    assert named.estimated is True
    assert {stage, named, CoordinateSystem("stage")} == {stage, named}   # hashable


@pytest.mark.parametrize("bad", ["no", "False", 1, 0, None, np.bool_(True)])
def test_coordinate_system_refuses_an_estimated_that_is_not_a_bool(bad):
    with pytest.raises(TypeError, match=r"estimated must be True or False.*bool\(\.\.\.\)"):
        CoordinateSystem("x", estimated=bad)


@pytest.mark.parametrize("bad", ["", "world space", "a/b", None, 3])
def test_coordinate_system_refuses_an_id_that_is_not_pattern_safe(bad):
    with pytest.raises(ValueError, match="not pattern-safe.*system_id"):
        CoordinateSystem(bad)


def test_coordinate_system_refuses_bad_axes():
    with pytest.raises(TypeError, match="Axis objects"):
        CoordinateSystem("stage", axes=["x"])
    with pytest.raises(ValueError, match="repeat"):
        CoordinateSystem("stage", axes=[Axis("x"), Axis("x")])


# -- Transformation ---------------------------------------------------------------

def test_transformation_endpoints_are_ids_keys_or_none():
    key = (("series", 0), ("view", 1))
    edge = Transformation("affine", key, "global", {"affine": [[1, 0, 0], [0, 1, 0]]})
    assert (edge.input, edge.output, edge.name) == (key, "global", None)
    inner = Transformation("scale", None, None, {"scale": [1, 1]}, name="unit")
    seq = Transformation("sequence", key, "global", {"transformations": (inner,)})
    assert seq.params["transformations"][0] is inner
    for bad in (5, ["series", 0], (("series",),), ((0, 0),), "a b"):
        with pytest.raises(TypeError, match="coordinate system id, an image key"):
            Transformation("translation", bad, "global")
        with pytest.raises(TypeError):
            Transformation("translation", "global", bad)
    with pytest.raises(ValueError, match="type must be"):
        Transformation("", key, "global")


# -- Collection -------------------------------------------------------------------

def test_collection_holds_collections_and_images_and_lists_its_leaves():
    a, b, c = (_ToyImage(n, {"well": 0, "field": n}) for n in range(3))
    well = Collection("B2", "well", 0, [Collection("field0", "field", 0, [a, b]), c])
    root = Collection("plate.czi", children=[well])
    assert root.kind is None and root.index is None and root.children == (well,)
    assert root.images == [a, b, c] and well.children[0].images == [a, b]
    assert "well 0" in repr(well) and "2 children" in repr(well)
    with pytest.raises(TypeError, match="Collections or Images"):
        Collection("x", children=["image"])
    with pytest.raises(ValueError):
        Collection(None)


# -- ids (3.23) -------------------------------------------------------------------

def test_the_ids_of_the_plan():
    assert system_id("stage") == "stage" and system_id("slide") == "slide"
    assert system_id("stage", "s3") == "stage-s3"
    assert system_id("well", "B2") == "well-B2"
    assert system_id("well", "p1", "B2") == "well-p1-B2"
    assert system_id("global", "registered") == "global-registered"
    assert system_id("stage", 3) == "stage-3"                 # numbers are text


NASTY = ["My Plate.czi", "a--b", "-B2-", "---", "µm", "well/B 2", "a.b.c", "é", "_", "x-y",
         "tab\tname", "日本", ".", "-", "a-", "-a", "a - b"]


@pytest.mark.parametrize("text", NASTY)
def test_cleaned_parts_are_pattern_safe_and_never_make_dots_or_double_dashes(text):
    part = clean_id_part(text)
    assert len(part) == len(text)                        # one character for one
    assert MADE_ID.fullmatch(part), part
    for made in (system_id(part), system_id("well", part), system_id(part, "s0"),
                 system_id("well", part, part)):
        assert MADE_ID.fullmatch(made) and PATTERN_SAFE.fullmatch(made), made
        assert "." not in made and "--" not in made
        CoordinateSystem(made)
        check_declared_systems([CoordinateSystem(made)])


def test_cleaning_examples_and_the_empty_part():
    assert clean_id_part("My Plate.czi") == "My_Plate_czi"
    assert clean_id_part("a--b") == "a-_b" and clean_id_part("x-y") == "x-y"
    assert clean_id_part("-B2-") == "_B2_"
    with pytest.raises(ValueError, match="empty id part"):
        system_id("well", "")
    with pytest.raises(ValueError):
        system_id("")


def test_a_none_part_raises_like_an_empty_one():
    with pytest.raises(ValueError, match=r"empty id part \(None\)"):
        clean_id_part(None)
    with pytest.raises(ValueError, match="empty id part"):
        system_id("well", None)                         # never 'well-None'
    assert clean_id_part(0) == "0" and clean_id_part(False) == "False"   # only None is nothing


def test_image_key_follows_the_index_order_and_drops_the_resolution():
    image = _ToyImage(0, {"series": 2, "tile": 7, "resolution": 1})
    assert image_key(image) == (("series", 2), ("tile", 7))
    assert image_key({"tile": 7, "series": 2}) == (("tile", 7), ("series", 2))
    assert image_key({}) == ()
    assert Transformation("identity", image_key(image), "stage").input == image_key(image)


# -- Metadata.coordinate_system (3.4) -----------------------------------------------

def test_metadata_has_no_system_by_default_and_keeps_it_through_scaling():
    assert Metadata().coordinate_system is None and Metadata.unknown(2).coordinate_system is None
    placed = Metadata(position={"x": Quantity(1.0, UM)}, coordinate_system="stage",
                      pixel_sizes={"x": Quantity(0.5, UM)})
    level = placed.scaled({"x": 2.0})
    assert level.coordinate_system == "stage" and level.position == placed.position
    assert placed.unplaced() == Metadata(pixel_sizes={"x": Quantity(0.5, UM)})


def test_metadata_refuses_a_system_without_a_position():
    with pytest.raises(ValueError, match="empty position.*I3"):
        Metadata(coordinate_system="stage")
    assert Metadata(position={"x": Quantity(1.0, UM)}).coordinate_system is None   # tolerated


# -- File API (3.3, 4.2) -------------------------------------------------------------

def test_a_file_declares_nothing_by_default():
    f = File("plain.tif", [Image(0, "yx", (4, 4), "u1")])
    assert f.coordinate_systems == () and f.coordinate_transformations == ()
    tree = f.collection                                 # built from the indices (3.17)
    assert (tree.name, tree.kind, tree.children) == ("plain", None, (f.images[0],))


def test_declared_systems_are_checked_once_and_cached():
    stage = CoordinateSystem("stage")
    f = _file("a.toy", systems=[stage])
    assert f.coordinate_systems == (stage,) and f.coordinate_systems is f.coordinate_systems
    f._systems = ()                                       # declared once: not read again
    assert f.coordinate_systems == (stage,)
    assert all(image.metadata_reads == 0 for image in f.images)


@pytest.mark.parametrize("systems, message", [
    ([CoordinateSystem("stage"), CoordinateSystem("stage")], "declared twice"),
    ([CoordinateSystem("a.b")], "not one micro-reader makes"),
    ([CoordinateSystem("a--b")], "not one micro-reader makes"),
    (["stage"], "must be a CoordinateSystem"),
])
def test_declared_systems_that_break_i1_raise(systems, message):
    with pytest.raises(InvariantError, match=message):
        _file("a.toy", systems=systems).coordinate_systems


@pytest.mark.parametrize("returned", [None, CoordinateSystem("stage"), "stage",
                                      {CoordinateSystem("stage")},
                                      (s for s in [CoordinateSystem("stage")])])
def test_a_systems_hook_that_returns_no_tuple_raises_naming_the_hook(returned):
    with pytest.raises(InvariantError, match=r"a.toy: File._declare_coordinate_systems\(\) "
                                             r"returned .*: return a tuple of CoordinateSystem"):
        _file("a.toy", systems=returned).coordinate_systems
    with pytest.raises(InvariantError, match="_declare_coordinate_systems"):
        check_declared_systems(returned)


@pytest.mark.parametrize("returned", [None, Transformation("identity", "stage", "stage"), "e"])
def test_an_edges_hook_that_returns_no_tuple_raises_naming_the_hook(returned):
    with pytest.raises(InvariantError, match=r"b.toy: File._declare_coordinate_transformations"
                                             r"\(\) returned .*: return a tuple of Transformation"):
        _file("b.toy", systems=(CoordinateSystem("stage"),),
              edges=returned).coordinate_transformations
    with pytest.raises(InvariantError, match="_declare_coordinate_transformations"):
        check_edges(returned, (), ())


def test_declared_edges_are_checked_against_systems_and_image_keys():
    image = _ToyImage(0, {"series": 0}, "global")
    key = image_key(image)
    good = Transformation("affine", key, "global", {"affine": [[1, 0, 0], [0, 1, 0]]})
    f = _file("b.toy", image, systems=[CoordinateSystem("global")], edges=[good])
    assert f.coordinate_transformations == (good,)
    for edge, message in [
        (Transformation("affine", key, "world"), "output 'world' is neither"),
        (Transformation("affine", (("series", 9),), "global"), "input .* is neither"),
        (Transformation("affine", None, "global"), "input None is neither"),
    ]:
        with pytest.raises(InvariantError, match=message):
            _file("b.toy", image, systems=[CoordinateSystem("global")],
                  edges=[edge]).coordinate_transformations
    with pytest.raises(InvariantError, match="must be a Transformation"):
        _file("b.toy", image, systems=[CoordinateSystem("global")],
              edges=["edge"]).coordinate_transformations


# -- the invariant check (3.22) -----------------------------------------------------

def test_check_placement_i2_and_i3():
    systems = [CoordinateSystem("stage"), CoordinateSystem("stage-registered", estimated=True)]
    at = {"x": Quantity(1.0, UM)}
    check_placement(Metadata(position=at, coordinate_system="stage"), systems)
    check_placement(Metadata(), systems)
    with pytest.raises(InvariantError, match="without a coordinate system .I3"):
        check_placement(Metadata(position=at), systems, where="f: img")
    check_placement(Metadata(position=at), systems, require_system=False)
    with pytest.raises(InvariantError, match="not one the file declares.*I2"):
        check_placement(Metadata(position=at, coordinate_system="slide"), systems)
    with pytest.raises(InvariantError, match="estimated system .I2"):
        check_placement(Metadata(position=at, coordinate_system="stage-registered"), systems)


def test_check_invariants_over_a_file():
    stage = CoordinateSystem("stage")
    placed = _ToyImage(0, {"series": 0}, "stage")
    unplaced = _ToyImage(1, {"series": 1}, placed=False)
    check_invariants(_file("c.toy", placed, unplaced, systems=[stage]))
    with pytest.raises(InvariantError, match="c.toy: 'image2'.*'slide' is not one the file"):
        check_invariants(_file("c.toy", placed, _ToyImage(2, {"series": 2}, "slide"),
                               systems=[stage]))
    no_system = _ToyImage(3, {"series": 3})
    with pytest.raises(InvariantError, match="'image3': position .* without a coordinate"):
        check_invariants(_file("c.toy", no_system, systems=[stage]))
    check_invariants(_file("c.toy", no_system, systems=[stage]), require_systems=False)


# -- through micro_reader.open: the session check of conftest.py -------------------

def test_a_placed_toy_file_passes_the_session_check(tmp_path, monkeypatch):
    """Opened through ``micro_reader.open``, so the conftest fixture declares
    the systems at open (no metadata read) and checks I2 / I3 on each
    metadata read; the toy backend is not in its allowlist, so the check is
    strict.  Levels and axes wrappers are reached too."""
    stage = CoordinateSystem("stage")
    monkeypatch.setattr(_registry, "BACKENDS", _registry.BACKENDS + [_module(
        lambda path: _file(path, _ToyImage(0, {"series": 0}, "stage"),
                           _ToyImage(1, {"series": 1}, placed=False), systems=[stage]))])
    path = str(tmp_path / "placed.toy")
    with micro_reader.open(path, backend="toy") as f:
        assert f.coordinate_systems == (stage,) and f.backend == "toy"
        assert all(image.metadata_reads == 0 for image in f.images)
        assert f.images[0].metadata.coordinate_system == "stage"
        assert f.images[1].metadata.coordinate_system is None
        assert f.images[1].metadata.position == {}
    with micro_reader.open(path, backend="toy", axes="tczyx") as f:
        assert f.images[0].metadata.coordinate_system == "stage"


def test_a_backend_that_reads_metadata_to_declare_is_caught():
    """What the session check reports; here through ``check_invariants``'
    siblings, so the session itself stays green."""
    f = _file("lazy.toy", systems=[CoordinateSystem("stage")], read_metadata=True)
    f.coordinate_systems
    assert f.images[0].metadata_reads == 1          # the hook read it: the fixture flags this


def test_check_edges_and_declared_systems_helpers_return_tuples():
    stage = CoordinateSystem("stage")
    assert check_declared_systems([stage]) == (stage,)
    key = (("series", 0),)
    edge = Transformation("translation", key, "stage", {"translation": [1, 2]})
    assert check_edges([edge], [stage], [key]) == (edge,)
    assert check_edges([], [], []) == ()
