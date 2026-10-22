"""``File.collection``, the tree of what contains what (plan 3.1, 3.17, 3.8;
``micro_reader._spatial.file_collection``): plate > well > field, or series,
built from the current images' indices and ``File.plates``, never from their
metadata.  On the synthetic writers: a CZI plate with tiles and a label, a
multi-series CZI, an OIR tiling set, a Harmony plate and one-image files; a
toy backend for the naming rules."""
from __future__ import annotations

import numpy as np
import pytest

import micro_reader
from micro_reader import Collection, InvariantError, Plate, Well
from micro_reader._base import File, Image
from micro_reader._spatial import check_tree, file_collection, image_name

from _czi_writer import raw_payload, write_czi
from test_harmony import _write_harmony
from test_hcs import H, SCENES, W, _plate_xml, _write
from test_omp2info import _write_set

def _counted_metadata_reads(f, monkeypatch) -> list:
    """A counter of ``_read_metadata`` calls on every class of every image *f*
    gave out (levels, parts and the native images behind ``axes=``)."""
    calls = [0]
    classes = {cls for image in f._every_image() + list(f.images)
               for cls in type(image).__mro__ if "_read_metadata" in cls.__dict__}
    for cls in classes:
        real = cls.__dict__["_read_metadata"]

        def counting(self, real=real):
            calls[0] += 1
            return real(self)
        monkeypatch.setattr(cls, "_read_metadata", counting)
    return calls


def _fresh_tree(f, monkeypatch) -> Collection:
    """``f.collection`` built again, asserting that building it reads no
    image's metadata (the session check made it once at open)."""
    calls = _counted_metadata_reads(f, monkeypatch)
    f._tree = None
    tree = f.collection
    assert calls[0] == 0, "building the tree read image metadata"
    check_tree(tree, f.images, where=f.path)
    return tree


def _shape(node) -> list:
    """The tree as nested (name, kind, index, children) / image-name lists."""
    out = []
    for name, child in zip(node.names, node.children):
        if isinstance(child, Collection):
            out.append((name, child.kind, child.index, _shape(child)))
        else:
            out.append(name)
    return out


def _ancestors_agree(tree) -> None:
    """3.8: every ancestor's (kind, index) pair is in the image's index."""
    def walk(node, pairs):
        if node.kind is not None:
            pairs = pairs + [(node.kind, node.index)]
        for child in node.children:
            if isinstance(child, Collection):
                walk(child, pairs)
            else:
                assert all(child.index[k] == v for k, v in pairs), (child.index, pairs)
    walk(tree, [])


# -- the synthetic writers ------------------------------------------------------------

def _label_czi(path):
    a = np.random.default_rng(3).integers(0, 255, (6, 8)).astype("u1")
    write_czi(path, [dict(series=0, plane={"C": 0}, x=0, y=0, height=6, width=8, dtype="u1",
                          compression=0, payload=raw_payload(a))])


def _plate_with_label(tmp_path, tiles_per_scene=2):
    """The CZI plate of test_hcs (well B2 with 2 fields, C5 with 1), each
    field of *tiles_per_scene* mosaic tiles, and a slide label attached."""
    _label_czi(str(tmp_path / "label.czi"))
    tiles = [dict(series=s, plane={"C": 0}, x=1000 * s + W * m, y=0, height=H, width=W,
                  dtype="u2", compression=0, payload=raw_payload(np.zeros((H, W), "u2")),
                  mosaic=m)
             for s in SCENES for m in range(tiles_per_scene)]
    path = str(tmp_path / "plate.czi")
    write_czi(path, tiles, _plate_xml(SCENES),
              attachments=[("Label", "CZI", (tmp_path / "label.czi").read_bytes())])
    return path


def test_a_czi_plate_with_tiles_nests_plate_well_field_and_keeps_the_label_at_the_root(
        tmp_path, monkeypatch):
    """6.1's shape: tiles under their field; a field of tiles never collapses;
    the label sits under the root as a series of its own (3.17)."""
    path = _plate_with_label(tmp_path)
    with micro_reader.open(path, plates=True) as f:
        tree = _fresh_tree(f, monkeypatch)
        assert (tree.name, tree.kind, tree.index) == ("plate", None, None)
        assert _shape(tree) == [
            ("plate0", "plate", 0, [
                ("B2", "well", 0, [("field0", "field", 0, ["acquisition0_tile0",
                                                           "acquisition0_tile1"]),
                                   ("field1", "field", 1, ["acquisition0_tile0",
                                                           "acquisition0_tile1"])]),
                ("C5", "well", 1, [("field0", "field", 0, ["acquisition0_tile0",
                                                           "acquisition0_tile1"])])]),
            "series0"]                                            # the label: no plate
        assert tree.images == f.images                            # File.images order
        assert all(any(leaf is image for image in f.images) for leaf in tree.images)
        assert f.images[-1].index == {"series": 0}               # the only series
        _ancestors_agree(tree)
        # the tree and the plate navigation reach the same objects
        (plate,) = tree.children[:1]
        for well, nav in zip(plate.children, f.plates[0].wells):
            for field, nav_field in zip(well.children, nav.fields):
                assert field.images == list(nav_field.tiles)


def test_a_czi_plate_of_one_image_per_field_collapses_the_fields_not_the_wells(
        tmp_path, monkeypatch):
    path = str(tmp_path / "plate.czi")
    _write(path, SCENES, _plate_xml(SCENES))
    with micro_reader.open(path, plates=True) as f:
        tree = _fresh_tree(f, monkeypatch)
        assert _shape(tree) == [("plate0", "plate", 0, [
            ("B2", "well", 0, ["acquisition0_field0", "acquisition0_field1"]),
            ("C5", "well", 1, ["acquisition0_field0"])])]          # a well never collapses
        _ancestors_agree(tree)
    with micro_reader.open(path, plates=True, tiles="mosaic") as f:  # one mosaic per field
        assert _shape(_fresh_tree(f, monkeypatch))[0][3][1] == \
            ("C5", "well", 1, ["acquisition0_field0"])


def test_a_multi_series_czi(tmp_path, monkeypatch):
    """Series of tiles are collections; a series of one image collapses into
    it, its index keeping the series."""
    path = str(tmp_path / "scenes.czi")
    _write(path, SCENES, "", tiles_per_scene=2)
    with micro_reader.open(path) as f:
        tree = _fresh_tree(f, monkeypatch)
        assert _shape(tree) == [(f"series{s}", "series", s, ["tile0", "tile1"])
                                for s in range(3)]
        _ancestors_agree(tree)
    with micro_reader.open(path, tiles="mosaic") as f:
        tree = _fresh_tree(f, monkeypatch)
        assert _shape(tree) == ["series0", "series1", "series2"]
        assert tree.children == tuple(f.images)
        assert [image.index for image in tree.children] == [{"series": s} for s in range(3)]


def test_an_oir_tiling_set_is_one_series_of_tiles(tmp_path, monkeypatch):
    """6.4's shape: series > tiles (``omp2info``)."""
    path, _tiles = _write_set(str(tmp_path / "set"))
    with micro_reader.open(path) as f:
        tree = _fresh_tree(f, monkeypatch)
        assert tree.name == "matl"                               # matl.omp2info
        assert _shape(tree) == [("series0", "series", 0, [f"tile{k}" for k in range(6)])]
        assert tree.children[0].images == f.images
        _ancestors_agree(tree)


def test_a_harmony_plate(tmp_path, monkeypatch):
    """6.6's shape: plate > well > field; one-image fields collapse."""
    index, _data = _write_harmony(str(tmp_path / "harmony"), fields=2)
    with micro_reader.open(index) as f:
        tree = _fresh_tree(f, monkeypatch)
        assert tree.name == "Index.idx"
        fields = ["acquisition0_field0", "acquisition0_field1"]
        assert _shape(tree) == [("Screen plate 1", "plate", 0, [
            ("A1", "well", 0, fields), ("B3", "well", 1, fields)])]
        assert tree.images == f.images
        _ancestors_agree(tree)


def test_a_one_image_file_is_the_root_holding_its_image(tmp_path, monkeypatch):
    path = str(tmp_path / "one.czi")
    _write(path, {0: None}, "")
    with micro_reader.open(path) as f:
        tree = _fresh_tree(f, monkeypatch)
        assert (tree.name, tree.kind, tree.index) == ("one", None, None)
        assert tree.children == (f.images[0],) and tree.names == ("series0",)
    tif = str(tmp_path / "plain.tif")
    import tifffile
    tifffile.imwrite(tif, np.zeros((4, 5), "u1"))
    with micro_reader.open(tif) as f:
        tree = _fresh_tree(f, monkeypatch)
        assert tree.name == "plain" and tree.children == (f.images[0],)


# -- the current images (C2-07) ---------------------------------------------------------

def test_after_open_with_axes_every_leaf_is_an_object_in_images(tmp_path, monkeypatch):
    """``open(axes=..., samples=...)`` replaces ``images`` with axes images
    (``_registry.open``): the tree holds those, not the native ones."""
    path = _plate_with_label(tmp_path)
    with micro_reader.open(path, plates=True, axes="tczyx", samples="channels") as f:
        tree = _fresh_tree(f, monkeypatch)
        assert len(tree.images) == len(f.images) == 7
        for leaf, image in zip(tree.images, f.images):
            assert leaf is image and leaf.source is not leaf
        assert _shape(tree)[0][3][0][3][0][3] == ["acquisition0_tile0", "acquisition0_tile1"]
    index, _data = _write_harmony(str(tmp_path / "harmony"))
    with micro_reader.open(index, axes="tczyx", samples="channels") as f:
        tree = _fresh_tree(f, monkeypatch)
        assert all(any(leaf is image for image in f.images) for leaf in tree.images)
        assert len(tree.images) == len(f.images)


def test_the_tree_is_rebuilt_when_images_change(tmp_path):
    path = str(tmp_path / "scenes.czi")
    _write(path, SCENES, "")
    with micro_reader.open(path) as f:
        first = f.collection
        assert f.collection is first                              # cached
        f.images = [image.as_axes("tczyx") for image in f.images]
        second = f.collection
        assert second is not first and second.images == f.images
        f.images = f.images[:1]
        assert f.collection.images == f.images


def test_only_full_resolution_images_are_leaves(tmp_path):
    """``resolutions="all"`` lists the levels in ``images``; the tree holds
    level 0 only, as ``File.plates`` navigation does."""
    from test_resolutions import _pyramid_tiff
    path = str(tmp_path / "pyramid.ome.tif")
    _pyramid_tiff(path, np.zeros((2, 96, 80), "u2"))
    with micro_reader.open(path, resolutions="all") as f:
        assert len(f.images) > 1
        tree = f.collection
        assert tree.name == "pyramid"                              # .ome.tif dropped (M1-2)
        assert tree.images == [image for image in f.images if image.is_pyramid]
        assert tree.names == ("series0",)


# -- the naming rules and the failures, on a toy file -------------------------------------

def _toy(n, index):
    return Image(n, "yx", (2, 2), "u1", index=index)


def test_images_are_named_by_the_index_values_no_ancestor_holds():
    """``view1_illumination0`` (orthogonal indices never make collections),
    ``image`` when every value is held, ``resolution`` never shown."""
    images = [_toy(0, {"series": 0, "view": 0, "illumination": 0, "resolution": 0}),
              _toy(1, {"series": 0, "view": 1, "illumination": 0, "resolution": 0}),
              _toy(2, {"series": 1, "resolution": 0}),
              _toy(3, {})]
    tree = File("dir/views.xml", images).collection
    assert tree.name == "views"
    assert _shape(tree) == [("series0", "series", 0, ["view0_illumination0",
                                                      "view1_illumination0"]),
                            "series1", "image"]
    assert image_name(images[2], ("series",)) == "image"
    assert image_name(images[0]) == "series0_view0_illumination0"


def test_what_a_collection_holds_follows_from_its_kind():
    leaf = _toy(0, {"series": 0, "tile": 2})
    assert Collection("x", "series", 0, [leaf]).names == ("tile2",)
    assert Collection("x", "tile", 2, [leaf]).names == ("series0",)


def test_a_plate_image_without_a_plate_or_beyond_the_plates_raises():
    plate = Plate("P", ("A",), ("1",), (Well(0, 0, "A1"),))
    with pytest.raises(InvariantError, match="has a well but no plate"):
        file_collection(File("p.x", [_toy(0, {"well": 0, "field": 0})], plates=[plate]))
    with pytest.raises(InvariantError, match="plate 1 are in no plate of File.plates"):
        file_collection(File("p.x", [_toy(0, {"plate": 1, "well": 0, "field": 0})],
                             plates=[plate]))
    with pytest.raises(InvariantError, match="well 3 are in no well of plate 0"):
        file_collection(File("p.x", [_toy(0, {"plate": 0, "well": 3, "field": 0})],
                             plates=[plate]))


def test_check_tree_catches_a_tree_that_disagrees_with_the_index_or_the_images():
    a, b = _toy(0, {"series": 0, "tile": 0}), _toy(1, {"series": 0, "tile": 1})
    good = Collection("f", None, None, [Collection("series0", "series", 0, [a, b])])
    check_tree(good, [a, b])
    with pytest.raises(InvariantError, match="sits in series 1"):
        check_tree(Collection("f", None, None, [Collection("s", "series", 1, [a, b])]), [a, b])
    with pytest.raises(InvariantError, match="not in File.images"):
        check_tree(good, [a])
    with pytest.raises(InvariantError, match="twice"):
        check_tree(Collection("f", None, None, [a, a, b]), [a, b])
    with pytest.raises(InvariantError, match="missing from the tree"):
        check_tree(Collection("f", None, None, [a]), [a, b])
