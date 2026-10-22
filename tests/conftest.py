"""Session-wide checks shared by every test module."""
import weakref

import pytest

#: backends whose images carry a position without a coordinate system today:
#: the spatial check below tolerates that for them (plan I3 cannot hold before
#: a backend declares its systems).  Each backend task of plan 9.7 removes its
#: backend's name (``File.backend``) once the backend declares the file's
#: systems and sets ``metadata.coordinate_system`` beside every position; the
#: check is strict for every backend not listed.  Companion sets opened
#: through a member are the "tiff" backend.
POSITIONS_WITHOUT_A_SYSTEM = {"ndpis"}


def strict_placement(f) -> bool:
    """Whether a position without a system fails the run for file *f*."""
    return f.backend not in POSITIONS_WITHOUT_A_SYSTEM


@pytest.fixture(autouse=True, scope="session")
def every_index_is_an_image_index():
    """Every image ``micro_reader.open`` gives out in the whole session -- its
    images, their resolution levels and parts, also with ``axes=``,
    ``samples=`` and ``resolutions="all"`` -- has an ``ImageIndex`` (the index
    guard, plan 3.24): a backend that stores a plain dict bypasses the guard."""
    import micro_reader
    from micro_reader._base import ImageIndex

    real_open = micro_reader.open
    bad, checked = [], [0]

    def checking_open(*args, **kwargs):
        f = real_open(*args, **kwargs)
        for image in list(f.images) + f._every_image():
            checked[0] += 1
            if type(image.index) is not ImageIndex:
                bad.append(f"{f.path}: {image.name!r} has a {type(image.index).__name__}")
        return f

    micro_reader.open = checking_open
    try:
        yield
    finally:
        micro_reader.open = real_open
    assert not bad, "images without an ImageIndex:\n" + "\n".join(bad[:20])


@pytest.fixture(autouse=True, scope="session")
def every_file_keeps_the_spatial_invariants():
    """Every file ``micro_reader.open`` gives out in the whole session keeps
    the spatial model's invariants (plan 3.22, ``micro_reader._spatial``):

    - declaring ``coordinate_systems``, ``coordinate_transformations`` and
      ``collection`` at open reads no image's metadata (a count of first
      metadata reads stays put: the backend hook may use the indices, the
      plates and the file's header, never ``_read_metadata``);
    - the tree (``collection``) agrees with the indices and holds exactly
      the full-resolution objects of ``images`` (I7, ``check_tree``);
    - I1 and I5 hold for what the file declares (checked inside the
      properties: a violation raises at open);
    - every image metadata read later in the session (levels and parts too)
      keeps I2 and I3: ``coordinate_system`` is a declared, non-estimated
      system, and set exactly when ``position`` is not empty.  For the
      backends in ``POSITIONS_WITHOUT_A_SYSTEM`` a position without a system
      is tolerated until their 9.7 task lands (and removes them from it).
      Images whose metadata is already there when ``open`` returns (a backend
      that fills ``_metadata`` while opening, e.g. mrc) are checked at open.

    Failures are collected and reported once, at the end of the session."""
    import micro_reader
    from micro_reader._base import Image
    from micro_reader._errors import InvariantError
    from micro_reader._spatial import check_placement, check_tree

    real_open, real_metadata = micro_reader.open, Image.metadata
    bad, reads = [], [0]
    declared = weakref.WeakKeyDictionary()      # native image -> (path, strict, systems)

    def check(image, meta, info):
        path, strict, systems = info
        try:
            check_placement(meta, systems, where=f"{path}: {image.name!r}",
                            require_system=strict)
        except InvariantError as exc:
            bad.append(str(exc))

    def counting_metadata(self):
        first = self._metadata is None
        meta = real_metadata.fget(self)
        if first:
            reads[0] += 1
            info = declared.get(self)
            if info is not None:
                check(self, meta, info)
        return meta

    def checking_open(*args, **kwargs):
        f = real_open(*args, **kwargs)
        before = reads[0]
        systems = f.coordinate_systems                  # I1, I5: raise at open
        f.coordinate_transformations
        tree = f.collection
        if reads[0] != before:
            bad.append(f"{f.path}: declaring the coordinate systems or the tree read the "
                       f"metadata of {reads[0] - before} image(s) (plan 4.2: a backend "
                       "declares them from what opening already read)")
        try:
            check_tree(tree, f.images, where=f.path)    # I7: tree and index agree (3.8)
        except InvariantError as exc:
            bad.append(str(exc))
        for image in f._every_image():
            declared[image] = info = (f.path, strict_placement(f), systems)
            if image._metadata is not None:             # read while opening: no first read
                check(image, image._metadata, info)     # comes later, so check it now
        return f

    Image.metadata = property(counting_metadata)
    micro_reader.open = checking_open
    try:
        yield
    finally:
        micro_reader.open = real_open
        Image.metadata = real_metadata
    assert not bad, "spatial invariants broken:\n" + "\n".join(bad[:20])


#: the formats micro-reader places nowhere (plan 4.12): no position they
#: record is verified, so none is given (MetaMorph's and Micro-Manager's
#: ``stage_position`` stays as recorded, 3.27).  Keyed by (``File.backend``,
#: ``File.format``).  A file of one of these declares no coordinate system and
#: no transformation, and every image it gives out has ``position == {}`` and
#: ``coordinate_system`` None.  A task that places one of them -- after
#: checking its positions on real files -- removes the format's pair.
UNPLACED_FORMATS = {
    # the plan's list (4.12)
    ("metamorph", "metamorph"), ("incell", "incell"), ("oif", "oif"), ("oif", "oib"),
    ("nd2", "nd2-legacy"), ("oir", "oir"), ("vsi", "vsi"), ("vsi", "ets"),
    ("tiff", "bif"), ("tiff", "philips"), ("dv", "dv"), ("mrc", "mrc"),
    ("tiff", "tiff"), ("tiff", "bigtiff"),
    ("hdf5", "luxendo"), ("hdf5", "fiji-hdf5"), ("ilastik", "ilastik"),
    ("metaxpress", "metaxpress"),
    # no position either, and no row of 4.3-4.11 places them
    ("ics", "ics"), ("micromanager", "micromanager-files"),
    ("raster", "png"), ("raster", "jpeg"), ("raster", "bmp"),
    ("tiff", "imagej"), ("tiff", "lsm"), ("tiff", "stk"), ("tiff", "eer"), ("tiff", "ndpi"),
    ("tiff", "ndtiff"), ("tiff", "micromanager"), ("tiff", "scanimage"),
}


@pytest.fixture(autouse=True, scope="session")
def unplaced_formats_stay_unplaced():
    """Every file ``micro_reader.open`` gives out in the whole session in one
    of ``UNPLACED_FORMATS`` (plan 4.12) declares no coordinate system and no
    transformation, and every image metadata read from it later (levels and
    parts too) has ``position == {}`` and ``coordinate_system`` None.  A
    position appearing for one of these formats fails the run until a task
    has checked it and removed the format's pair.  Failures are collected and
    reported once, at the end of the session."""
    import micro_reader
    from micro_reader._base import Image

    real_open, real_metadata = micro_reader.open, Image.metadata
    bad = []
    unplaced = weakref.WeakKeyDictionary()      # native image -> (path, (backend, format))

    def checking_metadata(self):
        first = self._metadata is None
        meta = real_metadata.fget(self)
        if first and self in unplaced:
            path, pair = unplaced[self]
            if meta.position or meta.coordinate_system is not None:
                bad.append(f"{path}: {self.name!r} ({pair[0]}, {pair[1]}) has position "
                           f"{sorted(meta.position)} in {meta.coordinate_system!r} (plan 4.12: "
                           "this format is placed nowhere; verify its positions, then remove "
                           "it from UNPLACED_FORMATS)")
        return meta

    def checking_open(*args, **kwargs):
        f = real_open(*args, **kwargs)
        pair = (f.backend, f.format)
        if pair in UNPLACED_FORMATS:
            if f.coordinate_systems or f.coordinate_transformations:
                bad.append(f"{f.path}: ({pair[0]}, {pair[1]}) declares "
                           f"{[s.id for s in f.coordinate_systems]} and "
                           f"{len(f.coordinate_transformations)} transformation(s) (plan 4.12: "
                           "this format is placed nowhere)")
            for image in f._every_image():
                unplaced[image] = (f.path, pair)
        return f

    Image.metadata = property(checking_metadata)
    micro_reader.open = checking_open
    try:
        yield
    finally:
        micro_reader.open = real_open
        Image.metadata = real_metadata
    assert not bad, "unplaced formats placed:\n" + "\n".join(bad[:20])
