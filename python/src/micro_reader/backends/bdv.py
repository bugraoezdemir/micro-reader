"""BigDataViewer / BigStitcher (``[hdf5]`` extra, h5py): an XML file
(``SpimData``) describing the views, and an HDF5 file holding the pixels.

XML (the main file; it names the HDF5 file)::

    SpimData/BasePath                                  folder the paths are relative to
    SpimData/SequenceDescription/ImageLoader[@format="bdv.hdf5"]/hdf5   the HDF5 file
        .../ImageLoader/partition/path                 partitioned: one file per part
    SpimData/SequenceDescription/ViewSetups/ViewSetup  id, name, size (x y z),
        voxelSize (unit, size x y z), attributes (illumination, channel, tile, angle)
    SpimData/SequenceDescription/ViewSetups/Attributes[@name]   id -> name per attribute
    SpimData/SequenceDescription/Timepoints            range / pattern / list
    SpimData/SequenceDescription/MissingViews          views not acquired
    SpimData/ViewRegistrations/ViewRegistration        affine transforms per view

HDF5::

    s00/resolutions          per level, the downsampling factors (x, y, z)
    s00/subdivisions         per level, the chunk shape (x, y, z)
    t00000/s00/0/cells       one view at one time point and level, (z, y, x)

Images: one per tile, angle and illumination (indices ``tile``, ``view``,
``illumination``, each only when it takes several values), its channels the
``c`` axis and its time points the ``t`` axis: ``tczyx``.  Stored levels are
the image's ``resolutions``.  BigDataViewer stores unsigned 16-bit pixels as
signed 16-bit HDF5 data; they are read as ``uint16``.

Opening the HDF5 file opens the XML next to it with the same name, when that
XML names the HDF5 file (BigDataViewer's and BigStitcher's own naming); a
partition file ``<name>-<tt>-<ss>.h5`` opens the XML ``<name>.xml`` beside it
that lists it, the whole dataset.  An HDF5 file without such an XML is read alone, with a ``CompositeWarning``:
its view setups, whose meaning only the XML gives, are the ``sequence``
index, without pixel sizes.  Other XMLs naming the file (a second
registration of the same data) are separate documents, read only when opened
themselves; the warning names those in the file's folder.
"""
from __future__ import annotations

import builtins
import os
import re
import warnings
import xml.etree.ElementTree as ET
from typing import Optional

import numpy as np

from .. import _parts
from .._base import File, Image
from .._errors import (CompositeWarning, CorruptFileError, MissingImagesWarning,
                       UnsupportedFormatError)
from .._metadata import Channel, Metadata, Quantity, quantity
from . import _hdf5

NAME = "bdv"
#: group_files: the XML is an index file naming the HDF5 file(s)
SET_INDEX = True
HDF5_SUFFIXES = (".h5", ".hdf5")
#: what an attribute of a view setup becomes (BigStitcher's "angle" is a
#: light-sheet view; channels are the c axis)
ATTRIBUTE_INDEX = {"tile": "tile", "angle": "view", "illumination": "illumination"}
#: XML read whole to find the HDF5 file it names (owner): BigStitcher projects
#: with many registrations are a few MB
MAX_XML_BYTES = 256 << 20


def may_index(name: str) -> bool:
    """``group_files``' cheap test (reads nothing): an XML file."""
    return name.lower().endswith(".xml")


def sniff(head: bytes, path: str) -> bool:
    if path.lower().endswith(".xml"):
        return b"<SpimData" in head
    return _hdf5.is_hdf5(head) and _hdf5.kind(path) == NAME


# -- the HDF5 file opens its XML ------------------------------------------------

#: bdv-core's partition files (Partition.split): ``<name>-<tt>-<ss>.h5`` beside the
#: master ``<name>.h5``, one per group of time points and setups
PARTITION = re.compile(r"^(?P<stem>.+)-\d{2,}-\d{2,}\.h5$", re.IGNORECASE)


def _stems(name: str) -> list:
    """The XML names (lower case, without ``.xml``) an HDF5 file *name* may
    belong to: its own stem, and for a partition file the master's stem."""
    lower = name.lower()
    m = PARTITION.match(lower)
    return [os.path.splitext(lower)[0]] + ([m["stem"]] if m else [])


#: BigDataViewer's viewer settings beside a dataset XML: ``<name>.settings.xml``
SETTINGS = ".settings.xml"
#: ViewTransform names (lower case) that place a view where the acquisition
#: put it -- nominal, not registered (docs/PROCESSED.md, "Spatial operations"):
#: BigStitcher's calibration and grid / tile-configuration translations.  An
#: unnamed transform (bdv-core export, npy2bdv, Luxendo) counts as nominal;
#: any other name ("Stitching Transform", "Tile ICP Refinement", bead
#: registrations) is a registration and ends the nominal run.
NOMINAL_TRANSFORMS = frozenset({"calibration", "translation to regular grid",
                                "translation from tile configuration"})
#: an HDF5 file read alone: XMLs of its folder checked for naming it, for the
#: warning only (never read from)
MAX_NAMING_XMLS = 32


def may_own(folder: str, name: str, folders) -> bool:
    """``group_files``' cheap test: an HDF5 file with an XML of its name beside
    it (or, a partition file, of its master's name), or a viewer settings file
    beside its dataset XML."""
    lower = name.lower()
    if lower.endswith(SETTINGS):
        return lower[:-len(SETTINGS)] + ".xml" in folders.names(folder)
    return lower.endswith(HDF5_SUFFIXES) and \
        any(stem + ".xml" in folders.names(folder) for stem in _stems(lower))


def _listing(xml: str, path: str) -> bool:
    """Whether the BigDataViewer XML *xml* names the HDF5 file *path* (as its
    file or one of its partition files)."""
    try:
        if os.path.getsize(xml) > MAX_XML_BYTES:
            return False
        with builtins.open(xml, "rb") as fh:
            if b"<SpimData" not in fh.read(_parts.HEAD_BYTES):
                return False
        files = _Xml(xml).pixel_files()
    except (OSError, ET.ParseError, CorruptFileError, UnsupportedFormatError):
        return False
    mine = os.path.normcase(os.path.abspath(path))
    return any(os.path.normcase(os.path.abspath(p)) == mine for p in files)


def naming_xmls(path: str) -> list:
    """The BigDataViewer XMLs in *path*'s folder that name it (other than the
    one of its name, which would have opened it): e.g. a second registration
    of the same data (``kidney_v1.xml``).  For the warning of an HDF5 file read
    alone -- a look to write the message, never a source of metadata (the
    pointing rule, docs/COMPOSITE.md)."""
    folder = os.path.dirname(os.path.abspath(path))
    try:
        names = sorted(e.name for e in os.scandir(folder) if e.is_file()
                       and e.name.lower().endswith(".xml") and not e.name.lower().endswith(SETTINGS))
    except OSError:
        return []
    return [n for n in names[:MAX_NAMING_XMLS] if _listing(os.path.join(folder, n), path)]


def owner(path: str, head: Optional[bytes]) -> Optional[_parts.Part]:
    """An HDF5 file is a part of the BigDataViewer XML of its name that names
    it -- a partition file ``<name>-<tt>-<ss>.h5`` of the XML ``<name>.xml``
    that lists it; ``<name>.settings.xml`` (viewer settings) of the dataset
    XML ``<name>.xml``."""
    if path.lower().endswith(SETTINGS):
        name = os.path.basename(path)[:-len(SETTINGS)] + ".xml"
        xml = _parts.find(os.path.dirname(path), name)
        if xml is None:
            return None
        try:
            with builtins.open(xml, "rb") as fh:
                return _parts.Part(xml, xml) if b"<SpimData" in fh.read(_parts.HEAD_BYTES) else None
        except OSError:
            return None
    if head is None or not path.lower().endswith(HDF5_SUFFIXES) or not _hdf5.is_hdf5(head):
        return None
    folder = os.path.dirname(path)
    for stem in _stems(os.path.basename(path)):
        xml = _parts.find(folder, stem + ".xml")
        if xml is not None and _listing(xml, path):
            return _parts.Part(xml, xml, standalone=True)
    return None


# -- the XML --------------------------------------------------------------------

def _text(node, tag: str, default=None):
    found = node.find(tag) if node is not None else None
    return found.text.strip() if found is not None and found.text else default


def _numbers(text: Optional[str]) -> list:
    return [float(v) for v in (text or "").split()]


def integer_pattern(text: str) -> list:
    """BigDataViewer's integer patterns: "0-9", "1,3,5", "0-20:5"."""
    out = []
    for part in (text or "").replace(" ", "").split(","):
        if not part:
            continue
        m = re.fullmatch(r"(-?\d+)(?:-(-?\d+)(?::(\d+))?)?", part)
        if not m:
            raise CorruptFileError(f"timepoint pattern {text!r}: cannot read {part!r}")
        first = int(m.group(1))
        last = int(m.group(2)) if m.group(2) is not None else first
        out.extend(range(first, last + 1, int(m.group(3) or 1)))
    return out


class _Setup:
    def __init__(self, node, where: str):
        self.id = int(_text(node, "id"))
        self.name = _text(node, "name")
        size = [int(v) for v in _numbers(_text(node, "size"))]
        self.size = tuple(reversed(size)) if len(size) == 3 else None      # z, y, x
        voxel = node.find("voxelSize")
        self.unit = _text(voxel, "unit")
        self.voxel = _numbers(_text(voxel, "size")) if voxel is not None else []   # x, y, z
        attrs = node.find("attributes")
        self.attributes = {child.tag: int(child.text) for child in (attrs if attrs is not None else [])
                           if child.text and child.text.strip().lstrip("-").isdigit()}


class _Xml:
    """What micro-reader needs from a SpimData XML."""

    def __init__(self, path: str):
        self.path = path
        try:
            self.root = ET.parse(path).getroot()
        except ET.ParseError as exc:
            raise CorruptFileError(f"{path!r}: not a readable XML file ({exc})") from exc
        if self.root.tag != "SpimData":
            raise UnsupportedFormatError(f"{path!r}: not a BigDataViewer XML (root {self.root.tag!r})")
        sequence = self.root.find("SequenceDescription")
        if sequence is None:
            raise CorruptFileError(f"{path!r}: SpimData without a SequenceDescription")
        self.sequence = sequence
        self.loader = sequence.find("ImageLoader")
        if self.loader is None:
            raise CorruptFileError(f"{path!r}: SpimData without an ImageLoader")
        self.format = self.loader.get("format", "")
        base = self.root.find("BasePath")
        here = os.path.dirname(os.path.abspath(path))
        if base is not None and base.text and base.text.strip():
            self.base = base.text.strip() if base.get("type") == "absolute" \
                else os.path.normpath(os.path.join(here, base.text.strip()))
        else:
            self.base = here

    def _resolve(self, node) -> str:
        ref = node.text.strip() if node is not None and node.text else ""
        if not ref:
            raise CorruptFileError(f"{self.path!r}: an ImageLoader path is empty")
        if node.get("type") == "absolute" and os.path.exists(ref):
            return ref
        if node.get("type") == "absolute":
            raise CorruptFileError(f"{self.path!r}: the HDF5 file {ref!r} is not found")
        return _parts.resolve(self.base, ref, self.path)

    def pixel_files(self) -> list:
        """The HDF5 file, and the partition files of a partitioned one."""
        if self.format != "bdv.hdf5":
            raise UnsupportedFormatError(
                f"{self.path!r}: BigDataViewer image loader {self.format!r}; micro-reader reads "
                "'bdv.hdf5' (an N5, OME-Zarr or file-list dataset: open its own files)")
        files = [self._resolve(self.loader.find("hdf5"))]
        #: partition files the XML lists that are not there (their views read as zeros)
        self.missing_files = []
        for partition in self.loader.findall("partition"):
            try:
                files.append(self._resolve(partition.find("path")))
            except CorruptFileError:
                self.missing_files.append(_text(partition, "path") or "?")
        return files

    def setups(self) -> list:
        node = self.sequence.find("ViewSetups")
        if node is None:
            raise CorruptFileError(f"{self.path!r}: SpimData without ViewSetups")
        return sorted((_Setup(s, self.path) for s in node.findall("ViewSetup")), key=lambda s: s.id)

    def attribute_names(self) -> dict:
        """{attribute: {id: name}} from the ``Attributes`` blocks."""
        out = {}
        node = self.sequence.find("ViewSetups")
        for block in node.findall("Attributes") if node is not None else []:
            names = {}
            for entry in block:
                ident = _text(entry, "id")
                if ident is not None and ident.lstrip("-").isdigit():
                    names[int(ident)] = _text(entry, "name")
            out[block.get("name")] = names
        return out

    def timepoints(self) -> list:
        node = self.sequence.find("Timepoints")
        if node is None:
            return [0]
        kind = node.get("type", "range")
        if kind == "range":
            return list(range(int(_text(node, "first", "0")), int(_text(node, "last", "0")) + 1))
        if kind == "pattern":
            return integer_pattern(_text(node, "integerpattern", ""))
        if kind == "list":                          # spimdata writes Timepoint; TimePoint seen too
            ids = [int(_text(t, "id")) for t in node.findall("Timepoint") + node.findall("TimePoint")
                   if _text(t, "id")]
            return sorted(ids) or integer_pattern(_text(node, "integerpattern", ""))
        raise UnsupportedFormatError(f"{self.path!r}: Timepoints of type {kind!r}")

    def missing(self) -> set:
        node = self.sequence.find("MissingViews")
        return {(int(m.get("timepoint")), int(m.get("setup")))
                for m in (node.findall("MissingView") if node is not None else [])}

    def transforms(self, timepoint: int, setup: int) -> Optional[list]:
        """The view's transforms as listed: [(name or None, 4 x 4)], the first
        listed applied last; None when the XML has none or one is malformed."""
        regs = self.root.find("ViewRegistrations")
        if regs is None:
            return None
        for reg in regs.findall("ViewRegistration"):
            if int(reg.get("timepoint", -1)) == timepoint and int(reg.get("setup", -1)) == setup:
                out = []
                for vt in reg.findall("ViewTransform"):
                    values = _numbers(_text(vt, "affine"))
                    if len(values) != 12:
                        return None
                    out.append((_text(vt, "Name"),
                                np.vstack([np.reshape(values, (3, 4)), [0, 0, 0, 1]])))
                return out
        return None

    def registration(self, timepoint: int, setup: int) -> Optional[np.ndarray]:
        """The view's NOMINAL placement (pixel -> global coordinates), 4 x 4:
        the run of nominal transforms at the end of the list (applied first),
        composed.  What is listed before it (stitching, ICP, bead
        registration) is a registration estimated from the content: not a
        position (pending history, docs/PROCESSED.md).  None when the XML has
        no transforms for the view, or places it nowhere: a BigStitcher
        "calibration" alone, without a translation, only scales."""
        listed = self.transforms(timepoint, setup)
        if listed is None:
            return None
        nominal, placed = np.eye(4), False
        for name, matrix in reversed(listed):
            if name is not None and name.lower() not in NOMINAL_TRANSFORMS:
                break
            nominal = matrix @ nominal
            placed = (placed or name is None or name.lower() != "calibration"
                      or bool(np.any(matrix[:3, 3])))
        return nominal if placed else None


# -- images ---------------------------------------------------------------------

def _cells(timepoint: int, setup: int, level: int) -> str:
    return f"t{timepoint:05d}/s{setup:02d}/{level}/cells"


class _BdvImage(Image):
    """One tile / view / illumination at one level: time points x channels."""

    def __init__(self, position: int, name: str, index: dict, datasets: list, shape: tuple,
                 dtype, view, reader: _hdf5.ChunkReader, key, metadata_of, members: set):
        self._datasets = datasets               # [t][c] -> h5py dataset or None (missing)
        self._view = view
        self._reader = reader
        self._key = key
        self._metadata_of = metadata_of
        self._members = members
        self._closed = False
        present = next((ds for row in datasets for ds in row if ds is not None), None)
        chunks = present.chunks if present is not None else None
        super().__init__(position, "tczyx", shape, dtype, name=name,
                         read_unit=(1, 1, *chunks) if chunks else None, index=index)

    def _read_block(self, block):
        if self._closed:
            raise ValueError("this BigDataViewer image has been closed: its file was closed "
                             "(File.close() or the end of a with block)")
        ts, cs, zs, ys, xs = block
        out = np.zeros(tuple(s.stop - s.start for s in block), dtype=self.dtype)
        read, jobs = 0, []
        for t in range(ts.start, ts.stop):
            for c in range(cs.start, cs.stop):
                ds = self._datasets[t][c]
                if ds is None:
                    continue                     # a missing view: zeros
                read += self._reader.plan((self._key, t, c), ds, (zs, ys, xs),
                                          out[t - ts.start, c - cs.start], jobs, view=self._view)
        return out, read + self._reader.run(jobs)

    def _read_metadata(self):
        return self._metadata_of(self)

    def _member_paths(self) -> set:
        return set(self._members)

    def close(self) -> None:
        self._closed = True


def _pixel_type(datasets, declared: set, where: str):
    """(dtype the image has, view to apply to the stored values or None).
    bdv-core (Hdf5ImageLoader): a setup group's ``dataType`` attribute names the
    type; without it the file is of the original format, which stores
    unsigned 16-bit pixels as signed 16-bit data."""
    stored = {ds.dtype for ds in datasets}
    if len(stored) != 1 or len(declared) > 1:
        raise UnsupportedFormatError(f"{where}: views of one image differ in type "
                                     f"{sorted(map(str, stored | declared))}")
    dtype = stored.pop()
    name = next(iter(declared), None)
    if name is not None:
        try:
            want = np.dtype(name)
        except TypeError:
            raise UnsupportedFormatError(f"{where}: BigDataViewer dataType {name!r}") from None
        if want.itemsize != dtype.itemsize or want.kind not in "iuf":
            raise CorruptFileError(f"{where}: dataType {name!r} but the data is {dtype}")
        if want.kind == dtype.kind:
            return dtype.newbyteorder("="), None
        view = want.newbyteorder(dtype.byteorder)
        return view.newbyteorder("="), view
    if dtype.kind == "i" and dtype.itemsize == 2:             # original format: uint16 as int16
        view = _hdf5.unsigned(dtype)
        return view.newbyteorder("="), view
    return dtype.newbyteorder("="), None


def _data_type(h5, setup: int) -> Optional[str]:
    """A setup's ``dataType`` attribute (bdv-core 10.4.6 and later), or None."""
    group = h5.get(f"s{setup:02d}")
    return _hdf5.attr_text(group.attrs, "dataType") if group is not None else None


def _dataset(h5, name: str):
    """The dataset *name*, or None when it is not there -- also when it is an
    external link (partitioned files) to a file that is missing."""
    try:
        return h5[name] if name in h5 else None
    except (KeyError, OSError):
        return None


def _levels(h5, setup: int) -> int:
    name = f"s{setup:02d}/resolutions"
    return int(h5[name].shape[0]) if name in h5 else 1


class _Group:
    """The view setups of one image: one per channel."""

    def __init__(self, key: tuple, setups: list):
        self.key = key
        self.setups = setups


def _build(h5, path: str, xml: Optional[_Xml], resolutions: str, tiles: str,
           reader: _hdf5.ChunkReader, members: set) -> list:
    """The images of a BigDataViewer dataset (``xml`` None: the HDF5 file alone)."""
    where = os.path.basename(xml.path if xml else path)
    if xml is not None:
        setups = xml.setups()
        timepoints = xml.timepoints()
        missing = xml.missing()
        names = xml.attribute_names()
    else:
        # a setup behind a broken link is no setup (h5py: KeyError), as in classify
        ids = sorted(int(n[1:]) for n in h5 if _hdf5.SETUP.match(n)
                     and hasattr(h5.get(n), "keys") and "resolutions" in h5.get(n))
        setups = []
        for i in ids:
            s = _Setup.__new__(_Setup)
            s.id, s.name, s.size, s.unit, s.voxel = i, None, None, None, []
            s.attributes = {"sequence": i}
            setups.append(s)
        timepoints = sorted(int(n[1:]) for n in h5 if _hdf5.TIMEPOINT.match(n))
        missing, names = set(), {}
    if not setups or not timepoints:
        raise CorruptFileError(f"{where}: no view setups or no time points")

    # images: setups grouped by their non-channel attributes; channels stacked
    if xml is not None:
        dims = [a for a in ATTRIBUTE_INDEX if len({s.attributes.get(a, 0) for s in setups}) > 1]
        groups: dict = {}
        for s in setups:
            groups.setdefault(tuple(s.attributes.get(a, 0) for a in dims), []).append(s)
        for members_ in groups.values():
            members_.sort(key=lambda s: (s.attributes.get("channel", 0), s.id))
        if any(len({s.attributes.get("channel", 0) for s in m}) != len(m) for m in groups.values()):
            # setups that no attribute micro-reader knows tells apart (an old XML
            # without attributes: setups named "Angle 0", "Angle 60" -- a name is
            # no evidence): each setup is an image of its own, index "sequence"
            for s in setups:
                s.attributes["sequence"] = s.id
            dims = dims + ["sequence"]
            groups = {}
            for s in setups:
                groups.setdefault(tuple(s.attributes.get(a, 0) for a in dims), []).append(s)
        index_names = [ATTRIBUTE_INDEX.get(a, a) for a in dims]
        ranks = {a: sorted({s.attributes.get(a, 0) for s in setups}) for a in dims}
    else:
        dims, index_names, ranks = ["sequence"], ["sequence"], {"sequence": [s.id for s in setups]}
        groups = {(s.id,): [s] for s in setups}
    if tiles == "mosaic" and "tile" in index_names:
        raise UnsupportedFormatError(f"{where}: tiles='mosaic': micro-reader does not compose "
                                     "BigDataViewer tiles (each tile is its own image)")

    absent = []
    images_by_level = []
    for g, (gkey, group) in enumerate(sorted(groups.items())):
        index = {"series": 0}
        index.update({name: ranks[a].index(v) for name, a, v in zip(index_names, dims, gkey)})
        n_levels = {_levels(h5, s.id) for s in group}
        if len(n_levels) != 1:
            raise UnsupportedFormatError(f"{where}: the channels of {index} store "
                                         f"different numbers of levels {sorted(n_levels)}")
        pyramid = []
        for level in range(n_levels.pop()):
            datasets = []
            for tp in timepoints:
                row = []
                for s in group:
                    name = _cells(tp, s.id, level)
                    ds = _dataset(h5, name) if (tp, s.id) not in missing else None
                    if ds is None and level == 0 and (tp, s.id) not in missing:
                        absent.append(name)
                    row.append(ds)
                datasets.append(row)
            present = [ds for row in datasets for ds in row if ds is not None]
            if not present:
                break                            # nothing stored at this level
            shapes = {ds.shape for ds in present}
            if len(shapes) != 1 or len(next(iter(shapes))) != 3:
                raise UnsupportedFormatError(f"{where}: the views of {index} at level {level} "
                                             f"have different shapes {sorted(shapes)}")
            zyx = shapes.pop()
            if level == 0 and xml is not None:
                sizes = {s.size for s in group if s.size}
                if sizes and sizes != {zyx}:
                    raise CorruptFileError(f"{where}: view setup size {sorted(sizes)} (z, y, x) "
                                           f"but the HDF5 data is {zyx}")
            declared = {d for d in (_data_type(h5, s.id) for s in group) if d}
            dtype, view = _pixel_type(present, declared, where)
            label = " ".join(f"{a} {v}" for a, v in zip(dims, gkey))
            name = f"{where} {label}".strip()
            level_index = dict(index, resolution=level) if (level or resolutions == "all") else dict(index)
            image = _BdvImage(0, name, level_index, datasets, (len(timepoints), len(group), *zyx),
                              dtype, view, reader, (g, level),
                              lambda im, group=group: _metadata(xml, group, timepoints, names, im),
                              members)
            pyramid.append(image)
        if not pyramid:
            continue
        for image in pyramid:
            image._resolutions = pyramid
        images_by_level.append(pyramid)
    lost = getattr(xml, "missing_files", [])
    lost_text = (f"; {len(lost)} of the partition files the XML lists are missing (e.g. "
                 f"{lost[0]!r})" if lost else "")
    if absent and images_by_level:
        warnings.warn(MissingImagesWarning(
            f"{where}: {len(absent)} views are neither stored nor listed as missing (e.g. "
            f"{absent[0]!r}): read as zeros{lost_text}"), stacklevel=4)
    if not images_by_level:
        raise CorruptFileError(f"{where}: no view is stored in the HDF5 file{lost_text}")
    listed = resolutions == "all"
    images = [im for pyramid in images_by_level for im in (pyramid if listed else pyramid[:1])]
    for n, image in enumerate(images):
        image._position = n
    return images


def _metadata(xml: Optional[_Xml], group: list, timepoints: list, names: dict, image) -> Metadata:
    """Pixel sizes (the first channel's voxel size), channel names (the
    ``channel`` attribute's names) and the position of the first pixel, from
    the nominal placement (``_Xml.registration``: stage / grid, not a
    stitching or registration result) when it only scales and shifts."""
    n_channels = len(group)
    if xml is None:
        return Metadata.unknown(n_channels)
    first = group[0]
    sizes = {}
    for axis, value in zip("xyz", first.voxel):
        q = quantity(value, first.unit)
        if q is not None:
            sizes[axis] = q
    channel_names = names.get("channel", {})
    channels = tuple(Channel(name=channel_names.get(s.attributes.get("channel")))
                     for s in group)
    position = {}
    matrix = xml.registration(timepoints[0], first.id)
    if matrix is not None and sizes:
        linear = matrix[:3, :3]
        diagonal = np.diag(linear)
        if np.allclose(linear, np.diag(diagonal)) and np.all(diagonal > 0):
            for k, axis in enumerate("xyz"):
                q = sizes.get(axis)
                if q is not None:
                    position[axis] = Quantity(float(matrix[k, 3] * q.value / diagonal[k]), q.unit)
    return Metadata(pixel_sizes=sizes, position=position, channels=channels)


class _BdvFile(File):
    backend = NAME

    def __init__(self, path, images, h5, reader, fmt):
        self.format = fmt
        super().__init__(path, images)
        self._h5 = h5
        self._reader = reader

    def close(self) -> None:
        super().close()
        self._reader.close()
        self._h5.close()


def open(path: str, tiles: str = "separate", resolutions: str = "base") -> _BdvFile:  # noqa: A001 - backend protocol
    h5py = _hdf5.h5py_module("reading BigDataViewer files")
    xml = None
    if path.lower().endswith(".xml"):
        xml = _Xml(path)
        files = xml.pixel_files()
        h5_path = files[0]
        members = set(files)
    else:
        h5_path, members = path, set()
        others = naming_xmls(path)
        hint = (f"open one of the XMLs naming it, {', '.join(repr(n) for n in others)}, for them"
                if others else "open the dataset's .xml for them")
        warnings.warn(CompositeWarning(
            f"{path!r}: a BigDataViewer HDF5 file without the XML of its name: its view "
            f"setups are read as the 'sequence' index, without pixel sizes or channel names "
            f"({hint})"), stacklevel=4)
    try:
        h5 = h5py.File(h5_path, "r")
    except OSError as exc:
        raise CorruptFileError(f"{h5_path!r}: {exc}") from exc
    reader = _hdf5.ChunkReader(h5_path)
    try:
        images = _build(h5, path, xml, resolutions, tiles, reader, members)
        if not images:
            raise CorruptFileError(f"{path!r}: no view is stored in {h5_path!r}")
    except BaseException:
        reader.close()
        h5.close()
        raise
    return _BdvFile(path, images, h5, reader, "bdv-xml" if xml is not None else "bdv-hdf5")
