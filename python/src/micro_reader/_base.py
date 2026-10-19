"""The interface every backend implements: a File holding Images.

Each image is one spatial frame of the file; its index says which (scene,
tile, rotation, view: see docs/AXES.md)."""
from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np

from ._index import normalise_index


@dataclass
class ReadStats:
    """What an image has read so far; the over-read check of the test harness.

    ``pixels_returned`` counts the contiguous spans requested (before any step
    subsampling); ``pixels_read`` what the backend actually had to read or
    decode.  A backend that reads exact rectangles keeps them equal.
    """
    reads: int = 0
    pixels_returned: int = 0
    pixels_read: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False,
                                  compare=False)

    @property
    def over_read(self) -> float:
        return (self.pixels_read / self.pixels_returned
                if self.pixels_returned else 1.0)

    def add(self, returned: int, read: int) -> None:
        with self._lock:
            self.reads += 1
            self.pixels_returned += returned
            self.pixels_read += read

    def reset(self) -> None:
        with self._lock:
            self.reads = self.pixels_returned = self.pixels_read = 0


class Image:
    """One image of a file (one spatial frame), read by rectangle.

    Backends implement ``_read_block``; everything numpy-like -- integer
    indices, steps, Ellipsis, the keyword form of ``read`` -- is handled here.
    Results are always native-endian and never views into file-backed memory.
    """

    def __init__(self, position: int, axes: str, shape: Sequence[int], dtype,
                 name: Optional[str] = None,
                 read_unit: Optional[Sequence[int]] = None,
                 index: Optional[dict] = None):
        if len(axes) != len(shape):
            raise ValueError(f"axes {axes!r} do not match shape {tuple(shape)}")
        #: place in the file's ``images`` (internal; ``f.images.index(image)``)
        self._position = int(position)
        self.axes = axes.lower()
        self.shape = tuple(int(n) for n in shape)
        self.dtype = np.dtype(dtype).newbyteorder("=")
        self.name = name if name is not None else f"image{position}"
        #: which image of the file: {"scene": .., "tile": .., "rotation": ..,
        #: "view": .., "resolution": ..} -- only the indices the file has
        #: (plates: "well", "field", ... instead of "scene"; see INDEX_NAMES)
        self.index = dict(index) if index is not None else {"scene": self._position}
        self._resolutions: Optional[list] = None
        #: a ragged image's rectangular parts / a part's whole (see ``parts``)
        self._parts: Optional[list] = None
        self._whole: Optional["Image"] = None
        self._metadata = None           # backends set a _metadata.Metadata
        #: None: any rectangle is read exactly.  Otherwise the smallest block
        #: the backend reads at once (e.g. a compressed tile or strip).
        self.read_unit = None if read_unit is None else tuple(int(n) for n in read_unit)
        self.stats = ReadStats()

    @property
    def ndim(self) -> int:
        return len(self.shape)

    @property
    def metadata(self):
        """Pixel and channel metadata (docs/METADATA.md): pixel sizes, stage
        position, channels (name, colour, wavelengths), essential axis
        values.  What the file does not say is absent / None.  Read on first
        access; a resolution level's is level 0's with its pixel sizes scaled
        by the shape ratio."""
        if self._metadata is None:
            from ._metadata import Metadata
            base = self._resolutions[0] if self._resolutions else self
            if base is not self:
                factors = {a: base.shape[base.axes.index(a)] / self.shape[self.axes.index(a)]
                           for a in "xyz" if a in self.axes and a in base.axes}
                self._metadata = base.metadata.scaled(factors)
            else:
                n = self.shape[self.axes.index("c")] if "c" in self.axes else 1
                self._metadata = self._read_metadata() or Metadata.unknown(n)
        return self._metadata

    def _read_metadata(self):
        """This image's ``_metadata.Metadata`` from the file, or None when the
        backend reads none (backends override it)."""
        return None

    @property
    def is_pyramidal(self) -> bool:
        """Whether the file stores lower resolution levels of this image
        (``resolutions`` holds more than the image itself)."""
        return len(self.resolutions) > 1

    # An image is either a pyramid (the logical image: level 0, also when the
    # file stores no lower levels) or a lower resolution level of one.

    @property
    def is_pyramid(self) -> bool:
        """Whether this image is a pyramid, the image itself at full
        resolution, rather than a lower resolution level of one.  True for
        every image without lower levels too."""
        return self.pyramid is self

    @property
    def level(self) -> int:
        """This image's resolution level in its pyramid: 0 for the pyramid
        itself (full resolution), 1, 2, ... for lower levels."""
        for n, member in enumerate(self.resolutions):
            if member is self:
                return n
        raise RuntimeError(f"{self.name!r} is not in its own pyramid")

    @property
    def pyramid(self) -> "Image":
        """The pyramid this image belongs to: itself at level 0, the
        full-resolution image for a lower level."""
        return self.resolutions[0]

    # A ragged image: its channels do not share one sampling (e.g. one channel
    # a z-stack, the others single planes; docs/AXES.md).  It is a whole made of
    # rectangular parts, each an ordinary lazy Image with its own z / t and
    # metadata.  The whole reads as one array, padded with zeros, when the
    # parts lie on a common grid; otherwise reading it raises.

    @property
    def parts(self) -> list:
        """This image's rectangular parts, channel groups that share one
        sampling: several for a ragged image, just this image otherwise."""
        return list(self._parts) if self._parts else [self]

    @property
    def is_ragged(self) -> bool:
        """Whether this image's channels do not share one sampling
        (``parts`` holds more than one part)."""
        return len(self.parts) > 1

    @property
    def whole(self) -> "Image":
        """The image this part belongs to: itself for a whole (or an image
        that is not ragged)."""
        return self._whole if self._whole is not None else self

    @property
    def resolutions(self) -> list:
        """This image's pyramid, full resolution first: one image per
        resolution level the file stores (just this image without levels)."""
        return list(self._resolutions) if self._resolutions else [self]

    @property
    def nbytes(self) -> int:
        return int(np.prod(self.shape)) * self.dtype.itemsize

    def __getitem__(self, key) -> np.ndarray:
        spans, squeeze, steps = normalise_index(key, self.shape)
        block = tuple(slice(a, b) for a, b in spans)
        requested = int(np.prod([b - a for a, b in spans]))
        if requested == 0:
            out = np.empty(tuple(b - a for a, b in spans), dtype=self.dtype)
            pixels_read = 0
        else:
            out, pixels_read = self._read_block(block)
        self.stats.add(requested, pixels_read)
        if any(step != 1 for step in steps):
            out = out[tuple(slice(None, None, step) for step in steps)]
        if squeeze:
            out = out.squeeze(axis=squeeze)
        return out

    def read(self, **selection) -> np.ndarray:
        """Keyword form: ``image.read(z=3, y=slice(0, 512))``; omitted axes: all."""
        unknown = set(selection) - set(self.axes)
        if unknown:
            raise KeyError(f"unknown axes {sorted(unknown)}; this image has {self.axes!r} "
                           "(open(path, axes='canonical') or image.as_axes('canonical') "
                           "adds the missing standard axes with size 1)")
        return self[tuple(selection.get(ax, slice(None)) for ax in self.axes)]

    def as_axes(self, axes: Optional[str] = None, *, samples: str = "keep",
                squeeze=False) -> "Image":
        """This image presented with *axes*: ``None`` keeps its order,
        ``"canonical"`` is ``tczyx`` (+ ``s``), or explicit letters such as
        ``"zyx"``.  Missing axes are inserted with size 1, the others
        reordered; reads stay exact.  ``samples="channels"`` folds RGB
        samples into the channels (C x S); ``squeeze=True`` drops size-1 axes
        except y / x.  An axis *axes* cannot hold is an error that names it.
        The native image is ``.source`` of the result."""
        from ._axes import AxesImage
        return AxesImage(self, axes, samples=samples, squeeze=squeeze)

    def squeeze(self, axes: Optional[str] = None) -> "Image":
        """This image without its size-1 axes: all but ``y`` / ``x``, or just
        those in *axes* (each must have size 1).  They also leave ``axes``."""
        from ._axes import AxesImage
        return AxesImage(self, None, squeeze=axes or True)

    def _read_block(self, block: tuple[slice, ...]) -> tuple[np.ndarray, int]:
        """Read contiguous unit-step spans -> (array of exactly that shape,
        number of pixels actually read or decoded to produce it)."""
        raise NotImplementedError

    def close(self) -> None:
        """Release file handles; the image is unusable afterwards."""

    def __repr__(self) -> str:
        unit = "any rectangle" if self.read_unit is None else f"unit {self.read_unit}"
        return (f"<micro_reader.Image {self.name!r} index={self.index} axes={self.axes!r} "
                f"shape={self.shape} dtype={self.dtype} ({unit})>")


#: the indices that say which image of a file an image is, outermost first.
#: Arrays have only t c z y x (+ s); every other dimension is an index:
#: plate, acquisition (imaging run), well and field for plates (docs/HCS.md;
#: on a plate, well and field take the place of scene), illumination
#: (light-sheet sides), phase, block (CZI), emission / excitation (spectral
#: scans; the wavelength in the image's metadata), lifetime (FLIM bins) and
#: sequence (a dimension the file does not name; open(...,
#: rename={"sequence": "z"}) makes it an axis)
INDEX_NAMES = ("plate", "acquisition", "well", "field",
               "scene", "tile", "rotation", "view", "illumination", "phase", "block",
               "emission", "excitation", "lifetime", "sequence", "part", "resolution")
#: the indices of plates (docs/HCS.md), on every image of a plate;
#: ``plate`` and ``acquisition`` are 0 when a file holds one of each
PLATE_INDICES = ("plate", "acquisition", "well", "field")
#: the indices that are not frames of the image but dimensions sampled at the
#: same points (what other readers keep as array axes)
SAMPLED_INDICES = ("illumination", "phase", "block", "emission", "excitation", "lifetime",
                   "sequence")


def _counted(result, bad: list) -> int:
    """One member checked (``File.check_members``): *result* is 1, or the
    text saying why it does not fit (kept in *bad*)."""
    if result != 1:
        bad.append(result)
    return 1


class File:
    """An opened file: its images, the backend that reads them (``backend``:
    "tiff", "czi", "nd2", "lif", "oir", "oif", "vsi", "ims", "dv", "mrc", "raster") and the
    format variant it is (``format``: e.g. "ome-tiff", "imagej", "lsm",
    "nd2-legacy", "lof", "xlef", "png").  Every format opens as a File; a
    backend's own class is an implementation detail."""

    backend: str = "base"

    def __init__(self, path: str, images: Sequence[Image], plates: Sequence = ()):
        #: the file read: for a composite format its main file (docs/COMPOSITE.md)
        self.path = str(path)
        #: the path ``open()`` was given: a part of a composite, a folder, or ``path``
        self.opened = self.path
        self.images = list(images)
        #: the plates the images belong to (``_plate.Plate``), like ``images``
        #: a list; empty for a file that is not a plate (docs/HCS.md)
        self.plates = list(plates)
        if not getattr(self, "format", None):
            #: the format variant; the backend's name unless it reads several
            self.format = self.backend
        self._index_resolutions()
        self._bind_plates()

    def _bind_plates(self) -> None:
        """Fill in each plate's wells -> fields -> images from ``images``
        (again whenever ``images`` is replaced, e.g. by ``open(axes=...)``)."""
        if self.plates:
            from ._plate import navigate
            self.plates = navigate(self.plates, self.images)

    def _index_resolutions(self) -> None:
        """``resolution`` is in every index when the file stores a pyramid of
        any of its images (0 = full resolution, also for an image without
        lower levels), and in none otherwise: an index holds the dimensions
        the file has.  The same with resolutions="base" or "all"."""
        members, seen = [], set()
        for image in self.images:
            for member in image.resolutions:
                if id(member) not in seen:
                    seen.add(id(member))
                    members.append(member)
        pyramidal = any(image.is_pyramidal for image in self.images)
        for member in members:
            index = {k: v for k, v in member.index.items() if k != "resolution"}
            if pyramidal:
                index["resolution"] = member.index.get("resolution", 0)
            member.index = index

    @property
    def indices(self) -> list:
        """Every image's index, in the order of ``images``."""
        return [image.index for image in self.images]

    @property
    def index_range(self) -> dict:
        """The widest range of each index in the file: how many distinct
        values it takes, e.g. ``{"scene": 2, "tile": 5}``.  Not a total and
        not a shape: with 3 tiles in scene 0 and 5 in scene 1, tile indices
        run 0-2 and 0-4, so the range is 5 (not 8, and scene 0 has only 3).
        Tile, rotation, view, well and field indices count from 0, so the
        values are ``range(n)``; scene indices are the file's own (a CZI's
        may skip numbers), so for scenes it is the number of distinct
        values.  A well index is a place in ``plates[p].wells``."""
        counts = {}
        for name in INDEX_NAMES:
            values = {image.index[name] for image in self.images if name in image.index}
            if values:
                counts[name] = len(values)
        return counts

    def select(self, **index) -> list:
        """The images whose index matches, e.g. ``select(scene=1, view=0)``."""
        unknown = set(index) - set(INDEX_NAMES)
        if unknown:
            raise KeyError(f"unknown indices {sorted(unknown)}; the indices are {INDEX_NAMES}")
        return [image for image in self.images
                if all(image.index.get(name) == value for name, value in index.items())]

    def _member_files(self) -> set:
        """The other files this file's images are read from -- a file set's
        members -- for ``group_files``.  Empty for a file that reads only itself."""
        out = set()
        for image in self.images:
            image = getattr(image, "source", image)
            for each in [image] + [p for p in image.parts if p is not image]:
                paths = getattr(each, "_member_paths", None)
                if paths is not None:
                    out.update(paths())
        out.discard(self.path)
        return out

    def _side_files(self) -> set:
        """Files that belong to this file set without being read as its images
        (CellVoyager correction images) -- for ``group_files``, which lists them
        among the set's parts.  They still open alone.  Empty by default."""
        return set()

    def check_members(self) -> int:
        """For a file set (a plate, an OME companion set): open every member
        file once and check that it is the plane its index describes (size,
        type, one plane).  Reading makes the same check plane by plane, at no
        cost; this finds a file that does not fit before anything is read, at
        the cost of opening every member.  Returns the number of members
        checked (0 for a file that is not a file set); raises
        ``CorruptFileError`` naming the members that do not fit."""
        from . import _pool
        from ._errors import CorruptFileError
        seen, bad = set(), []

        def tasks():                        # lazy: a plate may have millions of members
            for image in self.images:
                image = getattr(image, "source", image)        # open(axes=...) wraps it
                for each in [image] + [p for p in image.parts if p is not image]:
                    check = getattr(each, "_check_members", None)
                    if check is not None:
                        for task in check(seen):
                            yield lambda task=task: _counted(task(), bad)
        checked = _pool.run(tasks())
        if bad:
            shown = "; ".join(bad[:5]) + (f"; ... ({len(bad)} in all)" if len(bad) > 5 else "")
            raise CorruptFileError(f"{self.path!r}: {len(bad)} of {checked} member files do "
                                   f"not fit the index: {shown}")
        return checked

    def _every_image(self) -> list:
        """Every image this file gave out, each once: ``images``, their
        resolution levels and the parts of ragged / mixed-type images (also
        the levels ``resolutions="base"`` leaves out of ``images``).  An
        axes image (``open(axes=...)``) counts as the native image it shows."""
        out, seen = [], set()
        todo = [getattr(image, "source", image) for image in reversed(self.images)]
        while todo:
            image = todo.pop()
            if id(image) in seen:
                continue
            seen.add(id(image))
            out.append(image)
            todo.extend(reversed([m for m in image.resolutions + image.parts
                                  if id(m) not in seen]))
        return out

    def close(self) -> None:
        """Close every image the file gave out (``_every_image``): reading
        any of them afterwards -- a resolution level or a part too -- raises
        ``ValueError``."""
        for image in self._every_image():
            image.close()

    def __enter__(self) -> "File":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def outline(self) -> str:
        """The file's structure (docs/AXES.md): its plates, wells and fields,
        or its scenes, each with its tiles, the orthogonal indices that vary
        in it (``x view 2``), the image shape and the pyramid levels."""
        from ._outline import outline_lines
        return "\n".join([self._header()] + ["  " + line for line in outline_lines(self)])

    def _header(self) -> str:
        plates = f" plates={len(self.plates)}" if self.plates else ""
        return (f"<micro_reader.File {self.path!r} backend={self.backend!r} "
                f"format={self.format!r} images={len(self.images)}{plates}>")

    #: outline lines shown by ``repr``; ``outline()`` shows all
    REPR_LINES = 6

    def __repr__(self) -> str:
        from ._outline import outline_lines
        try:
            lines = outline_lines(self)
        except Exception:                               # a repr never fails
            return self._header()
        shown = lines[:self.REPR_LINES]
        if len(lines) > len(shown):
            shown.append(f"... {len(lines) - len(shown)} more lines: .outline()")
        return "\n".join([self._header()] + ["  " + line for line in shown])
