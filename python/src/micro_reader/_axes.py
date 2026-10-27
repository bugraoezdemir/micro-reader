"""An image presented with chosen axes (``Image.as_axes``, ``Image.squeeze``,
``open(axes=..., squeeze=...)``).

Arrays have only ``t c z y x`` and ``s`` (RGB samples); every other
dimension a file has is an image index (docs/AXES.md).  Backends report the
axes an image has: formats with an order of their own (TIFF, MRC) in that
order, the others in ``tczyx``.  An axes image rearranges them without
guessing:

- ``axes=None`` keeps the image's order; ``"canonical"`` gives ``t c z y x``,
  then ``s``; any other string is the exact axes wanted;
- axes the file lacks are inserted with size 1;
- RGB samples ``s`` stay a trailing ``s`` when asked for, or are folded into
  the channels with ``samples="channels"`` (channel ``k`` = sample ``k % S``
  of channel ``k // S``, as eubi-bridge stores them; ``open``'s default).
  The metadata then lists the folded channels: "<name> R" / "G" / "B" ("A"),
  tinted red / green / blue (white for A); a single unnamed channel gives
  unnamed, tinted channels; other sample counts are numbered, untinted -- the
  names eubi-bridge gives (``sample_channels.sample_channel_labels``);
- ``squeeze=True`` drops every size-1 axis except ``y`` and ``x``;
  ``squeeze="tz"`` drops just those (each must have size 1);
- an axis the requested ones cannot hold is an error that names it.

Reads stay exact: each request is translated into the native axes, read
there, and rearranged; the native image stays available as ``.source``.
"""
from __future__ import annotations

from typing import Optional, Union

import numpy as np

from ._base import Image

SAMPLE_POLICIES = ("keep", "channels")
#: labels and colours of the channels that samples become (as eubi-bridge's)
SAMPLE_LABELS = {3: ("R", "G", "B"), 4: ("R", "G", "B", "A")}
SAMPLE_COLORS = {3: ("#FF0000", "#00FF00", "#0000FF"),
                 4: ("#FF0000", "#00FF00", "#0000FF", "#FFFFFF")}
#: the array axes, outermost first; ``s`` (RGB samples) after them
STANDARD = "tczyx"
CANONICAL = STANDARD + "s"
#: never dropped by ``squeeze=True``: a plane stays a plane
KEPT_BY_SQUEEZE = "yx"


def folded_channels(channels, n: int, samples: int) -> tuple:
    """The channels of *n* channels of *samples* samples each, folded into
    ``n * samples`` channels, channel-major.  A file that lists them already
    (Bio-Formats-style: one channel per sample) is kept; otherwise each
    channel becomes "<name> R" / "G" / "B" ("A"), tinted from ``SAMPLE_COLORS``,
    keeping its wavelengths; an unnamed one among several is "Channel {c} R";
    a single unnamed channel gives unnamed, tinted channels; sample counts
    without labels are numbered ("<name> 0") and untinted."""
    from ._metadata import Channel
    channels = tuple(channels)
    if len(channels) == n * samples:
        return channels
    if len(channels) != n:
        channels = (Channel(),) * n
    labels = SAMPLE_LABELS.get(samples, tuple(str(i) for i in range(samples)))
    colors = SAMPLE_COLORS.get(samples) or (None,) * samples
    single_unnamed = n == 1 and not channels[0].name
    out = []
    for c, ch in enumerate(channels):
        for i in range(samples):
            name = None if single_unnamed else f"{ch.name or f'Channel {c}'} {labels[i]}"
            out.append(Channel(name=name, color=colors[i],
                               emission_wavelength=ch.emission_wavelength,
                               excitation_wavelength=ch.excitation_wavelength))
    return tuple(out)


class AxesImage(Image):
    """*source* presented with *axes* (see the module docstring)."""

    def __init__(self, source: Image, axes: Optional[str] = None, *, samples: str = "keep",
                 squeeze: Union[bool, str] = False, _select: Optional[dict] = None):
        if samples not in SAMPLE_POLICIES:
            raise ValueError(f"samples={samples!r}: one of {SAMPLE_POLICIES}")
        names = list(source.axes)
        #: native size-1 axes left out (squeezed): read at index 0
        select = dict(_select or {})
        axes = self._resolve(source, axes, names, select, samples)
        self.source = source
        self._names, self._select = names, select
        self._samples_policy = samples
        sizes = dict(zip(names, source.shape))
        # RGB samples folded into the channels
        self._fold = "s" in names and "s" not in axes and "s" not in select \
            and samples == "channels"
        if self._fold:
            sizes["c"] = sizes.get("c", 1) * sizes.pop("s")
        left = [a for a in names if a not in select and a not in axes
                and not (self._fold and a == "s") and not (self._fold and a == "c")]
        if left:
            hint = ("; samples='channels' folds RGB samples into the channels"
                    if "s" in left else "")
            raise ValueError(f"{source.name!r} has axes {''.join(left)!r} that {axes!r} cannot "
                             f"hold{hint}")
        if self._fold and "c" not in axes:
            raise ValueError(f"samples='channels' needs 'c' in the axes (got {axes!r})")
        if squeeze:
            axes = self._squeeze(axes, sizes, squeeze, names, select, source.name)
        shape = [sizes.get(a, 1) for a in axes]
        self._samples = source.shape[names.index("s")] if self._fold else 1
        unit = None
        if source.read_unit is not None:
            native_unit = dict(zip(names, source.read_unit))
            if self._fold:
                native_unit["c"] = native_unit.get("c", 1) * native_unit.pop("s")
            unit = [native_unit.get(a, 1) for a in axes]
        super().__init__(source._position, axes, shape, source.dtype, name=source.name,
                         read_unit=unit, index=source.index)

    @staticmethod
    def _resolve(source, axes, names, select, samples) -> str:
        """The axes string asked for: None, "canonical" or explicit letters."""
        free = [a for a in names if a not in select]
        fold = samples == "channels" and "s" in free
        if axes is None:                                     # the image's own order
            if fold:
                free.remove("s")
                if "c" not in free:                          # RGB without channels: c
                    at = next((i for i, a in enumerate(free) if a in "zyx"), len(free))
                    free.insert(at, "c")                     # where the canonical order has it
            return "".join(free)
        axes = axes.lower()
        if axes == "canonical":
            return STANDARD + ("s" if "s" in free and not fold else "")
        if len(set(axes)) != len(axes) or not axes.isalpha():
            raise ValueError(f"axes {axes!r}: one letter per axis, each once")
        stray = sorted(set(axes) - set(CANONICAL))
        if stray:
            raise ValueError(f"axes {axes!r}: {stray} are not array axes (arrays have "
                             f"{CANONICAL!r}; other dimensions are image indices)")
        return axes

    def _squeeze(self, axes, sizes, squeeze, names, select, name) -> str:
        """*axes* without the squeezed ones; squeezed native axes are read at
        index 0 (*select* is updated in place)."""
        if squeeze is True:
            drop = [a for a in axes if sizes.get(a, 1) == 1 and a not in KEPT_BY_SQUEEZE]
        else:
            drop = list(squeeze.lower())
            for a in drop:
                if a not in axes:
                    raise ValueError(f"squeeze {a!r}: {name!r} has axes {axes!r}")
                if sizes.get(a, 1) != 1:
                    raise ValueError(f"squeeze {a!r}: axis {a!r} of {name!r} has "
                                     f"{sizes[a]}, not 1")
        for a in drop:
            if a in names and not (self._fold and a == "c"):
                select[a] = 0
        return "".join(a for a in axes if a not in drop)

    def _read_block(self, block):
        names, select, axes = self._names, self._select, self.axes
        want = dict(zip(axes, block))
        S = self._samples
        if self._fold:                          # folded channels k0:k1 -> channels, all samples
            k = want.get("c", slice(0, 1))
            c0, c1 = k.start // S, (k.stop - 1) // S + 1
        native_key = []
        for a, size in zip(names, self.source.shape):
            if a in select:
                native_key.append(slice(select[a], select[a] + 1))
            elif self._fold and a == "s":
                native_key.append(slice(0, S))
            elif self._fold and a == "c":
                native_key.append(slice(c0, c1))
            else:
                native_key.append(want[a])
        data, read = self.source._read_block(tuple(native_key))
        order = list(names)
        # drop squeezed axes
        for a in select:
            data = data.squeeze(axis=order.index(a))
            order.remove(a)
        if self._fold:
            if "c" not in order:                  # RGB without channels: one channel
                data = np.expand_dims(data, order.index("s"))
                order.insert(order.index("s"), "c")
            ci, si = order.index("c"), order.index("s")
            data = np.moveaxis(data, si, ci + 1 if si > ci else ci)
            order.remove("s")
            ci = order.index("c")
            shape = list(data.shape)
            shape[ci:ci + 2] = [shape[ci] * shape[ci + 1]]
            data = data.reshape(shape)
            k = want.get("c", slice(0, 1))
            index = [slice(None)] * data.ndim
            index[ci] = slice(k.start - c0 * S, k.stop - c0 * S)
            data = data[tuple(index)]
        data = data.transpose([order.index(a) for a in axes if a in order])
        for i, a in enumerate(axes):                         # axes the file lacks
            if a not in order:
                data = np.expand_dims(data, i)
        return np.ascontiguousarray(data), read

    @property
    def metadata(self):
        """The native image's metadata; with samples folded into the channels,
        one channel per folded channel (``folded_channels``)."""
        meta = self.source.metadata
        if not self._fold:
            return meta
        from dataclasses import replace
        n = self.source.shape[self._names.index("c")] if "c" in self._names else 1
        return replace(meta, channels=folded_channels(meta.channels, n, self._samples))

    @property
    def resolutions(self) -> list:
        """The native image's pyramid, each level with these axes."""
        levels = self.source.resolutions
        if len(levels) == 1:
            return [self]
        return [self if level is self.source else
                AxesImage(level, self.axes, samples=self._samples_policy,
                          _select=dict(self._select))
                for level in levels]

    def as_axes(self, axes: Optional[str] = None, **policies) -> "AxesImage":
        """An axes image of the native image with other axes (they do not stack)."""
        return self.source.as_axes(axes, **policies)

    def squeeze(self, axes: Optional[str] = None) -> "AxesImage":
        """This axes image without its size-1 axes (all but ``y`` / ``x``, or those
        in *axes*), rebuilt over the native image."""
        return AxesImage(self.source, self.axes, samples=self._samples_policy,
                         squeeze=axes or True, _select=dict(self._select))

    def close(self) -> None:
        self.source.close()

    def __getattr__(self, name):
        """The native image's own public attributes (a CZI series' ``origin``, ...):
        an RGB image folded by ``open``'s default stays the backend's image to read
        them from.  Private ones are not passed on."""
        source = self.__dict__.get("source")
        if source is None or name.startswith("_"):
            raise AttributeError(f"{type(self).__name__!r} object has no attribute {name!r}")
        return getattr(source, name)

    def __repr__(self) -> str:
        return super().__repr__()[:-1] + f" of {self.source.axes!r}>"
