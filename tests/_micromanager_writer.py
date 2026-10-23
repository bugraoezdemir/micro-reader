"""Micro-Manager image-per-file acquisitions for tests, laid out as the OME
samples (``formats/tiff/micromanager/ome-serge``, ``ome-thomas/test/*_sep``)
and the Zenodo 6659532 2.0.0 sample: ``<acquisition>/display_and_comments.txt``
and one folder per stage position holding ``metadata.txt`` (JSON:
``"Summary"``, then ``"FrameKey-<t>-<c>-<z>"`` entries (1.4), ``"Coords-<file>"``
entries (2.0 betas) or ``"Coords-<folder>/<file>"`` + ``"Metadata-<folder>/<file>"``
entries (2.0.0)) and one ImageJ TIFF per plane, its plane metadata as JSON in
the ImageJ ``Info`` (tifffile)."""
from __future__ import annotations

import json
import os

import numpy as np


def name_14(t, channel, z):
    return f"img_{t:09d}_{channel}_{z:03d}.tif"


def name_20(t, c, p, z):
    return f"img_channel{c:03d}_position{p:03d}_time{t:09d}_z{z:03d}.tif"


def summary(*, frames, channels, slices, positions, labels=None, pixel_size=0.0, z_step=0.0,
            interval=0.0, custom=None, colors=None, prefix="acq", uuid="79ab3210-c64c",
            style="1.4"):
    """A Summary as the samples write it (*style* "1.4", "2.0" (beta) or
    "2.0.0"); *labels*: the position list's labels (None: no list,
    ``InitialPositionList`` null; the 2.0.0 sample lists none)."""
    if style == "2.0.0":
        return {"Prefix": prefix, "MicroManagerVersion": "2.0.0", "MetadataVersion": "12.0.0",
                "ComputerName": "Dobby", "ChNames": list(channels),
                "AxisOrder": ["channel", "time", "z", "position"], "TimeFirst": False,
                "SlicesFirst": True, "Slices": slices, "z-step_um": z_step,
                "IntendedDimensions": {"channel": len(channels), "position": positions,
                                       "time": frames, "z": slices},
                "Frames": frames, "Positions": positions, "Channels": len(channels),
                "Width": 0, "Height": 0, "IJType": 1, "PixelType": "GRAY16"}
    if style == "2.0":
        return {"ChannelGroup": "Channel", "Slices": slices, "ChNames": list(channels),
                "AxisOrder": ["channel", "time"], "Frames": frames, "Prefix": prefix,
                "Channels": len(channels), "CustomIntervals_ms": [], "MetadataVersion": "11.0.0",
                "z-step_um": z_step, "Positions": positions, "WaitInterval": 0,
                "Directory": "", "MicroManagerVersion": "2.0.0-beta3-20160124",
                "IntendedDimensions": {"position": positions, "time": frames, "z": slices,
                                       "channel": len(channels)}}
    out = {"Slices": slices, "Interval_ms": interval, "UUID": uuid, "PixelType": "GRAY16",
           "z-step_um": z_step, "MetadataVersion": 10, "PositionIndex": 0, "SlicesFirst": False,
           "MicroManagerVersion": "1.4.16", "ChNames": list(channels), "Prefix": prefix,
           "PixelSize_um": pixel_size, "Frames": frames, "Channels": len(channels),
           "CustomIntervals_ms": custom or [], "TimeFirst": False, "Positions": positions,
           "Directory": "F:\\data", "Time": "2018-01-18 15:02:44 +0100",
           "InitialPositionList": None if labels is None else [
               {"DeviceCoordinatesUm": {"XY": [0.0, 0.0]}, "Label": label,
                "GridRowIndex": 0, "GridColumnIndex": 0} for label in labels]}
    if colors is not None:
        out["ChColors"] = list(colors)
    return out


def write_plane(path, plane, info: dict):
    import tifffile
    tifffile.imwrite(path, plane, imagej=True, metadata={"Info": json.dumps(info, indent=2)})


def write_acquisition(root, *, frames=2, channels=("DAPI", "GFP"), slices=1, positions=1,
                      labels=None, style="1.4", skip=(), skip_positions=(), unlisted=(),
                      stage=None, shape=(8, 12), seed=3, display=True, plane_pixel_size=None,
                      tiff_pixel_size=None, **summary_kw):
    """An acquisition in folder *root*: random uint16 planes.  *labels*: the
    position folders (and the Summary's position list; None: ``Pos<k>``
    without a list).  *skip*: (p, t, c, z) planes listed but not written;
    *skip_positions*: position folders not written; *unlisted*: (p, t, c, z)
    planes neither listed nor written; *stage*: p -> (x, y, z) recorded per
    plane.  -> ({(p, t, c, z): plane}, {p: metadata.txt path})."""
    rng = np.random.default_rng(seed)
    os.makedirs(root, exist_ok=True)
    folders = list(labels) if labels is not None else [f"Pos{p}" for p in range(positions)]
    data, files = {}, {}
    for p, folder in enumerate(folders):
        if p in skip_positions:
            continue
        here = os.path.join(root, folder)
        os.makedirs(here, exist_ok=True)
        s = summary(frames=frames, channels=channels, slices=slices, positions=positions,
                    labels=labels, style=style, **summary_kw)
        if style == "1.4":
            s["PositionIndex"] = p
        md = {"Summary": s}
        for t in range(frames):
            for c, channel in enumerate(channels):
                for z in range(slices):
                    if (p, t, c, z) in unlisted:
                        continue
                    plane = rng.integers(1, 60000, shape).astype("u2")
                    data[(p, t, c, z)] = plane
                    x, y, zz = stage(p) if stage else (0, 0, 0)
                    if style == "1.4":
                        name = name_14(t, channel, z)
                        entry = {"FileName": name, "PositionName": folder, "PositionIndex": p,
                                 "FrameIndex": t, "ChannelIndex": c, "SliceIndex": z,
                                 "Channel": channel, "XPositionUm": x, "YPositionUm": y,
                                 "ZPositionUm": zz, "PixelSizeUm": plane_pixel_size
                                 if plane_pixel_size is not None else s["PixelSize_um"],
                                 "Width": shape[1], "Height": shape[0]}
                        md[f"FrameKey-{t}-{c}-{z}"] = entry
                        info = dict(entry)
                    elif style == "2.0.0":
                        # the file name carries the position folder (as the sample)
                        name = name_20(t, c, p, z)
                        coords = {"Frame": t, "FrameIndex": t, "PositionIndex": p, "Slice": z,
                                  "SliceIndex": z, "ChannelIndex": c}
                        md[f"Coords-{folder}/{name}"] = coords
                        info = dict(coords, Width=shape[1], Height=shape[0], PixelType="GRAY16",
                                    PixelSizeUm=plane_pixel_size if plane_pixel_size is not None
                                    else 0.0, PositionName=folder, XPositionUm=x, YPositionUm=y,
                                    ZPositionUm=zz)
                        md[f"Metadata-{folder}/{name}"] = info
                    else:
                        name = name_20(t, c, p, z)
                        md[f"Coords-{name}"] = {"position": p, "time": t, "z": z, "channel": c}
                        info = {"XPositionUm": x, "YPositionUm": y, "ZPositionUm": zz,
                                "PositionName": {"PropVal": folder, "PropType": "String"}}
                    if tiff_pixel_size is not None:
                        info["PixelSizeUm"] = tiff_pixel_size
                    if (p, t, c, z) not in skip:
                        write_plane(os.path.join(here, name), plane, info)
        path = os.path.join(here, "metadata.txt")
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(md, fh, indent=2)
        files[p] = path
    if display:
        with open(os.path.join(root, "display_and_comments.txt"), "w", encoding="utf-8") as fh:
            json.dump({"Channels": [{"Name": c, "Color": -1} for c in channels],
                       "Comments": {"Summary": ""}}, fh, indent=2)
    return data, files
