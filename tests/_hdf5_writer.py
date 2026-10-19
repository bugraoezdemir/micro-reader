"""Synthetic HDF5 files for the HDF5 backends: BigDataViewer (XML + HDF5, as
BigDataViewer / BigStitcher write them) and ilastik (datasets with vigra's
``axistags``)."""
from __future__ import annotations

import json

import numpy as np

h5py = __import__("pytest").importorskip("h5py")


def axistags(keys: str) -> str:
    flags = {"t": 8, "c": 1, "z": 2, "y": 2, "x": 2}
    return json.dumps({"axes": [{"key": k, "typeFlags": flags[k], "resolution": 0,
                                 "description": ""} for k in keys]})


def write_ilastik(path, data: np.ndarray, keys: str, name="exported_data", chunks=None,
                  compression=None, extra=None, shuffle=False):
    """One ilastik-style dataset (more with *extra*: {name: (data, keys)})."""
    with h5py.File(path, "w") as f:
        for n, (arr, k) in {name: (data, keys), **(extra or {})}.items():
            ds = f.create_dataset(n, data=arr, chunks=chunks, compression=compression,
                                  shuffle=shuffle)
            ds.attrs["axistags"] = axistags(k)
    return data


def write_luxendo(path, data: np.ndarray, size_zyx=(2.0, 0.26, 0.26), levels=(2,),
                  lasers_on=(445,), band="BP 459-499", description="Cyan"):
    """A Luxendo .lux.h5 (github.com/Luxendo/luxendo-image): Data (z, y, x),
    uncompressed 64^3-style chunks, Data_f_f_f levels, element_size_um as
    float32, JSON metadata.  -> {level name: array}."""
    meta = {"processingInformation": {"channel": "0", "channel_description": description,
                                      "voxel_size_um": {"width": size_zyx[2], "height": size_zyx[1],
                                                        "depth": size_zyx[0]}},
            "metaData": {"lasers": [{"name": f"{nm} nm", "on": nm in lasers_on, "intensity": 50,
                                     "wavelength_nm": nm} for nm in (445, 488, 561)]
                         + [{"name": "LED", "on": False, "wavelength_nm": 100}],
                         "filterwheels": [{"name": "fw_long", "selection": band},
                                          {"name": "fw_short", "selection": "empty"}]},
            "imagingBranch": {"detection": {"filterwheels": ["fw_long"]}}}
    out = {}
    with h5py.File(path, "w") as f:
        for name, arr in [("Data", data)] + [
                (f"Data_{k}_{k}_{k}", data[::k, ::k, ::k].copy()) for k in levels]:
            ds = f.create_dataset(name, data=arr, chunks=tuple(min(16, n) for n in arr.shape))
            ds.attrs["element_size_um"] = np.array(size_zyx, np.float32)
            out[name] = arr
        f["metadata"] = json.dumps(meta)
    return out


def write_fiji(path, datasets: dict, size_zyx=(0.5, 0.2, 0.2)):
    """Fiji's HDF5 plugin (Vibez): {name: array}, each with element_size_um."""
    with h5py.File(path, "w") as f:
        for name, arr in datasets.items():
            f.create_dataset(name, data=arr).attrs["element_size_um"] = np.array(size_zyx, np.float32)


def _xml(h5_name, setups, timepoints, voxel=(0.5, 0.5, 2.0), unit="µm", missing=(),
         registrations=None, loader="bdv.hdf5", partitions=(), channel_names=None,
         timepoint_xml=None):
    """A SpimData XML: *setups* = [(id, attributes dict, size zyx)]."""
    lines = ['<?xml version="1.0" encoding="UTF-8"?>', '<SpimData version="0.2">',
             '  <BasePath type="relative">.</BasePath>', '  <SequenceDescription>',
             f'    <ImageLoader format="{loader}">',
             f'      <hdf5 type="relative">{h5_name}</hdf5>']
    for part in partitions:
        lines.append(f'      <partition><path type="relative">{part}</path></partition>')
    lines += ['    </ImageLoader>', '    <ViewSetups>']
    for sid, attrs, (z, y, x) in setups:
        lines += ['      <ViewSetup>', f'        <id>{sid}</id>', f'        <name>setup {sid}</name>',
                  f'        <size>{x} {y} {z}</size>',
                  f'        <voxelSize><unit>{unit}</unit><size>{voxel[0]} {voxel[1]} {voxel[2]}</size></voxelSize>',
                  '        <attributes>']
        lines += [f'          <{k}>{v}</{k}>' for k, v in attrs.items()]
        lines += ['        </attributes>', '      </ViewSetup>']
    names = {"channel": channel_names or {}}
    for attr in ("illumination", "channel", "tile", "angle"):
        ids = sorted({a.get(attr, 0) for _, a, _ in setups})
        tag = attr.capitalize()
        lines.append(f'      <Attributes name="{attr}">')
        for i in ids:
            lines.append(f'        <{tag}><id>{i}</id><name>{names.get(attr, {}).get(i, i)}</name></{tag}>')
        lines.append('      </Attributes>')
    lines.append('    </ViewSetups>')
    lines.append(timepoint_xml or (f'    <Timepoints type="range"><first>{timepoints[0]}</first>'
                                   f'<last>{timepoints[-1]}</last></Timepoints>'))
    if missing:
        lines.append('    <MissingViews>')
        lines += [f'      <MissingView timepoint="{t}" setup="{s}" />' for t, s in missing]
        lines.append('    </MissingViews>')
    lines += ['  </SequenceDescription>', '  <ViewRegistrations>']
    for t in timepoints:
        for sid, _, _ in setups:
            lines.append(f'    <ViewRegistration timepoint="{t}" setup="{sid}">')
            for affine in (registrations or {}).get(sid, [(1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 4, 0)]):
                name = ""
                if isinstance(affine[0], str):              # (Name, affine): BigStitcher names them
                    name, affine = f"<Name>{affine[0]}</Name>", affine[1]
                lines.append(f'      <ViewTransform type="affine">{name}<affine>'
                             + " ".join(str(v) for v in affine) + '</affine></ViewTransform>')
            lines.append('    </ViewRegistration>')
    lines += ['  </ViewRegistrations>', '</SpimData>']
    return "\n".join(lines)


def write_bdv(folder, name="dataset", tiles=2, channels=2, timepoints=(0, 1), shape=(6, 40, 50),
              levels=((1, 1, 1), (2, 2, 1)), chunks=(4, 16, 16), dtype=np.int16, missing=(),
              compression="gzip", seed=0, xml=True, scaleoffset=False, value_range=None,
              transforms=None, **xml_options):
    """A BigDataViewer dataset: setups = tiles x channels (setup id = tile *
    channels + channel), tile k shifted by k * 45 pixels in x.  *scaleoffset*:
    as BigDataViewer writes (HDF5 scaleoffset, then *compression*);
    *value_range*: (low, high) of the uint16 values (default: all of them).
    *transforms*: tile -> [(Name, affine)] instead of the unnamed grid + calibration.
    Returns {(timepoint, tile, channel, level): stored array (z, y, x)}."""
    low, high = value_range or (0, 65535)
    rng = np.random.default_rng(seed)
    h5_path = folder / f"{name}.h5"
    setups, stored = [], {}
    registrations = {}
    with h5py.File(h5_path, "w") as f:
        for tile in range(tiles):
            for ch in range(channels):
                sid = tile * channels + ch
                setups.append((sid, {"illumination": 0, "channel": ch, "tile": tile, "angle": 0},
                               shape))
                # BigStitcher: grid translation (first listed, applied last), calibration last
                registrations[sid] = transforms(tile) if transforms else                     [(1, 0, 0, 45.0 * tile, 0, 1, 0, 0, 0, 0, 1, 0),
                     (1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 4, 0)]
                g = f.create_group(f"s{sid:02d}")
                g["resolutions"] = np.array(levels, dtype=np.float64)
                g["subdivisions"] = np.array([tuple(reversed(chunks))] * len(levels), dtype=np.int32)
                for t in timepoints:
                    if (t, sid) in missing:
                        continue
                    for level, factors in enumerate(levels):
                        fx, fy, fz = (int(v) for v in factors)
                        lshape = (max(1, shape[0] // fz), max(1, shape[1] // fy), max(1, shape[2] // fx))
                        if np.dtype(dtype) == np.int16:      # uint16 values, stored as int16
                            arr = rng.integers(low, high, lshape, dtype=np.uint16,
                                               endpoint=True).view(np.int16)
                        else:
                            arr = rng.integers(0, 200, lshape).astype(dtype)
                        f.create_dataset(f"t{t:05d}/s{sid:02d}/{level}/cells", data=arr,
                                         chunks=tuple(min(c, n) for c, n in zip(chunks, lshape)),
                                         compression=compression,
                                         scaleoffset=0 if scaleoffset else None)
                        stored[(t, tile, ch, level)] = arr
    if xml:
        (folder / f"{name}.xml").write_text(
            _xml(h5_path.name, setups, list(timepoints), missing=missing,
                 registrations=registrations, **xml_options), encoding="utf-8")
    return stored


def _ims_text(value):
    """Imaris attributes: arrays of 1-byte strings."""
    return np.array(list(str(value)), dtype="S1")


def _ims_info(f, n_c, extent_um):
    info = f.require_group("DataSetInfo/Image")
    for k, v in enumerate(extent_um):                   # x, y, z
        info.attrs[f"ExtMin{k}"] = _ims_text(0.0)
        info.attrs[f"ExtMax{k}"] = _ims_text(v)
    info.attrs["Unit"] = _ims_text("um")
    for c in range(n_c):
        f.require_group(f"DataSetInfo/Channel {c}").attrs["Name"] = _ims_text(f"ch{c}")


def write_ims_multifile(folder, name="set", n_t=2, n_c=2, shape=(3, 20, 24), padded=(4, 32, 32),
                        chunks=(2, 16, 16), levels=2, skip=(), copy=True, seed=0):
    """FileConverter's multi-file Imaris (as BioImage Archive S-BIAD13):
    ``<name>.ims`` (and, *copy*, the same ``<name>.h5`` FileConverter also
    writes) holds DataSetInfo; each ``DataSet/ResolutionLevel r/TimePoint
    t/Channel c`` is an external link to group ``ResolutionLevel r`` of
    ``./<name>--C<cc>--T<ttttt>.h5``, which holds ``Data`` (padded) with the
    real size in ImageSize attributes.  Parts whose (t, c) is in *skip* are
    not written.  -> real data (T, C, Z, Y, X) of level 0 (zeros where skipped)."""
    rng = np.random.default_rng(seed)
    data = rng.integers(1, 250, (n_t, n_c, *shape)).astype(np.uint8)
    for t in range(n_t):
        for c in range(n_c):
            if (t, c) in skip:
                data[t, c] = 0
                continue
            with h5py.File(folder / f"{name}--C{c:02d}--T{t:05d}.h5", "w") as part:
                for r in range(levels):
                    real = data[t, c, :, ::2 ** r, ::2 ** r]
                    pad = (padded[0], -(-padded[1] // 2 ** r), -(-padded[2] // 2 ** r))
                    g = part.create_group(f"ResolutionLevel {r}")
                    ds = g.create_dataset("Data", shape=pad, dtype=np.uint8,
                                          chunks=tuple(min(a, b) for a, b in zip(chunks, pad)),
                                          compression="gzip", fillvalue=0)
                    ds[:real.shape[0], :real.shape[1], :real.shape[2]] = real
                    for axis, size in zip("ZYX", real.shape):
                        g.attrs[f"ImageSize{axis}"] = _ims_text(size)
    for suffix in (".ims", ".h5") if copy else (".ims",):
        with h5py.File(folder / f"{name}{suffix}", "w") as f:
            f.attrs["ImarisVersion"] = _ims_text("5.5.0")
            for r in range(levels):
                for t in range(n_t):
                    group = f.require_group(f"DataSet/ResolutionLevel {r}/TimePoint {t}")
                    for c in range(n_c):
                        group[f"Channel {c}"] = h5py.ExternalLink(
                            f"./{name}--C{c:02d}--T{t:05d}.h5", f"ResolutionLevel {r}")
            _ims_info(f, n_c, (shape[2] * 0.5, shape[1] * 0.5, shape[0] * 2.0))
    return data


def write_ims_luxendo(folder, name="imaris", n_t=2, n_c=2, shape=(3, 20, 24), skip=(), seed=0):
    """Luxendo's link-only Imaris file (as S-BIAD3686's ``imaris.ims``): its
    Channel groups carry the ImageSize attributes, ``Data`` is an external
    link to ``Data`` of ``raw/stack_0-Zstack_channel_<c>/Cam_long_<ttttt>.lux.h5``
    (Luxendo .lux.h5 stacks, write_luxendo).  Stacks whose (t, c) is in
    *skip* are not written.  -> data (T, C, Z, Y, X), zeros where skipped."""
    rng = np.random.default_rng(seed)
    data = rng.integers(0, 4000, (n_t, n_c, *shape)).astype(np.uint16)
    with h5py.File(folder / f"{name}.ims", "w") as f:
        f.attrs["ImarisVersion"] = np.frombuffer(b"5.5.0", np.uint8)
        for t in range(n_t):
            for c in range(n_c):
                stack = f"raw/stack_0-Zstack_channel_{c}/Cam_long_{t:05d}.lux.h5"
                if (t, c) in skip:
                    data[t, c] = 0
                else:
                    (folder / stack).parent.mkdir(parents=True, exist_ok=True)
                    write_luxendo(folder / stack, data[t, c], levels=())
                g = f.require_group(f"DataSet/ResolutionLevel 0/TimePoint {t}/Channel {c}")
                for axis, size in zip("ZYX", shape):
                    g.attrs[f"ImageSize{axis}"] = np.frombuffer(str(size).encode(), np.uint8)
                g["Data"] = h5py.ExternalLink(stack, "Data")
        _ims_info(f, n_c, (shape[2] * 0.26, shape[1] * 0.26, shape[0] * 2.0))
    return data
