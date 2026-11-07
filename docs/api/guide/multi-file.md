# Multi-file images

Many microscopy images are spread over several files. micro-reader handles two kinds:

- **Composite formats.** The format itself fixes where its other files are. Examples are an OIF and its `.oif.files` folder, a VSI and its `_name_` folder, and a DeltaVision file and its `.log`.
- **Index files.** One file lists the files that hold the planes. Examples are a plate's `Index.idx.xml`, an OME `.companion.ome` and a Leica `.xlef`.

In both cases, you open one path and get one `File`.

## Composite formats

| Format | Files | Open any of |
| --- | --- | --- |
| OIR | `name.oir`, `name_00001`, `name_00002`, ... (continuations) | any of them |
| Olympus OIF | `name.oif` + `name.oif.files/` (one TIFF per plane) | the `.oif`, the folder, or a file in it |
| Olympus VSI | `name.vsi` + `_name_/stack1/frame_t.ets`, ... | the `.vsi`, the `_name_` folder, a `stack` folder or an `.ets` |
| DeltaVision | `name.dv` + `name.dv.log` (softWoRx log, for channel names) | the `.dv`, or the `.log` |
| Micro-Manager MMStack | `name_MMStack_Pos0.ome.tif`, `name_MMStack_Pos0_1.ome.tif`, ... | any stack file |
| NDTiff | a dataset folder with `NDTiff.index` and stack files | the folder, or any file in it |
| Plates | index + plane TIFFs + side files (`Assaylayout/`, `FFC_Profile/`, ...) | the index, the measurement folder, or a side file |
| Opera Flex plates | a folder of `.flex` files, one per barcode | any `.flex` file, or the folder |
| Cellomics plates | one `.C01` file per field and channel | any `.C01` file, or the folder |
| scanR plates | `experiment_descriptor.xml`, `data/` folder | the descriptor, a folder, or a data file |
| NIS-Elements well-plate runs | one ND2 file per well or per point | any of the ND2 files |

```python
f = micro_reader.open("cord_60x.oif.files/s_C001Z003.tif")
f.path       # 'cord_60x.oif'                     the main file
f.opened     # 'cord_60x.oif.files/s_C001Z003.tif' what you passed
f.images[0].shape    # (1, 2, 6, 1024, 1024): the whole image, not one plane
```

The rules:

- Parts are found by the format's own naming rule, never by searching. Names that differ only in letter case are found too, for files copied from Windows.
- If the main file is missing, a part that is an image on its own is read alone, with a `CompositeWarning`. For example, an `.ets` file of a VSI is read alone when it is a plain 2-D image. A part that is not an image on its own raises an error.

### Opening a folder

```python
micro_reader.open("Measurement1/")        # a Harmony measurement folder
micro_reader.open("dataset_1/")           # an NDTiff dataset
micro_reader.open("one_image_folder/")    # a folder that holds exactly one image
```

A folder opens the image it belongs to, or the one image its direct contents hold. A plate or companion index inside it comes first. A folder with no image or with several images raises an error that lists what it found. Subfolders are not searched.

## OME companion sets

An OME companion file (`.companion.ome`) is an OME-XML file that describes images stored in separate OME-TIFF files.

**Open the companion** to read the whole set. Its planes are grouped into `t c z y x` images, with the companion's metadata:

```python
f = micro_reader.open("timelapse.companion.ome")
f.backend, f.format      # ('companion', 'ome-companion')
```

A companion file that describes a plate reads as a plate (see [Plates](plates.md)).

**Open one of its TIFF files** to read only that file. Pixel sizes, channels and position are taken from the companion:

```python
f = micro_reader.open("timelapse_t001.ome.tif")
f.backend, f.format      # ('tiff', 'ome-tiff')
```

If the companion is missing, the file is read without pixel sizes and you get a `CompanionWarning`. A TIFF whose metadata lives in another OME-TIFF, instead of a `.companion.ome`, takes it from that file in the same way.

### Multi-file OME-TIFF without a companion

Some OME-TIFF sets keep the full OME-XML, describing every file, in each of their TIFF files. Opening any of them reads the whole set. To read only the planes of the file you opened:

```python
f = micro_reader.open("set_t001.ome.tif", multifile=False)
```

## Checking the members of a file set

For a plate or companion set, `check_members()` opens every member file once and checks that each one is the plane its index describes:

```python
n = f.check_members()     # number of member files checked; 0 if f is not a file set
```

It raises `CorruptFileError` naming the members that do not fit. Reads make the same check plane by plane, so you only need `check_members()` to find problems before a long job starts.

## Grouping many files

When you have a folder of files, for example everything you want to convert, you need to know which files make one image. Opening each file to find out would open plate members thousands of times. `group_files()` sorts the files without reading most of them:

```python
g = micro_reader.group_files("experiment/")     # a folder (walked down), a file, or a list

for group in g.groups:
    group.kind       # 'composite' or 'index'
    group.main       # the file to open (None: the main file is missing)
    group.format     # the backend that reads it
    group.parts      # the other given files of the same image
    group.members    # index: every file its images are read from
    group.notes      # index: what opening it said (missing files, errors)

g.singles            # every other file, as given: open each on its own
```

A real example, the DeltaVision folder of the OME sample data:

```python
g = micro_reader.group_files("dv/U2OS")
for grp in g.groups:
    print(grp.kind, grp.main and os.path.basename(grp.main),
          [os.path.basename(p) for p in grp.parts])
print([os.path.basename(s) for s in g.singles])
```

```text
composite U2OS_AurB_AurA_001_R3D.dv ['U2OS_AurB_AurA_001_R3D.dv.log']
composite None ['U2OS_AurB_AurA_100x_001_R3D.dv.log']
composite None ['U2OS_Eg5_PH3_100x_002_R3D.dv.log']
composite U2OS_Eg5_PH3_100x_002_R3D_D3D.dv ['U2OS_Eg5_PH3_100x_002_R3D_D3D_log.txt']
['U2OS_AurB_AurA_100x_001_R3D_D3D.dv', 'readme.txt']
```

The two groups with `main=None` are logs whose raw `.dv` file was not shared. Their `expected` field names the missing file. The deconvolved `_D3D.dv` next to them is a single file, and it still gets its channel names from the raw file's log when opened.

And a Harmony measurement folder:

```python
g = micro_reader.group_files("HipSci ... Measurement1/")
grp = g.groups[0]
grp.kind, grp.format          # ('index', 'harmony')
len(grp.parts)                # 2: Assaylayout/*.xml and FFC_Profile/*.xml
len(grp.members)              # 24 plane TIFFs
grp.notes                     # ("...: 2564 image files are missing ...",)
g.singles                     # []: every plane belongs to the index
```

How it works and what it costs:

- Index files are recognised by name first. Only those are read and opened.
- Every other file is matched against the composite formats' naming rules, which needs no reading. Only a file that a rule may apply to has its first bytes read.
- A plain file is never read.
- A file named by an index is listed in that index's group only, not as a single.
- A folder of ND2 files from a NIS-Elements well-plate run is one group (`group_files`' default, `plates=True`). With `plates=False`, each file is a single.
- When two index files list the same members (a CellVoyager `.wpi`, `.mrf` and `.mlf`), one is `main` and the others are its `parts`.

### Converting a folder

```python
import micro_reader

g = micro_reader.group_files(folder)
to_open = [grp.main for grp in g.groups if grp.main] + g.singles

for path in to_open:
    try:
        with micro_reader.open(path) as f:
            convert(f)
    except micro_reader.UnsupportedFormatError:
        print("skipped", path)

for grp in g.groups:
    if grp.main is None:
        print("incomplete: expected", grp.expected, "for", grp.parts)
```
