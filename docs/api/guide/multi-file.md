# Multi-file images

Many microscopy images are stored across several files. micro-reader supports two types of multi-file images:

- **Composite formats:** The format defines where its associated files are located. Examples include an OIF file and its `.oif.files` folder, a VSI file and its `_name_` folder, and a DeltaVision file and its `.log` file.
- **Index files:** A separate file lists the files that contain the image planes. Examples include a plate's `Index.idx.xml`, an OME `.companion.ome` file, and a Leica `.xlef` file.

In both cases, you open a single path and get one `File` object.

## Composite formats

| Format | Files | Open any of |
| --- | --- | --- |
| OIR | `name.oir`, `name_00001`, `name_00002`, ... (continuation files) | Any of them |
| Olympus OIF | `name.oif` and `name.oif.files/` (one TIFF per plane) | The `.oif` file, the folder, or a file inside it |
| Olympus VSI | `name.vsi` and `_name_/stack1/frame_t.ets`, ... | The `.vsi` file, the `_name_` folder, a `stack` folder, or an `.ets` file |
| DeltaVision | `name.dv` and `name.dv.log` (softWoRx log containing channel names) | The `.dv` file or the `.log` file |
| Micro-Manager MMStack | `name_MMStack_Pos0.ome.tif`, `name_MMStack_Pos0_1.ome.tif`, ... | Any stack file |
| NDTiff | A dataset folder containing `NDTiff.index` and stack files | The folder or any file inside it |
| Micro-Manager 1.4 | A folder per stage position with `metadata.txt` and one TIFF per plane | Any file or folder of the acquisition |
| MetaMorph | `name.nd` and the TIFF / STK files named after it | The `.nd` file, any of its files, or the folder |
| ICS 1.0 | `name.ics` (header) and `name.ids` (pixels) | Either file |
| Luxendo experiments | A folder with `raw/` (one folder of `.lux.h5` stacks per channel) and link files next to it | The folder, `raw/`, a channel folder, a `.lux.h5` file, or a link file such as `imaris.ims` or `bdv.xml` |
| Plates | An index file, plane TIFFs, and supporting files (`Assaylayout/`, `FFC_Profile/`, ...) | The index, measurement folder, or a supporting file |
| Opera Flex plates | A folder of `.flex` files, one per barcode | Any `.flex` file or the folder |
| Cellomics plates | One `.C01` file per field and channel | Any `.C01` file or the folder |
| scanR plates | `experiment_descriptor.xml` and a `data/` folder | The descriptor, a folder, or a data file |
| NIS-Elements well-plate runs | One ND2 file per well or per point | Any of the ND2 files |

```python
f = micro_reader.open("cord_60x.oif.files/s_C001Z003.tif")
f.path                  # 'cord_60x.oif'                       the main file
f.opened                # 'cord_60x.oif.files/s_C001Z003.tif'  the path you passed
f.images[0].shape       # (1, 2, 6, 1024, 1024), the whole image, not one plane
```

micro-reader locates associated files using each format's naming rules, not by searching the directory. Matching is case-insensitive, which also helps when files have been copied from Windows.

If the main file is missing, a part that can be read as an image on its own is opened independently, with a `CompositeWarning`. For example, a VSI `.ets` file is read on its own if it contains a plain 2-D image. A part that cannot be read as an image independently raises an error.

### Opening a folder

```python
micro_reader.open("Measurement1/")       # a Harmony measurement folder
micro_reader.open("dataset_1/")          # an NDTiff dataset
micro_reader.open("one_image_folder/")   # a folder containing exactly one image
```

A folder opens the image it belongs to, or the single image represented by its direct contents. A plate or companion index inside the folder takes precedence.

A folder that holds no image itself, such as a downloaded sample folder with the export one or two levels down, opens the one dataset found in its subfolders (up to six levels down). If the folder contains no image, or several images or datasets, opening it raises an error listing what was found.

## OME companion sets

An OME companion file (`.companion.ome`) is an OME-XML file that describes images stored in separate OME-TIFF files.

**Open the companion file** to read the entire set. Its planes are grouped into `t c z y x` images, which carry the companion's metadata:

```python
f = micro_reader.open("timelapse.companion.ome")
f.backend, f.format      # ('companion', 'ome-companion')
```

If the companion describes a plate, it is opened as a plate (see [Plates](plates.md)).

**Opening one of the TIFF files** also reads the entire set, as if you had opened the companion. `f.path` is then the companion:

```python
f = micro_reader.open("timelapse_t001.ome.tif")
f.backend, f.format      # ('companion', 'ome-companion')
f.path                   # 'timelapse.companion.ome'
```

To read only that file, pass `multifile=False`. Its pixel sizes, channel information and position are still taken from the companion:

```python
f = micro_reader.open("timelapse_t001.ome.tif", multifile=False)
f.backend, f.format      # ('tiff', 'ome-tiff')
```

If the companion is missing, or does not list the file, opening the whole set raises an error that names `multifile=False`. With `multifile=False`, the TIFF is then read without pixel sizes and a `CompanionWarning` is issued. A TIFF whose metadata is stored in another OME-TIFF file rather than a `.companion.ome` file works in the same way, with that OME-TIFF in the companion's place.

### Multi-file OME-TIFF without a companion

Some OME-TIFF sets store the complete OME-XML metadata in each TIFF file. This metadata describes all files in the set, so opening any one of them reads the entire set.

To read only the planes in the file you opened, pass `multifile=False`:

```python
f = micro_reader.open("set_t001.ome.tif", multifile=False)
```

## Checking the members of a file set

For a plate or companion set, `check_members()` opens each member file once and checks that it contains the planes described by the index:

```python
n = f.check_members()  # number of member files checked; 0 if f is not a file set
```

If a member does not match its description, the method raises `CorruptFileError` and identifies the affected files.

Pixel reads perform the same check plane by plane. You therefore only need to call `check_members()` when you want to detect problems before starting a long-running job.

## Grouping many files

When working with a folder of files, for example, a collection to convert, you may need to determine which files belong to the same image. Opening every file to find out can be expensive, especially when plate members would otherwise be opened thousands of times.

`group_files()` groups files using their naming rules and index metadata, without opening most of them:

```python
g = micro_reader.group_files("experiment/")  # a folder (walked recursively), a file, or a list

for group in g.groups:
    group.kind       # 'composite' or 'index'
    group.main       # the file to open (None if the main file is missing)
    group.format     # the backend that reads it
    group.parts      # other given files belonging to the same image
    group.members    # index: every file from which its images are read
    group.notes      # index: information reported during opening (missing files, errors)

g.singles            # all remaining files, as given; open each independently
```

For example, consider the DeltaVision folder from the OME sample data:

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

The two groups with `main=None` contain log files whose corresponding raw `.dv` files were not shared. Their `expected` field identifies the missing file.

The deconvolved `_D3D.dv` file is independent and can still be opened on its own. When opened, it obtains its channel names from the raw file's log.

The same call on a Harmony measurement folder produces an index group:

```python
g = micro_reader.group_files("HipSci ... Measurement1/")
grp = g.groups[0]
grp.kind, grp.format          # ('index', 'harmony')
len(grp.parts)                # 2: Assaylayout/*.xml and FFC_Profile/*.xml
len(grp.members)              # 24 plane TIFFs
grp.notes                     # ("...: 2564 image files are missing ...",)
g.singles                     # [], since every plane belongs to the index
```

`group_files()` works in the following steps:

- **Index files:** Recognised by name first. Only these files are read and opened.
- **Other files:** Matched against the composite formats' naming rules, which do not require reading file contents. Only files that might match a rule have their first bytes read. Plain files are never read.
- **Files listed in an index:** Added only to that index's group, not listed as singles.
- **ND2 well-plate runs:** By default, a folder of ND2 files from a NIS-Elements well-plate run is treated as one group (`plates=True`). With `plates=False`, each file is treated as a single.
- **Indexes sharing members:** Index files that list the same members, such as CellVoyager `.wpi`, `.mrf`, and `.mlf` files, are merged into one group. One becomes `main`, and the others become `parts`.

### Converting a folder

You can use `group_files()` to identify the files to open before converting them:

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