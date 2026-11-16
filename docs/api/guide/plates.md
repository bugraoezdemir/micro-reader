# Plates (HCS)

High-content screening (HCS) plates open like any other image file, as a `File` whose images you can read. In addition, `f.plates` describes the plate layout and lets you navigate from a plate to its wells, fields, and images.

| Format | Open |
| --- | --- |
| PerkinElmer / Revvity Harmony (Operetta, Opera Phenix) | `Images/Index.idx.xml` or the measurement folder |
| PerkinElmer Columbus | `ImageIndex.ColumbusIDX.xml`, `MeasurementIndex.ColumbusIDX.xml`, or the export folder |
| PerkinElmer Opera Flex | Any `.flex` file or the folder containing the `.flex` files |
| Molecular Devices MetaXpress | The plate's `.HTD` file |
| Molecular Devices ImageXpress HCS.ai | The `.jdce` file |
| GE / Cytiva IN Cell Analyzer | The `.xdce` file |
| Yokogawa CellVoyager (CV7000, CV8000) | The measurement folder or its `.wpi`, `.mrf`, or `.mlf` file |
| Thermo Fisher Cellomics | Any `.C01` file or the folder |
| Olympus / Evident scanR | `experiment_descriptor.xml` or the experiment folder |
| Zeiss CZI | The `.czi` file |
| Nikon ND2 | The `.nd2` file or any file from a well-plate run |
| OME companion file, OME-TIFF (including Yokogawa CQ1) | The file |

## Plate indices

In a plate, images are indexed by `plate`, `acquisition`, `well`, and `field` instead of `series`:

```python
f = micro_reader.open("Measurement1/Images/Index.idx.xml")
f.index_range
# {'plate': 1, 'acquisition': 1, 'well': 3, 'field': 2}

f.images[0].index
# {'plate': 0, 'acquisition': 0, 'well': 0, 'field': 0}
```

- `plate` and `acquisition` are both 0 when the file contains only one of each.
- `well` is the well's position in `f.plates[p].wells`, not its row or column number.
- `field` is the field number defined by the plate format, starting at 0.
- Other indices, such as tiles and views, are added as needed, just as they are for other files.

## The layout

```python
plate = f.plates[0]
plate
# <micro_reader.Plate 'HipSci Experiment 10 Plate 1' 8 x 12, 3 wells with images>

plate.name            # 'HipSci Experiment 10 Plate 1' ('' if the file has no name)
plate.row_names       # ('A', 'B', 'C', 'D', 'E', 'F', 'G', 'H')
plate.column_names    # ('1', '2', ..., '12')
plate.acquisitions    # (Acquisition(id=0, name='...'), ...), empty if not recorded

plate.wells           # only wells that contain images
plate.wells[0]
# Well(row=0, column=0, name='A1')
```

Rows and columns are numbered from 0, as in OME-Zarr. `plate.row_names[well.row]` gives a well's row name. A well's `name` consists of its row letter(s) followed by its column number, for example `"B2"`. Plates with more than 26 rows continue with `Y`, `Z`, `AA`, `AB`, and so on.

## Navigating to an image

You can navigate through the plate hierarchy using attributes:

```python
well = plate.wells[0]
field = well.fields[0]           # a Field: <micro_reader.Field 0: 1 images>
image = field.image              # the field's one image
image.read(c=0).shape            # (1, 1, 1024, 1360)
```

| From | Attribute | Gives |
| --- | --- | --- |
| `File` | `plates` | List of `Plate` objects |
| `Plate` | `wells` | Tuple of `Well` objects that contain images |
| `Well` | `fields` | Tuple of `Field` objects that contain images, in field order |
| `Field` | `image` | The single image, if the field contains exactly one |
| `Field` | `tiles` | Images in tile order, for tiled fields |
| `Field` | `images` | All images in the field, in storage order |
| `Field` | `select(**index)` | The field containing only images that match the given indices |

These are the same `Image` objects available through `f.images`, so navigation and `select()` lead to the same images.

Each level identifies its own type and parent. The `kind` attribute is `"file"`, `"plate"`, `"well"`, or `"field"`, and `parent` refers to the level above. Accessors navigate down the hierarchy only. For example, asking a well for `plates` raises an `AttributeError` explaining what that well contains.

```python
well.kind               # 'well'
well.parent is plate    # True
```

## Selecting images

`f.select()` lets you select images by plate location, using either names or positions. Each keyword identifies one location:

| Place | By position | By name |
| --- | --- | --- |
| Plate | `plate=0` | `plate_name="Plate 1"` |
| Acquisition | `acquisition=0` | `acquisition_name="..."` |
| Well | `well=12` | `well_name="C3"` |
| Row | `row=2` | `row_name="C"` |
| Column | `column=2` | `column_name="3"` |

Positions start at 0. `well` is the well's position in `plate.wells`. `row` and `column` are zero-based numbers, matching `well.row` and `well.column`. For example, `column=2` selects the column named `"3"`.

A name can also be passed to a positional keyword. For example, `well="C3"` is equivalent to `well_name="C3"`.

```python
f.select(plate=0, well="C3")                            # well C3
f.select(plate=0, row="C", column="3")                  # the same well, by row and column
f.select(plate=0, row=["B", "C"], column=range(2, 5))   # rows B and C, columns 3 to 5
f.select(plate=0, well=["B2", "C5"], field=[0, 1])      # two wells, their first two fields
f.select(plate=0, row=slice(2, None))                   # row C onward
f.select(plate=0, column=slice(0, None, 2))             # every second column
```

Selection values and rules:

- A value can be a single item or a list, tuple, range, or set. Multiple values select images matching any of them.
- Slices select positions, like ranges. Bounds must be zero or positive, and the step must be at least 1.
- `row` and `column` provide another way to address a well, so they cannot be combined with `well`.
- A name keyword requires text. For example, `well_name=3` raises a `TypeError` that points to `well=3`.
- Plate locations are available only when the file is opened as a plate. On a CZI or ND2 file opened with `plates=False`, selecting by plate location raises a `ValueError` explaining this.
- A `Well`, `Field`, or `Plate` object is not a valid index value. Pass the name or use `plate.well_index("C3")` to get the well's index.

!!! warning "Name the plate too"

    `select()` matches every level you do not name. Without `plate` or `plate_name`, `well="C3"` selects well C3 in every plate the file holds, and without `acquisition`, in every acquisition. In code that may meet files with several plates, name every level you want to search within:

    ```python
    plate = f.plates[0]
    plate.name                                     # 'Plate 1'

    f.select(plate_name=plate.name, well_name="C3")
    f.select(plate=0, acquisition=0, row="C", column=range(2, 5))
    ```

    A plate whose file records no name has the name `""`. Select it by position instead, with `plate=0`.

`Field.select()` also accepts lists, ranges, and slices for the other indices.

### A part of a plate

`plate.rows` and `plate.columns` let you select a region of a plate. The result is itself a plate:

```python
part = plate.rows[2:5].columns[3:7]  # rows C to E, columns 4 to 7
part.is_part                          # True
len(part.wells)                       # wells in this part that contain images
```

You can use an index, slice, name, or list of these inside the brackets. The part retains the positions of the original plate, so `part.well_index("D5")` returns the same index as it does for the full plate.

### Finding a well by name

```python
plate = f.plates[0]
i = plate.well_index("B2")                 # KeyError if B2 has no images
plate.wells[i]
f.select(plate=0, well=i)                  # every image in well B2
f.select(plate=0, well="B2", field=3)      # the same well by name, selecting one field
```

### Tiled fields and orthogonal indices

A field can contain multiple images, such as mosaic tiles, views, acquisitions, or other independent indices.

```python
f = micro_reader.open("W96_B2+B4.czi")
field = f.plates[0].wells[0].fields[0]
len(field.tiles)                 # 45 (tiles 0-44)
field.tiles[7].index             # {'plate': 0, 'acquisition': 0, 'well': 0, 'field': 0, 'tile': 7}
```

`field.image` raises a `ValueError` if the field contains more than one image. `field.tiles` raises a `ValueError` if another index also varies within the field, because tile order alone is not enough to identify an image. Select a value for the other index first:

```python
field.select(view=1).tiles[7]
field.select(acquisition=0).image
```

## Walking a whole plate

You can iterate over the plate hierarchy to process every image:

```python
with micro_reader.open("plate/Index.idx.xml") as f:
    for p, plate in enumerate(f.plates):
        for well in plate.wells:
            for field in well.fields:
                im = field.image
                print(plate.name, well.name, field.index, im.shape)
```

You can also work directly with image indices, without navigating the layout:

```python
plate = f.plates[0]
for im in f.images:
    well = plate.wells[im.index["well"]]
    print(well.name, im.index["field"], im.shape)
```

## Missing and incorrect plane files

Plate datasets contain many plane files, and some may be missing. If a field has **some** of its plane files, it is read with zeros in place of the missing planes. If a field has **none** of its files, it is omitted.

Both cases are reported once, when the file is opened, through a `MissingImagesWarning`:

```text
MissingImagesWarning: '.../Index.idx.xml': 2564 image files are missing
(e.g. 'r01c01f03p01-ch1sk1fk1fl1.tiff'): 641 fields have no files and are left out (of 647 fields)
```

A plane file whose size or pixel type differs from what the index specifies raises `CorruptFileError` when it is read. Reads check files plane by plane. To identify problematic files before starting a long conversion, check all members at once:

```python
checked = f.check_members()  # opens every plane file once
print(checked, "plane files fit the index")
```

`check_members()` returns the number of files checked, or 0 if the file is not a file set. It raises `CorruptFileError` if files do not match their descriptions, naming up to five affected files and reporting the total number.

## Ragged fields

In some datasets, channels within a field are sampled differently. For example, one channel may contain a z-stack while the others contain only single planes. Such an image is called *ragged*.

The following IN Cell plate has one channel with a 16-plane z-stack and three channels with a single plane each:

```text
<micro_reader.File 'Training_Demo_1.xdce' backend='incell' format='incell' images=4 plates=1>
  plate 0 'Training_Demo_1': 8 x 12, 2 wells with images
    well C5
      field 0: tczyx (1, 4, 16, 1024, 1024) uint16, ragged: (1, 3, 1, 1024, 1024) + (1, 1, 16, 1024, 1024)
      ...
```

```python
im = f.images[0]
im.is_ragged                      # True
im.shape                          # (1, 4, 16, 1024, 1024), the whole image
[p.shape for p in im.parts]
# [(1, 3, 1, 1024, 1024), (1, 1, 16, 1024, 1024)]
im.parts[1].index
# {'plate': 0, 'acquisition': 0, 'well': 0, 'field': 0, 'part': 1}
```

Each part is an ordinary image containing the channels that share the same sampling. Each has its own `z` and `t` dimensions and its own metadata. Its index includes a `part` entry, and `part.whole` refers back to the complete image.

If all parts lie on a common grid, the whole image can be read as one array, with zeros where a channel has no plane. If they do not share a common grid, reading the whole image raises an error, and you must read the parts individually:

```python
for part in im.parts:
    data = part.read()
    z_step = part.metadata.pixel_sizes.get("z")
```

## CZI plates

A Zeiss CZI recorded on a plate opens as a plate by default. Pass `plates=False` to read it by series instead.

The same behaviour applies to ND2 files whose positions are named after wells, folders of ND2 files from NIS-Elements well-plate runs, and OME companion files or OME-TIFFs that describe a plate.

=== "plates=False"

    ```python
    f = micro_reader.open("W96_B2+B4.czi")
    f.index_range        # {'series': 2, 'tile': 45}
    ```

=== "plates=True (default)"

    ```python
    f = micro_reader.open("W96_B2+B4.czi")
    print(f)
    ```

    ```text
    <micro_reader.File 'W96_B2+B4.czi' backend='czi' format='czi' images=90 plates=1>
      plate 0: 8 x 12, 2 wells with images
        well B2
          field 0: tiles 0-44, tczyx (2, 3, 4, 420, 640) uint8
        well B4
          field 0: tiles 0-44, tczyx (2, 3, 4, 420, 640) uint8
    ```