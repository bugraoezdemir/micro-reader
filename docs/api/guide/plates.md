# Plates (HCS)

High-content screening plates open like any other file: as a `File` whose images you can read. In addition, `f.plates` describes the plate layout, and you can walk from a plate to its wells, fields and images.

| Format | Open |
| --- | --- |
| PerkinElmer / Revvity Harmony (Operetta, Opera Phenix) | `Images/Index.idx.xml`, or the measurement folder |
| PerkinElmer Columbus | `ImageIndex.ColumbusIDX.xml`, `MeasurementIndex.ColumbusIDX.xml`, or the export's folder |
| PerkinElmer Opera Flex | any `.flex` file, or the folder of `.flex` files |
| Molecular Devices MetaXpress | the plate's `.HTD` file |
| Molecular Devices ImageXpress HCS.ai | the `.jdce` file |
| GE / Cytiva IN Cell Analyzer | the `.xdce` file |
| Yokogawa CellVoyager (CV7000, CV8000) | the measurement folder, or its `.wpi` / `.mrf` / `.mlf` |
| Thermo Fisher Cellomics | any `.C01` file, or the folder |
| Olympus / Evident scanR | `experiment_descriptor.xml`, or the experiment folder |
| Zeiss CZI | the `.czi` file |
| Nikon ND2 | the `.nd2` file, or any file of a well-plate run |
| OME companion file, OME-TIFF (including Yokogawa CQ1) | the file |

## Plate indices

On a plate, images are indexed by `plate`, `acquisition`, `well` and `field` instead of `series`:

```python
f = micro_reader.open("Measurement1/Images/Index.idx.xml")
f.index_range
# {'plate': 1, 'acquisition': 1, 'well': 3, 'field': 2}

f.images[0].index
# {'plate': 0, 'acquisition': 0, 'well': 0, 'field': 0}
```

- `plate` and `acquisition` are 0 when the file holds one of each.
- `well` is a position in `f.plates[p].wells`, not a row or column number.
- `field` is the field number of the plate format, counting from 0.
- Tiles, views and other indices add to these, as in any file.

## The layout

```python
plate = f.plates[0]
plate
# <micro_reader.Plate 'HipSci Experiment 10 Plate 1' 8 x 12, 3 wells with images>

plate.name            # 'HipSci Experiment 10 Plate 1' ('' when the file has none)
plate.row_names       # ('A', 'B', 'C', 'D', 'E', 'F', 'G', 'H')
plate.column_names    # ('1', '2', ..., '12')
plate.acquisitions    # (Acquisition(id=0, name='...'), ...): empty when not recorded

plate.wells           # only the wells that hold images
plate.wells[0]
# Well(row=0, column=0, name='A1')
```

Rows and columns count from 0, as in OME-Zarr. `plate.row_names[well.row]` is a well's row name. A well's `name` is its row letter(s) followed by its column, for example `"B2"`. Plates with more than 26 rows continue `Y, Z, AA, AB, ...`.

## Navigating to an image

Containment indices (plate › well › field › tile) are navigated by attribute:

```python
well = plate.wells[0]
field = well.fields[0]           # a Field: <micro_reader.Field 0: 1 images>
image = field.image              # the field's one image
image.read(c=0).shape            # (1, 1, 1024, 1360)
```

| From | Attribute | Gives |
| --- | --- | --- |
| `File` | `plates` | list of `Plate` |
| `Plate` | `wells` | tuple of `Well` that have images |
| `Well` | `fields` | tuple of `Field` that have images, in field order |
| `Field` | `image` | the one image, when the field has exactly one |
| `Field` | `tiles` | the images in tile order, for tiled fields |
| `Field` | `images` | every image of the field, as stored |
| `Field` | `select(**index)` | the field with only the matching images |

These are the same `Image` objects as in `f.images`. Navigation and `select()` lead to the same images.

Every level knows what it is and what holds it: `kind` is `"file"`, `"plate"`, `"well"` or `"field"`, and `parent` is the level above. Accessors only go down: asking a well for `plates` raises an `AttributeError` that says what a well holds.

```python
well.kind          # 'well'
well.parent is plate    # True
```

## Selecting images

`f.select()` takes a plate's places by name or by position, one keyword each:

| Place | By position | By name |
| --- | --- | --- |
| plate | `plate=0` | `plate_name="Plate 1"` |
| acquisition | `acquisition=0` | `acquisition_name="..."` |
| well | `well=12` | `well_name="C3"` |
| row | `row=2` | `row_name="C"` |
| column | `column=2` | `column_name="3"` |

Positions count from 0. `well` is the well's place in `plate.wells`. `row` and `column` are 0-based numbers (`column=2` is the column named "3"), as in `well.row` and `well.column`. A name given to a position keyword is taken as its name, so `well="C3"` is `well_name="C3"`.

```python
f.select(well="C3")                        # well C3
f.select(row="C", column="3")              # the same well, by row and column
f.select(row=["B", "C"], column=range(2, 5))   # rows B and C, columns 3 to 5
f.select(well=["B2", "C5"], field=[0, 1])      # two wells, their first two fields
f.select(row=slice(2, None))               # row C onward
f.select(column=slice(0, None, 2))         # every second column
```

- A value can be a single value, or a list, tuple, range or set meaning any of them.
- Slices select positions, like a range. They need 0 or positive bounds, and a step of at least 1.
- `row` and `column` are another address of a well, so they do not go with `well`.
- A name given to a name keyword must be text: `well_name=3` raises a `TypeError` that names `well=3`.
- Plate places need a file opened as a plate. On a CZI or ND2 file opened with `plates=False`, they raise a `ValueError` that says so.
- A `Well`, `Field` or `Plate` object is not an index value. Give `plate.well_index("C3")` or the name.

`Field.select()` takes lists, ranges and slices of the other indices too.

### A part of a plate

`plate.rows` and `plate.columns` cut out a part of a plate. The part is still a plate:

```python
part = plate.rows[2:5].columns[3:7]      # rows C to E, columns 4 to 7
part.is_part                             # True
len(part.wells)                          # the wells of the part that hold images
```

Inside the brackets you can use an index, a slice, a name or a list of them. The part keeps the whole plate's positions: `part.well_index("D5")` is the same number as in the whole plate.

### Finding a well by name

```python
i = plate.well_index("B2")       # KeyError if B2 has no images
plate.wells[i]
f.select(well=i)                 # every image of well B2
f.select(well="B2", field=3)     # the same, by name; one field
```

### Tiled fields and orthogonal indices

A field can hold several images: mosaic tiles, or views, acquisitions and other orthogonal indices.

```python
f = micro_reader.open("W96_B2+B4.czi")
field = f.plates[0].wells[0].fields[0]
len(field.tiles)                 # 45: tiles 0-44
field.tiles[7].index             # {'plate': 0, 'acquisition': 0, 'well': 0, 'field': 0, 'tile': 7}
```

`field.image` raises `ValueError` when there is more than one image. `field.tiles` raises `ValueError` when an orthogonal index also varies in the field, because tile order alone does not identify an image then. Pick one value first:

```python
field.select(view=1).tiles[7]
field.select(acquisition=0).image
```

## Walking a whole plate

```python
with micro_reader.open("plate/Index.idx.xml") as f:
    for p, plate in enumerate(f.plates):
        for well in plate.wells:
            for field in well.fields:
                im = field.image
                print(plate.name, well.name, field.index, im.shape)
```

Or, without the layout, from the indices alone:

```python
plate = f.plates[0]
for im in f.images:
    well = plate.wells[im.index["well"]]
    print(well.name, im.index["field"], im.shape)
```

## Missing and wrong plane files

Plates are made of many plane files, and some may be missing:

- A field with **some** of its plane files is read with zeros in place of the missing planes.
- A field with **none** of its files is left out.
- Both cases are reported once, when the file is opened, as a `MissingImagesWarning`:

```text
MissingImagesWarning: '.../Index.idx.xml': 2564 image files are missing
(e.g. 'r01c01f03p01-ch1sk1fk1fl1.tiff'): 641 fields have no files and are left out (of 647 fields)
```

A plane file whose size or pixel type differs from what the index says raises `CorruptFileError` when it is read. Reading checks this plane by plane, at no extra cost. To find such files before you start a long conversion, check them all at once:

```python
checked = f.check_members()      # opens every plane file once
print(checked, "plane files fit the index")
```

`check_members()` returns the number of files it checked (0 for a file that is not a file set). It raises `CorruptFileError` naming up to five files that do not fit, and how many in all.

## Ragged fields

Sometimes the channels of a field are not sampled the same way. For example, one channel is a z-stack and the others are single planes. Such an image is *ragged*:

An IN Cell plate where one channel is a 16-plane z-stack and the other three are single planes:

```text
<micro_reader.File 'Training_Demo_1.xdce' backend='incell' format='incell' images=4 plates=1>
  plate 0 'Training_Demo_1': 8 x 12, 2 wells with images
    well C5
      field 0: tczyx (1, 4, 16, 1024, 1024) uint16, ragged: (1, 3, 1, 1024, 1024) + (1, 1, 16, 1024, 1024)
      ...
```

```python
im = f.images[0]
im.is_ragged                     # True
im.shape                         # (1, 4, 16, 1024, 1024): the whole
[p.shape for p in im.parts]
# [(1, 3, 1, 1024, 1024), (1, 1, 16, 1024, 1024)]
im.parts[1].index
# {'plate': 0, 'acquisition': 0, 'well': 0, 'field': 0, 'part': 1}
```

Each part is an ordinary image holding the channels that share one sampling, with its own `z` and `t` and its own metadata. Its index has a `part` entry, and `part.whole` leads back to the whole image.

The whole image reads as one array, with zeros where a channel has no plane, when the parts lie on a common grid. When they do not, reading the whole raises an error, and you read the parts instead:

```python
for part in im.parts:
    data = part.read()
    z_step = part.metadata.pixel_sizes.get("z")
```

## CZI plates

A Zeiss CZI recorded on a plate opens as a plate. Use `plates=False` to read it by its series instead. The same holds for ND2 files whose positions are named after wells, folders of ND2 files from a NIS-Elements well-plate run, and OME companion files or OME-TIFFs that describe a plate:

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
