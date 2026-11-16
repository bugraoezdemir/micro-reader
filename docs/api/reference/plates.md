# Plate classes

The plate layout of a high-content screening file, from `File.plates`. All classes are immutable. See the [Plates guide](../guide/plates.md).

```text
File.plates[p]                 Plate
  .wells[w]                    Well       index "well": w
    .fields[j]                 Field      .index = field number
      .image / .tiles[k]       Image      the same objects as in File.images
```

## Containment

Every level knows what it is and what holds it.

| Level | `kind` | `parent` | Holds |
| --- | --- | --- | --- |
| `File` | `"file"` | `None` | `.plates` |
| `Plate` | `"plate"` | its file | `.wells` |
| `Well` | `"well"` | its plate | `.fields` |
| `Field` | `"field"` | its well | `.images`, `.tiles`, `.image` |

Accessors go down only. An accessor of another level, such as `well.plates`, `file.wells` or `field.wells`, raises `AttributeError`. The message names what the object holds and says that `.parent` is what holds it. For a plate that was cut with `rows` or `columns` (below), `parent` is still the file.

Rows and columns are part of the address of a well. They are not levels of containment, so a plate does not hold rows.

## Plate

```python
@dataclass(frozen=True)
class micro_reader.Plate:
    name: str
    row_names: tuple[str, ...]
    column_names: tuple[str, ...]
    wells: tuple[Well, ...]
    acquisitions: tuple[Acquisition, ...] = ()
    parent: File | None = None
    whole: tuple[Well, ...] = ()
```

| Field | Description |
| --- | --- |
| `name` | The name of the plate. It is `""` when the file has none. |
| `row_names` | The row names of the plate layout. `row_names[i]` names row `i`, for example `"A"`. |
| `column_names` | The column names. `column_names[j]` names column `j`, for example `"1"`. |
| `wells` | The wells that hold images. The image index `"well": i` is `wells[i]` of the whole plate. |
| `acquisitions` | The imaging runs of the plate. This is empty when the file does not record them. |
| `parent` | The [`File`](file.md) that holds the plate. |
| `whole` | For a part of a plate (below), the wells of the whole plate. It is empty for a whole plate. |

| Property | Description |
| --- | --- |
| `kind` | Always `"plate"`. |
| `rows` | A slicer over the rows of the plate (below). `len(plate.rows)` is the number of rows of the layout. |
| `columns` | A slicer over the columns of the plate. |
| `is_part` | `True` for a plate cut with `rows` or `columns` that holds fewer wells than the whole plate. |

!!! warning "0.3.0: `rows` and `columns` changed meaning"
    Before 0.3.0, `Plate.rows` and `Plate.columns` held the row and column *names*. Those are now `Plate.row_names` and `Plate.column_names`. Change `plate.rows[well.row]` to `plate.row_names[well.row]`. `len(plate.rows)` still gives the number of rows. Iterating `plate.rows` raises a `TypeError` that points to `row_names`.

The layout is the layout of the plate itself and not only of the wells that hold images. If the file describes a layout that is too small for its wells, the layout is grown to hold them, and a [`PlateLayoutWarning`](errors.md#platelayoutwarning) is issued.

### Cutting a plate: `rows` and `columns`

```python
Plate.rows[key] -> Plate
Plate.columns[key] -> Plate
```

Returns the part of the plate in those rows or columns, which is still a plate. `key` is a position, a `slice`, a name or a list of them:

```python
plate.rows[2]                      # row C
plate.rows[2:5]                    # rows C, D, E
plate.rows["B"]                    # by name
plate.rows[["B", "D"]]             # rows B and D
plate.rows[-2:]                    # the last two rows (negative bounds count from the end of the layout)
plate.rows[2:5].columns[3:7]       # rows C-E, columns 4-7: the two combine
```

Positions refer to the whole layout, so `rows[3]` of a part is still row D. A name that is not in the layout raises `KeyError`, and a position out of range raises `IndexError`. The `wells` of the result are the wells of the part that hold images. `well_index` answers with the index that the images carry. `File.select(row=..., column=...)` picks the images of the same wells.

### `well_index`

```python
Plate.well_index(name) -> int
```

Returns the index of the well called `name`, for example `"B2"`, as its images carry it. You can pass it to `File.select(well=...)`. For a part of a plate, it is still the place in the wells of the whole plate. It raises `KeyError` when there is no such well, and the message lists the wells with images.

```python
plate = f.plates[0]
f.select(plate=0, well=plate.well_index("B2"))
f.select(plate=0, well="B2")                    # the same
```

## Well

```python
@dataclass(frozen=True)
class micro_reader.Well:
    row: int
    column: int
    name: str
    fields: tuple[Field, ...] = ()
    parent: Plate | None = None
```

| Field | Description |
| --- | --- |
| `row` | The row, counting from 0. |
| `column` | The column, counting from 0. |
| `name` | The row name followed by the column name, for example `"B2"` for row 1, column 1. |
| `fields` | The fields that have images, in field order. `fields[j]` is the j-th field present, and its `.index` is its field number. |
| `parent` | The plate that holds the well. |

`kind` is `"well"`.

## Field

```python
class micro_reader.Field:
    index: int
    images: tuple[Image, ...]
    parent: Well | None
```

One field (site) of a well.

| Member | Description |
| --- | --- |
| `index` | The field number. It is the same as the `"field"` index of the images. |
| `images` | All images of the field at full resolution, in `File.images` order. |
| `parent` | The well that holds the field. |
| `image` | The one image of the field, for a field that is not tiled or that is placed as one mosaic with `tiles="mosaic"`. It raises `ValueError` when the field has more than one image. In that case, use `tiles` or `select()`. |
| `tiles` | The images in tile order. `tiles[k]` is the k-th tile present, and `.index["tile"]` is its tile index. It raises `ValueError` when an orthogonal index such as view, acquisition or illumination also varies in the field. |
| `select(**index) -> Field` | Returns this field with only the images whose index matches, for example `field.select(view=1).tiles[7]`. A value may be an `int`, a `list`, `tuple`, `range` or `set` of them, which means any of them, or a `slice` of positions, as in [`File.select`](file.md#select). It raises `KeyError` for an unknown index name. The `row` and `column` of a plate are not accepted, because a field sits in one well. |

`kind` is `"field"`.

## Acquisition

```python
@dataclass(frozen=True)
class micro_reader.Acquisition:
    id: int
    name: str = ""
```

One imaging run of a plate. The images carry the run in their `"acquisition"` index. `File.select(acquisition_name=...)` picks a run by `name`.
