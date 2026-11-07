# Plate classes

The plate layout of a high-content screening file, from `File.plates`. All classes are immutable. See the [Plates guide](../guide/plates.md).

```text
File.plates[p]                 Plate
  .wells[w]                    Well       index "well": w
    .fields[j]                 Field      .index = field number
      .image / .tiles[k]       Image      the same objects as in File.images
```

## Containment

Every level knows what it is and what holds it:

| Level | `kind` | `parent` | Holds |
| --- | --- | --- | --- |
| `File` | `"file"` | `None` | `.plates` |
| `Plate` | `"plate"` | its file | `.wells` |
| `Well` | `"well"` | its plate | `.fields` |
| `Field` | `"field"` | its well | `.images`, `.tiles`, `.image` |

Accessors go down only. An accessor of another level (`well.plates`, `file.wells`, `field.wells`) raises `AttributeError` that names what the object holds and says `.parent` is what holds it. For a plate that was cut with `rows` / `columns` (below), `parent` is still the file.

Rows and columns are a well's address, not levels of containment: a plate does not hold rows.

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
| `name` | The plate's name; `""` when the file has none. |
| `row_names` | Row names of the plate layout: `row_names[i]` names row `i` (e.g. `"A"`). |
| `column_names` | Column names: `column_names[j]` names column `j` (e.g. `"1"`). |
| `wells` | The wells that hold images. Image index `"well": i` is `wells[i]` of the whole plate. |
| `acquisitions` | The plate's imaging runs. Empty when the file does not record them. |
| `parent` | The [`File`](file.md) that holds the plate. |
| `whole` | For a part of a plate (below): the whole plate's wells. Empty for a whole plate. |

| Property | Description |
| --- | --- |
| `kind` | `"plate"`. |
| `rows` | A slicer over the plate's rows (below). `len(plate.rows)` is the number of rows of the layout. |
| `columns` | A slicer over the plate's columns. |
| `is_part` | `True` for a plate cut with `rows` / `columns` that holds fewer wells than the whole. |

!!! warning "0.3.0: `rows` and `columns` changed meaning"
    Before 0.3.0, `Plate.rows` and `Plate.columns` held the row and column *names*. Those are now `Plate.row_names` and `Plate.column_names`. Change `plate.rows[well.row]` to `plate.row_names[well.row]`. `len(plate.rows)` still gives the number of rows. Iterating `plate.rows` raises `TypeError` that points to `row_names`.

The layout is the plate's own, not just the wells that hold images. A layout the file describes too small for its wells is grown to hold them, with a [`PlateLayoutWarning`](errors.md#platelayoutwarning).

### Cutting a plate: `rows` and `columns`

```python
Plate.rows[key] -> Plate
Plate.columns[key] -> Plate
```

The part of the plate in those rows (or columns), still a plate. `key` is a position, a `slice`, a name or a list of them:

```python
plate.rows[2]                      # row C
plate.rows[2:5]                    # rows C, D, E
plate.rows["B"]                    # by name
plate.rows[["B", "D"]]             # rows B and D
plate.rows[-2:]                    # the last two rows (negative bounds count from the layout's end)
plate.rows[2:5].columns[3:7]       # rows C-E, columns 4-7: the two combine
```

Positions are the whole layout's, so a part's `rows[3]` is still row D. A name that is not in the layout raises `KeyError`; a position out of range `IndexError`. The result's `wells` are the wells of the part that hold images; `well_index` answers with the index the images carry. `File.select(row=..., column=...)` picks the same wells' images.

### `well_index`

```python
Plate.well_index(name) -> int
```

The index of the well called `name` (e.g. `"B2"`) as its images carry it, for `File.select(well=...)`. For a part of a plate it is still the place in the whole plate's wells. Raises `KeyError` listing the wells with images when there is no such well.

```python
f.select(well=plate.well_index("B2"))
f.select(well="B2")                    # the same
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
| `row` | Row, counting from 0. |
| `column` | Column, counting from 0. |
| `name` | Row name + column name, e.g. `"B2"` for row 1, column 1. |
| `fields` | The fields that have images, in field order. `fields[j]` is the j-th field present. Its `.index` is its field number. |
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
| `index` | The field number, the same as the images' `"field"` index. |
| `images` | All images of the field (full resolution), in `File.images` order. |
| `parent` | The well that holds the field. |
| `image` | The field's one image (a field that is not tiled, or placed as one mosaic with `tiles="mosaic"`). Raises `ValueError` when the field has more than one: use `tiles` or `select()`. |
| `tiles` | The images in tile order. `tiles[k]` is the k-th tile present, and `.index["tile"]` is its tile index. Raises `ValueError` when an orthogonal index (view, acquisition, illumination, ...) also varies in the field. |
| `select(**index) -> Field` | This field with only the images whose index matches, e.g. `field.select(view=1).tiles[7]`. A value may be an `int`, a `list`, `tuple`, `range` or `set` of them (any of them), or a `slice` of positions, as in [`File.select`](file.md#select). Raises `KeyError` for an unknown index name (a plate's `row` / `column` are not accepted: a field sits in one well). |

`kind` is `"field"`.

## Acquisition

```python
@dataclass(frozen=True)
class micro_reader.Acquisition:
    id: int
    name: str = ""
```

One imaging run of a plate. Images carry the run in their `"acquisition"` index. `File.select(acquisition_name=...)` picks by `name`.
