# Grouping classes

The result of [`group_files()`](functions.md#group_files). See [Grouping many files](../guide/multi-file.md#grouping-many-files).

## Grouping

```python
@dataclass(frozen=True)
class micro_reader.Grouping:
    groups: list[FileGroup]
    singles: list[str]
```

| Field | Description |
| --- | --- |
| `groups` | One `FileGroup` for each multi-file image or index file found, sorted by `expected`. |
| `singles` | Every other file, as given, sorted. Each one is opened on its own, or is not an image at all. `group_files` does not read them. |

## FileGroup

```python
@dataclass(frozen=True)
class micro_reader.FileGroup:
    kind: str
    main: str | None
    expected: str
    format: str | None = None
    parts: tuple[str, ...] = ()
    members: frozenset[str] = frozenset()
    notes: tuple[str, ...] = ()
```

| Field | Description |
| --- | --- |
| `kind` | `"composite"` is a main file and its parts, found by the naming rules of the format. `"index"` is an index file and the files its images are read from. |
| `main` | The file to pass to `open()`. It is `None` when the main file of a composite is missing. |
| `expected` | Where the main file is, or where it should be. |
| `format` | The backend that reads the group, for example `"oif"`, `"dv"` or `"harmony"`. |
| `parts` | The given files that belong to `main`, not including `main` itself. For an index, it also holds other index files of the same images, such as the CellVoyager `.wpi`, `.mrf` and `.mlf` files, and side files. |
| `members` | **Index only.** Every file that its images are read from, including files you did not give. |
| `notes` | **Index only.** What opening the index said, which is the missing files or the reason it could not be read. |

```python
for grp in g.groups:
    if grp.kind == "index" and grp.notes:
        print(grp.main, "->", *grp.notes, sep="\n  ")
```
