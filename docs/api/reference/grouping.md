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
| `groups` | One `FileGroup` per multi-file image or index file found, sorted by `expected`. |
| `singles` | Every other file, as given, sorted. Each one is opened on its own (or is not an image at all: `group_files` does not read them). |

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
| `kind` | `"composite"`: a main file and its parts, found by the format's naming rules. `"index"`: an index file and the files its images are read from. |
| `main` | The file to pass to `open()`. `None` when a composite's main file is missing. |
| `expected` | Where the main file is, or should be. |
| `format` | The backend that reads the group, e.g. `"oif"`, `"dv"`, `"harmony"`. |
| `parts` | The given files that belong to `main` (not `main` itself). For an index, also other index files of the same images (CellVoyager `.wpi` / `.mrf` / `.mlf`) and side files. |
| `members` | **Index only.** Every file its images are read from, including files you did not give. |
| `notes` | **Index only.** What opening it said: missing files, or why it could not be read. |

```python
for grp in g.groups:
    if grp.kind == "index" and grp.notes:
        print(grp.main, "->", *grp.notes, sep="\n  ")
```
