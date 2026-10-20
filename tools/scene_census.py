"""Rename census for plan 7.1 / 7.4 / 7.7 / 7.9: where the word "scene" is, by kind.

    python -I scene_census.py <root> [<root> ...]

Kinds: KEY = a quoted "scene" / 'scene' or a scene= keyword (an index key or select
argument: must be renamed or guarded); WORD = any other "scene" (docs, comments, names
like scene_index; reviewed by hand). Counts files and occurrences per top folder.
Exit 1 if --strict is given and any KEY occurrence remains outside the allow-list.

The reference check of plan 9.6 (micro-reader 0.1.0 renamed the index to
"series"): ``python -I tools/scene_census.py python tests harness smoke_tests
--strict`` with CENSUS_ALLOW a regex of the KEY sites allowed to keep the old
name, as ``<path relative to its root>:<line>`` (the index guard, its tests,
other readers' own keywords); docs/RENAME_NOTE.md has the current list.
"""
import os, re, sys, collections

KEY = re.compile(r"""["']scene["']|\bscene\s*=(?!=)""")
WORD = re.compile(r"scene", re.I)
SKIP_DIRS = {".git", "__pycache__", "build", "dist", "site", "legacy_0_1", ".venv", "venv"}
EXTS = {".py", ".md", ".json", ".csv", ".txt", ".yml", ".yaml", ".toml", ".ini", ".html"}
ALLOW = re.compile(os.environ.get("CENSUS_ALLOW", r"$^"))   # e.g. a regex of allowed file:line


def main(argv):
    strict = "--strict" in argv
    roots = [a for a in argv if not a.startswith("--")]
    bad = 0
    for root in roots:
        files = collections.Counter(); occ = collections.Counter(); keys = []
        for d, dirs, names in os.walk(root):
            dirs[:] = [x for x in dirs if x not in SKIP_DIRS]
            for n in names:
                if os.path.splitext(n)[1].lower() not in EXTS:
                    continue
                p = os.path.join(d, n)
                rel = os.path.relpath(p, root)
                top = rel.split(os.sep)[0] if os.sep in rel else "."
                try:
                    text = open(p, encoding="utf-8", errors="replace").read()
                except OSError:
                    continue
                w = len(WORD.findall(text))
                if not w:
                    continue
                files[top] += 1; occ[top] += w
                if n.endswith(".py"):
                    for i, line in enumerate(text.splitlines(), 1):
                        if KEY.search(line):
                            where = f"{rel}:{i}"
                            keys.append((where, line.strip()[:100]))
                            if not ALLOW.search(where):
                                bad += 1
        print(f"== {root}")
        for top in sorted(files):
            print(f"  {top:28s} {files[top]:5d} files  {occ[top]:6d} occurrences of 'scene' (any case)")
        by_top = collections.Counter(w.split(os.sep)[0] for w, _ in keys)
        print(f"  KEY sites in .py (quoted 'scene' or scene=): {len(keys)} lines; per folder {dict(by_top)}")
        mods = sorted({w.rsplit(':', 1)[0] for w, _ in keys})
        print(f"  .py files with a KEY site: {len(mods)}")
        for m in mods:
            print(f"    {m}")
    if strict and bad:
        print(f"STRICT: {bad} KEY sites outside the allow-list"); return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
