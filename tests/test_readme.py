"""The README is micro-reader's user guide in two places: the repository's
top-level README.md (GitHub) and python/README.md (PyPI; hatchling cannot
take a readme from outside the package folder).  python/README.md must be
exactly the top-level README above its "Repository and development"
section."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MARKER = "\n## Repository and development\n"


def test_the_pypi_readme_is_the_user_guide():
    top = (ROOT / "README.md").read_text(encoding="utf-8")
    assert MARKER in top, f"README.md has no {MARKER.strip()!r} section"
    guide = top.split(MARKER)[0].rstrip("\n") + "\n"
    pypi = (ROOT / "python" / "README.md").read_text(encoding="utf-8")
    assert pypi == guide, (
        "python/README.md differs from the user guide in README.md: edit README.md, then "
        "copy everything above '## Repository and development' to python/README.md")
