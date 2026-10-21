"""The import guard against eubi-bridge releases that read the old index name (plan 8.28, E2).

micro-reader 0.1.0 renames the image index "scene" to "series". eubi-bridge 0.1.3 up to 0.2.x
and eubi-bridge-lite below 0.3.0 still read "scene": full eubi-bridge would switch silently to
its older readers, eubi-bridge-lite would fail every file with a wrong message. So
``import micro_reader`` refuses to load beside them, with an ``ImportError`` that names both
versions and both ways out.

The versions come from the installed distributions' metadata (``importlib.metadata``); eubi
itself is never imported. An editable install is checked like any other (its recorded version
may be stale: the message then says to reinstall it). Only ``MICRO_READER_SKIP_EUBI_CHECK=1``
skips the check, with one warning.

Removal: when every eubi-bridge / eubi-bridge-lite line below 0.3.0 has left the support
window, checked by the release version matrix (release policy rule 4); not before.
"""
import os as _os
import re as _re
import warnings as _warnings

from ._version import __version__

#: the first eubi-bridge / eubi-bridge-lite release that reads "series" (T, decision E2)
FIRST_SERIES_EUBI = "0.3.0"
#: the distributions checked, the spellings tried, and the lowest version that reads micro-reader
#: (eubi-bridge 0.1.0-0.1.2 have no micro-reader dependency; every eubi-bridge-lite has one)
CHECKED = (("eubi-bridge", ("eubi-bridge", "eubi_bridge"), "0.1.3"),
           ("eubi-bridge-lite", ("eubi-bridge-lite", "eubi_bridge_lite"), "0"))
SKIP_ENV = "MICRO_READER_SKIP_EUBI_CHECK"

_PRE = {"a": 0, "alpha": 0, "b": 1, "beta": 1, "c": 2, "rc": 2, "pre": 2, "preview": 2}
_VERSION = _re.compile(
    r"v?(?:(?P<epoch>\d+)!)?(?P<release>\d+(?:\.\d+)*)"
    r"(?:[-_.]?(?P<pre>alpha|a|beta|b|preview|pre|c|rc)[-_.]?(?P<pre_n>\d*))?"
    r"(?:-(?P<post_n1>\d+)|[-_.]?(?:post|rev|r)[-_.]?(?P<post_n2>\d*))?"
    r"(?:[-_.]?dev[-_.]?(?P<dev_n>\d*))?"
    r"(?:\+[a-z0-9]+(?:[-_.][a-z0-9]+)*)?",
    _re.IGNORECASE)
_warned = set()     # one warning per process for each reason


def version_key(version):
    """A sortable key for a PEP 440 version, or None when *version* is not one.

    Release numbers first (``0.3`` equals ``0.3.0``), then ``.devN`` < ``aN`` < ``bN`` <
    ``rcN`` < the final release < ``.postN``; a dev release of a pre or post release sorts just
    before it. Local versions (``+...``) are ignored. No ``packaging`` dependency.
    """
    m = _VERSION.fullmatch(version.strip()) if isinstance(version, str) else None
    if m is None:
        return None
    release = tuple(int(p) for p in m["release"].split("."))
    while len(release) > 1 and release[-1] == 0:
        release = release[:-1]
    post = m["post_n1"] or m["post_n2"]
    has_post = m["post_n1"] is not None or m["post_n2"] is not None
    has_dev = m["dev_n"] is not None
    if m["pre"]:
        pre = (_PRE[m["pre"].lower()], int(m["pre_n"] or 0))
    elif has_dev and not has_post:
        pre = (-1, 0)                   # 0.3.0.dev0 sorts before 0.3.0a1
    else:
        pre = (3, 0)                    # the final release
    return (int(m["epoch"] or 0), release, pre, (int(post or 0),) if has_post else (-1,),
            (0, int(m["dev_n"] or 0)) if has_dev else (1, 0))


def _warn_once(reason, message):
    if reason not in _warned:
        _warned.add(reason)
        _warnings.warn(message, UserWarning, stacklevel=4)


def may_be_installed():
    """False only when no eubi distribution can be found on this interpreter's path.

    Importing ``importlib.metadata`` and asking it costs about 35 ms, a quarter of
    ``import micro_reader``, paid again by every worker process. So the folders on ``sys.path``
    are listed first (about 1 ms), the way ``importlib.metadata`` itself searches them: a
    ``*.dist-info``, ``*.egg-info`` or ``*.egg`` child, or a path entry, whose name starts with
    "eubi" (any case or separator) means "maybe". Whatever this listing cannot see answers
    "maybe" too: a path entry that is a file (a zip), a folder that cannot be listed, and any
    import finder other than the standard one that can report distributions."""
    import sys
    from importlib.machinery import PathFinder
    for finder in sys.meta_path:
        if finder is not PathFinder and getattr(finder, "find_distributions", None) is not None:
            return True
    for entry in sys.path:
        if not isinstance(entry, str):
            return True
        root = entry or "."
        try:
            children = _os.listdir(root)
        except FileNotFoundError:
            continue
        except OSError:                 # a file (zip) or a folder we may not list
            return True
        if _os.path.basename(root.rstrip("\\/")).lower().startswith("eubi"):
            return True
        for child in children:
            low = child.lower()
            if low.startswith("eubi") and low.endswith((".dist-info", ".egg-info", ".egg")):
                return True
    return False


def _distribution(spellings):
    import importlib.metadata as _md   # here, not at the top: most imports never need it
    for name in spellings:
        try:
            return _md.distribution(name)
        except _md.PackageNotFoundError:
            continue
    return None


def _editable_folder(dist):
    """The source folder of an editable install ("" when it is editable but the folder is not
    recorded), or None when *dist* is not an editable install."""
    try:
        text = dist.read_text("direct_url.json")
    except OSError:
        return None
    if not text:
        return None
    import json
    try:
        record = json.loads(text)
    except ValueError:
        return None
    if not (isinstance(record, dict) and isinstance(record.get("dir_info"), dict)
            and record["dir_info"].get("editable") is True):
        return None
    url = record.get("url")
    if not (isinstance(url, str) and url.startswith("file:")):
        return ""
    from urllib.parse import urlparse
    from urllib.request import url2pathname
    return url2pathname(urlparse(url).path)


def refusal(name, version, editable_folder=None):
    """The ImportError message for *name* *version* (an editable install adds how to reinstall)."""
    text = (f"{name} {version} does not support micro-reader {__version__}: this micro-reader names "
            f"the image index 'series' (it was 'scene'), and {name} {version} reads 'scene'. "
            f"Either upgrade: pip install -U '{name}>={FIRST_SERIES_EUBI}', or keep {name} {version} "
            f"with the micro-reader it was built for: pip install 'micro-reader<0.1'. "
            f"(Working on an eubi checkout? Set {SKIP_ENV}=1 to skip this check.)")
    if editable_folder is not None:
        if not editable_folder:             # not recorded: a bare placeholder, never quoted
            folder = "<its folder>"
        elif " " in editable_folder:
            folder = f'"{editable_folder}"'
        else:
            folder = editable_folder
        text += (f" {name} is an editable install: its recorded version may be stale; reinstall it "
                 f"(pip install -e {folder}) so the version matches, then import again.")
    return text


def check():
    """Raise ImportError when an installed eubi-bridge / eubi-bridge-lite reads "scene"."""
    if _os.environ.get(SKIP_ENV) == "1":
        _warn_once("skip", f"micro-reader {__version__}: {SKIP_ENV}=1 is set, so the check for "
                           f"eubi-bridge / eubi-bridge-lite releases below {FIRST_SERIES_EUBI} (which "
                           f"read the old index name 'scene') is skipped. Unset it outside eubi "
                           f"development.")
        return
    if not may_be_installed():
        return
    below = version_key(FIRST_SERIES_EUBI + ".dev0")    # dev builds of the first series release pass
    for name, spellings, lowest in CHECKED:
        dist = _distribution(spellings)
        if dist is None:
            continue
        version = dist.version
        key = version_key(version)
        if key is None:
            what = (f"cannot read the version {version!r} of the installed {name}"
                    if version is not None else f"the installed {name} records no version")
            _warn_once(("unparseable", name),
                       f"micro-reader {__version__}: {what}, so it is not checked for the old "
                       f"index name 'scene'. "
                       f"If it is below {FIRST_SERIES_EUBI}, upgrade it: pip install -U "
                       f"'{name}>={FIRST_SERIES_EUBI}'.")
            continue
        if version_key(lowest) <= key < below:
            raise ImportError(refusal(name, version, _editable_folder(dist)), name="micro_reader")
