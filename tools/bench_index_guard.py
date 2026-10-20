"""Cost of the image index guard (plan 3.24) on File.index_range and File.select.

    python -I tools/bench_index_guard.py [--n 100000] [--repeat 7] [--baseline-src DIR]

Times the real methods over *n* images indexed {"series": i // 100, "tile":
i % 100} (a File with a list of Image objects, no file behind it), best of
*repeat*, against two baselines: the micro-reader 0.0.4 code of both methods
(copied below) over plain-dict indices, and -- with --baseline-src, the
python/src folder of a micro-reader 0.0.4 tree -- that tree's own File and
Image (run in a subprocess).  Prints one JSON line per run and the ratios.
"""
import argparse
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(os.path.dirname(HERE), "python", "src")


def best(fn, repeat):
    out = float("inf")
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        out = min(out, time.perf_counter() - t0)
    return round(out * 1e3, 2)


def measure(src: str, n: int, repeat: int) -> dict:
    """index_range / select times (ms) of the micro-reader in *src*."""
    sys.path.insert(0, src)
    from micro_reader import _base
    first = "series" if "series" in _base.INDEX_NAMES else "scene"   # 0.1.0 or 0.0.4
    images = [_base.Image(i, "yx", (1, 1), "u1", index={first: i // 100, "tile": i % 100})
              for i in range(n)]
    f = object.__new__(_base.File)
    f.images = images
    assert f.index_range == {first: -(-n // 100), "tile": min(n, 100)}
    assert len(f.select(**{first: 3, "tile": 5})) == 1
    return {"src": src, "index": type(images[0].index).__name__,
            "index_range_ms": best(lambda: f.index_range, repeat),
            "select_two_ms": best(lambda: f.select(**{first: 3, "tile": 5}), repeat),
            "select_one_ms": best(lambda: f.select(**{first: 3}), repeat)}


# -- micro-reader 0.0.4's File.index_range and File.select, over plain dicts --

class _Plain:
    """An image as 0.0.4 had it: ``index`` a plain dict attribute."""

    def __init__(self, index):
        self.index = index


def _old_index_range(images, names):
    counts = {}
    for name in names:
        values = {image.index[name] for image in images if name in image.index}
        if values:
            counts[name] = len(values)
    return counts


def _old_select(images, names, **index):
    unknown = set(index) - set(names)
    if unknown:
        raise KeyError(f"unknown indices {sorted(unknown)}")
    return [image for image in images
            if all(image.index.get(name) == value for name, value in index.items())]


def measure_old_code(src: str, n: int, repeat: int) -> dict:
    sys.path.insert(0, src)
    from micro_reader import _base
    names = _base.INDEX_NAMES
    images = [_Plain({"series": i // 100, "tile": i % 100}) for i in range(n)]
    return {"src": "0.0.4 code, plain dicts", "index": "dict",
            "index_range_ms": best(lambda: _old_index_range(images, names), repeat),
            "select_two_ms": best(lambda: _old_select(images, names, series=3, tile=5), repeat),
            "select_one_ms": best(lambda: _old_select(images, names, series=3), repeat)}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100_000)
    ap.add_argument("--repeat", type=int, default=7)
    ap.add_argument("--src", default=SRC, help="python/src of the micro-reader measured")
    ap.add_argument("--baseline-src", help="python/src of a micro-reader 0.0.4 tree")
    ap.add_argument("--one", action="store_true", help=argparse.SUPPRESS)
    args = ap.parse_args(argv)
    if args.one:                                        # a subprocess run
        print(json.dumps(measure(args.src, args.n, args.repeat)))
        return 0

    def run(src):
        out = subprocess.run([sys.executable, "-I", os.path.abspath(__file__), "--one",
                              "--src", src, "--n", str(args.n), "--repeat", str(args.repeat)],
                             capture_output=True, text=True, check=True)
        return json.loads(out.stdout)

    new = run(args.src)
    baselines = [json.loads(subprocess.run(
        [sys.executable, "-I", "-c",
         "import json, sys; sys.path.insert(0, sys.argv[1]); import bench_index_guard as b; "
         "print(json.dumps(b.measure_old_code(sys.argv[2], int(sys.argv[3]), int(sys.argv[4]))))",
         HERE, args.src, str(args.n), str(args.repeat)],
        capture_output=True, text=True, check=True).stdout)]
    if args.baseline_src:
        baselines.append(run(args.baseline_src))
    print(json.dumps(new))
    for base in baselines:
        print(json.dumps(base))
        ratios = {k: round(new[k] / base[k], 2) for k in new if k.endswith("_ms")}
        print(f"after / before ({base['src']}): {json.dumps(ratios)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
