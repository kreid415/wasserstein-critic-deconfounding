#!/usr/bin/env python
"""Run the outline's mock-figure script (synthetic panels, fixed seed) from a command line.

paper/drafts_20261022/make_mock_figures.py lives on the branch `paper`, expects apply_figure_style and panel_letter to be in scope and
writes figures/mock/fig1-6.png relative to the working directory. This runner reads it with `git show <ref>:<path>`, supplies the
stand-in helpers (figure_helpers.py), works inside <out> so no tracked figure is overwritten, and fails unless all six PNGs exist.
The script reads no analysis output: it is a check that it still runs, not part of the data path.

Usage: python scripts/dryrun/run_mock_figures.py --out <dir> [--ref origin/paper]
"""
import argparse
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import matplotlib  # noqa: E402

matplotlib.use("Agg")
from figure_helpers import apply_figure_style, panel_letter  # noqa: E402

MOCK_PATH = "paper/drafts_20261022/make_mock_figures.py"


def main(argv=None):  # pragma: no cover  # needs the paper branch and its mock-figure script
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--ref", default="origin/paper")
    a = ap.parse_args(argv)
    src = subprocess.run(["git", "show", f"{a.ref}:{MOCK_PATH}"], capture_output=True, text=True, check=True).stdout
    os.makedirs(os.path.join(a.out, "figures", "mock"), exist_ok=True)
    os.chdir(a.out)
    apply_figure_style(sizes=(8, 7, 6))
    exec(compile(src, MOCK_PATH, "exec"), dict(apply_figure_style=apply_figure_style, panel_letter=panel_letter))
    missing = [f"fig{i}.png" for i in range(1, 7) if not os.path.getsize(f"figures/mock/fig{i}.png")]
    if missing:
        raise SystemExit(f"mock figures missing: {missing}")
    return 0


if __name__ == "__main__":  # pragma: no cover  # command-line entry
    sys.exit(main())
