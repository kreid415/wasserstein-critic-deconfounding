"""Shared paths and the input-file ledger for the scripts in this folder (DOC-R04, 2026-10-15).

Blinding (SI-68, SI-78): these scripts read committed files of this repository, the obs tables of the prepared datasets
and the scores of the pilot experiments A1 and A1_ext_r1_round1. They never read an X1 or X13 score. `data_path` refuses
every path that contains the name of a blinded folder, and `python common.py` shows the refusal.

Usage from a clone of the repository: python experiments/doc_numbers_20261015/<script>.py (see README.md).
"""
import csv
import hashlib
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]                      # experiments/doc_numbers_20261015 -> repository root
DATA = Path(os.environ.get("WCD_DATA", "/home/kendall/experiment_data/wasserstein-critic-deconfounding"))
RESULTS = HERE / "results"
BLINDED = ("X1_local", "jhpce_tier12", "review_sealed", "identity", "/logs/", "fillers_x1_x13", "x13_cpu_baselines")
sys.path.insert(0, str(REPO / "scripts"))   # PYTHONSAFEPATH=1 does not add the script folder


def data_path(*parts):
    """A path under the data folder, refused if it names a blinded folder."""
    p = DATA.joinpath(*parts)
    if any(b in str(p) for b in BLINDED):
        raise PermissionError(f"blinded path refused: {p}")
    return p


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_inputs(name, paths):
    """Write results/<name>_inputs.csv (path, bytes, sha256), one row per input file; return the SHA-256 of that file."""
    RESULTS.mkdir(exist_ok=True)
    out = RESULTS / f"{name}_inputs.csv"
    with open(out, "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["path", "bytes", "sha256"])
        for p in sorted(map(Path, paths)):
            rel = p.relative_to(REPO) if REPO in p.parents else Path("$WCD_DATA") / p.relative_to(DATA)
            w.writerow([str(rel), p.stat().st_size, sha256(p)])
    return sha256(out)


if __name__ == "__main__":     # positive control for the blinding guard
    try:
        data_path("tier12_v2", "X1_local", "scores")
    except PermissionError as e:
        print("refused as intended:", e)
    else:
        raise SystemExit("the blinding guard did not refuse a blinded path")
