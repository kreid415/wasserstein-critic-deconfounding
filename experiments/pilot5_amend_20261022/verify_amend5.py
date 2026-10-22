"""Re-derives the numbers quoted in the 2026-10-22 amendment (docs/PREREG.md section 9) from the repository and the
pilot data root. Standard library only; reads no X1 or X13 score, latent or ledger. Exit 1 on any mismatch.

Usage (repo root): python3 experiments/pilot5_amend_20261022/verify_amend5.py [--data-root D]
"""
import argparse
import collections
import csv
import hashlib
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, os.pardir, os.pardir))
NEW = "manifests/paper_manifest_stock_pilot_u5_b10_v3.pilot5.tsv"
OLD = "manifests/paper_manifest_stock_pilot_u5_b10_v3.tsv"
NEW_SHA = "5dd65550fdb18e5c80e73dcfa731d72bdeb6128bdeac7ffb36df6c54450396ff"
OLD_SHA = "6efbe422c8f09f9884001a84f7d7b363506f435803d238280e00bbb7b7e1cf24"
NEW_TASKS = ["atac_large", "immune_hum_mou", "lung", "pancreas", "sim2"]
OLD_TASKS = ["atac_small", "immune", "sim1"]
EDGE = {("discriminator", "0", 0.01), ("discriminator", "0", 0.03), ("sinkhorn", "1", 10000.0), ("sinkhorn", "1", 30000.0)}
SCORER_TREES = ["scripts", "src", "configs", "docs", "manifests", "cluster/envspec"]
bad = []


def check(name, ok, detail=""):
    print(("PASS " if ok else "FAIL ") + name + (f": {detail}" if detail else ""))
    if not ok:
        bad.append(name)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def rows(path):
    with open(path) as f:
        lines = [x for x in f if not x.startswith("#")]
    return list(csv.DictReader(lines, delimiter="\t"))


def git(*args):
    return subprocess.run(["git", "-C", REPO, *args], capture_output=True, text=True, check=True).stdout.strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="/home/kendall/experiment_data/wasserstein-critic-deconfounding")
    a = ap.parse_args()

    new = rows(os.path.join(REPO, NEW))
    check("new manifest sha256", sha256(os.path.join(REPO, NEW)) == NEW_SHA, sha256(os.path.join(REPO, NEW)))
    check("new manifest rows", len(new) == 1890, str(len(new)))
    per_task = collections.Counter(r["task"] for r in new)
    check("378 rows for each of the 5 new datasets", per_task == {t: 378 for t in NEW_TASKS}, str(dict(per_task)))
    check("seeds 100-102 only", {r["seed"] for r in new} == {"100", "101", "102"})
    check("experiment A1 only", {r["experiment"] for r in new} == {"A1"})
    edge = collections.Counter((r["arm"], r["cond"], float(r["lam"])) for r in new if (r["arm"], r["cond"], float(r["lam"])) in EDGE)
    check("12 grid-edge rows per dataset (4 strengths x 3 seeds)", set(edge) == EDGE and set(edge.values()) == {15},
          str(dict(edge)))  # 5 datasets x 3 seeds = 15 rows per (arm, decoder, strength)
    check("366 base rows per dataset", len(new) - sum(edge.values()) == 5 * 366, str(len(new) - sum(edge.values())))

    check("old pilot manifest sha256", sha256(os.path.join(REPO, OLD)) == OLD_SHA)
    ext = os.path.join(a.data_root, "tier12_v3/rules/r1_pass1/a1_extension.tsv")
    if os.path.isfile(ext):
        e = rows(ext)
        got = collections.Counter(r["task"] for r in e)
        check("first-round extension file: 12 rows for each of the 3 old datasets", got == {t: 12 for t in OLD_TASKS}, str(dict(got)))
        check("its strengths equal the new grid-edge strengths", {(r["arm"], r["cond"], float(r["lam"])) for r in e} == EDGE)
        print(f"     extension file sha256 {sha256(ext)}")
    else:
        check("first-round extension file present", False, ext)
    old = [r for r in rows(os.path.join(REPO, OLD)) if r["experiment"] == "A1" and r["task"] in OLD_TASKS]
    check("old base rows: 366 for each of the 3 old datasets", collections.Counter(r["task"] for r in old) == {t: 366 for t in OLD_TASKS})
    print(f"     fits in the 8-dataset R1 input: {len(old) + 36} (old) + {len(new)} (new) = {len(old) + 36 + len(new)}")

    for path in ("scripts/score_scib_native.py", "scripts/host_info.py", "scripts/run_stage.py"):
        x, y = git("rev-parse", f"10a1ca5:{path}"), git("rev-parse", f"a4989ff:{path}")
        check(f"{path} blob equal at 10a1ca5 and a4989ff", x == y, x[:8])
    for tree in SCORER_TREES:
        x, y = git("rev-parse", f"10a1ca5:{tree}"), git("rev-parse", f"a4989ff:{tree}")
        check(f"tree {tree}/ equal at 10a1ca5 and a4989ff", x == y, x[:8])
    check("tag prereg-tier12-v3 is 10a1ca5", git("rev-parse", "prereg-tier12-v3^{commit}").startswith("10a1ca5"))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
