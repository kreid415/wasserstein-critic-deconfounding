#!/usr/bin/env python
"""Compare two sets of score files (one-row CSVs of scripts/score_scib_native.py) latent by latent, exactly.

PASS rule (also printed): PASS only if, for EVERY latent of the list, both files hold the same latent_sha256 and the same
scorer_versions (package versions) and EVERY metric column is identical. Metric columns = all columns except tag, the
identity columns (latent_sha256, experiment, task, arm, lam, cond, seed), the provenance columns (scorer_git_sha,
scorer_dirty, score_host, cpu_model, cpu_simd, numba_cpu_name, scorer_versions) and score_seconds; read as float64, an
empty cell is NaN, NaN equals NaN only when both are NaN. Otherwise the output gives, per metric column, the largest
absolute difference and the number of latents that differ, and the differing latents by the candidate's CPU type.

Usage: python cluster/jhpce/compare_scores.py --list TSV --cand DIR [--ref DIR]
  TSV  header row with the columns tag and ref_score (path of the reference score file; cluster/jhpce/score_check/);
       --ref DIR reads DIR/<tag>.csv instead (e.g. the first of two cluster runs, to test that the cluster repeats itself)
  DIR  DIR/<tag>.csv, as written by score_array.sh in OUT_DIR/scores
Exit 0 PASS | 1 FAIL | 2 a file, row or column is missing
"""
import argparse
import csv
import math
import os
import sys
from collections import Counter

IDENTITY = ("latent_sha256", "experiment", "task", "arm", "lam", "cond", "seed")
PROVENANCE = ("scorer_git_sha", "scorer_dirty", "score_host", "cpu_model", "cpu_simd", "numba_cpu_name", "scorer_versions")
NOT_METRIC = {"tag", "score_seconds", *IDENTITY, *PROVENANCE}
RULE = ("PASS only if, for every latent, latent_sha256 and scorer_versions are equal and every metric column is "
        "identical (float64; NaN only against NaN)")


def die(msg):
    print(msg, file=sys.stderr)
    raise SystemExit(2)


def one_row(path):
    if not os.path.isfile(path):
        die(f"{path}: no such file")
    with open(path) as f:
        rows = list(csv.DictReader(f))
    if len(rows) != 1:
        die(f"{path}: {len(rows)} rows, expected 1")
    return rows[0]


def num(s, where):
    try:
        return math.nan if s == "" else float(s)
    except ValueError:
        die(f"{where}: non-numeric metric value {s!r}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--list", required=True)
    ap.add_argument("--cand", required=True)
    ap.add_argument("--ref")
    a = ap.parse_args(argv)
    with open(a.list) as f:
        items = list(csv.DictReader(f, delimiter="\t"))
    if not items:
        die(f"{a.list}: no latents")
    cols, n_differ, max_diff, nan_vs_value = None, Counter(), {}, Counter()
    differing, problems, cpu = {}, [], {}
    for it in items:
        tag = it["tag"]
        ref = one_row(os.path.join(a.ref, tag + ".csv") if a.ref else it["ref_score"])
        cand = one_row(os.path.join(a.cand, tag + ".csv"))
        if ref["tag"] != tag or cand["tag"] != tag:
            die(f"{tag}: a score file holds the tag {ref['tag']!r} / {cand['tag']!r}")
        here = [c for c in ref if c not in NOT_METRIC]
        if set(here) != {c for c in cand if c not in NOT_METRIC} or (cols and set(here) != set(cols)):
            die(f"{tag}: the metric columns of the two files (or of the other latents) differ")
        cols = here
        problems += [f"{tag}: {c} differs" for c in ("latent_sha256", "scorer_versions") if ref[c] != cand[c]]
        cpu[tag] = f"{cand['score_host']} ({cand['cpu_simd']})"
        for c in cols:
            x, y = num(ref[c], f"{tag} {c}"), num(cand[c], f"{tag} {c}")
            if (math.isnan(x) and math.isnan(y)) or x == y:
                continue
            n_differ[c] += 1
            differing[tag] = differing.get(tag, 0) + 1
            if math.isnan(x) != math.isnan(y):
                nan_vs_value[c] += 1
            else:
                max_diff[c] = max(max_diff.get(c, 0.0), abs(x - y))
    print(f"PASS rule: {RULE}")
    print(f"{len(items)} latents, {len(cols)} metric columns")
    print(f"{'metric column':28s}{'latents differ':>15s}{'max |diff|':>13s}{'NaN vs value':>13s}")
    for c in cols:
        print(f"{c:28s}{n_differ[c]:15d}{max_diff.get(c, 0.0):13.3g}{nan_vs_value[c]:13d}")
    for p in problems:
        print(f"PROBLEM {p}")
    for tag, n in differing.items():
        print(f"differs: {tag}: {n} columns, scored on {cpu[tag]}")
    print("latents by candidate CPU: " + "; ".join(
        f"{k}: {v} scored, {sum(1 for t in differing if cpu[t] == k)} differ" for k, v in sorted(Counter(cpu.values()).items())))
    ok = not differing and not problems
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
