#!/usr/bin/env python
"""Identity check of the JHPCE fixed-seed scoring against workstation scores (CONSTRAINTS.md SI-115).

For every latent of --ref-list (TSV with the columns tag and ref_score: the workstation's one-row score file), the JHPCE score file
--cand/<tag>.csv must carry the same latent_sha256 and scorer_versions and an identical value in EVERY metric column (all columns except
tag, the identity columns, the provenance columns incl. the three JHPCE site columns, and score_seconds): float64, an empty cell is NaN,
NaN equals NaN only. The result goes to --out as JSON (docs/scorer_identity_check_20261028.json is read by the rule check:
all_columns_identical, n_latents, with the SHA-256 of every file compared). Exit 0 PASS | 1 FAIL | 2 a file, row or column is missing.
Usage: python identity_check.py --ref-list TSV --cand DIR --out JSON
"""
import argparse
import csv
import datetime
import hashlib
import json
import math
import os
import sys

IDENTITY = ("latent_sha256", "experiment", "task", "arm", "lam", "cond", "seed")
PROVENANCE = ("scorer_git_sha", "scorer_dirty", "score_host", "cpu_model", "cpu_simd", "numba_cpu_name", "scorer_versions",
              "score_site", "lisi_random_device", "lisi_preload_sha256")
NOT_METRIC = {"tag", "score_seconds", *IDENTITY, *PROVENANCE}


def one_row(path):
    if not os.path.isfile(path):
        raise SystemExit(f"{path}: no such file") from None
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    if len(rows) != 1:
        raise SystemExit(f"{path}: {len(rows)} rows, expected 1")
    return rows[0]


def sha256_file(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def num(s):
    return math.nan if s == "" else float(s)


def compare(ref, cand):
    """({metric column: identical?}, [columns present on one side only]) for two score rows."""
    metrics = lambda r: {c for c in r if c not in NOT_METRIC}          # noqa: E731
    only = sorted(metrics(ref) ^ metrics(cand))
    same = {}
    for c in sorted(metrics(ref) & metrics(cand)):
        a, b = num(ref[c]), num(cand[c])
        same[c] = (math.isnan(a) and math.isnan(b)) or a == b
    return same, only


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--ref-list", required=True)
    ap.add_argument("--cand", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    with open(a.ref_list, newline="") as f:
        items = list(csv.DictReader(f, delimiter="\t"))
    per, ok = [], True
    for it in items:
        ref_path, cand_path = it["ref_score"], os.path.join(a.cand, it["tag"] + ".csv")
        ref, cand = one_row(ref_path), one_row(cand_path)
        same, only = compare(ref, cand)
        bad = sorted(c for c, v in same.items() if not v)
        good = (ref["tag"] == cand["tag"] == it["tag"] and ref["latent_sha256"] == cand["latent_sha256"]
                and ref["scorer_versions"] == cand["scorer_versions"] and not bad and not only)
        ok &= good
        per.append(dict(tag=it["tag"], identical=good, n_columns=len(same), differing_columns=bad, columns_on_one_side_only=only,
                        same_latent=ref["latent_sha256"] == cand["latent_sha256"], same_package_versions=ref["scorer_versions"] == cand["scorer_versions"],
                        ref_file=ref_path, ref_sha256=sha256_file(ref_path), ref_scorer_git_sha=ref["scorer_git_sha"], ref_cpu=ref["cpu_model"],
                        cand_file=cand_path, cand_sha256=sha256_file(cand_path), cand_scorer_git_sha=cand["scorer_git_sha"],
                        cand_cpu=cand["cpu_model"], cand_score_host=cand["score_host"], cand_lisi_random_device=cand.get("lisi_random_device")))
    res = dict(status="done", all_columns_identical=bool(ok and per), n_latents=len(per), n_metric_columns=sorted({p["n_columns"] for p in per}),
               rule="PASS only if, for every latent, latent_sha256 and scorer_versions are equal and every metric column is identical (float64; NaN only against NaN)",
               made=datetime.datetime.now().astimezone().isoformat(timespec="seconds"), latents=per)
    with open(a.out, "w") as f:
        json.dump(res, f, indent=1, sort_keys=True)
        f.write("\n")
    print(("PASS" if res["all_columns_identical"] else "FAIL"), f"{sum(p['identical'] for p in per)}/{len(per)} latents identical in every column")
    return 0 if res["all_columns_identical"] else 1


if __name__ == "__main__":  # pragma: no cover  # entry point
    sys.exit(main())
