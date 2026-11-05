#!/usr/bin/env python
"""Per-unit validity of the replication scores (input of the replication run-status board).
A unit (manifest row) is valid when exactly one score CSV named <tag>.csv exists in the score directories, it has one row whose tag
matches, and every required metric (prereg_rules.BATCH_METRICS + bio_metrics_for(task)) is finite and inside the scIB scaled range.
Usage: python score_validity.py --rules-scripts DIR --manifest TSV --scores DIR [DIR ...] --out CSV. Exit 1 if any unit is not valid."""
import argparse, os, sys, datetime
import numpy as np, pandas as pd


def unit_check(P, task, tag, files):
    """-> (valid, reason, score_file). files: all score CSVs found for the tag."""
    if len(files) != 1:
        return False, f"{len(files)} score files", ""
    d = pd.read_csv(files[0])
    if len(d) != 1 or d.tag.iloc[0] != tag:
        return False, f"rows={len(d)} tag={list(d.tag)[:2]}", files[0]
    need = P.BATCH_METRICS + P.bio_metrics_for(task)
    absent = [m for m in need if m not in d.columns]
    if absent:
        return False, f"missing columns {absent}", files[0]
    v = pd.to_numeric(d.loc[0, need], errors="raise").astype(float)
    bad = [m for m in need if not np.isfinite(v[m]) or not -1 - 1e-6 <= v[m] <= 1 + 1e-6]
    return (not bad), (f"non-finite or out-of-range {bad}" if bad else ""), files[0]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--rules-scripts", required=True); ap.add_argument("--manifest", required=True)
    ap.add_argument("--scores", nargs="+", required=True); ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    sys.path.insert(0, a.rules_scripts)
    import prereg_rules as P
    M, _ = P.read_manifest(a.manifest)
    found = {}
    for d in a.scores:
        for f in os.listdir(d):
            if f.endswith(".csv"):
                found.setdefault(f[:-4], []).append(os.path.join(d, f))
    extra = sorted(set(found) - set(M.tag))
    if extra:
        raise SystemExit(f"{len(extra)} score files are not in the manifest, e.g. {extra[:3]}")
    rows = []
    for r in M.itertuples():
        ok, why, f = unit_check(P, r.task, r.tag, found.get(r.tag, []))
        t = "" if not f else datetime.datetime.fromtimestamp(os.path.getmtime(f)).astimezone().isoformat(timespec="seconds")
        rows.append(dict(tag=r.tag, task=r.task, arm=r.arm, seed=r.seed, valid=ok, reason=why, score_file=f, finished_at=t))
    R = pd.DataFrame(rows); R.to_csv(a.out, index=False)
    print(f"units={len(R)} valid={int(R.valid.sum())} invalid={int((~R.valid).sum())}")
    return 0 if R.valid.all() else 1


if __name__ == "__main__":  # pragma: no cover  # command-line entry
    sys.exit(main())
