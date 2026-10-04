#!/usr/bin/env python
"""Compare the metric columns of score rows exactly (regression check for scorer changes; fail-loud).

Usage: python scripts/compare_score_rows.py REFERENCE.csv CANDIDATE.csv [--out table.csv]
Each file holds one row of scripts/score_scib_native.py. Compared: every metric of BATCH_METRICS + BIO_METRICS
(read from score_scib_native.py by ast, no scib import), kbet_seed and kbet_r_calls, bitwise as float64 (NaN equals
NaN only when both are NaN). Exit 0 iff all are identical; a missing column or row-count != 1 is an error.
"""
import argparse
import ast
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))


def metric_names(path=os.path.join(HERE, "score_scib_native.py")):
    tree = ast.parse(open(path).read())
    vals = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            if node.targets[0].id in ("BATCH_METRICS", "BIO_METRICS"):
                vals[node.targets[0].id] = ast.literal_eval(node.value)
    if set(vals) != {"BATCH_METRICS", "BIO_METRICS"}:
        raise RuntimeError(f"{path}: BATCH_METRICS / BIO_METRICS not found")
    return vals["BATCH_METRICS"] + vals["BIO_METRICS"]


def one_row(p):
    d = pd.read_csv(p)
    if len(d) != 1:
        raise ValueError(f"{p}: {len(d)} rows, expected 1")
    return d.iloc[0]


def compare(ref, cand, cols):
    rows = []
    for c in cols:
        if c not in ref.index or c not in cand.index:
            raise KeyError(f"column {c!r} missing ({'reference' if c not in ref.index else 'candidate'})")
        a, b = np.float64(ref[c]), np.float64(cand[c])
        same = (np.isnan(a) and np.isnan(b)) or (a == b)
        rows.append(dict(column=c, reference=repr(float(a)), candidate=repr(float(b)), identical=bool(same)))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("reference")
    ap.add_argument("candidate")
    ap.add_argument("--out")
    a = ap.parse_args()
    ref, cand = one_row(a.reference), one_row(a.candidate)
    if str(ref["tag"]) != str(cand["tag"]):
        raise ValueError(f"tags differ: {ref['tag']} vs {cand['tag']}")
    T = compare(ref, cand, metric_names() + ["kbet_seed", "kbet_r_calls"])
    T.insert(0, "tag", ref["tag"])
    if a.out:
        T.to_csv(a.out, index=False)
    n_diff = int((~T.identical).sum())
    print(f"{ref['tag']}: {len(T)} columns compared, {n_diff} differ")
    if n_diff:
        print(T[~T.identical].to_string(index=False))
    sys.exit(1 if n_diff else 0)


if __name__ == "__main__":
    main()
