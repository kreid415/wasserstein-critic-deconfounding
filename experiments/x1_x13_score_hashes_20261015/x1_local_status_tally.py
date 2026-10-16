#!/usr/bin/env python3
"""Count-only tally of the local X1 ledger (user decision NB-20261016-16, SI-79 as amended 2026-10-16).

Reads ONLY the status column (named `state`) of the ledger file(s) under tier12_v2/X1_local/ledger and prints the
number of rows and the number of rows per status value over ALL rows. No other column is kept or printed (no
tag, arm, task, seed, score), so no result is examined (SI-78).

Usage (standard library only):
    python experiments/x1_x13_score_hashes_20261015/x1_local_status_tally.py \
        --data-root /home/kendall/experiment_data/wasserstein-critic-deconfounding
"""
import argparse, csv, glob, os
from collections import Counter


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data-root", default="/home/kendall/experiment_data/wasserstein-critic-deconfounding")
    a = ap.parse_args()
    files = sorted(glob.glob(os.path.join(a.data_root, "tier12_v2", "X1_local", "ledger", "*.csv")))
    if not files:
        raise SystemExit("no ledger csv found under tier12_v2/X1_local/ledger")
    n, per_status = 0, Counter()
    for f in files:
        with open(f, newline="") as fh:
            rd = csv.reader(fh)
            col = next(rd).index("state")
            for row in rd:
                n += 1
                per_status[row[col]] += 1
    print("ledger files", len(files))
    print("rows", n)
    for k, v in sorted(per_status.items()):
        print("status", repr(k), v)


if __name__ == "__main__":
    main()
