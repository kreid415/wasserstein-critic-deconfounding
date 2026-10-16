#!/usr/bin/env python3
"""Count-only read of the local X1 runner record failures.csv (user decision NB-20261016-25, CR-05).

Prints ONLY: whether the file still matches the SHA-256 in score_file_hashes.csv, the number of data rows (header
excluded), and, if a column named status, reason, state, outcome or category exists, the number of rows per label of
that column with every digit replaced by '#'. No other column is kept or printed (no tag, arm, task, seed or
score), so no result is examined (SI-78).

Usage (standard library only):
    python experiments/x1_x13_score_hashes_20261015/x1_local_failures_count.py \
        --data-root /home/kendall/experiment_data/wasserstein-critic-deconfounding
"""
import argparse, csv, hashlib, os, re
from collections import Counter

REL = "tier12_v2/X1_local/failures.csv"
LABEL_COLUMNS = ("status", "reason", "state", "outcome", "category")
HERE = os.path.dirname(os.path.abspath(__file__))


def listed_sha256():
    with open(os.path.join(HERE, "score_file_hashes.csv"), newline="") as fh:
        hit = [r for r in csv.DictReader(fh) if r["path"] == REL]
    if len(hit) != 1:
        raise SystemExit(f"{REL} must be listed exactly once in score_file_hashes.csv, found {len(hit)}")
    return hit[0]["sha256"]


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data-root", default="/home/kendall/experiment_data/wasserstein-critic-deconfounding")
    a = ap.parse_args()
    path = os.path.join(a.data_root, REL)
    with open(path, "rb") as fh:
        raw = fh.read()
    print("sha256 matches the hash list:", hashlib.sha256(raw).hexdigest() == listed_sha256())
    rd = csv.reader(raw.decode().splitlines())
    header = next(rd)
    cols = [i for i, h in enumerate(header) if h.strip().lower() in LABEL_COLUMNS]
    tallies = {i: Counter() for i in cols}
    n = 0
    for row in rd:
        if not row:
            continue
        n += 1
        for i in cols:
            tallies[i][re.sub(r"\d", "#", row[i])] += 1
    print("data rows", n)
    print("label columns found:", [header[i].strip() for i in cols])
    for i in cols:
        for k, v in sorted(tallies[i].items()):
            print("label", header[i].strip(), repr(k), v)


if __name__ == "__main__":
    main()
