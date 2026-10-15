#!/usr/bin/env python
"""How far from the 50-digit known answers do the code and three planted grid errors fall? (AN-R08, 2026-10-15)

For each case of known_answers_small_s.csv: the largest absolute difference (over P(Delta above / below / within the margin), the
posterior mean, the 95% interval ends and the heterogeneity) between the 50-digit value and the code, and the same for the code with
the spread grid cut to 151 points (planted error O22), to 101 points (M24) and with the grid's lower end at 1e-9 instead of 1e-15 (O02).
The test `test_pool_known_answers_small_s` accepts 1e-8.

Run (conda env `scvi-api` or `python`, one thread; a few seconds), from the repo root:
    python experiments/analysis_numerics_20261014/known_answers_effects.py > experiments/analysis_numerics_20261014/known_answers_effects.txt
"""
import csv
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "scripts"))
import analysis_x1 as A  # noqa: E402

KEYS = ("p_above", "p_below", "p_within", "mean", "lo", "hi", "heterogeneity")
VARIANTS = (("code", 201, 1e-15), ("GRID_N 151", 151, 1e-15), ("GRID_N 101", 101, 1e-15), ("TAU_FLOOR 1e-9", 201, 1e-9))


def worst(row, grid_n, floor):
    saved = A.GRID_N, A.TAU_FLOOR
    A.GRID_N, A.TAU_FLOOR = grid_n, floor
    try:
        r = A.combine(row["tasks"].split(";"), [float(x) for x in row["d"].split(";")], [float(x) for x in row["s"].split(";")],
                      row["structure"], float(row["scale"]), float(row["margin"]))
    finally:
        A.GRID_N, A.TAU_FLOOR = saved
    return max(abs(r[k] - float(row[k])) for k in KEYS)


def main():
    with open(os.path.join(HERE, "known_answers_small_s.csv"), newline="") as f:
        for row in csv.DictReader(f):
            print(f"{row['name']:28s} difference from the 50-digit value: " +
                  "; ".join(f"{name} {worst(row, n, floor):.1e}" for name, n, floor in VARIANTS))


if __name__ == "__main__":
    main()
