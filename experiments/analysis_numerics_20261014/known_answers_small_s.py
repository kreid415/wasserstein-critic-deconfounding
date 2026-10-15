#!/usr/bin/env python
"""Known answers of the pooled model for standard errors of 1e-6 to 3e-6 (just above S_MIN), for tests/analysis
(finding AN-R08, 2026-10-15). Inputs are synthetic hand-made sets of six datasets. The first three ('H2' sets of the independent
re-check) have differences d that straddle the margin 0.01 almost exactly, so the declaration probability is far from 0 and 1; the
last two ('spread') have differences spread by half a prior scale of 0.2, where the size of the spread grid changes the
result by more than the tolerance of the test (a grid of 151 points moves the independent case by 4.9e-8, one of 101 points by 2e-5).

Values come from the independent reference of ref_mp.py (never imports scripts/analysis_x1.py): block algebra in 50-digit
arithmetic (mpmath) and composite Gauss-Legendre quadrature in log(tau) over [1e-18, 12 prior scales] with 8 nodes per
panel (panels 1.0 wide for the family cases, 0.25 wide for the one-spread cases), instead of the code's trapezoid rule on 201
points from 1e-15 to 5 prior scales. Each case is checked against the same reference in long double arithmetic on the
finest rule of the re-check (panels 0.25 wide, 16 nodes); the script stops if the two differ by more than 1e-9. precision_mpmath.csv holds relative errors of the posterior mean and variance at
fixed spreads, not values of these probabilities, so it cannot serve as the known answer.

Run (conda env `python`: numpy 2.4.6, scipy 1.17.1, mpmath 1.4.1; about 3 minutes):
    python experiments/analysis_numerics_20261014/known_answers_small_s.py
Output: known_answers_small_s.csv (one row per case; lists are separated by semicolons).
"""
import csv
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import ref_mp as M  # noqa: E402
import ref_pool as R  # noqa: E402

TASKS = ["pancreas", "lung", "immune", "immune_hum_mou", "atac_small", "atac_large"]
FAMILY = ["pancreas", "lung", "immune", "immune", "atac", "atac"]       # as prereg_rules.TASK_FAMILY
D = [0.0100005, 0.0100001, 0.0100009, 0.0099996, 0.0100012, 0.0100003]
S = [1.0e-6, 1.1e-6, 1.3e-6, 1.6e-6, 2.0e-6, 3.0e-6]
PATTERN = [1.0, -0.8, 0.3, 1.2, -1.1, 0.4]
MEAN = sum(PATTERN) / 6
SD = (sum((x - MEAN) ** 2 for x in PATTERN) / 5) ** 0.5
SPREAD = [0.01 + 0.1 * (x - MEAN) / SD for x in PATTERN]        # standard deviation 0.1 = half a prior scale of 0.2
CASES = [("H2_family_0.1", "family", 0.1, D), ("H2_family_0.2", "family", 0.2, D), ("H2_independent_0.1", "independent", 0.1, D),
         ("H2_spread_family_0.2", "family", 0.2, SPREAD), ("H2_spread_independent_0.2", "independent", 0.2, SPREAD)]
MARGIN = 0.01
KEYS = ("p_above", "p_below", "p_within", "mean", "lo", "hi", "heterogeneity")
PANEL = {"family": 1.0, "independent": 0.25}     # panel widths in log(tau); the one-spread cases are cheap, and panels 1.0 leave 3.4e-9 in the spread case
METHOD = "mpmath 50 digits, block algebra, Gauss-Legendre in log tau on [1e-18, 12 scales] (panels {}, 8 nodes)"


def main():
    rows = []
    for name, structure, scale, d in CASES:
        mp = M.reference_mp(d, S, FAMILY, structure, scale, MARGIN, panel_w=PANEL[structure])
        ld = R.reference(d, S, FAMILY, structure, scale, MARGIN, panel_w=0.25, g=16)
        worst = max(abs(mp[k] - ld[k]) for k in KEYS)
        print(f"{name}: nodes {mp['nodes']}, largest difference of the 50-digit and the long-double reference {worst:.2e}", flush=True)
        if worst > 1e-9:
            raise SystemExit(f"{name}: the two reference computations differ by {worst:.2e} (> 1e-9)")
        rows.append(dict(name=name, structure=structure, scale=scale, margin=MARGIN, tasks=";".join(TASKS),
                         d=";".join(repr(x) for x in d), s=";".join(repr(x) for x in S),
                         **{k: repr(float(mp[k])) for k in KEYS}, method=METHOD.format(PANEL[structure])))
    with open(os.path.join(HERE, "known_answers_small_s.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
