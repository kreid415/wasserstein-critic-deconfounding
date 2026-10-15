#!/usr/bin/env python
"""The pooled model of scripts/analysis_x1.py (`combine`, as analyse() calls it) against the independent reference of ref_pool.py
(block algebra in long double arithmetic, Gauss-Legendre quadrature in log tau over [1e-18, 12 prior scales]); synthetic inputs only.
Written from compare_code.py, headline_cases.py and trunc_scan.py of the independent re-check of 2026-10-14 (evidence archive, artifact
version 4fc04d2a-1c2e-481c-bb7e-5157820f755f); the 38 hand-made sets of that re-check include 20 calibrated to lie 1e-6 and 1e-7 from the
declaration threshold, which are not repeated here.

Part 1, `cases`: the 18 other hand-made sets (standard errors near 1e-4, near 1e-6 and ordinary; one dataset; strong heterogeneity; one tiny
standard error among six; P2's scale) in the structures the analysis uses.
Part 2, `spread_scan`: six datasets (standard error 0.01) whose differences have a standard deviation of 1 to 8 prior scales, at prior
scales 0.05 and 0.1: how the error of the code grows when the data call for a spread beyond the grid's end at 5 prior scales.
Errors: largest absolute difference over P(Delta > r), P(Delta < -r), P(|Delta| <= r) ('prob') and over those plus the posterior
mean, the 95% interval ends and the heterogeneity ('all'). 'cut5' compares with the reference cut at 5 prior scales, as the code is.

Run (conda env `python`: numpy 2.4.6, scipy 1.17.1, pandas 2.3.3; about 15 seconds):
    python experiments/analysis_numerics_20261014/compare_with_reference.py
Output: compare_with_reference.json (both parts and the largest differences).
"""
import json
import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(HERE)), "scripts"))
import analysis_x1 as A  # noqa: E402
import prereg_rules as P  # noqa: E402
import ref_pool as R  # noqa: E402

KEYS = ("p_above", "p_below", "p_within", "mean", "lo", "hi", "heterogeneity")
PROBS = ("p_above", "p_below", "p_within")
TASKS = list(P.REAL_TASKS)


def prep(tasks, d, s, structure):
    """What combine() does before pool(): the units, and for 'pairs' the family averages (re-implemented here)."""
    fams = [P.TASK_FAMILY[t] for t in tasks]
    if structure == "pairs":
        order = list(dict.fromkeys(fams))
        g = [[i for i, f in enumerate(fams) if f == u] for u in order]
        return ([sum(d[i] for i in gi) / len(gi) for gi in g], [math.sqrt(sum(s[i] ** 2 for i in gi)) / len(gi) for gi in g], order)
    return list(d), list(s), fams


def run_ref(tasks, d, s, structure, scale, margin, cut=False):
    d2, s2, fam = prep(tasks, d, s, structure)
    lo, span = (1e-15, 5.0) if cut else (1e-18, 12.0)
    return R.reference(d2, s2, fam, "family" if structure == "family" else "independent", scale, margin, lo=lo, span=span)


def compare(case):
    tasks, d, s, structure, scale, margin = (case[k] for k in ("tasks", "d", "s", "structure", "scale", "margin"))
    r = A.combine(tasks, list(d), list(s), structure, scale, margin)
    code = {k: float(r[k]) for k in KEYS}
    wide, cut = (run_ref(tasks, d, s, structure, scale, margin, cut=c) for c in (False, True))
    return dict(name=case["name"], structure=structure, scale=scale, n=len(d), s_min=float(min(s)), s_max=float(max(s)),
                declaration=r["declaration"], reference_declaration=wide["declaration"],
                err_prob=max(abs(code[k] - wide[k]) for k in PROBS), err_all=max(abs(code[k] - wide[k]) for k in KEYS),
                err_prob_cut5=max(abs(code[k] - cut[k]) for k in PROBS), err_all_cut5=max(abs(code[k] - cut[k]) for k in KEYS),
                cut_effect=max(abs(cut[k] - wide[k]) for k in KEYS), reference_mass_above_5_scales=wide["tau_mass_above_5"])


def hand_made_cases():
    s1, d1 = [1.0e-4, 1.3e-4, 0.9e-4, 1.1e-4, 1.2e-4, 1.4e-4], [0.0125, 0.0118, 0.0123, 0.0121, 0.0119, 0.0122]
    s2, d2 = [1.0e-6, 1.1e-6, 1.3e-6, 1.6e-6, 2.0e-6, 3.0e-6], [0.0100005, 0.0100001, 0.0100009, 0.0099996, 0.0100012, 0.0100003]
    d4, s4 = [0.35, -0.20, 0.05, 0.40, -0.30, 0.15], [0.01] * 6
    d4b = [0.30, -0.25, 0.20, 0.35, -0.30, 0.25]
    out = []

    def add(name, tasks, d, s, structure="family", scale=0.1, margin=0.01):
        out.append(dict(name=name, tasks=list(tasks), d=[float(x) for x in d], s=[float(x) for x in s], structure=structure,
                        scale=scale, margin=margin))
    for scale, structure in ((0.1, "family"), (0.2, "family"), (0.05, "family"), (0.1, "independent"), (0.1, "pairs")):
        add(f"H1_s1e-4_{structure}_{scale}", TASKS, d1, s1, structure, scale)
    for scale, structure in ((0.1, "family"), (0.2, "family"), (0.1, "independent")):
        add(f"H2_s1e-6_{structure}_{scale}", TASKS, d2, s2, structure, scale)
    add("H3_single_s1e-6", ["pancreas"], [0.0100012], [1.0e-6])
    add("H3_single_s1e-6_independent", ["pancreas"], [0.0100012], [1.0e-6], "independent")
    add("H3_single_s0.005", ["lung"], [0.03], [0.005])
    add("H3_single_s1e-4", ["lung"], [0.0105], [1.0e-4], scale=0.2)
    add("H4_strong_het_0.1", TASKS, d4, s4)
    add("H4_strong_het_0.05", TASKS, d4, s4, scale=0.05)
    add("H4b_strong_het_0.05", TASKS, d4b, s4, scale=0.05)
    add("H4_strong_het_independent_0.05", TASKS, d4, s4, "independent", 0.05)
    add("H5_one_tiny_s", TASKS, [0.02, 0.019, 0.021, 0.0205, 0.018, 0.0195], [1.0e-6, 0.01, 0.01, 0.012, 0.011, 0.01])
    add("H6_P2_scale", TASKS, [0.04, 0.12, 0.02, 0.08, 0.1, 0.07], [0.09, 0.1, 0.11, 0.09, 0.12, 0.1], scale=0.25, margin=0.05)
    return out


def spread_scan():
    pat = np.array([1.0, -0.8, 0.3, 1.2, -1.1, 0.4])
    pat = (pat - pat.mean()) / pat.std(ddof=1)
    rows = []
    for scale in (0.05, 0.1):
        for ratio in (1, 2, 3, 4, 5, 6, 8):
            case = dict(name=f"scale{scale}_ratio{ratio}", tasks=TASKS, d=list(0.02 + ratio * scale * pat), s=[0.01] * 6,
                        structure="family", scale=scale, margin=0.01)
            rows.append(dict(sd_of_d_over_prior_scale=ratio, **compare(case)))
    return rows


def main():
    t0 = time.time()
    cases = []
    for c in hand_made_cases():
        cases.append(compare(c))
        print(f"{cases[-1]['name']:34s} err_prob {cases[-1]['err_prob']:.2e} err_all {cases[-1]['err_all']:.2e} "
              f"declaration {cases[-1]['declaration']}/{cases[-1]['reference_declaration']}", flush=True)
    scan = spread_scan()
    for r in scan:
        print(f"spread {r['sd_of_d_over_prior_scale']} prior scales (scale {r['scale']}): err_prob {r['err_prob']:.2e} err_all {r['err_all']:.2e}",
              flush=True)
    summary = dict(cases=len(cases), largest_err_prob=max(r["err_prob"] for r in cases), largest_err_all=max(r["err_all"] for r in cases),
                   declarations_that_differ=sum(r["declaration"] != r["reference_declaration"] for r in cases + scan))
    json.dump(dict(summary=summary, cases=cases, spread_scan=scan), open(os.path.join(HERE, "compare_with_reference.json"), "w"), indent=1)
    print(json.dumps(summary))
    print(f"run time {time.time() - t0:.0f} s (not stored, so that a re-run reproduces the output file byte for byte)")


if __name__ == "__main__":
    main()
