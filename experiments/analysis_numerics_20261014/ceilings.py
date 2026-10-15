#!/usr/bin/env python
"""How large can the probability behind a declaration get when only one, two, ... datasets are evaluable? (finding AN-R12, 2026-10-15)

Synthetic inputs, family structure, prior scale 0.1, margin 0.01, every evaluable dataset with the same difference d and the same
standard error s. For each number of datasets and each s it records, over the grid of d below, the largest P(Delta > 0.01) (by
symmetry the same ceiling holds for P(Delta < -0.01) with -d), the smallest grid value of d that reaches 0.95, and the largest
P(|Delta| <= 0.01) over d between 0 and 0.015. 'two_same_family' is immune and immune_hum_mou (one family, so one family spread);
the other sets of two or more datasets are taken in the order pancreas, lung, immune, atac_small, immune_hum_mou, atac_large.
Written from ceilings2.py of the independent re-check of 2026-10-14 (evidence archive, artifact version
4fc04d2a-1c2e-481c-bb7e-5157820f755f), extended to the equivalence side and to the same-family pair.

Run (conda env `python`: numpy 2.4.6, scipy 1.17.1, pandas 2.3.3; about 1.5 minutes):
    python experiments/analysis_numerics_20261014/ceilings.py
Output: ceilings.json.
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(HERE)), "scripts"))
import analysis_x1 as A  # noqa: E402

ORDER = ["pancreas", "lung", "immune", "atac_small", "immune_hum_mou", "atac_large"]
SETS = {"one": ORDER[:1], "two_different_families": ORDER[:2], "two_same_family": ["immune", "immune_hum_mou"],
        "three": ORDER[:3], "four": ORDER[:4], "five": ORDER[:5], "six": ORDER[:6]}
D_ABOVE = [0.015, 0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 1.0, 2.0]
D_WITHIN = [0.0, 0.0025, 0.005, 0.01, 0.015]
S_VALUES = [1e-6, 1e-4, 0.005, 0.01]


def main():
    out = {}
    for s in S_VALUES:
        for name, tasks in SETS.items():
            n = len(tasks)
            up = [A.combine(tasks, [d] * n, [s] * n, "family", 0.1, 0.01)["p_above"] for d in D_ABOVE]
            within = [A.combine(tasks, [d] * n, [s] * n, "family", 0.1, 0.01)["p_within"] for d in D_WITHIN]
            first = next((d for d, p in zip(D_ABOVE, up) if p >= 0.95), None)
            out[f"s{s:g}_{name}"] = dict(s=s, datasets=n, tasks=tasks, max_p_above=max(up), d_at_max=D_ABOVE[up.index(max(up))],
                                         smallest_d_reaching_0_95=first, max_p_within=max(within),
                                         p_above_by_d=dict(zip(map(str, D_ABOVE), up)), p_within_by_d=dict(zip(map(str, D_WITHIN), within)))
            r = out[f"s{s:g}_{name}"]
            print(f"s={s:g} {name:22s} max P(above)={r['max_p_above']:.4f} (d={r['d_at_max']}) smallest d with P>=0.95: "
                  f"{first}  max P(within)={r['max_p_within']:.4f}", flush=True)
    json.dump(out, open(os.path.join(HERE, "ceilings.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
