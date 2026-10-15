"""DOC-07 (documents review of 2026-10-14): power of the P1 contrast with and without the simulation sim1.

Re-derives `power` of docs/prereg/r1_record.json from its stored per-block summaries with the formulas of
prereg_rules.power_report (and stops if the committed values are not reproduced), then drops the sim1 blocks and repeats.
The root search starts at 0.01 because the search from 1.0 in power_report stopped, until SI-86 (the user's decision of
2026-10-15 to restart it at 0.01 when it fails), for some standard errors, among them the one without sim1: with scipy
1.17.1 the non-central t (scipy.stats.nct.cdf) returns NaN (not a number) at isolated large non-centralities (the effect
divided by its standard error), the first at 42.15 for 5 degrees of freedom and at 57.24 for 3; 7 of 230 tested pairs of
standard error and degrees of freedom failed (script 05_power_report_nan_probe.py of the documents re-check, 2026-10-14).
Usage (repo root, env scvi-api): python experiments/final_review_amendments_20261013/power_without_sim1.py
Writes results/power_without_sim1.json."""
import json
import math
import os
import sys

import numpy as np
from scipy import optimize, stats

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import prereg_rules as P  # noqa: E402

DESIGNS = (("real_tasks", len(P.REAL_TASKS)), ("families", len(set(P.TASK_FAMILY.values()))))


def power(blocks):
    """sigma, tau and the two designs of prereg_rules.power_report, from {'task|decoder': {n, mean, sd}}."""
    num = sum((b["n"] - 1) * b["sd"] ** 2 for b in blocks.values())
    den = sum(b["n"] - 1 for b in blocks.values())
    sigma, by_cond = math.sqrt(num / den), {}
    for k, b in blocks.items():
        by_cond.setdefault(k.split("|")[1], []).append((b["mean"], b["n"]))
    taus = [max(0.0, float(np.var([m for m, _ in v], ddof=1)) - sigma ** 2 / float(np.mean([n for _, n in v])))
            for v in by_cond.values() if len(v) >= 2]
    tau = math.sqrt(float(np.mean(taus)))
    res = dict(sigma=sigma, sigma_df=den, tau=tau, tau_df=sum(len(v) - 1 for v in by_cond.values()), designs={})
    for label, D in DESIGNS:
        se, df, hi = math.sqrt((tau ** 2 + sigma ** 2 / P.POWER_SEEDS) / D), D - 1, 0.01
        while P._power(hi, se, df) < 0.8:
            hi *= 2
        res["designs"][label] = dict(
            D=D, df=df, se=se, mde80=optimize.brentq(lambda x: P._power(x, se, df) - 0.8, 1e-9, hi),
            power_at_delta=P._power(P.POWER_DELTA, se, df),
            equivalence_feasible={str(h): bool(stats.t.ppf(0.975, df) * se < h) for h in P.EQUIV_BOUNDS})
    return res


if __name__ == "__main__":
    rec = json.load(open(os.path.join(ROOT, "docs", "prereg", "r1_record.json")))["power"]
    full = power(rec["blocks"])
    assert abs(full["sigma"] - rec["sigma"]) < 1e-8 and abs(full["tau"] - rec["tau"]) < 1e-8, "record not reproduced"
    for label in full["designs"]:
        for key in ("se", "mde80", "power_at_delta"):
            assert abs(full["designs"][label][key] - rec["designs"][label][key]) < 1e-8, (label, key)
    without = power({k: v for k, v in rec["blocks"].items() if not k.startswith("sim1|")})
    out = os.path.join(HERE, "results", "power_without_sim1.json")
    with open(out, "w") as f:
        json.dump(dict(all_blocks=full, without_sim1=without), f, indent=1)
    print(f"all six blocks: sigma {full['sigma']:.4f}, tau {full['tau']:.4f}; without sim1: sigma {without['sigma']:.4f}, "
          f"tau {without['tau']:.4f}; wrote {out}")
