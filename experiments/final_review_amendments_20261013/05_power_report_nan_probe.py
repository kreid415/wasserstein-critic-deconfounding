"""05: does prereg_rules.power_report's root search (start 1.0) fail at small standard errors? Usage: python 05_power_report_nan_probe.py <repo>.
Pure computation on scipy; no data."""
import os, sys, warnings
import numpy as np
from scipy import optimize, stats
REPO = os.path.abspath(sys.argv[1]); sys.path.insert(0, os.path.join(REPO, "scripts"))
import prereg_rules as P
import scipy; print("scipy", scipy.__version__)
def search(se, df, start):
    hi = start
    while P._power(hi, se, df) < 0.8:
        hi *= 2; P._require(hi < 1e6, "never reaches 0.8")
    return optimize.brentq(lambda x: P._power(x, se, df) - 0.8, 1e-9, hi)
for se, df, what in ((0.02355132223101683, 5, "without sim1, six tasks"), (0.04498519570869656, 5, "all six blocks, six tasks (the committed record)")):
    for start in (1.0, 0.01):
        try: print(f"{what}: se {se:.5f} df {df} start {start}: mde80 {search(se, df, start):.10f}")
        except Exception as ex: print(f"{what}: se {se:.5f} df {df} start {start}: {type(ex).__name__}: {ex}")
warnings.simplefilter("ignore")
for df in (3, 5):
    tc = stats.t.ppf(0.95, df); g = np.arange(0.5, 400, 0.01)
    cn = [float(x) for x in g if np.isnan(stats.nct.cdf(tc, df, x))]; sn = [float(x) for x in g if np.isnan(stats.nct.sf(tc, df, x))]
    print(f"df {df}: nct.cdf is NaN at {len(cn)} of {len(g)} non-centralities on 0.5-400 (first {cn[0]:.2f}); nct.sf is NaN at {len(sn)}")
bad = []
for D in (6, 4):
    for se in np.arange(0.003, 0.0601, 0.0005):
        try: search(se, D - 1, 1.0)
        except Exception: bad.append((D, round(float(se), 4)))
print("standard errors (0.003-0.06, step 0.0005) at which the start-1.0 search raises:", bad)
# standard error of the planned recompute, approximated with the committed pilot blocks of the R1 record (same formulas as power_report):
# the four blocks that are not flagged low-signal (the flagged ones are immune|1 and sim1|1), D = 6 real tasks, 5 seeds
import json, math
blk = json.load(open(os.path.join(REPO, "docs/prereg/r1_record.json")))["power"]["blocks"]
def se_of(keep, D=6, seeds=5):
    b = {k: v for k, v in blk.items() if k in keep}
    sigma2 = sum((v["n"] - 1) * v["sd"] ** 2 for v in b.values()) / sum(v["n"] - 1 for v in b.values())
    by = {}
    for k, v in b.items(): by.setdefault(k.split("|")[1], []).append((v["mean"], v["n"]))
    taus = [max(0.0, float(np.var([m for m, _ in v], ddof=1)) - sigma2 / float(np.mean([n for _, n in v]))) for v in by.values() if len(v) >= 2]
    return math.sqrt((float(np.mean(taus)) + sigma2 / seeds) / D)
allb, no_sim1, evalb = set(blk), {k for k in blk if not k.startswith("sim1")}, {k for k in blk if k not in ("immune|1", "sim1|1")}
for name, keep in (("all six blocks", allb), ("without sim1", no_sim1), ("four blocks not flagged low-signal", evalb)):
    print(f"proxy standard error, {name}: {se_of(keep):.5f}")
