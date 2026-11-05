"""Coverage of the exact (Clopper-Pearson) 95% interval for a penalty's collapse rate in the sweep design
(NS = 3 seeds x NL = 10 strengths = 30 fits; 12 strengths = 36 fits where edge strengths were fitted).
Replaces the six-strength, five-seed simulation of the first review (experiments/analysis_numerics_20261014/first_review/p2_clustering_sim.py,
coverage 0.91 to 1.00); same two scenarios, synthetic, parameters stretched from six to NL points by linear interpolation:
  A  runs independent, collapse probability rises with strength (target = mean probability over the grid);
  B  each seed has one latent collapse threshold, shared by all of its strengths (collapse at every strength index >= J; J = NL: never).
Control: constant probability, independent runs, must cover at least 0.95 (exact interval is conservative).
Run (python with numpy, scipy): python paper/analysis_pass10/coverage_sim_sweep.py   -> paper/analysis_pass10/coverage_sim_sweep.json (seed 7, 20,000 replicates per line)."""
import json, os
import numpy as np
from scipy.stats import beta

R, SEED, NS = 20000, 7, 3
rng = np.random.default_rng(SEED)
pA6 = np.array([0.0, 0.0, 0.05, 0.2, 0.6, 0.95])                      # first review's scenario A, six strengths
thr6 = np.array([0.0, 0.0, 0.0, 0.1, 0.15, 0.2, 0.55])               # scenario B, P(J = j), j = 0..6

def cp(x, n, a=0.05):
    lo = 0.0 if x == 0 else beta.ppf(a / 2, x, n - x + 1)
    hi = 1.0 if x == n else beta.ppf(1 - a / 2, x + 1, n - x)
    return lo, hi

def stretch(v, n):                                                   # linear interpolation of a profile onto n points
    return np.interp(np.linspace(0, 1, n), np.linspace(0, 1, len(v)), v)

def coverage(draw, target, n):
    cache, hit = {}, 0
    for _ in range(R):
        x = int(draw().sum())
        if x not in cache: cache[x] = cp(x, n)
        lo, hi = cache[x]; hit += lo <= target <= hi
    return hit / R

out = {"replicates": R, "seed": SEED, "seeds_per_fit_set": NS}
for NL in (10, 12):
    n = NL * NS
    pA = stretch(pA6, NL)
    thr = stretch(thr6, NL + 1); thr = thr / thr.sum()
    tB = float(sum(thr[j] * (np.arange(NL) >= j).mean() for j in range(NL + 1)))
    simA = lambda: rng.random((NS, NL)) < pA[None, :]
    simB = lambda: np.arange(NL)[None, :] >= rng.choice(NL + 1, size=NS, p=thr)[:, None]
    simC = lambda: rng.random((NS, NL)) < 0.3
    out[f"NL{NL}"] = {"fits": n, "A_target": float(pA.mean()), "A_coverage": coverage(simA, float(pA.mean()), n),
                      "B_target": tB, "B_coverage": coverage(simB, tB, n), "control_constant_p0.3_coverage": coverage(simC, 0.3, n)}
    assert out[f"NL{NL}"]["control_constant_p0.3_coverage"] >= 0.95, out
    print(NL, out[f"NL{NL}"])
json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "coverage_sim_sweep.json"), "w"), indent=1)
