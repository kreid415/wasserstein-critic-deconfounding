"""Do the P2 confidence intervals (frozen clopper_pearson / agresti_caffo, 30 runs = 6 lambda x 5 seeds treated as independent) hold their nominal 95% coverage
for the registered-grid collapse rate when (A) runs are independent but the collapse probability rises with lambda, or (B) collapse is driven by a per-seed latent
threshold (the same seed shares its initialisation across lambda, PREREG CR-07)? Target = expected collapse fraction over the registered grid. Synthetic.
Written by the first independent rigour review (2026-10-14, review bundle artifact 3de51e25-da4a-49d3-b248-7a8143d3197d); only the
paths changed. It supports the limitation 3 of docs/ANALYSIS_SPEC.md section 16 (coverage 0.91 to 1.00).
Run from anywhere (conda env scvi-api): python experiments/analysis_numerics_20261014/first_review/p2_clustering_sim.py
Output: p2_clustering_sim.json (seed 7, 20000 replicates per line).
"""
import json
import os
import sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', '..', '..', 'scripts'))
import numpy as np  # noqa: E402
import analysis_x1 as A  # noqa: E402
rng = np.random.default_rng(7)
NL, NS, R = 6, 5, 20000
def simA(p_lam):
    return (rng.random((NS, NL)) < p_lam[None, :]).astype(int)
def simB(thr_probs):       # per-seed threshold index J in 0..NL (collapse at every lambda index >= J; J=NL means never)
    J = rng.choice(NL + 1, size=NS, p=thr_probs); return (np.arange(NL)[None, :] >= J[:, None]).astype(int)
def cov_cp(sim, target):
    hit = 0
    for _ in range(R):
        x = int(sim().sum()); lo, hi = A.clopper_pearson(x, NL * NS); hit += lo <= target <= hi
    return hit / R
pA = np.array([0.0, 0.0, 0.05, 0.2, 0.6, 0.95])
thr = np.array([0.0, 0.0, 0.0, 0.1, 0.15, 0.2, 0.55])           # P(J=j); collapses start at lambda index 3..5 or never
tB = float(sum(thr[j] * (np.arange(NL) >= j).mean() for j in range(NL + 1)))
OUT = {}
OUT['A_target'] = float(pA.mean()); OUT['A_cp_coverage'] = cov_cp(lambda: simA(pA), pA.mean()); print('A: grid-average collapse probability %.3f; CP coverage %.3f' % (OUT['A_target'], OUT['A_cp_coverage']))
OUT['B_target'] = tB; OUT['B_cp_coverage'] = cov_cp(lambda: simB(thr), tB); print('B: grid-average collapse probability %.3f; CP coverage %.3f' % (OUT['B_target'], OUT['B_cp_coverage']))
# difference of two arms: paired seeds (shared latent) vs independent, Agresti-Caffo 95% interval vs true difference
def cov_ac(simpair, target):
    hit = 0; w = []
    for _ in range(R):
        a, b = simpair(); (d, se) = A.agresti_caffo(int(a.sum()), NL * NS, int(b.sum()), NL * NS); hit += abs(d - target) <= 1.96 * se; w.append(se)
    return hit / R, float(np.mean(w))
thr2 = np.array([0.0, 0.0, 0.0, 0.0, 0.05, 0.15, 0.80]); t2 = float(sum(thr2[j] * (np.arange(NL) >= j).mean() for j in range(NL + 1)))
def pair_shared():        # both arms share the seed's latent resilience: arm 2 collapses at a later index (shift +1) for the same seed
    u = rng.random(NS)
    J1 = np.searchsorted(np.cumsum(thr), u, side='right'); J2 = np.searchsorted(np.cumsum(thr2), u, side='right')
    return [(np.arange(NL)[None, :] >= J[:, None]).astype(int) for J in (J1, J2)]
def pair_indep():
    J1 = rng.choice(NL + 1, size=NS, p=thr); J2 = rng.choice(NL + 1, size=NS, p=thr2)
    return [(np.arange(NL)[None, :] >= J[:, None]).astype(int) for J in (J1, J2)]
OUT['AC_target_diff'] = tB - t2; print('AC (Agresti-Caffo) target diff %.3f' % (tB - t2))
OUT['AC_paired_coverage'], OUT['AC_paired_se'] = cov_ac(pair_shared, tB - t2); print('  paired: coverage %.3f, mean SE %.3f' % (OUT['AC_paired_coverage'], OUT['AC_paired_se']))
OUT['AC_indep_coverage'], OUT['AC_indep_se'] = cov_ac(pair_indep, tB - t2); print('  independent: coverage %.3f, mean SE %.3f' % (OUT['AC_indep_coverage'], OUT['AC_indep_se']))
OUT['R'] = R
OUT['seed'] = 7
json.dump(OUT, open(os.path.join(HERE, 'p2_clustering_sim.json'), 'w'), indent=1)
