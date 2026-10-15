# Provenance: written by the independent re-check of 2026-10-14 (evidence archive, artifact version
# 4fc04d2a-1c2e-481c-bb7e-5157820f755f, scripts/validate_algebra.py); copied here without change to the code.
"""Check my block closed form (longdouble and mp) against direct n x n 50-digit algebra at random nodes."""
import sys, json
import numpy as np, mpmath as mp
import os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ref_pool as R, ref_mp as M
rng = np.random.default_rng(7)
FAM = ['a', 'a', 'b', 'b', 'c', 'd']
worst = dict(ld_ll=0, ld_mean=0, ld_var=0, mp_ll=0, mp_mean=0, mp_var=0)
rows = []
for trial in range(60):
    n = int(rng.integers(1, 7))
    idx = sorted(rng.choice(6, n, replace=False))
    fam = [FAM[i] for i in idx]
    structure = ['family', 'independent'][trial % 2]
    s = 10 ** rng.uniform(-6, -1.5, n)
    d = 0.02 * rng.standard_normal() + 10 ** rng.uniform(-6, -0.7) * rng.standard_normal(n) + s * rng.standard_normal(n)
    xf = mp.mpf(10) ** (2 * mp.mpf(float(rng.uniform(-15, -0.3)))); xd = mp.mpf(10) ** (2 * mp.mpf(float(rng.uniform(-15, -0.3))))
    if structure != 'family':
        xf = mp.mpf(0)
    delta2 = mp.mpf('0.01')
    ex = M.direct_node(d, s, fam, structure, xf, xd, delta2)
    bm = M.block_node(d, s, fam, structure, xf, xd, delta2)
    famidx = np.array([sorted(set(fam)).index(x) for x in fam]) if structure == 'family' else np.zeros(n, int)
    ld = R.node_ld(d, s, famidx, np.array([float(xf)], R.LD) if False else np.array([R.LD(str(xf))]), np.array([R.LD(str(xd))]), R.LD('0.01'))
    for name, val in (('ld', [mp.mpf(str(x[0])) for x in ld]), ('mp', list(bm))):
        for k, key in enumerate(('ll', 'mean', 'var')):
            ref = ex[k]
            rel = abs(val[k] - ref) / max(abs(ref), mp.mpf('1e-300'))
            worst[f'{name}_{key}'] = max(worst[f'{name}_{key}'], float(rel))
print(json.dumps(worst, indent=1))
