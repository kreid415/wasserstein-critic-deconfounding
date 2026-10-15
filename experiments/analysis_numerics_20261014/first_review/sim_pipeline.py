"""Operating characteristics of the frozen P1 decision rule, simulated with the frozen code's own functions
(read_p1 with leave-one-seed-out standard error, combine/pool, declarations) on synthetic seed-level data with a known truth.

World: 6 real datasets (registered families), one decoder (cond 0 grids from docs/prereg/r1_record.json), 6 adversarial arms
(discriminator, reference, pooled, barycenter, mmd, sinkhorn) x 6 grid points x 5 seeds, plain scVI x 5 seeds.
Truth: both P1 arms follow the same bio-versus-batch law, the reference shifted by theta_t in biology
(theta_t = Delta + family effect + dataset effect), so the true P1 difference is theta_t when the law is linear (kappa = 0).
b* = b0 + 0.5 (b_common - b0) is recomputed from every replicate's own data (as R4 does); the summary also gives the readings with b* frozen
at its noise-free value.
Seed noise: per point and arm SD in B and C drawn log-normally around the pilot medians (0.008, 0.015), B-C correlation 0.2.
Written by the first independent rigour review (2026-10-14, review bundle artifact 3de51e25-da4a-49d3-b248-7a8143d3197d). Changes
made when it was committed: the paths; the progress lines go to stderr; and the family effects are drawn in sorted family order (the
order of a set of strings depends on the process's hash seed, which made the pooled results of S1 and S2 differ from run to run).
It supports limitation 1 of docs/ANALYSIS_SPEC.md section 16. `run_simulations.sh` runs the six scenarios with their seeds; the
summaries are in sim_pipeline_<scenario>.txt.
"""
import argparse
import json
import math
import os
import sys
import time
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, '..', '..', '..')
sys.path.insert(0, os.path.join(ROOT, 'scripts'))
import numpy as np  # noqa: E402
import analysis_x1 as A  # noqa: E402
import prereg_rules as P  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('--scenario', required=True)
ap.add_argument('--delta', type=float, default=0.0); ap.add_argument('--tau_f', type=float, default=0.0); ap.add_argument('--tau_d', type=float, default=0.0)
ap.add_argument('--kappa', type=float, default=0.0)
ap.add_argument('--worlds', type=int, default=20); ap.add_argument('--reps', type=int, default=200)
ap.add_argument('--pool_reps', type=int, default=500)
ap.add_argument('--seed', type=int, default=1)
ap.add_argument('--noise_scale', type=float, default=1.0)
ap.add_argument('--collapse', type=float, default=0.0, help='bio loss at the strongest penalties (0 = none)')
ap.add_argument('--prior_truth', type=int, default=0, help='1: draw Delta, tau_f, tau_d per world from the model prior (N(0,0.1^2), half-normal 0.1)')
ap.add_argument('--wiggle', type=int, default=0, help='1: the discriminator curve touches b* at its second point and dips below it at the third (several crossings)')
ap.add_argument('--out', required=True)
a = ap.parse_args()

R1 = json.load(open(os.path.join(ROOT, 'docs', 'prereg', 'r1_record.json')))
cond = 0
GRIDS = {'discriminator': R1['families'][f'js|{cond}']['x1_grid'], 'reference': R1['families'][f'w1|{cond}']['x1_grid'],
         'pooled': R1['families'][f'w1|{cond}']['x1_grid'], 'barycenter': R1['families'][f'w1|{cond}']['x1_grid'],
         'mmd': R1['families'][f'mmd|{cond}']['x1_grid'], 'sinkhorn': R1['families'][f'sinkhorn|{cond}']['x1_grid']}
GAIN = {'discriminator': 0.22, 'reference': 0.28, 'pooled': 0.30, 'barycenter': 0.26, 'mmd': 0.32, 'sinkhorn': 0.20}
TASKS = list(P.REAL_TASKS)
FAMS = [P.TASK_FAMILY[t] for t in TASKS]
SEEDS = [0, 1, 2, 3, 4]
rng = np.random.default_rng(a.seed)
sig = lambda x: 1 / (1 + np.exp(-x))


def make_world(rng):
    """Dataset-level truth: b0, c0, per arm/point true (B, C) *without* the reference shift, noise SDs, and theta_t."""
    delta_w, tf_w, td_w = (rng.normal(0, 0.1), abs(rng.normal(0, 0.1)), abs(rng.normal(0, 0.1))) if a.prior_truth else (a.delta, a.tau_f, a.tau_d)
    fam_eff = {f: rng.normal(0, tf_w) for f in sorted(set(FAMS))}
    theta = np.array([delta_w + fam_eff[f] + rng.normal(0, td_w) for f in FAMS])
    W = []
    for t in range(6):
        b0 = rng.uniform(0.35, 0.65); c0 = rng.uniform(0.65, 0.80)
        arms = {}
        for arm, grid in GRIDS.items():
            x = np.log10(grid) - (np.log10(grid[2]) + 0.25)
            B = b0 + GAIN[arm] * sig(x / 0.35)
            # a common bio-versus-batch law: slope -0.3, curvature kappa; collapse at the strongest points
            C = c0 - 0.3 * (B - b0) - a.kappa * (B - b0) ** 2 - a.collapse * sig((x - 1.0) / 0.3)
            if arm == 'reference':
                C = C + theta[t]
            if a.wiggle and arm == 'discriminator':
                pass
            sB = np.clip(0.008 * a.noise_scale * np.exp(0.8 * rng.normal(size=6)), 5e-4, 0.15)
            sC = np.clip(0.015 * a.noise_scale * np.exp(0.8 * rng.normal(size=6)), 5e-4, 0.15)
            arms[arm] = dict(grid=grid, B=B, C=C, sB=sB, sC=sC)
        if a.wiggle:
            Mmin = min(v['B'].max() for v in arms.values())
            v = arms['discriminator']; B = v['B'].copy()
            B[1] = b0 + 0.50 * (Mmin - b0); B[2] = b0 + 0.35 * (Mmin - b0)
            x = np.log10(v['grid']) - (np.log10(v['grid'][2]) + 0.25)
            v['B'] = B
            v['C'] = c0 - 0.3 * (B - b0) - a.kappa * (B - b0) ** 2 - a.collapse * sig((x - 1.0) / 0.3)
        W.append(dict(b0=b0, c0=c0, arms=arms, sB0=0.005 * a.noise_scale, sC0=0.008 * a.noise_scale))
    return theta, W, delta_w


def draw(rng, w):
    """Noise draw for one dataset: zero {seed: (B, C)}, points per arm in the _cell_points format."""
    zero = {s: (w['b0'] + rng.normal(0, w['sB0']), w['c0'] + rng.normal(0, w['sC0'])) for s in SEEDS}
    pts = {}
    for arm, v in w['arms'].items():
        out = []
        for k, lam in enumerate(v['grid']):
            e1 = rng.normal(size=5); e2 = 0.2 * e1 + math.sqrt(1 - 0.04) * rng.normal(size=5)
            out.append(dict(lam=float(lam), n_fail=0, B={s: float(v['B'][k] + v['sB'][k] * e1[s]) for s in SEEDS},
                            C={s: float(v['C'][k] + v['sC'][k] * e2[s]) for s in SEEDS}))
        pts[arm] = out
    return zero, pts


def bstar_of(zero, pts):
    b0 = np.mean([zero[s][0] for s in SEEDS])
    M = [max(np.mean(list(p['B'].values())) for p in pts[arm]) for arm in GRIDS]
    return b0, b0 + 0.5 * (min(M) - b0)


t0 = time.time()
cell_rows, pool_rows = [], []
for wi in range(a.worlds):
    theta, W, delta_w = make_world(rng)
    # true b* from noise-free curves (for the fixed-b* comparison)
    zero_true = {s: (w_['b0'], w_['c0']) for s in SEEDS for w_ in [W[0]]}
    per_ds = [[] for _ in range(6)]
    d_poly = []
    for t in range(6):
        zt = {s_: (W[t]['b0'], W[t]['c0']) for s_ in SEEDS}
        pt = {arm: [dict(lam=float(v['grid'][k]), n_fail=0, B={s_: float(v['B'][k]) for s_ in SEEDS}, C={s_: float(v['C'][k]) for s_ in SEEDS}) for k in range(6)] for arm, v in W[t]['arms'].items()}
        b0_, bs_ = bstar_of(zt, pt)
        d_poly.append(A.read_p1(zt, pt['reference'], pt['discriminator'], bs_, 'mean')['d'])
    d_poly = np.array(d_poly)
    for rep in range(a.reps):
        recs = []
        for t in range(6):
            zero, pts = draw(rng, W[t])
            b0, bstar = bstar_of(zero, pts)
            r = A.read_p1(zero, pts['reference'], pts['discriminator'], bstar, 'mean')
            # same data, b* frozen at the noise-free value
            Mtrue = min(W[t]['arms'][arm]['B'].max() for arm in GRIDS)
            bstar_true = W[t]['b0'] + 0.5 * (Mtrue - W[t]['b0'])
            r_fix = A.read_p1(zero, pts['reference'], pts['discriminator'], bstar_true, 'mean')
            recs.append((r, r_fix))
            per_ds[t].append(dict(d=r['d'], s=r['s'], reason=r['reason'], d_fix=r_fix['d'], s_fix=r_fix['s'], reason_fix=r_fix['reason'], theta=float(theta[t]), d_poly=float(d_poly[t])))
        if rep < a.pool_reps // a.worlds + 1:
            tasks, d, s = [], [], []
            for t, (r, _) in enumerate(recs):
                if r['reason'] is None:
                    tasks.append(TASKS[t]); d.append(r['d']); s.append(r['s'])
            if tasks:
                c = A.combine(tasks, d, s, 'family', 0.1, 0.01)
                pool_rows.append(dict(world=wi, n=len(tasks), delta_true=float(delta_w), mean=c['mean'], lo=c['lo'], hi=c['hi'], p_above=c['p_above'],
                                      p_below=c['p_below'], p_within=c['p_within'], decl=c['declaration'],
                                      mean_theta_eval=float(np.mean([theta[TASKS.index(x)] for x in tasks])),
                                      theta_cov=[bool(th['lo'] <= theta[TASKS.index(x)] <= th['hi']) for x, th in zip(tasks, c['theta'])]))
    for t in range(6):
        cell_rows.append(dict(world=wi, ds=t, rows=per_ds[t]))
    print(f'world {wi+1}/{a.worlds}  {time.time()-t0:.0f}s', file=sys.stderr, flush=True)
json.dump(dict(args=vars(a), cells=cell_rows, pool=pool_rows), open(a.out, 'w'))

# summary: variance calibration of the leave-one-seed-out SE and coverage
sd_true, rms_s, cover, cover_fix, sd_fix, rms_s_fix, evalfrac, bias, bias_th = [], [], [], [], [], [], [], [], []
for cr in cell_rows:
    ok = [r for r in cr['rows'] if r['reason'] is None]
    okf = [r for r in cr['rows'] if r['reason_fix'] is None]
    evalfrac.append(len(ok) / len(cr['rows']))
    if len(ok) > 20:
        d = np.array([r['d'] for r in ok]); s = np.array([r['s'] for r in ok]); th = cr['rows'][0]['d_poly']
        sd_true.append(d.std(ddof=1)); rms_s.append(math.sqrt(np.mean(s ** 2))); bias.append(d.mean() - th); bias_th.append(d.mean() - cr['rows'][0]['theta'])
        cover.append(np.mean(np.abs(d - th) <= 1.96 * s))
    if len(okf) > 20:
        d = np.array([r['d_fix'] for r in okf]); s = np.array([r['s_fix'] for r in okf]); th = cr['rows'][0]['d_poly']
        sd_fix.append(d.std(ddof=1)); rms_s_fix.append(math.sqrt(np.mean(s ** 2))); cover_fix.append(np.mean(np.abs(d - th) <= 1.96 * s))
sd_true, rms_s = np.array(sd_true), np.array(rms_s)
print(f"scenario {a.scenario}: {len(sd_true)} dataset cells")
if len(sd_true) == 0:
    cell_rows = []
if len(sd_true):
    print(f"  P1 evaluable fraction {np.mean(evalfrac):.3f}; bias of d (mean d - noise-free polyline reading) mean {np.mean(bias):+.4f} sd {np.std(bias):.4f}")
    print(f"  bias of d relative to the smooth truth theta_t (interpolation + noise): mean {np.mean(bias_th):+.4f}, sd {np.std(bias_th):.4f}, max |.| {np.max(np.abs(bias_th)):.4f}")
    print(f"  ratio rms(jackknife s) / true SD of d: median {np.median(rms_s/sd_true):.3f}, 10-90% {np.percentile(rms_s/sd_true,10):.3f}-{np.percentile(rms_s/sd_true,90):.3f}")
    print(f"  coverage of d +/- 1.96 s about theta: mean {np.mean(cover):.3f}, min {np.min(cover):.3f}")
    if sd_fix:
        sd_fix, rms_s_fix = np.array(sd_fix), np.array(rms_s_fix)
        print(f"  with b* frozen at its noise-free value: ratio median {np.median(rms_s_fix/sd_fix):.3f}; coverage {np.mean(cover_fix):.3f}; true SD median {np.median(sd_fix):.4f} vs {np.median(sd_true):.4f} with b* re-estimated")
pr = pool_rows
n = len(pr)
print(f"  pooled: {n} replicate worlds; declarations above {sum(r['decl']=='above' for r in pr)/n:.3f} below {sum(r['decl']=='below' for r in pr)/n:.3f} equivalent {sum(r['decl']=='equivalent' for r in pr)/n:.3f}")
print(f"  coverage of the 95% interval for Delta (truth {'drawn from the prior' if a.prior_truth else a.delta}): {np.mean([r['lo']<=r['delta_true']<=r['hi'] for r in pr]):.3f};  mean posterior mean {np.mean([r['mean'] for r in pr]):+.4f}")
print(f"  coverage of the 95% interval for the mean of evaluable theta: {np.mean([r['lo']<=r['mean_theta_eval']<=r['hi'] for r in pr]):.3f}; per-dataset theta coverage {np.mean([c for r in pr for c in r['theta_cov']]):.3f}")
