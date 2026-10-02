#!/usr/bin/env python
"""Build the pre-registered Tier 1 + 2 manifest (docs/PAPER_PLAN.md, revised 2026-10-02).

Decisions from 2026-10-02:
  * the eight standard scIB tasks only, with scIB batch/label keys (no new datasets);
  * the discriminator always takes ONE update per generator step;
  * within a task every arm and every neural baseline uses the same backbone, epochs, batch size
    and latent size (BACKBONES; epochs = the scvi-tools heuristic min(400, round(20000/n*400)));
  * barycenter critic rebuilt (free-support W2 barycenter target, wcd.barycenter);
  * critics use WGAN-GP Algorithm 1 defaults (n_critic 5, lambda_GP 10, Adam 1e-4, betas (0, 0.9)).

Wall-time design (no sequential calibration stage):
  * X1 uses ONE fixed log grid of lambda for every arm (LAMBDA_GRID). This replaces the A1
    calibration pilot, so no lambda is chosen from the data before the main benchmark.
  * Follow-up experiments use 'matched' lambdas, resolved from X1 by the pre-registered rule in
    scripts/freeze_matched_lambda.py (per task x arm x conditioning: the grid point whose seed-mean
    scIB batch score is closest to the target b*; lo / hi = its grid neighbours). Their X1 tasks are
    queued first, so they never wait.
  * Rows identical to an X1 row (all settings equal, seed included) are not refitted; the analysis
    reads the X1 fit (column reuses_x1 in the summary).

Usage: python scripts/build_paper_manifest.py --backbone scib --out scripts/paper_manifest_scib.tsv
"""
import argparse
import hashlib
import itertools
import json

TASKS = {  # name: (cells, batches under the scIB key) -- prepped_scib/<task>__scib.h5ad
    "pancreas": (16382, 9), "lung": (32472, 16), "immune": (33506, 10), "immune_hum_mou": (97861, 23),
    "sim1": (12097, 6), "sim2": (19318, 16), "atac_small": (11270, 3), "atac_large": (84813, 11),
}
BACKBONES = {
    # scIB's scVI (scib 1.1.7 scib.integration.scvi): n_latent 30, 2 layers, n_hidden 128, NB, all cells
    "scib":  dict(decoder="SCVI", n_latent=30, n_layers=2, n_hidden=128, likelihood="nb", train_size=1.0, batch_size=128),
    # scvi-tools 1.4.2 SCVI defaults
    "stock": dict(decoder="SCVI", n_latent=10, n_layers=1, n_hidden=128, likelihood="zinb", train_size=0.9, batch_size=128),
}
LAMBDA_GRID = [0.1, 0.3, 1, 3, 10, 30, 100, 300, 1000, 3000]   # half-decade steps, 4.5 decades
FAMILY_GRID = [f"g{i}" for i in range(1, 7)]   # design 'pilot': 6 per-family points fixed from A1
CRITICS = ["reference", "pooled", "barycenter"]
ARMS = ["discriminator"] + CRITICS + ["mmd", "sinkhorn"]
N_CRITIC = 5
COLS = ["tag", "experiment", "task", "counts", "arm", "lam", "n_critic", "adv_input", "zstd", "cond", "decoder",
        "n_latent", "n_layers", "n_hidden", "likelihood", "batch_size", "max_epochs", "train_size", "seed",
        "reference", "extra"]
# X3: depletion target = largest batch; types = its two most abundant types present in >= 3 batches
# (read from prepped_scib/<task>__scib.h5ad on 2026-10-02)
X3_TARGETS = {
    "atac_small": ("Cusanovich et al. - WholeBrainA_62216", ["Inhibitory Neurons", "Excitatory Neurons"]),
    "immune_hum_mou": ("MCA_BM_2", ["Neutrophils", "Monocyte progenitors"]),
    "sim2": ("Batch3Sub1", ["Group1", "Group2"]),
    "pancreas": ("inDrop3", ["alpha", "acinar"]),
}
# X8: total cells fixed per task, equal cells per batch; the largest N for which V = 2 still has
# >= 3 eligible batches and V = 16 has >= 16 (batch sizes in prepped_scib, 2026-10-02)
X8_TOTAL = {"immune_hum_mou": 16000, "lung": 6000, "sim2": 3000}


def scvi_epochs(n):
    """scvi-tools get_max_epochs_heuristic: min(400, round(20000 / n * 400))."""
    return int(min(400, round(20000 / n * 400)))


def n_adv_steps(arm):
    base = arm[:-3] if arm.endswith("_sn") else arm
    if base in CRITICS + ["reference_fixed"]:
        return N_CRITIC
    return 1 if base == "discriminator" else 0


BARY_WARM_ITER = None   # set by --bary-warm-iter; recorded in each barycenter row's extra


def row(exp, task, arm, lam, seed, cond, bb, n_cells=None, **over):
    r = dict(experiment=exp, task=task, counts="scib", arm=arm, lam=lam, n_critic=n_adv_steps(arm),
             adv_input="mean", zstd=0, cond=int(cond), seed=seed, reference="auto", extra="{}", **bb)
    r.update(over)
    if arm.startswith("barycenter") and BARY_WARM_ITER:
        ex = json.loads(r["extra"]); ex["bary_warm_iter"] = int(BARY_WARM_ITER); r["extra"] = json.dumps(ex)
    r["max_epochs"] = scvi_epochs(n_cells or TASKS[task][0])
    return r


def build(bb, design="shared", pilot_seeds=3, uncond_seeds=5, x12_runs=8):
    R = []
    grid = LAMBDA_GRID if design == "shared" else FAMILY_GRID
    if design == "pilot":
        # ---- A1: calibration pilot on the full shared grid; fixes each family's 6-point X1 grid
        for t, s, cond in itertools.product(["atac_small", "immune", "sim1"], range(pilot_seeds), [True, False]):
            R += [row("A1", t, a, lam, s, cond, bb) for a, lam in itertools.product(ARMS, LAMBDA_GRID)]
    # ---- X1/X2: core benchmark. Conditioned: 5 seeds; unconditioned: 3 seeds.
    for t in TASKS:
        for s in range(5):
            R.append(row("X1", t, "none", 0, s, True, bb))
            R.append(row("X1", t, "scvi_adv", 0, s, True, bb))
            R += [row("X1", t, a, lam, s, True, bb) for a, lam in itertools.product(ARMS, grid)]
        for s in range(uncond_seeds):
            R.append(row("X1", t, "none", 0, s, False, bb))
            R += [row("X1", t, a, lam, s, False, bb) for a, lam in itertools.product(ARMS, grid)]
    # ---- X13: neural baselines, same backbone/epochs/batch/latent (scANVI: scIB protocol; sysVI cycle weight grid)
    for t, s in itertools.product(TASKS, range(5)):
        R.append(row("X13", t, "scanvi", 0, s, True, bb))
        R += [row("X13", t, "sysvi", w, s, True, bb) for w in [1, 2, 5, 10, 20, 50]]
    # ---- A2: adversary input, posterior mean vs sample (mean half = X1 rows)
    for t, s, inp, arm, lam in itertools.product(["atac_small", "immune"], range(3), ["mean", "sample"],
                                                 ["discriminator", "reference", "pooled"], ["matched_lo", "matched", "matched_hi"]):
        R.append(row("A2", t, arm, lam, s, True, bb, adv_input=inp))
    # ---- A3: per-dimension standardisation of the adversary input (off half = X1 rows)
    for t, s, z, arm, lam in itertools.product(["atac_small", "immune"], range(3), [0, 1],
                                               ["discriminator", "reference", "pooled", "mmd", "sinkhorn"],
                                               ["matched_lo", "matched", "matched_hi"]):
        R.append(row("A3", t, arm, lam, s, True, bb, zstd=z))
    # ---- X7: every batch as reference (K >= 3): reference W1, fixed-reference W1, MMD to reference
    for t in ["pancreas", "sim1", "atac_small"]:
        for ref, arm, s in itertools.product(range(TASKS[t][1]), ["reference", "reference_fixed", "mmd_ref"], range(3)):
            R.append(row("X7", t, arm, "matched", s, True, bb, reference=str(ref)))
    # ---- X8: number of batches V at fixed total cells and equal cells per batch
    for t, n in X8_TOTAL.items():
        for V, sub, s in itertools.product([2, 4, 8, 16], [0, 1], range(3)):
            ex = json.dumps(dict(subsample=dict(kind="batches", n_batches=V, subset=sub, n_cells=n)))
            for arm in ["none", "discriminator", "mmd", "reference", "pooled"]:
                R.append(row("X8", t, arm, 0 if arm == "none" else "matched", s, True, bb, n_cells=n, extra=ex))
    # ---- X3 (tier 2): composition-shift dose-response
    for t, (b, types) in X3_TARGETS.items():
        n = min(20000, TASKS[t][0])
        for dose, s in itertools.product([0, 50, 80, 95, 100], range(3)):
            ex = json.dumps(dict(subsample=dict(kind="composition", batch=b, types=types, deplete_pct=dose, n_cells=n)))
            R.append(row("X3", t, "none", 0, s, True, bb, n_cells=n, extra=ex))
            R += [row("X3", t, a, lam, s, True, bb, n_cells=n, extra=ex)
                  for a, lam in itertools.product(["discriminator", "reference", "pooled", "mmd"], ["matched_lo", "matched_hi"])]
    # ---- X6 (tier 2): divergence x Lipschitz control x update budget (pooled = symmetric target)
    for t, s, (arm, k), lam in itertools.product(["atac_small", "immune", "pancreas"], range(3),
                                                 [("discriminator", 1), ("discriminator_sn", 1), ("pooled", 1), ("pooled", 5),
                                                  ("pooled", 10), ("pooled_sn", 1), ("pooled_sn", 5)],
                                                 ["matched_lo", "matched", "matched_hi"]):
        R.append(row("X6", t, arm, lam, s, True, bb, n_critic=k))
    # ---- X12 (tier 2): 5 two-level factors. 8 runs = 2^(5-2), generators D = AB, E = AC (resolution III:
    #      main effects aliased with two-factor interactions). 16 runs = 2^(5-1), E = ABCD (resolution V).
    for t, s, run, arm, lam in itertools.product(["atac_small", "immune", "pancreas"], range(3), range(x12_runs),
                                                 ["discriminator", "reference"], ["matched_lo", "matched_hi"]):
        A, B, C = run & 1, (run >> 1) & 1, (run >> 2) & 1
        if x12_runs == 8:
            D, E = A ^ B, A ^ C
        else:
            D = (run >> 3) & 1
            E = A ^ B ^ C ^ D
        ex = json.dumps(dict(factorial_run=run, adv_width=[32, 128][B], adv_lr=[1e-4, 1e-3][C]))
        R.append(row("X12", t, arm, lam, s, True, bb, extra=ex, n_latent=[10, 30][A],
                     batch_size=[128, 512][D], decoder=["SCVI", "LinearSCVI"][E]))
    return R


def finalize(R):
    """Tag rows; mark rows whose settings equal an X1 row (they reuse that fit). A 'matched*' lambda
    is resolved to an X1 grid value at freeze time, so it is compared with lambda as a wildcard."""
    def key(r, wild=False):
        return "|".join(f"{c}={'*' if (wild and c == 'lam') else r[c]}" for c in COLS if c not in ("tag", "experiment"))
    x1 = {key(r) for r in R if r["experiment"] == "X1"} | {key(r, wild=True) for r in R if r["experiment"] == "X1"}
    out, seen = [], set()
    for r in R:
        k = key(r)
        r["tag"] = f"{r['experiment']}_{r['task']}_{r['arm']}_l{r['lam']}_c{r['cond']}_s{r['seed']}_" + \
                   hashlib.sha1((r["experiment"] + "|" + k).encode()).hexdigest()[:8]
        matched = isinstance(r["lam"], str) and r["lam"].startswith("matched")
        r["reuses_x1"] = int(r["experiment"] != "X1" and (key(r, wild=True) if matched else k) in x1)
        if (r["experiment"], k) in seen:
            raise ValueError(f"duplicate row {r['tag']}")
        seen.add((r["experiment"], k))
        out.append(r)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", required=True, choices=sorted(BACKBONES))
    ap.add_argument("--out", required=True)
    ap.add_argument("--design", choices=["shared", "pilot"], default="shared")
    ap.add_argument("--pilot-seeds", type=int, default=3)
    ap.add_argument("--uncond-seeds", type=int, default=5, help="seeds of the unconditioned X1 block (plan: 5)")
    ap.add_argument("--x12-runs", type=int, choices=[8, 16], default=8, help="X12 fractional factorial size (plan: 8)")
    ap.add_argument("--bary-warm-iter", type=int, default=None,
                    help="barycenter: fixed-point iterations warm-started from the previous step (default: cold, 10)")
    a = ap.parse_args()
    global BARY_WARM_ITER
    BARY_WARM_ITER = a.bary_warm_iter
    R = finalize(build(BACKBONES[a.backbone], a.design, a.pilot_seeds, a.uncond_seeds, a.x12_runs))
    fit = [r for r in R if not r["reuses_x1"]]
    with open(a.out, "w") as f:
        f.write(f"# paper manifest, backbone={a.backbone}, design={a.design}, uncond_seeds={a.uncond_seeds}, "
                f"x12_runs={a.x12_runs}, bary_warm_iter={a.bary_warm_iter}; "
                f"lambda grid {LAMBDA_GRID}; 'g*' = per-family "
                f"grid from A1; 'matched*' lambdas are "
                f"resolved from X1 by scripts/freeze_matched_lambda.py. Rows reusing an X1 fit are omitted "
                f"({len(R) - len(fit)} of {len(R)}).\n")
        f.write("\t".join(COLS) + "\n")
        for r in fit:
            f.write("\t".join(str(r[c]) for c in COLS) + "\n")
    print(f"{len(fit)} fits ({len(R) - len(fit)} design rows reuse X1 fits) -> {a.out}")


if __name__ == "__main__":
    main()
