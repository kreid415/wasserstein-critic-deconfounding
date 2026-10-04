#!/usr/bin/env python
"""Build the pre-registered Tier 1 + 2 manifest (docs/PAPER_PLAN.md; decision rules docs/PREREG.md).

Decisions in force (CONSTRAINTS.md):
  * the eight standard scIB tasks only, with scIB batch/label keys (SI-02, SI-03);
  * the discriminator always takes ONE update per generator step (SI-01);
  * within a task every arm and every neural baseline uses the same backbone, epochs, batch size
    and latent size (SI-04; BACKBONES; epochs = the scvi-tools heuristic min(400, round(20000/n*400)));
  * backbone of record: scvi-tools defaults ('stock', SI-10); barycenter target = free-support W2
    barycenter, 10 cold fixed-point iterations per step (SI-14); critics use WGAN-GP Algorithm 1
    defaults (n_critic 5, lambda_GP 10, Adam 1e-4, betas (0, 0.9)).
  * X3 rows name their reference batch (the dose-0 automatic choice, fixed for every dose) and deplete a
    non-reference batch (SI-31..SI-33, X3_TARGETS); every dose of a task has the same cell count, the task's
    dose-100 size (X3_N), drawn as nested subsamples of one dose-independent permutation (draw X3_DRAW,
    scripts/fit_paper_config.py x3_draw; SI-41); X13 CPU baselines (harmony / scanorama knob x 6 values at
    the backbone's latent size, plus the tool default dimensions; PCA once; SI-34, SI-36, SI-46, CPU_BASELINES) are rows
    for scripts/run_cpu_baselines.py, which fit_paper_config.py refuses.

Design of record: --design pilot (SI-16), staged by the pre-registered rules of docs/PREREG.md:
  A1     atac_small, immune, sim1 x both decoders x seeds A1_SEED0.. (100-102) x (lambda=0 + 6 arms x
         LAMBDA_GRID), posterior-mean input, no standardisation. A1 seeds are disjoint from the X1
         seeds (0-4), so no A1 fit is ever an X1 fit.
  R1     scripts/freeze_a1_grid.py fixes a 6-point lambda grid per arm family x decoder (SI-18) and
         resolves X1's lam g1..g6.
  R4-a1  scripts/freeze_matched_lambda.py --stage a1 resolves the A2/A3 window (a1_matched_lo /
         a1_matched / a1_matched_hi) from A1 scores (SI-21).
  A2/A3  pilots, run BEFORE X1, on A1's tasks, seeds and both decoders (SI-27): posterior sample vs mean
         (A2), per-dimension standardisation on vs off (A3). Their mean / off halves are A1 fits, so only
         the new halves (sample; standardisation on) are rows of this manifest.
  R2/R3  scripts/decide_a2_a3.py resolves X1's adv_input ('A2') and zstd ('A3') placeholders and sets
         the follow-ups' adversarial rows (built with mean / 0) to the same decision (SI-19, SI-20).
  X1     8 tasks x both decoders x seeds 0-4. Then R4-x1 (freeze_matched_lambda.py --stage x1)
         resolves the follow-ups' 'matched*' lambdas: per task x arm x decoder, the grid point whose
         seed-mean unscaled batch score is closest to b*; lo / hi = its grid neighbours. The follow-ups
         (X3, X6, X7, X8, X12, X15) run on FOLLOWUP_SEEDS 10-12, disjoint from X1 and A1 (SI-26); X13 is not
         lambda-matched and keeps seeds 0-4.
A row that holds a placeholder cannot be fitted: fit_paper_config.py refuses a non-numeric lambda and
a non-integer zstd, and the training plan refuses an adv_input other than mean / sample.
Rows identical to an X1 row (all settings equal, seed included) are not refitted; the analysis reads
the X1 fit (column reuses_x1 in the summary). In the pilot design no follow-up row can equal an X1 row
(fresh seeds); freeze_matched_lambda.py --stage x1 stops if one does after resolution.

--design shared (costing alternative without A1): X1 runs the 10-point LAMBDA_GRID for every arm and
A2/A3 come after X1, with X1-matched lambdas and their mean / off halves taken from X1 rows.

Usage: python scripts/build_paper_manifest.py --backbone stock --design pilot --uncond-seeds 5 \\
           --bary-iter 10 --out scripts/paper_manifest.tsv
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
A1_TASKS = ["atac_small", "immune", "sim1"]    # design 'pilot': A1 calibration tasks (3 of 8; disclosed, PREREG.md)
A1_SEED0 = 100                                  # A1/A2/A3 seeds 100, 101, ...: disjoint from the X1 seeds 0-4
A1_MATCHED = ["a1_matched_lo", "a1_matched", "a1_matched_hi"]   # A2/A3 window, resolved from A1 (R4 stage a1)
ADV_INPUT_PENDING, ZSTD_PENDING = "A2", "A3"    # X1 placeholders, resolved by decide_a2_a3.py (R2 / R3)
FOLLOWUP_SEEDS = [10, 11, 12]                   # X3/X6/X7/X8/X12/X15: fresh seeds, disjoint from X1 (0-4) and A1 (SI-26)
CRITICS = ["reference", "pooled", "barycenter"]
ARMS = ["discriminator"] + CRITICS + ["mmd", "sinkhorn"]
N_CRITIC = 5
COLS = ["tag", "experiment", "task", "counts", "arm", "lam", "n_critic", "adv_input", "zstd", "cond", "decoder",
        "n_latent", "n_layers", "n_hidden", "likelihood", "batch_size", "max_epochs", "train_size", "seed",
        "reference", "extra"]
# X3: (depleted batch, depleted types, reference). Reference = the repo rule (select_reference_batch, max cell-type
# entropy) on the dose-0 subsample, fixed for every dose (SI-31); depleted batch = the largest NON-reference batch;
# types = its two most abundant types present in >= 3 batches, except sim2, where only Group1 is depleted so that
# Batch3Sub1 (Group1 and Group2 only) stays at every dose (SI-32); atac_small target signed off as SI-33.
# Read from prepped_scib/<task>__scib.h5ad (2026-10-02/03); tests/scvi/test_x3_x13_design.py re-derives every entry.
X3_TARGETS = {
    "atac_small": ("Fang et al. - CEMBA180305_2B", ["Excitatory Neurons", "Inhibitory Neurons"],
                   "Cusanovich et al. - WholeBrainA_62216"),
    "immune_hum_mou": ("MCA_BM_2", ["Neutrophils", "Monocyte progenitors"], "Oetjen_A"),
    "sim2": ("Batch3Sub1", ["Group1"], "Batch4Sub2"),
    "pancreas": ("inDrop3", ["alpha", "acinar"], "inDrop1"),
}
# X3 cell count per task (SI-41, code check CR-04): the dose-100 size = cells left after removing every declared
# cell, capped at 20,000 (the cap X3 used before; it binds on immune_hum_mou only). Read from prepped_scib on
# 2026-10-03 (cells - declared cells: atac_small 11,270 - 2,868; pancreas 16,382 - 1,973; sim2 19,318 - 1,446;
# immune_hum_mou 97,861 - 10,873 > 20,000); tests/scvi/test_x3_x13_design.py re-derives every entry.
X3_N = {"atac_small": 8402, "immune_hum_mou": 20000, "sim2": 17872, "pancreas": 14409}
X3_DRAW = "nested_v1"   # = fit_paper_config.X3_DRAW; the fitter refuses composition specs without it
# X13 CPU baselines (scripts/run_cpu_baselines.py; scripts/fit_paper_config.py refuses these arms): each method's
# strength knob x 6 values at the backbone's 10 dimensions, run once (deterministic), plus the tool's default
# dimensions at the default knob value as sensitivity. PCA has no knob: once, the uncorrected anchor (SI-36).
CPU_BASELINES = {   # arm: (knob, values (SI-34 / SI-46), default value, tool-default dimensions)
    "harmony": ("theta", [0, 0.5, 1, 2, 4, 8], 2, 50),
    # SI-46 (2026-10-04) replaces SI-35's knn 160 by knn 2: on immune_hum_mou knn 160 did not finish within 8 h
    "scanorama": ("knn", [2, 5, 10, 20, 40, 80], 20, 100),
    "pca": (None, [0], 0, 50),
}
# X15 KL warm-up sensitivity (SI-44; PAPER_PLAN N8): extra {"kl_warmup": "stock"} keeps scvi-tools' 400-epoch warm-up,
# "complete" sets its length to the row's max_epochs (scripts/fit_paper_config.py KL_WARMUP)
X15_TASKS = ["immune", "atac_large"]
X15_ARMS = ["discriminator", "pooled", "mmd"]
X15_KL_WARMUP = ["stock", "complete"]
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


BARY_WARM_ITER = None   # set by --bary-warm-iter (diagnostic only: converges to a different fixed point)
BARY_ITER = None        # set by --bary-iter: cold fixed-point iterations per step (runner default 10)


def row(exp, task, arm, lam, seed, cond, bb, n_cells=None, **over):
    r = dict(experiment=exp, task=task, counts="scib", arm=arm, lam=lam, n_critic=n_adv_steps(arm),
             adv_input="mean", zstd=0, cond=int(cond), seed=seed, reference="auto", extra="{}", **bb)
    r.update(over)
    if arm.startswith("barycenter") and (BARY_WARM_ITER or BARY_ITER):
        ex = json.loads(r["extra"])
        if BARY_ITER:
            ex["bary_iter"] = int(BARY_ITER)
        if BARY_WARM_ITER:
            ex["bary_warm_iter"] = int(BARY_WARM_ITER)
        r["extra"] = json.dumps(ex)
    r["max_epochs"] = scvi_epochs(n_cells or TASKS[task][0])
    return r


def cpu_row(task, arm, knob, value, dims):
    """X13 CPU baseline row: knob value in 'lam', dimensions in 'n_latent', seed 0 (run once); the scVI-backbone
    fields do not apply ('na' / 0) and scripts/fit_paper_config.py refuses the arm."""
    return dict(experiment="X13", task=task, counts="scib", arm=arm, lam=value, n_critic=0, adv_input="na", zstd=0,
                cond=0, decoder="na", n_latent=dims, n_layers=0, n_hidden=0, likelihood="na", batch_size=0,
                max_epochs=0, train_size=0, seed=0, reference="auto", extra=json.dumps({"knob": knob} if knob else {}))


def build(bb, design="shared", pilot_seeds=3, uncond_seeds=5, x12_runs=8):
    R = []
    grid = LAMBDA_GRID if design == "shared" else FAMILY_GRID
    a2_tasks, a2_arms = ["atac_small", "immune"], ["discriminator", "reference", "pooled"]
    a3_arms = ["discriminator", "reference", "pooled", "mmd", "sinkhorn"]
    if design == "pilot":
        # ---- A1: calibration pilot (docs/PREREG.md R1). Seeds are disjoint from every X1 seed (0-4), so no
        #      A1 fit is an X1 fit. lambda=0 ('none') rows are the seed-paired reference of R1 and R4.
        #      Posterior-mean input and no standardisation: the defaults that A2 / A3 test against.
        a1_seeds = [A1_SEED0 + i for i in range(pilot_seeds)]
        if set(a1_seeds) & set(range(5)):
            raise ValueError(f"A1 seeds {a1_seeds} overlap the X1 seeds 0-4")
        for t, s, cond in itertools.product(A1_TASKS, a1_seeds, [True, False]):
            R.append(row("A1", t, "none", 0, s, cond, bb))
            R += [row("A1", t, a, lam, s, cond, bb, adv_input="mean", zstd=0)
                  for a, lam in itertools.product(ARMS, LAMBDA_GRID)]
        # ---- A2 / A3 pilots, scheduled BEFORE X1 (R2 / R3 set X1's adversary input and standardisation).
        #      lambda window a1_matched_lo / a1_matched / a1_matched_hi = R4 applied to A1 scores
        #      (freeze_matched_lambda.py --stage a1), so every window value is an A1 grid point. Same tasks,
        #      seeds and decoders as A1: the posterior-mean (A2) and standardisation-off (A3) halves ARE A1
        #      fits and are not emitted; only the new halves are rows.
        if not (set(a2_tasks) <= set(A1_TASKS) and set(a2_arms + a3_arms) <= set(ARMS)):
            raise ValueError("A2/A3 tasks and arms must be A1 tasks and arms (their default halves are A1 fits)")
        #      Both decoders (SI-27): one pooled R2 / R3 decision over task x arm x decoder cells.
        for t, cond, s, arm, lam in itertools.product(a2_tasks, [True, False], a1_seeds, a2_arms, A1_MATCHED):
            R.append(row("A2", t, arm, lam, s, cond, bb, adv_input="sample", zstd=0))
        for t, cond, s, arm, lam in itertools.product(a2_tasks, [True, False], a1_seeds, a3_arms, A1_MATCHED):
            R.append(row("A3", t, arm, lam, s, cond, bb, adv_input="mean", zstd=1))
    # ---- X1/X2: core benchmark, both decoders; conditioned 5 seeds, unconditioned --uncond-seeds (plan 5).
    #      Pilot design: adversarial rows carry lam g1..g6 (R1 family grid) and the adv_input / zstd
    #      placeholders of R2 / R3, so none can be fitted before its rule has run. 'none' and 'scvi_adv'
    #      rows keep the inert mean / 0 and can run while A1-A3 are in progress.
    pend = dict(adv_input=ADV_INPUT_PENDING, zstd=ZSTD_PENDING) if design == "pilot" else {}
    for t in TASKS:
        for s in range(5):
            R.append(row("X1", t, "none", 0, s, True, bb))
            R.append(row("X1", t, "scvi_adv", 0, s, True, bb))
            R += [row("X1", t, a, lam, s, True, bb, **pend) for a, lam in itertools.product(ARMS, grid)]
        for s in range(uncond_seeds):
            R.append(row("X1", t, "none", 0, s, False, bb))
            R += [row("X1", t, a, lam, s, False, bb, **pend) for a, lam in itertools.product(ARMS, grid)]
    # ---- X13: neural baselines, same backbone/epochs/batch/latent (scANVI: scIB protocol; sysVI cycle weight grid)
    for t, s in itertools.product(TASKS, range(5)):
        R.append(row("X13", t, "scanvi", 0, s, True, bb))
        R += [row("X13", t, "sysvi", w, s, True, bb) for w in [1, 2, 5, 10, 20, 50]]
    for t, (arm, (knob, values, default, tool_dims)) in itertools.product(TASKS, CPU_BASELINES.items()):
        R += [cpu_row(t, arm, knob, v, bb["n_latent"]) for v in values] + [cpu_row(t, arm, knob, default, tool_dims)]
    if design == "shared":
        # ---- A2 (no-A1 design): adversary input, posterior mean vs sample (mean half = X1 rows)
        for t, s, inp, arm, lam in itertools.product(a2_tasks, range(3), ["mean", "sample"], a2_arms,
                                                     ["matched_lo", "matched", "matched_hi"]):
            R.append(row("A2", t, arm, lam, s, True, bb, adv_input=inp))
        # ---- A3 (no-A1 design): per-dimension standardisation of the adversary input (off half = X1 rows)
        for t, s, z, arm, lam in itertools.product(a2_tasks, range(3), [0, 1], a3_arms,
                                                   ["matched_lo", "matched", "matched_hi"]):
            R.append(row("A3", t, arm, lam, s, True, bb, zstd=z))
    # ---- follow-ups (X7, X8, X3, X6, X12, X15): lambda matched from X1, so FRESH seeds (SI-26): no follow-up cell
    #      reuses an X1 fit from which its matched lambda was chosen
    if set(FOLLOWUP_SEEDS) & (set(range(max(5, uncond_seeds))) | {A1_SEED0 + i for i in range(pilot_seeds)}):
        raise ValueError(f"follow-up seeds {FOLLOWUP_SEEDS} overlap the X1 or A1 seeds")
    # ---- X7: every batch as reference (K >= 3): reference W1, fixed-reference W1, MMD to reference
    for t in ["pancreas", "sim1", "atac_small"]:
        for ref, arm, s in itertools.product(range(TASKS[t][1]), ["reference", "reference_fixed", "mmd_ref"], FOLLOWUP_SEEDS):
            R.append(row("X7", t, arm, "matched", s, True, bb, reference=str(ref)))
        # reference JS (docs/SPECS_missing_arms.md section 2, CONSTRAINTS.md SI-23): one adversary step
        for ref, s in itertools.product(range(TASKS[t][1]), FOLLOWUP_SEEDS):
            R.append(row("X7", t, "discriminator_ref", "matched", s, True, bb, reference=str(ref), n_critic=1))
    # ---- X8: number of batches V at fixed total cells and equal cells per batch
    for t, n in X8_TOTAL.items():
        for V, sub, s in itertools.product([2, 4, 8, 16], [0, 1], FOLLOWUP_SEEDS):
            ex = json.dumps(dict(subsample=dict(kind="batches", n_batches=V, subset=sub, n_cells=n)))
            for arm in ["none", "discriminator", "mmd", "reference", "pooled"]:
                R.append(row("X8", t, arm, 0 if arm == "none" else "matched", s, True, bb, n_cells=n, extra=ex))
            # critics also with the per-batch stratified sampler (SPECS section 3, SI-24)
            ex_s = json.dumps(dict(subsample=dict(kind="batches", n_batches=V, subset=sub, n_cells=n),
                                   sampler="stratified"))
            for arm in ["reference", "pooled"]:
                R.append(row("X8", t, arm, "matched", s, True, bb, n_cells=n, extra=ex_s))
    # ---- X3 (tier 2): composition-shift dose-response
    for t, (b, types, ref) in X3_TARGETS.items():
        if ref == b or ref.isdigit():
            raise ValueError(f"X3 {t}: reference {ref!r} must be a batch name other than the depleted batch")
        n = X3_N[t]
        for dose, s in itertools.product([0, 50, 80, 95, 100], FOLLOWUP_SEEDS):
            ex = json.dumps(dict(subsample=dict(kind="composition", batch=b, types=types, deplete_pct=dose, n_cells=n,
                                                draw=X3_DRAW)))
            R.append(row("X3", t, "none", 0, s, True, bb, n_cells=n, extra=ex, reference=ref))
            R += [row("X3", t, a, lam, s, True, bb, n_cells=n, extra=ex, reference=ref)
                  for a, lam in itertools.product(["discriminator", "reference", "pooled", "mmd"], ["matched_lo", "matched_hi"])]
            # oracle importance-weighted control (SPECS section 4, SI-25): weights restore the dose-0 composition
            # (fit_paper_config.depletion_oracle_weights); at dose 0 every weight is 1 (= the unweighted rows above)
            # and at dose 100 the depleted pairs are empty, so IW rows at 50 / 80 / 95 only
            if dose in (50, 80, 95):
                ex_iw = json.dumps(dict(subsample=dict(kind="composition", batch=b, types=types, deplete_pct=dose,
                                                       n_cells=n, draw=X3_DRAW), iw="depletion_oracle"))
                R += [row("X3", t, a, lam, s, True, bb, n_cells=n, extra=ex_iw, reference=ref)
                      for a, lam in itertools.product(["discriminator", "reference", "pooled", "mmd"], ["matched_lo", "matched_hi"])]
    # ---- X6 (tier 2): divergence x Lipschitz control x update budget (pooled = symmetric target)
    for t, s, (arm, k), lam in itertools.product(["atac_small", "immune", "pancreas"], FOLLOWUP_SEEDS,
                                                 [("discriminator", 1), ("discriminator_sn", 1), ("pooled", 1), ("pooled", 5),
                                                  ("pooled", 10), ("pooled_sn", 1), ("pooled_sn", 5)],
                                                 ["matched_lo", "matched", "matched_hi"]):
        R.append(row("X6", t, arm, lam, s, True, bb, n_critic=k))
    # JS + R1 penalty, gamma 10, one adversary step (SPECS section 1, SI-22)
    for t, s, lam in itertools.product(["atac_small", "immune", "pancreas"], FOLLOWUP_SEEDS, ["matched_lo", "matched", "matched_hi"]):
        R.append(row("X6", t, "discriminator_r1", lam, s, True, bb, n_critic=1, extra=json.dumps(dict(r1_gamma=10))))
    # ---- X12 (tier 2): 5 two-level factors. 8 runs = 2^(5-2), generators D = AB, E = AC (resolution III:
    #      main effects aliased with two-factor interactions). 16 runs = 2^(5-1), E = ABCD (resolution V).
    for t, s, run, arm, lam in itertools.product(["atac_small", "immune", "pancreas"], FOLLOWUP_SEEDS, range(x12_runs),
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
    # ---- X15 (SI-44, PAPER_PLAN N8): KL warm-up sensitivity, conditioned decoder: lambda=0 and discriminator / pooled
    #      critic / MMD at the matched lo / hi lambda x FOLLOWUP_SEEDS x {stock 400-epoch warm-up, warm-up that
    #      completes (length = the row's max_epochs)}. Appended last, so every earlier row keeps its position.
    for t, kl, s in itertools.product(X15_TASKS, X15_KL_WARMUP, FOLLOWUP_SEEDS):
        ex = json.dumps(dict(kl_warmup=kl))
        R.append(row("X15", t, "none", 0, s, True, bb, extra=ex))
        R += [row("X15", t, a, lam, s, True, bb, extra=ex)
              for a, lam in itertools.product(X15_ARMS, ["matched_lo", "matched_hi"])]
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
    ap.add_argument("--bary-iter", type=int, default=None,
                    help="barycenter: cold fixed-point iterations per step, init = minibatch cells (default 10)")
    ap.add_argument("--bary-warm-iter", type=int, default=None,
                    help="DIAGNOSTIC ONLY: warm start from the previous step converges to a different fixed point")
    a = ap.parse_args()
    global BARY_WARM_ITER, BARY_ITER
    BARY_WARM_ITER, BARY_ITER = a.bary_warm_iter, a.bary_iter
    R = finalize(build(BACKBONES[a.backbone], a.design, a.pilot_seeds, a.uncond_seeds, a.x12_runs))
    fit = [r for r in R if not r["reuses_x1"]]
    with open(a.out, "w") as f:
        f.write(f"# paper manifest, backbone={a.backbone}, design={a.design}, uncond_seeds={a.uncond_seeds}, "
                f"x12_runs={a.x12_runs}, bary_iter={a.bary_iter or 10}, bary_warm_iter={a.bary_warm_iter}; "
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
