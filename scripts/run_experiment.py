"""Unified driver for the reviewer-response experiments E1/E2/E4/E8.

# WHY: E1 (lambda_adv Pareto), E2 (backbone sweep), E4 (reference design), E8 (multibatch
#      scaling) are the SAME loop over (config) -> evaluate_config -> row, differing only
#      in which axis is swept. One audited driver keeps the training/eval path identical
#      across experiments so cross-experiment comparisons are apples-to-apples.
#      (E3 external baselines and E5 biology use their own scripts; E6 has its own.)
# HOW: --experiment selects the config grid; each config is trained with the shared
#      harness.evaluate_config and appended to a tidy CSV (one row per fitted model).
"""
import argparse
import json
import os

import pandas as pd

from wcd_vae.wcd.experiment import evaluate_config, load_task

# lambda_adv grid for the Pareto front (R2.2 / R3.3). 0 == no adversarial term.
LAMBDA_GRID = [0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0]
# Rebuilt E2 (de-scCRAFT): native VAEs spanning likelihood x decoder batch-conditioning.
# NB/ZINB/LDVAE are run both conditioned and unconditioned so decoder-side integration is a
# measured axis; Gaussian/Poisson are simple unconditioned controls. scCRAFT is dropped.
BACKBONES = [
    "Gaussian", "Poisson",
    "NB", "NB_uncond",
    "ZINB", "ZINB_uncond",
    "LDVAE", "LDVAE_uncond",
]
SEEDS = [0, 1, 2]

# E10 -- optimisation sensitivity. WHY these values: bs 1024 is the production setting;
# 4096 is a 4x increase (gradient steps 2400 -> 600 on pancreas) chosen because it is the
# largest that fits the 8 GB card at every dataset. The lr multipliers span "no
# compensation" (1x), the sqrt-scaling rule (2x = sqrt(4)) and linear scaling (4x), so the
# arm tests whether a larger batch can be compensated at all.
BATCH_SIZES = [1024, 4096]
LR_MULTS = [1.0, 2.0, 4.0]
BASE_LR = 1e-3


def _resume_key(method, backbone, d_coef, seed, batch_size=None, lr_g=None):
    """Identity of a configuration for --resume.

    MUST include every field the experiment grid varies. E10 varies batch_size and lr
    while holding method/backbone/d_coef/seed fixed, so a key without them collapses its
    24 configs to 6 and silently skips 18 as already done. Defaults (None -> the
    production settings) keep keys stable for rows written before these columns existed,
    so an in-progress E1/E2/E8 wave still resumes correctly.
    """
    bs = 1024 if batch_size is None or (isinstance(batch_size, float) and batch_size != batch_size) else int(batch_size)
    lr = 1e-3 if lr_g is None or (isinstance(lr_g, float) and lr_g != lr_g) else float(lr_g)
    return (method, backbone, float(d_coef), int(seed), bs, round(lr, 12))


def configs_for(experiment, task_entry, reference_name=None, batch_size_only=None):
    """Yield config dicts (kwargs for evaluate_config) for the requested experiment.

    ``reference_name`` is the ENTROPY-selected reference batch from load_task; it is
    passed by name so training resolves the correct index (the literal
    ``reference_batch=0`` retained alongside it is the legacy positional fallback).
    """
    if experiment == "E1":
        # Pareto front: both heads x full lambda grid x 3 seeds, default backbone.
        for critic in (False, True):
            for lam in LAMBDA_GRID:
                for seed in SEEDS:
                    yield {"critic": critic, "d_coef": lam, "seed": seed,
                           "reference_batch": 0 if critic else None,
                       "reference_batch_name_str": reference_name if critic else None}
    elif experiment == "E2":
        # Architecture generality: 4 backbones x both heads x 3 seeds at the paper's
        # operating point (d_coef=0.2). Capacity sweep handled by --zdim/--override.
        for backbone in BACKBONES:
            for critic in (False, True):
                for seed in SEEDS:
                    yield {"critic": critic, "d_coef": 0.2, "seed": seed, "backbone": backbone,
                           "reference_batch": 0 if critic else None,
                       "reference_batch_name_str": reference_name if critic else None}
    elif experiment == "E8":
        # Multibatch scaling handled by caller varying --batch-count; here just both
        # heads x 3 seeds at the operating point with the full metric suite.
        for critic in (False, True):
            for seed in SEEDS:
                yield {"critic": critic, "d_coef": 0.2, "seed": seed,
                       "reference_batch": 0 if critic else None,
                       "reference_batch_name_str": reference_name if critic else None}
    elif experiment == "E10":
        # Optimisation sensitivity: does the critic-vs-discriminator comparison depend on
        # the optimiser settings? Both heads x {bs} x {lr} x 3 seeds at the operating
        # point. bs=1024/lr=1x is the production setting and serves as the baseline cell;
        # it is included so every other cell has a matched within-experiment reference.
        # batch_size_only lets the wave split E10 by batch size. WHY THAT SPLIT EXISTS:
        # the bs=4096 arm exhausts the 8 GiB card whenever ~6 workers share it (observed
        # on BOTH heads, three times in one wave), so those shards must run alone -- but
        # the bs=1024 arm is the production setting and has never OOM'd, so serialising
        # it too would waste ~11 h of lane time for nothing.
        for critic in (False, True):
            for bs in BATCH_SIZES:
                if batch_size_only is not None and bs != batch_size_only:
                    continue
                for mult in LR_MULTS:
                    if bs == 1024 and mult != 1.0:
                        continue  # only the production batch size needs its lr baseline
                    for seed in SEEDS:
                        yield {"critic": critic, "d_coef": 0.2, "seed": seed,
                               "batch_size": bs, "lr_g": BASE_LR * mult, "lr_d": BASE_LR * mult,
                               "reference_batch": 0 if critic else None,
                               "reference_batch_name_str": reference_name if critic else None}
    else:
        raise ValueError(f"run_experiment does not handle '{experiment}' (see dedicated script)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", required=True, choices=["E1", "E2", "E8", "E10"])
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--registry", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch-count", type=int, default=None)
    ap.add_argument("--balance", action="store_true")
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--zdim", type=int, default=256)
    ap.add_argument("--backbone", default=None,
                    help="backbone for E1/E8 (default NB primary post-scCRAFT-drop); "
                         "E2 sweeps its own set and ignores this")
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--head", choices=["discriminator", "critic", "both"], default="both",
                    help="run only one adversarial head (lets E1 split into two shorter jobs)")
    ap.add_argument("--d-coef-only", type=float, default=None,
                    help="E1: run only this single lambda value")
    ap.add_argument("--batch-size-only", type=int, default=None,
                    help="E10: run only this batch size (lets the wave serialise the "
                         "VRAM-heavy bs=4096 arm without serialising bs=1024)")
    ap.add_argument("--seed-only", type=int, default=None,
                    help="run only this single seed (makes each SLURM task exactly one config)")
    ap.add_argument("--embed-out", default=None,
                    help="directory to persist latents (.npz per config); use SCRATCH. "
                         "OMITTING THIS IS ALMOST ALWAYS A MISTAKE on a production wave -- "
                         "see the warning emitted at startup.")
    ap.add_argument("--no-embed-ok", action="store_true",
                    help="acknowledge running WITHOUT --embed-out (suppresses the warning); "
                         "only appropriate for smoke tests and timing probes")
    ap.add_argument("--resume", action="store_true",
                    help="skip configs already present in --out (by method,backbone,d_coef,seed)")
    args = ap.parse_args()

    # WHY THIS WARNING EXISTS: metrics-only CSVs cannot be re-analysed. Adding ANY new
    # embedding-derived metric (PAGA, kBET, probes, trajectory) to a finished wave costs a
    # full retraining run, because the latent is gone. This has now happened TWICE on this
    # project: once for PAGA, and again on 2026-08-11 when kbet was requested after a
    # 918-config / 24 h wave that had been launched without --embed-out. The flag is
    # optional and defaults to None, so the omission is silent and only surfaces weeks
    # later as a re-run bill. Embeddings are ~19 GB for the full light wave -- always
    # cheaper than the compute needed to regenerate them.
    if not args.embed_out and not args.no_embed_ok:
        import warnings
        warnings.warn(
            "\n"
            "*** RUNNING WITHOUT --embed-out: latents will NOT be persisted. ***\n"
            "    Any future embedding-derived metric (kBET, PAGA, probes, trajectory)\n"
            "    will require RETRAINING every config in this run.\n"
            "    Pass --embed-out <dir> (dataset-specific; a flat dir makes datasets\n"
            "    clobber each other), or --no-embed-ok if this is a smoke/timing run.\n",
            stacklevel=2,
        )

    with open(args.registry) as fh:
        registry = json.load(fh)
    entry = registry[args.dataset]

    # create output dir upfront so the incremental per-config CSV writes below succeed.
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    adata, batch_key, celltype_key, largest = load_task(
        args.dataset, batch_count=args.batch_count, balance=args.balance,
        data_root=args.data_root, registry=registry,
    )
    # WHY: load_task now returns the ENTROPY-selected reference batch (max cell-type
    #   Shannon entropy, ties by size). We pass it by NAME via reference_batch_name_str so
    #   training resolves the correct index -- the old `reference_batch=0` was the
    #   ALPHABETICALLY first batch, which on 4/6 datasets was one of the smallest.
    print(f"[{args.dataset}] n_obs={adata.n_obs} batch_key={batch_key} reference={largest} "
          f"n_batches={adata.obs[batch_key].nunique()}")

    # resume: load already-computed configs so a re-run skips them.
    done = set()
    rows = []
    if args.resume and os.path.exists(args.out) and os.path.getsize(args.out) > 0:
        try:
            prev = pd.read_csv(args.out)
        except pd.errors.EmptyDataError:
            prev = None
        if prev is not None and len(prev) > 0:
            rows = prev.to_dict("records")
            for r in rows:
                # WHY: a FAILED row records a gap, not a completed config -- treating it
                #      as done would make the failure permanent across every retry.
                if isinstance(r.get("failed"), str) and r["failed"]:
                    continue
                done.add(_resume_key(
                    r["method"], r.get("backbone", "NB"), r["d_coef"], r["seed"],
                    r.get("batch_size"), r.get("lr_g")))
        print(f"[resume] loaded {len(done)} completed configs from {args.out}")

    for i, cfg in enumerate(configs_for(args.experiment, entry, reference_name=largest, batch_size_only=args.batch_size_only)):
        # head filter: lets E1 be split into two shorter jobs (one per adversarial head).
        if args.head != "both":
            want_critic = args.head == "critic"
            if bool(cfg.get("critic")) != want_critic:
                continue
        if args.d_coef_only is not None and abs(float(cfg["d_coef"]) - args.d_coef_only) > 1e-9:
            continue
        # Seed-level split: one SLURM task == exactly one config (wall-safe fan-out).
        if args.seed_only is not None and int(cfg["seed"]) != args.seed_only:
            continue
        # E2 fan-out: when --backbone names a config that E2 already carries, run ONLY
        # that backbone's arm (one short job per backbone across idle CPU nodes).
        if args.experiment == "E2" and args.backbone is not None and cfg.get("backbone") != args.backbone:
            continue
        cfg = dict(cfg, epochs=args.epochs, z_dim=args.zdim)
        # --backbone sets the E1/E8 backbone (E2 configs already carry their own).
        if args.backbone is not None and "backbone" not in cfg:
            cfg["backbone"] = args.backbone
        method = "critic" if cfg.get("critic") else "discriminator"
        key = _resume_key(method, cfg.get("backbone", "NB"), cfg["d_coef"], cfg["seed"],
                          cfg.get("batch_size"), cfg.get("lr_g"))
        if key in done:
            continue
        try:
            row = evaluate_config(adata, batch_key, celltype_key, **cfg, embed_out=(os.path.join(args.embed_out, args.dataset) if args.embed_out else None))
            row.update({"experiment": args.experiment, "dataset": args.dataset,
                        "balanced": args.balance, "batch_count": adata.obs[batch_key].nunique()})
            rows.append(row)
            print(f"  [{i}] {row['method']:13s} bb={row['backbone']:8s} lam={row['d_coef']:<5} "
                  f"seed={row['seed']} | iLISI={row['ilisi']:.3f} cLISI={row['clisi']:.3f} "
                  f"ASWb={row['asw_batch']:.3f} ARI={row['ari']:.3f}", flush=True)
            # incremental save so a timeout still yields partial results
            pd.DataFrame(rows).to_csv(args.out, index=False)
        except Exception as e:
            # WHY: a dropped config leaves NO trace in the CSV, so an incomplete wave
            #      looks identical to a complete one. Record the failure as a row with
            #      NaN metrics so completeness audits can see it.
            print(f"  [{i}] cfg={cfg} FAILED {type(e).__name__}: {e}", flush=True)
            rows.append({
                "method": "critic" if cfg.get("critic") else "discriminator",
                "backbone": cfg.get("backbone", "NB"),
                "d_coef": cfg.get("d_coef"),
                "seed": cfg.get("seed"),
                "experiment": args.experiment,
                "dataset": args.dataset,
                "failed": f"{type(e).__name__}: {e}",
            })
            pd.DataFrame(rows).to_csv(args.out, index=False)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    pd.DataFrame(rows).to_csv(args.out, index=False)
    print(f"\nWrote {len(rows)} rows -> {args.out}")


if __name__ == "__main__":
    main()
