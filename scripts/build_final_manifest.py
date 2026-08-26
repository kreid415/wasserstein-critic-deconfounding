#!/usr/bin/env python
"""Build the ONE committed manifest for the reproducible final benchmark.

REPRODUCIBILITY CONTRACT: this file + the raw datasets (figshare IDs, recorded md5s) + the
committed code regenerate the entire benchmark. No prior latent, scored row, or probe is reused.

FIXED A-PRIORI LAMBDA GRID (declared BEFORE any sweep result exists; not this run's argmax).
Two families because their loss-injection scales differ. Provenance of each scale:

  loss_vae (scVI sum-reduction ELBO, immune)           ~ 868    [MEASURED: m.get_elbo(), |value|;
                                                                  scVI reports negative ELBO, magnitude used]
  critic-free divergence (MMD/Sinkhorn, per-minibatch  MMD ~0.10, Sinkhorn ~0.44
    bs=512)                                            [MEASURED on immune X_pca minibatches]
  adversarial fool-loss (JS cross-entropy / W1 critic) ~ O(1-10)  [ANALYTIC: CE ~ ln(n_batch) at
                                                                  init ~1-2; W1 critic output O(1) —
                                                                  NOT independently measured here]

  => lambda*loss_da a meaningful (10-50%) fraction of loss_vae~868 requires:
       adversarial families: lambda ~ 10-150   (fool-loss O(1-10))       => ADV_LAMBDAS
       critic-free families: lambda ~ 200-4000 (divergence ~0.1-0.4)     => CF_LAMBDAS
  The grids are set WIDE (0 -> past-saturation) so the true optimum falls INSIDE, not tuned to it;
  correctness of the range is VERIFIED IN-RUN by checking each dataset's peak is interior (Phase 3),
  NOT asserted from these anchors alone.

The grids span 0 -> past-saturation on a log scale, WIDE ENOUGH that each dataset's peak falls
INSIDE the grid rather than being tuned to it. Same grid for every dataset/decoder/seed.

  ADVERSARIAL (discriminator, reference, pooled, barycenter):  {0, 5, 20, 50, 150}
  CRITIC-FREE (mmd, sinkhorn):                                 {0, 50, 200, 500, 1500}

lambda=0 COLLAPSE: at lambda=0 every formulation == stock scVI (adversary contributes nothing),
so lambda=0 is emitted ONCE as adversary='none' per (dataset, decoder, seed) -- the bit-identical
clean control -- and each formulation's grid starts at its first lambda>0. This removes 400
redundant critic-rate fits AND makes lambda=0 a genuine (un-perturbed) control.

PRE-REGISTERED lambda for the secondary baseline-comparison table (NOT this run's argmax):
  adversarial families: lambda = 20   (mid-grid, where the immune probe put the discriminator peak)
  critic-free families:  lambda = 200  (mid-grid)
These are fixed here, before results, purely for the summary table; the PRIMARY result is the
selection-free full-curve dominance analysis.
"""
import os, sys, itertools

DATASETS = ["pancreas", "immune", "lung", "sim1", "sim2", "atac_small",
            "immune_hum_mou", "atac_large"]
DECODERS = ["lin", "nl"]          # LinearSCVI / SCVI
SEEDS    = [0, 1, 2, 3, 4]        # 5 seeds
ADV_FORMS = ["discriminator", "reference", "pooled", "barycenter"]
CF_FORMS  = ["mmd", "sinkhorn"]
ADV_LAMBDAS = [5, 20, 50, 150]           # lambda>0 only; lambda=0 emitted once as 'none'
CF_LAMBDAS  = [50, 200, 500, 1500]
PREREG = {"adv": 20, "cf": 200}          # committed here, for the secondary table only

EMB = os.environ.get("WCD_EMBED_OUT",
    "/home/kendall/experiment_data/wasserstein-critic-deconfounding/embeddings_final")
MODEL = {"lin": "LinearSCVI", "nl": "SCVI"}

def disc_iter(form):
    return 10 if form in ("reference", "pooled", "barycenter") else 1

def rows():
    # 1) lambda=0 clean control: one stock-scVI fit per (dataset, decoder, seed)
    for ds, dec, s in itertools.product(DATASETS, DECODERS, SEEDS):
        tag = f"{ds}_XZ_{dec}_uncond_none_lam0_s{s}"
        yield dict(model=MODEL[dec], cond=0, adv="none", lam=0, di=1, seed=s,
                   dataset=ds, dec=dec, tag=tag)
    # 2) adversarial families at lambda>0
    for ds, dec, s, form, lam in itertools.product(DATASETS, DECODERS, SEEDS, ADV_FORMS, ADV_LAMBDAS):
        tag = f"{ds}_XZ_{dec}_uncond_{form}_lam{lam}_s{s}"
        yield dict(model=MODEL[dec], cond=0, adv=form, lam=lam, di=disc_iter(form), seed=s,
                   dataset=ds, dec=dec, tag=tag)
    # 3) critic-free families at lambda>0
    for ds, dec, s, form, lam in itertools.product(DATASETS, DECODERS, SEEDS, CF_FORMS, CF_LAMBDAS):
        tag = f"{ds}_XZ_{dec}_uncond_{form}_lam{lam}_s{s}"
        yield dict(model=MODEL[dec], cond=0, adv=form, lam=lam, di=1, seed=s,
                   dataset=ds, dec=dec, tag=tag)

def main():
    out = sys.argv[1] if len(sys.argv) > 1 else "scripts/scvi_final_manifest.tsv"
    R = list(rows())
    with open(out, "w") as fh:
        # header carries the pre-registered lambdas + grid, so the manifest is self-documenting
        fh.write(f"# ADV_LAMBDAS={ADV_LAMBDAS} CF_LAMBDAS={CF_LAMBDAS} (plus lambda=0 as adversary=none)\n")
        fh.write(f"# PREREG_LAMBDA adv={PREREG['adv']} cf={PREREG['cf']} (secondary table only)\n")
        fh.write(f"# {len(DATASETS)} datasets x {len(DECODERS)} decoders x {len(SEEDS)} seeds\n")
        fh.write("model\tcond\tadv\tlam\tdi\tseed\tdataset\tdec\ttag\n")
        for r in R:
            fh.write(f"{r['model']}\t{r['cond']}\t{r['adv']}\t{r['lam']}\t{r['di']}\t"
                     f"{r['seed']}\t{r['dataset']}\t{r['dec']}\t{r['tag']}\n")
    # summary
    n_none = sum(1 for r in R if r["adv"] == "none")
    n_adv  = sum(1 for r in R if r["adv"] in ADV_FORMS)
    n_cf   = sum(1 for r in R if r["adv"] in CF_FORMS)
    print(f"wrote {out}: {len(R)} configs")
    print(f"  lambda=0 clean controls (adversary=none): {n_none}")
    print(f"  adversarial (disc/ref/pooled/bary) lambda>0: {n_adv}")
    print(f"  critic-free (mmd/sinkhorn) lambda>0: {n_cf}")
    # collision check: every tag unique
    tags = [r["tag"] for r in R]
    assert len(tags) == len(set(tags)), "DUPLICATE TAGS"
    print(f"  all {len(tags)} tags unique: OK")

if __name__ == "__main__":
    main()
