#!/usr/bin/env python
raise SystemExit('RETIRED (code check CR-10, 2026-10-03; retired at the integration merge 2026-10-04): driver of the pre-revision sweeps; it references files already retired to legacy/. Current pipeline: legacy/README.md. git history keeps the runnable version.')
"""Build the UNIFIED-lambda EXTENSION manifest: the same lambda grid across ALL formulations.

MOTIVATION. The original final manifest used two family-specific grids (adversarial
{5,20,50,150}, critic-free {50,200,500,1500}) chosen so lambda*loss_da is a comparable
fraction of the VAE ELBO for each family's loss scale. That answers "where does each family
peak on its own scale" but NOT "how do the families compare at a MATCHED nominal lambda", and
it left the low-lambda regime (lambda<5) unexplored -- where the JS discriminator was observed
to keep improving on the heavy ATAC data (atac_large discriminator scIB DECLINES monotonically
from the lambda=5 floor: lin 0.467->0.283->0.247, nl 0.503->0.493->0.434).

THIS MANIFEST puts every formulation on one wide log grid spanning the whole spectrum:

    UNIFIED_GRID = {0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 1500}   (3.5 decades)

CAVEAT (stated, not hidden): because the formulations inject loss at different scales, the SAME
lambda regularises each arm differently, so each peaks at a different point on the shared grid.
That is expected and is the informative content of a matched-lambda comparison; the grid is wide
enough that every arm's peak falls INSIDE it.

REUSE (no recompute, no cherry-pick). The original manifest already covers, on this grid:
    adversarial arms (disc/ref/pooled/bary): {5, 20, 50}   (150 stays as a bonus off-grid point)
    critic-free arms (mmd/sinkhorn):         {50, 200, 500, 1500}
So this extension emits ONLY the missing (arm, lambda) points. The union of the original manifest
+ this extension gives every arm the full 12-point grid. Existing scored rows are reused unchanged;
the analysis merges both and reads lambda from the per-tag scored rows.

Tag convention is IDENTICAL to the original (<ds>_XZ_<dec>_uncond_<arm>_lam<lam>_s<seed>); the new
lambda values are disjoint from the originals, so no tag collides (asserted at build time).
"""
import os, sys, itertools

DATASETS = ["pancreas", "immune", "lung", "sim1", "sim2", "atac_small",
            "immune_hum_mou", "atac_large"]
DECODERS = ["lin", "nl"]
SEEDS    = [0, 1, 2, 3, 4]
ADV_FORMS = ["discriminator", "reference", "pooled", "barycenter"]
CF_FORMS  = ["mmd", "sinkhorn"]

UNIFIED_GRID = [0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 1500]
ORIG_ADV = {5, 20, 50, 150}          # already computed for adversarial arms
ORIG_CF  = {50, 200, 500, 1500}      # already computed for critic-free arms
NEW_ADV = [l for l in UNIFIED_GRID if l not in ORIG_ADV]   # per adversarial arm
NEW_CF  = [l for l in UNIFIED_GRID if l not in ORIG_CF]    # per critic-free arm

MODEL = {"lin": "LinearSCVI", "nl": "SCVI"}
ORIG_MANIFEST = os.path.join(os.path.dirname(__file__), "scvi_final_manifest.tsv")

def lam_str(lam):
    return f"{lam:g}"           # 0.5 -> "0.5", 1 -> "1", 1500 -> "1500"

def disc_iter(form):
    return 10 if form in ("reference", "pooled", "barycenter") else 1

def rows():
    for ds, dec, s, form, lam in itertools.product(DATASETS, DECODERS, SEEDS, ADV_FORMS, NEW_ADV):
        tag = f"{ds}_XZ_{dec}_uncond_{form}_lam{lam_str(lam)}_s{s}"
        yield dict(model=MODEL[dec], cond=0, adv=form, lam=lam_str(lam),
                   di=disc_iter(form), seed=s, dataset=ds, dec=dec, tag=tag)
    for ds, dec, s, form, lam in itertools.product(DATASETS, DECODERS, SEEDS, CF_FORMS, NEW_CF):
        tag = f"{ds}_XZ_{dec}_uncond_{form}_lam{lam_str(lam)}_s{s}"
        yield dict(model=MODEL[dec], cond=0, adv=form, lam=lam_str(lam),
                   di=1, seed=s, dataset=ds, dec=dec, tag=tag)

def main():
    out = sys.argv[1] if len(sys.argv) > 1 else "scripts/scvi_unified_manifest.tsv"
    R = list(rows())
    tags = [r["tag"] for r in R]
    assert len(tags) == len(set(tags)), "DUPLICATE TAGS within extension"
    # disjoint from the original manifest (no config re-run, no collision)
    if os.path.exists(ORIG_MANIFEST):
        orig = set()
        for line in open(ORIG_MANIFEST):
            if line.startswith("#") or line.startswith("model"):
                continue
            orig.add(line.rstrip("\n").split("\t")[-1])
        clash = set(tags) & orig
        assert not clash, f"{len(clash)} tags collide with original manifest, e.g. {sorted(clash)[:3]}"
    with open(out, "w") as fh:
        fh.write(f"# UNIFIED_GRID={UNIFIED_GRID} applied to ALL 6 formulations\n")
        fh.write(f"# EXTENSION over scvi_final_manifest.tsv: NEW_ADV={NEW_ADV} NEW_CF={NEW_CF}\n")
        fh.write(f"# {len(DATASETS)} datasets x {len(DECODERS)} decoders x {len(SEEDS)} seeds\n")
        fh.write("model\tcond\tadv\tlam\tdi\tseed\tdataset\tdec\ttag\n")
        for r in R:
            fh.write(f"{r['model']}\t{r['cond']}\t{r['adv']}\t{r['lam']}\t{r['di']}\t"
                     f"{r['seed']}\t{r['dataset']}\t{r['dec']}\t{r['tag']}\n")
    n_adv = sum(1 for r in R if r["adv"] in ADV_FORMS)
    n_cf  = sum(1 for r in R if r["adv"] in CF_FORMS)
    print(f"wrote {out}: {len(R)} NEW configs")
    print(f"  adversarial (disc/ref/pooled/bary) x NEW_ADV{NEW_ADV}: {n_adv}")
    print(f"  critic-free (mmd/sinkhorn) x NEW_CF{NEW_CF}: {n_cf}")
    print(f"  all {len(tags)} tags unique AND disjoint from original manifest: OK")

if __name__ == "__main__":
    main()
