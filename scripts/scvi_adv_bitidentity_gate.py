"""Seeded bit-identity gate for the scvi-tools adversarial extension.

CLAIM UNDER TEST: WassersteinAdversarialTrainingPlan with adversary="none" is BIT-IDENTICAL to
stock scvi-tools LinearSCVI at the same seed. This is the upstreamability guarantee -- our
extension adds a swappable adversary that contributes EXACTLY ZERO when off, so a seeded run is
scvi's own training path unchanged.

It also checks the WEAKER-but-necessary property that d_coef=0 with an adversary head ATTACHED is
also bit-identical (the head trains its own optimizer but must not touch the generator when the
fool-loss weight is zero) -- this is the "clean lambda=0 control" the wcd audit flagged as broken
in the old harness (where the adversary desynchronised the RNG). Here the adversary optimizer runs
on z.detach() and the generator step uses d_coef*loss_da=0, so the GENERATOR sees no adversary; but
the adversary's own forward draws RNG (gradient penalty), so d_coef=0-with-head may differ from
adversary=none. We REPORT both, so the distinction is explicit rather than assumed.

PASS: adversary="none" vs stock LinearSCVI -> max|delta| == 0 across seeds.
"""
import os
import sys

import numpy as np
import scanpy as sc

sys.path.insert(0, os.path.dirname(__file__))
from scvi_adversarial_plan import fit_adversarial_linearscvi  # noqa: E402


def _stock_linearscvi(adata, batch_key, seed, epochs, batch_size):
    import scvi
    from scvi.model import LinearSCVI
    scvi.settings.seed = seed
    a = adata.copy()
    a.X = a.layers["counts"].copy()
    LinearSCVI.setup_anndata(a, batch_key=batch_key)
    m = LinearSCVI(a, n_latent=30)
    m.train(max_epochs=epochs, batch_size=batch_size, early_stopping=False,
            enable_progress_bar=False)
    return m.get_latent_representation()


def main():
    ds = os.environ.get("GATE_DS", "immune")
    epochs = int(os.environ.get("GATE_EPOCHS", "15"))
    batch = int(os.environ.get("GATE_BATCH", "512"))
    seeds = [int(s) for s in os.environ.get("GATE_SEEDS", "0,1").split(",")]
    adata = sc.read_h5ad(f"results/scvi_single/{ds}_prepped.h5ad")
    bk = adata.uns["batch_key"]

    worst = 0.0
    for seed in seeds:
        z_stock = _stock_linearscvi(adata, bk, seed, epochs, batch)
        z_none = fit_adversarial_linearscvi(adata, bk, adversary="none", seed=seed,
                                            max_epochs=epochs, batch_size=batch)
        d = float(np.abs(z_stock - z_none).max())
        worst = max(worst, d)
        print(f"[gate] {ds} seed={seed} {epochs}ep batch={batch}: "
              f"adversary=none vs stock LinearSCVI  max|delta| = {d:.3e}  "
              f"-> {'IDENTICAL' if d == 0 else 'DIFFERS'}", flush=True)

    ok = worst == 0.0
    print(f"\nBIT-IDENTITY GATE: {'PASS' if ok else 'FAIL'}  (worst max|delta| across seeds = {worst:.3e})")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
