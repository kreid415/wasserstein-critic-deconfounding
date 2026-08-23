# Swappable adversary for scvi-tools (`WassersteinAdversarialTrainingPlan`)

A drop-in scvi-tools `TrainingPlan` that adds a **swappable batch-adversary** on the latent `z`
of any scvi module (LinearSCVI / SCVI), so the *same* scvi generative model can be trained with:

| `adversary` | objective on z | inner loop |
|---|---|---|
| `"none"` | — (stock scvi training) | — |
| `"discriminator"` | V-way cross-entropy domain classifier (JS adversary) | 1 |
| `"reference"` | reference Wasserstein critic (WGAN-GP, align to a reference batch) | `disc_iter` |
| `"barycenter"` | barycenter Wasserstein critic (align to a learnable centre) | `disc_iter` |

## Why this is a genuine scvi-tools extension (not a fork)

The generative model **is** scvi-tools' own module, unchanged. The adversary is bolted on via
scvi's native `AdversarialTrainingPlan` (two optimizers, manual optimization): the VAE step
minimises `loss_vae − λ·loss_da`, the adversary trains on `z.detach()`. Because the adversary
contributes **exactly zero** when `adversary="none"`, a seeded run reproduces stock training
**bit-for-bit**:

```
scvi_adv_bitidentity_gate.py  (immune, seeds 0 & 1, 15 epochs, batch 512):
  adversary="none" vs stock LinearSCVI  ->  max|delta| = 0.000e+00   (IDENTICAL)
```

This is the correctness guarantee a harness reimplementation cannot give — two codebases in
different environments (GPU vs CPU float, different training loops) can only match within seed
noise, but the *same* codebase with the adversary off matches to the bit.

## Usage

```python
import sys; sys.path.insert(0, "scripts")
from scvi_adversarial_plan import fit_adversarial_linearscvi

# conditioned decoder (scvi's setup_anndata(batch_key=...)); adversary reads BATCH_KEY
Z = fit_adversarial_linearscvi(adata, "batch", adversary="barycenter", d_coef=50.0,
                               disc_iter=10, n_latent=30, max_epochs=239, batch_size=512, seed=0)

# unconditioned decoder (decoder gets NO batch covariate; adversary is the SOLE integrator).
# batch is registered via labels_key so the adversary still sees it; pass conditioned=False.
Z = fit_adversarial_linearscvi(adata, "batch", adversary="discriminator", d_coef=50.0,
                               conditioned=False, ...)
```

Or drive the plan directly on a scvi model:

```python
from scvi.model import LinearSCVI
from scvi_adversarial_plan import WassersteinAdversarialTrainingPlan
LinearSCVI.setup_anndata(adata, batch_key="batch")
model = LinearSCVI(adata, n_latent=30)
model._training_plan_cls = WassersteinAdversarialTrainingPlan          # inject the plan
model.train(max_epochs=239, batch_size=512, early_stopping=False,
            plan_kwargs=dict(adversary="barycenter", d_coef=50.0, disc_iter=10))
Z = model.get_latent_representation()
```

## Files

- `scvi_adversarial_plan.py` — the plan + `fit_adversarial_linearscvi` helper.
- `scvi_adv_bitidentity_gate.py` — seeded bit-identity gate (the upstreamability proof).
- `scvi_adv_fit.py` / `run_scvi_adv_sweep.sh` — fit one config / drive the immune sweep.
- The authored adversary heads (`Discriminator`, `ReferenceWassersteinLoss`, gradient penalty)
  live in `src/wcd_vae/wcd/{critic,adversarial}.py`; the plan loads them by file path so it needs
  no heavy harness import (the wcd package `__init__` pulls in `scib`, absent from `scvi-env`).

## Environment

Runs in `scvi-env` (scvi-tools importable; CPU torch). Scoring of the resulting latents uses the
project's `full_metric_suite` in `wcd-kbet`. The two are separate on purpose — scvi-tools and the
harness metric stack have incompatible torch/torchmetrics pins.

## Immune result (single seed, `results/scvi_adv_full.csv`)

- **Unconditioned decoder (adversary is the sole integrator):** the adversary lifts scIB from
  ~0.47 (λ=0, no integration) to 0.54–0.57 (λ=50). Ranking: discriminator (0.566) > reference
  (0.543) ≈ barycenter (0.537). The adversary does large, real work here.
- **Conditioned decoder (scvi already corrects batch):** the adversary adds little (λ 0→50 moves
  scIB by −0.006 to +0.013); only the discriminator improves mixing. scvi's decoder-side batch
  channel starves the critic.
- **λ=0 controls agree across arms** within a backbone (RNG-artifact spread only) — expected,
  since at λ=0 every arm is stock scvi.
