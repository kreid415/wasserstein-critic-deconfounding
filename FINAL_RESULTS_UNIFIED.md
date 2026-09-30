# Unified-λ sweep — final results

_Complementary exploratory extension to the pre-registered benchmark. Generated 2026-09-30 from `scored_all.csv` (6,305 scored configs) via `scripts/analyze_final.py` + `scripts/build_metric_curves.py`._

## What this run is

The pre-registered benchmark tested each formulation on a coarse, family-specific λ grid. This extension applies ONE **matched log grid** `{0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 1500}` to all six adversarial formulations (discriminator, reference/pooled/barycenter Wasserstein critics, MMD, Sinkhorn), across 8 datasets × 2 decoders (linear, non-linear) × 5 seeds. Only the (arm, λ) points missing from the original manifest were computed; existing scored rows were reused unchanged, so the two runs union into a fully-populated matched grid. **4,121 producible latents** finished; the 39 non-produced configs are all high-λ **discriminator divergences** (see Robustness).

The same nominal λ regularizes each arm differently (loss-scale differences), so each peaks at a different grid point — the grid is wide enough to bracket every peak, and comparisons are read off the whole curve, not a single operating point.

## Headline result — Wasserstein critics beat the JS discriminator across the grid

Selection-free frontier comparison (`scvi_final_frontier_dominance.csv`): per (dataset, decoder) we compare the discriminator's scIB λ-curve against the three Wasserstein critics across the whole grid.

- The **mean-of-3-critics beats the discriminator on 12 of 16 (dataset, decoder) cells**, mean advantage **+0.017 scIB**.
- The four exceptions are ties within ±0.008 (lung, sim2/lin, atac_small/nl).
- The critic advantage is **largest exactly where the discriminator is unstable**: atac_large/lin **+0.114** (the discriminator collapses at high λ), sim1/nl +0.045, sim2/nl +0.015.
- Pooled fraction of λ where the discriminator ≥ each critic: reference **0.27**, pooled **0.28**, barycenter **0.39** — i.e. the critics sit above the discriminator across most of the spectrum. The discriminator only edges Sinkhorn (0.63) and ties MMD (0.51).

## Robustness — every divergence is the discriminator

55 training divergences across the entire matched grid, **all in the JS discriminator arm** (atac_large 33, immune_hum_mou 22), **zero in any Wasserstein critic** on any dataset, decoder, λ, or seed. No gradient clipping was added, so this is a property of the objective, not the optimizer. The Wasserstein critics are numerically robust where the discriminator is not.

## Honest bottom line vs plain scVI

At the pre-registered λ, mean-rank across datasets (`scvi_final_ranks.csv`, 1 = best):

| rank | lin | nl |
|---|---|---|
| 1 | scANVI* (1.0) | scANVI* (1.0) |
| 2 | scVI (3.75) | scVI (3.9) |
| 3 | pooled-W (5.0) | discriminator (5.1) |
| 4 | discriminator (5.4) | barycenter-W (5.75) |

\* scANVI uses cell-type labels during training — not a fair unsupervised peer.

Neither the adversarial critics nor the discriminator beat plain **scVI** in aggregate scIB; scVI (the strong base) ranks second behind label-using scANVI. The adversarial contribution is a *within-family* one: **given that you add an adversary, a Wasserstein critic is both more effective and more stable than a JS discriminator.**

## Deliverables

- `scvi_final_lambda_curves.png` — scIB λ-response per dataset × decoder (primary, selection-free; divergences marked with ×)
- `metric_curves_unified_{lin,nl}.png` — per-metric decomposition (batch block + bio block), baselines as dashed horizontals
- `scvi_final_ranking.png` — mean-rank across datasets per formulation
- `scvi_final_full_curve.csv` — every (dataset, decoder, formulation, λ): 5-seed mean ± 95% CI for scIB/batch/bio
- `scvi_final_frontier_dominance.csv`, `scvi_final_baselines.csv`, `scvi_final_ranks.csv`
- `scored_all.csv` — 6,305 scored configs (original + unified), all metric columns

## Caveats

- Composite scIB is the mean of the 10 computed metrics; **kBET, cell-cycle, and PAGA-spearman were not computed** in this scoring pass (all-NaN).
- This is a complementary exploratory sweep, not a replacement for the pre-registered benchmark — the pre-registered λ table above is the confirmatory comparison; the frontier curves are the exploratory full picture.
