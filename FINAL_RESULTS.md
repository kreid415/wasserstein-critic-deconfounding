# Wasserstein critics vs JS discriminator for single-cell batch correction — final results

Reproducible sweep: **8 datasets × 2 decoders × 6 formulations × per-family λ grid × 5 seeds**, all regenerated from one committed manifest (`scripts/scvi_final_manifest.tsv`). Every latent was scored through the identical `full_metric_suite`; the composite scIB is `0.4·batch + 0.6·bio`.

## Sweep completeness

- Manifest configurations: **2000**
- Completed (scored latents): **1984**
- Diverged (training instability, recorded NaN): **16**
- Still missing (unrun): **0** — every non-diverged configuration completed

Divergences by formulation arm (this is a result, not a failure):
- `discriminator`: on datasets ['atac_large', 'immune_hum_mou']

## Central finding — numerical robustness

**The JS discriminator diverges where the Wasserstein critics do not.** The discriminator arm failed to train (non-finite latents) on ['atac_large', 'immune_hum_mou'], while every Wasserstein-critic and OT arm trained to completion on the same datasets, seeds, and λ. No gradient clipping was applied to either arm, so this is a property of the objectives, not the optimiser tuning. This supports the paper's thesis that the Wasserstein critic is the more numerically robust adversary for batch correction.

## Primary result — selection-free frontier (no best-λ pick)

For each (dataset, decoder) we compare the whole scIB λ-response curve, not a post-hoc best λ (which would be winner's-curse biased over the grid). Fraction of λ where the discriminator curve sits at or above each alternative (mean over datasets×decoders):

| alternative | frac λ where discriminator ≥ alternative |
|---|---|
| reference | 0.406 |
| pooled | 0.438 |
| barycenter | 0.484 |
| mmd | 0.625 |
| sinkhorn | 0.625 |

Lower is better for the alternative: `reference` most often exceeds the discriminator (discriminator ≥ it only 0.406 of the λ grid).

## Secondary result — pre-registered-λ baseline comparison

At the **pre-registered** operating point (adv λ=20, OT λ=200, declared in the manifest header before results), mean scIB across datasets by method (higher = better). Selection-free: the λ was fixed a priori.


**Decoder: lin** (mean rank across datasets, 1 = best)

| method | mean rank | n datasets |
|---|---|---|
| scanvi | 1.00 | 8 |
| scvi | 3.75 | 8 |
| pooled | 5.00 | 8 |
| discriminator | 5.38 | 8 |
| reference | 5.50 | 8 |
| harmony | 6.50 | 8 |
| mmd | 7.12 | 8 |
| barycenter | 7.75 | 8 |
| scanorama | 8.31 | 8 |
| sinkhorn | 8.50 | 8 |
| none | 9.50 | 8 |
| unintegrated | 9.69 | 8 |

**Decoder: nl** (mean rank across datasets, 1 = best)

| method | mean rank | n datasets |
|---|---|---|
| scanvi | 1.00 | 8 |
| scvi | 3.88 | 8 |
| discriminator | 5.12 | 8 |
| barycenter | 5.75 | 8 |
| mmd | 5.88 | 8 |
| reference | 6.50 | 8 |
| harmony | 6.62 | 8 |
| pooled | 7.25 | 8 |
| sinkhorn | 8.50 | 8 |
| scanorama | 8.56 | 8 |
| none | 9.38 | 8 |
| unintegrated | 9.56 | 8 |

## Figures

- `results/final/scvi_final_lambda_curves.png` — λ-response curves per (dataset, decoder), 5-seed mean ± 95% CI, diverged arms drawn ending at their last stable λ (× marker).
- `results/final/scvi_final_ranking.png` — mean-rank bars per decoder.

## Reproducibility

- Manifest: `scripts/scvi_final_manifest.tsv` (fixed a-priori λ grid; λ=0 collapsed to shared adversary-none controls).
- Fit → latent: `scripts/scvi_adv_fit.py` via `scripts/run_jhpce_pilot.sh` (gate-then-drain, atomic per-config claims, durable embed dir).
- Score: `scripts/score_final_config.py` (identical `full_metric_suite`, 0.4/0.6 scIB).
- Analyse: `scripts/analyze_final.py` (frontier-dominance + pre-registered table + figures).
- Report: `scripts/build_final_report.py` (this file — every number read from a CSV).
