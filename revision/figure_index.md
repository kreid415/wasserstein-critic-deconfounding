# Revision figure index — BIOINF-2026-1434

All figures produced for the revision, the experiment they come from, the reviewer
comment(s) they address, and the underlying data table. Artifact IDs are the latest
version at time of writing.

## Headline / summary

| Figure | What it shows | Data |
|---|---|---|
| `fig_headline_summary.png` | 4-panel synthesis: (A) λ_adv Pareto front, (B) backbone-invariant ΔARI, (C) biological over-correction, (D) multibatch scaling — the discriminator is preferable on cell-type conservation (ARI) throughout and dominates outright at the operating point; the critic's only edge is more aggressive batch mixing at low λ_adv / low batch counts, which does not persist and comes at a conservation cost | E1/E2/E5/E8 |

## Per-experiment figures

| Figure | Experiment | Addresses | Data table |
|---|---|---|---|
| `fig_E1_pareto_immune.png` | E1 λ_adv Pareto front (immune, both heads, 3 seeds) | R2.2, R3.3 | `E1_immune_pareto.csv` |
| `fig_E2_backbones.png` | E2 critic-vs-discriminator on 4 VAE backbones (scCRAFT/scVI-NB/Gaussian/ZINB) | R2.1, R3.1 | `E2_backbone_generalization.csv` |
| `fig_E4_refdesign.png` | E4 reference-design isolation (fixed/rotating/joint vs discriminator) | R2.5, R1.minor.1 | `E4_refdesign_combined.csv` |
| `fig_E5_biology.png` | E5 direct biological readouts (purity, rare-cell retention, cell-type collapse) | R2.6, R1.minor.2 | `E5_summary_all.csv`, `E5_purity_*.csv` |
| `fig_E6_local_global.png` | E6 local-vs-global metric decomposition | R1.major.1 | `E6_local_global_decomposition.csv` |
| `fig_support_overlap.png` | Batch support-overlap characterization across 8 datasets | R1.major.2 | `support_overlap.csv` |
| `fig_E8_multibatch.png` | E8 multibatch scaling with full metric suite | R1.minor.1, R3.4 | `E8_multibatch_summary.csv` |

## Notes
- All embeddings scored with the identical local+global metric suite (iLISI, cLISI,
  graph-connectivity, kBET; ASW-batch, ASW-celltype, ARI, NMI, PCR, isolated-label ASW).
- The critic-vs-discriminator switch is the single flag `Discriminator(critic=True/False)`;
  λ_adv is the generator loss coefficient `d_coef`.
- Cross-species E1 Pareto panel and z_dim capacity / ATAC-large trade-off numbers are
  appended as those runs complete (see reviewer_response.md R3.2 / R3.minor.1).
