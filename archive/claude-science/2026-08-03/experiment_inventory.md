# Experiment inventory — status, implementation, and decisions that affect results

Source: `main` @ dfb5b48, definitions read from code (`configs_for()` in
`scripts/run_experiment.py`, `ARMS`/design loops in the dedicated harnesses), not from notes.

## Status summary

| exp | what it measures | implementation | status |
|---|---|---|---|
| **E1** | λ_adv Pareto frontier | `run_experiment.py --experiment E1` | **rerun** |
| **E2** | backbone × conditioning generality | `run_experiment.py --experiment E2` | valid; **rerun for PAGA** |
| **E3** | external baselines | `run_baselines.py` | **rerun** |
| **E4** | reference-design sweep | `run_reference.py` | **rerun (invalid)** |
| **E5** | biological readouts | `run_biology.py` | **valid**, unanalysed |
| **E6** | batch support overlap | `support_overlap.py` | **valid** |
| **E8** | multi-batch scaling | `run_experiment.py --experiment E8` | **rerun (invalid)** |
| **E9** | critic formulation comparison | `run_formulations.py` | **rerun (invalid)** |
| **CV+NB** | per-dataset significance | `hyperparameter_search.py` (needs patches) | **new** |

The single reason most reruns are needed: everything except E5/E6 was completed on the
**batch-conditioned** backbone, where the decoder receives the batch label directly, the
encoder has no incentive to encode batch, and the adversary sits at exactly chance
(`loss_da = ln(V)`). Every arm therefore collapsed to the same value — those results
compare untrained adversaries, not critic vs discriminator.

---

## E1 — λ_adv Pareto sweep

`for critic in (False, True): for lam in LAMBDA_GRID: for seed in (0,1,2)` =
**2 heads × 10 λ × 3 seeds = 60 configs/dataset**, one backbone.

`LAMBDA_GRID = [0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0]` — dense at the low
end, where the heads' early-mixing behaviour differs most. λ=0 is the no-adversary
conservation anchor.

**Decisions that affect results.** The critic gets `reference_batch=0`; the discriminator
gets `None` — i.e. the critic always aligns to *the first batch*, which is an arbitrary
choice E4 exists to test. λ is the only swept parameter; everything else is fixed.

**Why rerun:** valid (unconditioned) data covers only the 4-point subset {0, 0.05, 0.2,
0.5}; the full 10-point grid exists only for the conditioned backbone. The selection
criterion is still rising at 0.5 in 11/12 dataset × head arms, so the interior optimum —
if any — is unresolved. Filling in the remaining 6 grid points needs **no code change**.
Note `--d-coef-only` silently selects zero configs for a value not in the grid.

## E2 — backbone × conditioning generality

**8 backbones × 2 heads × 3 seeds = 48 configs/dataset** at fixed λ=0.2.

The 8 backbones are a **likelihood × conditioning grid**: Gaussian (MSE), Poisson, NB,
ZINB, LDVAE (linear decoder). **5 are unconditioned** (Gaussian, Poisson, NB_uncond,
ZINB_uncond, LDVAE_uncond) and **3 conditioned** (NB, ZINB, LDVAE).

**Decisions that affect results.** λ is pinned at 0.2 — the "operating point" — so E2
compares heads at one adversarial strength rather than at each head's best. Defensible as
a controlled comparison, but it is not a best-case-per-method comparison and must be
described as such.

**Status:** results are valid and complete (288 configs across 6 datasets, 0 NaN). Rerun
only to add `paga_spearman` and to persist embeddings. The headline claim needs restating
at the dataset level (see the statistics note below).

## E3 — external baselines

`harmony`, `scanorama`, `scVI`, `scANVI`, `combat`. No adversary, so cheap (~67 CPU-h).

**Decisions that affect results.** Harmony and scanorama are called through their own APIs
with explicit output-shape assertions rather than the scanpy wrappers, because the wrapper
`obsm` writes were version-fragile — an orientation bug here previously transposed the
embedding silently.

**Why rerun:** last run in the scCRAFT era, so not comparable to current numbers.

## E4 — reference-design sweep

`fixed_ref0 … fixed_ref{N-1}` + `rotating` + `joint` + `discriminator`, each × 3 seeds.
Design count is **dataset-specific**: it depends on that dataset's batch count, so a
2-batch dataset legitimately has fewer arms than a 9-batch one.

**Decisions that affect results.** `rotating` cycles the reference as
`(base + epoch) % n_batches`; `joint` draws one uniformly at random per epoch. This is the
experiment that tests whether the critic's pathologies come from the arbitrary
`reference_batch=0` choice rather than from the Wasserstein objective.

**Why rerun:** completed on the conditioned backbone — every arm gave iLISI ≈ 0.50 on
pancreas because the adversary never trained. Invalid.

## E5 — biological readouts

Five direct readouts on each embedding: marker preservation, per-cell-type neighbourhood
purity, rare-cell retention, cell-type confusion pairs, and cross-batch label transfer.

**Valid** — 24/24 complete on unconditioned backbones, and **not yet analysed**. This is
free signal sitting on disk. PAGA is not needed here; it has its own readouts.

## E6 — batch support overlap

Three model-free overlap measures in PCA space, with the cross-species task as the
expected-disjoint control and the simulations as known-overlapping ground truth.

**Valid and immune to the conditioning bug** — it trains no model. It answers R1.major.2
(do these datasets actually exhibit the disjoint-support setting the critic is motivated
by?).

## E8 — multi-batch scaling

2 heads × 3 seeds at λ=0.2, with the caller varying `--batch-count`. Excludes the two
2-batch datasets by construction.

**Why rerun:** conditioned backbone, adversary inert.

## E9 — critic formulation comparison

Four arms: `reference` (align to batch 0), `pooled` (align to the global pool — the
V-way counterpart of the discriminator), `barycenter` (align to learnable anchors
approximating a Fréchet mean), and `discriminator` as the reference-free control.

**Decisions that affect results.** The barycenter anchors are owned by the *generator*
optimizer and excluded from the critic optimizer — correct, since they must be pulled
toward the batches rather than pushed away adversarially. Verified in the code audit.

**Why rerun:** conditioned backbone; all four arms collapsed to ASW_batch ≈ 0.95.

---

## Global hyperparameters (fixed across every revision experiment)

| parameter | value | note |
|---|---|---|
| `z_dim` | 256 | capacity sweep found the trade-off invariant over {128, 256} |
| `epochs` | 150 | **under test** — convergence array `1999135` running |
| `warmup_epoch` | 5 | adversary does not step during warmup |
| `batch_size` | 1024 | |
| `disc_iter` | **10 critic / 1 discriminator** | standard WGAN vs GAN convention; already implemented |
| optimizer | Adam lr=1e-3, β=(0.5, 0.9) | same for generator and adversary |
| `kl_coef` | 0.005 | |
| λ_adv | 0.2 (operating point) | swept only in E1 |
| HVG | 2000 top genes | in `prep_data` |

**No hyperparameter search is performed in any E1–E9 run.** The paper's original numbers
came from a different path (`hyperparameter_search.py` → nested CV) which *does* search λ
over (0.01, 0.05, 0.1, 0.2, 0.5) with 5 outer × 3 inner folds, selecting by
`composite = iLISI − 10·cLISI`, at `epochs=500`, `inner_epochs=100`, `warmup_epoch=10`,
`disc_iter=10`. So the paper reports *tuned-per-arm* results and the revision reports
*fixed-config* results. This inconsistency is reviewer-visible and must be stated.

Two problems with that original criterion: over the sweep iLISI ranges 0.021 while cLISI
ranges 0.036, so the ×10 weight makes the cLISI term dominate by **17.7×** — it is
effectively a pure cLISI criterion, not a trade-off. A fairer replacement is
within-dataset z-scoring of each metric, oriented higher-is-better, averaging an
integration group (iLISI, −batch-probe-lift) and a conservation group (−cLISI,
+label-probe-lift) at equal weight.

## Statistics — the constraint on every across-dataset claim

The comparison unit is the **dataset** (n=6), not the dataset × backbone pair (n=30).
Treating backbones as independent replicates is pseudo-replication: for the E2 batch-lift
effect it moves p from 0.0008 to **0.0705** with an unchanged effect size of −0.041.

Nadeau-Bengio corrects for *overlapping training sets across CV folds of one dataset*. It
does not apply to the across-dataset comparison, where the six datasets share no data.
CV therefore buys **per-dataset** significance in the paper's Table 1 format; it cannot
raise the across-dataset n. Nesting a λ search inside CV costs 11× (2,563 vs 233 CPU-h)
and is not worth it — keep CV at fixed λ and answer tuning separately via E1.

## Cost of the full rerun

| component | CPU-h | days @20 |
|---|---|---|
| E1 fill in 6 remaining λ points | 839 | 1.75 |
| E2 PAGA + embeddings | 1,118 | 2.33 |
| E4 reference designs | 1,837 | 3.83 |
| E8 multi-batch | 373 | 0.78 |
| E9 formulations | 339 | 0.71 |
| E3 baselines | 67 | 0.14 |
| CV+NB (fixed λ) | 233 | 0.49 |
| **total** | **4,806** | **10.0** |

E4 alone is 38% of the total. Costs assume the 6 light datasets and the measured runtime
model `t(h) = 1.05e-07 · n_cells^1.69`; adding the two heavy datasets roughly quadruples
any component containing them.

## Suggested order

1. **Analyse E5** — already valid, costs nothing, and it is a headline conservation claim.
2. **CV+NB** (233 CPU-h) — fixes the statistics problem that currently blocks every claim.
3. **E1 grid fill** (839 CPU-h) — resolves whether an interior λ optimum exists and
   completes the Pareto frontier.
4. **E9 + E8** (712 CPU-h) — mechanism experiments, moderate cost.
5. **E2 PAGA rerun** (1,118 CPU-h) — only if PAGA is wanted in the backbone table.
6. **E4** (1,837 CPU-h) — most expensive; decide whether the reference-design question
   still matters given that E9's `pooled` arm already tests reference-freeness.
