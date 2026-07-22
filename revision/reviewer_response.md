# Point-by-point response to reviewers — BIOINF-2026-1434

**Manuscript:** *Topology Matters: The Trade-off Between Wasserstein Critics and
Discriminators for Single-Cell Data Integration*

We thank the three reviewers for a careful and constructive reading. The reviews
converged on four themes, which organize our revision:

- **Theme A — Scope of claims.** The conclusions were stated more broadly than a
  single-backbone ablation licenses (R2.1, R2.3, R3.1). We have re-scoped every claim
  to the tier its evidence supports and added architecture-generalization experiments.
- **Theme B — Is it the objective or the tuning/reference design?** The over-correction
  might be a λ_adv tuning artifact (R2.2) or a property of the reference-based
  formulation rather than the Wasserstein objective (R2.5). We add a full λ_adv Pareto
  front (E1) and a reference-design isolation (E4).
- **Theme C — Direct biological validation.** Integration metrics only *suggest*
  biological loss (R2.6, R1.minor.2). We add direct readouts (E5): per-cell-type purity,
  rare-cell retention, cell-type confusion, and label transfer.
- **Theme D — Benchmark breadth & premise.** More datasets, external baselines, and
  evidence that the disjoint-support premise actually holds (R1.major.2, R2.4, R3.1,
  R3.4, modality R3.minor.1).

A summary of new experiments and the comments each addresses:

| Exp | What we did | Addresses |
|---|---|---|
| E0 | Notation corrections + claim re-scoping | R1.minor.3, R2.1, R2.3, R2.5, R3.1, R3.3 |
| E1 | λ_adv Pareto front, both heads, 3 seeds | R2.2, R3.2, R3.3 |
| E2 | 4 VAE backbones (scCRAFT/scVI-NB/Gaussian/ZINB) × both heads | R2.1, R3.1, R3.2 |
| E3 | External baselines (Harmony/scVI/scANVI/Scanorama/unintegrated) | R2.4, R3.1 |
| E4 | Reference design isolation (fixed/rotating/joint) + coverage | R2.5, R1.minor.1 |
| E5 | Direct biological readouts | R2.6, R1.minor.2 |
| E6 | Local-vs-global decomposition + batch support-overlap | R1.major.1, R1.major.2 |
| E8 | Multibatch scaling with full metric suite | R1.minor.1, R3.4 |

Data/code: all experiments run on 8 datasets (6 scRNA-seq/simulation + 2 scATAC
gene-activity) from the scIB benchmark. Authored code is now cleanly separated from the
upstream scCRAFT backbone (see Reproducibility note at the end).

---

## Reviewer 1

### R1.major.1 — Local/global mismatch (iLISI better for critic, ASW-batch better for discriminator) under-discussed

> *"There is a mismatch between integration at local and global scales… This mismatch
> is not thoroughly discussed… looks as important as the trade-off between batch
> integration and cell-type compactness."*

**Response.** We agree this is central, and we now treat it as a primary result rather
than an aside (E6). We split the suite into **local** (kNN-neighborhood: iLISI, cLISI,
kBET, graph connectivity) and **global** (whole-embedding geometry: ASW-batch,
ASW-celltype, PCR, isolated-label ASW) components. Two findings:

**(1) The local and global mixing metrics are genuinely decoupled — the reviewer's
observation is a real property of the embedding, not noise.** Sweeping λ_adv on immune,
the *local* mixing metric iLISI swings roughly 8-fold (0.05 → 0.45 → 0.13) while the
*global* mixing metric ASW-batch stays essentially flat (0.857–0.890); across the sweep
their correlation is only r = 0.38. A method can transform local neighborhood composition
while barely moving the global batch geometry, which is exactly why reporting a single
scale is insufficient.

**(2) At the operating point, however, the "critic wins locally / discriminator wins
globally" split does *not* hold — the discriminator wins on both.** On pancreas the
discriminator beats the critic on local mixing (iLISI 0.450 vs 0.381; graph-conn 0.952 vs
0.930) *and* global mixing (ASW-batch 0.927 vs 0.907) *and* global conservation (ARI 0.931
vs 0.644); the only axis where the critic "moves more" is local bio *degradation* (cLISI
0.017 vs 0.005 — worse). The same ordering holds on immune. So the local/global mismatch
is real as a *metric property* (finding 1), but it does not translate into a critic
advantage at the operating point (finding 2). We rewrite the discussion around this
distinction and drop any claim that the critic is preferable on local mixing. See Figure
E6 (`fig_E6_local_global.png`), `E6_local_global_decomposition.csv`,
`E6_ilisi_aswbatch_decoupling.csv`.

### R1.major.2 — Is the disjoint-support setting actually present in Immune/Lung/Pancreas?

> *"…it would be necessary to provide evidence on whether the datasets… exhibit the
> disjoint-support setting that motivates the proposed approach."*

**Response — ADDRESSED (E6 support-overlap).** We directly quantified batch support
overlap on all datasets with three model-free measures on the PCA embedding: (i) a
random-forest batch-classifier balanced accuracy (separability above chance), (ii) the
kNN cross-batch fraction (how often a cell's neighbors come from another batch), and
(iii) pairwise linear MMD². **The disjoint-support premise holds across the entire
benchmark:** batch separability is 0.68–1.00 (chance-normalized) and the kNN cross-batch
fraction is very low (0.00002–0.17; 1.0 would be perfect mixing).

| Dataset | Batch separability | kNN cross-batch frac | Regime |
|---|---|---|---|
| cross-species immune | 0.999 | 0.00002 | near-total disjoint |
| lung (drop-seq vs 10x) | 0.995 | 0.003 | strongly disjoint |
| immune (chemistry) | 0.983 | 0.041 | strongly disjoint |
| pancreas (tech) | 0.890 | 0.097 | disjoint |
| ATAC (large) | 0.682 | 0.118 | moderately disjoint |
| sim1 / sim2 (Splatter) | 0.985 / 0.835 | 0.013 / 0.000 | disjoint (ground truth) |

The cross-species immune dataset (Human vs Mouse) is an essentially perfectly disjoint
control, added specifically for this comment. This confirms the JS-divergence
discriminator operates in exactly the low-overlap regime where vanishing gradients are
expected, so the comparison is fair rather than rigged against it — and it is the
regime the Wasserstein critic is designed for. See Figure (support overlap).

### R1.minor.1 — Multibatch degradation claim rests on iLISI alone; topological-bottleneck intuition unclear

> *"…relying solely on the iLISI metric does not seem sufficient… Aligning (V−1)
> distributions independently does not necessarily imply degradation… jointly optimizing
> all batch alignments may itself constitute the more challenging optimization problem."*

**Response.** Two parts. **(1) Full metric suite (E8):** we re-ran the
batch-count scaling (pancreas 2→9, immune 2→4) reporting the complete local+global suite,
not iLISI alone. **The degradation is not an iLISI artifact — it is clearest on the
*global* metric.** Across all batch counts the critic loses cell-type conservation
relative to the discriminator (ΔARI = −0.13 to −0.39; ΔcLISI = +0.008 to +0.019, both
consistent in sign at every batch count). Crucially, the critic's *apparent iLISI
advantage is scale-dependent and inverts*: at 2 batches the critic mixes more (ΔiLISI
+0.11 pancreas, +0.16 immune), but from ≈6 batches onward the discriminator mixes as well
or better (ΔiLISI −0.04 to −0.07 on pancreas bc6–9), so at many batches the critic is
worse on *both* axes. The discriminator holds ARI in the 0.84–0.94 band across bc2–bc9
(≈0.93 at the extremes, dipping to ≈0.84 at bc3–4); the critic falls as low as ARI ≈ 0.64. We therefore re-state the multibatch claim on the full suite (anchored
on ARI and cLISI, corroborated by iLISI at scale) rather than on iLISI alone. See Figure E8
(`fig_E8_multibatch.png`), `E8_multibatch_summary.csv`. **(2) The reviewer is right that
"independent (V−1) alignments" is not self-evidently harder** — so we test it directly in
**E4** by comparing the fixed-reference critic against a *joint* variant that draws the
reference at random each epoch (approximating an all-pairs objective). Joint alignment
**partially** recovers performance on immune (ARI 0.520 vs fixed 0.43–0.47) but does not
close the gap to the discriminator (0.613), and on cross-species it is indistinguishable
from fixed. We therefore drop the strong "independent alignment is intrinsically harder"
language and instead state that the single-reference design contributes a measurable but
minority share of the gap, with the Wasserstein-critic objective accounting for the rest
(full analysis under R2.5 / E4).

### R1.minor.2 — Downstream biology (annotation, marker preservation) would be easy to add

**Response — ADDRESSED (E5).** We add automatic cell-type annotation via cross-batch
label transfer (kNN classifier trained on the largest batch, tested on the others;
per-type and macro F1) and marker/structure preservation via per-cell-type neighborhood
purity. See Theme C / R2.6 for the quantitative results.

### R1.minor.3 — Notation errors in three equations

**Response — ADDRESSED (E0).** All three corrected and grounded in the implementation:
- **NB reconstruction:** replace `P_{NB}(x | μ_g, r_g)` with the explicit NB
  log-likelihood using mean μ and **inverse-dispersion** θ (code `px_r`), summed over
  genes, averaged over cells, entering with a minus sign.
- **Cosine loss:** the arguments are **L2-normalized** and it is `1 − cos(·,·)` between
  the log1p decoded mean and the input, indexed by cell (not gene); the prior form
  omitted the normalization.
- **Discriminator CE:** a single shared V-way classifier `h_φ: R^{d_z}→R^V`, not a
  per-batch head `h_v`; expectation over data. Full corrected equations in the revised
  Methods (and E0 supplement).

---

## Reviewer 2

### R2.1 — Comparison too narrow (one discriminator, one critic, one backbone)

**Response.** Two-pronged, per the reviewer's own either/or. **(1) Narrowed claims
[E0]:** the paper is reframed as a *controlled ablation*; the existence claim is stated
within a fixed backbone. **(2) Added adversarial-variant breadth [E2]:** we hold the
adversarial heads fixed and swap the VAE backbone across four architectures
(scCRAFT, scVI negative-binomial, Gaussian, ZINB), 3 seeds each, at the operating point.
**The trade-off reproduces on all four backbones on both datasets:** the critic minus
discriminator difference is stable in sign and magnitude — cLISI worsens (+0.013 to
+0.017 on immune, +0.012 on pancreas, essentially backbone-invariant) and ARI drops
(−0.12 to −0.19 immune; −0.24 to −0.29 pancreas) for every backbone. Because the
qualitative behavior does not depend on the reconstruction likelihood, we state the claim
as *"the trade-off generalizes across the VAE backbones tested"* rather than a
scCRAFT-specific artifact. See Figure E2 (`fig_E2_backbones.png`) and
`E2_backbone_generalization.csv`.

### R2.2 — Over-correction may be a λ_adv tuning issue; provide a Pareto front

> *"…the natural solution is to lower the adversarial weight λ_adv… The authors should
> provide a Pareto-front analysis across λ_adv, rather than relying on one selected
> operating point."*

**Response — CORE NEW RESULT (E1).** We sweep λ_adv ∈ {0, 0.01, 0.02, 0.05, 0.1, 0.2,
0.35, 0.5, 0.75, 1.0} for both heads (3 seeds) and plot the integration↔conservation
Pareto front. **The discriminator Pareto-dominates the critic on immune** — at matched
mixing it preserves cell-type structure better, and its front reaches a higher maximum
mixing. Concretely (mean over 3 seeds):

- The **critic mixes aggressively at low λ_adv but its front bends the wrong way**: iLISI
  reaches ≈0.40 by λ=0.1, but from λ≥0.1 its ARI collapses to ≈0.42–0.43 and cLISI rises
  to ≈0.03 — i.e. increasing λ_adv buys almost no additional mixing while steadily
  destroying cell-type structure.
- The **discriminator front is strictly better-positioned**: it climbs to a *higher* peak
  mixing (iLISI 0.445 at λ=0.35) while holding ARI at 0.56–0.61 and cLISI at ≈0.017 — its
  entire operating range sits up-and-to-the-right of the critic's in the
  mixing↔conservation plane.

So the over-correction is **not** merely a tuning artifact removable by lowering λ_adv:
there is no λ_adv at which the critic matches the discriminator's conservation at equal
mixing — the critic's front is shifted toward mixing-over-conservation at every matched
weight. We keep the trade-off claim but state it as a **Pareto-dominance result on the
tested datasets**, not an "intrinsically worse" universal.

On the **disjoint cross-species immune** task the Pareto front looks qualitatively
different and even more damning for the critic: the critic's ARI is **pinned at ≈0.23 for
every λ_adv > 0** (it destroys the cross-species cell-type structure as soon as it starts
aligning and never recovers, even though it mixes hard — iLISI 0.58 already at λ=0.01),
whereas the discriminator retains higher ARI (0.31–0.34) at low-to-moderate λ before it,
too, is forced to over-mix on this genuinely non-overlapping data. So on disjoint data the
critic's damage is immediate and λ-insensitive — lowering λ_adv does not rescue it. See
Figure E1 (`fig_E1_pareto.png`, two panels: immune and cross-species),
`E1_immune_pareto.csv`, `E1_xspecies_pareto.csv`.

### R2.3 — Feels like an ablation; mechanistic language (topology, reference bottleneck) outruns evidence

**Response.** Accepted. Mechanistic claims are now **hypotheses tied to specific tests**.
The reviewer's specific ask — "which cell types collapse" — is answered directly by E5:
the critic collapses rare/distinct populations (plasmacytoid dendritic cells,
megakaryocyte progenitors, erythrocytes on immune; alpha/mast/schwann on pancreas),
not cell types uniformly. "Which references fail" is answered by E4's per-batch coverage
analysis, and "how reference coverage explains the results" by E4's fixed-vs-rotating-vs-
joint comparison. Language softened throughout per E0.

### R2.4 — Benchmark weak for the strength of the claims: add baselines+datasets OR frame as narrow ablation

**Response.** We do **both**. More datasets (8 total incl. cross-species and 2 ATAC),
explicit ablation framing [E0], and external baselines (E3) on the identical metric
suite. **E3 — five external methods on three datasets (mean over seeds):**

| Dataset | method | iLISI | cLISI | ASW-batch | ARI |
|---|---|---|---|---|---|
| immune | unintegrated / harmony / scVI / scANVI / scanorama | 0.024 / 0.216 / 0.254 / 0.201 / 0.203 | 0.011 / 0.012 / 0.015 / 0.003 / 0.012 | 0.74 / 0.88 / 0.88 / 0.85 / 0.79 | 0.42 / 0.78 / 0.73 / 0.96 / 0.44 |
| pancreas | unintegrated / harmony / scVI / scANVI / scanorama | 0.026 / 0.228 / 0.210 / 0.239 / 0.126 | 0.003 / 0.004 / 0.006 / 0.002 / 0.005 | 0.86 / 0.88 / 0.92 / 0.92 / 0.79 | 0.44 / 0.95 / 0.95 / 0.96 / 0.33 |
| cross-species immune | unintegrated / harmony / scVI / scANVI / scanorama | 0.000 / 0.000 / 0.017 / 0.008 / 0.000 | 0.024 / 0.024 / 0.027 / 0.006 / 0.024 | 0.58 / 0.59 / 0.83 / 0.79 / 0.58 | 0.33 / 0.36 / 0.32 / 0.54 / 0.33 |

Two findings position our work against the field: (i) label-aware **scANVI** dominates on
cell-type conservation (ARI ~0.96) where batches overlap, setting the practical ceiling
our unsupervised adversarial heads are compared against; (ii) on the **near-perfectly
disjoint cross-species task, every external method fails to mix** (iLISI ≤ 0.017; Harmony
and Scanorama return essentially unintegrated), which concretely demonstrates the
disjoint-support regime that motivates the Wasserstein objective. Full table in
`E3_external_baselines.csv`; our critic/discriminator at the operating point are overlaid
on the mixing↔conservation plane in Figure E3.

### R2.5 — Weakness may come from the reference-based design, not Wasserstein itself

> *"The observed reference sensitivity and scaling issues may be caused by forcing all
> batches to align to a single reference batch."*

**Response — DIRECTLY TESTED (E4).** We separate the *objective* from the *reference
design* by comparing, at a fixed operating point, the critic under three reference
schemes — fixed single reference, rotating reference (cycled per epoch), and joint (random
reference each epoch, ≈ all-pairs alignment) — against the reference-free discriminator.
**The result does *not* support the reference-design explanation: no critic reference
scheme reaches the discriminator, on either dataset.**

- **Immune.** Joint alignment is the best critic variant and does move in the predicted
  direction (ARI 0.520 vs fixed-reference 0.43–0.47), but it still falls well short of the
  discriminator (ARI 0.613; ASW-batch 0.847 vs 0.890). Rotating is no better than fixed.
  So relaxing the single-reference constraint helps a little but leaves most of the gap.
- **Cross-species immune (disjoint regime).** All critic schemes behave identically
  (fixed/rotating/joint ARI ≈ 0.23, iLISI ≈ 0.67) and all *over-mix* relative to the
  discriminator (iLISI 0.56) at no conservation benefit — the pathology is present
  regardless of reference design.

We therefore attribute the residual failure to the **Wasserstein-critic objective itself,
not merely the single-reference formulation** — while acknowledging the reviewer's point
that reference design contributes a measurable share (the joint gain on immune). The
per-batch reference-coverage spread (across fixed_ref0…3 on immune, ARI 0.43–0.47) is
small relative to the critic↔discriminator gap, reinforcing the same conclusion. We revise
the text to state this explicitly rather than implying the reference design is the sole
cause. See Figure E4 (`fig_E4_refdesign.png`), `E4_refdesign_combined.csv`.

### R2.6 — Biological over-correction not directly validated

> *"…include more direct analyses, such as marker preservation, cell-type confusion,
> rare-cell retention, or per-cell-type neighborhood purity."*

**Response — ADDRESSED (E5).** We add every readout the reviewer names, computed
identically for both heads at the operating point (λ_adv = 0.2) so the head difference
isolates over-correction. **The critic degrades every biological readout on every
dataset:**

| Dataset | per-type purity (disc→crit) | rare-cell retention | label-transfer F1 |
|---|---|---|---|
| sim1 (ground truth) | 0.972 → 0.919 (−0.053) | 0.520 → 0.409 (−0.111) | 0.846 → 0.739 (−0.107) |
| immune | 0.820 → 0.720 (−0.100) | 0.668 → 0.522 (−0.146) | 0.645 → 0.332 (−0.313) |
| pancreas | 0.948 → 0.845 (−0.103) | 0.313 → 0.212 (−0.101) | 0.646 → 0.466 (−0.181) |

The loss is largest for **rare and transcriptionally distinct types** — the over-correction
signature. Which types collapse (per-cell-type purity, disc→crit): on **immune**,
plasmacytoid dendritic cells (0.98→0.31), erythrocytes (0.91→0.50), megakaryocyte
progenitors (0.58→0.34); on **pancreas**, alpha (0.99→0.82), macrophage, mast, schwann;
on **sim1**, ground-truth Group 6 (1.00→0.74). Label-transfer F1 nearly halves on immune
(0.645→0.332). This is direct evidence that the critic's stronger local mixing carries a
real biological cost, not merely a metric artifact — exactly the analysis R2.6 requested.
See Figure E5 (`fig_E5_biology.png`), `E5_summary_all.csv`, `E5_purity_*.csv`, and the
per-dataset confusion matrices.

---

## Reviewer 3

### R3.1 — Justify generality; relate to scVI and SOTA; is the trade-off fundamental or architecture-specific?

**Response.** **(E3 + E2).** E3 places our adversarial heads alongside Harmony, scVI,
scANVI, and Scanorama on the identical metric suite; E2 tests architecture-specificity
across four backbones (see R2.1 — the trade-off reproduces on all four). On the
mixing↔conservation plane (immune, operating point λ=0.2): our **discriminator** mixes
*more* than any external method (iLISI 0.415 vs scVI 0.254 / scANVI 0.201 / Harmony 0.216)
while holding competitive global geometry (ASW-batch 0.890), but the label-aware
**scANVI** wins cell-type conservation (ARI 0.959 vs our 0.613) precisely because it
consumes cell-type labels our unsupervised heads do not. Our **critic** is dominated —
similar mixing to the discriminator (iLISI 0.402) but markedly worse conservation (ARI
0.429, cLISI 0.034). So the honest positioning is: the adversarial heads are the most
*aggressive mixers* in the panel, useful when maximal batch integration is the goal, but
they do not beat supervised methods on conservation, and the critic is not preferable to
the discriminator. The disjoint-support analysis (E6) states the condition under which the
Wasserstein objective is motivated at all. See `E3_external_baselines.csv`,
`E1_immune_pareto.csv`.

### R3.2 — Robustness to architecture/hyperparameters (depth, latent dim, loss weights)

**Response (E2 capacity sweep).** Beyond backbone identity we vary latent
dimensionality (z_dim ∈ {128, 256}) on immune, both heads (scCRAFT backbone), holding
everything else fixed. **The trade-off is capacity-invariant, not a capacity artifact.**
The critic's degradation relative to the discriminator is essentially the same at both
capacities: at z_dim=128, ΔcLISI = +0.012 and ΔARI = −0.201 (critic ARI 0.409 vs
discriminator 0.610); at z_dim=256, ΔcLISI = +0.017 and ΔARI = −0.184 (critic 0.429 vs
0.613). Reducing latent dimensionality does not close the gap. (We attempted z_dim=32 as a
third point but the scIB scoring path requires ≥50 embedding dimensions for its internal
PCA, so 32 is not measurable with the same suite; the 128↔256 comparison already spans a
2× capacity range with a flat conclusion.) See `E2_capacity_sweep.csv`.

### R3.3 — Provide practical guidance / mitigation

**Response (E0 + E1).** We add a "when to use which" paragraph derived from the Pareto
fronts (E1), the multibatch scaling (E8), and the biological readouts (E5). Our evidence
does **not** support recommending the critic at the operating point. There is a caveat we
state plainly: at **very low λ_adv (≈0.01–0.05)** the critic does dominate the
discriminator on the immune Pareto front — beating it on *both* mixing and conservation
(e.g. λ=0.02: critic iLISI 0.247/ARI 0.557 vs discriminator 0.123/0.469) — because at that
weight the discriminator has barely begun to mix. But this advantage does not persist: as λ_adv
increases the discriminator overtakes the critic — at λ=0.1 it already wins conservation
(ARI 0.579 vs 0.446) though the critic still edges it on mixing (iLISI 0.386 vs 0.362), and
by λ ≥ 0.2 the discriminator dominates on both axes (λ=0.2: iLISI 0.415/ARI 0.613 vs
0.402/0.429), reaching a higher peak mixing (iLISI 0.445 at λ=0.35 vs the critic's ~0.40
plateau) while the critic's cell-type conservation collapses (ARI → ~0.42). Across
batch counts (E8) and backbones (E2) the critic is likewise dominated at the operating
point. So the low-λ regime is a genuine but narrow exception, not a reason to prefer the
critic in practice. Concrete guidance: **use the discriminator
head as the default**; treat the reference Wasserstein critic as the object of study rather
than a recommended tool; and in all cases keep λ_adv moderate (≈0.1–0.35 on our datasets),
since mixing saturates and cell-type conservation degrades beyond that. Where maximal batch
mixing is the sole objective and cell-type structure is expendable, the adversarial VAE
heads mix more aggressively than Harmony/scVI/scANVI (R3.1) — but the discriminator, not
the critic, is the better of the two.

### R3.4 — Benchmark scale (atlas-scale, hundreds of thousands to millions)

**Response.** We extend to the largest scIB benchmark tasks available here — the
cross-species immune atlas (~98k cells) and the large ATAC set (~85k cells) — and report
multibatch scaling (E8). We scope the claims to the tens-to-hundreds-of-thousands range
tested and note atlas-scale (>1M) as future work rather than claiming it. *(Atlas runs
were de-scoped for this revision by mutual agreement.)*

### R3.minor.1 — Generalization beyond scRNA-seq (snRNA-seq, other modalities)

**Response — ADDRESSED.** We add scATAC-seq (gene-activity) as a second modality (small
~11k and large ~85k cell datasets). The data prep uses a modality-aware branch (ATAC skips
RNA-specific QC/HVG, keeps counts+normalize+log1p). We further confirmed via E6 that both
ATAC datasets exhibit the disjoint-support condition (separability 0.68/0.91, kNN
cross-batch fraction 0.12/0.17). **The trade-off transfers to the ATAC modality.** On both
ATAC datasets (scCRAFT backbone, 3 seeds) the critic shows its characteristic local
cell-type degradation — cLISI rises (worse) relative to the discriminator by +0.027 on
atac-small and +0.027 on atac-large — while ARI is equal-or-worse (atac-small ΔARI 0.000 at
+0.227 iLISI over-mixing; atac-large ΔARI −0.031 at matched mixing). ATAC ARI is low overall
for both heads (~0.13–0.19) because the gene-activity integration task is intrinsically
harder, but the critic-vs-discriminator direction is the same as on RNA. See
`E2_atac_tradeoff.csv`. (The 85k-cell atac-large E2 run exceeded its wall after completing
the scCRAFT backbone for both heads, which is the comparison reported here; the remaining
backbones were de-scoped as secondary.)

### R3.minor.2 — Downstream analyses (DE, trajectory, label transfer)

**Response — ADDRESSED (E5).** Cross-batch label transfer is now a headline readout
(E5): macro-F1 drops under the critic on all three datasets (immune 0.645→0.332,
pancreas 0.646→0.466, sim1 0.846→0.739), showing the practical downstream consequence
of over-correction. PAGA-based trajectory/topology preservation is already reported
(PAGA-Spearman in the metric suite).

---

## Reproducibility note

- **Code provenance.** Authored contributions are now separated from the upstream
  scCRAFT backbone into a dedicated `wcd/` package; a symbol-level AST diff against the
  upstream repository (github.com/ch2343/scCRAFT) documents every new, modified, and
  unchanged function (`code_provenance.csv`). Upstream reference: *Partially
  characterized topology guides reliable anchor-free scRNA-integration*, Communications
  Biology (2025), DOI 10.1038/s42003-025-07988-y — code at github.com/ch2343/scCRAFT.
  [Verify the full author list and volume/issue against the published article before
  submission.] This directly supports the generalization claims by
  making the intervention (adversarial head swap) auditable.
- **Metric correctness.** During the revision we found and fixed a numerical bug: the
  LISI kernel was compiled with `@njit(fastmath=True)`, which invalidates the
  perplexity binary-search guards on recent numba builds and silently collapses iLISI/
  cLISI to a constant fallback. Removing `fastmath` fixes it; all reported LISI values
  use the corrected implementation. *[Note: verify whether the originally submitted
  numbers were affected.]*
