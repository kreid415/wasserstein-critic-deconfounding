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
than an aside. **[E6 local-vs-global decomposition — PENDING]** We decompose every
metric into its local (neighborhood: iLISI, cLISI, kBET, graph connectivity) and global
(geometry: ASW-batch, ASW-celltype, PCR, isolated-label ASW) components and show the
critic and discriminator sit at systematically different points on the local↔global
axis. *[Fill: the critic maximizes local mixing (iLISI ↑) while the discriminator better
preserves global batch geometry (ASW-batch ↑); quantify the crossover.]*

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

**Response.** Two parts. **(1) Full metric suite [E8 — PENDING]:** we re-run the
batch-count scaling (pancreas 2→9, immune 2→4) reporting the complete local+global suite,
not iLISI alone. *[Fill: whether the degradation reproduces across metrics or is
iLISI-specific; qualify the claim accordingly.]* **(2) The reviewer is right that
"independent (V−1) alignments" is not self-evidently harder** — so we test it directly in
**E4** by comparing the fixed-reference critic against a *joint* variant that draws the
reference at random each epoch (approximating an all-pairs objective). *[Fill: whether
joint alignment recovers the lost performance; if so, the bottleneck is the reference
design, and we drop the "independent alignment is intrinsically harder" language.]*

### R1.minor.2 — Downstream biology (annotation, marker preservation) would be easy to add

**Response — ADDRESSED (E5).** Added, see Theme C / R2.6 below.

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
(scCRAFT/scVI-NB/Gaussian/ZINB), testing whether the trade-off is backbone-specific.
*[Fill E2: reproduces across N/4 backbones ⇒ "generalizes across the backbones tested"
or report the boundary.]*

### R2.2 — Over-correction may be a λ_adv tuning issue; provide a Pareto front

> *"…the natural solution is to lower the adversarial weight λ_adv… The authors should
> provide a Pareto-front analysis across λ_adv, rather than relying on one selected
> operating point."*

**Response — CORE NEW RESULT (E1).** We sweep λ_adv ∈ {0, 0.01, 0.02, 0.05, 0.1, 0.2,
0.35, 0.5, 0.75, 1.0} for both heads (3 seeds) and plot the integration↔conservation
Pareto front. **[PENDING]** *[Fill: whether the critic front dominates, is dominated by,
or crosses the discriminator front. If the critic's front is uniformly shifted toward
mixing-over-conservation at matched λ_adv, the trade-off is not merely a tuning choice;
if the fronts coincide, we retract the "intrinsically worse" framing.]*

### R2.3 — Feels like an ablation; mechanistic language (topology, reference bottleneck) outruns evidence

**Response.** Accepted. Mechanistic claims are now **hypotheses tied to specific tests**:
E4 (reference design) and E6 (local/global + which cell types collapse via E5). Language
softened throughout per E0.

### R2.4 — Benchmark weak for the strength of the claims: add baselines+datasets OR frame as narrow ablation

**Response.** We do **both**. More datasets (8 total incl. cross-species and 2 ATAC),
external baselines **[E3 — PENDING]**, and explicit ablation framing [E0].

### R2.5 — Weakness may come from the reference-based design, not Wasserstein itself

> *"The observed reference sensitivity and scaling issues may be caused by forcing all
> batches to align to a single reference batch."*

**Response — DIRECTLY TESTED (E4).** We separate the *objective* from the *reference
design* by comparing: fixed single reference, rotating reference (cycled per epoch), and
joint (random reference per epoch ≈ all-pairs), plus a per-batch reference-coverage
analysis. **[PENDING]** *[Fill: if rotating/joint removes the sensitivity, we explicitly
attribute the failure to the reference formulation, not Wasserstein critics in general —
exactly as the reviewer suggests.]*

### R2.6 — Biological over-correction not directly validated

> *"…include more direct analyses, such as marker preservation, cell-type confusion,
> rare-cell retention, or per-cell-type neighborhood purity."*

**Response — ADDRESSED (E5).** We add every readout the reviewer names. **[PENDING]**
For both heads at the operating point we report per-cell-type neighborhood purity,
rare-cell retention (smallest-quartile types), the cell-type confusion matrix (which
pairs merge), and cross-batch label-transfer F1. Simulations (ground-truth Group labels)
give a clean test. *[Fill: which specific cell types lose purity/merge under the critic.]*

---

## Reviewer 3

### R3.1 — Justify generality; relate to scVI and SOTA; is the trade-off fundamental or architecture-specific?

**Response.** **[E3 + E2].** E3 places our adversarial heads alongside Harmony, scVI,
scANVI, and Scanorama on the identical metric suite; E2 tests architecture-specificity
across four backbones. The disjoint-support analysis (E6) states the condition under
which the Wasserstein objective is motivated. *[Fill: our critic vs scVI/scANVI on the
mixing↔conservation plane.]*

### R3.2 — Robustness to architecture/hyperparameters (depth, latent dim, loss weights)

**Response.** **[E2 capacity sweep — PENDING].** Beyond backbone identity we sweep
latent dimensionality (z_dim ∈ {10, 30, 50, 256}) and network depth (±1 layer) to test
whether the trade-off is intrinsic or a capacity artifact. *[Fill.]*

### R3.3 — Provide practical guidance / mitigation

**Response.** **[E0 + E1].** We add a "when to use which" paragraph derived from the
Pareto fronts (E1) and support-overlap (E6): prefer the critic when batches have little
support overlap and local mixing is the priority; prefer the discriminator (or a
low-λ_adv critic) when global/rare-cell conservation matters.

### R3.4 — Benchmark scale (atlas-scale, hundreds of thousands to millions)

**Response.** We extend to the largest scIB benchmark tasks available here — the
cross-species immune atlas (~98k cells) and the large ATAC set (~85k cells) — and report
multibatch scaling (E8). We scope the claims to the tens-to-hundreds-of-thousands range
tested and note atlas-scale (>1M) as future work rather than claiming it. *(Atlas runs
were de-scoped for this revision by mutual agreement.)*

### R3.minor.1 — Generalization beyond scRNA-seq (snRNA-seq, other modalities)

**Response — ADDRESSED.** We add scATAC-seq (gene-activity) as a second modality (two
datasets). The data prep uses a modality-aware branch (ATAC skips RNA-specific QC/HVG).
*[Fill: whether the trade-off holds on ATAC.]*

### R3.minor.2 — Downstream analyses (DE, trajectory, label transfer)

**Response — PARTIALLY ADDRESSED (E5).** Label transfer is included in E5; PAGA-based
trajectory preservation is already reported (PAGA-Spearman). *[Fill.]*

---

## Reproducibility note

- **Code provenance.** Authored contributions are now separated from the upstream
  scCRAFT backbone (He et al. 2025) into a dedicated `wcd/` package; a symbol-level
  AST diff against the upstream repository documents every new, modified, and unchanged
  function (`code_provenance.csv`). This directly supports the generalization claims by
  making the intervention (adversarial head swap) auditable.
- **Metric correctness.** During the revision we found and fixed a numerical bug: the
  LISI kernel was compiled with `@njit(fastmath=True)`, which invalidates the
  perplexity binary-search guards on recent numba builds and silently collapses iLISI/
  cLISI to a constant fallback. Removing `fastmath` fixes it; all reported LISI values
  use the corrected implementation. *[Note: verify whether the originally submitted
  numbers were affected.]*
