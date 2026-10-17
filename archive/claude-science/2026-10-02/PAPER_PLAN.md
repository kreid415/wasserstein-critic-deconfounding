# Paper plan: rigorous revision of BIOINF-2026-1434

*Design document, 2026-10-02. Code state: commit `5709a98`. Every number marked provisional comes from the atac_small counts pilot (1 dataset, 3 seeds, pre-stock-default schedule) or from short single-seed checks, and is used here only to size and design the experiments.*

**Working title (proposal):** *Wasserstein critics versus discriminators for adversarial batch correction: a controlled, budget-matched comparison on scVI.* The original title "Topology Matters" should go: nothing topological was measured (R2.3, R1.minor.1).

**Sources:** original submission {{artifact:9d87cbf0-0080-45a8-a897-d4f02ec3b0dd}}; recovered decision letter with all 17 comments [BIOINF-2026-1434_decision_letter.txt]({{artifact:80d0fc2d-d3b1-4c9f-aa5e-f74dfac313cf}}); manuscript audit [original_manuscript_audit.md]({{artifact:0c9e00ea-b555-4cea-9a99-d04d4c2e56f8}}), [original_manuscript_claims_audit.csv]({{artifact:6c946526-684e-4e4b-9306-95c648230aae}}), [reviewer_comment_map.csv]({{artifact:eaabff29-bdcc-42c1-87da-73859e022bcc}}); methods/rigour review [rigour_review.md]({{artifact:281d44ff-8cbd-4adc-8169-0ac3051347d2}}); code review [code_review_5709a98_findings.csv]({{artifact:4eeb2df8-32dc-4935-98fe-1e2e33a6c964}}); earlier code audit [code_audit_findings.csv]({{artifact:5f4b1715-a22c-4ec1-b8b2-6bbf0417b094}}); counts pilot [counts_pilot_summary.csv]({{artifact:6f937ff2-fb9f-4cc6-b33a-939b86907831}}).

---

## 1. Where the original paper stands

- **25 substantive claims audited:** 11 invalidated by pipeline defects, 6 never tested, 5 partially supported, 3 contradicted by the paper's own data. No number in Table 2, Figs 2–5 or Table S1 can be reused, nor any E1–E8 result from the July response.
- **Submission-pipeline defects** (code at `cd64f05`; must be disclosed, since bioRxiv v1–v3 are public):
  - **D1** NB likelihood fit to log-normalised values.
  - **D2** LISI computed on a kNN graph mixing two unaligned PCA frames (train and test).
  - **D3** sampled z evaluated instead of the posterior mean.
  - **D4** gradient-penalty fallback in the binary and reference-sweep runs.
  - **D5** 10 critic updates vs 1 discriminator update.
  - **D6** unbounded discriminator generator loss.
  - **D7** λ selected on 40-epoch inner models, reported on 80-epoch outer models.
  - **D8** custom fastmath LISI.
  - **D9** single seed and overlapping folds as replicates; 36 uncorrected tests, of which 6 survive Holm.
- **Contradicted by the paper's own design:** the "balanced" scenario was not composition-independent (Cramér's V 0.33–0.42; pancreas balanced differs from unbalanced by 2 cells). Reference outliers occur equally in balanced data, where every reference has the same cell types, so coverage cannot be the "strict" driver.
- **What carries over:** the question, the narrative, the four Results sections (as pre-registered hypotheses with confirm/refute rules), Introduction paragraphs 1–3 after correction, the Related Work taxonomy, the figure designs, and the bibliography (51 entries; remove duplicates).

## 2. Framing

**Already known (verified by the methods review):**
- **W1 critics for domain invariance:** WDGRL (Shen 2018).
- **Reference-anchored WGAN-GP integration:** iMAP (Wang 2021). MMD to a reference batch: SAUCIE.
- **Controlled-backbone comparison of batch losses:** scIB-E (Yi 2025) compares GAN, HSIC, orthogonality and MI losses in scVI/scANVI, but has no W1 critic, MMD or OT arm.
- **Adversarial alignment mixes cell types under unbalanced composition:** sysVI (Hrovatin 2025).
- **Theory:** under label (composition) shift, alignment with any proper divergence forces errors, and tighter alignment forces more (Zhao 2019 Thm 4.3; Wu 2019).
- **In GANs the loss is second-order to tuning and regularisation** (Lucic 2018; Kurach 2019).

**Defensible gap:** no budget-matched JS vs W1 vs MMD/OT comparison on a stock scVI backbone trained on real counts (search-based absence, not exhaustive).

**Contribution, restated:** a controlled decomposition of the critic–discriminator trade-off. How much of it is:
- operating point;
- composition-driven over-alignment (predicted for every divergence);
- W1-specific geometry or latent scale;
- regularisation and update budget;
- target design (reference vs symmetric)?

Practical guidance for scVI users follows from the results, including the possibility that no adversary improves on stock scVI. That outcome was hinted in the pilot, provisionally: stock scVI 0.726 vs best conditioned arm 0.646.

**Drop or soften:**
- "JS fails on disjoint technology batches", unless X4 shows it.
- "First systematic comparison".
- "Critics over-correct" as a W1 property, unless shown at matched batch removal.
- "Weak integration is protective": that is the trade-off curve itself, unless shown at matched alignment.
- All "topology / bottleneck / radial projection" language.

## 3. Outline

**Abstract.** Motivation / Results (written to the pre-registered decision rules) / Availability (commit SHA, data sources with md5s).

**1 Introduction.**
1. Batch effects and adversarial integration (reuse, edited).
2. JS vs W1 stated correctly: WGAN Thm 1 (W1 continuous, differentiable a.e.); Arjovsky & Bottou 2017 (vanishing gradients for the saturating loss with a near-optimal discriminator; the non-saturating/bounded loss does not vanish but can be unstable; continuous noise breaks the disjoint-support assumption). The premise becomes a tested hypothesis.
3. Composition shift and over-alignment (Zhao, Wu, sysVI, Maan 2024).
4. Gap and contributions.

**2 Related work.** Reuse the taxonomy. Add WDGRL, iMAP, SAUCIE, scDREAMER, scIB-E, sysVI, trVAE, BERMUDA, uniPort, MMD-ResNet. Cite scCRAFT only as the earlier backbone.

**3 Methods.**
- **3.1 Backbone.** scvi-tools 1.4.2 SCVI (ZINB) at stock defaults: n_latent 10, batch 128, epoch heuristic, KL warm-up 400 epochs, no early stopping. Batch-conditioned (scVI as used) and unconditioned (adversary as sole integrator, the original paper's setting). λ=0 is bit-identical to stock SCVI (verified, max|Δz| = 0).
- **3.2 Adversaries** (implementation-faithful equations, R1.minor.3):
  - V-way discriminator with scvi-tools' bounded fool loss, at 1 and 10 adversary steps;
  - reference critic (per-head WGAN-GP, described as a star of pairwise W1 terms);
  - pooled critic (target = the other batches);
  - MMD;
  - Sinkhorn with a W1 (p=1) cost;
  - scvi-tools' own AdversarialTrainingPlan (κ = 1 − kl_weight) as the "as people use it" anchor.
- **3.3 Adversary input.** Posterior mean or sample, decided by pilot A2.
- **3.4 Training and disclosures.** Shared optimiser; adversary optimiser; inner-loop minibatch reuse; fixed λ vs the KL ramp, reporting final kl_weight per dataset (immune 0.595).
- **3.5 Data.** Integer counts only (guard checks every value). Per-dataset repairs (§7). scIB batch keys. HVGs by seurat_v3 on counts with batch_key. No min_genes filter on curated files. Table 1 regenerated.
- **3.6 Evaluation.**
  - `scib.metrics.metrics` (scib 1.1.7); raw metrics are primary.
  - The scIB overall (min–max within one pre-declared comparison set per dataset, 0.4 batch / 0.6 bio) is secondary.
  - Organism set per dataset; trajectory conservation where pseudotime exists.
  - Downstream panel and training diagnostics.
- **3.7 Statistics** (§5).

**4 Results** (one subsection per original section, plus R0 and R5):

| Section | Original claim | Hypothesis (two-sided) | Experiments | Primary endpoint and decision | Reviewers |
|---|---|---|---|---|---|
| **R0 Premise** | JS gradients vanish for disjoint technology batches | Batches are near-separable in the scVI latent before integration, and discriminator gradients vanish early while critic gradients persist | X4, model-free overlap | Confirm if pre-integration batch AUC ≈ 1 **and** discriminator ‖∇z‖ collapses with stalled batch metrics; otherwise remove the premise from the motivation | R1.major.2 |
| **R1 Integration vs conservation** (orig. §1) | Critic gives superior local mixing; critic over-corrects; discriminator protective; local/global mismatch | At matched batch removal the critic and discriminator conserve different biology; their ordering differs between local and global batch metrics | X1/X2, X3, X10, A2 (X11 optional) | **P1** Δbio@b\* (reference critic − discriminator ×10 steps) on unscaled aggregates, co-primary frontier hypervolume; **P2** collapse probability; local–global sign test; composition dose-response slope with importance-weighted control | R1.major.1, R2.2, R2.6, R1.minor.2 |
| **R2 Scalability and stability** (orig. §2) | Critic scales poorly with batch count ("bottleneck") | The critic–discriminator difference changes with log₂V when per-head sample size and total cells are fixed | X8, X9 | **P3** slope on log₂V (stratified sampler; ceiling-normalised iLISI plus the full suite); attribute to reference design only if the pooled critic shows no slope; divergence rates | R1.minor.1, R2.5 |
| **R3 Hyperparameter sensitivity** (orig. §3) | Broad λ windows; discriminator bio flat in λ, critic's rises | The difference is attributable to the divergence rather than regularisation, update budget, latent scale or architecture | λ-curves from X1, X6, A3, X12 | Factor effects with CIs; window width (decades within the seed CI of the maximum); sign stability across architectures | R2.2, R3.2, R2.3 |
| **R4 Reference selection** (orig. §4) | Critic depends on a "topologically dense" reference | Reference coverage affects the reference W1 critic more than a reference JS or MMD design with the same target | X7 | **P4** formulation × coverage interaction (coverage, size, entropy as covariates; seed noise floor); K ≥ 3 only; fixed-reference control | R2.3, R2.5 |
| **R5 Placement and guidance** (new) | — | Does any adversary improve on stock scVI at equal budget? Where do they sit relative to standard integrators? | X13 (X14 optional) | Best-of-k budget curves, frontier positions; scANVI in a label-aware stratum | R2.4, R3.1, R3.3, R3.4 |

**Figures (outcome-neutral):**
1. Design schematic (scVI encoder/decoder, optional batch conditioning, swappable adversary).
2. Per-dataset batch-vs-bio frontiers, with baselines as reference points.
3. Δbio@b\* forest plot with pooled posterior.
4. Per-metric λ-curves (individual metrics, 5-seed bands, baselines as horizontal lines).
5. Scaling.
6. Reference coverage.
7. Composition dose-response.
8. Factorial effect sizes.
9. Downstream panel.

**5 Discussion.**
- Guidance written from the decision rules; if no adversary beats stock scVI, say so.
- Limitations: a single categorical confounder; gene-activity ATAC; transductive evaluation; simulations treated as a separate family.

**Supplement.** Equations; disclosure of the earlier preprint's defects; per-dataset repairs with md5s; all metrics; sensitivity analyses.

## 4. Experiment matrix and compute

| ID | Section | Experiment | Seeds | Fits | GPU-h* | Days** | Tier |
|---|---|---|---|---|---|---|---|
| A1 | all | Stock-default re-pilot and calibration | 3 | 891 | 271 | 3.8 | 1 |
| A2 | RS1 | Adversary input: samples vs means | 3 | 108 | 63 | 0.9 | 1 |
| A3 | RS3 | Latent-scale shortcut (standardisation on/off) | 3 | 180 | 74 | 1.0 | 1 |
| X1/X2 | RS1 + RS3 | Core benchmark: frontiers, matched-alignment bio, local vs global batch metrics | 5 | 4,125 | 1,468 | 20.4 | 1 |
| X13 | RS5 (new) | Baselines at equal configuration budget | 5 | 385 | 21 | 0.3 | 1 |
| X3 | RS1 (mechanism) | Composition-shift dose-response (+ importance-weighted control) | 3 | 540 | 86 | 1.2 | 2 |
| X6 | RS3 | Divergence x regularisation x update budget | 3 | 270 | 102 | 1.4 | 2 |
| X7 | RS4 | Reference batch selection (every batch as reference, K>=3) | 3 | 216 | 83 | 1.2 | 1 |
| X8 | RS2 | Batch-count scaling, per-head sample size fixed | 3 | 504 | 328 | 4.6 | 1 |
| X9 | RS2 | Failure and variance accounting | - | 0 | 0 | 0.0 | 1 |
| X4 | RS1 (premise) | Disjoint-support and gradient diagnostics | - | 0 | 0 | 0.0 | 1 |
| X10 | RS1/RS5 | Downstream biology at matched batch removal | - | 0 | 0 | 0.0 | 1 |
| X12 | RS3 | Architecture/hyperparameter robustness (fractional factorial) | 3 | 288 | 134 | 1.9 | 2 |
| X11 | RS1 (ground truth) | Parametrised Splatter simulations | 3 | 324 | 21 | 0.3 | 3 |
| X14 | RS2/RS5 | Atlas scale (HLCA core) | 3 | 15 | 8 | 0.1 | 3 |

\*GPU-h = local RTX 3080 aggregate hours with 6 concurrent fits. \*\*Days = wall days on the local GPU plus one JHPCE GPU (taken as ≈2× local).

| Scope | Fits | GPU-h | Wall days (local + 1 JHPCE GPU) |
|---|---|---|---|
| Tier 1 (rigorous version of the four original sections + reviewer-required) | 6,409 | 2,308 | 32.1 |
| Tier 1 + 2 (adds X3, X6, X12) | 7,507 | 2,630 | 36.5 |
| Tier 1 + 2 + 3 (adds X11, X14) | 7,846 | 2,659 | 36.9 |

- **Throughput model.** Measured in the counts pilot (atac_small, 6 lanes): discriminator ≈ 240 and critic ≈ 37 optimizer steps per second aggregate. Critic cost is assumed linear in the number of batch heads; discriminator-10, MMD and Sinkhorn rates are assumptions. Treat every duration as ±2× until A1 measures stock-default throughput.
- **More GPUs.** Each additional JHPCE GPU shortens wall time roughly in proportion: Tier 1+2 on local + 3 JHPCE GPUs ≈ 15.7 days.
- **Scoring.** CPU-bound: ~2 min per latent on atac_small (pilot, 11k cells); the larger tasks will be slower (A1 measures this on immune). Runs concurrently on CPU nodes.
- **Sizing.** The core uses 6 λ per family and 5 seeds; new-dataset sizes are planning assumptions until prep.

**Phasing:**

| Phase | Content | Gate |
|---|---|---|
| A — pilot and calibration (~6 d) | A1–A3 on atac_small, immune, sim1 | Grid range per family spans λ=0 → collapse; σ and τ estimated; adversary input and standardisation decided |
| B — data (CPU, parallel with A) | Acquire HLCA subset, BMMC, Ding 2020, CellBench, Kang 2018; pancreas repairs; registry with md5s; integer guard passes | Table 1 regenerated |
| **Freeze** | Manifest hash, code SHA, analysis scripts and this pre-registration committed; disclose that the pilot informed the grid ranges | — |
| C — core (~20 d) | X1/X2 + X13 (X4/X9 logged) | All per-dataset optima interior, else extend the grid |
| D — mechanisms | X7, X8 (tier 1); X3, X6, X12 (tier 2) | — |
| E — analysis and writing | X10, figures, response letter from new results only | — |

## 5. Analysis plan (pre-registration)

- **Unit of inference: the dataset.**
  - Families enter as a random effect: immune and immune_hum_mou share all human cells; atac_small and atac_large share studies.
  - Simulations form a separate family, excluded from primary pooling.
  - Model: score ~ arm + (1 | family/dataset) + (1 | dataset:seed). Seeds are nested replicates, paired across arms (same init and minibatch order).
- **Primary endpoints P1–P4** (§3, Results table). They are computed on unscaled scIB metrics with `pcr_comparison`; batch and bio are reported separately and as the composite.
- **Decision rules.**
  - Bayesian hierarchical model (Corani 2017); ROPE = max(0.01, 2 × measurement SD).
  - Measurement SD comes from re-scoring 5% of embeddings on a second machine and with different kNN/Leiden seeds.
  - Declare a direction if its posterior probability is ≥ 0.95, equivalence if P(ROPE) ≥ 0.95, and inconclusive otherwise.
  - Frequentist companion: exact Wilcoxon on dataset means with Holm over P1–P4. With 9 real tasks the smallest attainable two-sided p is 0.0039, below Holm's first threshold of 0.0125. With 5 families nothing could reach significance, so the new datasets are needed.
- **λ handling.**
  - The frontier analysis needs no λ selection.
  - Single-λ tables use nested leave-one-dataset-out "transfer λ". Per-dataset oracle optima are shown only as labelled optimistic bounds.
  - Any optimum at the grid edge triggers grid extension before analysis.
- **Scaling.** One comparison set per dataset (all fixed-grid configurations, both decoders, baselines), never subsets; complete metric vectors required. Sensitivities: leave-one-method-out rescaling, z-score aggregation, a composite without the silhouette metrics; rank stability reported as Kendall τ.
- **Failures are outcomes.** Rates are reported with Clopper–Pearson intervals. Endpoints impute the unintegrated score, with a sensitivity analysis excluding failures; nothing is dropped silently.
- **Budget parity.** Every method family gets the same number of configurations, shown as best-of-k curves.
- **Power** (provisional; pilot paired seed SD 0.048). The pooled design can certify differences of ≳ 0.02 or equivalence within ±0.02. No per-dataset significance claims. σ and τ are re-estimated in A1 before seeds and N are frozen.

## 6. Methodological rigour: what changes

| Issue | Fix |
|---|---|
| λ not commensurable across objectives; "curve dominance" at matched nominal λ is not selection-free | Frontier endpoints (bio at matched batch removal, hypervolume); λ-curves descriptive only |
| Divergence confounded with gradient penalty, update budget and target design | Discriminator at 1 and 10 steps in the core; X6 factorial; X7 target factorial |
| Composition shift uncontrolled; "balanced" ≠ independent | X3 dose-response on composition-matched subsamples; d_JS(Y) as a covariate everywhere (measured 0.15–0.44 across tasks, [composition_shift_by_dataset.csv]({{artifact:824c0218-4cb8-450d-a667-8c5ef961aae7}})) |
| iLISI is label-agnostic and rewards cross-type mixing | Within-label vs cross-label decomposition; full metric suite; no iLISI-only claims |
| Batch-count "decay" partly a metric ceiling and composition change | Ceiling-normalised iLISI; fixed total cells and composition across V |
| K=2 tasks make reference choice irrelevant | Reference analyses on K ≥ 3 only; state the equivalence |
| Winner's curse in λ selection; pre-registered λ=20 was chosen from benchmark data | Transfer-λ rule; interior-optimum check |
| Unequal tuning budgets; scANVI uses labels | Equal configuration counts; scANVI in its own stratum |
| scIB min–max depends on the comparison set | Raw metrics primary; one fixed set |
| Dataset count too small for any significance across families | Add 4 raw-count real datasets (≈ 9 real tasks, 7 families) |
| Mechanistic language | Used only for factors isolated by X3–X8 |

## 7. Datasets

| Task | Family | Batch key (core) | V | Cells | Decision |
|---|---|---|---|---|---|
| pancreas | pancreas | tech | 8 | 14,890 | Drop smarter (RPKM, unrecoverable); invert celseq/celseq2 UMI-collision correction (exact); round fluidigmc1 expected counts (+ sensitivity without it) |
| immune | immune | batch (10 donors) | 10 | 33,506 | Round smart-seq2 expected counts (+ sensitivity without them) |
| immune_hum_mou | immune | species (core, disjoint-support case); batch (23) for X8 | 2 | 97,861 | Rounding keyed on 'chemistry'; no min_genes |
| atac_small | ATAC | batchname | 3 | 11,270 | Integer; describe as gene activity |
| atac_large | ATAC | batchname_all | 11 | 84,813 | Integer |
| HLCA subset (replaces lung) | lung | study | ~8 | ~40k† | CELLxGENE raw counts; scIB lung dropped (its 10x v2 batch, 70% of cells, is log-normalised and unrecoverable from the file) |
| BMMC (NeurIPS 2021) | bone marrow | site × donor | 13 | ~69k† | GSE194122; confirm the counts layer |
| Ding 2020 PBMC | PBMC technologies | technology | 7 | ~30k† | GSE132044 count matrices; low-composition-shift control |
| CellBench | cell lines | protocol | ~4 | ~5k† | GSE118767; identity known by design (ground truth for over-correction) |
| sim1, sim2 | simulation | Batch (sim2 SubBatch 16 for X8) | 6, 4 | 12,097; 19,318 | Integer; separate family |
| Kang 2018 | — | donor | 8 | — | GSE96583; downstream condition preservation only |

†Planning assumption; confirm at download.

## 8. Code: non-standard implementations and rejection risks

Each critical and high finding was re-checked against the source code. The empirical numbers behind N1–N3 come from 1-seed runs of ≤ 12 epochs and are provisional.

| ID | Sev. | Issue | Action | Status |
|---|---|---|---|---|
| N9 | critical | Regeneration path still rebuilds the rejected design: committed manifest (2,000 rows, all unconditioned, disc 1 step, lambda 5-1500), driver defaults 239 ep / batch 512 / n_latent 30, final scoring via score_final_config.py (raw PCR, wrong sign) | New manifest generator with every knob as a column; fit script refuses missing knobs; all scoring via score_scib_native.py; retire score_final_config.py; write config + git SHA + package versions into each latent file | must fix before freeze |
| N1 | critical | Adversary trains on posterior samples; the scored embedding is the posterior mean. Provisional run: at lambda=30 the critic is satisfied on samples while batch stays decodable from the means | Pilot A2 decides: adversary on posterior means (DANN/WDGRL-consistent) or samples (scvi-tools convention); log masking gap and sigma^2/Var(mu) for every run | decide in Phase A |
| C1 | critical | Composite added raw scib pcr() (higher = worse) with + sign | Fixed on the scIB-native path (pcr_comparison); retire old path (N9) | fixed (pilot path) |
| C2 | critical | Prep copied log-normalised X into layers['counts']; every NB/ZINB fit on log data | Fixed: source counts layer; integer guard over every value; recorded rounding only for quantifier expected counts | fixed (11bd8e6, ca6c3a0) |
| N4 | high | Pooled critic target includes the head's own batch: estimates (1-pi_k) W1(P_k, rest); for K=2 it is half the reference objective and reference choice cannot matter | Target = other batches (or reweight 1/(1-pi_k)); state K=2 equivalence; run reference analyses only on K>=3 | must fix before freeze |
| N3 | high | Barycenter = 64 anchors initialised near 0 and moved by the generator optimiser (eps 0.01): fixed point at small lambda, contracting quantiser at large lambda | Drop (recommended) or rebuild (k-means init after warm-up, >= batch-size anchors, own optimiser) | decision |
| N2 | high | z-standardisation divides by one pooled RMS of sampled z: anisotropy and noise-inflation shortcuts | Per-dimension standardisation using posterior-mean statistics; diagnostic arm unless spectrum matches lambda0 | must fix before A3 |
| H2/N16 | high | Sinkhorn: normalising scale under no_grad (drives latent contraction) and squared-Euclidean cost (W2^2, not W1) | Scale with gradient; p=1 cost for a W1-matched arm; describe MMD as a different IPM | must fix before freeze |
| N10 | high | Pancreas, lung, immune_hum_mou cannot be prepped; min_genes=300 removes 10-11% of cells from two curated scIB tasks, unevenly by cell type | Per-dataset repairs (table); rounding column field; skip min_genes on curated files; report cells vs scIB Table 1 | must fix before freeze |
| N11 | high | Batch keys differ from scIB on 5/8 datasets (lung protocol 2 vs 16 donors; immune chemistry 4 vs 10; ...) | scIB keys for the main benchmark; coarse technology keys as a labelled variant | decision (recommend scIB keys) |
| N12 | high | Baselines break the same-epochs/batch/latent rule (scVI 30 dims, 200 ep, early stopping; PCA/Harmony/Scanorama 50 dims); scANVI trained on the evaluation labels | Stock-schedule scVI/scANVI; 10 dims for PCA/Harmony/Scanorama (50 as sensitivity); scANVI in a label-aware stratum; deterministic baselines once | must fix before freeze |
| N13 | medium | scIB overall depends on the comparison set; organism default 'human' (wrong for mouse ATAC, sims); trajectory disabled although immune has dpt_pseudotime; Leiden grid differs from the paper | One pre-declared comparison set per dataset; raw metrics primary; organism per dataset; trajectory for immune; state the scib 1.1.7 clustering grid | must fix before freeze |
| N5 | medium | 'Reference' is not fixed: reference cells receive the summed pull of all K-1 heads (star of pairwise W1 terms) | Describe as star-shaped objective; add fixed-reference control (detach reference z) in X7 | add arm |
| N6 | medium | At batch 128 small batches give ~4-5 cells per head; critics weight batches equally, discriminator weights cells | Report n_k per minibatch; stratified-sampler arm in X8 | add arm |
| N7 | medium | Adversary optimiser/steps match neither WGAN-GP (lr 1e-4, betas (0,0.9), n_critic 5, fresh samples) nor scvi-tools' classifier (32 units, kappa schedule) | Stock scvi-tools adversary as an anchor arm; WGAN-GP-default critic setting in X12; disclose shared head and minibatch reuse | add arm + disclose |
| N8 | medium | Fixed lambda from step 0 against a KL warm-up that never completes on large datasets (final kl_weight immune 0.595) | Report final kl_weight per dataset; warm-up-complete sensitivity on 2 datasets | disclose + sensitivity |
| N14 | low | HVGs: scanpy 'seurat' flavour on log data (tutorial: seurat_v3 on counts with batch_key); ATAC = gene activity only | seurat_v3 on counts with batch_key; describe ATAC as gene activity | must fix before freeze |
| N15 | low | No adversary diagnostics logged (gap, GP, gradient norms, accuracy, ELBO terms) | Log per epoch and save history with each latent | must fix before freeze |
| M6 | low | Deterministic baselines (Harmony/Scanorama/PCA) counted as 5 identical seeds | Report once without seed CIs | must fix before freeze |
| L6 | low | Critic inner loop reuses one minibatch for all 10 steps (WDGRL does the same; WGAN-GP samples fresh) | Disclose | disclose |
| models | medium | Only latents are saved; marker/DE analyses need the trained decoder | Save model weights for every core fit (~1-2 MB each) | must fix before freeze |

**Verified correct at `5709a98`:**
- Loss signs and gradient penalty: per-head WGAN-GP, λ_gp 10, two-sided, excluded from the generator step.
- The bounded fool loss matches scvi-tools line by line.
- λ=0 is bit-identical to stock SCVI, conditioned and unconditioned.
- The unconditioned decoder is truly unconditioned.
- Fixed-seed determinism, with no cross-config leakage.
- The scored latent is the posterior mean.
- The KL warm-up and epoch heuristic are stock scvi-tools behaviour.
- `scib.metrics.metrics` is called correctly for embeddings (k = 15, pcr_comparison vs the prep's X_pca, kBET via R).
- Counts are integer in the new preps.
- The entropy-rule reference batch is applied.

## 9. Decisions needed

1. **Scope:** Tier 1 (≈32.1 d) or Tier 1+2 (≈36.5 d). *Recommend Tier 1+2:* X3, X6 and X12 answer R1.major.2, R2.3 and R3.2 directly.
2. **Adversary input:** posterior means (DANN/WDGRL-consistent; matches what is scored) or samples (scvi-tools convention). *Recommend deciding from pilot A2*, defaulting to means.
3. **Batch keys:** scIB standard keys (comparable to scIB; real multi-batch structure) or technology keys. *Recommend scIB keys*, with technology keys as a labelled variant.
4. **New datasets:** HLCA subset, BMMC, Ding 2020, CellBench (+ Kang 2018). Needed to reach ≥ 8 independent real tasks.
5. **Barycenter critic:** drop (recommended) or rebuild properly.
6. **GPUs:** stay at local + 1 JHPCE GPU, or allow more when your other agents are idle.
7. **Public record:** post a new preprint version superseding v1–v3 (including the D1 disclosure) and update the README.
