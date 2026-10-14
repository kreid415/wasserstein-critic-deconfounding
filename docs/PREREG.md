# Pre-registered decision rules for the Tier 1+2 rerun

*Status: signed off by the user 2026-10-02, revised with the user's answers of 2026-10-02/03 (v2, §7); branch `prereg-v2`. The rules freeze when this file, `scripts/prereg_rules.py` and the three rule scripts are committed and tagged before the first A1 fit. Every rule is a deterministic function of scored results; nothing in this file is chosen after seeing A1, A2, A3 or X1 data.*

Implementation: `scripts/prereg_rules.py` (all rule logic), `scripts/freeze_a1_grid.py` (R1 and the power report), `scripts/freeze_matched_lambda.py` (R4; stages `a1` and `x1`), `scripts/decide_a2_a3.py` (R2, R3). Tests: `tests/prereg/`. Governs `scripts/build_paper_manifest.py --design pilot` (design of record, CONSTRAINTS.md SI-16).

**Disclosure (for Methods).** The X1 λ ranges come from 3 of the 8 tasks (atac_small, immune, sim1), fitted on seeds 100–102, which are disjoint from every X1 seed (0–4). No A1 fit is used as an X1 fit, so every λ chosen on A1 is evaluated on fresh seeds (SI-13). Likewise the λ-matched follow-ups (X3, X6, X7, X8, X12) run on seeds 10–12, disjoint from X1 and A1, so no follow-up cell reuses an X1 fit from which its matched λ was chosen (SI-26).

## 0. Stage order

| Stage | Fits | Starts after | Resolves | Script |
|---|---|---|---|---|
| A1 | 3 tasks × {conditioned, unconditioned} × seeds 100–102 × (λ=0 + 6 arms × A1 grid) | — | — | — |
| R1 | — | A1 scored | X1 `lam` = g1…g6 | `freeze_a1_grid.py` |
| R4-a1 | — | R1 frozen | A2/A3 `lam` = a1_matched_lo / a1_matched / a1_matched_hi | `freeze_matched_lambda.py --stage a1` |
| A2, A3 | both decoders (SI-27); new halves only (posterior sample; standardisation on); the other halves are A1 fits | R4-a1 | — | — |
| R2, R3 | — | A2, A3 scored | X1 `adv_input` = A2, `zstd` = A3 | `decide_a2_a3.py` |
| X1 | 8 tasks, both decoders, seeds 0–4 | R1, R2, R3 | — | — |
| R4-x1 | — | X1 scored | follow-up `lam` = matched_lo / matched / matched_hi | `freeze_matched_lambda.py --stage x1` |
| X3, X6, X7, X8, X12 | seeds 10–12 (SI-26) | R4-x1 | | |

Fits that depend on no rule (X1 arms `none` and `scvi_adv`, X13) may run while A1–A3 are in progress. A row that still holds a placeholder cannot be fitted: `fit_paper_config.py` refuses a non-numeric `lam`, `int("A3")` fails for `zstd`, and the training plan refuses an `adv_input` other than `mean`/`sample`.

## 1. Definitions shared by all rules

- **Scores.** Raw `scib.metrics.metrics` values written by `scripts/score_scib_native.py`, one CSV row per fit, joined to the manifest by `tag`. kBET is deterministic: scib 1.1.7's wrapper sets no seed and R's kBET samples test cells (`testSize`, `n_repeat`), so the scorer sets R's RNG with `set.seed(KBET_SEED)` immediately before every R kBET call (default `KBET_SEED_DEFAULT` = 0) and writes `kbet_seed` and `kbet_r_calls` to each row; `tests/scoring/` shows that the same latent scored twice gives identical kBET and that a different seed changes it. The measurement-SD step of PAPER_PLAN §5 (re-scoring 5 % of embeddings) varies `KBET_SEED` explicitly, together with the second machine and the kNN/Leiden seeds. Those re-scores are kept apart from the primary scores: a rule script stops if the scores it reads mix kBET seeds, and it rejects score files without `kbet_seed` (unseeded output of the scorer before 2026-10-03).
- **Batch score B.** Unweighted mean of the 5 scIB batch metrics in `score_scib_native.BATCH_METRICS` (PCR_batch = `pcr_comparison`, ASW_label/batch, iLISI, graph_conn, kBET). scib 1.1.7 returns each on [0, 1] with 1 = best (checked in `scib/metrics/{pcr,silhouette,lisi,kbet}.py`), so B needs no rescaling and does not change when runs are added. The min–max scaled `scib_overall()` is not used by any rule.
- **Bio score C.** Unweighted mean of the task's metrics in `score_scib_native.BIO_METRICS`: NMI, ARI, ASW_label, isolated_label_F1, isolated_label_silhouette, cLISI, plus cell_cycle_conservation on pancreas, lung, immune and immune_hum_mou (RNA, human; read from the prepped files' `uns`), plus trajectory conservation on immune and immune_hum_mou (the tasks whose prepped files hold `obs['dpt_pseudotime']`). Trajectory joined `BIO_METRICS` on 2026-10-03 (SI-28) because scIB's bio score includes it: `group_bio` in scib-reproducibility `visualization/plotSingleTaskRNA.R` lines 45–46, bio score = `rowMeans` over the available scaled bio metrics, lines 142–143 (main @ 3afbffd367); scib-pipeline `scripts/metrics/metrics.py` computes it iff `dpt_pseudotime` is in `obs`. The two immune tasks therefore average 8 bio metrics, pancreas and lung 7, ATAC and simulations 6. The rule code reads both lists from `score_scib_native.py`, so there is one definition. Known-answer check: `experiments/metric_bio_trajectory/` (PREFLIGHT.md, `check_bio_trajectory.py`).
- **Required values.** For every scored fit, all 5 batch metrics and the task's bio metrics must be finite. A NaN is a pipeline fault, not an outcome: the rule script stops.
- **Fit outcome.** Each manifest row of a stage is exactly one of *scored* (one score row) or *failed* (one row in the failures table, status `diverged` = training loss became non-finite, or `nonfinite_latent` = the saved posterior mean has a non-finite value). A row that is neither is *missing* and the script stops; so does a row that is both, or a tag scored twice. Infrastructure errors (out of memory, killed job, I/O) are not failures: the fit is rerun.
- **Pairing.** Seeds are paired across settings (same seed = same initialisation and minibatch order). Every difference is taken within a seed, over the seeds where both fits succeeded.
- **Seed mean.** Mean over the successful seeds of a point; a point whose seeds all failed has no mean.
- **Ties.** Values within 1e-9 are equal.
- **λ sequence.** The A1 grid is 0.1, 0.3, 1, 3, …, 3000 (half-decade steps). Extensions continue the 1–3 sequence: 0.03, 0.01, 0.003, 0.001 below; 10000, 30000, 100000, 300000 above.

| Constant | Value | Used in |
|---|---|---|
| δ_min (smallest effect) | 0.01 | R1 (PAPER_PLAN §5 ROPE floor) |
| k (noise multiple) | 3 | R1 |
| C (collapse: bio loss vs λ=0) | 0.10 | R1 |
| q (b* position) | 0.5 | R4 |
| ROPE | 0.01 | R2, R3 |
| grid points per family | 6 | R1 (SI-16) |
| extension | 2 points per round, ≤ 2 rounds per edge (R1); 1 point per round, ≤ 2 rounds per edge (R4) | R1, R4 |

## 2. R1 — A1 → 6-point X1 λ grid per arm family

**A1 design.** Tasks atac_small, immune, sim1; arms discriminator, reference, pooled, barycenter, mmd, sinkhorn on the A1 grid; λ=0 (`none`) for every task × decoder × seed; conditioned and unconditioned decoders; seeds 100, 101, 102; posterior-mean adversary input, no standardisation (the A2/A3 defaults).

**Families.** *(user sign-off 2026-10-02, §7)* Divergence families, one grid per decoder condition: JS {discriminator}; W1 {reference, pooled, barycenter}; MMD {mmd}; Sinkhorn {sinkhorn} → 8 grids. The three W1 critics share a grid because λ multiplies the same quantity for each: the mean over active heads of a WGAN-GP W1 estimate under the same Lipschitz penalty (`wcd/critic.py` `ReferenceWassersteinLoss`, reduction `mean`; `scvi_adversarial_plan.py` `adv_term = -loss_g`). Equal λ therefore means equal adversarial weight, and their λ-curves can be overlaid. The discriminator (bounded cross-entropy fool loss), MMD (kernel scale) and Sinkhorn (entropic OT, critic-free) have different loss scales. Conditioned and unconditioned decoders get separate grids because the decoder can absorb batch in one case and not the other, so their responses can sit at different λ.

**Per block** β = (task, decoder, arm), at every grid λ:
- Δb(λ) = seed mean of B(λ, s) − B(0, s); Δc(λ) likewise for C; n_fail(λ) = failed seeds.
- Noise: σ_b = median over grid points (≥ 2 paired seeds) of the seed SD of B(λ, s) − B(0, s), each SD divided by its median-unbiasing constant √(χ²_{n−1, 0.5}/(n−1)); σ_c likewise. Thresholds δ_b(λ) = max(δ_min, k σ_b / √n(λ)), δ_c(λ) likewise.
- **No effect** at λ: n_fail = 0, |Δb| < δ_b and |Δc| < δ_c.
- **Floor** f_β = the largest index i such that every grid point ≤ i has no effect.
- **Collapsed** at λ: n_fail ≥ 1, or Δc ≤ −C.
- **Batch peak** p_β = the first index above f_β with Δb ≥ max(Δb) − δ_b (none if max(Δb) < δ_b).
- **Upper end** h_β = the smallest index i > f_β such that every grid point ≥ i is collapsed or at/after p_β: the onset of over-correction (batch removal no longer increases, within noise) or of collapse, whichever comes first, as a persistent regime.

**Edges.** A block reaches the *low edge* when the lowest grid point already shows an effect (no floor). It reaches the *high edge* when no upper end exists, or when the upper end is the top point and the top point is not collapsed (batch removal still rising at the top). A block with no effect anywhere is *non-responsive* and counts as a high edge. If any block of a family reaches an edge, A1 is extended for the whole family (every member arm × task × seed of that decoder) by the next 2 values of the λ sequence beyond that edge, the extension is fitted and scored, and R1 is re-run. After 2 rounds at an edge the edge is accepted: the floor is set to the lowest point (or the upper end to the top point) and flagged `low_edge_unresolved` / `high_edge_unresolved`; a block still non-responsive is left out of the interval and flagged. A family whose blocks are all non-responsive stops the script.

**Grid.** i_lo = min_β f_β, i_hi = max_β h_β over the family's blocks, so at λ_{i_lo} every block shows no effect and at λ_{i_hi} every block has reached over-correction or collapse. With w = i_hi − i_lo + 1 points:
- w ≥ 6: indices ⌊i_lo + j (i_hi − i_lo)/5 + ½⌋, j = 0…5 (6 points evenly spaced in log λ, both ends included);
- w < 6: all w points, then grid points added alternately above i_hi and below i_lo, starting above, skipping a side that has run off the A1 grid.

g1 < … < g6 are the six λ values; X1 rows of the family and decoder take them. Each X1 value has been fitted on all three A1 tasks.

## 3. R2 and R3 — A2/A3 → X1 adversary input and standardisation

**Design.** atac_small and immune, both decoders (SI-27), seeds 100–102 (the A1 seeds, so the default halves are A1 fits of the same decoder and are not refitted). λ window per task × decoder × arm = (lo, matched, hi) from R4 applied to A1 (stage `a1`).
- A2: arms discriminator, reference, pooled; input ∈ {posterior mean (A1 fits), posterior sample (new)} → 12 cells (2 tasks × 2 decoders × 3 arms), 108 new fits.
- A3: arms discriminator, reference, pooled, mmd, sinkhorn; per-dimension standardisation ∈ {off (A1 fits), on (new)} → 20 cells (2 × 2 × 5), 180 new fits.

**Endpoint per cell** (task, decoder, arm): D = bio@b*(alternative) − bio@b*(default). bio@b*(setting) is the seed-mean C linearly interpolated at B = b* (the stage-`a1` b* of that task and decoder) along the setting's window in λ order: the first adjacent pair whose seed-mean B values bracket b*; an exact hit takes that point. A window point whose seeds all failed is skipped. If a setting's B stays below b* at every window point it is *unreached-low*; above at every point, *unreached-high*. A cell is evaluable when both settings bracket b*. [Amended 2026-10-13: a cell of a task × decoder flagged bstar_low_signal is not evaluable and stays in n; section 9, SI-56, SI-66.]

**Decision.** One pooled decision over the cells of both decoders (SI-27). The alternative replaces the default for X1 only if all of:
1. at least ⌈2n/3⌉ of the n cells are evaluable (A2: 8 of 12; A3: 14 of 20), with at least one per task × decoder;
2. the mean D over evaluable cells is > ROPE (strictly);
3. the mean D is > 0 within each task × decoder;
4. the alternative has no more failed fits than the default over the windows.

Otherwise X1 uses the default. Additionally for A2 (*masking ⇒ mean*): if in any cell the posterior-sample input is unreached-low while the posterior mean brackets b*, the sample input is disqualified and X1 uses the mean. This is the N1 failure mode: the adversary is satisfied on noisy samples while the scored means keep batch.

- **R2 default** *(user sign-off 2026-10-02, §7)*: posterior mean. It is the scored embedding (`get_latent_representation`) and what DANN/WDGRL align. scvi-tools' own adversary trains on a sample (`AdversarialTrainingPlan`, `z = inference_outputs["z"]`, scvi-tools 1.4.2 `train/_trainingplans.py:689`); that convention is kept for the `scvi_adv` anchor arm, which is not governed by R2.
- **R3 default** *(user sign-off 2026-10-02, §7)*: standardisation off, the conventional implementation; A3 measures the latent-scale route that scale-dependent losses (W1, MMD, Sinkhorn) could use.

**Scope.** One decision for all six adversarial arms (barycenter inherits it without being tested), both decoder blocks of X1 (A2/A3 test both), and every downstream adversarial row: follow-up rows are built with the A1 defaults (mean, off) and `decide_a2_a3.py` sets them to the decision; a follow-up row with any other value stops the script. The two decisions are made independently from the A1 default; if both switch, the combination was not tested and is disclosed.

## 4. R4 — matched λ (stage `a1` for A2/A3, stage `x1` for the follow-ups)

Per task t and decoder c, on the stage's own fits (A1: A1 grid, seeds 100–102; X1: the family grid plus any R4 extension, seeds 0–4):
- b0 = seed mean of B for λ=0 (`none`). A failed λ=0 fit stops the script.
- M_a = the highest seed-mean B of arm a over its grid points with no failed seed; b_common = min over the six adversarial arms of M_a.
- **b\*** *(user sign-off 2026-10-02, §7)* = b0 + q (b_common − b0), q = 0.5: half-way from scVI to the batch removal that every arm reaches. Unscaled, fixed once the stage is complete, and inside every arm's range by construction. Flag `bstar_low_signal` when b_common − b0 < 2 δ_min. The same b* is proposed for P1 (bio at matched batch removal).
- **Matched λ** for (t, c, a) = the grid point with no failed seed whose seed-mean B is closest to b*; ties go to the smaller λ. lo / hi = its neighbours in the arm's sorted grid.
- **Edge.** If the matched point is the lowest or highest grid point, the next value of the λ sequence beyond it is fitted for that (t, c, a) on the stage's seeds and R4 is re-run. After 2 rounds at an edge the missing neighbour is set to the next λ-sequence value without a fit and flagged `edge_unresolved`. Only cells that a row of the stage takes a value from can ask for an extension (stage a1: the A2/A3 cells; stage x1: the base-arm cells of the follow-up rows); other edge cells get the next λ-sequence value without a fit, flagged `edge_not_extended`.
- **Variant arms** inherit the matched λ of their base arm on the same task and decoder: discriminator_sn → discriminator; reference_sn, reference_fixed → reference; pooled_sn and pooled at other n_critic → pooled; barycenter_sn → barycenter; mmd_ref → mmd; discriminator_r1 (X6, R1 penalty) and discriminator_ref (X7, reference-anchored JS) → discriminator (added at the 2026-10-03 merge by the same rule, by analogy with discriminator_sn and mmd_ref). Subsampled tasks (X3, X8) and architecture variants (X12) inherit the full task's value. An arm without a mapping stops the script.
- R4 extension points are used for matching only; frontier and P1 analyses use the six pre-registered points per family.
- **Fresh seeds** *(user sign-off, §7; SI-26)*: the follow-ups X3, X6, X7, X8 and X12 run on seeds 10–12, disjoint from X1 (0–4) and A1/A2/A3 (100–102); X13 is not λ-matched and keeps seeds 0–4. No follow-up cell therefore reuses an X1 fit from which its matched λ was chosen. `freeze_matched_lambda.py --stage x1` stops if any resolved follow-up row equals an X1 row, and `prereg_rules.check_seed_separation` stops on any seed overlap between the three seed sets.

## 5. Power (reported by `freeze_a1_grid.py`; the design is fixed)

From A1, for the P1 contrast (reference critic − discriminator), per task × decoder: each arm at its R4-a1 matched λ, d_s = C_reference(s) − C_discriminator(s) per seed. σ̂ = pooled within-(task, decoder) SD of d_s; τ̂² = max(0, Var_t(mean d) − σ̂²/n) per decoder, pooled. For the fixed design (5 seeds, SI-16) with D = 6 real tasks (pancreas, lung, immune, immune_hum_mou, atac_small, atac_large; simulations are a separate family, PAPER_PLAN §5) and, conservatively, D = 4 families: SE = √((τ̂² + σ̂²/5)/D); power of a one-sided posterior-probability-0.95 call (non-central t, D − 1 df) for Δ = 0.02; the minimal detectable effect at 80 %; whether the 95 % interval at Δ = 0 fits inside ±0.01 and ±0.02 (equivalence). The design is flagged UNDERPOWERED when the minimal detectable effect exceeds 0.02 or equivalence within ±0.02 is unreachable.

Known before any data: the frequentist companion (exact Wilcoxon on dataset means, Holm over P1–P4) cannot reject with 6 real tasks. The smallest two-sided p is 2/2⁶ = 0.031, above Holm's first threshold 0.05/4 = 0.0125. PAPER_PLAN §5 relied on 9 real tasks; SI-02 removed the new datasets. [Amended 2026-10-14: the frozen X1 analysis applies Holm over six primary results (P1, the frontier area and P2, each for both decoders), not four, so its first threshold is 0.05/6 = 0.0083. The six-test rule is the agent's proposal in docs/ANALYSIS_SPEC.md section 15, item 5, and awaits the diff review. docs/prereg/r1_record.json still stores the four-test value, 0.0125 (N_ENDPOINTS = 4 in scripts/prereg_rules.py). The smallest two-sided p, 0.031, is above both thresholds, so the companion cannot reject at the Holm-adjusted level in either version; unadjusted, 0.031 is below 0.05.]

## 6. Freeze protocol

1. Commit and tag this file, `prereg_rules.py`, the three scripts and the manifest builder before the first A1 fit. Record the manifest SHA-256 in the tag message.
2. Each rule script writes a JSON decision record holding its inputs' SHA-256, the git SHA, the constants above, every per-block quantity it used and the decision. It writes a resolved manifest whose tags are unchanged, so tags identify design cells across stages. Exit status: 0 = frozen, 3 = extension needed (an extension manifest is written and nothing is frozen), any other = error.
3. Commit each decision record and resolved manifest before the stage it governs starts.

## 7. Sign-off answers

The user answered on 2026-10-02 (ask_user, recommended option first in each question). Each answer is also recorded in CONSTRAINTS.md.

| Rule | Question | Answer |
|---|---|---|
| R1 | Which arms share one 6-point λ grid? | "Divergence × decoder (Recommended)": {discriminator}, {reference, pooled, barycenter}, {mmd}, {sinkhorn}, each with a separate grid for conditioned and unconditioned decoders (8 grids) |
| R2 | Which adversary input is the default? | "Posterior mean (Recommended)": switch to sample only if the §3 criteria hold; masking ⇒ mean |
| R3 | Which standardisation setting is the default? | "Off (Recommended)": switch on only if the §3 criteria hold |
| R4 | How is b* defined? | "Midpoint of common range (Recommended)": b* = b0 + 0.5 (b_common − b0) |

The thresholds stated with the questions (δ_min 0.01, k = 3, C = 0.10, ROPE 0.01) were not changed.

v2: the user answered the open items of v1 on 2026-10-02/03 (ask_user; recorded in CONSTRAINTS.md):

| Item | Question (abridged) | Answer |
|---|---|---|
| A2/A3 decoders | "... Should A2/A3 also run with the unconditioned decoder?" | "Add it, one pooled decision (Recommended)" (SI-27; §3) |
| Follow-up seeds | "... Which seeds should the follow-ups use?" | "Fresh seeds for all follow-ups (Recommended)" (SI-26; §4) |
| Trajectory | "... Our scorer computes it for immune and immune_hum_mou but leaves it out of the bio aggregate. Should it be included? ..." | "Include trajectory (Recommended)" (SI-28; §1) |

## 8. Limitations and open items

- The λ ranges come from 3 of 8 tasks; X1 tasks whose matched λ falls on a grid edge are extended by R4 (disclosed per task).
- R1 is frozen before R4-a1 adds any cell-specific A1 point; re-running `freeze_a1_grid.py` after an R4-a1 extension stops on the uneven grid by design.
- Extension rows (R1, R4) belong to a task and must run on that task's machine (SI-17).
- The failures table (tag, status, detail) has to be written by the fit harvester; no runner writes it yet.
- Resolved in v2 (2026-10-03): A2/A3 now cover both decoders (SI-27); follow-ups run on fresh seeds, so no follow-up cell reuses an X1 fit and the builder's fit count no longer includes rows that are deduplicated later (SI-26); trajectory is in `BIO_METRICS` (SI-28).

## 9. Amendments after the freeze (tag prereg-tier12-v1)

Each amendment is dated, states what changes and why, and is committed before the stage it affects. The 2026-10-03
amendment changes no fit. Those of 2026-10-04 change only stages that had not started, and A1 restarts from zero at
the tag prereg-tier12-v2, which freezes this file again before the restarted A1's first fit (section 6).

- **2026-10-03, sensitivity analysis without trajectory (user decision, SI-38).** Every X1 comparison on immune and
  immune_hum_mou is also reported with the bio score C recomputed without trajectory conservation (C over the
  remaining 7 bio metrics). The primary analysis is unchanged (C with trajectory, as scIB; SI-28). Reason: on the
  64 gate latents (immune, 3 epochs) trajectory had SD 0.121 for barycenter and 0.057 for the discriminator versus
  <= 0.010 for the other arms, and one latent scored 0.50 locally and 0.87 / 0.89 on two other GPUs
  (docs/jhpce/gate_data_v2/g3/scores_192_latents_wcd-kbet.csv). In scib 1.1.7 (metrics/trajectory.py) pseudotime is
  computed on the largest connected component of the kNN graph only (other cells get 0) and the root cell is chosen by
  a vote over diffusion components, so small latent changes can change the value. A1's rule R1 is unaffected: one
  metric of 8 moving 0.39 in one seed shifts the 3-seed mean C by about 0.016, against the collapse bound 0.10. [Corrected 2026-10-13: R1 does depend on trajectory, through its
  noise estimate; see the I091 entry below.]

- **2026-10-04, A1 restarts at prereg-tier12-v2 (user decision SI-43, under SI-40).** A code check of prereg-tier12-v1
  (2026-10-03; 14 findings: 0 critical, 3 high, 3 medium, 8 low) led to fixes in the fitting driver, the scorer, the stage
  runner and the CPU baselines. A1's training code (scripts/scvi_adversarial_plan.py, src/wcd_vae/wcd/) is unchanged
  apart from an optional KL warm-up argument used only by X15; the final code gives latents bit-identical to the tag for
  all 7 A1 arms x 2 decoders (14/14; atac_small, 1 epoch, CPU), and the fixed scorer gives identical values on two A1 rows and one immune fit (15/15 columns each). A1 nevertheless restarts from
  zero, so that every fit and score of the run comes from one tagged commit. The 202 A1 fits made at prereg-tier12-v1
  (2026-10-03 17:07 to 2026-10-04 09:27 EDT) stay on disk unused; no A1 metric value informed any decision (two A1 rows
  were rescored only to test scorer identity).
- **2026-10-04, X3 uses equal cell counts per dose (user decision SI-41; implementation SI-42; code check CR-04).** At
  prereg-tier12-v1 each depletion dose removed cells (up to 25% fewer cells and steps at dose 100) and drew its cells with
  a different seed. Now every dose of a task has the task's dose-100 size (atac_small 8,402, pancreas 14,409, sim2
  17,872, immune_hum_mou 20,000 cells), drawn from one dose-independent permutation: K0 = its first N cells; dose d
  removes the first round(d x h0_t) cells of each declared type t of K0 in permutation order and refills with the next
  non-declared cells, so doses are exactly nominal, nested and maximally overlapping. The oracle importance weights
  (SI-25) become w[b, y] = n_K0(b, y) / n_Kd(b, y) per batch and cell type from the realised draws (doses 50/80/95,
  same arms, self-normalised); they restore each batch's dose-0 composition exactly and equal SI-25's formula when
  nothing is refilled. Every X3 row carries draw nested_v1, so all 828 X3 tags change (docs/x3_si41_tag_map.csv).
- **2026-10-04, subsets are scored with the all-feature PCR reference (code check CR-01, CR-14).** For X3 and X8
  subsets the tagged scorer built an HVG-only PCA as the PCR reference; full-data scoring, like scib-pipeline, lets
  scib recompute the PCA of all features. Subsets now use the full-data convention (X3 pancreas, uncorrected 10-PC HVG embedding:
  PCR_batch 0.573 at dose 50 and 0.560 at dose 0 with the all-feature reference, 0.000 with the HVG one). Full-data scoring (A1, A2, A3, X1) is unchanged.
- **2026-10-04, scorer provenance and scib fallback flags (code check CR-05, CR-08).** Every score row records the
  scorer's git SHA and cleanliness, host, CPU model and SIMD level, NUMBA_CPU_NAME and package versions, and flags
  trajectory's root-cell fallback (value 0) and kBET labels forced to 1 or skipped. The stage runner refuses a stage
  with mixed scorer provenance, and prereg_rules.py refuses a rule input with mixed or missing provenance. The flags
  are recorded in every score row and their counts per rule cell are reported with the results; the rule scripts do
  not read them, so no decision depends on them. The values remain scib's own.
- **2026-10-04, pairing wording (code check CR-07).** Section 1's "same seed = same initialisation and minibatch order"
  holds for the scVI module initialisation and the train/validation split in every arm. The minibatch order is shared
  only within an arm and with the arms that have no adversary head (MMD, Sinkhorn): building the adversary head draws
  from torch's generator before the first epoch's sampler permutation. Differences against lambda=0 remain unbiased;
  they are less tightly paired than stated, which the empirical paired SD already reflects.
- **2026-10-04, declared differences between arms (code check CR-12).** Within a task every arm has the same epochs,
  batch size, latent size and backbone. Declared differences: each critic takes 5 Adam steps (lr 1e-4, betas 0 / 0.9)
  per generator step on the same minibatch (WGAN-GP draws fresh data for each critic step); the discriminator takes 1
  Adam step (lr 1e-3, eps 0.01, weight decay 1e-6); scANVI adds min(10, max(2, round(E/3))) epochs after its scVI
  pretraining at train_size 0.9; scvi-tools' SysVI trains without KL warm-up. These are design choices or tool
  defaults and are stated in Methods.
- **2026-10-04, X13 Scanorama grid knn {2, 5, 10, 20, 40, 80} (user decision SI-46, under SI-37).** On immune_hum_mou
  (97,861 cells; 2 threads, beside A1) knn 20 took 1,843 s and 10.5 GB, knn 80 7,457 s and 23.2 GB, and knn 160 did not
  finish within 8 h (43.9 GB at the limit; tier12/x13_scanorama_timing/scanorama_timing.csv). knn 160 is replaced by 2,
  keeping 6 values; the dimred-100 tool default at knn 20 stays as the sensitivity run.
- **2026-10-04, edge_unresolved window points in A2/A3 (user decision SI-45; code check CR-11).** If R4 stage a1 sets a
  window neighbour without an A1 fit (edge_unresolved, section 4), the default setting (mean / off) is fitted at that
  lambda on A1's seeds (100-102, same decoder) as A1 extension rows; R2 and R3 then apply as written on 3-point windows.
- **2026-10-04, X15: KL warm-up sensitivity (user decision SI-44; PAPER_PLAN N8).** With the stock 400-epoch KL warm-up
  the last epoch trains at KL weight 0.23 on atac_large, 0.20 on immune_hum_mou, 0.595 on immune and 0.61 on lung (399/400
  = 0.998 on the four 400-epoch tasks),
  while the adversary acts at full lambda from the first step. X15 fits immune (local) and atac_large (JHPCE L40S) x
  {lambda=0, discriminator, pooled critic, MMD at the matched lo / hi lambda} x seeds 10-12 x {stock warm-up, a warm-up
  that completes (length = the task's epoch count E, so the last epoch trains at KL weight (E-1)/E: immune 0.996,
  atac_large 0.989; the 400-epoch tasks end the stock warm-up the same way, at 399/400)}: 84 fits, conditioned decoder,
  after X1 with the follow-ups. It is reported as a sensitivity analysis of the X1 comparison. Its 12 stock-warm-up rows
  for the discriminator and the pooled critic on immune have the same settings as X6 rows; they are fitted separately
  (docs/x15_stock_equal_followups.csv).
- **2026-10-04, manifest of record.** manifests/paper_manifest_stock_pilot_u5_b10_v3.tsv, 6,930 rows, sha256
  6efbe422c8f09f9884001a84f7d7b363506f435803d238280e00bbb7b7e1cf24 (= python3 scripts/build_paper_manifest.py --backbone
  stock --design pilot --uncond-seeds 5 --bary-iter 10). Against the tagged manifest (6,846 rows): the 828 X3 rows are
  re-tagged (SI-41, SI-42), the 8 Scanorama knn-160 rows become knn 2 (SI-46) and 84 X15 rows are added (SI-44); the A1,
  A2, A3 and X1 rows and all other rows are byte-identical.
- **2026-10-04, re-review of the fixes (RIGOR_REVIEWER, integrate-fixes 080f417).** (a) Section 8's "no runner writes
  it yet" is superseded: scripts/run_stage.py writes OUT/failures.csv (tag, status, detail). (b) Every stage is fitted
  and scored from one worktree pinned at the tag prereg-tier12-v2. prereg_rules.py refuses rule inputs whose rows
  differ in scorer provenance, and scorer_git_sha is the checkout's HEAD, so a stage scored from any later commit would
  force rescoring of the stages it is combined with. Record and notebook commits go through a separate worktree.
  (c) Input guards added before the tag, with no change to any fit, score or decision rule: the JHPCE harvest refuses
  destinations under any tier12* directory; freeze_matched_lambda.py refuses to overwrite an existing
  --extension-manifest when it writes SI-45 rows; and the scANVI/sysVI guard test names the v1 fitter by commit
  (129142b) instead of by tag, because the JHPCE bundle clone carries no tags.
- **2026-10-13, correction: R1 depends on trajectory through its noise estimate (final review I091; user decision
  SI-53).** The 2026-10-03 sentence "A1's rule R1 is unaffected" is wrong. R1 uses C twice: in the collapse bound
  (-0.10), which one unstable metric barely moves, and in the bio noise SD sigma_c of the no-effect test (section 2,
  delta_c = max(0.01, 3 sigma_c / sqrt(n))). On immune with the conditioned decoder trajectory inflates sigma_c 3.3 to
  5.4 times (with / without trajectory: barycenter 0.073 / 0.014, pooled 0.060 / 0.017, reference 0.061 / 0.014,
  discriminator 0.064 / 0.017, MMD 0.060 / 0.018, Sinkhorn 0.059 / 0.013, the last from the pre-fix Sinkhorn fits), so
  delta_c reaches 0.10 to 0.13 and the no-effect test on bio cannot fail there. R1 re-run on the A1 fits without
  trajectory changes three conditioned-decoder families (W1 critics, MMD, Sinkhorn) from frozen to an extension at the
  low edge (lambda 0.03, 0.01); the other five families are unchanged
  (experiments/final_review_amendments_20261013/r1_without_trajectory.py, which first reproduces the record). R1 stays
  as registered (SI-53). The frozen X1 grids stand for every arm except Sinkhorn: the Sinkhorn grid will be re-derived
  by R1 on the refitted Sinkhorn fits (SI-57; entry on the Sinkhorn arm below), so docs/prereg/r1_record.json, which
  still holds the superseded Sinkhorn grid, will be superseded by the redone record. The Sinkhorn figures in this
  entry (sigma_c 0.059 / 0.013, and the move of the Sinkhorn family to an extension) are pre-fix: they come from the
  discarded Sinkhorn fits. The trajectory-free R1 is reported as a sensitivity analysis, its Sinkhorn part recomputed
  on the refitted Sinkhorn fits.
- **2026-10-13, PAPER_PLAN corrected to this file (final review I006, D-01, D-02; user decision SI-55).** (a) Grid
  edge: PAPER_PLAN promised that any X1 optimum at a grid edge triggers an extension before analysis; as section 4
  states, R4 extends grids for matching only, frontier and P1 use the six registered lambda points, and an optimum at a
  grid edge is disclosed, not extended. (b) P1 comparator: the discriminator at 1 adversary step (SI-01; every
  discriminator row of the manifest has n_critic 1), not "x10 steps"; the update-budget control is X6's pooled critic at
  1, 5 and 10 critic steps, so it exists only for the pooled W1 critic. (c) Power: section 5 flags the frozen design
  UNDERPOWERED (docs/prereg/r1_record.json, power). Its figures are those of the matched-lambda contrast of section 5
  (reference critic minus discriminator, each at its R4-a1 matched lambda), estimated from all six A1 pilot blocks (3
  tasks x 2 decoders), which include the simulation sim1 and the two blocks flagged low-signal (entry on low-signal
  blocks below); they are not the power of the frozen P1, which reads biology at the target batch score b\* by
  interpolation (docs/ANALYSIS_SPEC.md section 2.2). For D = 6 real tasks: SE 0.045, minimal detectable effect at 80 %
  power 0.131, power at Delta = 0.02 0.104; D = 4 families 0.182 and 0.088; equivalence within ±0.01 or ±0.02
  unreachable; sigma 0.030 (12 degrees of freedom), tau 0.109 (4 degrees of freedom); A1 per-block means of opposite
  sign, and in the conditioned decoder sim1 supplies 67 % of the squared spread of the three block means. Without
  sim1, the same formulas on the four remaining blocks give sigma 0.033 (8 degrees of freedom), tau 0.056 (2 degrees
  of freedom), SE 0.0236, minimal detectable effect 0.068 and power 0.182 for D = 6 (0.095 and 0.137 for D = 4);
  equivalence is still unreachable (experiments/final_review_amendments_20261013/power_without_sim1.py). The
  conclusion, underpowered, holds either way. P1 is therefore reported as the pooled posterior together with tau, the
  per-task posteriors and a per-task sign table; PAPER_PLAN's equivalence claim is dropped, and the power estimate
  will be recomputed with the frozen P1 reading and only the evaluable blocks when R4-a1 is re-run after the Sinkhorn
  redo (SI-57). (d) The frequentist companion cannot reject with 6 real tasks: the smallest attainable two-sided p,
  2/2⁶ = 0.031, is above Holm's first threshold, which is 0.0125 for the four tests of section 5 and 0.0083 for the
  six primary results of the frozen X1 analysis (bracketed note in section 5); it cannot reject at the Holm-adjusted
  level in either version. PAPER_PLAN's 9-task statement is replaced.
- **2026-10-13, two commits declared result-identical (user decisions SI-54, SI-60).** This amends the 2026-10-04 entry
  (b): rule inputs and stages may now combine rows from two commits. docs/EQUIVALENT_COMMITS.json declares that fits and
  scores made at c88cce3 (tag prereg-tier12-v2) are result-identical to those made at the new tag prereg-tier12-v3 for
  every arm except sinkhorn. The evidence is the fit and score identity checks of
  experiments/integration_checks_20261013, which are re-run at the candidate commit before it is tagged (SI-52).
  prereg_rules.py (scorer_git_sha) and run_stage.py (scorer_git_sha and the latents' git_sha) read c88cce3 as the tag's
  commit for those arms when rows mix commits, and still require one commit after that. Sinkhorn output made at c88cce3
  is refused: prereg_rules refuses the rule input, and the run_stage preflight refuses the stage, in report-only runs
  too. Sinkhorn rows are refitted and rescored at the tag (SI-57); no other row is rescored. If the tag is absent from a
  checkout, the mix stays and is refused, so a JHPCE bundle that combines rows from both commits must carry the tag
  (lesson RR-01).
- **2026-10-13, Sinkhorn arm corrected; its pilot fits are redone in rule order before X1 (final review I031, I032;
  user decisions SI-51, SI-57, SI-58, SI-61, SI-62).** The tagged Sinkhorn loss counted the gradient through its final
  transform twice (on the validation minibatches below, the tagged gradient had a median norm 2.00 times that of the
  converged gradient on A1 fits and 2.27 times on A3 fits; on the R1 extension rows 14.4 times, with median cosine
  0.28, over the 204 of 216 minibatches where the tagged code gave a finite gradient) and stopped the solve before it
  converged at the final epsilon. Since commit 3cbaca4 the gradient is taken once (envelope theorem) and every
  transport plan at the final epsilon converges to an L1 marginal error of 1e-4 (SINKHORN_TOL), in float64 with
  direct-difference distances. Limits: at most 500,000 iterations at the final epsilon, and reaching the cap raises an
  error that stops the stage (SI-61); a NaN marginal error ends the solve at once, a +infinity error does not, because
  finite latents of about 1e18 or more give +infinity and then converge a few iterations later (re-review R2-01,
  commit 89d5e2c). The solve stays in float64 (SI-58, SI-62): float32 met the agreement criterion (worst gradient
  cosine at least 0.99999, relative value error at most 1e-4) on the 598 hardest validation minibatches, but was no
  faster at 128-cell minibatches and failed the value criterion on all 3,076. The 180 A1 Sinkhorn fits will be
  refitted and rescored first, at the tag prereg-tier12-v3. Then, in rule order (SI-57): R1 for the Sinkhorn family,
  with the extension rows it asks for; R4 stage a1 over all six arms with the refitted Sinkhorn arm; the A3 rows it
  then resolves (the Sinkhorn rows, and any non-Sinkhorn cell whose window moved); R2 and R3 in full. The extension
  rows and the A3 Sinkhorn rows are therefore outputs of R1 and R4-a1 on the new fits, not a fixed list: before the
  fix there were 18 and 36 of them, 234 rows with the 180 A1 fits (the count named in SI-51), and the numbers may
  change (NB-20261013-09). No existing non-Sinkhorn fit is redone (SI-54 entry above); non-Sinkhorn A3 rows are fitted
  anew only where a window moves. The power estimate of section 5 will be recomputed with R4-a1. The solver was
  validated on 3,076 minibatches from the 234 pre-fix fits of atac_small, immune and sim1 (3, 10 and 6 batches; review
  note F6): 2,808 final-epoch minibatches (12 from each fit) and 268 captured earlier in training or in training mode
  (192 and 76); the check is repeated on the refitted latents before convergence is claimed for them or for the X1
  tasks with more batches (up to 23, immune_hum_mou). The records of the redo will be committed beside the frozen
  ones, which stay unchanged (names proposed for the diff review): docs/prereg/r1_record_sinkhorn_redo.json (R1),
  docs/prereg/r4_a1_record_sinkhorn_redo.json (R4 stage a1) and docs/prereg/r2r3_record_sinkhorn_redo.json (R2 and
  R3). The six non-Sinkhorn grids of the redone R1 record will be checked against docs/prereg/r1_record.json, and
  docs/ANALYSIS_SPEC.md will name the redone R1 record as its source of the six registered lambda per arm.
- **2026-10-13, guards before the tag (final review I033, I034, I042, I050, I051, I094, I008; user decisions SI-52,
  SI-63, SI-64; guard review follow-ups G-01 to G-14).** None changes a completed fit or score: at commit c3fd92d, 59 of
  59 non-Sinkhorn fits are identical to c88cce3 and 7 of 7 scored latents give identical values in all 32 compared
  columns (experiments/integration_checks_20261013; NB-20261013-39). (I033, c4be14b) A minibatch in which a critic-free
  loss has no pair of batches to compare (one batch present; for reference-anchored MMD, no reference cell or no other
  batch) contributes a zero that keeps the gradient graph, instead of stopping the fit; the fitter counts such
  minibatches and writes no_pair_steps into the latent's plan record when there are any (G-01, 96f4258). In a
  model-based estimate (the guard review, reproduced by the documents review of 2026-10-14; not simulated runs) they
  are expected in X7 reference-anchored MMD on pancreas (27 fits, 25.9 expected to meet at least one) and sim1 (all 18
  fits), and in about 14% of the 48 X3 atac_small MMD fits, 4.5% of the 24 X8 sim2 MMD fits and 0.6% of the 60 X1 sim1
  fits of each of MMD and Sinkhorn. The counts will be tabulated by arm, task and experiment from the latents' plan
  records by a short reporting script written after the runs. No code reads no_pair_steps yet; the script affects no
  rule, so it can follow the tag (proposal for the diff review). (I034, a4f4be0) A non-finite critic loss at any inner
  critic step, not only the last, marks the fit diverged. (I042, 5f30f17; SI-63) A fit refuses manifest settings and
  extra keys its arm does not read. (I050, I051, f5ea0c3) The scorer refuses a latent whose cell names or labels are
  not those of the prepped file in the same order, and the two pilot writers whose latents carry no cell names are
  retired to legacy/ (SI-64, c3fd92d); a kBET of exactly 0 with no seeded R call (scib set every tested label to
  rejection 1) is recorded, any other result without an R call refused. (I094, 46d79f2) A fit counts as complete, is
  resumed and is harvested from the cluster only with its model weights (GPU fits) and training history. (I008,
  2b52542) Each chained cluster job's clock is (chain position + 1) x (walltime + queue allowance).
- **2026-10-13, P1 starts at lambda = 0; low-signal cells are not evaluable (final review I072, D-04; user decisions
  SI-56, SI-66).** Every arm's bio-versus-batch curve starts at the lambda = 0 point, the plain-scVI scores that
  define b0, so a curve whose lowest registered point already lies beyond b\* is read between lambda = 0 and that
  point. A task x decoder flagged bstar_low_signal (b_common - b0 < 2 delta_min = 0.02, section 4) has too small a
  batch range for b\* to be read reliably: its cells are not evaluable for P1 or for R2/R3, and their readings are
  reported separately, never pooled. In R2/R3 such cells stay in n (the user chose this over leaving them out; SI-66),
  so while a block is flagged, criterion 1 of section 3 ("at least one per task x decoder") fails and R2 and R3 keep
  their defaults, posterior mean and standardisation off. The user decided SI-56 and SI-66 on 2026-10-13; the rule was
  implemented on 2026-10-14, in scripts/decide_a2_a3.py (commit f3c3415, 07:50; a flagged cell's readings carry status
  low_signal) and scripts/analysis_x1.py (commit ff88666, 08:04). Until f3c3415 the rule code did not apply them
  (NB-20261014-14). The frozen R4-a1 record (made at c88cce3) flags two blocks: immune with the conditioned decoder
  (b0 0.6382; b_common 0.6570, the barycenter critic's highest batch score; difference 0.0188) and sim1 with the
  conditioned decoder (b0 0.8120; b_common 0.8189, set by MMD; difference 0.0069). sim1 is not an A2/A3 task, so only
  the immune block bears on R2 and R3. The Sinkhorn redo cannot clear either flag: b_common is the minimum over the
  six arms and Sinkhorn sets it in no block of the record, so replacing the Sinkhorn arm can only lower it (and could
  flag further blocks). R2 and R3 will therefore keep the defaults whatever A2 and A3 show. The R2/R3 record
  (docs/prereg/r2r3_record.json) was made at c88cce3 on 2026-10-12, before the rule was implemented. Applying the rule
  to its stored cells changes no decision: R2 had 12 of 12 cells evaluable and a mean change in biology score
  (alternative minus default) of -0.0014, and has 9 of 12 and -0.0042 with the rule; R3 had 17 of 20 and +0.0010, and
  has 13 of 20 (14 needed) and -0.0014; a switch needs a mean change above 0.01, which neither version reaches. The
  pilot R2/R3 counts quoted elsewhere come from the frozen record made at c88cce3; the current code reports the new
  counts for the same inputs (checked 2026-10-14 by running prereg_rules.decide_binary on the record's stored cells
  with the low-signal marking that decide_a2_a3.cells_for applies). A2 and A3 exist (all 288 rows fitted and scored;
  the runner ended on 2026-10-11); only the Sinkhorn A3 rows (36 before the fix) are to be redone (SI-57). SI-57 still
  has them refitted and R2 and R3 re-run; since neither can change what X1 uses while the flag holds, whether to keep
  them is a question for the diff review.
- **2026-10-14, the X1 analyses are specified and frozen before X1 runs (final review I001, D-03, D-08, D-09; user
  decisions SI-68 to SI-77).** docs/ANALYSIS_SPEC.md specifies, and scripts/analysis_x1.py implements with known-answer
  tests (tests/analysis/), every analysis of X1 and X13 data: P1 per decoder with the frontier area as co-primary, P2
  (collapse rates and the critic-minus-discriminator comparison), local versus global batch removal, single-lambda
  tables with the transfer lambda, best-of-k placement against standard integrators, the lambda-window width and the
  sensitivity analyses. The tag prereg-tier12-v3 freezes all of them before any X1 adversarial-arm score is
  examined, and the script runs once, after every X1 and X13 row is scored or failed. The follow-up analyses (X3, X6,
  X7, X8, X12, X15) are specified later, with the X7/X8 redesign, while X1 runs and before any X1 result is examined;
  the premise check R0 and X10 are exploratory. This entry does not change R1-R4. Where PAPER_PLAN section 5 differs,
  the specification applies (its section 13 lists the replaced statements).
