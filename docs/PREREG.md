# Pre-registered decision rules for the Tier 1+2 rerun

*Status: signed off by the user 2026-10-02 (§7); branch `prereg-rules`. The rules freeze when this file, `scripts/prereg_rules.py` and the three rule scripts are committed and tagged before the first A1 fit. Every rule is a deterministic function of scored results; nothing in this file is chosen after seeing A1, A2, A3 or X1 data.*

Implementation: `scripts/prereg_rules.py` (all rule logic), `scripts/freeze_a1_grid.py` (R1 and the power report), `scripts/freeze_matched_lambda.py` (R4; stages `a1` and `x1`), `scripts/decide_a2_a3.py` (R2, R3). Tests: `tests/prereg/`. Governs `scripts/build_paper_manifest.py --design pilot` (design of record, CONSTRAINTS.md SI-16).

**Disclosure (for Methods).** The X1 λ ranges come from 3 of the 8 tasks (atac_small, immune, sim1), fitted on seeds 100–102, which are disjoint from every X1 seed (0–4). No A1 fit is used as an X1 fit, so every λ chosen on A1 is evaluated on fresh seeds (SI-13).

## 0. Stage order

| Stage | Fits | Starts after | Resolves | Script |
|---|---|---|---|---|
| A1 | 3 tasks × {conditioned, unconditioned} × seeds 100–102 × (λ=0 + 6 arms × A1 grid) | — | — | — |
| R1 | — | A1 scored | X1 `lam` = g1…g6 | `freeze_a1_grid.py` |
| R4-a1 | — | R1 frozen | A2/A3 `lam` = a1_matched_lo / a1_matched / a1_matched_hi | `freeze_matched_lambda.py --stage a1` |
| A2, A3 | new halves only (posterior sample; standardisation on); the other halves are A1 fits | R4-a1 | — | — |
| R2, R3 | — | A2, A3 scored | X1 `adv_input` = A2, `zstd` = A3 | `decide_a2_a3.py` |
| X1 | 8 tasks, both decoders, seeds 0–4 | R1, R2, R3 | — | — |
| R4-x1 | — | X1 scored | follow-up `lam` = matched_lo / matched / matched_hi | `freeze_matched_lambda.py --stage x1` |
| X3, X6, X7, X8, X12 | | R4-x1 | | |

Fits that depend on no rule (X1 arms `none` and `scvi_adv`, X13) may run while A1–A3 are in progress. A row that still holds a placeholder cannot be fitted: `fit_paper_config.py` refuses a non-numeric `lam`, `int("A3")` fails for `zstd`, and the training plan refuses an `adv_input` other than `mean`/`sample`.

## 1. Definitions shared by all rules

- **Scores.** Raw `scib.metrics.metrics` values written by `scripts/score_scib_native.py`, one CSV row per fit, joined to the manifest by `tag`.
- **Batch score B.** Unweighted mean of the 5 scIB batch metrics in `score_scib_native.BATCH_METRICS` (PCR_batch = `pcr_comparison`, ASW_label/batch, iLISI, graph_conn, kBET). scib 1.1.7 returns each on [0, 1] with 1 = best (checked in `scib/metrics/{pcr,silhouette,lisi,kbet}.py`), so B needs no rescaling and does not change when runs are added. The min–max scaled `scib_overall()` is not used by any rule.
- **Bio score C.** Unweighted mean of the task's metrics in `score_scib_native.BIO_METRICS`: NMI, ARI, ASW_label, isolated_label_F1, isolated_label_silhouette, cLISI, plus cell_cycle_conservation on pancreas, lung, immune and immune_hum_mou (RNA, human; read from the prepped files' `uns`). The rule code reads both lists from `score_scib_native.py`, so there is one definition.
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

**Design.** atac_small and immune, conditioned decoder, seeds 100–102 (the A1 seeds, so the default halves are A1 fits and are not refitted). λ window per task × arm = (lo, matched, hi) from R4 applied to A1 (stage `a1`).
- A2: arms discriminator, reference, pooled; input ∈ {posterior mean (A1 fits), posterior sample (new)} → 6 cells, 54 new fits.
- A3: arms discriminator, reference, pooled, mmd, sinkhorn; per-dimension standardisation ∈ {off (A1 fits), on (new)} → 10 cells, 90 new fits.

**Endpoint per cell** (task, arm): D = bio@b*(alternative) − bio@b*(default). bio@b*(setting) is the seed-mean C linearly interpolated at B = b* (the stage-`a1` b* of the task, conditioned) along the setting's window in λ order: the first adjacent pair whose seed-mean B values bracket b*; an exact hit takes that point. A window point whose seeds all failed is skipped. If a setting's B stays below b* at every window point it is *unreached-low*; above at every point, *unreached-high*. A cell is evaluable when both settings bracket b*.

**Decision.** The alternative replaces the default for X1 only if all of:
1. at least ⌈2/3⌉ of the cells are evaluable (A2: 4 of 6; A3: 7 of 10), with at least one per task;
2. the mean D over evaluable cells is > ROPE (strictly);
3. the mean D is > 0 within each task;
4. the alternative has no more failed fits than the default over the windows.

Otherwise X1 uses the default. Additionally for A2 (*masking ⇒ mean*): if in any cell the posterior-sample input is unreached-low while the posterior mean brackets b*, the sample input is disqualified and X1 uses the mean. This is the N1 failure mode: the adversary is satisfied on noisy samples while the scored means keep batch.

- **R2 default** *(user sign-off 2026-10-02, §7)*: posterior mean. It is the scored embedding (`get_latent_representation`) and what DANN/WDGRL align. scvi-tools' own adversary trains on a sample (`AdversarialTrainingPlan`, `z = inference_outputs["z"]`, scvi-tools 1.4.2 `train/_trainingplans.py:689`); that convention is kept for the `scvi_adv` anchor arm, which is not governed by R2.
- **R3 default** *(user sign-off 2026-10-02, §7)*: standardisation off, the conventional implementation; A3 measures the latent-scale route that scale-dependent losses (W1, MMD, Sinkhorn) could use.

**Scope.** One decision for all six adversarial arms (barycenter inherits it without being tested), both decoder blocks of X1 (A2/A3 are conditioned only), and every downstream adversarial row: follow-up rows are built with the A1 defaults (mean, off) and `decide_a2_a3.py` sets them to the decision; a follow-up row with any other value stops the script. The two decisions are made independently from the A1 default; if both switch, the combination was not tested and is disclosed.

## 4. R4 — matched λ (stage `a1` for A2/A3, stage `x1` for the follow-ups)

Per task t and decoder c, on the stage's own fits (A1: A1 grid, seeds 100–102; X1: the family grid plus any R4 extension, seeds 0–4):
- b0 = seed mean of B for λ=0 (`none`). A failed λ=0 fit stops the script.
- M_a = the highest seed-mean B of arm a over its grid points with no failed seed; b_common = min over the six adversarial arms of M_a.
- **b\*** *(user sign-off 2026-10-02, §7)* = b0 + q (b_common − b0), q = 0.5: half-way from scVI to the batch removal that every arm reaches. Unscaled, fixed once the stage is complete, and inside every arm's range by construction. Flag `bstar_low_signal` when b_common − b0 < 2 δ_min. The same b* is proposed for P1 (bio at matched batch removal).
- **Matched λ** for (t, c, a) = the grid point with no failed seed whose seed-mean B is closest to b*; ties go to the smaller λ. lo / hi = its neighbours in the arm's sorted grid.
- **Edge.** If the matched point is the lowest or highest grid point, the next value of the λ sequence beyond it is fitted for that (t, c, a) on the stage's seeds and R4 is re-run. After 2 rounds at an edge the missing neighbour is set to the next λ-sequence value without a fit and flagged `edge_unresolved`. Only cells that a row of the stage takes a value from can ask for an extension (stage a1: the A2/A3 cells; stage x1: the base-arm cells of the follow-up rows); other edge cells get the next λ-sequence value without a fit, flagged `edge_not_extended`.
- **Variant arms** inherit the matched λ of their base arm on the same task and decoder: discriminator_sn → discriminator; reference_sn, reference_fixed → reference; pooled_sn and pooled at other n_critic → pooled; barycenter_sn → barycenter; mmd_ref → mmd. Subsampled tasks (X3, X8) and architecture variants (X12) inherit the full task's value. An arm without a mapping stops the script.
- R4 extension points are used for matching only; frontier and P1 analyses use the six pre-registered points per family.

## 5. Power (reported by `freeze_a1_grid.py`; the design is fixed)

From A1, for the P1 contrast (reference critic − discriminator), per task × decoder: each arm at its R4-a1 matched λ, d_s = C_reference(s) − C_discriminator(s) per seed. σ̂ = pooled within-(task, decoder) SD of d_s; τ̂² = max(0, Var_t(mean d) − σ̂²/n) per decoder, pooled. For the fixed design (5 seeds, SI-16) with D = 6 real tasks (pancreas, lung, immune, immune_hum_mou, atac_small, atac_large; simulations are a separate family, PAPER_PLAN §5) and, conservatively, D = 4 families: SE = √((τ̂² + σ̂²/5)/D); power of a one-sided posterior-probability-0.95 call (non-central t, D − 1 df) for Δ = 0.02; the minimal detectable effect at 80 %; whether the 95 % interval at Δ = 0 fits inside ±0.01 and ±0.02 (equivalence). The design is flagged UNDERPOWERED when the minimal detectable effect exceeds 0.02 or equivalence within ±0.02 is unreachable.

Known before any data: the frequentist companion (exact Wilcoxon on dataset means, Holm over P1–P4) cannot reject with 6 real tasks. The smallest two-sided p is 2/2⁶ = 0.031, above Holm's first threshold 0.05/4 = 0.0125. PAPER_PLAN §5 relied on 9 real tasks; SI-02 removed the new datasets.

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

## 8. Limitations and open items

- The λ ranges come from 3 of 8 tasks; X1 tasks whose matched λ falls on a grid edge are extended by R4 (disclosed per task).
- A2/A3 are conditioned only; their decisions also govern the unconditioned block.
- Follow-up cells identical to an X1 cell (X6 discriminator at 1 step and pooled at 5 steps, seeds 0–2) reuse X1 fits on the seeds from which the matched λ was chosen, so their batch scores are biased towards b* by selection. Fresh follow-up seeds would remove the bias.
- Because X1 rows now hold the A2/A3 placeholders, the builder no longer recognises those 54 X6 rows as X1 fits; they are listed as fits until `freeze_matched_lambda.py --stage x1` removes them after resolution (recorded in `reuses_x1`). Giving follow-up adversarial rows the same placeholders (in `row()` or the follow-up blocks, outside this change) would restore the build-time count.
- R1 is frozen before R4-a1 adds any cell-specific A1 point; re-running `freeze_a1_grid.py` after an R4-a1 extension stops on the uneven grid by design.
- Extension rows (R1, R4) belong to a task and must run on that task's machine (SI-17).
- `score_scib_native.BIO_METRICS` omits scIB's trajectory conservation, which the scorer computes for immune and immune_hum_mou; scIB counts it as a bio metric. Adding it changes C on those two tasks; until then C follows `BIO_METRICS` as committed.
- The failures table (tag, status, detail) has to be written by the fit harvester; no runner writes it yet.
