# Pre-registered decision rules for the Tier 1+2 rerun

*Status, 2026-10-16 (ledger SI-78 to SI-89). Version 2 of the rules (tag `prereg-tier12-v2`, commit c88cce3, 2026-10-04) was signed off by the user on 2026-10-02 and revised with the user's answers of 2026-10-02/03 (§7). It was frozen before the restarted A1's first fit (§9), and every rule in it is a deterministic function of scored results. This file on branch `final-review-fixes` also holds a draft of version 3, the amendments of 2026-10-13 to 2026-10-15 in §9. Version 3 will be frozen by the tag `prereg-tier12-v3` (user decision SI-60) only after the user has reviewed the full diff (SI-52); that tag has not been made. The user answered the 19 questions of that review on 2026-10-15 (list at the end of §9) and the findings of the independent re-check on 2026-10-16 (SI-78, SI-81, SI-83, SI-85 and SI-89; amendment-table row 9). The sentence "nothing in this file is chosen after seeing A1, A2, A3 or X1 data" holds for version 2 only. The version 3 amendments were decided after A1, A2 and A3 had been fitted and scored and after the local X1 run had started and when part of the earlier X1 and X13 baseline fits already had scores (§9, entry on the disclosure); the table of amendments in §9 gives, for each one, when the user decided it, the commit, the data that existed at that time and whether any result or decision changed.*

Implementation: `scripts/prereg_rules.py` (all rule logic), `scripts/freeze_a1_grid.py` (R1 and the power report), `scripts/freeze_matched_lambda.py` (R4; stages `a1` and `x1`), `scripts/decide_a2_a3.py` (R2, R3). Tests: `tests/prereg/`. Version 3 adds `docs/ANALYSIS_SPEC.md` (the X1 analyses; the user approved the agent's proposals in it on 2026-10-15, SI-80), `scripts/analysis_x1.py` (their code) with its known-answer tests in `tests/analysis/`, `docs/EQUIVALENT_COMMITS.json` (the commits declared result-identical, SI-54), `experiments/doc_numbers_20261015/` (scripts that regenerate numbers quoted in these documents) and `experiments/x1_x13_score_hashes_20261015/` (the SHA-256 hash list of the existing X1 and X13 score files, SI-79). Governs `scripts/build_paper_manifest.py --design pilot` (design of record, CONSTRAINTS.md SI-16).

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

Fits that depend on no rule (X1 arms `none` and `scvi_adv`, X13) may run while A1–A3 are in progress; 250 of them already exist, and part of their scores (section 9, entry of 2026-10-14 on the disclosure). A row that still holds a placeholder cannot be fitted: `fit_paper_config.py` refuses a non-numeric `lam`, `int("A3")` fails for `zstd`, and the training plan refuses an `adv_input` other than `mean`/`sample`.

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

From A1, for the P1 contrast (reference critic − discriminator), per task × decoder: each arm at its R4-a1 matched λ, d_s = C_reference(s) − C_discriminator(s) per seed. σ̂ = pooled within-(task, decoder) SD of d_s; τ̂² = max(0, Var_t(mean d) − σ̂²/n) per decoder, pooled. For the fixed design (5 seeds, SI-16) with D = 6 real tasks (pancreas, lung, immune, immune_hum_mou, atac_small, atac_large; simulations are a separate family, PAPER_PLAN §5) and, conservatively, D = 4 dataset families: SE = √((τ̂² + σ̂²/5)/D); power of a one-sided posterior-probability-0.95 call (non-central t, D − 1 df) for Δ = 0.02; the minimal detectable effect at 80 %; whether the 95 % interval at Δ = 0 fits inside ±0.01 and ±0.02 (equivalence). The design is flagged UNDERPOWERED when the minimal detectable effect exceeds 0.02 or equivalence within ±0.02 is unreachable.

Known before any data: the frequentist companion (exact Wilcoxon on dataset means, Holm over P1–P4) cannot reject with 6 real tasks. The smallest two-sided p is 2/2⁶ = 0.031, above Holm's first threshold 0.05/4 = 0.0125. PAPER_PLAN §5 relied on 9 real tasks; SI-02 removed the new datasets. [Amended 2026-10-14: the frozen X1 analysis applies Holm over six primary results (P1, the frontier area and P2, each for both decoders), not four, so its first threshold is 0.05/6 = 0.0083. The six-test rule was the agent's proposal in docs/ANALYSIS_SPEC.md section 15, item 5; the user approved it on 2026-10-15 (SI-80). docs/prereg/r1_record.json still stores the four-test value, 0.0125 (N_ENDPOINTS = 4 in scripts/prereg_rules.py). The smallest two-sided p, 0.031, is above both thresholds, so the companion cannot reject at the Holm-adjusted level in either version; unadjusted, 0.031 is below 0.05.]

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

The amendments of 2026-10-13 to 2026-10-15 (version 3) follow a different order. Each was decided after A1, A2 and A3
had been fitted and scored and after the local X1 run had started on 2026-10-12, so "committed before the stage it
affects" holds for them only for the stages that have not yet run: the Sinkhorn redo, the X1 fits still to be made and
the single X1 analysis run. R2 and R3 had already been run once on the A2/A3 results (2026-10-12, NB-20261012-01)
before the rule for low-signal blocks (SI-56, SI-66) was decided; SI-57 has them run again after the Sinkhorn redo.
The table after the glossary lists every version 3 amendment with the data that existed when the user decided it.
Times are from the notebook and `git log`. Each entry dated 2026-10-13 to 2026-10-15 below starts with a status line
that says what is done, with date and commit, and what is planned; a step that has not been done is written with "will
be". The entries of 2026-10-03 and 2026-10-04 describe work that is done and frozen in version 2.

**Glossary.** Labels and terms that the entries below use.

| Label | What it is |
|---|---|
| SI-nn | Entry nn of the decision ledger (CONSTRAINTS.md). "User decision SI-nn" is the user's own answer to a question; entries marked as agent decisions are not the user's. |
| NB-yyyymmdd-nn | Entry nn of the lab notebook of that date (notebook/yyyy-mm-dd.md). |
| I0nn, for example I091 | Finding nn of the final pre-publication review of 2026-10-13. docs/review_final/REVIEW_FINAL.md tabulates the 17 it names; the full list (verified_issues.csv) is not in this repository. |
| D-nn, for example D-01 | Finding nn of that review's recheck of the design documents. |
| CR-nn, RR-nn | A finding of the code review of tag prereg-tier12-v1 (2026-10-03), or of the re-review of its fixes (2026-10-04). |
| F1 to F8, R2-01 to R2-09 | Notes of the first and the second independent review of the Sinkhorn fix (NB-20261013-13, NB-20261013-25). |
| G-01 to G-18 | Findings of the independent review of the guard commits (NB-20261013-13). experiments/guard_followups_20261013/README.md lists those that were addressed (G-01 to G-05 and G-14). |
| DOC-nn | Finding nn of the documents review of 2026-10-14 (NB-20261014-20). |
| 018c8f1 and similar | A commit of this repository (seven hexadecimal characters). |

| Term | What it is |
|---|---|
| version 2, version 3 | Version 2 is the frozen set of rules (tag prereg-tier12-v2, commit c88cce3). Version 3 is this file with the amendments of 2026-10-13 and 2026-10-14; its tag prereg-tier12-v3 has not been made. |
| arm | One method in the comparison. Plain scVI (single-cell variational inference, the base model) is arm none (penalty strength 0). The six adversarial arms are discriminator, reference, pooled, barycenter, mmd and sinkhorn. |
| scvi_adv | The anchor arm that uses scvi-tools' own adversarial training plan, the way people usually run an adversarial scVI (PAPER_PLAN section 3.2). It is not one of the six adversarial arms, and rule R2 does not govern it (section 3). |
| penalty strength, lambda | The weight of the batch-alignment term that an arm adds to the scVI loss. A grid is the list of strengths that are fitted; lambda = 0 is plain scVI. |
| discriminator, critic | The discriminator is a network trained to tell the batches apart (a Jensen-Shannon, JS, divergence). A critic is a network trained to estimate the Wasserstein-1 (W1) distance between batches: the reference critic compares each batch with a reference batch, the pooled critic with the other batches together, and the barycenter critic with a barycenter (an average distribution) of the batches. |
| critic-free loss | A loss computed directly from the cells' latent coordinates, with no trained network: MMD (maximum mean discrepancy), reference-anchored MMD, and Sinkhorn. |
| Sinkhorn, epsilon | The debiased Sinkhorn divergence, a regularised optimal-transport distance between batches, found by Sinkhorn iterations. Epsilon is its smoothing strength, lowered step by step; the final epsilon is the last value (src/wcd_vae/wcd/alignment.py). |
| transport plan, L1 marginal error, envelope theorem | The transport plan says how much of each cell's weight moves to each cell of the other batch; its row sums and column sums must equal the cells' weights. The L1 marginal error is the total absolute amount by which they miss; the solver runs until it is below 1e-4 (SINKHORN_TOL). By the envelope theorem, the cost of the converged plan does not change to first order when the plan changes, so the gradient can be taken with the plan held fixed. |
| gradient graph | The record of operations from which the optimiser computes gradients. A zero loss that lacks it cannot be differentiated and stops the fit. |
| no_pair_steps | The number of training steps whose minibatch gave a critic-free loss no pair of batches to compare. The fitter writes it into the latent's plan record only when it is above zero. |
| conditioned, unconditioned decoder | The conditioned decoder receives the batch label (scVI as usually used). The unconditioned decoder does not, so the adversary is the only batch-removal mechanism (PAPER_PLAN section 3.1). |
| task, dataset family | A task is one of the eight prepared datasets: pancreas, lung, immune, immune_hum_mou, atac_small and atac_large (the six real tasks), and sim1 and sim2 (simulations). A dataset family groups related real tasks (SI-73): pancreas; lung; immune (immune, immune_hum_mou); ATAC (chromatin-accessibility data: atac_small, atac_large). |
| penalty-strength family | The arms that share one penalty-strength grid (section 2 calls it a divergence family): JS {discriminator}; W1 {reference, pooled, barycenter}; MMD {mmd}; Sinkhorn {sinkhorn}. Each has one grid per decoder condition. |
| block, task-by-decoder block, cell | R1 calls a task x decoder x arm combination a block. R2 and R3 call it a cell (SI-27), whereas docs/ANALYSIS_SPEC.md uses cell for a task x decoder pair. In this section a task-by-decoder block is one task with one decoder condition and all its arms, and a cell of the tissue is called a single cell. |
| B, C | B is the batch-removal score, the mean of five scIB (single-cell integration benchmark) batch metrics. C is the biology-conservation score, the mean of the task's scIB biology metrics (section 1). Higher is better for both. |
| trajectory, kBET | Trajectory is the scIB biology metric for how well the order of cells along a developmental path (pseudotime) is kept; it is part of C on immune and immune_hum_mou. kBET is the k-nearest-neighbour batch-effect test, one of the five metrics in B. |
| b0, b_common, b\* | b0 is the batch score B of plain scVI. b_common is the highest B that every arm reaches. b\* = b0 + 0.5 (b_common - b0) is the target batch score at which arms are matched (section 4). |
| matched lambda | Among the grid points with no failed seed, the penalty strength at which an arm's seed-mean B is closest to b\*; ties go to the smaller strength (section 4, rule R4). |
| bstar_low_signal | A flag on a task-by-decoder block where b_common - b0 is below 0.02, so that b\* cannot be placed reliably (SI-56). |
| sigma_c, delta_c, delta_min | sigma_c is the between-seed standard deviation of the change in C. delta_c is the smallest change in C that rule R1 counts as an effect, max(delta_min, 3 sigma_c / sqrt(n)) with n the number of seeds; delta_min = 0.01 is the registered smallest difference that counts (section 2, SI-74). |
| R1 to R4 | The four decision rules (sections 2 to 4). R1 sets each penalty-strength family's six-point grid for X1 from A1. R2 decides whether the adversary reads the posterior mean or a posterior sample, R3 whether its input is standardised, and R4 sets the matched lambdas. PAPER_PLAN uses R0 to R5 for research questions instead. |
| A1, A2, A3 | The three pilot experiments, run before X1 to set the rules: A1 fits the penalty-strength grid, A2 tests the adversary input, A3 tests standardisation. |
| X1 | The main experiment, not a pilot: 8 tasks, both decoders, seeds 0 to 4, with the six adversarial arms and the arms none and scvi_adv (3,000 model fits in the manifest of record). |
| X13 | The comparison with standard integration tools: Harmony and Scanorama (the CPU baselines), sysVI (the SysVI model of scvi-tools) and scANVI (scVI trained with cell-type labels), with plain PCA (principal component analysis) as the uncorrected reference. It is not matched on penalty strength and keeps seeds 0 to 4 (section 4). |
| X3, X6, X7, X8, X12, X15 | Follow-up experiments that start after X1 (section 0, PAPER_PLAN section 4). |
| P1, P2, trade-off area | P1 is the difference in C between the reference critic and the discriminator at the target batch score b\*, one result per decoder. The trade-off area (also called frontier area) is the area between an arm's curve of C against B and the collapse line, counted only where the arm removes more batch effect than plain scVI (SI-72). P2 is how often runs collapse, that is, fail or lose 0.10 or more of C against the same seed's plain scVI run (SI-75). |
| Holm | Holm's step-down correction for testing several results: the smallest p-value is compared with 0.05 divided by the number of tests, the next smallest with 0.05 divided by one fewer, and so on. |
| power, SE, sigma, tau, D | In section 5, sigma is the between-seed standard deviation of the per-seed difference between two arms, tau is the standard deviation of the true difference between tasks, and SE is the standard error of the mean difference over tasks. Power is the probability that a call at 95 % posterior probability detects a difference of 0.02; the minimal detectable effect is the smallest difference found with 80 % power. D is the number of tasks (or dataset families) in section 5, and a difference in C at b\* in section 3. |
| transfer lambda, best-of-k | The transfer lambda is the single penalty strength that a method uses on datasets it was not tuned on (docs/ANALYSIS_SPEC.md section 8; the rule was the agent's proposal, approved by the user on 2026-10-15, SI-80). Best-of-k is the best score among k configurations; methods are compared at equal k (section 9 of that file). Equal numbers of tries is the user's decision (SI-68, SI-77); the best-of-k estimator and the composition of the method families were the agent's proposals, approved on 2026-10-15 (SI-80). |
| diff review, proposal | The diff review is the user's review of the full diff of branch final-review-fixes before the tag is made (SI-52). The user answered its 19 questions on 2026-10-15; the list is at the end of this section. A proposal is text that the agent suggested; it becomes a rule only when the user approves it. The user approved the agent's proposals in docs/ANALYSIS_SPEC.md (SI-80) and the three process proposals of this file (SI-81) on 2026-10-15. |
| agent, lead agent, sub-agent | An agent is an AI assistant that works on this project; the lead agent is the one that coordinates the others, and a sub-agent is one that another agent starts for a single task. The user is the person who decides; decisions marked as agent decisions in this file are not the user's. |

**Amendments of version 3.**

| # | Amendment | Decided by the user (2026-10-13 to 15, New York time) | Committed (New York time) | Data behind the decision | Result or decision changed? |
|---|---|---|---|---|---|
| 1 | R1 depends on the trajectory through its noise estimate (I091, SI-53) | 2026-10-13 11:46 | 018c8f1 (10-13 14:46) | A1 scores (bio-score noise estimates) | No registered rule changes. R1 without trajectory is a sensitivity analysis; the frozen X1 grids stand except Sinkhorn (row 4) |
| 2 | PAPER_PLAN corrected to this file: grid edge (SI-55), P1 comparator, power, Holm (I006, D-01, D-02) | Grid edge (SI-55): 2026-10-13 11:46. The other corrections are not user decisions | 018c8f1 (10-13 14:46) | A1 record (power figures) | No. PAPER_PLAN text only; Holm over six results was a later proposal (row 8), approved on 2026-10-15 (SI-80) |
| 3 | Two commits declared result-identical (SI-54; SI-60 names the tag) | SI-54: 2026-10-13 11:46. SI-60: 20:03 | 8ce5a54 (20:10), 7c32579 (20:30), b2c68c8 (20:30) | No X1 or X13 score value cited. The identity checks rescored 7 latents by script, one of them an X1 adversarial-arm latent and one an X13 sysVI latent, and printed two metrics of each (disclosure entry of 2026-10-14) | No |
| 4 | Sinkhorn arm corrected, its pilot fits redone in rule order (SI-51, SI-57, SI-58, SI-61, SI-62) | 2026-10-13: 07:45 (SI-51) to 20:48 (SI-61, SI-62) | 3cbaca4 (09:56), 89d5e2c (19:46); text 4d86a55 (10-14 07:54) | Pilot Sinkhorn fits (A1, A3); no X1 score cited | Yes. Sinkhorn A1, R1, R4-a1, A3 and R2/R3 results will be redone; the X1 Sinkhorn grid will change |
| 5 | Guards before the tag (I033, I034, I042, I050, I051, I094, I008; G-01 to G-05 and G-14; the NMI-0 case, SI-65; the check of divergence records in the stage runner, SI-85, widened on 2026-10-16 to every arm; the same check for a latent that names no commit, SI-89) | SI-52: 2026-10-13 07:45. SI-63, SI-64, SI-65: 20:48. SI-85: 2026-10-15 09:54, widened to every arm on 2026-10-16 (NB-20261016-09). SI-89: 2026-10-16 (NB-20261016-10) | c4be14b (10:09) to c3fd92d (21:00); SI-85: 2026-10-15, before the tag; SI-89: 4b02068 (2026-10-16), before the tag | No score cited | No completed fit or score changes: one-epoch test refits of code paths, not the production fits, give identical latents for 59 of 59 non-Sinkhorn code paths, and 7 of 7 rescored stored latents give identical scores (NB-20261013-39, -40). SI-85 and SI-89 add checks to the stage runner and change no fit or score |
| 6 | P1 starts at lambda = 0; low-signal blocks are not evaluable (I072, D-04, SI-56) | 2026-10-13 11:46 | Text 4d86a55 (10-14 07:54); code f3c3415 (R2/R3, 07:50) and ff88666 (P1, 08:04) | A1 and A2/A3 results; the R4-a1 record of 2026-10-10 already flagged two low-signal blocks | No. R2 and R3 keep their defaults with or without the rule |
| 7 | Low-signal blocks count as not evaluable in R2 and R3 (SI-66) | 2026-10-13 20:53 | f3c3415 (10-14 07:50); text 4d86a55 | A2/A3 results; R2 and R3 had been run once on them on 2026-10-12 (NB-20261012-01, cea50ca) | No. On the stored cells R2 has 9 of 12 cells evaluable (mean change -0.0042) and R3 13 of 20 (-0.0014); both keep their defaults |
| 8 | The X1 analyses are specified (I001, D-03, D-08, D-09; SI-68 to SI-77, plus the agent's proposals marked in ANALYSIS_SPEC, approved on 2026-10-15, SI-80) | SI-68 to SI-72: 2026-10-14 06:45. SI-73 to SI-77: 06:57. SI-78 to SI-82: 2026-10-15 09:37. SI-83, SI-84: 09:54 | 6d0ca1f (draft, 10-13 21:22), 56c0762 (06:47), 990d37d (06:58), a1d497b (07:42), ff88666 (08:04), de741a0 (08:48); revised after the independent review: 8b90d79 and 6aaa3ac (14:08), 56febe4 (14:12), 4af5069 (15:46); revised again on 2026-10-15 for SI-78, SI-80, SI-83 and SI-84, before the tag | A1 pilot ranges and collapse rates; 19 X1 rows had been scored on 2026-10-12, earlier X1 and X13 baseline scores existed, and two displays of values had occurred (X13 runner tests on 2026-10-02; two metrics of one X1 and one X13 rescore on 2026-10-13); the disclosure entry gives the facts, and the user lists the displays as shown to the agent only (SI-88). The question behind SI-75 quoted 8 of 45 pilot runs for the discriminator with the unconditioned decoder; the recount is 8 of 54 (NB-20261014-22) | New rules only; nothing already frozen changes. SI-75 stands after the P2 recount (user, 2026-10-15, NB-20261015-07): for the unconditioned decoder the discriminator collapsed in 8 of 54 pilot runs, not 8 of 45, so the quoted critic-minus-discriminator difference of +0.7 points is +3.7 points, both inside the 5-point margin (experiments/doc_numbers_20261015/p2_pilot_recount.py) |
| 9 | The user's answers of 2026-10-15 to the diff review that amend a rule or a procedure: 'examined' covers every X1 and X13 score (SI-78); a hash list of the existing score files (SI-79); the agent's proposals in ANALYSIS_SPEC are rules (SI-80); NMI-0 cases, iteration-cap hits and a reporting script after the runs (SI-81); scaling checks are exploratory (SI-82); a zero-spread dataset is kept with a floor (SI-83); five analysis designs (SI-84); the stage runner's check of divergence records (SI-85); a fallback in the power function (SI-86); the Sinkhorn redo overwrites its record files (SI-87); two displays of scores are listed as shown to the agent only (SI-88). The user's answers of 2026-10-16: SI-85 covers divergence records of every arm (NB-20261016-09); a latent that names no commit is refused (SI-89, NB-20261016-10); SI-81 (b) keeps its clause that each iteration-cap decision is listed with the results (NB-20261016-12); SI-81 (c) means the tag freezes the promise of the reporting script, not the script (NB-20261016-11); 'examined' means a person reads a value, and a value shown to an AI agent is disclosed, not examined (SI-78, NB-20261016-14); iteration-cap and NMI-0 cases stay decided case by case, with no default outcome (SI-81, NB-20261016-15); the number of datasets with a raised standard error is reported next to each pooled declaration (SI-83, NB-20261016-17) | SI-78 to SI-82: 2026-10-15 09:37. SI-83 to SI-87: 09:54. SI-88: 2026-10-15. Answers of 2026-10-16: NB-20261016-09 to -12, -14, -15 and -17 | 2026-10-15 and 2026-10-16, before the tag; identified by ledger id | The X1 and X13 scores that existed on 2026-10-15 (entry on the disclosure) | Yes: the freeze condition covers every X1 and X13 score (SI-78); a zero-spread dataset is kept with a floor, where the draft left it out (SI-83); the power function has a fallback where it stopped with an error (SI-86); the redo overwrites its record files (SI-87). No completed fit, score or decision of R1 to R4 changes |

- **2026-10-03, sensitivity analysis without trajectory (user decision, SI-38).** Every X1 comparison on immune and
  immune_hum_mou is also reported with the bio score C recomputed without trajectory conservation (C over the
  remaining 7 bio metrics). The primary analysis is unchanged (C with trajectory, as scIB; SI-28). Reason: on the
  64 gate latents (immune, 3 epochs) trajectory had SD 0.121 for barycenter and 0.057 for the discriminator versus
  <= 0.010 for the other arms, and one latent scored 0.50 locally and 0.87 / 0.89 on two other GPUs
  (docs/jhpce/gate_data_v2/g3/scores_192_latents_wcd-kbet.csv). In scib 1.1.7 (metrics/trajectory.py) pseudotime is
  computed on the largest connected component of the kNN graph only (other cells get 0) and the root cell is chosen by
  a vote over diffusion components, so small latent changes can change the value. A1's rule R1 is unaffected: one
  metric of 8 moving 0.39 in one seed shifts the 3-seed mean C by about 0.016, against the collapse bound 0.10. [Corrected 2026-10-13: R1 does depend on trajectory, through its
  noise estimate; see the entry of 2026-10-13 on R1 and the trajectory score below (finding I091).]

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
- **2026-10-13, correction: R1 depends on trajectory through its noise estimate (final review finding I091: R1 was
  wrongly said to be unaffected by the trajectory score; user decision SI-53: keep R1 as registered and disclose
  this).**
  Status: text committed 2026-10-13 (018c8f1), with the trajectory-free R1 script
  (experiments/final_review_amendments_20261013/r1_without_trajectory.py); R1 for the Sinkhorn family is planned,
  after the Sinkhorn redo.
  The 2026-10-03 sentence "A1's rule R1 is unaffected" is wrong. R1, the rule that sets each penalty-strength family's
  grid from A1, uses the biology score C twice: in the collapse bound (-0.10), which one unstable metric barely moves,
  and in sigma_c, the between-seed standard deviation of the change in C, which sets delta_c, the smallest change in C
  that counts as an effect (section 2; delta_c = max(0.01, 3 sigma_c / sqrt(n)), n the number of seeds). On immune
  with the conditioned decoder the trajectory metric (how well the order of cells along a developmental path is kept)
  inflates sigma_c 3.3 to 5.4 times (with / without trajectory: barycenter 0.073 / 0.014, pooled 0.060 / 0.017,
  reference 0.061 / 0.014, discriminator 0.064 / 0.017, MMD 0.060 / 0.018, Sinkhorn 0.059 / 0.013, the last from the
  pre-fix Sinkhorn fits), so delta_c reaches 0.10 to 0.13 and the no-effect test on bio cannot fail there. R1 re-run
  on the A1 fits without trajectory changes three conditioned-decoder penalty-strength families (W1 critics, MMD,
  Sinkhorn) from frozen to an extension at the low edge (lambda 0.03, 0.01); the other five penalty-strength families
  are unchanged (experiments/final_review_amendments_20261013/r1_without_trajectory.py, which first reproduces the
  record). R1 stays as registered (SI-53). The frozen X1 grids stand for every arm except Sinkhorn: the Sinkhorn grid
  will be re-derived by R1 on the refitted Sinkhorn fits (SI-57; entry on the Sinkhorn arm below), so
  docs/prereg/r1_record.json, which still holds the superseded Sinkhorn grid, will be superseded by the redone record.
  The Sinkhorn figures in this entry (sigma_c 0.059 / 0.013, and the move of the Sinkhorn family to an extension) are
  pre-fix: they come from the discarded Sinkhorn fits. The trajectory-free R1 will be reported as a sensitivity
  analysis, its Sinkhorn part recomputed on the refitted Sinkhorn fits.
- **2026-10-13, PAPER_PLAN corrected to this file (final review finding I006, the X1 grid-edge rule, and
  design-recheck findings D-01, the P1 comparator, and D-02, the power claim; user decision SI-55 for part (a)).**
  Status: done (2026-10-13, 018c8f1); the power estimate of part (c) is planned to be recomputed after the Sinkhorn
  redo.
  Part (a) is the user's decision SI-55; parts (b) to (d) correct PAPER_PLAN to the registered text of this file (and,
  for (b), to the user's earlier decision SI-01); they are not separate user decisions. (a) Grid edge: PAPER_PLAN
  promised that any X1 optimum at a grid edge triggers an extension before analysis; as section 4 states, R4 extends
  grids for matching only, frontier and P1 use the six registered lambda points, and an optimum at a grid edge is
  disclosed, not extended. (b) P1 comparator: the discriminator at 1 adversary step (SI-01; n_critic, the number of adversary updates per training step, is 1 in every discriminator row of the manifest), not "x10 steps"; the update-budget control is X6's pooled critic at 1, 5 and 10 critic
  steps, so it exists only for the pooled W1 critic. (c) Power: section 5 flags the frozen design UNDERPOWERED
  (docs/prereg/r1_record.json, power). Its figures are those of the matched-lambda contrast of section 5 (reference
  critic minus discriminator, each at its R4-a1 matched lambda), estimated from all six A1 pilot blocks (3 tasks x 2
  decoders), which include the simulation sim1 and the two blocks flagged low-signal (entry on low-signal blocks
  below); they are not the power of the frozen P1, which reads biology at the target batch score b\* by interpolation
  (docs/ANALYSIS_SPEC.md section 2.2). For D = 6 real tasks (D is the number of tasks or dataset families, section 5): SE (standard error of the mean difference over tasks) 0.045, minimal detectable effect at 80 % power 0.131,
  power at Delta = 0.02 0.104; D = 4 dataset families 0.182 and 0.088; equivalence within ±0.01 or ±0.02 unreachable; sigma (between-seed standard deviation of the per-seed difference) 0.030 (12 degrees of freedom), tau (standard deviation of the true difference between tasks) 0.109 (4 degrees of freedom); A1 per-block means of opposite sign, and in the
  conditioned decoder sim1 supplies 67 % of the squared spread of the three block means. Without sim1, the same
  formulas on the four remaining blocks give sigma 0.033 (8 degrees of freedom), tau 0.056 (2 degrees of freedom), SE
  0.0236, minimal detectable effect 0.068 and power 0.182 for D = 6 (0.095 and 0.137 for D = 4); equivalence is still
  unreachable (experiments/final_review_amendments_20261013/power_without_sim1.py). The conclusion, underpowered,
  holds either way. P1 is therefore reported as the pooled posterior together with tau, the per-task posteriors and a
  per-task sign table; PAPER_PLAN's equivalence claim is dropped, and the power estimate will be recomputed with the
  frozen P1 reading and only the evaluable blocks when R4-a1 is re-run after the Sinkhorn redo (SI-57). (d) The
  frequentist companion cannot reject with 6 real tasks: the smallest attainable two-sided p, 2/2⁶ = 0.031, is above
  Holm's first threshold, which is 0.0125 for the four tests of section 5 and 0.0083 for the six primary results of
  the frozen X1 analysis (bracketed note in section 5); it cannot reject at the Holm-adjusted level in either version.
  PAPER_PLAN's 9-task statement is replaced.
- **2026-10-13, two commits declared result-identical (user decision SI-54: treat fits and scores from the two commits
  as one set for every arm except Sinkhorn; user decision SI-60: name the new tag prereg-tier12-v3).**
  Status: done (2026-10-13): the equivalence list and the check that reads it (8ce5a54), their tests (7c32579) and the
  identity evidence at c3fd92d (NB-20261013-39, -40). Planned: the tag, the Sinkhorn refits, and a repeat of the
  identity checks if a later commit changes code that a fit or a score runs.
  This amends the 2026-10-04 entry
  (b): rule inputs and stages may now combine rows from two commits. docs/EQUIVALENT_COMMITS.json declares that fits and
  scores made at c88cce3 (tag prereg-tier12-v2) are result-identical to those made at the planned tag prereg-tier12-v3 for
  every arm except sinkhorn. The evidence is a pair of identity checks made at commit c3fd92d on 2026-10-13
  (NB-20261013-39 for fits, NB-20261013-40 for scores; files fit_identity_c3fd92d.csv, score_identity_c3fd92d.csv and
  run_c3fd92d.sh in experiments/integration_checks_20261013/results). Fits: one row for each of 63 code paths of the
  committed manifests was refitted for one epoch on CPU with one thread at both commits; the latents of 59 of 59
  non-Sinkhorn rows are identical, and the 4 Sinkhorn rows differ, as intended (SI-51). Scores: 7 stored latents
  (plain scVI, a pooled critic on immune, a simulation, a standardised fit, the barycenter critic, one X1 fit and one
  X13 sysVI row) were rescored by both versions of the scorer and are identical in all 32 compared columns.
  The X1 fit is one of the 19 early X1 rows and the X13 row is sysVI on pancreas; the check also printed two metrics
  of each of these two rescores (section 9, entry of 2026-10-14 on the disclosure). An earlier
  run at 2b52542 (fit_identity.csv, score_identity.csv) came before the guard follow-ups G-01 to G-05 and G-14 changed fit code and
  is kept as history. The evidence is narrow: one fit per code path, one epoch, one latent per scorer path. It holds
  for a later candidate commit only if that commit changes no code that a fit or a score runs. Before the tag is made,
  `git diff c3fd92d <candidate commit>` over src/, scripts/fit_paper_config.py, scripts/scvi_adversarial_plan.py,
  scripts/score_scib_native.py and the files they import will be checked for changes; if there is one, the identity
  checks will be re-run at the candidate commit (run_c3fd92d.sh; SI-52). prereg_rules.py (scorer_git_sha) and
  run_stage.py (scorer_git_sha, and the git_sha of latents and of divergence records, rigour review CF-02) read c88cce3 as the tag's commit for those arms when rows mix
  commits, and still require one commit after that. Two limits follow from this design. First, score rows carry the
  scorer's commit, not the fit's, so a rule input cannot show a Sinkhorn latent that was fitted at c88cce3 and
  rescored at the tag; the stage runner's start-up check catches it through the commit that fitted the latent. Second,
  "Sinkhorn rows must carry the new tag" is enforced as "no Sinkhorn row at c88cce3, and one commit per input once
  c88cce3 is read as the tag": a stage made entirely at one other commit is not refused by this list, so every stage
  is fitted and scored from one worktree pinned at the tag (2026-10-04 entry (b)). Sinkhorn output made at c88cce3 (a score, a latent or a
  divergence record) is refused, by different checks.
  The rule scripts see scores only, so prereg_rules refuses only a Sinkhorn score row made at c88cce3. The commit of a
  divergence record is checked by the stage runner's start-up check (the run_stage preflight) only: it reads the
  commit of every latent and of every divergence record, refuses the stage in report-only runs too, and, by the user's
  decision SI-85 of 2026-10-15, also refuses a divergence record of any arm (user, 2026-10-16) that carries no commit or was made from a checkout
  with uncommitted changes (a nonfinite_latent record that the runner writes itself stays accepted), and, by SI-89 (2026-10-16), a latent that names no commit. Sinkhorn rows will be refitted and rescored at the tag (SI-57); no other row will be rescored. If the tag is absent from
  a checkout, the mix stays and is refused, so a JHPCE bundle that combines rows from both commits must carry the tag
  (lesson RR-01).
- **2026-10-13, Sinkhorn arm corrected; its pilot fits will be redone in rule order before X1 (final review findings
  I031, the gradient counted twice, and I032, the solve stopped before it converged; user decisions SI-51, fix the arm
  and redo all 234 pilot fits, SI-57, redo in rule order, SI-58 and SI-62, solver precision, and SI-61, the iteration
  cap).**
  Status: the code fix is done (2026-10-13, 3cbaca4 and 89d5e2c) and so is the solver validation (NB-20261013-03,
  -19). The redo described below is planned; neither the notebook nor the repository records a start (2026-10-14).
  The tagged Sinkhorn loss counted the gradient through its final
  transform twice (on the validation minibatches below, the tagged gradient had a median norm 2.00 times that of the
  converged gradient on A1 fits and 2.27 times on A3 fits; on the R1 extension rows 14.4 times, with median cosine
  0.28, over the 204 of 216 minibatches where the tagged code gave a finite gradient) and stopped the solve before it
  converged at the final epsilon. Since commit 3cbaca4 the gradient is taken once, with the transport plan held fixed (envelope theorem), and every transport plan at the final epsilon is iterated until its L1 marginal error, the total absolute amount by which its row or column sums miss the weights they must match, is below 1e-4 (SINKHORN_TOL), in float64 (double precision) with
  direct-difference distances. Limits: at most 500,000 iterations at the final epsilon, and reaching the cap raises an
  error that stops the stage (SI-61); a NaN marginal error ends the solve at once, a +infinity error does not, because
  finite latents of about 1e18 or more give +infinity and then converge a few iterations later (re-review R2-01,
  commit 89d5e2c). A fit that reaches the cap is neither scored nor recorded as failed: the stage stops (SI-61), and a
  rerun with the same seed and code can be expected to reach the cap again. Section 1 treats infrastructure errors as
  rerunnable and requires every row to end as scored or failed, so a persistent cap hit had no registered outcome.
  The user decided on 2026-10-15 (SI-81 (b)): the stage stays stopped until the user decides whether such a fit is
  recorded as a failed fit; a fit recorded as failed counts as a collapse in P2, and the decision is listed with the
  results. The solve stays in float64 (SI-58,
  SI-62): float32 met the agreement criterion (worst gradient cosine at least 0.99999, relative value error at most
  1e-4) on the 598 hardest validation minibatches, but was no faster at 128-cell minibatches and failed the value
  criterion on all 3,076. The 180 A1 Sinkhorn fits will be refitted and rescored first, at the tag prereg-tier12-v3.
  Then, in rule order (SI-57): R1 for the Sinkhorn family, with the extension rows it asks for; R4 stage a1 over all
  six arms with the refitted Sinkhorn arm; the A3 rows it then resolves (the Sinkhorn rows, and any non-Sinkhorn cell
  whose window moved); R2 and R3 in full. The extension rows and the A3 Sinkhorn rows are therefore outputs of R1 and
  R4-a1 on the new fits, not a fixed list: before the fix there were 18 and 36 of them, 234 rows with the 180 A1 fits
  (the count named in SI-51), and the numbers may change (NB-20261013-09). No existing non-Sinkhorn fit is redone
  (SI-54 entry above); non-Sinkhorn A3 rows are fitted anew only where a window moves. The power estimate of section 5
  will be recomputed with R4-a1. The solver was validated on 3,076 minibatches in all (review note F6;
  experiments/doc_numbers_20261015/validation_minibatch_counts.py recomputes the counts from
  experiments/sinkhorn_fix_20261013/results/validation_12mb.csv): 2,808 final-epoch minibatches from the 234 pre-fix
  fits of atac_small, immune and sim1 (3, 10 and 6 batches; 12 from each fit), 76 training-mode minibatches from 19 of
  those fits, and 192 minibatches captured earlier in training in separate replays (160 replay tags, none of them
  among the 234 fits). The check will be repeated on the refitted latents before convergence is claimed for them or
  for the X1 tasks with more batches (up to 23, immune_hum_mou).
  The Sinkhorn redo overwrites docs/prereg/r1_record.json, r1_record_round1.json, r4_a1_record.json, r4_a1_record_round1.json and r2r3_record.json in place; the versions before the redo stay at tag prereg-tier12-v3, which is made before the redo.
  The five files were last changed by the commits 2f17f77 (r1_record.json), 03882dc (r1_record_round1.json), dae54b0
  (r4_a1_record.json), 55e66d4 (r4_a1_record_round1.json) and cea50ca (r2r3_record.json), so those commits also hold
  the versions before the redo. The tag prereg-tier12-v2 was made before the restarted A1 pilot ran and does not hold
  these files. The redo's notebook entry will list every decision for the other five arms that changed. The six
  non-Sinkhorn grids of the redone R1 record are to be checked by hand against the version at tag prereg-tier12-v3
  (the analysis script reads the grid from the record it is given and does not compare grids), and
  docs/ANALYSIS_SPEC.md will name the redone R1 record as its source of the six registered lambda per arm.
- **2026-10-13, guards before the tag (final review findings I033, I034, I042, I050, I051, I094 and I008, each
  explained in the list below; user decisions SI-52, SI-63, SI-64; follow-ups to the independent review of the guard
  commits, findings G-01 to G-05 and G-14).**
  Status: done (2026-10-13, c4be14b to c3fd92d; identity checks at c3fd92d, NB-20261013-39, -40). A guard is a
  stricter check or an extra record in the code. None of them changes a completed fit or score: at commit c3fd92d the
  one-epoch test refits of 59 of 59 non-Sinkhorn code paths give the same latents as at c88cce3 (the 4 Sinkhorn code
  paths differ, as intended), and 7 of 7 stored latents give identical scores in all 32 compared columns
  (experiments/integration_checks_20261013; NB-20261013-39 for the fits, NB-20261013-40 for the scores).
  - Minibatches with nothing to compare (I033, commit c4be14b; follow-up G-01, commit 96f4258). MMD,
    reference-anchored MMD and Sinkhorn are critic-free losses: they compare the batches present in a minibatch of
    cells directly, with no trained network. When a minibatch holds only one batch (for reference-anchored MMD: no
    reference cell, or no other batch), there is nothing to compare. The loss is then a zero that keeps the gradient
    graph, instead of stopping the fit. The fitter counts such minibatches and writes the count, no_pair_steps, into
    the latent's plan record when it is above zero. A model-based estimate (a calculation, not simulated runs; first
    made in the guard review, recomputed by experiments/doc_numbers_20261015/no_pair_expectations.py from the manifest
    of record and the prepared data, with the scvi-tools assumptions checked by no_pair_assumptions.py in the same
    folder) expects them in X7 reference-anchored MMD on pancreas (27 fits, 25.9 expected to meet at least one) and on
    sim1 (all 18 fits), and in about 14% of the 48 X3 atac_small MMD fits, 4.5% of the 24 X8 sim2 MMD fits and 0.6% of
    the 60 X1 sim1 fits of each of MMD and Sinkhorn. The counts, by arm, task and experiment from the latents' plan
    records, and, per arm, the number of cells that are not evaluable because the arm never reaches the target batch
    score b* (docs/ANALYSIS_SPEC.md) are tabulated by a reporting script that is written after the runs (user decision
    SI-81 (c), 2026-10-15). No code reads no_pair_steps yet; the script affects no rule, so it can follow the tag: the tag freezes this promise, not the script (SI-81 (c), user, 2026-10-16, NB-20261016-11).
  - Failures inside the critic's training loop (I034, commit a4f4be0). The critic can be updated more than once inside
    each training step. A loss that is not finite (not a number, NaN, or infinite) at any of these inner steps, not
    only the last, now marks the fit as diverged.
  - Settings that an arm does not read (I042, commit 5f30f17; SI-63). A fit refuses manifest settings and extra keys
    that its arm does not read, so that a setting cannot be changed without any effect and without notice.
  - Checks in the scorer (I050 and I051, commit f5ea0c3; SI-64, commit c3fd92d). The scorer refuses a latent whose
    cell names or labels are not those of the prepared data file in the same order, and the two pilot writers whose
    latents carry no cell names are retired to legacy/ (SI-64). A kBET result of exactly 0 with no seeded call to R
    (the statistical language in which kBET is implemented) is recorded, because scib then sets the rejection of every
    tested label to 1; any other result without a call to R is refused.
  - A normalised mutual information of 0 (SI-65). Where scib 1.1.7 finds NMI, the normalised mutual information, equal
    to 0 at every clustering resolution, it keeps no clustering, and the scorer keeps scib's loud error; the case is
    then decided case by case (SI-65, 2026-10-13). The user decided on 2026-10-15 (SI-81 (a)) that he decides each such
    case and that every case is listed with the results.
  - Complete fits only (I094, commit 46d79f2). A fit counts as complete, is resumed and is harvested from the cluster
    only if its model weights (graphics processing unit, GPU, fits) and training history are present.
  - Job time limits (I008, commit 2b52542). Each job in a chain of cluster jobs gets a time limit of (chain position +
    1) x (walltime + queue allowance), so a later job is not killed because the jobs before it ran long.
- **2026-10-13, P1 starts at lambda = 0; low-signal cells are not evaluable (final review finding I072 and
  design-recheck finding D-04, on how P1 reads the target batch score b\* when it lies outside an arm's grid; user
  decisions SI-56, start at lambda = 0 and treat low-signal blocks as not evaluable, and SI-66, count them as not
  evaluable in R2 and R3).**
  Status: decided 2026-10-13; rule code done 2026-10-14 (f3c3415 for R2 and R3, ff88666 for P1); the R2 and R3 re-runs
  are planned, after the Sinkhorn redo (SI-57).
  Every arm's bio-versus-batch curve starts at the lambda = 0 point, the plain-scVI scores that
  define b0, so a curve whose lowest registered point already lies beyond b\* is read between lambda = 0 and that
  point. A task x decoder flagged bstar_low_signal (b0 is plain scVI's batch score and b_common the highest batch score that every arm reaches; the flag is set when b_common - b0 < 2 delta_min = 0.02, section 4) has too small a
  batch range for b\* to be read reliably: its cells are not evaluable for P1 or for R2/R3, and their readings are
  reported separately, never pooled. In R2/R3 such cells stay in n, the number of cells the rule counts (the user chose this over leaving them out; SI-66),
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
  the runner ended on 2026-10-11); only the Sinkhorn A3 rows (36 before the fix) are to be redone (SI-57). SI-57 has
  them refitted and R2 and R3 re-run. Neither can change what X1 uses while the flag holds, and the user decided on
  2026-10-15 to keep every step of SI-57 in rule order all the same (NB-20261015-08).
- **2026-10-14, disclosure: part of X1 and X13 was fitted and scored before the X1 analyses were specified (finding
  DOC-01 of the documents review of 2026-10-14; widened on 2026-10-15 by the user's decisions SI-78, SI-79 and SI-88
  and by findings DOC-R01, DOC-R02 and DOC-R14 of the documents re-check).**
  Status: written 2026-10-14, revised 2026-10-15 after the user's decisions SI-78, SI-79 and SI-88 and the audit of
  2026-10-15. Planned: the time stamp from GitHub for the tag, after the tag is made (SI-79, below).
  The local part of X1 was launched on 2026-10-12 (NB-20261012-04, 12:40 New York time; the user chose to run X1
  locally) after R1, R2 and R3 were frozen, which is the registered order (section 0), and was stopped the same day to
  wait for the final review (NB-20261012-05, 16:36). At that point 19 X1 rows were scored, 1 was fitted but not scored
  and 1,097 still needed a fit. The 19 scored rows are kept as X1 rows: they were fitted at the tag checkout c88cce3
  (NB-20261012-04), which SI-54 declares result-identical to the tag prereg-tier12-v3 for every arm except Sinkhorn,
  and none of the 19 is a Sinkhorn row. Counted from the names of the score files, not from their contents
  (NB-20261014-21), the 19 scored rows are 17 adversarial-arm rows, all on immune with the conditioned decoder and
  seed 0 (discriminator at the first to sixth registered penalty strength, pooled critic at the first to fifth,
  reference critic at the first to sixth), and 2 rows of the arms none and scvi_adv.
  Fits that depend on no rule (the X1 arms none and scvi_adv, X13; section 0 lets them run while A1 to A3 are in
  progress) also exist, and their scores are partly made. The project's compute cluster (called JHPCE in the notebook)
  fitted 250 of them at tag prereg-tier12-v2: the X1 arms none and scvi_adv and the X13 baselines scANVI and sysVI, on
  atac_large and immune_hum_mou (pass A, 100 fits) and on lung, pancreas and sim2 (pass B, 150 fits). Both jobs were
  complete on 2026-10-06 (NB-20261006-01); the counts are those of the manifest of record
  (experiments/doc_numbers_20261015/filler_fit_counts.py). Their latents are scored on this machine. Pass B was scored
  in full, 150 of 150, by 2026-10-07; its entry says that the check was of file counts, finiteness and provenance and
  that the metric values were not inspected (NB-20261007-01). Of pass A, 1 of 100 was scored by 2026-10-07
  (NB-20261006-04, NB-20261007-01) and 99 were still to score when it resumed on 2026-10-12 (NB-20261012-02); it was
  paused the same day for the final review (NB-20261012-05), and the notebook records no later progress.
  The SHA-256 hash of every existing X1 and X13 score file is committed in experiments/x1_x13_score_hashes_20261015/
  (SI-79). Hashing reads a file's bytes and shows no value, so a later reader can use the list to check that no file
  has changed since the list was made. The README of that folder groups the files and is the inventory of what exists.
  The groups, found from file names, folder names and dates only, are:
  - (1) the 19 early X1 rows above (README group X1_local);
  - (2) the scores of the earlier fits above (group fillers);
  - (3) the rescoring by the scorer identity checks of 2026-10-13: 7 stored latents, among them one of the 19 early X1
    rows (tag X1_immune_discriminator_lg1_c1_s0_122afe99) and one X13 row (sysVI on pancreas, tag
    X13_pancreas_sysvi_l10_c1_s0_60b085ea); the old and the new scores were compared with each other and with the
    stored row by script, column by column (NB-20261013-12, -16 and -40; group identity_check);
  - (4) two test runs of the X13 baseline runner on atac_small on 2026-10-02, before the first tag prereg-tier12-v1,
    each scoring 6 latents of Harmony, Scanorama and PCA (NB-20261002-09 and -10; group x13_cpu_test); they are not registered rows;
  - (5) the probe outputs of the final pre-publication review of 2026-10-12 and 13: its probes scored 49 registered
    configurations, each with the task, arm, penalty strength, decoder and seed of a registered X1 or X13 row, 42 from
    fits that the reviewers made themselves on sim1, atac_small and pancreas (adversarial arms included), at the code
    of 2026-10-12 and with training settings not yet checked, and 7 that rescore 6 real X1 baseline latents of arm
    none. These are not the registered production fits (group review_probes; the README's sixth group, sealed, holds the 2 non-score files of review_sealed).
  The date that GitHub shows for the tag prereg-tier12-v3 will be recorded in the notebook after the tag is made, as a
  time stamp from outside this repository (SI-79).
  **Definition of "examined" (user decision SI-78, 2026-10-15, amended 2026-10-16).**
  A score is examined when a person reads its value or an analysis script uses it; checks that print only counts or flags do not count. This applies to every X1 and X13 score, of any arm.
  Only a person reading a value counts: a value shown to an AI agent is disclosed, not examined (user, 2026-10-16, NB-20261016-14).
  The listing of the two displays below as "shown to the agent only" (SI-88) follows from this definition.
  For example, a check that prints how many required metrics are finite does not examine a score; a person who opens a
  score file and reads a batch score does, and so does a script that averages scores. "Before any X1 or X13 score is
  examined" is read in this sense in the entry below, in docs/ANALYSIS_SPEC.md and in PAPER_PLAN section 5.
  The audit of 2026-10-14 (NB-20261014-21), made by the agent that ran the checks, searched the logs of the agent's
  working sessions on this project for reads of tier12_v2/X1_local/scores (12 messages) and for the printed output of
  the scripts that read them, and checked two log statements (run_stage.py and score_scib_native.py, which log only
  tags and timings). Three automated integrity checks read the 19 rows (scorer provenance and versions; finite
  required metrics; counts of kBET zeros and R calls) and printed only counts pooled over all stages. That audit
  searched one folder, and NB-20261015-23 corrects its conclusion in the light of the audit of 2026-10-15 below.
  On 2026-10-15 the user answered the question whether anyone, the agent or anyone else, had read the value of an X1
  or X13 score: no, nobody read a value (NB-20261015-15). The audit made later that day (NB-20261015-22; its method
  and results are in the README of experiments/x1_x13_score_hashes_20261015/) found two displays of values:
  - On 2026-10-02, before the first tag prereg-tier12-v1, the output of a sub-agent's tool call showed all metrics of
    six test fits (Harmony d10 and d50, Scanorama d10 and d100, PCA d10 and d50) from one of the two X13 runner-test
    runs on atac_small (test configurations, which are not registered rows). NB-20261002-10 and the notebook page of 2026-10-02 also record run-to-run differences of NMI,
    ARI and iLISI and a kBET value from these tests.
  - On 2026-10-13, before 17:13, the scorer identity check printed in the lead agent's session the old and new values
    of two metrics, PCR_batch (the principal-component-regression batch score) and cell-cycle conservation, for one X1
    adversarial-arm run (immune, discriminator, first registered penalty strength, conditioned decoder, seed 0; one of
    the 19 early local X1 rows) and for one X13 run (pancreas, sysVI). The first commit of the X1 analysis
    specification was at 21:22 that day, and the user decided its first rules on 2026-10-14.
  The audit of 2026-10-15 found no value from the 49 probe score files of group (5) in 56 session transcripts or
  26,722 review output files. Its limits: it is a pattern search of printed output; a value printed without a file,
  folder or metric name nearby can be missed; the sessions of Codex, a second AI assistant that took part in the
  reviews, left only output files; and what an agent read without printing cannot be known.
  The user decided on 2026-10-15 (SI-88, NB-20261015-25) that these two displays are listed as shown to the agent only
  and do not count as examined under SI-78. This entry gives their facts, dates and limits in full.
- **2026-10-14, the X1 analyses are specified and will be frozen before any X1 or X13 score is examined
  (final review finding I001, that the P1 to P4 analysis code had to be frozen before X1 scores are read;
  design-recheck findings D-03, D-08 and D-09; user decisions SI-68 to SI-77, and SI-80, which approved the agent's proposals
  named below).**
  Status: specification and code first committed 2026-10-14 (56c0762, 990d37d, a1d497b, ff88666; three test gaps
  closed in 858101b) and revised the same day after the independent review of 2026-10-14 (code and tests 8b90d79,
  numerical checks 6aaa3ac, specification 56febe4 and 4af5069). On 2026-10-15 the specification, the code and the
  tests were revised again, after the independent re-check (NB-20261015-01) and the user's answers (SI-78, SI-80,
  SI-83, SI-84). The tag prereg-tier12-v3, not any commit named here, fixes the registered version.
  Planned: the freeze by the tag, and the single analysis run after every X1 and X13 row is scored or failed.
  docs/ANALYSIS_SPEC.md specifies, and scripts/analysis_x1.py implements with known-answer tests (tests/analysis/),
  every analysis of X1 and X13 data: P1 (biology kept at equal batch removal) per decoder with the frontier (trade-off) area as co-primary, P2 (how often runs collapse, and the critic-minus-discriminator comparison), local versus global batch removal, single-lambda tables (one penalty strength per method, chosen on the other datasets: the transfer lambda), best-of-k placement (best score among k tries) against standard integrators, the width of each method's range of good penalty strengths (the lambda window) and the sensitivity analyses. The
  user's decisions are SI-68 to SI-77. Details that no question covered were the agent's proposals, marked as such in
  docs/ANALYSIS_SPEC.md; the user approved them as rules on 2026-10-15 (SI-80): section 2.3 (the not-evaluable rule),
  section 5 (marginal declarations; Holm over the six primary results, for the descriptive Wilcoxon companion only),
  section 8 (the leave-one-family-out rule for the transfer lambda; SI-77 decides only the selection score), section 9
  (best-of-k placement against standard tools; equal numbers of tries is the user's decision, SI-68 and SI-77, and the
  best-of-k estimator and the composition of the method families were the agent's proposals), section 10 (the width of
  the range of good strengths), the sensitivity analyses of section 11 that no decision names, and the implementation
  details of section 15. The parts of the specification that carry out designs from the rigour review, asked
  separately at the same review, follow the user's own answers: SI-83 (a dataset whose leave-one-seed-out standard
  error is below 1e-6 is kept, with its standard error raised to 1e-6) and SI-84 (the five analysis designs). The tag
  prereg-tier12-v3 will freeze all of them. No X1 or X13 score is to be examined (as defined in the
  disclosure above) before the tag is made, and the script is then run once, after every X1 and X13 row is scored or
  failed (a procedure: the script refuses to overwrite an existing record). The follow-up analyses (X3, X6, X7, X8,
  X12, X15) are specified later, with the X7/X8 redesign, while X1 runs and before any X1 result is examined (SI-68,
  whose words are "before anyone looks at an X1 result", any arm; the user confirmed on 2026-10-15 that these words
  stay, NB-20261015-09); the premise check R0 and X10 are exploratory. This entry does not change R1-R4. Where PAPER_PLAN
  section 5 differs, the specification applies (its section 13 lists the replaced statements).

- **2026-10-15, the user's answers to the 19 questions of the diff review (documents re-check finding DOC-R13).**
  Status: all 19 questions were answered on 2026-10-15 (New York time). This list gives each question in plain words,
  the answer, the record of the answer, and the file where the matter is explained. A record is a ledger entry (SI-nn,
  CONSTRAINTS.md) and its notebook entry, or a notebook entry only (NB-yyyymmdd-nn, notebook/2026-10-15.md); the
  user's own words are in the sources quoted there. Questions 15 to 18 concern code housekeeping and change no rule of
  this file.

| Q | Question | Answer | Record | Where explained |
|-----|----------------------|------------------------------------|-------------|------------------------|
| 1 | What should happen to a dataset whose run-to-run uncertainty comes out as exactly zero? | A dataset whose leave-one-seed-out standard error is below 0.000001 (1e-6) is kept, with its standard error raised to 1e-6; the same floor applies in the paired-datasets sensitivity analysis, and every raised dataset is listed in the analysis record. | SI-83, NB-20261015-16 | docs/ANALYSIS_SPEC.md sections 2.3 and 4 |
| 2 | Do you approve five design choices that the agent made in the X1 analysis code after the rigour review? | Approved, all five: the log-spaced 201-point grid for the between-dataset spreads; the solve-based formula for the overall difference; recomputing b0, b_common, b* and the low-signal flag from the same rows and stopping on a mismatch; t-based limits in the sign table; the per-run sensitivity limited to P1 and the local-versus-global test. The specification states where the grid is accurate and that the sign-table limits are approximate. | SI-84, NB-20261015-17 | docs/ANALYSIS_SPEC.md sections 1, 4 and 15 |
| 3 | Should the stage runner refuse a Sinkhorn divergence record (the file a fit writes when the Sinkhorn solve breaks down) that lacks the code version, or that was made from code with uncommitted changes? | Yes, as it refuses such results (latents and scores); a nonfinite_latent record that the runner writes itself stays accepted; one test covers the refusals for records and results together. On 2026-10-16 the user widened the rule to divergence records of every arm (NB-20261016-09) and added that a latent which names no commit is refused too (SI-89, NB-20261016-10; code at 4b02068). | SI-85, NB-20261015-18; widened on 2026-10-16 to every arm (NB-20261016-09) | this file, entry of 2026-10-13 on the two commits; scripts/run_stage.py |
| 4 | Should the error in the power function be fixed before the tag? | Yes: prereg_rules.power_report restarts its root search at 0.01 when the search from 1.0 fails, and every result that works today stays bit-identical. | SI-86, NB-20261015-19 | this file, section 5 and the entry of 2026-10-13 on PAPER_PLAN (part (c)); scripts/prereg_rules.py |
| 5 | What counts as having examined an X1 result? | A score is examined when a person reads its value or an analysis script uses it; checks that print only counts or flags do not count. This applies to every X1 and X13 score, of any arm. Amended on 2026-10-16: only a person reading a value counts; a value shown to an AI agent is disclosed, not examined. | SI-78, NB-20261015-02, NB-20261016-14 | this file, entry of 2026-10-14 on the disclosure |
| 6 | Has anyone, you or anyone else, read the value of an X1 or X13 score? | No, nobody read a value (the user's statement). | NB-20261015-15 (notebook entry only) | this file, entry of 2026-10-14 on the disclosure, which also gives the audit of 2026-10-15 and the user's decision SI-88 on its two findings |
| 7 | Should we record a fingerprint (a SHA-256 hash) of every existing X1 and X13 score file? | Yes: the SHA-256 hash of every existing X1 and X13 score file, and, after the tag, the date that GitHub shows for the tag prereg-tier12-v3 as a time stamp from outside this repository. | SI-79, NB-20261015-03 | this file, entry of 2026-10-14 on the disclosure; experiments/x1_x13_score_hashes_20261015/ |
| 8 | Do you approve the agent's proposals in the analysis specification (docs/ANALYSIS_SPEC.md)? | Approve all: sections 2.3, 5 (Holm over the six primary results, for the descriptive Wilcoxon companion only), 8, 9, 10, the section 11 sensitivity analyses that no decision names, and 15; the parts that implement the rigour-review designs of questions 1 and 2 follow those answers. | SI-80, NB-20261015-04 | docs/ANALYSIS_SPEC.md sections 2.3, 5, 8 to 11 and 15; this file, entry of 2026-10-14 on the X1 analyses |
| 9 | Do you approve three process proposals in the pre-registration? | Approve all three: (a) each NMI-0 case (scib finds zero normalised mutual information at every clustering resolution, SI-65) is decided by the user and listed with the results; (b) a Sinkhorn iteration-cap hit (SI-61) stops the stage until the user decides whether the fit counts as failed, and failed fits count as collapses in P2; (c) the no-pair training-step counts and, per arm, the number of cells not evaluable because the arm never reaches b* are tabulated by a reporting script written after the runs; the tag freezes this promise, not the script (user, 2026-10-16, NB-20261016-11). On 2026-10-16 the user also kept in (b) the clause that each iteration-cap decision is listed with the results (NB-20261016-12). | SI-81, NB-20261015-05, NB-20261016-11, NB-20261016-12 | this file, entries of 2026-10-13 on the guards (NMI-0 and no-pair counts) and on the Sinkhorn arm (iteration cap); docs/ANALYSIS_SPEC.md section 1 (cells that never reach b*) |
| 10 | Does your decision SI-75 still stand after a recount? | It stands: P2 collapse rates plus the formal critic-versus-discriminator comparison with a 5-point margin. After the recount the discriminator with the unconditioned decoder collapsed in 8 of 54 pilot runs, not 8 of 45, so the quoted difference is +3.7 points, not +0.7. | NB-20261015-07 (notebook entry only) | this file, amendment-table row 8; docs/ANALYSIS_SPEC.md section 6; experiments/doc_numbers_20261015/p2_pilot_recount.py |
| 11 | Should we keep two Sinkhorn redo steps that cannot change X1? | Keep them, in rule order: every step of SI-57 stays, including the A3 Sinkhorn refits and the R2/R3 re-run, although neither can change X1 while the immune block with the conditioned decoder is flagged low-signal. | NB-20261015-08 (notebook entry only) | this file, entries of 2026-10-13 on low-signal cells and on the Sinkhorn arm |
| 12 | Do you approve these file names for the redone Sinkhorn decision records? | The Sinkhorn redo overwrites docs/prereg/r1_record.json, r1_record_round1.json, r4_a1_record.json, r4_a1_record_round1.json and r2r3_record.json in place; the versions before the redo stay at tag prereg-tier12-v3, which is made before the redo. | SI-87, NB-20261015-20, corrected by NB-20261015-21 | this file, entry of 2026-10-13 on the Sinkhorn arm |
| 13 | What should happen to the paper plan's four scaling checks? | Report all four as exploratory: leave-one-method-out rescaling, z-score aggregation, a composite without the silhouette metrics and rank stability by Kendall's tau are reported, labelled not pre-registered, and used for no conclusion. | SI-82, NB-20261015-06 | docs/PAPER_PLAN.md section 5 (Scaling) |
| 14 | Do you confirm a wording repair to the citations of your decision SI-68 (which analyses are locked when)? | Confirm: where PREREG and PAPER_PLAN cite SI-68 for the later lock of the follow-up analyses, they keep its own words 'an X1 result'; with SI-78 the remaining 'X1 adversarial-arm score' wording for the freeze is widened to every X1 and X13 score. | NB-20261015-09 (notebook entry only) | this file, entry of 2026-10-14 on the X1 analyses; docs/PAPER_PLAN.md section 5 |
| 15 | Which starting commit should the test-coverage record use? | 973a97d for the fixes, plus a merged record of the scripts, source and scoring test runs for the whole review range from 5cb75c0. | NB-20261015-10 (notebook entry only) | notebook/2026-10-15.md; code housekeeping, no rule of this file |
| 16 | How should the fail-loud lint tool be recorded? | In a notebook note: the tool's source (the fail-loud-pipelines skill), its version, rule count and checksum, the ruff result and why the tests run in two environments. | NB-20261015-11 (notebook entry only) | notebook/2026-10-15.md; code housekeeping, no rule of this file |
| 17 | What should we do about small tidiness warnings in the code? | Leave them: no code change for the 28 ruff warnings of 08f1e3c, 23 of them outside the identity-proof code; the count goes in the note of question 16. | NB-20261015-12 (notebook entry only) | notebook/2026-10-15.md; code housekeeping, no rule of this file |
| 18 | Should we delete 11 old review copies of the code and two stray files? | Delete after a check: each of the 11 old review worktrees is checked for uncommitted work first, and any that has some is reported and kept; the folders and the two stray files go to the Trash through the approval prompt. | NB-20261015-13 (notebook entry only) | notebook/2026-10-15.md; housekeeping, no rule of this file |
| 19 | Do you approve the remaining wording changes from earlier rounds? | Approved as shown in the diff: the wording of I091 (R1 depends on the trajectory score through its noise estimate, SI-53), D-01 (PAPER_PLAN's P1 comparator), D-02 (the power claim) and the format of the amendment table at the top of this section (DOC-03). | NB-20261015-14 (notebook entry only) | this file, entries of 2026-10-13 on R1 and the trajectory score and on PAPER_PLAN, and the amendment table; docs/PAPER_PLAN.md sections 3.2 and 5 |
