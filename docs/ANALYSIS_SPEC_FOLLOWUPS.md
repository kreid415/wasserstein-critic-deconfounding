# Analysis specification for the follow-up experiments X3, X6, X7, X8, X12 and X15 (draft; not yet locked)

*Status, 2026-10-17: draft written on branch `followup-analysis-spec` (cut from `final-review-fixes` at c241af6) by the analysis
track. It is to be locked, together with `scripts/analysis_followups.py` and its tests `tests/analysis_followups/`, by the new code
tag that follows `prereg-tier12-v3` (STATUS.md step 10 (b)), before R4-x1 runs and before anyone examines an X1 result (user
decision SI-68; "examined" as defined in SI-78: a person reads a value or an analysis script uses it). No X1, X13 or follow-up score,
latent, failure table or status record was opened to write it; the code was run on synthetic data only. `docs/ANALYSIS_SPEC.md`
(the frozen X1 specification) is not changed: this file reuses its rules and says where it departs from them.*

*Round 2 (2026-10-17): the user answered the two questions of section 12 (both recommended options), and the names were
reconciled with the build track by merging branch `followup-build` at 3c682dc (manifest v4,
`manifests/paper_manifest_stock_pilot_u5_b10_v4.tsv`, 10,908 rows); the script now reads every design name from
`scripts/build_paper_manifest.py` and the manifest instead of assuming it, and its tests run on the v4 follow-up rows.*

*Decisions it follows: SI-26 (fresh seeds 10-12 for the follow-ups), SI-41/SI-42 (X3 equal cell count per dose), SI-56
(low-signal cells), SI-69 to SI-73 (reading, failures, decoders, pooling), SI-80 (agent proposals approved as rules), SI-82
(exploratory labels), SI-83 (standard-error floor), SI-93 to SI-98 (the follow-up designs of 2026-10-17), the defaults of
NB-20261017-07 (section 4 of `docs/followup_design_decisions_20261017.md`, including the D-09 endpoint definitions, which are
"written into the follow-up analysis specification and approved with it") and NB-20261017-08. Each choice that no decision
settles is marked **agent default** and listed in section 11; the two questions put to the user and their answers are in section 12.*

**Words used.** *Run* = one model trained with one seed. *Point* = one penalty strength (lambda, λ) of one arm in one cell, i.e.
3 runs (seeds 10, 11, 12). *Cell* = task x decoder, as in ANALYSIS_SPEC. *Decoder*: the batch-conditioned decoder (cond = 1, scVI
as normally used) or the unconditioned decoder (cond = 0, the adversary alone removes batch effects; SI-93 adds it to X3, X6, X7,
X8 and X12). *Decoder architecture* (X12 factor E) is a different thing: the ordinary non-linear SCVI decoder or the linear one
(LinearSCVI). *B* = batch-removal score, *C* = biology-conservation score (PREREG section 1, unscaled scIB means). *b0, b\** = plain
scVI's batch score and the matched batch-removal target of rule R4 for the same task and decoder, read from the frozen R4 stage-x1
record (computed on X1 seeds 0-4). *lo / matched / hi* = the three penalty strengths R4-x1 assigns to a follow-up arm: the X1
point whose batch score is closest to b\* and its two neighbours on the X1 grid (PREREG section 4; variant arms inherit their base
arm's values, PREREG line 98). *JS* = Jensen-Shannon divergence (the discriminator), *W1* = Wasserstein-1 distance (the critics),
*MMD* = maximum mean discrepancy, *SN* = spectral normalisation, *GP* = gradient penalty on interpolated points, *R1* = gradient
penalty on data points. *iLISI* = integration local inverse Simpson's index (scIB scales it to 0-1 by the number of batches).
*TV* = total-variation distance between two cell-type compositions. *LOSO* = leave-one-seed-out.

## 0. Scope and status of each analysis

| Follow-up (plain question) | Endpoint | Status | Real datasets pooled | Section |
|---|---|---|---|---|
| X3 composition dose-response: does removing cell types from one batch make an adversary over-correct, and does oracle reweighting undo it? | P2-dose: reference-critic minus discriminator slope of the biology cost on dose, at each fixed strength (lo, hi), per decoder | confirmatory (4 results) | 3 (atac_small, immune_hum_mou, pancreas); sim2 per task | 2 |
| X6 divergence vs regularisation vs adversary steps | R3-divergence: W1+SN critic minus JS+SN discriminator with the critic's optimiser, biology at b\* (SI-98) | confirmatory (2 results) | 3 (atac_small, immune, pancreas) | 3 |
| X7 reference choice | P4: each depleted level minus the same-size control, reference critic minus reference JS (SI-96; user answer 2026-10-17); natural-reference association secondary | confirmatory, reported per dataset with no verdict (4 results) | depleted: 1 (pancreas), sim1 per task; association: 2 (pancreas, atac_small) | 4 |
| X8 batch count | P3: slope on log2 V of reference-critic minus discriminator ceiling-normalised iLISI at fixed strength | confirmatory, reported per dataset with no verdict (2 results) | 2 (immune_hum_mou, lung); sim2 per task | 5 |
| X12 architecture robustness | sign stability of reference minus discriminator over 16 configurations; factor effects | secondary (descriptive and pooled main effects) | 3 | 6 |
| X15 KL warm-up | change of the X1-type comparisons when the KL (Kullback-Leibler) warm-up completes | sensitivity, reported per dataset with no verdict | 2 (immune, atac_large) | 7 |

Confirmatory = registered here with a declaration rule (section 1.7). Secondary = registered estimate and interval, declared with the
same rule but not counted among the confirmatory results and outside the Holm family. Sensitivity = reported to show whether X1's
conclusions depend on a setting; changes no declaration (as ANALYSIS_SPEC section 11). Exploratory = labelled not pre-registered
in the paper, used for no conclusion (SI-82's wording): the X8 adjusted model (section 5.5), the X8 attribution to batch count rather
than data volume (section 5.6), X12's two-factor interactions (section 6.4), and any analysis not written in this file.

## 1. Shared rules

### 1.1 Inputs and order

- The resolved follow-up manifest (after `freeze_matched_lambda.py --stage x1`), the follow-up score directories and one failures
  table covering every follow-up row, read through `prereg_rules.outcomes` (same provenance and kBET-seed checks as X1). A row
  neither scored nor failed stops the analysis. The analysis runs once, after every follow-up row is scored or failed.
- From the R4 stage-x1 record only `targets.<task>|<cond>` (b0, b\*, b_common, `bstar_low_signal`) is used. No X1 score row is
  read. The R4-x1 record is itself built from all X1 scores, so the follow-up analysis runs after the X1 analysis.
- A cell-type count table (columns `task, draw, batch, cell_type, n`) for the natural-reference coverage of X7 and the X8
  covariates. `draw` is `full` (the prepped task) or `x8_V<V>_s<subset>` (an X8 subset). With `--prepped-dir` the script builds
  it from the `prepped_scib/<task>__scib.h5ad` obs (`batch`, `celltype`) with the fitter's own X8 draw
  (`fit_paper_config.x8_draw`, draw `nested_chain_v1`) for every X8 spec of the manifest, writes it next to the record and
  records its hash; `--celltype-counts` takes a table built the same way. It holds no score. The depleted-reference contrast needs
  no count table (its levels differ only in the removed types, section 4.4).
- The script stops on any departure from the design of 2026-10-17: missing task x decoder cells (`EXPECTED` in the script), seeds
  other than 10-12, unknown keys in `extra`, unknown arms, an X12 run whose settings do not satisfy E = ABCD, a lambda label that
  cannot be read from the tag, a failed lambda = 0 fit, a missing point (section 1.9 lists the names, read from the build).

### 1.2 Points and failed runs

A point is the mean B and mean C over its successful runs (ANALYSIS_SPEC section 2.4; SI-71). Failed runs (status `diverged` or
`nonfinite_latent`) are left out of every endpoint and counted in the collapse and failure tables (section 1.8). A point whose 3
runs all failed is skipped where the endpoint allows it (a b\* reading walks over it) and otherwise makes the endpoint not
evaluable for that cell, with the reason recorded. Plain scVI (lambda = 0) fits must not fail; one that fails stops the run.

### 1.3 Uncertainty within a dataset

Average first, then leave one seed out (SI-69): every endpoint of a dataset is computed on the point means of seeds 10-12 and then
3 more times, each time without one seed (the same seed removed from every arm and from lambda = 0). Its standard error is the
jackknife s^2 = ((m - 1)/m) x sum_i (d_(i) - mean d_(.))^2 over the m readings that exist. For a mean of paired differences this
equals SD/sqrt(3). **Agent default:** all 3 readings must exist (m = 3); X1's rule "at least 4 of 5" allows one missing reading,
which with 3 seeds would leave a jackknife over 2 values. Per-dataset intervals in the sign tables are d ± t(0.975, m - 1) s, here
t with 2 degrees of freedom = 4.30: each dataset's interval is wide, and no per-dataset significance claim is made (as X1).
Two endpoints use a different, stated standard error: the X7 association (ordinary least-squares error of a slope across
references, section 4.3) and the X12 factor effects (pure seed error pooled over the 16 runs, section 6.3).

### 1.4 Reading biology at b\* (X6, X7) and fixed-strength reporting (X3, X8, X12, X15)

Rule I048 (NB-20261017-07): "interpolate biology at b\* only where the cell's own fits bracket b\*; otherwise mark the cell 'not
evaluable at b\*' and report it at fixed strength; never call a single-strength contrast 'matched batch removal'".

- *Where b\* is read.* X6 and X7 run on full X1 tasks with the lo / matched / hi window (X7: SI-95), so the X1 target b\* of the
  same task and decoder applies. bio@b\* is read on the curve of the three point means in increasing lambda with
  `prereg_rules.bio_at_bstar`: the first point equal to b\* (within 1e-9) or the first adjacent pair that brackets b\* (linear
  interpolation in B). This is the R2/R3 reading of PREREG section 3, not P1's: X6 and X7 have no lambda = 0 rows on seeds 10-12,
  so the curve has no anchor at (b0, c0). A cell is not evaluable at b\* when it is low-signal (SI-56), when either curve of a
  contrast never brackets b\* (status `unreached_low`, `unreached_high`), or when fewer than 3 LOSO readings exist. Every
  not-evaluable cell is listed with its reason and is also reported at the fixed `matched` strength (C difference and B
  difference), labelled fixed-strength.
- *Where b\* is not read.* X3 and X8 run on subsampled tasks, where the full task's b\* is not a defined target (the plain-scVI
  batch score changes with the subsample; brief Q3), and they have only lo/hi (X3) or one strength (X8, SI-95). X12 changes the
  architecture, so the response to lambda is not the X1 response (D-06). X15 has lo/hi only. These are compared at fixed strength,
  each point with its achieved B and, on full tasks (X12, X15), B - b\* (I048). For X3 and X8 the achieved B minus the same-seed
  lambda = 0 B of the same subsample is reported instead.

### 1.5 Combining datasets

The model of ANALYSIS_SPEC section 4, unchanged and called through `analysis_x1.combine`: d_t ~ N(theta_t, s_t^2), theta_t = Delta +
u_family(t) + v_t, priors Delta ~ N(0, 0.1^2), tau_f, tau_d ~ half-normal(0.1), the same deterministic grid, the 1e-6 floor on s
with the `raised` list (SI-83). Only evaluable real datasets enter; sim1 and sim2 are reported per task and never pooled. One
result per decoder, never pooled across decoders (SI-70, now for every follow-up with both decoders, SI-93). The families of
`prereg_rules.TASK_FAMILY` apply; no follow-up pools two datasets of the same family (X3: atac, immune, pancreas; X6 and
X12: atac, immune, pancreas; X7: pancreas, atac; X8: immune, lung; X15: immune, atac).

**What the pool can declare (ANALYSIS_SPEC section 15.4, AN-R12).** With one dataset P(Delta > 0.01) did not exceed 0.82, with two
datasets of different families 0.94, so a direction cannot be declared from 1 or 2 datasets at the 0.95 threshold however large the
difference; P(|Delta| <= 0.01) did not exceed 0.15 and 0.33, so equivalence cannot be declared either. Three datasets need a
difference of at least 0.03 to 0.08, depending on s. Consequences, stated before any data: the confirmatory results of X7 and X8 and
the X15 sensitivity results are inconclusive by construction under this model; only X3, X6 and X12 can reach a declaration.

**User answer, 2026-10-17 (question 1): report per dataset, no verdict.** The registered pooled model is kept and its
posterior reported, but whenever a pool has 1 or 2 evaluable real datasets the record's declaration is
`not_decidable_1_2_real_datasets` (the model's own output stays in `model_declaration`) and the text says "not decidable with
1-2 real datasets"; the result is reported through the per-dataset sign table (estimate and interval of every dataset,
simulations included). No per-dataset decision rule is added and no dataset is added. By design this concerns X7, X8 and
X15; it also applies when a low-signal cell leaves 2 datasets in an X3, X6 or X12 pool. It changes no declaration, because
with 1 or 2 datasets the model cannot declare (above).

### 1.6 Margin

r = 0.01 (PREREG section 1, delta_min; ANALYSIS_SPEC section 5) for every endpoint, in the endpoint's own unit: biology score for
X6, X7 natural-reference points and X15; biology score per 100 percentage points of depletion for X3; biology score per unit of
coverage share for the X7 slopes; ceiling-normalised iLISI (or B, C) per doubling of the batch count for X8; biology score per
two-level change of a factor for X12. **Agent default** (no decision names a follow-up margin; one constant keeps one rule).

### 1.7 Declarations and multiplicity

Declarations follow ANALYSIS_SPEC section 5: a direction if P(Delta > r) >= 0.95 or P(Delta < -r) >= 0.95, practically
equivalent if P(|Delta| <= r) >= 0.95, otherwise inconclusive; each on its own posterior. The confirmatory results are 12:
X3 P2-dose (2 strengths x 2 decoders), X6 R3-divergence (2 decoders), X7 P4 (2 depletion levels x 2 decoders), X8 P3 (2 decoders)
(`CONFIRMATORY` in the script). The report states that 12 were judged, and that the 6 of X7 and X8 are reported per dataset
without a verdict (section 1.5). Frequentist companion: the exact two-sided Wilcoxon signed-rank test over the
pooled datasets' d, with Holm's adjustment over the 12 (as ANALYSIS_SPEC section 5 and PREREG "Holm over P1-P4"). It cannot reject:
the smallest attainable p is 2/2^n = 0.25 with 3 datasets, 0.5 with 2 and 1 with 1. It is reported because it is registered,
and used for no conclusion. Secondary results are declared with the same rule and are not in the Holm family.

### 1.8 Collapse and failure (the P2 definition)

As ANALYSIS_SPEC section 6 and SI-75: a run collapses when it failed or when its C is at least 0.10 below the C of the same seed's
lambda = 0 run of the same design point (same dose, same subset, same warm-up), within 1e-9. Rates with Clopper-Pearson 95%
intervals and failure counts are reported: X3 per arm, strength and dose; X8 per arm and sampler over all V and subsets; X15 per
warm-up. X6, X7 and X12 have no lambda = 0 rows on seeds 10-12, so only failure counts are reported for them (X6 per point, X7 per arm
and reference over its window, X12 per arm and run; labelled failures, not collapses); a lambda = 0 anchor from X1 seeds would mix seeds and is not used. **Agent default.** No pooled
collapse comparison is registered for the follow-ups (P2's pooled comparison is an X1 result).

### 1.9 Names, read from the build (branch `followup-build` at 3c682dc, manifest v4)

| Item | Form in manifest v4 | Read from | Where used |
|---|---|---|---|
| lambda label | from the tag: `_l(matched_lo|matched|matched_hi|0)_c<cond>_s<seed>_<hash8>`, written before R4-x1 resolves `lam` | `build_paper_manifest.finalize` | all |
| cells | X3, X6, X7, X8, X12 with cond 0 and 1; X15 cond 1 only (`EXPECTED`) | SI-93, manifest v4 | all |
| seeds | exactly 10, 11, 12 in every point | SI-26 | all |
| `extra` keys | `subsample, sampler, iw, factorial_run, adv_width, adv_lr, kl_warmup, r1_gamma`; any other key stops the run | manifest v4 | all |
| subsample kinds | `composition` with draw `nested_v1` (X3), `batches` with `nested_chain_v1` (X8), `reference_composition` with `reference_fixed_v1` (X7); any other kind or draw stops the run | `BPM.X3_DRAW`, `X8_DRAW`, `X7_DRAW` | X3, X7, X8 |
| X3 | `deplete_pct` 0/50/80/95/100; weighted rows `iw = depletion_oracle` at 50/80/95 under both decoders; arms `none` (lambda 0), `discriminator, reference, pooled, mmd` at lo, hi | manifest v4 | X3 |
| X6 | (discriminator,1), (discriminator_sn,1), (discriminator_r1,1), (`discriminator_sn_critic_opt`,1), (pooled,1/5/10), (pooled_sn,1/5), at lo/matched/hi | `BASE_ARM`, manifest v4 | X6 |
| X7 natural rows | arms `reference, reference_fixed, mmd_ref, discriminator_ref`; `reference` = index into the sorted batch names, every batch; lo/matched/hi | manifest v4 | X7 |
| X7 depleted rows | same arms; `reference` = batch name = `subsample.batch` (pancreas inDrop1, sim1 Batch5); `n_reference` (1,572 / 1,005); level = the entry of `BPM.X7_DEPLETED[task]` whose removed-type list equals `subsample.types`: `none` (no type removed, the same-size control), `two_thirds`, `one_third` (rarest types removed first); a list that matches no entry stops the run | `BPM.X7_DEPLETED` | X7 |
| X8 | `n_batches` 2/4/8/16, `subset` = nested chain 0/1, `cells_per_batch` = X8_TOTAL // 16; arms `none` (lambda 0) and `discriminator, mmd, reference, pooled` at `matched`; `sampler = stratified` rows for reference and pooled | manifest v4, `fit_paper_config.x8_draw` | X8 |
| X12 | `factorial_run` 0-15; A = n_latent 30 (else 10), B = adv_width 128 (else 32), C = adv_lr 1e-3 (else 1e-4), D = batch_size 512 (else 128), E = decoder `LinearSCVI` (else SCVI), run bits A = run & 1, B bit 1, C bit 2, D bit 3, E = A xor B xor C xor D; arms discriminator (1 step), reference (5 steps) at lo, hi | `build_paper_manifest.py` X12 block | X12 |
| X15 | `kl_warmup` in `stock, complete`; arms `none`, `discriminator, pooled, mmd` at lo, hi | manifest v4 | X15 |

A test (`test_v4_rows_enumerated`) reads every v4 follow-up row through the analysis and checks the count per experiment,
decoder and lambda label (and per X7 level and X3 weighted row) against hand counts of the design.

## 2. X3: composition dose-response (P2-dose)

**2.1 Question.** When cell types are removed from one non-reference batch (dose = percentage of the two declared types removed:
0, 50, 80, 95, 100, equal cell count per dose, SI-41), does an adversary lose more biology than plain scVI loses on the same
subsample, does the reference W1 critic lose more than the discriminator, and does oracle reweighting of the alignment objective
(importance weights that restore the dose-0 composition) undo it?

**2.2 Estimand.** For arm a at fixed strength l (lo or hi) in a cell: the biology cost at dose x, cost_a(x) = C_a(x) - C_none(x)
(point means; the lambda = 0 run of the same dose removes the effect of the subsample on scVI itself), and its ordinary
least-squares slope beta_a on x / 100 over the 5 doses (units: biology per 100 percentage points). Over-correction under
composition shift is beta_a < 0.

**2.3 Confirmatory endpoint.** d = beta_reference - beta_discriminator (the P1 pair: reference critic minus the discriminator at 1
step), per strength (lo, hi) and decoder: 4 results. D-09 defines the response "at each fixed lambda"; **agent default:** the two
strengths are separate results rather than averaged. The reference-minus-discriminator contrast is primary because the registered
hypothesis concerns the critic against the discriminator (PAPER_PLAN section 3, R1 row); **agent default.** The original claim
"the critic over-corrects, the discriminator is protective" predicts d < 0.

**2.4 Secondary.** (i) each arm's own slope beta_a (discriminator, reference, pooled, MMD), per strength and decoder, pooled;
(ii) the batch-score slope of B_a(x) - B_none(x), which shows whether batch removal itself moves with dose at fixed strength;
(iii) oracle recovery: the mean over doses 50, 80, 95 of C_a,weighted(x) - C_a,unweighted(x), per arm and strength (dose 0 has
weight 1 everywhere; dose 100 has no depleted cells to reweight), per dataset with LOSO intervals; (iv) collapse rates per dose.
The effective sample size of the weights per dose is reported from the weight profile (`docs/x3_iw_weight_profile.csv`,
column `ess_frac_batch_b`; the weights do not depend on the decoder), and the text states that the unweighted dose effect includes
a shift of the depleted batch's share of cells (I088).

**2.5 Unit of independence and pooling.** Dataset. Seeds 10-12 give the within-dataset error (section 1.3). Real datasets atac_small,
immune_hum_mou, pancreas pooled (3 families); sim2 per task.

**2.6 Not evaluable.** A dose point whose 3 runs all failed (for the arm or its weighted copy) leaves the slope undefined for that
cell (reason `not_readable`); fewer than 3 LOSO readings (reason `readings`). Low-signal flags do not apply (no b\* is read).

**2.7 Confirm / refute.** Confirmed (critic over-corrects more under composition shift) if P(Delta < -0.01) >= 0.95 for a strength
and decoder; refuted in the opposite direction if P(Delta > 0.01) >= 0.95; practically equivalent if P(|Delta| <= 0.01) >= 0.95;
otherwise inconclusive. The oracle control is described as "oracle reweighting of the alignment objective", and its recovery is
reported, not declared (no rule ties it to a claim).

## 3. X6: divergence vs regularisation vs adversary steps (R3 factor effects)

**3.1 Question.** Is a critic-discriminator difference in biology at matched batch removal due to the divergence (W1 vs JS), or to
the Lipschitz control (gradient penalty, spectral normalisation, R1), the optimiser, or the number of adversary steps per update?

**3.2 Configurations.** JS with no penalty, SN, R1 (gamma 10) and SN with the critic's Adam settings (SI-98), each at 1 step; the
pooled W1 critic with GP at 1, 5, 10 steps and with SN at 1, 5 steps (spectral-norm critics compute no gradient penalty once the
I036 fix is on the follow-up tag). Each at lo/matched/hi.

**3.3 Estimand.** For a contrast (a, b): d = bio@b\*(a) - bio@b\*(b) (section 1.4), per task and decoder.

**3.4 Confirmatory endpoint.** The divergence contrast with matched optimiser: W1+SN at 1 step (`pooled_sn`, n_critic 1) minus
JS+SN with the critic's optimiser (`discriminator_sn_critic_opt`), per decoder: 2 results. With the I036 fix and SI-98 these two differ
only in the loss (both spectrally normalised, both 1 step, both Adam lr 1e-4, betas (0, 0.9)), and the target (the pooled
critic's symmetric target versus the discriminator's batch classifier) as stated in 3.7. **Agent default** that this, not the
own-optimiser contrast, is the confirmatory one: SI-98 was added to remove the optimiser confound (brief Q6).

**3.5 Secondary contrasts (the identified contrasts of I087).** Divergence with each family's own optimiser (W1+SN 1 step minus
JS+SN); optimiser within JS+SN (critic's settings minus the discriminator's); JS regularisation (SN minus none; R1 minus none);
W1 regularisation (SN minus GP at 1 and at 5 steps); update budget (GP 5 minus 1, GP 10 minus 1, SN 5 minus 1); as used in X1 (W1+GP
at 1 step minus JS at 1 step). Each pooled per decoder with the same rule; not in the Holm family. Also the R3 window width on the
biology score (D-09): the points whose mean C is at least the lower end of the two-sided 95% t interval (2 degrees of freedom) of
the best point's mean; width = log10(largest lambda) - log10(smallest), over 3 points, descriptive.

**3.6 Not evaluable; pooling.** Section 1.4 rules (low-signal; either curve unreached; fewer than 3 readings); the fixed `matched`
contrast is reported for every cell. Real datasets atac_small, immune, pancreas pooled; with the conditioned decoder the immune
cell was low-signal in the A1 pilot (SI-56), so that pool may hold 2 datasets and cannot then declare (section 1.5).

**3.7 Confirm / refute; disclosed limits.** The divergence explains the difference if the confirmatory contrast has a declared
direction with the same sign as X1's P1 for that decoder and the regularisation and step contrasts are smaller; this joint reading is
descriptive (only the divergence contrast is declared). Disclosed: X6 is an incomplete factorial; R1 (penalty on data points) and
GP (penalty on interpolates) differ in form; per-batch versus per-cell weighting was not probed; the pooled critic and the
discriminator differ in target as well as divergence; the X6 critic is the pooled one, while P1 uses the reference critic.

## 4. X7: reference choice (P4)

**4.1 Question.** Does the reference W1 critic depend on which batch is the reference (in particular on how well the reference covers
the task's cell types) more than a reference-anchored JS discriminator or a reference MMD with the same target?

**4.2 Coverage (D-09; natural references).** coverage = share of the task's cell types (types present in the prepped task, draw `full`) that the reference
batch holds with at least 10 cells. **Agent default: 10 cells** (no decision gives the minimum count; a fixed small count keeps types that the
reference holds only as a few stray cells from counting as covered, and is the same for every task and level).
Size = cells in the reference batch; entropy = Shannon entropy (natural log) of its cell-type shares.

**4.3 Natural references (association; secondary).** Every batch is the reference once (pancreas 9, sim1 6, atac_small 3). For
formulation f and reference k: y_f,k = bio@b\* (section 1.4). Paired difference across references, g_k = y_reference,k -
y_comparator,k, with comparator `discriminator_ref` (secondary companion: `mmd_ref`). Estimate: ordinary least-squares slope of g_k
on coverage_k over the references where both curves are read (at least 3; otherwise reason `fewer_than_3_references`), with its
classical standard error (n - 2 degrees of freedom; the references are the units of this association, so their scatter is the
error). Pancreas and atac_small pooled per decoder (2 datasets: inconclusive by construction, section 1.5); sim1 per task. Where at
least 6 references are read (pancreas, sim1) the coefficient on coverage adjusted for log(size) and entropy is reported (no
interval; none when the covariates are collinear). Also per formulation: the between-reference SD of y_f,k next to the seed-noise
floor (the square root of the mean variance of a 3-run point mean), the "seed noise floor" of PAPER_PLAN's P4 row. Wording: an
association; coverage moves with entropy, size and technology across references (I089), so no causal reading.

**4.4 Depleted reference (causal; confirmatory; user answer 2026-10-17).** On pancreas (reference inDrop1) and sim1 (reference
Batch5) the reference batch is drawn at three levels with one size per task (1,572 and 1,005 cells): `none` keeps every type (the
same-size control), `two_thirds` and `one_third` keep round(2c/3) and round(c/3) of its c types by removing the rarest types first;
every other batch is whole (`BPM.X7_DEPLETED`, `fit_paper_config.x7_draw`). The levels therefore differ in coverage only, while
technology and reference size are fixed. For each depleted level L and decoder:
d_L = [bio@b\*(reference, L) - bio@b\*(discriminator_ref, L)] - [bio@b\*(reference, none) - bio@b\*(discriminator_ref, none)],
the change of the critic-minus-JS difference caused by removing types, against the control of the same size (never against the
full-size natural reference, whose size differs). Companion: the same with `mmd_ref`. Standard error by LOSO (the reference is
fixed, so seed noise is the error). Confirmatory results: d_two_thirds and d_one_third for each decoder (4). The dense-reference
claim predicts d_L < 0, more negative at one_third. Not evaluable: low-signal; any of the 4 curves of a contrast unreached; fewer
than 3 readings. Reporting: per dataset with intervals; pancreas is the only real dataset, so the pooled entry is "not decidable
with 1-2 real datasets" (section 1.5); sim1 is reported per task and never pooled. The record keeps the reference, its size and
the number of types removed at each level.

**4.5 Labels and limits.** Only the depleted-reference contrast is described causally (I089 default as amended by the user's Q4
answer). The reference arm's automatic target uses cell-type labels, and the depletion is chosen from the reference's most abundant
types (I092 disclosure). `reference_fixed` (the critic with a fixed reference target) is reported per reference, descriptive.

## 5. X8: batch count (P3)

**5.1 Question.** Does the critic-discriminator difference change as the number of batches V grows (2, 4, 8, 16)?

**5.2 Design facts the analysis must respect (SI-94).** Batch sets are nested (the 2 inside the 4, inside the 8, inside the 16) and
every batch keeps N/16 cells of the task's X8 budget at every V, so total cells are V x N/16 and grow with V, and so do the
training steps per epoch. Cells per batch per minibatch are batch size / V (128/V for equal batches: 64 at V = 2, 8 at V = 16).
X8 is compared at one fixed strength, the full task's matched lambda, with achieved batch score as a covariate (SI-95). The
default sampler is used for every arm in the primary contrast; the stratified sampler exists for the reference and pooled critics
only (SI-24, D-07).

**5.3 Ceiling-normalised iLISI (D-09).** iLISI_cn = iLISI / ceiling_V, where iLISI is the scIB-scaled score stored by the scorer and
ceiling_V = (1 / sum_b p_b^2 - 1) / (V - 1) is its largest attainable value for the realised batch shares p_b of that draw (1 when
batches are equal; below 1 where floor division leaves unequal batches, e.g. sim2, I015 (b)).

**5.4 Confirmatory endpoint.** For each nested chain (subset) and V: g(V) = iLISI_cn(reference) - iLISI_cn(discriminator) (point
means, default sampler); the least-squares slope of g on log2 V over the 4 values of V; d = mean of the chains' slopes (units:
ceiling-normalised iLISI per doubling of V). Per decoder: 2 results, pooled over immune_hum_mou and lung (2 datasets: inconclusive
by construction, section 1.5); sim2 per task. The "bottleneck" claim predicts d < 0. Chains that select identical batch sets at a V
(listed in the record) are not independent composition replicates (I016): averaging them adds no information, and the standard
error comes from seeds only.

**5.5 Secondary.** The same slope for B and for C; the pooled critic minus the discriminator (PAPER_PLAN: "attribute to reference
design only if the pooled critic shows no slope", read descriptively); the stratified minus default sampler slope for each critic
(critic-only contrast, D-07); per-arm collapse rates; B minus the lambda = 0 B for every arm, V and chain (achieved batch removal).
Exploratory (SI-95, I078): the biology gap regressed on log2 V, the achieved batch-score gap and the composition distance (TV) from
the chain's V = 2 draw, over the 8 chain x V points, without an interval. When the batch-score gap itself moves with V, the
coefficients of log2 V and of the batch gap are not separately identified (only their combination along the data is), and the
record says so by reporting the coefficients without an interpretation; it is reported as none when the covariates are exactly
collinear. Cells per batch per minibatch is a function of V and cannot be a separate covariate.

**5.6 What P3 can and cannot say.** Because total cells and steps grow exactly with V (log2 total cells = log2 V + a constant), a
slope on log2 V is the effect of adding batches together with their cells. A claim that the number of batches, rather than data
volume, drives it is exploratory and not identified by X8 (SI-94's confound). The reference critic's automatic reference batch may
change between V within a chain; the record of the build track should list it per draw (section 13), and the analysis reports
it as a limitation.

## 6. X12: architecture robustness

**6.1 Question.** Is the sign of the reference-critic-minus-discriminator difference stable over 16 backbone and adversary
configurations, and which settings move it?

**6.2 Design.** 2^(5-1) fraction with generator E = ABCD (resolution V, SI-97): A latent size 10/30, B adversary width 32/128, C
adversary learning rate 1e-4/1e-3, D minibatch size 128/512, E decoder architecture SCVI/LinearSCVI. Main effects are aliased only
with four-factor interactions and two-factor interactions with three-factor interactions. The script checks every row against
this design.

**6.3 Estimand.** Per run, strength (lo, hi), task and decoder: d_run = mean over seeds of the paired difference C_reference,s -
C_discriminator,s (seeds where both arms succeeded). Main effect of a factor (or a two-factor interaction, using the product of the
+1/-1 codes) = mean of d_run over its 8 "+" runs minus the mean over its 8 "-" runs. Standard error from pure seed error: the
within-run variance of the paired differences pooled over the 16 runs (up to 32 degrees of freedom), Var = sigma^2 x (sum over +
runs of 1/n_run + sum over - runs of 1/n_run) / 64. Results are at fixed strength with achieved B and B - b\* per arm and run, and
are not called matched (D-06).

**6.4 Reported.** (i) Sign stability (secondary, descriptive): per task, decoder and strength, the counts of runs with d_run > 0.01,
< -0.01, within ±0.01, and whether no two runs have opposite signs beyond the margin. (ii) Main effects (secondary): per dataset with
intervals, pooled over the 3 real datasets per decoder and strength with the section 1.5 model and declared with the section 1.7
rule; not in the Holm family. (iii) Two-factor interactions: per dataset only, exploratory. A run with no paired seeds makes that
task x decoder x strength not evaluable for effects (reason recorded).

## 7. X15: KL warm-up (sensitivity)

**7.1 Question.** Do the X1 comparisons change when the KL warm-up completes within training instead of scvi-tools' stock 400-epoch
warm-up (conditioned decoder; immune, atac_large; SI-44)?

**7.2 Estimand.** At each fixed strength: the change of the pooled-critic-minus-discriminator biology difference, [C_pooled -
C_disc]_complete - [C_pooled - C_disc]_stock; and for each arm its biology cost against lambda = 0, [C_a - C_none]_complete -
[C_a - C_none]_stock. LOSO standard errors; B - b\* for every point. X15 has no reference-critic arm, so the P1 pair itself is not
tested; the pooled critic is the nearest W1 arm.

**7.3 Reading.** Sensitivity: per dataset with intervals and a pooled posterior over the 2 datasets (inconclusive by construction,
section 1.5); no declaration and no change to any X1 conclusion. The 12 stock-warm-up rows with X6 settings are fitted separately
(PREREG section 9).

## 8. What would confirm or refute each claim (summary)

| Claim (original paper or reviewer) | Endpoint | Supports | Contradicts | Can be declared with this design? |
|---|---|---|---|---|
| Critic over-corrects under composition shift; discriminator protective | X3 P2-dose d | P(Delta < -0.01) >= 0.95 | P(Delta > 0.01) >= 0.95 or equivalence | yes (3 real datasets), if the difference is about 0.03 or more |
| The difference is due to the divergence | X6 R3-divergence d | direction declared with P1's sign | equivalence, or opposite sign | yes (3 datasets; 2 if immune is low-signal: no) |
| Critic needs a dense reference | X7 d_L (depleted level minus same-size control) | d_L < -0.01 | d_L > 0.01 | no: 1 real dataset; per dataset, no verdict (user answer) |
| Critic scales poorly with batch count | X8 P3 d | slope < -0.01 per doubling | slope > 0.01 | no: 2 real datasets; per dataset, no verdict (user answer) |
| Sign is stable across architectures | X12 sign counts | no opposite signs beyond ±0.01 | runs of both signs | descriptive |
| X1 holds without the warm-up artefact | X15 | changes within ±0.01 | changes beyond ±0.01 | descriptive (sensitivity) |

## 9. Limitations stated with the results

1. Three seeds per point: per-dataset intervals use t with 2 degrees of freedom; the jackknife of a non-smooth reading (a crossing)
   from 3 values is rougher than X1's from 5 (ANALYSIS_SPEC section 16, limitation 1, applies more strongly).
2. Pooled declarations need 3 or more real datasets (section 1.5); X7, X8 and X15 have 1 or 2.
3. The follow-up strengths come from X1 (R4-x1): a variant arm inherits its base arm's lambda, so its batch removal can land off b\*;
   where its window does not bracket b\* it is reported at fixed strength only (I048).
4. X3 and X8 subsamples have no defined b\*; their comparisons are at fixed strength and mix any change of batch removal with the
   change of biology (the batch-score slopes are reported next to them).
5. X8: batch count is aliased with total cells and training steps (section 5.6); the automatic reference may move with V.
6. X7: natural coverage barely varies in pancreas (8 of 9 references cover 93-100% of the types, brief Q4), so the association rests
   mostly on one reference; the depleted arm uses one reference batch per task, thinned to one size below its natural size (pancreas
   1,572 of 1,937, sim1 1,005 of 1,761), and sim1's deepest level keeps 2 of 7 types (followup-build review guide).
7. X6: incomplete factorial and the target difference between pooled critic and discriminator (section 3.7).
8. The mechanism follow-ups with the unconditioned decoder (SI-93) use the same rules; low-signal cells (SI-56) drop out per decoder.

## 10. Implementation, tests and lock

- `scripts/analysis_followups.py` (usage at its top). It imports `build_paper_manifest` (the design tables), `fit_paper_config`
  (only `x8_draw`, when it builds the count table), `analysis_x1` (`combine`, `wilcoxon_exact`, `holm`,
  `clopper_pearson`, `PRIOR_SCALE`) and `prereg_rules` (`outcomes`, `bio_at_bstar`, `read_record`, `write_record`, the constants)
  and changes neither; no X1 fit or analysis path is touched. One JSON record per run (inputs with SHA-256, git state, constants,
  every per-cell quantity, the not-evaluable reasons, the 10 confirmatory declarations with Wilcoxon and Holm values); it refuses
  to overwrite an existing record. Rows are sorted by tag before any computation.
- Tests (`tests/analysis_followups/test_analysis_followups.py`, synthetic scores on the real v4 follow-up rows): the v4 row
  enumeration (section 1.9); known answers (jackknife of a mean = SD/sqrt(3),
  least-squares slope, iLISI ceiling 0.6 for shares 1/4 and 3/4, coverage, bio window, a constant predictor); a planted-effect world
  run end to end through `main` in which every follow-up's endpoint recovers its planted value (X3 slope contrast -0.10 declared
  below; X6 divergence +0.05 declared above; X7 association slope +0.2 and depleted-level contrasts -0.2/3 and -0.4/3 against the control; X8 P3 slope -0.02 per doubling; X12 factor A
  effect 0.06 with the other main effects and interactions near 0, declared above; X15 null) and the not-evaluable paths
  (low-signal, unreached, fewer than 3 references, a run without paired seeds, a failed run, collapse at the 0.10 bound); a null
  world in which no confirmatory result declares a direction; the count table built from prepped h5ad files and from obs;
  refusals of eight departures from the design (including an unknown subsample kind and a removed-type list that matches no
  level); no overwrite.
- Command (repo root, env scvi-api): `python scripts/check_tests.py --base origin/followup-build --src scripts --
  tests/analysis_followups` (the build track's own changes are tested by its tests; about 1.5 minutes; on 2026-10-17, round 2, it
  printed OK: 13 passed).
- Lock: this file, the script and the tests are frozen by the follow-up code tag, after reconciliation with the build track's names
  (section 1.9) and before R4-x1. A PREREG section 9 amendment (written by the lead with the build track's) points to this file.

## 11. Choices made by the agent (no prior decision settles them; SI-92 defaults, each listed for review)

1. All 3 LOSO readings required (section 1.3).
2. Margin r = 0.01 in each endpoint's unit (1.6).
3. Collapse only where a same-design lambda = 0 run exists; failure counts elsewhere (1.8).
4. X3: the reference-minus-discriminator slope contrast is confirmatory, per strength (2.3).
5. X6: the optimiser-matched divergence contrast is confirmatory (3.4).
6. X7: both depleted levels are confirmatory, each against the same-size control (4.4).
7. X7: coverage minimum count 10 cells (4.2); natural association error = least-squares error across references (4.3).
8. X8: the confirmatory response is ceiling-normalised iLISI (PAPER_PLAN P3, D-09); B and C secondary (5.4-5.5).
9. X12: effects with pure seed error; sign stability and main effects secondary; two-factor interactions exploratory (6.3-6.4).
10. X15: sensitivity only (7.3).

## 12. Questions put to the user (answered 2026-10-17)

1. Follow-ups with fewer than 3 real datasets (X7, X8, X15): answered "Report per dataset, no verdict (Recommended)": the
   registered pooled model is kept, per-dataset estimates are reported with intervals, and the result is stated as "not decidable
   with 1-2 real datasets"; no per-dataset decision rule and no added datasets (section 1.5).
2. X7's confirmatory P4 result: answered "Depleted-reference comparison (Recommended)": the depleted-reference contrast,
   read against the same-size control; the natural-reference association is secondary (section 4).

## 13. For the lead

- The cell-type count table is built by the script from the prepped files (`--prepped-dir`); no separate producer is needed.
- Record the automatic reference batch of each X8 draw (the reference critic's target may change with V within a chain).
- Write the PREREG section 9 amendment (draft `docs/prereg/amendment_20261017_followups.md`) so that it points to this file; lock
  both with the follow-up code tag before R4-x1.
