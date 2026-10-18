# Analysis specification for the follow-up experiment X6 (draft; not yet locked)

**Pruned 2026-10-17 (user decision SI-103): X3, X7, X8, X12 and X15 are dropped and this file covers X6 only; see the amendment of
2026-10-17 in `docs/PREREG.md` section 9. Section numbers are those of the draft; sections 2 and 4 to 7 were removed with their
experiments.**

*Status, 2026-10-17: draft written on branch `followup-analysis-spec` (cut from `final-review-fixes` at c241af6) by the analysis
track. It is to be locked, together with `scripts/analysis_followups.py` and its tests `tests/analysis_followups/`, by the new code
tag that follows `prereg-tier12-v3` (STATUS.md step 10 (b)), before R4-x1 runs and before anyone examines an X1 result (user
decision SI-68; "examined" as defined in SI-78: a person reads a value or an analysis script uses it). No X1, X13 or follow-up score,
latent, failure table or status record was opened to write it; the code was run on synthetic data only. `docs/ANALYSIS_SPEC.md`
(the frozen X1 specification) is not changed: this file reuses its rules and says where it departs from them.*

*Round 2 (2026-10-17): the user answered the two questions of section 12 (both recommended options), and the names were
reconciled with the build track by merging branch `followup-build` at 3c682dc (manifest v4,
`manifests/paper_manifest_stock_pilot_u5_b10_v4.tsv`, 10,908 rows; the pruned manifest v5 has 5,280 rows, with the same X6 rows); the script now reads every design name from
`scripts/build_paper_manifest.py` and the manifest instead of assuming it, and its tests run on the v4 follow-up rows.*

*Decisions it follows: SI-26 (fresh seeds 10-12 for the follow-ups), SI-56
(low-signal cells), SI-69 to SI-73 (reading, failures, decoders, pooling), SI-80 (agent proposals approved as rules), SI-82
(exploratory labels), SI-83 (standard-error floor), SI-93 and SI-98 (X6 with both decoders; the spectral-norm discriminator arm),
SI-103 (the pruning), the defaults of
NB-20261017-07 (section 4 of `docs/followup_design_decisions_20261017.md`, including the D-09 endpoint definitions, which are
"written into the follow-up analysis specification and approved with it") and NB-20261017-08. Each choice that no decision
settles is marked **agent default** and listed in section 11; the two questions put to the user and their answers are in section 12.*

**Words used.** *Run* = one model trained with one seed. *Point* = one penalty strength (lambda, λ) of one arm in one cell, i.e.
3 runs (seeds 10, 11, 12). *Cell* = task x decoder, as in ANALYSIS_SPEC. *Decoder*: the batch-conditioned decoder (cond = 1, scVI
as normally used) or the unconditioned decoder (cond = 0, the adversary alone removes batch effects; SI-93 adds it to X6).
*B* = batch-removal score, *C* = biology-conservation score (PREREG section 1, unscaled scIB means). *b0, b\** = plain
scVI's batch score and the matched batch-removal target of rule R4 for the same task and decoder, read from the frozen R4 stage-x1
record (computed on X1 seeds 0-4). *lo / matched / hi* = the three penalty strengths R4-x1 assigns to a follow-up arm: the X1
point whose batch score is closest to b\* and its two neighbours on the X1 grid (PREREG section 4; variant arms inherit their base
arm's values, PREREG line 98). *JS* = Jensen-Shannon divergence (the discriminator), *W1* = Wasserstein-1 distance (the critics),
*SN* = spectral normalisation, *GP* = gradient penalty on interpolated points, *R1* = gradient
penalty on data points. *LOSO* = leave-one-seed-out.

## 0. Scope and status of each analysis

| Follow-up (plain question) | Endpoint | Status | Real datasets pooled | Section |
|---|---|---|---|---|
| X6 divergence vs regularisation vs adversary steps | R3-divergence: W1+SN critic minus JS+SN discriminator with the critic's optimiser, biology at b\* (SI-98) | confirmatory (2 results) | 3 (atac_small, immune, pancreas) | 3 |

Dropped by SI-103 (no fit, no analysis): X3 composition dose-response, X7 reference choice, X8 batch count, X12 architecture
robustness, X15 KL warm-up. The original claims that the critic scales poorly with batch count and that it depends on a dense
reference are removed, not tested.

Confirmatory = registered here with a declaration rule (section 1.7). Secondary = registered estimate and interval, declared with the
same rule but not counted among the confirmatory results and outside the Holm family. Exploratory = labelled not pre-registered
in the paper, used for no conclusion (SI-82's wording): any analysis not written in this file.

## 1. Shared rules

### 1.1 Inputs and order

- The resolved follow-up manifest (after `freeze_matched_lambda.py --stage x1`), the follow-up score directories and one failures
  table covering every follow-up (X6) row, read through `prereg_rules.outcomes` (same provenance and kBET-seed checks as X1). A row
  neither scored nor failed stops the analysis. The analysis runs once, after every follow-up row is scored or failed.
- From the R4 stage-x1 record only `targets.<task>|<cond>` (b0, b\*, b_common, `bstar_low_signal`) is used. No X1 score row is
  read. The R4-x1 record is itself built from all X1 scores, so the follow-up analysis runs after the X1 analysis.
- No cell-type count table is needed (it served X7 and X8 only).
- The analysis stops on any departure from the X6 design of 2026-10-17: missing task x decoder cells, seeds other than 10-12,
  unknown keys in `extra`, unknown arms, a lambda label that cannot be read from the tag, a missing point (section 1.9 lists the
  names, read from the build).

### 1.2 Points and failed runs

A point is the mean B and mean C over its successful runs (ANALYSIS_SPEC section 2.4; SI-71). Failed runs (status `diverged` or
`nonfinite_latent`) are left out of every endpoint and counted in the collapse and failure tables (section 1.8). A point whose 3
runs all failed is skipped where the endpoint allows it (a b\* reading walks over it) and otherwise makes the endpoint not
evaluable for that cell, with the reason recorded.

### 1.3 Uncertainty within a dataset

Average first, then leave one seed out (SI-69): every endpoint of a dataset is computed on the point means of seeds 10-12 and then
3 more times, each time without one seed (the same seed removed from every arm and from lambda = 0). Its standard error is the
jackknife s^2 = ((m - 1)/m) x sum_i (d_(i) - mean d_(.))^2 over the m readings that exist. For a mean of paired differences this
equals SD/sqrt(3). **Agent default:** all 3 readings must exist (m = 3); X1's rule "at least 4 of 5" allows one missing reading,
which with 3 seeds would leave a jackknife over 2 values. Per-dataset intervals in the sign tables are d ± t(0.975, m - 1) s, here
t with 2 degrees of freedom = 4.30: each dataset's interval is wide, and no per-dataset significance claim is made (as X1).

### 1.4 Reading biology at b\* (X6)

Rule I048 (NB-20261017-07): "interpolate biology at b\* only where the cell's own fits bracket b\*; otherwise mark the cell 'not
evaluable at b\*' and report it at fixed strength; never call a single-strength contrast 'matched batch removal'".

- *Where b\* is read.* X6 runs on full X1 tasks with the lo / matched / hi window, so the X1 target b\* of the
  same task and decoder applies. bio@b\* is read on the curve of the three point means in increasing lambda with
  `prereg_rules.bio_at_bstar`: the first point equal to b\* (within 1e-9) or the first adjacent pair that brackets b\* (linear
  interpolation in B). This is the R2/R3 reading of PREREG section 3, not P1's: X6 has no lambda = 0 rows on seeds 10-12,
  so the curve has no anchor at (b0, c0). A cell is not evaluable at b\* when it is low-signal (SI-56), when either curve of a
  contrast never brackets b\* (status `unreached_low`, `unreached_high`), or when fewer than 3 LOSO readings exist. Every
  not-evaluable cell is listed with its reason and is also reported at the fixed `matched` strength (C difference and B
  difference), labelled fixed-strength.

### 1.5 Combining datasets

The model of ANALYSIS_SPEC section 4, unchanged and called through `analysis_x1.combine`: d_t ~ N(theta_t, s_t^2), theta_t = Delta +
u_family(t) + v_t, priors Delta ~ N(0, 0.1^2), tau_f, tau_d ~ half-normal(0.1), the same deterministic grid, the 1e-6 floor on s
with the `raised` list (SI-83). Only evaluable real datasets enter; sim1 and sim2 are reported per task and never pooled. One
result per decoder, never pooled across decoders (SI-70, now for every follow-up with both decoders, SI-93). The families of
`prereg_rules.TASK_FAMILY` apply; X6 pools atac_small, immune and pancreas (three families).

**What the pool can declare (ANALYSIS_SPEC section 15.4, AN-R12).** With one dataset P(Delta > 0.01) did not exceed 0.82, with two
datasets of different families 0.94, so a direction cannot be declared from 1 or 2 datasets at the 0.95 threshold however large the
difference; P(|Delta| <= 0.01) did not exceed 0.15 and 0.33, so equivalence cannot be declared either. Three datasets need a
difference of at least 0.03 to 0.08, depending on s. Consequence, stated before any data: X6 can reach a declaration only with its
3 datasets; a pool reduced to 1 or 2 by a low-signal cell is inconclusive by construction.

**User answer, 2026-10-17 (question 1): report per dataset, no verdict.** The registered pooled model is kept and its
posterior reported, but whenever a pool has 1 or 2 evaluable real datasets the record's declaration is
`not_decidable_1_2_real_datasets` (the model's own output stays in `model_declaration`) and the text says "not decidable with
1-2 real datasets"; the result is reported through the per-dataset sign table (estimate and interval of every dataset,
simulations included). No per-dataset decision rule is added and no dataset is added. The answer was given for X7, X8 and X15
(dropped by SI-103); it now applies when a low-signal cell leaves 2 datasets in an X6 pool. It changes no declaration, because
with 1 or 2 datasets the model cannot declare (above).

### 1.6 Margin

r = 0.01 (PREREG section 1, delta_min; ANALYSIS_SPEC section 5) for every endpoint, in the endpoint's own unit: biology score for
X6. **Agent default** (no decision names a follow-up margin; one constant keeps one rule).

### 1.7 Declarations and multiplicity

Declarations follow ANALYSIS_SPEC section 5: a direction if P(Delta > r) >= 0.95 or P(Delta < -r) >= 0.95, practically
equivalent if P(|Delta| <= r) >= 0.95, otherwise inconclusive; each on its own posterior. The confirmatory results are 2:
X6 R3-divergence (2 decoders). The report states that 2 were judged (before the pruning of SI-103 the draft had 12, 10 of them
in the dropped experiments). Frequentist companion: the exact two-sided Wilcoxon signed-rank test over the
pooled datasets' d, with Holm's adjustment over the 2 (as ANALYSIS_SPEC section 5 and PREREG "Holm over P1-P4"); a result with no
p-value (no evaluable real dataset) enters the adjustment with p = 1, so the family size stays m = 2 (recorded as `holm_m`). It cannot reject:
the smallest attainable p is 2/2^n = 0.25 with 3 datasets, 0.5 with 2 and 1 with 1. It is reported because it is registered,
and used for no conclusion. Secondary results are declared with the same rule and are not in the Holm family.

### 1.8 Collapse and failure (the P2 definition)

As ANALYSIS_SPEC section 6 and SI-75: a run collapses when it failed or when its C is at least 0.10 below the C of the same seed's
lambda = 0 run of the same design point, within 1e-9. X6 has no lambda = 0 rows on seeds 10-12, so only failure counts are
reported for it (per point; labelled failures, not collapses); a lambda = 0 anchor from X1 seeds would mix seeds and is not used.
**Agent default.** No pooled collapse comparison is registered for the follow-ups (P2's pooled comparison is an X1 result).

### 1.9 Names, read from the build (branch `followup-build` at 3c682dc, manifest v4)

| Item | Form in manifest v4 | Read from | Where used |
|---|---|---|---|
| lambda label | from the tag: `_l(matched_lo|matched|matched_hi)_c<cond>_s<seed>_<hash8>`, written before R4-x1 resolves `lam` | `build_paper_manifest.finalize` | X6 |
| cells | X6 with cond 0 and 1 (atac_small, immune, pancreas) | SI-93, manifest v4 and v5 | X6 |
| seeds | exactly 10, 11, 12 in every point | SI-26 | X6 |
| `extra` keys | `r1_gamma` (on the R1 rows); any other key stops the run | manifest v5 | X6 |
| X6 | (discriminator,1), (discriminator_sn,1), (discriminator_r1,1), (`discriminator_sn_critic_opt`,1), (pooled,1/5/10), (pooled_sn,1/5), at lo/matched/hi | `BASE_ARM`, manifest v5 | X6 |

Manifest v5 (`manifests/paper_manifest_stock_pilot_u5_b10_v5.tsv`, built by `build_paper_manifest.py --exclude-experiments X3 X7 X8 X12 X15`)
holds 486 X6 rows (243 per decoder), identical to v4's. The draft's test (`test_v4_rows_enumerated`) reads every follow-up row
through the analysis and checks the count per decoder and lambda label against a hand count of the design; it is to be cut to X6
(section 10).

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

## 8. What would confirm or refute each claim (summary)

| Claim (original paper or reviewer) | Endpoint | Supports | Contradicts | Can be declared with this design? |
|---|---|---|---|---|
| The difference is due to the divergence | X6 R3-divergence d | direction declared with P1's sign | equivalence, or opposite sign | yes (3 datasets; 2 if immune is low-signal: no) |

The claims on composition shift (X3), a dense reference (X7), batch-count scaling (X8), architecture (X12) and KL warm-up (X15) are not
tested (SI-103); the original claims that the critic scales poorly with batch count and that it depends on a dense reference are
removed.

## 9. Limitations stated with the results

1. Three seeds per point: per-dataset intervals use t with 2 degrees of freedom; the jackknife of a non-smooth reading (a crossing)
   from 3 values is rougher than X1's from 5 (ANALYSIS_SPEC section 16, limitation 1, applies more strongly).
2. Pooled declarations need 3 or more real datasets (section 1.5); X6's conditioned-decoder pool may fall to 2 if immune is low-signal.
3. The follow-up strengths come from X1 (R4-x1): a variant arm inherits its base arm's lambda, so its batch removal can land off b\*;
   where its window does not bracket b\* it is reported at fixed strength only (I048).
4. X6: incomplete factorial and the target difference between pooled critic and discriminator (section 3.7).
5. X6 with the unconditioned decoder (SI-93) uses the same rules; low-signal cells (SI-56) drop out per decoder.
6. Scope (SI-103): X3, X7, X8, X12 and X15 are not run. Architecture and KL warm-up are not varied, and reference choice,
   batch count and composition shift are not tested; the paper states these as scope limitations.

## 10. Implementation, tests and lock

- `scripts/analysis_followups.py` (usage at its top). It imports `build_paper_manifest` (the design tables), `analysis_x1`
  (`combine`, `wilcoxon_exact`, `holm`, `clopper_pearson`, `PRIOR_SCALE`) and `prereg_rules` (`outcomes`, `bio_at_bstar`,
  `read_record`, `write_record`, the constants) and changes neither; no X1 fit or analysis path is touched. One JSON record per
  run (inputs with SHA-256, git state, constants, every per-cell quantity, the not-evaluable reasons, the 2 confirmatory declarations
  with Wilcoxon and Holm values); it refuses to overwrite an existing record. Rows are sorted by tag before any computation.
- **Not done on this branch (open):** the script and its tests (`tests/analysis_followups/`) still contain the code and the
  planted-effect checks of the dropped experiments, and `CONFIRMATORY` and `EXPECTED` still list them. They must be cut to X6, with the
  Holm family set to 2, and the tests re-run on manifest v5, before the lock; only this document was changed here.
- Tests (as written for the draft, synthetic scores on the real follow-up rows): the X6 row enumeration (section 1.9); known answers
  (jackknife of a mean = SD/sqrt(3), least-squares slope, bio window, a constant predictor); a planted-effect world run end to end
  through `main` in which X6's divergence contrast (+0.07, declared above, with JS+SN given different biology under the two
  optimisers so the confirmatory subtrahend is pinned: own-optimiser contrast +0.05) recovers its planted value; the not-evaluable
  paths (low-signal, unreached, a failed run); a null world in which no confirmatory result declares a direction; refusals of
  departures from the design; no overwrite.
- Command (repo root, env scvi-api): `python scripts/check_tests.py --base origin/followup-build --src scripts --
  tests/analysis_followups` (before the pruning, on 2026-10-17, round 2, it printed OK: 13 passed; not re-run after the pruning).
- Lock: this file, the script and the tests are frozen by the follow-up code tag, after the script is cut to X6 and reconciled with
  the build track's names (section 1.9) and before R4-x1. The PREREG section 9 amendment of 2026-10-17 (SI-103) points to the pruning.

## 11. Choices made by the agent (no prior decision settles them; SI-92 defaults, each listed for review)

1. All 3 LOSO readings required (section 1.3).
2. Margin r = 0.01 in the biology score (1.6).
3. Collapse only where a same-design lambda = 0 run exists; failure counts elsewhere (1.8).
4. X6: the optimiser-matched divergence contrast is confirmatory (3.4).
5. Holm family of 2 (X6 R3-divergence, per decoder) after the pruning; the draft's family of 12 included 10 results of dropped
   experiments (1.7). A consequence of SI-103, not a separate user decision.

## 12. Questions put to the user (answered 2026-10-17)

1. Follow-ups with fewer than 3 real datasets (X7, X8, X15): answered "Report per dataset, no verdict (Recommended)": the
   registered pooled model is kept, per-dataset estimates are reported with intervals, and the result is stated as "not decidable
   with 1-2 real datasets"; no per-dataset decision rule and no added datasets (section 1.5). X7, X8 and X15 are since dropped
   (SI-103); the answer now covers an X6 pool reduced to 1-2 datasets.
2. X7's confirmatory P4 result: answered "Depleted-reference comparison (Recommended)" (SI-102). Moot: X7 is dropped (SI-103).

## 13. For the lead

- Cut `scripts/analysis_followups.py` and its tests to X6 (section 10), then run the tests on manifest v5.
- Lock this file, the script and the tests with the follow-up code tag before R4-x1; the PREREG section 9 amendment of 2026-10-17
  (SI-103) already records the pruning. The draft `docs/prereg/amendment_20261017_followups.md` of the build track covers the
  earlier design (SI-93 to SI-98) and needs the same pruning.
