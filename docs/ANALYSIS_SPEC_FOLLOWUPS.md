# Analysis specification for the follow-up experiment X6

**Status: locked by the follow-up code tag `prereg-x6-v1`.** That tag is made after the user's review of this text (it did not
exist when this text was written, 2026-10-20) and freezes this file, `scripts/analysis_followups.py` and its tests
`tests/analysis_followups/` together with the PREREG section 9 entry of 2026-10-20, before R4-x1 runs and before anyone examines an
X1 result (user decision SI-68; "examined" as defined in SI-78: a person reads a value or an analysis script uses it). No X1, X13
or follow-up score, latent, failure table or status record was opened to write or change this file or the script; the script was
run on synthetic data only. `docs/ANALYSIS_SPEC.md` (the frozen X1 specification) is not changed: this file reuses its rules and
lists where it departs from them (section 1.11).

**Scope.** This file covers X6 only. The follow-ups X3, X7, X8, X12 and X15 were dropped by the user's decision SI-103 of
2026-10-17 and have no analysis here. Section numbers are those of the earlier, longer draft; sections 2 and 4 to 7 were removed with
their experiments. The design that this file analyses is put in force by the PREREG section 9 entry of 2026-10-20 (both decoders,
the arm `discriminator_sn_critic_opt` and its base-arm mapping, no gradient penalty for spectral-norm critics, the strength
pre-check of section 1.10 for that arm and for `pooled_sn` at 1 step). The earlier draft amendment `docs/prereg/amendment_20261017_followups.md` is superseded by that entry.

*Decisions it follows: SI-26 (fresh seeds 10-12 for the follow-ups), SI-56 (low-signal cells), SI-69 to SI-73 (reading, failures,
decoders, pooling), SI-80 (agent proposals approved as rules), SI-82 (exploratory labels), SI-83 (standard-error floor), SI-93 and
SI-98 (X6 with both decoders; the spectral-norm discriminator arm), SI-103 (the pruning), the pre-check decision of 2026-10-20 and its
extension to `pooled_sn` at 1 step (SI-107, 2026-10-21), and the
defaults of NB-20261017-07 (section 4 of `docs/followup_design_decisions_20261017.md`, including the D-09 endpoint definitions, which
are "written into the follow-up analysis specification and approved with it") and NB-20261017-08. Each choice that no decision settles
is marked **agent default** and listed in section 11; the two questions put to the user and their answers are in section 12.*

**Words used.** *SI-nn* = entry nn of the project's decision ledger (`CONSTRAINTS.md`); *NB-yyyymmdd-nn* = entry nn of the lab
notebook of that date; *Ixxx* and *D-nn* = findings of the project's final review (named in
`docs/followup_design_decisions_20261017.md`). *Run* = one model trained with one seed. *Point* = one penalty strength (lambda, λ) of
one arm in one cell, i.e. 3 runs (seeds 10, 11, 12). *Cell* = task x decoder, as in ANALYSIS_SPEC. *Decoder*: the batch-conditioned
decoder (cond = 1, scVI as normally used) or the unconditioned decoder (cond = 0, the adversary alone removes batch effects; SI-93
adds it to X6). *B* = batch-removal score, *C* = biology-conservation score (PREREG section 1, unscaled scIB means). *b0, b\** =
plain scVI's batch score and the matched batch-removal target of rule R4 for the same task and decoder, read from the frozen R4
stage-x1 record (computed on X1 seeds 0-4). *lo / matched / hi* = the three penalty strengths R4-x1 assigns to an X6 arm: the X1 point
whose batch score is closest to b\* and its two neighbours on the X1 grid (PREREG section 4; a variant arm takes the strengths of its
base arm, in X6: `discriminator_sn`, `discriminator_r1` and `discriminator_sn_critic_opt` take those of `discriminator`;
`pooled_sn` and `pooled` at other step counts take those of `pooled`; the pre-check of section 1.10 then moves the strengths of
`discriminator_sn_critic_opt` and of `pooled_sn` at 1 step, each by its own decision, one step at most). *JS* = Jensen-Shannon divergence (the discriminator), *W1* =
Wasserstein-1 distance (the critics), *SN* = spectral normalisation, *GP* = gradient penalty on interpolated points, *R1* = gradient
penalty on data points. *LOSO* = leave-one-seed-out. *P1* = the main study's primary result: biology at b\* of the reference W1 critic
minus that of the discriminator, per decoder (ANALYSIS_SPEC section 2).

## 0. Scope and status of each analysis

| Follow-up (plain question) | Endpoint | Status | Real datasets pooled | Section |
|---|---|---|---|---|
| X6 divergence vs regularisation vs adversary steps | W1+SN critic minus JS+SN discriminator with the critic's optimiser, biology at b\* (SI-98) | confirmatory (2 results, one per decoder) | 3 (atac_small, immune, pancreas) | 3 |

Dropped by SI-103 (no fit, no analysis): X3 composition dose-response, X7 reference choice, X8 batch count, X12 architecture
robustness, X15 KL warm-up. The original claims that the critic scales poorly with batch count and that it depends on a dense
reference are removed, not tested.

Confirmatory = registered here with a declaration rule (section 1.7). Secondary = registered estimate and interval, declared with the
same rule but not counted among the confirmatory results and outside the Holm family. Exploratory = labelled not pre-registered
in the paper, used for no conclusion (SI-82's wording): any analysis not written in this file.

## 1. Shared rules

### 1.1 Inputs and order

- The resolved follow-up manifest (after `freeze_matched_lambda.py --stage x1 --precheck-decision <decision.json>`, section 1.10),
  the X6 score directories and one failures table covering every X6 row, read through `prereg_rules.outcomes` (same provenance
  and kBET-seed checks as X1). The score files and the failures table must hold X6 rows only: any score or failure row whose tag is not
  an X6 tag of the manifest stops the script (the blinding rule SI-78 made a check). A row neither scored nor failed stops the
  analysis. The analysis runs once, after every X6 row is scored or failed.
- From the R4 stage-x1 record only `targets.<task>|<cond>` (b0, b\*, b_common, `bstar_low_signal`) is used. No X1 score row is
  read. The R4-x1 record is itself built from all X1 scores, so the X6 analysis runs after the X1 analysis. The penalty strengths
  enter through the resolved manifest (`lam`), not through the record's `cells`; for the arm `discriminator_sn_critic_opt` and for
  `pooled_sn` at 1 step they include the pre-check shift of each arm's own decision (section 1.10; `pooled_sn` at 5 steps has none).
- The analysis stops on any departure from the X6 design: missing task x decoder cells; seeds other than 10-12; a tag with no lambda
  label among `matched_lo`, `matched` and `matched_hi` (a strength-0 tag stops it); `extra` other than `{"r1_gamma": 10}` on a
  `discriminator_r1` row or other than empty on any other row; a set of arm x step settings other than the 9 of section 3.2; a missing
  point (a strength label with no row, or several strength values under one label); a point that does not hold exactly seeds 10-12;
  a score or failure row that is not an X6 row; an existing output record (section 1.9 lists the names).

### 1.2 Points and failed runs

A point is the mean B and mean C over its successful runs (ANALYSIS_SPEC section 2.4; SI-71). Failed runs (status `diverged` or
`nonfinite_latent`) are left out of every endpoint and counted per point (section 1.8). A point whose 3 runs all failed is skipped
where the endpoint allows it: the b\* reading walks over it, and the window rule (section 3.5) counts it in
`n_all_failed_between` when it lies between the window's ends. An arm with no successful run at any of its three strengths in a cell
has the status `no_points` there; every contrast that uses it is not evaluable in that cell, with the reason `no_points`, and the
analysis continues with the other cells and arms.

### 1.3 Uncertainty within a dataset

Average first, then leave one seed out (SI-69): every endpoint of a dataset is computed on the point means of seeds 10-12 and then
3 more times, each time without one seed (the same seed removed from every arm). Its standard error is the jackknife
s^2 = ((m - 1)/m) x sum_i (d_(i) - mean d_(.))^2 over the m readings that exist. For a mean of paired differences this
equals SD/sqrt(3). **Agent default:** all 3 readings must exist (m = 3); X1's rule "at least 4 of 5" allows one missing reading,
which with 3 seeds would leave a jackknife over 2 values. Per-dataset intervals in the sign tables are d ± t(0.975, m - 1) s, here
t with 2 degrees of freedom = 4.30: each dataset's interval is wide, and no per-dataset significance claim is made (as X1). The sign
table's sign is 0 when |d| is at most the margin r of section 1.6 (0.01) and otherwise the sign of d; it is descriptive
(**agent default**).

### 1.4 Reading biology at b\* (X6)

Rule I048 (NB-20261017-07): "interpolate biology at b\* only where the cell's own fits bracket b\*; otherwise mark the cell 'not
evaluable at b\*' and report it at fixed strength; never call a single-strength contrast 'matched batch removal'".

- *Where b\* is read.* X6 runs on full X1 tasks with the lo / matched / hi window, so the X1 target b\* of the
  same task and decoder applies. bio@b\* is read on the curve of the three point means in increasing lambda with
  `prereg_rules.bio_at_bstar`: the first point equal to b\* (within 1e-9) or the first adjacent pair that brackets b\* (linear
  interpolation in B). This is the R2/R3 reading of PREREG section 3, not P1's: X6 has no strength-0 rows on seeds 10-12,
  so the curve has no anchor at (b0, c0). The status of an arm's curve in a cell is `bracketed`, `unreached_low` (every point
  has B below b\*), `unreached_high` (every point has B above b\*) or `no_points` (no successful run).
- *Not evaluable.* A contrast is not evaluable at b\* in a cell when the cell is low-signal (SI-56; reason `low_signal`), when
  either arm's curve does not bracket b\* (the reason is the status of the first such arm, the minuend first: `unreached_low`,
  `unreached_high` or `no_points`), or when fewer than 3 LOSO readings exist (reason `readings`). Every not-evaluable cell is listed
  with its reason.
- *Fixed-strength report.* Every contrast of every cell is also reported at fixed strength, whether or not it is evaluable at b\*:
  record key `fixed_strength`, label "fixed-strength, not matched batch removal", holding the C difference and the B difference of
  the two arms at their `matched` strengths. **The two arms of a contrast sit at their own inherited strengths**, which in general
  differ (for the confirmatory pair: the strength of `pooled_sn` at 1 step, which is the pooled critic's `matched` strength moved by
  that arm's pre-check decision, against the strength of `discriminator_sn_critic_opt`, which is the plain discriminator's `matched`
  strength moved by that arm's own pre-check decision, section 1.10), so this difference is not a matched comparison. The achieved batch score minus b\* is reported for every point (`B_minus_bstar`).

### 1.5 Combining datasets

The model of ANALYSIS_SPEC section 4, unchanged and called through `analysis_x1.combine`: d_t ~ N(theta_t, s_t^2), theta_t = Delta +
u_family(t) + v_t, priors Delta ~ N(0, 0.1^2), tau_f, tau_d ~ half-normal(0.1), the same deterministic grid, the 1e-6 floor on s
with the `raised` list (SI-83). Only evaluable real datasets enter; sim1 and sim2 are reported per task and never pooled (X6 has no
simulated task). One result per decoder, never pooled across decoders (SI-70, now for every follow-up with both decoders, SI-93). The
families of `prereg_rules.TASK_FAMILY` apply; X6 pools atac_small, immune and pancreas (three families).

**What the pool can declare (ANALYSIS_SPEC section 15.4, AN-R12).** With one dataset P(Delta > 0.01) did not exceed 0.82, with two
datasets of different families 0.94, so a direction cannot be declared from 1 or 2 datasets at the 0.95 threshold however large the
difference; P(|Delta| <= 0.01) did not exceed 0.15 and 0.33, so equivalence cannot be declared either. Three datasets need a
difference of at least 0.03 to 0.08, depending on s. Consequence, stated before any data: X6 can reach a declaration only with its
3 datasets; a pool reduced to 1 or 2 by a low-signal cell is inconclusive by construction.

**User answer, 2026-10-17 (question 1): report per dataset, no verdict.** The registered pooled model is kept and its
posterior reported, but whenever a pool has 1 or 2 evaluable real datasets the record's declaration is
`not_decidable_1_2_real_datasets` (the model's own output stays in `model_declaration`) and the text says "not decidable with
1-2 real datasets"; the result is reported through the per-dataset sign table (estimate and interval of every dataset). No
per-dataset decision rule is added and no dataset is added. The answer was given for X7, X8 and X15 (dropped by SI-103); it now
applies when a low-signal cell leaves 2 datasets in an X6 pool. It changes no declaration, because with 1 or 2 datasets the model
cannot declare (above).

### 1.6 Margin

r = 0.01 (PREREG section 1, delta_min; ANALYSIS_SPEC section 5) for every endpoint, in the endpoint's own unit: biology score for
X6. **Agent default** (no decision names a follow-up margin; one constant keeps one rule).

### 1.7 Declarations and multiplicity

Declarations follow ANALYSIS_SPEC section 5: a direction if P(Delta > r) >= 0.95 or P(Delta < -r) >= 0.95, practically
equivalent if P(|Delta| <= r) >= 0.95, otherwise inconclusive; each on its own posterior. A pool with no evaluable real dataset has
the declaration `not_evaluable` (the label `analysis_x1.combine` returns for an empty pool; it has no posterior and no p-value). The
confirmatory results are 2: the divergence contrast (section 3.4), one per decoder. The report states that 2 were judged (before the
pruning of SI-103 the draft had 12, 10 of them in the dropped experiments). Frequentist companion: the exact two-sided Wilcoxon
signed-rank test over the pooled datasets' d, with Holm's adjustment over the 2 (as ANALYSIS_SPEC section 5, whose X1 Holm family is
the frozen six primary results: P1, the trade-off area and P2, each per decoder; SI-80, `analysis_x1.py`:523-527); a result with no
p-value (no evaluable real dataset) enters the adjustment with p = 1, so the family size stays m = 2 (recorded as `holm_m`) and its
adjusted value is recorded as null. It cannot reject: the smallest attainable p is 2/2^n = 0.25 with 3 datasets, 0.5 with 2 and 1
with 1. It is reported because it is registered, and used for no conclusion. Secondary results are declared with the same rule and
are not in the Holm family.

### 1.8 Collapse and failure (the P2 definition)

As ANALYSIS_SPEC section 6 and SI-75: a run collapses when it failed or when its C is at least 0.10 below the C of the same seed's
strength-0 run of the same design point, within 1e-9. X6 has no strength-0 rows on seeds 10-12, so only failure counts are
reported for it (per point, `n_fail`; labelled failures, not collapses, and not split by failure type); a strength-0 anchor from X1
seeds would mix seeds and is not used. **Agent default.** No pooled collapse comparison is registered for the follow-ups (P2's pooled
comparison is an X1 result).

### 1.9 What the script reads and which names it hard-codes (manifest v5)

`scripts/analysis_followups.py` imports two frozen modules and changes neither: `analysis_x1` (frozen at the tag
`prereg-tier12-v3`; it uses `combine`, `window`, `wilcoxon_exact`, `holm` and `PRIOR_SCALE`) and `prereg_rules` (it uses
`read_manifest`, `check_seed_separation`, `read_scores`, `read_failures`, `outcomes`, `bio_at_bstar`, `read_record`, `write_record`,
`DELTA_MIN`, `REAL_TASKS` and the error helper `_require`; this file differs from the tag in two lines, see the PREREG entry of
2026-10-20). It does not import `build_paper_manifest`. The names below are hard-coded in the script and checked against the
manifest rows; they were read from `scripts/build_paper_manifest.py` and manifest v5 when this file was written, and any difference
stops the script.

| Item | Hard-coded form | Name in the script |
|---|---|---|
| strength label | read from the tag: `_l(matched_lo\|matched_hi\|matched)_c<cond>_s<seed>_<hash8>`, written before R4-x1 resolves `lam`; no strength-0 label | `TAG_LEVEL`, `LEVELS` |
| cells | atac_small, immune, pancreas, each with cond 0 and cond 1 (SI-93) | `EXPECTED` |
| seeds | exactly 10, 11, 12 in every point (SI-26) | `FU_SEEDS` |
| `extra` | `{"r1_gamma": 10}` on every `discriminator_r1` row, empty on every other row | `R1_EXTRA` |
| arm x steps | (discriminator,1), (discriminator_sn,1), (discriminator_r1,1), (`discriminator_sn_critic_opt`,1), (pooled,1/5/10), (pooled_sn,1/5), each at lo / matched / hi | `X6_CONFIGS`, `X6_COPT` |
| contrasts | the 11 of sections 3.4 and 3.5 | `X6_CONTRASTS` |

Manifest v5 (`manifests/paper_manifest_stock_pilot_u5_b10_v5.tsv`, SHA-256 in the `.sha256` file next to it, checked) holds 486 X6 rows
(243 per decoder), 81 per decoder and strength label (3 tasks x 9 settings x 3 seeds). The test `test_v5_rows_enumerated` reads every
X6 row of manifest v5 through the analysis and checks that count against this hand count.

### 1.10 Strengths of the two arms of the confirmatory contrast: the pre-check (user decisions, 2026-10-20 and 2026-10-21)

The arm `discriminator_sn_critic_opt` takes the matched strengths of the plain discriminator, which is trained at a ten times higher
learning rate; if the slower-learning arm removes less batch effect, its batch score never reaches b\*, and the confirmatory contrast
cannot be read at b\*. The other arm, `pooled_sn` at 1 step, takes the matched strengths of the main pooled critic (gradient penalty,
5 steps), was never piloted, and can miss b\* in the same way (user decision 2026-10-21, SI-107). The pre-check
(`experiments/x6_precheck_20261020/PLAN.md` and its addendum of 2026-10-21, rule fixed before any pre-check fit) fits both arms
on the pilot tasks (36 fits: 18 per arm, pilot seed 100) and writes `decision.json`, whose key `decisions` gives each arm its own
decision: `keep`, `shift_up` or `shift_down`. `freeze_matched_lambda.py --stage x1` requires the file (`--precheck-decision`) when it
resolves X6 rows of either arm and stops without it, with a file that lacks the key or an arm, or with any other value. For each of the two
arms, each of its three resolved strengths then moves one step up (`shift_up`) or down (`shift_down`) the sequence ..., 0.1, 0.3, 1, 3,
10, ... (`prereg_rules.lam_step`) by that arm's own decision; `keep` leaves them unchanged. `pooled_sn` at 5 steps is not checked and
not moved, and no other arm moves. The R4-x1 record holds the decisions (`precheck_decisions`), the number of rows each arm's decision
applies to (`precheck_arm_rows`) and the SHA-256 of the file. The analysis does not read the pre-check: it reads the shifted strengths
from the manifest. The decisions stay what the rule gave, whatever X1 shows; no second pre-check is run. If even a shifted window does not
bracket b\*, the contrast is not evaluable in that cell (section 1.4) and is reported at fixed strength.

### 1.11 Departures from ANALYSIS_SPEC

The follow-up analysis reuses the frozen rules, with these differences (none is a sensitivity analysis):

1. All 3 leave-one-seed-out readings must exist; X1 needs 4 of 5 (section 1.3).
2. **No sensitivity analysis is registered.** ANALYSIS_SPEC section 11 and 15.11 register them for X1: biology without the trajectory
   score (immune is a trajectory task), the per-run reading, dropping points with a failed run, prior scales 0.05 and 0.2, the
   pairs-first and the independent dataset structures. None is registered for X6; any of them run later is exploratory and labelled so.
3. The reading of a low-signal cell is not computed; ANALYSIS_SPEC section 2.3 shows such readings in a separate table. X6 reports the
   cell as not evaluable (reason `low_signal`) with its fixed-strength differences (section 1.4).
4. Further crossings of b\* (a curve that falls back past its batch peak) are not counted or reported; ANALYSIS_SPEC section 2.2
   counts and reports them. With 3 points per curve the first crossing is the only one that can be read.
5. The b\* curve has 3 points and no strength-0 anchor (section 1.4); P1's curve has the anchor and six points.
6. Per-dataset intervals use t with 2 degrees of freedom (section 1.3).
7. No pooled collapse comparison (section 1.8).
8. The sign table has the margin of section 1.3 (agent default).

Not a departure: the window width is computed by the frozen `analysis_x1.window` (section 3.5).

## 3. X6: divergence vs regularisation vs adversary steps (R3 factor effects)

**3.1 Question.** Is a critic-discriminator difference in biology at matched batch removal due to the divergence (W1 vs JS), or to
the Lipschitz control (gradient penalty, spectral normalisation, R1), the optimiser, or the number of adversary steps per update?

**3.2 Configurations.** JS with no penalty, SN, R1 (weight gamma = 10) and SN with the critic's Adam settings (SI-98), each at 1 step;
the pooled W1 critic with GP at 1, 5, 10 steps and with SN at 1, 5 steps (spectral-norm critics compute no gradient penalty: the
I036 fix, in the follow-up code tag). Each at lo/matched/hi.

**3.3 Estimand.** For a contrast (a, b): d = bio@b\*(a) - bio@b\*(b) (section 1.4), per task and decoder.

**3.4 Confirmatory endpoint.** The divergence contrast with matched optimiser: W1+SN at 1 step (`pooled_sn`, n_critic 1) minus
JS+SN with the critic's optimiser (`discriminator_sn_critic_opt`), per decoder: 2 results. With the I036 fix and SI-98 these two differ
only in the loss (both spectrally normalised, both 1 step, both Adam lr 1e-4, betas (0, 0.9)), and the target (the pooled
critic's symmetric target versus the discriminator's batch classifier) as stated in 3.7. **Agent default** that this, not the
own-optimiser contrast, is the confirmatory one: SI-98 was added to remove the optimiser confound (brief Q6).

**3.5 Secondary contrasts (the identified contrasts of I087).** Divergence with each family's own optimiser (W1+SN 1 step minus
JS+SN; `divergence_own_optimiser`); optimiser within JS+SN (critic's settings minus the discriminator's; `js_optimiser`); JS
regularisation (SN minus none; R1 minus none); W1 regularisation (SN minus GP at 1 and at 5 steps); update budget (GP 5 minus 1,
GP 10 minus 1, SN 5 minus 1); and, as in the main benchmark's critic, W1 with GP at 5 steps (X1's critics take 5 steps) minus JS at 1
step (`as_used`: "W1 critic with gradient penalty at 5 steps, as in the main benchmark, minus JS at 1 step"). Each pooled per decoder
with the same rule; not in the Holm family. Also the R3 window width on the biology score (D-09), computed by the frozen
`analysis_x1.window` (ANALYSIS_SPEC sections 10 and 15.10): the best point maximises the mean C, the window holds the points whose
mean C is at least the lower end of the two-sided 95% t interval (t with n - 1 degrees of freedom, n = the successful runs of the
best point) of the best point's mean; width = log10(largest lambda) - log10(smallest), over at most 3 points, descriptive. The record
gives the function's fields: `status` (`ok`, `no_interval` when the best point has fewer than 2 successful runs, `no_points`),
`best`, `lower`, `lams`, `width`, `n_points`, `gap` (a point between the window's ends lies below the bound) and
`n_all_failed_between` (points with no successful run between the window's ends).

**3.6 Not evaluable; pooling.** Section 1.4 rules (low-signal; either curve not bracketing b\*; fewer than 3 readings); the
fixed-strength report is given for every cell. Real datasets atac_small, immune, pancreas pooled; with the conditioned decoder the
immune cell was low-signal in the A1 pilot (SI-56), so that pool may hold 2 datasets and cannot then declare (section 1.5).

**3.7 Confirm / refute; disclosed limits.** *The reading,* made per decoder and only for the contrasts of section 3.5 named here.
The divergence explains the difference between critic and discriminator in the main benchmark if all three hold: (a) P1 for that
decoder has a declared direction (ANALYSIS_SPEC section 5); (b) the confirmatory contrast has a declared direction with the same sign
as P1's (both are critic minus discriminator); (c) the regularisation contrasts (JS: SN minus none, R1 minus none; W1: SN minus GP at 1
and at 5 steps) and the step contrasts (GP 5 minus 1, GP 10 minus 1, SN 5 minus 1) are each smaller, meaning that the absolute value
of the pooled posterior mean is below the absolute value of the confirmatory contrast's pooled posterior mean; a contrast with no
pooled estimate counts as not smaller. If P1 for that decoder is inconclusive, practically equivalent or not evaluable, the "explains"
reading is not made, and X6 is reported by its own sign: the confirmatory contrast's declaration and nothing about the X1 result.
If the confirmatory contrast itself is inconclusive, not decidable or not evaluable, the reading is not made either. This joint
reading is descriptive (only the divergence contrast is declared), and it is not computed by the script: it is made from the
record and P1 when the results are written up. *Disclosed limits:* X6 is an incomplete factorial; R1 (penalty on data points) and
GP (penalty on interpolates) differ in form; per-batch versus per-cell weighting was not probed; the pooled critic and the
discriminator differ in target as well as divergence; the X6 critic is the pooled one, while P1 uses the reference critic.

## 8. What would confirm or refute each claim (summary)

| Claim (original paper or reviewer) | Endpoint | Supports | Contradicts | Can be declared with this design? |
|---|---|---|---|---|
| The difference is due to the divergence | X6 divergence contrast d | direction declared with the sign of P1 (itself declared), regularisation and step contrasts smaller (section 3.7) | equivalence, or opposite sign | yes (3 datasets; 2 if immune is low-signal: no) |

The claims on composition shift (X3), a dense reference (X7), batch-count scaling (X8), architecture (X12) and KL warm-up (X15) are not
tested (SI-103); the original claims that the critic scales poorly with batch count and that it depends on a dense
reference are removed.

## 9. Limitations stated with the results

1. Three seeds per point: per-dataset intervals use t with 2 degrees of freedom; the jackknife of a non-smooth reading (a crossing)
   from 3 values is rougher than X1's from 5 (ANALYSIS_SPEC section 16, limitation 1, applies more strongly).
2. Pooled declarations need 3 or more real datasets (section 1.5); X6's conditioned-decoder pool may fall to 2 if immune is low-signal.
3. The X6 strengths come from X1 (R4-x1): a variant arm inherits its base arm's strengths, so its batch removal can land off b\*;
   for `discriminator_sn_critic_opt` and for `pooled_sn` at 1 step the pre-check (section 1.10) moves them, each by its own decision, by at
   most one step, from a single-seed pilot (`pooled_sn` at 5 steps is not checked and not moved); where a window does not bracket b\*
   the contrast is reported at fixed strength only (I048).
4. X6: incomplete factorial and the target difference between pooled critic and discriminator (section 3.7).
5. X6 with the unconditioned decoder (SI-93) uses the same rules; low-signal cells (SI-56) drop out per decoder.
6. Scope (SI-103): X3, X7, X8, X12 and X15 are not run. Architecture and KL warm-up are not varied, and reference choice,
   batch count and composition shift are not tested; the paper states these as scope limitations.
7. No sensitivity analysis is registered for X6 (section 1.11).

## 10. Implementation, tests and lock

- `scripts/analysis_followups.py` (usage at its top). It imports `analysis_x1` and `prereg_rules` (names in section 1.9) and changes
  neither; it imports no other project module itself (`prereg_rules` imports `build_paper_manifest`, for the manifest columns and
  grid), and no X1 fit or analysis path is touched. The design names it checks (cells,
  seeds, `extra`, arm x step settings, contrasts, strength labels) are hard-coded in it (section 1.9). One JSON record per run
  (inputs with SHA-256, git state, constants, every per-cell quantity, the not-evaluable reasons, the 2 confirmatory declarations
  with Wilcoxon and Holm values); it refuses to overwrite an existing record. Rows are sorted by tag before any computation.
- Tests, `tests/analysis_followups/test_analysis_followups.py` (synthetic scores on the real manifest-v5 X6 rows, no X1 or X13
  score): the X6 row enumeration (section 1.9); known answers (jackknife of a mean = SD/sqrt(3), Holm with a missing p); a
  planted-effect world run end to end through `main` in which the divergence contrast (+0.07, declared above, with JS+SN given
  different biology under the two optimisers so the confirmatory subtrahend is pinned: own-optimiser contrast +0.05) recovers its
  planted value, with one failed run, one point whose 3 runs all failed, one arm whose 9 runs all failed in a cell (status and
  window `no_points`, the analysis continues), a curve below and a curve above b\* (`unreached_low`, `unreached_high` as the contrast
  reasons), the fixed-strength values and label, and the Wilcoxon and Holm-adjusted p-values; a second record with every
  unconditioned-decoder cell low-signal (a pool with no evaluable dataset: declaration `not_evaluable`, p = 1 in the Holm family); a
  null world in which no confirmatory result declares a direction; refusals of departures from the design (cells, seeds, `extra` in
  three forms, a strength-0 tag, a tag without a label, arms, a missing point); refusal of score or failure rows of another experiment;
  no overwrite. The pre-check shift of section 1.10 is tested in `tests/prereg/test_cli_e2e.py` (both arms, each taking its own
  decision, `pooled_sn` at 5 steps unmoved, a missing decision, an invalid decision, a decision file without the key or an arm, the
  record and the file's SHA-256) and in `tests/x6_precheck/` (the 36-row manifest and the per-arm decisions).
- Command, run on 2026-10-20 from the repo root with the environment `scvi-api` (`/home/kendall/.claude-science/conda/envs/scvi-api`):
  `python -m pytest -q tests/analysis_followups tests/x6_precheck tests/prereg`. Output: `168 passed, 1 skipped in 70.22s
  (0:01:10)`. Fewest-tests check, `python scripts/check_tests.py --base d034db7 --src scripts -- tests/analysis_followups
  tests/x6_precheck tests/prereg`: `168 passed, 1 skipped in 135.89s (0:02:15)` and `check_tests: OK (every changed --src line and
  branch is tested; no redundant new tests; base d034db7)`. Both runs were made on the tree whose `scripts/` and `tests/` files are
  those of the commit that records this output; the later edits to this file change no code.
- Lock: this file, the script and the tests are frozen by the follow-up code tag `prereg-x6-v1`, which the user makes after review, after
  the X1 fit-identity check and before R4-x1. The PREREG section 9 entry of 2026-10-20 puts the design in force and lists the files
  that differ from the tag `prereg-tier12-v3`.

## 11. Choices made by the agent (no prior decision settles them; SI-92 defaults, each listed for review)

1. All 3 LOSO readings required (section 1.3).
2. Margin r = 0.01 in the biology score (1.6).
3. Collapse only where a same-design strength-0 run exists; failure counts elsewhere (1.8).
4. X6: the optimiser-matched divergence contrast is confirmatory (3.4).
5. Holm family of 2 (the divergence contrast, per decoder) after the pruning; the draft's family of 12 included 10 results of dropped
   experiments (1.7). A consequence of SI-103, not a separate user decision.
6. The sign table's zero band of width r (1.3).
7. The "explains" reading of 3.7: which contrasts count, "smaller" as the absolute pooled posterior mean, and no reading when P1 or
   the confirmatory contrast has no declared direction.
8. No sensitivity analysis, no low-signal readings, no extra crossings (1.11).

## 12. Questions put to the user (answered 2026-10-17)

1. Follow-ups with fewer than 3 real datasets (X7, X8, X15): answered "Report per dataset, no verdict (Recommended)": the
   registered pooled model is kept, per-dataset estimates are reported with intervals, and the result is stated as "not decidable
   with 1-2 real datasets"; no per-dataset decision rule and no added datasets (section 1.5). X7, X8 and X15 are since dropped
   (SI-103); the answer now covers an X6 pool reduced to 1-2 datasets.
2. X7's confirmatory P4 result: answered "Depleted-reference comparison (Recommended)" (SI-102). Moot: X7 is dropped (SI-103).

## 13. For the lead

- Review this file, the script and the PREREG entry of 2026-10-20; then run the X1 fit-identity check on the commit to be tagged
  (`experiments/followup_build_20261017/README.md`) and make the tag `prereg-x6-v1` on that commit, before R4-x1.
- The strength pre-check (section 1.10) must have produced its `decision.json`, with a decision for each of its two arms, before R4-x1
  resolves the X6 rows.
