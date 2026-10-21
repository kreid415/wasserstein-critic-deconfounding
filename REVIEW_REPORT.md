# Pre-lock review of the X6 follow-up analysis rule

Reviewer: Rigor Reviewer (read-only), 2026-10-20.
Object: the X6 follow-up analysis. X6 asks whether the difference between the Wasserstein-1 (W1) critic and the Jensen-Shannon (JS)
discriminator comes from the divergence itself, from regularisation (spectral normalisation, SN; gradient penalty, GP; the R1
penalty on data points) or from the number of adversary update steps.
Checkouts read: `fas/` (branch followup-analysis-spec), `fb/` (branch followup-build), `frf/` (branch final-review-fixes, the frozen
main-study rules). No file under them was changed. No test was run, because shell commands were not available in this session.
No network command was run. No main-benchmark (X1) or baseline (X13) score, failure or status file was opened.

**Verdict: FIX-FIRST.** 3 BLOCKER, 7 SHOULD-FIX, 9 NOTE, 1 open question.

Terms used below. *b\** = the matched batch-removal target per task and decoder, read from the frozen record of rule R4 on X1
(R4 picks the matched penalty strength). *lo / matched / hi* = the three penalty strengths (lambda) R4 assigns to each X1 arm, the
point closest to b\* and its two grid neighbours. A variant arm takes its base arm's three strengths. *LOSO* = leave-one-seed-out.
*Holm* = Holm's step-down adjustment of p-values for multiple tests. *Low-signal* = a cell where the common batch range
b_common - b0 is below 0.02 (decision SI-56). *Cell* = task x decoder. *cond 1 / cond 0* = decoder that receives the batch label /
decoder without it. *P1* = the main-study primary endpoint: biology at b\* for the reference W1 critic minus the discriminator.
*SI-nn* = entry nn of the project's decision ledger (CONSTRAINTS.md).

---

## 1. Findings, most severe first

### BLOCKER 1. The design the spec locks is not yet in force in the pre-registration
- **Spec side.** `fas/docs/ANALYSIS_SPEC_FOLLOWUPS.md:33-34`: "variant arms inherit their base arm's values, PREREG line 98".
  `:149`: the arm list includes `discriminator_sn_critic_opt`. `:161-163`: "spectral-norm critics compute no gradient penalty
  once the I036 fix is on the follow-up tag". `:214-216`: the script "imports ... `prereg_rules` ... and changes neither".
- **Frozen side.** `frf/docs/PREREG.md:98` lists the variant arms and their base arms. `discriminator_sn_critic_opt` is not in the
  list, and the same line says "An arm without a mapping stops the script." `frf/scripts/prereg_rules.py:59-61`: `BASE_ARM` has no
  `discriminator_sn_critic_opt` entry. `fas/scripts/prereg_rules.py:62` and `fb/scripts/prereg_rules.py:62` add it, so the
  `prereg_rules.py` the analysis imports differs from the frozen one.
- **Amendment state.** `fb/docs/prereg/amendment_20261017_followups.md:3` says it is a "draft, not in force". It covers SI-93 (both
  decoders), SI-98 (the new arm and its base-arm mapping, line 30) and I036 (no gradient-penalty pass for SN critics, line 26). It
  still describes X3, X7, X8 and X12. The in-force SI-103 pruning entry (`frf/docs/PREREG.md:663-694`) only says X6 is kept "as
  designed in SI-93 and SI-98" (`:674-675`). It does not register the base-arm mapping, the cond 0 cells or the I036 change as rule
  text. The spec's own `:253-255` says the draft "needs the same pruning".
- **Consequence.** If the lock is made now, the confirmatory arm's strengths come from an unregistered mapping in a changed frozen
  script, and an amendment is needed afterwards.
- **Proposed fix.** Prune the draft amendment to X6 (SI-93 for X6, SI-98, I036, the `BASE_ARM` line). Paste it into PREREG
  section 9. Commit it in the same tree as the lock tag, and correct spec `:214-216` to say which `prereg_rules.py` lines changed.

### BLOCKER 2. The window-width code stops the whole analysis when one arm has no successful run in a cell, and it departs from the frozen window rule
- **Code.** `fas/scripts/analysis_followups.py:152-153`: `rows = [... if len(cc)]`, then `best = max(rows, ...)`. If all 9 runs of
  one arm in one cell failed (status `diverged` or `nonfinite_latent`), `rows` is empty and `max()` raises `ValueError`.
  `x6()` calls it for every arm and cell (`:172`), so the script stops before any confirmatory result is written.
- **Spec.** `:69-71` (section 1.2): a point whose runs all failed "is skipped ... and otherwise makes the endpoint not evaluable
  for that cell, with the reason recorded". It does not say the analysis stops.
- **Departures from the frozen rule.** The spec says "(2 degrees of freedom)" (`:177`). The code uses `len(best[1]) - 1`
  (`:156`), which is 1 degree of freedom when a run of the best point failed. The frozen rule is t with n - 1 degrees of freedom
  (`frf/docs/ANALYSIS_SPEC.md:279`, `:411`). The frozen `analysis_x1.window` (`frf/scripts/analysis_x1.py:343-365`) also returns
  `status="no_points"`, breaks ties within 1e-9 and reports `gap` and `n_all_failed_between` (frozen spec `:413-416`, review
  findings AN-14 and AN-R13). `bio_window` does none of these.
- **Proposed fix.** Replace `bio_window` with `A.window([(lam, list(cc.values())) for lam, _, cc, _ in pts])`, the frozen and
  tested function. Change spec 3.5 to "t with n - 1 degrees of freedom, as ANALYSIS_SPEC sections 10 and 15.10".

### BLOCKER 3. Spec section 10, the header and section 1.9 describe code that no longer exists, and the tests were not re-run after the cut
- **Spec says the script is not cut yet.** `:219-221`: "the script and its tests ... still contain the code ... of the dropped
  experiments, and `CONFIRMATORY` and `EXPECTED` still list them. They must be cut to X6". The script is already cut:
  `analysis_followups.py:31-33` has `EXPECTED` with X6 only, and `:189` has `CONFIRMATORY` with 2 entries.
- **Spec says the script reads design names from the build; it does not.** `:16-17`: "the script now reads every design name
  from `scripts/build_paper_manifest.py` and the manifest instead of assuming it". `:143-149`: a "Read from" column naming
  `build_paper_manifest.finalize` and `BASE_ARM`. `:214-215`: the script "imports `build_paper_manifest` ... `clopper_pearson`".
  The script imports neither (`:12-23`). It hard-codes `FU_SEEDS`, `EXPECTED`, `EXTRA_KEYS` and `X6_CONFIGS` (`:25-43`).
- **Tests not re-run.** `:228-229`: "13 passed; not re-run after the pruning". The current test file has 10 test cases
  (`test_known_answers`, `test_planted_effects`, `test_null_world`, `test_v4_rows_enumerated`, 5 `test_refusals` cases,
  `test_main_refuses_overwrite`). No result on the cut code is recorded.
- **Proposed fix.** Rewrite the header, section 1.9 and section 10 to describe the cut script as it is. Run the section 10 command
  (`python scripts/check_tests.py --base origin/followup-build --src scripts -- tests/analysis_followups`) on the exact tree to be
  tagged, and record its output in the spec before tagging.

### SHOULD-FIX 1. The "as used in X1" secondary contrast is mislabelled
- **Spec and code.** Spec `:176`: "as used in X1 (W1+GP at 1 step minus JS at 1 step)". Script `:43`: `"as_used": ("w1_gp_1", "js")`.
- **Design.** X1's critics take 5 steps. `fb/scripts/build_paper_manifest.py:85` has `N_CRITIC = 5`. Manifest v5 X1 rows have
  `n_critic` 5 for pooled and reference (e.g. `fb/manifests/paper_manifest_stock_pilot_u5_b10_v5.tsv:3653`) and 1 for the
  discriminator (`:3641`).
- **Proposed fix.** Define `as_used = ("w1_gp_5", "js")`, or keep the definition and rename it "W1+GP at 1 step minus JS at 1 step".

### SHOULD-FIX 2. The confirm/refute reading depends on X1's P1 sign without saying what happens when P1 has no declared sign
- **Spec.** `:184-186`: "The divergence explains the difference if the confirmatory contrast has a declared direction with the
  same sign as X1's P1 for that decoder and the regularisation and step contrasts are smaller". `:194`: "Supports: direction
  declared with P1's sign".
- **Gaps.** No rule covers the cases where P1 for that decoder is inconclusive, equivalent or not evaluable. "Smaller" is
  undefined: smaller in absolute posterior mean, or in the interval?
- **Code.** The script does not read P1, and this reading is in no output, so it lives only in the text.
- **Proposed fix.** Add one sentence for each P1 case (for example: "if P1 declares no direction, X6 is reported by its own sign
  and the 'explains' reading is not made"). Define "smaller" as |posterior mean| below the confirmatory contrast's |posterior mean|.

### SHOULD-FIX 3. Silent departures from the frozen main-study analysis
The spec says it "reuses its rules and says where it departs from them" (`:11-12`). These departures are not stated:
- **Sensitivity analyses.** The frozen `ANALYSIS_SPEC.md:283-292` registers biology without the trajectory score (immune is a
  trajectory task, `frf/scripts/prereg_rules.py:65`), the per-run reading, dropping points with a failed run, prior scales 0.05 and 0.2, and
  the pairs-first and independent dataset structures. The follow-up spec registers none of them and does not say so. Any of them
  added later would be exploratory only.
- **Readings of low-signal cells.** Frozen `ANALYSIS_SPEC.md:123-124`: "Readings of low-signal cells are shown in a separate
  table". The follow-up code returns `d=None` and computes no reading for a low-signal cell (`analysis_followups.py:140-142`).
- **Extra crossings.** Frozen `ANALYSIS_SPEC.md:115-116`: further crossings of b\* "are counted and reported". The follow-up code
  counts none.
- **Proposed fix.** Add a "Departures from ANALYSIS_SPEC" list. It should say that no sensitivity analysis is registered (or
  register the trajectory-free one for immune) and that low-signal readings and crossings are not reported (or compute them).

### SHOULD-FIX 4. Stop conditions are less complete than spec section 1.1 says
- **Spec.** `:62-64`: the analysis "stops on any departure from the X6 design".
- **Lambda = 0 tags pass.** `TAG_LEVEL` (`:44`) accepts `_l0_` tags, a leftover from the dropped X3 and X8, which had lambda = 0
  rows. `window_points` (`:113-114`) reads only the three matched labels, so an X6 lambda = 0 row of a known arm would be ignored
  without a stop.
- **R1 penalty weight not checked.** The script checks only the `extra` key name (`:57-59`), not its value. The design is gamma 10
  (spec `:161`; all 54 R1 rows in v5 carry `{"r1_gamma": 10}`), and gamma is also not tied to the `discriminator_r1` rows.
- **Resolved strengths not checked.** The resolved lambda values are never compared with the R4 record's `cells` values.
- **Proposed fix.** Drop `|0` from `TAG_LEVEL`. Require `extra == {"r1_gamma": 10}` exactly on `discriminator_r1` rows and `{}`
  elsewhere. Optionally, check each resolved lambda against `r4["cells"][task|cond|BASE_ARM]`.

### SHOULD-FIX 5. Confirmatory-path rules that no test covers
The existing coverage is in the table in section 3 below. These are uncovered:
- **Fixed-strength report.** The `fixed_matched` values, required for every not-evaluable cell (spec `:93-94`), are never asserted.
- **Missing point.** The "a missing point" stop (spec `:64`; code `:115`) is never triggered.
- **`unreached_high`.** Only `unreached_low` is tested (test `:115`).
- **All runs of a point failed.** No test has a point whose 3 runs all failed, so the b\* reading that skips over it (spec
  `:70`) is untested. The world has one failed run (test `:64`).
- **Holm values.** Adjusted p-values are never asserted end to end; only `holm_m == 2` is (test `:118`). Holm with a missing p is
  tested in a unit test only (test `:102`).
- **A pool with no evaluable dataset.** A pool with 0 evaluable real datasets (declaration `not_evaluable`, p entering Holm as 1)
  is never run end to end.
- **Proposed fix.** Extend `world()` in `test_planted_effects`. Fail all 9 runs of one arm in one cell (this also catches
  BLOCKER 2), give one curve B above b\*, and mark all three cond 0 cells low-signal in a second record. Then assert
  `fixed_matched`, `unreached_high`, `holm_p` and `not_evaluable`. Add a "missing point" case to `test_refusals`.

### SHOULD-FIX 6. The fixed-strength output is named "matched" and has no label
- **Spec.** `:84-85` (rule I048): "never call a single-strength contrast 'matched batch removal'". `:93-94`: "labelled fixed-strength".
- **Code.** The record key is `fixed_matched` (`:178`) and has no label field.
- **Strengths differ between arms.** The fixed-strength d compares each arm at its own inherited `matched` lambda. For the
  confirmatory pair, these are the pooled critic's strength and the discriminator's strength. The spec does not say this.
- **Proposed fix.** Rename the key to `fixed_strength`, add `label="fixed-strength, not matched batch removal"`, and state in
  spec 1.4 that the two arms sit at different lambda values.

### SHOULD-FIX 7. Not-evaluable reasons are coarser than the spec's statuses
- **Spec.** `:92`: "(status `unreached_low`, `unreached_high`)".
- **Code.** The contrast reason is always `"unreached"` (`:145-146`). It also overwrites `"not_readable"`, so a contrast with no
  successful run anywhere is recorded as "unreached". The per-arm status (`:172`) can also be `no_points`
  (`prereg_rules.py:614-615`), which the spec never names. The per-arm status does carry `unreached_low` and `unreached_high`.
- **Proposed fix.** Record the contrast reason as the failing arm's status (`unreached_low`, `unreached_high` or `no_points`) and
  add `no_points` to spec 1.4.

### NOTE 1. The declaration label for a pool with zero datasets is not in the spec
`A.combine([])` returns `declaration="not_evaluable"` (`frf/scripts/analysis_x1.py:234-235`). Spec 1.7 (`:130`) names the case
("no evaluable real dataset") but not the label. Add the label to spec 1.7.

### NOTE 2. Stale cross-references and leftovers
- Script `:30` and `:59` cite "spec section 1.4" for names (now 1.9). `:39` cites "spec section 4" (now 3.4). `:9` says "design v4".
- Test docstring `:4` cites "section 1.4".
- Spec `:76` says "and from lambda = 0", but X6 has no lambda = 0 rows.
- The spec 1.9 heading says "manifest v4".
- Fix: update the references in the same commit.

### NOTE 3. Tests read manifest v4, the run uses v5
Test `:29` reads `manifests/paper_manifest_stock_pilot_u5_b10_v4.tsv`, and `fas/` has no v5 file. I counted 486 X6 rows in both v4
and v5, and sampled tags agree (e.g. `X6_atac_small_discriminator_sn_critic_opt_lmatched_c0_s10_39448145`, v4 line 9427 and
v5 line 5035). Byte identity of all 486 rows was not checked (no shell). Point the test at v5 after the merge.

### NOTE 4. Blinding is protected by procedure, not by code
`prereg_rules.read_scores` loads every CSV in each `--scores` directory (`fas/scripts/prereg_rules.py:270-287`), and
`read_failures` loads the whole failures file (`:290-298`), before `outcomes` filters to X6 tags (`:303-311`). A wrong path to an X1
directory would load X1 values into memory. They would not be used or printed. `main` prints only a count (`:227`). The full R4
record is also loaded, including X1 per-arm best batch scores (`targets.M`), but only `bstar` and `bstar_low_signal` are used
(`:168`, `:176`). Fix: refuse any score or failure row whose tag is not an X6 manifest tag.

### NOTE 5. Leftover follow-up list
`prereg_rules.FOLLOWUPS` still lists X3, X7, X8, X12 and X15 (`fas/scripts/prereg_rules.py:70`). With no rows for them in v5, it
changes no behaviour of the X6 analysis.

### NOTE 6. Failure counts are not split by failure type
X6 reports `n_fail` per point (`:116`, `:137`). X1 splits failures into `diverged` and `nonfinite_latent`
(`analysis_x1.py:300-302`). Consider reporting both.

### NOTE 7. The per-dataset sign table uses an undocumented margin
`sign_row` sets the sign to 0 when |d| <= 0.01 (`:106`). The spec does not mention this. Describe it in spec 1.3 or drop it.

### NOTE 8. Claim in spec 3.4 checked against the fit code: holds
The claim is "these two differ only in the loss ... and the target".
- **Same network.** Both arms use the same three-layer head with spectral normalisation on every layer
  (`fb/src/wcd_vae/wcd/adversarial.py:48-52`).
- **One step.** Both take one adversary step (`fb/scripts/scvi_adversarial_plan.py:213-220`).
- **Same optimiser.** Both use Adam with lr 1e-4, betas (0, 0.9), eps 1e-8 and no weight decay
  (`scvi_adversarial_plan.py:266-267`, defaults `:168`; `fit_paper_config.py:422` passes 1e-4; test
  `fb/tests/scvi/test_adversarial_plan_v2.py:72, 77`).
- **No gradient penalty for spectrally normalised arms.** SN critics skip the gradient-penalty pass (`with_gp=not
  self.spectral_norm`, `:410`; zeroed again in `_head`, `:300-301`; test `:79`). The SN discriminator computes none (`:421`).
- **Remaining differences.** The generator term (fool cross-entropy `:441` versus the W1 estimate `:434-435`) is "the loss". The
  lambda window differs because the arms inherit from different base arms (SHOULD-FIX 6).

### NOTE 9. Section and file names in the task brief
`tests/analysis_followups/test_followup_v4.py` does not exist in `fas/`. The file of that name is in `fb/tests/scvi/` (build-side
tests, read only for NOTE 8).

### Open question (cannot be decided from the files)
Will the inherited windows bracket b\*? `discriminator_sn_critic_opt` trains its adversary at lr 1e-4 instead of the plain
discriminator's 1e-3 (`scvi_adversarial_plan.py:267-269`), but at the plain discriminator's three strengths. If its batch removal
lands below the window, it is `unreached_low` in many cells. The confirmatory pools would then be `not_decidable` or
`not_evaluable` by construction. The rule handles this case (it reports per dataset and at fixed strength), but the chance of an
undecidable confirmatory result is not stated. A pre-launch preflight could estimate it from the A1 pilot. It is logged as a
question, not an issue.

---

## 2. Spec versus script, rule by rule (check 1)

| Rule | Spec text (file:line) | Code (file:line) | Verdict |
|---|---|---|---|
| Confirmatory contrast | "W1+SN at 1 step (`pooled_sn`, n_critic 1) minus JS+SN with the critic's optimiser (`discriminator_sn_critic_opt`), per decoder: 2 results" (spec:167-168) | `"divergence": ("w1_sn_1", "js_sn_copt")` (:40); `"w1_sn_1": ("pooled_sn", 1)`, `"js_sn_copt": (X6_COPT, 1)` (:36-38); `CONFIRMATORY = [f"X6|divergence|{c}" for c in (0, 1)]` (:189) | Matches |
| bio at b\*, bracketing only | "the first point equal to b\* (within 1e-9) or the first adjacent pair that brackets b\* (linear interpolation in B)"; no lambda = 0 anchor (spec:88-91) | `read_bstar` → `P.bio_at_bstar(curve, bstar)`, value only if `"bracketed"` (:120-124); curve is the 3 window points only (:122); `bio_at_bstar` (prereg_rules:611-625) | Matches |
| Otherwise fixed strength, labelled | "reported at the fixed `matched` strength (C difference and B difference), labelled fixed-strength" (spec:93-94) | `res["fixed_matched"] = dict(dC=..., dB=...)` for every contrast (:177-179); no label | Partial (SHOULD-FIX 6) |
| unreached_low / unreached_high | "(status `unreached_low`, `unreached_high`)" (spec:92) | per-arm `status` from `bio_at_bstar` (:127-129, :172); contrast `reason="unreached"` (:145-146) | Partial (SHOULD-FIX 7) |
| All 3 LOSO readings | "all 3 readings must exist (m = 3)" (spec:78) | `MIN_READINGS = 3`; `reason = ... "readings" if m < MIN_READINGS` (:26, :83) | Matches |
| Jackknife standard error | "s^2 = ((m - 1)/m) x sum_i (d_(i) - mean d_(.))^2" (spec:77) | `np.sqrt((m - 1) / m * np.sum((existing - existing.mean()) ** 2))` (:84) | Matches |
| Same seed dropped from both arms | "the same seed removed from every arm" (spec:76) | one `keep` list for both arms (:143-144) | Matches |
| Low-signal | "not evaluable at b\* when it is low-signal (SI-56)" (spec:91) | `bstar_contrast(..., tg["bstar_low_signal"])` → `reason="low_signal"` (:140-142, :176) | Matches (departure from X1 in SHOULD-FIX 3) |
| Failed runs | "left out of every endpoint ... A point whose 3 runs all failed is skipped" (spec:69-71) | `seed_values` keeps `status == "ok"` (:67-68); `read_bstar` drops points with no B (:122); `bio_window` raises (:153) | Fails in the window code (BLOCKER 2) |
| Pooling the 3 datasets | "`analysis_x1.combine` ... Only evaluable real datasets enter ... One result per decoder" (spec:98-101) | `A.combine(keep, d, s, "family", A.PRIOR_SCALE, MARGIN)`, keep = real and evaluable (:91-92); loop per decoder (:181-184) | Matches |
| A pool of 1-2 cannot declare | "`not_decidable_1_2_real_datasets` (the model's own output stays in `model_declaration`)" (spec:111-113) | `if 0 < len(keep) < 3: res = dict(res, model_declaration=..., declaration="not_decidable_1_2_real_datasets")` (:93-94) | Matches |
| Margin r = 0.01 | spec:120 | `MARGIN = P.DELTA_MIN` (:27); `DELTA_MIN = 0.01` (prereg_rules:43) | Matches |
| Declaration at 0.95 | "a direction if P(Delta > r) >= 0.95 or P(Delta < -r) >= 0.95, practically equivalent if P(\|Delta\| <= r) >= 0.95, otherwise inconclusive" (spec:125-126) | frozen `pool` used unchanged: `DECLARE = 0.95`, above / below / equivalent / inconclusive (analysis_x1:37, :206-213) | Matches |
| Holm over exactly 2, missing p = 1 | "enters the adjustment with p = 1, so the family size stays m = 2 (recorded as `holm_m`)" (spec:130) | `A.holm([1.0 if x is None else x for x in p])` (:195); `res["n_confirmatory"] = res["holm_m"] = len(CONFIRMATORY)` (:205) | Matches |
| Secondary contrasts outside Holm | spec:132, :175-176 | Holm input is only the 2 `CONFIRMATORY` keys (:201); 10 secondary contrasts declared with the same pool (:183-184) | Matches; "as used" label wrong (SHOULD-FIX 1) |
| Window width (3.5) | "lower end of the two-sided 95% t interval (2 degrees of freedom) ... width = log10(largest lambda) - log10(smallest)" (spec:177-178) | `stats.t.ppf(0.975, len(best[1]) - 1)` (:156); `max()` on an empty list (:153) | Does not match (BLOCKER 2) |
| Stop on departures (1.1) | "missing task x decoder cells, seeds other than 10-12, unknown keys in `extra`, unknown arms, a lambda label that cannot be read from the tag, a missing point" (spec:62-64) | cells (:52-53), seeds (:54, :66), lambda label (:55-56), extra keys (:57-59), arms x steps (:164-165), missing point (:115) | Partial (SHOULD-FIX 4) |
| X1 outputs read | "only `targets.<task>\|<cond>` ... No X1 score row is read" (spec:59) | `read_record(..., "R4", stage="x1")` (:220); only `tg["bstar"]` and `tg["bstar_low_signal"]` used (:168, :172, :176); matched strengths enter through the resolved manifest `lamf` (:115-116) | Matches |

## 3. Spec versus tests (check 2; read, not run)

| Confirmatory-path rule | Test (`fas/tests/analysis_followups/test_analysis_followups.py`) |
|---|---|
| Contrast identity (critic-optimiser subtrahend) | `test_planted_effects` :110-111 |
| Direction declared on a planted +0.07 | :108, :112 |
| Low-signal → not evaluable; 2 datasets → `not_decidable_1_2_real_datasets` | :113-114, :119 |
| `unreached_low` / reason `unreached` | :115 |
| One failed run keeps m = 3 | :116 |
| CONFIRMATORY = 2, holm_m = 2 | :117-118 |
| Jackknife, `readings` reason, Holm with a missing p, window | `test_known_answers` :94-102 (units) |
| Null world declares no direction | `test_null_world` :122-125 |
| Departure refusals: cells, seeds, extra, lambda label, arms | `test_refusals` :139-148 |
| No overwrite | :151-153 |
| **Uncovered** | fixed-strength values; `unreached_high`; a point with all runs failed; the missing-point stop; Holm-adjusted values end to end; a pool with 0 datasets; the window when every run of an arm fails (SHOULD-FIX 5, BLOCKER 2) |

## 4. Spec versus design (check 3), counted with Grep on `fb/manifests/paper_manifest_stock_pilot_u5_b10_v5.tsv`

| Item | Spec | Manifest v5 | Verdict |
|---|---|---|---|
| X6 rows | 486, 243 per decoder (spec:152) | 486 total; cond 0: 243; cond 1: 243 | Matches |
| Per task | atac_small, immune, pancreas | 162 each | Matches |
| Per arm x steps | 9 configurations (spec:149) | discriminator 54, discriminator_sn 54, discriminator_r1 54, discriminator_sn_critic_opt 54, pooled n_critic 1: 54, 5: 54, 10: 54, pooled_sn 1: 54, 5: 54 (= 486) | Matches |
| Strengths | lo / matched / hi | matched_lo 162, matched 162, matched_hi 162; no lambda = 0 row | Matches |
| Seeds | 10, 11, 12 | seed 10: 162, seeds 11-12: 324 | Matches |
| `extra` | `r1_gamma` on the R1 rows only | 54 rows `{"r1_gamma": 10}`, all discriminator_r1; 432 rows `{}` | Matches |
| Other columns | — | adv_input mean, zstd 0 (equals the R2/R3 decision, `fb/docs/prereg/r2r3_record.json:2, :7911`); max_epochs 239 for immune and 400 otherwise, as in X1 (v5:2153) | Consistent |
| `discriminator_sn_critic_opt` = critic's Adam + SN, 1 step | spec:168-169 | `_CRITIC_OPT_ARMS` → parsed as `discriminator_sn` (scvi_adversarial_plan:64, :170); SN head (:248; adversarial.py:48-52); Adam(lr 1e-4, betas (0, 0.9)) (:266-267); 1 step (:213-216) | Matches |
| No gradient penalty for SN critics | spec:162-163 | `with_gp=not self.spectral_norm` (:410); `_head` zeroes it (:300-301) | Matches |

## 5. Spec versus the frozen main rules (check 4)

| Rule | Frozen (`frf/`) | Follow-up | Verdict |
|---|---|---|---|
| Margin | r = 0.01 (ANALYSIS_SPEC:207) | 0.01 (spec:120) | Same |
| Declaration rule | ANALYSIS_SPEC:203-205; `pool` | same `pool` via `combine` | Same |
| Pooled model | family/dataset random effects, prior 0.1, 1e-6 floor (ANALYSIS_SPEC:167-178) | `combine(..., "family", PRIOR_SCALE, MARGIN)` | Same |
| LOSO | at least 4 of 5 readings (ANALYSIS_SPEC:122) | 3 of 3 (agent default, disclosed spec:78) | Disclosed departure |
| Low-signal flag | same R4 flag; readings shown separately (ANALYSIS_SPEC:121-124) | same flag; no reading computed | Silent departure (SHOULD-FIX 3) |
| Task families | atac_small → atac, immune → immune, pancreas → pancreas (frf/scripts/prereg_rules.py:67-68) | 3 families (spec:102) | Same |
| Window width | n - 1 degrees of freedom, gap, n_all_failed_between (ANALYSIS_SPEC:279, :411-416) | 2 degrees of freedom in text, n - 1 in code, no gap fields | Departure (BLOCKER 2) |
| Sensitivity analyses | 8 registered (ANALYSIS_SPEC:283-292; analysis_x1:44-47) | none, not stated | Silent departure (SHOULD-FIX 3) |
| Variant-arm mapping | PREREG:98 lacks `discriminator_sn_critic_opt` | needed for the confirmatory arm | Not registered (BLOCKER 1) |
| Ceiling with 1-2 datasets | 0.82 / 0.94 / 0.88; 3 datasets need 0.03 to 0.08 (ANALYSIS_SPEC:378-382) | quoted correctly (spec:104-107) | Same |

## 6. Lock readiness of the text (check 5)
- **Draft wording.** "draft; not yet locked" (spec:1, :7) is expected before the lock. The "Not done on this branch (open)"
  paragraph (spec:219-221) is stale (BLOCKER 3).
- **Dropped experiments that still change behaviour.** Only `TAG_LEVEL` accepting `_l0_` (SHOULD-FIX 4). Mentions of X3, X7, X8,
  X12 and X15 elsewhere are history and change nothing (spec:3-5, :44-46, :114-116, :196-198, :244-248; `FOLLOWUPS`, NOTE 5).
- **Agent defaults.** All five in section 11 (spec:235-241) read one way. Undefined edge cases: P1 with no declared sign and the
  meaning of "smaller" (SHOULD-FIX 2); the window's degrees of freedom when a run fails (BLOCKER 2); the label for a pool with 0
  datasets (NOTE 1).

## 7. Blinding (check 6)
- **Script.** It uses only `bstar` and `bstar_low_signal` from the R4-x1 record and X6 rows. It prints one count line
  (`:227`), and the record holds X6 quantities plus `B_minus_bstar` (`:137`).
- **Tests.** Synthetic scores only. The real manifest is read for design rows (test :32-36). No X1 or X13 value is read.
- **Residual risk.** Wrong input paths would load X1 values into memory before filtering (NOTE 4).

## 8. Scope of this review
- **Read.** The spec in full; the script in full; the test file in full; `prereg_rules.py` (fas, relevant parts; frf `BASE_ARM`
  and `FOLLOWUPS`); `analysis_x1.py` (frf, lines 1-536); `ANALYSIS_SPEC.md` (frf, sections 1-5, 10-16); `PREREG.md` (frf,
  section 4 and the SI-103 entry); the fit code (`scvi_adversarial_plan.py` in full; relevant lines of `fit_paper_config.py`,
  `adversarial.py`, `build_paper_manifest.py`, `freeze_matched_lambda.py`, `decide_a2_a3.py`); the build-side test
  `test_adversarial_plan_v2.py:55-114`; the draft amendment; the manifest counts.
- **Not done.** Running the tests or `check_tests.py` (no shell); byte comparison of v4 and v5 X6 rows; a literature or
  prior-work search (out of scope for a spec-versus-code check, and network commands were not allowed); updating the project
  rigor ledger (writing under `fas/`, `fb/` and `frf/` was not allowed).
