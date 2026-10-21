Verdict: FIX-FIRST

# Second pre-lock review of the X6 follow-up analysis rule (after the fix pass)

Reviewer: Rigor Reviewer (read-only), 2026-10-21. Tree reviewed: branch `x6-lock` at 22b6b29. Fix pass reviewed: d034db7 (exclusive) to
22b6b29 (7796e48, 75be0fd, af58b6c, 461dd72, 6a2ed00, 22b6b29).

**Counts: 1 BLOCKER, 4 SHOULD-FIX, 11 NOTE, 1 open question. Prior findings: all 3 blockers and all 7 should-fix items are fixed;
none is fixed partly or with a new defect in the confirmatory path.** The one blocker is a precondition the project set itself and
has not met. It is not a defect in the rule.

Terms. *X1* = the main benchmark (8 datasets x 2 decoders x 5 seeds over the adversary arms). *X13* = the standard-method baselines.
*X6* = the follow-up that asks whether the critic-versus-discriminator difference comes from the divergence, the regularisation or
the number of adversary update steps. *W1* = Wasserstein-1 distance (the critics). *JS* = Jensen-Shannon divergence (the
discriminator). *SN* = spectral normalisation. *GP* = gradient penalty. *R4* = the pre-registered rule that picks matched penalty
strengths. *R4-x1* = R4 run on X1 scores. It sets the follow-ups' three strengths *lo / matched / hi* and the target *b\** (the
batch-removal score at which biology is compared). *copt arm* = `discriminator_sn_critic_opt`, the JS + SN discriminator trained with
the critic's optimiser (Adam, learning rate 1e-4, betas (0, 0.9)). *Pre-check* = the 18-fit pilot of the copt arm
(`experiments/x6_precheck_20261020/`). It decides whether that arm's three strengths move one step. *LOSO* = leave-one-seed-out.
*Holm* = Holm's step-down adjustment of p-values. *P1* = the main study's primary result (reference W1 critic minus discriminator,
biology at b\*). *SI-nn* = entry nn of the decision ledger `CONSTRAINTS.md`. *I036* = the fix under which SN critics compute no
gradient penalty.

---

## 1. Prior findings: status and evidence

| Prior id (plain meaning) | Status | Evidence (file:line, all in this tree) |
|---|---|---|
| BLOCKER 1 (the design the spec locks was not in force in PREREG) | **Fixed** | A new in-force entry, `docs/PREREG.md:663-732`, covers both decoders (:687-691), the copt arm and its mapping to `discriminator` (:692-701), I036 (:702-707), the pre-check (:708-721), the analysis rule (:722-727) and every script that differs from the tag (:728-732). The draft is marked superseded (`docs/prereg/amendment_20261017_followups.md:3-6`). Spec 1.9 now says `prereg_rules.py` differs in two lines (`docs/ANALYSIS_SPEC_FOLLOWUPS.md:168-169`). `git diff prereg-tier12-v3 HEAD -- scripts/prereg_rules.py` confirms this: one line edited (61, trailing comma) and one line added (62, `discriminator_sn_critic_opt="discriminator"`). What is left is branch divergence, not this finding (SHOULD-FIX S-3). |
| BLOCKER 2 (window code stopped the whole analysis when an arm had no successful run, and departed from the frozen window rule) | **Fixed** | `bio_window` was removed. The cell record now calls the frozen `analysis_x1.window` (`scripts/analysis_followups.py:165`; frozen function `scripts/analysis_x1.py:343-365`, unchanged since the tag). That function returns `no_points` / `no_interval` and gives `gap` and `n_all_failed_between`. An arm with no successful run gets status `no_points` from `prereg_rules.bio_at_bstar` (`scripts/prereg_rules.py:614-615`). The spec now says "t with n - 1 degrees of freedom" (`ANALYSIS_SPEC_FOLLOWUPS.md:241-247`). Tests: 9 failed runs of one arm in one cell (`tests/analysis_followups/test_analysis_followups.py:70`), with status, window and reason all `no_points` (:129-131). A weak window assertion remains (NOTE N-8). |
| BLOCKER 3 (spec described code that no longer existed; tests not re-run) | **Fixed** | Header, 1.9, 10 and 13 are rewritten (`ANALYSIS_SPEC_FOLLOWUPS.md:1-15, :163-185, :291-319, :342-346`). They now match the imports (`analysis_followups.py:23-24`) and the hard-coded names (:26-46). Test output is recorded at :311-316. Its claim that the runs were on the scripts and tests of HEAD holds: the later commits 6a2ed00 and 22b6b29 add only `review_out/CHANGES.md` and `REVIEW_REPORT.md` (`git show --stat`). The coordinator's run on HEAD (41 passed) is consistent with this. The recorded run covers only 3 test folders; the project's standard suites were not run (S-2). |
| SHOULD-FIX 1 ("as used in X1" contrast mislabelled) | **Fixed** | `"as_used": ("w1_gp_5", "js")` (`analysis_followups.py:44`), with the spec label at `ANALYSIS_SPEC_FOLLOWUPS.md:239-240`. Checked in manifest v5: all 480 X1 `pooled` rows have n_critic 5, and all 480 X1 `discriminator` rows have n_critic 1 (Grep counts). |
| SHOULD-FIX 2 (confirm/refute reading undefined when P1 has no declared sign; "smaller" undefined) | **Fixed** | Section 3.7 now has conditions (a) to (c), with "smaller" defined as the absolute pooled posterior mean. The contrasts that count are named. When P1, or the confirmatory contrast itself, is not declared, the reading is not made (`ANALYSIS_SPEC_FOLLOWUPS.md:253-262`; agent default 11.7 at :330-331). Leftover: the section 8 table (NOTE N-5). |
| SHOULD-FIX 3 (silent departures from the frozen X1 analysis) | **Fixed** | Section 1.11 lists 8 departures: no sensitivity analysis, no low-signal readings, no further crossings, no strength-0 anchor, 2 degrees of freedom, no collapse comparison, all 3 LOSO readings required, and the sign margin (`ANALYSIS_SPEC_FOLLOWUPS.md:200-217`). |
| SHOULD-FIX 4 (stop conditions incomplete) | **Fixed** (the optional part was not done, with a reason) | Strength-0 tags are no longer accepted: `TAG_LEVEL` no longer has `\|0` (`analysis_followups.py:46`). `extra` must equal `{"r1_gamma": 10}` on `discriminator_r1` rows and be empty on every other row (:60-62). Refusal tests cover a wrong gamma, gamma on every arm and a strength-0 tag (`test_analysis_followups.py:179-181`). Manifest v5 has 54 R1 rows with `{"r1_gamma": 10}` and 432 X6 rows with `{}` (Grep counts). The optional check of resolved strengths against the R4 record was skipped. The reason given is that the pre-check shift makes the copt arm differ from the record on purpose (`review_out/CHANGES.md:61-62`); I accept it. |
| SHOULD-FIX 5 (confirmatory-path rules without tests) | **Fixed** | The failures are 1 run, 1 whole point and 1 whole arm (`test_analysis_followups.py:67-70`). Assertions cover `unreached_low` / `unreached_high` reasons (:122-123), a point with all runs failed being skipped (:126-127), `no_points` (:129-131), the fixed-strength values and label (:134-137), and Wilcoxon p [0.25, 0.5] with Holm-adjusted [0.5, 0.5] (:142-143; checked by hand: 2 x 0.25 = 0.5, then max(0.5, 0.5)). A pool with 0 evaluable datasets is tested: `not_evaluable`, null adjusted p, and Holm of [1, 0.5] gives 1.0 (:146-155). Refusal of a missing point: :184-185. |
| SHOULD-FIX 6 (fixed-strength output named "matched", no label) | **Fixed** | The key is `fixed_strength`, with `label=FIXED_LABEL` = "fixed-strength, not matched batch removal" (`analysis_followups.py:45, :171-173`). The spec says each arm sits at its own inherited strength, and that the copt arm's strength includes the pre-check shift (`ANALYSIS_SPEC_FOLLOWUPS.md:107-112`). |
| SHOULD-FIX 7 (not-evaluable reasons coarser than the statuses) | **Fixed** | The reason is now the status of the first arm that does not bracket b\*, minuend first (`analysis_followups.py:148-149`). `next()` cannot run empty: d is None exactly when one arm's full-seed curve is not `bracketed`, and `bstar_status` builds the same curve as `read_bstar` with all seeds (:125, :131). `no_points` is in spec 1.4 (`ANALYSIS_SPEC_FOLLOWUPS.md:101-105`). Wording clash with 1.2: NOTE N-4. |
| NOTE 1-4, 7 (prior) | Fixed | Label `not_evaluable` (spec :144-145); cross-references updated (script :1-14, :31, :40); tests read v5 (test :29-36); `only_x6` refuses non-X6 rows (script :203-207, :224; test :197-203); sign margin documented (spec :88-90). |
| NOTE 5, 6 (prior) | Left as is, as instructed | `prereg_rules.FOLLOWUPS` still lists dropped follow-ups (`prereg_rules.py:70`), which changes no X6 behaviour. Failure counts are not split by type (disclosed at spec :159). |
| Open question (prior: will the inherited windows bracket b\*?) | Answered for the copt arm by the pre-check | `experiments/x6_precheck_20261020/PLAN.md`. Still open for the other arm of the confirmatory contrast (Q-1 below). |

---

## 2. Answers to the review questions

### 2.1 Main-benchmark (X1) safety, HEAD versus tag `prereg-tier12-v3`
`git diff --stat prereg-tier12-v3 HEAD -- scripts src`: 7 files under `scripts/`, nothing under `src/`.

| File | Change | Touches X1? |
|---|---|---|
| `scripts/prereg_rules.py` | `BASE_ARM` gains `discriminator_sn_critic_opt="discriminator"` (:61-62) | `BASE_ARM` is used only in `freeze_matched_lambda.py` (:62-63, :84-86) and in `constants()` (:794), which writes it into every rule record. Nothing reads `constants` back (Grep for `["constants"]` finds only the guard test). The set of cells R4-x1 needs is unchanged, because `discriminator` is already needed by the X6 `discriminator` rows (`freeze_matched_lambda.py:86`). **No X1 decision, row, strength or output changes.** Every new rule record carries one extra `constants.BASE_ARM` key. |
| `scripts/freeze_matched_lambda.py` | `--precheck-decision`. The shift applies only when `arm == SHIFT_ARM` (:37, :70-72) | `resolve` touches only placeholder rows of follow-up experiments: placeholders anywhere else stop the script (:55-58). The R4-x1 cells and targets are computed before `resolve` and do not depend on the decision (:122, :137). Only `discriminator_sn_critic_opt` rows can move. The record gains `precheck_decision`, `precheck_arm_rows` and `inputs.precheck_decision` (:131-136, :158). **No X1 effect.** A stray docstring edit is at :12 (N-1). |
| `scripts/cost_model.py` | `arm_label` strips `_critic_opt` (:34) | Cost estimates only; the copt arm only. |
| `scripts/analysis_followups.py` | new | X6 only. |
| `scripts/fit_paper_config.py` (+83 lines), `scripts/scvi_adversarial_plan.py` (+71), `scripts/build_paper_manifest.py` (+132) | Build side (followup-build, before the fix pass): the copt arm (`scvi_adversarial_plan.py:64, :170, :190`), I036 (`with_gp=not self.spectral_norm`, :410) and options of the dropped X3, X7 and X8 (for example `fit_paper_config.py:268`) | **Fit code did change** relative to the tag. No X1 row uses an SN critic: all 3,000 X1 rows of manifest v5 are in the arms discriminator, reference, pooled, barycenter, mmd, sinkhorn or the no-adversary arms (Grep count 3,000 of 3,000; 0 X1 tags contain `_sn_`). So I036 cannot reach X1 by its condition. The fix pass changed no fit code: `git diff --stat d034db7 HEAD -- src scripts` lists only `analysis_followups.py` and `freeze_matched_lambda.py`. |
| Scoring and X1 analysis | `score_scib_native.py`, `run_stage.py`, `analysis_x1.py`, `docs/ANALYSIS_SPEC.md`, `src/` | Identical to the tag (`git diff --stat` empty). |

**Identity proof.** The no-fit plan-identity run compared NEW = 9de6426 with the tag: 57 of 57 rows and 5 of 5 control rows identical
(`experiments/followup_build_20261017/results/plan_identity_run_20261017/RUN.md:9-13`). It carries over to HEAD by file identity:
`git diff --stat 9de6426 HEAD -- src scripts/fit_paper_config.py scripts/scvi_adversarial_plan.py scripts/score_scib_native.py
scripts/run_stage.py` is empty. Only `build_paper_manifest.py` changed after 9de6426 (4685d50: `--exclude-experiments`), and
`fit_paper_config.py` does not import it (its imports are at :24-32). By its own limits, the no-fit run does not test the training
step (RUN.md:16-20). The training step is where I036 lives. The one-epoch fit identity that does test it has not been run (BLOCKER B-1).

### 2.2 The edit to `tests/prereg/test_rules_unchanged.py:90-91`
**Narrow and legitimate.** It pops exactly one key with an exact expected value (`== "discriminator"`) from the new record only. It then
requires every other field to be equal (:92) and the inputs' SHA-256 to be equal (:93). The resolved manifests must still be byte
identical (:78-81). If the entry were missing or had another value, the test would fail. Since `constants` is in every record, applying
the pop to all 4 records is correct. Limits (NOTE N-2): the baseline is `eb444fe` (:17), not the tag. The rule scripts differ between
eb444fe and the tag by 177 lines (`git diff --stat eb444fe prereg-tier12-v3`), so the equality with the tag holds only through
eb444fe. The synthetic rows (`tests/prereg/synth.py:31-33`, default design) contain no copt rows, so the guard never runs the
pre-check path. The docstring (:1-5) still names the SI-45 field as the only allowed difference.

### 2.3 The confirmatory contrast: no drift
| Element | Approved draft (`git show d034db7:docs/ANALYSIS_SPEC_FOLLOWUPS.md`, section 3.4 / 1.6 / 1.7) | HEAD | Drift |
|---|---|---|---|
| Contrast | `pooled_sn` n_critic 1 minus `discriminator_sn_critic_opt`, per decoder, 2 results | Spec :230-234 (same text); code `"divergence": ("w1_sn_1", "js_sn_copt")` (:41), configs (:37-39), `CONFIRMATORY` (:183) | none |
| Endpoint | biology at b\*, bracketing only | Spec :97-105; `read_bstar` / `bio_at_bstar` (:123-127), unchanged in the fix pass | none |
| Margin | r = 0.01 | Spec :138; `MARGIN = P.DELTA_MIN` (:28) = 0.01 (`prereg_rules.py:43`) | none |
| Declaration | ANALYSIS_SPEC section 5 at 0.95 | Spec :143-145; frozen `combine` / `pool` (`analysis_x1.py:233-255`, unchanged) | none (adds the `not_evaluable` label for an empty pool) |
| Wilcoxon companion with Holm over 2 | as draft | Spec :147-152; `holm_all` and `analyse` (:186-199), unchanged in the fix pass | none (adds: adjusted value null when p is missing) |
| Strengths of the subtrahend arm | inherited from `discriminator` | inherited, then moved by the pre-check decision (spec :186-198) | **Design change, registered** (user decision 2026-10-20, PREREG :708-721). It changes what is run, not the analysis rule. |

`pooled()`, `loso()`, `read_bstar()` and `holm_all()` are not in the fix-pass diff. The fix pass changed the confirmatory path only in
the reason labels and in input refusals.

### 2.4 The strength pre-check rule and its implementation
- **Rule as written** (`PLAN.md:27-32`): per cell, `reached` if min B <= b\* <= max B, `unreached_low` if max B < b\*,
  `unreached_high` if min B > b\*, and `not_readable` with fewer than 2 successful fits. Decision: `shift_up` if at least 3 of the 4
  counted cells are `unreached_low`, `shift_down` if at least 3 are `unreached_high`, otherwise `keep`.
- **`decide.py` implements this exactly:** `cell_status` at :11-20 and `decide` at :23-27. Tests are at `tests/x6_precheck/test_precheck.py:17-33`.
- **Applied exactly:** one `lam_step` per strength (`freeze_matched_lambda.py:70-72`; `prereg_rules.py:131-136`). The test
  checks all three decisions on the real 486 X6 rows (`tests/prereg/test_cli_e2e.py:134-153`).
- **Refuses to run without the file:** `precheck_steps(None)` raises as soon as a copt row is resolved (:45-48, :70). This is tested
  in the unit test (`test_cli_e2e.py:135, :142-145`) and through the command line (:167-168).
- **Cannot move another arm:** the shift requires an exact match on `arm == SHIFT_ARM` (:70). The test checks that every other arm
  keeps its base arm's strengths (`test_cli_e2e.py:151-153`).
- Robustness gaps are listed as NOTE N-6 and N-7. The rule and the PREREG entry disagree on timing (SHOULD-FIX S-1).

### 2.5 Does the spec describe the code? Remaining mismatches
The spec matches the code on inputs, stop conditions, b\* reading, reasons, fixed-strength output, window fields, pooling, margin,
declarations, Holm handling and the names in 1.9. I checked each name in 1.9 against the code (`analysis_followups.py:23-46`, the `A.`
and `P.` uses). Manifest v5 checks: SHA-256 `ce8dfbf2...a46432` matches its `.sha256` file; 486 X6 rows; 81 rows at cond 0 under
`matched`; 54 copt rows. The remaining mismatches are wording only: N-4 (spec 1.2 versus 1.4 on the reason), N-5 (section 8 table)
and N-9 (PREREG :723-725, Holm).

### 2.6 Tree hygiene for the tag
- `REVIEW_REPORT.md` (root) is **byte-identical** to the prior review (`diff -q` against `x6review2_inputs/prior_review.md`: no
  difference). An automatic checkpoint hook added it (22b6b29). Its file:line references point to other checkouts (`fas/`, `fb/`,
  `frf/`).
- `review_out/CHANGES.md` is the fix pass's own log. This report (`review_out2/REPORT.md`) is not committed.
- `git ls-files --others --exclude-standard` lists nothing.
- There are two stray whitespace edits (N-1).
- Recommendation: S-4.

---

## 3. New findings

### BLOCKER

**B-1. The X1 fit-identity check, which the project made a precondition of this tag, has not been run.**
- **The precondition.** The spec says the tag is made "after the X1 fit-identity check" (`ANALYSIS_SPEC_FOLLOWUPS.md:317-318`, :344-345).
  The README calls it "precondition of the new code tag, review item F5" (`experiments/followup_build_20261017/README.md:19`).
- **Why it matters.** Fit code differs from the tag (section 2.1). The only identity evidence is the no-fit plan check, which does not
  test the training step (`RUN.md:16-20`), and I036 is in that step (`scvi_adversarial_plan.py:410`). The PREREG entry lists the check
  as "not done" (`docs/PREREG.md:669-672`).
- **Fix.** On the final commit M, run README Step 1 (the 57 one-epoch fits plus the 5 control fits). Commit its record. Then tag either M
  or a commit whose `git diff M -- scripts src tests` is empty and which updates PREREG :669-672. Alternatively, the user records in
  the PREREG entry, before tagging, a decision that the tag comes first and the identity run comes before the first X6 fit.

### SHOULD-FIX

**S-1. The pre-check plan and the PREREG entry disagree on when the decision is registered, and nothing requires `decision.json` to be committed before R4-x1.**
- **The two texts.** `PLAN.md:35-36` says "The decision is written into the follow-up amendment before the follow-up analysis is
  locked". The tag would lock the analysis before the decision exists, which `review_out/CHANGES.md:54` confirms.
- **What the code does instead.** The decision is applied from a file at R4-x1, and only its SHA-256 is recorded
  (`docs/PREREG.md:714-721`; spec :192-196).
- **Fix.** Add one sentence to the PREREG entry: it departs from PLAN.md:35-36, the decision is applied mechanically at R4-x1, and
  `decision.json` is committed under `experiments/x6_precheck_20261020/results/`, with a notebook entry giving its SHA-256, before
  R4-x1 runs. The other option is to wait for the decision and write it into the entry before tagging.

**S-2. The project's standard test suites have not been run on the tree to be tagged.**
- **What was run.** The recorded runs cover only `tests/analysis_followups tests/x6_precheck tests/prereg` (spec :311-316). The
  coordinator ran 41 tests on 3 files.
- **What CLAUDE.md requires.** The cloud-form `check_tests.py` command over `tests/prereg tests/stage tests/analysis tests/scvi
  tests/jhpce tests/test_legacy_retired.py tests/test_gate_task_split.py`, and on the workstation the full `tests/`.
- **Why it matters.** `review_out/CHANGES.md:30-32` reports that `test_rules_unchanged.py` was already failing at d034db7 without
  anyone noticing. That shows the wider suites were not being run on the follow-up branches, and the frozen rule and fit code changed
  on them. I could not run tests (no pytest).
- **Fix.** On the exact commit to be tagged, run the CLAUDE.md command plus `tests/analysis_followups tests/x6_precheck`, and on the
  workstation `tests/scvi` and `tests/wcd_vae`. Record the outputs in spec section 10.

**S-3. Decisions that the frozen texts cite are missing from this tree.**
- **What is missing.** `CONSTRAINTS.md` holds SI-93 to SI-98 and no SI-99 to SI-103 (Grep). `notebook/` has no entry dated
  2026-10-20 or 2026-10-21 and none on the pre-check (Grep).
- **Where they are cited.** The spec cites SI-102 and SI-103 (`ANALYSIS_SPEC_FOLLOWUPS.md:11, :19, :340`). PREREG :678-680 says the
  SI-103 entry is on `final-review-fixes`, a branch that is not in this clone (`git branch -a` shows only `x6-lock`), so I could not
  check it.
- **The rule broken.** CLAUDE.md rule 4 requires a notebook entry for every decision; neither the pre-check decision nor this fix pass
  has one.
- **Fix.** Merge `final-review-fixes` into `x6-lock`, putting the SI-103 PREREG entry before the 2026-10-20 entry. Add the ledger and
  notebook entries on `final-review-fixes` (the pre-check decision, the fix pass and this review), then tag the merged commit.

**S-4. Tree hygiene.**
- **The files.** `REVIEW_REPORT.md` (root, from the automatic checkpoint 22b6b29) and `review_out/CHANGES.md` are review records that
  sit outside the repo's review folder.
- **Fix.** `git mv` them to `docs/review_final/x6_prelock_review_20261020.md` and `docs/review_final/x6_prelock_fixes_20261020.md`;
  they are the record the fix-pass commits cite. Keep `review_out2/` out of the tagged tree, or move it the same way.
- **At tag time.** Check that the automatic checkpoint hook added nothing: `git status` clean, and `git diff --stat <reviewed commit> <tag commit>` lists only the intended files.

### NOTE

**N-1. Two stray whitespace edits.** `scripts/freeze_matched_lambda.py:12` reads "b* =b0" (the tag has "b* = b0"), and
`tests/analysis_followups/test_analysis_followups.py:57` reads "C =0.70". Neither changes behaviour, but the first is an unregistered
edit to a frozen rule script. Fix: revert both so the frozen-file diff against the tag holds only the registered change.

**N-2. The guard test's docstring is out of date, and its baseline is not the tag.** The docstring of `tests/prereg/test_rules_unchanged.py:1-5`
is stale, and the baseline is `eb444fe`, not the tag (:17). Fix: add the `BASE_ARM` entry to the docstring, and consider making
`prereg-tier12-v3` a second baseline (parametrise `PREVIOUS`).

**N-3. No test checks that the pre-check path leaves X1 untouched.** The claim rests on reading the code, which is sound
(`freeze_matched_lambda.py:55-58, :70`). Fix: in `test_x6_precheck_decision_enters_the_record` (`tests/prereg/test_cli_e2e.py:156-178`),
assert that every non-copt row of the output manifest, and the record's `cells` and `targets`, equal those of the run without copt rows
(`pipeline` `m.r4x1.tsv` / `r4x1.json`).

**N-4. Spec 1.2 says the reason is always `no_points`; the code says otherwise.** Spec 1.2 says every contrast that uses a `no_points`
arm has the reason `no_points` (`ANALYSIS_SPEC_FOLLOWUPS.md:77-79`). The code and spec 1.4 give the minuend's status first
(`analysis_followups.py:148-149`; spec :104). Fix: reword 1.2 to "...is not evaluable in that cell (reason: the first non-bracketing
arm's status, section 1.4)".

**N-5. The section 8 table is not tied to P1.** Its "Contradicts" column (`ANALYSIS_SPEC_FOLLOWUPS.md:271`) reads "equivalence, or
opposite sign", without the condition, now in 3.7 (:259-262), that P1 is declared. Fix: write "a declared direction opposite to a
declared P1, or equivalence".

**N-6. `decide.py` hard-codes the counted cells.** `COUNTED` is fixed in `decide.py:8`, while PLAN.md:29 defines the counted cells as
those not low-signal in the pilot record, and nothing checks the two against each other. The shape check is an `assert` (:49), which
is skipped under `python -O`. Fix: raise if `{k for k, c in cells.items() if not c["low_signal"]} != set(COUNTED)`, and turn the
`assert` into a raise. I did not read the pilot record to compare, because the brief bars reads under `tier12*`.

**N-7. `freeze_matched_lambda.py` trusts the decision file.** It reads only the `decision` key (:133) and does not check that the file
is a `decide.py` output. Optional fix: refuse a file whose `decision` differs from `decide.decide` applied to its own
`n_unreached_low` and `n_unreached_high`.

**N-8. The window has no known-answer test.** The cell-window assertion only checks `status == "ok"` and `n_points >= 1`
(`test_analysis_followups.py:132`), so passing B instead of C into `A.window` (`analysis_followups.py:165`) would survive. Fix: assert
`window["best"]` equals the cell's smallest lambda. In the synthetic world C falls with lambda (:57) and B rises.

**N-9. PREREG wording on Holm.** `docs/PREREG.md:724-725` says "Holm's adjustment over the 2 results" without saying that the
adjustment applies to the exact Wilcoxon companion p-values, which are reported and used for no conclusion (spec :147-152). Fix:
add that clause.

**N-10. A pre-check preflight item is out of date.** The evidence for PF-18 in `experiments/x6_precheck_20261020/PREFLIGHT.md:37`
("waits for the stage-2 A3 runner's exit line") was overtaken by 05d2403 (21:42:25), which made the launcher wait for a free GPU two
minutes after the GO (b471742, 21:40:33). Fix: re-record PF-18 against the current `run_precheck.sh`.

**N-11. `docs/claude/STATUS.md:86` is out of date.** It still lists X3, X7, X8 and X12 and gives R4-x1 without `--precheck-decision`.
This file is not frozen. Fix: update it in the next status commit.

### Open question (cannot be settled from the files; it changes what is run, so it is the user's)
**Q-1. The confirmatory contrast's other arm was never piloted or pre-checked.** That arm is `pooled_sn` at 1 step (W1 + SN). It takes
the strengths of the X1 `pooled` critic, which uses a GP at 5 steps. Its curve may also fail to bracket b\*, and if so the
confirmatory contrast cannot be evaluated in that cell whatever the pre-check decides.
- **No pilot or pre-check.** `pooled_sn` is not an A1 pilot arm (`prereg_rules.py:54-56`), and the pre-check fits only the copt arm
  (`PLAN.md:15`).
- **What already covers it.** The rule handles this case through rule I048 (fixed-strength report, `ANALYSIS_SPEC_FOLLOWUPS.md:107-112`).
- **What is missing.** Spec limitation 9.3 (:282-284) does not name this risk.
- **Options.** State it in 9.3, or extend the pre-check to `pooled_sn` at 1 step (a user decision).

---

## 4. Scope
- **Read in full:** the prior review; `review_out/CHANGES.md`; the fix-pass diff of `scripts/` and `tests/`; `scripts/analysis_followups.py`;
  `docs/ANALYSIS_SPEC_FOLLOWUPS.md` at HEAD and at d034db7; the PREREG entry (:663-732); `tests/prereg/test_rules_unchanged.py`;
  `tests/x6_precheck/test_precheck.py`; `experiments/x6_precheck_20261020/PLAN.md`, `decide.py`, `PREFLIGHT.md`;
  `experiments/followup_build_20261017/README.md` and `plan_identity_run_20261017/RUN.md`.
- **Read in part:** `prereg_rules.py` (:36-160, :265-324, :605-625, :780-830); `freeze_matched_lambda.py` (:1-164);
  `analysis_x1.py` (:225-369, :385-404); `test_analysis_followups.py` (:60-204); `scvi_adversarial_plan.py` (Grep);
  `tests/prereg/synth.py:31-33`.
- **Commands run:**
  - `git log`, `git diff --stat` and `git show --stat` against the tag, d034db7, 9de6426 and eb444fe;
  - `diff -q` of `REVIEW_REPORT.md` against the prior review;
  - `sha256sum` of manifest v5;
  - Grep counts on manifest v5, a design file with no scores.
- **Not done:**
  - **Tests.** I ran none; no pytest was available. Test results are the coordinator's 41 passed and the recorded outputs.
  - **Blinded files.** I read nothing under `tier12*` and no X1 or X13 score, failure or status file, so the pilot record behind N-6 was
    not checked.
  - **Other branch.** `final-review-fixes` is not in this clone.
  - **Literature.** No literature or prior-work search: this is a spec-versus-code lock review.
  - **Records.** No rigor-ledger or notebook update and no commit (the brief allows edits to this report only).
