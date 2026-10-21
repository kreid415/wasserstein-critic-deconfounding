# Changes made for the X6 pre-lock review (branch x6-lock, from d034db7)

Commits (not pushed, no tag made): 7796e48 (strength pre-check in R4-x1), 75be0fd (analysis script and tests), af58b6c (PREREG
entry, draft marked superseded), 461dd72 (spec). Plain-language terms: X6 = the follow-up experiment on divergence versus
regularisation versus adversary steps; R4-x1 = the rule that sets the follow-ups' penalty strengths from the main benchmark (X1);
b* = the matched batch-removal target.

## Findings

| Id | What changed | Where |
|---|---|---|
| BLOCKER 1 | New 2026-10-20 entry at the end of PREREG section 9: X6 only, both decoders (SI-93), arm `discriminator_sn_critic_opt` and its base-arm mapping to `discriminator` (SI-98), no gradient penalty for spectral-norm critics (I036), follow-up spec locked by the follow-up code tag, strength pre-check, list of scripts that differ from the tag. `git diff prereg-tier12-v3 -- scripts/prereg_rules.py` shows two changed lines, 61 and 62; the entry says so. Draft amendment marked superseded (not deleted). The spec's claim that `prereg_rules` is unchanged is replaced by the two-line statement | `docs/PREREG.md:663` (entry, to end of file); `docs/prereg/amendment_20261017_followups.md:3-6`; `docs/ANALYSIS_SPEC_FOLLOWUPS.md:163-170` |
| BLOCKER 2 | `bio_window` removed; the cell record uses the frozen `analysis_x1.window` (status, best, lower, lams, width, n_points, gap, n_all_failed_between). An arm with no successful run in a cell gets status `no_points`; the analysis continues. Spec: t with n - 1 degrees of freedom as ANALYSIS_SPEC sections 10 and 15.10, and the fields | `scripts/analysis_followups.py:165`; `docs/ANALYSIS_SPEC_FOLLOWUPS.md:72-80, 236-248` |
| BLOCKER 3 | Spec header, section 1.9, section 10 and section 13 rewritten for the script as it is (X6 only; imports `analysis_x1` and `prereg_rules`; hard-coded names listed); "draft; not yet locked" removed, lock by tag `prereg-x6-v1` stated (tag not made); test command and real output recorded | `docs/ANALYSIS_SPEC_FOLLOWUPS.md:1-37, 163-185, 291-319, 342-346` |
| 4 (pre-check) | `freeze_matched_lambda.py --precheck-decision <decision.json>`: for arm `discriminator_sn_critic_opt` only, each resolved strength moves one `prereg_rules.lam_step` up (`shift_up`) or down (`shift_down`); `keep` unchanged; the script stops when X6 rows of that arm are resolved without the file or with another value. The record holds the decision, the arm's row count and the file's SHA-256. Described in the spec (1.10) and the PREREG entry | `scripts/freeze_matched_lambda.py:37-38, 45-48, 70-72, 109, 131-137, 158`; `docs/ANALYSIS_SPEC_FOLLOWUPS.md:186-198`; `docs/PREREG.md` entry |
| SHOULD-FIX 1 | `as_used` = (`w1_gp_5`, `js`); label "W1 critic with gradient penalty at 5 steps, as in the main benchmark, minus JS at 1 step" (spec). Checked: X1 `pooled` rows have n_critic 5 and `discriminator` rows 1 in manifest v5 | `scripts/analysis_followups.py:44`; `docs/ANALYSIS_SPEC_FOLLOWUPS.md:236-242` |
| SHOULD-FIX 2 | Section 3.7: one sentence for P1 inconclusive, equivalent or not evaluable (the "explains" reading is not made; X6 reported by its own sign); "smaller" = absolute pooled posterior mean below the confirmatory contrast's; which contrasts count is named | `docs/ANALYSIS_SPEC_FOLLOWUPS.md:253-265` |
| SHOULD-FIX 3 | New section 1.11 "Departures from ANALYSIS_SPEC": no sensitivity analysis registered, no low-signal readings, no extra crossings of b*, no anchor, 2-degree-of-freedom intervals, no collapse comparison, all-3-readings rule, sign margin | `docs/ANALYSIS_SPEC_FOLLOWUPS.md:200-217` |
| SHOULD-FIX 4 | `|0` dropped from `TAG_LEVEL`; `extra` must equal `{"r1_gamma": 10}` on `discriminator_r1` rows and be empty on every other row (`EXTRA_KEYS` replaced by `R1_EXTRA`) | `scripts/analysis_followups.py:35, 46, 58-62` |
| SHOULD-FIX 5 | Tests: world has one failed run, one failed point (3 runs), one failed arm (9 runs), a curve above b*; asserts fixed-strength values and label, `unreached_low`/`unreached_high` reasons, `no_points`, Wilcoxon and Holm-adjusted p (0.25 and 0.5 give 0.5 and 0.5); second record with every unconditioned cell low-signal (pool with no evaluable dataset: `not_evaluable`, null adjusted p, p = 1 for cond 1 gives 1.0); refusals added: wrong gamma, gamma on every arm, strength-0 tag, a missing point | `tests/analysis_followups/test_analysis_followups.py:67-100, 111-156, 176-190` |
| SHOULD-FIX 6 | `fixed_matched` renamed `fixed_strength` with `label="fixed-strength, not matched batch removal"` (`FIXED_LABEL`); spec 1.4 says the two arms sit at their own inherited strengths | `scripts/analysis_followups.py:45, 171-174`; `docs/ANALYSIS_SPEC_FOLLOWUPS.md:92-112` |
| SHOULD-FIX 7 | Contrast reason = status of the first arm whose curve does not bracket b* (`unreached_low`, `unreached_high`, `no_points`); `no_points` added to spec 1.4 | `scripts/analysis_followups.py:148-149`; `docs/ANALYSIS_SPEC_FOLLOWUPS.md:96-107` |
| NOTE 1 | Label `not_evaluable` for a pool with no evaluable dataset added to spec 1.7 | `docs/ANALYSIS_SPEC_FOLLOWUPS.md:141-153` |
| NOTE 2 | Stale cross-references fixed (script docstring and comments now cite 1.9 and 3.4, "design v5"; test docstring cites 1.9; spec "from lambda = 0" removed from 1.3; 1.9 heading says manifest v5) | `scripts/analysis_followups.py:1-14, 31, 40`; `tests/analysis_followups/test_analysis_followups.py:1-6`; `docs/ANALYSIS_SPEC_FOLLOWUPS.md:81-90, 163` |
| NOTE 3 | Tests read `manifests/paper_manifest_stock_pilot_u5_b10_v5.tsv`; test renamed `test_v5_rows_enumerated` | `tests/analysis_followups/test_analysis_followups.py:29-36, 164` |
| NOTE 4 | `only_x6` stops the script on any score or failure row whose tag is not an X6 manifest tag, before `outcomes` | `scripts/analysis_followups.py:203-208, 224`; test `test_only_x6_rows_are_accepted` |
| NOTE 5, 6 | Left as is (as instructed) | none |
| NOTE 7 | Sign margin documented in spec 1.3 and listed as an agent default (1.11, 11) | `docs/ANALYSIS_SPEC_FOLLOWUPS.md:81-90` |

Other change needed to make "all tests pass": `tests/prereg/test_rules_unchanged.py:91` now expects exactly the one registered rule
difference (the `BASE_ARM` entry of `discriminator_sn_critic_opt`). Without it 4 tests of that file fail; they fail identically at the
unchanged base commit d034db7 (run in a temporary worktree, removed afterwards), because the `BASE_ARM` entry was already in the merged tree.

## Test output (repo root, env scvi-api)

`python -m pytest -q tests/analysis_followups tests/x6_precheck tests/prereg`:

    168 passed, 1 skipped in 70.22s (0:01:10)

`python scripts/check_tests.py --base d034db7 --src scripts -- tests/analysis_followups tests/x6_precheck tests/prereg`:

    168 passed, 1 skipped in 135.89s (0:02:15)
    check_tests: OK (every changed --src line and branch is tested; no redundant new tests; base d034db7)

(one CoverageWarning about `prereg_rules.py<mutant>` was printed by the mutation step; the check still exited OK.)
Test counts: `tests/analysis_followups` 16 (was 10 test cases before: 5 refusal cases became 9, plus `test_pool_without_an_evaluable_dataset`
and `test_only_x6_rows_are_accepted`); `tests/prereg` gained 6 cases (5 parametrised resolve cases, 1 CLI test).

## Not done, with reasons

- No tag `prereg-x6-v1` (instructed); no push.
- The X1 fit-identity check on the commit to be tagged (`experiments/followup_build_20261017/README.md`) was not run: needs the data and
  a long fit; it is listed as "not yet done" in the PREREG entry and in spec section 13.
- The pre-check fits and `decision.json` do not exist from this work; the script only requires them at R4-x1.
- No lab-notebook entry or ledger (`CONSTRAINTS.md`) entry was written: notebook appends are allowed only with `final-review-fixes`
  checked out (CLAUDE.md rule 4), and this branch is `x6-lock`. The user's pre-check decision of 2026-10-20 and this pre-lock fix need
  entries there.
- The review's open question (will the inherited windows bracket b*?) is not answered by this work; the pre-check is its answer.
- The ledger entry SI-103 and its PREREG entry are on `origin/final-review-fixes`, not in this tree (checked with `git grep`); the new
  PREREG entry says so and does not restate them.
- Optional check in SHOULD-FIX 4 (compare each resolved strength with the R4 record's `cells`) was not added: the pre-check shift makes
  the arm's strengths differ from the record on purpose, and the report marked it optional.
- NOTE 5 and NOTE 6 left as instructed; no sensitivity analysis added.
