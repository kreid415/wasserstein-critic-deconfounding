# AMEND5_REPORT: pre-registration amendment of 2026-10-22 (documents only; nothing run, nothing submitted, nothing pushed)

Branch `pilot5-amend`. Documents commit: `98732ff`. This report is in the commit after it (a file cannot hold its own commit hash). Plain names: A1 = penalty-strength pilot; X1 = main benchmark; R1 = strength-grid rule; R4 = matched-strength rule; A2/A3 = companion pilots (R2/R3 their rules); SI-nn = ledger entry in CONSTRAINTS.md; JHPCE = the Johns Hopkins cluster.

## 1. What was added where

| File | Lines | Content |
|---|---|---|
| `docs/PREREG.md` | 663-813 (end of section 9; the previous last line, the question-19 table row, is unchanged) | One entry dated 2026-10-22: status paragraph, then items (1) main benchmark stopped and the blinding statement, (2) pilot extended to 5 datasets, (3) companion pilots not extended, (4) R1 re-run on 8 datasets with one shared grid, (5) keep-or-refit rule, (6) scoring-site check and its three cases, (7) scorer-commit check, (8) which earlier statements change |
| `CONSTRAINTS.md` | 657-697 | SI-108 (stop X1), SI-109 (extend pilot), SI-110 (companions not extended), SI-111 (R1 on 8 datasets), SI-112 (keep-or-refit), SI-113 (scoring site); each with rule / source / added / keywords / files |
| `experiments/pilot5_amend_20261022/verify_amend5.py` | new | Standard-library script that re-derives the manifest numbers and the commit-equality check below; exit 1 on any mismatch |
| `docs/EQUIVALENT_COMMITS.json` | unchanged | See section 3 |

No frozen rule code, no test and no other file changed.

## 2. Verified here (command, result)

`python3 experiments/pilot5_amend_20261022/verify_amend5.py` (all 21 checks PASS): new manifest sha256 `5dd65550...0396ff`, 1,890 rows, 378 per dataset, seeds 100-102 only, experiment A1 only; 12 grid-edge rows per dataset (discriminator unconditioned 0.01 and 0.03, Sinkhorn conditioned 10,000 and 30,000; 15 rows per strength over 5 datasets); 366 base rows per dataset; old manifest sha256 `6efbe422...cf24`; first-round extension file 12 rows for each of atac_small, immune, sim1, sha256 `6c38c116...ecffe`; 1,098 + 36 + 1,890 = 3,024 pilot fits for the R1 re-run.
`python3 scripts/compare_a1_pilot5.py ...` (command of PILOT5_BUILD.md section 4), re-run: PASS, 18 columns equal, grid-edge strengths agree in the R1 record, old rows, builder and new rows; R1 rounds used: js|0 low 1, sinkhorn|1 high 1, all others 0.
One A1 Sinkhorn score file (`tier12_v3/A1_sinkhorn/scores/A1_sim1_sinkhorn_l30_c0_s100_ebcbe68a.csv`, a pilot score, not blinded): scorer_git_sha `10a1ca58...`, host gimli, AMD Ryzen 5 3600, avx2, numba setting empty.
Tag `prereg-tier12-v3` peels to `10a1ca5874fbcdc033b148525b5e71d866306bdb`.
Rule-code lines quoted in the entry were read from `scripts/prereg_rules.py` (outcomes 311-313, grid-equality check 486-488, rounds 50, provenance columns 73, provenance check 247-266, R4 targets 561-573, equivalence functions 220-244) and the two tests that read the equivalence file (`tests/stage/test_run_stage.py` 670, `tests/prereg/test_provenance_rule.py` 108-112).

## 3. Item 7: scorer-commit equivalence check and why `docs/EQUIVALENT_COMMITS.json` is unchanged

```
git diff --stat 10a1ca5 a4989ff           -> 7 files, 833 insertions, 0 deletions:
  SCORE5_BUILD.md, cluster/README.md (+2), cluster/jhpce/compare_scores.py, score_array.sh, score_plan.py,
  score_check/check_latents.tsv, tests/jhpce/test_score_check.py
git diff --stat 10a1ca5 a4989ff -- scripts src configs docs manifests cluster/envspec CONSTRAINTS.md   -> empty
git diff --stat af67f3a a4989ff           -> SCORE5_BUILD.md only (198 insertions)
git rev-parse <commit>:<path>, same id at 10a1ca5 and a4989ff:
  scripts/score_scib_native.py 800d78ee   scripts/host_info.py e47b195c   scripts/run_stage.py 5fe6fb35
  trees scripts/ 8b822d98  src/ 1feaa7eb  configs/ 26579f07  docs/ 3654a663  manifests/ fa95787e  cluster/envspec/ 64c62bae
```
The scorer's only repository import is `scripts/host_info.py` (imports checked with grep; it has no imports). Result: byte-identical, so the equivalence holds for the scorer code and its settings in git. Not covered by git: installed package versions, processor and the environment variables of the new job script `score_array.sh`; the first two are recorded in every score row and compared by the 20-latent check, the third is covered by `tests/jhpce/test_score_check.py` (SCORE5_BUILD.md section 10, not re-run here).

**The declaration could not be added "in its existing format", so I did not edit the file.** `docs/EQUIVALENT_COMMITS.json` is one object with one declared commit (`"commit": "c88cce3..."`, read as the tag's commit); `prereg_rules.refuse_uncovered` and `equivalent_commits` (lines 220-244) read only that key, and `tests/prereg/test_provenance_rule.py::test_the_committed_equivalence_list` pins its content. A second key or a list would be ignored by the rule scripts, so an edit would look like a declaration without having an effect. Without it, a rule input that mixes scores at c88cce3/10a1ca5 and a4989ff (case (i) of the scoring check) is refused with "mix scorer provenance". The entry says this in item 7. Proposed change, needing your approval (frozen code, CLAUDE.md hard rule 2): let the file hold a list of `{commit, commit_ref}` pairs, read them in `refuse_uncovered`/`equivalent_commits`, and extend the three tests that read the single entry (`test_run_stage.py` 670, `test_provenance_rule.py` 110, and the pinned-content test). Case (ii) and (iii) need no declaration.

## 4. Test result

`python3 /home/kendall/scratch/run_in_env.py -m pytest tests/prereg -q -p no:cacheprovider`: 140 passed, 1 skipped in 36.09 s (same counts as PILOT5_BUILD.md section 5; no test needed extending because the equivalence file did not change).

## 5. Things you should check or decide

1. **SI numbers.** This clone's ledger stops at SI-92, but the research line already holds SI-93 to SI-107 for other decisions (read in the worktree `/home/kendall/scratch/fixtests`, SI-107 added 2026-10-21; the highest I could find). I numbered the six new entries SI-108 to SI-113 to avoid a clash; the live ledger may be further on, so confirm or renumber at merge (the numbers also appear in `docs/PREREG.md` 663-813). The new entries carry no `notebook:` line: no notebook entry was written (the notebook is hash-chained, written from `final-review-fixes` only; CLAUDE.md hard rule 4 asks for one per decision).
2. **Source lines.** The brief gave your answers in paraphrase, not verbatim; each SI `source:` says so. Replace with the ask_user wording if you want it.
3. **Dates.** Only item 1 has a day (2026-10-21); the others are recorded as 2026-10-21/22, as in the brief.
4. **Blinding statement (item 1).** The repository's blinding records (hash list README, PREREG disclosure, notebook, user statement NB-20261015-15) end on 2026-10-16. I could not verify "no X1 score read" for 2026-10-17 to 2026-10-22; the entry says so. It rests on: JHPCE jobs run `run_stage.py --no-score` (`cluster/jhpce/prod_fit_job.sh` 2, 137) and I opened no X1 file. Not checked: whether the workstation run scored latents after 2026-10-16.
5. **Not verified (copied from your brief; no cluster contact, X1 files not opened):** job numbers 36518712, 36518713, 36518737, 36518738, 36518744, 36551148-36551152, 36550869, job record 6bbb6be0; the times 21:58, 19:49, 19:30, 08:10, 23.6 h; the counts 572 / 405 / 167 / 10.
6. **My additions beyond the brief:** (a) in item 6, a comparison that ends with exit status 2 (missing file, row or column) is neither a pass nor a failure and is completed or repeated before a case is chosen, marked "clarification added by the writer"; strike it if unwanted. (b) item 4 states that the first extension round counts as round 1 at two edges, so one round is left there (read from the R1 record's `rounds` and PILOT5_BUILD.md section 7). (c) item 6 states the limit that the 20 latents come from the three small datasets and none from the two largest.
7. **If the match check's result was already known when you commit this,** the entry's "submitted before its result was known" is still true but add one sentence saying so.
8. **Not touched:** the status paragraph at the top of `docs/PREREG.md` (line 3) and the stage table (section 0) still describe three pilot datasets; item (8) of the entry says what supersedes them. R4 stage a1, R2, R3 and the power report (section 5) are not re-run by the entry; the brief does not say whether the power report is.
