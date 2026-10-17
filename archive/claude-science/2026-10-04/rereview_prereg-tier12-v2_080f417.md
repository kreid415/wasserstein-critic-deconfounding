# Re-review before tag prereg-tier12-v2: origin/integrate-fixes @ 080f417

Reviewer: Rigor Reviewer (sub-agent), 2026-10-04. Review only: no commit, push, tag or cluster command, and no edit to the reviewed code (fresh clone of origin; `git status --porcelain` empty throughout; 'clean=0' in the test summary is that count of changed files). Baseline: tag
prereg-tier12-v1 (129142b). Commits 4fde2b6..080f417 change only docs/PREREG.md and experiments/integration_checks_20261004/
(checked with git diff --stat), so every code result below holds at 4fde2b6 and 080f417 alike.

## Verdicts

| question | verdict | condition |
|---|---|---|
| A1: may the lead tag prereg-tier12-v2 and restart A1 from zero with run_A1.sh? | **yes** | Valid at 080f417. Recommended: tag one commit later, after the RR-01 fix (test or bundle recipe) and the sec 9 text fixes (RR-02, RR-03, RR-05), so that A1 and the JHPCE jobs share one tag. These touch no fit or score code; re-run tests/scvi and the A1 identity check at that commit. |
| JHPCE jobA + jobB at that tag | **NO-GO** | Blocking: RR-01 (both jobs exit 2 at the node test step, before any fit). GO once RR-01 is fixed and local_rehearsal.sh passes for both jobs at the tagged commit (with the tag in the bundle both pass now), and the user gives the go (SI-29). |

Preflight records written by this review (workspace, not committed): `PREFLIGHT_jhpce_fillers_x1_x13_prod.md` (46 items:
17 pass, 28 n/a, 1 fail; blocking PF-22 = RR-01 → NO-GO) and `PREFLIGHT_A1_lambda_pilot_v2_restamp.md` (the committed
A1 record re-stamped with evidence from 080f417, same plan hash: 31 pass, 15 n/a, 0 blocking → GO).

## Findings (new, severity order)

| id | sev | finding | fix |
|---|---|---|---|
| RR-01 | high | The JHPCE job clones a single-ref git bundle (no tags) and runs `pytest tests/scvi`; `tests/scvi/test_baseline_guard.py:63` calls `git show prereg-tier12-v1:...`. local_rehearsal.sh at 080f417: both jobs exit 2, "2 failed, 108 passed, 1 skipped". With `refs/tags/prereg-tier12-v1` in the bundle: both exit 0 (110 passed, fingerprints 8/8, dry runs 100 / 150 rows). | Pin the test to commit 129142b, or bundle the tag (and update local_rehearsal.sh and the submit recipe); then re-run the rehearsal for both jobs. |
| RR-02 | low | PREREG sec 9 (l.178-179): "X3 pancreas dose 0: PCR_batch 0.573 ... 0.000". The evidence row is dose 50 (tag ..._31994c65) and its latent is an uncorrected HVG-PCA embedding, not a fit. Re-run: dose 50 0.573082 vs 0.0; dose 0 0.560350 vs 0.0. | Reword before the tag. |
| RR-03 | low | Sec 9 (l.184) says the scib fallback flags are reported with every R1 and R4 input; no rule script reads them. | Add flag counts to the R1/R4 records (reporting only), or reword. |
| RR-04 | low | `scorer_git_sha` = HEAD, and prereg_rules refuses mixed values. R2/R3 combine A1 rows with A2/A3, SI-45 and R4-extension rows scored later; one notebook commit on the run checkout splits the input and forces re-scoring ~1,100 rows. run_A1.sh says "v2 or later". | Run every stage from one worktree pinned at the tag (say so in the launchers), or key provenance on scorer file hashes. |
| RR-05 | low | CR-11 remainder: PREREG sec 8 l.140 still says no runner writes the failures table. | One dated line in sec 9. |
| RR-06 | low | The committed A1 PREFLIGHT.md (GO, 129142b) cites v1 artefacts (manifest v1, tests 72 passed, tier12/A1). | Commit a re-stamped record with the tag (draft attached). |
| RR-07 | low | harvest_local.py:182 protects only a `tier12` path component; the new A1 tree is tier12_v2/A1. | Refuse `tier12*`. |
| RR-08 | low | SI-45 rows overwrite `--extension-manifest` (os.replace); passing the previous round's path deletes that round's R4 extension manifest. | Refuse an existing path. |

No finding affects A1's fits or scores.

## Prior findings CR-01..CR-14 at 080f417 (own checks)

| id | state | evidence (summary; full text in prior_issues_status_080f417.csv) |
|---|---|---|
| CR-01 | fixed | Subset branch lets scib recompute the all-feature PCA. Dose-50 scorer value = all-feature reference (0.573082; HVG 0.0). The full-data path is unchanged: a stored A1 latent re-scored gives 15/15 identical columns. |
| CR-02 | fixed | `--tags-file` (own stage key) and CPU lanes. Dry runs: CPU rows refused without lanes; 35 GPU + 16 CPU fits with lanes; jobB tags give 150 rows. Stage suite 62 passed. |
| CR-03 | fixed (job blocked by RR-01) | Production job with witness, versions, node tests, slots, hashes, claims, stop guard and SHA-256 harvest; jhpce suite 36 passed. |
| CR-04 | fixed | nested_v1 on the real obs: equal N (8,402 / 20,000 / 14,409 / 17,872), nested, overlap formula exact. The smallest type kept at dose 95 has 30 / 30 / 37 / 67 cells. Weights restore K0 counts exactly. |
| CR-05 | fixed | Scorer columns, runner gate and rule-side refusal (check_provenance in outcomes); 7 tests pass. Caveat RR-04. The rule-side refusal is the lead's commit 443b17b to the v1-frozen prereg_rules.py (disclosed in sec 9); the reviewer edited nothing (clone clean; prereg_rules.py blob 4c8c12c0 = 080f417's). |
| CR-06 | fixed | Guard on scANVI/sysVI plans; bit identity vs the tagged fitter (test passes); GPU smoke exit 0. |
| CR-07 | disclosed | Sec 9 wording matches the first review's measurement (0/5 vs 5/5 shared minibatches). |
| CR-08 | fixed (reporting open: RR-03) | Pass-through wrappers with call accounting checked against scib 1.1.7 kbet.py:110-189. |
| CR-09 | fixed | Resolved plan recorded and checked; the GPU smoke shows GP/λ_GP and KL warm-up per arm. |
| CR-10 | fixed | Legacy drivers retired with a guard; test passes. |
| CR-11 | open (partial) | SI-19/20 annotated; SI-45 implemented and tested. The sec 8 sentence is still stale (RR-05). |
| CR-12 | disclosed | The declared-differences paragraph matches the code; scANVI comment fixed. |
| CR-13 | open (partial) | N8 is now X15. Still missing: the SI-38 script, the N1/N6 logs and the clustering-grid statement. |
| CR-14 | fixed | Missing or duplicate cell names refused; tests pass. |

## Review of the new code (what was checked)

- **Runner** (`run_stage.py`). The tags file is parsed strictly (UTF-8, one tag per line, no duplicates). Every listed tag
  must lie inside --experiments x --tasks, and the file's hash goes into the stage key and ledger. Rows are routed to GPU
  or CPU by row kind. A CPU row must record device 'cpu' and a CPU model, and its only outcome record is
  nonfinite_latent. The scorer-provenance gate runs in the preflight and in the completion gate. Dry runs gave the
  expected refusals and counts.
- **Scorer.** The subset PCR reference and indexer checks are correct, and score rows now carry provenance. The fallback
  flags follow scib 1.1.7's label rule: labels with ≥10 cells and >1 batch are tested; the 75% rule and NeighborsError
  set a label to 1; an R read error gives NaN. The wrappers re-raise and pass values through.
- **X3 (SI-41/42).** x3_draw and the weights match the SI-42 text. Checked on all 4 tasks × 5 doses with the fitter's own
  functions (table x3_nested_v1_check.csv). Weight 0 arises only for refill pairs absent from K0 (1-2 pancreas cells); the
  weighted critic treats a batch whose weights sum to zero as inactive.
- **X15 (SI-44).** `kl_warmup_epochs` passes nothing for 'stock' or a missing key, so A1 and X1 rows are unchanged
  (14/14 identity). It passes max_epochs for 'complete', and check_kl_warmup verifies the trained plan. KL weights were
  recomputed from scvi-tools 1.4.2 `_compute_kl_weight` (_trainingplans.py:32-72).
- **SI-45.** r4_stage records `edge_unresolved` neighbours. default_half_extensions writes A1 rows (mean, zstd 0, seeds
  100-102, same decoder). decide_a2_a3 refuses the cell until those rows are scored. 6 tests pass. RR-08 is a robustness
  note.
- **Manifest v3.** The stated command rebuilds sha256 6efbe422c8f0...e1cf24 exactly. Against the tag: 828 X3 rows
  re-tagged with only n_cells and the draw changed, 8 Scanorama rows knn 160→2, 84 X15 rows added. 6,010 shared rows are
  byte-identical and in the same order, including all A1/A2/A3/X1 rows.
- **run_A1.sh.** Uses manifest v3 and writes to tier12_v2/A1 (absent today). Dry run: 1,098 rows, all needs_fit.
- **JHPCE job.** The tags files equal the 250 pre-freeze GPU rows (X1 none/scvi_adv, X13 scANVI/sysVI) of the 5 SI-39
  tasks: no placeholders, jobA 100 + jobB 150, disjoint. Job order: commit and clean-tree check, node witness and tests,
  slot, input hashes, fingerprints, claims, dry run, run under the stop guard, summary, pack. The rehearsal at the
  review commit fails at the test step (RR-01).
- **GPU smoke (local RTX 3080, 1-2 epochs).** 24 rows ran on 080f417: 14 A1 rows plus X1/X13 pancreas fillers, X15
  'complete' and an X3 importance-weight row. All exited 0 with finite latents. The resolved plans show λ_GP 10 for
  critics only, KL warm-up 400 / 2 / 0 as intended, draw nested_v1 and a 9-batch weight table.

## PREREG sec 9 fact check (21 statements; prereg_sec9_factcheck.csv)

18 statements are correct or consistent. One was not checkable: "no A1 metric value informed any decision" is a process
statement. Two are wrong: the subset-PCR example (RR-02) and the "flags reported with every R1 and R4 input" promise
(RR-03).

Verified values:
- X3 sizes.
- A1 identity 14/14 (re-run).
- Scorer identity 15/15 (plus a 4th latent).
- 202 A1 fits: 202 latents and 202 scores, latent times 17:17:59 to 09:13:36 EDT.
- Scanorama times and memory against scanorama_timing.csv. The knn 80 value of 23.2 GB is the main-process column; the
  tree column, 21.56 GB, is the one PRODUCTION_JOB.md quotes.
- KL weights: 93/400, 81/400, 238/400, 245/400, 399/400, 238/239, 93/94.
- 84 X15 rows. Exactly 12 of them equal X6 rows, as listed in docs/x15_stock_equal_followups.csv.
- Manifest of record and byte-identity.

Against sections 1-8 and the ledger: only the stale sec 8 sentence (RR-05) conflicts. SI-41..SI-46 agree with the code
and the manifest.

## Checks run (all nice 19, 1 thread, CUDA hidden unless stated)

| check | result |
|---|---|
| Five suites at 080f417 (run_suites_rr.sh) | stage+e2e 62 passed; scvi 110 passed / 1 skipped (WCD_REQUIRE_DATA=1); x13+scoring 29 passed; other 311 passed; jhpce 36 passed |
| Manifest rebuild + diff vs tag | sha256 identical; differences exactly as stated |
| A1 identity, tag worktree vs 080f417 (lead's a1_fit_identity.sh, run by the reviewer) | 14/14 identical, max\|dz\| 0 |
| Re-score of stored A1 latent (barycenter λ30 c0 s100) with the 080f417 scorer | 15/15 columns identical |
| check_x3_nested_v1.py (prepped obs, fitter functions) | all checks pass (see table) |
| check_pcr_subset_rr.sh (X3 pancreas dose 0 and 50) | dose 50 0.573082 = all-feature; HVG 0.0; dose 0 0.560350 / 0.0 |
| Runner dry runs (X13 with and without CPU lanes, jobB tags, A1, X1 pancreas) | refusals and counts as designed |
| local_rehearsal.sh jobA / jobB at 080f417 | exit 2 both (RR-01) |
| Same, tag added to the bundle (reviewer's scratch copy of the script) | exit 0 both; 110 passed; fingerprints 8/8; 100 / 150 rows |
| GPU smoke (gpu_smoke_080f417.sh, local RTX 3080) | 24/24 exit 0, finite, plan records as intended |
| Preflight (experiment-preflight + single-cell pack) | JHPCE NO-GO (PF-22); A1 re-stamp GO |

## Not reviewed

- prod_helpers.py: reviewed only for inputs and slots, plus the jhpce simulation tests. The summary, stagekey, pack and
  claims logic was not reviewed line by line.
- Nothing was executed on an L40S node. Real nvidia-smi, env-version and Slurm behaviour remain the job's own on-node
  gates.
- Not reviewed: the gate_report.py / gate_task_split.py / x3_si41_report.py / manifest_v3_report.py changes, the
  wall-time docs, the notebook entries, X8 subset scoring on real latents, and the manuscript.
- Walltime estimate for the JHPCE jobs. The L40S 8-lane factor 4.62 comes from 3-epoch immune fits only and is applied
  untested to scANVI/sysVI (175 of 250 rows). The 24-h request and the stop guard cover this, so it is recorded as a
  caveat, not a finding.
- No novelty or prior-work search; this is a code re-review. No literature was fetched in this round.
