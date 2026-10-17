# Pipeline track review (Part B items 5, 6, 7, 8, 10, 11; Part 0 item 3 environments)

Reviewer: claude-opus-5-5. Review copy: `review/wcd_review_copy` (single commit aa7a27d, a snapshot of main bc69b8c = tag c88cce3 plus the X1 files). Every probe ran from this copy, on CPU (`CUDA_VISIBLE_DEVICES=`), at nice 19, with 2 threads. All outputs are in this folder. Nothing was written under `tier12_v2`, `jhpce_tier12`, `prepped_scib` or the copy's tracked files. `X1_local` and `fillers_x1_x13` were only read (listing files and reading score/latent configs).

## 1. Verdicts

- **Cluster launch (5 JHPCE jobs, 1,800 fits): GO AFTER FIXES.** Two items must be fixed before launch:
  - **F-01.** The R1 noise estimate in the copy's `prereg_rules.py` does not reproduce the frozen R1 record. The record matches the PREREG text. Someone must establish which code produced the record before 1,800 fits are launched on its λ values.
  - **F-06.** The run clock of the chained jobs is too short.

  None of the findings changes a value in the X1 design file: my independent R1–R4a1–R2/R3 recomputation reproduces every λ, adv_input and zstd in it.
- **Local X1 fits (19 scored + 1 fitted, paused): VALID.** All 20 latents were fitted at c88cce3, clean, on the RTX 3080, and their configs equal the X1 manifest rows. All 19 score rows have every required metric finite and kBET seed 0. Condition: F-05 (the runner gate accepts a NaN bio metric) must be fixed, or the final X1_local scores re-checked, before the local stage's `.done` marker is trusted.

## 2. Findings (severity-ordered)

| ID | Part | Severity | Location | Design statement / claim | Evidence (command → output) | Consequence | Fix |
|---|---|---|---|---|---|---|---|
| F-01 | B5 | fix before launch | scripts/prereg_rules.py:370 `float(np.std(d))` (ddof=0) | PREREG §2: "seed SD … divided by its median-unbiasing constant √(χ²_{n−1,0.5}/(n−1))" (that constant belongs to the n−1 sample SD) | `freeze_a1_grid.py` from the copy on the stored A1 + R1-extension scores (rules/r1.json) gives w1\|0 x1_grid [0.1,0.3,1,3,10,30] vs record [0.3,1,3,10,30,100], and mmd\|1 [1,3,30,100,1000,3000] vs record [3,10,30,300,1000,3000]. sigma_b for w1\|0 atac_small\|reference is 0.016143 (copy) vs 0.019771 (record); the ratio is √(3/2). My ddof=1 implementation reproduces all 8 grids, 36 blocks and 372 points of r1_record.json exactly (max abs diff 0). The record says git_sha c88cce3, git_dirty false. | Either the code under review is not what produced the frozen record (a provenance contradiction at the tag), or R1 cannot be reproduced from committed code. A re-run moves 200 X1 rows (120 W1-unconditioned, 80 MMD-conditioned) to λ values that were never fitted. No test catches it: 241 tests pass, and the one test that would catch it (test_rules_unchanged) needs git history. | Establish whether the tag holds ddof=0. Restore `np.std(d, ddof=1)`. Add a known-answer unit test on robust_noise_sd (n=3 → the √(3/2) factor). Re-run freeze_a1_grid and require a byte-identical r1 manifest. |
| F-06 | B8 | fix before launch | experiments/X1_jhpce/x1_jhpce_command.py:58,108 | SUBMISSION_PLAN: chains 1→3 and 2→4→5 with afterany | `run_timeout_s = wall_s(walltime) + 120 h` = 72 + 120 = 192 h for every job. Job 5 starts only after jobs 2 and 4 end (up to 2 × 72 h), then runs up to 72 h: 216 h plus queue time. | The harness run clock can expire on job 4 or 5 while it is still fitting or queued. GPU-days are lost and a resume is forced. | Set run_timeout_s per job to (number of jobs in the chain up to and including this one) × walltime + queue allowance, or submit each chained job only after its predecessor ends. |
| F-05 | B7 | fix before analysis (before any X1 gate is trusted) | scripts/run_stage.py:459 (`nonfinite = [m for m in batch …]`); PREFLIGHT SC-17 | PREREG §1: "all 5 batch metrics and the task's bio metrics must be finite". run_stage docstring l.433-436. SC-17: "the runner fails any row with a non-finite required metric" | Probe (probe_gate): A1 immune score with trajectory or NMI set to '' or 'nan' → `Outputs.state` = "scored"; kBET → refused. Full stage (probe_stage/out_nanbio): NMI blanked, `--report-only` → "[gate] PASS: 3/3 final", exit 0. The rule side refuses the same row (prereg_rules.outcomes). | The completion gate and `.done` marker can certify a stage that the rule scripts and P1–P4 must reject. The SC-17 preflight claim is false. | Apply the finiteness check to `bio` as well (line 459); add a test with a blank bio metric. |
| F-03 | B5 | fix before analysis (before R4-x1) | prereg_rules.py:537-539 | PREREG §4: "M_a = the highest seed-mean B of arm a over its grid points with no failed seed" | Probe: one failed seed at barycenter atac_small c0 λ=10 (the arm's argmax) → code M_a 0.63828 (mean of the 2 remaining seeds) vs 0.61936 per the text | With any failure in X1, b_common and b* (and so every follow-up's matched λ and P1's b*) differ from the pre-registered rule. No effect on R4-a1 (0 A1 failures). | Use only points with `n_fail == 0` in M_a (as `_closest` already does); add a test. |
| F-02 | B5 | fix before analysis | prereg_rules.py:619 | PREREG §3 criterion 4: "the alternative has no more failed fits than the default over the windows" | Probe: 12 synthetic cells, the alternative has 2 failures in the one non-evaluable cell → `switch=True`, failures {mean:0, sample:0} | Failures that make a cell non-evaluable are invisible to criterion 4. R2/R3 are frozen with 0 failures, so there is no effect now. | Sum n_fail over all cells' windows. |
| F-07 | B7 | fix before analysis | run_stage.py:1148-1152 | PREREG §9(b): every stage fitted and scored from the tag; "the runner's mixed-commit check" | Probe stage: latents fitted at aa7a27d (not the tag) → gate ok, latent_git_shas [aa7a27d…]. The gate requires one SHA, not the tag; latents without git_sha are skipped by the set. | Local scoring of harvests (or a re-run) from another commit would pass. Cluster jobs are protected by prod_fit_job.sh's HEAD == tag check. | Add `--expect-git-sha` (gate: latent_git_shas == [it], and none missing). Use it in the scoring launcher. |
| F-08 | B7/B8/A | fix before analysis (not implemented) | — | PAPER_PLAN §5 (P1–P4, Bayesian model, Wilcoxon/Holm, failure imputation + Clopper–Pearson, transfer λ, measurement-SD re-scoring); PREREG §4 R4-x1 application; SI-38; launcher that downloads, verifies and scores X1 harvests | grep: no P1–P4/endpoint code, no SI-38 code (CR-13 still open), no X1 harvest scoring launcher (only fillers_x1_x13_local_scoring/run_scoring.sh); harvest_local.py verifies and merges only | Failures could still be dropped silently at analysis time, contrary to "Failures are outcomes". | Write and pre-review these before X1 scoring ends. The scoring launcher needs `--expect-device L40S`, the tag checkout and the expected latent SHA (F-07). |
| F-04 | B5 | fix (text) before analysis | docs/PREREG.md §1 constants table | "ROPE \| 0.02 \| R2, R3" | §7: "ROPE 0.01 … were not changed"; SI-19/20 "> 0.01"; prereg_rules.py:47 ROPE = 0.01; all 3 records constants ROPE 0.01 | The frozen text is internally inconsistent. The R2/R3 decisions do not change at either value (mean D −0.0014 / 0.0010). | Errata entry: table → 0.01. |
| F-09 | B5 | note | prereg_rules.py:73 | PREREG §9: rule inputs with "mixed or missing provenance" refused (host, CPU model listed) | Probe: score_host and cpu_model altered on 2 A1 rows → accepted (5 of 7 columns checked) | The cross-host scoring guarantee rests on cpu_simd/numba only. | Add score_host/cpu_model, or reword §9. |
| F-10 | A/B6 | note (claims) | docs/PAPER_PLAN.md §3 table, P1 | "Δbio@b* (reference critic − discriminator ×10 steps)" | 0 discriminator rows with n_critic > 1 in 6,975 manifest rows (SI-01); PREREG §5 uses the 1-step discriminator | P1 as written cannot be computed. | Amend P1 to the 1-step discriminator. |
| F-11 | Part 0.3 | note | environment.yaml, conda-lock.yml, Dockerfile | Reproducible environment | environment.yaml has no scvi-tools/torch and an unpinned scib; conda-lock pins numba 0.61.2 and torch 2.5–2.7; the envs used are scvi 1.4.2/torch 2.13.0/numba 0.67.0 (fit) and scib 1.1.7/numba 0.66.0 (score) | A reader rebuilding from the root env files gets a different stack. | Point README to cluster/envspec/*.explicit.txt + pip freezes; retire the root files. |
| F-12 | B5 | note | PREREG §4 | "A1: A1 grid, seeds 100–102" | The R4-a1 record is reproduced only if the R1 extension points (0.01, 0.03; 1e4, 3e4) are part of the stage-a1 grid (prereg_rules.py:536) | Ambiguous text; the code's reading is reasonable. | Clarify in errata. |
| F-13 | B5 | note | docs/prereg/r4_a1_record.json, r2r3_record.json inputs | §6: records hold their inputs | 3 inputs point to `…/workspaces/<id>/scratch/wt_int/…` (absent). Their SHA-256 equal the repo copies of r1 manifest, r1 record and r4a1 record. | Paths are not analysable later; the hashes are. | Record repo-relative paths. |
| F-14 | B7 | note | run_stage.py:409, 703 | SI-17 device check | `--expect-device cpu` accepted CPU fits of GPU arms (substring) | Only a misuse risk; the launchers hardcode L40S / RTX 3080. | Optional: an exact device list. |
| F-15 | B8 | note | prod_fit_job.sh:68; run_stage.py:1037-1055 | Two concurrent jobs, one OUT (tier12/x1_jhpce) | Claims are per tag and ledgers per tags-file key. failures.csv is rewritten by both runners from all status records (atomic; union), and each harvest packs that shared failures.csv. | Harmless if the local scorer rebuilds failures from status/ (harvest_local does). | None needed; document. |
| F-16 | B8 | note | experiments/X1/PREFLIGHT.md PF-24, PF-17 | "walltime 3 d vs ≤ 1.9 d predicted per job (2.7 d at 1.5×)" | SUBMISSION_PLAN table: immune_hum_mou 2.2 d predicted; PF-17 "52.27 h" does not match the plan's 8.5 GPU-days; the adversarial arms have not been timed on L40S | Walltime margin is overstated in the preflight. | Reconcile the numbers; time one adversarial arm on L40S in job 1's first hour (the ledger shows it). |
| F-17 | B8 | note | prod_helpers.py slot / afterany chain | Resume path | If job 1 ends at walltime with fits left, afterany starts job 3. With job 2 or 4 running, both slots are held and a resume job is refused (exit 7) until one ends. | Manual sequencing is needed. | State this in SUBMISSION_PLAN. |
| F-18 | A (R4) | note | r4_a1_record targets immune\|1, sim1\|1 | `bstar_low_signal` | immune\|1: b_common − b0 = 0.6570 − 0.6382 = 0.0188 < 0.02. Half the A2/A3 immune-conditioned cells compare bio at a b* only 0.0094 above λ=0. | The R2/R3 evidence for the conditioned immune block is weak. | Disclose with R2/R3. |

## 3. Traceability matrix

[traceability_matrix_pipeline.csv](traceability_matrix_pipeline.csv): 75 rows covering PREREG §0–§9 (rule-relevant), PAPER_PLAN §3/§5 statements in this track, SI-01/13/16–21/26/27/34/36/44/46, the runner, the cluster checks, environments and CR-01…CR-14. Each row gives the design statement, file:line, evidence and verdict. Verdicts: 41 matches, 11 deviates (incl. text errors), 3 not implemented / open, 1 untestable here; the remainder are CR-fix and qualified-match verdicts (see the CSV).

## 4. Independent recomputation of R1, R4-a1, R2, R3

I implemented the rules from the PREREG text before reading prereg_rules.py. Code: [independent_rules.py](independent_rules.py); summary: [independent_rules_comparison.json](independent_rules_comparison.json).

- **R1 round 1** (A1 only): extension needed for JS cond0 (low edge: immune, sim1) and Sinkhorn cond0 (high edge: sim1). This equals a1_extension_r1_round1.tsv; the copy-code extension manifest is byte-identical (`cmp`).
- **R1 final** (A1 + R1 extension): 8/8 grids equal r1_record.json:
  - js\|0 [0.03,0.1,0.3,1,3,10]; js\|1 [0.1,0.3,1,3,10,30];
  - w1\|0 [0.3,1,3,10,30,100]; w1\|1 [0.1,0.3,3,10,100,300];
  - mmd\|0 [3,10,30,100,300,1000]; mmd\|1 [3,10,30,300,1000,3000];
  - sinkhorn\|0 [0.3,3,10,100,300,3000]; sinkhorn\|1 [0.1,0.3,1,3,10,30].

  Per-block floor, peak, upper, σ_b and σ_c agree on 36/36 blocks; db, dc, δ_b, δ_c, no-effect and collapsed agree on 372/372 points (max abs diff 0). **The copy's code differs (F-01).**
- **Power** (copy code = record): σ 0.029837, τ 0.109380, MDE80 (D=6) 0.1307, underpowered = True.
- **R4-a1**: round-1 extensions are immune c1 discriminator/reference/pooled at 0.03 (= a1_extension_r4_round1.tsv). After the extension, all 36 cells (matched, lo, hi) and 6 targets (b0, b_common, b*, low-signal flag) match r4_a1_record.json. The 4 sim1\|1 cells with lo = 0.03 are flagged edge_not_extended, as in the record. The copy code regenerates the r4a1 manifest body byte-identically.
- **R2**: 12/12 cells evaluable, mean D −0.00145 → keep posterior mean. No masking cell.
- **R3**: 17/20 evaluable (atac_small c0 and immune c0 Sinkhorn unreached-high; immune c1 MMD unreached-low), mean D 0.00101 → keep standardisation off.

  Per-cell bio@b* and statuses for both rules are identical to r2r3_record.json (max abs diff 0). The copy code regenerates the committed X1 manifest byte-identically: whole-file sha256 7c2fe8679631c882…, equal to the design-file hash. The regenerated r4a1 manifest is body-identical; its whole-file hash differs (846caf9d… vs ca28d991…) only through the header comment lines naming the record path.
- **Design files**:
  - The builder reproduces the manifest of record (sha256 6efbe422c8f09f98…).
  - The manifest chain changes only the columns the rules own: v3→r1 changes the lam of 2,880 X1 rows and adds 36 A1 extension rows; r1→r4a1 changes the lam of 288 A2/A3 rows and adds 9 A1 rows; r4a1→x1 changes adv_input/zstd of 2,880 X1 rows. No tag changed.
  - The seed sets are disjoint: pilots 100–102, X1 0–4, follow-ups 10–12; X13 uses 0–4 by design.
  - The 250 filler rows are identical between v3 and x1.
  - The 1,800 JHPCE tags plus 75 filler rows cover exactly the 1,875 X1 rows of the 5 JHPCE tasks.
  - The only placeholders left are the follow-ups' matched/matched_lo/matched_hi (1,992 rows), each mapping to an X1 base cell.

## 5. Runner, cluster and earlier findings

- **Runner probe** (probe_stage): a 3-row X1 atac_small stage at 1 epoch on CPU, fitted and scored by the copy's runner.
  - Dry run exit 0; run exit 0, gate PASS 3/3; peak RSS 2.1 GB.
  - Resume plans 0 fits, gate PASS.
  - With a blanked NMI the gate still passes (F-05).
- **prod_helpers probes** (probe_helpers):
  - inputs: OK → exit 0 (360 tags); wrong task → 8; wrong manifest SHA → 8.
  - deadline: 4,000 s left → 9.
  - slots: overlapping task → 7; third live job → 7; slot reclaimed after COMPLETED.
- **Every check that SUBMISSION_PLAN lists is in prod_fit_job.sh and can fail:**
  - HEAD/clean → exit 3;
  - one L40S, version record, tests/scvi → exit 2;
  - slot → exit 7;
  - inputs and fingerprints → exit 8;
  - claims → exit 7;
  - dry run → exit 4;
  - stage key → exit 11;
  - deadline guard → exit 9 at start.
- **X1 builder vs the tagged prod_command.py:** the same Slurm lines and body. The X1 builder adds the x1_inputs staging (sha256sum -c, cp), an optional afterany dependency, a 28,800 s fit timeout and the tag-only check.
- **X1 rehearsal vs local_rehearsal.sh:** the diff is limited to the builder call and the input-staging block.
- **CR-01…CR-14:** fixes are present for CR-01, 02, 03, 05 (partial: F-09), 06, 07, 08, 09, 10, 11, 12 and 14. CR-04 is present (correctness of the nested draw was not re-derived here). CR-13 is still open (F-08). Details are in the matrix.
- **Tests** (scvi-api env, CPU): tests/prereg, tests/stage, tests/jhpce, test_legacy_retired and test_gate_task_split gave 241 passed, 1 failed, 9 errors and 2 skipped.
  - The failure (test_manifest_v3::test_v2_is_the_builder_output_of_the_previous_commit) and the 9 errors (test_rules_unchanged) need git history (`git show eb444fe`). I do not count them as code findings.
  - Skips: scib is absent in scvi-api; the e2e test is opt-in.
- **Fail-loud scan** (failloud_grep.txt, 64 hits in the active rule, runner, fitter, scorer, cluster and X1 files):
  - The broad excepts in experiments/X1_local/env_record.py record the error string (provenance only).
  - run_stage.py:1000 re-raises.
  - score_scib_native.py:175-176 `skipna` only feeds the unused scib_overall.
  - src/wcd_vae/wcd/{experiment,hyperparameter,training}.py have broad excepts; whether that code is reached by the active path belongs to the coverage track.
  - No result-selecting silent fallback was found in this track's files beyond F-02, F-03 and F-05.
- **Environments:**
  - Local fit (scvi-api): scvi 1.4.2, torch 2.13.0+cu130, numba 0.67.0.
  - JHPCE expected: identical modules except torch +cu126 (declared). By SI-17 a task's arms never mix machines.
  - Score (wcd-kbet, local only): scib 1.1.7, scanpy 1.11.5, numba 0.66.0, pynndescent 0.5.13, R 4.5.2, kBET 0.99.6.
  - The tag's cluster/jhpce/env.sh and build_envs.sh (the ones the jobs run) differ from the copy's and could not be read (blinding).

## 6. Not checked (with reasons)

- **The tag's versions of cluster/jhpce/env.sh, build_envs.sh, PRODUCTION_GO_SUMMARY.md and tests/jhpce/test_prod_job.py, which the cluster jobs actually run:** the copy holds main's versions and reading other copies is forbidden.
- **rehearse_x1_job.sh end-to-end:** it needs commit c88cce3 in the repo, and the single-commit copy lacks it. I ran its component checks (prod_helpers) instead.
- **R4-x1 on real data:** there are no X1 scores yet. Code was read, and every follow-up matched row maps to an X1 base cell.
- **harvest_local.py merge and conflict paths:** no harvest of X1 exists; I did not probe it with a synthetic tarball.
- **Measured line coverage (Part 0 item 2):** not in this track. My probes were not run under coverage.
- **Correctness of x3_draw (CR-04), scorer metrics and loss functions:** other tracks.
- **JHPCE read-only commands:** none were run.
- **Re-scoring of existing latents:** scoring track. I re-scored none; the probe stage scored 3 new 1-epoch latents.

## 7. High-impact areas (for escalated double-check)

1. **F-01, R1 reproducibility.** Which prereg_rules.py produced r1_record.json? A wrong answer means either the frozen X1 grid is not reproducible from the tag, or (if the record is wrong) 200 X1 rows carry the wrong λ. My check rests on one independent implementation plus the copy code; the tag's file itself was not read.
2. **F-03, R4-x1 M_a with failures.** It sets every follow-up's matched λ and P1's b*. It only bites if X1 has failures; untested on real data.
3. **F-05 and F-08, endpoint robustness.** The runner gate passes NaN bio metrics, and the P1–P4 / failure-imputation code does not exist. An error there changes the published endpoints.
4. **F-06, chained-job run clock.** It can waste GPU-days and force resumes. The harness semantics of run_timeout_s (submit-relative clock) are taken from prod_command.py's own comment, not verified on the harness.
5. **Weakest checks:** the tag-only cluster env scripts; the rehearsal (not run); harvest_local (not probed).
