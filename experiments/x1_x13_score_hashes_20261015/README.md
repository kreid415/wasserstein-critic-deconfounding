# SHA-256 list of the X1 and X13 score files (user decision SI-79)

**What this is.** A fingerprint (SHA-256 hash) of every X1 and X13 score file that exists before the pre-registration tag prereg-tier12-v3, so that anyone can later confirm that none of these files was changed. X1 is the main experiment of the pre-registration (docs/PREREG.md) and X13 is its comparison with standard tools. User decision SI-79 (CONSTRAINTS.md; notebook NB-20261015-03) asks for this list and for the date GitHub shows for the tag, which is recorded in the notebook after the tag as a time stamp from outside this repository.

**No value was read.** `hash_score_files.py` opens each selected file in binary mode only to compute its hash. It never decodes, parses or prints a file's contents; it prints counts only. Under SI-78 a score is examined when a person reads its value or an analysis script uses it; checks that print only counts or flags do not count.

**Files.**
- `hash_score_files.py`: the script (Python standard library only).
- `score_file_hashes.csv`: one row per file: group, kind, path relative to the data folder, size in bytes, last-modified time (UTC), SHA-256.
- `summary.json`: counts per group and kind, hashing time, SHA-256 of the list and of the script.

**Run** (2026-10-15, Python 3.11.15, conda environment `python`; hashing time in `summary.json`: 2026-10-15T14:25:49Z):

    python experiments/x1_x13_score_hashes_20261015/hash_score_files.py --data-root /home/kendall/experiment_data/wasserstein-critic-deconfounding

Result: 827 files; SHA-256 of score_file_hashes.csv `519e7335f2f38f77a59ec7d51ddb52a3ae99bff7dcedd8964a36ab57d4298cb4`.

**Check later.** Run the same command with `--out-dir <new folder>` and compare the new score_file_hashes.csv with this one (for example `diff`). Unchanged files give identical rows. New score files written after this date (the X1 runs, the rest of the filler scoring) appear as new rows; a changed row means a file listed here was modified.

**Which files are selected** (rules in `kind_of()`):
- score files: names starting `X1_` or `X13_` inside any folder whose name starts with `scores` (kind `score`; their scoring logs are kind `scoring_log`, as are `X1_`/`X13_` logs in `logs/score`);
- the identity check's comparison tables `score_identity.csv` (kind `comparison`);
- runner ledgers whose names start with `X1` in a `ledger` folder (kind `ledger`), the runners' `failures.csv` tables (kind `runner_record`) and the X1 and cluster runner logs (kind `runner_log`);
- everything in `review_sealed/` (kind `sealed`).
Not selected: the code checkouts under `code/`, test scratch folders (names containing `pytest`, `codex_work`), latents and model files, embeddings and prepared data, and all pilot (A1, A2, A3) score files except the five pilot rows inside the identity check's comparison tables.

**Inventory** (counts from `summary.json`; descriptions from folder names and the cited notebook entries):

| group | score | scoring_log | comparison | ledger | runner_record | runner_log | sealed | what it is |
|---|---|---|---|---|---|---|---|---|
| X1_local | 19 | 20 | 0 | 2 | 1 | 1 | 0 | The 19 early local X1 runs of 2026-10-12 (immune task; 17 of them adversarial arms), their scoring logs, the runner's ledger and failure table, and the X1 runner log (tier12_v2/X1_local, tier12_v2/logs). Disclosed in docs/PREREG.md; NB-20261014-21 records the audit of how they were read. |
| fillers | 158 | 162 | 0 | 11 | 3 | 4 | 0 | Scores of the 250 X1 and X13 baseline fits made on the JHPCE cluster on 2026-10-06 (X1 arms none and scvi_adv; X13 scANVI and sysVI), scored on this machine in passes A and B (NB-20261007-01: pass B 150 of 150; NB-20261012-02: pass A resumed, now paused), with scoring logs, ledgers, failure tables (including the cluster harvest copies) and the cluster job logs (jhpce_tier12). |
| identity_check | 12 | 12 | 3 | 0 | 0 | 0 | 0 | The identity check of the scorer (2026-10-13; NB-20261013-40, 7 of 7 latents scored identically by the old and new scorer): old and new scores of one X1 adversarial-arm run (X1 immune discriminator, grid point 1) and one X13 run (pancreas sysVI), their scoring logs, and the three comparison tables score_identity.csv (review/identity_20261013, review/identity_c3fd92d_20261014). The comparison tables also hold the five A1 and A3 pilot rows of the check. |
| review_probes | 49 | 42 | 0 | 302 | 0 | 0 | 0 | Outputs of the final pre-publication review of 2026-10-12/13 (docs/review_final/REVIEW_FINAL.md) under review/out and review/verify: 49 score files whose tags are registered X1 or X13 configurations in the committed manifests (same task, arm, grid point, decoder and seed as a registered row) - 42 from fits the reviewers made themselves on sim1, atac_small and pancreas, including adversarial arms, at the code of 2026-10-12 with training settings not checked here (these are not the registered production fits), and 7 that rescore 6 real X1 baseline latents (arm none: five filler latents at seed 0 and the local immune run, the latter twice) - their scoring logs, and runner ledgers, including runner test cases with constructed inputs (folders runner_cases*, test_tmp2, verify/codex_hi). |
| x13_cpu_test | 12 | 12 | 0 | 0 | 0 | 0 | 0 | Two test runs of the X13 standard-tool runner (Harmony, Scanorama and PCA) on atac_small on 2026-10-02, before tag prereg-tier12-v1 (x13_cpu_baselines_atac_small_test; NB-20261002-09; NB-20261002-10 records that kBET and Harmony were not reproducible run to run in these tests). |
| sealed | 0 | 0 | 0 | 0 | 0 | 0 | 2 | Two files made on 2026-10-12 (clean_snapshot.bundle, planted_key.json in review_sealed); by their names, the material of the blinded planted-error test reported in docs/review_final/REVIEW_FINAL.md. Not score files; hashed because the reviewers' blinding rules list this folder. |

## Audit: were X1 or X13 score values displayed? (2026-10-15)

`audit_displayed_values.py` searched (a) the log, text, markdown and JSON files of the final pre-publication review under review/out and review/verify (26722 files; 268 mention one of the score files) and (b) the app's stored transcripts of every session of this project active since 2026-10-02 (56 sessions, the three running round-3 fixers excluded; 45 mention one of the score files). Identifiers: the file names of the score files of groups review_probes, x13_cpu_test and identity_check above. A line is flagged when it holds an identifier and three or more decimal numbers, or a metric name and two or more decimal numbers inside a block that mentions an identifier or one of the folders. Only counts and lines with every digit replaced by # were printed, so the audit itself showed no value. Output: `audit_displayed_values.json`.

Result: two displays.
1. 2026-10-02, before tag prereg-tier12-v1: a sub-agent's tool output (session 'Missing arms', message 327) showed all metrics of six test fits (Harmony d10 and d50, Scanorama d10 and d100, PCA d10 and d50) from one of the two X13 runner-test runs on atac_small; these test configurations are not registered rows. Notebook NB-20261002-10 and the 2026-10-02 page also record run-to-run differences of NMI, ARI and iLISI and a kBET value from these tests.
2. 2026-10-13: in the lead agent's session (message 554), the scorer identity check printed the old and new values, to 16 digits, of two metrics (PCR_batch, a batch-removal metric, and cell-cycle conservation, a biology metric) for one X1 adversarial-arm run (immune, discriminator, grid point 1, decoder 1, seed 0) and one X13 run (pancreas, sysVI). The X1 analysis specification was drafted later that day (2026-10-13 evening) and decided on 2026-10-14.
No value from the 49 score files of the final review's probes was found in any transcript or review file. Every flagged line was classified. Of the 355 lines flagged in review files, 351 are verbatim copies of lines of the committed notebook page 2026-10-02.md (repository copies inside test folders; the same X13 runner-test entries as above) and 4 are lines of code-coverage reports (coverage percentages). The other flagged transcript lines are power-report numbers, test counts, project descriptions and known-answer fixture values of 2026-10-02.

Limits: the patterns can miss a value printed with no identifier, folder name or metric name nearby; sessions run by Codex left only their output files, which were scanned; what an agent read without printing cannot be known. This audit corrects NB-20261014-21, which searched only for reads of tier12_v2/X1_local/scores.
