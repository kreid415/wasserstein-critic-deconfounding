# Reproducing the paper (BMC Bioinformatics resubmission, branch `paper`)

Repository: github.com/kreid415/wasserstein-critic-deconfounding. Every number in the paper is listed with its source in the
ledgers `paper/pass7_numbers.csv`, `paper/pass9_numbers.csv` to `paper/pass16_numbers.csv`, the supplement ledgers
`paper/supp-methods_numbers.csv`, `supp-analysis_numbers.csv`, `supp-records_numbers.csv` and `paper/pass17_numbers.csv`, which
`paper/tools/check_ledger.py` re-reads from those sources. `D` below is the data root
`/home/kendall/experiment_data/wasserstein-critic-deconfounding` on the workstation; it holds the inputs, fits and scores, which are
not committed (they are large). Paths of the form `$D/code/wt_*` are git worktrees of this repository at the commits named.

## 1. Which commit produces what

| result | branch | commit | worktree used |
|---|---|---|---|
| fit and scoring code of the sweep and the replication | (tag at 10a1ca5) | 10a1ca5 | `$D/code/wcd_prereg-tier12-v3` |
| sweep analysis (tables, `paper_numbers.json`) | sweep-paper | 8595a16 | `$D/code/wt_sweep_paper` |
| independent check of the sweep placement numbers | sweep-paper | 3280bd4 | same |
| replication across training seeds, with the rule's extension step | sweep-replication | a475ea3 (prediction and rule: tag `replication-prediction-v1`, 3a90d0f) | `$D/code/wt_sweep_replication` |
| direct biology check (run at c4a793f; summary page fixed at 697cc6c) | biology-check | 697cc6c | `$D/code/wt_biology_check` |
| figures, supplement tables, paper text, PDFs | paper | this commit (Pass 15 commits from 0596835 on; supplement merged and placed in Pass 17) | `$D/code/wcd_paper` |

## 2. Data provenance

- The eight tasks are the integration-task datasets of the single-cell integration benchmark (scIB), figshare article 12420968,
  version 8 (doi:10.6084/m9.figshare.12420968.v8). Download: `curl -L -o data_download.zip
  'https://figshare.com/ndownloader/articles/12420968/versions/8'` (README.md, "Data Acquisition"; the dataset-page link in that
  section names version 1, the download is version 8).
- Per-file figshare ids, byte sizes and md5 checksums: `git show 3202725:archive/claude-science/2026-10-02/scib_raw_sources.tsv`
  (on origin/final-review-fixes). `cluster/jhpce/stage_scib.sh download|prep SOURCES_TSV REPORT_DIR` downloads serially, checks bytes
  and md5, runs `scripts/prep_scib_task.py --counts scib` and compares the prepared files with `docs/prepped_fingerprints_scib.json`.
  The prepared files used by every run are `$D/prepped_scib/<task>__scib.h5ad`.

## 3. Environments (exports committed)

| use | environment | export |
|---|---|---|
| fits (workstation) | `scvi-api` conda env (Python 3.11.15, scvi-tools 1.4.2, torch 2.13.0+cu130) | `cluster/envspec/scvi-api.explicit.txt`, `cluster/envspec/local_scvi-api_pip_freeze.txt` |
| scoring (scIB, kBET in R) | `wcd-kbet` conda env, R libraries in `$D/Rlib_kbet` | `cluster/envspec/wcd-kbet.explicit.txt`, `cluster/envspec/local_wcd-kbet_pip_freeze.txt` |
| fits and scoring on JHPCE | `/users/kreid/envs/wcd-fit`, scorer checkout `scorer_10a1ca5` | `cluster/envspec/fit-pip.txt`, `cluster/envspec/score-pip.txt` |
| sweep analysis and its check | `scvi-api` | `experiments/sweep_checks/environment_scvi-api.yml` (sweep-paper); `$D/sweep_analysis_final_20261105/ENVIRONMENT_scvi-api.yml` |
| replication agreement test | `~/wcd_agreement_env` (conda -p, Python 3.11) | `experiments/sweep_replication/agreement_result/environment_agreement.txt` (sweep-replication) |
| biology check | `scvi-api` | `experiments/biology_check/environment_scvi-api.txt` (biology-check) |
| figures, supplement tables, paper tools, tests | `/home/kendall/.claude-science/conda/envs/python` (Python 3.11.15, pandas 2.3.3, numpy 2.4.6, matplotlib 3.11.0, scipy 1.17.1, pytest 9.1.1) | `paper/ENVIRONMENT_paper_tools.explicit.txt`; versions also in `paper/figure_scripts/ENVIRONMENT.txt`, `ENVIRONMENT_supp.txt` |
| PDF build | `latex` conda env: tectonic 0.16.9, pandoc 3.10 | `paper/ENVIRONMENT_latex.explicit.txt` |

## 4. Seeds

- Sweep fits: seeds 100, 101, 102 (`scripts/sweep_inputs.py`, `SWEEP_SEEDS`); baselines (scvi-tools adversary, scANVI, sysVI): seeds
  0-4; Harmony, Scanorama, PCA: one run each at seed 0 (`scripts/run_cpu_baselines.py` at 10a1ca5 refuses any other seed).
- Replication fits: seeds 0-4 (`manifests/sweep_replication_x1.tsv`, `manifests/sweep_replication_ext1.tsv`, column `seed`).
- Scoring: kBET seed 0 (column `kbet_seed` of every score file); on JHPCE the local inverse Simpson's index uses a fixed seed
  (`lisi_fix/fixed_random_device.so`).
- No random numbers in the sweep analysis, the agreement test (exact enumeration of all 2^13 sign patterns), the biology check, the
  supplement tables, `make_replication.py` (2^8 patterns) or the figures, except a jitter of dots in Supplementary Figure S5a
  (`figS5.py`, `np.random.default_rng(0)`).

## 5. Re-run commands, in order (from the repository root of the worktree named)

### 5.1 Sweep fits and scores (commit 10a1ca5)
Run lists `$D/tier12_v3/rules/r4_pass2/m.r4a1.tsv` (atac_small, immune, sim1; workstation) and
`manifests/paper_manifest_stock_pilot_u5_b10_v3.pilot5.tsv` (other five tasks; JHPCE, scores in `$D/jhpce_tier12/pilot5_jhpce/scores`);
baselines `manifests/paper_manifest_stock_pilot_u5_b10_v3.x1_sinkhorn_redo.tsv`. Runner: `scripts/run_stage.py` (workstation) and
`cluster/jhpce/prod_fit_job.sh` (JHPCE). The command lines of those runs are in their logs on the data root, not in the repository.

### 5.2 Sweep analysis (sweep-paper 8595a16), as recorded in `$D/sweep_analysis_final_20261105/RUN_RECORD.md`
```
PYTHON=/home/kendall/.claude-science/conda/envs/scvi-api/bin/python NUMBA_CACHE_DIR=/tmp/wcd-numba-cache \
  bash scripts/run_sweep_analysis.sh $D/sweep_analysis_final_20261105 --baseline-dirs $D/jhpce_tier12/fillers_x1_x13 $D/sweep_baselines
```
(`--require-baselines` was not passed: one baseline row, Scanorama at 80 neighbours on atac_large, could not be fitted; RUN_RECORD.md
explains.) Output and checksums: `$D/sweep_analysis_final_20261105/MANIFEST.sha256`.

### 5.3 Independent check of the sweep placement numbers (sweep-paper 3280bd4)
```
python experiments/sweep_checks/check_sweep_placement.py $D/sweep_analysis_final_20261105 experiments/sweep_checks/out
```

### 5.4 Replication across training seeds (sweep-replication a475ea3)
Design: `experiments/sweep_replication/replication_design.py --out DIR` (from the 10a1ca5 checkout's `scripts/`), run list
`manifests/sweep_replication_x1.tsv` (855 rows, sha256 d8d35481...f225f1), prediction and rule `PREDICTION.json` and `AGREEMENT.md`.
Fits: workstation tasks with `scripts/run_stage.py --no-score` at 10a1ca5 (tags `experiments/sweep_replication/jhpce_stage/local3.tags`,
log `$D/tier12_v3/sweep_replication/tags/local3.log`; the launcher of the extension step, `experiments/sweep_replication/run_ext_local.sh`,
shows the full option list); JHPCE tasks with `bash experiments/sweep_replication/jhpce_stage/run_shard.sh <shard>` (Slurm jobs and
shards in `experiments/sweep_replication/README.md`). Extension step (40 fits, `manifests/sweep_replication_ext1.tsv`):
`run_ext_local.sh` and `run_ext_local_score.sh` (workstation), `jhpce_stage/run_shard_ext.sh` and `jhpce_stage/scoring_ext/*.sbatch`
(JHPCE). Agreement result:
```
PY=~/wcd_agreement_env/bin/python; OUT=experiments/sweep_replication/agreement_result_with_extension
$PY experiments/sweep_replication/score_validity.py --rules-scripts scripts --manifest manifests/sweep_replication_x1_ext1.tsv \
    --scores $D/tier12_v3/sweep_replication/scores $D/sweep_replication/jhpce_scores/scores --out $OUT/score_validity.csv
$PY experiments/sweep_replication/agreement_test.py --rules-scripts scripts --manifest manifests/sweep_replication_x1_ext1.tsv \
    --scores $D/tier12_v3/sweep_replication/scores $D/sweep_replication/jhpce_scores/scores \
    --failures $D/tier12_v3/sweep_replication/failures.csv $D/sweep_replication/jhpce_scores/failures.csv --out $OUT
(cd experiments/sweep_replication && $PY make_report.py)
```
The result before the extension step: the same commands with `manifests/sweep_replication_x1.tsv` and
`--out experiments/sweep_replication/agreement_result` (its README.md).

### 5.5 Direct biology check (biology-check; rules 639c302, run at c4a793f)
```
PY=/home/kendall/.claude-science/conda/envs/scvi-api/bin/python; OUT=$D/biology_check_20261105
$PY scripts/sweep_inputs.py --out $OUT/inputs --baseline-dirs $D/jhpce_tier12/fillers_x1_x13 $D/sweep_baselines
$PY scripts/biology_check.py --inputs $OUT/inputs --prepped-dir $D/prepped_scib --out-dir $OUT/run --jobs 6
$PY experiments/biology_check/make_summary.py $OUT/run $OUT/run/summary.md
```
Committed copy of the outputs with checksums: `docs/biology_check/` (`MANIFEST.sha256`).

### 5.6 Figures, supplement tables and paper text (branch paper; Python of the paper-tools environment)
```
bash paper/figure_scripts/make_figures.sh          # Figures 1-6 from paper/figure_data (checks its MANIFEST.sha256 first)
bash paper/figure_scripts/make_supp_figures.sh     # copies inputs from $D and sweep-paper, Figures S1-S6, Tables S6-S7
python paper/tools/make_replication.py --check     # Table S9 and replication_numbers.json from sweep-replication a475ea3
python paper/tools/make_supp_tables.py --check-table4
python paper/tools/make_supp_supp-analysis.py                         # Table S8 and the label-transfer table of Supplementary Note 9
python paper/tools/make_supp_supp-methods.py --collect                # inputs of Tables S1, S2, S4, S5 from $D (paper/figure_data/supp_methods)
python paper/tools/make_supp_supp-methods.py --check                  # Tables S1, S2, S4 and S5 (run counts) identical to the committed ones
python paper/tools/make_supp_supp-records.py collect                  # facts of Notes 10-11 from $D and git (paper/supp/supp-records_facts.json)
python paper/tools/make_supp_supp-records.py --check                  # generated blocks of Supplementary Notes 10 and 11 up to date
python paper/tools/check_ledger.py paper/supp-analysis_numbers.csv   # numbers in Supplementary Notes 4-7 and 9
```
Supplement: Notes 2-12 and Tables S1, S2, S4, S5 and S8 are input in the supplementary list of `paper/outline.tex` by
`paper/tools/pass17_edits.py` (from dc21cfc; it also writes the Pass 17 corrections). Text passes: each `paper/tools/passNN_edits.py` rewrites the text from the commit its change log names (`paper/passNN_changelog.md`,
section "Re-run"); `paper/tools/test_pass11_12_edits.py` re-runs Passes 11-20 from their starting commits and compares with their
result commits. Clean copy: `python paper/tools/accept_changes.py` writes `paper/outline_clean.tex` (generated, never edited by hand;
`--check` exits 1 when it is out of date).

### 5.7 Number, test and build checks
```
python paper/tools/check_ledger.py                                     # every ledger value re-read from its source
python paper/tools/abstract_words.py                                   # abstract word counts (limit 350)
python scripts/check_tests.py --base f20fe27 --src paper/tools -- paper/tools   # Pass 20 (Pass 19: 2ff8608; Pass 18: 6e97ebf; Pass 17: a2a2c1c; earlier: bfdd82d)
cd paper && tectonic -X compile outline.tex && tectonic -X compile outline_clean.tex   # tracked and clean PDFs
pandoc revision/reviewer_response.md -o reviewer_response.pdf --pdf-engine=tectonic      # response PDF (pandoc 3.10)
sha256sum -c paper/MANIFEST.sha256                                     # from the repository root
```
The PDFs in `paper/build/` were built from the zip of `paper/outline.tex`, `outline_clean.tex`, `refs.bib`, `supp_items.tex`,
`supp/` and `figures/` at d28b747 (Pass 20; identical to this commit for those files). PDF bytes carry a creation time, so a rebuild has
other checksums; the source checksums in `paper/MANIFEST.sha256` are the ones to compare.

## 6. Audit of 2026-11-06 (Pass 15)

Re-run on the workstation, each into a scratch folder and compared byte for byte with the committed result:
- sweep placement check (3280bd4): exit 0; `placement_summary.csv` and `check_summary.json` identical;
- replication agreement with the extension step (a475ea3): exit 0; `replication_agreement_summary.json`,
  `replication_large_gaps.csv`, `replication_readings.csv`, `replication_task_readings.csv` identical; folder MANIFEST.sha256 OK;
- biology check (697cc6c): not re-run (about 1 h 40 min); `docs/biology_check/MANIFEST.sha256` OK (27 files);
- paper: `paper/figure_data/MANIFEST.sha256` OK (16 files), `paper/figures/MANIFEST.sha256` OK (20 files),
  `make_replication.py --check` exit 0, `make_supp_tables.py --check-table4` exit 0 with no change to the tables;
- `check_ledger.py`: 501 values re-read from their sources, 507 pointers checked for the path only, 0 failures;
- `check_tests.py --base bfdd82d --src paper/tools`: OK, 61 tests passed, no WARN;
- both PDFs built with tectonic from the saved zip: tracked 46 pages and clean 40 pages, 13 of 13 figures each, no failing log class
  (2 underfull boxes each).

Not re-run here: the fits and scores (GPU and cluster time) and the sweep analysis itself (about 100 min); their outputs are covered
by the checksum files named above.

## 7. Known gaps

- The command lines of the sweep fits and of the first 855 replication fits and scores are in run logs on the data root, not in the
  repository (5.1, 5.4).
- Older commits of README.md contain a figshare browser session cookie in the data-download command (also tags
  prereg-tier12-v1/v2/v3, replication-prediction-v1 and pull-request refs 2-4). Since 2026-11-06 no branch tip has it (Pass 18
  change log, section Repository); the history was deliberately not rewritten, so the commit IDs cited here stay valid.
- Until Pass 18, `paper/pass16_numbers.csv` was not committed (`*.csv` is git-ignored), so `check_ledger.py` failed on a fresh
  clone; it is committed now.

## 8. Checks of Pass 17 (2026-11-06)

Supplement merged and placed (`paper/pass17_changelog.md`): `check_ledger.py` 1,117 values re-read, 507 pointers checked for the
path only, 0 failures; `check_tests.py --base a2a2c1c --src paper/tools`: OK, 82 passed, no WARN; supplement generators `--check`
identical; tracked PDF 64 pages and clean PDF 55 pages built with tectonic from the saved zip, 13 of 13 figures, no failing log
class; response PDF 8 pages.

## 9. Checks of Pass 18 (2026-11-06)

Supplementary Table S11 numbered, supplementary headings kept with their text, Pass 16 ledger committed
(`paper/pass18_changelog.md`): `check_ledger.py` from a fresh clone on the workstation: 1,117 values re-read, 507 pointers checked
for the path only, 0 failures; tests 83 passed there; `check_tests.py --base 6e97ebf --src paper/tools`: OK, no WARN; tracked PDF 64
pages and clean PDF 55 pages built with tectonic from the saved zip, 13 of 13 figures, no failing log class; every page of the
tracked PDF viewed; response PDF 8 pages.

## 10. Checks of Pass 19 (2026-11-06)

AI-tool statement in Methods (`paper/pass19_changelog.md`): `check_ledger.py` on the workstation: OK, 0 failures;
`check_tests.py --base 2ff8608 --src paper/tools`: OK, no WARN; tracked PDF 64 pages and clean PDF 55 pages built with tectonic from
the saved zip, 13 of 13 figures, no failing log class; every changed page viewed.

## 11. Checks of Pass 20 (2026-11-06)

Declarations: ethics wording drafted, competing interests none (`paper/pass20_changelog.md`): `check_ledger.py` on the workstation: OK,
0 failures; `check_tests.py --base f20fe27 --src paper/tools`: OK, no WARN; tracked PDF 64 pages and clean PDF 55 pages built with
tectonic from the saved zip, 13 of 13 figures, no failing log class; every changed page viewed.
