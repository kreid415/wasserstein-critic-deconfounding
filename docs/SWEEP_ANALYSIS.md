# Sweep analysis pipeline (eight-task penalty-strength sweep with baselines)

Written 2026-11-01 on branch `sweep-paper`, after the rescope of 2026-11-02 (`docs/RESCOPE_20261102.md`, SI-116, SI-117): the main
benchmark (X1) is not run, and the paper's results are the penalty-strength sweep (experiment A1) on all eight scIB tasks, every fitted
strength, both decoders, seeds 100-102, compared with the baselines of the committed run list. The analysis definitions are those of
`docs/ANALYSIS_SPEC.md`; section 3 below lists every point where the specification could not be applied as written and what the code
does instead.

## 1. Command

```
# from the repo root; python = /home/kendall/.claude-science/conda/envs/scvi-api/bin/python (the default when it exists; else set PYTHON)
scripts/run_sweep_analysis.sh <out-dir> --baseline-dirs $D/jhpce_tier12/fillers_x1_x13 $D/sweep_baselines --jobs 8 [--require-baselines]
```

`D` = `/home/kendall/experiment_data/wasserstein-critic-deconfounding` (environment variable `WCD_DATA_ROOT`; read only). The directory
`<out-dir>` must not exist. Stages, each timed in `timings.tsv` and logged in `logs/`:

| stage | script | writes |
|---|---|---|
| inputs | `scripts/sweep_inputs.py` | `inputs/manifest.tsv` (every row the analysis reads), `inputs/scores_list.txt`, `inputs/failures.csv`, `inputs/latents.tsv` (tag to latent file), `inputs/baseline_coverage.csv`, `inputs/inputs.json` (sources with SHA-256, counts, missing baseline rows) |
| analysis | `scripts/analysis_sweep.py` (shared code in `scripts/analysis_x1.py`) | `analysis_sweep.json`: the record (inputs' SHA-256, git commit, constants, primary analysis, four sensitivity analyses, label transfer per embedding, baseline coverage) |
| report | `scripts/sweep_report.py` | `tables/*.csv` (13 tables), `figures/*.png` and `.pdf` (5 figures), `paper_numbers.json` |

Then `MANIFEST.sha256` over the record, the tables, the figures and the input files. Options:

- `--baseline-dirs DIR ...`: runner-layout directories (`scores/<tag>.csv`, `latents/<tag>.npz`, `failures.csv`). Only files whose tag is
  one of the 448 baseline rows are read; a tag scored in two directories is refused. Baselines arriving later in `D/sweep_baselines/` are
  picked up by adding that directory.
- `--require-baselines`: any of the 448 rows neither scored nor failed stops the run and the message names the rows. Without it the run
  proceeds with what exists; the record (`baselines.complete`, `baselines.missing`, `baselines.coverage`) and `paper_numbers.json`
  (`_meta.baselines_complete`, `baseline_coverage.*`) say so.
- `--jobs N`: processes for label transfer. Each embedding is one independent computation collected by tag, so the output is identical
  for every N (checked by `tests/analysis/test_sweep_analysis.py::test_sweep_label_transfer`).
- `--tasks T ...`: a subset of the eight tasks (plumbing checks only; the record is marked `partial_tasks`).
- `--debug-label-transfer-subset N`: DEBUG ONLY, label transfer on the first N embeddings (sorted tags) per task; everything else is
  reported as not computed, and the record and `paper_numbers.json` carry the flag (`label_transfer_debug_subset`).

Inputs, by source (section 2 of `scripts/sweep_inputs.py`'s docstring has the paths):

| rows | run list | scores | latents | failures |
|---|---|---|---|---|
| atac_small, immune, sim1 (1,143 A1 rows) | `D/tier12_v3/rules/r4_pass2/m.r4a1.tsv` | the 1,143 files of `scores_list.txt` (the Sinkhorn redo superseded `tier12_v2/A1` Sinkhorn files under the same tags: the listed file wins) | `latents/<tag>.npz` next to the listed score file | `failures_merged.csv` (empty) |
| immune_hum_mou, atac_large, lung, sim2, pancreas (1,890 A1 rows) | `manifests/paper_manifest_stock_pilot_u5_b10_v3.pilot5.tsv` | `D/jhpce_tier12/pilot5_jhpce/scores/` | `.../latents/` | `.../failures.csv` (empty) |
| baselines (448 rows: X13, plus scvi_adv rows labelled X1) | `manifests/paper_manifest_stock_pilot_u5_b10_v3.x1_sinkhorn_redo.tsv` (restored to this branch from commit 2168d75) | `--baseline-dirs` | same | same |

Provenance checks are unchanged and on: one scorer commit after the SI-54 equivalence (`c88cce3` rows of non-Sinkhorn arms read as the
tag `prereg-tier12-v3`, commit `10a1ca5`), one kBET seed, one package-version string, and the SI-115 site rule for the rows scored on
JHPCE with the fixed LISI seed (immune_hum_mou, atac_large). SI-115 was ported from `origin/pilot5-jhpce-scoring` (commits a7aed00,
abd4e5f): `prereg_rules.refuse_unknown_scorer_sites`, `docs/SCORER_EQUIVALENCE.json`, `docs/scorer_identity_check_20261028.json`, its
tests (`tests/prereg/checks.py::check_scorer_sites`, five mutants in `test_mutation.py`, two tests in `test_provenance_rule.py`) and the
ledger entry SI-115 in `CONSTRAINTS.md`. In atac_large every score file has blank `cell_cycle_conservation`, `hvg_overlap` and
`trajectory`; none of them is a required metric of that task (`prereg_rules.bio_metrics_for`), so the rows are complete.

## 2. Design

`scripts/analysis_x1.py` keeps its X1 entry point (`main`, `analyse`) and its tests; the ranking analysis of one set of pairs was moved
into `analyse_pairs(out, exp, seeds, targets, grids, matched, ...)`, which the X1 path calls with the six-strength grids of the R1 record
and the R4 record's targets, and the sweep path (`scripts/analysis_sweep.py`) calls with every fitted strength and targets computed from
the sweep rows. Label transfer (`transfer_labels`, `label_transfer_many`) computes each embedding once, in a process pool when
`--jobs` > 1, and the per-embedding results are kept in the record (`label_transfer.per_tag`).

What the sweep analysis computes, per pair (task x decoder) from the three seeds, both decoders, all eight tasks:

- b0 (plain scVI's mean batch score), each penalty's best failure-free seed-mean batch score M, b_common = min M, b* = b0 + 0.5 (b_common - b0),
  low-signal flag (b_common - b0 < 0.02), by `prereg_rules.pair_targets` (the formula of rule R4, factored out of `r4_stage`).
- Per penalty: the curve (anchor at plain scVI, then every fitted strength in increasing order, each point the mean over its successful
  runs), the reading at b* (`bio_at_bstar`), reading status, crossings, rank within the pair (bottom places shared, average ties), the
  three leave-one-seed-out repeats with their standard error and rank range, local minus global batch removal at the crossing, frontier
  area, collapse rate with the exact interval, good-strength width, label transfer at the matched strength and along the whole curve.
- Per decoder: mean rank over the ranked real tasks; simulations listed by task.
- Baselines (as in spec sections 10 and 11): single-setting methods as they are (plain scVI, scvi_adv, scANVI, PCA, the tools' default
  dimensions), transfer setting for Harmony, Scanorama and sysVI, best-of-k per family and its mean over the real tasks, frontier positions.
- The four sensitivity analyses of spec section 12 (biology without trajectory, per-run reading, points with a failed run dropped,
  area corner at biology 0).

`paper_numbers.json` is flat: `{key: {"value", "description"}}`, keys such as `rank.<penalty>.<task>.cond<0|1>`,
`bio_at_bstar.*`, `mean_rank.cond<c>.<penalty>`, `mean_rank_n_tasks.*`, `collapse_rate.*`, `collapse_ci.*`, `area.*`, `width_decades.*`,
`local_minus_global.*`, `label_transfer_f1.*`, `best_of_k_real_mean.cond<c>.<family>.k<k>`, `transfer_strength.*`, `single.*`,
`n_fits.*`, `n_failed.*`, `n_pairs_no_reading.cond<c>.<penalty>`, `baseline_coverage.<task>.<method>.<expected|scored|failed|missing>`,
and `_meta.*` (record SHA-256, git commit, partial-tasks and debug flags, baseline completeness).

## 3. Where the specification cannot apply as written

Each item: what the specification says, what the code does, why.

1. **Points.** Spec 1 and 2.1: the six grid strengths of the R1 record per family and decoder; extension rows are not points. Code: every
   strength fitted in the sweep is a point (10 base strengths; 12 for the discriminator with the unconditioned decoder and for Sinkhorn with
   the conditioned decoder on all eight tasks; 11 for the discriminator, pooled and reference critics with the conditioned decoder on
   immune, whose R4 extension row at 0.03 was fitted). Why: the rescope reads every sweep strength and no grid rule is re-run (SI-116).
   Consequence stated in the record's counts: the penalties have 10-12 tries in best-of-k where the tools have 6, so compare at equal k.
2. **Records.** Spec 1 and 16.1: b0, b_common, b* and the low-signal flag are read from the frozen R4 stage-x1 record and recomputed from the
   rows; the R1 record gives the grids. Code: no R1 or R4 record exists for the sweep; the targets are computed from the sweep rows with the
   same formula (`pair_targets`), every point counted, and written into the record (`cells.*.b0, b_common, bstar, best_batch`).
3. **Seeds.** Spec: five seeds 0-4 and `X1_SEEDS`. Code: the sweep seeds 100-102 (three); the leave-one-seed-out analysis has three repeats.
4. **Standard error threshold.** Spec 4 (i): the jackknife standard error is reported only when m >= 4 of the 5 repeats give a reading (at most
   one repeat without a reading). Code: m >= 2 of 3 (`min_readings = n_seeds - 1`, the same "at most one repeat missing"); `loso_m` is
   reported beside it. With the five-seed threshold no standard error could ever be reported. A standard error from two or three
   repeats describes the data weakly; the user may prefer to drop it.
5. **Every curve reaches b*.** With every strength a point, b_common is the lowest of the penalties' best failure-free batch scores, and each
   curve starts at b0 < b* and contains a point at or above b_common >= b*, so in a ranked pair every penalty's three-seed curve brackets b*
   and the bottom-places rule of spec 3.1 applies only in the leave-one-seed-out repeats and in the per-run sensitivity analysis. The
   record keeps the status of every reading; `n_pairs_no_reading.*` counts the pairs whose curve gives no reading (expected 0 outside
   low-signal pairs).
6. **Anchor at or above b* (the known mismatch, fixed).** Spec 4 (i) and 16.2: an anchor at or above b* gives no reading. The old code
   (`prereg_rules.bio_at_bstar`) returned a reading for a curve that starts above b* and falls through it. Code: `analysis_x1.read_arms`
   gives status `anchor_at_or_above` and no reading when the anchor's batch score is >= b* - 1e-9 (full reading and every repeat; such a
   penalty takes the bottom places in that repeat). `bio_at_bstar` itself is unchanged because rules R2/R3 read unanchored windows with it.
   Where this occurs in the sweep: a pair in which one penalty's best failure-free batch score lies below plain scVI's (b_common < b0, so
   b* < b0); such a pair is low-signal and not ranked, so the fix changes no rank in the three-seed reading.
7. **Matched strength for label transfer.** Spec 9: the grid point R4 matched (the failure-free point whose seed-mean batch score is nearest b*,
   ties to the smaller strength, extension points counted). Code: the same choice among every fitted strength (`prereg_rules._closest`),
   made in `analysis_sweep.sweep_targets`; the record stores it with each label-transfer entry (`lam`) and the distance of its batch score from b*.
8. **Label transfer scope.** Spec 9: on the embeddings at the matched strength, plain scVI and scvi_adv. Code: on every successful penalty
   embedding of the sweep (about 3,000), plain scVI and scvi_adv; the matched-strength table is read from those results and each penalty
   also gets its label-transfer curve over strength (`tables/label_transfer_curve.csv`). Why: the sweep framing reads every strength; the
   matched-strength table is unchanged by the extra readings. The other baselines (scANVI, sysVI, Harmony, Scanorama, PCA) are not
   label-transferred, as in the specification.
9. **Transfer strength across tasks (spec 10, 16.7).** Spec: the eligible strengths are grid points failure-free in every training task,
   and every task holds the same grid. Code: the tasks' strength sets differ (item 1), so a strength is eligible when it was fitted and is
   failure-free in every training task and was fitted in the held-out task (so its scores there can be reported); a tool with no scored
   configuration in a training task has no eligible setting and its transfer entry is "missing" while its own best setting is still
   reported. Why: without the last condition the chosen strength (for example immune's 0.03) can have no fit in the held-out task.
10. **Baseline completeness (spec 16.1).** Spec: the analysis refuses an input lacking any X13 configuration. Code: baselines enter as scored;
    `--require-baselines` restores the refusal; the coverage table (expected / scored / failed / missing per task and method) is in the
    inputs, the record and the paper numbers. Why: the baselines are being fitted and scored over several days (SI-117).
11. **Baseline seeds.** The baselines keep their run-list seeds (scvi_adv, scANVI, sysVI: five seeds 0-4; Harmony, Scanorama, PCA: one run), so
    their configuration scores are means over five runs where the sweep's are means over three. The best-of-k estimator takes each
    configuration's mean (spec 16.8) and is applied as written; the record gives each family's number of configurations and failed runs.
12. **Collapse rate (spec 6).** Spec: 30 runs (six grid points x five seeds). Code: every fitted strength x three seeds (30-36 runs); the
    Clopper-Pearson interval as written. Limitation 3 of the spec applies with the smaller count.
13. **Good-strength width (spec 8) and frontier area (spec 5).** Applied as written over every fitted strength; the width can now span more
    than the six-point grid allowed.
14. **Experiment labels.** The sweep rows are experiment A1, the scvi_adv baseline rows are labelled X1 in the run list and the other
    baselines X13; `analyse_pairs` takes the penalty experiment as a parameter and reads scvi_adv rows by arm. `check_seed_separation`
    still holds (A1 seeds 100-102, X1 seeds 0-4).
15. **Spec section 15 run commands and the X1 tag.** The X1 analysis was to run once after a freeze tag; the sweep analysis is exploratory
    (SI-116) and runs whenever baselines arrive. The record and `paper_numbers.json` carry the git commit and `git_dirty`.

Nothing in `scripts/prereg_rules.py` changed a rule: `pair_targets` is the R4 target formula factored out of `r4_stage` (same records,
`tests/prereg/test_rules_unchanged.py`), `refuse_unknown_scorer_sites` is the SI-115 port, and `_read_score_file` reads hash columns
as text (section 6).

## 4. Baseline coverage found on 2026-11-01

`D/jhpce_tier12/fillers_x1_x13/` holds 158 score files, of which 124 are baseline rows: scANVI (5), scvi_adv (5) and sysVI (30) on lung,
pancreas and sim2, and 4 scvi_adv rows on immune_hum_mou; the other 34 files are X1 plain-scVI rows and are not read. 76 further baseline
latents there have no score yet (atac_large: scANVI 5, scvi_adv 5, sysVI 30; immune_hum_mou: scANVI 5, scvi_adv 1, sysVI 30). No Harmony,
Scanorama or PCA row is scored, and nothing is scored on atac_small, immune or sim1. `D/sweep_baselines/` does not exist yet. So 124 of 448
rows are available; the three-task run below therefore has no baselines, and the full run should be made after the baselines land
(`--require-baselines` then checks that all 448 are in).

## 5. Runs made and timings (workstation, 12 CPUs; one run each)

All runs read the real score files; baselines from `D/jhpce_tier12/fillers_x1_x13` (none of them is on the three original tasks).
During every run another session's scoring jobs kept the load average at about 17 on the 12 CPUs, so the wall times below are
inflated and are upper bounds for an idle machine. Output directories: `D/sweep_analysis_dev/` (scratch, not committed).

| run | tasks | label transfer | `--jobs` | inputs | analysis | report | total |
|---|---|---|---|---|---|---|---|
| `smoke3b` | atac_small, immune, sim1 | debug subset 3 per task (9 embeddings) | 4 | 1.7 s | 30.6 s | 5.4 s | 38 s |
| `run3b` | atac_small, immune, sim1 | all 1,143 embeddings | 8 | 1.8 s | 803.0 s | 4.8 s | 13.5 min |
| `all8_debug` | all eight | debug subset 2 per task (16 embeddings, the largest tasks included) | 4 | 2.2 s | 154.4 s | 6.6 s | 2.7 min |

What the analysis time is made of: reading the score files (3,157 single-row files: about 30 s), the ranking analysis and four
sensitivity analyses (about 25 s for 16 pairs), the SHA-256 sums of the inputs for the record, and label transfer. Label transfer
dominates: 1,143 embeddings of the three small tasks took about 770 s of wall time on 8 processes under load, about 5.4 CPU-seconds per
embedding. The eight-task run reads about 3,064 embeddings (3,024 sweep fits plus 40 scvi_adv); immune_hum_mou (97,861 cells) and
atac_large (84,813 cells) take about 30-40 s each (756 embeddings), so the full run is estimated at 60-90 minutes of wall time on 8
processes on an idle machine, and about twice that at the load seen here. A first attempt of the three-task run was stopped at a
10-minute limit before it finished (`run3`); its directory was left as is. The eight-task full run was not made (the first author runs it).

## 6. Tests

New: `tests/analysis/test_sweep_analysis.py` (seven tests). Its module fixture builds a synthetic data root in the real layout (the
r4_pass2 score list with a superseded Sinkhorn copy under the same tags, the pilot5 runner layout, two baseline directories, the
committed pilot5 manifest and baseline run list) and runs `scripts/run_sweep_analysis.sh` end to end; planted cases: a tie, a
low-signal pair, a pair whose anchor lies above b* (b_common < b0), a penalty that reaches b* only with one seed (2 of 3 repeats), sweep
and baseline failures, differing strength sets across tasks, partial baseline coverage, a task whose label transfer is not evaluable.
The tests check the inputs and the coverage table, the `--require-baselines` refusal, the readings and ranks, label transfer (every
embedding, the matched strength, the process pool against one process), the tables, figures and `paper_numbers.json`, the debug subset
and task subset, the incomplete baseline seed group, simulation-only reporting, and the anchor rule (unit test). The SI-115 provenance tests cover invalid JHPCE CPU and scorer metadata (`tests/prereg`).

Results on 2026-11-01 (environment scvi-api, workstation under the load described above):

| command | result | exit code |
|---|---|---|
| `python -m pytest -q tests/analysis` | 59 passed in 144 s (53 existing + 6 new) | 0 |
| `python scripts/check_tests.py --base 2bc5b00 --src scripts -- tests/analysis tests/prereg` | 211 passed, 1 skipped in 350 s; `check_tests: OK` (every changed line and branch of `scripts/` run by a test; no redundant new test) | 0 |
| `python -m pytest -q tests/prereg/test_rules_unchanged.py tests/prereg/test_rules.py` | 58 passed (the `pair_targets` refactor leaves every rule record identical) | 0 |

Two faults found and fixed on the way: (1) pandas 2.3.3 crashes (segmentation fault, both parsers) when it infers a number from a hex
digest that reads like a float with a huge exponent (a synthetic `latent_sha256` beginning `33e05422...`); `prereg_rules._read_score_file`
now reads the tag, hash and provenance columns as text (the 3,191 real score files read identically before and after). (2) The
label-transfer pool uses spawned workers with one BLAS thread each (threadpoolctl, a scikit-learn dependency), not forked ones.

## 7. Open questions

1. Standard error from 2 of 3 repeats (item 4): keep, or report "not available" throughout?
2. Label transfer on every embedding (item 8) costs about 1.5 h of wall time on 8 processes for the eight tasks (estimate from the
   three-task timing and the dry-run per-embedding times); the matched-strength table alone would need about one tenth of it.
3. Transfer-strength eligibility restricted to strengths fitted in the held-out task (item 9).
4. Which baseline directories the final run reads (`fillers_x1_x13` plus `sweep_baselines`), and whether a baseline scored in both is
   possible (the pipeline refuses it; the directory of the earlier scoring would have to be dropped from the list).

## 8. Review fixes

- SI-115 accepts JHPCE scores only from the documented Zen 2 CPU with an allowed scorer commit and a clean scorer checkout, while retaining the site, seed, preload and provenance checks.
- Baseline settings with missing seeds remain in per-setting coverage with status `incomplete` and do not enter placement or strength selection until each expected seed is scored or failed.
- Exported tables retain the `label_aware` stratum, and scANVI has a separate label-aware label in the best-of-k figure.
- Low-signal pairs go to `low_signal_pairs.csv` and are omitted from `ranking_pairs.csv`.
- Simulation-only reports omit figures that summarize real tasks and still write the remaining tables and figures.
