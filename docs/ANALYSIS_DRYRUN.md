# Analysis dry run, 2026-10-28

Branch `analysis-dryrun` (off `ranking-analysis`), run on the workstation (cs-local). No main-benchmark score, ledger or runner log was read: the rows called "X1" below are either synthetic or pilot rows relabelled X1 (notebook NB-20261028-02).

## 1. What was shown

One command takes a score table to the analysis record (JSON), three tables and the ranking figure, and runs the outline's mock-figure script as a check that it still runs:

```
# from the repo root, python = an environment with numpy, pandas, scipy, scikit-learn, matplotlib
PYTHON=/home/kendall/.claude-science/conda/envs/scvi-api/bin/python bash scripts/run_all_analysis.sh synthetic <new-out-dir>
PYTHON=... WCD_DATA_ROOT=/home/kendall/experiment_data/wasserstein-critic-deconfounding bash scripts/run_all_analysis.sh pilot <new-out-dir>
```

The script refuses an existing output directory, stops at the first failed stage and prints that stage's log, and writes `timings.tsv` and `MANIFEST.sha256` (SHA-256 of the record and the tables). It must not be pointed at main-benchmark scores before the analysis tag exists (SI-78: running the analysis on a score counts as examining it).

### Timings (seconds, one run each, from `timings.tsv`)

| stage | synthetic table | pilot (3 tasks) |
|---|---|---|
| build inputs (generator / pilot mapping) | 5.5 | 5.0 |
| `scripts/analysis_x1.py` (record + 4 sensitivity analyses) | 25.9 | 365.1 |
| tables and ranking figure (`scripts/dryrun/figures_x1.py`) | 1.0 | 0.8 |
| `build_metric_curves.py` (older figure script) | not applicable | 6.0 |
| mock figures (`make_mock_figures.py`) | 3.5 | 3.6 |
| total | 35.9 | 380.5 |

An earlier pilot run took 318 s and one took 391 s while other jobs shared the workstation; read the pilot analysis time as 320-390 s. Most of it is label transfer (spec section 9, 15-nearest-neighbour classification on every used embedding; the synthetic run, whose embeddings are tiny, needs 26 s for everything else): a separate timing probe took 0.89 s per embedding for atac_small (11,270 cells, 3 batches), 1.84 s for sim1 (12,097 cells, 6 batches) and 6.67 s for immune (33,506 cells, 10 batches). Extrapolating those three points (log-log slope 1.6) to the main benchmark's 75 embeddings per task gives about 1.8 hours in total (immune_hum_mou 0.78 h, atac_large 0.62 h, the other six tasks under 0.15 h each). This is provisional: three points, one machine. Each embedding is independent, so the work can be run in parallel on the 12 workstation CPUs; that was not done here (see open questions).

## 2. Pilot: how pilot rows were mapped to the main-benchmark format

`scripts/dryrun/build_pilot_inputs.py` (reads `tier12_v3/rules/r4_pass2` and `r1_pass2`, which are not blinded):

| main-benchmark input | built from |
|---|---|
| manifest | the pilot's A1 rows of atac_small, immune and sim1, experiment relabelled X1, seeds 100-102 relabelled 0-2, tags kept. Kept: plain scVI and each penalty at the six strengths of `r1_pass2` `families.<family>|<decoder>.x1_grid`; then, as `freeze_matched_lambda.py` does, the extension rows R4 asks for, taken from the pilot fits (687 rows in 2 rounds) |
| scores | the pilot's own score files, linked (not copied or changed) |
| failures | `r4_pass2/failures_merged.csv` (header only: no pilot fit failed) |
| R1 record | `r1_pass2/r1_record.json` |
| R4 stage-x1 record | `prereg_rules.r4_stage` on the mapped rows, written with `write_record` |
| latents | one pilot latent per tag, linked from the directory next to its score file |
| `scvi_adv` and the X13 baselines (scANVI, sysVI, Harmony, Scanorama, PCA) | **stand-ins, not pilot data**: 168 synthetic rows from the generator for the same three tasks, in directories whose names start `standin_`, carrying the pilot scorer's provenance values and a `synthetic_standin` column |

Check against numbers recorded earlier in the project notes (from `r4_a1_record.json`): the recomputed pilot targets agree for atac_small with the unconditioned decoder (b0 0.402, b_common 0.642, b* 0.522) and for immune with the conditioned decoder (b0 0.638, b_common 0.657, low-signal). For sim1 with the conditioned decoder b_common is 0.818 here against 0.819 in the notes, which fits failure 1 below (the five missing extension strengths). Values read from `/tmp/final_pilot/analysis_x1.json`.

All pilot-based numbers are provisional: three seeds instead of five, stand-in baselines, and five extension strengths missing (failure 1 below). The pilot ranking figure shows two real tasks (atac_small, immune) because sim1 is a simulation.

## 3. Failures, in the order they appeared, with causes

| # | where | what happened | cause | what was done |
|---|---|---|---|---|
| 1 | pilot, mapping | `r4_stage` on the mapped rows asks for lambda 0.03 for five penalties on sim1 with the conditioned decoder (barycenter, discriminator, pooled, reference, sinkhorn) | the main grid starts at 0.1 there; the pilot never fitted below 0.1 | R4 frozen by hand, the five requests are listed in the R4 record (`unmet_extensions`). sim1 with the conditioned decoder is a low-signal pair, so no rank changes, but its b_common may |
| 2 | pilot, analysis | refused: scores mix scorer provenance | the synthetic stand-ins carry a different scorer commit and version string than the pilot rows; the refusal is correct | stand-ins carry the pilot's provenance values and a flag column |
| 3 | pilot, analysis (`--strict`) | refused at the first check of `analyse()`: all eight tasks, both decoders, `scvi_adv` and X13 required (1 s after the inputs were read) | the script fixes 8 tasks and plain-scVI seeds 0-4; the pilot has 3 tasks, 3 seeds, no `scvi_adv` and no X13 scores | `scripts/dryrun/run_analysis.py` sets the task list, real-task list and seeds in memory (no file changed); `--strict` keeps the unchanged behaviour |
| 4 | pilot, label transfer | refused: 2 latent files for the tag `A1_atac_small_sinkhorn_l3_c0_s100_9ba7945a` | the superseded Sinkhorn fits (`tier12_v2/A1`) and the redone ones (`tier12_v3/A1_sinkhorn`) share tags; the analysis demands exactly one file per tag | the mapping links one latent per tag, from the directory of the score file the score row came from |
| 5 | pilot, figures | no existing script reads the analysis record. `make_mock_figures.py` draws fixed-seed synthetic panels (no input) and stops with `NameError: panel_letter` unless the figure-style helpers are in scope | helpers live in the Claude Science skill, not in the repo | `figures_x1.py` written (ranking tables and ranking figure only); `scripts/dryrun/figure_helpers.py` supplies minimal stand-ins; `run_mock_figures.py` runs the script from a temporary directory so no tracked figure is overwritten |
| 6 | pilot, figures | `build_metric_curves.py` needs a scored table in an older format; `compare_methods_stratified.py` needs the old benchmark's wave CSVs, an E3 CSV and Seurat output | different generation of the project | the first was fed a table made from the pilot (decoder mapped cond 0 -> `lin`, cond 1 -> `nl` for this dry run only; ran, 6.0 s); the second cannot be fed pilot data (no equivalent inputs) and was not run beyond its argument check |
| 7 | mock figures | the script's own overlap check prints overlaps for figures 1-3 and a label outside figure 6 with my stand-in helpers | probably the stand-in font settings differ from the real figure-style helpers; not checked against them | not fixed; the mock figures are the writer's, not touched |
| 8 | tooling | a 60 s limit on single commands cut the first pilot run | the label-transfer stage takes about 6 minutes | all runs went through the job interface |

## 4. Synthetic main-shaped table

`scripts/dryrun/make_synthetic_x1.py --out <dir>`, seed 20261028, every value a function of the seed and the row's tag (independent of row order; two runs agree exactly apart from latent file paths inside the record). Shape: 8 tasks (6 real, sim1, sim2) x 2 decoders x plain scVI + 6 penalties x the six strengths of `r1_pass2` (`GRIDS` in the script) x seeds 0-4, `scvi_adv`, and the X13 rows of the committed manifest; 3,573 rows in 3 rounds (the base table plus two R4 extension rounds, each with its own manifest and score directory, one score file per fit as the scorer writes them), 16 failed fits, R4 frozen. Planted cases (named in `SCENARIO`):

- ties: pooled = reference on pancreas (conditioned), barycenter = pooled on sim2 (unconditioned): identical scores, rank 2.5 each;
- curves that never reach b* on the six grid strengths: sinkhorn on lung (unconditioned), mmd and sinkhorn on atac_large (conditioned): status `unreached_low`, rank 6 and 5.5 each; b* counts the extension rows, as in the real flow;
- one low-signal pair: immune with the conditioned decoder (barycenter gain 0.010);
- failed fits: all five seeds of one strength (diverged), three strengths with some seeds, one strength of one penalty (mmd, immune_hum_mou, unconditioned) with all five seeds `nonfinite_latent`, counted in the collapse rate;
- a curve that rises through b* and falls back (2 crossings): pooled on lung (conditioned); three penalties with a 0.15 collapse of biology at high strength.

Result: the analysis ran at the first attempt on all planted cases (25.9 s). Checks on the record (all passed): the six ranks of every ranked pair sum to 21; a reading exists exactly when the status is `bracketed`; the standard error is absent exactly when fewer than 4 leave-one-seed-out readings exist or the penalty has no reading; the rank range lies within 1-6 with its low end not above its high end; mean ranks equal the average of the per-task ranks (5 real tasks with the conditioned decoder, 6 with the unconditioned); the unreached penalties sit at the bottom places; area is non-negative, collapse intervals contain the rate; label-transfer macro F1 lies in [0, 1]. Test: `tests/analysis/test_dryrun_synthetic.py` (one end-to-end test; expected values follow from the planted cases). The whole `tests/analysis` folder: 53 passed (52 existing + 1 new). `scripts/check_tests.py --base ranking-analysis --src scripts -- tests/analysis`: OK. Fail-loud lint (`fl_lint`, medium) on the new scripts: no findings.

## 5. Analysis bugs found and fixed

None in `scripts/analysis_x1.py` or `scripts/prereg_rules.py`. I found no disagreement with the specification that changes a result in the planted cases, except the open question 1 below, which I did not change. All fixes were in the new dry-run code (the generator's planted "never reaches b*" curve first did not start below the plain-scVI batch score, so R4 matched an interior point; corrected by giving that curve a batch score 0.03 below plain scVI until its top grid strength). The `analysis_x1.py` guards were not stressed beyond the existing `test_refusals`.

## 6. Open questions (not decided here; no analysis rule was changed)

1. Spec sections 4(i) and 16.2 say a leave-one-seed-out repeat whose plain-scVI anchor is at or above b* gives no reading. `prereg_rules.bio_at_bstar` (used unchanged) returns a reading when a curve starts above b* and falls through it (checked: curve (B, C) = (0.6, 0.7), (0.5, 0.6) with b* = 0.55 gives biology 0.65, status `bracketed`). Which is intended? It matters only in rare repeats, and only for the standard error and the rank range.
2. Which R1 record goes to the main analysis? The committed X1 manifest has the same strengths as `docs/prereg/r1_record.json` for every penalty, but `tier12_v3/rules/r1_pass2/r1_record.json` differs for Sinkhorn (committed manifest: 0.3, 3, 10, 100, 300, 3000 unconditioned and 0.1, 0.3, 1, 3, 10, 30 conditioned; r1_pass2: 0.3, 1, 3, 30, 100, 300 and 0.3, 3, 30, 100, 1000, 10000). The analysis would refuse the r1_pass2 grids on the committed manifest. The synthetic table uses the r1_pass2 grids, as the task asked.
3. Only the ranking figure and three tables are produced from the record; Figures 3-6 and Tables 1-4 of the outline are still mock-ups. Write record-to-figure code for them before the analysis tag, or after?
4. Label transfer is about 95% of the analysis time (estimated 1.8 h on the main benchmark). Run the embeddings in parallel (results must then be shown to be identical to the serial run on the pilot before use)? Not done.
5. `analysis_x1.py` cannot run on pilot-sized inputs without the in-memory settings of `run_analysis.py`. Keep it strict for the main benchmark (my recommendation) or add a documented pilot mode?

## 7. Declared limits and deviations

- Pilot case: scvi_adv and X13 rows are synthetic stand-ins; three seeds; five R4 extension strengths missing; seeds relabelled; decoder mapping for `build_metric_curves.py` is for the dry run only. Pilot numbers are provisional.
- `compare_methods_stratified.py` was not run on pilot data (no equivalent inputs).
- Figure-style helpers are stand-ins, not the skill's code; the mock figures were not compared with the committed PNGs.
- Timings: one run per case on a shared workstation; the pilot analysis time varied 318-391 s between runs.
- `ranking-analysis` on the workstation holds one commit not on GitHub (61b884a, an automatic checkpoint that contains the rewritten `analysis_x1.py` and tests); pushing `analysis-dryrun` publishes it too.
