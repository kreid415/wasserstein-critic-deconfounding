# Inputs of Figures 1-5 and Supplementary Figures S7-S8 (Pass 24 numbering; copied unchanged; the figure scripts read nothing else; Figure 1 panel b reads only paper_numbers.json and curves.csv)

| File | Source |
|---|---|
| paper_numbers.json | Claude Science artifact 52c258c0-4e2f-4eae-b277-36d219820e7d, version v1b0c059e (paper_numbers.json of the sweep analysis, 2,999 keys); analysis commit 8595a169ed853db61eaf54501fb42f6735dea911 on sweep-paper, `_meta.git_dirty` false |
| curves.csv, baseline_points.csv, best_of_k_real_mean.csv, transfer_strength.csv, single_methods.csv | `tables/` of the workstation directory /home/kendall/experiment_data/wasserstein-critic-deconfounding/sweep_analysis_final_20261105 (same run, 2026-11-05); SHA-256 of the originals checked against these copies on 2026-11-05 |

Baselines are 447 of 448 rows: Scanorama on atac_large at strength 80 could not be fitted, so Scanorama has 6 of 7 settings there (in `best_of_k_real_mean.csv` Scanorama at k = 6 is the mean of 5 of 6 tasks).
Since Pass 23 no figure or table reads `best_of_k.csv`, `best_of_k_real_mean.csv` or `placement_summary.csv`: the best-of-k analysis (the expected best selection score among k settings, k = 1 to 6) was removed from the paper (`paper/pass23_changelog.md`, Removed claims), and with it the former Figure 6a, Supplementary Figure S6 (best of k per task) and panel b of Supplementary Table S7. The files stay here as the analysis record, with their checksums in MANIFEST.sha256.
SHA-256 of every file: MANIFEST.sha256 (`sha256sum -c MANIFEST.sha256`).


## Added for Supplementary Tables S6-S7 and Figures S1-S6 (branch supp-items; `figure_scripts/make_supp_figures.sh` rebuilds everything)

Copied unchanged from `tables/` of the sweep analysis directory D/sweep_analysis_final_20261105 (same run as above; `figure_scripts/copy_supp_inputs.sh`; SHA-256 in MANIFEST.sha256):
`ranking_pairs.csv`, `low_signal_pairs.csv` (Table S6, Figures S1, S2, S5), `best_of_k.csv` (read by no figure or table since Pass 23), `label_transfer_nearest_any_crossing.csv` (the sweep analysis's `label_transfer.csv`, kept for the record; since 2026-11-09 `label_transfer.csv` is rebuilt from it, see below), `label_transfer_curve.csv` (Figure S3), `secondary.csv` (Supplementary Table S8; added on branch supp-analysis), and
`placement_summary.csv`, the output of `experiments/sweep_checks/check_sweep_placement.py` (branch sweep-paper, commit 3280bd4; the output file itself is not tracked in that repository), copied from `experiments/sweep_checks/out/` of the worktree wt_sweep_paper; its SHA-256 equals that of Claude Science artifact version 13ab3936 (read by no table since Pass 23: it fed panel b of Supplementary Table S7, removed with the best-of-k analysis).

Derived (scripts record the inputs; SHA-256 in MANIFEST.sha256):
| File | Made by | From |
|---|---|---|
| `sensitivity_ranks.csv` | `figure_scripts/prep_supp_figure_data.py --only sensitivity` | `analysis_sweep.json` (sha256 197104bb7ebb...) of the sweep analysis, `sensitivity` block |
| `metric_scores.csv` | `prep_supp_figure_data.py --only metrics` | the 3,480 score files listed in `inputs/scores_list.txt` (sha256 f0ad343ab7df...), joined to `inputs/manifest.tsv`; columns are the individual scIB metrics and the sha256 of the latent each row was scored from |
| `label_transfer_types.csv` | `figure_scripts/prep_label_transfer_types.py` (scvi-api environment) | the saved latents of the 120 label-transfer rows (checked against the latent sha256 of each score row); the macro F1 recomputed from the per-type values equals `label_transfer.csv` for all 120 rows (1e-9). The committed copy was made from `label_transfer_nearest_any_crossing.csv`; it must be re-made from the rebuilt `label_transfer.csv` (below) |
| `scaling_check_ranks.csv` | `figure_scripts/figS5.py` via `scaling_checks.py` | `metric_scores.csv`; the unscaled reading reproduces the 78 readings and ranks of `ranking_pairs.csv` before any variant is run |

## Label transfer at the first crossing of b* (2026-11-09; changed after the first results had been read)

The paper reads each penalty where its batch-score curve first reaches the matched batch removal b*. Label transfer had used the failure-free strength whose seed-mean batch score is nearest b* at any crossing, which for some penalties is a collapsed strength far past the first crossing. Rule, in `paper/tools/first_crossing_strength.py`: from plain scVI (strength 0) up the penalty's strengths, take the first point with batch score at least b*; of that point and the one before it choose the one nearer b* (a tie goes to the smaller strength), or the upper one when the one before is plain scVI. 31 of the 96 penalty rows changed strength.

| File | Script | From |
|---|---|---|
| first_crossing_strength.csv (not in MANIFEST.sha256) | `paper/tools/first_crossing_strength.py` | curves.csv, ranking_pairs.csv, low_signal_pairs.csv |
| label_transfer.csv (in MANIFEST.sha256; same columns and row order as before) | `paper/tools/label_transfer_first_crossing.py` | penalty rows: label_transfer_curve.csv at the strengths of first_crossing_strength.csv, with n_runs and not_evaluable read from analysis_sweep.json of the sweep analysis; plain scVI and scvi_adv rows: label_transfer_nearest_any_crossing.csv |

## Derived in Pass 23 (written by scripts in `paper/tools/`, not copies; not in MANIFEST.sha256)

| File | Script | Used by |
|---|---|---|
| pareto_frontier.csv, pareto_frontier_median.csv, pareto_frontier.json | pareto_frontier.py (from curves.csv, paper_numbers.json, secondary.csv; checks that the uncapped hypervolume reproduces secondary.csv) | Figure 3, Supplementary Figure S8, Supplementary Table S8 (HV, HV_all), Res-3b, Supplementary Note 7 |
| collapse_common_range.csv, collapse_common_range.json | collapse_common_range.py (from metric_scores.csv, curves.csv, paper_numbers.json, secondary.csv, ranking_pairs.csv; checks that the per-fit rebuild reproduces the collapse counts of secondary.csv) | Figure 4, Supplementary Table S8 (Coll. common), Res-4, abstract, Supplementary Note 7 |
| res2_seed_noise.json | res2_seed_noise.py | Res-2 |
| res5_label_transfer.json | res5_label_transfer.py (from label_transfer.csv, ranking_pairs.csv) | Res-5 |
| replication_mean_ranks.json | replication_mean_ranks.py | Supplementary Note 13, Res-2b |
| res3_curve_numbers.json | res3_curve_numbers.py | Res-3 |
