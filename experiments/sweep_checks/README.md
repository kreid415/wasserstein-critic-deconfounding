# Independent check of the sweep analysis' placement numbers

`check_sweep_placement.py <analysis-out-dir> <check-out-dir>` recomputes, without importing any pipeline module, from the raw score CSVs
(`inputs/scores_list.txt`, with the run-list rows of `inputs/manifest.tsv`): per-task and real-task-mean expected best of k for every method
family (equal tries), single-setting S (plain scVI, scANVI, scvi_adv, PCA), b_common, hypervolume and share of non-dominated points of every
penalty in every task x decoder cell (all 8 tasks x 2 decoders x 6 penalties). It compares each value with `paper_numbers.json` (tolerance 1e-9;
per-task E_k, which paper_numbers.json does not hold, with `tables/best_of_k.csv`) and exits 1 on any mismatch.

Run (workstation, scvi-api environment, see environment_scvi-api.yml):
`python experiments/sweep_checks/check_sweep_placement.py $D/sweep_analysis_final_20261105 experiments/sweep_checks/out`

Output in `out/`: `check_results.csv` (every comparison), `check_summary.json`, `placement_equal_tries.csv` (penalty vs tool at equal k, per task and
decoder), `placement_summary.csv` (number of real tasks on which the penalty's E_k exceeds the tool's, at k = 1 and k = 6).
Tools with one setting (plain scVI, scANVI, scvi_adv, PCA) are compared at their S for every k. Scanorama on atac_large has 5 settings (l80 is not scored), so its k = 6 counts are over 5 tasks.
Mutation test (2026-11-05): four planted changes of paper_numbers.json (a hypervolume, a non-dominated share, a best-of-k mean, a null made non-null) were all reported as mismatches.
