# Inputs of Figures 1-6 (copied unchanged; the figure scripts read nothing else; Figure 1 panel b reads only paper_numbers.json and curves.csv)

| File | Source |
|---|---|
| paper_numbers.json | Claude Science artifact 52c258c0-4e2f-4eae-b277-36d219820e7d, version v1b0c059e (paper_numbers.json of the sweep analysis, 2,999 keys); analysis commit 8595a169ed853db61eaf54501fb42f6735dea911 on sweep-paper, `_meta.git_dirty` false |
| curves.csv, baseline_points.csv, best_of_k_real_mean.csv, transfer_strength.csv, single_methods.csv | `tables/` of the workstation directory /home/kendall/experiment_data/wasserstein-critic-deconfounding/sweep_analysis_final_20261105 (same run, 2026-11-05); SHA-256 of the originals checked against these copies on 2026-11-05 |

Baselines are 447 of 448 rows: Scanorama on atac_large at strength 80 could not be fitted, so Scanorama has 6 of 7 settings there (Figure 6a draws Scanorama k = 6 as an open marker, 5 of 6 tasks).
SHA-256 of every file: MANIFEST.sha256 (`sha256sum -c MANIFEST.sha256`).
