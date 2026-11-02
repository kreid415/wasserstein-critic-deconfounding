# Fresh-seed replication of the eight-task sweep (rescope 2026-11-02)

Design: replication_design.py (docstring), outputs in design_20261102/, run list ../../manifests/sweep_replication_x1.tsv (855 rows,
sha256 d8d35481cd06b28c58045216e86bf7fee7c470cb69636fe80b291d0da4f225f1). Prediction and agreement rule: PREDICTION.json, AGREEMENT.md
(tag replication-prediction-v1, commit 3a90d0f, fixed before any replication score). Budget: BUDGET.json (JHPCE, approved 2026-11-02).

Where each task is fitted and scored (same kind of machine as in the sweep):
| tasks | fits | fitted on | code | scored on |
|---|---|---|---|---|
| atac_small, immune, sim1 | 270 | workstation RTX 3080 (driver 580.178.04), run_stage.py --no-score, 4 lanes | tag prereg-tier12-v3 (10a1ca5) clean checkout | workstation, scorer 10a1ca5 |
| lung, sim2 | 200 | JHPCE L40S, shard 3 (Slurm 36823804) | 10a1ca5 from wcd.bundle via cluster/jhpce/prod_fit_job.sh | JHPCE, fixed LISI seed (SI-115) |
| pancreas | 130 | JHPCE L40S, shard 4b (Slurm 36825656) | same | JHPCE, fixed LISI seed |
| atac_large | 130 | JHPCE L40S, shard 1 (Slurm 36825657, after quick test 36823803) | same | JHPCE, fixed LISI seed |
| immune_hum_mou | 125 | JHPCE L40S, shard 2b (Slurm 36825658, after quick test) | same | JHPCE, fixed LISI seed |
The sweep fitted atac_small, immune and sim1 on the same workstation GPU (the driver was updated on 2026-11-02 from 580.173 to
580.178), and the other five tasks on JHPCE L40S. JHPCE stage files: /fastscratch/myscratch/kreid/wcd/replication_20261102
(run_shard.sh, validate_shard.py, shard tag files with shard_tags.sha256); local tag files in D/tier12_v3/sweep_replication/tags.
