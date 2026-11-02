# Scoring of the JHPCE-fitted baseline latents (atac_large, immune_hum_mou)

Run plan for rescope SI-117 (docs/RESCOPE_20261102.md). 76 latents fitted by the JHPCE filler jobs (Slurm 36155707/36155708,
commit c88cce3; local mirror D/jhpce_tier12/fillers_x1_x13) have no score: sysVI 30 + scANVI 5 + scvi_adv 5 on atac_large,
sysVI 30 + scANVI 5 + scvi_adv 1 on immune_hum_mou. (The plan said 78; the exact count from the run list is 76.)

- Scorer: clean checkout of 10a1ca5 (JHPCE /fastscratch/myscratch/kreid/wcd/scorer_10a1ca5), wrapper
  cluster/jhpce/score_array_fixed_seed.sh from repo_9d23c20 (LISI seed fixed, SI-115), Zen 2 nodes.
- List: list.tsv here (tag, JHPCE latent path, prepped input, metadata with latent SHA-256 from the local mirror).
- Quick test run: Slurm 36823510 (2 tasks: one atac_large sysVI, one immune_hum_mou scANVI; 1 CPU, 48G, 4 h).
- Budget: BUDGET.json (checked by the cluster-rules budget_check.py before every submission).
- Output: JHPCE /fastscratch/myscratch/kreid/wcd/sweep_baselines_scoring/out/{scores,logs}; pulled to
  D/sweep_baselines/jhpce_scores/ on the workstation.
