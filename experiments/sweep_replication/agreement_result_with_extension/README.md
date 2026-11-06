# Replication agreement result with the rule's extension step (final; branch sweep-replication)

Rule: ../AGREEMENT.md line 9 (tag replication-prediction-v1): where the new-seed means do not bracket b*, add the next sweep-grid strength once (five seeds, 0-4). User decision 2026-11-05: "Run all 40, as the rule says". Report: AGREEMENT_RESULT.md. The result before the extension step stays in ../agreement_result/.

Data provenance
- Extension run list: manifests/sweep_replication_ext1.tsv (40 rows, sha256 13013d7d18215784e7d8da39a316884a26b6961f35e1eee12a6d1b0c0ee16494; built by ../extension_rows.py, committed in 648f7e8 before any extension fit). 35 rows are committed X1 rows (sweep manifest, x1_sinkhorn_redo), 5 rows (pancreas d1 Sinkhorn, strength 10) were built with prereg_rules.extension_rows as replication_design.py did.
- Combined 895-row manifest for the test: manifests/sweep_replication_x1_ext1.tsv (sha256 b00f7529ef18ad8e9dcd833d137a28c3eaf337f2f3bb51b990f8864a77c812ef) = replication list (d8d35481...) + extension list.
- Fits, same code (10a1ca5, tag prereg-tier12-v3), same machine type per task as the replication: workstation RTX 3080 15 fits (atac_small, sim1; run_stage.py --no-score, 5 lanes, ../run_ext_local.sh, tags ../ext1_local.tags); JHPCE L40S 25 fits (immune_hum_mou, pancreas, sim2; Slurm 36899485, 3 h 03 min, ../jhpce_stage/run_shard_ext.sh, tags ../jhpce_stage/replication_sE.tags).
- Scoring (scorer 10a1ca5): workstation ../run_ext_local_score.sh (15 scores); JHPCE fixed LISI seed, helper 36901327 (lists) and arrays 36901360 (immune_hum_mou 5), 36901365 (pancreas 15), 36901380 (sim2 5), files ../jhpce_stage/scoring_ext/. The 25 JHPCE score files were copied to D/sweep_replication/jhpce_scores/scores with SHA-256 checked (list D/sweep_replication/jhpce_scores/ext_scores_sha256.txt; D = /home/kendall/experiment_data/wasserstein-critic-deconfounding). All 895 score files pass score_validity.py (finite, in range, one file per tag).
- Efficiency (Slurm accounting read back after the run): fit job MaxRSS 38.4 GiB of 48 GiB, CPU use 69% of allocated; scoring MaxRSS immune_hum_mou 23.5-28.5 of 48 GiB (49-59%, below the 70% target), pancreas 12.5-14.0 of 20 GiB (62-70%), sim2 30.0-37.4 of 36 GiB.
- Budget: experiments/sweep_replication/BUDGET.json (ack recorded 2026-11-05 in BUDGET.jobs.jsonl; every submit passed budget_check.py check with decision allow).

Re-run (from the repo root, on the workstation):
    PY=~/wcd_agreement_env/bin/python; D=/home/kendall/experiment_data/wasserstein-critic-deconfounding
    OUT=experiments/sweep_replication/agreement_result_with_extension
    $PY experiments/sweep_replication/score_validity.py --rules-scripts scripts --manifest manifests/sweep_replication_x1_ext1.tsv \
        --scores $D/tier12_v3/sweep_replication/scores $D/sweep_replication/jhpce_scores/scores --out $OUT/score_validity.csv
    $PY experiments/sweep_replication/agreement_test.py --rules-scripts scripts --manifest manifests/sweep_replication_x1_ext1.tsv \
        --scores $D/tier12_v3/sweep_replication/scores $D/sweep_replication/jhpce_scores/scores \
        --failures $D/tier12_v3/sweep_replication/failures.csv $D/sweep_replication/jhpce_scores/failures.csv --out $OUT
    (cd experiments/sweep_replication && $PY make_report.py)      # writes AGREEMENT_RESULT.md
Environment: ~/wcd_agreement_env (environment_agreement.txt in ../agreement_result). The test is deterministic (exact permutation over all 2^13 sign patterns; no random numbers; no seeds needed).
Tests: tests/analysis/test_replication_extension.py (new), test_replication_agreement.py, test_score_validity.py; scripts/check_tests.py --base 648f7e8~1 --src experiments/sweep_replication exits 0 with no WARN.
