# Replication agreement result (branch sweep-replication)

Agreement rule: ../AGREEMENT.md, prediction ../PREDICTION.json (tag replication-prediction-v1). Result report: AGREEMENT_RESULT.md.

Data provenance: run list manifests/sweep_replication_x1.tsv (sha256 d8d35481cd06b28c58045216e86bf7fee7c470cb69636fe80b291d0da4f225f1).
Scores (855 CSVs, scorer 10a1ca5): workstation 270 in D/tier12_v3/sweep_replication/scores; JHPCE 585 copied from
/fastscratch/myscratch/kreid/wcd/sweep_replication_scores/scores to D/sweep_replication/jhpce_scores/scores (SHA-256 of all 585 checked: jhpce_scores_sha256.txt).
D = /home/kendall/experiment_data/wasserstein-critic-deconfounding. Last 40 scores: Slurm 36889227 (list ../jhpce_stage/scoring_w3).
Environment: environment_agreement.txt (conda -p ~/wcd_agreement_env, python 3.11). Seeds: the test is deterministic (exact permutation over all 2^13 sign patterns; no random numbers).

Re-run (from the repo root, on the workstation):
    PY=~/wcd_agreement_env/bin/python; D=/home/kendall/experiment_data/wasserstein-critic-deconfounding
    $PY experiments/sweep_replication/score_validity.py --rules-scripts scripts --manifest manifests/sweep_replication_x1.tsv \
        --scores $D/tier12_v3/sweep_replication/scores $D/sweep_replication/jhpce_scores/scores --out experiments/sweep_replication/agreement_result/score_validity.csv
    $PY experiments/sweep_replication/agreement_test.py --rules-scripts scripts --manifest manifests/sweep_replication_x1.tsv \
        --scores $D/tier12_v3/sweep_replication/scores $D/sweep_replication/jhpce_scores/scores \
        --failures $D/tier12_v3/sweep_replication/failures.csv $D/sweep_replication/jhpce_scores/failures.csv --out experiments/sweep_replication/agreement_result
    $PY scripts/check_tests.py --base HEAD --src experiments/sweep_replication -- tests/analysis/test_replication_agreement.py tests/analysis/test_score_validity.py
