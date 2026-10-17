# Follow-up build, 2026-10-17 (SI-93..SI-98): re-run commands

Branch `followup-build` (cut from c241af6). Environment `scvi-api` (`/home/kendall/.claude-science/conda/envs/scvi-api`),
run from the repo root, `OMP_NUM_THREADS=1`. Nothing here reads an X1 or X13 score.

```
# 1. v4 manifest (v3 is rebuilt byte for byte by the same command without the two v4 flags)
python scripts/build_paper_manifest.py --backbone stock --design pilot --uncond-seeds 5 --bary-iter 10 \
    --x12-runs 16 --followup-design v4 --out manifests/paper_manifest_stock_pilot_u5_b10_v4.tsv
# 2. row diff v3 -> v4 and cost (scripts/cost_model.py); fails if any row outside X3/X6/X7/X8/X12 differs
python experiments/followup_build_20261017/diff_report.py            # writes results/manifest_diff_v3_v4.csv
# 3. static check: which X1/X13 rows and code paths the changed code can reach (no fit)
python experiments/followup_build_20261017/static_x1_paths.py        # writes results/x1_static_paths.csv
# 4. tests (touched files only)
WCD_SRC=src PYTHONPATH=. PREPPED_DIR=<prepped_scib> python scripts/check_tests.py --base c241af6 --src scripts -- \
    tests/scvi/test_followup_v4.py tests/scvi/test_adversarial_plan_v2.py tests/scvi/test_runner_and_manifest_missing_arms.py
```

## Not run yet: X1 fit and score identity against the tag (needs the CPU free of the Sinkhorn redo)
```
C=/home/kendall/.claude-science/conda/envs; D=/home/kendall/experiment_data/wasserstein-critic-deconfounding
# worktrees: OLD = D/code/wcd_prereg-tier12-v3 (tag checkout, in use by the stage-1 run: make a second checkout), NEW = candidate commit
python select_x1_rows.py results/x1_identity_rows.tsv          # 53 rows: every X1 arm x decoder x adv_input x zstd + scanvi, sysvi
PREPPED_DIR=$D/prepped_scib FIT_PY=$C/scvi-api/bin/python bash fit_identity.sh OLD NEW OUT results/x1_identity_rows.tsv 4
```
`fit_identity.sh` exits 1 unless every row gives equal latents with exit code 0 at both commits. Scoring code is untouched
(`static_x1_paths.py` lists every changed function; none is in `score_scib_native.py` or `src/`), so `score_identity.sh` is a
copy kept for completeness and is not needed unless a scoring file changes.
