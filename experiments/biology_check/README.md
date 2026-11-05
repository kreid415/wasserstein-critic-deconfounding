# Biology check (reviewer comment R2.6): direct biology readouts on the saved sweep embeddings

Rules: `docs/BIOLOGY_CHECK.md` (committed 639c302 before any value was computed; written after sweep scores were read).
Code: `scripts/biology_check.py` (tests `tests/analysis/test_biology_check.py`), summary page `experiments/biology_check/make_summary.py`.
Exploratory and descriptive: no new fits, no hypothesis test.

## Re-run (workstation, from the repo root; D = /home/kendall/experiment_data/wasserstein-critic-deconfounding)

```
PY=/home/kendall/.claude-science/conda/envs/scvi-api/bin/python      # environment: environment_scvi-api.txt (conda list --export)
OUT=$D/biology_check_20261105
$PY scripts/sweep_inputs.py --out $OUT/inputs --baseline-dirs $D/jhpce_tier12/fillers_x1_x13 $D/sweep_baselines
$PY scripts/biology_check.py --inputs $OUT/inputs --prepped-dir $D/prepped_scib --out-dir $OUT/run --jobs 6
$PY experiments/biology_check/make_summary.py $OUT/run $OUT/run/summary.md
```

Inputs (read only): the sweep run lists, score files and saved latents listed by `sweep_inputs.py` (every input's SHA-256 is in
`docs/biology_check/biology_check.json`, `inputs`; each latent is checked against its score row's `latent_sha256`), and the prepped
scIB files `$D/prepped_scib/<task>__scib.h5ad` (SHA-256 in the record, `prepped`). The inputs listing is byte-identical to the one of
the sweep analysis run `$D/sweep_analysis_final_20261105/inputs` except `inputs.json` (which records its own output path). No random
numbers are drawn; the output does not depend on `--jobs`.

Run of 2026-11-05: commit c4a793f, 783 fits (8 tasks; atac_large has 97 because one Scanorama baseline row is missing from the baseline
run), 6 worker processes, about 1 h 40 min of wall time on a shared workstation.

Outputs copied to `docs/biology_check/`: the record, `MANIFEST.sha256`, `summary.md` and `tables/`.
