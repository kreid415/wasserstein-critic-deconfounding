# Baseline integration + geometry pipeline

Adds Seurat/scVI baselines and geometry metrics (trustworthiness, continuity)
to the benchmark, and the dataset-stratified method comparison used in the paper.

## Baselines (two-stage: fit in a dedicated env, score through the harness suite)

Both follow the same pattern so their scIB numbers are directly comparable to
the VAE arms: export the harness-preprocessed data, fit the model in its own
env, then score the embedding through `full_metric_suite` in `wcd-kbet`.

**Seurat (CCA integration), env `seurat-env` (r-seurat 5.5.1):**
```
export WCD_DATA=<data> R_HOME=$WCD_KBET/lib/R R_LIBS=<Rlib_kbet> PYTHONPATH=src
for DS in pancreas immune lung sim1 sim2 atac_small; do
  SEURAT_DS=$DS SEURAT_DIR=results/seurat_$DS python scripts/seurat_export.py
  SEURAT_DIR=results/seurat_$DS Rscript scripts/seurat_integrate_ds.R   # seurat-env
  SEURAT_DS=$DS SEURAT_DIR=results/seurat_$DS python scripts/seurat_score_ds.py
done
```
(pancreas legacy output is `results/seurat_in/`; `seurat_integrate.R`/`seurat_score.py`
are the original pancreas-hardcoded versions, kept for provenance.)

**scVI, env `scvi-env`:** prep the AnnData in `wcd-kbet`, fit in `scvi-env`
reading the h5ad (avoids importing the harness package, which pulls scib), score
in `wcd-kbet`: `scripts/scvi_fit_pancreas.py` + `scripts/scvi_score.py`.

## Geometry (trustworthiness / continuity)

`scripts/geometric_fidelity.py` — VERBATIM port of
SteinOBrienLab/disentanglement-benchmarking `calculate_geometric_fidelity`:
high_d = ambient `load_task` X, low_d = latent, global 5000-cell subsample
(seed 42), k=15. Reads latents from `$WCD_EMBED_OUT` (VAE) and `$WCD_BASE_EMB`
(baselines); Seurat from `results/seurat_<ds>/seurat_emb.csv`.
```
WCD_EMBED_OUT=<val latents> WCD_BASE_EMB=<baseline latents> WCD_DATA=<data> \
  python scripts/geometric_fidelity.py   # -> tc_barycenter.csv, tc_baselines.csv
```

## Stratified comparison (the paper analysis)

`scripts/compare_methods_stratified.py` — compares methods WITHIN each dataset
and aggregates by mean RANK (Friedman-blocked on dataset), never pooling raw
scores across datasets (datasets differ too much in baseline difficulty for a
pooled mean or between-dataset CI to be meaningful). Emits composite/batch/bio
matrices, ranks, and figures.
```
python scripts/compare_methods_stratified.py --e3-csv <E3 baseline rows csv> --make-figures
```

## NOTE — variance

The barycenter arms are 3 seeds; the classical baselines are single-run
(near-deterministic). More seeds are needed to tighten the within-dataset CIs
before the head/method gaps are submission-ready — see the mean-rank analysis,
which is robust to this, as the primary result meanwhile.
