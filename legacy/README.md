# Retired pipeline (kept for provenance only)

Code check of tag prereg-tier12-v1, finding CR-10 (2026-10-03); code review N9 / C1. These files produced
the pre-revision results (FINAL_RESULTS.md). That design was rejected: the committed manifest
`scvi_final_manifest.tsv` holds 2,000 unconditioned rows with the discriminator taking one step at lambda 5-1500, and
`score_final_config.py` adds scib's raw PCR with the wrong sign (C1). Every file here stops with a RETIRED message
(Python: `raise SystemExit` before any import; shell: `exit 1`). git history keeps the runnable versions.

| file | role |
|---|---|
| scvi_final_manifest.tsv | rejected manifest |
| scvi_adv_fit.py | its fitter (replaced by scripts/fit_paper_config.py) |
| score_final_config.py | its scorer (replaced by scripts/score_scib_native.py: scib.metrics.metrics, pcr_comparison) |
| run_jhpce_pilot.sh | JHPCE driver of the rejected manifest |
| score_parallel.sh, score_baselines_parallel.sh | parallel drivers of score_final_config.py |
| run_final_baselines.py | baselines in the rejected npz/scoring format |

Current path: `scripts/prep_scib_task.py` -> `scripts/build_paper_manifest.py` -> `scripts/run_stage.py`
(fits with `scripts/fit_paper_config.py`, X13 CPU rows with `scripts/run_cpu_baselines.py`, scores with
`scripts/score_scib_native.py`) -> pre-registered rules `scripts/prereg_rules.py`, `scripts/freeze_a1_grid.py`,
`scripts/freeze_matched_lambda.py`, `scripts/decide_a2_a3.py` (docs/PREREG.md).
Scripts still in scripts/ that called these files (jhpce_gate_then_drain.sh, run_final_sweep.sh,
run_scvi_*_sweep.sh, run_scvi_xop_probe.sh, build_final_manifest.py, build_unified_manifest.py, build_final_report.py,
analyze_final.py) belong to the same retired path; they now fail because these files are not in scripts/.
