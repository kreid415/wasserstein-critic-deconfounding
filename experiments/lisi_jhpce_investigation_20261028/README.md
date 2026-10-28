# LISI run-to-run variation on JHPCE: investigation (2026-10-28)

Report: `lisi_jhpce_investigation_20261028.md`. Everything here was produced by the scripts in this folder from the latents of the 2026-10-23 score-match check
(SHA-256 list: `latents.sha256`; source files: JHPCE `/fastscratch/myscratch/kreid/wcd/score_check/latents`, copied to the workstation scratch).

Environments: JHPCE `/users/kreid/envs/wcd-score` (versions: `results/control/env_jhpce_versions_score.json`), workstation `wcd-kbet` (`results/control/env_workstation.txt`).
Seeds: scib's kNN uses `random_state=0`; the one random element is the helper's row subsample (the subject of the study); kBET seed 0 (scorer default). Thread settings: OMP/MKL/OPENBLAS/NUMBA threads 1, MKL_THREADING_LAYER=SEQUENTIAL.
Provenance of the stored workstation rows: `ws_stored_reference_lisi.json` (read from `tier12_v2/A1/scores/<tag>.csv` on the workstation; used only where the stored latent hash matches).

Re-run (each line needs the environment's python; `D` = directory holding the 20 latents):
1. RDRAND state: `python rdrand_probe.py`
2. Build the preload library (JHPCE, via a Slurm job): `g++ -O1 -fPIC -shared -nostdlib -fno-exceptions -fno-rtti -o fixed_random_device.so fixed_random_device.cpp -lc`  (expected SHA-256 in the report)
3. No-fix run: `bash run_set.sh PYTHON D OUT natural_run1 4`
4. Fixed-seed run: `LD_PRELOAD=$PWD/fixed_random_device.so LISI_RANDOM_DEVICE_VALUE=0xFFFFFFFF bash run_set.sh PYTHON D OUT final_shim_run1 4`
5. Compare: `python compare_probe.py results/comparison.csv ws_stored_reference_lisi.json latents.sha256 jh_nat_1=results/jh:natural_run1 jh_final_1=results/jh:final_shim_run1 jh_final_2=results/jh:final_shim_run2 ws_nat_1=results/ws:natural_run1 ws_nat_2=results/ws:natural_run2 ws_final_1=results/ws:final_shim_run1`
6. Tables: `python summarize_probe.py results`; full-scorer comparison: `python compare_full_scores.py results/full_scorer latents.sha256`
7. Test: `python -m pytest tests/test_lisi_investigation_compare.py` (repo root)

Full-scorer check command (JHPCE, scorer commit a4989ff, repo clean): `LD_PRELOAD=... LISI_RANDOM_DEVICE_VALUE=0xFFFFFFFF` exported, then `SLURM_ARRAY_TASK_ID=i bash cluster/jhpce/score_array.sh a4989ff80206e03b721e0adf874bc5604803e335 plan/list_atac_small.tsv OUT` for i = 1..6.
Note: `results/jh/*shim_run*` files named `shim_run1/2` (not `final_`) were made with a first preload build that also linked libstdc++; the final build (libc only) is `final_shim_run*`.
The `stored_ws_*` columns of `results/lisi_values_by_latent.csv` are not valid for the 3 sinkhorn latents (stored latent hash differs).
