# JHPCE cross-host gate (2026-10-02/03 (provisional: G3 local half and G1/G2/G4 complete; GPU part pending))

Status: PROVISIONAL. G1 (CPU build checks), G2 and G4 are final. The JHPCE half of G3, the GPU rows of G1, and G5 wait on the pinned-GPU gate job, which is still queued (A100 36131135, L40S copy 36133219; the user chose to keep both queued). The split below uses a PLACEHOLDER per-GPU speed ratio of 1.0 until G5 is measured.

Local: RTX 3080 10 GB (driver 580.173.02, CUDA 13.0), AMD Ryzen 5 3600 (AVX2), envs scvi-api (fits) / wcd-kbet (scoring). JHPCE: fit env wcd-fit on one pinned GPU model (A100 job queued, L40S copy queued); scoring env wcd-score on 'shared' nodes (G4 ran on compute-174, AMD EPYC 9555, AVX-512). Repo `jhpce-setup` @ 6675f8c (setup and A100 gate job), 2f29004 (diagnosis jobs, L40S copy); fit code identical to main 8cda6de and to the local calibration commit fe56044 (docstring-only diff), confirmed by 8 local refits that reproduce the calibration latents bitwise. JHPCE jobs (this session's ledger): setup 695f3107 (Slurm 36131132, COMPLETED), G4 diagnosis ed9b0733 (36132749, FAILED: AVX2-pinned scoring on a pre-AVX2 Opteron) and 3fba33c9 (36132943, COMPLETED), gate A100 8749bada (36131135, PENDING), gate L40S copy a2b6ae1f (36133219, PENDING).

| item | result | key numbers |
|---|---|---|
| G1 versions | PASS (CPU build checks); GPU tests PENDING | fit env vs local scvi-api: identical=90 differing=34 unexpected=0; score env vs local wcd-kbet: identical=152 differing=1 unexpected=0; tests/scvi on the JHPCE GPU: PENDING (GPU job) |
| G2 prepped fingerprints | PASS | fingerprint_prepped.py --compare exit 0; 8/8 downloads size+md5 verified |
| G3 latents + scores | PENDING | the 64 JHPCE latents come from the pending GPU job. The 64 local latents are already scored in local wcd-kbet (64/64, 0 failures). |
| G4 scoring equivalence | FAIL | deterministic metrics (all except kBET; hvg_overlap is NaN on both): 5/12 within 1e-6 at default settings, failing ARI_cluster/label 0.018, NMI_cluster/label 0.0027, cLISI 0.00017, graph_conn 0.00029, iLISI 0.0028, isolated_label_F1 0.022, trajectory 3.7e-06; with NUMBA_CPU_NAME=generic on both hosts 10/12 (failing cLISI, iLISI). kBET reported separately (unseeded): cross-host 0.0025 vs same-machine repeat 0.002 (local) / 0.0029 (JHPCE) |
| G5 throughput | PENDING | needs the pending GPU job's queue_times.tsv (local reference: makespan 433.9 s, window 386.6 s, 8-lane factor 3.77) |

## G1 versions

| env | item | local | jhpce | same |
|---|---|---|---|---|
| fit | anndata | 0.12.19 | 0.12.19 | yes |
| fit | h5py | 3.16.0 | 3.16.0 | yes |
| fit | lightning | 2.6.5 | 2.6.5 | yes |
| fit | numba | 0.67.0 | 0.67.0 | yes |
| fit | numpy | 2.4.6 | 2.4.6 | yes |
| fit | ot | 0.9.7.post1 | 0.9.7.post1 | yes |
| fit | pandas | 2.3.3 | 2.3.3 | yes |
| fit | pynndescent | 0.6.0 | 0.6.0 | yes |
| fit | pyro | 1.9.1 | 1.9.1 | yes |
| fit | scanpy | 1.11.5 | 1.11.5 | yes |
| fit | scipy | 1.17.1 | 1.17.1 | yes |
| fit | scvi | 1.4.2 | 1.4.2 | yes |
| fit | sklearn | 1.9.0 | 1.9.0 | yes |
| fit | torch | 2.13.0+cu130 | 2.13.0+cu126 | declared swap (cu130 -> cu126 build) |
| fit | python | 3.11.15 | 3.11.15 | yes |
| fit | torch_cuda | 13.0 | 12.6 | declared swap |
| fit | cudnn | 92000 | 91002 | declared swap (wheel of the cu126 build) |
| fit | gpu | NVIDIA GeForce RTX 3080 | PENDING (GPU job) | pending |
| fit | gpu_capability | [8, 6] | PENDING (GPU job) | pending |
| fit | nvidia-smi | NVIDIA GeForce RTX 3080, 580.173.02, 10240 MiB | PENDING (GPU job) | pending |
| score | anndata | 0.11.4 | 0.11.4 | yes |
| score | anndata2ri | 1.3.2 | 1.3.2 | yes |
| score | h5py | 3.16.0 | 3.16.0 | yes |
| score | igraph | 0.11.9 | 0.11.9 | yes |
| score | leidenalg | 0.10.2 | 0.10.2 | yes |
| score | llvmlite | 0.48.0 | 0.48.0 | yes |
| score | numba | 0.66.0 | 0.66.0 | yes |
| score | numpy | 1.26.4 | 1.26.4 | yes |
| score | pandas | 1.5.3 | 1.5.3 | yes |
| score | pynndescent | 0.5.13 | 0.5.13 | yes |
| score | rpy2 | 3.6.5 (dist metadata) | 3.6.5 (dist metadata) | yes |
| score | scanpy | 1.11.5 | 1.11.5 | yes |
| score | scib | 1.1.7 | 1.1.7 | yes |
| score | scipy | 1.14.1 | 1.14.1 | yes |
| score | sklearn | 1.9.0 | 1.9.0 | yes |
| score | torch | 2.4.1+cu121 | 2.4.1+cu121 | yes |
| score | umap | 0.5.12 | 0.5.12 | yes |
| score | python | 3.11.15 | 3.11.15 | yes |
| score | R | R version 4.5.2 (2025-10-31) | R version 4.5.2 (2025-10-31) | yes |
| score | kbet | kBET version=0.99.6 objects=21 md5=3b541efa166047024303705943926906 | kBET version=0.99.6 objects=21 md5=3b541efa166047024303705943926906 | yes |
| score | runtime BLAS/OpenMP | mkl 2025.3-Product libmkl_rt.so.2 intel; openblas 0.3.27.dev libscipy_openblas-c128ec02.so Zen; openblas 0.3.34 libopenblasp-r0.3.34.so Zen; openmp None libgomp-a34b3233.so.1; openmp None libomp.so | mkl 2025.3-Product libmkl_rt.so.2 intel; openblas 0.3.27.dev libscipy_openblas-c128ec02.so Cooperlake; openblas 0.3.34 libopenblasp-r0.3.34.so Cooperlake; openmp None libgomp-a34b3233.so.1; openmp None libomp.so | same libraries; CPU code path differs (OpenBLAS arch) |

wcd-fit vs local scvi-api `pip list`: identical=90 differing=34 unexpected=0; every difference is the declared torch 2.13.0+cu126 / CUDA 12 wheel substitution (local torch 2.13.0+cu130 needs a CUDA 13 driver; JHPCE GPU nodes run driver 555). wcd-score vs local wcd-kbet: identical=152 differing=1 unexpected=0 (igraph: local pip metadata reads a stale 0.11.8 record; the imported igraph is 0.11.9 on both hosts). Conda layers: package file names + md5 identical to the local explicit specs (both envs); `pip check` identical to local; kBET namespace fingerprint identical (theislab/kBET afc5f431). GPU rows come from the pending GPU job; the fit-env rows above are from the CPU build node.

## G2 prepped-input fingerprints

```
atac_large__scib.h5ad: OK
atac_small__scib.h5ad: OK
immune__scib.h5ad: OK
immune_hum_mou__scib.h5ad: OK
lung__scib.h5ad: OK
pancreas__scib.h5ad: OK
sim1__scib.h5ad: OK
sim2__scib.h5ad: OK
```

| file | bytes | md5 | status |
|---|---|---|---|
| human_pancreas_norm_complexBatch.h5ad | 315955785 | 5aa155652b998c90395a0c9bd731b4b3 | downloaded_153s |
| Lung_atlas_public.h5ad | 1019664664 | aac0832c3e3413ae279343f5de02a2c6 | downloaded_277s |
| Immune_ALL_human.h5ad | 2064748344 | 482a6a3117432d41a46effd8376b8ee0 | downloaded_1714s |
| Immune_ALL_hum_mou.h5ad | 4267730467 | 3cc1ba9f35b21551dd7a81162998b727 | downloaded_145s |
| small_atac_gene_activity.h5ad | 26991491 | 8eb4af0e5e2d07842e516d5b8047b449 | downloaded_2s |
| large_atac_gene_activity.h5ad | 148888428 | bab0bd97840692f938f4361bd07fac8e | downloaded_7s |
| sim1_1_norm.h5ad | 2207895247 | ea2589604bec1fd184ee489d1491e54f | downloaded_78s |
| sim2_norm.h5ad | 4208083958 | 4d22e971f1e199726f99748cf282e689 | downloaded_3204s |

## G3

PENDING: the 64 JHPCE latents come from the pending GPU job. The 64 local latents are already scored in local wcd-kbet (64/64, 0 failures).

## G4 scoring equivalence

Per metric. G4 = the 2 latents (Q_none_0, Q_barycenter_1) and the prepped file, md5-identical on both hosts, scored locally (wcd-kbet) and on JHPCE (wcd-score, compute-174 = AMD EPYC 9555, AVX-512); tolerance 1e-6 absolute. G4b = the same on JHPCE against the JHPCE-regenerated prepped file. local repeat = the same latents re-scored on the local machine (inside the G3 run); JHPCE repeat = a second job on compute-174; numba generic = both hosts with NUMBA_CPU_NAME=generic. The host verdict uses the deterministic metrics only: kBET is unseeded in scib 1.1.7 (R kBET samples test cells) and changes between runs on one machine (local repeat column; the lead measured up to 0.0047), so its cross-host difference is not a host difference; a fixed R seed is being added to score_scib_native.py on prereg-v2. Diagnosis of the deterministic metrics: (i) the scorer's kNN graph (pynndescent, numba) depends on the CPU vector width: AVX-512 EPYC vs local AVX2 differ in 251 of 33,506 rows, which moves ARI, NMI, isolated-label F1, graph connectivity and trajectory; an AVX-only Opteron (compute-053) builds the same graph as local; with NUMBA_CPU_NAME=generic the graph is identical on all three CPU types and those metrics agree to <=1e-16; (ii) graph iLISI / cLISI still differ across hosts AND between two runs on compute-174 (identical graph; ARI/NMI unchanged), but are identical between local runs (<=1.4e-17). The cause is not scib's LISI subsampling RNG (knn_graph.cpp draws a random number per row but skips nothing at subsample=100); unresolved. PCR and cell-cycle conservation agree to <=5.7e-7.

| metric | max diff G4 | status G4 | max diff G4b | status G4b | max diff local repeat | status local repeat | max diff JHPCE repeat | status JHPCE repeat | max diff numba generic | status numba generic |
|---|---|---|---|---|---|---|---|---|---|---|
| ARI_cluster/label | 0.01831 | exceeds tol | 0.01831 | exceeds tol | 0 | ok | 0 | ok | 0 | ok |
| ASW_label | 0 | ok | 0 | ok | 0 | ok | 0 | ok | 0 | ok |
| ASW_label/batch | 0 | ok | 0 | ok | 0 | ok | 0 | ok | 0 | ok |
| NMI_cluster/label | 0.002678 | exceeds tol | 0.002678 | exceeds tol | 0 | ok | 0 | ok | 1.11e-16 | ok |
| PCR_batch | 1.277e-07 | ok | 1.277e-07 | ok | 0 | ok | 0 | ok | 1.277e-07 | ok |
| cLISI | 0.000167 | exceeds tol | 0.0001147 | exceeds tol | 0 | ok | 0.0002724 | exceeds tol | 0.0001549 | exceeds tol |
| cell_cycle_conservation | 5.686e-07 | ok | 5.686e-07 | ok | 0 | ok | 0 | ok | 5.686e-07 | ok |
| graph_conn | 0.0002894 | exceeds tol | 0.0002894 | exceeds tol | 0 | ok | 0 | ok | 0 | ok |
| hvg_overlap | 0 | ok | 0 | ok | 0 | ok | 0 | ok | 0 | ok |
| iLISI | 0.002805 | exceeds tol | 0.004603 | exceeds tol | 0 | ok | 0.001974 | exceeds tol | 0.00196 | exceeds tol |
| isolated_label_F1 | 0.02188 | exceeds tol | 0.02188 | exceeds tol | 0 | ok | 0 | ok | 0 | ok |
| isolated_label_silhouette | 0 | ok | 0 | ok | 0 | ok | 0 | ok | 0 | ok |
| kBET | 0.002518 | exceeds tol | 0.003888 | exceeds tol | 0.002005 | exceeds tol | 0.002908 | exceeds tol | 0.003577 | exceeds tol |
| trajectory | 3.728e-06 | exceeds tol | 3.728e-06 | exceeds tol | 0 | ok | 0 | ok | 0 | ok |

kNN graph of Q_none_0 as built by the scorer (sc.pp.neighbors on the latent), compared entry by entry:

| pair | target | cells | knn_rows_identical | knn_identical | dist_identical | conn_identical |
|---|---|---|---|---|---|---|
| local_vs_jhpce053 | host | 33506 | 33506 | True | True | True |
| local_vs_jhpce174 | host | 33506 | 33255 | False | False | False |
| local_vs_jhpce053 | generic | 33506 | 33506 | True | True | True |
| local_vs_jhpce174 | generic | 33506 | 33506 | True | True | True |

## G5

PENDING: needs the pending GPU job's queue_times.tsv

## Task split

Stage-gated model (docs/PREREG.md sec 0 on prereg-rules @ 2e4f24d; scripts/gate_task_split.py): S1 A1 -> S2 A2+A3 -> S3 X1 adversarial rows -> S4 X3/X6/X7/X8/X12, every gate global and including the scoring of the stage's last latents; fillers (X1 none/scvi_adv rows, X13) run in a host's gate waits; whole tasks per host (SI-17); JHPCE fits of a task may spread over G GPUs of the one pinned model. Fit cost: cost_model.py on the 6,061-fit design of record; local 8-lane factor 3.77 (mixed-arm queue); scoring 16.24 ms per cell per latent, 4 local CPUs while the GPU fits (12 when idle), 48 JHPCE CPUs. PROVISIONAL: the JHPCE factor is a placeholder (ratio 1.0) until G5; see the ratio sweep at the end of this section.

Speed ratio used: 1.0 (PLACEHOLDER until G5); all-local: 24.02 days. Manifest md5 179f969e271ebf0eb8eba0bfb6b3cf4b (6061 fits, 2168.6 local lane-hours, 785.9 scoring CPU-hours).

| jhpce_gpus | wall_days | local | jhpce | local_busy_days | jhpce_busy_days | jhpce_job_starts | stage_days_local_jhpce |
|---|---|---|---|---|---|---|---|
| 1 | 12.69 | atac_large immune lung pancreas | atac_small immune_hum_mou sim1 sim2 | 12.37 | 11.69 | 6 | S1 1.95|2.15 / S2 0.35|0.19 / S3 7.52|6.36 / S4 2.16|2.66 |
| 2 | 8.21 | atac_large pancreas sim1 | atac_small immune immune_hum_mou lung sim2 | 7.61 | 8.25 | 10 | S1 1.14|1.49 / S2 0.0|0.27 / S3 4.66|4.82 / S4 1.56|1.64 |
| 3 | 6.09 | immune_hum_mou sim1 | atac_large atac_small immune lung pancreas sim2 | 5.94 | 6.08 | 15 | S1 1.14|0.99 / S2 0.0|0.18 / S3 3.36|3.52 / S4 1.25|1.2 |
| 4 | 5.2 | immune_hum_mou | atac_large atac_small immune lung pancreas sim1 sim2 | 3.44 | 5.2 | 16 | S1 0.0|1.03 / S2 0.0|0.14 / S3 2.23|2.97 / S4 1.06|0.94 |

Sensitivity to the two pending design decisions (base-optimal split re-evaluated; scenario optimum):

| gpus | scenario | base_split_wall_days | scenario_optimum_wall_days |
|---|---|---|---|
| 1 | base | 12.69 | 12.69 |
| 1 | x6_plus54 | 12.74 | 12.74 |
| 1 | a23_uncond_plus144 | 13.03 | 13.03 |
| 1 | both | 13.08 | 13.08 |
| 2 | base | 8.21 | 8.21 |
| 2 | x6_plus54 | 8.29 | 8.29 |
| 2 | a23_uncond_plus144 | 8.48 | 8.48 |
| 2 | both | 8.55 | 8.55 |
| 3 | base | 6.09 | 6.09 |
| 3 | x6_plus54 | 6.15 | 6.15 |
| 3 | a23_uncond_plus144 | 6.26 | 6.26 |
| 3 | both | 6.33 | 6.33 |
| 4 | base | 5.2 | 5.2 |
| 4 | x6_plus54 | 5.25 | 5.24 |
| 4 | a23_uncond_plus144 | 5.33 | 5.33 |
| 4 | both | 5.39 | 5.39 |

JHPCE queue-wait sensitivity (wait per 3-day GPU job; free assignment vs the 3 A1 tasks kept local):

| queue_wait_days | gpus | constraint | wall_days | jhpce | local | local_busy_days | jhpce_busy_days | jhpce_job_starts |
|---|---|---|---|---|---|---|---|---|
| 0 | 1 | free | 12.69 | atac_small immune_hum_mou sim1 sim2 | atac_large immune lung pancreas | 12.37 | 11.69 | 6 |
| 0 | 1 | A1_tasks_local | 14.84 | atac_large lung pancreas sim2 | atac_small immune immune_hum_mou sim1 | 13.91 | 10.15 | 4 |
| 0 | 2 | free | 8.21 | atac_small immune immune_hum_mou lung sim2 | atac_large pancreas sim1 | 7.61 | 8.25 | 10 |
| 0 | 2 | A1_tasks_local | 11.16 | atac_large immune_hum_mou lung pancreas sim2 | atac_small immune sim1 | 10.51 | 6.79 | 6 |
| 0 | 3 | free | 6.09 | atac_large atac_small immune lung pancreas sim2 | immune_hum_mou sim1 | 5.94 | 6.08 | 15 |
| 0 | 3 | A1_tasks_local | 10.51 | atac_large immune_hum_mou lung pancreas sim2 | atac_small immune sim1 | 10.51 | 4.54 | 9 |
| 0 | 4 | free | 5.2 | atac_large atac_small immune lung pancreas sim1 sim2 | immune_hum_mou | 3.44 | 5.2 | 16 |
| 0 | 4 | A1_tasks_local | 10.51 | atac_large immune_hum_mou lung pancreas sim2 | atac_small immune sim1 | 10.51 | 3.41 | 8 |
| 0.5 | 1 | free | 13.99 | immune immune_hum_mou lung | atac_large atac_small pancreas sim1 sim2 | 13.38 | 13.69 | 6 |
| 0.5 | 1 | A1_tasks_local | 15.42 | immune_hum_mou lung sim2 | atac_large atac_small immune pancreas sim1 | 15.59 | 10.47 | 4 |
| 0.5 | 2 | free | 10.06 | atac_small immune_hum_mou lung sim1 sim2 | atac_large immune pancreas | 10.06 | 9.52 | 10 |
| 0.5 | 2 | A1_tasks_local | 12.66 | atac_large immune_hum_mou lung pancreas sim2 | atac_small immune sim1 | 10.51 | 8.29 | 6 |
| 0.5 | 3 | free | 8.43 | atac_small immune immune_hum_mou lung sim2 | atac_large pancreas sim1 | 7.61 | 8.02 | 15 |
| 0.5 | 3 | A1_tasks_local | 10.49 | atac_large immune_hum_mou lung pancreas sim2 | atac_small immune sim1 | 10.51 | 6.04 | 9 |
| 0.5 | 4 | free | 6.61 | atac_large immune immune_hum_mou lung pancreas sim1 | atac_small sim2 | 5.78 | 6.61 | 20 |
| 0.5 | 4 | A1_tasks_local | 10.51 | atac_large immune_hum_mou lung pancreas sim2 | atac_small immune sim1 | 10.51 | 4.41 | 8 |
| 1 | 1 | free | 15.39 | immune_hum_mou sim1 sim2 | atac_large atac_small immune lung pancreas | 15.39 | 12.67 | 5 |
| 1 | 1 | A1_tasks_local | 16.07 | lung pancreas sim2 | atac_large atac_small immune immune_hum_mou sim1 | 15.98 | 11.09 | 5 |
| 1 | 2 | free | 11.68 | atac_large atac_small lung pancreas sim1 | immune immune_hum_mou sim2 | 11.12 | 11.49 | 10 |
| 1 | 2 | A1_tasks_local | 13.25 | atac_large immune_hum_mou lung pancreas | atac_small immune sim1 sim2 | 13.25 | 8.41 | 6 |
| 1 | 3 | free | 10.25 | immune immune_hum_mou lung pancreas sim1 | atac_large atac_small sim2 | 7.85 | 9.44 | 15 |
| 1 | 3 | A1_tasks_local | 11.99 | atac_large immune_hum_mou lung pancreas sim2 | atac_small immune sim1 | 10.51 | 7.54 | 9 |
| 1 | 4 | free | 8.44 | atac_large immune lung pancreas sim1 sim2 | atac_small immune_hum_mou | 6.46 | 8.44 | 20 |
| 1 | 4 | A1_tasks_local | 10.53 | atac_large immune_hum_mou lung pancreas sim2 | atac_small immune sim1 | 10.51 | 5.41 | 12 |
| 2 | 1 | free | 16.45 | atac_large pancreas sim1 | atac_small immune immune_hum_mou lung sim2 | 16.45 | 15.61 | 7 |
| 2 | 1 | A1_tasks_local | 17.74 | atac_large lung pancreas | atac_small immune immune_hum_mou sim1 sim2 | 16.65 | 13.41 | 6 |
| 2 | 2 | free | 13.94 | atac_large pancreas sim1 sim2 | atac_small immune immune_hum_mou lung | 13.71 | 13.2 | 12 |
| 2 | 2 | A1_tasks_local | 15.52 | atac_large lung pancreas sim2 | atac_small immune immune_hum_mou sim1 | 13.91 | 11.09 | 10 |
| 2 | 3 | free | 11.42 | atac_large immune_hum_mou lung pancreas sim1 | atac_small immune sim2 | 10.75 | 10.46 | 12 |
| 2 | 3 | A1_tasks_local | 13.53 | atac_large immune_hum_mou lung sim2 | atac_small immune pancreas sim1 | 13.53 | 7.53 | 9 |
| 2 | 4 | free | 11.02 | atac_large immune_hum_mou lung sim1 sim2 | atac_small immune pancreas | 11.02 | 9.29 | 16 |
| 2 | 4 | A1_tasks_local | 11.91 | atac_large immune_hum_mou lung pancreas sim2 | atac_small immune sim1 | 10.51 | 7.41 | 12 |

Provisional split versus the per-GPU speed ratio (no queue wait):

| ratio | jhpce_gpus | wall_days | jhpce | local_busy_days | jhpce_busy_days |
|---|---|---|---|---|---|
| 0.75 | 1 | 14.24 | immune lung pancreas | 13.76 | 13.73 |
| 0.75 | 2 | 9.92 | atac_small immune_hum_mou lung sim1 sim2 | 10.06 | 9.35 |
| 0.75 | 3 | 7.78 | atac_small immune immune_hum_mou lung sim2 | 7.61 | 7.34 |
| 0.75 | 4 | 6.09 | atac_large atac_small immune lung pancreas sim2 | 5.94 | 6.08 |
| 1 | 1 | 12.69 | atac_small immune_hum_mou sim1 sim2 | 12.37 | 11.69 |
| 1 | 2 | 8.21 | atac_small immune immune_hum_mou lung sim2 | 7.61 | 8.25 |
| 1 | 3 | 6.09 | atac_large atac_small immune lung pancreas sim2 | 5.94 | 6.08 |
| 1 | 4 | 5.2 | atac_large atac_small immune lung pancreas sim1 sim2 | 3.44 | 5.2 |
| 1.25 | 1 | 10.89 | atac_large atac_small pancreas sim1 sim2 | 10.7 | 10.71 |
| 1.25 | 2 | 7.29 | atac_large atac_small immune lung pancreas sim2 | 5.94 | 7.29 |
| 1.25 | 3 | 5.27 | immune immune_hum_mou lung pancreas sim1 sim2 | 5.11 | 5.1 |
| 1.25 | 4 | 4.3 | atac_large atac_small immune immune_hum_mou lung pancreas sim1 | 2.76 | 4.3 |
| 1.5 | 1 | 9.92 | atac_small immune_hum_mou lung sim1 sim2 | 10.06 | 9.35 |
| 1.5 | 2 | 6.09 | atac_large atac_small immune lung pancreas sim2 | 5.94 | 6.08 |
| 1.5 | 3 | 4.7 | atac_large atac_small immune lung pancreas sim1 sim2 | 3.44 | 4.63 |
| 1.5 | 4 | 3.59 | atac_large atac_small immune immune_hum_mou lung pancreas sim1 | 2.76 | 3.59 |
| 2 | 1 | 8.21 | atac_small immune immune_hum_mou lung sim2 | 7.61 | 8.25 |
| 2 | 2 | 5.2 | atac_large atac_small immune lung pancreas sim1 sim2 | 3.44 | 5.2 |
| 2 | 3 | 3.59 | atac_large atac_small immune immune_hum_mou lung pancreas sim1 | 2.76 | 3.59 |
| 2 | 4 | 3.05 | atac_large atac_small immune immune_hum_mou lung pancreas sim1 sim2 | 0 | 3.05 |

## Caveats and deviations

- G4 FAILED at the specified 1e-6 tolerance, so all 128 G3 latents are scored in local wcd-kbet, as the task specifies; G3 paired differences therefore include kBET's run-to-run noise (local repeat: up to 2.0e-3 in kBET, i.e. ~4e-4 in the 5-metric batch mean).
- Gate fits read a staged copy of the LOCAL immune__scib.h5ad (md5 3888d256...), so G3 compares hosts on identical input bytes; the JHPCE-regenerated file passes the fingerprint check (G2) and differs in bytes only (uns prep_code_sha, X_pca at BLAS precision).
- GPU choice: A100 (gpu:tesa100:1) chosen at 17:10 EDT 2026-10-02 because no A100 or L40S was free and the A100 queue looked shorter; an L40S copy was queued at 20:13 as a hedge. The user decided to keep both queued; whichever runs defines that model's measurement, and the JHPCE tasks of the benchmark must then use that one model.
- JHPCE GPU access for this account is slow: priority 1 (fairshare), the 2-h single-GPU gate job has been pending since 17:17 EDT, squeue's start estimate was 2026-10-04 11:05 EDT. The split's makespans exclude queue waits; the queue-wait table shows the effect of 0.5-2 days per 3-day GPU job.
- Gate GPU job requests 10 CPUs for 8 lanes (local: 12 logical CPUs); JHPCE GPU nodes have 24 CPUs per 4 GPUs, so benchmark jobs may get fewer CPUs per GPU than the gate.
- Diagnosis job 36132749 failed because I pinned AVX2 code paths (OPENBLAS_CORETYPE=Haswell, MKL_CBWR=AVX2) and Slurm placed it on compute-053 (Opteron, no AVX2); its kNN graphs were harvested and the scoring check was redone on compute-174 (36132943).
- Split model limits: rule-freeze and sign-off time not modelled; per-stage gates assumed global; X6 +54 fits scaled proportionally per task; +144 A2/A3 fits modelled as a copy of the existing A2/A3 rows (unconditioned decoder at the same cost); local scoring assumes 4 free CPUs while fitting.
