# JHPCE cross-host gate (2026-10-03; final for the gate configs, both pinned-GPU gate jobs finished and harvested)

Status: Final for the gate configs. Both JHPCE gate jobs (A100 Slurm 36131135, L40S Slurm 36133219) finished; their latents were re-harvested from the job workdirs and md5-verified, and all 192 gate latents were scored in one local env with the current scorer. G4 fails (scores of identical latents differ between local and JHPCE CPUs), so every benchmark latent is scored locally and the split models that.

Local: RTX 3080 10 GB (driver 580.173.02, CUDA 13.0), AMD Ryzen 5 3600 (6 cores / 12 threads, AVX2); envs scvi-api (fits), wcd-kbet (scoring). JHPCE: fit env wcd-fit (torch 2.13.0+cu126) on compute-128 (A100 80GB PCIe; 2x Intel Xeon Gold 5317) and compute-171 (L40S 46 GB; AMD EPYC 7443P), driver 555.42.06; each gate job 1 GPU, 10 CPUs, 64 GB, 8 fit lanes. Report built on branch `jhpce-gate-v2` (from main @ 878d8ea). Jobs: setup 36131132 (shared, done); G4 diagnosis 36132749 (failed on compute-053) and 36132943 (done); gate A100 36131135 and L40S 36133219 (done; re-harvested 2026-10-03).

| item | A100 | L40S |
|---|---|---|
| G1 env (CPU build checks, shared) | PASS: fit identical=90 differing=34 unexpected=0; score identical=152 differing=1 unexpected=0 | PASS: fit identical=90 differing=34 unexpected=0; score identical=152 differing=1 unexpected=0 |
| G1 tests/scvi on the GPU node | PASS: `27 passed, 1 skipped in 23.77s` (NVIDIA A100 80GB PCIe, torch 2.13.0+cu126, driver 555.42.06) | PASS: `27 passed, 1 skipped in 12.26s` (NVIDIA L40S, torch 2.13.0+cu126, driver 555.42.06) |
| G2 prepped inputs (shared) | PASS: fingerprint_prepped.py --compare exit 0, 8/8 downloads md5 verified | PASS: fingerprint_prepped.py --compare exit 0, 8/8 downloads md5 verified |
| harvest integrity | tar md5 OK (8d827557dfe577e3a2d6bfa1b0f1eeec); gate_completion 64/64 valid, problems 0 | tar md5 OK (e863e27f3b9f9531bd16ded5374b6b55); gate_completion 64/64 valid, problems 0 |
| G3 latents vs local (64 configs) | bitwise 0/64; per-dim r median 0.9857 (min 0.513); 15-NN overlap mean 0.515 (min 0.205; two local seeds: 0.150) | bitwise 0/64; per-dim r median 0.9875 (min 0.515); 15-NN overlap mean 0.512 (min 0.288; two local seeds: 0.150) |
| G3 scores (flag rule) | FLAGGED: 2 of 14 (arm, cond) groups flagged (none/1, pooled/1); sign-flip null: 2.0 flags expected by chance over 28 tests, P(>= 2) = 0.6253 | FLAGGED: 2 of 14 (arm, cond) groups flagged (mmd/1, none/1); sign-flip null: 2.875 flags expected by chance over 28 tests, P(>= 2) = 0.8407 |
| G4 scoring on JHPCE CPUs (shared) | FAIL: 5/12 deterministic metrics within 1e-6; all scoring stays local | FAIL: 5/12 deterministic metrics within 1e-6; all scoring stays local |
| G5 8-lane throughput | makespan 599.2 s (local 433.9 s); window 532.8 s (local 386.6 s); factor 2.7 vs 3.77 -> per-GPU ratio 0.716 | makespan 353.7 s (local 433.9 s); window 314.6 s (local 386.6 s); factor 4.62 vs 3.77 -> per-GPU ratio 1.225 |
| queue wait observed (submit -> start) | 9.41 h (2026-10-02T17:17:57 -> 2026-10-03T02:42:37, 2-h 1-GPU 10-CPU job) | 7.82 h (2026-10-02T20:13:04 -> 2026-10-03T04:02:30, 2-h 1-GPU 10-CPU job) |
| split, base case (SI-30, observed wait) | 1 GPU: 18.85 d; 2 GPU: 15.93 d; 3 GPU: 13.94 d; 4 GPU: 13.8 d (all-local 26.25 d) | 1 GPU: 15.99 d; 2 GPU: 13.77 d; 3 GPU: 13.78 d; 4 GPU: 13.78 d (all-local 26.25 d) |

## Recommendation (the user picks)

Rule: per GPU model, the smallest JHPCE GPU count whose base-case makespan (SI-30, observed queue wait, all scoring local) is within 0.25 d of that model's minimum over 1-4 GPUs; between models, fewer G3-flagged (arm, cond) groups first, then the shorter makespan, then the faster G5.

- A100: 3 GPU(s) -> 13.94 d (minimum over 1-4 GPUs 13.8 d; all-local 26.25 d); JHPCE tasks atac_large, immune_hum_mou, lung, pancreas, sim2; local atac_small, immune, sim1; local GPU busy 11.29 d, JHPCE busy 6.96 d per GPU, 12 JHPCE job starts; G3 flagged groups 2; G5 ratio 0.716.
- L40S: 2 GPU(s) -> 13.77 d (minimum over 1-4 GPUs 13.77 d; all-local 26.25 d); JHPCE tasks atac_large, immune_hum_mou, lung, pancreas, sim2; local atac_small, immune, sim1; local GPU busy 11.46 d, JHPCE busy 6.1 d per GPU, 8 JHPCE job starts; G3 flagged groups 2; G5 ratio 1.225.

**Rule outcome: L40S, 2 GPU(s).** Rationale: the L40S is faster per GPU than both the local RTX 3080 and the A100 on these fits, its G3 agreement matches the A100's (same number of flagged groups, both consistent with the sign-flip null), and its observed queue wait was shorter. More than two L40S GPUs do not shorten the makespan: every latent is scored on the 12 local CPUs, which become the bottleneck in S3 and S4 once JHPCE latents join their backlog, and with SI-30 the pilot stages S1-S2 run on the local GPU only. JHPCE fits of the X1 adversarial rows can start only after the A1-A3 rules freeze (S3); before that only filler rows (X1 none/scvi_adv, X13) can run there. Every JHPCE submission still needs the user's go (SI-29).

## G1 versions

| env | item | local | jhpce A100 | jhpce L40S | same |
|---|---|---|---|---|---|
| fit | anndata | 0.12.19 | 0.12.19 | 0.12.19 | yes |
| fit | h5py | 3.16.0 | 3.16.0 | 3.16.0 | yes |
| fit | lightning | 2.6.5 | 2.6.5 | 2.6.5 | yes |
| fit | numba | 0.67.0 | 0.67.0 | 0.67.0 | yes |
| fit | numpy | 2.4.6 | 2.4.6 | 2.4.6 | yes |
| fit | ot | 0.9.7.post1 | 0.9.7.post1 | 0.9.7.post1 | yes |
| fit | pandas | 2.3.3 | 2.3.3 | 2.3.3 | yes |
| fit | pynndescent | 0.6.0 | 0.6.0 | 0.6.0 | yes |
| fit | pyro | 1.9.1 | 1.9.1 | 1.9.1 | yes |
| fit | scanpy | 1.11.5 | 1.11.5 | 1.11.5 | yes |
| fit | scipy | 1.17.1 | 1.17.1 | 1.17.1 | yes |
| fit | scvi | 1.4.2 | 1.4.2 | 1.4.2 | yes |
| fit | sklearn | 1.9.0 | 1.9.0 | 1.9.0 | yes |
| fit | torch | 2.13.0+cu130 | 2.13.0+cu126 | 2.13.0+cu126 | declared swap (cu130 -> cu126 build) |
| fit | python | 3.11.15 | 3.11.15 | 3.11.15 | yes |
| fit | torch_cuda | 13.0 | 12.6 | 12.6 | declared swap |
| fit | cudnn | 92000 | 91002 | 91002 | declared swap (cu126 wheel) |
| fit | gpu | NVIDIA GeForce RTX 3080 | NVIDIA A100 80GB PCIe | NVIDIA L40S | hardware |
| fit | gpu_capability | [8, 6] | [8, 0] | [8, 9] | hardware |
| fit | host | localhost | compute-128.cm.cluster | compute-171.cm.cluster | hardware |
| fit | nvidia-smi | NVIDIA GeForce RTX 3080, 580.173.02, 10240 MiB | NVIDIA A100 80GB PCIe, 555.42.06, 81920 MiB | NVIDIA L40S, 555.42.06, 46068 MiB | hardware |

| env | item | local | jhpce | same |
|---|---|---|---|---|
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

wcd-fit vs local scvi-api `pip list`: identical=90 differing=34 unexpected=0; every difference is the declared torch 2.13.0+cu126 / CUDA 12 wheel substitution (local torch 2.13.0+cu130 needs a CUDA 13 driver; JHPCE GPU nodes run driver 555). wcd-score vs local wcd-kbet: identical=152 differing=1 unexpected=0. Conda layers identical to the local explicit specs; `pip check` identical to local; kBET namespace fingerprint identical (theislab/kBET afc5f431). GPU-node tests ran at the commit each gate job cloned (A100 6675f8c, L40S 2f29004; identical src/).

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

## G3 latent agreement (JHPCE vs local, same config and seed)

Per config: per-dimension Pearson r between the two latents (dims matched by index, identical init), max |dz|, relative Frobenius difference, and the mean 15-NN overlap (exact Euclidean kNN). Reference: two DIFFERENT local seeds of the same (arm, cond).

| tag | arm | cond | seed | r_min A100 | r_median A100 | knn A100 | r_min L40S | r_median L40S | knn L40S |
|---|---|---|---|---|---|---|---|---|---|
| Q_barycenter_1 | barycenter | 0 | 1 | 0.972 | 0.984 | 0.368 | 0.973 | 0.984 | 0.361 |
| Q_barycenter_3 | barycenter | 0 | 3 | 0.765 | 0.966 | 0.322 | 0.515 | 0.974 | 0.327 |
| Q_barycenter_5 | barycenter | 0 | 5 | 0.940 | 0.956 | 0.314 | 0.923 | 0.960 | 0.321 |
| Q_barycenter_7 | barycenter | 0 | 7 | 0.968 | 0.978 | 0.412 | 0.941 | 0.962 | 0.401 |
| Q_barycenter_9 | barycenter | 0 | 9 | 0.869 | 0.979 | 0.322 | 0.903 | 0.988 | 0.362 |
| Q_barycenter_0 | barycenter | 1 | 0 | 0.944 | 0.985 | 0.308 | 0.981 | 0.987 | 0.350 |
| Q_barycenter_2 | barycenter | 1 | 2 | 0.966 | 0.981 | 0.405 | 0.947 | 0.981 | 0.387 |
| Q_barycenter_4 | barycenter | 1 | 4 | 0.975 | 0.979 | 0.330 | 0.966 | 0.985 | 0.348 |
| Q_barycenter_6 | barycenter | 1 | 6 | 0.970 | 0.992 | 0.491 | 0.972 | 0.986 | 0.484 |
| Q_barycenter_8 | barycenter | 1 | 8 | 0.870 | 0.942 | 0.205 | 0.921 | 0.974 | 0.288 |
| Q_discriminator_1 | discriminator | 0 | 1 | 0.986 | 0.990 | 0.493 | 0.903 | 0.949 | 0.343 |
| Q_discriminator_3 | discriminator | 0 | 3 | 0.953 | 0.978 | 0.308 | 0.947 | 0.961 | 0.362 |
| Q_discriminator_5 | discriminator | 0 | 5 | 0.958 | 0.975 | 0.381 | 0.899 | 0.943 | 0.350 |
| Q_discriminator_7 | discriminator | 0 | 7 | 0.920 | 0.976 | 0.330 | 0.938 | 0.971 | 0.311 |
| Q_discriminator_9 | discriminator | 0 | 9 | 0.956 | 0.966 | 0.376 | 0.933 | 0.965 | 0.325 |
| Q_discriminator_0 | discriminator | 1 | 0 | 0.848 | 0.958 | 0.334 | 0.908 | 0.978 | 0.362 |
| Q_discriminator_2 | discriminator | 1 | 2 | 0.931 | 0.969 | 0.352 | 0.972 | 0.986 | 0.457 |
| Q_discriminator_4 | discriminator | 1 | 4 | 0.936 | 0.983 | 0.412 | 0.938 | 0.967 | 0.369 |
| Q_discriminator_6 | discriminator | 1 | 6 | 0.977 | 0.988 | 0.398 | 0.990 | 0.994 | 0.493 |
| Q_discriminator_8 | discriminator | 1 | 8 | 0.935 | 0.977 | 0.404 | 0.914 | 0.970 | 0.435 |
| Q_mmd_1 | mmd | 0 | 1 | 0.998 | 0.999 | 0.809 | 0.999 | 0.999 | 0.826 |
| Q_mmd_3 | mmd | 0 | 3 | 0.997 | 0.998 | 0.752 | 0.996 | 0.998 | 0.721 |
| Q_mmd_5 | mmd | 0 | 5 | 0.980 | 0.994 | 0.650 | 0.982 | 0.995 | 0.662 |
| Q_mmd_7 | mmd | 0 | 7 | 0.995 | 0.998 | 0.755 | 0.995 | 0.998 | 0.751 |
| Q_mmd_9 | mmd | 0 | 9 | 0.996 | 0.998 | 0.785 | 0.995 | 0.997 | 0.711 |
| Q_mmd_0 | mmd | 1 | 0 | 0.994 | 0.998 | 0.719 | 0.995 | 0.997 | 0.711 |
| Q_mmd_2 | mmd | 1 | 2 | 0.994 | 0.998 | 0.705 | 0.996 | 0.998 | 0.707 |
| Q_mmd_4 | mmd | 1 | 4 | 0.992 | 0.998 | 0.851 | 0.958 | 0.996 | 0.680 |
| Q_mmd_6 | mmd | 1 | 6 | 0.993 | 0.997 | 0.746 | 0.993 | 0.998 | 0.764 |
| Q_mmd_8 | mmd | 1 | 8 | 0.991 | 0.997 | 0.666 | 0.976 | 0.993 | 0.595 |
| Q_none_0 | none | 1 | 0 | 0.995 | 0.998 | 0.762 | 0.996 | 0.998 | 0.762 |
| Q_none_1 | none | 1 | 1 | 0.993 | 0.998 | 0.693 | 0.995 | 0.998 | 0.721 |
| Q_pooled_1 | pooled | 0 | 1 | 0.986 | 0.994 | 0.604 | 0.985 | 0.994 | 0.603 |
| Q_pooled_3 | pooled | 0 | 3 | 0.975 | 0.993 | 0.597 | 0.971 | 0.988 | 0.576 |
| Q_pooled_5 | pooled | 0 | 5 | 0.986 | 0.995 | 0.643 | 0.981 | 0.993 | 0.633 |
| Q_pooled_7 | pooled | 0 | 7 | 0.929 | 0.983 | 0.471 | 0.969 | 0.986 | 0.509 |
| Q_pooled_9 | pooled | 0 | 9 | 0.980 | 0.991 | 0.591 | 0.988 | 0.992 | 0.599 |
| Q_pooled_0 | pooled | 1 | 0 | 0.974 | 0.990 | 0.568 | 0.975 | 0.989 | 0.545 |
| Q_pooled_2 | pooled | 1 | 2 | 0.976 | 0.985 | 0.504 | 0.961 | 0.988 | 0.497 |
| Q_pooled_4 | pooled | 1 | 4 | 0.930 | 0.983 | 0.507 | 0.955 | 0.980 | 0.505 |
| Q_pooled_6 | pooled | 1 | 6 | 0.956 | 0.992 | 0.558 | 0.969 | 0.990 | 0.535 |
| Q_pooled_8 | pooled | 1 | 8 | 0.964 | 0.982 | 0.522 | 0.976 | 0.987 | 0.541 |
| Q_reference_1 | reference | 0 | 1 | 0.981 | 0.995 | 0.598 | 0.978 | 0.994 | 0.572 |
| Q_reference_3 | reference | 0 | 3 | 0.972 | 0.993 | 0.587 | 0.977 | 0.993 | 0.582 |
| Q_reference_5 | reference | 0 | 5 | 0.974 | 0.994 | 0.598 | 0.982 | 0.995 | 0.641 |
| Q_reference_7 | reference | 0 | 7 | 0.960 | 0.990 | 0.524 | 0.960 | 0.989 | 0.535 |
| Q_reference_9 | reference | 0 | 9 | 0.971 | 0.990 | 0.562 | 0.986 | 0.991 | 0.578 |
| Q_reference_0 | reference | 1 | 0 | 0.979 | 0.992 | 0.581 | 0.965 | 0.988 | 0.555 |
| Q_reference_2 | reference | 1 | 2 | 0.971 | 0.983 | 0.495 | 0.962 | 0.989 | 0.508 |
| Q_reference_4 | reference | 1 | 4 | 0.979 | 0.987 | 0.509 | 0.963 | 0.988 | 0.523 |
| Q_reference_6 | reference | 1 | 6 | 0.971 | 0.986 | 0.540 | 0.955 | 0.989 | 0.520 |
| Q_reference_8 | reference | 1 | 8 | 0.950 | 0.988 | 0.518 | 0.896 | 0.977 | 0.449 |
| Q_scvi_adv_0 | scvi_adv | 1 | 0 | 0.994 | 0.997 | 0.688 | 0.997 | 0.998 | 0.761 |
| Q_scvi_adv_1 | scvi_adv | 1 | 1 | 0.994 | 0.998 | 0.738 | 0.998 | 0.999 | 0.782 |
| Q_sinkhorn_1 | sinkhorn | 0 | 1 | 0.847 | 0.978 | 0.457 | 0.933 | 0.979 | 0.417 |
| Q_sinkhorn_3 | sinkhorn | 0 | 3 | 0.818 | 0.925 | 0.408 | 0.833 | 0.926 | 0.396 |
| Q_sinkhorn_5 | sinkhorn | 0 | 5 | 0.935 | 0.973 | 0.485 | 0.948 | 0.973 | 0.499 |
| Q_sinkhorn_7 | sinkhorn | 0 | 7 | 0.513 | 0.785 | 0.332 | 0.659 | 0.919 | 0.385 |
| Q_sinkhorn_9 | sinkhorn | 0 | 9 | 0.920 | 0.951 | 0.416 | 0.930 | 0.949 | 0.411 |
| Q_sinkhorn_0 | sinkhorn | 1 | 0 | 0.898 | 0.968 | 0.457 | 0.894 | 0.948 | 0.376 |
| Q_sinkhorn_2 | sinkhorn | 1 | 2 | 0.930 | 0.985 | 0.493 | 0.926 | 0.985 | 0.448 |
| Q_sinkhorn_4 | sinkhorn | 1 | 4 | 0.930 | 0.952 | 0.398 | 0.897 | 0.955 | 0.369 |
| Q_sinkhorn_6 | sinkhorn | 1 | 6 | 0.832 | 0.980 | 0.458 | 0.873 | 0.970 | 0.430 |
| Q_sinkhorn_8 | sinkhorn | 1 | 8 | 0.811 | 0.965 | 0.386 | 0.931 | 0.982 | 0.448 |

Per arm (r_min = worst dimension over the arm's configs; knn = mean overlap):

| arm | r_min A100 | knn A100 | r_min L40S | knn L40S |
|---|---|---|---|---|
| barycenter | 0.765 | 0.348 | 0.515 | 0.363 |
| discriminator | 0.848 | 0.379 | 0.899 | 0.381 |
| mmd | 0.980 | 0.744 | 0.958 | 0.713 |
| none | 0.993 | 0.727 | 0.995 | 0.741 |
| pooled | 0.929 | 0.556 | 0.955 | 0.554 |
| reference | 0.950 | 0.551 | 0.896 | 0.546 |
| scvi_adv | 0.994 | 0.713 | 0.997 | 0.772 |
| sinkhorn | 0.513 | 0.429 | 0.658 | 0.418 |

A100 vs L40S latents (both JHPCE, same torch build): per-dim r median 0.9862 (min 0.523); 15-NN overlap mean 0.514 (min 0.244); bitwise 0/64.

### G3 score differences

All 192 latents (64 local, 64 per JHPCE model) scored in the local wcd-kbet env with the CURRENT scorer (scripts/score_scib_native.py at main 878d8ea, blob f1b92b3f83: trajectory in BIO_METRICS, kBET seeded with set.seed(0)), 10 processes at a time, one thread each; started 2026-10-03T11:09:19-04:00, finished 2026-10-03T16:59:24-04:00. Raw batch mean = mean of PCR_batch, ASW_label/batch, iLISI, graph_conn, kBET; raw bio mean = mean of NMI_cluster/label, ARI_cluster/label, ASW_label, isolated_label_F1, isolated_label_silhouette, cLISI, cell_cycle_conservation, trajectory. Flag = |mean paired diff| > 2 SE and > 0.5 x local seed SD.

#### A100 - local

| arm | cond | n_seeds | batch_mean_mean_diff | batch_mean_se | batch_mean_local_seed_sd | batch_mean_abs_diff_over_se | batch_mean_flag | bio_mean_mean_diff | bio_mean_se | bio_mean_local_seed_sd | bio_mean_abs_diff_over_se | bio_mean_flag |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| barycenter | 0 | 5 | 0.0002617 | 0.005709 | 0.009999 | 0.04584 | False | 0.001739 | 0.006711 | 0.02955 | 0.2591 | False |
| barycenter | 1 | 5 | -0.002314 | 0.001021 | 0.0131 | 2.266 | False | 0.007727 | 0.01225 | 0.04543 | 0.6308 | False |
| discriminator | 0 | 5 | 0.005081 | 0.00867 | 0.02724 | 0.586 | False | -0.006896 | 0.005121 | 0.03157 | 1.347 | False |
| discriminator | 1 | 5 | 0.0131 | 0.006579 | 0.02928 | 1.991 | False | -0.00304 | 0.005466 | 0.01876 | 0.5562 | False |
| mmd | 0 | 5 | 0.001891 | 0.001497 | 0.001622 | 1.264 | False | 0.003657 | 0.004239 | 0.008699 | 0.8626 | False |
| mmd | 1 | 5 | 0.0009809 | 0.002534 | 0.006881 | 0.3871 | False | -0.00241 | 0.001576 | 0.009724 | 1.529 | False |
| none | 1 | 2 | 0.003092 | 0.0001964 | 0.0007654 | 15.75 | True | -0.01171 | 0.01096 | 0.0171 | 1.069 | False |
| pooled | 0 | 5 | -0.0003083 | 0.003142 | 0.01011 | 0.09815 | False | 0.00302 | 0.005818 | 0.01644 | 0.5191 | False |
| pooled | 1 | 5 | -0.0008123 | 0.0005852 | 0.003591 | 1.388 | False | -0.003827 | 0.001702 | 0.005605 | 2.249 | True |
| reference | 0 | 5 | -0.001405 | 0.001827 | 0.00538 | 0.7688 | False | -0.0003649 | 0.004207 | 0.01352 | 0.08674 | False |
| reference | 1 | 5 | -0.0005553 | 0.001714 | 0.003427 | 0.324 | False | 0.004422 | 0.002493 | 0.006224 | 1.774 | False |
| scvi_adv | 1 | 2 | -0.00301 | 0.002779 | 0.009621 | 1.083 | False | 0.006717 | 0.009092 | 0.02049 | 0.7388 | False |
| sinkhorn | 0 | 5 | 0.001386 | 0.002208 | 0.00221 | 0.6275 | False | -0.006059 | 0.005853 | 0.00752 | 1.035 | False |
| sinkhorn | 1 | 5 | -0.001199 | 0.001324 | 0.0008347 | 0.9054 | False | 0.004854 | 0.005075 | 0.01144 | 0.9565 | False |

Calibration of the flag rule (sign flips of the paired differences, i.e. no systematic host effect): 2.0 flags expected over 28 tests; P(>= 2 flags) = 0.6253. Groups with only 2 seeds (none/1, scvi_adv/1) estimate the SE from two differences, so each of their tests flags with probability up to 0.5 under that null. Largest single-metric paired difference: trajectory on Q_barycenter_8 (0.498 local vs 0.874 A100); local seed SD of trajectory within that (arm, cond): 0.162.

Per metric over the 64 paired configs:

| metric | mean_diff | mean_abs_diff | max_abs_diff | local_sd_over_all_latents |
|---|---|---|---|---|
| PCR_batch | 0.006627 | 0.01473 | 0.1197 | 0.1733 |
| ASW_label/batch | 0.0004634 | 0.00311 | 0.01479 | 0.02496 |
| iLISI | -0.00125 | 0.003238 | 0.01085 | 0.04051 |
| graph_conn | 0.002572 | 0.01283 | 0.05722 | 0.02619 |
| kBET | -0.002107 | 0.005797 | 0.02097 | 0.0237 |
| NMI_cluster/label | -0.002863 | 0.01058 | 0.03823 | 0.04356 |
| ARI_cluster/label | 0.0004911 | 0.03852 | 0.1611 | 0.09933 |
| ASW_label | 0.0003243 | 0.003067 | 0.02007 | 0.03561 |
| isolated_label_F1 | -0.003749 | 0.0207 | 0.08416 | 0.03925 |
| isolated_label_silhouette | 0.001347 | 0.004006 | 0.049 | 0.04202 |
| cLISI | 2.126e-05 | 0.0002906 | 0.001557 | 0.001737 |
| cell_cycle_conservation | -0.001936 | 0.01868 | 0.1205 | 0.09772 |
| trajectory | 0.006879 | 0.01402 | 0.3767 | 0.07316 |
| batch_mean | 0.001261 | 0.005383 | 0.03349 | 0.04414 |
| bio_mean | 6.447e-05 | 0.008796 | 0.05554 | 0.03671 |

#### L40S - local

| arm | cond | n_seeds | batch_mean_mean_diff | batch_mean_se | batch_mean_local_seed_sd | batch_mean_abs_diff_over_se | batch_mean_flag | bio_mean_mean_diff | bio_mean_se | bio_mean_local_seed_sd | bio_mean_abs_diff_over_se | bio_mean_flag |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| barycenter | 0 | 5 | -0.001069 | 0.004781 | 0.009999 | 0.2236 | False | -0.006584 | 0.006265 | 0.02955 | 1.051 | False |
| barycenter | 1 | 5 | -0.003184 | 0.002079 | 0.0131 | 1.532 | False | 0.01301 | 0.01433 | 0.04543 | 0.908 | False |
| discriminator | 0 | 5 | 0.003766 | 0.005784 | 0.02724 | 0.6511 | False | 0.002773 | 0.009408 | 0.03157 | 0.2948 | False |
| discriminator | 1 | 5 | 0.007832 | 0.004412 | 0.02928 | 1.775 | False | -0.005494 | 0.003713 | 0.01876 | 1.48 | False |
| mmd | 0 | 5 | 0.002877 | 0.001696 | 0.001622 | 1.696 | False | 0.006448 | 0.004384 | 0.008699 | 1.471 | False |
| mmd | 1 | 5 | -0.00359 | 0.002077 | 0.006881 | 1.729 | False | -0.005553 | 0.00216 | 0.009724 | 2.571 | True |
| none | 1 | 2 | 0.0008343 | 0.001685 | 0.0007654 | 0.4952 | False | -0.009505 | 0.004517 | 0.0171 | 2.104 | True |
| pooled | 0 | 5 | 0.001673 | 0.001695 | 0.01011 | 0.9874 | False | -0.005789 | 0.004518 | 0.01644 | 1.281 | False |
| pooled | 1 | 5 | -0.0004883 | 0.000722 | 0.003591 | 0.6762 | False | -0.001572 | 0.00321 | 0.005605 | 0.4898 | False |
| reference | 0 | 5 | 0.001093 | 0.001219 | 0.00538 | 0.8967 | False | -0.001564 | 0.003843 | 0.01352 | 0.4069 | False |
| reference | 1 | 5 | -0.0005594 | 0.0008776 | 0.003427 | 0.6374 | False | 0.005297 | 0.002903 | 0.006224 | 1.825 | False |
| scvi_adv | 1 | 2 | -0.003655 | 0.001847 | 0.009621 | 1.979 | False | -0.005003 | 0.01551 | 0.02049 | 0.3226 | False |
| sinkhorn | 0 | 5 | 0.004055 | 0.002075 | 0.00221 | 1.955 | False | -0.001538 | 0.003039 | 0.00752 | 0.5062 | False |
| sinkhorn | 1 | 5 | -0.0009382 | 0.0009658 | 0.0008347 | 0.9714 | False | -0.0001764 | 0.004608 | 0.01144 | 0.03829 | False |

Calibration of the flag rule (sign flips of the paired differences, i.e. no systematic host effect): 2.875 flags expected over 28 tests; P(>= 2 flags) = 0.8407. Groups with only 2 seeds (none/1, scvi_adv/1) estimate the SE from two differences, so each of their tests flags with probability up to 0.5 under that null. Largest single-metric paired difference: trajectory on Q_barycenter_8 (0.498 local vs 0.887 L40S); local seed SD of trajectory within that (arm, cond): 0.162.

Per metric over the 64 paired configs:

| metric | mean_diff | mean_abs_diff | max_abs_diff | local_sd_over_all_latents |
|---|---|---|---|---|
| PCR_batch | 0.0039 | 0.01276 | 0.09334 | 0.1733 |
| ASW_label/batch | 0.0002618 | 0.002734 | 0.0117 | 0.02496 |
| iLISI | -0.0006457 | 0.002973 | 0.01272 | 0.04051 |
| graph_conn | 0.001635 | 0.0114 | 0.05232 | 0.02619 |
| kBET | -0.001112 | 0.005843 | 0.02008 | 0.0237 |
| NMI_cluster/label | -0.00373 | 0.009937 | 0.04154 | 0.04356 |
| ARI_cluster/label | -0.008224 | 0.03482 | 0.162 | 0.09933 |
| ASW_label | -0.0002793 | 0.003339 | 0.01619 | 0.03561 |
| isolated_label_F1 | 0.0008384 | 0.02244 | 0.1002 | 0.03925 |
| isolated_label_silhouette | 0.0002204 | 0.004827 | 0.03105 | 0.04202 |
| cLISI | -7.398e-05 | 0.000327 | 0.001091 | 0.001737 |
| cell_cycle_conservation | -0.003292 | 0.02967 | 0.1593 | 0.09772 |
| trajectory | 0.01045 | 0.01535 | 0.389 | 0.07316 |
| batch_mean | 0.0008077 | 0.004535 | 0.02047 | 0.04414 |
| bio_mean | -0.0005113 | 0.008713 | 0.06979 | 0.03671 |

## G4 scoring equivalence (unchanged since the provisional report)

Per metric. G4 = the 2 latents (Q_none_0, Q_barycenter_1) and the prepped file, md5-identical on both hosts, scored locally (wcd-kbet) and on JHPCE (wcd-score, compute-174 = AMD EPYC 9555, AVX-512); tolerance 1e-6 absolute. G4b = the same on JHPCE against the JHPCE-regenerated prepped file. local repeat = the same latents re-scored on the local machine (inside the G3 run); JHPCE repeat = a second job on compute-174; numba generic = both hosts with NUMBA_CPU_NAME=generic. The host verdict uses the deterministic metrics only: kBET is unseeded in scib 1.1.7 (R kBET samples test cells) and changes between runs on one machine (local repeat column; the lead measured up to 0.0047), so its cross-host difference is not a host difference; a fixed R seed is being added to score_scib_native.py on prereg-v2. Diagnosis of the deterministic metrics: (i) the scorer's kNN graph (pynndescent, numba) depends on the CPU vector width: AVX-512 EPYC vs local AVX2 differ in 251 of 33,506 rows, which moves ARI, NMI, isolated-label F1, graph connectivity and trajectory; an AVX-only Opteron (compute-053) builds the same graph as local; with NUMBA_CPU_NAME=generic the graph is identical on all three CPU types and those metrics agree to <=1e-16; (ii) graph iLISI / cLISI still differ across hosts AND between two runs on compute-174 (identical graph; ARI/NMI unchanged), but are identical between local runs (<=1.4e-17). The cause is not scib's LISI subsampling RNG (knn_graph.cpp draws a random number per row but skips nothing at subsample=100); unresolved. PCR and cell-cycle conservation agree to <=5.7e-7. G4 was measured with the scorer before the kBET seed was added (main 8cda6de) and was not re-run: every benchmark latent is scored locally.

| metric | max diff default | status default | max diff numba generic | status numba generic |
|---|---|---|---|---|
| ARI_cluster/label | 0.01831 | exceeds tol | 0 | ok |
| ASW_label | 0 | ok | 0 | ok |
| ASW_label/batch | 0 | ok | 0 | ok |
| NMI_cluster/label | 0.002678 | exceeds tol | 1.11e-16 | ok |
| PCR_batch | 1.277e-07 | ok | 1.277e-07 | ok |
| cLISI | 0.000167 | exceeds tol | 0.0001549 | exceeds tol |
| cell_cycle_conservation | 5.686e-07 | ok | 5.686e-07 | ok |
| graph_conn | 0.0002894 | exceeds tol | 0 | ok |
| hvg_overlap | nan | NaN on both (not computed) | nan | NaN on both (not computed) |
| iLISI | 0.002805 | exceeds tol | 0.00196 | exceeds tol |
| isolated_label_F1 | 0.02188 | exceeds tol | 0 | ok |
| isolated_label_silhouette | 0 | ok | 0 | ok |
| kBET | 0.002518 | exceeds tol | 0.003577 | exceeds tol |
| trajectory | 3.728e-06 | exceeds tol | 0 | ok |

## G5 throughput

Same 64-fit 8-lane queue (immune, 3 epochs, queue order of the local calibration), 1 OMP/NUMBA thread per lane, 10 CPUs.

| host | node | makespan_s | window_s | factor_window | factor_makespan | per_gpu_ratio | single_lane | job_cpu_time | job_elapsed |
|---|---|---|---|---|---|---|---|---|---|
| JHPCE A100 | compute-128 | 599.2 | 532.8 | 2.7 | 2.63 | 0.716 | [fit] S_immune_barycenter_iter5 41s z(33506, 10) finite=True; [fit] S_immune_barycenter_iter5_e1 21s z(33506, 10) finite=True | 01:15:01 | 00:12:36 |
| JHPCE L40S | compute-171 | 353.7 | 314.6 | 4.62 | 4.46 | 1.225 | [fit] S_immune_barycenter_iter5 30s z(33506, 10) finite=True; [fit] S_immune_barycenter_iter5_e1 12s z(33506, 10) finite=True | 45:31.183 | 00:07:36 |
| local RTX 3080 | local | 433.9 | 386.6 | 3.77 | 3.64 | 1 |  |  |  |

## Task split

Model (scripts/gate_task_split.py): whole tasks per host (SI-17), pilot tasks forced local (SI-30); stage-gated S1 A1 -> S2 A2/A3 -> S3 X1 adversarial -> S4 X3/X6/X7/X8/X12, fillers (X1 none/scvi_adv, X13) in GPU gaps. ALL latents are scored on the local CPUs: 4 scoring processes while the local GPU fits (8 lanes), 12 otherwise, each at the docs/scoring_time.csv single-thread rate times the efficiency measured in the 192-latent run; JHPCE latents join the local backlog of their stage. JHPCE fit rate = G x that model's measured 8-lane factor; one queue wait per 3-day JHPCE job (base case: the wait observed for that model's gate job). Rule-freeze time and copying latents back are not modelled (re-harvest of the two 109 MB latent tars plus reports took 44 s).

Design of record: manifest_dor.tsv (md5 cabda46269f31913d8deb2cbea8ab729): 6718 fits, 2375.1 local lane-hours, 1627.9 scoring hours (31.71 ms per cell per process after the measured scoring efficiency 0.512); fits per experiment {'A1': 1098, 'A2': 108, 'A3': 180, 'X1': 3000, 'X12': 288, 'X13': 280, 'X3': 828, 'X6': 216, 'X7': 216, 'X8': 504}; per task {'atac_large': 410, 'atac_small': 1331, 'immune': 1088, 'immune_hum_mou': 785, 'lung': 578, 'pancreas': 893, 'sim1': 848, 'sim2': 785}. Forced local (SI-30): atac_small, immune, sim1. All-local makespan 26.25 d.

### A100 (per-GPU ratio 0.716; base queue wait 9.41 h per 3-day job)

| jhpce_gpus | wall_days | jhpce | local | local_gpu_busy_days | jhpce_busy_days_per_gpu | jhpce_gpu_days | jhpce_job_starts | stage_end_days |
|---|---|---|---|---|---|---|---|---|
| 1 | 18.85 | atac_large pancreas sim2 | atac_small immune immune_hum_mou lung sim1 | 17.68 | 11.97 | 11.97 | 6 | S1 4.1 / S2 1.07 / S3 8.98 / S4 4.69 |
| 2 | 15.93 | atac_large immune_hum_mou lung pancreas | atac_small immune sim1 sim2 | 14.71 | 8.26 | 16.51 | 8 | S1 4.1 / S2 1.07 / S3 7.8 / S4 2.96 |
| 3 | 13.94 | atac_large immune_hum_mou lung pancreas sim2 | atac_small immune sim1 | 11.29 | 6.96 | 20.89 | 12 | S1 4.1 / S2 1.07 / S3 6.17 / S4 2.59 |
| 4 | 13.8 | atac_large immune_hum_mou lung pancreas sim2 | atac_small immune sim1 | 11.48 | 5.22 | 20.89 | 16 | S1 4.1 / S2 1.07 / S3 6.37 / S4 2.25 |

Makespan (days) by JHPCE queue wait per 3-day GPU job; SI-30 = pilot tasks forced local, free = unconstrained:

| constraint | gpus | wait 0.0 d | wait 0.392 d | wait 0.5 d | wait 1.0 d | wait 2.0 d |
|---|---|---|---|---|---|---|
| SI-30 | 1 | 17.80 | 18.85 | 19.07 | 20.27 | 20.38 |
| SI-30 | 2 | 15.33 | 15.93 | 15.93 | 16.36 | 17.91 |
| SI-30 | 3 | 13.68 | 13.94 | 14.07 | 14.96 | 16.70 |
| SI-30 | 4 | 13.68 | 13.80 | 13.84 | 14.11 | 16.27 |
| free | 1 | 16.38 | 17.72 | 17.79 | 18.79 | 19.99 |
| free | 2 | 12.21 | 13.04 | 13.19 | 14.70 | 16.83 |
| free | 3 | 9.70 | 11.12 | 11.42 | 12.89 | 15.64 |
| free | 4 | 8.24 | 9.81 | 10.16 | 11.82 | 14.55 |

X3 / X13 counts still change on missing-arms-v2: base-case split re-evaluated with both counts halved or +50%:

| gpus | scenario | fits | base_split_wall_days | scenario_optimum_wall_days | scenario_optimum_jhpce |
|---|---|---|---|---|---|
| 1 | x3_x13_half | 6164 | 17.46 | 17.46 | atac_large pancreas sim2 |
| 1 | x3_x13_plus50pct | 7272 | 20.02 | 20.02 | atac_large pancreas sim2 |
| 2 | x3_x13_half | 6164 | 15.08 | 15.08 | atac_large immune_hum_mou lung pancreas |
| 2 | x3_x13_plus50pct | 7272 | 16.7 | 16.7 | atac_large immune_hum_mou lung pancreas |
| 3 | x3_x13_half | 6164 | 13.32 | 13.32 | atac_large immune_hum_mou lung pancreas sim2 |
| 3 | x3_x13_plus50pct | 7272 | 14.56 | 14.56 | atac_large immune_hum_mou lung pancreas sim2 |
| 4 | x3_x13_half | 6164 | 13.33 | 13.33 | atac_large immune_hum_mou lung pancreas sim2 |
| 4 | x3_x13_plus50pct | 7272 | 14.27 | 14.27 | atac_large immune_hum_mou lung pancreas sim2 |

### L40S (per-GPU ratio 1.225; base queue wait 7.82 h per 3-day job)

| jhpce_gpus | wall_days | jhpce | local | local_gpu_busy_days | jhpce_busy_days_per_gpu | jhpce_gpu_days | jhpce_job_starts | stage_end_days |
|---|---|---|---|---|---|---|---|---|
| 1 | 15.99 | atac_large immune_hum_mou lung pancreas | atac_small immune sim1 sim2 | 14.42 | 9.65 | 9.65 | 5 | S1 4.1 / S2 1.07 / S3 7.49 / S4 3.32 |
| 2 | 13.77 | atac_large immune_hum_mou lung pancreas sim2 | atac_small immune sim1 | 11.46 | 6.1 | 12.21 | 8 | S1 4.1 / S2 1.07 / S3 6.34 / S4 2.26 |
| 3 | 13.78 | atac_large immune_hum_mou lung pancreas sim2 | atac_small immune sim1 | 11.48 | 4.07 | 12.21 | 9 | S1 4.1 / S2 1.07 / S3 6.36 / S4 2.24 |
| 4 | 13.78 | atac_large immune_hum_mou lung pancreas sim2 | atac_small immune sim1 | 11.48 | 3.05 | 12.21 | 12 | S1 4.1 / S2 1.07 / S3 6.36 / S4 2.24 |

Makespan (days) by JHPCE queue wait per 3-day GPU job; SI-30 = pilot tasks forced local, free = unconstrained:

| constraint | gpus | wait 0.0 d | wait 0.326 d | wait 0.5 d | wait 1.0 d | wait 2.0 d |
|---|---|---|---|---|---|---|
| SI-30 | 1 | 15.80 | 15.99 | 16.52 | 17.77 | 18.45 |
| SI-30 | 2 | 13.68 | 13.77 | 13.80 | 14.39 | 16.49 |
| SI-30 | 3 | 13.68 | 13.78 | 13.84 | 13.99 | 14.93 |
| SI-30 | 4 | 13.68 | 13.78 | 13.84 | 13.99 | 14.61 |
| free | 1 | 13.06 | 13.85 | 14.31 | 15.87 | 17.72 |
| free | 2 | 9.32 | 10.15 | 10.64 | 12.11 | 15.13 |
| free | 3 | 7.14 | 7.95 | 8.71 | 10.94 | 13.07 |
| free | 4 | 5.82 | 7.12 | 7.82 | 9.22 | 12.74 |

X3 / X13 counts still change on missing-arms-v2: base-case split re-evaluated with both counts halved or +50%:

| gpus | scenario | fits | base_split_wall_days | scenario_optimum_wall_days | scenario_optimum_jhpce |
|---|---|---|---|---|---|
| 1 | x3_x13_half | 6164 | 15.27 | 15.27 | atac_large immune_hum_mou lung pancreas |
| 1 | x3_x13_plus50pct | 7272 | 17.03 | 17.03 | atac_large immune_hum_mou lung pancreas |
| 2 | x3_x13_half | 6164 | 13.31 | 13.31 | atac_large immune_hum_mou lung pancreas sim2 |
| 2 | x3_x13_plus50pct | 7272 | 14.2 | 14.2 | atac_large immune_hum_mou lung pancreas sim2 |
| 3 | x3_x13_half | 6164 | 13.31 | 13.31 | atac_large immune_hum_mou lung pancreas sim2 |
| 3 | x3_x13_plus50pct | 7272 | 14.25 | 14.25 | atac_large immune_hum_mou lung pancreas sim2 |
| 4 | x3_x13_half | 6164 | 13.31 | 13.31 | atac_large immune_hum_mou lung pancreas sim2 |
| 4 | x3_x13_plus50pct | 7272 | 14.25 | 14.25 | atac_large immune_hum_mou lung pancreas sim2 |

Sensitivity to the local scoring efficiency (base case, SI-30, observed queue wait):

| model | score_efficiency | G1 days | G2 days | G3 days | G4 days | all_local_days |
|---|---|---|---|---|---|---|
| A100 | 0.4 | 19.57 | 17.15 | 14.97 | 15 | 26.25 |
| A100 | 0.512 | 18.85 | 15.93 | 13.94 | 13.8 | 26.25 |
| A100 | 0.6 | 18.74 | 15.37 | 13.45 | 13.17 | 26.25 |
| L40S | 0.4 | 17.09 | 14.99 | 14.99 | 14.99 | 26.25 |
| L40S | 0.512 | 15.99 | 13.77 | 13.78 | 13.78 | 26.25 |
| L40S | 0.6 | 15.85 | 13.1 | 13.15 | 13.15 | 26.25 |

## Caveats

- G3 and G5 use one task (immune, 3 epochs, 64 configs); per-GPU ratios may differ for larger tasks or full-length fits.
- Queue waits come from one 2-h, 1-GPU, 10-CPU job per model, submitted on a Friday evening; multi-day jobs may wait longer (grid: 0.5-2 d per job).
- Scoring efficiency was measured with 10 concurrent processes while the local GPU was idle; with 8 fit lanes running, the 4 scoring processes may be slower (efficiency 0.40 / 0.60 table).
- X3 and X13 counts still change on missing-arms-v2 (robustness tables: both counts halved or +50%).
- The gate jobs ran jhpce-setup 6675f8c (A100) and 2f29004 (L40S), whose src/ is identical; main 878d8ea adds the missing-arm code (adversarial.py, alignment.py, critic.py changed; discriminator_losses.py, sampling.py new), which no gate job ran.
- No cross-host latent is bitwise identical to its local counterpart, and A100 and L40S differ from each other as much as from local: every fit of a task (and any refit) must stay on one GPU model (SI-17).
- trajectory (now in BIO_METRICS) is the most host-sensitive metric for barycenter latents, whose local seed SD of trajectory is large (largest paired difference in the G3 sections).
- Makespan differences below about 0.05 d (e.g. L40S with 2 vs 3 GPUs) are within the model's resolution.
- fastscratch is purged after 30 days: the harvested latents are saved as artifacts and both envs are rebuildable from the recipes in the JHPCE host notes.
- experiment-preflight skills are not available in this profile; the earlier docs/jhpce/PREFLIGHT_GATE.md stands in for them.

## Addendum (lead, 2026-10-03): split on the tagged design of record

The split above was computed on the 6,718-fit design (main 878d8ea). Recomputed on the tagged manifest (prereg-tier12-v1,
6,846 rows; X13 CPU baselines carry no GPU time, their latents' scoring is included) with the same tool and inputs
(`split/final_6846/`, commands in `split/final_6846/COMMANDS.md`):

| scenario (score efficiency 0.512, observed queue wait) | wall days |
|---|---|
| all local | 26.27 |
| SI-30, L40S x 1 / x 2 / x 3 / x 4 | 16.34 / 13.77 / 13.80 / 13.80 |
| SI-30, A100 x 1 / x 2 / x 3 / x 4 | 18.83 / 15.94 / 13.95 / 13.82 |
| SI-30, L40S x 2, score efficiency 0.40 / 0.60 | 15.01 / 13.11 |
| no SI-30, L40S x 2 / x 3 | 10.20 / 8.09 |

Task assignment for SI-30 + L40S x 2 is unchanged: JHPCE atac_large, immune_hum_mou, lung, pancreas, sim2; local
atac_small, immune, sim1. A1 started locally on 2026-10-03, so most of SI-30's cost is already committed: keeping A1-A3
local and placing X1 onward freely gives about 13.16 d with L40S x 2 and 11.22 d with x 3 (additive approximation: local
A1 4.10 d + A2/A3 1.07 d + the free model's X1 and follow-up stages).
