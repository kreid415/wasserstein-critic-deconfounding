# Why graph iLISI and cLISI vary run to run on JHPCE but repeat on the workstation (2026-10-28)

Investigation only. No production scoring was run and no score rule was changed. Branch `lisi-jhpce-investigation` (off `final-review-fixes`), folder `experiments/lisi_jhpce_investigation_20261028/`.
Terms: iLISI / cLISI = graph integration / cell-type Local Inverse Simpson's Index (scib 1.1.7); RDRAND = the CPU instruction that returns a hardware random number; "helper" = scib's compiled `knn_graph.o`, which scib starts as a separate program for each LISI call; "preload library" = a small shared library loaded before the program that replaces named functions.

## 1. Cause (established, with a test that separates it from the alternatives)

1. `scib.metrics.metrics()` calls `ilisi_graph` and `clisi_graph` with `subsample=0.5*100`. The helper therefore scores a random 50% of the cells (49.2-51.2% on JHPCE).
2. The helper chooses that half with `mt19937 gen(random_device())` (source: `scib/knn_graph/knn_graph.cpp`). `std::random_device` in libstdc++ reads the CPU's RDRAND.
3. JHPCE Zen 2 nodes (EPYC 7702) return a different random number every time, so every LISI call scores a different half of the cells. This is the jitter. iLISI differed on 19/20 latents and cLISI on 13/20 between two JHPCE runs (2026-10-23); the same 19/13 counts reappear between the new no-fix JHPCE run and the workstation.
4. The workstation CPU (Ryzen 5 3600, also Zen 2) returns the constant `0xffffffff` from both RDRAND and RDSEED (6 of 6 reads each; `results/control/rdrand_workstation.json`; the kernel hides the flag in `/proc/cpuinfo`, `cpuid` still reports it). The seed is therefore always the same, so the workstation scores the same half every time, and the same half for iLISI and cLISI. This is the workstation's "repeatability": it depends on a hardware fault, not on any scoring setting.

Alternatives excluded by measurement (all 20 latents, every run):
- The kNN graph (pynndescent/numba): the SHA-256 of the connectivities is identical for every latent in all conditions on both hosts, and the graph rebuilt inside each LISI call equals the scorer's own graph. Library builds, the MKL/OpenBLAS/OpenMP runtimes, glibc/libm and the CPU model do not matter here.
- The input matrix handed to the helper: identical SHA-256 everywhere.
- Thread settings: identical (`thread_env(1)`) in every run.
- The Python LISI arithmetic (numba `Hbeta`, BLAS): once the helper keeps the same rows, JHPCE and workstation give bit-identical iLISI and cLISI on all 20 latents.
- The helper binary: identical SHA-256 (`05acd7b0...`) in `wcd-score` (JHPCE) and `wcd-kbet` (workstation).

Separating test: fix the seed on JHPCE to `0xffffffff` and compare (section 3). Control: a different fixed value (`0x12345678`) gives a different iLISI (0.5333142 vs 0.5316830 for the atac_small `none` latent), so the value matters.

## 2. Setting that makes JHPCE repeatable and equal to the workstation

`fixed_random_device.cpp` (committed) builds a 16 KB preload library that replaces only the three `std::random_device` functions the helper calls. It reads the value from `LISI_RANDOM_DEVICE_VALUE` and aborts if that variable is missing or not a 32-bit number (tested: exit 134 for missing and for `banana`, exit 0 for `0xFFFFFFFF`; Slurm 36752996).
Build (g++ 11.4 on JHPCE; the workstation has no compiler, so the binary is copied; it links libc only and cannot pull a second libstdc++ into the scorer): `g++ -O1 -fPIC -shared -nostdlib -fno-exceptions -fno-rtti -o fixed_random_device.so fixed_random_device.cpp -lc`. SHA-256 of the built file: `fda4d45779c0971626018a0e68abb405b1a1b8f5810841a6bef4da6f5d9feea3`.
Use: set `LD_PRELOAD=<path>/fixed_random_device.so` and `LISI_RANDOM_DEVICE_VALUE=0xFFFFFFFF` in the environment of the scorer. `score_array.sh` passes its environment on to the scorer unchanged, so the full-scorer check below used it without editing the repository.
It is not applied to any production script; adopting it is your decision (section 5).

## 3. Exact-equality results

LISI-only probe (`probe_lisi.py`, same scib calls as the scorer), 20 latents (6 atac_small 11,270 cells, 7 immune 33,506, 7 sim1 12,097), 1 CPU per process, `thread_env(1)`. Counts are latents (of 20) with bit-identical iLISI / cLISI / helper output files:

| comparison | iLISI / cLISI / helper |
|---|---|
| JHPCE fixed-seed run 1 vs run 2 (nodes compute-122, compute-120) | 20 / 20 / 20 |
| JHPCE fixed-seed run 1 vs workstation without preload (run 1) | 20 / 20 / 20 |
| JHPCE fixed-seed run 2 vs workstation without preload (run 2) | 20 / 20 / 20 |
| workstation run 1 vs run 2 (no preload) | 20 / 20 / 20 |
| workstation with preload vs without | 20 / 20 / 20 |
| JHPCE no-fix run (compute-119) vs workstation (negative control) | 1 / 7 / 0 |
| stored workstation production scores vs JHPCE fixed-seed run 1 and run 2 (17 latents whose stored latent hash matches) | 17 / 17 each |
| same stored scores vs JHPCE no-fix run | 0 / 6 |

Full scorer (all 19 metric columns, scorer commit a4989ff, `score_array.sh` unchanged, preload set outside it), 6 atac_small latents, two JHPCE jobs on different nodes (compute-121, compute-120):
- run A vs run B: all 19 columns bit-identical on 6/6 latents.
- run A and run B vs the stored workstation production rows: all 19 columns bit-identical on 5/5 latents whose stored latent hash matches (`results/full_scorer_comparison.txt`).
- the no-fix JHPCE run of 2026-10-23 vs the same stored rows: iLISI and cLISI differ on 5/5, the other 17 columns are identical. This is the negative control; the comparison script exits non-zero if it is not flagged.

The 3 sinkhorn latents have no usable stored score: their latents were remade after the stored score was written (stored latent hash differs from the hash of the file scored). For them the workstation re-score from this session is the baseline.

## 4. What this means for existing results

- Every workstation LISI value in the project is the score of one fixed, arbitrary half of the cells (the half seed `0xffffffff` produces), the same half for iLISI and cLISI. It is reproducible only while RDRAND on that machine stays stuck. A firmware update or another machine would silently change all LISI values by about 0.005 (iLISI) and 0.004 (cLISI).
- The earlier cross-host note (G4, 2026-10-02/03) that put an iLISI difference of 0.0028 and cLISI 1.7e-4 down to CPU SIMD width is probably in part this subsample effect. Not re-tested here.
- The score-match check of 2026-10-23 failed for this reason alone: the other 17 columns already matched.

## 5. Decision for you (not taken here)

1. Use the preload library in the JHPCE scoring job (and on the workstation, where it changes nothing today: 20/20 identical). All existing and new scores stay on the same fixed half and JHPCE scoring can stand in for workstation scoring. No score changes.
2. Change the rule: a designed seed, or LISI on all cells (`subsample=100`). This changes every LISI value and needs a stated rule change; not tested here.
Either way, add the RDRAND state of the scoring host to the score record (`rdrand_probe.py`).

## 6. Not tested / limits (also listed in the deviations)

- Non-Zen 2 JHPCE nodes (AVX-512): the CPU-dependent kNN graph is a separate known effect; the preload does not address it.
- Full-scorer equality was run on the 6 atac_small latents only; immune and sim1 were checked LISI-only.
- The preload is inherited by the whole scorer process, including embedded R (kBET); the 19 columns on 6 latents show no change. A narrower variant (setting `LD_PRELOAD` inside the scorer only for the helper child process) was not tested.
- The probe builds each AnnData from the latent's own `batch` / `celltype` arrays instead of reading the prepared h5ad; the full-scorer check uses the real files.
- JHPCE no-fix runs: one set of 20 latents plus two repeats of one latent in this session (the 2026-10-23 runs are the other evidence).

## 7. Resource use (measured with sacct)

- LISI probe jobs (4 CPUs, 6 GB): peak memory 3.2-3.5 GiB (53-59% of the request), CPU about 95-97% busy, 6.7-9.1 min per 20 latents. A 5 GB request would still leave a margin.
- Full-scorer jobs (3 CPUs, 24 GB): peak 5.6 GiB (23%), CPU about 94% busy, 9-10 min. 24 GB was oversized (carried over from the plan's 8 GB per task); three tasks need about 8 GB in total.

## 8. Jobs submitted by this session (all finished)

| Slurm id | harness id | purpose |
|---|---|---|
| 36752157 | 34a80d5a | first probe, 1 latent x2, no preload |
| 36752721 | e75582e4 | quick test: first preload build + 2 latents |
| 36752791 | be422eef | 20 latents, first preload build, run 1 |
| 36752794 | 236f66a6 | 20 latents, first preload build, run 2 |
| 36752795 | 5be8ea9c | 20 latents, no preload (no-fix run) |
| 36752849 | 3bab0bc9 | libc-only preload build + different-value control |
| 36752854 | 846e8635 | 20 latents, final preload, run 1 |
| 36752855 | 0fa1b452 | 20 latents, final preload, run 2 |
| 36752925 | d85f1587 | abort test v1 (script error) |
| 36752993 | 559b0146 | abort test v2 (same script error) |
| 36752996 | bb9d8656 | abort test v3 |
| 36752994 | 9ac7abe0 | full scorer + preload, 6 atac_small, run A |
| 36752995 | 0eaffb50 | full scorer + preload, 6 atac_small, run B |

The two abort-test jobs 36752925 and 36752993 stopped at the first command because the job wrapper runs with `set -e`: my script error, fixed in the third job. Workstation jobs (harness ids): 873cd765 (no-fix runs 1-2, 4 processes) and 48b95e80 (preload run, 3 processes). Nothing else on the account was touched; the 128 atac_large fit controllers (36749854, 36750324, 36750325) were left alone.

## 9. Notebook

NB-20261028-01 to -05 on this branch (RESULT helper subsample, RESULT RDRAND, FINDING cause, RESULT fixed-seed equality, ISSUE hardware dependence). Entry numbers restart on each branch; the pilot5-scorecheck-verdict, speed-diagnosis and chaining-board branches already use the same numbers for 2026-10-28, so the pages need renumbering when the branches are merged.

## 10. Files (all under `experiments/lisi_jhpce_investigation_20261028/`)

`probe_lisi.py`, `run_set.sh`, `rdrand_probe.py`, `fixed_random_device.cpp` (+ `.so`), `compare_probe.py`, `compare_full_scores.py`, `summarize_probe.py`, test `tests/test_lisi_investigation_compare.py`; results in `results/` (per-latent JSON, `comparison.csv`, `lisi_values_by_latent.csv`, `helper_rows_kept.csv`, `full_scorer/`, `control/`); `MANIFEST.sha256`; `README.md` with the re-run commands.
