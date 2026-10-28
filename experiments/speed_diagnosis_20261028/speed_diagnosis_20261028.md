# Speed diagnosis, 2026-10-28

Pilot data only. No main-benchmark score was read (one main-benchmark ledger was read for fit timing columns only: notebook NB-20261028-02, corrected in NB-20261028-03). All numbers below come from the CSV/JSON files in `experiments/speed_diagnosis_20261028/results/` (branch `speed-diagnosis-20261028`), produced by `scripts/analysis.py`. Notebook entries: NB-20261028-01 (plan), -04, -05, -06 (results), -07 (issue), -09, -10 (findings).

## 1. Summary

1. The atac_large slowdown is a wait **before** the timed part of each fit, not a slower fit. It starts abruptly at 17:27 on 2026-10-22 (8.3 h into job 36551149) and then hits every fit: 169 of 169 fits started afterwards waited 1.67–3.17 h (median 2.24 h); 81 of 81 fits started before waited at most 29.6 s. The cause is **not established**. The best-supported candidate is a node-level block on compute-172 early in each fit process (GPU driver start-up or the shared file system). One GPU test separates the candidates (section 2).
2. Each scoring run is single-threaded and kBET takes 40–44% of it. On the workstation, scoring throughput stops improving at 6 scorers at once (2.8 times one scorer, on a small latent). Large latents are limited by memory (18.4 GB each), not by CPU: 3 at once run at full speed.
3. Only the baseline thread setting (`thread_env(1)`) and NUMBA 2 threads reproduce the existing scores exactly; NUMBA 2 threads is not faster, so keep `thread_env(1)`. Extra BLAS/OpenMP threads change PCR_batch and cell_cycle_conservation in the 8th–9th digit and are not faster.
4. Side finding: scoring on JHPCE does not reproduce the workstation scores for iLISI and cLISI, and JHPCE scores differ between identical runs (section 5).

## 2. atac_large: where the time went

Method: the runner log lines carry no date, so the date comes from the first line of the Slurm output (2026-10-22 09:10:18) and day rollover is handled. For each fit, wall time = `[done]` line − `[fit] start` line; the timed fit ends ~4.5 s before the `[done]` line (the child writes the latent and model after timing; checked on one pooled fit: latent file written 4 s and log 0.5 s before the done line). So wait before the timed fit = wall − fit seconds − 4.5 s.

![wait before the timed fit and lane states](results/fig_atac_large_wait.png)

| Dataset (Slurm id) | fits | median wait (s) | max wait (s) | wall / fit seconds | lane use* | CPU busy | CPU s per logged fit s | sacct MaxRSS (GB) |
|---|---|---|---|---|---|---|---|---|
| immune_hum_mou (36551148) | 378 | 12 | 67 | 1.00 | 0.992 | 0.790 | 0.997 | 30.4 |
| atac_large (36551149) | 250 | 7191 | 11398 | 3.13 | 0.316 | 0.209 | 0.836 | 35.3 |
| lung (36551150) | 378 | 7 | 16 | 1.00 | 0.994 | 0.790 | 0.996 | 22.1 |
| sim2 (36551151) | 378 | 15 | 74 | 1.01 | 0.991 | 0.790 | 0.999 | 116.5 |
| pancreas (36551152) | 378 | 8 | 17 | 1.00 | 0.993 | 0.789 | 0.996 | 21.2 |

*lane use = sum of logged fit seconds / (8 lanes × time from first start to last done). The earlier figure of 46% in the project notes used another denominator; by this definition atac_large is 0.316 for the whole job.

What the evidence shows:

1. **The wait is before the fit timer.** The timed fit is the last part of the wall time; the file times of one slow fit (latent written 20:44:51, model 20:44:53, log 20:44:54, done line 20:44:55) leave only ~4 s after it. Before the timer the child only imports packages, loads the prepared file (173 MB, the smallest of the eight), subsets genes and builds the model inputs (`fit_paper_config.py` lines 285–333). The source of that file, `scvi_adversarial_plan.py`, `fit_outcome.py` and `data.py` contain no sleep, lock or wait call (grep).
2. **The wait is abrupt and permanent.** Last fast fit started 17:25:38; the first slow fit 17:29:12. After that the wait varies smoothly between 1.7 and 3.2 h with start time. It makes up 77% of lane wall time after the onset; on average 1.8 lanes are in a timed fit and 6.1 wait (fits that finished; the 8 fits killed at the time limit are not counted).
3. **Waiting lanes use no CPU.** TotalCPU / logged fit seconds is 0.836 for atac_large and 0.996–0.999 for the other four jobs (each lane uses one core in its timed fit).
4. **Not memory, not CPU oversubscription.** sacct MaxRSS is 35 GB of 96 GB; the sim2 job reached 116 GB without trouble and ran at wall/fit 1.007. The other four jobs ran 8 lanes on 10 CPUs identically and did not wait.
5. **Not the data file or the imports.** Slurm 36746301 (shared CPU node compute-116, 8 CPUs, no GPU, no fits) ran the start-up steps of `fit_paper_config.py` with `atac_large` alone and with 8 processes at once: 21.5 s alone and 20.5–21.0 s with 8 at once (lung: 16.2 s alone, 25.6–25.9 s with 8). Peak RSS 2.3 GB per process.
6. **Not other users' jobs on the node.** 10 of the 117 release times lie within 60 s of the start or end of another job on compute-172; by chance 10.6% would. (Node-level list kept out of the repo because it names other users.)
7. **Not a node-wide Slurm event.** No Slurm node events are listed for compute-172; slurmd restarted on 10-23 02:43 and the wait did not change.
8. **Release is common to a group.** Lanes started within 3 min of each other began their timed fit within ±3 s of each other (6 of 8 lanes at 22:43:04–07 on 10-22; 7 of 8 at 19:30:47–50). A shared block released all of them at once fits this; a per-process slowness does not.
9. **Other nodes were fine.** The four sibling jobs (compute-171 and compute-173) waited at most 74 s.

Candidates, best first:

| Candidate | Fits | Does not fit / untested |
|---|---|---|
| A. Block in the GPU driver or CUDA start-up on compute-172 (for example the first CUDA call of each process waiting on a node-wide lock) | zero CPU in the wait; common release of a group; one node only; onset abrupt; the CPU test shows the non-GPU start-up steps are fast | no GPU job could be run by me; the code before the timer does not obviously touch the GPU (importing scvi/torch may) |
| B. Stall of the shared file system on that node (reads of the Python environment in /users, or the fast-scratch directories) | zero CPU in the wait; one node; common release | the same files are read by the siblings; no I/O counters survive |
| C. Something in the runner | none of the evidence points here | runner starts the child and logs immediately |

**The one test that separates A from B:** one GPU job on compute-172 (and the same on another L40S node) that starts 8 processes at once and prints a time stamp after each start-up phase (`import torch`, `torch.cuda.is_available()`, first CUDA tensor, `import scvi`, `read_h5ad`), with `scripts/lane_sampler.sh` recording each process's state and kernel wait point every 60 s. About 15 minutes; needs 1 GPU. I did not run it (instruction: no GPU jobs).

Fix that cannot change fit results (proposed, not applied):

1. Run the 128 missing atac_large fits (counts by arm: sinkhorn 26, discriminator 21, barycenter 20, mmd 20, pooled 20, reference 20, none 1) in a new job on another node (`--exclude=compute-172`), same code commit 10a1ca5, same 8 lanes. Provisional estimate with no wait: 11–14 h (fit seconds taken from the 8-lane periods of the same job, 8 lanes at 99% use) instead of the 3-day limit.
2. Put a start-up check that must pass into the job before any fit: 8 simultaneous probe processes doing the start-up steps and one CUDA tensor; stop the job if any takes more than 120 s (normal: about 21 s without GPU). It finds this condition in minutes instead of after 8 hours.
3. Start `scripts/lane_sampler.sh` in every GPU job (one JSON line per minute: process state, wait point, memory; GPU use; job memory peak). It also supplies the GPU-busy and real memory numbers the efficiency rules ask for, which no job has recorded (Slurm here records no GPU use: `AccountingStorageTRES` has the GPU count only). Tested for output format only; the cgroup and GPU fields are untested on JHPCE.
4. Print phase time stamps in `fit_paper_config.py` and have the runner record a `pre_fit_s` column and a warning when a lane has not reached its timer after 10 minutes. Logging only; needs a test before use.

## 3. Scorer profile (workstation: Ryzen 5 3600, 6 cores / 12 threads, 94 GB RAM shared with other sessions)

Scorer = `score_scib_native.py` at commit 10a1ca5, run unchanged with timers around each scIB call (`scripts/profile_score.py`); CPU seconds equal wall seconds in every metric, so every metric is single-threaded. Latents: pilot5 immune_hum_mou (98k cells) and pilot5 pancreas (16k cells).

![scorer profile](results/fig_scorer_profile.png)

| Metric | large: seconds | large: % | small: seconds | small: % |
|---|---|---|---|---|
| kBET (local batch-mixing test, run in R) | 797 | 43.8 | 68.9 | 39.5 |
| clustering at the best resolution (for NMI/ARI) | 283 | 15.5 | 7.8 | 4.5 |
| isolated-label F1 | 208 | 11.4 | 9.4 | 5.4 |
| cell-cycle conservation | 136 | 7.5 | 28.4 | 16.3 |
| cell-type silhouette | 100 | 5.5 | 2.5 | 1.4 |
| isolated-label silhouette | 90 | 5.0 | 2.7 | 1.5 |
| PC regression (batch) | 88 | 4.8 | 15.9 | 9.1 |
| neighbour graph (4 builds) | 71 | 3.9 | 28.4 | 16.3 |
| cLISI | 31 | 1.7 | 6.4 | 3.7 |
| iLISI | 29 | 1.6 | 5.6 | 3.2 |
| batch silhouette | 11 | 0.6 | 0.6 | 0.3 |
| trajectory conservation | 10 | 0.5 | 0.0 | 0.0 |
| read prepared file | 8 | 0.4 | 5.1 | 2.9 |
| NMI | 2 | 0.1 | 0.2 | 0.1 |
| graph connectivity | 0 | 0.0 | 0.1 | 0.0 |
| ARI | 0 | 0.0 | 0.0 | 0.0 |
| **total** | 1820 | 100 | 174.6 | 100 |

(Each latent scored alone, one process. Percentages are of the total run time of that latent.)

Process count (pancreas latent, same latent in every copy; machine load 1–3 from other sessions before the runs):

| Scorers at once | per-scorer time (s) | per-scorer speed vs 1 | scores per hour | throughput vs 1 | peak RSS per scorer (GB) |
|---|---|---|---|---|---|
| 1 | 175 | 1.00 | 20.2 | 1.00 | 8.35 |
| 3 | 246 | 0.71 | 43.3 | 2.14 | 8.35 |
| 6 | 379 | 0.46 | 56.2 | 2.78 | 8.34 |
| 8 | 495 | 0.35 | 56.9 | 2.81 | 8.36 |

* PC regression and cell-cycle conservation slow down 5.0× and 4.9× at 6 scorers (they work on large dense matrices: memory-bandwidth limited); kBET and the LISI metrics slow down 1.1×.
* A first sweep at 9 and 12 scorers (noisier machine; up to 92 GB of scorer memory against ~67 GB free) gave 2.62× and 2.73×; no gain beyond 6 and a likely memory shortage, so it is not used.
* Large latent: alone 1,820 s; 3 at once 1,736 s and 1,786 s, 18.4 GB each (no slowdown). Memory (3 × 18.4 = 55 GB of ~67 GB free) limits the number, not CPU.
* JHPCE shared-partition node compute-147 (Intel Xeon Silver 4310; 8 CPUs allocated = 16 threads on 8 cores), same pancreas latent: per-scorer time 212 s with 1 scorer, 233 s with 4, 262 s with 8; throughput 3.67× at 4 and 5.68× at 8 (Slurm 36746374; 8 on 8 CPUs; 16 on 8 CPUs not tested because of memory). During the 8-scorer step CPU busy was 83% (8 × 257.8 CPU s over 8 × 310 s wall).

Thread and process settings tested for exact reproduction (3 A1 latents with existing score rows: atac_small, sim1, immune; 16 score columns each):

| Setting | Matches existing rows | Time for the 3 latents at once | CPU seconds |
|---|---|---|---|
| baseline `thread_env(1)` (first run; machine busier) | 16/16 on all 3 | 585 s for the immune latent | – |
| baseline re-run on the quiet machine | 16/16 on all 3, identical to first run | 803 s summed over latents | 796 |
| NUMBA 2 threads | 16/16 on all 3 | 804 s | 797 |
| BLAS/OpenMP/MKL 2 threads | PCR_batch differs (3e-8 to 1.6e-7); cell_cycle_conservation differs on immune (1e-8) | 809 s | 1112 |
| all 2 threads | same differences as above | 807 s | 1111 |

The first baseline run was slower only because the machine was busier (immune latent 585 s then, 455 s on re-run); with equal load no setting is faster. Result: **keep `thread_env(1)`; running more single-thread scorers is the only speed lever tested.** Process count does not change scores: all copies at 3, 6 and 8 scorers are bit-identical to the single-scorer row.

## 4. Recommended lanes and scorers (the usage rules: memory ≥70% used, CPU ≥85% busy, GPU ≥70% busy)

Fit lanes per L40S job (pilot measurements; four datasets with no wait):

| Item | Measured | Recommendation |
|---|---|---|
| Lanes | 8 lanes gave lane use 0.991–0.994. On atac_large (natural variation in lanes in a timed fit) throughput of the critic arms rose 1.6–2.0× going from 2.7 to 8 lanes (3× the lanes), the light arms (discriminator, mmd) ~linearly; nothing was measured above 8 | keep **8 lanes**; try 9–10 only after the sampler shows GPU use below 70% |
| CPUs | each lane uses 0.997 core; 8 lanes on 10 CPUs = 79% busy (fails 85%) | request **9 CPUs**: 8 × 0.997 / 9 = 89% (leave one core for the runner) |
| Memory (sacct MaxRSS + 20%) | immune_hum_mou 30.4 GB, atac_large 35.3, lung 22.1, pancreas 21.2 (all of 96 GB requested: 22–37%); sim2 116.5 GB (above the 96 GB request without a kill, so sacct counts cache) | request **37 / 43 / 27 / 26 GB** for immune_hum_mou / atac_large / lung / pancreas (used/requested 83%); sim2 stays 96 GB until the sampler gives its real peak |
| GPU busy | **never measured** (no job recorded it; Slurm keeps no GPU use) | use `lane_sampler.sh`; accept the lane count only if GPU use ≥70% |
| atac_large | lane use 0.32 because of the wait | not usable until the start-up check passes (section 2) |

Scorers per CPU node:

| Node | Scorers | Basis | Memory request |
|---|---|---|---|
| Workstation (6 cores, ~67 GB free) | **6 small-latent scorers** (pancreas size), or **3 large-latent scorers** | 6 small: 56 scores/h = 2.8× one; 8 gives 2.81× (no gain). 3 large: no slowdown. Memory used 6 × 8.4 = 50 GB (75% of free) or 3 × 18.4 = 55 GB (82%) | limit set by memory; 3 large + 1 small = 63 GB of ~67 GB free is too tight |
| JHPCE shared partition (8 CPUs = 8 cores) | **8 scorers per 8-CPU job**, one per CPU | 5.68× one scorer, 83% CPU busy; memory 8 × 8.4 = 67 GB measured | 80 GB (84% used); sacct reported a 98.9 GB peak for it (includes cache), no kill |

Caveats: JHPCE scoring is not reproducible for iLISI/cLISI (section 5), so the JHPCE row is only for scores that need not match the workstation bit for bit; large-latent scorers were measured on the workstation only; the workstation is shared with other sessions, so free memory and speed vary. Provisional scale of the pilot scoring on the workstation: immune_hum_mou 378 latents × 1,820 s at 3 at once ≈ 64 h; pancreas 378 latents at 56 per hour ≈ 6.7 h (lung, sim2, atac_large not timed).

## 5. Side finding: JHPCE scores differ from workstation scores for iLISI and cLISI

Source: earlier 20-latent check (Slurm 36569613, run1 and run2 on compute-121) against the 20 local A1 rows, and my 13 copies of one pancreas latent on compute-147 (Slurm 36746374). File: `results/jhpce_vs_workstation_score_match_20latents.csv`.

1. 17 of the 20 latents have the same latent hash as the local row (the 3 sinkhorn ones do not: the local rows come from a different latent; excluded).
2. On the 17: the 14 non-LISI columns are identical to the workstation in all 17 (and identical between run1 and run2 on all 20). iLISI differs in 17 (median 1.9e-3, max 4.2e-3) and cLISI in 11.
3. JHPCE run1 and run2 differ from each other on 19 of 20 latents (max 5.7e-3), and the 13 pancreas copies on compute-147 all differ in iLISI and cLISI. Workstation copies are identical to each other. Package versions are the same on both hosts (scorer_versions column); the hosts differ in CPU (Xeon with AVX-512 vs Ryzen with AVX2).
4. One test to locate it: run the LISI step three times in one process on a JHPCE node, with and without the neighbour graph rebuilt.

This is evidence for the score-match verdict record, not the verdict itself (I do not know the pass criterion given to that check).

## 6. Declared limits

1. GPU-side test not run (no GPU jobs allowed): cause of the wait is a ranked candidate list, not a finding; GPU busy percentages are not measured.
2. The start-up test ran on a shared CPU node (compute-116), not on a GPU node, and without a GPU.
3. Large-latent scoring measured on the workstation only (immune_hum_mou; atac_large, lung and sim2 latents were not timed); JHPCE scorer test used the small pancreas latent only (1, 4, 8 scorers).
4. Thread-setting exactness shown on 3 latents (the minimum asked); concurrency numbers on the shared workstation vary with other sessions' load.
5. The throughput-versus-lanes table for atac_large mixes arms and conditions and time periods (before and after the wait started); read it for direction only (`results/atac_large_throughput_vs_lanes.csv`).
6. No production code was changed. The lane sampler and probe scripts are proposals (sampler tested for output format on the workstation).
