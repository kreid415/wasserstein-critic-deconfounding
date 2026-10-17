# JHPCE cross-host GPU check at the v3 tag (review item CR2-01): plan, prepared 2026-10-17, NOT submitted

Status: prepared and rehearsed locally; waits for the user's explicit go (project GO rule). Nothing was submitted.
Branch `jhpce-gate-v3` (from c241af6). Frozen files (`src/`, `scripts/`, `docs/PREREG.md`) are untouched.

## 1. Why this check exists

The earlier cluster check (`docs/jhpce/GATE_JHPCE.md`, 2026-10-03) ran code from before the final-review fixes (commits 6675f8c / 2f29004).
Since then 7 files of the fit path changed (review item CR2-01), the largest being the corrected Sinkhorn solver in
`src/wcd_vae/wcd/alignment.py`. Before 1,800 X1 fits (5 jobs, ~8.5 GPU-days) run on JHPCE's L40S graphics cards, one short
job must show that the tagged code (`prereg-tier12-v3`, commit 10a1ca5874fbcdc033b148525b5e71d866306bdb) still runs
correctly there and gives the same latents (the 10-dimensional cell embeddings the fits produce) as the last check.
Of the 1,800 X1 cluster fits, 300 are Sinkhorn fits (X1 manifest `paper_manifest_stock_pilot_u5_b10_v3.x1.tsv`).

## 2. What runs (one Slurm job, one L40S)

The unchanged gate of 2026-10-03 (`cluster/jhpce/gate_job.sh` -> `gate_gpu.sh`, both identical to the tag), now at the v3 commit:

1. Checks before any fit: exactly one visible GPU, and it is an L40S; the fit environment is marked verified; the checkout is
   clean; the queue manifest (md5 9cca9c8c38e767ebf13a98ad44bb9e95) and the staged immune data (md5 3888d2563a7243f9854e41de973f3607) match.
2. GPU witness: package versions and CUDA visible (`env_versions.py --kind fit`).
3. The repository's `tests/scvi` test suite (111 tests at the last local run, 2b52542) on the GPU node; any failure stops the job.
4. Two single-lane fits (barycenter, 3 and 1 epochs), then the 64-fit queue (immune dataset, 3 epochs, 8 fits at a time, one thread
   each, manifest order of the local calibration): per arm x conditioning, 5 fits for barycenter, discriminator, mmd, pooled,
   reference and sinkhorn, plus 2 `none` and 2 `scvi_adv` fits.
5. Completion gate on the node: 64 of 64 latents present, shape (33506, 10), finite, written by this commit with a clean tree
   on an L40S; fit exit codes recorded. Then a throughput calibration (`calibrate_concurrency.py`).

Same manifest and same input bytes as the 2026-10-02/03 gate and the local calibration, so results are comparable.
No benchmark fit, no scoring (scoring stays on the workstation, gate item G4).

## 3. Where and with what (read-only probe of ssh:jhpce, 2026-10-17 12:22 EDT; raw output in `docs/jhpce/gate_data_v3/`)

| item | value |
|---|---|
| submit | partition `gpu`, account `jhpce`, `--gres=gpu:l40s:1` (pinned: one GPU model for all gate and X1 fits, SI-17), 10 CPUs, 64 GB, `--no-requeue` |
| walltime | 02:00:00. Measured: the L40S gate job of 2026-10-03 took 7 min 36 s (Slurm 36133219: tests 12 s, queue 354 s, peak memory 19.4 GB). v3 adds the corrected Sinkhorn solver (up to 2.8x slower on the hardest fits) and a longer test suite, so the estimate is 10 to 25 min; 2 h is at least 5 times the estimate. A fit that runs over 50 min is stopped by the script and fails the gate loudly |
| code | tag commit shipped as a git bundle (27 MB; the login node has no GitHub); the job clones it into `/fastscratch/myscratch/kreid/wcd/gate_v3/job_<id>/repo`, checks out 10a1ca5874fb..., and exits if HEAD differs or the tree is dirty. `~/wcd_repo` on the cluster (a September clone, commit a54fa05) is not used |
| environment | `/users/kreid/envs/wcd-fit` (in $HOME): exists, `.verified` written 2026-10-04 19:01, python 3.11.15 starts, 69,224 recorded files with the 1 known benign gap (`bin/pip3.12`). Built from the committed recipe on 2026-10-04 (torch 2.13.0+cu126, scvi-tools 1.4.2) |
| inputs on scratch | `/fastscratch/myscratch/kreid/wcd/ref_local/` holds `immune__scib.h5ad` (270,288,496 bytes), `bench_queue_manifest.tsv` (md5 matches), `queue_order.txt` (64 lines), `.staged`; all written 2026-10-02. Fastscratch deletes files older than 30 days, so these (and `prepped_scib/*.h5ad`, written 2026-10-02 18:51 to 18:52, which X1 needs) are due for deletion from about 2026-11-01. Run the gate and X1 before then, or refresh the files' dates with a short `shared` job |
| job data | all on fastscratch (`gate/`, caches, TMPDIR via `cluster/jhpce/env.sh`); fastscratch has 15 TB free; the gate writes about 0.6 GB. Only the report and the 2 latent tar parts (60 MB each, 109 MB in all) go to the job's harness folder under `/users/kreid/.claude-science-scratch` (inside $HOME; $HOME free space was not measurable today, last measured 46,978 MiB of the 100 GB cap on 2026-10-04) |
| queue (2026-10-17 12:22 EDT) | `gpu` partition: 17 running, 36 pending (8 of them ask for an L40S); all 3 L40S nodes (compute-171, -172, -173; 12 GPUs) show 4 of 4 GPUs allocated. Observed waits for a 2 h, 1-GPU job pinned to an L40S: 7.8 h (2026-10-02). Expect hours to a day; multi-day waits are unmeasured |
| cost | no core-hour accounting at JHPCE (checked 2026-08-26); about 0.2 GPU-hours if it runs as before. The per-user cap is 5 GPUs (QOS `public-gpu-limit`); this job uses 1 |
| jobs | exactly 1 Slurm job; 66 fits (2 single-lane + 64 queue) |

## 4. Review item G-15: the 384 h / 576 h harness limits

- The cluster side: `gpu` partition `MaxTime=3-00:00:00` (72 h per Slurm job), QOS `public-gpu-limit` allows `gres/gpu=5` per user and no other limit; X1's `--time=3-00:00:00` is the maximum allowed.
- The 384 h and 576 h are the compute tool's own run clock (`run_timeout_s`, counted from submission), not a Slurm limit. The tool reports its ceiling only when a submission exceeds it (`TimeoutCeiling`, with the limit in the error); it cannot be read beforehand without submitting.
- Proposal: submit this gate with `--run-timeout-h 576`. Slurm still ends the job after 2 h, so the long clock is only a backstop. If 576 h is above the ceiling the submission is refused before anything runs and the error names the limit; resubmit with 384, then 192. This settles G-15 at no extra cost. If the ceiling turns out below 576 h, X1 job 5 cannot be chained behind job 4 with its current clock; the fallback is to submit job 5 only after job 4 ends, with a 192 h clock.

## 5. How results come back and are checked (locally, no further cluster use)

1. After `compute_done`: the files arrive under `hpc/<job_id>/` (`gate_report/*`, `gate_latents_jhpce.tar.part00/01`, `.md5`, `slurm-*.out`). Mirror them to a new folder `D/jhpce_gate_v3/` (not `D/jhpce_tier12`, which holds the blinded X1/X13 stages; this folder holds no X1 score).
2. Join the parts (`cat part00 part01 > gate_latents_jhpce.tar`); `md5sum` must equal the `.md5` file; extract to `D/jhpce_gate_v3/latents/`.
3. `python cluster/jhpce/gate_v3_check.py --new D/jhpce_gate_v3/latents --old <latents of artifact 19d6ee4d (L40S gate of 2026-10-03)> --manifest <bench_queue_manifest.tsv of artifact fa640ada> --expected-sha 10a1ca5874fbcdc033b148525b5e71d866306bdb`.
   Rehearsed on the real 2026-10-03 L40S latents: 66 rows, 56 non-Sinkhorn rows compared, 0 invalid, exit 0 with their own commit; exit 1 with a wrong commit.
4. `scripts/gate_compare_latents.py` of new L40S latents against the local latents (command in `docs/jhpce/GATE_V2_COMMANDS.md` step 2) for the record.
5. Save the report folder as an artifact; add a short results page next to `docs/jhpce/GATE_JHPCE.md`.

## 6. Pass criteria (declared before the run; nothing of this job has been seen)

| # | criterion | outcome if it fails |
|---|---|---|
| P1 | Job exit 0 on one L40S; `tests/scvi`: 0 failed; 64 of 64 queue fits and both single-lane fits exit 0; on-node completion gate 64/64 valid (`gate_completion.json`, problems 0), provenance 10a1ca5, clean | FAIL: no X1 cluster launch; ISSUE entry; diagnose from `slurm-*.out` and `gate_report/` |
| P2 | Harvest tar md5 matches; `gate_v3_check.py` exits 0: all 66 latents valid and all 56 non-Sinkhorn latents are bit-identical to the 2026-10-03 L40S gate latents (same GPU model, same environment recipe; local runs of these arms were already shown bit-identical across the fixes, `experiments/integration_checks_20261013/README.md`, CPU fits) | exit 2 (differs): go to P3; exit 1: FAIL as P1 |
| P3 (only if P2 gives exit 2) | Bitwise equality can fail from GPU non-determinism that the gate never measured. Then, per (arm, conditioning) group, the mean 15-neighbour overlap with the local latents must be within 0.05 and the median per-dimension correlation within 0.02 of the 2026-10-03 L40S values in `docs/jhpce/GATE_JHPCE.md` (G3). These tolerances are my proposal, set before any v3 gate output was seen; change them before the run if you want different ones | FAIL: no X1 cluster launch; decide with the user |
| P4 | Sinkhorn: the 10 queue fits finish with finite latents (a solver cap hit raises and fails the fit, so exit 0 means convergence at tolerance 1e-4); fit times recorded | A cap hit is the user's call (SI-81 b) even in this gate |
| P5 | Throughput recorded (`concurrency_calibration_jhpce.json`, `queue_summary.txt`); informational: compare queue wall time with 353.7 s (L40S, 2026-10-03) and re-check the X1 plan's predicted 8.5 GPU-days if the Sinkhorn fits are far slower | no pass/fail |

Limits: the Sinkhorn rows have no valid earlier comparator (the old solver gradient was wrong, SI-51), so P2 excludes them; P4 checks only that they run and converge. Gate G4 (scoring on JHPCE CPUs) is not repeated: scoring stays local.

## 7. Go request (text for the user)

> JHPCE go, 1 job: cross-host GPU check at tag prereg-tier12-v3 (CR2-01). Partition `gpu`, account `jhpce`, 1x L40S, 10 CPU, 64 GB, walltime 2 h (last run took 7.6 min). Command: `python cluster/jhpce/gate_command.py --bundle wcd_gate_v3.bundle --run-timeout-h 576`, then submit its plan. Runs tests/scvi plus 66 immune fits (no benchmark fits). Outputs: report and 109 MB latents under `D/jhpce_gate_v3/`. Queue wait: hours to a day. The 576 h clock only probes the tool's limit (G-15).

## 8. Commands (exact)

```
# local, repo root, at any checkout containing the tag
git clone -q --no-checkout . /tmp/bclone && git -C /tmp/bclone update-ref refs/heads/gate-v3 10a1ca5874fbcdc033b148525b5e71d866306bdb
git -C /tmp/bclone bundle create wcd_gate_v3.bundle gate-v3            # 27 MB
python cluster/jhpce/gate_command.py --bundle /tmp/bclone/wcd_gate_v3.bundle --run-timeout-h 576 > gate_plan.json
# in the repl kernel after the go: c = host.compute.create('ssh:jhpce'); c.submit_job(**json.load(open('gate_plan.json'))['submit'])
```
Updating the cluster's code to the tag needs no command of its own: every job clones the bundle fresh. (To update the old clone anyway:
stage the bundle, then `git -C ~/wcd_repo fetch /path/wcd_gate_v3.bundle gate-v3` and `git -C ~/wcd_repo checkout 10a1ca5874fbcdc033b148525b5e71d866306bdb`; not needed and not done.)

## 9. Not done here (for the lead)

- `experiments/X1_jhpce/x1_jhpce_command.py` still has TAG_SHA c88cce3 (G-17) and `SUBMISSION_PLAN.md` still describes v2; they must move to the v3 commit before X1 jobs are built.
- Peer jobs on the account (5 pending CPU jobs named `p2a-cpu-*` at the time of the probe) belong to another session; none were touched.
