# X1 cluster part: submission plan (prepared 2026-10-12, not submitted)

Status: preflight GO (experiments/X1/PREFLIGHT.md, 46 items: 32 pass, 14 not applicable, 0 fail). Waiting for your go.

## The 5 jobs (1,800 fits = the six adversarial methods x 6 strengths x 5 seeds x 2 decoders on each dataset)

Each job: JHPCE partition `gpu`, account `jhpce`, 1 L40S GPU, 10 CPUs, 96 GB RAM, 3-day limit, 8 fits at a time on the GPU, fit only (scoring happens on the workstation).
At most 2 jobs run at once (the job script's own limit); each chain starts with a large dataset so the most expensive models reach local scoring first.

| Job | Dataset | Cells | Fits | Pilot ratio (its 15 L40S fits / cost model) | Predicted GPU-days | If fits run 1.5x slower than the cost model | Longest single fit | Starts |
|---|---|---|---|---|---|---|---|---|
| 1 | immune_hum_mou | 97,861 | 360 | 1.21 | 2.2 | 2.7 | 2.0 h | at submission |
| 2 | atac_large | 84,813 | 360 | 0.90 | 1.4 | 2.4 | 1.6 h | at submission |
| 3 | lung | 32,472 | 360 | 1.11 | 1.9 | 2.5 | 1.8 h | after job 1 ends |
| 4 | sim2 | 19,318 | 360 | 0.90 | 1.5 | 2.4 | 1.7 h | after job 2 ends |
| 5 | pancreas | 16,382 | 360 | 1.19 | 1.5 | 1.9 | 1.3 h | after job 4 ends |

Predicted = tagged cost model x that dataset's pilot ratio (its 15 rule-free X1 fits on the same nodes, Oct 4-5). Chains: 1 -> 3 about 4.0 days, 2 -> 4 -> 5 about 4.4 days, plus queue waits; total 8.5 GPU-days. The six adversarial methods have not been timed on an L40S yet.
If a job runs short of time, its runner is stopped 45 min before the Slurm limit and the finished models are kept; one resume job then fits the rest.

## What each job checks before it fits anything (tagged job script cluster/jhpce/prod_fit_job.sh)
1. The checkout is the tag prereg-tier12-v2 (c88cce3) and clean.
2. Exactly one L40S is visible, and the fit environment matches its recorded package versions.
3. tests/scvi pass on that node, including the test that lambda = 0 is bit-identical to stock scVI.
4. The design file and the run list match their SHA-256 sums; every listed run is an X1 run of that dataset.
5. A concurrency slot is free, the 8 prepped datasets match their fingerprints, no live job holds the same runs, and the runner's dry run plans 360 fits.

## Code and inputs
- Code: the tag prereg-tier12-v2 (c88cce3), unchanged. This differs from my earlier description ("a new git tag"). PREREG sec 7 requires every stage to be fitted and scored from the one tagged commit,
  and the 75 X1 baseline fits already made on JHPCE are at c88cce3. Fits from a new commit would trip the runner's mixed-commit check.
  So the X1 design file and each job's run list travel as job inputs, copied next to the checkout and checked by SHA-256 inside the job.
- Design file: manifests/paper_manifest_stock_pilot_u5_b10_v3.x1.tsv (sha256 7c2fe8679631c882...). Run lists: experiments/X1_jhpce/tags/x1_jhpce_<dataset>.tags (360 runs each, 60 per method).
- Builder experiments/X1_jhpce/x1_jhpce_command.py: refuses any commit other than the tag, a walltime over 3 days, or uncommitted inputs.
- Rehearsal experiments/X1_jhpce/rehearse_x1_job.sh: all 5 jobs passed locally (2026-10-12; tests/scvi 110 passed, dry run 360 fits each).

## After each job
The harvest (about 2.5 GB per job: 6.8 MB per fit, measured on the filler jobs, x 360) is downloaded, checked against its SHA-256 sums and scored on the workstation with the tagged scorer.
Each download needs an approval card, because /fastscratch/myscratch/kreid is not a JHPCE data root. Adding it in the JHPCE compute settings would remove these 5 cards.

## Submit command of job 1 (immune_hum_mou), exactly as it would be sent
```
#SBATCH --partition=gpu
#SBATCH --account=jhpce
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=96G
#SBATCH --time=3-00:00:00
#SBATCH --signal=B:TERM@300
#SBATCH --no-requeue
set -euo pipefail
SCR="/fastscratch/myscratch/$USER/wcd"
JOBDIR="$SCR/prod/job_${SLURM_JOB_ID}"
mkdir -p "$JOBDIR/x1_inputs"
printf "%s  %s\n" 7c2fe8679631c882abf6a4d020c6c3ba7f327261903bf47fc968e6cf80feaf85 paper_manifest_stock_pilot_u5_b10_v3.x1.tsv 743bb8d9a40b17670dd389b7ac6b849bf1bb6e7c4cbd689366805fc23ab10d87 x1_jhpce_immune_hum_mou.tags | sha256sum -c --quiet
cp paper_manifest_stock_pilot_u5_b10_v3.x1.tsv x1_jhpce_immune_hum_mou.tags "$JOBDIR/x1_inputs/"
git clone -q --no-checkout "$PWD/wcd.bundle" "$JOBDIR/repo"
git -C "$JOBDIR/repo" -c advice.detachedHead=false checkout -q c88cce3c569e9de47f2d65c6794ca3890637786a
rc=0
bash "$JOBDIR/repo/cluster/jhpce/prod_fit_job.sh" --expected-sha c88cce3c569e9de47f2d65c6794ca3890637786a --stage x1_jhpce --tags-file ../x1_inputs/x1_jhpce_immune_hum_mou.tags --tags-sha256 743bb8d9a40b17670dd389b7ac6b849bf1bb6e7c4cbd689366805fc23ab10d87 --manifest ../x1_inputs/paper_manifest_stock_pilot_u5_b10_v3.x1.tsv --manifest-sha256 7c2fe8679631c882abf6a4d020c6c3ba7f327261903bf47fc968e6cf80feaf85 --experiments X1 --tasks immune_hum_mou --fit-lanes 8 --fit-timeout-s 28800 --max-attempts 2 --guard-margin-s 2700 || rc=$?
for f in "$JOBDIR/logs/job_summary.json" "$JOBDIR/harvest.json"; do [ ! -f "$f" ] || cp "$f" .; done   # KB-sized; the harvest stays on fastscratch
ls -l .
exit "$rc"
```
