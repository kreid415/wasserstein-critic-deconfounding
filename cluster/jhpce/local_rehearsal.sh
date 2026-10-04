#!/usr/bin/env bash
# Local rehearsal of one JHPCE production job's full dry-run path: what the submitted job does on the L40S node up to
# and including the runner's dry run, at one commit, on this machine. Nothing is submitted and nothing is fitted.
#
# Real (as on the node): the plan from cluster/jhpce/prod_command.py (the job's exact arguments), a git bundle of the
# commit cloned and checked out the way the submit command does it, env.sh, the .verified check, the commit and
# clean-tree check, WCD_SRC=src pytest -q tests/scvi at that commit (CPU here), the concurrency slot, the manifest /
# tags-file SHA-256 and tags-in-manifest checks, fingerprint_prepped.py --compare on the local prepped scIB files, the
# claims guard, scripts/run_stage.py --dry-run with the job's arguments, the stage-key read and the dry-run summary.
# Stubbed, because they need the GPU node or Slurm: nvidia-smi (one L40S), the env_versions.py record (the job gets
# expected_fit_versions.json, i.e. the L40S node's record; the real local record is kept next to it and compared, for
# information), sacct / squeue / SLURM_JOB_ID, and the JHPCE storage roots (WCD_SCRATCH, and HOME, because the job
# refuses any path under HOME).
#
# Usage: cluster/jhpce/local_rehearsal.sh --job jobA|jobB --sha <40-hex> --work DIR --fit-python PY --prepped-dir DIR
# Prints one JSON record (exit code, stage key, plan, tests, fingerprints, version differences) and exits with the
# job's exit code. Run it at nice 19; it is CPU-only (CUDA_VISIBLE_DEVICES is emptied).
set -euo pipefail

JOB= SHA= WORK= REAL_PY= PREPPED=
while [ $# -gt 0 ]; do
  case "$1" in
    --job) JOB=$2;; --sha) SHA=$2;; --work) WORK=$2;; --fit-python) REAL_PY=$2;; --prepped-dir) PREPPED=$2;;
    *) echo "unknown argument $1" >&2; exit 2;;
  esac
  shift 2
done
for v in JOB SHA WORK REAL_PY PREPPED; do [ -n "${!v}" ] || { echo "FATAL: --${v,,} missing" >&2; exit 2; }; done
[[ "$SHA" =~ ^[0-9a-f]{40}$ ]] || { echo "FATAL: --sha must be a full 40-hex commit id" >&2; exit 2; }
[ -x "$REAL_PY" ] || { echo "FATAL: $REAL_PY is not executable" >&2; exit 2; }
[ -d "$PREPPED" ] || { echo "FATAL: $PREPPED is not a directory" >&2; exit 2; }
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)
mkdir -p "$WORK"
WORK=$(cd "$WORK" && pwd -P)
case "/$WORK/" in */tier12/*) echo "FATAL: --work $WORK is inside a 'tier12' directory" >&2; exit 2;; esac
[ ! -e "$WORK/scratch" ] || { echo "FATAL: $WORK/scratch exists: use a fresh --work" >&2; exit 2; }

# ---- plan and bundle (as for a submission) -------------------------------------------------------------------------
mkdir -p "$WORK/harness" "$WORK/bin" "$WORK/home"
TMPREF="refs/heads/rehearsal-tmp-$$"                    # git clone of a bundle takes refs/heads/* only
git -C "$REPO" update-ref "$TMPREF" "$SHA"
git -C "$REPO" bundle create -q "$WORK/harness/wcd.bundle" "$TMPREF" || { git -C "$REPO" update-ref -d "$TMPREF"; exit 2; }
git -C "$REPO" update-ref -d "$TMPREF"
"$REAL_PY" "$REPO/cluster/jhpce/prod_command.py" --job "$JOB" --expected-sha "$SHA" --bundle "$WORK/harness/wcd.bundle" \
  > "$WORK/plan.json"
mapfile -t JOB_ARGS < <("$REAL_PY" -c 'import json,sys; print("\n".join(json.load(open(sys.argv[1]))["job_args"]))' "$WORK/plan.json")
"$REAL_PY" - "$WORK/plan.json" <<'EOF'
import json, shlex, sys
p = json.load(open(sys.argv[1]))
line = 'bash "$JOBDIR/repo/cluster/jhpce/prod_fit_job.sh" ' + " ".join(shlex.quote(x) for x in p["job_args"]) + " || rc=$?"
if line not in p["submit"]["command"].splitlines():
    sys.exit("FATAL: plan job_args differ from the job line of the submit command")
EOF

# ---- stand-ins for the node and Slurm ------------------------------------------------------------------------------
SCR="$WORK/scratch"
JOBID=990001
FIT_ENV="$SCR/conda/envs/wcd-fit"
mkdir -p "$FIT_ENV/bin"
ln -s "$PREPPED" "$SCR/prepped_scib"                       # read-only use: fingerprints and the runner's existence check
cat > "$WORK/bin/versions_stub.py" <<'EOF'
"""env_versions.py on the L40S node: write expected_fit_versions.json's record (the node's) to --out, and the real
local record to <out>.local.json with its differences, for information."""
import json, os, subprocess, sys
args = sys.argv[1:]
script, out = args[0], args[args.index("--out") + 1]
local = out + ".local.json"
r = subprocess.run([sys.executable, script] + [a if a != out else local for a in args[1:]], capture_output=True, text=True)
exp = json.load(open(os.path.join(os.path.dirname(os.path.abspath(script)), "expected_fit_versions.json")))
rec = dict(kind="fit", python=exp["python"], torch_cuda=exp["torch_cuda"], cudnn=exp["cudnn"], cuda_available=True,
           gpu="NVIDIA L40S (rehearsal stub)", modules=dict(exp["modules"]), stub=True)
json.dump(rec, open(out, "w"), indent=1)
diff = {"local_record_exit": r.returncode}
if r.returncode == 0:
    got = json.load(open(local))
    diff["modules"] = {k: [got.get("modules", {}).get(k), v] for k, v in exp["modules"].items()
                       if got.get("modules", {}).get(k) != v}
    diff.update({k: [got.get(k), exp[k]] for k in ("python", "torch_cuda", "cudnn") if got.get(k) != exp[k]})
    diff["local_gpu"], diff["local_cuda_available"] = got.get("gpu"), got.get("cuda_available")
json.dump(diff, open(out + ".local_diff.json", "w"), indent=1)
EOF
cat > "$FIT_ENV/bin/python" <<EOF
#!/bin/sh
case "\$1" in */cluster/jhpce/env_versions.py) exec "$REAL_PY" "$WORK/bin/versions_stub.py" "\$@";; esac
exec "$REAL_PY" "\$@"
EOF
chmod +x "$FIT_ENV/bin/python"
echo "rehearsal: $(date -Is)" > "$FIT_ENV/.verified"
printf '#!/bin/sh\necho "NVIDIA L40S, 555.42.06, 46068 MiB, GPU-rehearsal"\n' > "$WORK/bin/nvidia-smi"
printf '#!/bin/sh\ncase " $* " in *" %%L "*) echo 23:59:00;; esac\n' > "$WORK/bin/squeue"
printf '#!/bin/sh\nexit 0\n' > "$WORK/bin/sacct"                  # no Slurm here: no state is known (= live)
chmod +x "$WORK/bin/nvidia-smi" "$WORK/bin/squeue" "$WORK/bin/sacct"

# ---- the submit command's steps, with SCR on this machine ----------------------------------------------------------
JOBDIR="$SCR/prod/job_$JOBID"
mkdir -p "$JOBDIR"
git clone -q --no-checkout "$WORK/harness/wcd.bundle" "$JOBDIR/repo"
git -C "$JOBDIR/repo" -c advice.detachedHead=false checkout -q "$SHA"
RC=0
env -i PATH="$WORK/bin:/usr/bin:/bin" HOME="$WORK/home" USER="${USER:-rehearsal}" LANG=C.UTF-8 \
  SLURM_JOB_ID="$JOBID" SLURM_CPUS_PER_TASK=10 WCD_SCRATCH="$SCR" CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 \
  nice -n 19 bash "$JOBDIR/repo/cluster/jhpce/prod_fit_job.sh" "${JOB_ARGS[@]}" --dry-run \
  > "$WORK/job_stdout.txt" 2> "$WORK/job_stderr.txt" || RC=$?

L="$JOBDIR/logs"
"$REAL_PY" - "$WORK" "$L" "$RC" "$JOB" "$SHA" <<'EOF'
import json, os, sys
work, logs, rc, job, sha = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4], sys.argv[5]
rd = lambda p: json.load(open(p)) if os.path.isfile(p) else None
tail = lambda p: [x.rstrip() for x in open(p)][-1] if os.path.isfile(p) and os.path.getsize(p) else None
s = rd(os.path.join(logs, "job_summary.json")) or {}
st = rd(os.path.join(logs, "stage_key.dryrun.json")) or {}
rec = dict(job=job, sha=sha, exit=rc, stage_key=s.get("stage_key"), planned=s.get("planned"), problems=s.get("problems"),
           runner_line=next((x.strip() for x in open(os.path.join(logs, "dryrun.log")) if "[stage] " in x and " rows " in x), None)
           if os.path.isfile(os.path.join(logs, "dryrun.log")) else None,
           tests_scvi=tail(os.path.join(logs, "pytest_tests_scvi.txt")),
           fingerprints=(lambda ls: dict(ok=sum(x.endswith(": OK") for x in ls), lines=len(ls)))(
               [x.rstrip() for x in open(os.path.join(logs, "prepped_fingerprints.txt")) if x.strip()]
               if os.path.isfile(os.path.join(logs, "prepped_fingerprints.txt")) else []),
           inputs=rd(os.path.join(logs, "inputs.json")), versions_check=rd(os.path.join(logs, "versions_check.json")),
           local_versions_vs_l40s=rd(os.path.join(logs, "versions_fit.json.local_diff.json")),
           stage_key_record=st, logs=logs, stdout=os.path.join(work, "job_stdout.txt"),
           stderr_tail=[x.rstrip() for x in open(os.path.join(work, "job_stderr.txt"))][-5:])
json.dump(rec, open(os.path.join(work, "rehearsal.json"), "w"), indent=1, sort_keys=True)
print(json.dumps(rec, indent=1, sort_keys=True))
EOF
exit "$RC"
