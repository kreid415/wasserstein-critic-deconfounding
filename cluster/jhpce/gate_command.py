#!/usr/bin/env python
"""Submit plan (JSON) of the JHPCE cross-host GPU check at the tag prereg-tier12-v3 (review item CR2-01): one 'gpu'
batch job on one L40S that runs cluster/jhpce/gate_job.sh of the tag (the unchanged v2 gate: GPU witness, tests/scvi,
the 64-fit queue + 2 single-lane fits, completion gate). Nothing is submitted: the job needs the user's explicit go
(project GO rule); the agent then passes plan['submit'] to host.compute.create('ssh:jhpce').submit_job(**plan['submit']).
The repo travels as a git bundle (the login node has no GitHub); the node checks out GATE_SHA, asserts HEAD and a clean
tree, and gate_gpu.sh re-asserts a clean tree and writes git_sha into every latent.
Bundle (repo root): git branch gate-v3 prereg-tier12-v3^{commit} && git bundle create wcd_gate_v3.bundle gate-v3
Usage: python cluster/jhpce/gate_command.py --bundle wcd_gate_v3.bundle [--run-timeout-h 122]
"""
import argparse
import json
import os
import re
import subprocess

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TAG, GATE_SHA = "prereg-tier12-v3", "10a1ca5874fbcdc033b148525b5e71d866306bdb"
WALL_RE = re.compile(r"^(\d+-)?\d{1,2}:\d{2}:\d{2}$")


def git(*a, cwd=REPO):
    return subprocess.run(["git", "-C", cwd, *a], capture_output=True, text=True)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", required=True, help="workspace path of a git bundle with a head at the tag's commit")
    ap.add_argument("--expected-sha", default=GATE_SHA)
    ap.add_argument("--walltime", default="02:00:00", help="measured L40S gate job: 7.6 min (docs/jhpce/GATE_JHPCE.md)")
    ap.add_argument("--run-timeout-h", type=float, default=122, help="harness clock from submission: walltime + 120 h queue; "
                    "576 also tests the harness ceiling for review item G-15 (Slurm still ends the job at --walltime)")
    a = ap.parse_args(argv)
    if a.expected_sha != GATE_SHA:
        raise SystemExit(f"--expected-sha must be {TAG} ({GATE_SHA}); this gate exists to check that commit")
    if git("rev-parse", f"{TAG}^{{commit}}").stdout.strip() != GATE_SHA:
        raise SystemExit(f"tag {TAG} is missing here or does not point at {GATE_SHA}")
    if not WALL_RE.match(a.walltime):
        raise SystemExit(f"--walltime {a.walltime}: D-HH:MM:SS")
    if not os.path.isfile(a.bundle):
        raise SystemExit(f"bundle {a.bundle} not found")
    v = git("bundle", "verify", os.path.abspath(a.bundle))
    heads = subprocess.run(["git", "bundle", "list-heads", a.bundle], capture_output=True, text=True)
    if v.returncode != 0 or GATE_SHA not in heads.stdout.split():
        raise SystemExit(f"bundle {a.bundle} is invalid or has no head at {GATE_SHA}")
    command = "\n".join([
        "#SBATCH --partition=gpu",
        "#SBATCH --account=jhpce",
        "#SBATCH --gres=gpu:l40s:1",
        "#SBATCH --cpus-per-task=10",
        "#SBATCH --mem=64G",
        f"#SBATCH --time={a.walltime}",
        "#SBATCH --no-requeue",
        "set -euo pipefail",
        'W="$PWD"',
        'JOBDIR="/fastscratch/myscratch/$USER/wcd/gate_v3/job_${SLURM_JOB_ID}"',
        'mkdir -p "$JOBDIR"',
        'git clone -q --no-checkout "$W/wcd.bundle" "$JOBDIR/repo"',
        f'git -C "$JOBDIR/repo" -c advice.detachedHead=false checkout -q {GATE_SHA}',
        f'[ "$(git -C "$JOBDIR/repo" rev-parse HEAD)" = {GATE_SHA} ] || {{ echo "FATAL: wrong commit"; exit 1; }}',
        '[ -z "$(git -C "$JOBDIR/repo" status --porcelain)" ] || { echo "FATAL: checkout is dirty"; exit 1; }',
        "rc=0",
        'bash "$JOBDIR/repo/cluster/jhpce/gate_job.sh" "$W" L40S 2 || rc=$?   # parts + report land in the job dir',
        "ls -l . gate_report || true",
        'exit "$rc"',
    ])
    rt = int(a.run_timeout_h * 3600)
    plan = dict(job="gate_v3_L40S", expected_sha=GATE_SHA,
                submit=dict(command=command, inputs=[{"src": a.bundle, "dst": "wcd.bundle"}],
                            outputs=["gate_report/*", "gate_latents_jhpce.tar.part*", "gate_latents_jhpce.tar.md5",
                                     "slurm-*.out"],
                            run_timeout_s=rt,
                            intent=(f"JHPCE gpu/jhpce 1x L40S, 10 CPU 64G, wall {a.walltime}: cross-host GPU check "
                                    f"(CR2-01) at {TAG} {GATE_SHA[:10]}: tests/scvi + 66 immune fits, no benchmark fits; "
                                    f"harness clock {a.run_timeout_h:g} h")))
    print(json.dumps(plan, indent=1))


if __name__ == "__main__":  # pragma: no cover (entry point)
    main()
