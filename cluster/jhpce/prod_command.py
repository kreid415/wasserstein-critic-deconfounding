#!/usr/bin/env python
"""Print the submission plan of one JHPCE production fit job (JSON): the harness `command` (#SBATCH header first,
then: clone the staged bundle on fastscratch, check out the expected commit, run cluster/jhpce/prod_fit_job.sh), the
inputs (repo bundle), the small outputs copied back, run_timeout_s and the approval-card intent. Nothing is submitted:
each submission needs the user's explicit go (CONSTRAINTS.md SI-29); the agent passes the plan to
host.compute.create('ssh:jhpce').submit_job(**plan['submit']).

Job definitions (JOBS) follow CONSTRAINTS.md SI-39 (2 L40S; atac_large, immune_hum_mou, lung, pancreas, sim2) and the
tags files of cluster/jhpce/make_tags.py. Resources (docs/jhpce/PRODUCTION_JOB.md): 1 L40S, 10 CPUs (8 fit lanes + 2),
96 GB, 24 h wall (estimate x >= 4; stop guard at end - 45 min), --signal=B:TERM@300, --no-requeue.

Usage: python cluster/jhpce/prod_command.py --expected-sha SHA --job jobA [--bundle wcd.bundle] [--walltime 1-00:00:00]
"""
import argparse
import hashlib
import json
import os
import re
import shlex
import subprocess

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MANIFEST = "manifests/paper_manifest_stock_pilot_u5_b10.tsv"
JOBS = {
    "jobA": dict(stage="fillers_x1_x13", tags="cluster/jhpce/tags/fillers_x1_x13_jobA.tags", experiments="X1 X13",
                 tasks="atac_large immune_hum_mou"),
    "jobB": dict(stage="fillers_x1_x13", tags="cluster/jhpce/tags/fillers_x1_x13_jobB.tags", experiments="X1 X13",
                 tasks="lung pancreas sim2"),
}
WALL_RE = re.compile(r"^(\d+-)?\d{1,2}:\d{2}:\d{2}$")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def wall_s(w):
    d, rest = (w.split("-", 1) if "-" in w else ("0", w))
    h, m, s = (int(x) for x in rest.split(":"))
    return ((int(d) * 24 + h) * 60 + m) * 60 + s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--expected-sha", required=True)
    ap.add_argument("--job", required=True, choices=sorted(JOBS))
    ap.add_argument("--bundle", default="wcd.bundle", help="workspace path of the git bundle holding --expected-sha")
    ap.add_argument("--walltime", default="1-00:00:00")
    ap.add_argument("--mem", default="96G")
    ap.add_argument("--cpus", type=int, default=10)
    ap.add_argument("--fit-lanes", type=int, default=8)
    ap.add_argument("--fit-timeout-s", type=int, default=7200)
    ap.add_argument("--guard-margin-s", type=int, default=2700)
    ap.add_argument("--queue-allowance-h", type=float, default=120, help="added to the walltime for run_timeout_s "
                    "(the harness counts its run clock from submit, queue time included)")
    a = ap.parse_args()
    if not re.fullmatch(r"[0-9a-f]{40}", a.expected_sha):
        raise SystemExit("--expected-sha must be a full 40-hex commit id")
    if not WALL_RE.match(a.walltime) or wall_s(a.walltime) > 3 * 86400:
        raise SystemExit(f"--walltime {a.walltime}: D-HH:MM:SS up to the gpu partition limit of 3 days")
    if a.cpus < a.fit_lanes + 2:
        raise SystemExit(f"--cpus {a.cpus} < fit lanes {a.fit_lanes} + 2")
    j = JOBS[a.job]
    if subprocess.run(["git", "-C", REPO, "cat-file", "-e", f"{a.expected_sha}^{{commit}}"]).returncode != 0:
        raise SystemExit(f"{a.expected_sha} is not a commit of {REPO}")
    show = lambda p: subprocess.run(["git", "-C", REPO, "show", f"{a.expected_sha}:{p}"], check=True, capture_output=True).stdout
    tags_sha = hashlib.sha256(show(j["tags"])).hexdigest()          # the files as committed at the expected SHA
    man_sha = hashlib.sha256(show(MANIFEST)).hexdigest()
    n_tags = sum(1 for l in show(j["tags"]).decode().splitlines() if l.split("#", 1)[0].strip())
    args = ["--expected-sha", a.expected_sha, "--stage", j["stage"], "--tags-file", j["tags"], "--tags-sha256", tags_sha,
            "--manifest", MANIFEST, "--manifest-sha256", man_sha, "--experiments", j["experiments"], "--tasks", j["tasks"],
            "--fit-lanes", str(a.fit_lanes), "--fit-timeout-s", str(a.fit_timeout_s), "--max-attempts", "2",
            "--guard-margin-s", str(a.guard_margin_s)]
    command = "\n".join([
        "#SBATCH --partition=gpu",
        "#SBATCH --account=jhpce",
        "#SBATCH --gres=gpu:l40s:1",
        f"#SBATCH --cpus-per-task={a.cpus}",
        f"#SBATCH --mem={a.mem}",
        f"#SBATCH --time={a.walltime}",
        "#SBATCH --signal=B:TERM@300",
        "#SBATCH --no-requeue",
        "set -euo pipefail",
        'SCR="/fastscratch/myscratch/$USER/wcd"',
        'JOBDIR="$SCR/prod/job_${SLURM_JOB_ID}"',
        'mkdir -p "$JOBDIR"',
        'git clone -q --no-checkout "$PWD/wcd.bundle" "$JOBDIR/repo"',
        f'git -C "$JOBDIR/repo" -c advice.detachedHead=false checkout -q {a.expected_sha}',
        "rc=0",
        'bash "$JOBDIR/repo/cluster/jhpce/prod_fit_job.sh" ' + " ".join(shlex.quote(x) for x in args) + " || rc=$?",
        'for f in "$JOBDIR/logs/job_summary.json" "$JOBDIR/harvest.json"; do [ ! -f "$f" ] || cp "$f" .; done   # KB-sized; the harvest stays on fastscratch',
        'ls -l .',
        'exit "$rc"',
    ])
    rt = int(wall_s(a.walltime) + a.queue_allowance_h * 3600)
    plan = dict(job=a.job, tasks=j["tasks"].split(), n_tags=n_tags, tags_sha256=tags_sha, manifest_sha256=man_sha,
                expected_sha=a.expected_sha,
                submit=dict(command=command, inputs=[{"src": a.bundle, "dst": "wcd.bundle"}],
                            outputs=["job_summary.json", "harvest.json", "slurm-*.out"], run_timeout_s=rt,
                            intent=(f"JHPCE gpu/jhpce 1x L40S (gpu:l40s:1), {a.cpus} CPU {a.mem}, wall {a.walltime}: "
                                    f"production fits {a.job} ({n_tags} rows: X1 none/scvi_adv + X13 scanvi/sysvi; "
                                    f"{j['tasks']}) at {a.expected_sha[:10]}, fit only, harvest on fastscratch")))
    print(json.dumps(plan, indent=1))


if __name__ == "__main__":
    main()
