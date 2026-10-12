#!/usr/bin/env python
"""Submit plan (JSON) of one X1 JHPCE fit job (one task): the tagged job body at prereg-tier12-v2, X1 inputs beside it.

PREREG sec 7 (re-review, 2026-10-04): every stage is fitted and scored from one worktree pinned at the tag
prereg-tier12-v2. The tag's cluster/jhpce/prod_command.py only knows the two filler jobs, and the X1 manifest
(R1/R4-a1/R2/R3-resolved, committed on main) is not in the tag. So this builder writes the same submit command as
prod_command.py (same Slurm lines, bundle clone + checkout of the tag, cluster/jhpce/prod_fit_job.sh of the tag) with
one addition: the X1 manifest and the job's tags file are staged as job inputs and copied to $JOBDIR/x1_inputs/, beside
the checkout, and passed as ../x1_inputs/<file> (prod_fit_job.sh resolves them against the repo and checks their
SHA-256, the tags' membership, experiments and tasks; the checkout stays clean at the tag).
Inputs are read from this repo's HEAD commit (the working-tree files must equal it), so the plan names committed files.
Usage (main repo root): python experiments/X1_jhpce/x1_jhpce_command.py --task immune_hum_mou --bundle wcd_tag.bundle
"""
import argparse
import hashlib
import json
import os
import re
import shlex
import subprocess

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TAG_SHA = "c88cce3c569e9de47f2d65c6794ca3890637786a"                     # prereg-tier12-v2
MANIFEST = "manifests/paper_manifest_stock_pilot_u5_b10_v3.x1.tsv"
TAGS_DIR = "experiments/X1_jhpce/tags"
TASKS = ("atac_large", "immune_hum_mou", "lung", "pancreas", "sim2")
STAGE = "x1_jhpce"
WALL_RE = re.compile(r"^(\d+-)?\d{1,2}:\d{2}:\d{2}$")


def wall_s(w):
    d, rest = (w.split("-", 1) if "-" in w else ("0", w))
    h, m, s = (int(x) for x in rest.split(":"))
    return ((int(d) * 24 + h) * 60 + m) * 60 + s


def committed(path):
    """Bytes of path at this repo's HEAD; refuses if the working-tree file differs or is untracked."""
    blob = subprocess.run(["git", "-C", REPO, "show", f"HEAD:{path}"], capture_output=True)
    if blob.returncode != 0:
        raise SystemExit(f"{path} is not committed at HEAD of {REPO}")
    if open(os.path.join(REPO, path), "rb").read() != blob.stdout:
        raise SystemExit(f"{path} differs from its HEAD version; commit it first")
    return blob.stdout


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=TASKS)
    ap.add_argument("--expected-sha", default=TAG_SHA)
    ap.add_argument("--bundle", required=True, help="workspace path of a git bundle holding --expected-sha")
    ap.add_argument("--walltime", default="3-00:00:00")
    ap.add_argument("--mem", default="96G")
    ap.add_argument("--cpus", type=int, default=10)
    ap.add_argument("--fit-lanes", type=int, default=8)
    ap.add_argument("--fit-timeout-s", type=int, default=28800)
    ap.add_argument("--guard-margin-s", type=int, default=2700)
    ap.add_argument("--queue-allowance-h", type=float, default=120)
    ap.add_argument("--after-slurm-job", default=None,
                    help="Slurm job id this job waits for (afterany: it starts once that job has ended, however it ended)")
    a = ap.parse_args(argv)
    if a.expected_sha != TAG_SHA:
        raise SystemExit(f"--expected-sha must be prereg-tier12-v2 ({TAG_SHA}): PREREG sec 7 pins every stage to the tag")
    if a.after_slurm_job is not None and not re.fullmatch(r"[0-9]+", a.after_slurm_job):
        raise SystemExit(f"--after-slurm-job {a.after_slurm_job!r}: a numeric Slurm job id")
    if not WALL_RE.match(a.walltime) or wall_s(a.walltime) > 3 * 86400:
        raise SystemExit(f"--walltime {a.walltime}: D-HH:MM:SS up to the gpu partition limit of 3 days")
    if a.cpus < a.fit_lanes + 2:
        raise SystemExit(f"--cpus {a.cpus} < fit lanes {a.fit_lanes} + 2")
    if not os.path.isfile(a.bundle):
        raise SystemExit(f"bundle {a.bundle} not found")
    v = subprocess.run(["git", "bundle", "list-heads", a.bundle], capture_output=True, text=True)
    if v.returncode != 0 or a.expected_sha not in v.stdout.split():
        raise SystemExit(f"bundle {a.bundle} has no head at {a.expected_sha}")
    tags_rel = f"{TAGS_DIR}/x1_jhpce_{a.task}.tags"
    man_b, tags_b = committed(MANIFEST), committed(tags_rel)
    man_sha, tags_sha = hashlib.sha256(man_b).hexdigest(), hashlib.sha256(tags_b).hexdigest()
    n_tags = sum(1 for l in tags_b.decode().splitlines() if l.split("#", 1)[0].strip())
    man_name, tags_name = os.path.basename(MANIFEST), os.path.basename(tags_rel)
    args = ["--expected-sha", a.expected_sha, "--stage", STAGE, "--tags-file", f"../x1_inputs/{tags_name}",
            "--tags-sha256", tags_sha, "--manifest", f"../x1_inputs/{man_name}", "--manifest-sha256", man_sha,
            "--experiments", "X1", "--tasks", a.task, "--fit-lanes", str(a.fit_lanes),
            "--fit-timeout-s", str(a.fit_timeout_s), "--max-attempts", "2", "--guard-margin-s", str(a.guard_margin_s)]
    command = "\n".join([
        "#SBATCH --partition=gpu",
        "#SBATCH --account=jhpce",
        "#SBATCH --gres=gpu:l40s:1",
        f"#SBATCH --cpus-per-task={a.cpus}",
        f"#SBATCH --mem={a.mem}",
        f"#SBATCH --time={a.walltime}",
        "#SBATCH --signal=B:TERM@300",
        "#SBATCH --no-requeue",
        *([f"#SBATCH --dependency=afterany:{a.after_slurm_job}"] if a.after_slurm_job else []),
        "set -euo pipefail",
        'SCR="/fastscratch/myscratch/$USER/wcd"',
        'JOBDIR="$SCR/prod/job_${SLURM_JOB_ID}"',
        'mkdir -p "$JOBDIR/x1_inputs"',
        f'printf "%s  %s\\n" {man_sha} {shlex.quote(man_name)} {tags_sha} {shlex.quote(tags_name)} | sha256sum -c --quiet',
        f'cp {shlex.quote(man_name)} {shlex.quote(tags_name)} "$JOBDIR/x1_inputs/"',
        'git clone -q --no-checkout "$PWD/wcd.bundle" "$JOBDIR/repo"',
        f'git -C "$JOBDIR/repo" -c advice.detachedHead=false checkout -q {a.expected_sha}',
        "rc=0",
        'bash "$JOBDIR/repo/cluster/jhpce/prod_fit_job.sh" ' + " ".join(shlex.quote(x) for x in args) + " || rc=$?",
        'for f in "$JOBDIR/logs/job_summary.json" "$JOBDIR/harvest.json"; do [ ! -f "$f" ] || cp "$f" .; done   # KB-sized; the harvest stays on fastscratch',
        'ls -l .',
        'exit "$rc"',
    ])
    rt = int(wall_s(a.walltime) + a.queue_allowance_h * 3600)
    plan = dict(job=f"x1_{a.task}", tasks=[a.task], after_slurm_job=a.after_slurm_job, n_tags=n_tags, tags_sha256=tags_sha, manifest_sha256=man_sha,
                expected_sha=a.expected_sha, job_args=args, inputs_dir="../x1_inputs",
                submit=dict(command=command,
                            inputs=[{"src": a.bundle, "dst": "wcd.bundle"},
                                    {"src": os.path.join(REPO, MANIFEST), "dst": man_name},
                                    {"src": os.path.join(REPO, tags_rel), "dst": tags_name}],
                            outputs=["job_summary.json", "harvest.json", "slurm-*.out"], run_timeout_s=rt,
                            intent=(f"JHPCE gpu/jhpce 1x L40S (gpu:l40s:1), {a.cpus} CPU {a.mem}, wall {a.walltime}: "
                                    f"X1 fits {a.task} ({n_tags} rows: X1 adversarial arms) at prereg-tier12-v2 "
                                    f"{a.expected_sha[:10]}, fit only, harvest on fastscratch"
                                    + (f"; starts after Slurm job {a.after_slurm_job} ends" if a.after_slurm_job else ""))))
    print(json.dumps(plan, indent=1))


if __name__ == "__main__":
    main()
