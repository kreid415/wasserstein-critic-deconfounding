#!/usr/bin/env python
"""Fit-to-scoring chain: submit a throttled CPU scoring array for latents the moment they pass the runner's validity check.

Every --interval seconds: read the expected rows (manifest + tags files), judge each latent with scripts/run_stage.py's own
output contract (cluster/jhpce/chain_lib.py), and for each dataset ('task') send the valid, not yet scored, not yet submitted
latents to one Slurm array of `cluster/jhpce/score_array.sh` from --score-repo (one latent per array task, 1 CPU, the thread
settings of run_stage.thread_env(1), so scores equal the runner's own). A batch goes when it holds --min-batch latents, or when
the dataset's fits are all final, or when every fit job is over. The arrays together never hold more than --max-cpus CPUs or
--max-mem-gb of memory (sum of their throttles; each array's `%throttle`). Scores land in --score-out/scores/<tag>.csv.
A task whose size in score_sizes.json is not 'measured' sends --probe-n latents first; the peak RSS of that array (sacct) x 1.3
sets the memory of the rest (state/measured.json).
Ends: exit 0 when every expected unit has a valid score (or a recorded fit failure), 3 when nothing more can happen (fits over,
no live array, units still without a score: listed in state/finished.json), 10 at --max-ticks / --max-hours (state kept; run again).
A failed scoring task is never resubmitted: it is listed and ends the chain at exit 3 (resubmit by hand after reading its log).
State (all in --state-dir): submissions.jsonl (append only; intent then submitted per batch), batches/<id>/ (list + sbatch file),
status.json (written every tick), finished.json. Run under job-watch: cluster/jhpce/chain_job.sh.
"""
import argparse
import datetime
import json
import os
import shlex
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "vendor"))
import chain_lib  # noqa: E402
import run_status_board as RSB  # noqa: E402

EXIT_DONE, EXIT_STALLED, EXIT_RUNNING = 0, 3, 10
MEM_MARGIN = 1.3           # memory asked = measured peak RSS x this, rounded up to whole GB


def now():
    return datetime.datetime.now().astimezone().isoformat(timespec="seconds")


def plan_batches(ready, submitted, task_final, min_batch):
    """ready {task: [tags that have a valid latent and no valid score]}; submitted: tags already sent; task_final {task: True
    when no more latents will arrive for the task}. -> {task: [tags]} to send now, in the order given."""
    out = {}
    for task, tags in ready.items():
        new = [t for t in tags if t not in submitted]
        if new and (len(new) >= min_batch or task_final.get(task)):
            out[task] = new
    return out


def admit(live, mem_gb, n, max_cpus, max_mem_gb):
    """Throttle of a new array of n tasks of mem_gb each, given the live arrays [(throttle, mem_gb)]: the arrays' throttles
    and their memory (throttle x mem_gb) summed over all live arrays may not exceed the budgets. 0 means wait."""
    cpu_left = max_cpus - sum(t for t, _ in live)
    mem_left = max_mem_gb - sum(t * m for t, m in live)
    return max(0, min(n, cpu_left, int(mem_left // mem_gb)))


def sbatch_text(job, n, throttle, size, score_repo, sha, lst, out, logs, constraint=None, script=None, env=()):
    head = ["#!/bin/bash", f"#SBATCH --job-name={job}", "#SBATCH --partition=shared", "#SBATCH --account=jhpce",
            f"#SBATCH --array=1-{n}%{throttle}", "#SBATCH --cpus-per-task=1", f"#SBATCH --mem={size['mem_gb']}G",
            f"#SBATCH --time={size['walltime']}", "#SBATCH --no-requeue", f"#SBATCH --output={logs}/slurm-%x_%A_%a.out"]
    if constraint:
        head.append(f"#SBATCH --constraint={constraint}")
    script = script or f"{score_repo}/cluster/jhpce/score_array.sh"       # --score-array: a wrapper that calls the repo's script
    exports = [f"export {k}={shlex.quote(v)}" for k, v in (e.split("=", 1) for e in env)]
    return "\n".join(head + ["set -euo pipefail", *exports, f"exec bash {script} {sha} {lst} {out}", ""])


def read_submissions(path):
    recs = [json.loads(line) for line in open(path)] if os.path.exists(path) else []
    sub = {r["batch"]: r for r in recs if r["event"] == "submitted"}
    return sub, [r for r in recs if r["event"] == "intent" and r["batch"] not in sub]


def append(path, rec):
    with open(path, "a") as f:
        f.write(json.dumps(rec) + "\n")
        f.flush()
        os.fsync(f.fileno())


def run(cmd):
    return subprocess.run(cmd, capture_output=True, text=True)


def live_tasks(job_ids):
    """{job id: {state: n tasks}} from squeue for explicit job ids only (read-only)."""
    out = {j: {} for j in job_ids}
    if not job_ids:
        return out
    r = run(["squeue", "-h", "-r", "-j", ",".join(job_ids), "-o", "%i|%T"])
    if r.returncode != 0 and "Invalid job id" not in r.stderr:
        raise RuntimeError(f"squeue failed: {r.stderr.strip()}")
    for line in r.stdout.splitlines():
        jid, state = line.split("|")
        d = out[jid.split("_")[0]]
        d[state] = d.get(state, 0) + 1
    return out


def measured_mem_gb(job_id):
    """Largest peak RSS (GB) over the finished tasks of an array, x MEM_MARGIN, rounded up; None unless every task COMPLETED
    and reported a peak."""
    r = run(["sacct", "-P", "-n", "-j", job_id, "--format", "JobID,State,MaxRSS"])
    peaks, states = [], []
    for line in r.stdout.splitlines():
        jid, state, rss = line.split("|")
        if ".batch" in jid:
            peaks.append(RSB.rs_mem_mb(rss)[0])
        elif ".extern" not in jid:
            states.append(state)
    if r.returncode or not peaks or None in peaks or set(states) != {"COMPLETED"}:
        return None
    return int(-(-max(peaks) * MEM_MARGIN // 1024))


def adopt_dangling(path, intents):
    """An intent without a 'submitted' record: sbatch may have run before a crash. Look the job up by its unique name."""
    for rec in intents:
        r = run(["sacct", "-X", "-n", "-P", "--name", rec["job"], "-S", rec["at"][:10], "--format", "JobID"])
        ids = sorted({x.split("_")[0] for x in r.stdout.split() if x.strip()})
        if len(ids) > 1:
            raise RuntimeError(f"{rec['job']}: {len(ids)} jobs carry this name {ids}: resolve by hand")
        if ids:
            append(path, dict(rec, event="submitted", job_id=ids[0], adopted=True, at=now()))


def make_batch(a, rows_by_tag, fit_detail, tags, size, throttle, bid):
    bdir = os.path.join(a.state_dir, "batches", bid)
    os.makedirs(os.path.join(bdir, "logs"), exist_ok=True)
    lines = []
    for t in tags:
        r = rows_by_tag[t]
        pre = os.path.join(a.prepped_dir, f"{r['task']}__{r['counts']}.h5ad")
        if not os.path.isfile(pre):
            raise SystemExit(f"{t}: prepped file {pre} is missing")
        meta = dict(latent_sha256=fit_detail[t]["sha256"], experiment=r["experiment"], task=r["task"], arm=r["arm"],
                    lam=r["lam"], cond=r["cond"], seed=r["seed"])
        lines.append("\t".join([t, os.path.join(a.fit_out, "latents", t + ".npz"), pre, json.dumps(meta)]))
    lst = os.path.join(bdir, "list.tsv")
    with open(lst, "w") as f:
        f.write("\n".join(lines) + "\n")
    sb = os.path.join(bdir, "score.sbatch")
    with open(sb, "w") as f:
        f.write(sbatch_text(f"wcdscore-{bid}", len(tags), throttle, size, a.score_repo, a.score_sha, lst, a.score_out,
                            os.path.join(bdir, "logs"), a.constraint, a.score_array, a.score_env))
    return sb


def tick(a, rows, ck, sizes):
    """One pass. Returns (status dict, True when work is still in progress or about to be submitted)."""
    sub_path = os.path.join(a.state_dir, "submissions.jsonl")
    sub, intents = read_submissions(sub_path)
    if intents:
        adopt_dangling(sub_path, intents)
        sub, intents = read_submissions(sub_path)
    rows_by_tag = {r["tag"]: r for r in rows}
    mpath = os.path.join(a.state_dir, "measured.json")
    measured = json.load(open(mpath)) if os.path.exists(mpath) else {}
    fit = {r["tag"]: ck.fit(r) for r in rows}
    score = {r["tag"]: ck.score(r) for r in rows}
    ck.save()
    fit_detail = {t: f[2] for t, f in fit.items() if f and f[0] and f[2]["scorable"]}
    settled = {t for t in fit_detail
               if time.time() - os.path.getmtime(os.path.join(a.fit_out, "latents", t + ".npz")) >= a.settle_s}
    live = live_tasks(sorted(b["job_id"] for b in sub.values()))
    live_batches = [b for b in sub.values() if live[b["job_id"]]]
    in_flight = {t for b in live_batches for t in b["tags"]}
    final = {t for t, s in score.items() if s and s[0]}                 # a valid score, or a recorded fit failure
    sent = {t for b in sub.values() for t in b["tags"]}
    failed = sorted(t for t in sent if t not in final and t not in in_flight)
    fit_jobs_live = any(live_tasks(a.fit_jobs).values()) if a.fit_jobs else True
    tasks = sorted({r["task"] for r in rows})
    task_final = {k: (not fit_jobs_live) or all(fit[r["tag"]] and fit[r["tag"]][0] for r in rows if r["task"] == k)
                  for k in tasks}
    ready = {k: [r["tag"] for r in rows if r["task"] == k and r["tag"] in settled and r["tag"] not in final] for k in tasks}
    # A task whose memory was not measured sends one small probe batch first; its measured peak sets the size of the rest.
    for b in sub.values():
        if b["task"] not in measured and not sizes[b["task"]].get("measured") and not live[b["job_id"]]:
            gb = measured_mem_gb(b["job_id"])
            if gb:
                measured[b["task"]] = dict(mem_gb=gb, basis=f"measured peak RSS of job {b['job_id']} x {MEM_MARGIN}")
                with open(mpath, "w") as f:
                    json.dump(measured, f)
    size_of = {k: dict(v, **measured.get(k, {}), measured=bool(v.get("measured") or k in measured)) for k, v in sizes.items() if not k.startswith("_")}
    probing = {t for t in tasks if not size_of[t]["measured"] and any(b["task"] == t for b in sub.values())}
    todo = plan_batches({k: v for k, v in ready.items() if k not in probing}, sent, task_final, a.min_batch)
    now_sent = []
    for task, tags in todo.items():
        size = size_of[task]
        if not size["measured"]:
            tags = tags[:a.probe_n]
        throttle = admit([(b["throttle"], b["mem_gb"]) for b in live_batches], size["mem_gb"], len(tags), a.max_cpus,
                         a.max_mem_gb)
        if throttle < 1:
            continue
        bid = f"{task}-{datetime.datetime.now().strftime('%Y%m%dT%H%M%S')}-{len(sub) + len(now_sent)}"
        rec = dict(batch=bid, job=f"wcdscore-{bid}", task=task, tags=tags, throttle=throttle, mem_gb=size["mem_gb"],
                   walltime=size["walltime"], score_sha=a.score_sha, at=now())
        sb = make_batch(a, rows_by_tag, fit_detail, tags, size, throttle, bid)
        if a.dry_run:
            now_sent.append(dict(rec, sbatch=sb, job_id="dry-run"))
            continue
        append(sub_path, dict(rec, event="intent"))
        r = run(["sbatch", "--parsable", sb])
        if r.returncode != 0:
            raise RuntimeError(f"sbatch {sb} failed ({r.returncode}): {r.stderr.strip()}; not retried")
        job_id = r.stdout.strip().split(";")[0]
        append(sub_path, dict(rec, event="submitted", job_id=job_id, at=now()))
        live_batches.append(dict(rec, job_id=job_id))
        now_sent.append(dict(rec, job_id=job_id))
    status = dict(at=now(), expected=len(rows), fit_valid=sum(1 for f in fit.values() if f and f[0]), final=len(final),
                  submitted_total=len(sent) + (0 if a.dry_run else 0), in_flight=len(in_flight), failed_scores=failed,
                  live_arrays=len(live_batches), cpus_running=sum(live[b["job_id"]].get("RUNNING", 0) for b in sub.values()),
                  cpus_bound=sum(b["throttle"] for b in live_batches), fit_jobs_live=fit_jobs_live,
                  submitted_now=[(s["batch"], len(s["tags"]), s["throttle"]) for s in now_sent],
                  fit_missing=[r["tag"] for r in rows if not (fit[r["tag"]] and fit[r["tag"]][0])],
                  waiting_for_measurement=sorted(probing))
    busy = bool(live_batches or now_sent or todo)       # a task waiting for its probe is not busy: nothing can start it but the probe
    return status, busy


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    for k in ("stage-repo", "score-repo", "manifest", "manifest-sha256", "fit-out", "score-out", "prepped-dir", "state-dir"):
        ap.add_argument("--" + k, required=True)
    ap.add_argument("--experiments", nargs="+", required=True)
    ap.add_argument("--tags-files", nargs="*", default=[])
    ap.add_argument("--expect-device", required=True, help="substring of the GPU name every latent was fitted on")
    ap.add_argument("--sizes", default=os.path.join(HERE, "score_sizes.json"))
    ap.add_argument("--fit-jobs", nargs="*", default=[], help="Slurm job ids of the fits (explicit); none = fits assumed live")
    ap.add_argument("--max-cpus", type=int, default=200)
    ap.add_argument("--max-mem-gb", type=int, default=2000, help="below the shared QOS cap of 2.5 TB")
    ap.add_argument("--min-batch", type=int, default=10)
    ap.add_argument("--probe-n", type=int, default=3, help="latents of the first batch of a task whose memory is not measured")
    ap.add_argument("--settle-s", type=float, default=60, help="a latent is scored only when this old (the fitter is done writing)")
    ap.add_argument("--constraint", help="one Slurm feature, e.g. znver2 (same CPU family for all scores)")
    ap.add_argument("--score-array", help="array-task script to run instead of <score-repo>/cluster/jhpce/score_array.sh (same arguments)")
    ap.add_argument("--score-env", nargs="*", default=[], metavar="NAME=VALUE", help="exported in every array task before the script")
    ap.add_argument("--interval", type=float, default=300)
    ap.add_argument("--max-ticks", type=int, default=0)
    ap.add_argument("--max-hours", type=float, default=0)
    ap.add_argument("--dry-run", action="store_true", help="write batch files, submit nothing, record nothing")
    ap.add_argument("--check-finished", action="store_true", help="exit 0 iff state/finished.json says done")
    a = ap.parse_args(argv)
    fin_path = os.path.join(a.state_dir, "finished.json")
    if a.check_finished:
        return 0 if os.path.exists(fin_path) and json.load(open(fin_path))["ok"] else 1
    if not (a.fit_jobs or a.max_hours or a.max_ticks):
        raise SystemExit("give --fit-jobs (so the chain knows when fits are over) or --max-hours / --max-ticks")
    if any("=" not in e or e.startswith("=") for e in a.score_env):
        raise SystemExit("--score-env takes NAME=VALUE items")
    if not 1 <= a.max_cpus <= 200:
        raise SystemExit("--max-cpus must be 1..200: the project limit for scoring arrays is 200 CPUs held")
    os.makedirs(a.state_dir, exist_ok=True)
    a.score_out, a.state_dir, a.score_repo = (os.path.abspath(p) for p in (a.score_out, a.state_dir, a.score_repo))
    head = run(["git", "-C", a.score_repo, "rev-parse", "HEAD"])
    if head.returncode:
        raise SystemExit(f"{a.score_repo} is not a git checkout: {head.stderr.strip()}")
    a.score_sha = head.stdout.strip()
    with open(a.sizes) as f:
        sizes = json.load(f)
    rs = chain_lib.load_run_stage(a.stage_repo)
    rows = chain_lib.read_expected(rs, a.manifest, a.manifest_sha256, a.experiments, a.tags_files)
    missing = sorted({r["task"] for r in rows} - set(sizes))
    if missing:
        raise SystemExit(f"{a.sizes} has no size for tasks {missing}")
    ck = chain_lib.Checker(rs, a.fit_out, a.score_out, a.expect_device, os.path.join(a.state_dir, "validity_cache.json"))
    t0, n = time.time(), 0
    while True:
        status, busy = tick(a, rows, ck, sizes)
        n += 1
        tmp = os.path.join(a.state_dir, f"status.json.tmp{os.getpid()}")
        with open(tmp, "w") as f:
            json.dump(status, f, indent=1)
        os.replace(tmp, os.path.join(a.state_dir, "status.json"))
        print(json.dumps({k: v for k, v in status.items() if k != "fit_missing"}), flush=True)
        if status["final"] == len(rows):
            with open(fin_path, "w") as f:
                json.dump(dict(ok=True, at=now(), expected=len(rows)), f)
            return EXIT_DONE
        if not status["fit_jobs_live"] and not busy:
            with open(fin_path, "w") as f:
                json.dump(dict(ok=False, at=now(), expected=len(rows), final=status["final"],
                               failed_scores=status["failed_scores"], fit_missing=status["fit_missing"],
                               waiting_for_measurement=status["waiting_for_measurement"]), f)
            return EXIT_STALLED
        if (a.max_ticks and n >= a.max_ticks) or (a.max_hours and time.time() - t0 > a.max_hours * 3600):
            return EXIT_RUNNING
        time.sleep(a.interval)


if __name__ == "__main__":  # pragma: no cover  # entry point
    sys.exit(main())
