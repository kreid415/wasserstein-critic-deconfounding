#!/usr/bin/env python
"""Helpers of cluster/jhpce/prod_fit_job.sh (JHPCE production fit job). Each subcommand prints one JSON record and
exits non-zero on refusal, so the job script can stop before any fit. Needs numpy-free stdlib + pandas only.

  inputs    manifest and tags-file SHA-256 against the expected values; every tag in the manifest once; the tags'
            experiments / tasks equal --experiments / --tasks                               (exit 8 on refusal)
  slot      acquire / release one of N concurrency slots (atomic mkdir under --slots-dir); a slot whose owner job is
            no longer live in Slurm is reclaimed; live slots must hold disjoint tasks       (exit 7 on refusal)
  claims    run_stage claims (OUT/claims/<tag>/owner.json) on this job's tags: clearable only if the owning Slurm
            job is terminal (or the claim has no owner record for > 60 s, the runner's stale rule)  (exit 7)
  versions  env_versions.py --kind fit record against expected_fit_versions.json (modules, CUDA, GPU name) (exit 2)
  deadline  epoch seconds at which the stop guard fires: Slurm end time (squeue %L) - margin (exit 9 if unknown)
  summary   job summary from the runner's ledger JSON/CSV and the job facts
  pack      harvest of this job's tags: per-file SHA-256 manifest, one tar, parts <= --part-mb MiB, SHA256SUMS
Slurm states are read with sacct (fallback squeue); a state that cannot be read counts as live (fail safe).
"""
import argparse
import glob
import hashlib
import io
import json
import os
import re
import shutil
import socket
import subprocess
import sys
import tarfile
import time

EXIT_REFUSED_CONC, EXIT_REFUSED_INPUTS, EXIT_ENV = 7, 8, 9
LIVE = {"PENDING", "RUNNING", "REQUEUED", "REQUEUE_FED", "REQUEUE_HOLD", "RESIZING", "SUSPENDED", "COMPLETING",
        "CONFIGURING", "STAGE_OUT", "SIGNALING", "STOPPED", "UNKNOWN"}


def out(rec, code=0):
    print(json.dumps(rec, indent=1, sort_keys=True), flush=True)
    sys.exit(code)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def read_tags(path):
    tags = []
    with open(path) as f:
        for line in f:
            t = line.split("#", 1)[0].strip()
            if t:
                tags.append(t)
    return tags


def job_state(job_id):
    """Slurm state of a job id ('UNKNOWN' if neither sacct nor squeue answers)."""
    job_id = str(job_id).strip()
    if not re.fullmatch(r"\d+(_\d+)?", job_id):
        return "UNKNOWN"
    errors = []
    for cmd, pick in ((["sacct", "-X", "-n", "-P", "-j", job_id, "-o", "State"], lambda s: s.split()[0]),   # 'CANCELLED by 1'
                      (["squeue", "-h", "-j", job_id, "-o", "%T"], lambda s: s)):
        try:
            p = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        except (OSError, subprocess.TimeoutExpired) as e:
            errors.append(f"{cmd[0]}: {e!r}")
            continue
        s = [l.strip() for l in p.stdout.splitlines() if l.strip()]
        if p.returncode == 0 and s:
            return pick(s[0])
        errors.append(f"{cmd[0]}: exit {p.returncode}, {len(s)} lines")
    print(f"[slurm] state of job {job_id} unknown ({'; '.join(errors)}): counted as live", file=sys.stderr)
    return "UNKNOWN"


# ---- inputs ------------------------------------------------------------------------------------------------------
def cmd_inputs(a):
    import pandas as pd
    problems = []
    msha, tsha = sha256(a.manifest), sha256(a.tags)
    if msha != a.manifest_sha256:
        problems.append(f"manifest {a.manifest} sha256 {msha} != expected {a.manifest_sha256}")
    if tsha != a.tags_sha256:
        problems.append(f"tags file {a.tags} sha256 {tsha} != expected {a.tags_sha256}")
    tags = read_tags(a.tags)
    dup = sorted({t for t in tags if tags.count(t) > 1})
    if dup:
        problems.append(f"duplicate tags in {a.tags}: {dup[:5]}")
    M = pd.read_csv(a.manifest, sep="\t", comment="#", dtype=str, keep_default_na=False)
    if M.tag.duplicated().any():
        problems.append(f"{a.manifest}: duplicate manifest tags")
    sel = M[M.tag.isin(tags)]
    missing = sorted(set(tags) - set(sel.tag))
    if missing:
        problems.append(f"{len(missing)} tags not in the manifest: {missing[:5]}")
    if not tags:
        problems.append("empty tags file")
    ex, ta = set(sel.experiment), set(sel.task)
    if ex != set(a.experiments):
        problems.append(f"tags span experiments {sorted(ex)}, job declares {sorted(a.experiments)}")
    if ta != set(a.tasks):
        problems.append(f"tags span tasks {sorted(ta)}, job declares {sorted(a.tasks)}")
    counts = {f"{t}/{arm}": int(n) for (t, arm), n in sel.groupby(["task", "arm"]).size().items()}
    out(dict(ok=not problems, problems=problems, manifest_sha256=msha, tags_sha256=tsha, n_tags=len(tags),
             counts=counts), 0 if not problems else EXIT_REFUSED_INPUTS)


# ---- concurrency slots -------------------------------------------------------------------------------------------
def cmd_slot(a):
    os.makedirs(a.slots_dir, exist_ok=True)
    if a.action == "release":
        rel, unreadable = [], []
        for d in sorted(glob.glob(os.path.join(a.slots_dir, "slot*"))):
            try:
                o = json.load(open(os.path.join(d, "owner.json")))
            except (OSError, ValueError) as e:
                unreadable.append(f"{os.path.basename(d)}: {e!r}")
                continue
            if str(o.get("slurm_job_id")) == str(a.job):
                shutil.rmtree(d)
                rel.append(os.path.basename(d))
        out(dict(released=rel, unreadable=unreadable), 0 if rel else 1)
    me = set(a.tasks)
    held, free = [], []
    for i in range(1, a.n + 1):
        d = os.path.join(a.slots_dir, f"slot{i}")
        if not os.path.isdir(d):
            free.append(d)
            continue
        try:
            o = json.load(open(os.path.join(d, "owner.json")))
        except (OSError, ValueError) as e:             # no / partial owner record: liveness unknown (fail safe)
            print(f"[slot] {d}: owner record unreadable ({e!r})", file=sys.stderr)
            o = {}
        if str(o.get("slurm_job_id")) == str(a.job):
            out(dict(slot=os.path.basename(d), reused=True, owner=o))
        st = job_state(o.get("slurm_job_id", "")) if o else "UNKNOWN"
        if not o and time.time() - os.path.getmtime(d) > 60:
            st = "NO_OWNER_RECORD"
        if st in LIVE:
            held.append(dict(slot=os.path.basename(d), state=st, **{k: o.get(k) for k in ("slurm_job_id", "host", "tasks")}))
        else:                                       # owner job ended: move the record aside, the slot is free
            dst = os.path.join(a.slots_dir, "ended", f"{os.path.basename(d)}.{int(time.time())}.{os.getpid()}")
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            try:
                os.rename(d, dst)
                free.append(d)
            except OSError:
                held.append(dict(slot=os.path.basename(d), state="RACE"))
    overlap = [h for h in held if me & set(h.get("tasks") or [])]
    if overlap:
        out(dict(ok=False, reason="a live production job holds overlapping tasks", held=held), EXIT_REFUSED_CONC)
    for d in free:
        try:
            os.mkdir(d)
        except FileExistsError:     # fl: allow FL102 another job took this slot first; try the next one
            continue
        rec = dict(slurm_job_id=str(a.job), host=socket.gethostname(), tasks=sorted(me), acquired_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"))
        tmp = os.path.join(d, f".owner.{os.getpid()}")
        with open(tmp, "w") as f:
            json.dump(rec, f)
        os.replace(tmp, os.path.join(d, "owner.json"))
        out(dict(ok=True, slot=os.path.basename(d), owner=rec, held_by_others=held))
    out(dict(ok=False, reason=f"all {a.n} concurrency slots are held by live jobs", held=held), EXIT_REFUSED_CONC)


# ---- claims ------------------------------------------------------------------------------------------------------
def cmd_claims(a):
    tags = read_tags(a.tags)
    found, clearable, refuse = [], [], []
    for t in tags:
        d = os.path.join(a.out_dir, "claims", t)
        if not os.path.isdir(d):
            continue
        f = os.path.join(d, "owner.json")
        o = json.load(open(f)) if os.path.isfile(f) else None
        if o is None:
            age = time.time() - os.path.getmtime(d)
            rec = dict(tag=t, owner=None, age_s=round(age))
            (clearable if age > 60 else refuse).append(rec)
            found.append(rec)
            continue
        jid = str(o.get("slurm_job_id", ""))
        st = job_state(jid) if jid else "NO_SLURM_JOB_ID"
        rec = dict(tag=t, slurm_job_id=jid, state=st, host=o.get("host"), claimed_at=o.get("claimed_at"))
        found.append(rec)
        if jid == str(a.job):
            refuse.append(dict(rec, why="claim carries this job's id before the runner started"))
        elif jid and st not in LIVE:
            clearable.append(rec)
        else:
            refuse.append(rec)
    flags = ["--clear-foreign-claims", "--clear-stale-claims"] if found and not refuse else []
    out(dict(ok=not refuse, n_claims=len(found), clearable=clearable, refuse=refuse, flags=flags),
        0 if not refuse else EXIT_REFUSED_CONC)


# ---- versions ----------------------------------------------------------------------------------------------------
def cmd_versions(a):
    got, exp = json.load(open(a.got)), json.load(open(a.expected))
    bad = [f"{k}: {got.get('modules', {}).get(k)!r} != {v!r}" for k, v in exp["modules"].items()
           if got.get("modules", {}).get(k) != v]
    for k in ("python", "torch_cuda", "cudnn"):
        if k in exp and got.get(k) != exp[k]:
            bad.append(f"{k}: {got.get(k)!r} != {exp[k]!r}")
    if got.get("cuda_available") is not True:
        bad.append("cuda_available is not true")
    if exp.get("gpu_contains", "") not in str(got.get("gpu", "")):
        bad.append(f"gpu {got.get('gpu')!r} does not contain {exp.get('gpu_contains')!r}")
    out(dict(ok=not bad, problems=bad, gpu=got.get("gpu"), torch=got.get("modules", {}).get("torch")), 0 if not bad else 2)


# ---- deadline ----------------------------------------------------------------------------------------------------
def parse_slurm_time(s):
    """'D-HH:MM:SS' | 'HH:MM:SS' | 'MM:SS' -> seconds; None for UNLIMITED / NOT_SET / unparsable."""
    m = re.fullmatch(r"(?:(\d+)-)?(?:(\d+):)?(\d+):(\d+)", s.strip())
    if not m:
        return None
    d, h, mi, se = (int(x) if x else 0 for x in m.groups())
    return ((d * 24 + h) * 60 + mi) * 60 + se


def cmd_deadline(a):
    left, src = None, None
    if a.time_left_s is not None:
        left, src = a.time_left_s, "--time-left-s"
    elif os.environ.get("SLURM_JOB_ID"):
        try:
            p = subprocess.run(["squeue", "-h", "-j", os.environ["SLURM_JOB_ID"], "-o", "%L"], capture_output=True,
                               text=True, timeout=60)
            left, src = parse_slurm_time(p.stdout.strip().splitlines()[0]) if p.stdout.strip() else None, "squeue %L"
        except (OSError, subprocess.TimeoutExpired, IndexError):
            left = None
    if left is None:
        out(dict(ok=False, reason="time left of this job is unknown (squeue %L unreadable, no --time-left-s)"), EXIT_ENV)
    now = int(time.time())
    if left <= a.margin_s + a.min_run_s:
        out(dict(ok=False, reason=f"only {left} s left, stop margin {a.margin_s} s + minimum run {a.min_run_s} s"), EXIT_ENV)
    out(dict(ok=True, now=now, time_left_s=left, source=src, margin_s=a.margin_s, deadline=now + left - a.margin_s))


# ---- summary -----------------------------------------------------------------------------------------------------
def stage_key_from_log(log):
    """The runner prints '[stage] <key>: N rows ...' before work; the key names ledger/<key>.json."""
    try:
        with open(log) as f:
            for line in f:
                m = re.search(r"\[stage\] (\S+): \d+ rows", line)
                if m:
                    return m.group(1)
    except OSError as e:
        print(f"[summary] runner log {log} unreadable ({e!r}); using the default stage key", file=sys.stderr)
    return None


def cmd_summary(a):
    key = stage_key_from_log(a.runner_log) or a.default_key
    led = os.path.join(a.out_dir, "ledger", f"{key}.json")
    rec = dict(stage_key=key, runner_exit=a.rc, stopped_by=a.stopped_by or None, ledger=led if os.path.isfile(led) else None,
               done_marker=os.path.isfile(os.path.join(a.out_dir, "ledger", f"{key}.done")))
    if os.path.isfile(led):
        L = json.load(open(led))
        rec.update({k: L.get(k) for k in ("counts", "gate", "stop_reason", "git_shas", "devices") if k in L})
    csvp = os.path.join(a.out_dir, "ledger", f"{key}.csv")
    if os.path.isfile(csvp):
        import pandas as pd
        C = pd.read_csv(csvp, dtype=str, keep_default_na=False)
        rec["state_by_task"] = {f"{t}/{s}": int(n) for (t, s), n in C.groupby(["task", "state"]).size().items()}
    for kv in a.fact or []:
        k, v = kv.split("=", 1)
        rec[k] = v
    with open(a.out, "w") as f:
        json.dump(rec, f, indent=1, sort_keys=True)
    out(rec)


# ---- pack --------------------------------------------------------------------------------------------------------
def harvest_files(out_dir, tags, key):
    rel = []
    for t in tags:
        for p in (f"latents/{t}.npz", f"status/{t}.json", f"attempts/{t}.jsonl"):
            if os.path.isfile(os.path.join(out_dir, p)):
                rel.append(p)
        for root in (f"models/{t}",):
            for dp, _, fns in os.walk(os.path.join(out_dir, root)):
                rel += [os.path.relpath(os.path.join(dp, fn), out_dir) for fn in fns]
        rel += [os.path.relpath(p, out_dir) for p in glob.glob(os.path.join(out_dir, "logs", "*", f"{t}.a*.log"))]
        for q in glob.glob(os.path.join(out_dir, "quarantine", f"{t}.*")):      # run_stage: <tag>.<kind>.a<N>.<name>
            if os.path.isfile(q):
                rel.append(os.path.relpath(q, out_dir))
            for dp, _, fns in os.walk(q):
                rel += [os.path.relpath(os.path.join(dp, fn), out_dir) for fn in fns]
    for p in [f"ledger/{key}.csv", f"ledger/{key}.json", f"ledger/{key}.done", "failures.csv"] + \
            [os.path.relpath(p, out_dir) for p in glob.glob(os.path.join(out_dir, "manifests", "*.tsv"))]:
        if os.path.isfile(os.path.join(out_dir, p)):
            rel.append(p)
    return sorted(set(rel))


def cmd_pack(a):
    tags = read_tags(a.tags)
    key = stage_key_from_log(a.runner_log) or a.default_key
    os.makedirs(a.dest, exist_ok=True)
    rel = harvest_files(a.out_dir, tags, key)
    job_files = sorted(os.path.relpath(os.path.join(dp, fn), a.job_dir) for dp, _, fns in os.walk(a.job_dir)
                       for fn in fns if not os.path.relpath(dp, a.job_dir).split(os.sep)[0] in ("repo",))
    entries = [(os.path.join(a.out_dir, r), f"out/{r}") for r in rel] + \
              [(os.path.join(a.job_dir, r), f"job/{r}") for r in job_files]
    files = [dict(path=arc, bytes=os.path.getsize(src), sha256=sha256(src)) for src, arc in entries]
    base = f"{a.stage}__job{a.job}"
    man = dict(stage=a.stage, slurm_job_id=str(a.job), stage_key=key, tags_file=os.path.basename(a.tags),
               tags_sha256=sha256(a.tags), n_tags=len(tags), created=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
               host=socket.gethostname(), n_files=len(files), total_bytes=sum(f["bytes"] for f in files), files=files,
               n_latents=sum(f["path"].startswith("out/latents/") for f in files))
    man_bytes = json.dumps(man, indent=1, sort_keys=True).encode()
    tar_path = os.path.join(a.dest, f"{base}.tar")
    with tarfile.open(tar_path, "w", format=tarfile.PAX_FORMAT) as tf:
        ti = tarfile.TarInfo("harvest_manifest.json")
        ti.size, ti.mtime = len(man_bytes), int(time.time())
        tf.addfile(ti, io.BytesIO(man_bytes))
        for src, arc in entries:
            tf.add(src, arcname=arc, recursive=False)
    tar_sha = sha256(tar_path)
    part_b = a.part_mb * 1024 * 1024
    parts = []
    with open(tar_path, "rb") as f:
        i = 0
        while True:
            chunk = f.read(part_b)
            if not chunk:
                break
            pp = os.path.join(a.dest, f"{base}.tar.part{i:03d}")
            with open(pp, "wb") as g:
                g.write(chunk)
            parts.append(dict(name=os.path.basename(pp), bytes=len(chunk), sha256=hashlib.sha256(chunk).hexdigest()))
            i += 1
    os.remove(tar_path)
    with open(os.path.join(a.dest, f"{base}.harvest_manifest.json"), "wb") as f:
        f.write(man_bytes)
    sums = [f"{p['sha256']}  {p['name']}" for p in parts] + [f"{tar_sha}  {base}.tar",
                                                              f"{hashlib.sha256(man_bytes).hexdigest()}  {base}.harvest_manifest.json"]
    with open(os.path.join(a.dest, f"{base}.SHA256SUMS"), "w") as f:
        f.write("\n".join(sums) + "\n")
    rec = dict(ok=True, dest=a.dest, base=base, tar_sha256=tar_sha, parts=parts, n_files=len(files),
               total_bytes=man["total_bytes"], n_latents=man["n_latents"], stage_key=key)
    with open(os.path.join(a.dest, f"{base}.harvest.json"), "w") as f:
        json.dump(rec, f, indent=1, sort_keys=True)
    out(rec)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("inputs")
    for k in ("manifest", "manifest-sha256", "tags", "tags-sha256"):
        p.add_argument(f"--{k}", required=True)
    p.add_argument("--experiments", nargs="+", required=True)
    p.add_argument("--tasks", nargs="+", required=True)
    p = sub.add_parser("slot")
    p.add_argument("action", choices=["acquire", "release"])
    p.add_argument("--slots-dir", required=True)
    p.add_argument("--n", type=int, default=2)
    p.add_argument("--job", required=True)
    p.add_argument("--tasks", nargs="*", default=[])
    p = sub.add_parser("claims")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--tags", required=True)
    p.add_argument("--job", required=True)
    p = sub.add_parser("versions")
    p.add_argument("--got", required=True)
    p.add_argument("--expected", required=True)
    p = sub.add_parser("deadline")
    p.add_argument("--margin-s", type=int, required=True)
    p.add_argument("--min-run-s", type=int, default=1800)
    p.add_argument("--time-left-s", type=int, default=None)
    p = sub.add_parser("summary")
    for k in ("out-dir", "runner-log", "default-key", "out"):
        p.add_argument(f"--{k}", required=True)
    p.add_argument("--rc", type=int, required=True)
    p.add_argument("--stopped-by", default="")
    p.add_argument("--fact", action="append")
    p = sub.add_parser("pack")
    for k in ("out-dir", "tags", "runner-log", "default-key", "job-dir", "dest", "stage", "job"):
        p.add_argument(f"--{k}", required=True)
    p.add_argument("--part-mb", type=int, default=250)
    a = ap.parse_args()
    dict(inputs=cmd_inputs, slot=cmd_slot, claims=cmd_claims, versions=cmd_versions, deadline=cmd_deadline, summary=cmd_summary, pack=cmd_pack)[a.cmd](a)


if __name__ == "__main__":
    main()
