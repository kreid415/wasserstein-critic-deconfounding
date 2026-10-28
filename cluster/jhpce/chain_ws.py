#!/usr/bin/env python
"""Workstation backend of the fit-to-scoring chain: copy finished JHPCE pilot latents to this machine and score them here.

Used for the penalty-strength pilot while its scoring site is the workstation (PREREG section 9 item 6, case iii). Every
--interval seconds, for one task (default atac_large):
  1. list the task's latents (and failure records) on JHPCE that are not in the local mirror and are older than --min-age-min
     (a fit that is still writing is never copied);
  2. copy them (rsync) into <mirror>/_incoming, then compare the SHA-256 of every copied file with a hash computed on JHPCE by a
     1-CPU / 1-GB `srun` after the copy (a file that changed or broke in transit does not match and is copied again next tick);
  3. move the verified files into the mirror (latents/, models/, attempts/, status/);
  4. when anything is new, score the mirror's tags of the task with experiments/pilot5_workstation_scoring/run_pilot_scoring.sh (scripts/run_stage.py at the
     tag commit 10a1ca5, 3 scorers) unless another scoring run holds the lock (the four-dataset run keeps it): then try again
     next tick, and the copy goes on.
Ends: exit 0 when every expected tag has a valid score file or a failure record, 10 at --max-hours / --max-ticks (state kept;
run again; run it under job-watch), 2 on a usage or setup error. Every other failure raises (nothing is skipped silently).
State: <state-dir>/state.json (written every tick) and chain_ws.log.
"""
import argparse
import datetime
import hashlib
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
EXIT_DONE, EXIT_RUNNING, EXIT_USAGE = 0, 10, 2
BUSY = 75                       # flock -E: another scoring run holds the lock


def now():
    return datetime.datetime.now().astimezone().isoformat(timespec="seconds")


def read_tags(path):
    """Tags of a run list: first column, '#' lines and the 'tag' header skipped."""
    with open(path) as f:
        return [ln.split("\t")[0].strip() for ln in f if ln.strip() and not ln.startswith("#") and not ln.startswith("tag")]


def files_for(tag, failed=False):
    """Mirror-relative files of one tag: a fit has latent, model and attempts; a failure record has status and attempts."""
    return ([f"status/{tag}.json"] if failed else [f"latents/{tag}.npz", f"models/{tag}/model.pt"]) + [f"attempts/{tag}.jsonl"]


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def verify(local_sums, remote_sums):
    """({file: sha}, {file: sha}) -> (ok files, bad files). A file missing on either side is bad: nothing is assumed."""
    ok = sorted(f for f, s in local_sums.items() if remote_sums.get(f) == s)
    return ok, sorted((set(local_sums) | set(remote_sums)) - set(ok))


class Remote:
    def __init__(self, host, root, hash_prefix):
        self.host, self.root, self.hash_prefix = host, root, hash_prefix

    def run(self, cmd, stdin=None):
        argv = ["ssh", "-o", "BatchMode=yes", self.host, cmd] if self.host else ["bash", "-c", cmd]
        return subprocess.run(argv, input=stdin, check=True, capture_output=True, text=True).stdout

    def old_files(self, sub, pattern, min_age_min):
        out = self.run(f"cd {self.root} && find {sub} -maxdepth 1 -name '{pattern}' -mmin +{int(min_age_min)} -printf '%f\\n' 2>/dev/null || true")
        return set(out.split())

    def sums(self, files):
        out = self.run(f"cd {self.root} && {self.hash_prefix} xargs -d '\\n' sha256sum", stdin="\n".join(files) + "\n")
        return {ln.split(None, 1)[1].strip(): ln.split()[0] for ln in out.splitlines() if ln.strip()}

    def fetch(self, files, dest):
        os.makedirs(dest, exist_ok=True)
        lst = os.path.join(dest, ".files")
        with open(lst, "w") as f:
            f.write("\n".join(files) + "\n")
        src = f"{self.host}:{self.root}/" if self.host else f"{self.root}/"
        subprocess.run(["nice", "-n", "10", "rsync", "-a", "--partial", "--ignore-missing-args", f"--files-from={lst}", src, dest + "/"],
                       check=True, capture_output=True, text=True)


def local_state(mirror, tags):
    """(have {tag: 'fit'|'failed'}, scored set): what the mirror holds for the expected tags."""
    have, scored = {}, set()
    for t in tags:
        if os.path.exists(os.path.join(mirror, "latents", f"{t}.npz")):
            have[t] = "fit"
        elif os.path.exists(os.path.join(mirror, "status", f"{t}.json")):
            have[t] = "failed"
        if os.path.exists(os.path.join(mirror, "scores", f"{t}.csv")):
            scored.add(t)
    return have, scored


def tick(a, rem, tags, log):
    have, scored = local_state(a.mirror, tags)
    ready_fit = rem.old_files("latents", f"A1_{a.task}_*.npz", a.min_age_min)
    ready_fail = rem.old_files("status", f"A1_{a.task}_*.json", a.min_age_min)
    new = [(t, False) for t in tags if t not in have and f"{t}.npz" in ready_fit] + \
          [(t, True) for t in tags if t not in have and f"{t}.json" in ready_fail]
    installed, bad = [], []
    for i in range(0, len(new), a.batch):
        chunk = new[i:i + a.batch]
        files = [f for t, failed in chunk for f in files_for(t, failed)]
        rem.fetch(files, os.path.join(a.mirror, "_incoming"))
        local = {f: sha256_file(os.path.join(a.mirror, "_incoming", f)) for f in files}   # raises if a file did not arrive
        ok, nok = verify(local, rem.sums(files))
        for t, failed in chunk:
            fs = files_for(t, failed)
            if all(f in ok for f in fs):
                for f in fs:
                    os.makedirs(os.path.dirname(os.path.join(a.mirror, f)), exist_ok=True)
                    os.replace(os.path.join(a.mirror, "_incoming", f), os.path.join(a.mirror, f))
                installed.append(t)
            else:
                bad.append(t)
    log(f"copied and verified {len(installed)} tags; {len(bad)} failed verification (copied again next tick)")
    have, scored = local_state(a.mirror, tags)
    fits = sorted(t for t, k in have.items() if k == "fit")
    rc = None
    if a.after_jw and json.load(open(os.path.join(a.after_jw, "state.json")))["status"] != "done":
        log(f"scoring waits: job-watch run {a.after_jw} is not done")        # the four-dataset run needs the memory first
    elif fits and set(fits) - scored:
        sub = os.path.join(a.state_dir, "tags_present.tags")
        with open(sub, "w") as f:
            f.write("\n".join(fits) + "\n")
        env = dict(os.environ, TAGS_FILE=sub)
        rc = subprocess.run(["flock", "-n", "-E", str(BUSY), a.lock, "bash", a.score_script, a.task, str(a.workers), "--clear-stale-claims"],
                            env=env, stdout=open(os.path.join(a.state_dir, "score.log"), "a"), stderr=subprocess.STDOUT).returncode
        if rc not in (0, BUSY):
            raise RuntimeError(f"scoring run exited {rc} (see {a.state_dir}/score.log); not retried")
        log("scoring: lock held by another run, will retry" if rc == BUSY else "scoring run finished")
        have, scored = local_state(a.mirror, tags)
    done = sum(1 for t in tags if t in scored or have.get(t) == "failed")
    return dict(at=now(), expected=len(tags), mirrored=len(have), scored=len(scored), failed_fits=sum(k == "failed" for k in have.values()),
                final=done, installed=len(installed), bad=len(bad), score_rc=rc)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--task", default="atac_large")
    ap.add_argument("--tags-file", required=True, help="the task's full run list (378 tags)")
    ap.add_argument("--mirror", required=True, help="local mirror of the stage folder (latents/, models/, attempts/, scores/)")
    ap.add_argument("--state-dir", required=True)
    ap.add_argument("--host", default="jhpce", help="ssh host; '' = the remote folder is local (tests)")
    ap.add_argument("--remote-root", default="/fastscratch/myscratch/kreid/wcd/tier12/pilot5_jhpce")
    ap.add_argument("--hash-prefix", default="srun -p shared -c 1 --mem 1G -t 30", help="runs the remote sha256sum as a 1-CPU job")
    ap.add_argument("--score-script", default=os.path.join(REPO, "experiments", "pilot5_workstation_scoring", "run_pilot_scoring.sh"))
    ap.add_argument("--lock", default=None, help="flock file shared with the four-dataset run (default <mirror>/.ws_scoring.lock)")
    ap.add_argument("--after-jw", default=None, help="job-watch state dir; score only once that run is done (memory is shared)")
    ap.add_argument("--workers", type=int, default=3, help="scorers at once (large dataset: 3)")
    ap.add_argument("--min-age-min", type=float, default=20)
    ap.add_argument("--batch", type=int, default=40)
    ap.add_argument("--interval", type=float, default=900)
    ap.add_argument("--max-ticks", type=int, default=0)
    ap.add_argument("--max-hours", type=float, default=0)
    a = ap.parse_args(argv)
    a.lock = a.lock or os.path.join(a.mirror, ".ws_scoring.lock")
    if not os.path.isdir(a.mirror):
        print(f"mirror {a.mirror} does not exist", file=sys.stderr)
        return EXIT_USAGE
    os.makedirs(a.state_dir, exist_ok=True)
    tags = read_tags(a.tags_file)
    if len(tags) != len(set(tags)) or not tags:
        print(f"{a.tags_file}: {len(tags)} tags, duplicates or none", file=sys.stderr)
        return EXIT_USAGE
    rem = Remote(a.host, a.remote_root, a.hash_prefix)
    t0, n = time.time(), 0

    def log(msg):
        with open(os.path.join(a.state_dir, "chain_ws.log"), "a") as f:
            f.write(f"[{now()}] {msg}\n")
    while True:
        st = tick(a, rem, tags, log)
        n += 1
        with open(os.path.join(a.state_dir, "state.json.tmp"), "w") as f:
            json.dump(st, f)
        os.replace(os.path.join(a.state_dir, "state.json.tmp"), os.path.join(a.state_dir, "state.json"))
        if st["final"] == st["expected"]:
            return EXIT_DONE
        if (a.max_ticks and n >= a.max_ticks) or (a.max_hours and time.time() - t0 > a.max_hours * 3600):
            return EXIT_RUNNING
        time.sleep(a.interval)


if __name__ == "__main__":  # pragma: no cover  # entry point
    sys.exit(main())
