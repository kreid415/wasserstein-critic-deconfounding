#!/usr/bin/env python
"""Fail-loud stage runner: fit, and optionally score, every manifest row of a stage (docs/PREREG.md section 0).
Host-agnostic: the local RTX 3080 (fit + score) and JHPCE GPU jobs (--no-score; their latents are scored locally).

    python scripts/run_stage.py --manifest M.tsv --experiments A1 --tasks atac_small immune sim1 \
        --out-dir OUT --prepped-dir PREPPED --fit-python FIT_PY --score-python SCORE_PY \
        --r-home R_HOME --r-libs R_LIBS --expect-device "RTX 3080"

Outcome of each manifest row (docs/PREREG.md section 1, "Fit outcome"):
    scored            finite latent and one valid score row               OUT/latents/<tag>.npz, OUT/scores/<tag>.csv
    fitted            finite latent, not scored (final only with --no-score)
    diverged          the fitter exited fit_outcome.EXIT_DIVERGED and wrote OUT/status/<tag>.json (training loss
                      became non-finite)
    nonfinite_latent  the saved posterior mean has a non-finite value: this runner writes OUT/status/<tag>.json and
                      never scores the latent
    infrastructure    any other fitter / scorer failure (exit status, signal, timeout, output that breaks the fitter
                      or scorer contract): retried up to --max-attempts times in this run, then the stage stops (no new
                      work; running work finishes) and the runner exits EXIT_INFRA naming the log. Never recorded as
                      a failure.
Per-tag files (the source of truth; the views below are rebuilt from them):
    claims/<tag>/owner.json      atomic claim (mkdir) while a runner works on the tag: host, machine id, boot time,
                                 pid namespace, pid, process start, runner id, SLURM job id. Live claims of another
                                 runner are skipped; stale claims (owner verifiably dead) and foreign claims (liveness
                                 not checkable here: other host, container or sandbox) stop the preflight unless
                                 --clear-stale-claims / --clear-foreign-claims is given.
    attempts/<tag>.jsonl         append-only start / end record of every fit and score attempt (claim holder only)
    logs/{fit,score}/<tag>.a<N>.log   one log per attempt (N counts the starts recorded for the tag)
    quarantine/                  outputs of an attempt that broke the fitter or scorer contract, moved aside
Views (atomic rewrites):
    failures.csv                 tag, status, detail, ... of every OUT/status/*.json (prereg_rules.read_failures)
    ledger/<stage>.csv           one row per selected manifest row: state, attempts, timings, logs, host, device
    ledger/<stage>.json          counts per state, completion gate, kBET seed, git SHAs, devices, stop reason
    ledger/<stage>.done          written only when the completion gate passes (removed at the start of every run)
Scores are written by scripts/score_scib_native.py to OUT/scores/<tag>.csv (prereg_rules.read_scores(OUT/scores));
every row carries the kBET seed of the run and the SHA-256 of the latent it scores.
Completion gate: exit 0 only if every selected row is final (scored, diverged or nonfinite_latent; with --no-score
also fitted), none is both scored and failed and none is missing. Exit EXIT_PREFLIGHT (nothing ran), EXIT_INFRA,
EXIT_GATE, or 128 + signal after SIGINT / SIGTERM (running work is stopped and its claims released).
The runner itself needs numpy, pandas and scipy (prereg_rules); fits and scores run in --fit-python / --score-python.
"""
import argparse
import ast
import collections
import ctypes
import hashlib
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import time
import uuid
import zipfile

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import fit_outcome  # noqa: E402  (statuses, EXIT_DIVERGED, status records)
import fit_paper_config as FPC  # noqa: E402  (REQUIRED: the fitter's row contract)
import prereg_rules as PR  # noqa: E402  (BATCH_METRICS, bio_metrics_for, read_failures, read_scores)

FIT_SCRIPT = os.path.join(HERE, "fit_paper_config.py")
SCORE_SCRIPT = os.path.join(HERE, "score_scib_native.py")
EXIT_PREFLIGHT, EXIT_INFRA, EXIT_GATE = 4, 5, 6
FINAL = {True: ("scored", "diverged", "nonfinite_latent"), False: ("fitted", "scored", "diverged", "nonfinite_latent")}
NO_ADV_INPUT = ("none", "scvi_adv", "scanvi", "sysvi")      # arms that never read adv_input
LEDGER_COLS = ["tag", "experiment", "task", "arm", "lam", "cond", "seed", "state", "detail", "fit_attempts",
               "score_attempts", "fit_host", "fit_started", "fit_ended", "fit_wall_s", "fit_seconds", "score_started",
               "score_ended", "score_wall_s", "fit_log", "score_log", "device", "git_sha", "claimed_by"]
FAILURE_COLS = ["tag", "status", "detail", "experiment", "task", "arm", "lam", "cond", "seed", "epoch", "step",
                "host", "recorded_at"]
ENV_PROBE = ("import json, torch, scvi; print(json.dumps(dict(torch=torch.__version__, scvi=scvi.__version__, "
             "device=(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'))))")
SCORE_PROBE = ("import scib, rpy2.robjects; from rpy2.robjects.packages import importr; importr('kBET'); "
               "print('kBET stack OK')")


class StageError(RuntimeError):
    """The stage cannot run as specified (bad input, invalid output, refused preflight)."""


class InvalidOutput(StageError):
    """A file in OUT_DIR does not belong to its manifest row or breaks the fitter / scorer contract."""


def now_iso(t=None):
    return time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime(time.time() if t is None else t))


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic_write(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f"{path}.tmp{os.getpid()}"
    with open(tmp, "w") as f:
        f.write(text)
    os.replace(tmp, path)


def kbet_seed_default(path=SCORE_SCRIPT):
    """score_scib_native.KBET_SEED_DEFAULT, read without importing scanpy / scib (one definition)."""
    with open(path) as f:
        tree = ast.parse(f.read())
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and \
                getattr(node.targets[0], "id", None) == "KBET_SEED_DEFAULT":
            return int(ast.literal_eval(node.value))
    raise StageError(f"{path}: KBET_SEED_DEFAULT not found")


def thread_env(n):
    """Thread caps for one child process (the settings of cluster/jhpce/gate_gpu.sh and scripts/score_list.sh)."""
    env = {k: str(n) for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS")}
    env.update(KMP_AFFINITY="disabled", PYTHONWARNINGS="ignore", PYTHONUNBUFFERED="1")
    if n == 1:
        env["MKL_THREADING_LAYER"] = "SEQUENTIAL"
    return env


# ---- process identity (Linux /proc): is the owner of a claim still running? ----------------------------------------
def process_domain():
    """Where a pid identifies one process: hostname, /etc/machine-id (None if absent), boot time (/proc/stat btime)
    and pid namespace (/proc/self/ns/pid; containers and sandboxes have their own)."""
    machine = None
    if os.path.isfile("/etc/machine-id"):
        with open("/etc/machine-id") as f:
            machine = f.read().strip() or None
    btime = None
    with open("/proc/stat") as f:
        for line in f:
            if line.startswith("btime "):
                btime = int(line.split()[1])
    if btime is None:
        raise StageError("/proc/stat has no btime line")
    return dict(host=socket.gethostname(), machine_id=machine, boot_time=btime, pid_ns=os.readlink("/proc/self/ns/pid"))


def start_ticks(pid):
    """Start time of process `pid` in clock ticks since boot (/proc/<pid>/stat field 22); None if it is not running
    (no such process, or a zombie / dead process: state Z or X, field 3)."""
    try:
        with open(f"/proc/{int(pid)}/stat") as f:
            stat = f.read()
    except (FileNotFoundError, ProcessLookupError):   # no such process: that is the answer, not an error
        return None
    fields = stat[stat.rindex(")") + 2:].split()
    return None if fields[0] in ("Z", "X") else int(fields[19])


_LIBC = ctypes.CDLL(None, use_errno=True)
PR_SET_PDEATHSIG = 1


def child_setup(parent_pid):
    """preexec_fn: the kernel sends SIGKILL to the child when the runner dies, even by SIGKILL, so a killed runner
    leaves no orphaned fit or scorer writing into OUT_DIR (its claims are then verifiably stale)."""
    def setup():
        if _LIBC.prctl(PR_SET_PDEATHSIG, signal.SIGKILL) != 0:
            raise OSError(ctypes.get_errno(), "prctl(PR_SET_PDEATHSIG) failed")
        if os.getppid() != parent_pid:      # the runner died before prctl took effect
            os._exit(70)
    return setup


class Claims:
    """Atomic per-tag claims: OUT/claims/<tag>/ is created with mkdir (fails if it exists), then owner.json."""

    def __init__(self, out_dir, runner_id, stage):
        self.dir = os.path.join(out_dir, "claims")
        self.domain = process_domain()
        self.pid, self.ticks = os.getpid(), start_ticks(os.getpid())
        self.runner_id, self.stage = runner_id, stage
        self.held = set()

    def path(self, tag):
        return os.path.join(self.dir, tag)

    def owner(self, tag):
        """None: no claim; {}: a claim without an owner record; else the owner record."""
        if not os.path.isdir(self.path(tag)):
            return None
        f = os.path.join(self.path(tag), "owner.json")
        if not os.path.isfile(f):
            return {}
        with open(f) as fh:
            return json.load(fh)

    def classify(self, tag):
        """('free' | 'mine' | 'live' | 'stale' | 'foreign', owner).
        live    another runner whose process is running (same host, machine id, boot and pid namespace);
        stale   owner verifiably dead: same host and machine id, and the host rebooted since, or same boot and pid
                namespace and the process is gone (or the pid was reused); also a claim without an owner record
                after 60 s;
        foreign anything whose liveness cannot be checked from here: another host or machine id, a host without
                /etc/machine-id, another pid namespace (container / sandbox) of this host."""
        o = self.owner(tag)
        if o is None:
            return "free", None
        if not o:
            return ("stale" if time.time() - os.path.getmtime(self.path(tag)) > 60 else "live"), o
        if o.get("runner_id") == self.runner_id:
            return "mine", o
        d = self.domain
        if o.get("host") != d["host"] or d["machine_id"] is None or o.get("machine_id") != d["machine_id"]:
            return "foreign", o
        if o.get("boot_time") != d["boot_time"]:
            return "stale", o
        if o.get("pid_ns") != d["pid_ns"]:
            return "foreign", o
        return ("live" if start_ticks(o["pid"]) == o.get("start_ticks") else "stale"), o

    def acquire(self, tag, purpose):
        os.makedirs(self.dir, exist_ok=True)
        try:
            os.mkdir(self.path(tag))
        except FileExistsError:             # held by someone else: the caller classifies it
            return False
        rec = dict(tag=tag, **self.domain, pid=self.pid, start_ticks=self.ticks, runner_id=self.runner_id,
                   stage=self.stage, purpose=purpose, claimed_at=now_iso(), slurm_job_id=os.environ.get("SLURM_JOB_ID", ""))
        atomic_write(os.path.join(self.path(tag), "owner.json"), json.dumps(rec, indent=1))
        self.held.add(tag)
        return True

    def release(self, tag):
        o = self.owner(tag)
        if not o or o.get("runner_id") != self.runner_id:
            raise StageError(f"claim {self.path(tag)} is not held by this runner (owner {o}); not removed")
        shutil.rmtree(self.path(tag))
        self.held.discard(tag)

    def clear(self, tag):
        shutil.rmtree(self.path(tag))


# ---- manifest -----------------------------------------------------------------------------------------------------
def load_manifest(path):
    """The fitter's reading of the manifest (fit_paper_config.load_row): tab-separated, '#' comments, strings."""
    m = pd.read_csv(path, sep="\t", comment="#", dtype=str, keep_default_na=False)
    missing = [c for c in FPC.REQUIRED if c not in m.columns]
    if missing:
        raise StageError(f"{path}: manifest lacks columns {missing}")
    dup = m.tag[m.tag.duplicated()].tolist()
    if dup:
        raise StageError(f"{path}: duplicate tags {dup[:5]}")
    return m


def select_rows(m, experiments, tasks):
    unknown = sorted(set(experiments) - set(m.experiment))
    if unknown:
        raise StageError(f"experiments {unknown} have no manifest rows")
    sel = m[m.experiment.isin(experiments)]
    if tasks:
        unknown = sorted(set(tasks) - set(sel.task))
        if unknown:
            raise StageError(f"tasks {unknown} have no rows in experiments {sorted(experiments)}")
        sel = sel[sel.task.isin(tasks)]
    return sel.reset_index(drop=True)


def unfittable(row):
    """Placeholders that fit_paper_config.py / the training plan refuse (docs/PREREG.md section 0)."""
    out = []
    try:
        float(row["lam"])
    except ValueError:
        out.append(f"lam={row['lam']!r}")
    if row["arm"] not in NO_ADV_INPUT and row["adv_input"] not in ("mean", "sample"):
        out.append(f"adv_input={row['adv_input']!r}")
    if not row["zstd"].lstrip("-").isdigit():
        out.append(f"zstd={row['zstd']!r}")
    return out


# ---- outputs: one reading of the per-tag files for the preflight, the post-run checks, the ledger and the gate ----
class Outputs:
    def __init__(self, out_dir, kbet_seed, expect_device, allow_dirty):
        self.out, self.kbet_seed = out_dir, int(kbet_seed)
        self.expect_device, self.allow_dirty = expect_device, allow_dirty

    def latent_path(self, tag):
        return os.path.join(self.out, "latents", f"{tag}.npz")

    def model_path(self, tag):
        return os.path.join(self.out, "models", tag, "model.pt")

    def score_path(self, tag):
        return os.path.join(self.out, "scores", f"{tag}.csv")

    def status_path(self, tag):
        return fit_outcome.status_path(self.out, tag)

    def latent(self, row):
        """None if there is no latent; else dict(finite, n_nonfinite, n_cells_nonfinite, shape, sha256, git_sha,
        device, fit_seconds). InvalidOutput if it is unreadable, malformed or not this row's fit."""
        p = self.latent_path(row["tag"])
        if not os.path.exists(p):
            return None
        try:
            with np.load(p, allow_pickle=False) as d:
                missing = sorted({"z", "obs_names", "batch", "celltype", "config"} - set(d.files))
                if missing:
                    raise InvalidOutput(f"{p}: latent lacks {missing}")
                z, n_obs = d["z"], len(d["obs_names"])
                n_b, n_c, cfg = len(d["batch"]), len(d["celltype"]), json.loads(str(d["config"]))
        except (OSError, ValueError, zipfile.BadZipFile, json.JSONDecodeError) as e:
            raise InvalidOutput(f"{p}: unreadable latent ({e!r})") from e
        if z.ndim != 2 or z.dtype.kind != "f" or z.shape[1] != int(row["n_latent"]) or not z.shape[0] == n_obs == n_b == n_c:
            raise InvalidOutput(f"{p}: z {z.shape} {z.dtype} vs n_latent {row['n_latent']}, {n_obs} obs_names, "
                                f"{n_b} batch, {n_c} celltype")
        diff = [c for c in FPC.REQUIRED if str(cfg.get("row", {}).get(c)) != row[c]]
        if diff:
            raise InvalidOutput(f"{p}: fitted with other settings than the manifest row ({diff})")
        if self.expect_device not in str(cfg.get("gpu")):
            raise InvalidOutput(f"{p}: fitted on {cfg.get('gpu')!r}, expected {self.expect_device!r} (SI-17)")
        if cfg.get("git_dirty") is not False and not self.allow_dirty:
            raise InvalidOutput(f"{p}: fitted from a dirty or unknown checkout (git_dirty={cfg.get('git_dirty')})")
        bad = ~np.isfinite(z)
        return dict(finite=not bool(bad.any()), n_nonfinite=int(bad.sum()), n_cells_nonfinite=int(bad.any(1).sum()),
                    shape=list(z.shape), sha256=sha256_file(p), git_sha=cfg.get("git_sha"), device=cfg.get("gpu"),
                    fit_seconds=cfg.get("fit_seconds"))

    def status(self, row):
        p = self.status_path(row["tag"])
        if not os.path.exists(p):
            return None
        try:
            rec = fit_outcome.read_status(p, tag=row["tag"])
        except (ValueError, json.JSONDecodeError) as e:
            raise InvalidOutput(f"{p}: invalid status record ({e})") from e
        diff = [c for c in FPC.REQUIRED if str(rec["row"].get(c)) != row[c]]
        if diff:
            raise InvalidOutput(f"{p}: recorded for other settings than the manifest row ({diff})")
        return rec

    def score(self, row, latent_sha):
        """None if there is no score; else dict(score_seconds). InvalidOutput unless the file holds exactly one row
        of this tag, with this run's kBET seed, the SHA-256 of the current latent and finite, in-range required
        metrics (docs/PREREG.md section 1, 'Required values'; ranges as prereg_rules.outcomes)."""
        p = self.score_path(row["tag"])
        if not os.path.exists(p):
            return None
        s = pd.read_csv(p, dtype={"tag": str, "latent_sha256": str})
        batch, bio = PR.BATCH_METRICS, PR.bio_metrics_for(row["task"])
        missing = [c for c in ["tag", "kbet_seed", "latent_sha256"] + batch + bio if c not in s.columns]
        if len(s) != 1 or missing:
            raise InvalidOutput(f"{p}: {len(s)} rows, missing columns {missing}")
        r = s.iloc[0]
        problems = []
        if r["tag"] != row["tag"]:
            problems.append(f"tag {r['tag']!r}")
        if int(r["kbet_seed"]) != self.kbet_seed:
            problems.append(f"kbet_seed {r['kbet_seed']} (this run {self.kbet_seed})")
        if r["latent_sha256"] != latent_sha:
            problems.append("scores a different latent (latent_sha256 differs)")
        vals = {m: float(pd.to_numeric(r[m], errors="raise")) for m in batch + bio}
        nonfinite = [m for m, v in vals.items() if not np.isfinite(v)]
        out_b = [m for m in batch if np.isfinite(vals[m]) and not -1e-6 <= vals[m] <= 1 + 1e-6]
        out_c = [m for m in bio if np.isfinite(vals[m]) and not -1 - 1e-6 <= vals[m] <= 1 + 1e-6]
        if nonfinite:
            problems.append(f"non-finite required metrics {nonfinite} (a pipeline fault, not an outcome)")
        if out_b or out_c:
            problems.append(f"metrics outside the scIB range {out_b + out_c}")
        if problems:
            raise InvalidOutput(f"{p}: " + "; ".join(problems))
        return dict(score_seconds=float(r["score_seconds"]) if "score_seconds" in s.columns else None)

    def state(self, row, score_mode):
        """(state, info) from the files alone: scored | fitted | diverged | nonfinite_latent | needs_fit |
        needs_score | nonfinite_unrecorded. InvalidOutput for contradictory files (e.g. scored and failed)."""
        tag = row["tag"]
        st, lat, has_score = self.status(row), self.latent(row), os.path.exists(self.score_path(tag))
        if st is not None:
            if has_score:
                raise InvalidOutput(f"{tag}: both a failure record ({st['status']}) and a score")
            if st["status"] == "diverged" and lat is not None:
                raise InvalidOutput(f"{tag}: both a divergence record and a latent")
            if st["status"] == "nonfinite_latent" and (lat is None or lat["finite"]):
                raise InvalidOutput(f"{tag}: nonfinite_latent record but the latent is missing or finite")
            return st["status"], dict(status=st, latent=lat)
        if lat is None:
            if has_score:
                raise InvalidOutput(f"{tag}: a score without a latent")
            return "needs_fit", {}
        if not lat["finite"]:
            if has_score:
                raise InvalidOutput(f"{tag}: a score of a non-finite latent")
            return "nonfinite_unrecorded", dict(latent=lat)
        if has_score:
            return "scored", dict(latent=lat, score=self.score(row, lat["sha256"]))
        return ("needs_score" if score_mode else "fitted"), dict(latent=lat)


class Job:
    def __init__(self, kind, tag, attempt, popen, log, fh):
        self.kind, self.tag, self.attempt, self.popen, self.log, self.fh = kind, tag, attempt, popen, log, fh
        self.start, self.timed_out = time.time(), False


class Stage:
    def __init__(self, a, rows):
        self.a, self.score = a, not a.no_score
        self.rows = {r["tag"]: r for r in rows.to_dict("records")}
        self.order = list(self.rows)
        self.out = os.path.abspath(a.out_dir)
        self.key = "+".join(sorted(a.experiments)) + "__" + ("+".join(sorted(a.tasks)) if a.tasks else "all")
        self.runner_id = uuid.uuid4().hex
        self.claims = Claims(self.out, self.runner_id, self.key)
        self.files = Outputs(self.out, a.kbet_seed, a.expect_device, a.allow_dirty)
        self.fit_q, self.score_q = collections.deque(), collections.deque()
        self.running, self.tries = [], collections.Counter()
        self.mem = {}               # tag -> (state, detail) known to this run (ledger between gate evaluations)
        self.stop_reason, self.signal, self.infra_errors = None, None, []
        self.started = time.time()
        self.head = self.git_state()
        self.manifest_sha, self.snapshot = sha256_file(a.manifest), None   # rows were read from these bytes
        self.n_need_fit = 0

    # ---- small helpers ----------------------------------------------------------------------------------------
    def git_state(self):
        sha = subprocess.run(["git", "-C", REPO, "rev-parse", "HEAD"], check=True, capture_output=True,
                             text=True).stdout.strip()
        dirty = subprocess.run(["git", "-C", REPO, "status", "--porcelain", "--untracked-files=no"], check=True,
                               capture_output=True, text=True).stdout.strip() != ""
        return dict(git_sha=sha, git_dirty=dirty)

    def say(self, msg):
        print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

    def attempts_path(self, tag):
        return os.path.join(self.out, "attempts", f"{tag}.jsonl")

    def attempts(self, tag):
        p = self.attempts_path(tag)
        if not os.path.exists(p):
            return []
        with open(p) as f:
            return [json.loads(line) for line in f if line.strip()]

    def append_attempt(self, tag, rec):
        if tag not in self.claims.held:
            raise StageError(f"{tag}: attempt records are written by the claim holder only")
        os.makedirs(os.path.dirname(self.attempts_path(tag)), exist_ok=True)
        with open(self.attempts_path(tag), "a") as f:
            f.write(json.dumps(rec) + "\n")

    def quarantine(self, tag, kind, attempt, paths):
        """Move the outputs of a contract-breaking attempt aside (kept as evidence; a rerun starts clean)."""
        moved = []
        for p in paths:
            if os.path.exists(p):
                dest = os.path.join(self.out, "quarantine", f"{tag}.{kind}.a{attempt}.{os.path.basename(p)}")
                os.makedirs(os.path.dirname(dest), exist_ok=True)
                os.replace(p, dest)
                moved.append(dest)
        return moved

    def manifest_snapshot(self):
        """OUT/manifests/<sha256[:16]>.tsv: the exact manifest bytes this run selected its rows from; the fitter reads
        this copy, so editing the original during a run cannot change a fit (and the copy is the provenance)."""
        if self.snapshot is None:
            with open(self.a.manifest, "rb") as f:
                data = f.read()
            sha = hashlib.sha256(data).hexdigest()
            if sha != self.manifest_sha:
                raise StageError(f"{self.a.manifest} changed since the rows were selected")
            path = os.path.join(self.out, "manifests", f"{sha[:16]}.tsv")
            if not os.path.exists(path):
                os.makedirs(os.path.dirname(path), exist_ok=True)
                tmp = f"{path}.tmp{os.getpid()}"
                with open(tmp, "wb") as f:
                    f.write(data)
                os.replace(tmp, path)
            elif sha256_file(path) != sha:
                raise StageError(f"{path} does not hold the manifest it is named after")
            self.snapshot = path
        return self.snapshot

    # ---- preflight ----------------------------------------------------------------------------------------------
    def preflight(self, launching):
        """Refuse before any work: placeholders, missing inputs, dirty checkout, uncleared stale / foreign claims
        (not in --report-only), missing R paths, invalid or contradictory outputs. Queues the remaining work."""
        problems, a = [], self.a
        for tag, r in self.rows.items():
            why = unfittable(r)
            if why:
                problems.append(f"{tag}: placeholder(s) {why}: resolve the rule first (docs/PREREG.md section 0)")
        for t in sorted({(r["task"], r["counts"]) for r in self.rows.values()}):
            p = os.path.join(a.prepped_dir, f"{t[0]}__{t[1]}.h5ad")
            if not os.path.isfile(p):
                problems.append(f"prepped file {p} missing")
        if launching and self.head["git_dirty"] and not a.allow_dirty:
            problems.append(f"repo {REPO} has uncommitted changes: results must come from committed code "
                            f"(--allow-dirty for tests only)")
        claims = collections.defaultdict(list)
        for tag in self.order:
            kind, owner = self.claims.classify(tag)
            if kind != "free":
                claims[kind].append((tag, owner))
        for kind, flag in (("stale", a.clear_stale_claims), ("foreign", a.clear_foreign_claims)):
            for tag, owner in claims[kind]:
                who = {k: (owner or {}).get(k) for k in ("host", "pid", "slurm_job_id", "claimed_at", "runner_id")}
                if flag and launching:
                    self.claims.clear(tag)
                    self.say(f"[claim] cleared {kind} claim {tag} {who}")
                elif a.report_only:
                    self.mem[tag] = (f"{kind}_claim", f"{who}")
                else:
                    problems.append(f"{kind} claim {self.claims.path(tag)} {who}: "
                                    f"{'--clear-stale-claims' if kind == 'stale' else '--clear-foreign-claims'} "
                                    f"to clear it")
        for tag, owner in claims["live"]:
            self.mem[tag] = ("claimed_elsewhere", f"live claim of runner {owner.get('runner_id')} pid {owner.get('pid')}")
        if not a.no_score and launching:
            for v, name in ((a.r_home, "--r-home / R_HOME"), (a.r_libs, "--r-libs / R_LIBS")):
                if not v or not os.path.isdir(v):
                    problems.append(f"scoring needs {name} (an existing directory), got {v!r}")
        states = collections.Counter()
        for tag in self.order:
            r = self.rows[tag]
            try:
                st, info = self.files.state(r, self.score)
            except InvalidOutput as e:
                problems.append(f"invalid output: {e}")
                continue
            states[st] += 1
            if tag in self.mem:
                continue
            self.mem[tag] = (st, "")
            self.n_need_fit += st == "needs_fit"
            if st in ("needs_fit", "nonfinite_unrecorded"):    # the latter is recorded once claimed (recheck)
                self.fit_q.append(tag)
            elif st == "needs_score":
                self.score_q.append(tag)
        if problems:
            raise StageError("preflight refused:\n  " + "\n  ".join(problems))
        return states

    def check_envs(self):
        """The fit env must see the expected device (only if a fit is needed: scoring harvested latents runs on
        another host than the fits); the scoring env must import scib, rpy2 and R's kBET."""
        a = self.a
        if self.n_need_fit:
            self.check_fit_env()
        if self.score:
            env = dict(os.environ, R_HOME=a.r_home, R_LIBS=a.r_libs, CUDA_VISIBLE_DEVICES="", **thread_env(1))
            p = subprocess.run([a.score_python, "-c", SCORE_PROBE], env=env, capture_output=True, text=True,
                               timeout=600)
            if p.returncode != 0:
                raise StageError(f"scoring env probe failed (exit {p.returncode}): {p.stderr.strip()[-500:]}")
            self.say(f"[env] score {a.score_python}: {p.stdout.strip().splitlines()[-1]}")

    def check_fit_env(self):
        a = self.a
        env = dict(os.environ, **thread_env(a.fit_threads))
        p = subprocess.run([a.fit_python, "-c", ENV_PROBE], env=env, capture_output=True, text=True, timeout=600)
        if p.returncode != 0:
            raise StageError(f"fit env probe failed (exit {p.returncode}): {p.stderr.strip()[-500:]}")
        probe = json.loads(p.stdout.strip().splitlines()[-1])
        if a.expect_device not in probe["device"]:
            raise StageError(f"fit env {a.fit_python} sees {probe['device']!r}, expected {a.expect_device!r}")
        self.fit_env = probe
        self.say(f"[env] fit {a.fit_python}: torch {probe['torch']}, scvi {probe['scvi']}, device {probe['device']}")

    # ---- launching ----------------------------------------------------------------------------------------------
    def claim(self, tag, purpose):
        """Hold the claim of `tag`; False (and the reason in the ledger) if another runner holds it."""
        if tag in self.claims.held:
            return True
        if self.claims.acquire(tag, purpose):
            return True
        kind, owner = self.claims.classify(tag)
        self.mem[tag] = ("claimed_elsewhere" if kind == "live" else f"{kind}_claim",
                         f"claim appeared during the run: {kind} {(owner or {}).get('host')} pid {(owner or {}).get('pid')}")
        self.say(f"[claim] {tag} skipped: {self.mem[tag][1]}")
        return False

    def recheck(self, tag, kind):
        """After claiming: the files may have moved on since the preflight (another runner, a resumed run)."""
        st, info = self.files.state(self.rows[tag], self.score)
        if st == f"needs_{kind}":
            return True
        if st in FINAL[self.score]:
            self.finish(tag, st, "already final when claimed")
        elif st == "nonfinite_unrecorded":
            self.record_nonfinite(tag, info["latent"])
            self.finish(tag, "nonfinite_latent", "non-finite latent recorded when claimed")
        elif st == "needs_score":
            self.score_q.append(tag)                     # claim kept
        elif st == "needs_fit":
            if not self.n_need_fit:
                self.check_fit_env()
                self.n_need_fit = 1
            self.fit_q.append(tag)                       # the latent is gone: refit; claim kept
        else:
            raise StageError(f"{tag}: unexpected state {st!r} on recheck")
        return False

    def next_attempt(self, tag, kind):
        return 1 + sum(1 for rec in self.attempts(tag) if rec["kind"] == kind and rec["event"] == "start")

    def launch(self, tag, kind):
        a, r = self.a, self.rows[tag]
        n = self.next_attempt(tag, kind)
        log = os.path.join(self.out, "logs", kind, f"{tag}.a{n}.log")
        if os.path.exists(log):
            raise StageError(f"{log} exists although {n - 1} {kind} attempts are recorded: attempts/{tag}.jsonl and "
                             f"logs/ disagree")
        os.makedirs(os.path.dirname(log), exist_ok=True)
        if kind == "fit":
            env = dict(os.environ, MANIFEST=self.manifest_snapshot(), TAG=tag, PREPPED_DIR=a.prepped_dir,
                       OUT_DIR=self.out, WCD_SRC=a.wcd_src, **thread_env(a.fit_threads))
            cmd = [a.fit_python, FIT_SCRIPT]
        else:
            lat = self.files.latent(r)
            meta = dict(latent_sha256=lat["sha256"], experiment=r["experiment"], task=r["task"], arm=r["arm"],
                        lam=r["lam"], cond=r["cond"], seed=r["seed"])
            os.makedirs(os.path.join(self.out, "scores"), exist_ok=True)
            env = dict(os.environ, PREPPED=os.path.join(a.prepped_dir, f"{r['task']}__{r['counts']}.h5ad"),
                       NPZ=self.files.latent_path(tag), TAG=tag, META=json.dumps(meta),
                       OUT_CSV=self.files.score_path(tag), KBET_SEED=str(a.kbet_seed), R_HOME=a.r_home,
                       R_LIBS=a.r_libs, CUDA_VISIBLE_DEVICES="", **thread_env(1))
            cmd = [a.score_python, SCORE_SCRIPT]
        fh = open(log, "w")
        popen = subprocess.Popen(cmd, env=env, stdout=fh, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                 start_new_session=True, preexec_fn=child_setup(os.getpid()))
        job = Job(kind, tag, n, popen, log, fh)
        self.running.append(job)
        self.tries[(kind, tag)] += 1
        self.append_attempt(tag, dict(event="start", kind=kind, attempt=n, runner_id=self.runner_id,
                                      host=socket.gethostname(), pid=popen.pid, start=now_iso(job.start), log=log,
                                      git_sha=self.head["git_sha"], slurm_job_id=os.environ.get("SLURM_JOB_ID", "")))
        self.mem[tag] = ("fitting" if kind == "fit" else "scoring", f"attempt {n}")
        self.say(f"[{kind}] start {tag} (attempt {n})")

    def fill(self):
        a = self.a
        n_fit = sum(j.kind == "fit" for j in self.running)
        while n_fit < a.fit_lanes and self.fit_q and self.stop_reason is None:
            tag = self.fit_q.popleft()
            if self.claim(tag, "fit") and self.recheck(tag, "fit"):
                self.launch(tag, "fit")
                n_fit += 1
        if not self.score:
            return
        cap = a.score_workers_during if (self.fit_q or n_fit) else a.score_workers_after
        n_score = sum(j.kind == "score" for j in self.running)
        while n_score < cap and self.score_q and self.stop_reason is None:
            tag = self.score_q.popleft()
            if self.claim(tag, "score") and self.recheck(tag, "score"):
                self.launch(tag, "score")
                n_score += 1

    # ---- results ------------------------------------------------------------------------------------------------
    def reap(self):
        for job in list(self.running):
            rc = job.popen.poll()
            timeout = self.a.fit_timeout_s if job.kind == "fit" else self.a.score_timeout_s
            if rc is None and timeout and time.time() - job.start > timeout:
                os.killpg(job.popen.pid, signal.SIGKILL)
                rc = job.popen.wait()
                job.timed_out = True
            if rc is None:
                continue
            job.fh.close()
            self.running.remove(job)
            (self.fit_done if job.kind == "fit" else self.score_done)(job, rc)

    def end_record(self, job, rc, outcome, detail):
        end = time.time()
        self.append_attempt(job.tag, dict(event="end", kind=job.kind, attempt=job.attempt, runner_id=self.runner_id,
                                          host=socket.gethostname(), end=now_iso(end),
                                          wall_s=round(end - job.start, 1), returncode=rc, outcome=outcome,
                                          detail=detail, log=job.log))

    def fit_done(self, job, rc):
        tag, r = job.tag, self.rows[job.tag]
        try:
            if job.timed_out:
                outcome, detail = "infrastructure", f"fit timed out after {self.a.fit_timeout_s} s"
            elif rc == 0:
                lat = self.files.latent(r)
                if lat is None:
                    raise InvalidOutput("fitter exited 0 without a latent")
                if not os.path.isfile(self.files.model_path(tag)):
                    raise InvalidOutput(f"fitter exited 0 without {self.files.model_path(tag)}")
                if os.path.exists(self.files.status_path(tag)):
                    raise InvalidOutput("fitter exited 0 and wrote a status record")
                if lat["finite"]:
                    outcome, detail = "fitted", f"fit {lat['fit_seconds']} s on {lat['device']}"
                else:
                    self.record_nonfinite(tag, lat)
                    outcome, detail = "nonfinite_latent", self.files.status(r)["detail"]
            elif rc == fit_outcome.EXIT_DIVERGED:
                st = self.files.status(r)
                if st is None or st["status"] != "diverged":
                    raise InvalidOutput(f"fitter exit {rc} (EXIT_DIVERGED) without a divergence record")
                if os.path.exists(self.files.latent_path(tag)):
                    raise InvalidOutput("fitter recorded a divergence and wrote a latent")
                outcome, detail = "diverged", st["detail"]
            else:
                sig = f" ({signal.Signals(-rc).name})" if rc < 0 else ""
                outcome, detail = "infrastructure", f"fitter exit {rc}{sig}"
        except InvalidOutput as e:
            moved = self.quarantine(tag, "fit", job.attempt, [self.files.latent_path(tag), self.files.status_path(tag),
                                                               os.path.join(self.out, "models", tag)])
            outcome, detail = "infrastructure", f"fitter contract broken: {e}; moved aside {moved}"
        self.end_record(job, rc, outcome, detail)
        if outcome == "fitted":
            self.say(f"[fit] {tag} fitted: {detail}")
            if self.score:
                self.mem[tag] = ("queued_score", "")
                self.score_q.append(tag)
            else:
                self.finish(tag, "fitted", detail)
        elif outcome in fit_outcome.FAILURE_STATUSES:
            self.finish(tag, outcome, detail)
        else:
            self.infra(job, detail)

    def score_done(self, job, rc):
        tag, r = job.tag, self.rows[job.tag]
        try:
            if job.timed_out:
                raise InvalidOutput(f"scorer timed out after {self.a.score_timeout_s} s")
            if rc != 0:
                sig = f" ({signal.Signals(-rc).name})" if rc < 0 else ""
                raise InvalidOutput(f"scorer exit {rc}{sig}")
            lat = self.files.latent(r)
            if lat is None or not lat["finite"]:
                raise InvalidOutput("the latent disappeared or changed while it was scored")
            sc = self.files.score(r, lat["sha256"])
            if sc is None:
                raise InvalidOutput("scorer exited 0 without a score CSV")
            outcome, detail = "scored", f"score {sc['score_seconds']} s"
        except InvalidOutput as e:
            moved = self.quarantine(tag, "score", job.attempt, [self.files.score_path(tag)])
            outcome, detail = "infrastructure", f"{e}; moved aside {moved}"
        self.end_record(job, rc, outcome, detail)
        if outcome == "scored":
            self.finish(tag, "scored", detail)
        else:
            self.infra(job, detail)

    def record_nonfinite(self, tag, lat):
        fit_outcome.write_status(self.out, tag, "nonfinite_latent", row=self.rows[tag],
                                 detail=f"{lat['n_nonfinite']} of {lat['shape'][0] * lat['shape'][1]} posterior-mean "
                                        f"values are non-finite ({lat['n_cells_nonfinite']} cells)",
                                 latent_sha256=lat["sha256"], device=lat["device"], git_sha=lat["git_sha"])

    def finish(self, tag, state, detail):
        self.mem[tag] = (state, detail)
        if tag in self.claims.held:
            self.claims.release(tag)
        self.say(f"[done] {tag}: {state} {detail}")

    def infra(self, job, detail):
        key = (job.kind, job.tag)
        self.say(f"[INFRA] {job.kind} {job.tag} attempt {job.attempt}: {detail}; log {job.log}")
        if self.tries[key] < self.a.max_attempts and self.stop_reason is None:
            (self.fit_q if job.kind == "fit" else self.score_q).append(job.tag)      # retried later; claim kept
            self.mem[job.tag] = (f"queued_{job.kind}", f"retry after: {detail}")
            return
        self.infra_errors.append(dict(tag=job.tag, kind=job.kind, attempts=self.tries[key], detail=detail,
                                      log=job.log))
        if self.stop_reason is None:
            self.stop_reason = (f"{job.kind} of {job.tag} failed {self.tries[key]} time(s) with infrastructure "
                                f"errors; last: {detail}; log {job.log}")
            self.say(f"[STOP] {self.stop_reason}. No new work is started; running work finishes.")
        self.mem[job.tag] = ("infrastructure_error", f"{detail} (log {job.log})")
        if job.tag in self.claims.held:
            self.claims.release(job.tag)

    # ---- main loop ----------------------------------------------------------------------------------------------
    def on_signal(self, signum, frame):
        self.signal = signum

    def run(self):
        """Returns None when the queues are drained (or the stage stopped), 128 + signal after SIGINT / SIGTERM."""
        for sig in (signal.SIGTERM, signal.SIGINT):
            signal.signal(sig, self.on_signal)
        last_view = 0.0
        try:
            while True:
                if self.signal is not None:
                    return self.interrupt(signal.Signals(self.signal).name)
                self.reap()
                if self.signal is not None:
                    return self.interrupt(signal.Signals(self.signal).name)
                self.fill()
                if not self.running and (self.stop_reason is not None or not (self.fit_q or self.score_q)):
                    self.release_queued()
                    return None
                if time.time() - last_view > self.a.view_interval_s:
                    self.write_views(final=False)
                    last_view = time.time()
                time.sleep(self.a.poll_s)
        except BaseException as e:      # a runner error: stop the children and release claims, then re-raise
            self.interrupt(f"runner error {e!r}")
            raise

    def release_queued(self):
        """Nothing runs any more: release the claims of rows still queued (only after a stop) so the next run
        does not see them as stale."""
        for tag in sorted(self.claims.held):
            self.claims.release(tag)
            self.mem[tag] = ("pending", f"not started: {self.stop_reason}")

    def interrupt(self, why):
        """Stop every running job (SIGTERM, then SIGKILL after --kill-grace-s), record it as interrupted and release
        this runner's claims; the rows stay unfinished and are rerun by the next run."""
        self.say(f"[stage] {why}: stopping {len(self.running)} running job(s)")
        for job in self.running:
            if job.popen.poll() is None:
                os.killpg(job.popen.pid, signal.SIGTERM)
        deadline = time.time() + self.a.kill_grace_s
        for job in self.running:
            try:
                job.popen.wait(timeout=max(0.1, deadline - time.time()))
            except subprocess.TimeoutExpired:     # did not stop on SIGTERM: kill it
                os.killpg(job.popen.pid, signal.SIGKILL)
                job.popen.wait()
            job.fh.close()
            self.end_record(job, job.popen.returncode, "interrupted", f"stopped by the runner: {why}")
            self.mem[job.tag] = ("pending", f"interrupted ({why})")
        self.running = []
        for tag in sorted(self.claims.held):
            self.claims.release(tag)
            if self.mem.get(tag, ("",))[0] not in FINAL[self.score]:
                self.mem[tag] = ("pending", f"interrupted ({why})")
        self.write_views(final=False)
        return None if self.signal is None else 128 + self.signal

    # ---- views and the completion gate -------------------------------------------------------------------------
    def write_failures(self):
        """failures.csv from every status record in OUT/status (validated with prereg_rules.read_failures)."""
        d = os.path.join(self.out, "status")
        names = sorted(f for f in os.listdir(d) if f.endswith(".json")) if os.path.isdir(d) else []
        rows = []
        for f in names:
            rec = fit_outcome.read_status(os.path.join(d, f))
            rows.append({**{k: rec["row"].get(k, "") for k in ("experiment", "task", "arm", "lam", "cond", "seed")},
                         **{k: rec.get(k, "") for k in ("tag", "status", "detail", "epoch", "step", "host",
                                                         "recorded_at")}})
        df = pd.DataFrame(rows, columns=FAILURE_COLS)
        path = os.path.join(self.out, "failures.csv")
        tmp = f"{path}.tmp{os.getpid()}"
        df.to_csv(tmp, index=False)
        F = PR.read_failures(tmp)
        if len(F) != len(names):
            raise StageError(f"failures table has {len(F)} rows for {len(names)} status records")
        os.replace(tmp, path)
        return F

    def file_states(self):
        out = {}
        for tag in self.order:
            try:
                st, info = self.files.state(self.rows[tag], self.score)
                out[tag] = (st, info)
            except InvalidOutput as e:
                out[tag] = ("invalid", dict(error=str(e)))
        return out

    def gate(self, states):
        final = FINAL[self.score]
        counts = collections.Counter(st for st, _ in states.values())
        not_final = {t: st for t, (st, _) in states.items() if st not in final}
        both = sorted(t for t, (st, info) in states.items() if st == "invalid" and "both" in info["error"])
        missing = sorted(t for t, st in not_final.items() if st in ("needs_fit", "needs_score"))
        F = self.write_failures()
        failed = {t for t, (st, _) in states.items() if st in fit_outcome.FAILURE_STATUSES}
        absent = sorted(failed - set(F.tag))
        scored = sorted(t for t, (st, _) in states.items() if st == "scored")
        dup = []
        if scored:
            S, _ = PR.read_scores(os.path.join(self.out, "scores"))
            n = S.tag.value_counts()
            dup = sorted(t for t in scored if n.get(t, 0) != 1)
        n_final = sum(counts[s] for s in final)
        if n_final + len(not_final) != len(self.order):
            raise StageError(f"gate count mismatch: {n_final} final + {len(not_final)} not final != {len(self.order)}")
        ok = not not_final and not absent and not dup
        return dict(ok=ok, n_rows=len(self.order), counts=dict(sorted(counts.items())), n_final=n_final,
                    not_final=dict(sorted(not_final.items())), missing=missing, both=both,
                    failures_without_row=absent, scored_not_once=dup)

    def ledger_rows(self, states):
        out = []
        for tag in self.order:
            r = self.rows[tag]
            st, info = states[tag] if states is not None else (self.mem.get(tag, ("pending", ""))[0], {})
            detail = self.mem.get(tag, ("", ""))[1] if states is None else (
                info.get("error") or (info.get("status") or {}).get("detail", "") or self.mem.get(tag, ("", ""))[1])
            if states is not None and st in ("needs_fit", "needs_score") and tag in self.mem and \
                    self.mem[tag][0] in ("infrastructure_error", "claimed_elsewhere", "stale_claim", "foreign_claim"):
                st, detail = self.mem[tag]
            recs = self.attempts(tag)
            last = {}
            for rec in recs:
                last.setdefault(rec["kind"], {})[rec["event"]] = rec
            fs, fe = last.get("fit", {}).get("start", {}), last.get("fit", {}).get("end", {})
            ss, se = last.get("score", {}).get("start", {}), last.get("score", {}).get("end", {})
            lat = info.get("latent") or {}
            owner = self.claims.owner(tag) or {}
            out.append(dict(tag=tag, experiment=r["experiment"], task=r["task"], arm=r["arm"], lam=r["lam"],
                            cond=r["cond"], seed=r["seed"], state=st, detail=detail,
                            fit_attempts=sum(1 for x in recs if x["kind"] == "fit" and x["event"] == "start"),
                            score_attempts=sum(1 for x in recs if x["kind"] == "score" and x["event"] == "start"),
                            fit_host=fs.get("host", ""), fit_started=fs.get("start", ""), fit_ended=fe.get("end", ""),
                            fit_wall_s=fe.get("wall_s", ""), fit_seconds=lat.get("fit_seconds", ""),
                            score_started=ss.get("start", ""), score_ended=se.get("end", ""),
                            score_wall_s=se.get("wall_s", ""), fit_log=fs.get("log", ""), score_log=ss.get("log", ""),
                            device=lat.get("device", ""), git_sha=lat.get("git_sha", ""),
                            claimed_by=(f"{owner.get('host')}:{owner.get('pid')}" if owner else "")))
        return out

    def write_views(self, final):
        """failures.csv, ledger and summary; final=True re-reads every file and runs the completion gate."""
        if not final:
            self.write_failures()           # (the gate writes it when final)
        states = self.file_states() if final else None
        rows = self.ledger_rows(states)
        d = os.path.join(self.out, "ledger")
        df = pd.DataFrame(rows, columns=LEDGER_COLS)
        tmp = os.path.join(d, f"{self.key}.csv.tmp{os.getpid()}")
        os.makedirs(d, exist_ok=True)
        df.to_csv(tmp, index=False)
        os.replace(tmp, os.path.join(d, f"{self.key}.csv"))
        summary = dict(stage=self.key, experiments=sorted(self.a.experiments), tasks=sorted(self.a.tasks or []),
                       manifest=os.path.abspath(self.a.manifest), manifest_sha256=self.manifest_sha,
                       manifest_snapshot=self.snapshot,
                       out_dir=self.out, score=self.score, kbet_seed=int(self.a.kbet_seed),
                       expect_device=self.a.expect_device, runner_id=self.runner_id, host=socket.gethostname(),
                       runner_git=self.head, started_at=now_iso(self.started), updated_at=now_iso(),
                       counts=dict(collections.Counter(x["state"] for x in rows)), stop_reason=self.stop_reason,
                       infrastructure_errors=self.infra_errors)
        if final:
            g = self.gate(states)
            summary["gate"] = g
            summary["latent_git_shas"] = sorted({str(x["git_sha"]) for x in rows if x["git_sha"]})
            summary["devices"] = sorted({str(x["device"]) for x in rows if x["device"]})
            if len(summary["latent_git_shas"]) > 1 and not self.a.allow_mixed_sha:
                g["ok"] = False
                g["mixed_git_sha"] = summary["latent_git_shas"]
            atomic_write(os.path.join(d, f"{self.key}.json"), json.dumps(summary, indent=1))
            return g
        atomic_write(os.path.join(d, f"{self.key}.json"), json.dumps(summary, indent=1))
        return None


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0], allow_abbrev=False)   # flags exactly as written
    ap.add_argument("--manifest", required=True, help="builder / resolved / extension manifest (TSV)")
    ap.add_argument("--experiments", nargs="+", required=True, help="experiment ids of the stage, e.g. A1")
    ap.add_argument("--tasks", nargs="+", default=None, help="task filter (default: every task of the stage)")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--prepped-dir", required=True)
    ap.add_argument("--fit-python", required=True, help="python of the fit env (scvi-tools)")
    ap.add_argument("--score-python", default=None, help="python of the scoring env (scib, rpy2, kBET)")
    ap.add_argument("--r-home", default=os.environ.get("R_HOME"))
    ap.add_argument("--r-libs", default=os.environ.get("R_LIBS"))
    ap.add_argument("--expect-device", required=True,
                    help="substring of torch.cuda.get_device_name(0) every fit must run on ('cpu' without CUDA)")
    ap.add_argument("--wcd-src", default=os.path.join(REPO, "src"))
    ap.add_argument("--fit-lanes", type=int, default=8)
    ap.add_argument("--fit-threads", type=int, default=1)
    ap.add_argument("--score-workers-during", type=int, default=4, help="scorers while fits are queued or running")
    ap.add_argument("--score-workers-after", type=int, default=12, help="scorers once every fit has finished")
    ap.add_argument("--no-score", action="store_true", help="fit only (JHPCE jobs); latents are scored elsewhere")
    ap.add_argument("--kbet-seed", type=int, default=kbet_seed_default())
    ap.add_argument("--max-attempts", type=int, default=2, help="attempts per tag and kind in this run")
    ap.add_argument("--fit-timeout-s", type=float, default=0, help="0 = none")
    ap.add_argument("--score-timeout-s", type=float, default=0, help="0 = none")
    ap.add_argument("--clear-stale-claims", action="store_true")
    ap.add_argument("--clear-foreign-claims", action="store_true",
                    help="clear claims held by other hosts: only after checking that their runners are gone")
    ap.add_argument("--allow-dirty", action="store_true", help="tests only: accept an uncommitted checkout")
    ap.add_argument("--allow-mixed-sha", action="store_true", help="accept latents fitted at different commits")
    ap.add_argument("--dry-run", action="store_true", help="preflight and plan; start nothing")
    ap.add_argument("--report-only", action="store_true", help="rebuild failures.csv, ledger and gate; start nothing")
    ap.add_argument("--poll-s", type=float, default=1.0)
    ap.add_argument("--view-interval-s", type=float, default=30.0)
    ap.add_argument("--kill-grace-s", type=float, default=30.0)
    return ap


def validate_args(a):
    positive = dict(fit_lanes=a.fit_lanes, fit_threads=a.fit_threads, max_attempts=a.max_attempts)
    if not a.no_score:
        positive.update(score_workers_during=a.score_workers_during, score_workers_after=a.score_workers_after)
    bad = [k for k, v in positive.items() if v < 1]
    if bad:
        raise StageError(f"{bad} must be >= 1")
    if a.dry_run and a.report_only:
        raise StageError("--dry-run and --report-only are exclusive")
    if not a.no_score and not a.score_python and not a.report_only:
        raise StageError("scoring needs --score-python (or --no-score)")
    for p in [a.fit_python] + ([a.score_python] if a.score_python else []):
        if not (os.path.isfile(p) and os.access(p, os.X_OK)):
            raise StageError(f"{p} is not an executable file")
    a.prepped_dir, a.wcd_src = os.path.abspath(a.prepped_dir), os.path.abspath(a.wcd_src)


def main(argv=None):
    """Exit status: 0 gate passed; EXIT_PREFLIGHT refused before any work; EXIT_INFRA stopped after infrastructure
    errors; EXIT_GATE incomplete; 128 + signal interrupted. A runner error during the run stops the children,
    releases the claims and propagates (traceback, exit 1)."""
    a = build_parser().parse_args(argv)
    launching = not (a.dry_run or a.report_only)
    try:                                                 # ---- preflight: refuse before any work
        validate_args(a)
        rows = select_rows(load_manifest(a.manifest), a.experiments, a.tasks)
        stage = Stage(a, rows)
        done = os.path.join(stage.out, "ledger", f"{stage.key}.done")
        if launching and os.path.exists(done):
            os.remove(done)                              # a marker is valid only for the run that wrote it
        states = stage.preflight(launching)
        n_fit, n_score = len(stage.fit_q), len(stage.score_q)
        stage.say(f"[stage] {stage.key}: {len(rows)} rows {dict(sorted(states.items()))}; to run: {n_fit} fits"
                  + (f", up to {n_fit + n_score} scores" if stage.score else " (no scoring)") + f"; out {stage.out}")
        if a.dry_run:
            return 0
        if launching:
            stage.check_envs()
    except StageError as e:
        print(f"[stage] REFUSED: {e}", file=sys.stderr, flush=True)
        return EXIT_PREFLIGHT
    if launching:
        rc = stage.run()
        if rc is not None:
            stage.say(f"[stage] interrupted; exit {rc}")
            return rc
    g = stage.write_views(final=True)
    stage.say(f"[gate] {'PASS' if g['ok'] else 'FAIL'}: {g['n_final']}/{g['n_rows']} final; counts {g['counts']}")
    if stage.stop_reason is not None:
        stage.say(f"[stage] STOPPED: {stage.stop_reason}")
        return EXIT_INFRA
    if not g["ok"]:
        for k in ("not_final", "both", "failures_without_row", "scored_not_once", "mixed_git_sha"):
            if g.get(k):
                items = list(g[k].items()) if isinstance(g[k], dict) else g[k]
                stage.say(f"[gate] {k} ({len(items)}): {items[:50]}")
        return EXIT_GATE
    if launching:
        atomic_write(done, json.dumps(dict(gate=g, at=now_iso(), runner_id=stage.runner_id), indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
