"""scripts/run_stage.py: classification, claims, retry bound, completion gate, failures table, resume, the
--tags-file filter, X13 CPU-baseline lanes and the scorer-provenance gate (CR-02, CR-05).
CPU only, no fits: the fit, CPU-baseline and scoring interpreters are tests/stage/fake_env.py, which emulates the
contracts of scripts/fit_paper_config.py, scripts/run_cpu_baselines.py and scripts/score_scib_native.py. Run in any
env with numpy, pandas, scipy, pytest:
    CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 python -m pytest -q tests/stage
"""
import hashlib
import json
import os
import signal
import stat
import subprocess
import sys
import time

import pandas as pd
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import fit_outcome  # noqa: E402
import fit_paper_config as fpc  # noqa: E402
import prereg_rules as pr  # noqa: E402
import run_stage as rs  # noqa: E402

RUNNER = os.path.join(ROOT, "scripts", "run_stage.py")


def _row(tag, task="atac_small", **over):
    r = dict(tag=tag, experiment="T1", task=task, counts="scib", arm="discriminator", lam="1", n_critic="1",
             adv_input="mean", zstd="0", cond="1", decoder="SCVI", n_latent="10", n_layers="1", n_hidden="128",
             likelihood="zinb", batch_size="128", max_epochs="2", train_size="0.9", seed="100", reference="auto",
             extra="{}")
    r.update(over)
    return r


def _cpu_row(tag, arm="harmony", **over):
    """An X13 CPU-baseline row as the builder writes it (cpu_row): knob in lam, backbone fields 'na' / 0, seed 0."""
    knob = rs.RCB.CPU_ARMS[arm]
    r = _row(tag, experiment="X13", arm=arm, lam={"harmony": "2", "scanorama": "20", "pca": "0"}[arm], n_critic="0",
             adv_input="na", zstd="0", cond="0", decoder="na", n_layers="0", n_hidden="0", likelihood="na",
             batch_size="0", max_epochs="0", train_size="0", seed="0", extra=json.dumps({"knob": knob}) if knob else "{}")
    r.update(over)
    return r


class Stage:
    """A temporary stage: manifest, prepped dir (empty files: the fake fitter reads nothing), fake interpreters."""

    def __init__(self, tmp, rows, plan=None, score_plan=None, experiments=("T1",)):
        self.tmp, self.out = tmp, str(tmp / "out")
        self.experiments = list(experiments)
        self.key = "+".join(sorted(self.experiments)) + "__all"
        self.manifest = tmp / "manifest.tsv"
        with open(self.manifest, "w") as f:
            f.write("# test manifest\n")
            pd.DataFrame(rows)[fpc.REQUIRED].to_csv(f, sep="\t", index=False)
        self.tags = [r["tag"] for r in rows]
        (tmp / "prepped").mkdir()
        for t in {r["task"] for r in rows}:
            (tmp / "prepped" / f"{t}__scib.h5ad").write_text("")
        for d in ("rhome", "rlibs", "state"):
            (tmp / d).mkdir()
        self.fakepy = tmp / "fakepy"
        self.fakepy.write_text(f"#!/bin/sh\nexec \"{sys.executable}\" \"{HERE}/fake_env.py\" \"$@\"\n")
        self.fakepy.chmod(self.fakepy.stat().st_mode | stat.S_IEXEC)
        self.set_plan(plan or {}, score_plan or {})

    def set_plan(self, plan, score_plan=None, score_prov=None):
        (self.tmp / "plan.json").write_text(json.dumps(plan))
        (self.tmp / "score_plan.json").write_text(json.dumps(score_plan or {}))
        (self.tmp / "score_prov.json").write_text(json.dumps(score_prov or {}))

    def args(self, *extra, lanes=2):
        return ["--manifest", str(self.manifest), "--experiments", *self.experiments, "--out-dir", self.out,
                "--prepped-dir", str(self.tmp / "prepped"), "--fit-python", str(self.fakepy),
                "--score-python", str(self.fakepy), "--r-home", str(self.tmp / "rhome"),
                "--r-libs", str(self.tmp / "rlibs"), "--expect-device", "cpu", "--fit-lanes", str(lanes),
                "--score-workers-during", "1", "--score-workers-after", "2", "--allow-dirty", "--poll-s", "0.05",
                "--view-interval-s", "0.5", "--kill-grace-s", "5", *extra]

    def env(self, **kw):
        return dict(os.environ, FAKE_PLAN=str(self.tmp / "plan.json"), FAKE_SCORE_PLAN=str(self.tmp / "score_plan.json"),
                    FAKE_SCORE_PROV=str(self.tmp / "score_prov.json"), FAKE_STATE=str(self.tmp / "state"),
                    CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1", **kw)

    def run(self, *extra, lanes=2, **env):
        p = subprocess.run([sys.executable, RUNNER, *self.args(*extra, lanes=lanes)], env=self.env(**env),
                           capture_output=True, text=True, timeout=300)
        return p.returncode, p.stdout + p.stderr

    def start(self, *extra, lanes=2, **env):
        return subprocess.Popen([sys.executable, RUNNER, *self.args(*extra, lanes=lanes)], env=self.env(**env),
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

    def ledger(self, key=None):
        return pd.read_csv(os.path.join(self.out, "ledger", f"{key or self.key}.csv"), dtype=str,
                           keep_default_na=False).set_index("tag")

    def summary(self, key=None):
        with open(os.path.join(self.out, "ledger", f"{key or self.key}.json")) as f:
            return json.load(f)

    def tags_file(self, text, name="tags.txt"):
        """(path, sha256) of a --tags-file holding `text`."""
        p = self.tmp / name
        p.write_text(text)
        return str(p), hashlib.sha256(text.encode()).hexdigest()

    def attempts(self, tag):
        p = os.path.join(self.out, "attempts", f"{tag}.jsonl")
        return [json.loads(x) for x in open(p)] if os.path.exists(p) else []

    def claims(self):
        d = os.path.join(self.out, "claims")
        return sorted(os.listdir(d)) if os.path.isdir(d) else []


def _wait(cond, timeout=30.0):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if cond():
            return True
        time.sleep(0.05)
    return False


def _dead_pid():
    p = subprocess.Popen([sys.executable, "-c", "pass"])
    p.wait()
    return p.pid


def _write_claim(out, tag, **owner):
    d = os.path.join(out, "claims", tag)
    os.makedirs(d)
    rec = dict(tag=tag, **rs.process_domain(), pid=os.getpid(), start_ticks=rs.start_ticks(os.getpid()),
               runner_id="other", stage="T1__all", purpose="fit", claimed_at=rs.now_iso(), slurm_job_id="")
    rec.update(owner)
    with open(os.path.join(d, "owner.json"), "w") as f:
        json.dump(rec, f)


# ---- outcomes -----------------------------------------------------------------------------------------------------
def test_failure_statuses_match_prereg():
    assert fit_outcome.FAILURE_STATUSES == pr.FAILURE_STATUSES


def check_outcomes(st):
    """Every row classified per PREREG sec. 1; failures.csv and scores readable by the rule code."""
    led = st.ledger()
    assert led.state.to_dict() == {"ok1": "scored", "ok2": "scored", "nan": "nonfinite_latent",
                                   "div": "diverged"}, led.state.to_dict()
    F = pr.read_failures(os.path.join(st.out, "failures.csv"))
    assert dict(zip(F.tag, F.status)) == {"nan": "nonfinite_latent", "div": "diverged"}
    assert (F.detail.str.len() > 0).all() and F.loc[F.tag == "div", "epoch"].item() == "3"
    S, files = pr.read_scores(os.path.join(st.out, "scores"))
    assert sorted(S.tag) == ["ok1", "ok2"] and set(S.kbet_seed) == {0} and len(files) == 2
    assert not os.path.exists(os.path.join(st.out, "scores", "nan.csv"))       # a non-finite latent is never scored
    g = st.summary()["gate"]
    assert g["ok"] and g["n_rows"] == 4 and g["counts"] == {"diverged": 1, "nonfinite_latent": 1, "scored": 2}
    assert st.claims() == [] and os.path.exists(os.path.join(st.out, "ledger", "T1__all.done"))


def test_outcomes_are_classified(tmp_path):
    rows = [_row("ok1"), _row("ok2", task="immune"), _row("nan"), _row("div")]
    st = Stage(tmp_path, rows, plan={"nan": "nan", "div": "diverge"})
    rc, log = st.run()
    assert rc == 0, log
    check_outcomes(st)
    rec = fit_outcome.read_status(os.path.join(st.out, "status", "nan.json"), tag="nan")
    assert rec["detail"].startswith("1 of 300 posterior-mean values are non-finite (1 cells)"), rec["detail"]
    assert "threads=1" in open(st.ledger().loc["ok1", "fit_log"]).read()        # thread cap reached the child


def test_rerun_of_a_complete_stage_does_nothing(tmp_path):
    st = Stage(tmp_path, [_row("ok1"), _row("div")], plan={"div": "diverge"})
    assert st.run()[0] == 0
    st.set_plan({"ok1": "crash", "div": "crash"})          # anything launched now would fail
    rc, log = st.run()
    assert rc == 0, log
    assert [len(st.attempts(t)) for t in ("ok1", "div")] == [4, 2]               # fit+score start/end; fit start/end


def test_no_score_then_score_elsewhere(tmp_path):
    st = Stage(tmp_path, [_row("ok1"), _row("div")], plan={"div": "diverge"})
    rc, log = st.run("--no-score")
    assert rc == 0, log
    assert st.ledger().state.to_dict() == {"ok1": "fitted", "div": "diverged"}
    assert not os.path.isdir(os.path.join(st.out, "scores")) or os.listdir(os.path.join(st.out, "scores")) == []
    st.set_plan({"ok1": "crash"})                          # the scoring run must not refit
    rc, log = st.run()
    assert rc == 0, log
    assert st.ledger().state.to_dict() == {"ok1": "scored", "div": "diverged"}
    assert [a["kind"] for a in st.attempts("ok1")] == ["fit", "fit", "score", "score"]


# ---- infrastructure errors ----------------------------------------------------------------------------------------
def test_infrastructure_retry_bound_then_stop(tmp_path):
    st = Stage(tmp_path, [_row("crash"), _row("ok1"), _row("ok2")], plan={"crash": "crash"})
    rc, log = st.run(lanes=1)
    assert rc == rs.EXIT_INFRA, log
    ends = [a for a in st.attempts("crash") if a["event"] == "end"]
    assert [(a["attempt"], a["outcome"], a["returncode"]) for a in ends] == [(1, "infrastructure", 1),
                                                                               (2, "infrastructure", 1)]
    led = st.ledger()
    assert led.loc["crash", "state"] == "infrastructure_error" and "crash.a2.log" in led.loc["crash", "detail"]
    assert set(led.loc[["ok1", "ok2"], "state"]) <= {"scored", "needs_score"}   # queued scores do not start
    F = pr.read_failures(os.path.join(st.out, "failures.csv"))
    assert len(F) == 0                                      # infrastructure errors are never failure rows
    s = st.summary()
    assert "crash" in s["stop_reason"] and "crash.a2.log" in s["stop_reason"] and not s["gate"]["ok"]
    assert st.claims() == []


def test_transient_error_is_retried(tmp_path):
    st = Stage(tmp_path, [_row("flaky")], plan={"flaky": ["crash", "ok"]})
    rc, log = st.run()
    assert rc == 0, log
    assert [(a["event"], a["attempt"]) for a in st.attempts("flaky") if a["kind"] == "fit"] == \
        [("start", 1), ("end", 1), ("start", 2), ("end", 2)]
    assert st.ledger().loc["flaky", "state"] == "scored"


@pytest.mark.parametrize("behaviour, msg", [("no_latent", "without a latent"),
                                            ("diverge_no_status", "without a divergence record"),
                                            ("wrong_device", "expected 'cpu'")])
def test_contract_violations_are_infrastructure(tmp_path, behaviour, msg):
    st = Stage(tmp_path, [_row("bad")], plan={"bad": behaviour})
    rc, log = st.run("--max-attempts", "1")
    assert rc == rs.EXIT_INFRA, log
    assert msg in st.ledger().loc["bad", "detail"], st.ledger().loc["bad", "detail"]
    assert len(pr.read_failures(os.path.join(st.out, "failures.csv"))) == 0
    if behaviour == "wrong_device":                          # the offending latent is kept, moved aside
        assert sorted(os.listdir(os.path.join(st.out, "quarantine"))) == ["bad.fit.a1.bad", "bad.fit.a1.bad.npz"]
        assert not os.path.exists(os.path.join(st.out, "latents", "bad.npz"))


def test_nan_metric_is_a_pipeline_fault(tmp_path):
    st = Stage(tmp_path, [_row("ok1")], score_plan={"ok1": "nan"})
    rc, log = st.run()
    assert rc == rs.EXIT_INFRA, log
    assert "non-finite required metrics ['kBET']" in st.ledger().loc["ok1", "detail"]
    assert os.listdir(os.path.join(st.out, "scores")) == []      # quarantined: never read by read_scores
    assert len(pr.read_failures(os.path.join(st.out, "failures.csv"))) == 0


# ---- claims -------------------------------------------------------------------------------------------------------
def test_claim_is_atomic_and_owned(tmp_path):
    a, b = rs.Claims(str(tmp_path), "runner-a", "s"), rs.Claims(str(tmp_path), "runner-b", "s")
    assert a.acquire("t", "fit") and not b.acquire("t", "fit")
    assert b.classify("t")[0] == "live" and a.classify("t")[0] == "mine"
    with pytest.raises(rs.StageError):
        b.release("t")
    a.release("t")
    assert b.classify("t") == ("free", None)


@pytest.mark.parametrize("kind, owner, flag", [("stale", dict(pid=None), "--clear-stale-claims"),
                                               ("stale", dict(boot_time=1), "--clear-stale-claims"),
                                               ("foreign", dict(host="node-elsewhere"), "--clear-foreign-claims"),
                                               ("foreign", dict(pid_ns="pid:[1]"), "--clear-foreign-claims")])
def test_stale_and_foreign_claims_need_a_flag(tmp_path, kind, owner, flag):
    st, owner = Stage(tmp_path, [_row("ok1"), _row("ok2")]), dict(owner)
    if owner.get("pid", 0) is None:
        owner["pid"] = _dead_pid()
    _write_claim(st.out, "ok1", **owner)
    rc, log = st.run()
    assert rc == rs.EXIT_PREFLIGHT and f"{kind} claim" in log and flag in log, log
    assert not os.path.isdir(os.path.join(st.out, "latents"))         # nothing ran
    rc, log = st.run(flag)
    assert rc == 0, log
    assert f"cleared {kind} claim ok1" in log and st.ledger().state.tolist() == ["scored", "scored"]


def test_live_claim_of_another_runner_is_skipped(tmp_path):
    st = Stage(tmp_path, [_row("ok1"), _row("held")])
    _write_claim(st.out, "held")                           # owner: this (live) test process
    rc, log = st.run()
    assert rc == rs.EXIT_GATE, log
    led, g = st.ledger(), st.summary()["gate"]
    assert led.loc["held", "state"] == "claimed_elsewhere" and led.loc["ok1", "state"] == "scored"
    assert g["not_final"] == {"held": "needs_fit"} and g["missing"] == ["held"]
    assert st.claims() == ["held"]                          # another runner's claim is never removed


# ---- preflight and gate -------------------------------------------------------------------------------------------
def test_placeholders_are_refused(tmp_path):
    st = Stage(tmp_path, [_row("ok1"), _row("g", lam="g1", adv_input="A2", zstd="A3")])
    rc, log = st.run()
    assert rc == rs.EXIT_PREFLIGHT and "lam='g1'" in log and "adv_input='A2'" in log and "zstd='A3'" in log, log


def test_contradictory_outputs_are_refused(tmp_path):
    st = Stage(tmp_path, [_row("ok1"), _row("div")], plan={"div": "diverge"})
    assert st.run()[0] == 0
    with open(os.path.join(st.out, "scores", "div.csv"), "w") as f:     # a score for a failed row
        f.write(open(os.path.join(st.out, "scores", "ok1.csv")).read().replace("ok1", "div"))
    rc, log = st.run()
    assert rc == rs.EXIT_PREFLIGHT and "both a failure record (diverged) and a score" in log, log
    rc, log = st.run("--report-only")
    assert rc == rs.EXIT_PREFLIGHT, log


def test_stale_latent_from_other_settings_is_refused(tmp_path):
    st = Stage(tmp_path, [_row("ok1")])
    assert st.run("--no-score")[0] == 0
    m = pd.read_csv(st.manifest, sep="\t", comment="#", dtype=str, keep_default_na=False)
    m.loc[0, "max_epochs"] = "3"                             # same tag, other setting (e.g. a resolved manifest)
    m.to_csv(st.manifest, sep="\t", index=False)
    rc, log = st.run()
    assert rc == rs.EXIT_PREFLIGHT and "other settings than the manifest row (['max_epochs'])" in log, log


def test_score_of_another_latent_or_seed_is_refused(tmp_path):
    st = Stage(tmp_path, [_row("ok1")])
    assert st.run()[0] == 0
    rc, log = st.run("--kbet-seed", "7")
    assert rc == rs.EXIT_PREFLIGHT and "kbet_seed 0 (this run 7)" in log, log


def test_fits_read_a_snapshot_of_the_manifest(tmp_path):
    st = Stage(tmp_path, [_row("ok1")])
    assert st.run("--no-score")[0] == 0
    s = st.summary()
    data = open(st.manifest, "rb").read()
    assert s["manifest_snapshot"] == os.path.join(st.out, "manifests", f"{s['manifest_sha256'][:16]}.tsv")
    assert open(s["manifest_snapshot"], "rb").read() == data


def test_scoring_harvested_latents_needs_no_fit_env(tmp_path):
    """JHPCE fits (--no-score, device A100) scored on a host whose fit env sees another device."""
    st = Stage(tmp_path, [_row("ok1")])
    rc, log = st.run("--no-score", "--expect-device", "A100", FAKE_DEVICE="NVIDIA A100-SXM4-80GB")
    assert rc == 0, log
    rc, log = st.run("--expect-device", "A100")             # fake fit env now reports 'cpu'; nothing to fit
    assert rc == 0 and "[env] fit" not in log, log
    assert st.ledger().loc["ok1", "state"] == "scored"
    m = pd.read_csv(st.manifest, sep="\t", comment="#", dtype=str, keep_default_na=False)
    pd.concat([m, m.assign(tag="new")]).to_csv(st.manifest, sep="\t", index=False)
    rc, log = st.run("--expect-device", "A100")             # a fit is needed: the probe refuses the wrong device
    assert rc == rs.EXIT_PREFLIGHT and "sees 'cpu', expected 'A100'" in log, log


def test_dry_run_starts_nothing(tmp_path):
    st = Stage(tmp_path, [_row("ok1"), _row("ok2")])
    rc, log = st.run("--dry-run")
    assert rc == 0 and "to run: 2 fits, up to 2 scores" in log, log
    assert not os.path.exists(st.out) or sorted(os.listdir(st.out)) == []


def test_gate_counts(tmp_path):
    """The gate's own predicate, run on the files of a finished stage and on known-bad variants of it."""
    st = Stage(tmp_path, [_row("ok1"), _row("nan"), _row("div")], plan={"nan": "nan", "div": "diverge"})
    assert st.run()[0] == 0
    assert gate_ok(st.out, st.manifest)
    os.remove(os.path.join(st.out, "scores", "ok1.csv"))
    assert not gate_ok(st.out, st.manifest)


def gate_ok(out, manifest, score=True):
    """Re-run the completion gate on an output directory (report-only); True iff it passes."""
    p = subprocess.run([sys.executable, RUNNER, "--manifest", str(manifest), "--experiments", "T1", "--out-dir", out,
                        "--prepped-dir", os.path.join(os.path.dirname(out), "prepped"), "--fit-python", sys.executable,
                        "--expect-device", "cpu", "--report-only", "--allow-dirty"] + ([] if score else ["--no-score"]),
                       capture_output=True, text=True, timeout=300)
    if p.returncode not in (0, rs.EXIT_GATE, rs.EXIT_PREFLIGHT):
        raise RuntimeError(p.stdout + p.stderr)
    return p.returncode == 0


# ---- resume -------------------------------------------------------------------------------------------------------
def _running_fits(st, n):
    return _wait(lambda: len(st.claims()) >= n and sum(a["event"] == "start" for t in st.tags
                                                       for a in st.attempts(t)) >= n)


def test_resume_after_killed_run(tmp_path):
    st = Stage(tmp_path, [_row("s1"), _row("s2"), _row("s3")], plan={t: "sleep" for t in ("s1", "s2", "s3")})
    p = st.start()
    assert _running_fits(st, 2), "two fits did not start"
    pids = [a["pid"] for t in st.tags for a in st.attempts(t) if a["event"] == "start"]
    p.send_signal(signal.SIGKILL)
    p.wait()
    assert _wait(lambda: all(rs.start_ticks(x) is None for x in pids), 10), "fits outlived the killed runner"
    held = st.claims()
    assert len(held) == 2
    rc, log = st.run()
    assert rc == rs.EXIT_PREFLIGHT and log.count("stale claim") == 2, log
    st.set_plan({})                                         # the resumed fits finish
    rc, log = st.run("--clear-stale-claims")
    assert rc == 0, log
    led = st.ledger()
    assert led.state.tolist() == ["scored"] * 3
    for t in held:                                          # killed attempt kept (start, no end); a2 completed it
        fits = [(a["event"], a["attempt"]) for a in st.attempts(t) if a["kind"] == "fit"]
        assert fits == [("start", 1), ("start", 2), ("end", 2)], fits
        assert os.path.exists(os.path.join(st.out, "logs", "fit", f"{t}.a1.log"))


def test_sigterm_stops_children_and_releases_claims(tmp_path):
    st = Stage(tmp_path, [_row("s1"), _row("s2"), _row("s3")], plan={t: "sleep" for t in ("s1", "s2", "s3")})
    p = st.start()
    assert _running_fits(st, 2)
    pids = [a["pid"] for t in st.tags for a in st.attempts(t) if a["event"] == "start"]
    p.send_signal(signal.SIGTERM)
    out, _ = p.communicate(timeout=60)
    assert p.returncode == 128 + signal.SIGTERM, out
    assert all(rs.start_ticks(x) is None for x in pids) and st.claims() == []
    ends = [a for t in st.tags for a in st.attempts(t) if a["event"] == "end"]
    assert len(ends) == 2 and {a["outcome"] for a in ends} == {"interrupted"}
    st.set_plan({})
    rc, log = st.run()                                      # no flag needed: the claims were released
    assert rc == 0, log
    assert st.ledger().state.tolist() == ["scored"] * 3


# ---- unchanged stage semantics (no --tags-file, no CPU rows) ------------------------------------------------------
LEDGER_COLS_C9568FD = ["tag", "experiment", "task", "arm", "lam", "cond", "seed", "state", "detail", "fit_attempts",
                       "score_attempts", "fit_host", "fit_started", "fit_ended", "fit_wall_s", "fit_seconds",
                       "score_started", "score_ended", "score_wall_s", "fit_log", "score_log", "device", "git_sha",
                       "claimed_by"]


def test_unfiltered_gpu_stage_is_unchanged(tmp_path):
    """An A1-style stage (every row on the GPU lanes, no tags filter): the stage key, ledger files and columns, states
    and gate of c9568fd; the new summary fields are present and neutral."""
    st = Stage(tmp_path, [_row("ok1"), _row("nan"), _row("div")], plan={"nan": "nan", "div": "diverge"})
    rc, log = st.run()
    assert rc == 0, log
    assert sorted(os.listdir(os.path.join(st.out, "ledger"))) == ["T1__all.csv", "T1__all.done", "T1__all.json"]
    led = pd.read_csv(os.path.join(st.out, "ledger", "T1__all.csv"), dtype=str, keep_default_na=False)
    assert list(led.columns) == LEDGER_COLS_C9568FD == rs.LEDGER_COLS
    assert dict(zip(led.tag, led.state)) == {"ok1": "scored", "nan": "nonfinite_latent", "div": "diverged"}
    assert dict(zip(led.tag, led.device)) == {"ok1": "cpu", "nan": "cpu", "div": ""}
    s = st.summary()
    assert s["stage"] == "T1__all" and s["tags_file"] is None and s["row_kinds"] == {"gpu": 3} and s["cpu_lanes"] == 0
    g = s["gate"]
    assert g["ok"] and g["counts"] == {"diverged": 1, "nonfinite_latent": 1, "scored": 1} and g["n_final"] == 3
    assert g["mixed_scorer_provenance"] == {} and g["scorer_provenance"]["cpu_simd"] == "avx2"
    assert {a["lane"] for t in st.tags for a in st.attempts(t) if a["event"] == "start"} == {"gpu", "score"}
    assert "CPU-baseline" not in log and "[env] cpu" not in log


# ---- --tags-file (CR-02) ------------------------------------------------------------------------------------------
def test_tags_file_restricts_the_stage(tmp_path):
    rows = [_row("ok1"), _row("ok2"), _row("ph", lam="g1", adv_input="A2"), _row("ok3", task="immune")]
    st = Stage(tmp_path, rows)
    rc, log = st.run()
    assert rc == rs.EXIT_PREFLIGHT and "ph: placeholder(s)" in log, log        # unfiltered: the placeholder blocks
    path, sha = st.tags_file("# the rows to run\nok1\n\n   ok3   # immune\n")
    rc, log = st.run("--tags-file", path)
    assert rc == 0, log                                     # a placeholder outside the filter is not this stage's
    key = f"T1__all__tags-{sha[:12]}"
    assert st.ledger(key).state.to_dict() == {"ok1": "scored", "ok3": "scored"}
    s = st.summary(key)
    assert s["stage"] == key and s["tags_file"] == dict(path=path, sha256=sha, n_tags=2) and s["gate"]["n_rows"] == 2
    assert sorted(os.listdir(os.path.join(st.out, "latents"))) == ["ok1.npz", "ok3.npz"]      # nothing else ran
    assert os.path.exists(os.path.join(st.out, "ledger", f"{key}.done"))
    assert not os.path.exists(os.path.join(st.out, "ledger", "T1__all.done"))   # the full stage is not complete
    path2, _ = st.tags_file("ok2\nph\n", "tags2.txt")
    rc, log = st.run("--tags-file", path2)                  # a placeholder inside the filter is refused
    assert rc == rs.EXIT_PREFLIGHT and "ph: placeholder(s) [\"lam='g1'\", \"adv_input='A2'\"]" in log, log
    assert not os.path.exists(os.path.join(st.out, "latents", "ok2.npz"))


@pytest.mark.parametrize("text, extra, msg", [
    ("ok1\nnope\n", (), "1 tags are not in the manifest, e.g. ['nope']"),
    ("ok1\nother\n", (), "1 tags are outside --experiments ['T1'] / --tasks [], e.g. ['other']"),
    ("ok1\nimm\n", ("--tasks", "atac_small"), "outside --experiments ['T1'] / --tasks ['atac_small'], e.g. ['imm']"),
    ("ok1\nok1\n", (), "1 tags listed more than once, e.g. ['ok1']"),
    ("# no tags\n\n", (), "no tags"),
    ("ok1 imm\n", (), ":1: one tag per line, got ['ok1', 'imm']"),
])
def test_tags_file_is_checked(tmp_path, text, extra, msg):
    st = Stage(tmp_path, [_row("ok1"), _row("imm", task="immune"), _row("other", experiment="T2")])
    path, _ = st.tags_file(text)
    rc, log = st.run("--tags-file", path, *extra)
    assert rc == rs.EXIT_PREFLIGHT and msg in log, log
    assert not os.path.isdir(os.path.join(st.out, "latents"))


def test_missing_tags_file_is_refused(tmp_path):
    st = Stage(tmp_path, [_row("ok1")])
    rc, log = st.run("--tags-file", str(tmp_path / "absent.txt"))
    assert rc == rs.EXIT_PREFLIGHT and "is not a file" in log, log


# ---- X13 CPU baselines (CR-02) ------------------------------------------------------------------------------------
def _cpu_stage(tmp_path, **kw):
    rows = [_row("g1"), _cpu_row("h1"), _cpu_row("s1", arm="scanorama"), _cpu_row("p1", arm="pca")]
    return Stage(tmp_path, rows, experiments=("T1", "X13"), **kw)


def test_cpu_rows_need_cpu_lanes(tmp_path):
    st = _cpu_stage(tmp_path)
    assert [rs.row_kind(r) for r in rs.load_manifest(st.manifest).to_dict("records")] == ["gpu", "cpu", "cpu", "cpu"]
    rc, log = st.run()
    assert rc == rs.EXIT_PREFLIGHT and "3 X13 CPU-baseline rows need a fit" in log and "--cpu-lanes N" in log, log
    assert not os.path.isdir(os.path.join(st.out, "latents"))           # refused before any work
    rc, log = st.run("--dry-run")                                        # a dry run refuses it too
    assert rc == rs.EXIT_PREFLIGHT and "3 X13 CPU-baseline rows need a fit" in log, log
    rc, log = st.run("--cpu-lanes", "1")
    assert rc == rs.EXIT_PREFLIGHT and "--cpu-lanes needs --cpu-python" in log, log
    rc, log = st.run("--dry-run", "--cpu-lanes", "2", "--cpu-python", str(st.fakepy))
    assert rc == 0 and "to run: 1 fits + 3 CPU-baseline fits, up to 4 scores" in log, log


def test_cpu_rows_run_on_cpu_lanes(tmp_path):
    st = _cpu_stage(tmp_path)
    rc, log = st.run("--cpu-lanes", "2", "--cpu-python", str(st.fakepy), "--cpu-threads", "3",
                     "--expect-device", "FakeGPU", FAKE_DEVICE="FakeGPU 10GB")
    assert rc == 0, log
    led = st.ledger()
    assert led.state.to_dict() == {t: "scored" for t in ("g1", "h1", "s1", "p1")}
    assert led.device.to_dict() == {"g1": "FakeGPU 10GB", "h1": "cpu: fake cpu", "s1": "cpu: fake cpu",
                                    "p1": "cpu: fake cpu"}             # each kind gated on its own device
    for t in ("h1", "s1", "p1"):
        flog = open(led.loc[t, "fit_log"]).read()
        assert f"fake cpu fit {t}: ok threads=3 --threads=3 cuda=''" in flog, flog
        assert [a["lane"] for a in st.attempts(t) if a["event"] == "start"] == ["cpu", "score"]
        assert not os.path.exists(os.path.join(st.out, "models", t))   # CPU baselines save no model
    assert [a["lane"] for a in st.attempts("g1") if a["event"] == "start"] == ["gpu", "score"]
    assert "[env] cpu" in log and "[env] fit" in log
    s = st.summary()
    assert s["gate"]["ok"] and s["row_kinds"] == {"cpu": 3, "gpu": 1} and s["cpu_lanes"] == 2
    S, _ = pr.read_scores(os.path.join(st.out, "scores"))
    assert sorted(S.tag) == ["g1", "h1", "p1", "s1"]                    # scored like every other latent
    rc, log = st.run("--expect-device", "FakeGPU")          # complete: a rerun needs neither lanes nor fit envs
    assert rc == 0 and "[env]" not in log.replace("[env] score", ""), log


@pytest.mark.parametrize("behaviour, msg", [
    ("wrong_device", "CPU baseline recorded device 'cuda' and cpu_model 'fake cpu', expected device 'cpu'"),
    ("no_device", "CPU baseline recorded device None and cpu_model None, expected device 'cpu'"),
    ("no_latent", "run_cpu_baselines.py exited 0 without a latent"),
    ("diverge", "run_cpu_baselines.py exit 23 with a status record"),
    ("crash", "run_cpu_baselines.py exit 1"),
])
def test_cpu_baseline_contract_violations_are_infrastructure(tmp_path, behaviour, msg):
    st = Stage(tmp_path, [_cpu_row("h1")], plan={"h1": behaviour}, experiments=("X13",))
    rc, log = st.run("--cpu-lanes", "1", "--cpu-python", str(st.fakepy), "--max-attempts", "1")
    assert rc == rs.EXIT_INFRA, log
    detail = st.ledger().loc["h1", "detail"]
    assert msg in detail, detail
    assert len(pr.read_failures(os.path.join(st.out, "failures.csv"))) == 0     # never an outcome of the row
    if behaviour in ("wrong_device", "no_device", "diverge"):          # the offending output is moved aside
        moved = {"diverge": ["h1.fit.a1.h1.json"]}.get(behaviour, ["h1.fit.a1.h1.npz"])
        assert sorted(os.listdir(os.path.join(st.out, "quarantine"))) == moved
    assert not os.path.exists(os.path.join(st.out, "latents", "h1.npz"))


@pytest.mark.parametrize("behaviour, detail", [
    ("nan", "1 of 300 embedding values are non-finite (1 cells)"),             # recorded by the runner, as for GPU rows
    ("nan_record", "fake: the saved embedding has non-finite values"),         # recorded by run_cpu_baselines.py
])
def test_cpu_nonfinite_latent_is_an_outcome(tmp_path, behaviour, detail):
    """Lead decision 2026-10-03: X13 CPU baselines have the nonfinite_latent outcome of every manifest fit."""
    st = Stage(tmp_path, [_cpu_row("h1"), _cpu_row("p1", arm="pca")], plan={"h1": behaviour}, experiments=("X13",))
    rc, log = st.run("--cpu-lanes", "1", "--cpu-python", str(st.fakepy))
    assert rc == 0, log
    assert st.ledger().state.to_dict() == {"h1": "nonfinite_latent", "p1": "scored"}
    F = pr.read_failures(os.path.join(st.out, "failures.csv"))
    assert dict(zip(F.tag, F.status)) == {"h1": "nonfinite_latent"} and F.detail.tolist() == [detail]
    rec = fit_outcome.read_status(os.path.join(st.out, "status", "h1.json"), tag="h1")
    with open(os.path.join(st.out, "latents", "h1.npz"), "rb") as f:
        assert rec["latent_sha256"] == hashlib.sha256(f.read()).hexdigest()   # the record names the kept latent
    assert not os.path.exists(os.path.join(st.out, "scores", "h1.csv"))        # a non-finite latent is never scored
    assert not os.path.isdir(os.path.join(st.out, "quarantine"))
    g = st.summary()["gate"]
    assert g["ok"] and g["counts"] == {"nonfinite_latent": 1, "scored": 1}
    st.set_plan({"h1": "crash", "p1": "crash"})              # final: a rerun starts nothing
    rc, log = st.run("--cpu-lanes", "1", "--cpu-python", str(st.fakepy))
    assert rc == 0 and "[fit] start" not in log, log


@pytest.mark.parametrize("behaviour, msg, moved", [
    ("nan_record_no_latent", "run_cpu_baselines.py exited 0 without a latent", ["h1.fit.a1.h1.json"]),
    ("finite_record", "nonfinite_latent record but the latent is missing or finite",
     ["h1.fit.a1.h1.json", "h1.fit.a1.h1.npz"]),
    ("nan_record_other_sha", "nonfinite_latent record of another latent (latent_sha256 differs)",
     ["h1.fit.a1.h1.json", "h1.fit.a1.h1.npz"]),
    ("nan_record_exit23", "run_cpu_baselines.py exit 23 with a status record", ["h1.fit.a1.h1.json", "h1.fit.a1.h1.npz"]),
])
def test_cpu_nonfinite_records_outside_the_contract_are_infrastructure(tmp_path, behaviour, msg, moved):
    st = Stage(tmp_path, [_cpu_row("h1")], plan={"h1": behaviour}, experiments=("X13",))
    rc, log = st.run("--cpu-lanes", "1", "--cpu-python", str(st.fakepy), "--max-attempts", "1")
    assert rc == rs.EXIT_INFRA, log
    detail = st.ledger().loc["h1", "detail"]
    assert msg in detail, detail
    assert len(pr.read_failures(os.path.join(st.out, "failures.csv"))) == 0     # never an outcome
    assert sorted(os.listdir(os.path.join(st.out, "quarantine"))) == moved      # record and latent moved aside


def test_gpu_fitter_record_of_its_own_nonfinite_latent_is_refused(tmp_path):
    """The GPU contract is unchanged: fit_paper_config.py writes no nonfinite_latent record (the runner does)."""
    st = Stage(tmp_path, [_row("g1")], plan={"g1": "nan_record"})
    rc, log = st.run("--max-attempts", "1")
    assert rc == rs.EXIT_INFRA and "fitter exited 0 and wrote a status record" in st.ledger().loc["g1", "detail"], log
    assert len(pr.read_failures(os.path.join(st.out, "failures.csv"))) == 0


def test_cpu_env_probe_must_pass(tmp_path):
    st = Stage(tmp_path, [_cpu_row("h1")], experiments=("X13",))
    rc, log = st.run("--cpu-lanes", "1", "--cpu-python", str(st.fakepy), FAKE_CPU_ENV_BROKEN="1")
    assert rc == rs.EXIT_PREFLIGHT and "CPU-baseline env probe failed" in log, log


@pytest.mark.parametrize("over, msg", [
    (dict(max_epochs="400"), "scVI-backbone fields set on a CPU row (they would be ignored): {'max_epochs': '400'}"),
    (dict(seed="1"), "expected extra knob 'theta' and seed 0, got {'knob': 'theta'} and seed 1"),
    (dict(extra="{}"), "expected extra knob 'theta' and seed 0, got {} and seed 0"),
    (dict(experiment="T1"), "not X13 CPU rows"),
])
def test_cpu_rows_follow_the_cpu_runner_contract(tmp_path, over, msg):
    st = Stage(tmp_path, [_cpu_row("h1", **over)], experiments=(over.get("experiment", "X13"),))
    rc, log = st.run("--cpu-lanes", "1", "--cpu-python", str(st.fakepy))
    assert rc == rs.EXIT_PREFLIGHT and "refused by run_cpu_baselines.select_rows" in log and msg in log, log


# ---- scorer provenance (CR-05) ------------------------------------------------------------------------------------
@pytest.mark.parametrize("column, value", [("scorer_git_sha", "0123abc"), ("scorer_dirty", 1),
                                           ("cpu_simd", "avx512f"), ("numba_cpu_name", "generic"),
                                           ("scorer_versions", json.dumps({"scib": "1.1.6"}))])
def test_mixed_scorer_provenance_fails_the_gate(tmp_path, column, value):
    st = Stage(tmp_path, [_row("ok1"), _row("ok2"), _row("ok3")])
    st.set_plan({}, {}, score_prov={"ok2": {column: value}})
    rc, log = st.run()
    assert rc == rs.EXIT_GATE and "[gate] mixed_scorer_provenance (1)" in log, log
    g = st.summary()["gate"]
    assert not g["ok"] and list(g["mixed_scorer_provenance"]) == [column]
    assert g["mixed_scorer_provenance"][column][str(value)] == dict(n=1, tags=["ok2"])
    assert sorted(g["scorer_provenance"]) == sorted(set(rs.SCORER_PROVENANCE) - {column})
    assert st.ledger().state.tolist() == ["scored"] * 3    # every row is scored, yet the stage is not complete
    assert not os.path.exists(os.path.join(st.out, "ledger", "T1__all.done"))
    rc, log = st.run()                                      # a rerun refuses before any work
    assert rc == rs.EXIT_PREFLIGHT and "existing score rows disagree on scorer provenance" in log, log
    rc, log = st.run("--report-only")                       # the report still rebuilds the failing gate
    assert rc == rs.EXIT_GATE and "mixed_scorer_provenance" in log, log


def test_other_scorer_fields_may_differ(tmp_path):
    """Only SCORER_PROVENANCE is gated: two scoring hosts of one CPU class (same SIMD level and numba target)."""
    st = Stage(tmp_path, [_row("ok1"), _row("ok2")])
    st.set_plan({}, {}, score_prov={"ok2": {"score_host": "node2", "cpu_model": "other cpu", "kbet_labels_skipped": 2}})
    rc, log = st.run()
    assert rc == 0, log


def test_scores_without_provenance_are_refused(tmp_path):
    st = Stage(tmp_path, [_row("ok1")], score_plan={"ok1": "old_scorer"})
    rc, log = st.run("--max-attempts", "1")
    assert rc == rs.EXIT_INFRA, log
    detail = st.ledger().loc["ok1", "detail"]
    assert f"no scorer provenance {list(rs.SCORER_PROVENANCE)} (written by a scorer older than CR-05" in detail, detail
    assert os.listdir(os.path.join(st.out, "scores")) == []          # quarantined: never read by read_scores
    q = os.path.join(st.out, "quarantine")
    (old,) = [f for f in os.listdir(q) if f.endswith(".csv")]
    os.replace(os.path.join(q, old), os.path.join(st.out, "scores", "ok1.csv"))   # an old score file in place
    rc, log = st.run()
    assert rc == rs.EXIT_PREFLIGHT and "written by a scorer older than CR-05" in log, log

