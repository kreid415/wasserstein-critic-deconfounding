"""scripts/run_stage.py: classification, claims, retry bound, completion gate, failures table, resume.
CPU only, no fits: the fit and scoring interpreters are tests/stage/fake_env.py, which emulates the contracts of
scripts/fit_paper_config.py and scripts/score_scib_native.py. Run in any env with numpy, pandas, scipy, pytest:
    CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 python -m pytest -q tests/stage
"""
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


class Stage:
    """A temporary stage: manifest, prepped dir (empty files: the fake fitter reads nothing), fake interpreters."""

    def __init__(self, tmp, rows, plan=None, score_plan=None):
        self.tmp, self.out = tmp, str(tmp / "out")
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

    def set_plan(self, plan, score_plan=None):
        (self.tmp / "plan.json").write_text(json.dumps(plan))
        (self.tmp / "score_plan.json").write_text(json.dumps(score_plan or {}))

    def args(self, *extra, lanes=2):
        return ["--manifest", str(self.manifest), "--experiments", "T1", "--out-dir", self.out,
                "--prepped-dir", str(self.tmp / "prepped"), "--fit-python", str(self.fakepy),
                "--score-python", str(self.fakepy), "--r-home", str(self.tmp / "rhome"),
                "--r-libs", str(self.tmp / "rlibs"), "--expect-device", "cpu", "--fit-lanes", str(lanes),
                "--score-workers-during", "1", "--score-workers-after", "2", "--allow-dirty", "--poll-s", "0.05",
                "--view-interval-s", "0.5", "--kill-grace-s", "5", *extra]

    def env(self, **kw):
        return dict(os.environ, FAKE_PLAN=str(self.tmp / "plan.json"), FAKE_SCORE_PLAN=str(self.tmp / "score_plan.json"),
                    FAKE_STATE=str(self.tmp / "state"), CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1", **kw)

    def run(self, *extra, lanes=2, **env):
        p = subprocess.run([sys.executable, RUNNER, *self.args(*extra, lanes=lanes)], env=self.env(**env),
                           capture_output=True, text=True, timeout=300)
        return p.returncode, p.stdout + p.stderr

    def start(self, *extra, lanes=2, **env):
        return subprocess.Popen([sys.executable, RUNNER, *self.args(*extra, lanes=lanes)], env=self.env(**env),
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

    def ledger(self):
        return pd.read_csv(os.path.join(self.out, "ledger", "T1__all.csv"), dtype=str, keep_default_na=False) \
            .set_index("tag")

    def summary(self):
        with open(os.path.join(self.out, "ledger", "T1__all.json")) as f:
            return json.load(f)

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
