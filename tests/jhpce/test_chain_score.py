"""Fit-to-scoring chain (cluster/jhpce/chain_score.py): trigger logic, throttle and one pass-by-pass run against the real
scripts/run_stage.py output contract. The fitter and the scorer are tests/stage/fake_env.py; Slurm is a set of stub commands
(sbatch runs the batch's list through the fake scorer at once; squeue answers from a JSON file the test writes).
    CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 python -m pytest -q tests/jhpce/test_chain_score.py
"""
import json
import os
import stat
import subprocess
import sys
import textwrap

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "cluster", "jhpce"))
from tests.stage.test_run_stage import Stage, _row  # noqa: E402
import chain_lib  # noqa: E402
import chain_score as cs  # noqa: E402

FAKE = os.path.join(ROOT, "tests", "stage", "fake_env.py")
SB_STUB = textwrap.dedent('''\
    #!{py}
    import json, os, re, subprocess, sys
    st = os.environ["STUB_DIR"]
    n = len(os.listdir(st)) + 100
    text = open(sys.argv[-1]).read()
    open(os.path.join(st, f"sbatch_{{n}}.json"), "w").write(json.dumps(dict(file=sys.argv[-1], text=text)))
    if os.environ.get("STUB_SBATCH_FAIL"):
        sys.exit("sbatch: error: invalid account (stub)")
    sha, lst, out = re.search(r"score_array.sh (\\S+) (\\S+) (\\S+)", text).groups()
    if not os.environ.get("STUB_NO_SCORES"):
        for line in open(lst).read().splitlines():
            tag, npz, pre, meta = line.split("\\t")
            os.makedirs(out + "/scores", exist_ok=True)
            env = dict(os.environ, TAG=tag, META=meta, KBET_SEED="0", OUT_CSV=f"{{out}}/scores/{{tag}}.csv", FAKE_STATE=st)
            subprocess.run([sys.executable, "{fake}", "score_scib_native.py"], env=env, check=True)
    print(n)
    ''')
SQ_STUB = textwrap.dedent('''\
    #!{py}
    import json, os, sys
    live = json.load(open(os.environ["STUB_DIR"] + "/live.json")) if os.path.exists(os.environ["STUB_DIR"] + "/live.json") else {{}}
    if os.environ.get("STUB_SQUEUE_FAIL"):
        sys.exit("squeue: error: socket timed out")
    ids = sys.argv[sys.argv.index("-j") + 1].split(",")
    for i in ids:
        for state, k in live.get(i, {{}}).items():
            for t in range(k):
                print(f"{{i}}_{{t + 1}}|{{state}}")
    ''')
SA_STUB = textwrap.dedent('''\
    #!{py}
    import os, sys
    p = os.environ["STUB_DIR"] + ("/sacct_by_name.txt" if "--name" in sys.argv else "/sacct_job.txt")
    print(open(p).read() if os.path.exists(p) else "", end="")
    ''')


@pytest.fixture
def env(tmp_path, monkeypatch):
    rows = [_row(f"a{i}", task="atac_small", experiment="A1") for i in range(4)] + \
           [_row(f"i{i}", task="immune", experiment="A1") for i in range(2)]
    st = Stage(tmp_path, rows, experiments=("A1",))
    bin_, stub = tmp_path / "bin", tmp_path / "stub"
    bin_.mkdir(), stub.mkdir()
    for name, body in (("sbatch", SB_STUB), ("squeue", SQ_STUB), ("sacct", SA_STUB)):
        p = bin_ / name
        p.write_text(body.format(py=sys.executable, fake=FAKE))
        p.chmod(p.stat().st_mode | stat.S_IEXEC)
    monkeypatch.setenv("PATH", f"{bin_}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.setenv("STUB_DIR", str(stub))
    repo = tmp_path / "scorerepo"
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    subprocess.run(["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "--allow-empty", "-m", "x"],
                   check=True)
    st.stub, st.repo, st.score_out, st.state = stub, repo, tmp_path / "scores", tmp_path / "state"
    live(st)
    return st


def fit(st, tags):
    tf, _ = st.tags_file("\n".join(tags) + "\n", name=f"fit_{'_'.join(tags)}.txt")
    code, out = st.run("--no-score", "--tags-file", tf)
    assert code == 0, out


def chain(st, *extra, ticks=1):
    args = ["--stage-repo", ROOT, "--score-repo", str(st.repo), "--manifest", str(st.manifest),
            "--manifest-sha256", chain_lib.sha256_file(st.manifest), "--experiments", "A1", "--fit-out", st.out,
            "--score-out", str(st.score_out), "--prepped-dir", str(st.tmp / "prepped"), "--state-dir", str(st.state),
            "--expect-device", "cpu", "--settle-s", "0", "--interval", "0", "--min-batch", "3", "--max-ticks", str(ticks),
            "--fit-jobs", "999", *extra]
    return cs.main(args)


def subs(st):
    p = st.state / "submissions.jsonl"
    return [json.loads(x) for x in open(p)] if p.exists() else []


def live(st, fits=True, **arrays):
    """What squeue shows: the fit job 999 while `fits`, plus the given scoring arrays {job id: {state: tasks}}."""
    (st.stub / "live.json").write_text(json.dumps(dict(arrays, **({"999": {"RUNNING": 1}} if fits else {}))))


@pytest.mark.parametrize("ready,submitted,final,min_batch,want", [
    ({"t": ["a", "b"]}, set(), {}, 3, {}),                           # below the batch size while fits are coming
    ({"t": ["a", "b"]}, set(), {"t": True}, 3, {"t": ["a", "b"]}),   # the dataset's fits are final: flush
    ({"t": ["a", "b", "c", "d"]}, {"a"}, {}, 3, {"t": ["b", "c", "d"]}),   # already-sent latents are not sent twice
    ({"t": ["a", "b", "c"], "u": ["x"]}, set(), {}, 3, {"t": ["a", "b", "c"]}),
])
def test_plan_batches(ready, submitted, final, min_batch, want):  # test-reason: pins the trigger rules (batch size, flush, no double send); run tests use one setting
    assert cs.plan_batches(ready, submitted, final, min_batch) == want


@pytest.mark.parametrize("live_arrays,mem,n,want", [
    ([], 8, 500, 200),                       # CPU budget
    ([(150, 8)], 8, 500, 50),                # CPUs left
    ([], 32, 500, 62),                       # memory budget (2000 GB / 32 GB)
    ([(200, 8)], 8, 5, 0),                   # nothing left: wait
    ([], 8, 5, 5),                           # never more than the tasks
])
def test_admit(live_arrays, mem, n, want):  # test-reason: pins the CPU and memory budgets given live arrays; run tests only reach the CPU limit
    assert cs.admit(live_arrays, mem, n, 200, 2000) == want


def test_chain_runs_fit_to_scoring(env):
    st = env
    fit(st, ["a0", "a1", "a2", "i0"])
    wrap = str(st.tmp / "wrap" / "score_array.sh")
    code = chain(st, "--constraint", "znver2", "--max-cpus", "2", "--score-array", wrap, "--score-env", "SCORER_REPO=/a b", "X=1")
    s = subs(st)
    assert code == cs.EXIT_RUNNING and [r["event"] for r in s] == ["intent", "submitted"]
    assert s[1]["tags"] == ["a0", "a1", "a2"] and s[1]["task"] == "atac_small" and s[1]["throttle"] == 2
    text = json.load(open(sorted(st.stub.glob("sbatch_*.json"))[0]))["text"]
    assert "--array=1-3%2" in text and "--mem=3G" in text and "--constraint=znver2" in text and str(st.score_out) in text
    assert f"exec bash {wrap} " in text and "export SCORER_REPO='/a b'\nexport X=1\n" in text           # --score-array / --score-env
    lines = [x.split("\t") for x in open(st.state / "batches" / s[1]["batch"] / "list.tsv").read().splitlines()]
    assert [x[0] for x in lines] == ["a0", "a1", "a2"]
    assert json.loads(lines[0][3])["latent_sha256"] == chain_lib.sha256_file(os.path.join(st.out, "latents", "a0.npz"))
    # pass 2: the three scores are valid, the lone immune latent waits (fits still live), nothing is sent twice
    assert chain(st) == cs.EXIT_RUNNING and len(subs(st)) == 2
    status = json.load(open(st.state / "status.json"))
    assert status["final"] == 3 and status["submitted_now"] == [] and status["fit_missing"] == ["a3", "i1"]
    # more fits; with a live array the CPU/memory bound counts it
    fit(st, ["a3", "i1"])
    live(st, **{s[1]["job_id"]: {"RUNNING": 2, "PENDING": 1}})
    assert chain(st, "--max-cpus", "2") == cs.EXIT_RUNNING and len(subs(st)) == 2          # no CPU left: wait
    # fits over (no live fit job) and the array over: flush both datasets, then everything has a score
    live(st, fits=False)
    assert chain(st, ticks=3) == cs.EXIT_DONE
    sent = [r["tags"] for r in subs(st) if r["event"] == "submitted"]
    assert sorted(map(sorted, sent)) == [["a0", "a1", "a2"], ["a3"], ["i0", "i1"]]
    assert json.load(open(st.state / "finished.json"))["ok"] is True and chain(st, "--check-finished") == 0


def test_chain_does_not_send_invalid_latents_and_ends_when_stuck(env):
    st = env
    fit(st, ["a0", "a1", "a2"])
    open(os.path.join(st.out, "latents", "a1.npz"), "wb").write(b"not a zip")       # a latent the contract refuses
    live(st, fits=False)
    assert chain(st, "--min-batch", "2", ticks=2) == cs.EXIT_STALLED
    assert [r["tags"] for r in subs(st) if r["event"] == "submitted"] == [["a0", "a2"]]
    fin = json.load(open(st.state / "finished.json"))
    assert fin["ok"] is False and "a1" in fin["fit_missing"] and chain(st, "--check-finished") == 1
    fit(st, ["a3"])
    os.remove(st.tmp / "prepped" / "atac_small__scib.h5ad")                          # a missing input file stops the batch
    with pytest.raises(SystemExit, match="prepped file"):
        chain(st, "--min-batch", "1")


# test-reason: a scoring array that ends without a score file is listed and never resubmitted; no other test ends this way
def test_chain_lists_failed_scores_and_never_resubmits(env, monkeypatch):
    st = env
    fit(st, ["a0", "a1", "a2"])
    monkeypatch.setenv("STUB_NO_SCORES", "1")                                        # the array ends, no score file
    live(st, fits=False)
    assert chain(st, ticks=3) == cs.EXIT_STALLED
    assert len([r for r in subs(st) if r["event"] == "submitted"]) == 1
    assert json.load(open(st.state / "finished.json"))["failed_scores"] == ["a0", "a1", "a2"]


def test_chain_submit_failure_dry_run_and_adoption(env, monkeypatch):
    st = env
    fit(st, ["a0", "a1", "a2"])
    monkeypatch.setenv("STUB_SQUEUE_FAIL", "1")
    with pytest.raises(RuntimeError, match="squeue failed"):
        chain(st, "--dry-run")
    monkeypatch.delenv("STUB_SQUEUE_FAIL")
    assert chain(st, "--dry-run") == cs.EXIT_RUNNING and subs(st) == []
    assert (st.state / "batches").is_dir()
    monkeypatch.setenv("STUB_SBATCH_FAIL", "1")
    with pytest.raises(RuntimeError, match="not retried"):
        chain(st)
    monkeypatch.delenv("STUB_SBATCH_FAIL")
    # the intent is on record but no 'submitted' line (a crash after sbatch): adopt the job found by name, send nothing new
    assert chain(st, "--dry-run") == cs.EXIT_RUNNING and subs(st) == subs(st)[:1]               # no such job found: stays an intent
    (st.stub / "sacct_by_name.txt").write_text("4242\n4242_1\n")
    assert chain(st, "--dry-run") == cs.EXIT_RUNNING
    adopted = [r for r in subs(st) if r["event"] == "submitted"]
    assert len(adopted) == 1 and adopted[0]["job_id"] == "4242" and adopted[0]["adopted"] is True
    (st.stub / "sacct_by_name.txt").write_text("1\n2\n")
    st2 = subs(st)[0]
    open(st.state / "submissions.jsonl", "a").write(json.dumps(dict(st2, event="intent", batch="b2", job="wcdscore-b2")) + "\n")
    with pytest.raises(RuntimeError, match="resolve by hand"):
        chain(st, "--dry-run")


@pytest.mark.parametrize("extra,match", [
    (["--max-cpus", "201"], "1..200"),
    (["--score-env", "oops"], "NAME=VALUE"),
    (["--sizes", "MISSING"], "no size"),
])
def test_chain_refuses_bad_settings(env, tmp_path, extra, match):
    if "--sizes" in extra:
        extra = ["--sizes", str(tmp_path / "sizes.json")]
        (tmp_path / "sizes.json").write_text(json.dumps({"atac_small": {"mem_gb": 1, "walltime": "00:10:00"}}))
    with pytest.raises(SystemExit, match=match):
        chain(env, *extra)


def test_chain_needs_a_stop_rule_and_a_git_score_repo(env, tmp_path):
    st = env
    args = ["--stage-repo", ROOT, "--score-repo", str(st.repo), "--manifest", str(st.manifest), "--manifest-sha256", "0",
            "--experiments", "A1", "--fit-out", st.out, "--score-out", str(st.score_out), "--prepped-dir", str(st.tmp),
            "--state-dir", str(st.state), "--expect-device", "cpu"]
    with pytest.raises(SystemExit, match="--fit-jobs"):
        cs.main(args)
    with pytest.raises(SystemExit, match="not a git checkout"):
        cs.main(args[:3] + [str(tmp_path / "nogit")] + args[4:] + ["--max-ticks", "1"])


PROBE_SIZES = {"atac_small": {"mem_gb": 8, "walltime": "00:30:00", "measured": False}, "immune": {"mem_gb": 1, "walltime": "00:30:00", "measured": True}}


def test_unmeasured_task_sends_a_probe_first_then_uses_its_measured_memory(env, tmp_path):
    st = env
    sizes = tmp_path / "sizes.json"
    sizes.write_text(json.dumps(PROBE_SIZES))
    fit(st, ["a0", "a1", "a2", "a3", "i0", "i1"])
    args = ["--sizes", str(sizes), "--min-batch", "1", "--probe-n", "2"]
    assert chain(st, *args) == cs.EXIT_RUNNING
    first = [r for r in subs(st) if r["event"] == "submitted"]
    assert [r["tags"] for r in first] == [["a0", "a1"], ["i0", "i1"]] and [r["mem_gb"] for r in first] == [8, 1]   # probe: 2 of 4
    (st.stub / "sacct_job.txt").write_text("101_1|COMPLETED|\n101_1.batch|COMPLETED|2000000K\n101_1.extern|COMPLETED|0\n"
                                           "101_2|COMPLETED|\n101_2.batch|COMPLETED|1800000K\n")
    assert chain(st, *args) == cs.EXIT_RUNNING                                                # probe scored: size measured, rest sent
    sent = [r for r in subs(st) if r["event"] == "submitted"]
    assert [r["tags"] for r in sent] == [["a0", "a1"], ["i0", "i1"], ["a2", "a3"]] and sent[2]["mem_gb"] == 3
    assert json.load(open(st.state / "measured.json")) == {"atac_small": {"mem_gb": 3, "basis": json.load(open(st.state / "measured.json"))["atac_small"]["basis"]}}
    # 1.91 GiB x 1.3, rounded up; the immune size was measured already and is not replaced
    # a probe that did not complete cleanly measures nothing: the task waits, and the chain ends once nothing else can happen
    st2 = st.tmp / "state2"
    st.state = st2
    st.score_out = st.tmp / "scores2"
    (st.stub / "sacct_job.txt").write_text("201_1|FAILED|\n201_1.batch|FAILED|1000K\n")
    live(st, fits=False)
    assert chain(st, *args, ticks=3) == cs.EXIT_STALLED
    assert json.load(open(st2 / "finished.json"))["waiting_for_measurement"] == ["atac_small"]
    assert [r["tags"] for r in subs(st) if r["event"] == "submitted"] == [["a0", "a1"], ["i0", "i1"]]   # nothing more for a waiting task
