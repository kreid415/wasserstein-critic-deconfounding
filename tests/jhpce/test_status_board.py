"""Status board (cluster/jhpce/status_board.py): the four groups against a real stage output (fake fitter and scorer, as
tests/jhpce/test_chain_score.py), offline Slurm text, the presence-only rule for blinded main files and the two-site
export / import.   CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 python -m pytest -q tests/jhpce/test_status_board.py
"""
import builtins
import datetime
import json
import os
import subprocess
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "cluster", "jhpce"))
from tests.stage.test_run_stage import Stage, _row  # noqa: E402
import chain_lib  # noqa: E402
import status_board as sbd  # noqa: E402

FAKE = os.path.join(ROOT, "tests", "stage", "fake_env.py")


def sacct_text(*rows):
    cap = datetime.datetime.now().astimezone().strftime("%Y-%m-%dT%H:%M:%S%z")
    head = "JobID|JobName|State|ExitCode|Elapsed|Start|End|NodeList|ReqMem|MaxRSS|Timelimit|AllocCPUS"
    return "\n".join([f"#CAPTURED {cap}", head, *rows]) + "\n"


def job(jid, state, exit_="0:0", end="2026-10-27T02:00:00"):
    return [f"{jid}|n|{state}|{exit_}|01:00:00|2026-10-27T01:00:00|{end}|c1|96G||3-00:00:00|10",
            f"{jid}.batch|batch|{state}|{exit_}|01:00:00|2026-10-27T01:00:00|{end}|c1||1K||10"]


def write_score(st, tag, sha, task="atac_small"):
    meta = dict(latent_sha256=sha, experiment="A1", task=task, arm="discriminator", lam="1", cond="1", seed="100")
    os.makedirs(os.path.join(st.out, "scores"), exist_ok=True)
    env = dict(os.environ, TAG=tag, META=json.dumps(meta), KBET_SEED="0", OUT_CSV=os.path.join(st.out, "scores", tag + ".csv"),
               FAKE_STATE=str(st.tmp / "state"))
    subprocess.run([sys.executable, FAKE, "score_scib_native.py"], env=env, check=True)


def run_board(cfg_path, out, *args):
    return sbd.main(["--config", str(cfg_path), "--out-dir", str(out), *args])


@pytest.fixture
def world(tmp_path):
    rows = [_row(f"a{i}", task="atac_small", experiment="A1") for i in range(4)] + \
           [_row(f"i{i}", task="immune", experiment="A1") for i in range(2)]
    st = Stage(tmp_path, rows, plan={"a2": "diverge", "i0": "nan"}, experiments=("A1",))
    tf, _ = st.tags_file("a0\na1\na2\na3\ni0\n")
    st.run("--no-score", "--tags-file", tf)                                    # a2 diverges, i0 saves a non-finite latent
    open(os.path.join(st.out, "latents", "a1.npz"), "wb").write(b"not a zip")               # present, fails the contract
    os.remove(os.path.join(sbd.chain_lib.load_run_stage(ROOT).fit_outcome.status_path(st.out, "i0")))   # i0: no failure record
    write_score(st, "a0", chain_lib.sha256_file(os.path.join(st.out, "latents", "a0.npz")))
    write_score(st, "a3", "0" * 64)                                                            # a score of another latent
    main = {s: tmp_path / "main" / s for s in ("jhpce", "local")}
    for s, d in main.items():
        (d / "latents").mkdir(parents=True), (d / "scores").mkdir()
    (main["jhpce"] / "latents" / "a0.npz").write_bytes(b"blinded bytes")
    (main["local"] / "latents" / "a1.npz").write_bytes(b"blinded bytes")
    (main["jhpce"] / "latents" / "a2.npz").write_bytes(b"x")
    (main["local"] / "latents" / "a2.npz").write_bytes(b"x")                                # at both sites
    (main["jhpce"] / "latents" / "a3.npz").write_bytes(b"")                                 # empty: not output
    (main["local"] / "scores" / "a0.csv").write_bytes(b"blinded score bytes")
    subm = tmp_path / "submissions.jsonl"
    subm.write_text(json.dumps(dict(event="intent", job_id="1", tags=["a1"])) + "\n" +
                    json.dumps(dict(event="submitted", job_id="222", tags=["a0", "a3"])) + "\n")
    pilot = dict(manifest=str(st.manifest), manifest_sha256=chain_lib.sha256_file(st.manifest), experiments=["A1"],
                 stage_repo=ROOT, check="full", expect_device="cpu", sites={"jhpce": st.out})
    x1 = dict(manifest=str(st.manifest), manifest_sha256=pilot["manifest_sha256"], experiments=["A1"], stage_repo=ROOT,
              check="presence", sites={s: str(d) for s, d in main.items()})
    cfg = {"groups": {
        "pilot_fit": dict(pilot, title="pilot fits", kind="fit", jobs={"atac_small": "111"}),
        "pilot_score": dict(pilot, title="pilot scores", kind="score", score_sites={"jhpce": st.out}, submissions=str(subm)),
        "main_fit": dict(x1, title="main fits", kind="fit", jobs={"atac_small": "111"}),
        "main_score": dict(x1, title="main scores", kind="score")}}
    p = tmp_path / "cfg.json"
    p.write_text(json.dumps(cfg))
    sa = tmp_path / "sacct.txt"
    sa.write_text(sacct_text(*job("111", "FAILED", "15:0"), *job("222_1", "COMPLETED"), *job("222_2", "COMPLETED")))
    sq = tmp_path / "squeue.txt"
    sq.write_text(sacct_text()[: sacct_text().index("\n") + 1] + "JOBID|NAME|STATE|TIME|TIME_LIMIT|SUBMIT_TIME|START_TIME|NODELIST(REASON)\n")
    return dict(st=st, cfg=p, sa=sa, sq=sq, main=main, tmp=tmp_path)


def totals(out, name):
    return json.load(open(out / "status_board.json"))["groups"][name]["board"]["totals"]


def test_board_states_presence_only_and_two_sites(world, monkeypatch):
    w, out = world, world["tmp"] / "board"
    opened = []
    real_open = builtins.open
    monkeypatch.setattr(builtins, "open", lambda f, *a, **k: (opened.append(str(f)), real_open(f, *a, **k))[1])
    assert run_board(w["cfg"], out, "--sites", "jhpce", "local", "--sacct-file", str(w["sa"]), "--squeue-file", str(w["sq"])) == 0
    monkeypatch.undo()
    assert not [p for p in opened if os.sep + "main" + os.sep in p]                  # blinded main files are only stat-ed
    t = totals(out, "pilot_fit")      # a0, a3 valid; a2 a recorded divergence (final); a1 corrupt under a FAILED job; i0 no record; i1 absent
    assert (t["done_valid"], t["failed"], t["done_invalid"], t["missing"]) == (3, 1, 1, 1)
    units = {u["unit"]: u for u in json.load(open(out / "status_board.json"))["groups"]["pilot_fit"]["board"]["units"]}
    assert "partial output invalid" in units["a1"]["reason"] and "final outcome diverged" in units["a2"]["reason"]
    s = totals(out, "pilot_score")    # a0 scored; a2 needs no score; a3's score is of another latent (done_invalid); a1, i0, i1 have no valid fit
    assert (s["done_valid"], s["done_invalid"], s["missing"]) == (2, 1, 3)
    f = json.load(open(out / "status_board.json"))["groups"]["main_fit"]["board"]
    assert f["validity_basis"] == "presence_only" and f["totals"]["done_valid"] == 3          # a0 (jhpce), a1 (local), a2 (both)
    assert f["totals"]["failed"] == 1                                                          # a3: empty file, its job FAILED
    assert any("also at" in u["reason"] for u in f["units"] if u["unit"] == "a2") and totals(out, "main_score")["done_valid"] == 1
    md = (out / "status_board.md").read_text()
    assert "PRESENCE ONLY" in md and "not applicable: no live job" in md
    # the board is rebuilt from the cache of the first pass (no latent is read twice); an unchanged tree gives the same totals
    assert run_board(w["cfg"], out, "--sites", "jhpce", "local", "--sacct-file", str(w["sa"]), "--squeue-file", str(w["sq"])) == 0
    assert totals(out, "pilot_fit") == t and (out / "validity_cache_jhpce.json").exists()


def test_board_export_import_and_blind_site(world, monkeypatch):
    w, out = world, world["tmp"] / "board2"
    cfg = json.load(open(w["cfg"]))
    cfg["groups"]["pilot_score"]["submissions"] = str(w["tmp"] / "no_such_submissions.jsonl")        # before the first submission
    (w["tmp"] / "cfg_nosub.json").write_text(json.dumps(cfg))
    run_board(w["tmp"] / "cfg_nosub.json", w["tmp"] / "board_nosub", "--sites", "jhpce", "--groups", "pilot_score")
    assert totals(w["tmp"] / "board_nosub", "pilot_score")["done_valid"] == 2
    (w["tmp"] / "empty.json").write_text("{}")
    with pytest.raises(SystemExit, match="no groups"):
        run_board(w["tmp"] / "empty.json", out, "--sites", "jhpce")
    inv = w["tmp"] / "inv_local.json"
    assert run_board(w["cfg"], out, "--sites", "local", "--groups", "main_fit", "main_score", "--export-inventory", str(inv)) == 0
    assert set(json.load(open(inv))["main_fit"]) == {"local"} and not (out / "status_board.md").exists()
    with pytest.raises(SystemExit, match="neither readable"):
        run_board(w["cfg"], out, "--sites", "jhpce", "--groups", "main_fit")
    run_board(w["cfg"], out, "--sites", "jhpce", "--groups", "main_fit", "--import", f"local={inv}")
    assert totals(out, "main_fit")["done_valid"] == 3                                   # a1 only exists at the local site
    live = w["tmp"] / "sq_live.txt"
    live.write_text(sacct_text("111_1|n|RUNNING|2:00|3-00:00:00|2026-10-27T01:00:00|2026-10-27T01:00:00|c1"))
    sa = w["tmp"] / "sa_live.txt"
    sa.write_text(sacct_text(*job("111", "RUNNING", end="Unknown")))
    run_board(w["cfg"], out, "--sites", "jhpce", "--groups", "main_fit", "--import", f"local={inv}", "--sacct-file", str(sa),
              "--squeue-file", str(live))
    eta = json.load(open(out / "status_board.json"))["groups"]["main_fit"]["eta"]
    assert eta["refused"] is True                                                       # too few timestamped completions
    # no saved Slurm text: sacct and squeue are asked (stubs here) for explicit job ids; a group without job ids asks nothing
    bin_ = w["tmp"] / "bin"
    bin_.mkdir()
    for name, text in (("sacct", sa.read_text().split("\n", 1)[1]), ("squeue", "111_1|n|RUNNING|2:00|3-00:00:00|2026-10-27T01:00:00|2026-10-27T01:00:00|c1\n")):
        (bin_ / name).write_text(f"#!/bin/sh\ncat <<'EOF'\n{text}EOF\n")
        (bin_ / name).chmod(0o755)
    monkeypatch.setenv("PATH", f"{bin_}{os.pathsep}{os.environ['PATH']}")
    run_board(w["cfg"], out, "--sites", "jhpce", "--groups", "main_fit", "--import", f"local={inv}")
    assert (out / "sacct.txt").read_text().startswith("#CAPTURED") and totals(out, "main_fit")["running"] == 1
    run_board(w["cfg"], out, "--sites", "jhpce", "--groups", "main_score", "--import", f"local={inv}")
    assert totals(out, "main_score")["done_valid"] == 1


def test_expected_rows_refuses_a_changed_manifest(world):
    with pytest.raises(SystemExit, match="scripts/run_stage.py"):
        chain_lib.load_run_stage(str(world["tmp"]))
    rs = chain_lib.load_run_stage(ROOT)
    assert chain_lib.Checker(rs, world["st"].out, world["st"].out, "cpu").save() is None      # no cache file: nothing to save
    tf, _ = world["st"].tags_file("a0\n")
    got = chain_lib.read_expected(rs, str(world["st"].manifest), chain_lib.sha256_file(str(world["st"].manifest)), ["A1"], [tf])
    assert [r["tag"] for r in got] == ["a0"]
    with pytest.raises(SystemExit, match="pinned"):
        chain_lib.read_expected(rs, str(world["st"].manifest), "0" * 64, ["A1"])
    with pytest.raises(SystemExit, match="more than one tags file"):
        tf, _ = world["st"].tags_file("a0\n")
        chain_lib.read_expected(rs, str(world["st"].manifest), chain_lib.sha256_file(str(world["st"].manifest)), ["A1"], [tf, tf])
