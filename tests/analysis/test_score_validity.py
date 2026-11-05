"""Tests of experiments/sweep_replication/score_validity.py (per-unit score validity for the replication board)."""
import importlib.util, os, sys
import numpy as np, pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import prereg_rules as P  # noqa: E402

spec = importlib.util.spec_from_file_location("score_validity", os.path.join(ROOT, "experiments/sweep_replication/score_validity.py"))
V = importlib.util.module_from_spec(spec); spec.loader.exec_module(V)


def _write(tmp_path, task="sim1", tag="t1", **over):
    row = {"tag": tag, **{m: 0.5 for m in P.BATCH_METRICS + P.bio_metrics_for(task)}, **over}
    f = tmp_path / f"{tag}.csv"; pd.DataFrame([row]).to_csv(f, index=False)
    return [str(f)]


# test-reason: one valid file and each way a score file is invalid (wrong tag, NaN, out of range)
@pytest.mark.parametrize("over,valid", [({}, True), ({"tag": "other"}, False), ({P.BATCH_METRICS[0]: np.nan}, False), ({P.BATCH_METRICS[0]: 1.5}, False)])
def test_unit_check(tmp_path, over, valid):
    assert V.unit_check(P, "sim1", "t1", _write(tmp_path, **over))[0] is valid


def test_unit_check_file_count_and_columns(tmp_path):  # test-reason: missing/duplicate score file and missing metric column
    assert not V.unit_check(P, "sim1", "t1", [])[0] and not V.unit_check(P, "sim1", "t1", ["a", "b"])[0]
    f = tmp_path / "t1.csv"; pd.DataFrame([{"tag": "t1"}]).to_csv(f, index=False)
    ok, why, _ = V.unit_check(P, "sim1", "t1", [str(f)])
    assert not ok and "missing columns" in why


# test-reason: command-line path: exit status, written table, and refusal of score files outside the manifest
@pytest.mark.parametrize("case,code", [("ok", 0), ("bad", 1), ("extra", "stray")])
def test_main(tmp_path, monkeypatch, case, code):
    d = tmp_path / "scores"; d.mkdir()
    for tag in ("t1", "t2"):
        _write(d, tag=tag, **({P.BATCH_METRICS[0]: np.nan} if case == "bad" and tag == "t2" else {}))
    (d / "notes.txt").write_text("not a score")
    if case == "extra":
        _write(d, tag="t3")
    monkeypatch.setattr(P, "read_manifest", lambda p: (pd.DataFrame(dict(tag=["t1", "t2"], task="sim1", arm="none", seed=0)), None))
    argv = ["--rules-scripts", os.path.join(ROOT, "scripts"), "--manifest", "m", "--scores", str(d), "--out", str(tmp_path / "v.csv")]
    if case == "extra":
        with pytest.raises(SystemExit, match="not in the manifest"):
            V.main(argv)
        return
    assert V.main(argv) == code
    assert pd.read_csv(tmp_path / "v.csv").valid.sum() == (2 if case == "ok" else 1)
