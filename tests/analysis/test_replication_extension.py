"""Tests of the replication extension builders (experiments/sweep_replication/extension_rows.py, jhpce_stage/scoring_ext/make_list_ext.py)."""
import importlib.util, os, sys
import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
os.chdir(ROOT)  # extension_rows.py puts the working directory on sys.path (it is run from the rules checkout's scripts/)
sys.path.insert(0, os.path.join(ROOT, "scripts"))


def load(rel, name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(ROOT, rel)); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


E = load("experiments/sweep_replication/extension_rows.py", "extension_rows")
L = load("experiments/sweep_replication/jhpce_stage/scoring_ext/make_list_ext.py", "make_list_ext")
REPM = pd.read_csv(os.path.join(ROOT, "manifests/sweep_replication_x1.tsv"), sep="\t", comment="#", dtype=str)


def cell(task, cond, arm, lams):
    return REPM[(REPM.task == task) & (REPM.cond == cond) & (REPM.arm == arm) & REPM.lam.isin(lams)]


def test_next_lam():  # test-reason: grid walk and refusal at the ends of the grid
    assert E.next_lam(1.0) == 3 and E.next_lam(30.0) == 100
    for bad in (2.0, 3000.0):
        with pytest.raises(AssertionError):
            E.next_lam(bad)


def test_extension_rows_copied_and_built():  # test-reason: both paths (committed row copied, missing row built) give five seeds at the next strength
    at3 = cell("pancreas", "1", "sinkhorn", ["3"])  # highest committed replication strength of this cell
    committed10 = at3.assign(lam="10", tag=at3.tag.str.replace("lg2", "lg9"))  # stand-in for a committed lam-10 row
    rows, built = E.extension_for("pancreas", 1, "sinkhorn", 3, pd.concat([at3, committed10]), REPM)
    assert not built and set(rows.lam) == {"10"} and sorted(rows.seed) == list("01234") and set(rows.tag) == set(committed10.tag)
    rows, built = E.extension_for("pancreas", 1, "sinkhorn", 3, at3, REPM)
    assert built and set(rows.lam) == {"10"} and sorted(rows.seed) == list("01234") and not set(rows.tag) & set(REPM.tag)


def test_extension_rows_rejects_wrong_lam_max():  # test-reason: the reading's highest strength must match the run list
    with pytest.raises(AssertionError):
        E.extension_for("pancreas", 1, "sinkhorn", 1, cell("pancreas", "1", "sinkhorn", ["3"]), REPM)


def test_build_lists(tmp_path, capsys):  # test-reason: per-task scoring lists, missing-latent and already-scored refusals
    R, OUT, S = tmp_path / "R", tmp_path / "OUT", tmp_path / "S"
    for d in (R / "scoring", OUT / "latents", S / "sweep_replication_scores/scores"):
        d.mkdir(parents=True)
    rows = pd.concat([cell("pancreas", "1", "sinkhorn", ["3"]), cell("sim2", "0", "sinkhorn", ["10"]).head(2)])
    (R / "sweep_replication_ext1.tsv").write_text("# c\n" + rows.to_csv(sep="\t", index=False))
    for t in rows.tag:
        (OUT / "latents" / f"{t}.npz").write_bytes(b"x")
    expect = {"pancreas": 5, "sim2": 2}
    with pytest.raises(AssertionError):  # a score already exists
        (S / "sweep_replication_scores/scores" / f"{rows.tag.iloc[0]}.csv").write_text("x")
        L.build_lists(str(R), str(OUT), str(S), expect)
    (S / "sweep_replication_scores/scores" / f"{rows.tag.iloc[0]}.csv").unlink()
    L.build_lists(str(R), str(OUT), str(S), expect)
    assert len((R / "scoring/ext_pancreas.tsv").read_text().splitlines()) == 5 and len((R / "scoring/ext_sim2.tsv").read_text().splitlines()) == 2
    (OUT / "latents" / f"{rows.tag.iloc[0]}.npz").unlink()
    with pytest.raises(AssertionError):  # a latent is missing
        L.build_lists(str(R), str(OUT), str(S), expect)
