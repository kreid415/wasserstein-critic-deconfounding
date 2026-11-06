"""Tests of make_supp_supp-methods.py (run from the repository root: python -m pytest paper/tools/test_supp_methods.py).

The collect step runs on a small fake data root (two tasks, h5py files of the real sizes, one fit per task); the table
step runs on the committed inputs in paper/figure_data/supp_methods and must reproduce the committed tables."""
import importlib.util
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pytest

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("make_supp_methods", HERE / "make_supp_supp-methods.py")
sm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sm)
FP = json.loads(sm.git_show(*sm.GIT["fingerprints"]))
TWO = ["pancreas", "sim2"]


def write_h5(path, names, cols, missing=None):
    with h5py.File(path, "w") as h:
        o = h.create_group("obs")
        o.attrs["_index"] = "_index"
        o.create_dataset("_index", data=np.array(names, dtype="S"))
        for k, v in cols.items():
            cats, codes = np.unique(np.asarray(v), return_inverse=True)
            g = o.create_group(k)
            codes = codes.astype(np.int8)
            if k == missing:
                codes[0] = -1
            g.create_dataset("codes", data=codes)
            g.create_dataset("categories", data=cats.astype("S"))


def task_obs(t):
    """Cells, batches and cell types of the right sizes for a task of the fingerprint record."""
    fp = FP[f"{t}__scib.h5ad"]
    status = json.loads(fp["uns"]["count_status_as_distributed"])
    batch = np.concatenate([[b] * s["cells"] for b, s in status.items()])
    cols = dict(batch=batch, celltype=np.array([f"c{i % 4}" for i in range(len(batch))]))
    if t == "sim2":   # sub-batches nested in four batches
        parent = {b: f"P{i // 4}" for i, b in enumerate(status)}
        cols["Batch"] = np.array([parent[b] for b in batch])
    return [f"{t}{i}" for i in range(len(batch))], cols


ROW = dict(counts="scib", lam=0.0, n_critic=0, adv_input="mean", zstd=0, cond=1, decoder="SCVI", n_latent=10, n_layers=1,
           n_hidden=128, likelihood="zinb", batch_size=128, max_epochs=400, train_size=0.9, reference="auto", extra="{}")
PLAN = {"fit": {"lr": 0.001, "n_epochs_kl_warmup": 400, "d_coef": 0.0,
                "optimizers": [{"cls": "Adam", "param_groups": [{"lr": 0.001}]}]}}


@pytest.fixture
def root(tmp_path, monkeypatch):
    """A data root with the two tasks, one plain-scVI fit each and one Harmony fit; returns (root, site)."""
    d = tmp_path / "D"
    (d / "prepped_scib").mkdir(parents=True)
    inp, tab = d / sm.SWEEP / "inputs", d / sm.SWEEP / "tables"
    inp.mkdir(parents=True)
    tab.mkdir()
    rows, lat, scores = [], [], []
    metrics = sm.constants(sm.git_show(*sm.GIT["scorer"]), ["BATCH_METRICS", "BIO_METRICS"])
    for t in TWO:
        names, cols = task_obs(t)
        write_h5(d / "prepped_scib" / f"{t}__scib.h5ad", names, cols)
        for tag, exp, arm, cfg in ((f"A1_{t}", "A1", "none", {"scvi": "1.4.2", "plan": PLAN}),
                                   (f"X13_{t}", "X13", "harmony",
                                    {"tool_kwargs": {"n_comps": 10, "theta": 2.0}, "versions": {"harmony-pytorch": "0.1.7"}})):
            rows.append(dict(tag=tag, experiment=exp, task=t, arm=arm, seed=100, **ROW))
            np.savez(d / f"{tag}.npz", config=json.dumps(cfg))
            lat.append(f"{tag}\t{d / f'{tag}.npz'}")
            vals = {m: 0.5 for m in metrics["BATCH_METRICS"] + metrics["BIO_METRICS"]}
            if t == "sim2":
                vals.update(cell_cycle_conservation=np.nan, trajectory=np.nan)
            vals["trajectory"] = np.nan
            pd.DataFrame([dict(tag=tag, hvg_overlap=np.nan, scorer_versions=json.dumps({"scib": "1.1.7"}), **vals)]).to_csv(
                d / f"{tag}.csv", index=False)
            scores.append(str(d / f"{tag}.csv"))
    pd.DataFrame(rows).to_csv(inp / "manifest.tsv", sep="\t", index=False)
    (inp / "latents.tsv").write_text("\n".join(lat) + "\n")
    (inp / "scores_list.txt").write_text("\n".join(scores) + "\n")
    pd.DataFrame([dict(arm="harmony", expected=1, failed=0, missing=0, scored=1, task=t) for t in TWO]).to_csv(
        tab / "baseline_coverage.csv", index=False)
    site = tmp_path / "site"
    (site / "scib" / "metrics").mkdir(parents=True)
    for f in sm.SCIB_FILES:
        (site / "scib" / "metrics" / f).write_text(f"# {f}\n")
    (site / "scib-1.1.7.dist-info" / "licenses").mkdir(parents=True)
    (site / "scib-1.1.7.dist-info" / "licenses" / "LICENSE.txt").write_text("MIT\n")
    real = sm.git_show
    rep = pd.DataFrame([dict(tag=f"X1_{t}", experiment="X1", task=t, arm="none", seed=s, **ROW) for t in TWO for s in range(5)])
    monkeypatch.setattr(sm, "git_show", lambda rev, path: rep.to_csv(sep="\t", index=False)
                        if (rev, path) == sm.GIT["replication"] else real(rev, path))
    return d, site


def test_collect_writes_task_records_settings_and_checksums(root, tmp_path):  # test-reason: the only run of the collect step
    d, site = root
    out = sm.collect(d, tmp_path / "out", TWO, site)
    t = pd.read_csv(out / "tasks.csv", keep_default_na=False).set_index("task")
    assert t.loc["pancreas", "cells"] == FP["pancreas__scib.h5ad"]["shape"][0] and t.loc["pancreas", "nonint_batches"] == 4
    assert t.loc["sim2", "parent_batches"] == "4" and t.loc["pancreas", "parent_batches"] == ""   # nesting found only in sim2
    assert t.loc["pancreas", "n_bio"] == 7 and t.loc["sim2", "n_bio"] == 6                        # cell-cycle only for RNA tasks
    assert "cell_cycle_conservation" in t.loc["sim2", "blank_metrics"]
    s = pd.read_csv(out / "settings.csv", keep_default_na=False)
    assert set(s.field) >= {"fit.lr", "fit.optimizer0", "tool.theta", "version.harmony-pytorch", "manifest.seed"}
    assert "fit.d_coef" not in set(s.field)                                                        # the strength varies by design
    assert sm.setting(s, "all", "manifest.seed", "replication") == [0, 1, 2, 3, 4]
    listed = [line.split("  ")[1] for line in (out / "MANIFEST.sha256").read_text().splitlines()]
    assert all(sm.sha256(out / f) == h for f, h in zip(listed, [line.split()[0] for line in (out / "MANIFEST.sha256").read_text().splitlines()]))
    assert "scib_1.1.7/LICENSE.txt" in listed


@pytest.mark.parametrize("break_it, message", [
    (lambda d, site: (d / sm.SWEEP / "inputs" / "latents.tsv").write_text(
        (d / sm.SWEEP / "inputs" / "latents.tsv").read_text().splitlines()[0] + "\n"), "have no latent"),
    (lambda d, site: (d / sm.SWEEP / "inputs" / "scores_list.txt").write_text(
        "\n".join((d / sm.SWEEP / "inputs" / "scores_list.txt").read_text().split()[:-1])), "score rows for"),
    (lambda d, site: (site / "scib-1.1.7.dist-info" / "licenses" / "LICENSE.txt").unlink()
     or (site / "scib-1.1.7.dist-info" / "licenses").rmdir() or (site / "scib-1.1.7.dist-info").rename(site / "x"), "missing"),
])
def test_collect_refuses_incomplete_inputs(root, tmp_path, break_it, message):
    d, site = root
    break_it(d, site)
    with pytest.raises(ValueError, match=message):
        sm.collect(d, tmp_path / "out", TWO, site)


def task_inputs(tmp_path):
    names, cols = task_obs("pancreas")
    write_h5(tmp_path / "pancreas__scib.h5ad", names, cols)
    raw = sm.read_tsv(sm.git_show(*sm.GIT["raw_sources"]))
    rules = sm.constants(sm.git_show(*sm.GIT["rules"]), ["TASK_FAMILY", "CC_TASKS", "TRAJ_TASKS"])
    sc = sm.constants(sm.git_show(*sm.GIT["scorer"]), ["BATCH_METRICS", "BIO_METRICS"])
    metrics = dict(batch=sc["BATCH_METRICS"], bio=sc["BIO_METRICS"])
    man = pd.DataFrame([dict(tag="a", experiment="A1", task="pancreas", arm="none", seed=100, **ROW)])
    vals = {m: 0.5 for m in metrics["batch"] + metrics["bio"]}
    scores = pd.DataFrame([dict(tag="a", task="pancreas", hvg_overlap=np.nan, **dict(vals, trajectory=np.nan))])
    return dict(prepped=tmp_path, fingerprints=json.loads(json.dumps(FP)), raw=raw, rules=rules, metrics=metrics,
                manifest=man, scores=scores, tasks=["pancreas"])


def _fp(k, v):
    def f(a):
        a["fingerprints"]["pancreas__scib.h5ad"][k] = v
    return f


@pytest.mark.parametrize("change, message", [
    (_fp("shape", [1, 1]), "differ from the fingerprint"),
    (lambda a: a["fingerprints"]["pancreas__scib.h5ad"]["uns"].update(source_md5="0"), "disagree on the source md5"),
    (lambda a: a["manifest"].__setitem__("max_epochs", 399), "not the scvi-tools rule"),
    (lambda a: a["scores"].__setitem__("cell_cycle_conservation", np.nan), "differ from the rule's list"),
    (lambda a: a.update(scores=pd.concat([a["scores"], a["scores"].assign(cLISI=np.nan)])), "present in 1 of 2"),
])
def test_task_rows_refuses_inconsistent_records(tmp_path, change, message):
    a = task_inputs(tmp_path)
    change(a)
    with pytest.raises(ValueError, match=message):
        sm.task_rows(**a)


def test_task_rows_refuses_a_batch_in_two_parents_and_missing_labels(tmp_path):
    names, cols = task_obs("pancreas")
    cols["Batch"] = np.array([f"p{i % 2}" for i in range(len(names))])    # every batch split over two parents
    write_h5(tmp_path / "pancreas__scib.h5ad", names, cols)
    (tmp_path / "x").mkdir()
    a = dict(task_inputs(tmp_path / "x"), prepped=tmp_path)
    with pytest.raises(ValueError, match="more than one parent"):
        sm.task_rows(**a)
    write_h5(tmp_path / "pancreas__scib.h5ad", names, cols, missing="celltype")
    with pytest.raises(ValueError, match="missing values"):
        sm.h5_obs(tmp_path / "pancreas__scib.h5ad", ["celltype"])


def test_constants_and_replication_checks():
    with pytest.raises(ValueError, match="not found"):
        sm.constants("A = 1\n", ["A", "B"])
    man = pd.DataFrame([dict(tag="a", experiment="A1", task="pancreas", arm="none", seed=100, **ROW)])
    rep = man.assign(experiment="X1", seed=0, batch_size=64)
    with pytest.raises(ValueError, match="batch_size differs"):
        sm.replication_rows(rep, man)


def test_tables_reproduce_the_committed_tables(tmp_path):
    texts = sm.build()
    for f, t in texts.items():
        assert (sm.OUT / f).read_text() == t, f
    assert "0.2025" in texts["tableS2.tex"] and "(E-1)/400" in texts["tableS2.tex"]             # last-epoch KL weight
    assert "448 & 447" in texts["tableS4.tex"] and "1 of Scanorama on ATAC large" in texts["tableS4.tex"]
    assert "holds all cells and batches of Immune" in texts["tableS1.tex"]                       # containment, not name overlap
    assert "holds all cells and batches of ATAC small" in texts["tableS1.tex"]                   # barcodes with other suffixes
    assert sm.main(["--check"]) == 0
    assert sm.main(["--out", str(tmp_path)]) == 0 and (tmp_path / "tableS4.tex").read_text() == texts["tableS4.tex"]
    (tmp_path / "tableS1.tex").write_text("changed")
    assert sm.main(["--check", "--out", str(tmp_path)]) == 1


def test_main_collect_passes_the_data_root(monkeypatch, tmp_path):
    seen = {}
    monkeypatch.setattr(sm, "collect", lambda root, data: seen.update(root=root, data=data) or data)
    assert sm.main(["--collect", "--data-root", "R", "--data", str(tmp_path)]) == 0 and seen == dict(root="R", data=str(tmp_path))


def test_table_helpers_refuse_disagreeing_fits():
    data = sm.DATA
    settings = pd.read_csv(data / "settings.csv", keep_default_na=False)
    manifest = pd.read_csv(data / "manifest.tsv", sep="\t")
    tasks = pd.read_csv(data / "tasks.csv", keep_default_na=False)
    two = pd.concat([settings, settings[settings.field == "fit.adv_steps"].assign(value="7")])
    with pytest.raises(ValueError, match="distinct values"):
        sm.setting(two, "reference", "fit.adv_steps", "A1")
    with pytest.raises(ValueError, match="unexpected optimiser"):
        sm.adam({"cls": "SGD", "param_groups": [{}]})
    with pytest.raises(ValueError, match="sweep fits differ"):
        sm.table_s2(tasks, pd.concat([manifest, manifest.head(1).assign(n_latent=99)]), settings)
    gen = settings.field == "fit.optimizer0"
    other = settings[gen & (settings.arm == "mmd")].assign(value=json.dumps({"cls": "Adam", "param_groups": [{"lr": 1}]}))
    with pytest.raises(ValueError, match="differs from plain scVI"):
        sm.table_s2(tasks, manifest, pd.concat([settings[~(gen & (settings.arm == "mmd"))], other]))
    adv = settings[(settings.arm == "pooled") & (settings.field == "fit.optimizer1")]
    with pytest.raises(ValueError, match="adversary optimisers differ"):
        sm.table_s2(tasks, manifest, pd.concat([settings, adv.assign(value=json.dumps({"cls": "Adam", "param_groups": [{}]}))]))
    part = pd.Series(dict(cells=10, shares_cells_with="sim1:3"))                                   # partial overlap: batches only
    assert sm.relation(part, "sim1", 2, {"sim1": 20}) == "2 batches also in Simulation 1"
    assert sm.sci(0.001) == "$10^{-3}$" and sm.sci(0.01) == "0.01" and sm.seeds([0]) == "0" and sm.kl_final(400, 400) == 0.9975
