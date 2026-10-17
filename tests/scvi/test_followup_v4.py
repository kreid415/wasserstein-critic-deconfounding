"""Follow-up design v4 (CONSTRAINTS.md SI-93..SI-98): manifest, X8 nested draw, X7 depleted reference, X12 16-run design.
Run in the scvi env: WCD_SRC=src [PREPPED_DIR=<prepped_scib>] python -m pytest -q tests/scvi/test_followup_v4.py
(the optimiser and no-penalty checks of SI-98 / I036 are in tests/scvi/test_adversarial_plan_v2.py).

1. The v4 manifest differs from v3 in X3 / X6 / X7 / X8 / X12 only; row counts per experiment and decoder; X3's oracle-weighted
   rows under both decoders; X8 draws N // 16 cells per batch; the depleted-reference rows name the depleted batch; the new
   X6 arm resolves its lambda from the discriminator (BASE_ARM). The v3 manifest fails the same checks.
2. X12's 16 runs are a resolution V design: all 5 main effects and 10 two-factor interactions are mutually orthogonal; the
   8-run design (D = AB, E = AC) is rejected by the same check.
3. x8_draw: V * N // 16 cells, equal cells per batch, batches and cells nested from V = 2 to 16, two different chains,
   ineligible (small) batches never drawn. The earlier per-V draw is rejected by the same nesting check.
4. X7 depleted reference: x7_draw removes the listed types from the reference batch and thins it to n_reference cells at
   every level, other batches untouched; bad specs refused; the reference must be the depleted batch. With PREPPED_DIR: X7_DEPLETED and the X8 batch eligibility are re-derived from the prepped files.
"""
import json
import os
import sys

import numpy as np
import pandas as pd
import pytest

from tests.scvi._require import import_or_skip, needs_env  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import build_paper_manifest as bpm  # noqa: E402
import cost_model  # noqa: E402
import fit_paper_config as fpc  # noqa: E402
import prereg_rules as P  # noqa: E402

ad = import_or_skip("anndata")
CHANGED = ["X3", "X6", "X7", "X8", "X12"]
BB = bpm.BACKBONES["stock"]


def _build(design, runs=8):
    return pd.DataFrame(bpm.finalize(bpm.build(BB, "pilot", 3, 5, x12_runs=runs, followup_design=design))).astype(str)


def _spec(r):
    return json.loads(r.extra).get("subsample", {})


def _check_v4(m, v3):
    """The v4 design of the 2026-10-17 decisions; raises AssertionError for any other manifest."""
    out = ~m.experiment.isin(CHANGED)
    assert m.tag[out].tolist() == v3.tag[~v3.experiment.isin(CHANGED)].tolist()
    got = m[m.experiment.isin(CHANGED + ["X15"])].groupby(["experiment", "cond"]).size().to_dict()
    assert got == {("X3", "1"): 828, ("X3", "0"): 828, ("X6", "1"): 243, ("X6", "0"): 243, ("X7", "1"): 864,
                   ("X7", "0"): 864, ("X8", "1"): 504, ("X8", "0"): 504, ("X12", "1"): 576, ("X12", "0"): 576,
                   ("X15", "1"): 84}, got
    assert not m.tag.duplicated().any()
    iw = m.extra.str.contains('"iw"')
    assert iw.sum() == 576 and (m.cond[iw].value_counts() == 288).all()      # weighted rows under both decoders
    x8 = m[m.experiment == "X8"]
    for r in x8.itertuples():
        sp = _spec(r)
        per = bpm.X8_TOTAL[r.task] // 16
        assert (sp["draw"], sp["cells_per_batch"], sp["n_cells"]) == (fpc.X8_DRAW, per, sp["n_batches"] * per)
        assert r.max_epochs == str(bpm.scvi_epochs(sp["n_cells"])) and set(x8.lam) <= {"0", "matched"}
    x7 = m[m.experiment == "X7"]
    dep = x7[x7.extra.str.contains("reference_composition")]
    assert len(dep) == 432 and dep.extra.str.contains('"types": \\[\\]').sum() == 144 and set(dep.arm) == {"reference", "reference_fixed", "mmd_ref", "discriminator_ref"}
    assert all(r.reference == _spec(r)["batch"] for r in dep.itertuples())
    for r in dep.itertuples():
        ref, n_ref, n, levels = bpm.X7_DEPLETED[r.task]
        sp = _spec(r)
        assert (sp["draw"], sp["n_reference"], sp["n_cells"], r.max_epochs) == (fpc.X7_DRAW, n_ref, n, str(bpm.scvi_epochs(n)))
        assert sp["types"] in levels.values()
    assert set(x7.lam) == {"matched_lo", "matched", "matched_hi"}
    new = m[m.arm == "discriminator_sn_critic_opt"]
    assert len(new) == 54 and set(new.n_critic) == {"1"} and set(new.lam) == {"matched_lo", "matched", "matched_hi"}
    assert P.BASE_ARM["discriminator_sn_critic_opt"] == "discriminator"
    assert "discriminator_sn_critic_opt" in fpc.ADVERSARIAL
    assert cost_model.arm_label("discriminator_sn_critic_opt", {}) == "discriminator_sn"


def test_v4_manifest_design():
    v3, v4 = _build("v3"), _build("v4", runs=16)
    assert (bpm.X8_DRAW, bpm.X7_DRAW) == (fpc.X8_DRAW, fpc.X7_DRAW)
    _check_v4(v4, v3)
    with pytest.raises(AssertionError):                       # the checker rejects v3 itself
        _check_v4(v3, v3)
    with pytest.raises(ValueError, match="followup_design"):
        bpm.build(BB, "pilot", 3, 5, followup_design="v5")


def _res_v(f):
    """f: 0/1 DataFrame, columns A..E, one row per run. Raises unless the 5 main effects and 10 two-factor interactions
    are mutually orthogonal and orthogonal to the mean (resolution V)."""
    s = 2 * f.to_numpy(dtype=float) - 1
    cols = [np.ones(len(s))] + [s[:, i] for i in range(5)] + [s[:, i] * s[:, j] for i in range(5) for j in range(i + 1, 5)]
    X = np.stack(cols, axis=1)
    assert np.allclose(X.T @ X, len(s) * np.eye(X.shape[1])), "effects are aliased"


def _x12_factors(runs):
    m = _build("v4", runs=runs) if runs == 16 else _build("v3")
    r = m[(m.experiment == "X12") & (m.cond == "1") & (m.task == "atac_small") & (m.seed == "10")
          & (m.arm == "discriminator") & (m.lam == "matched_lo")]
    ex = r.extra.map(lambda s: json.loads(s))
    f = pd.DataFrame({"A": (r.n_latent == "30"), "B": ex.map(lambda e: e["adv_width"] == 128),
                      "C": ex.map(lambda e: e["adv_lr"] == 1e-3), "D": (r.batch_size == "512"),
                      "E": (r.decoder == "LinearSCVI")}).astype(int)
    assert len(f) == runs and not f.duplicated().any()
    return f


def test_x12_sixteen_runs_are_resolution_v():   # test-reason: the design property (no aliasing of effects up to two factors), not a code line
    _res_v(_x12_factors(16))
    with pytest.raises(AssertionError, match="aliased"):      # the 8-run design (resolution III) is rejected
        _res_v(_x12_factors(8))


def _obs(sizes, seed=0):
    rows = [(b, c) for (b, c), n in sizes.items() for _ in range(n)]
    rows = [rows[i] for i in np.random.default_rng(seed).permutation(len(rows))]
    return ad.AnnData(obs=pd.DataFrame(rows, columns=["batch", "celltype"], index=[f"c{i}" for i in range(len(rows))]))


def _nested(sets):
    """sets: {V: set of cell indices}; raises unless each set lies inside the next larger one."""
    vs = sorted(sets)
    for lo, hi in zip(vs, vs[1:]):
        assert sets[lo] <= sets[hi], f"V={lo} is not inside V={hi}"


def test_x8_draw_is_nested_with_equal_cells_per_batch():
    a = _obs({(f"b{i:02d}", f"t{j}"): 40 if i < 3 else 30 for i in range(20) for j in range(3)})   # b00-b02: 120, rest 90
    per = 100                                           # only b00-b02 have >= 100 cells: too few for V = 16
    sp = lambda v, s, per: dict(kind="batches", n_batches=v, subset=s, n_cells=v * per, cells_per_batch=per,  # noqa: E731
                                draw=fpc.X8_DRAW)
    with pytest.raises(ValueError, match="only 3 batches"):
        fpc.x8_draw(a.obs, sp(2, 0, per))
    per = 90                                            # all 20 batches eligible
    idx = {(v, s): fpc.x8_draw(a.obs, sp(v, s, per)) for v in (2, 4, 8, 16) for s in (0, 1)}
    for (v, s), i in idx.items():
        counts = a.obs.iloc[i].batch.value_counts()
        assert len(i) == v * per and len(counts) == v and (counts == per).all()
        assert np.array_equal(i, fpc.x8_draw(a.obs, sp(v, s, per)))
    for s in (0, 1):
        _nested({v: set(idx[(v, s)]) for v in (2, 4, 8, 16)})
        _nested({v: set(a.obs.iloc[idx[(v, s)]].batch) for v in (2, 4, 8, 16)})
    assert set(a.obs.iloc[idx[(2, 0)]].batch) != set(a.obs.iloc[idx[(2, 1)]].batch)    # two different chains
    # the earlier per-V draw (spec without `draw`) fails the nesting check: known-bad input
    old = {v: set(fpc.subsample(a, dict(kind="batches", n_batches=v, subset=0, n_cells=v * per), 0).obs_names)
           for v in (2, 4, 8, 16)}
    with pytest.raises(AssertionError, match="not inside"):
        _nested({v: {int(n[1:]) for n in names} for v, names in old.items()})
    sub = fpc.subsample(a, sp(4, 0, per), 0)
    assert sub.n_obs == 4 * per
    with pytest.raises(ValueError, match="composition"):
        fpc.subsample(a, sp(4, 0, per), 0, return_info=True)
    with pytest.raises(ValueError):                    # an unknown subsample kind is refused
        fpc.subsample(a, dict(kind="unknown"), 0)
    with pytest.raises(KeyError, match="does not consume"):
        fpc.x8_draw(a.obs, {**sp(2, 0, per), "extra_key": 1})
    with pytest.raises(ValueError, match="must equal"):
        fpc.x8_draw(a.obs, {**sp(2, 0, per), "n_cells": 5})


def test_x7_depleted_reference_spec():
    sizes = {("ref", f"t{j}"): 40 + 20 * j for j in range(5)}                       # 40, 60, 80, 100, 120 = 400 cells
    sizes.update({(b, f"t{j}"): 30 for b in ("o1", "o2") for j in range(5)})         # 300 other cells
    a = _obs(sizes)
    rare = {"none": [], "two_thirds": ["t0", "t1"], "one_third": ["t0", "t1", "t2"]}  # rarest first; keeps 5, 3 and 2 types
    n_ref = 220                                                                      # 100 + 120: the deeper level's cells
    spec = lambda t: dict(kind="reference_composition", batch="ref", types=t, n_reference=n_ref,  # noqa: E731
                          n_cells=n_ref + 300, draw=fpc.X7_DRAW)
    for level, types in rare.items():
        sub = fpc.subsample(a, spec(types), 0)
        ref = sub.obs[sub.obs.batch == "ref"]
        assert sub.n_obs == 520 and len(ref) == n_ref and not ref.celltype.isin(types).any()
        assert (sub.obs.batch != "ref").sum() == 300 and sub.obs_names.is_unique          # other batches untouched
        pos = np.where(a.obs.batch.values == "ref")[0]                                    # spec: the first n_reference cells of one
        order = pos[np.random.default_rng(fpc.X7_PERMUTATION_SEED).permutation(len(pos))]  # fixed permutation of the batch, among
        want = order[~a.obs.celltype.iloc[order].isin(types).values][:n_ref]               # the kept types: the same permutation
        assert set(ref.index) == set(a.obs_names[want]), level                             # at every level, control included
    none = fpc.subsample(a, spec([]), 0).obs
    assert none[none.batch == "ref"].celltype.nunique() == 5                       # same-size control keeps every type
    deep = fpc.subsample(a, spec(rare["one_third"]), 0)
    assert set(deep.obs[deep.obs.batch == "ref"].celltype) == {"t3", "t4"}               # all kept cells of t3, t4
    for bad, exc, msg in [(dict(n_reference=300, n_cells=600), ValueError, "keeps only"),
                          (dict(types=["t0", "zz"]), ValueError, "present"), (dict(n_cells=5), ValueError, "must equal"),
                          (dict(draw="nested_v1"), ValueError, "needs draw"), (dict(extra_key=1), KeyError, "does not consume")]:
        with pytest.raises(exc, match=msg):                                              # known-bad specs
            fpc.subsample(a, {**spec(rare["one_third"]), **bad}, 0)
    with pytest.raises(ValueError, match="composition"):
        fpc.subsample(a, spec(rare["one_third"]), 0, return_info=True)
    names = ["o1", "o2", "ref"]
    sp = spec(rare["one_third"])
    assert fpc.resolve_reference("t", "ref", names, sp) == "ref"
    for bad in ("auto", "1", "o1"):                        # not the depleted batch: known-bad inputs
        with pytest.raises(ValueError, match="must be the depleted batch"):
            fpc.resolve_reference("t", bad, names, sp)


@needs_env("PREPPED_DIR")
def test_x7_depleted_table_and_x8_eligibility_follow_the_rules():   # test-reason: re-derives the hard-coded design tables from the data
    import h5py
    import importlib.util
    spec = importlib.util.spec_from_file_location("wcd_data", os.path.join(ROOT, "src", "wcd_vae", "wcd", "data.py"))
    data = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(data)

    def obs_of(task):
        with h5py.File(os.path.join(os.environ["PREPPED_DIR"], f"{task}__scib.h5ad"), "r") as f:
            return ad.io.read_elem(f["obs"])[["batch", "celltype"]].astype(str)
    for task, (ref, n_ref, n, levels) in bpm.X7_DEPLETED.items():
        obs = obs_of(task)
        assert str(data.select_reference_batch(ad.AnnData(obs=obs), "batch", "celltype")) == ref
        cnt = obs[obs.batch == ref].celltype.value_counts()
        order = sorted(cnt.index, key=lambda c: (cnt[c], c))                       # rarest first, ties by name
        c = len(order)
        want = {"none": [], "two_thirds": order[:c - round(c * 2 / 3)], "one_third": order[:c - round(c / 3)]}
        assert levels == want, task
        kept = {k: int(cnt.sum() - cnt[t].sum()) for k, t in levels.items()}
        assert n_ref == min(kept.values()) and n == n_ref + len(obs) - int(cnt.sum()), task
    for task, total in bpm.X8_TOTAL.items():
        sizes = obs_of(task).batch.value_counts()
        assert (sizes >= total // 16).sum() >= 16, task
