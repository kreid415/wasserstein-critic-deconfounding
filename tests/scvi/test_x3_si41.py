"""X3 equal cell count per dose (CONSTRAINTS.md SI-41; code check CR-04; lead decisions 2026-10-03), draw 'nested_v1'
(scripts/fit_paper_config.py x3_draw) and the exact oracle importance weights (depletion_oracle_weights).
Run in the scvi env: WCD_SRC=src PREPPED_DIR=<prepped_scib> python -m pytest -q tests/scvi/test_x3_si41.py

1. manifests/paper_manifest_stock_pilot_u5_b10_v2.tsv is the builder's output; every non-X3 row (and the header) is
   byte-identical, in order, to the tagged manifest manifests/paper_manifest_stock_pilot_u5_b10.tsv.
2. X3: 828 rows at the same positions, every field but tag / extra unchanged, one n_cells per task (= X3_N) at every
   dose, draw 'nested_v1' in every spec, all 828 tags new and unique.
3. subsample() on the real prepped obs: exactly N cells at every dose, dose 0 = the first N cells of the permutation,
   every declared type keeps h0 - round(dose/100 * h0), declared cells nested (decreasing) and refill nested
   (increasing), no dose-0 cell of another kind removed, maximal overlap between doses, identical across calls.
   The checker rejects the literal rule of the first SI-41 draft (which empties immune_hum_mou at dose 50).
4. The dose-0 reference rule still selects each task's manifest reference (tests/scvi/test_x3_x13_design.py and
   below); N re-derived from the prepped files.
5. Importance weights: the weighted (batch, type) counts of every dose equal the dose-0 counts exactly (real data,
   doses 50/80/95, and a toy case), the depleted batch's weighted composition equals its dose-0 composition, the
   formula equals SI-25's 1 / keep fraction and 1 when nothing is refilled, refill-only pairs get weight 0, no batch
   is all-zero; doses 0 / 100, a lost pair and a spec without the draw key are refused. The checker rejects SI-25's
   old global kappa under refill.
"""
import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import build_paper_manifest as bpm  # noqa: E402
import fit_paper_config as fpc  # noqa: E402

ad = pytest.importorskip("anndata")
NEEDS_DATA = pytest.mark.skipif(not os.environ.get("PREPPED_DIR"), reason="needs PREPPED_DIR (prepped scIB h5ad files)")
TAGGED = os.path.join(ROOT, "manifests", "paper_manifest_stock_pilot_u5_b10.tsv")
V2 = os.path.join(ROOT, "manifests", "paper_manifest_stock_pilot_u5_b10_v2.tsv")
DOSES = [0, 50, 80, 95, 100]


def _lines(p):
    with open(p) as f:
        return f.read().splitlines()


def _exp(line):
    return line.split("\t")[1]


def _spec(task, dose):
    b, types, _ref = bpm.X3_TARGETS[task]
    return dict(kind="composition", batch=b, types=types, deplete_pct=dose, n_cells=bpm.X3_N[task], draw=bpm.X3_DRAW)


def _obs_adata(sizes, seed=0):
    """obs-only AnnData, cells shuffled: sizes[(batch, celltype)] = count."""
    rows = [(b, c) for (b, c), n in sizes.items() for _ in range(n)]
    rows = [rows[i] for i in np.random.default_rng(seed).permutation(len(rows))]
    obs = pd.DataFrame(rows, columns=["batch", "celltype"], index=[f"c{i}" for i in range(len(rows))])
    return ad.AnnData(obs=obs)


def _read_obs(task):
    import h5py
    with h5py.File(os.path.join(os.environ["PREPPED_DIR"], f"{task}__scib.h5ad"), "r") as f:
        obs = ad.io.read_elem(f["obs"])
    return ad.AnnData(obs=obs[["batch", "celltype"]].copy())


# ---- 1, 2: manifest ----------------------------------------------------------------------------------------------
def test_v2_manifest_is_the_builder_output(tmp_path):
    out = tmp_path / "v2.tsv"
    subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "build_paper_manifest.py"), "--backbone", "stock",
                    "--design", "pilot", "--uncond-seeds", "5", "--bary-iter", "10", "--out", str(out)], check=True)
    assert out.read_bytes() == open(V2, "rb").read()


def non_x3_violations(old_lines, new_lines):
    """Differences between two manifests outside X3 (header, column line, non-X3 rows in order)."""
    bad = []
    if old_lines[:2] != new_lines[:2]:
        bad.append("header or column line differs")
    o = [x for x in old_lines[2:] if _exp(x) != "X3"]
    n = [x for x in new_lines[2:] if _exp(x) != "X3"]
    if len(o) != len(n):
        bad.append(f"{len(o)} vs {len(n)} non-X3 rows")
    bad += [f"row {i} differs" for i, (x, y) in enumerate(zip(o, n)) if x != y][:5]
    return bad


def test_non_x3_rows_are_byte_identical_to_the_tagged_manifest():
    old, new = _lines(TAGGED), _lines(V2)
    assert len(old) == len(new) == 2 + 6846
    assert non_x3_violations(old, new) == []
    assert sum(_exp(x) != "X3" for x in new[2:]) == 6018


def test_non_x3_checker_detects_a_changed_row():
    """Mutation check (fail-loud R11): one changed non-X3 row (a seed) is reported."""
    old = _lines(TAGGED)
    i = next(k for k, x in enumerate(old) if k >= 2 and _exp(x) == "X1")
    bad = old.copy()
    bad[i] = bad[i].replace("\t0\tauto\t", "\t1\tauto\t", 1) if "\t0\tauto\t" in bad[i] else bad[i] + "x"
    assert bad[i] != old[i] and non_x3_violations(old, bad)


def test_x3_rows_equal_n_draw_and_new_tags():
    old, new = _lines(TAGGED), _lines(V2)
    cols = new[1].split("\t")
    pos_o = [k for k, x in enumerate(old[2:]) if _exp(x) == "X3"]
    pos_n = [k for k, x in enumerate(new[2:]) if _exp(x) == "X3"]
    assert pos_o == pos_n and len(pos_n) == 828
    O = pd.DataFrame([old[2 + k].split("\t") for k in pos_o], columns=cols)
    N = pd.DataFrame([new[2 + k].split("\t") for k in pos_n], columns=cols)
    for c in cols:
        if c not in ("tag", "extra"):
            assert (O[c] == N[c]).all(), c
    so = O.extra.map(lambda e: json.loads(e)["subsample"])
    sn = N.extra.map(lambda e: json.loads(e)["subsample"])
    for a, b in zip(so, sn):     # only n_cells changes and draw is added
        assert {k: v for k, v in a.items() if k != "n_cells"} == {k: v for k, v in b.items() if k not in ("n_cells", "draw")}
    assert bpm.X3_DRAW == fpc.X3_DRAW == "nested_v1"
    assert set(d["draw"] for d in sn) == {"nested_v1"}
    for task, g in N.assign(spec=sn).groupby("task"):
        assert sorted({d["n_cells"] for d in g.spec}) == [bpm.X3_N[task]], task
        assert sorted({d["deplete_pct"] for d in g.spec}) == DOSES
    assert (O.tag != N.tag).all() and N.tag.is_unique and not set(O.tag) & set(N.tag)
    allv2 = pd.read_csv(V2, sep="\t", comment="#", dtype=str)
    assert allv2.tag.is_unique and len(allv2) == 6846
    assert set(N.max_epochs) == {"400"}        # scvi heuristic at N <= 20,000, as before


# ---- 3, 4: the draw on the real prepped obs -----------------------------------------------------------------------
def draw_violations(obs_adata, task, draw):
    """SI-41 / nested_v1 properties of draw(adata, spec) -> AnnData over the five doses of one task."""
    b, types, _ref = bpm.X3_TARGETS[task]
    n = bpm.X3_N[task]
    names = np.asarray(obs_adata.obs_names)
    o = obs_adata.obs.astype(str)
    decl = set(names[((o.batch == b) & o.celltype.isin(types)).to_numpy()])
    perm = np.random.default_rng(fpc.X3_PERMUTATION_SEED).permutation(obs_adata.n_obs)
    K = {d: list(draw(obs_adata, _spec(task, d)).obs_names) for d in DOSES}
    bad = [f"dose {d}: {len(k)} cells" for d, k in K.items() if len(k) != n or len(set(k)) != n]
    if bad:
        return bad
    S = {d: set(k) for d, k in K.items()}
    if S[0] != set(names[perm[:n]]):
        bad.append("dose 0 is not the first N cells of the permutation")
    ct = dict(zip(names, o.celltype))
    h0 = {t: sum(1 for c in S[0] & decl if ct[c] == t) for t in types}
    for d in DOSES:
        for t in types:
            kept = sum(1 for c in S[d] & decl if ct[c] == t)
            if kept != h0[t] - int(round(h0[t] * d / 100)):
                bad.append(f"dose {d}: type {t} keeps {kept}, expected {h0[t] - int(round(h0[t] * d / 100))}")
    for d1, d2 in zip(DOSES, DOSES[1:]):
        if not (S[d2] & decl) <= (S[d1] & decl):
            bad.append(f"declared cells of dose {d2} not within dose {d1}")
        if not (S[d1] - decl) <= (S[d2] - decl):
            bad.append(f"refill of dose {d1} not within dose {d2}")
    for d1 in DOSES:
        for d2 in DOSES:
            want = n - abs(len(S[d1] & decl) - len(S[d2] & decl))
            if len(S[d1] & S[d2]) != want:
                bad.append(f"doses {d1}/{d2} share {len(S[d1] & S[d2])} cells, maximum {want}")
    if list(draw(obs_adata, _spec(task, 50)).obs_names) != K[50]:
        bad.append("not deterministic")
    return bad


def _literal_first_draft(a, spec):
    """Known-bad draw (the first SI-41 wording): drop the declared cells in permutation order over the whole task,
    keep the first N remaining cells of the permutation."""
    o = a.obs.astype(str)
    hit = ((o.batch == spec["batch"]) & o.celltype.isin(spec["types"])).to_numpy()
    perm = np.random.default_rng(fpc.X3_PERMUTATION_SEED).permutation(a.n_obs)
    h = perm[hit[perm]]
    dropped = np.zeros(a.n_obs, dtype=bool)
    dropped[h[:int(round(len(h) * spec["deplete_pct"] / 100))]] = True
    return a[np.sort(perm[~dropped[perm]][:spec["n_cells"]])].copy()


@NEEDS_DATA
@pytest.mark.parametrize("task", sorted(bpm.X3_TARGETS))
def test_draw_has_the_si41_properties_on_real_obs(task):
    a0 = _read_obs(task)
    assert draw_violations(a0, task, lambda a, s: fpc.subsample(a, s, 0)) == []
    o = a0.obs.astype(str)
    b, types, ref = bpm.X3_TARGETS[task]
    n_decl = int(((o.batch == b) & o.celltype.isin(types)).sum())
    assert bpm.X3_N[task] == min(20000, a0.n_obs - n_decl)                 # the dose-100 size (capped)
    spec0 = _spec(task, 0)
    k0 = fpc.subsample(a0, spec0, 0)
    k0.obs["batch"] = k0.obs["batch"].astype(str).astype("category")
    import x3_design_profile as x3p
    assert str(x3p._repo_data_module().select_reference_batch(k0, "batch", "celltype")) == ref   # SI-31 rule


@NEEDS_DATA
def test_checker_rejects_the_literal_first_draft():
    """Mutation check (fail-loud R11): the first-draft rule fails the checker (immune_hum_mou keeps 0 declared
    cells from dose 50 on; on the full-size tasks the per-type counts and the overlap are not as specified)."""
    for task in ("immune_hum_mou", "pancreas"):
        bad = draw_violations(_read_obs(task), task, _literal_first_draft)
        assert bad, task
    a = _literal_first_draft(_read_obs("immune_hum_mou"), _spec("immune_hum_mou", 50))
    b, types, _ = bpm.X3_TARGETS["immune_hum_mou"]
    assert int(((a.obs.batch.astype(str) == b) & a.obs.celltype.astype(str).isin(types)).sum()) == 0


def test_draw_refusals():
    a0 = _obs_adata({("b0", "t0"): 100, ("b0", "t1"): 60, ("b1", "t0"): 80, ("b1", "t1"): 40, ("b2", "t1"): 50})
    good = dict(kind="composition", batch="b0", types=["t0"], deplete_pct=50, n_cells=230, draw="nested_v1")
    fpc.subsample(a0, good, 0)
    for change, exc, msg in [({"draw": None}, ValueError, "draw"), ({"draw": "nested_v0"}, ValueError, "draw"),
                             ({"n_cells": 231}, ValueError, "n_cells"), ({"deplete_pct": 101}, ValueError, "deplete_pct"),
                             ({"types": ["t9"]}, ValueError, "no cell"), ({"extra_key": 1}, KeyError, "consume")]:
        spec = {k: v for k, v in {**good, **change}.items() if v is not None}
        with pytest.raises(exc, match=msg):
            fpc.subsample(a0, spec, 0)


# ---- 5: importance weights --------------------------------------------------------------------------------------
def iw_violations(a0, spec, weights_fn):
    """Weighted (batch, type) counts of dose d vs the dose-0 counts; depleted batch's weighted composition."""
    a, info = fpc.subsample(a0, spec, 0, return_info=True)
    w, _ = weights_fn(a, spec, info)
    k0 = fpc.group_counts(fpc.subsample(a0, dict(spec, deplete_pct=0), 0).obs)
    kd = fpc.group_counts(a.obs)
    bad = []
    for b in set(k0) | set(kd):
        for c in set(k0.get(b, {})) | set(kd.get(b, {})):
            got = w.get(b, {}).get(c, 0.0) * kd.get(b, {}).get(c, 0)
            if abs(got - k0.get(b, {}).get(c, 0)) > 1e-9 * max(1, k0.get(b, {}).get(c, 0)):
                bad.append(f"({b}, {c}): weighted {got}, dose 0 {k0.get(b, {}).get(c, 0)}")
    bb = str(spec["batch"])
    tot_w = sum(w[bb][c] * n for c, n in kd[bb].items())
    tot_0 = sum(k0[bb].values())
    for c, n0 in k0[bb].items():
        if abs(w[bb][c] * kd[bb].get(c, 0) / tot_w - n0 / tot_0) > 1e-12:
            bad.append(f"depleted batch composition of {c} differs")
    for b, row in w.items():
        if not any(v * kd[b][c] > 0 for c, v in row.items()):
            bad.append(f"batch {b} has only zero weights")
    return bad


def _si25_global_kappa(a, spec, info):
    """Known-bad weights: SI-25's global formula (1 / keep fraction on the declared pairs, 1 elsewhere)."""
    keep = sum(v["kept"] for v in info["per_type"].values()) / sum(v["h0"] for v in info["per_type"].values())
    types = set(map(str, spec["types"]))
    w = {}
    for b, row in fpc.group_counts(a.obs).items():
        w[b] = {c: (1 / keep if (b == str(spec["batch"]) and c in types) else 1.0) for c in row}
    return w, keep


TOY = {("b0", "t0"): 100, ("b0", "t1"): 60, ("b0", "t2"): 7, ("b1", "t0"): 80, ("b1", "t1"): 40, ("b2", "t1"): 50,
       ("b2", "t3"): 9}


@pytest.mark.parametrize("dose", [50, 80, 95])
def test_weights_restore_the_dose0_counts_toy(dose):
    a0 = _obs_adata(TOY, seed=3)
    spec = dict(kind="composition", batch="b0", types=["t0", "t1"], deplete_pct=dose, n_cells=180, draw="nested_v1")
    assert iw_violations(a0, spec, fpc.depletion_oracle_weights) == []
    assert iw_violations(a0, spec, _si25_global_kappa)                    # mutation check: old formula fails


@NEEDS_DATA
@pytest.mark.parametrize("task", sorted(bpm.X3_TARGETS))
def test_weights_restore_the_dose0_counts_real(task):
    a0 = _read_obs(task)
    for dose in (50, 80, 95):
        spec = _spec(task, dose)
        assert iw_violations(a0, spec, fpc.depletion_oracle_weights) == [], (task, dose)
        a, info = fpc.subsample(a0, spec, 0, return_info=True)
        w, keep = fpc.depletion_oracle_weights(a, spec, info)
        assert all(abs(k - (1 - dose / 100)) < 0.02 for k in keep.values()), keep


def test_weights_equal_si25_without_refill():
    """No refill (K_d = K0 minus the dropped cells): w = 1 / keep fraction on the depleted pair, 1 elsewhere."""
    a0 = _obs_adata({("b0", "t0"): 100, ("b0", "t1"): 60, ("b1", "t0"): 80, ("b1", "t1"): 40}, seed=1)
    spec = dict(kind="composition", batch="b0", types=["t0"], deplete_pct=80, n_cells=180, draw="nested_v1")
    _, info = fpc.subsample(a0, spec, 0, return_info=True)
    k0 = a0[np.asarray(fpc.x3_draw(a0.obs, dict(spec, deplete_pct=0))[0])].copy()
    o = k0.obs.astype(str)
    decl = np.flatnonzero(((o.batch == "b0") & (o.celltype == "t0")).to_numpy())
    h0 = len(decl)
    kd = k0[np.setdiff1d(np.arange(k0.n_obs), decl[:int(round(h0 * 0.8))])].copy()   # drop, no refill
    w, _ = fpc.depletion_oracle_weights(kd, spec, info)
    kappa = (h0 - int(round(h0 * 0.8))) / h0
    assert abs(w["b0"]["t0"] - 1 / kappa) < 1e-12
    assert w["b0"]["t1"] == w["b1"]["t0"] == w["b1"]["t1"] == 1.0


def test_refill_only_pairs_get_zero_weight_and_refusals():
    a0 = _obs_adata({("b0", "t0"): 100, ("b0", "t1"): 60, ("b1", "t0"): 80, ("b1", "t1"): 40, ("b2", "t9"): 3}, seed=7)
    spec = dict(kind="composition", batch="b0", types=["t0"], deplete_pct=95, n_cells=183, draw="nested_v1")
    a, info = fpc.subsample(a0, spec, 0, return_info=True)
    w, _ = fpc.depletion_oracle_weights(a, spec, info)
    k0 = info["k0_counts"]
    for b, row in w.items():
        for c, v in row.items():
            assert (v == 0) == (k0.get(b, {}).get(c, 0) == 0)
    for dose in (0, 100):
        s = dict(spec, deplete_pct=dose)
        a, info = fpc.subsample(a0, s, 0, return_info=True)
        with pytest.raises(ValueError, match="0 < deplete_pct < 100"):
            fpc.depletion_oracle_weights(a, s, info)
    a, info = fpc.subsample(a0, spec, 0, return_info=True)
    lost = a[(a.obs.batch.astype(str) != "b1") | (a.obs.celltype.astype(str) != "t1")].copy()
    with pytest.raises(ValueError, match="cannot be restored"):
        fpc.depletion_oracle_weights(lost, spec, info)
