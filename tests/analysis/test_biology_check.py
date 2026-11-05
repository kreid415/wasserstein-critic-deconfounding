"""scripts/biology_check.py (docs/BIOLOGY_CHECK.md) on a synthetic world: two tasks of 229 cells (pancreas: marker genes, sparse X;
sim1: none), a manifest, scores and latent files laid out as scripts/sweep_inputs.py writes them. Expected values come from the
planted case and from brute-force numpy (distance matrix, argsort, majority vote), not from the code under test. Synthetic data only."""
import hashlib
import itertools
import json
import os
import subprocess
import sys

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scanpy as sc
from scipy import sparse
from scipy.spatial.distance import cdist

ROOT = os.environ.get("ACCEPT_WORKDIR") or os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for _p in (os.path.join(ROOT, "scripts"), os.path.join(ROOT, "tests", "prereg")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import analysis_x1 as A  # noqa: E402
import biology_check as B  # noqa: E402
import prereg_rules as P  # noqa: E402
import synth  # noqa: E402

TASKS = ["pancreas", "sim1"]
GENES = [f"g{i}" for i in range(40)]
N_HVG = 30
TYPES = np.array([t for _ in range(3) for t in "ABC" for _ in range(25)] + ["R", "R", "D", "D"])      # R and D are rare (2 < 2.29 cells)
BATCHES = np.array([f"b{b}" for b in range(3) for _ in "ABC" for _ in range(25)] + ["b0", "b1", "b0", "b0"])   # D: one batch, dropped
OBS = np.array([f"c{i}" for i in range(len(TYPES))])
CENTRE = dict(A=(0, 0, 0), B=(10, 0, 0), C=(0, 10, 0), R=(10, 10, 0), D=(0, 0, 10))
MERGING = ("mmd", "pca")                  # these put B on top of A: purity of both falls
GAIN = {"1": 0.05, "10": 0.20, "100": 0.30}
FIT_TYPES_COLS = ("task tag method setting decoder seed celltype n_cells rare purity ref_purity purity_diff lt_recall ref_lt_recall "
                  "lt_recall_diff marker_jaccard ref_marker_jaccard marker_jaccard_diff").split()
FITS_COLS = ("task tag method setting decoder seed n_types n_rare purity purity_diff rare_purity rare_purity_diff rare_lt_recall "
             "rare_lt_recall_diff lt_f1 marker_jaccard marker_jaccard_diff lam").split()
CONFUSION_COLS = "task method setting decoder true_type pred_type n_fits share_mean share_sd".split()


def mrow(experiment, task, arm, lam, cond, seed, n_latent="30"):
    r = dict.fromkeys(P.BPM.COLS, "-")
    r.update(experiment=experiment, task=task, arm=arm, lam=lam, cond=str(cond), seed=str(seed), n_latent=n_latent,
             tag=f"{experiment}_{task}_{arm}_{lam or 'na'}_c{cond}_s{seed}_n{n_latent}")
    return r


def manifest_rows():
    R = []
    for task, cond, seed in itertools.product(TASKS, (0, 1), (100, 101, 102)):
        R.append(mrow("A1", task, "none", "0", cond, seed))
        R += [mrow("A1", task, arm, lam, cond, seed) for arm in P.ADV_ARMS for lam in GAIN]
    for task in TASKS:
        R += [mrow("X1", task, "scvi_adv", "3", 1, s, "10") for s in (0, 1)] + [mrow("X13", task, "scanvi", "0", 1, s, "10") for s in (0, 1)]
        R += [mrow("X13", task, a, lam, 1, 0, "10") for a, lam in (("sysvi", "5"), ("harmony", "1"), ("scanorama", "1"), ("pca", ""))]
    return pd.DataFrame(R, columns=P.BPM.COLS)


def world_scores(r):
    """(status, B, C): every penalty gains GAIN over plain scVI (mmd more at 1; sim1 decoder 1 hardly: low signal); two planted failures."""
    if (r["task"], r["arm"], r["lam"], r["cond"], r["seed"]) == ("pancreas", "sinkhorn", "100", "0", "101") or \
            (r["task"], r["arm"], r["seed"]) == ("sim1", "scanvi", "1"):
        return "diverged", 0.0, 0.0
    b = 0.40 + synth._noise(f"b{r['task']}|{r['cond']}|{r['seed']}", 0.005)       # seed effect shared by every arm
    if r["arm"] in P.ADV_ARMS:
        gain = dict(GAIN, **({"1": 0.16} if r["arm"] == "mmd" else {}))
        b += gain[r["lam"]] / (20 if (r["task"], r["cond"]) == ("sim1", "1") else 1)
    elif r["arm"] != "none":
        b += 0.1
    return "ok", b, 0.7


def expected_matched(M):
    """The rule of docs section 1 on the planted scores: the failure-free strength whose seed-mean B is nearest b*, ties to the smaller."""
    rows = pd.DataFrame([dict(r, status=world_scores(r)[0], B=world_scores(r)[1]) for r in M[M.experiment == "A1"].to_dict("records")])
    got = {}
    for (task, cond), g in rows.groupby(["task", "cond"]):
        b0 = g[g.arm == "none"].B.mean()
        pts = {(arm, float(lam)): (h.status.eq("ok").all(), h.B.mean()) for (arm, lam), h in g[g.arm != "none"].groupby(["arm", "lam"])}
        bstar = b0 + 0.5 * (min(max(b for (a, _), (ok, b) in pts.items() if a == arm and ok) for arm in P.ADV_ARMS) - b0)
        for arm in P.ADV_ARMS:
            cand = sorted((abs(b - bstar), lam) for (a, lam), (ok, b) in pts.items() if a == arm and ok)
            got[(task, int(cond), arm)] = cand[0][1]
    return got


def write_latent(path, tag, arm):
    rng = np.random.default_rng(int(hashlib.sha1(tag.encode()).hexdigest()[:8], 16))
    centre = dict(CENTRE, B=(0.2, 0, 0) if arm in MERGING else CENTRE["B"])
    z = np.array([centre[t] for t in TYPES]) + 0.3 * rng.normal(size=(len(TYPES), 3))
    np.savez(path, z=z.astype(np.float32), obs_names=OBS, batch=BATCHES, celltype=TYPES)


def write_prepped(path, X, sparse_x, modality, obs_names=OBS, types=TYPES):
    a = ad.AnnData(X=sparse.csr_matrix(X) if sparse_x else X.copy(),
                   obs=pd.DataFrame({"celltype": types, "batch": BATCHES}, index=obs_names),
                   var=pd.DataFrame({"highly_variable": [True] * N_HVG + [False] * (len(GENES) - N_HVG)}, index=GENES))
    a.layers["counts"] = np.full(X.shape, 7.0, np.float32)        # must never be read
    a.uns["celltype_key"], a.uns["modality"] = "celltype", modality
    a.write_h5ad(path)


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    root = tmp_path_factory.mktemp("biology")
    X = np.random.default_rng(1).gamma(2.0, 0.5, size=(len(TYPES), len(GENES))).astype(np.float32)
    for t, g in zip("ABC", (0, 10, 20)):
        X[TYPES == t, g:g + 10] += 3.0                              # marker genes of A, B, C
    prepped, inputs, latents = (root / d for d in ("prepped", "inputs", "latents"))
    for d in (prepped, inputs, latents):
        d.mkdir()
    write_prepped(prepped / "pancreas__scib.h5ad", X, True, "rna")
    write_prepped(prepped / "sim1__scib.h5ad", X, False, "sim")
    M = manifest_rows()
    S, F = synth.scores(M, world_scores)
    for tag, arm in zip(M.tag, M.arm):
        if tag not in set(F.tag):
            write_latent(latents / f"{tag}.npz", tag, arm)
    S["latent_sha256"] = [P.sha256(str(latents / f"{t}.npz")) for t in S.tag]
    S.to_csv(inputs / "scores.csv", index=False)
    F.to_csv(inputs / "failures.csv", index=False)
    P.write_manifest(M, ["# synthetic"], str(inputs / "manifest.tsv"), "test")
    (inputs / "scores_list.txt").write_text(str(inputs / "scores.csv") + "\n")
    (inputs / "latents.tsv").write_text("".join(f"{t}\t{latents / (t + '.npz')}\n" for t in S.tag))
    (inputs / "inputs.json").write_text(json.dumps(dict(tasks=TASKS)))
    base = M[M.experiment != "A1"].assign(failed=lambda d: d.tag.isin(set(F.tag)))
    cov = base.groupby(["task", "arm"]).agg(expected=("tag", "size"), failed=("failed", "sum")).reset_index()
    cov.insert(3, "scored", cov.expected - cov.failed)
    cov.assign(missing=0)[["task", "arm", "expected", "scored", "failed", "missing"]].to_csv(inputs / "baseline_coverage.csv", index=False)
    L = B.load_sweep(str(inputs))
    return dict(root=root, inputs=str(inputs), prepped=str(prepped), X=X, M=M, L=L,
                fits=B.select_fits(L["out"], L["matched"], L["latents"]))


def brute(path, X=None, base=None):
    """Brute-force readouts of one saved embedding (distance matrix, argsort, majority vote of the 15 nearest)."""
    with np.load(path) as d:
        z, batch = d["z"], d["batch"]
    D = cdist(z, z)
    np.fill_diagonal(D, np.inf)
    idx = np.argsort(D, axis=1, kind="stable")[:, :15]
    own = (TYPES[idx] == TYPES[:, None]).mean(axis=1)
    dropped = [t for t in set(TYPES) if len(set(batch[TYPES == t])) == 1]
    keep = ~np.isin(TYPES, dropped)
    kz, kt, kb = z[keep], TYPES[keep], batch[keep]
    pred = np.empty(len(kt), dtype=kt.dtype)
    for b in set(kb):
        test = kb == b
        nn = np.argsort(cdist(kz[test], kz[~test]), axis=1, kind="stable")[:, :15]
        for i, row in zip(np.flatnonzero(test), nn):
            labels, counts = np.unique(kt[~test][row], return_counts=True)
            pred[i] = labels[np.argmax(counts)]
    types = sorted(set(kt))
    out = dict(purity={t: own[TYPES == t].mean() for t in sorted(set(TYPES))}, recall={t: (pred[kt == t] == t).mean() for t in types},
               conf=np.array([[(pred[kt == a] == b).mean() for b in types] for a in types]), jaccard={}, kept=types)
    if X is not None:
        smoothed = np.stack([X[row].astype(np.float64).mean(axis=0) for row in idx])
        out["jaccard"] = {t: len(set(base[t]) & set(m[t])) / len(set(base[t]) | set(m[t])) for m in [rank_genes(smoothed)] for t in "ABC"}
    return out


def rank_genes(X):
    a = ad.AnnData(X, obs=pd.DataFrame({"t": pd.Categorical(TYPES)}, index=OBS), var=pd.DataFrame(index=GENES[:N_HVG]))
    sc.tl.rank_genes_groups(a, "t", method="wilcoxon", use_raw=False, reference="rest", n_genes=20)
    return {t: list(a.uns["rank_genes_groups"]["names"][t]) for t in "ABC"}


# ---------------------------------------------------------------------------------------------------------------------------------
# small functions, by hand
# ---------------------------------------------------------------------------------------------------------------------------------
def test_neighbours_purity_rare_smooth():  # test-reason: hand-computed purity, the strict 1% boundary, identical points, sparse and dense smoothing
    Z = np.random.default_rng(0).normal(size=(40, 3))
    Z[1] = Z[0]                                          # identical coordinates: each is the other's nearest, never its own
    idx = B.knn_indices(Z, k=5)
    D = cdist(Z, Z)
    np.fill_diagonal(D, np.inf)
    assert idx.shape == (40, 5) and idx[0, 0] == 1 and idx[1, 0] == 0
    assert all(np.allclose(D[i, row], np.sort(D[i])[:5]) for i, row in enumerate(idx))      # the 5 smallest distances, in order; ties may pick either cell
    assert B.purity_per_type(np.array(list("aabbb")), np.array([[1, 2], [0, 2], [3, 4], [2, 4], [2, 3]])) == {"a": 0.5, "b": 1.0}
    assert B.rare_types(np.array(["x"] * 197 + ["y"] * 2 + ["z"])) == ["z"]      # 2 of 200 is not below 1%
    X, nbr = np.random.default_rng(2).normal(size=(6, 4)), np.random.default_rng(3).integers(0, 6, size=(6, 3))
    want = np.stack([X[row].mean(axis=0) for row in nbr])
    for matrix in (X, sparse.csr_matrix(X)):
        got = B.smooth(matrix, nbr)
        assert isinstance(got, np.ndarray) and np.allclose(got, want, atol=1e-12)


def test_markers_and_jaccard():  # test-reason: planted marker genes in score order, types under min_cells left out, the 2-cell stop
    rng = np.random.default_rng(0)
    labels = np.array(["A"] * 30 + ["B"] * 30 + ["D"] * 4)
    X = rng.gamma(2.0, 0.5, size=(len(labels), 25))
    X[:30, :5] += 6.0
    X[30:60, 5:10] += 6.0
    got = B.top_markers(X, labels, [f"g{i}" for i in range(25)], n=8)
    assert list(got) == ["A", "B"] and all(len(v) == 8 for v in got.values())     # D has 4 cells < 10
    assert set(got["A"][:5]) == {f"g{i}" for i in range(5)} and set(got["B"][:5]) == {f"g{i}" for i in range(5, 10)}
    assert B.jaccard(["a", "b"], ["b", "c"]) == pytest.approx(1 / 3) and B.jaccard(got["A"], got["A"]) == 1.0
    with pytest.raises(P.PreregError, match="'D'"):
        B.top_markers(X[:61], np.append(labels[:60], "D"), [f"g{i}" for i in range(25)])


def test_label_transfer_readouts():  # test-reason: dropped type, predictions of separated blobs, hand-computed recall and confusion, the stop
    Z = np.array([CENTRE[t] for t in TYPES]) + 0.3 * np.random.default_rng(0).normal(size=(len(TYPES), 3))
    keep, pred, dropped = B.transfer_predictions(Z, TYPES, BATCHES)
    assert dropped == ["D"] and keep.sum() == len(TYPES) - 2 and (pred[TYPES[keep] != "R"] == TYPES[keep][TYPES[keep] != "R"]).all()
    assert B.per_type_recall(TYPES[keep], pred)["R"] == 0.0           # one R cell in the training batches: never the majority
    true, wrong = np.array(list("aabb")), np.array(list("abbb"))
    assert B.per_type_recall(true, wrong) == {"a": 0.5, "b": 1.0}
    table = B.confusion_table(true, wrong)
    assert table.to_numpy().tolist() == [[0.5, 0.5], [0.0, 1.0]] and list(table.index) == list(table.columns) == ["a", "b"]
    with pytest.raises(P.PreregError, match="'c'"):
        B.confusion_table(true, np.array(list("abbc")))


def test_prepped_files(world):  # test-reason: only the highly variable genes of X (sparse or dense), never the counts layer; labels without reading X
    got = B.load_expression(os.path.join(world["prepped"], "pancreas__scib.h5ad"))
    assert got["X"].dtype == np.float32 and np.array_equal(got["X"], world["X"][:, :N_HVG]) and got["genes"] == GENES[:N_HVG]
    assert got["modality"] == "rna" and np.array_equal(got["obs_names"], OBS) and np.array_equal(got["celltype"], TYPES)
    dense = B.load_expression(os.path.join(world["prepped"], "sim1__scib.h5ad"))
    assert np.array_equal(dense["X"], got["X"])
    light = B._read_obs(os.path.join(world["prepped"], "sim1__scib.h5ad"))
    assert light["modality"] == "sim" and np.array_equal(light["celltype"], TYPES) and np.array_equal(light["obs_names"], OBS)


# ---------------------------------------------------------------------------------------------------------------------------------
# the sweep inputs and the fits chosen
# ---------------------------------------------------------------------------------------------------------------------------------
def test_sweep_inputs_and_fit_selection(world):  # test-reason: matched strengths, the settings, decoders and counts of every method, the stops on a missing latent or method
    L, fits = world["L"], world["fits"]
    want = expected_matched(world["M"])
    assert L["matched"] == want and len(set(want.values())) > 1 and list(L["coverage"].columns) == ["task", "arm", "expected", "scored", "failed", "missing"]
    assert L["info"]["tasks"] == TASKS and L["targets"]["sim1|1"]["bstar_low_signal"] and not L["targets"]["sim1|0"]["bstar_low_signal"]
    assert list(fits.columns) == "task tag method setting decoder seed lam latent_path latent_sha256".split()
    assert fits.groupby("task").size().to_dict() == {"pancreas": 36 + 6 + 2 + 2 + 4, "sim1": 36 + 6 + 2 + 1 + 4}     # sim1: one scANVI fit failed
    pen = fits[fits.method.isin(P.ADV_ARMS)]
    assert (pen.setting == "matched").all() and (pen.groupby(["task", "method", "decoder"]).size() == 3).all()
    assert all(r.lam == want[(r.task, 0 if r.decoder == "unconditioned" else 1, r.method)] for r in pen.itertuples())
    plain = fits[fits.method == "scvi"]
    assert (plain.setting == "plain").all() and (plain.lam == 0.0).all() and sorted(plain.seed) == sorted([100, 101, 102] * 4)
    one = fits[(fits.task == "pancreas") & fits.method.isin(["scvi_adv", "scanvi", "sysvi", "harmony", "scanorama", "pca"])]
    assert dict(zip(one.method, one.setting)) == dict(scvi_adv="lam=3,dim=10", scanvi="lam=0,dim=10", sysvi="lam=5,dim=10",
                                                      harmony="lam=1,dim=10", scanorama="lam=1,dim=10", pca="lam=nan,dim=10")
    assert set(one.decoder[one.method.isin(B.NO_DECODER)]) == {"none"} and set(one.decoder[~one.method.isin(B.NO_DECODER)]) == {"conditioned"}
    assert all(P.sha256(r.latent_path) == r.latent_sha256 and r.latent_path.endswith(r.tag + ".npz") for r in fits.itertuples())
    with pytest.raises(P.PreregError, match="no latent file"):
        B.select_fits(L["out"], L["matched"], dict(L["latents"], paths={}))
    with pytest.raises(P.PreregError, match="seurat"):
        B.select_fits(L["out"].assign(arm=L["out"].arm.replace("harmony", "seurat")), L["matched"], L["latents"])


# ---------------------------------------------------------------------------------------------------------------------------------
# one task
# ---------------------------------------------------------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def pancreas(world):
    f = world["fits"]
    sub = f[(f.task == "pancreas") & f.method.isin(["scvi", "mmd", "reference", "scanvi", "harmony", "pca"])]
    return dict(fits=sub, res=B.analyse_task("pancreas", sub, os.path.join(world["prepped"], "pancreas__scib.h5ad"), True))


def test_task_readouts_match_brute_force(world, pancreas):  # test-reason: every readout, reference and aggregate against brute force on the saved embeddings
    fits, res, X = pancreas["fits"], pancreas["res"], world["X"][:, :N_HVG]
    ft, pf, cf = res["fit_types"], res["fits"], res["confusion"]
    assert list(ft.columns) == FIT_TYPES_COLS and list(pf.columns) == FITS_COLS and list(cf.columns) == CONFUSION_COLS
    base = rank_genes(X)
    ref = {r.tag: brute(r.latent_path, X, base) for r in fits.itertuples()}
    assert len(ft) == len(fits) * 5 and list(pf.tag) == list(fits.tag)
    for r in ft.itertuples():
        b = ref[r.tag]
        assert r.n_cells == (TYPES == r.celltype).sum() and r.rare == (r.celltype in "RD")
        assert r.purity == pytest.approx(b["purity"][r.celltype], abs=1e-12)
        assert (np.isnan(r.lt_recall) if r.celltype == "D" else r.lt_recall == pytest.approx(b["recall"][r.celltype], abs=1e-12))
        assert (np.isnan(r.marker_jaccard) if r.celltype in "RD" else r.marker_jaccard == pytest.approx(b["jaccard"][r.celltype], abs=1e-12))
    scvi = {(r.decoder, r.seed): r.tag for r in fits[fits.method == "scvi"].itertuples()}
    cond = [t for (d, _), t in scvi.items() if d == "conditioned"]
    for r in ft.itertuples():
        refs = [ref[scvi[(r.decoder, r.seed)]]] if r.method in B.SEEDED else [ref[t] for t in cond]
        for name, key in (("purity", "purity"), ("lt_recall", "recall"), ("marker_jaccard", "jaccard")):
            want = np.mean([x[key].get(r.celltype, np.nan) for x in refs])
            have, diff = getattr(r, f"ref_{name}"), getattr(r, f"{name}_diff")
            assert have == pytest.approx(want, abs=1e-12, nan_ok=True)
            assert diff == pytest.approx(getattr(r, name) - want, abs=1e-12, nan_ok=True)
    assert (ft[ft.method == "scvi"].filter(like="_diff").fillna(0) == 0).all().all()
    assert ft[(ft.method == "mmd") & (ft.celltype == "A")].purity_diff.max() < -0.2        # merged with B
    for r in pf.itertuples():
        f = ft[ft.tag == r.tag]
        x = f[f.rare]
        assert (r.n_types, r.n_rare) == (5, 2) and r.purity == pytest.approx(f.purity.mean()) and r.purity_diff == pytest.approx(f.purity_diff.mean())
        assert r.rare_purity == pytest.approx(x.purity.mean()) and r.rare_lt_recall == 0.0 and r.rare_lt_recall_diff == pytest.approx(0.0)
        assert r.marker_jaccard == pytest.approx(f.marker_jaccard.mean())
        with np.load(fits.latent_path[fits.tag == r.tag].iloc[0]) as d:
            assert r.lt_f1 == pytest.approx(A.label_transfer_f1(d["z"], TYPES, d["batch"])["mean"], abs=1e-12)
        assert r.lam == fits.lam[fits.tag == r.tag].iloc[0] or np.isnan(r.lam)
    groups = fits.groupby(["method", "setting", "decoder"]).groups
    assert len(cf) == len(groups) * 16 and cf.groupby(["method", "setting", "decoder"]).n_fits.first().to_dict() == {k: len(v) for k, v in groups.items()}
    for (method, setting, decoder), rows in groups.items():
        g = cf[(cf.method == method) & (cf.setting == setting) & (cf.decoder == decoder)]
        tables = np.stack([ref[fits.tag[i]]["conf"] for i in rows])
        assert list(g.true_type) == [a for a in "ABCR" for _ in "ABCR"] and list(g.pred_type) == list("ABCR") * 4
        assert np.allclose(g.share_mean, tables.mean(axis=0).ravel(), atol=1e-12)
        assert np.allclose(g.share_sd, tables.std(axis=0, ddof=1).ravel(), atol=1e-12) if len(rows) > 1 else g.share_sd.isna().all()


def test_parallel_equals_serial(world, pancreas):  # test-reason: the process pool (one expression matrix per worker) gives the serial tables, row order included
    par = B.analyse_task("pancreas", pancreas["fits"], os.path.join(world["prepped"], "pancreas__scib.h5ad"), True, jobs=2)
    for name, df in pancreas["res"].items():
        pd.testing.assert_frame_equal(par[name], df)


def test_task_without_markers(world):  # test-reason: marker columns stay empty when markers are off
    f = world["fits"]
    sub = f[(f.task == "sim1") & f.method.isin(["scvi", "harmony"])]
    res = B.analyse_task("sim1", sub, os.path.join(world["prepped"], "sim1__scib.h5ad"), False)
    assert res["fit_types"].filter(like="marker").isna().all().all() and res["fits"].filter(like="marker").isna().all().all()
    assert res["fit_types"].purity.notna().all()


@pytest.mark.parametrize("case, match", [("obs_order", "obs_names"), ("celltype", "celltype"), ("sha", "latent_sha256"),
                                         ("no_seed_match", "no plain-scVI fit with the same decoder and seed"),
                                         ("no_conditioned", "no conditioned-decoder")])
def test_task_stops(world, tmp_path, case, match):  # test-reason: each stop of the task analysis fires
    f = world["fits"]
    fits = f[(f.task == "sim1") & f.method.isin(["scvi", "mmd", "harmony"])].copy()
    path = os.path.join(world["prepped"], "sim1__scib.h5ad")
    if case in ("obs_order", "celltype"):
        path = str(tmp_path / "bad.h5ad")
        order = np.arange(len(OBS))[::-1] if case == "obs_order" else np.arange(len(OBS))
        types = TYPES.copy()
        types[0] = "B" if case == "celltype" else types[0]
        write_prepped(path, world["X"], False, "sim", OBS[order], types[order])
    if case == "sha":
        fits.loc[fits.index[0], "latent_sha256"] = "0" * 64
    if case == "no_seed_match":
        fits = fits.drop(fits.index[(fits.method == "scvi") & (fits.decoder == "conditioned") & (fits.seed == 101)])
    if case == "no_conditioned":
        fits = fits[(fits.method != "scvi") | (fits.decoder == "unconditioned")][lambda d: d.method != "mmd"]
    with pytest.raises(P.PreregError, match=match):
        B.analyse_task("sim1", fits, path, False)


# ---------------------------------------------------------------------------------------------------------------------------------
# summaries
# ---------------------------------------------------------------------------------------------------------------------------------
def test_summaries():  # test-reason: mean, SD (NaN for one fit), NaN handling and the real/simulated grouping by hand
    keys = dict(method="m", setting="s", decoder="d", lam=2.0)
    rows = [dict(task="pancreas", purity=0.5, marker_jaccard=np.nan, **keys), dict(task="pancreas", purity=0.7, marker_jaccard=np.nan, **keys),
            dict(task="lung", purity=0.3, marker_jaccard=np.nan, **keys), dict(task="sim1", purity=0.9, marker_jaccard=np.nan, **keys)]
    fits = pd.DataFrame(rows).reindex(columns=["task", "method", "setting", "decoder", "lam", *B.FIT_VALUES])
    s = B.summarise(fits).set_index("task")
    assert list(B.summarise(fits).columns) == ["task", "method", "setting", "decoder", "n_fits", "lam",
                                               *(f"{v}_{k}" for v in B.FIT_VALUES for k in ("mean", "sd"))]
    assert s.n_fits.to_dict() == dict(lung=1, pancreas=2, sim1=1) and s.lam.eq(2.0).all()
    assert s.purity_mean.to_dict() == pytest.approx(dict(lung=0.3, pancreas=0.6, sim1=0.9))
    assert s.purity_sd["pancreas"] == pytest.approx(np.std([0.5, 0.7], ddof=1)) and np.isnan(s.purity_sd["lung"]) and s.marker_jaccard_mean.isna().all()
    by = B.summarise_by_method(B.summarise(fits), P.REAL_TASKS).set_index("task_group")
    assert list(by.columns) == ["method", "setting", "decoder", *(c for v in B.FIT_VALUES for c in (v, f"{v}_n"))]
    assert by.purity.to_dict() == pytest.approx(dict(real=0.45, simulated=0.9)) and by.purity_n.to_dict() == dict(real=2, simulated=1)
    assert by.marker_jaccard.isna().all() and (by.marker_jaccard_n == 0).all()


# ---------------------------------------------------------------------------------------------------------------------------------
# the command line
# ---------------------------------------------------------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def cli(world, tmp_path_factory):
    out = str(tmp_path_factory.mktemp("cli") / "out")
    argv = ["--inputs", world["inputs"], "--prepped-dir", world["prepped"], "--out-dir", out, "--jobs", "1"]
    assert B.main(argv) == 0
    return dict(out=out, argv=argv)


def test_command_line_outputs(world, cli, tmp_path):  # test-reason: files, record, per-task counts and manifest written by main
    out, tables = cli["out"], os.path.join(cli["out"], "tables")
    assert sorted(os.listdir(tables)) == sorted([f"{t}_{n}.csv" for t in TASKS for n in ("fit_types", "fits", "confusion")]
                                                + ["summary.csv", "summary_by_method.csv"])
    rec = json.load(open(os.path.join(out, "biology_check.json")))
    git = subprocess.run(["git", "-C", os.path.join(ROOT, "scripts"), "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    assert rec["git_commit"] == git and rec["argv"] == cli["argv"] and rec["tasks"] == TASKS
    assert set(rec["versions"]) == {"python", "numpy", "pandas", "scipy", "sklearn", "scanpy", "anndata"}
    assert rec["constants"] == dict(K=15, RARE_FRAC=0.01, N_MARKERS=20, MIN_MARKER_CELLS=10, MARKER_TASKS=list(B.MARKER_TASKS))
    assert rec["inputs"] == {os.path.join(world["inputs"], f): P.sha256(os.path.join(world["inputs"], f)) for f in B.INPUT_FILES}
    assert rec["prepped"] == {os.path.join(world["prepped"], f"{t}__scib.h5ad"): P.sha256(os.path.join(world["prepped"], f"{t}__scib.h5ad")) for t in TASKS}
    assert rec["counts"]["pancreas"] == dict(scvi=6, scvi_adv=2, scanvi=2, sysvi=1, harmony=1, scanorama=1, pca=1, **dict.fromkeys(P.ADV_ARMS, 6))
    assert rec["counts"]["sim1"]["scanvi"] == 1
    out_frame, want = world["L"]["out"].set_index("tag"), expected_matched(world["M"])
    assert len(rec["matched"]) == 24 and [m["lam"] for m in rec["matched"]] == [want[(m["task"], int(m["decoder"] == "conditioned"), m["method"])] for m in rec["matched"]]
    for m in rec["matched"]:
        tags = world["fits"].tag[(world["fits"].task == m["task"]) & (world["fits"].method == m["method"]) & (world["fits"].decoder == m["decoder"])]
        assert m["batch_score"] == pytest.approx(out_frame.B[tags].mean()) and m["distance"] == pytest.approx(abs(m["batch_score"] - m["bstar"]))
        assert m["low_signal"] is (m["task"] == "sim1" and m["decoder"] == "conditioned")
    written = [os.path.join(d, n) for d, _, ns in os.walk(out) for n in ns if n != "MANIFEST.sha256"]
    lines = [ln.split("  ") for ln in open(os.path.join(out, "MANIFEST.sha256")).read().splitlines()]
    assert sorted(p for _, p in lines) == sorted(os.path.relpath(w, out) for w in written)
    assert all(P.sha256(os.path.join(out, p)) == h for h, p in lines)
    fits = pd.read_csv(os.path.join(tables, "pancreas_fits.csv"))
    assert len(fits) == 50 and fits.marker_jaccard.notna().all() and pd.read_csv(os.path.join(tables, "sim1_fits.csv")).marker_jaccard.isna().all()
    by = pd.read_csv(os.path.join(tables, "summary_by_method.csv"))
    assert sorted(set(by.task_group)) == ["real", "simulated"] and set(by.task_group[by.method == "mmd"]) == {"real", "simulated"}
    assert len(pd.read_csv(os.path.join(tables, "summary.csv"))) == 2 * 20
    sub = str(tmp_path / "sub")      # --tasks: the record lists only the chosen task's matched strengths
    assert B.main(["--inputs", world["inputs"], "--prepped-dir", world["prepped"], "--out-dir", sub, "--jobs", "1", "--tasks", "sim1"]) == 0
    assert {m["task"] for m in json.load(open(os.path.join(sub, "biology_check.json")))["matched"]} == {"sim1"}


@pytest.mark.parametrize("case, match", [("exists", "exists"), ("task", "nope"), ("counts", "penalty fits"), ("modality", "rna")])
def test_command_line_stops(world, tmp_path, monkeypatch, case, match):  # test-reason: the stops of main: existing out-dir, unknown task, wrong fit count, non-RNA marker task
    monkeypatch.setattr(B, "load_sweep", lambda _: world["L"])
    prepped = world["prepped"]
    argv = ["--inputs", world["inputs"], "--prepped-dir", prepped, "--out-dir", str(tmp_path / "out"), "--jobs", "1"]
    if case == "exists":
        (tmp_path / "out").mkdir()
    if case == "task":
        argv += ["--tasks", "nope"]
    if case == "counts":
        monkeypatch.setattr(B, "select_fits", lambda *a: world["fits"].iloc[1:])
    if case == "modality":
        write_prepped(str(tmp_path / "pancreas__scib.h5ad"), world["X"], False, "atac")
        argv[3] = str(tmp_path)
    with pytest.raises(P.PreregError, match=match):
        B.main(argv)
    assert not (tmp_path / "out" / "tables").exists()
