#!/usr/bin/env python
"""Direct biology check on the saved sweep embeddings (reviewer comment R2.6; the rules are docs/BIOLOGY_CHECK.md).
Per fit and cell type: neighbourhood purity, label-transfer recall and confusion, marker-gene preservation after smoothing over the
15 nearest neighbours; each as a difference from plain scVI. Exploratory and descriptive: no new fits, no test.

Usage: python scripts/biology_check.py --inputs <dir written by sweep_inputs.py> --prepped-dir <dir of <task>__scib.h5ad>
           --out-dir <new directory> [--jobs 6] [--tasks T ...] [--strengths CSV]
--strengths: CSV with columns task, cond, arm, lam (one row per penalty, task and decoder of the inputs); the penalty fits use these
strengths instead of the matched ones. The record keeps the file's path and SHA-256.
"""
import argparse
import importlib.metadata
import json
import multiprocessing
import os
import platform
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse
from sklearn.metrics import f1_score
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from threadpoolctl import threadpool_limits

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import analysis_sweep as S  # noqa: E402
import analysis_x1 as A  # noqa: E402
import prereg_rules as P  # noqa: E402

K = 15
RARE_FRAC = 0.01
N_MARKERS = 20
MIN_MARKER_CELLS = 10
MARKER_TASKS = ("pancreas", "lung", "immune", "immune_hum_mou")
DECODER = {0: "unconditioned", 1: "conditioned"}
NO_DECODER = ("harmony", "scanorama", "pca")
BASELINES = ("scvi_adv", "scanvi", "sysvi", *NO_DECODER)
SEEDED = (*P.ADV_ARMS, "scvi")          # the methods whose reference is the plain-scVI fit of the same decoder and seed
READOUTS = ("purity", "lt_recall", "marker_jaccard")
FIT_COLS = ["task", "tag", "method", "setting", "decoder", "seed", "lam", "latent_path", "latent_sha256"]
TYPE_COLS = ["task", "tag", "method", "setting", "decoder", "seed", "celltype", "n_cells", "rare",
             *(c for r in READOUTS for c in (r, f"ref_{r}", f"{r}_diff"))]
FIT_VALUES = ["purity", "purity_diff", "rare_purity", "rare_purity_diff", "rare_lt_recall", "rare_lt_recall_diff", "lt_f1",
              "marker_jaccard", "marker_jaccard_diff"]
GROUP = ["task", "method", "setting", "decoder"]
INPUT_FILES = ("inputs.json", "manifest.tsv", "scores_list.txt", "latents.tsv", "failures.csv", "baseline_coverage.csv")


def knn_indices(Z, k=K):
    """The k nearest OTHER cells of each cell, nearest first (sklearn leaves the query cell out, also among identical points)."""
    return NearestNeighbors(n_neighbors=k).fit(Z).kneighbors(return_distance=False)


def purity_per_type(labels, idx):
    """Per type: the mean over its cells of the share of the cell's neighbours (rows of idx) with the cell's own label."""
    codes, types = pd.factorize(np.asarray(labels), sort=True)
    same = (codes[idx] == codes[:, None]).mean(axis=1)
    return {str(t): float(m) for t, m in zip(types, np.bincount(codes, weights=same) / np.bincount(codes))}


def rare_types(labels, frac=RARE_FRAC):
    types, counts = np.unique(np.asarray(labels), return_counts=True)
    return [str(t) for t, c in zip(types, counts) if c < frac * len(labels)]


def smooth(X, idx):
    """Row i = mean of X over the rows idx[i]: a row-normalised neighbour matrix times X."""
    n, k = idx.shape
    W = sparse.csr_matrix((np.full(n * k, 1.0 / k), idx.ravel(), np.arange(0, n * k + 1, k)), shape=(n, n))
    out = W @ X
    return out.toarray() if sparse.issparse(out) else np.asarray(out)


def top_markers(X, labels, genes, n=N_MARKERS, min_cells=MIN_MARKER_CELLS):
    """Top n genes (scanpy's Wilcoxon score order) of every type with at least min_cells cells, each type against all other cells."""
    labels = np.asarray(labels)
    types, counts = np.unique(labels, return_counts=True)
    P._require((counts >= 2).all(), f"cell types with fewer than 2 cells cannot be ranked: {[str(t) for t in types[counts < 2]]}")
    adata = ad.AnnData(X, obs=pd.DataFrame({"type": pd.Categorical(labels)}, index=np.arange(len(labels)).astype(str)),
                       var=pd.DataFrame(index=list(genes)))
    sc.tl.rank_genes_groups(adata, "type", method="wilcoxon", use_raw=False, reference="rest", n_genes=n)
    names = adata.uns["rank_genes_groups"]["names"]
    return {str(t): [str(g) for g in names[str(t)]] for t, c in zip(types, counts) if c >= min_cells}


def jaccard(a, b):
    a, b = set(a), set(b)
    return len(a & b) / len(a | b)


def transfer_predictions(Z, celltype, batch, k=K):
    """The rule of analysis_x1.label_transfer_f1, keeping the predictions: (kept-cell mask, predicted label of each kept cell,
    dropped types). Types present in one batch only are dropped; each batch is predicted from the others."""
    Z, celltype, batch = np.asarray(Z), np.asarray(celltype), np.asarray(batch)
    P._require(len(Z) == len(celltype) == len(batch), "embedding and labels must list the same cells")
    dropped = sorted(t for t in set(celltype) if len(set(batch[celltype == t])) == 1)
    keep = ~np.isin(celltype, dropped)
    Z, celltype, batch = Z[keep], celltype[keep], batch[keep]
    pred = np.empty(len(celltype), dtype=celltype.dtype)
    for b in sorted(set(batch)):
        test = batch == b
        pred[test] = KNeighborsClassifier(n_neighbors=k).fit(Z[~test], celltype[~test]).predict(Z[test])
    return keep, pred, [str(t) for t in dropped]


def per_type_recall(true, pred):
    true, pred = np.asarray(true), np.asarray(pred)
    return {str(t): float((pred[true == t] == t).mean()) for t in sorted(set(true))}


def confusion_table(true, pred):
    """Rows = true type, columns = predicted type (the same sorted types), each row summing to 1."""
    true, pred = np.asarray(true), np.asarray(pred)
    types = np.unique(true)
    P._require(np.isin(pred, types).all(), f"predicted types {sorted(set(pred) - set(types))} do not occur among the true types")
    m = len(types)
    counts = np.bincount(np.searchsorted(types, true) * m + np.searchsorted(types, pred), minlength=m * m).reshape(m, m)
    names = [str(t) for t in types]
    return pd.DataFrame(counts / counts.sum(axis=1, keepdims=True), index=names, columns=names)


def _obs_info(a):
    return dict(obs_names=a.obs_names.to_numpy(dtype=str), celltype=a.obs[a.uns["celltype_key"]].astype(str).to_numpy(),
                modality=str(a.uns["modality"]))


def _read_obs(prepped_path):
    """Cell names, cell types and modality of a prepped file, without reading its expression matrix."""
    a = ad.read_h5ad(prepped_path, backed="r")
    info = _obs_info(a)
    a.file.close()
    return info


def load_expression(prepped_path):
    """X (log-normalised as distributed, never the counts layer) of the highly variable genes, as dense float32."""
    a = ad.read_h5ad(prepped_path)
    hv = a.var["highly_variable"].to_numpy(dtype=bool)
    X = a.X[:, hv]
    return dict(X=np.asarray(X.toarray() if sparse.issparse(X) else X, dtype=np.float32), genes=a.var_names[hv].tolist(),
                **_obs_info(a))


def load_sweep(inputs_dir):
    """The sweep-analysis inputs read as analysis_sweep.main reads them, up to the matched strengths."""
    i = inputs_dir
    with open(os.path.join(i, "inputs.json")) as f:
        info = json.load(f)
    M, _ = P.read_manifest(os.path.join(i, "manifest.tsv"))
    P.check_seed_separation(M)
    with open(os.path.join(i, "scores_list.txt")) as f:
        scores, _ = P.read_scores([ln.strip() for ln in f if ln.strip()])
    F = P.read_failures(os.path.join(i, "failures.csv"))
    out = A.add_scores(P.outcomes(M, scores, F, [S.EXP, "X1", "X13"]).sort_values("tag", ignore_index=True))
    sweep = out[out.experiment == S.EXP]
    tasks = sorted(set(sweep.task))
    P._require(tasks == sorted(info["tasks"]), f"manifest tasks {tasks} differ from inputs.json {sorted(info['tasks'])}")
    seeds = sorted(set(sweep.seed_i))
    P._require(seeds == [100, 101, 102], f"sweep seeds {seeds} are not 100-102")
    P._require(set(out.experiment[out.arm == "scvi_adv"]) <= {"X1"} and set(out.experiment) <= {S.EXP, "X1", "X13"},
               "unexpected experiment labels (sweep rows are A1, baselines X13 or scvi_adv rows labelled X1)")
    P._require("latent_sha256" in scores.columns, "scores lack latent_sha256: the embeddings cannot be tied to the scores")
    with open(os.path.join(i, "latents.tsv")) as f:
        paths = dict(ln.rstrip("\n").split("\t") for ln in f if ln.strip())
    latents = dict(paths=paths, sha=dict(zip(scores.tag, scores.latent_sha256)))
    targets, matched = S.sweep_targets(out, tasks, seeds)
    return dict(out=out, targets=targets, matched=matched, latents=latents, info=info,
                coverage=pd.read_csv(os.path.join(i, "baseline_coverage.csv")))


def read_strengths(path, matched):
    """--strengths: {(task, cond, arm): strength} from a CSV with columns task, cond, arm, lam; it must list exactly the keys of `matched`."""
    s = pd.read_csv(path, float_precision="round_trip")      # the default parser can change the last digit, and the fits are picked by exact equality
    P._require({"task", "cond", "arm", "lam"} <= set(s.columns) and s.lam.notna().all(), f"{path}: strengths need columns task, cond, arm, lam and a strength in every row")
    got = {(t, int(c), a): float(lam) for t, c, a, lam in zip(s.task, s.cond, s.arm, s.lam)}
    P._require(len(got) == len(s) and got.keys() == matched.keys(),
               f"{path}: strengths must list each penalty, task and decoder of the inputs exactly once ({len(s)} rows, {len(matched)} expected)")
    return got


def select_fits(out, matched, latents):
    """The fits of the check (docs section 1): penalties at the matched strength (or the --strengths one), plain scVI, every scored baseline row."""
    ok = out[out.status == "ok"]
    a1 = ok[ok.experiment == S.EXP]
    pen = a1[a1.arm.isin(P.ADV_ARMS)]
    pen = pen[pen.lamf.to_numpy() == [matched[(t, c, a)] for t, c, a in zip(pen.task, pen.cond_i, pen.arm)]]
    pen = pen.assign(method=pen.arm, setting="matched", decoder=pen.cond_i.map(DECODER), lam=pen.lamf)
    plain = a1[a1.arm == "none"]
    plain = plain.assign(method="scvi", setting="plain", decoder=plain.cond_i.map(DECODER), lam=0.0)
    base = ok[((ok.experiment == "X1") & (ok.arm == "scvi_adv")) | (ok.experiment == "X13")]
    P._require(set(base.arm) <= set(BASELINES), f"unknown baseline methods {sorted(set(base.arm) - set(BASELINES))}")
    base = base.assign(method=base.arm, setting=[f"lam={lam:g},dim={n}" for lam, n in zip(base.lamf, base.n_latent)],
                       decoder=base.cond_i.map(DECODER).where(~base.arm.isin(NO_DECODER), "none"), lam=base.lamf)
    fits = pd.concat([x.assign(seed=x.seed_i.astype(int)) for x in (pen, plain, base)], ignore_index=True)
    missing = sorted(set(fits.tag) - set(latents["paths"]) | set(fits.tag) - set(latents["sha"]))
    P._require(not missing, f"no latent file or SHA-256 for tags {missing[:5]}")
    return fits.assign(latent_path=fits.tag.map(latents["paths"]), latent_sha256=fits.tag.map(latents["sha"]))[FIT_COLS]


def _fit(row, ctx):
    """All readouts of one fit (ctx: obs_names, celltype, base markers and expression of the task)."""
    tag, where = row["tag"], f"{row['task']} {row['tag']}"
    path, z, latent_types, batch = A.read_latent(tag, dict(paths={tag: row["latent_path"]}, sha={tag: row["latent_sha256"]}))
    with np.load(path) as d:
        obs_names = d["obs_names"]
    P._require(np.array_equal(obs_names, ctx["obs_names"]), f"{where}: latent obs_names differ from the prepped file's, or are in another order")
    celltype = ctx["celltype"]
    P._require(np.array_equal(latent_types, celltype), f"{where}: latent celltype differs from the prepped file's")
    idx = knn_indices(z)
    keep, pred, dropped = transfer_predictions(z, celltype, batch)
    true, held_out = celltype[keep], np.asarray(batch)[keep]
    f1 = {b: float(f1_score(true[held_out == b], pred[held_out == b], labels=sorted(set(true[held_out == b])), average="macro",
                            zero_division=0)) for b in sorted(set(held_out))}
    expected = A.label_transfer_f1(z, celltype, batch)["per_batch"]
    P._require(f1.keys() == expected.keys() and all(abs(f1[b] - expected[b]) <= 1e-12 for b in f1),
               f"{where}: per-batch macro F1 of the saved predictions differs from analysis_x1.label_transfer_f1")
    jac = {}
    if ctx["expr"] is not None:
        found = top_markers(smooth(ctx["expr"]["X"], idx), celltype, ctx["expr"]["genes"])
        jac = {t: jaccard(ctx["base"][t], found[t]) for t in ctx["base"]}
    return dict(purity=purity_per_type(celltype, idx), recall=per_type_recall(true, pred), confusion=confusion_table(true, pred),
                jaccard=jac, dropped=dropped, lt_f1=float(np.mean(list(f1.values()))) if f1 else np.nan)


_WORKER = {}


def _worker_init(head, prepped_path):
    """Once per worker: the task's expression matrix (not once per fit)."""
    _WORKER.update(head, expr=load_expression(prepped_path) if prepped_path else None)


def _worker_fit(row):
    with threadpool_limits(1):        # `jobs` workers use `jobs` CPUs
        return _fit(row, _WORKER)


def _with_reference(ft, task):
    """ref_r and r_diff (docs section 4): the plain-scVI fit of the same decoder and seed for penalties and plain scVI itself,
    the per-type mean over the conditioned-decoder plain-scVI fits for every other method."""
    plain = ft[ft.method == "scvi"]
    seeded = ft.method.isin(SEEDED).to_numpy()
    need = set(zip(ft.decoder[seeded], ft.seed[seeded]))
    P._require(need <= set(zip(plain.decoder, plain.seed)),
               f"{task}: no plain-scVI fit with the same decoder and seed for {sorted(need - set(zip(plain.decoder, plain.seed)))}")
    conditioned = plain[plain.decoder == "conditioned"]
    P._require(seeded.all() or len(conditioned), f"{task}: no conditioned-decoder plain-scVI fit to compare the baselines with")
    cols = list(READOUTS)
    same = plain.set_index(["decoder", "seed", "celltype"])[cols].reindex(pd.MultiIndex.from_frame(ft[["decoder", "seed", "celltype"]]))
    mean = conditioned.groupby("celltype")[cols].mean().reindex(ft.celltype.to_numpy())
    ref = np.where(seeded[:, None], same.to_numpy(), mean.to_numpy())
    for j, r in enumerate(READOUTS):
        ft = ft.assign(**{f"ref_{r}": ref[:, j]})
        ft[f"{r}_diff"] = ft[r] - ft[f"ref_{r}"]
    return ft[TYPE_COLS]


def _check(ft, fits, dropped, markers, task):
    P._require(len(ft) == len(fits) * ft.celltype.nunique() and set(ft.tag) == set(fits.tag), f"{task}: a fit or a cell type is missing")
    defined = dict(purity=np.ones(len(ft), bool), lt_recall=~ft.celltype.isin(dropped).to_numpy(),
                   marker_jaccard=((ft.n_cells >= MIN_MARKER_CELLS) & markers).to_numpy())
    for r, d in defined.items():
        for col in (r, f"ref_{r}", f"{r}_diff"):
            P._require((ft[col].isna().to_numpy() == ~d).all(), f"{task}: {col} is missing where it is defined or present where it is not")
        for col in (r, f"ref_{r}"):
            v = ft[col].to_numpy(float)[d]
            P._require(((v >= -1e-12) & (v <= 1 + 1e-12)).all(), f"{task}: {col} lies outside [0, 1]")


def analyse_task(task, fits, prepped_path, markers, jobs=1):
    """Readouts of every fit of one task: dict(fit_types, fits, confusion)."""
    P._require(jobs >= 1, "jobs must be at least 1")
    fits = fits.reset_index(drop=True)
    P._require(len(fits) > 0 and (fits.task == task).all(), f"{task}: needs fits, all of this task")
    expr = load_expression(prepped_path) if markers else None
    obs = _read_obs(prepped_path) if expr is None else expr
    head = dict(obs_names=obs["obs_names"], celltype=obs["celltype"],
                base=top_markers(expr["X"], expr["celltype"], expr["genes"]) if markers else None)
    records = fits.to_dict("records")
    if jobs == 1:
        results = [_fit(r, dict(head, expr=expr)) for r in records]
    else:
        del expr, obs            # the workers load their own copy; the parent does not keep one
        with ProcessPoolExecutor(max_workers=jobs, mp_context=multiprocessing.get_context("spawn"), initializer=_worker_init,
                                 initargs=(head, prepped_path if markers else None)) as pool:
            results = list(pool.map(_worker_fit, records, chunksize=1))
    P._require(len({tuple(r["dropped"]) for r in results}) == 1, f"{task}: the fits' embeddings carry different batch labels")
    dropped = results[0]["dropped"]
    types, counts = np.unique(head["celltype"], return_counts=True)
    rare = set(rare_types(head["celltype"]))
    ft = pd.DataFrame([dict(task=task, tag=r["tag"], method=r["method"], setting=r["setting"], decoder=r["decoder"], seed=r["seed"],
                            celltype=str(t), n_cells=int(c), rare=str(t) in rare, purity=res["purity"][str(t)],
                            lt_recall=res["recall"].get(str(t), np.nan), marker_jaccard=res["jaccard"].get(str(t), np.nan))
                       for r, res in zip(records, results) for t, c in zip(types, counts)])
    ft = _with_reference(ft, task)
    _check(ft, fits, dropped, markers, task)
    by_tag = {tag: g for tag, g in ft.groupby("tag")}
    per_fit = []
    for r, res in zip(records, results):
        f = by_tag[r["tag"]]
        x = f[f.rare]
        per_fit.append(dict(task=task, tag=r["tag"], method=r["method"], setting=r["setting"], decoder=r["decoder"], seed=r["seed"],
                            n_types=len(f), n_rare=len(x), purity=f.purity.mean(), purity_diff=f.purity_diff.mean(),
                            rare_purity=x.purity.mean(), rare_purity_diff=x.purity_diff.mean(), rare_lt_recall=x.lt_recall.mean(),
                            rare_lt_recall_diff=x.lt_recall_diff.mean(), lt_f1=res["lt_f1"], marker_jaccard=f.marker_jaccard.mean(),
                            marker_jaccard_diff=f.marker_jaccard_diff.mean(), lam=r["lam"]))
    confusion = []
    for (method, setting, decoder), g in fits.groupby(["method", "setting", "decoder"]):
        tables = [results[i]["confusion"] for i in g.index]
        P._require(all(t.index.equals(tables[0].index) for t in tables), f"{task} {method} {setting} {decoder}: confusion tables differ in types")
        stack = np.stack([t.to_numpy() for t in tables])
        names = list(tables[0].index)
        true, pred = zip(*[(a, b) for a in names for b in names]) if names else ((), ())
        confusion.append(pd.DataFrame(dict(task=task, method=method, setting=setting, decoder=decoder, true_type=list(true),
                                           pred_type=list(pred), n_fits=len(g), share_mean=stack.mean(axis=0).ravel(),
                                           share_sd=(stack.std(axis=0, ddof=1) if len(g) > 1 else np.full(stack.shape[1:], np.nan)).ravel())))
    return dict(fit_types=ft, fits=pd.DataFrame(per_fit), confusion=pd.concat(confusion, ignore_index=True))


def summarise(fits):
    """Per task, method, setting and decoder: the number of fits and the mean and SD (over fits) of every value."""
    rows = []
    for key, g in fits.groupby(GROUP):
        row = dict(zip(GROUP, key), n_fits=len(g), lam=g.lam.iloc[0])
        for v in FIT_VALUES:
            row[f"{v}_mean"], row[f"{v}_sd"] = g[v].mean(), g[v].std()
        rows.append(row)
    return pd.DataFrame(rows)


def summarise_by_method(summary, real_tasks):
    """Per task group (real or simulated), method, setting and decoder: the mean over tasks of the per-task means."""
    summary = summary.assign(task_group=np.where(summary.task.isin(real_tasks), "real", "simulated"))
    rows = []
    for key, g in summary.groupby(["task_group", "method", "setting", "decoder"]):
        row = dict(zip(["task_group", "method", "setting", "decoder"], key))
        for v in FIT_VALUES:
            row[v], row[f"{v}_n"] = g[f"{v}_mean"].mean(), int(g[f"{v}_mean"].notna().sum())
        rows.append(row)
    return pd.DataFrame(rows)


def _check_counts(task, fits, coverage):
    n = fits.method.value_counts()
    P._require(fits.method.isin(P.ADV_ARMS).sum() == 36, f"{task}: {fits.method.isin(P.ADV_ARMS).sum()} penalty fits, expected 36 (6 penalties x 2 decoders x 3 seeds)")
    P._require(n.get("scvi", 0) == 6, f"{task}: {n.get('scvi', 0)} plain-scVI fits, expected 6")
    scored = coverage[coverage.task == task].set_index("arm").scored
    for m in sorted(set(scored.index) | (set(fits.method) & set(BASELINES))):
        P._require(n.get(m, 0) == scored.get(m, 0), f"{task} {m}: {n.get(m, 0)} fits, expected {scored.get(m, 0)} scored rows in baseline_coverage.csv")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--inputs", required=True, help="directory written by scripts/sweep_inputs.py")
    ap.add_argument("--prepped-dir", required=True, help="directory of <task>__scib.h5ad")
    ap.add_argument("--out-dir", required=True, help="must not exist")
    ap.add_argument("--jobs", type=int, default=6)
    ap.add_argument("--tasks", nargs="+", default=None, help="default: every task of the inputs")
    ap.add_argument("--strengths", default=None, help="CSV with columns task, cond, arm, lam: the penalty strengths to use instead of the matched ones")
    a = ap.parse_args(argv)
    P._require(not os.path.exists(a.out_dir), f"output directory {a.out_dir} exists")
    L = load_sweep(a.inputs)
    tasks = sorted(set(a.tasks or L["info"]["tasks"]))
    P._require(set(tasks) <= set(L["info"]["tasks"]), f"tasks {sorted(set(tasks) - set(L['info']['tasks']))} are not in the inputs")
    strengths = read_strengths(a.strengths, L["matched"]) if a.strengths else L["matched"]
    fits = select_fits(L["out"], strengths, L["latents"])
    prepped = {t: os.path.join(a.prepped_dir, f"{t}__scib.h5ad") for t in tasks}
    for t in tasks:      # every stop that does not need a fit comes before the first one
        _check_counts(t, fits[fits.task == t], L["coverage"])
        if t in MARKER_TASKS:
            P._require(_read_obs(prepped[t])["modality"] == "rna", f"{t}: marker genes need an RNA task (modality is not rna)")
    tables = os.path.join(a.out_dir, "tables")
    os.makedirs(tables)
    per_task = []
    for t in tasks:
        start = time.time()
        res = analyse_task(t, fits[fits.task == t], prepped[t], t in MARKER_TASKS, a.jobs)
        for name, df in res.items():
            df.to_csv(os.path.join(tables, f"{t}_{name}.csv"), index=False)
        per_task.append(res["fits"])
        print(f"{t}: {len(res['fits'])} fits, {time.time() - start:.0f} s", flush=True)
    summary = summarise(pd.concat(per_task, ignore_index=True))
    summary.to_csv(os.path.join(tables, "summary.csv"), index=False)
    summarise_by_method(summary, P.REAL_TASKS).to_csv(os.path.join(tables, "summary_by_method.csv"), index=False)
    B = L["out"].set_index("tag").B
    matched = []
    for (t, cond, arm), lam in sorted(strengths.items()):
        if t in tasks:
            target = L["targets"][f"{t}|{cond}"]
            score = float(B[fits.tag[(fits.task == t) & (fits.method == arm) & (fits.decoder == DECODER[cond])]].mean())
            matched.append(dict(task=t, decoder=DECODER[cond], method=arm, lam=float(lam), batch_score=score, bstar=float(target["bstar"]),
                                distance=abs(score - target["bstar"]), low_signal=bool(target["bstar_low_signal"])))
    sha = {os.path.join(a.inputs, f): P.sha256(os.path.join(a.inputs, f)) for f in INPUT_FILES}
    git = P.git_state()
    record = dict(git_commit=git["git_sha"], git_dirty=git["git_dirty"], argv=list(sys.argv[1:] if argv is None else argv), tasks=tasks,
                  versions=dict(python=platform.python_version(), **{k: importlib.metadata.version(v) for k, v in dict(
                      numpy="numpy", pandas="pandas", scipy="scipy", sklearn="scikit-learn", scanpy="scanpy", anndata="anndata").items()}),
                  constants=dict(K=K, RARE_FRAC=RARE_FRAC, N_MARKERS=N_MARKERS, MIN_MARKER_CELLS=MIN_MARKER_CELLS,
                                 MARKER_TASKS=list(MARKER_TASKS)),
                  inputs=sha, prepped={p: P.sha256(p) for p in prepped.values()}, matched=matched,
                  strengths=dict(path=a.strengths, sha256=P.sha256(a.strengths)) if a.strengths else None,
                  counts={t: {m: int(n) for m, n in fits[fits.task == t].method.value_counts().sort_index().items()} for t in tasks})
    with open(os.path.join(a.out_dir, "biology_check.json"), "w") as f:
        json.dump(record, f, indent=1, sort_keys=True, allow_nan=False)
        f.write("\n")
    written = sorted(os.path.relpath(os.path.join(d, n), a.out_dir) for d, _, ns in os.walk(a.out_dir) for n in ns)
    with open(os.path.join(a.out_dir, "MANIFEST.sha256"), "w") as f:
        f.write("".join(f"{P.sha256(os.path.join(a.out_dir, p))}  {p}\n" for p in written))
    return 0


if __name__ == "__main__":  # pragma: no cover  # command-line entry
    sys.exit(main())
