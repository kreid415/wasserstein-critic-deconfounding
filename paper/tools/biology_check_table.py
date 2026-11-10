"""Table S10 and Res-5 summaries from the archived first-crossing biology check.

Run from the tree root: python paper/tools/biology_check_table.py.
Only summary arithmetic from the existing check; no fits or new readouts.
"""
import hashlib
import json
from pathlib import Path

import pandas as pd

DATA = Path(__file__).resolve().parents[1] / "figure_data"
REAL = ["pancreas", "lung", "immune", "atac_small", "atac_large", "immune_hum_mou"]
PEN = ["discriminator", "reference", "pooled", "barycenter", "mmd", "sinkhorn"]
BASE = ["scvi_adv", "scanvi", "sysvi", "harmony", "scanorama", "pca"]
LABEL = dict(zip(PEN + ["scvi"] + BASE, ["JS discriminator", "Reference critic", "Pooled critic", "Barycenter critic",
             "MMD", "Sinkhorn", "Plain scVI", "scvi-tools adversary", "scANVI (labels)", "sysVI", "Harmony", "Scanorama", "PCA"]))
COLS = ["purity_diff", "rare_purity_diff", "rare_lt_recall_diff", "marker_jaccard", "marker_jaccard_diff"]


def summarize(data=DATA):
    run = data / "biology_check_first_crossing"
    record = json.loads((run / "biology_check.json").read_text())
    assert record["strengths"]["sha256"] == hashlib.sha256((data / "first_crossing_strength.csv").read_bytes()).hexdigest()
    strengths = pd.read_csv(data / "first_crossing_strength.csv").set_index(["task", "cond", "arm"])
    summ = pd.read_csv(run / "tables/summary.csv")
    bym = pd.read_csv(run / "tables/summary_by_method.csv")
    for r in summ[summ.method.isin(PEN)].itertuples():
        assert r.lam == strengths.loc[(r.task, int(r.decoder == "conditioned"), r.method), "lam"]
    rows = []
    for method in PEN + ["scvi"] + BASE:
        for decoder in bym[bym.method == method].decoder.unique():
            g = bym[(bym.task_group == "real") & (bym.method == method) & (bym.decoder == decoder)]
            s = summ[summ.task.isin(REAL) & (summ.method == method) & (summ.decoder == decoder)]
            assert len(g) and len(s) and g.purity_n.min() >= 5
            rows.append(dict(method=method, decoder=decoder, settings=len(g),
                             seed_sd=None if s.purity_diff_sd.isna().all() else float(s.purity_diff_sd.mean()),
                             **{c: float(g[c].mean()) for c in COLS}))
    rare = summ[summ.task.isin(REAL) & summ.method.isin(PEN) & summ.rare_purity_diff_mean.notna()]
    ft = pd.read_csv(run / "tables/lung_fit_types.csv")
    ion = ft[(ft.celltype == "Ionocytes") & (ft.method == "discriminator") & (ft.decoder == "conditioned")]
    assert len(ion) == 3 and ion.n_cells.nunique() == 1
    return dict(rows=rows, rare_cells=len(rare), rare_negative=int((rare.rare_purity_diff_mean < 0).sum()),
                ionocytes=dict(n=int(ion.n_cells.iloc[0]), purity_diff=float(ion.purity_diff.mean()),
                               recall=float(ion.lt_recall.mean()), plain_recall=float(ion.ref_lt_recall.mean())))


def signed(x):
    return "$0.000$" if abs(x) < 0.0005 else f"${x:+.3f}$"


def table_lines(summary):
    rows = []
    for method in PEN + ["scvi"] + BASE:
        for decoder in ("conditioned", "unconditioned", "none"):
            g = [r for r in summary["rows"] if r["method"] == method and r["decoder"] == decoder]
            if not g:
                continue
            r = g[0]
            name = LABEL[method] if decoder != "unconditioned" else ""
            if method in BASE:
                name += f" ({r['settings']})"
            dec = decoder if method in PEN + ["scvi", "scvi_adv"] else "--"
            sd = "n/a" if r["seed_sd"] is None else f"{r['seed_sd']:.3f}"
            rows.append(f"{name} & {dec} & {signed(r['purity_diff'])} & {sd} & {signed(r['rare_purity_diff'])} & "
                        f"{signed(r['rare_lt_recall_diff'])} & {r['marker_jaccard']:.3f} & {signed(r['marker_jaccard_diff'])} " + r"\\")
    return rows


if __name__ == "__main__":  # pragma: no cover  # CLI; summarize and table_lines are tested
    (DATA / "res5_biology.json").write_text(json.dumps(summarize(), indent=1) + "\n")
