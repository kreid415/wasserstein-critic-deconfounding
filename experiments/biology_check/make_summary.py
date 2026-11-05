"""One-page summary of the biology check (docs/BIOLOGY_CHECK.md) from the run's tables; re-checks counts and missing values.
Usage: python experiments/biology_check/make_summary.py <run-dir> <out.md>
Every number in the page is read from <run-dir>/tables and <run-dir>/biology_check.json."""
import json, os, sys
import numpy as np, pandas as pd

run, out_md = sys.argv[1], sys.argv[2]
T = os.path.join(run, "tables")
rec = json.load(open(os.path.join(run, "biology_check.json")))
PEN = ("discriminator", "reference", "pooled", "barycenter", "mmd", "sinkhorn")
BASE = ("scvi_adv", "scanvi", "sysvi", "harmony", "scanorama", "pca")
MARK = ("pancreas", "lung", "immune", "immune_hum_mou")
REAL = (*MARK, "atac_small", "atac_large")  # the six real tasks; seed SD uses the same tasks as the purity column

# --- re-checks (stop on failure)
fits = pd.concat([pd.read_csv(os.path.join(T, f"{t}_fits.csv")) for t in rec["tasks"]], ignore_index=True)
ft = pd.concat([pd.read_csv(os.path.join(T, f"{t}_fit_types.csv")) for t in rec["tasks"]], ignore_index=True)
assert len(rec["tasks"]) == 8, rec["tasks"]
for t in rec["tasks"]:
    got = fits[fits.task == t].method.value_counts().to_dict()
    assert got == rec["counts"][t], (t, got, rec["counts"][t])
    assert sum(got.get(p, 0) for p in PEN) == 36 and got["scvi"] == 6, t
assert fits.tag.is_unique and fits.purity.notna().all() and fits.purity_diff.notna().all() and fits.lt_f1.notna().all()
assert fits[fits.task.isin(MARK)].marker_jaccard.notna().all() and fits[~fits.task.isin(MARK)].marker_jaccard.isna().all()
assert (fits.rare_purity.isna() == (fits.n_rare == 0)).all()
assert ft.purity.between(0, 1).all() and ft.purity.notna().all()
n_fits = len(fits)

summ = pd.read_csv(os.path.join(T, "summary.csv"))
bym = pd.read_csv(os.path.join(T, "summary_by_method.csv"))
real = bym[bym.task_group == "real"]
sim = bym[bym.task_group == "simulated"]


def f(x, d=3):
    return "n/a" if pd.isna(x) else f"{x:+.{d}f}"


def g(x, d=3):
    return "n/a" if pd.isna(x) else f"{x:.{d}f}"


def seed_sd(method, decoder, col):
    s = summ[(summ.method == method) & (summ.decoder == decoder) & summ.task.isin(REAL)]
    assert s.task.nunique() == len(REAL), (method, decoder, sorted(s.task))
    return s[f"{col}_sd"].mean()


rows = []
for m in (*PEN, "scvi"):
    for dec in ("unconditioned", "conditioned"):
        r = real[(real.method == m) & (real.decoder == dec)].iloc[0]
        s = sim[(sim.method == m) & (sim.decoder == dec)].iloc[0]
        rows.append((f"{m}, {dec}",
                     r.purity_diff, seed_sd(m, dec, "purity_diff"), s.purity_diff, r.rare_purity_diff, r.rare_lt_recall_diff,
                     r.marker_jaccard, r.marker_jaccard_diff, int(r.purity_diff_n), int(r.marker_jaccard_n)))
for m in BASE:
    r = real[real.method == m]
    s = sim[sim.method == m]
    if not len(r):
        continue
    rng = f" [{r.purity_diff.min():+.3f}, {r.purity_diff.max():+.3f}]" if len(r) > 1 else ""
    rows.append((f"{m}, {len(r)} setting{'s' if len(r) > 1 else ''} (mean{', range' if len(r) > 1 else ''})",
                 r.purity_diff.mean(), summ[(summ.method == m) & summ.task.isin(REAL)].purity_diff_sd.mean(), s.purity_diff.mean(), r.rare_purity_diff.mean(), r.rare_lt_recall_diff.mean(),
                 r.marker_jaccard.mean(), r.marker_jaccard_diff.mean(), int(r.purity_diff_n.max()), int(r.marker_jaccard_n.max()), rng))

# rare-type pattern over (real task, penalty, decoder) cells
rp = summ[summ.method.isin(PEN) & summ.task.isin(["pancreas", "lung", "immune", "immune_hum_mou", "atac_small", "atac_large"])
          & summ.rare_purity_diff_mean.notna()]
n_rare_cells, n_neg = len(rp), int((rp.rare_purity_diff_mean < 0).sum())
worst = rp.loc[rp.rare_purity_diff_mean.idxmin()]
rare_tasks = sorted(set(rp.task))
n_rare_types = {t: int(fits[fits.task == t].n_rare.iloc[0]) for t in rec["tasks"]}

L = []
L.append("# Direct biology check on the saved sweep embeddings (reviewer comment R2.6)\n")
L.append(f"Exploratory and descriptive; rules in `docs/BIOLOGY_CHECK.md` (written after sweep scores were read, before any value of this "
         f"check was computed). Code commit `{rec['git_commit'][:7]}` (tree clean: {not rec['git_dirty']}). {n_fits} saved fits on "
         f"{len(rec['tasks'])} tasks; no new fits. Penalties at the label-transfer strength (batch score nearest b*), seeds 100-102; "
         "plain scVI both decoders; every scored baseline setting. 15 nearest neighbours in the latent.\n")
L.append("Differences are from plain scVI (penalties: same decoder and seed; baselines: conditioned plain scVI, mean of 3 seeds). "
         "Values: mean over tasks of the per-task seed means; 'seed SD' = mean over the 6 real tasks of the SD over the 3 seeds (corrected 2026-11-05; the first version of this page "
         "averaged all 8 tasks). Real = 6 real "
         "tasks for purity; the rare-type columns average the 4 real tasks that have rare types; markers = the 4 RNA tasks; sim = 2 "
         "simulations. Baselines with several settings: mean over settings (purity range in brackets); their seed SD is over seeds "
         "0-4 (n/a for one-run tools: Harmony, Scanorama, PCA).\n")
L.append("| method | purity diff (real) | seed SD | purity diff (sim) | rare purity diff | rare recall diff | marker Jaccard | Jaccard diff | n real / n RNA |")
L.append("|---|---|---|---|---|---|---|---|---|")
for r in rows:
    name = r[0] + (r[10] if len(r) > 10 else "")
    L.append(f"| {name} | {f(r[1])} | {g(r[2])} | {f(r[3])} | {f(r[4])} | {f(r[5])} | {g(r[6])} | {f(r[7])} | {r[8]} / {r[9]} |")
L.append("")
L.append(f"**Rare types** (< 1% of cells): per task {', '.join(f'{t} {n}' for t, n in n_rare_types.items())}. Over the "
         f"{n_rare_cells} task x penalty x decoder cells with rare types ({', '.join(rare_tasks)}), rare-type purity is below plain scVI's "
         f"in {n_neg}; the largest drop is {worst.rare_purity_diff_mean:+.3f} ({worst.method}, {worst.decoder}, {worst.task}).\n")
L.append("scANVI is trained with the cell-type labels; its readouts are not label-free. Low-signal pairs are included (flagged in "
         f"`biology_check.json`, `matched`: {sum(m['low_signal'] for m in rec['matched'])} of {len(rec['matched'])} task x decoder x "
         "penalty cells). No hypothesis test; numbers describe these fits only.\n")
L.append("Tables: `tables/<task>_fit_types.csv` (per fit x type), `<task>_fits.csv`, `<task>_confusion.csv` (seed-mean label-transfer "
         "confusion), `summary.csv`, `summary_by_method.csv`.")
open(out_md, "w").write("\n".join(L) + "\n")
print("n_fits", n_fits, "rare cells", n_rare_cells, "neg", n_neg)
