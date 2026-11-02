#!/usr/bin/env python
"""Tables, main figures and the flat results file from a sweep analysis record (scripts/analysis_sweep.py).

Reads only the record; values are copied, never recomputed. Writes to --out:
  tables/   ranking_pairs.csv, mean_rank.csv, secondary.csv, curves.csv, baseline_points.csv, label_transfer.csv,
            label_transfer_curve.csv, best_of_k.csv, best_of_k_real_mean.csv, transfer_strength.csv, single_methods.csv,
            baseline_coverage.csv, counts.csv
  figures/  fig_ranking, fig_tradeoff_cond1, fig_tradeoff_cond0, fig_best_of_k, fig_collapse (each .png at 300 dpi and .pdf)
  paper_numbers.json: every number a paper would quote, {key: {"value": ..., "description": ...}}, plus "_meta".

Usage: python scripts/sweep_report.py --record <analysis_sweep.json> --out <dir>
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
for p in (HERE, os.path.join(HERE, "dryrun")):
    sys.path.insert(0, p)
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import prereg_rules as P  # noqa: E402
from figure_helpers import apply_figure_style, panel_letter  # noqa: E402
from figures_x1 import figure as ranking_figure  # noqa: E402

ARMS = list(P.ADV_ARMS)
TOOLS = ["harmony", "scanorama", "sysvi"]
SINGLE = ["none", "scvi_adv", "scanvi", "pca"]
DECODER = {1: "conditioned decoder", 0: "unconditioned decoder"}
COLOUR = dict(zip(ARMS, ["#1b9e77", "#d95f02", "#7570b3", "#e7298a", "#66a61e", "#e6ab02"]))
BASE_MARK = dict(none=("k", "*"), scvi_adv=("k", "P"), scanvi=("k", "X"), sysvi=("0.35", "^"), harmony=("0.35", "s"),
                 scanorama=("0.35", "D"), pca=("0.6", "v"))


def tables(rec):
    prim, T = rec["primary"], {}
    pairs, sec, curves, bpts, lt, ltc, bok, single, setting_cov = [], [], [], [], [], [], [], [], []
    for key, c in prim["cells"].items():
        t, cond = c["task"], c["cond"]
        for arm in ARMS:
            r, w, col = c["penalties"][arm], c["window"][arm], c["collapse"][arm]
            ranked = "rank" in r
            pairs.append(dict(task=t, cond=cond, arm=arm, low_signal=c["low_signal"], b0=c["b0"], b_common=c["b_common"], bstar=c["bstar"],
                              best_batch=c["best_batch"][arm], n_points=c["n_points"][arm], bio_at_bstar=r["bio"], status=r["status"],
                              crossings=r["crossings"], rank=r.get("rank") if ranked else None, rank_lo=r.get("rank_lo") if ranked else None,
                              rank_hi=r.get("rank_hi") if ranked else None, loso_se=r.get("s") if ranked else None,
                              n_loso=r.get("m") if ranked else None, local_minus_global=r["local_minus_global"]))
            nd = c["nondominated"][arm]
            sec.append(dict(task=t, cond=cond, arm=arm, hypervolume=None if c["hypervolume"] is None else c["hypervolume"][arm],
                            nondominated_share=nd["share"], nondominated_n=nd["n"], collapse_n=col["n"],
                            collapsed=col["collapsed"], collapsed_bio=col["bio"], diverged=col["failed"]["diverged"],
                            nonfinite_latent=col["failed"]["nonfinite_latent"], collapse_rate=col["rate"], collapse_lo=col["lo"],
                            collapse_hi=col["hi"], window_status=w["status"], window_best=w["best"], window_width_decades=w["width"],
                            window_n_points=w["n_points"], window_gap=w["gap"]))
            curves += [dict(task=t, cond=cond, arm=arm, lam=lam, B=b, C=cc, n_fail=nf) for lam, b, cc, nf in c["curve"][arm]]
        bpts += [dict(task=t, cond=cond, name=n, **v) for n, v in c["baseline_points"].items()]
        setting_cov += c.get("baseline_settings", [])
        single += [dict(task=t, cond=cond, name=n, n_ok=v["n_ok"], n_fail=v["n_fail"], B=v["B"][0], B_sd=v["B"][1], C=v["C"][0],
                        C_sd=v["C"][1], S=v["S"][0], S_sd=v["S"][1], S_minus_plain_scvi=v["S_minus_plain_scvi"]) for n, v in c["single"].items()]
        for fam, v in c["best_of_k"].items():
            bok += [dict(task=t, cond=cond, family=fam, stratum=v["stratum"], k=k + 1, E=e, E_minus_plain_scvi=d, n_configs=v["n"], failed_runs=v["failed_runs"])
                    for k, (e, d) in enumerate(zip(v["E"], v["E_minus_plain_scvi"]))]
    L = prim["label_transfer"]
    for cond, d in L["decoders"].items():
        for t, arms in d["tasks"].items():
            for arm, x in arms.items():
                lt.append(dict(cond=int(cond), task=t, arm=arm, lam=x["lam"], f1_mean=x["mean"], f1_sd=x["sd"], n_runs=x["n_runs"],
                               computed=x["computed"], not_evaluable=x["not_evaluable"], batch_score=x["batch_score"], distance=x["distance"]))
                ltc += [dict(cond=int(cond), task=t, arm=arm, lam=y["lam"], f1_mean=y["mean"], f1_sd=y["sd"], computed=y["computed"],
                             batch_score=y["batch_score"], distance=y["distance"]) for y in x.get("curve", [])]
    mean = [dict(cond=int(k), arm=a, mean_rank=v["mean"], n_tasks=v["n_tasks"]) for k, d in prim["ranking"].items()
            for a, v in d["mean_rank"].items()]
    real_mean = [dict(cond=int(k), family=f, stratum=next(v["best_of_k"][f]["stratum"] for v in prim["cells"].values() if f in v["best_of_k"]), k=i + 1, mean_E=e, n_tasks=n) for k, d in prim["best_of_k_real_mean"].items()
                 for f, v in d.items() for i, (e, n) in enumerate(v)]
    tr = []
    for cond, d in prim["transfer"].items():
        for fam, v in d.items():
            for t, x in v.items():
                row = dict(cond=int(cond), family=fam, task=t, train=" ".join(x["train"]), lam=x["lam"], best_lam=x["best_lam"])
                for name, at in (("", x["at"]), ("best_", x["best"])):
                    for q in ("B", "C", "S"):
                        row[name + q], row[name + q + "_sd"] = (at[q] if at else (None, None))
                    row[name + "S_minus_plain_scvi"] = at["S_minus_plain_scvi"] if at else None
                tr.append(row)
    pair_df = pd.DataFrame(pairs)
    status = pair_df.groupby(["cond", "arm", "status"]).size().rename("n_pairs").reset_index()
    counts = [dict(kind="sweep_fits", task=t, n=n) for t, n in rec["counts"]["sweep_fits"].items()]
    counts += [dict(kind="sweep_failed", task=t, n=n) for t, n in rec["counts"]["sweep_failed"].items()]
    counts += [dict(kind=f"reading_status_{r.status}", task=f"cond{r.cond}|{r.arm}", n=int(r.n_pairs)) for r in status.itertuples()]
    T.update(ranking_pairs=pair_df[~pair_df.low_signal].reset_index(drop=True), low_signal_pairs=pair_df[pair_df.low_signal].reset_index(drop=True),
             mean_rank=pd.DataFrame(mean), secondary=pd.DataFrame(sec), curves=pd.DataFrame(curves),
             baseline_points=pd.DataFrame(bpts), label_transfer=pd.DataFrame(lt), label_transfer_curve=pd.DataFrame(ltc),
             best_of_k=pd.DataFrame(bok), best_of_k_real_mean=pd.DataFrame(real_mean), transfer_strength=pd.DataFrame(tr),
             single_methods=pd.DataFrame(single), baseline_coverage=pd.DataFrame(rec["baselines"]["coverage"]),
             baseline_setting_coverage=pd.DataFrame(setting_cov).drop_duplicates().reset_index(drop=True), counts=pd.DataFrame(counts))
    return T


def save(fig, path):
    for ext in (".png", ".pdf"):
        fig.savefig(path + ext, dpi=300)
    plt.close(fig)


def tradeoff_figure(rec, T, cond, path):
    """One panel per task: biology against batch removal along each penalty's curve (anchor = plain scVI), b* (dashed) and the
    collapse line c0 - 0.10 (dotted); baseline configurations as grey or black markers."""
    cells = {c["task"]: c for c in rec["primary"]["cells"].values() if c["cond"] == cond}
    tasks = sorted(cells)
    apply_figure_style(sizes=(8, 7, 6))
    ncol = 4
    nrow = int(np.ceil(len(tasks) / ncol))
    fig, axs = plt.subplots(nrow, ncol, figsize=(17.4 / 2.54, 4.6 * nrow / 2.54 + 1.4), squeeze=False,
                            gridspec_kw=dict(wspace=0.35, hspace=0.6, bottom=0.28 / nrow + 0.08, top=0.86, left=0.08, right=0.99))
    for ax, t in zip(axs.flat, tasks):
        c = cells[t]
        for arm in ARMS:
            cv = np.array([(b, cc) for _, b, cc, _ in c["curve"][arm]])
            ax.plot(cv[:, 0], cv[:, 1], "-o", color=COLOUR[arm], ms=2.2, lw=0.9, label=arm)
        for name, v in c["baseline_points"].items():
            col, mk = BASE_MARK[v["arm"]]
            ax.plot(v["B"], v["C"], mk, color=col, ms=3.5, mew=0.6, mfc="none", label=v["arm"])
        ax.axvline(c["bstar"], color="k", ls="--", lw=0.7)
        ax.axhline(c["curve"][ARMS[0]][0][2] - P.COLLAPSE_BIO, color="k", ls=":", lw=0.7)
        ax.set_title(t + (" (low-signal)" if c["low_signal"] else ""), loc="left")
        ax.set_xlabel("batch score B")
        ax.set_ylabel("biology score C")
    for ax in axs.flat[len(tasks):]:
        ax.axis("off")
    handles, labels = {}, []
    for ax in axs.flat:
        for h, lab in zip(*ax.get_legend_handles_labels()):
            if lab not in handles:
                handles[lab], labels = h, labels + [lab]
    fig.legend([handles[lab] for lab in labels], labels, loc="lower center", ncol=min(7, len(labels)), frameon=False, fontsize=6)
    fig.suptitle(f"{DECODER[cond]}: biology against batch removal over the strength sweep; dashed = b*, dotted = collapse line", x=0.01,
                 y=0.985, ha="left", fontsize=7)
    save(fig, path)


def best_of_k_figure(rec, T, path):
    """Per decoder, the mean over the real tasks of each method's expected best of k tries (spec 11, main figure)."""
    apply_figure_style(sizes=(8, 7, 6))
    fig, axs = plt.subplots(1, 2, figsize=(17.4 / 2.54, 7.5 / 2.54), gridspec_kw=dict(wspace=0.25, bottom=0.38, left=0.08, right=0.99))
    for ax, cond in zip(axs, (1, 0)):
        d = T["best_of_k_real_mean"][T["best_of_k_real_mean"].cond == cond]
        for fam, g in d.groupby("family"):
            g = g.sort_values("k")
            label = f"{fam} (label-aware)" if fam == "scanvi" else fam
            if fam in ARMS:
                ax.plot(g.k, g.mean_E, "-o", color=COLOUR[fam], ms=2.5, lw=0.9, label=fam)
            elif fam in TOOLS:
                col, mk = BASE_MARK[fam]
                ax.plot(g.k, g.mean_E, "--" + mk, color=col, ms=3, lw=0.8, mfc="none", label=label)
            else:
                col, mk = BASE_MARK[fam]
                ax.plot(g.k, g.mean_E, mk, color=col, ms=4, mfc="none", label=label)
        ax.set_title(DECODER[cond], loc="left")
        ax.set_xlabel("k tries")
        ax.set_ylabel("expected best selection score S")
        ax.set_xticks(sorted(set(d.k)))
    panel_letter(axs[0], "a", dx=-0.2)
    panel_letter(axs[1], "b", dx=-0.2)
    h, lab = axs[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", bbox_to_anchor=(0.5, 0.07), ncol=7, frameon=False, fontsize=6)
    n = sorted(set(T["best_of_k_real_mean"].n_tasks))
    fig.text(0.01, 0.01, f"mean over the real tasks with at least k configurations ({', '.join(map(str, n))} tasks); S = 0.4 B + 0.6 C", fontsize=6)
    save(fig, path)


def collapse_figure(T, path):
    """Per decoder: each penalty's collapse rate per task with its exact 95% interval."""
    apply_figure_style(sizes=(8, 7, 6))
    sec = T["secondary"]
    tasks = sorted(set(sec.task))
    fig, axs = plt.subplots(1, 2, figsize=(17.4 / 2.54, 7.5 / 2.54), gridspec_kw=dict(wspace=0.1, bottom=0.36, left=0.14, right=0.99))
    cmap = plt.cm.tab10
    for ax, cond in zip(axs, (1, 0)):
        d = sec[sec.cond == cond]
        for i, arm in enumerate(ARMS):
            for j, t in enumerate(tasks):
                r = d[(d.arm == arm) & (d.task == t)].iloc[0]
                y = i + (j - (len(tasks) - 1) / 2) * 0.8 / max(len(tasks), 1)
                ax.errorbar(r.collapse_rate, y, xerr=[[r.collapse_rate - r.collapse_lo], [r.collapse_hi - r.collapse_rate]], fmt="o",
                            color=cmap(j % 10), ms=2.2, elinewidth=0.6, capsize=0, label=t if i == 0 else None)
        ax.set_yticks(range(len(ARMS)), ARMS if cond == 1 else [])
        ax.set_ylim(len(ARMS) - 0.4, -0.6)
        ax.set_xlim(-0.02, 1.02)
        ax.set_xlabel("collapse rate (runs collapsed / runs)")
        ax.set_title(DECODER[cond], loc="left")
    panel_letter(axs[0], "a", dx=-0.32)
    panel_letter(axs[1], "b", dx=-0.1)
    h, lab = axs[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", bbox_to_anchor=(0.5, 0.07), ncol=min(8, len(tasks)), frameon=False, fontsize=6)
    fig.text(0.01, 0.01, "a run collapses when it failed or its biology is 0.10 or more below the same seed's plain scVI; bars = exact 95% interval",
             fontsize=6)
    save(fig, path)


def paper_numbers(rec, T, record_path):
    """Every number a paper would quote, under a short key with a one-line description."""
    N = {}

    def put(key, value, desc):
        P._require(key not in N, f"duplicate paper number key {key}")
        N[key] = dict(value=None if value is None or (isinstance(value, float) and not np.isfinite(value)) else value, description=desc)

    prim = rec["primary"]
    put("_meta.record_sha256", P.sha256(record_path), "SHA-256 of the analysis record these numbers were read from")
    put("_meta.git_sha", rec["git_sha"], "git commit of the analysis code")
    put("_meta.git_dirty", rec["git_dirty"], "analysis code had uncommitted changes")
    put("_meta.partial_tasks", rec["partial_tasks"], "fewer than the eight tasks were analysed (a plumbing run, not the result)")
    put("_meta.label_transfer_debug_subset", rec["label_transfer_debug_subset"],
        "label transfer limited to this many embeddings per task (DEBUG run, not a result) or null")
    put("_meta.baselines_complete", rec["baselines"]["complete"], "every baseline row of the run list was scored or failed")
    put("_meta.tasks", rec["tasks"], "tasks analysed")
    put("_meta.seeds", rec["seeds"], "sweep seeds")
    for t, n in rec["counts"]["sweep_fits"].items():
        put(f"n_fits.{t}", n, f"sweep fits (rows) on {t}, both decoders, all strengths and seeds")
        put(f"n_failed.{t}", rec["counts"]["sweep_failed"][t], f"sweep fits on {t} that failed (diverged or non-finite latent)")
    put("n_fits.total", sum(rec["counts"]["sweep_fits"].values()), "sweep fits in the analysis")
    put("n_failed.total", sum(rec["counts"]["sweep_failed"].values()), "sweep fits that failed")
    put("n_baseline_rows.in_analysis", rec["counts"]["baseline_rows"], "baseline rows (scored or failed) in the analysis")
    put("n_baseline_rows.missing", len(rec["baselines"]["missing"]), "baseline rows of the run list neither scored nor failed")
    for r in rec["baselines"]["coverage"]:
        for col in ("expected", "scored", "failed", "missing"):
            put(f"baseline_coverage.{r['task']}.{r['arm']}.{col}", int(r[col]), f"{r['arm']} rows on {r['task']}: {col}")
    for key, c in prim["cells"].items():
        t, cond = c["task"], c["cond"]
        for q, desc in (("b0", "plain scVI's mean batch score"), ("b_common", "highest batch score all six penalties reach"),
                        ("bstar", "matched batch-removal target b* (half-way from b0 to b_common)"), ("low_signal", "pair is low-signal (b_common - b0 < 0.02): not ranked")):
            put(f"pair.{t}.cond{cond}.{q}", c[q], f"{desc}, {t}, {DECODER[cond]}")
        for arm in ARMS:
            r, pre = c["penalties"][arm], f"{arm}.{t}.cond{cond}"
            put(f"bio_at_bstar.{pre}", r["bio"], f"biology at b* of {arm} on {t}, {DECODER[cond]} (null = curve gives no reading)")
            put(f"reading_status.{pre}", r["status"], "bracketed / unreached_low / unreached_high / anchor_at_or_above / no_points")
            put(f"crossings.{pre}", r["crossings"], "times the curve crosses or touches b*")
            put(f"local_minus_global.{pre}", r["local_minus_global"], "local minus global batch removal at the crossing")
            if "rank" in r:
                put(f"rank.{pre}", r["rank"], f"rank of {arm} among the six penalties on {t}, {DECODER[cond]} (1 = best; bottom places shared)")
                put(f"rank_range.{pre}", [r["rank_lo"], r["rank_hi"]], "smallest and largest rank over the leave-one-seed-out repeats")
                put(f"loso_se.{pre}", r["s"], f"leave-one-seed-out standard error of the reading (null when fewer than {rec['min_readings']} repeats read)")
                put(f"loso_m.{pre}", r["m"], "leave-one-seed-out repeats with a reading")
            put(f"hypervolume.{pre}", None if c["hypervolume"] is None else c["hypervolume"][arm],
                "hypervolume (Zitzler and Thiele 1998; Li et al. 2022) from the corner (b0, c0 - 0.10) over the common batch range (null in low-signal pairs)")
            nd = c["nondominated"][arm]
            put(f"nondominated_share.{pre}", nd["share"],
                f"share of the penalty's strengths that no other of its strengths beats on both batch and biology ({nd['n_nondominated']} of {nd['n']})")
            col = c["collapse"][arm]
            put(f"collapse_rate.{pre}", col["rate"], f"collapsed runs / runs ({col['collapsed']} of {col['n']})")
            put(f"collapse_ci.{pre}", [col["lo"], col["hi"]], "exact Clopper-Pearson 95% interval of the collapse rate")
            put(f"collapse_failed.{pre}", col["failed"], "failed runs by type among the collapsed")
            w = c["window"][arm]
            put(f"width_decades.{pre}", w["width"], "good-strength width in decades (window of strengths within the best point's 95% t interval)")
            put(f"best_strength.{pre}", w["best"], "strength with the highest mean selection score S")
            put(f"n_strengths.{pre}", c["n_points"][arm], "strengths fitted for this penalty in this pair")
    for cond, d in prim["ranking"].items():
        for arm, v in d["mean_rank"].items():
            put(f"mean_rank.cond{cond}.{arm}", v["mean"], f"mean rank of {arm} over the ranked real tasks, {DECODER[int(cond)]}")
            put(f"mean_rank_n_tasks.cond{cond}.{arm}", v["n_tasks"], "real tasks the mean rank is over")
        put(f"low_signal_tasks.cond{cond}", d["low_signal"], f"low-signal (unranked) tasks, {DECODER[int(cond)]}")
        for t, ranks in d["sim_ranks"].items():
            for arm, rk in ranks.items():
                put(f"sim_rank.{t}.cond{cond}.{arm}", rk, f"rank of {arm} on the simulation {t}, {DECODER[int(cond)]}")
    st = pd.concat([T["ranking_pairs"], T["low_signal_pairs"]]).groupby(["cond", "arm"]).status.apply(lambda s: int((s != "bracketed").sum()))
    for (cond, arm), n in st.items():
        put(f"n_pairs_no_reading.cond{cond}.{arm}", n, f"task pairs in which {arm}'s curve gives no reading at b*, {DECODER[cond]}")
    for cond, d in prim["best_of_k_real_mean"].items():
        for fam, v in d.items():
            for k, (e, n) in enumerate(v, 1):
                put(f"best_of_k_real_mean.cond{cond}.{fam}.k{k}", e, f"mean over {n} real tasks of {fam}'s expected best S of {k} tries")
    for cond, d in prim["transfer"].items():
        for fam, v in d.items():
            for t, x in v.items():
                put(f"transfer_strength.cond{cond}.{fam}.{t}", x["lam"], f"strength of {fam} chosen on the other families' real tasks for {t} (null = none eligible)")
                put(f"transfer_S_minus_plain.cond{cond}.{fam}.{t}", x["at"]["S_minus_plain_scvi"] if x["at"] else None,
                    "S at the transfer strength minus plain scVI's S")
                put(f"best_own_strength.cond{cond}.{fam}.{t}", x["best_lam"], "the task's own best strength (optimistic bound)")
    L = prim["label_transfer"]
    for cond, d in L["decoders"].items():
        for t, arms in d["tasks"].items():
            for arm, x in arms.items():
                put(f"label_transfer_f1.{arm}.{t}.cond{cond}", x["mean"], f"macro F1 of 15-nearest-neighbour label transfer, {arm} at its matched strength on {t}, {DECODER[int(cond)]}")
                put(f"label_transfer_f1_sd.{arm}.{t}.cond{cond}", x["sd"], "SD over seeds")
                put(f"label_transfer_distance.{arm}.{t}.cond{cond}", x["distance"], "distance of the embedding's seed-mean batch score from b*")
        for arm, v in d["summary"].items():
            put(f"label_transfer_mean.cond{cond}.{arm}", v["mean"], f"mean label-transfer F1 of {arm} over {v['n_tasks']} real tasks that are not low-signal")
    for r in T["single_methods"].itertuples():
        put(f"single.{r.name}.{r.task}.cond{r.cond}.S", r.S, f"selection score S of {r.name} on {r.task}, {DECODER[r.cond]} (as it is)")
        put(f"single.{r.name}.{r.task}.cond{r.cond}.S_minus_plain", r.S_minus_plain_scvi, "S minus plain scVI's S")
    return N


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--record", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    rec = json.load(open(a.record))
    P._require(rec.get("rule") == "analysis_sweep" and rec.get("status") == "done", f"{a.record} is not a finished sweep analysis record")
    tdir, fdir = os.path.join(a.out, "tables"), os.path.join(a.out, "figures")
    os.makedirs(tdir, exist_ok=True)
    os.makedirs(fdir, exist_ok=True)
    T = tables(rec)
    for name, df in T.items():
        df.to_csv(os.path.join(tdir, name + ".csv"), index=False)
    pairs = T["ranking_pairs"]
    real_tasks = rec.get("real_tasks", [])
    if real_tasks:
        figure_pairs = pd.concat([pairs, T["low_signal_pairs"]], ignore_index=True)
        for ext in (".png", ".pdf"):
            ranking_figure(rec["primary"], figure_pairs, T["mean_rank"], os.path.join(fdir, "fig_ranking" + ext))
    for cond in (1, 0):
        tradeoff_figure(rec, T, cond, os.path.join(fdir, f"fig_tradeoff_cond{cond}"))
    if real_tasks:
        best_of_k_figure(rec, T, os.path.join(fdir, "fig_best_of_k"))
    collapse_figure(T, os.path.join(fdir, "fig_collapse"))
    N = paper_numbers(rec, T, a.record)
    with open(os.path.join(a.out, "paper_numbers.json"), "w") as f:
        json.dump(N, f, indent=1, sort_keys=True)
    print(f"{len(T)} tables, real-task figures {'included' if real_tasks else 'skipped'}, {len(N)} paper numbers -> {a.out}")
    return 0


if __name__ == "__main__":  # pragma: no cover  # command-line entry
    sys.exit(main())
