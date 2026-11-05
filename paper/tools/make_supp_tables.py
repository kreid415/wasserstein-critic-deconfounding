#!/usr/bin/env python
"""Supplementary Tables S6 and S7 as LaTeX, generated from paper/figure_data (no typed numbers).

Inputs (paper/figure_data, checksums in MANIFEST.sha256; sweep analysis of 2026-11-05, commit 8595a16):
  ranking_pairs.csv, low_signal_pairs.csv   Table S6 (biology at b*, leave-one-seed-out SE and count, ranks, rank range)
  transfer_strength.csv, single_methods.csv Table S7a (selection score at the transferred strength, own best in parentheses)
  placement_summary.csv                     Table S7b (real tasks on which a penalty beats a tool, k = 1 and k = 6)
  paper_numbers.json                        cross-check of loso_se, loso_m, rank and bio_at_bstar against the S6 rows
Outputs: paper/supp/tableS6.tex, paper/supp/tableS7.tex.
Positive control: the same code, run for the batch-conditioned decoder, must reproduce the cells of Table 4 in paper/outline.tex
(python paper/tools/make_supp_tables.py --check-table4).
"""
import argparse
import json
import math
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]          # .../paper
DATA, OUT = ROOT / "figure_data", ROOT / "supp"
PEN = ["discriminator", "reference", "pooled", "barycenter", "mmd", "sinkhorn"]
PEN_LABEL = {"discriminator": "JS discriminator", "reference": "Reference critic", "pooled": "Pooled critic",
             "barycenter": "Barycenter critic", "mmd": "MMD", "sinkhorn": "Sinkhorn"}
REAL = ["pancreas", "lung", "immune", "atac_small", "atac_large", "immune_hum_mou"]
SIMS = ["sim1", "sim2"]
TASK_LABEL = {"pancreas": "Pancreas", "lung": "Lung", "immune": "Immune", "atac_small": "ATAC small",
              "atac_large": "ATAC large", "immune_hum_mou": "Human--mouse immune", "sim1": "Simulation 1",
              "sim2": "Simulation 2"}
DEC_LABEL = {1: "batch-conditioned", 0: "unconditioned"}


def f3(x):
    return f"{x:.3f}"


def fr(x):  # ranks are integers or averages of tied integers
    return f"{x:.0f}" if float(x).is_integer() else f"{x:.1f}"


def read(name):
    df = pd.read_csv(DATA / name)
    assert len(df), name
    return df


# ------------------------------------------------------------------ Table S6
def check_against_paper_numbers(rk, low):
    pn = json.load(open(DATA / "paper_numbers.json"))
    n = 0
    for _, r in rk.iterrows():
        key = f"{r.arm}.{r.task}.cond{int(r.cond)}"
        for field, col in (("bio_at_bstar", "bio_at_bstar"), ("loso_se", "loso_se"), ("rank", "rank")):
            v = pn[f"{field}.{key}"]["value"]
            assert math.isclose(v, r[col], rel_tol=1e-9, abs_tol=1e-12), (field, key, v, r[col])
            n += 1
        assert pn[f"loso_m.{key}"]["value"] == r.n_loso, key
        lo, hi = pn[f"rank_range.{key}"]["value"]
        assert (lo, hi) == (r.rank_lo, r.rank_hi), key
    for _, r in low.iterrows():
        key = f"{r.arm}.{r.task}.cond{int(r.cond)}"
        assert math.isclose(pn[f"bio_at_bstar.{key}"]["value"], r.bio_at_bstar, rel_tol=1e-9), key
    return n


def row_s6(r, min_readings=2):
    ok = r.n_loso >= min_readings if pd.notna(r.n_loso) else False   # rule of Methods: at least two of three readings
    se = f3(r.loso_se) if ok else "n/a"
    m = str(int(r.n_loso)) if pd.notna(r.n_loso) else "--"
    if pd.isna(r["rank"]):
        rank, rng = "not ranked", "--"
    else:
        rank = fr(r["rank"])
        rng = (fr(r.rank_lo) if r.rank_lo == r.rank_hi else f"{fr(r.rank_lo)}--{fr(r.rank_hi)}") if ok else "n/a"
    mark = "" if r.status == "bracketed" else r"$^\dagger$"
    return f"{PEN_LABEL[r.arm]}{mark} & {f3(r.bio_at_bstar)} & {se} & {m} & {rank} & {rng} & {int(r.crossings)} \\\\"


def block_s6(df, tasks):
    lines = []
    for cond in (1, 0):
        for t in tasks:
            g = df[(df.task == t) & (df.cond == cond)]
            if g.empty:
                continue
            assert len(g) == 6 and set(g.arm) == set(PEN), (t, cond)
            b = g.bstar.iloc[0]
            lines.append(r"\midrule")
            lines.append(rf"\multicolumn{{7}}{{@{{}}l}}{{\textit{{{TASK_LABEL[t]}, {DEC_LABEL[cond]} decoder; $b^*={f3(b)}$}}}} \\")
            for arm in PEN:
                lines.append(row_s6(g[g.arm == arm].iloc[0]))
    return "\n".join(lines)


def make_s6():
    rk, low = read("ranking_pairs.csv"), read("low_signal_pairs.csv")
    n_checked = check_against_paper_numbers(rk, low)
    assert not rk.low_signal.any() and low.low_signal.all()
    pairs = rk.groupby(["task", "cond"]).ngroups
    low_pairs = low.groupby(["task", "cond"]).ngroups
    n_nb = int((rk.status != "bracketed").sum() + (low.status != "bracketed").sum())
    n_se_na = int((rk.n_loso < 2).sum())
    unreached = ("In every pair every penalty's curve reaches $b^*$, so no reading is marked as not reached." if n_nb == 0
                 else r"A dagger marks a penalty whose curve does not reach $b^*$ (bottom rank in that pair).")
    cap_a = (rf"Ranking detail: biology score at the matched batch removal $b^*$ per penalty, task and decoder, with the "
             rf"leave-one-seed-out standard error (SE) of the reading (the same seed left out of plain scVI, $b^*$ held fixed), "
             rf"the number $m$ of the three leave-one-seed-out repeats that gave a reading, the rank (1 is the highest biology; "
             rf"ties share the average rank) and the lowest to highest rank over the three repeats. The SE and the rank range "
             rf"are given only where $m\geq2$ (otherwise n/a; {n_se_na} of {len(rk)} ranked readings). ``Cross.'' is the number of times "
             rf"the penalty's seed-mean curve crosses $b^*$; the reading is taken at the first crossing. {unreached} "
             rf"{pairs} task and decoder pairs are ranked; the immune task and both simulations with the batch-conditioned decoder "
             rf"are low-signal and are in Part B. Simulations are ranked separately from the real tasks "
             rf"(Supplementary Notes 5 and 6; Table~3 gives the mean ranks).")
    cap_b = (rf"Low-signal task and decoder pairs ({low_pairs}; the gap between plain scVI's batch score and the highest batch score "
             rf"every penalty reaches is under 0.02): biology score at $b^*$ read in the same way, not ranked and without standard errors "
             rf"(Supplementary Note 5).")
    rows_a = block_s6(rk, REAL + SIMS)
    rows_b = block_s6(low, ["immune", "sim1", "sim2"])
    cap = (rf"{cap_a} \textbf{{Part B (last block)}}: {cap_b}")
    part_b = (r"\midrule" + "\n" + r"\multicolumn{7}{@{}l}{\textbf{Part B: low-signal pairs (not ranked)}} \\" + "\n" + rows_b)
    tex = [r"% generated by paper/tools/make_supp_tables.py; do not edit",
           r"{\scriptsize\renewcommand{\tablename}{Supplementary Table}\renewcommand{\thetable}{S6}%",
           r"\setlength{\tabcolsep}{4pt}",
           r"\begin{longtable}{@{}lrrrrrr@{}}",
           rf"\caption{{{cap}}}\label{{tab:s6}}\\", r"\toprule Penalty & Biology at $b^*$ & SE & $m$ & Rank & Rank range & Cross. \\" + "\n" +
           r"\multicolumn{7}{@{}l}{\textbf{Part A: ranked pairs}} \\" + "\n" + rows_a + "\n" + part_b + r"\\ \bottomrule",
           r"\end{longtable}}"]
    (OUT / "tableS6.tex").write_text("\n".join(tex) + "\n")
    return dict(rows_a=len(rk), rows_b=len(low), crosschecked_values=n_checked, n_not_bracketed=n_nb, n_se_na=n_se_na)


# ------------------------------------------------------------------ Table S7
TOOLS = [("harmony", "Harmony"), ("scanorama", "Scanorama"), ("sysvi", "sysVI")]


def transfer_cells(cond):
    """{method label: {task: 'S (best S)' or 'S'}} and the counts above plain scVI (transferred, own best)."""
    tr, sm = read("transfer_strength.csv"), read("single_methods.csv")
    sm = sm.assign(tool=sm.name.str.split("|").str[0])
    cells, cnt = {}, {}

    def single(label, tool, name):
        g = sm[(sm.cond == cond) & (sm.name == name)].set_index("task")
        assert set(REAL) <= set(g.index), (label, name)
        cells[label] = {t: f3(g.loc[t, "S"]) for t in REAL}
        cnt[label] = (int((g.loc[REAL, "S_minus_plain_scvi"] > 0).sum()), None)

    single("Plain scVI", "none", "none|0|10")
    if cond == 1:                       # the scvi-tools adversary is run with the batch-conditioned decoder only
        single("scvi-tools adversary", "scvi_adv", "scvi_adv|0|10")
    fams = [(p, PEN_LABEL[p]) for p in PEN] + [(k, v) for k, v in TOOLS]
    for fam, label in fams:
        g = tr[(tr.cond == cond) & (tr.family == fam)].set_index("task")
        assert set(REAL) <= set(g.index), (fam, cond)
        cells[label] = {t: f"{f3(g.loc[t, 'S'])} ({f3(g.loc[t, 'best_S'])})" for t in REAL}
        cnt[label] = (int((g.loc[REAL, "S_minus_plain_scvi"] > 0).sum()), int((g.loc[REAL, "best_S_minus_plain_scvi"] > 0).sum()))
    single("scANVI (labels)", "scanvi", "scanvi|0|10")
    single("PCA (uncorrected)", "pca", "pca|0|10")
    return cells, cnt


S7_ORDER = (["Plain scVI", "scvi-tools adversary"] + [PEN_LABEL[p] for p in PEN] + ["Harmony", "Scanorama", "sysVI"]
            + ["scANVI (labels)", "PCA (uncorrected)"])


def tabular_s7a(cond, with_counts=True):
    cells, cnt = transfer_cells(cond)
    out = [r"\setlength{\tabcolsep}{3pt}", r"\begin{tabular}{@{}l" + "c" * len(REAL) + ("c" if with_counts else "") + r"@{}}", r"\toprule",
           "Method & " + " & ".join(TASK_LABEL[t] for t in REAL) + (r" & Above plain scVI" if with_counts else "") + r" \\", r"\midrule"]
    for i, m in enumerate(x for x in S7_ORDER if x in cells):
        if m == "Harmony" or m == "scANVI (labels)" or m == PEN_LABEL["discriminator"]:
            out.append(r"\midrule")
        c = cnt[m]
        extra = ""
        if with_counts:
            extra = ("" if m == "Plain scVI" else (f" & {c[0]} ({c[1]})" if c[1] is not None else f" & {c[0]}"))
        out.append(m + " & " + " & ".join(cells[m][t] for t in REAL) + extra + r" \\")
    out += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(out), cells


def tabular_s7b(cond=0):
    ps = read("placement_summary.csv")
    ps = ps[ps.cond == cond]
    tools = [("none", "Plain scVI"), ("harmony", "Harmony"), ("scanorama", "Scanorama"),
             ("sysvi", "sysVI"), ("scanvi", "scANVI (labels)"), ("pca", "PCA")]
    out = [r"\setlength{\tabcolsep}{3pt}", r"\begin{tabular}{@{}l" + "cc" * len(tools) + r"@{}}", r"\toprule",
           " & " + " & ".join(rf"\multicolumn{{2}}{{c}}{{{lab}}}" for _, lab in tools) + r" \\",
           "".join(rf"\cmidrule(lr){{{2 + 2 * i}-{3 + 2 * i}}}" for i in range(len(tools))),
           "Penalty & " + " & ".join(r"$k{=}1$ & $k{=}6$" for _ in tools) + r" \\", r"\midrule"]
    for p in PEN:
        cells = []
        for tool, _ in tools:
            for k in (1, 6):
                r = ps[(ps.penalty == p) & (ps.tool == tool) & (ps.k == k)]
                assert len(r) == 1, (p, tool, k)
                cells.append(f"{int(r.n_real_tasks_better.iloc[0])}/{int(r.n_real_tasks.iloc[0])}")
        out.append(PEN_LABEL[p] + " & " + " & ".join(cells) + r" \\")
    out += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(out)


def make_s7():
    cond = 0
    a, _ = tabular_s7a(cond)
    b = tabular_s7b(cond)
    cap = (r"Placement with the unconditioned decoder (counterpart of Table~4 and of Figure~6). \textbf{(a)} Mean selection score "
           r"($0.4\times$batch $+\,0.6\times$biology) per method and real task at the strength (Harmony, Scanorama, sysVI: the setting) with the best mean "
           r"selection score on real tasks of other task families (related tasks never choose for each other); in parentheses, the task's own best "
           r"strength or setting (optimistic bound). Plain scVI here receives no batch information. The scvi-tools adversary is run with the batch-conditioned decoder only and is not shown. scANVI (uses cell-type labels), PCA and the three tools have no decoder, so their values equal those of Table~4. The last column counts the real "
           r"tasks (of six) on which the method's score exceeds plain scVI's, at the transferred strength and, in parentheses, at the own best. "
           r"\textbf{(b)} Number of real tasks on which the penalty's expected best selection score among $k$ randomly drawn settings exceeds "
           r"that of the tool at $k$ settings of its own, $k=1$ and $k=6$; Scanorama at $k=6$ is compared on the five tasks where all six of its settings exist "
           r"(the 80-neighbour setting could not be fitted on the large chromatin-accessibility task). Counts are descriptive (Supplementary Notes 7 and 8).")
    tex = [r"% generated by paper/tools/make_supp_tables.py; do not edit",
           r"\begin{table}[t]", r"\centering\scriptsize\renewcommand{\tablename}{Supplementary Table}\renewcommand{\thetable}{S7}%",
           rf"\caption{{{cap}}}\label{{tab:s7}}", r"\textbf{(a)}\\[2pt]", a, r"\\[6pt]", r"\textbf{(b)}\\[2pt]", b, r"\end{table}"]
    (OUT / "tableS7.tex").write_text("\n".join(tex) + "\n")


def check_table4():
    """The cond=1 table must equal Table 4 of the outline (cell strings)."""
    txt = (ROOT / "outline.tex").read_text()
    blk = txt.split(r"\label{tab:4}")[1].split(r"\end{tabular}")[0]
    ref = {}
    for ln in blk.splitlines():
        if "&" in ln and not ln.startswith("Method"):
            parts = [p.strip() for p in ln.replace(r"\\", "").split("&")]
            ref[parts[0]] = parts[1:7]
    cells, _ = transfer_cells(1)
    name = {"Plain scVI": "Plain scVI", "scvi-tools adversary": "scvi-tools adversary", "JS discriminator": "JS discriminator",
            "Reference critic": "Reference critic", "Pooled critic": "Pooled critic", "Barycenter critic": "Barycenter critic", "MMD": "MMD",
            "Sinkhorn": "Sinkhorn", "Harmony": "Harmony", "Scanorama": "Scanorama", "sysVI": "sysVI", "scANVI (labels)": "scANVI (labels)",
            "PCA (uncorrected)": "PCA (uncorrected)"}
    order = [r"Pancreas", "Lung", "Immune", "ATAC small", "ATAC large", "Human--mouse immune"]
    assert [TASK_LABEL[t] for t in REAL] == order
    n = 0
    for m, row in cells.items():
        assert m in ref, m
        got = [row[t] for t in REAL]
        assert got == ref[m], (m, got, ref[m])
        n += len(got)
    print(f"Table 4 reproduced: {len(cells)} rows, {n} cells identical")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--check-table4", action="store_true")
    a = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    if a.check_table4:
        check_table4()
    print(make_s6())
    make_s7()
    print("wrote", OUT / "tableS6.tex", OUT / "tableS7.tex")
