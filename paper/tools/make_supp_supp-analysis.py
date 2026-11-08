#!/usr/bin/env python
"""Supplementary Table S8 and the label-transfer table of Supplementary Note 9 as LaTeX, generated from paper/figure_data (no typed numbers).

Inputs (paper/figure_data, checksums in its MANIFEST.sha256; sweep analysis of 2026-11-05, commit 8595a16):
  secondary.csv                          Table S8: hypervolume, non-dominated share, collapse counts and interval, failure types, good-strength window
  pareto_frontier.csv                    Table S8: hypervolume up to the highest common batch score (written by paper/tools/pareto_frontier.py, Pass 23)
  collapse_common_range.csv              Table S8: collapse rate in the common batch range (written by paper/tools/collapse_common_range.py, Pass 23)
  ranking_pairs.csv, low_signal_pairs.csv  b*, local minus global at b* (the analysis record's value), ranked or low-signal pair
  metric_scores.csv                      local and global batch removal at b*, re-read from the per-fit metrics on the same curves and
                                         checked against the record's local minus global (stops on a difference above 1e-9)
  label_transfer.csv, paper_numbers.json label-transfer table (matched strength, macro F1, distance to b*) and its cross-task means
Outputs: paper/supp/tableS8.tex, paper/supp/note9_table.tex. Run from the repository root: python paper/tools/make_supp_supp-analysis.py
"""
import json
import math
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]          # .../paper
DATA, OUT = ROOT / "figure_data", ROOT / "supp"
PEN = ["discriminator", "reference", "pooled", "barycenter", "mmd", "sinkhorn"]
PEN_LABEL = {"discriminator": "JS discriminator", "reference": "Reference critic", "pooled": "Pooled critic",
             "barycenter": "Barycenter critic", "mmd": "MMD", "sinkhorn": "Sinkhorn", "none": "Plain scVI",
             "scvi_adv": "scvi-tools adversary"}
REAL = ["pancreas", "lung", "immune", "atac_small", "atac_large", "immune_hum_mou"]
SIMS = ["sim1", "sim2"]
TASK_LABEL = {"pancreas": "Pancreas", "lung": "Lung", "immune": "Immune", "atac_small": "ATAC small",
              "atac_large": "ATAC large", "immune_hum_mou": "Human--mouse immune", "sim1": "Simulation 1",
              "sim2": "Simulation 2"}
DEC_LABEL = {1: "batch-conditioned", 0: "unconditioned"}
LOCAL, GLOBAL = ["iLISI", "kBET", "graph_conn"], ["PCR_batch", "ASW_label/batch"]
TOL = 1e-9


def read(name):
    df = pd.read_csv(DATA / name)
    assert len(df), name
    return df


def f3(x):
    return f"{x:.3f}"


def fs(x):
    """A signed value in math mode, so that the minus sign is typeset as one."""
    return f"${x:.3f}$"


def lam(x):
    """A strength as printed in the paper: 0.03, 0.3, 1, 30, 1{,}000."""
    x = float(x)
    return f"{x:g}" if x < 1 else f"{int(round(x)):,}".replace(",", "{,}")


def first_crossing(curve, bstar):
    """(L, G) where the batch curve [(B, L, G), ...] (anchor first) first reaches b*: an exact point within TOL, else linear interpolation
    in B between the first adjacent pair that brackets b* (the reading rule of Supplementary Note 5); None if the curve never reaches it."""
    for i, (b1, l1, g1) in enumerate(curve):
        if abs(b1 - bstar) <= TOL:
            return l1, g1
        if i + 1 < len(curve):
            b2, l2, g2 = curve[i + 1]
            if abs(b2 - bstar) > TOL and (b1 - bstar) * (b2 - bstar) < 0:
                w = (bstar - b1) / (b2 - b1)
                return l1 + w * (l2 - l1), g1 + w * (g2 - g1)
    return None


def local_global(pairs):
    """Local and global batch removal at b* for every row of `pairs`, from the per-fit metrics; checked against the record."""
    ms = read("metric_scores.csv")
    ms = ms[ms.experiment == "A1"].assign(L=lambda d: d[LOCAL].mean(axis=1), G=lambda d: d[GLOBAL].mean(axis=1))
    assert ms[LOCAL + GLOBAL].notna().all().all(), "a batch metric is missing"
    ms = ms.assign(B=(3 * ms.L + 2 * ms.G) / 5)                          # the batch score is the mean of the five metrics
    out = {}
    for _, r in pairs.iterrows():
        g = ms[(ms.task == r.task) & (ms.cond == r.cond)]
        anchor = g[g.arm == "none"][["B", "L", "G"]].mean()
        pts = g[g.arm == r.arm].groupby("lam")[["B", "L", "G"]].mean().sort_index()
        assert len(pts) == r.n_points, (r.task, r.cond, r.arm)
        got = first_crossing([tuple(anchor)] + [tuple(x) for x in pts.to_numpy()], r.bstar)
        assert got is not None, (r.task, r.cond, r.arm)
        assert math.isclose(got[0] - got[1], r.local_minus_global, abs_tol=TOL), (r.task, r.cond, r.arm, got, r.local_minus_global)
        out[(r.task, r.cond, r.arm)] = got
    return out


# ------------------------------------------------------------------ Table S8
def row_s8(s, hv1, cc):
    nd = round(s.nondominated_share * s.nondominated_n)       # secondary.csv: nondominated_n is the number of points
    assert math.isclose(nd, s.nondominated_share * s.nondominated_n, abs_tol=1e-9)
    hv = "n/a" if pd.isna(s.hypervolume) else f3(s.hypervolume)
    assert pd.isna(s.hypervolume) == (hv1 is None), (s.task, s.cond, s.arm)
    hv = ("n/a" if hv1 is None else f3(hv1)) + " & " + hv
    ci = f"{100 * s.collapse_rate:.0f} ({100 * s.collapse_lo:.0f}--{100 * s.collapse_hi:.0f})"
    assert s.diverged == s.nonfinite_latent == 0 and cc.n_all == s.collapse_n and cc.k_all == s.collapsed, (s.task, s.cond, s.arm)
    cic = f"{int(cc.n_common)} & {100 * cc.rate_common:.0f} ({100 * cc.lo_common:.0f}--{100 * cc.hi_common:.0f})"
    assert s.window_status == "ok", (s.task, s.cond, s.arm)
    width = f"{s.window_width_decades:.1f}" + (r"$^\ast$" if s.window_gap else "")
    return (f"{PEN_LABEL[s.arm]} & {hv} & {nd}/{int(s.nondominated_n)} & {int(s.collapse_n)} & {ci} & "
            f"{cic} & {lam(s.window_best)} & {width} & {int(s.window_n_points)} \\\\")


def blocks_s8(sec, pairs, tasks, hv1, cc):
    lines = []
    for cond in (1, 0):
        for t in tasks:
            p = pairs[(pairs.task == t) & (pairs.cond == cond)]
            if p.empty:
                continue
            assert len(p) == 6 and set(p.arm) == set(PEN), (t, cond)
            lines += [r"\midrule", rf"\multicolumn{{11}}{{@{{}}l}}{{\textit{{{TASK_LABEL[t]}, {DEC_LABEL[cond]} decoder; "
                                    rf"$b^*={f3(p.bstar.iloc[0])}$}}}} \\"]
            g = sec[(sec.task == t) & (sec.cond == cond)]
            lines += [row_s8(g[g.arm == a].iloc[0], hv1.get((t, cond, a)), cc.loc[(t, cond, a)]) for a in PEN]
    return "\n".join(lines)


def make_s8():
    sec, rk, low = read("secondary.csv"), read("ranking_pairs.csv"), read("low_signal_pairs.csv")
    assert len(sec) == len(rk) + len(low) == 96
    assert sec.hypervolume.isna().groupby([sec.task, sec.cond]).all().sum() == low.groupby(["task", "cond"]).ngroups
    local_global(pd.concat([rk, low]))  # checks the record's local_minus_global against the per-fit metrics; not printed since Pass 23
    pf = read("pareto_frontier.csv")  # paper/tools/pareto_frontier.py (Pass 23): hypervolume up to b_common
    hv1 = {(r.task, r.cond, r.arm): r.hypervolume_to_common for r in pf.itertuples()}
    assert len(hv1) == sec.hypervolume.notna().sum()
    cc = read("collapse_common_range.csv").set_index(["task", "cond", "arm"])  # paper/tools/collapse_common_range.py (Pass 23)
    assert len(cc) == len(sec)
    n_gap, n_fail = int(sec.window_gap.sum()), int((sec.diverged + sec.nonfinite_latent).sum())
    counts = sorted(sec.collapse_n.unique())
    cap = (r"Secondary results per penalty, task and decoder (definitions in Supplementary Note 7; all descriptive). "
           r"HV: hypervolume up to the highest common batch score (Figure 4), the area of the batch-score by biology-score plane reached "
           r"by the penalty's sweep points beyond the reference point (plain scVI's batch score, plain scVI's biology score minus 0.10), "
           r"with batch scores above the highest common batch score counted at it, divided by the batch gap that sets $b^*$; HV$_{\mathrm{all}}$: "
           r"the same without that cap, so that batch removal beyond the highest common batch score adds area; n/a in low-signal pairs. Non-dom.: strengths that no other strength of the penalty beats on both scores, of all its strengths. "
           rf"Coll.\ all: number of fits $n$ ({', '.join(str(int(c)) for c in counts[:-1])} or {int(counts[-1])}: all strengths times three seeds) and the "
           r"percentage that failed or lost 0.10 or more biology against the same seed's plain scVI, with the exact (Clopper--Pearson) 95\% "
           r"interval, which is nominal because the fits of one seed share one threshold; "
           rf"{n_fail} fits diverged or had a non-finite latent, so every collapse is a loss of biology. $n_{{\mathrm{{c}}}}$ and Coll.\ common: "
           r"the same for the fits at strengths whose mean batch score is at most the highest common batch score (the range of Figure 4 "
           r"and of the main-text collapse rate, Figure 5). Best: the strength with the highest mean selection score; Width: decades spanned by the strengths whose mean "
           r"selection score is at least the lower end of the best strength's 95\% $t$ interval; $^\ast$ a strength inside that span falls "
           rf"below the bound ({n_gap} windows); Good: number of such strengths. \textbf{{Part B}}: low-signal pairs, shown for completeness and left out of every "
           r"summary that uses $b^*$ (Supplementary Note 5).")
    head = (r"\toprule Penalty & HV & HV$_{\mathrm{all}}$ & Non-dom. & $n$ & Coll.\ all \% (95\% CI) & $n_{\mathrm{c}}$ & Coll.\ common \% (95\% CI) & Best & Width & Good \\")
    tex = [r"% generated by paper/tools/make_supp_supp-analysis.py; do not edit",
           r"{\scriptsize\renewcommand{\tablename}{Supplementary Table}\renewcommand{\thetable}{S8}%",
           r"\setlength{\tabcolsep}{3pt}\setlength{\LTcapwidth}{\textwidth}",
           r"\begin{longtable}{@{}lrrrrrrrrrr@{}}",
           rf"\caption{{{cap}}}\label{{tab:s8}}\\", head, r"\midrule", r"\endfirsthead",
           head.replace(r"\toprule", r"\toprule\multicolumn{11}{@{}l}{\textit{Supplementary Table S8, continued}} \\" + "\n"),
           r"\endhead",
           r"\multicolumn{11}{@{}l}{\textbf{Part A: ranked pairs}} \\", blocks_s8(sec, rk, REAL + SIMS, hv1, cc),
           r"\midrule", r"\multicolumn{11}{@{}l}{\textbf{Part B: low-signal pairs (not ranked)}} \\", blocks_s8(sec, low, REAL + SIMS, hv1, cc),
           r"\\ \bottomrule", r"\end{longtable}}",
           r"\addtocounter{table}{-1}% the caption stepped the main table counter; S8 is numbered by hand"]
    (OUT / "tableS8.tex").write_text("\n".join(tex) + "\n")
    return dict(rows=len(sec), gap_windows=n_gap, failed=n_fail, collapse_n=[int(c) for c in counts])


# ------------------------------------------------------------------ Note 9 table
def make_note9():
    lt, rk, low = read("label_transfer.csv"), read("ranking_pairs.csv"), read("low_signal_pairs.csv")
    pn = json.load(open(DATA / "paper_numbers.json"))
    assert lt.computed.all() and not lt.not_evaluable.any()
    bstar = pd.concat([rk, low]).groupby(["task", "cond"]).bstar.first()
    low_pairs = set(map(tuple, low[["task", "cond"]].drop_duplicates().to_numpy()))
    rows = []
    for cond in (1, 0):
        for t in REAL + SIMS:
            g = lt[(lt.task == t) & (lt.cond == cond)].set_index("arm")
            flag = " (low-signal)" if (t, cond) in low_pairs else ""
            rows += [r"\midrule", rf"\multicolumn{{7}}{{@{{}}l}}{{\textit{{{TASK_LABEL[t]}, {DEC_LABEL[cond]} decoder{flag}; "
                                  rf"$b^*={f3(bstar[(t, cond)])}$}}}} \\"]
            for arm in ["none", "scvi_adv"] + PEN:
                if arm not in g.index:
                    assert arm == "scvi_adv" and cond == 0       # the scvi-tools adversary runs with the batch-conditioned decoder only
                    continue
                r = g.loc[arm]
                assert math.isclose(r.distance, abs(r.batch_score - bstar[(t, cond)]), abs_tol=TOL)
                strength = "--" if pd.isna(r.lam) else lam(r.lam)
                rows.append(f"{PEN_LABEL[arm]} & {strength} & {int(r.n_runs)} & {f3(r.f1_mean)} & {f3(r.f1_sd)} & "
                            f"{f3(r.batch_score)} & {f3(r.distance)} \\\\")
    n_real = {c: len(set(rk[rk.cond == c].task) & set(REAL)) for c in (1, 0)}     # real tasks that are not low-signal
    summ = []
    for arm in ["none", "scvi_adv"] + PEN:
        v1, v0 = (pn[f"label_transfer_mean.cond{c}.{arm}"]["value"] for c in (1, 0))
        summ.append(f"{PEN_LABEL[arm]} & & & {f3(v1)} & & & {'--' if v0 is None else f3(v0)} \\\\")
    head = r"\toprule Method & Strength & Seeds & Macro F1 & SD & Batch score & $|B-b^*|$ \\"
    cap = (r"Label transfer at the matched strength, per method, task and decoder (definitions in Supplementary Note 9; descriptive). "
           r"Strength: the matched strength (-- for plain scVI and the scvi-tools adversary, which have none). Macro F1 and SD: mean and "
           r"standard deviation over the seeds of the held-out-batch macro F1. Batch score $B$: seed-mean batch score of the embeddings; "
           r"$|B-b^*|$: its distance from $b^*$. Last block: mean macro F1 over the real tasks that are not low-signal, per decoder.")
    tex = [r"% generated by paper/tools/make_supp_supp-analysis.py; do not edit",
           r"{\scriptsize\renewcommand{\tablename}{Supplementary Table}\renewcommand{\thetable}{S11}\def\theHtable{S11}%",
           r"\setlength{\tabcolsep}{4pt}\setlength{\LTcapwidth}{\textwidth}",
           r"\begin{longtable}{@{}lrrrrrr@{}}",
           rf"\caption{{{cap}}}\label{{tab:s11}}\\",
           head, r"\endfirsthead", r"\multicolumn{7}{@{}l}{\textit{Supplementary Table S11, continued}} \\", head, r"\endhead",
           "\n".join(rows), r"\midrule",
           r"\multicolumn{7}{@{}l}{\textit{Mean macro F1 over the real tasks that are not low-signal}} \\",
           rf"Decoder & & & conditioned ({n_real[1]}) & & & uncond. ({n_real[0]}) \\", "\n".join(summ),
           r"\bottomrule", r"\end{longtable}}",
           r"\addtocounter{table}{-1}% the caption stepped the main table counter; S11 is numbered by hand"]
    (OUT / "note9_table.tex").write_text("\n".join(tex) + "\n")
    return dict(rows=len(lt))


if __name__ == "__main__":  # pragma: no cover  # command-line entry; make_s8 and make_note9 are tested
    OUT.mkdir(exist_ok=True)
    print(make_s8(), make_note9())
    print("wrote", OUT / "tableS8.tex", OUT / "note9_table.tex")
