#!/usr/bin/env python
"""Replication across training seeds: Supplementary Table S9 and the numbers the paper quotes, generated from the committed
agreement result (no typed numbers).

Inputs, read with `git show` from commit a475ea3 (branch sweep-replication) and checked against that folder's MANIFEST.sha256:
  experiments/sweep_replication/agreement_result_with_extension/   final result (895 fits, the rule's extension step included)
  experiments/sweep_replication/agreement_result/                  the result before the extension step (855 fits)
  paper/figure_data/curves.csv                                     the sweep's plain scVI batch score (strength 0) per task and decoder
Outputs: paper/supp/tableS9.tex and paper/replication_data/replication_numbers.json ({key: {description, value}}).
The per-task sign-flip p (a sensitivity analysis added after the agreement rule) is computed here: the mean Kendall tau-b over the 13
ranked pairs with the signs flipped per task, both decoders of a task together, over all 2^8 patterns (exact, no random numbers).
Run from the repository root:  python paper/tools/make_replication.py [--check]   (--check: exit 1 if the committed outputs differ)
"""
import argparse, hashlib, io, itertools, json, subprocess, sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_supp_tables import PEN, PEN_LABEL, TASK_LABEL, DEC_LABEL, REAL, SIMS, f3, fr  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
REV = "a475ea3"
FINAL = "experiments/sweep_replication/agreement_result_with_extension"
BEFORE = "experiments/sweep_replication/agreement_result"
OUT_TEX = REPO / "paper/supp/tableS9.tex"
OUT_JSON = REPO / "paper/replication_data/replication_numbers.json"


def git_file(folder, name, rev=REV):
    """Bytes of a committed result file, checked against the folder's MANIFEST.sha256 at the same commit."""
    def show(path):
        r = subprocess.run(["git", "-C", str(REPO), "show", f"{rev}:{path}"], capture_output=True, check=True)
        return r.stdout
    manifest = dict(line.split()[::-1] for line in show(f"{folder}/MANIFEST.sha256").decode().splitlines() if line.strip())
    data = show(f"{folder}/{name}")
    assert hashlib.sha256(data).hexdigest() == manifest[name], (folder, name)
    return data


def sign_flip_p(values, groups):
    """Exact one-sided sign-flip p of the mean of `values`, flipping the signs of whole groups together:
    (patterns whose mean reaches the observed one, number of patterns)."""
    sums = pd.Series(list(values)).groupby(list(groups)).sum().to_numpy()
    obs = sums.sum()
    hits = sum(sum(s * x for s, x in zip(signs, sums)) >= obs - 1e-12 for signs in itertools.product((1, -1), repeat=len(sums)))
    return hits, 2 ** len(sums)


def lam_str(x):
    return "0" if x == 0 else (f"{int(x):,}".replace(",", "{,}") if x >= 1000 else f"{x:g}")


def pair_order(tk):
    keys = [(t, c) for t in REAL + SIMS for c in (1, 0)]
    have = set(zip(tk.task, tk.cond))
    assert have <= set(keys), have - set(keys)
    return [k for k in keys if k in have]


def numbers(rd, tk, gaps, summ, rd0, summ0, gaps0, curves):
    """Every replication value the paper text quotes, with a description; asserts that what is recomputed equals the result files."""
    assert len(rd) == summ["n_readings"] == 78 and len(tk) == summ["n_pairs"] == 13, (len(rd), len(tk))
    assert summ["n_scored"] == summ["n_manifest"] and summ["n_failed_rows"] == 0 and rd.n_fail.sum() == 0
    assert abs(tk.tau.mean() - summ["mean_tau"]) < 1e-12 and abs(tk.tau_slope_adj.mean() - summ["mean_tau_slope_adj"]) < 1e-12
    hits_pair, n_pair = sign_flip_p(tk.tau, range(len(tk)))
    assert abs(hits_pair / n_pair - summ["p_value"]) < 1e-15, (hits_pair, n_pair, summ["p_value"])  # the rule's test, recomputed
    hits_task, n_task = sign_flip_p(tk.tau, tk.task)
    plain = curves[curves.lam == 0].groupby(["task", "cond"]).B.agg(["min", "max"])
    assert (plain["max"] - plain["min"]).abs().max() < 1e-12  # one plain scVI curve start per task and decoder
    pd_ = tk.set_index(["task", "cond"]).b0_rep - plain["min"].reindex(pd.MultiIndex.from_frame(tk[["task", "cond"]])).to_numpy()
    reached = rd[rd.status == "bracketed"]
    diff = (reached.rep_bio - reached.sweep_bio).abs()
    ext = rd0[rd0.status != "bracketed"][["pair", "arm"]]
    ext_final = ext.merge(rd, on=["pair", "arm"])
    unreached = rd[rd.status != "bracketed"]
    assert len(unreached) == summ["n_unreached"] == 1
    u = unreached.iloc[0]
    close = gaps.loc[gaps.sweep_diff.abs().idxmin()]
    undefined0 = gaps0[~gaps0.defined]
    neg = tk[tk.tau < 0]
    lowest = curves[curves.lam > 0].groupby(["task", "cond", "arm"]).lam.min()
    from_plain = rd[rd.lam_lo == 0]
    passed = [r.lam_hi > lowest[(r.task, r.cond, r.arm)] for _, r in from_plain.iterrows()]  # lower refitted strength already past b*
    V = {
        "n_fits": (summ["n_manifest"], "replication fits in the final result (planned plus extension)"),
        "n_fits_planned": (summ0["n_manifest"], "replication fits planned before the extension step"),
        "n_fits_extension": (summ["n_manifest"] - summ0["n_manifest"], "fits of the rule's extension step"),
        "n_failed": (summ["n_failed_rows"], "failed replication fits"),
        "n_readings": (len(rd), "penalty readings (13 pairs x 6 penalties)"),
        "n_bracketed_before": (int((rd0.status == "bracketed").sum()), "readings bracketing b* before the extension step"),
        "n_extension_readings": (len(ext), "readings that needed the extension step"),
        "n_extension_bracketed": (int((ext_final.status == "bracketed").sum()), "of those, bracketed after it"),
        "n_reached": (len(reached), "readings that reach b* in the final result"),
        "unreached_pair": (u.pair, "the reading that did not replicate the crossing: pair"),
        "unreached_arm": (u.arm, "... penalty"),
        "unreached_bmax": (float(u.B_max), "... highest new-seed mean batch score reached"),
        "unreached_bstar": (float(u.bstar), "... b* of that pair"),
        "plain_diff_min": (float(pd_.min()), "plain scVI batch score, seeds 0-4 minus sweep seeds 100-102, 13 ranked pairs: min"),
        "plain_diff_max": (float(pd_.max()), "... max"),
        "n_from_plain": (len(from_plain), "readings interpolated from plain scVI at seeds 0-4 (lower point strength 0)"),
        "n_from_plain_passed": (int(sum(passed)), "... of which the new-seed mean at the lower refitted strength already reached b*"),
        "absdiff_median": (float(diff.median()), "|replication - sweep| biology at b*, readings that reach b*: median"),
        "absdiff_max": (float(diff.max()), "... max"),
        "pair_mad_min": (float(tk.mean_abs_reading_diff.min()), "per-pair mean |replication - sweep| (pairs with six readings): min"),
        "pair_mad_max": (float(tk.mean_abs_reading_diff.max()), "... max"),
        "n_top_same": (int((tk.top_sweep == tk.top_rep).sum()), "pairs whose first-ranked penalty is the same"),
        "n_rank_same": (int((rd.sweep_rank == rd.rep_rank).sum()), "readings with the same rank in sweep and replication"),
        "n_rank_within1": (int(((rd.sweep_rank - rd.rep_rank).abs() <= 1).sum()), "readings whose rank moved by at most one place"),
        "n_gaps": (len(gaps), "large sweep gaps (>= 0.04)"),
        "n_gaps_kept": (int(gaps.sign_kept.sum()), "large gaps whose sign is kept (raw reading)"),
        "close_gap_pair": (close.pair, "large gap closest to flipping: pair"),
        "close_gap_sweep": (float(close.sweep_diff), "... sweep difference"),
        "close_gap_rep": (float(close.rep_diff), "... replication difference"),
        "tau_min": (float(tk.tau.min()), "per-pair Kendall tau-b: min"),
        "tau_max": (float(tk.tau.max()), "... max"),
        "n_tau_positive": (int((tk.tau > 0).sum()), "pairs with tau-b > 0"),
        "n_tau_negative": (len(neg), "pairs with tau-b < 0"),
        "neg_pairs": ("; ".join(neg.pair), "pairs with tau-b < 0"),
        "mean_tau": (summ["mean_tau"], "mean tau-b over the 13 pairs"),
        "p_pair": (hits_pair / n_pair, "one-sided sign-flip p over pairs (the rule's test)"),
        "p_pair_hits": (hits_pair, "patterns reaching the observed mean, over pairs"),
        "p_pair_patterns": (n_pair, "sign patterns over pairs"),
        "p_task": (hits_task / n_task, "sensitivity: one-sided sign-flip p, signs flipped per task (both decoders together)"),
        "p_task_hits": (hits_task, "patterns reaching the observed mean, per task"),
        "p_task_patterns": (n_task, "sign patterns over tasks"),
        "n_tasks_tau_sum_positive": (int((tk.groupby("task").tau.sum() > 0).sum()), "tasks whose summed tau-b is positive"),
        "agree": (bool(summ["agree"]), "agreement rule met (final)"),
        "n_near_ties": (summ["n_near_ties"], "sweep near-ties (difference < 0.02)"),
        "near_ties_within": (summ["near_ties_within_0p04"], "near-ties whose replication difference stays within +/-0.04"),
        "near_ties_within_before": (summ0["near_ties_within_0p04"], "... before the extension step"),
        "n_gaps_kept_slope": (int(gaps.sign_kept_slope_adj.sum()), "large gaps kept, slope-adjusted reading"),
        "mean_tau_slope": (summ["mean_tau_slope_adj"], "mean tau-b, slope-adjusted reading"),
        "p_slope": (summ["p_value_slope_adj"], "sign-flip p over pairs, slope-adjusted reading"),
        "before_n_gaps_defined": (int(gaps0.defined.sum()), "before the extension step: large gaps with both readings"),
        "before_n_gaps_kept": (int(gaps0.sign_kept.sum()), "... kept"),
        "before_undefined_pair": ("; ".join(undefined0.pair), "... the pair of the gap that could not be read"),
        "before_mean_tau": (summ0["mean_tau"], "before the extension step: mean tau-b"),
        "before_p": (summ0["p_value"], "before the extension step: sign-flip p over pairs"),
        "before_agree": (bool(summ0["agree"]), "before the extension step: rule met as computed"),
    }
    out = {}
    for k, (v, d) in V.items():
        out[k] = {"description": d, "value": v.item() if hasattr(v, "item") else v}
    return out


def label(pair):
    t, c = pair.split("|")
    return f"{TASK_LABEL[t]}, {DEC_LABEL[int(c)]} decoder"


def row_s9(r, ext):
    """One penalty row: strengths, batch and biology at the two bracketing points, readings and ranks."""
    note = ("extension" if ext else "") + ("; not reached$^\\dagger$" if r.status != "bracketed" else "")
    rank = f"{fr(r.rep_rank)} ({fr(r.sweep_rank)})"
    if r.status != "bracketed":
        return (rf"{PEN_LABEL[r.arm]} & up to {lam_str(r.lam_max)} & {f3(r.B_max)} (max) & -- & -- & {f3(r.sweep_bio)} & "
                rf"{rank} & {note.lstrip('; ')} \\")
    return (rf"{PEN_LABEL[r.arm]} & {lam_str(r.lam_lo)}--{lam_str(r.lam_hi)} & {f3(r.B_lo)}/{f3(r.B_hi)} & "
            rf"{f3(r.C_lo)}/{f3(r.C_hi)} & {f3(r.rep_bio)} & {f3(r.sweep_bio)} & {rank} & {note} \\")


def table_s9(rd, tk, gaps, rd0, curves, V):
    ext = set(zip(rd0[rd0.status != "bracketed"].pair, rd0[rd0.status != "bracketed"].arm))
    plain = curves[curves.lam == 0].groupby(["task", "cond"]).B.min()
    blocks = []
    for t, c in pair_order(tk):
        p = f"{t}|{c}"
        k = tk[tk.pair == p].iloc[0]
        head = (rf"\multicolumn{{8}}{{@{{}}l}}{{\textit{{{label(p)}; $b^*={f3(k.bstar)}$; plain scVI batch score "
                rf"{f3(k.b0_rep)} (sweep {f3(plain[(t, c)])}); $\tau_b={k.tau:.3f}$}}}} \\")
        rows = [row_s9(r, (r.pair, r.arm) in ext) for _, r in rd[rd.pair == p].set_index("arm").loc[PEN].reset_index().iterrows()]
        blocks.append("\\midrule\n" + head + "\n" + "\n".join(rows))
    g_rows = [rf"\multicolumn{{2}}{{@{{}}l}}{{{label(g.pair)}}} & \multicolumn{{2}}{{l}}{{{PEN_LABEL[g.a]} $-$ {PEN_LABEL[g.b]}}} & "
              rf"${g.sweep_diff:+.3f}$ & ${g.rep_diff:+.3f}$ & {'yes' if g.sign_kept else 'no'} & \\" for _, g in gaps.iterrows()]
    v = {k: x["value"] for k, x in V.items()}
    cap = (rf"Replication across training seeds: biology at the matched batch removal $b^*$ (fixed at its sweep value) on new seeds "
           rf"0--4 beside the sweep reading (seeds 100--102), per penalty, task and decoder, for the {len(tk)} ranked pairs. "
           rf"``Strengths'' are the two bracketing strengths (0 is plain scVI at seeds 0--4); ``Batch'' and ``Biology'' give the seed-mean "
           rf"scores at those two points; the reading interpolates linearly in batch score between them at the first crossing of $b^*$, "
           rf"walking up from plain scVI at seeds 0--4 (in {v['n_from_plain_passed']} readings the new-seed mean at the lower refitted strength "
           rf"already reached $b^*$, so the reading starts at plain scVI). "
           rf"Rank: replication (sweep); 1 is the highest biology; a penalty that does not reach $b^*$ takes the bottom place. "
           rf"``extension'': the new-seed means did not bracket $b^*$ at the planned strengths, so the next sweep strength was fitted "
           rf"once with seeds 0--4, as the agreement rule prescribes ({v['n_extension_readings']} readings, {v['n_fits_extension']} fits). "
           rf"$^\dagger$Did not replicate the crossing: the highest new-seed mean batch score is shown. Each block header gives $b^*$, "
           rf"plain scVI's batch score at seeds 0--4 beside the sweep's and Kendall's $\tau_b$ between the sweep and replication ranks. "
           rf"All {v['n_fits']} fits were scored and {v['n_failed']} failed. The last block gives the four large sweep gaps "
           rf"(0.04 or more within a pair) and their sign on the new seeds (Supplementary Notes 4 and 10).")
    tex = [r"% generated by paper/tools/make_replication.py from the agreement result at commit " + REV + "; do not edit",
           r"{\scriptsize\renewcommand{\tablename}{Supplementary Table}\renewcommand{\thetable}{S9}%",
           r"\setlength{\tabcolsep}{3pt}",
           r"\begin{longtable}{@{}lllllrll@{}}",
           rf"\caption{{{cap}}}\label{{tab:s9}}\\",
           r"\toprule Penalty & Strengths & Batch & Biology & At $b^*$, new & At $b^*$, sweep & Rank & Note \\",
           "\n".join(blocks),
           r"\midrule",
           r"\multicolumn{8}{@{}l}{\textbf{Large sweep gaps}} \\",
           r"\multicolumn{2}{@{}l}{Pair} & \multicolumn{2}{l}{Difference} & Sweep & New & Sign kept & \\",
           "\n".join(g_rows),
           r"\bottomrule",
           r"\end{longtable}}"]
    return "\n".join(tex) + "\n"


def build():
    csv = lambda folder, name: pd.read_csv(io.BytesIO(git_file(folder, name)))  # noqa: E731
    js = lambda folder: json.loads(git_file(folder, "replication_agreement_summary.json"))  # noqa: E731
    rd, tk, gaps = csv(FINAL, "replication_readings.csv"), csv(FINAL, "replication_task_readings.csv"), csv(FINAL, "replication_large_gaps.csv")
    rd0, gaps0 = csv(BEFORE, "replication_readings.csv"), csv(BEFORE, "replication_large_gaps.csv")
    curves = pd.read_csv(REPO / "paper/figure_data/curves.csv")
    V = numbers(rd, tk, gaps, js(FINAL), rd0, js(BEFORE), gaps0, curves)
    return table_s9(rd, tk, gaps, rd0, curves, V), json.dumps(V, indent=1) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--check", action="store_true", help="compare with the committed outputs instead of writing them")
    a = ap.parse_args(argv)
    tex, js = build()
    if a.check:
        same = OUT_TEX.read_text() == tex and OUT_JSON.read_text() == js
        print("Table S9 and replication_numbers.json", "reproduced" if same else "DIFFER from the committed files")
        return 0 if same else 1
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_TEX.write_text(tex)
    OUT_JSON.write_text(js)
    v = json.loads(js)
    print(f"wrote {OUT_TEX.name} and {OUT_JSON.name}; per-task sign-flip p = {v['p_task']['value']} "
          f"({v['p_task_hits']['value']} of {v['p_task_patterns']['value']} patterns)")
    return 0


if __name__ == "__main__":  # pragma: no cover  # command-line entry; main() is tested
    sys.exit(main())
