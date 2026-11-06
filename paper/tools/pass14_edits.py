"""Pass 14 edits (tracked id P14): the replication across training seeds filled in from its result. Edits paper/outline.tex,
paper/supp_items.tex and revision/reviewer_response.md. Every number written comes from paper/replication_data/replication_numbers.json
(made by paper/tools/make_replication.py from the agreement result at commit a475ea3); every old text must occur exactly once or the
script stops. The numbers are re-read from their sources by paper/tools/check_ledger.py through paper/pass14_numbers.csv.
Run from the repository root:  python paper/tools/make_replication.py ; python paper/tools/pass14_edits.py
(PASS14_ROOT overrides the root of the edited files; the numbers file is read from the same root.)
Then: python paper/tools/check_ledger.py ; python paper/tools/abstract_words.py"""
import json, os
from pathlib import Path

ROOT = Path(os.environ.get("PASS14_ROOT") or Path(__file__).resolve().parents[2])
V = {k: x["value"] for k, x in json.loads((ROOT / "paper/replication_data/replication_numbers.json").read_text()).items()}

# the prose below names these cells in words: stop if the result names other cells
assert V["agree"] and not V["before_agree"]
assert V["close_gap_pair"] == V["before_undefined_pair"] == "atac_small|0"   # barycenter - reference, unconditioned decoder
assert (V["unreached_pair"], V["unreached_arm"]) == ("pancreas|1", "mmd")
assert V["neg_pairs"] == "immune_hum_mou|1; pancreas|1" and round(V["tau_min"], 3) == -0.067


def edit(path, pairs):
    p = ROOT / path
    s = p.read_text()
    for old, new in pairs:
        assert s.count(old) == 1, (path, s.count(old), old[:100])
        s = s.replace(old, new)
    p.write_text(s)


def res(old):
    """An \\added[id=P5]{\\res{old}} placeholder and its old text as the struck part of a P14 replacement."""
    return r"\added[id=P5]{\res{" + old + "}}", "{[result: " + old + "]}"


def fill(old, new):
    placeholder, struck = res(old)
    return placeholder, r"\replaced[id=P14]{" + new + "}" + struck


v = V
abstract = (rf"A replication across training seeds kept the sign of all four large within-pair differences (mean rank correlation "
            rf"{v['mean_tau']:.3f}, $p={v['p_pair']:.5f}$).")
b1 = (rf"All {v['n_fits']} replication fits were scored and {v['n_failed']} failed: the {v['n_fits_planned']} planned fits and "
      rf"{v['n_fits_extension']} fits of the extension step, which refitted the next sweep strength once, with seeds 0--4, for the "
      rf"{v['n_extension_readings']} readings whose new-seed means did not bracket $b^*$ (Supplementary Note 11). Plain scVI at seeds "
      rf"0--4 reached batch scores within ${v['plain_diff_min']:.3f}$ to ${v['plain_diff_max']:.3f}$ of the sweep's plain scVI in the "
      rf"13 ranked pairs. Before the extension step {v['n_bracketed_before']} of the {v['n_readings']} readings bracketed $b^*$; after "
      rf"it {v['n_reached']} did. One reading, MMD on the pancreas task with the batch-conditioned decoder, did not replicate the "
      rf"crossing (its highest new-seed mean batch score was {v['unreached_bmax']:.3f}, against $b^*={v['unreached_bstar']:.3f}$) and "
      rf"takes the bottom place in its pair.")
b2 = (rf"Biology at $b^*$ on the new seeds differed from the sweep reading by a median of {v['absdiff_median']:.3f} and at most "
      rf"{v['absdiff_max']:.3f} (absolute differences over the {v['n_reached']} readings that reach $b^*$); the mean absolute difference "
      rf"per pair ranged from {v['pair_mad_min']:.3f} to {v['pair_mad_max']:.3f}. In {v['n_from_plain_passed']} readings the new-seed "
      rf"mean at the lower refitted strength had already reached $b^*$, so the reading interpolates from plain scVI at seeds 0--4. The "
      rf"first-ranked penalty was the same as in the sweep in {v['n_top_same']} of 13 pairs; {v['n_rank_same']} of the "
      rf"{v['n_readings']} penalties kept their sweep rank and {v['n_rank_within1']} moved by at most one place. Supplementary Table S9 "
      rf"gives every reading with the batch and biology scores of its two bracketing points.")
b3 = (rf"Each of the four large sweep gaps kept its sign on the new seeds ({v['n_gaps_kept']} of {v['n_gaps']}); the closest to "
      rf"flipping was the barycenter critic minus the reference critic on the small chromatin-accessibility task with the "
      rf"unconditioned decoder (sweep ${v['close_gap_sweep']:+.3f}$, replication ${v['close_gap_rep']:+.3f}$). Kendall's $\tau_b$ "
      rf"ranged from ${v['tau_min']:.3f}$ to ${v['tau_max']:.3f}$ over the 13 pairs ({v['n_tau_positive']} positive), with mean "
      rf"{v['mean_tau']:.3f} and one-sided sign-flip $p={v['p_pair']:.5f}$ ({v['p_pair_hits']} of the $2^{{13}}$ sign patterns reach the "
      rf"observed mean). The agreement rule is met. Before the extension step that gap could not be read, because the barycenter "
      rf"critic's new-seed mean had not reached $b^*$; the other {v['before_n_gaps_kept']} gaps kept their sign and the mean $\tau_b$ "
      rf"was {v['before_mean_tau']:.3f} ($p={v['before_p']:.5f}$), so the rule was met only once the step it prescribes had been run.")
b4 = (rf"Sensitivity analysis, not part of the rule: with the signs flipped per task, both decoders of a task together (all $2^{{8}}$ "
      rf"patterns over the eight tasks), $p={v['p_task']:.4f}$. Only the observed pattern reaches the observed mean "
      rf"({v['p_task_hits']} of {v['p_task_patterns']}), because the summed $\tau_b$ is positive in all {v['n_tasks_tau_sum_positive']} "
      rf"tasks, so this is the smallest $p$ that eight tasks allow.")
b5 = (rf"Of the {v['n_near_ties']} sweep near-ties (differences under 0.02), {v['near_ties_within']} kept a replication difference "
      rf"within $\pm0.04$ ({v['near_ties_within_before']} before the extension step). With five seeds the replication resolves "
      rf"within-pair differences of about 0.04 (Methods), so near-tied penalties stayed close, but their order is not tested by it. The "
      rf"slope-adjusted reading, a sensitivity analysis whose formula was defined after the rule was fixed (the rule names it without "
      rf"one), kept the sign of {v['n_gaps_kept_slope']} of the {v['n_gaps']} large gaps, with mean $\tau_b$ {v['mean_tau_slope']:.3f} and $p={v['p_slope']:.5f}$.")
d1 = (rf"The replication across training seeds met its agreement rule: the four large within-pair gaps kept their sign, and the sweep "
      rf"and replication orders were positively correlated (mean $\tau_b$ {v['mean_tau']:.3f}, $p={v['p_pair']:.5f}$; "
      rf"$p={v['p_task']:.4f}$ with signs flipped per task). The orders did not hold in detail: the first-ranked penalty changed in "
      rf"{13 - v['n_top_same']} of 13 pairs, $\tau_b$ was slightly negative in {v['n_tau_negative']} pairs (pancreas and the "
      rf"human--mouse immune task, both with the batch-conditioned decoder; $\tau_b={v['tau_min']:.3f}$), and MMD on pancreas with the "
      rf"batch-conditioned decoder did not replicate its crossing. Because the replication refits the same tasks with new seeds, not "
      rf"new data, and resolves differences of about 0.04, it supports the large gaps and the broad order, not the order of "
      rf"near-tied penalties.")

outline = [
    (r"\definechangesauthor[name={Pass 13}, color=orange]{P13}",
     r"\definechangesauthor[name={Pass 14}, color=violet]{P14}% replication across training seeds filled in; change log paper/pass14_changelog.md, numbers in paper/pass14_numbers.csv" + "\n"
     r"\definechangesauthor[name={Pass 13}, color=orange]{P13}"),
    (r"re-read by \texttt{paper/tools/check\_ledger.py}.}",
     r"re-read by \texttt{paper/tools/check\_ledger.py}.}\added[id=P14]{ Pass 14 fills the replication across training seeds from "
     r"its result (change log \texttt{paper/pass14\_changelog.md}, numbers in \texttt{paper/pass14\_numbers.csv}, Supplementary Table "
     r"S9 generated by \texttt{paper/tools/make\_replication.py}).}"),
    fill(r"whether the replication across training seeds agrees with the sweep: the sign of the four large within-pair differences and "
         r"the mean rank correlation over the 13 ranked pairs.", abstract),
    fill(r"fits completed and failed of the 855 replication fits; plain scVI at seeds 0--4 beside the sweep's plain scVI per task and "
         r"decoder; cells whose new-seed means bracket $b^*$, cells that needed the extra outward strength, and cells that did not "
         r"replicate the crossing.", b1),
    fill(r"biology at $b^*$ on the new seeds per penalty, task and decoder beside the sweep reading, with the batch score each "
         r"replication point reached (Supplementary Table S9).", b2),
    fill(r"the sign of each of the four large sweep gaps on the new seeds; Kendall's $\tau_b$ per pair and its mean over the 13 pairs "
         r"with the permutation $p$; whether the agreement rule is met.", b3),
    (r"\added[id=P12]{\res{sensitivity analysis, not part of the rule: the permutation $p$ of the mean $\tau_b$ with signs flipped per "
     r"task, both decoders of a task together.}}",
     r"\replaced[id=P14]{" + b4 + r"}{[result: sensitivity analysis, not part of the rule: the permutation $p$ of the mean $\tau_b$ "
     r"with signs flipped per task, both decoders of a task together.]}"),
    fill(r"share of sweep near-ties (differences under 0.02) whose replication difference stays within $\pm0.04$; the slope-adjusted "
         r"reading as a sensitivity analysis.", b5),
    fill(r"whether the replication across training seeds met the agreement rule, and which of the four large gaps or pair orders did "
         r"not hold on the new seeds.", d1),
    (r"supports a decoder-specific description of the sweep; agreement in the replication across training seeds remains to be assessed.",
     r"supports a decoder-specific description of the sweep; \replaced[id=P14]{the replication across training seeds, on new seeds of "
     r"the same tasks, kept the four large within-pair gaps and agreed with the pair orders in part (Res-2b)}{agreement in the "
     r"replication across training seeds remains to be assessed}."),
    (r"855 fits, 775 with a penalty and 80 of plain scVI; the three low-signal",
     rf"855 fits, 775 with a penalty and 80 of plain scVI\added[id=P14]{{, and {v['n_fits_extension']} fits of the extension step "
     rf"for {v['n_extension_readings']} readings, {v['n_fits']} in all}}; the three low-signal"),
    (r"Replication across training seeds: 855 fits (775 with a penalty, 80 of plain scVI), completed",
     rf"Replication across training seeds: 855 fits (775 with a penalty, 80 of plain scVI)\added[id=P14]{{, plus "
     rf"{v['n_fits_extension']} fits of the extension step (15 on the workstation, 25 on the cluster), {v['n_fits']} in all, "
     rf"{v['n_failed']} failed}}, completed"),
]
edit("paper/outline.tex", outline)

edit("paper/supp_items.tex", [
    ("% supp_items.tex -- Supplementary Note 1, Tables S6-S7 and Figures S1-S6",
     "% supp_items.tex -- Supplementary Note 1, Tables S6, S7 and S9 and Figures S1-S6"),
    ("% Tables S6 and S7 are generated by paper/tools/make_supp_tables.py,",
     "% Tables S6 and S7 are generated by paper/tools/make_supp_tables.py, Table S9 by paper/tools/make_replication.py,"),
    (r"(Note 1, Tables S6--S7, Figures S1--S6)", r"(Note 1, Tables S6, S7 and S9, Figures S1--S6)"),
    ("\\input{supp/tableS7}\n", "\\input{supp/tableS7}\n\\clearpage\n\\input{supp/tableS9}\n"),
])

response = (
    f"**Done (Pass 14, 2026-11-06).** The prediction (the sweep's readings and ranks at $b^*$ in the 13 ranked task and decoder pairs)\n"
    f"and the agreement rule were committed under the tag `replication-prediction-v1` (commit 3a90d0f) before any replication fit was\n"
    f"scored. The replication refits the same eight tasks with new seeds (0--4); it uses no new data. All {v['n_fits']} fits were scored\n"
    f"and {v['n_failed']} failed: the {v['n_fits_planned']} planned fits and {v['n_fits_extension']} fits of the rule's extension step,\n"
    f"which refits the next sweep strength once where the new-seed means do not bracket $b^*$ ({v['n_extension_readings']} readings).\n"
    f"After that step {v['n_reached']} of the {v['n_readings']} readings reach $b^*$; MMD on pancreas with the batch-conditioned\n"
    f"decoder still does not (highest new-seed batch score {v['unreached_bmax']:.3f} against $b^*$ = {v['unreached_bstar']:.3f}) and\n"
    f"takes the bottom place in its pair, marked as not replicating the crossing.\n\n"
    f"- Large gaps: all four sweep differences of 0.04 or more kept their sign ({v['n_gaps_kept']} of {v['n_gaps']}). The closest to\n"
    f"  flipping was the barycenter critic minus the reference critic on the small chromatin-accessibility task with the unconditioned\n"
    f"  decoder (sweep {v['close_gap_sweep']:+.3f}, replication {v['close_gap_rep']:+.3f}).\n"
    f"- Order: Kendall's $\\tau_b$ between the sweep and replication ranks ranged from {v['tau_min']:.3f} to {v['tau_max']:.3f} over\n"
    f"  the 13 pairs ({v['n_tau_positive']} positive); the mean was {v['mean_tau']:.3f}, one-sided sign-flip $p$ = {v['p_pair']:.5f}\n"
    f"  over all $2^{{13}}$ sign patterns. **The agreement rule is met.**\n"
    f"- Before the extension step that gap could not be read (the barycenter critic had not reached $b^*$); the other\n"
    f"  {v['before_n_gaps_kept']} kept their sign and the mean $\\tau_b$ was {v['before_mean_tau']:.3f} ($p$ = {v['before_p']:.5f}).\n"
    f"  The rule was met only once the step it prescribes had been run; both results are reported (Res-2b).\n"
    f"- Sensitivity analyses, not part of the rule and defined after it was fixed: with signs flipped per task (both decoders of a\n"
    f"  task together, $2^{{8}}$ patterns) $p$ = {v['p_task']:.4f}, the smallest value eight tasks allow; the slope-adjusted reading\n"
    f"  kept the sign of {v['n_gaps_kept_slope']} of the {v['n_gaps']} large gaps, mean $\\tau_b$ {v['mean_tau_slope']:.3f}, $p$ = {v['p_slope']:.5f}.\n"
    f"- Near-ties: {v['near_ties_within']} of the {v['n_near_ties']} sweep near-ties (differences under 0.02) stayed within\n"
    f"  $\\pm$0.04 on the new seeds. Five seeds resolve within-pair differences of about 0.04, so the replication does not test the\n"
    f"  order of near-tied penalties; the first-ranked penalty was the same as in the sweep in {v['n_top_same']} of 13 pairs.\n\n"
    f"Where it is reported: abstract (Results), Res-2b, Disc-1 and Supplementary Table S9; the design is in Methods (\"Replication\n"
    f"across training seeds\") and Supplementary Notes 4, 10 and 11.\n")
edit("revision/reviewer_response.md", [
    ("Updated in Pass 12 (2026-11-05) after the Pass 11 rigour review.",
     "Updated in Pass 12 (2026-11-05) after the Pass 11 rigour review. The replication results were added in Pass 14 (2026-11-06;\n"
     "numbers in `paper/pass14_numbers.csv`)."),
    (" Its result had not reached this pass of the response; Res-2b and this section are completed from the agreement test output.",
     " The agreement rule is met (Res-2b; last section of this response)."),
    ("We did not add atlas-scale data (see R3.4).",
     "We did not add atlas-scale data (see R3.4). The replication across training seeds (last section) checks the sweep's readings\n"
     "on new seeds of the same tasks; it adds no new data."),
    ("Pending. When this response was written (2026-11-05, Pass 9), the result of the agreement test (whether the four large sweep gaps "
     "kept their sign, the mean Kendall rank correlation over the 13 ranked pairs and its permutation $p$, and the readings per task) had "
     "not been received. This section, abstract item 7, Res-2b and the Disc-1 bullet will be written from the agreement test output; "
     "until then no replication result is claimed anywhere in the paper or in this response.\n", response),
])
print("pass 14 edits written under", ROOT)
