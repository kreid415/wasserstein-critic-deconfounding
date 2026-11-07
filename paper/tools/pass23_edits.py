"""Pass 23 edits (2026-11-07, tracked id P23): decisions from the user's walkthrough of the Results, one result at a time.
Res-1: the fit-completion paragraph moves to the supplement (Supplementary Notes 5, 8 and 11 and Table S5 already hold every
number in it) and one sentence stays in the Results. Res-2: an opening sentence states that the order depends on the
decoder; the three worked examples of close readings become counts from paper/tools/res2_seed_noise.py
(paper/figure_data/res2_seed_noise.json); the simulation orders move to Supplementary Figure S1, which shows them.
Res-2b: the replication paragraph moves to a new Supplementary Note 13 (written by this script, with the replication mean
ranks of paper/tools/replication_mean_ranks.py added); one paragraph stays in the Results. Abstract, Disc-1 and Conclusions: the sweep leaders stay and a clause says that
the leaders changed on the replication's new seeds (user decision 2026-11-07).
Table 2 (ranking) repeats Figure 2b and Supplementary Table S6, so it moves to Supplementary Table S12 (user rule: no
repeated information in the main text); the label tab:3 moves with it. Figure 2: the caption follows the redrawn figure (panel a has tasks as rows
and the decoders stacked; panel b stacks the decoders, sorts penalties best first and reverses the rank axis).
Each edited line is rewritten as word-level tracked changes against its printed (accepted) old text, as in Pass 22. Every old
line must occur exactly once or the script stops.
Run from the repository root:  python paper/tools/pass23_edits.py   (PASS23_ROOT overrides the root of the edited files)
Then: python paper/tools/accept_changes.py ; python paper/tools/check_ledger.py"""
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pass22_edits as p22  # noqa: E402
from accept_changes import accept, group  # noqa: E402

ROOT = Path(os.environ.get("PASS23_ROOT") or Path(__file__).resolve().parents[2])
p22.TAG = "P23"  # rewrite() marks its changes with the module's TAG

EDITS = [  # (old source line, new printed text)
    ('All 3,033 sweep fits completed, with 0 failed fits across the penalties and decoders (Supplementary Table S5). The analysis includes 387 fits on the immune task and 378 on each other task; the immune total includes additional strengths for the discriminator, the reference critic and the pooled critic with the batch-conditioned decoder. Of the 448 baseline fits, 447 were scored and 0 failed: the scvi-tools adversary 40 of 40, scANVI 40 of 40, sysVI 240 of 240, Harmony 56 of 56, Scanorama 55 of 56 and PCA 16 of 16. The remaining fit, Scanorama with 80 neighbours on the large chromatin-accessibility task, could not be fitted (it ran out of memory, and a retry stopped at a 32-bit sparse-matrix index limit); it is reported as an incomplete setting, not as a failed fit, so on that task Scanorama is placed with five of its six settings (Supplementary Table S5). The low-signal pairs were the immune task and both simulations with the batch-conditioned decoder: plain scVI and the highest common batch scores were 0.638 and 0.657 (gap 0.019) for the immune task; 0.812 and 0.819 (gap 0.007) for simulation 1; 0.656 and 0.665 (gap 0.009) for simulation 2. Every penalty reached $b^*$ in every task and decoder pair, including the low-signal \\replaced[id=P22]{pairs, so}{pairs;} none received a bottom rank for a missing full-seed \\replaced[id=P22]{reading; low-signal}{reading. Low-signal} pairs are reported separately under the Methods rule (Supplementary Note 5).',
     'All 3,033 sweep fits and 447 of 448 baseline fits were scored, and every penalty reached $b^*$ in every task and decoder pair; the immune task and both simulations were low-signal with the batch-conditioned decoder (Supplementary Notes 5, 8 and 11; Supplementary Table S5).'),
    ('\\caption{\\replaced[id=P10]{Ranking of the six penalties by biology kept at matched batch removal $b^*$. (a) Biology score at $b^*$ (colour) and rank among the six penalties within each task column (number; tied penalties share the average rank) for each real task, with the batch-conditioned and the unconditioned decoder. Hatched cells: the immune task with the batch-conditioned decoder, a low-signal pair that is not ranked. A penalty whose curve does not reach $b^*$ would take the bottom rank; every curve reached it here. (b) Mean rank over the ranked real tasks (five with the batch-conditioned decoder, six with the unconditioned decoder) per penalty and decoder; rank 1 is best; bars span the lowest and highest rank over those tasks. The pooled critic had the best mean rank with the batch-conditioned decoder (2.2) and MMD with the unconditioned decoder (2.17). The leave-one-seed-out standard error of each reading in (a) and the range of each rank over the three leave-one-seed-out repeats are in Supplementary Table S6. Simulations are ranked separately (Supplementary Figure S1).}{MOCK -- synthetic data. Ranking of the six penalties by biology kept at matched batch removal $b^*$. (a) Biology score at $b^*$ (colour) and rank within the column (number) for each real task, conditioned and unconditioned decoder; hatched cells are low-signal pairs and are not ranked; a penalty whose curve does not reach $b^*$ takes the bottom rank, and tied penalties share the average rank. (b) Mean rank over the real tasks that are not low-signal per penalty and decoder (rank 1 is best); bars, range over tasks. The leave-one-seed-out standard error of each reading in (a) and the range of each rank over the five leave-one-seed-out repeats are in Supplementary Table S6. Simulations are ranked separately (Supplementary Figure S1).}}',
     '\\caption{Ranking of the six penalties by biology kept at matched batch removal $b^*$. (a) Biology score at $b^*$ (colour) and rank among the six penalties within each task row (number; tied penalties share the average rank) for each real task, with the batch-conditioned decoder (upper block) and the unconditioned decoder (lower block). Hatched row: the immune task with the batch-conditioned decoder, a low-signal pair that is not ranked. A penalty whose curve does not reach $b^*$ would take the bottom rank; every curve reached it here. (b) Mean rank over the ranked real tasks (five with the batch-conditioned decoder, upper block; six with the unconditioned decoder, lower block), with the penalties of each decoder sorted from best to worst mean rank (the value after each name, rounded to one decimal); dot, mean rank; bar, lowest to highest rank over those tasks; the rank axis runs from 6 at the left to 1 (best) at the right. The pooled critic had the best mean rank with the batch-conditioned decoder (2.20) and MMD with the unconditioned decoder (2.17). The leave-one-seed-out standard error of each reading in (a) and the range of each rank over the three leave-one-seed-out repeats are in Supplementary Table S6. Simulations are ranked separately (Supplementary Figure S1).}'),
    ("\\replaced[id=P7]{With the batch-conditioned decoder, the ranking rested on five real tasks. Mean ranks (range across those tasks) were the pooled critic 2.20 (1--3); the Sinkhorn divergence 2.40 (1--5); the barycenter critic 3.00 (1--5); the reference critic 3.80 (1--5); MMD 4.60 (2--6); the discriminator 5.00 (3--6). With the unconditioned decoder, all six real tasks contributed: MMD 2.17 (1--6); the reference critic 3.33 (1--6); the pooled critic 3.33 (2--5); the discriminator 3.50 (1--5); the barycenter critic 4.00 (1--6); the Sinkhorn divergence 4.67 (3--6). The Sinkhorn divergence ranked first most often with the batch-conditioned decoder, on the human--mouse immune and lung tasks; the discriminator ranked last on the small chromatin-accessibility, lung and pancreas tasks. With the unconditioned decoder, MMD ranked first most often, on the large chromatin-accessibility, human--mouse immune and pancreas tasks; the reference critic ranked last on the small chromatin-accessibility and human--mouse immune tasks, and the Sinkhorn divergence on the large chromatin-accessibility and lung tasks.}{Rule: for each task and decoder, biology at $b^*$ is read where a penalty's curve first reaches $b^*$, interpolating between neighbouring strengths, with the standard error of the reading and the range of its rank over repeats with each of the three seeds left out; the six penalties are ranked by it (rank 1 is best; ties share the average rank; a penalty whose curve does not reach $b^*$ takes the bottom rank), and the mean rank over the real tasks that are not low-signal is reported per decoder (Methods; Supplementary Notes 5 and 6). [result: mean rank per penalty and decoder with its range over tasks; how many real tasks each decoder's ranking rests on; which penalties are first or last in most tasks.] [result: biology at $b^*$ per penalty, task and decoder with leave-one-seed-out standard errors and rank ranges (Supplementary Table S6); whether the readings of neighbouring ranks lie within one standard error of each other; low-signal pairs shown separately and pairs where a curve never reaches $b^*$ marked; simulations ranked per task (Supplementary Figure S1).] Descriptive: no significance test or decision rule; sensitivity of the ranking to the reading rules in Supplementary Figure S2.}",
     'The order of the penalties depended on the decoder: MMD had the fifth-best mean rank with the batch-conditioned decoder and the best with the unconditioned decoder, and the Sinkhorn divergence the second-best and the worst. With the batch-conditioned decoder, the ranking rested on five real tasks. Mean ranks (range across those tasks) were the pooled critic 2.20 (1--3); the Sinkhorn divergence 2.40 (1--5); the barycenter critic 3.00 (1--5); the reference critic 3.80 (1--5); MMD 4.60 (2--6); the discriminator 5.00 (3--6). With the unconditioned decoder, all six real tasks contributed: MMD 2.17 (1--6); the reference critic 3.33 (1--6); the pooled critic 3.33 (2--5); the discriminator 3.50 (1--5); the barycenter critic 4.00 (1--6); the Sinkhorn divergence 4.67 (3--6). The Sinkhorn divergence ranked first most often with the batch-conditioned decoder, on the human--mouse immune and lung tasks; the discriminator ranked last on the small chromatin-accessibility, lung and pancreas tasks. With the unconditioned decoder, MMD ranked first most often, on the large chromatin-accessibility, human--mouse immune and pancreas tasks; the reference critic ranked last on the small chromatin-accessibility and human--mouse immune tasks, and the Sinkhorn divergence on the large chromatin-accessibility and lung tasks.'),
    ('\\added[id=P7]{Neighbouring biology readings often differed by less than at least one of their reported leave-one-seed-out standard errors (Methods; Supplementary Table S6). For example, with the batch-conditioned decoder on the small chromatin-accessibility task, the pooled critic and the Sinkhorn divergence read 0.632 and 0.631, with standard errors 0.007 and 0.004, respectively. With the unconditioned decoder on the large chromatin-accessibility task, MMD and the reference critic read 0.622 and 0.621, with standard errors 0.010 and 0.006. Some gaps exceeded both neighbouring standard errors: on the small chromatin-accessibility task with the batch-conditioned decoder, MMD exceeded the reference critic by 0.017 (standard errors 0.007 and 0.003). The simulations were reported separately (Supplementary Figure S1): with the unconditioned decoder, the order from highest to lowest biology was the discriminator, the reference critic, MMD, the Sinkhorn divergence, the pooled critic and the barycenter critic in simulation 1, and the discriminator, the pooled critic, MMD, the reference critic, the barycenter critic and the Sinkhorn divergence in simulation 2. Neither simulation was ranked with the batch-conditioned decoder because both pairs were low-signal.}',
     "Most neighbouring readings were close. In 44 of the 55 pairs of penalties next to each other in a task's order (17 of 25 with the batch-conditioned decoder, 27 of 30 with the unconditioned decoder), the two biology readings differed by less than the larger of their leave-one-seed-out standard errors, and 2 of the 11 first places stayed first in all three leave-one-seed-out repeats (Methods; Supplementary Table S6). The simulations are ranked with the unconditioned decoder only, because both pairs with the batch-conditioned decoder were low-signal (Supplementary Figure S1)."),
    ('\\item \\replaced[id=P14]{A replication across training seeds kept the sign of all four large within-pair differences (mean rank correlation 0.456, $p=0.00049$).}{[result: whether the replication across training seeds agrees with the sweep: the sign of the four large within-pair differences and the mean rank correlation over the 13 ranked pairs.]}',
     '\\item A replication across training seeds kept the sign of all four large within-pair differences (mean rank correlation 0.456, $p=0.00049$), but on new seeds the Sinkhorn divergence and the pooled critic led, and the discriminator stayed last with the batch-conditioned decoder.'),
    ('\\pp{Disc-1}{The penalty with the best mean rank depended on the decoder: the pooled critic \\deleted[id=P22]{ranked highest} with the batch-conditioned decoder, \\added[id=P22]{closely} followed \\deleted[id=P22]{closely} by the Sinkhorn divergence, \\replaced[id=P22]{and}{whereas} MMD \\deleted[id=P22]{ranked highest} with the unconditioned decoder. \\replaced[id=P22]{Task-level}{The task-level} orders varied, and many neighbouring biology readings were within a leave-one-seed-out standard \\replaced[id=P22]{error, so these}{error. These} descriptive rankings \\deleted[id=P22]{therefore} do not establish a consistent ordering across tasks. The secondary summaries \\deleted[id=P22]{also} gave different \\replaced[id=P22]{orders again:}{orders:} MMD had relatively low collapse rates despite ranking near the bottom with the batch-conditioned decoder, and the local-versus-global readings \\replaced[id=P22]{favoured no single}{did not give a consistent advantage to one} penalty. This exploratory comparison on eight tasks with three seeds (100--102) supports a decoder-specific description of the sweep; the replication across training seeds, on new seeds of the same tasks, kept the four large within-pair gaps and agreed with the pair orders in part (Res-2b).}',
     '\\pp{Disc-1}{The penalty with the best mean rank depended on the decoder: the pooled critic  with the batch-conditioned decoder, closely followed  by the Sinkhorn divergence, and MMD  with the unconditioned decoder. Task-level orders varied, and many neighbouring biology readings were within a leave-one-seed-out standard error, so these descriptive rankings  do not establish a consistent ordering across tasks. The secondary summaries  gave different orders again: MMD had relatively low collapse rates despite ranking near the bottom with the batch-conditioned decoder, and the local-versus-global readings favoured no single penalty. This exploratory comparison on eight tasks with three seeds (100--102) supports a decoder-specific description of the sweep; the replication across training seeds, on new seeds of the same tasks, kept the four large within-pair gaps and agreed with the pair orders in part, but the leaders changed: on the new seeds the Sinkhorn divergence had the best mean rank with the batch-conditioned decoder and the pooled critic with the unconditioned decoder, while the discriminator stayed last with the batch-conditioned decoder (Res-2b; Supplementary Note 13).}'),
    ("\\item \\added[id=P3]{On eight tasks of the single-cell integration benchmark, six alignment penalties added to one scVI model kept similar biology at matched batch removal: the best mean rank was the pooled critic's with the batch-conditioned decoder (2.20) and MMD's with the unconditioned decoder (2.17), but 160 of the 195 within-pair differences were under 0.02, so the ranking does not separate most penalties. At equal numbers of tries with the batch-conditioned decoder, the penalties exceeded Harmony, Scanorama and sysVI on most real tasks and plain scVI mainly when several strengths were tried\\added[id=P12]{ and scored on the same data}; scANVI, which uses cell-type labels, was exceeded only on pancreas.}",
     "\\item On eight tasks of the single-cell integration benchmark, six alignment penalties added to one scVI model kept similar biology at matched batch removal: the best mean rank was the pooled critic's with the batch-conditioned decoder (2.20) and MMD's with the unconditioned decoder (2.17), but 160 of the 195 within-pair differences were under 0.02, so the ranking does not separate most penalties, and on new training seeds the Sinkhorn divergence and the pooled critic led instead. At equal numbers of tries with the batch-conditioned decoder, the penalties exceeded Harmony, Scanorama and sysVI on most real tasks and plain scVI mainly when several strengths were tried and scored on the same data; scANVI, which uses cell-type labels, was exceeded only on pancreas."),
    ('\\pp{Res-2}{\\replaced{Penalties are ranked by the biology they keep at matched batch removal (Figure 2, Table \\ref{tab:3}).}{The premise check reports how separable batches are before integration and how adversary gradients behave in training (Figure 2).}}',
     '\\pp{Res-2}{Penalties are ranked by the biology they keep at matched batch removal (Figure 2; Supplementary Table \\ref{tab:3}).}'),
    ('\\item Table \\ref{tab:3}: \\replaced{benchmark ranking: mean rank per penalty and decoder over the real tasks that are not low-signal, range over tasks, \\added{mean biology at $b^*$, }simulation ranks, tasks not ranked, tasks ranked.}{primary results (primary comparison, trade-off area, collapse rate, per decoder): estimate, interval, probabilities, call, evaluable tasks.}',
     None),
]


def rewrite(old, new):
    """Captions are tracked inside \\caption{} so the caption command itself is never struck."""
    if new and new.startswith("\\caption{"):
        inner_old, _ = group(accept(old).strip(), len("\\caption"))
        inner_new, _ = group(new, len("\\caption"))
        return "\\caption{" + p22.tracked(inner_old, inner_new) + "}"
    return p22.rewrite(old, new)


NAMES = [("discriminator", "the discriminator"), ("reference", "the reference critic"), ("pooled", "the pooled critic"),
         ("barycenter", "the barycenter critic"), ("mmd", "MMD"), ("sinkhorn", "the Sinkhorn divergence")]
NOTE13_HEADS = ["Rule", "Fits", "Readings", "Agreement", "Sensitivity analyses", "Near-ties and slope-adjusted reading"]


def res2b_block(s):
    """The Res-2b heading line and its six \\item lines (the replication paragraph moved to Supplementary Note 13)."""
    lines = s.split("\n")
    i = [k for k, l in enumerate(lines) if l.startswith(r"\pp{\added[id=P5]{Res-2b}")]
    assert len(i) == 1 and lines[i[0] + 1] == r"\begin{itemize}" and lines[i[0] + 8] == r"\end{itemize}"
    items = lines[i[0] + 2:i[0] + 8]
    assert all(l.startswith(r"\item ") for l in items)
    return lines[i[0]], items


def note13(items, rep, mr):
    texts = [accept(l)[len(r"\item "):].strip() for l in items]
    sweep_first = {d: min((a for a, _ in NAMES), key=lambda a: mr[d][a]["sweep"]) for d in mr}
    assert sweep_first == {"batch_conditioned": "pooled", "unconditioned": "mmd"}  # named in the sentence below
    assert texts[0].startswith("Rule: ") and texts[0].count("in this paragraph") == 1
    texts[0] = "The " + texts[0][len("Rule: the "):].replace("in this paragraph", "in this note")
    order = lambda d: ", ".join(f"{name} {mr[d][a]['replication']:.2f} ({mr[d][a]['sweep']:.2f})" for a, name in NAMES)
    ranks = (f"Mean ranks over the ranked real tasks on the new seeds, recomputed from the replication ranks of Supplementary "
             f"Table~S9 (sweep mean ranks in brackets), were, with the batch-conditioned decoder ({mr['batch_conditioned']['n_tasks']}"
             f" tasks), {order('batch_conditioned')}; with the unconditioned decoder ({mr['unconditioned']['n_tasks']} tasks), "
             f"{order('unconditioned')}. The penalty with the best mean rank changed with both decoders: "
             f"{dict(NAMES)[mr['batch_conditioned']['replication_first']]} replaced the pooled critic with the batch-conditioned "
             f"decoder, and {dict(NAMES)[mr['unconditioned']['replication_first']]} replaced MMD with the unconditioned decoder. "
             "These mean ranks are descriptive, are not part of the agreement rule and were computed after the replication had "
             "been read (\\texttt{paper/tools/replication\\_mean\\_ranks.py}).")
    body = "\n\n".join(rf"\noindent\textit{{{h}.}} {t}" for h, t in zip(NOTE13_HEADS + ["Mean ranks on the new seeds"],
                                                                         texts + [ranks]))
    return ("% Supplementary Note 13: replication across training seeds. Written by paper/tools/pass23_edits.py from the Res-2b\n"
            "% paragraph of the main text (moved here in Pass 23) and paper/figure_data/replication_mean_ranks.json; do not edit.\n"
            "\\medskip\\noindent\\textbf{Supplementary Note 13. Replication across training seeds.}\\par\\smallskip\n"
            "\\begingroup\\small\\setlength{\\parskip}{3pt}\n" + body + "\n\\par\n\\endgroup\n")


def res2b_line(rep, mr):
    first = {d: dict(NAMES)[mr[d]["replication_first"]] for d in mr}
    return (f"A replication on {rep['n_fits']} new fits (seeds 0--4) met its agreement rule only after the extension step that the "
            f"rule prescribes: all four large sweep gaps kept their sign, and the sweep and replication orders were "
            f"positively correlated (mean Kendall's $\\tau_b$ {rep['mean_tau']:.3f}, $p={rep['p_pair']:.5f}$). The first-ranked "
            f"penalty changed in {rep['n_readings'] // 6 - rep['n_top_same']} of {rep['n_readings'] // 6} pairs, and by mean rank "
            f"on the new seeds {first['batch_conditioned']} led with the batch-conditioned decoder "
            f"({mr['batch_conditioned'][mr['batch_conditioned']['replication_first']]['replication']:.2f}) and "
            f"{first['unconditioned']} with the unconditioned decoder "
            f"({mr['unconditioned'][mr['unconditioned']['replication_first']]['replication']:.2f}) (Supplementary Note 13, "
            f"Supplementary Table S9).")


def move_res2b(s):
    rep = {k: v["value"] if isinstance(v, dict) and "value" in v else v for k, v in
           json.loads((ROOT / "paper/replication_data/replication_numbers.json").read_text()).items()}
    mr = json.loads((ROOT / "paper/figure_data/replication_mean_ranks.json").read_text())
    assert rep["n_gaps_kept"] == rep["n_gaps"] == 4 and rep["agree"] and not rep["before_agree"]
    head, items = res2b_block(s)
    (ROOT / "paper/supp/note13.tex").write_text(note13(items, rep, mr))
    old = "\n".join([head, r"\begin{itemize}"] + items) + "\n"
    new = "\n".join([head, rf"\added[id={p22.TAG}]{{{res2b_line(rep, mr)}}}", r"\begin{itemize}"] +
                    [p22.rewrite(l, None) for l in items]) + "\n"
    assert s.count(old) == 1
    s = s.replace(old, new)
    note12 = "\\Needspace{5\\baselineskip}\\input{supp/note12}\\par\n"
    assert s.count(note12) == 1
    s = s.replace(note12, note12 + "\\Needspace{5\\baselineskip}\\input{supp/note13}\\par\n")
    hd = [l for l in s.split("\n") if l.startswith(r"\hd{Supplementary material}")]
    assert len(hd) == 1 and accept(hd[0]).count("Notes 1--12,") == 1
    return s.replace(hd[0], accept(hd[0]).replace("Notes 1--12,", rf"Notes 1--\replaced[id={p22.TAG}]{{13}}{{12}},"))


T2_START = "% ---------------------------------------------------------------- Table 3\n"
T2_END = "% ---------------------------------------------------------------- Table 4\n"


def edit(path, pairs):
    p = ROOT / path
    s = p.read_text()
    for old, new in pairs:
        assert s.count(old) == 1, (path, s.count(old), old[:100])
        s = s.replace(old, new)
    p.write_text(s)


def move_table2(s):
    """Main-text Table 2 (label tab:3, the ranking) repeats Figure 2b and Supplementary Table S6, so it moves to the
    supplement as Supplementary Table S12; the label tab:3 moves with it, so every \\ref follows."""
    i, j = s.index(T2_START), s.index(T2_END)
    block = s[i:j]
    b, e = block.index("\\begin{table}[t]"), block.index("\\end{table}") + len("\\end{table}")
    table = accept(block[b:e])
    for old, new in [("& conditioned &", "& batch-conditioned &"), ("with the conditioned decoder (--)", "with the batch-conditioned decoder (--)")]:
        assert old in table, old
        table = table.replace(old, new)
    assert table.count("\\caption{") == 1 and table.count("\\label{tab:3}") == 1
    table = table.replace("\\centering\\small\n", "\\centering\\scriptsize\\renewcommand{\\tablename}{Supplementary Table}"
                          "\\renewcommand{\\thetable}{S12}%\n", 1)
    assert "renewcommand{\\thetable}{S12}" in table
    (ROOT / "paper/supp/tableS12.tex").write_text("% Supplementary Table S12: the main-text ranking table (former Table 2), moved by "
                                                 "paper/tools/pass23_edits.py; values unchanged.\n" + table + "\n")
    commented = "".join("%" + l + "\n" for l in block[b:e].split("\n"))
    s = s[:i] + block[:b] + (f"\\noindent\\deleted[id={p22.TAG}]{{Table 2 (benchmark ranking) moved to Supplementary Table S12; it "
                             "repeated Figure 2b and Supplementary Table S6.}\n") + commented + block[e:].lstrip("\n") + s[j:]
    item_s10 = [l for l in s.split("\n") if l.startswith(r"\item \added[id=P9]{Table S10, direct biology check")]
    assert len(item_s10) == 1
    s = s.replace(item_s10[0], item_s10[0] + "\n" + rf"\item \added[id={p22.TAG}]{{Table S12, benchmark ranking (the former "
                  "main-text Table 2): mean rank per penalty and decoder over the ranked real tasks, range over tasks, mean "
                  "biology at $b^*$, simulation ranks, tasks not ranked and tasks ranked.}", 1)
    hd = [l for l in s.split("\n") if l.startswith(r"\hd{Supplementary material}")]
    assert len(hd) == 1 and hd[0].count("S4--S11,") == 1
    s = s.replace(hd[0], hd[0].replace("S4--S11,", rf"S4--\replaced[id={p22.TAG}]{{S12}}{{S11}},"))
    edit("paper/supp_items.tex", [("Tables S6, S7 and S9 and Figures", "Tables S6, S7, S9 and S12 and Figures"),
                                  ("(Note 1, Tables S6, S7 and S9, Figures", "(Note 1, Tables S6, S7, S9 and S12, Figures"),
                                  ("\\input{supp/tableS9}\n", "\\input{supp/tableS9}\n\\clearpage\n\\input{supp/tableS12}\n")])
    edit("paper/supp/tableS6.tex", [("Table~\\ref{tab:3} gives the mean ranks", "Supplementary Table~\\ref{tab:3} gives the mean ranks")])
    edit("paper/supp/note6.tex", [("(Table~\\ref{tab:3})", "(Supplementary Table~\\ref{tab:3})")])
    edit("paper/tools/make_supp_tables.py", [(r"6; Table~\ref{{tab:3}} gives", r"6; Supplementary Table~\ref{{tab:3}} gives")])
    edit("paper/tools/test_supp.py", [(r'r"Table~\ref{tab:3} gives the mean ranks"', r'r"Supplementary Table~\ref{tab:3} gives the mean ranks"')])
    return s


def main():
    p = ROOT / "paper/outline.tex"
    s = p.read_text()
    for old, new in EDITS:
        assert s.count("\n" + old + "\n") == 1, old[:80]
        line = rewrite(old, new)
        assert " ".join(accept(line).split()) == " ".join((new or r"\item").split()), old[:80]
        s = s.replace("\n" + old + "\n", "\n" + line + "\n", 1)
    s = move_res2b(s)
    s = move_table2(s)
    anchor = "\\definechangesauthor[name={Pass 22}, color=red]{P22}"
    assert s.count(anchor) == 1
    s = s.replace(anchor, "\\definechangesauthor[name={Pass 23}, color=blue]{P23}% Results walkthrough decisions; change log "
                  "paper/pass23_changelog.md\n" + anchor)
    p.write_text(s)


if __name__ == "__main__":
    main()
