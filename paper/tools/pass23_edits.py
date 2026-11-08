"""Pass 23 edits (2026-11-07, tracked id P23): decisions from the user's walkthrough of the Results, one result at a time.
Res-1: the fit-completion paragraph moves to the supplement (Supplementary Notes 5, 8 and 11 and Table S5 already hold every
number in it) and one sentence stays in the Results. Res-2: an opening sentence states that the order depends on the
decoder; the three worked examples of close readings become counts from paper/tools/res2_seed_noise.py
(paper/figure_data/res2_seed_noise.json); the simulation orders move to Supplementary Figure S1, which shows them.
Res-2b: the replication paragraph moves to a new Supplementary Note 13 (written by this script, with the replication mean
ranks of paper/tools/replication_mean_ranks.py added); one paragraph stays in the Results. Abstract, Disc-1 and Conclusions: the sweep leaders stay and a clause says that
the leaders changed on the replication's new seeds (user decision 2026-11-07).
Main-text Tables 1-3 repeat main-text content (user rule SI-115: no repeated information in the main text), so they move to
Supplementary Tables S13 (methods), S12 (ranking) and S14 (placement); each label moves with its table.
Res-3: a main claim with numbers from paper/tools/res3_curve_numbers.py leads; the ATAC-large sentence names its decoder and
the full range of the other penalties; the repeated 'no failed fits' sentences go (Res-1 says it); the Figure 3 caption follows
the redrawn figure (strongest settings marked, low-signal panel grey, standard tools left to Figure 6b).
Figure 3 then became a dot summary (user choice): biology minus plain scVI at b* and at the strongest setting; the curve
figure moved to Supplementary Figure S7 (figure_scripts/figS7.py); Figure 4 and 5 captions now point to Figure 2a for the
task order. Figure 2: the caption follows the redrawn figure (panel a has tasks as rows
and the decoders stacked; panel b stacks the decoders, sorts penalties best first and reverses the rank axis).
Each edited line is rewritten as word-level tracked changes against its printed (accepted) old text, as in Pass 22. Every old
line must occur exactly once or the script stops.
Run from the repository root:  python paper/tools/pass23_edits.py   (PASS23_ROOT overrides the root of the edited files)
Then: python paper/tools/accept_changes.py ; python paper/tools/check_ledger.py"""
import difflib
import re
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
    ('The real-task curves overlapped near plain scVI and around many matched \\replaced[id=P22]{readings}{readings,} but differed more at stronger penalties (Figure 3; curve construction in Supplementary Note 5). With the batch-conditioned decoder, the lung and pancreas curves were close near $b^*$, whereas the discriminator lost more biology at stronger settings. On the large chromatin-accessibility task, the barycenter critic reached a lower maximum batch score, 0.674, than the reference critic, \\replaced[id=P22]{0.815; this}{0.815. This} shorter batch-removal range did not reflect failed fits or failure to reach $b^*$. At the strongest fitted setting on lung with the batch-conditioned decoder, the barycenter critic ended at batch and biology scores of 0.466 and 0.397, below plain scVI on both scores. With the unconditioned decoder, the chromatin-accessibility curves lost biology as batch removal increased, whereas on pancreas several penalties increased both scores over part of their strength range. \\replaced[id=P22]{A}{Thus, a} curve could \\added[id=P22]{therefore} turn back or lose biology after reaching the matched level; no real-task curve ended early because of failed fits.',
     "The penalties kept similar biology up to the matched level and differed at stronger settings (Figure 3; trade-off curves in Supplementary Figure S7, built as in Supplementary Note 5): over the 11 ranked real task and decoder pairs, biology at $b^*$ differed between penalties by a median of 0.025, against 0.213 at each penalty's strongest fitted setting. The barycenter critic reached the lowest biology score of the six penalties in all 12 real task and decoder pairs (0.287 to 0.516); on lung with the batch-conditioned decoder its strongest setting ended below plain scVI on both scores (batch 0.466 and biology 0.397, against 0.633 and 0.722). It also had the shortest batch-removal range on the large chromatin-accessibility task with the batch-conditioned decoder (highest batch score 0.674; the other penalties 0.683 to 0.834). With the unconditioned decoder, the chromatin-accessibility curves lost biology as batch removal increased, whereas on pancreas every penalty raised both scores over part of its strength range."),
    ("\\caption{\\replaced[id=P10]{Batch score against biology score for the six real tasks (columns) with the batch-conditioned decoder (a) and the unconditioned decoder (b). Each line joins plain scVI (star) and one penalty's fitted strengths in increasing order, by mean scores over three seeds; no real-task curve ended early because of failed fits. Dashed line, matched batch removal $b^*$. Grey open symbols, each standard tool at its best setting by selection score (0.4 batch plus 0.6 biology) among the settings run; scANVI uses cell-type labels. Panels in a column share axes.}{MOCK -- synthetic data. Batch removal against biology kept in the six real tasks (columns) for the conditioned (a) and unconditioned (b) decoder. Each line joins plain scVI (star) and one penalty's sweep strengths by seed-mean scores (three seeds); a curve ending early marks collapsed fits. Dashed line, matched batch removal $b^*$. Grey open symbols, standard tools at their best setting.}}",
     "\\caption{Biology kept by each penalty relative to plain scVI, at the matched batch removal $b^*$ (filled dots) and at the penalty's strongest fitted setting (open dots), with the batch-conditioned (a) and the unconditioned decoder (b). One dot per real task: biology score minus plain scVI's with the same decoder (higher is better); grey line, no change. Filled dots are drawn for the ranked tasks only (the immune task with the batch-conditioned decoder is low-signal); open dots for all six real tasks. The full trade-off curves are in Supplementary Figure S7.}"),
    ("\\pp{Res-3}{Figure 3 shows each method's trade-off between batch removal and biology in every real task.}",
     '\\pp{Res-3}{Figure 3 compares the biology each penalty keeps at the matched level with the biology it keeps at its strongest setting.}'),
]


def rewrite(old, new):
    """Captions are tracked inside \\caption{} so the caption command itself is never struck."""
    if new and new.startswith("\\caption{"):
        inner_old, _ = group(accept(old).strip(), len("\\caption"))
        inner_new, _ = group(new, len("\\caption"))
        return "\\caption{" + p22.tracked(inner_old, inner_new) + "}"
    acc = accept(old).strip()
    if new and not acc.startswith(("\\item ", "\\pp{")):  # a paragraph rewritten throughout reads better as one replacement
        a, b = p22.tokens(acc), p22.tokens(new)
        if difflib.SequenceMatcher(None, a, b, autojunk=False).ratio() < 0.5:
            return rf"\replaced[id={p22.TAG}]{{{new}}}{{{p22.struck(acc)}}}"
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


MARK = "% ---------------------------------------------------------------- "
MOVES = [  # (start marker, next marker, label, supplementary number, word fixes, why it repeats)
    ("Table 1\n", "Table 2 (removed in Pass 4)\n", "tab:1", "S13",
     [("with the conditioned decoder", "with the batch-conditioned decoder", 1)],
     "it repeated the Methods text, Figure 1a and Supplementary Table S2"),
    ("Table 3\n", "Table 4\n", "tab:3", "S12",
     [("& conditioned &", "& batch-conditioned &", 6), ("with the conditioned decoder (--)", "with the batch-conditioned decoder (--)", 1)],
     "it repeated Figure 2b and Supplementary Table S6"),
    ("Table 4\n", "Supplementary Table S10 (Pass 9)\n", "tab:4", "S14",
     [("real task, conditioned decoder,", "real task, batch-conditioned decoder,", 1)],
     "it repeated Figure 6b and the Res-7 counts"),
]
OLD_NUMBER = {"tab:1": 1, "tab:3": 2, "tab:4": 3}  # printed numbers before Pass 23


def edit(path, pairs):
    p = ROOT / path
    s = p.read_text()
    for old, new in pairs:
        assert s.count(old) == 1, (path, s.count(old), old[:100])
        s = s.replace(old, new)
    p.write_text(s)


def move_table(s, start, end, label, snum, fixes, why):
    """Move one main-text table to the supplement as Supplementary Table <snum>; its label moves with it."""
    i, j = s.index(MARK + start), s.index(MARK + end)
    block = s[i:j]
    b, e = block.index("\\begin{table}[t]"), block.index("\\end{table}") + len("\\end{table}")
    table = accept(block[b:e])
    for old, new, n in fixes:
        assert table.count(old) == n, (old, table.count(old))
        table = table.replace(old, new)
    assert table.count("\\caption{") == 1 and table.count("\\label{%s}" % label) == 1
    table = table.replace("\\centering\\small\n", "\\centering\\scriptsize\\renewcommand{\\tablename}{Supplementary Table}"
                          "\\renewcommand{\\thetable}{%s}%%\n" % snum, 1)
    assert "renewcommand{\\thetable}{%s}" % snum in table
    (ROOT / f"paper/supp/table{snum}.tex").write_text(f"% Supplementary Table {snum}: the former main-text Table "
                                                     f"{OLD_NUMBER[label]}, moved by paper/tools/pass23_edits.py; values unchanged.\n"
                                                     + table + "\n")
    commented = "".join("%" + l + "\n" for l in block[b:e].split("\n"))
    note = (f"\\noindent\\deleted[id={p22.TAG}]{{Table {OLD_NUMBER[label]} moved to Supplementary Table {snum}; {why}.}}\n")
    return s[:i] + block[:b] + note + commented + block[e:].lstrip("\n") + s[j:]


def tracked_line(s, startswith, pairs):
    """Rewrite the one line that (accepted) starts with `startswith`, applying `pairs` to its accepted text, as tracked changes."""
    hits = [l for l in s.split("\n") if accept(l).startswith(startswith)]
    assert len(hits) == 1, (startswith, len(hits))
    new = accept(hits[0]).strip()
    for old, rep in pairs:
        assert new.count(old) == 1, (old, new.count(old))
        new = new.replace(old, rep)
    line = rewrite(hits[0], new)
    assert " ".join(accept(line).split()) == " ".join(new.split())
    return s.replace(hits[0], line, 1)


def move_tables(s):
    """Main-text Tables 1-3 repeat main-text content (user rule SI-115), so all three move to the supplement."""
    for m in MOVES:
        s = move_table(s, *m)
    # references: the main text now has no tables
    s = tracked_line(s, r"\qquad Each method adds a fixed penalty", [(r"(Table~\ref{tab:1})", r"(Supplementary Table~\ref{tab:1})")])
    s = tracked_line(s, r"\pp{Res-7}", [(r"(Table \ref{tab:4}, Figure 6b)", r"(Figure 6b; Supplementary Table \ref{tab:4})")])
    s = tracked_line(s, r"\item At the strength chosen on real tasks of other families",
                     [(r"; Table \ref{tab:4})", r"; Supplementary Table \ref{tab:4})"),
                      (r"which Table \ref{tab:4} prints", r"which Supplementary Table \ref{tab:4} prints")])
    s = tracked_line(s, r"\noindent Every supplementary note and table listed here", [
        ("Supplementary Notes 2--12 and", "Supplementary Notes 2--13 and"),
        ("Supplementary Note 1, Tables S6, S7 and S9 and Figures S1--S6 follow the main figures and tables, and Table S10 follows "
         r"Table~\ref{tab:4}.", "Table S10, then Supplementary Note 1, Tables S6, S7, S9 and S12--S14 and Figures S1--S8, follow "
         "the main figures.")])
    s = tracked_line(s, r"\item Table S7, single-strength placement",
                     [(r"(the counterpart of Table~\ref{tab:4})", r"(the counterpart of Supplementary Table~\ref{tab:4})")])
    s = tracked_line(s, r"\item Table S10, direct biology check", [(r"(typeset after Table~\ref{tab:4})", "(typeset after the main figures)")])
    for start in (r"\item Table \ref{tab:1}:", r"\item Table \ref{tab:4}:"):  # list of main-text tables (tab:3 is struck in EDITS)
        hits = [l for l in s.split("\n") if accept(l).startswith(start)]
        assert len(hits) == 1, start
        s = s.replace(hits[0], p22.rewrite(hits[0], None), 1)
    item_s10 = [l for l in s.split("\n") if accept(l).startswith(r"\item Table S10, direct biology check")]
    assert len(item_s10) == 1
    s = s.replace(item_s10[0], item_s10[0] + "\n" + "\n".join(rf"\item \added[id={p22.TAG}]{{{t}}}" for t in [
        "Table S12, benchmark ranking (the former main-text Table 2): mean rank per penalty and decoder over the ranked real "
        "tasks, range over tasks, mean biology at $b^*$, simulation ranks, tasks not ranked and tasks ranked.",
        "Table S13, alignment methods (the former main-text Table 1): alignment term, network, update steps, gradient penalty "
        "and sweep strengths.",
        "Table S14, single-strength placement with the batch-conditioned decoder (the former main-text Table 3): selection "
        "score per method and real task at the strength chosen on other task families, with the task's own best in "
        "parentheses."]), 1)
    hd = [l for l in s.split("\n") if l.startswith(r"\hd{Supplementary material}")]
    assert len(hd) == 1 and hd[0].count("S4--S11,") == 1
    s = s.replace(hd[0], hd[0].replace("S4--S11,", rf"S4--\replaced[id={p22.TAG}]{{S14}}{{S11}},"))
    # the writer's note sits inside \added[id=P12]{}; changes cannot nest, so it is amended in place (change log)
    old_n = (r"Pass 13 numbers the main-text tables 1 to 3 (Table \ref{tab:1} the methods, Table \ref{tab:3} the ranking and "
             r"Table \ref{tab:4} the placement table)")
    assert s.count(old_n) == 1
    s = s.replace(old_n, r"Pass 13 numbered the main-text tables 1 to 3 (the methods, the ranking and the placement table; since "
                  r"Pass 23 they are Supplementary Tables \ref{tab:1}, \ref{tab:3} and \ref{tab:4})")
    edit("paper/supp_items.tex", [("Tables S6, S7 and S9 and Figures S1-S6", "Tables S6, S7, S9 and S12-S14 and Figures S1-S7"),
                                  ("(Note 1, Tables S6, S7 and S9, Figures S1--S6)", "(Note 1, Tables S6, S7, S9 and S12--S14, Figures S1--S7)"),
                                  ("\\input{supp/tableS9}\n", "\\input{supp/tableS9}\n" + "".join(
                                      f"\\clearpage\n\\input{{supp/table{n}}}\n" for n in ("S12", "S13", "S14")))])
    edit("paper/supp/tableS6.tex", [("Table~\\ref{tab:3} gives the mean ranks", "Supplementary Table~\\ref{tab:3} gives the mean ranks")])
    edit("paper/supp/note6.tex", [("(Table~\\ref{tab:3})", "(Supplementary Table~\\ref{tab:3})")])
    edit("paper/supp/note8.tex", [("(Table~\\ref{tab:4} and Supplementary Table~S7)", "(Supplementary Tables~\\ref{tab:4} and S7)")])
    edit("paper/supp/tableS7.tex", [("(counterpart of Table~\\ref{tab:4} and", "(counterpart of Supplementary Table~\\ref{tab:4} and"),
                                    ("equal those of Table~\\ref{tab:4}.", "equal those of Supplementary Table~\\ref{tab:4}.")])
    edit("paper/tools/make_supp_tables.py", [
        (r"6; Table~\ref{{tab:3}} gives", r"6; Supplementary Table~\ref{{tab:3}} gives"),
        (r"(counterpart of Table~\ref{tab:4} and", r"(counterpart of Supplementary Table~\ref{tab:4} and"),
        (r"equal those of Table~\ref{tab:4}.", r"equal those of Supplementary Table~\ref{tab:4}."),
        ('txt = (ROOT / "outline.tex").read_text()', 'txt = (ROOT / "supp/tableS14.tex").read_text()  # Pass 23: moved from the outline'),
        ("placement table (label tab:4) of the outline", "placement table (label tab:4, Supplementary Table S14)")])
    edit("paper/tools/test_supp.py", [(r'r"Table~\ref{tab:3} gives the mean ranks"', r'r"Supplementary Table~\ref{tab:3} gives the mean ranks"')])
    return s


FIGS7 = r"""
\begin{figure}[p]
\centering\renewcommand{\figurename}{Supplementary Figure}\renewcommand{\thefigure}{S7}%
\includegraphics[width=\textwidth]{figures/figS7.png}
\caption{Trade-off curves (the main-text Figure 3 until Pass 23): batch score against biology score for the six real tasks (columns) with the batch-conditioned decoder (a) and the unconditioned decoder (b). Each line joins plain scVI (star) and one penalty's fitted strengths in increasing order, by mean scores over three seeds; the open circle marks the strongest fitted setting. Dashed line, matched batch removal $b^*$. Grey panel, the low-signal immune pair, which is not ranked. Panels in a column share axes. Figure 3 summarises these curves at $b^*$ and at the strongest setting.}
\end{figure}
"""


def move_figure3(s):
    """Figure 3 becomes the dot summary of Res-3; the curve figure moves to Supplementary Figure S7 (user choice)."""
    for start in (r"\caption{Secondary summaries per penalty", r"\caption{Width of each penalty's range"):
        s = tracked_line(s, start, [("in the order of Figure 3 from top to bottom", "in the order of Figure 2a from top to bottom")])
    s = tracked_line(s, r"\item Figure 3:", [(accept([l for l in s.split("\n") if accept(l).startswith(r"\item Figure 3:")][0])
                                               .strip()[len(r"\item Figure 3: "):],
                                               "biology score minus plain scVI's per penalty and decoder, one dot per real task, at "
                                               "matched batch removal $b^*$ and at the strongest fitted setting.")])
    item_s6 = [l for l in s.split("\n") if accept(l).startswith(r"\item Figure S6, best of $k$ settings per task")]
    assert len(item_s6) == 1
    s = s.replace(item_s6[0], item_s6[0] + "\n" + rf"\item \added[id={p22.TAG}]{{Figure S7, trade-off curves (the former main-text "
                  "Figure 3): batch score against biology score per penalty, task and decoder, with plain scVI, $b^*$ and the "
                  "strongest fitted setting marked.}", 1)
    hd = [l for l in s.split("\n") if l.startswith(r"\hd{Supplementary material}")]
    assert len(hd) == 1 and hd[0].count("Figures S1--S6") == 1
    s = s.replace(hd[0], hd[0].replace("Figures S1--S6", rf"Figures S1--\replaced[id={p22.TAG}]{{S7}}{{S6}}"))
    edit("paper/supp/figures.tex", [("Supplementary Figures S1-S6", "Supplementary Figures S1-S7")])
    (ROOT / "paper/supp/figures.tex").write_text((ROOT / "paper/supp/figures.tex").read_text() + FIGS7)
    edit("paper/figure_scripts/make_supp_figures.sh", [("for n in 1 2 3 4 5 6; do", "for n in 1 2 3 4 5 6 7; do"),
                                                       ("and Figures S1-S6 (PNG", "and Figures S1-S7 (PNG")])
    return s


def pareto_text(pf):
    """Res-3b paragraph and Figure 4 caption, numbers from figure_data/pareto_frontier.json (paper/tools/pareto_frontier.py)."""
    f = lambda v: f"{v:.3f}"
    d = pf["text"]
    n = pf["n_real_pairs"]
    assert pf["n_pairs_barycenter_reach_1"] == n and set(pf["n_pairs_reach_beyond_1"].values()) == {0, n}
    assert [a for a, v in pf["n_pairs_reach_beyond_1"].items() if v == 0] == ["barycenter"]
    head = r"Each penalty's Pareto frontier gives the most biology it kept at each level of batch removal (Figure 4)."
    body = (r"We scaled batch removal per task and decoder so that 0 is plain scVI, 0.5 is $b^*$ and 1 is the highest batch score "
            r"every penalty reaches, and took at each level the best biology among a penalty's strengths at that level or beyond "
            rf"(Methods). Up to $b^*$, the median frontier over the ranked real tasks stayed within {f(pf['max_abs_front_up_to_bstar'])} "
            r"of plain scVI's biology for every penalty and decoder (Figure 4a,b). At the highest common batch score, the "
            rf"discriminator's median frontier with the batch-conditioned decoder was {f(d['drop_at_common_cond1_discriminator'])} below plain scVI, "
            rf"against {f(d['drop_at_common_cond1_others_min'])}--{f(d['drop_at_common_cond1_others_max'])} for the other penalties; with the unconditioned decoder, the six frontiers "
            rf"ended {f(d['drop_at_common_cond0_min'])}--{f(d['drop_at_common_cond0_max'])} below. The hypervolume up to that level, the area under a "
            r"task's frontier above 0.10 below plain scVI's biology (0.10 when plain scVI's biology is kept throughout), had medians of "
            rf"{f(pf['hv_median_others_min'])}--{f(pf['hv_median_others_max'])} for every penalty and decoder except the discriminator "
            rf"with the batch-conditioned decoder, {f(pf['cond1.discriminator']['hv_median'])} (Figure 4c,d; Supplementary Table S8). "
            rf"The barycenter critic's batch removal ended at the highest common batch score in all {n} ranked pairs, so it set that "
            rf"level, whereas every other penalty reached beyond it in all {n} (Supplementary Note 7). Each frontier takes the best "
            r"strength at each level after scoring, so it is optimistic, and differences of the size seen at $b^*$ were often within "
            r"the leave-one-seed-out standard error (Res-2).")
    cap = (rf"Pareto frontier of each penalty over the ranked real tasks ({pf['n_tasks']['1']} with the batch-conditioned decoder, a "
           rf"and c; {pf['n_tasks']['0']} with the unconditioned decoder, b and d; the low-signal immune pair is excluded). Batch "
           r"removal is scaled per task so that 0 is plain scVI, 0.5 is matched batch removal $b^*$ (dashed line) and 1 is the highest "
           r"batch score every penalty reaches. (a, b) Best biology score reachable at each level or beyond, minus plain scVI's, "
           r"median over tasks, from seed means over three seeds. (c, d) Hypervolume up to the highest common batch score, the area "
           r"under each task's frontier above 0.10 below plain scVI's biology (higher is better; dashed line, 0.10, plain scVI's "
           r"biology kept throughout); dots, tasks; black tick, median. The frontiers are assembled after scoring and are optimistic.")
    return head, body, cap


FIG4 = """% ---------------------------------------------------------------- Figure 4 (Pass 23: Pareto frontier)
\\begin{figure}[!htbp]
\\centering
\\includegraphics[width=\\textwidth]{figures/fig4.png}
\\caption{CAPTION}
\\label{fig:4}
\\end{figure}

"""


def whole(s, start, new):
    """Pair that replaces the whole accepted text of the line starting with `start` (for tracked_line)."""
    hits = [accept(l).strip() for l in s.split("\n") if accept(l).startswith(start)]
    assert len(hits) == 1, start
    return hits[0], new


CAP5 = (r"Collapse rate per penalty up to the highest common batch score, with the batch-conditioned decoder (a) and the "
        r"unconditioned decoder (b): the percentage of a penalty's fits, at strengths whose mean batch score is at most the highest "
        r"batch score every penalty reaches, that lose 0.10 or more biology against the same seed's plain scVI (lower is better; "
        r"rates over all strengths in Supplementary Table S8). One point per real task, in the order of Figure 2a from top to "
        r"bottom within each row, with its nominal exact 95\% interval; black tick, median over the six real tasks, including the "
        r"low-signal immune pair with the batch-conditioned decoder (ringed).")
CAP6 = (r"Local minus global batch removal (panel c of the main-text Figure 4 until Pass 23) at matched batch removal $b^*$ per penalty, with the batch-conditioned decoder (a) and "
        r"the unconditioned decoder (b). Local removal is the mean of the three neighbourhood batch metrics and global removal the "
        r"mean of the two whole-embedding metrics (Supplementary Note 7); positive means stronger local removal. One point per real "
        r"task, in the order of Figure 2a from top to bottom within each row; black tick, median over the ranked real tasks; the "
        r"ringed low-signal immune pair with the batch-conditioned decoder is not in the median.")
FIGS8 = r"""
\begin{figure}[p]
\centering\renewcommand{\figurename}{Supplementary Figure}\renewcommand{\thefigure}{S8}%
\includegraphics[width=\textwidth]{figures/figS8.png}
\caption{CAPTION}
\end{figure}
"""


def res5_pair(s):
    """Res-5: the local-versus-global readings shrink to one sentence (figure moved to Supplementary Figure S8, user choice);
    numbers from figure_data/res5_local_global.json (paper/tools/res5_local_global.py)."""
    lg = json.loads((ROOT / "paper/figure_data/res5_local_global.json").read_text())
    assert lg["positive_tasks"] == ["atac_large"] and lg["n_positive_pairs"] == 2
    line = [accept(l) for l in s.split("\n") if accept(l).startswith("At $b^*$, every penalty had more global than local")]
    assert len(line) == 1
    old = line[0].split("In the same penalty order")[0]
    new = (r"At $b^*$, every penalty removed batch effects more globally than locally on every ranked real task except the large "
           r"chromatin-accessibility task, where local removal was stronger with both decoders. Within a task and decoder, the six "
           rf"penalties' local-minus-global readings differed by at most {lg['within_pair_spread_max']:.3f}, whereas their medians "
           rf"ranged from ${lg['pair_median_min']:.3f}$ to ${lg['pair_median_max']:.3f}$ across the {lg['n_ranked_pairs']} ranked "
           r"pairs, so the reading describes the task rather than the penalty (Supplementary Figure S8; Methods and Supplementary "
           r"Note 7). ")
    return old, new


def add_pareto(s):
    """New main Figure 4, the Pareto frontier with its hypervolume (user choice, version A: hypervolume capped at b_common);
    the old Figure 4 keeps only its collapse panel as Figure 5 (user choice, one topic per figure), its local-versus-global panel
    moves to Supplementary Figure S8 (user choice: the penalties barely differ) and its hypervolume panel to Figure 4 (rule SI-115,
    no repeats), so the old Figures 5-6 become Figures 6-7."""
    pf = json.loads((ROOT / "paper/figure_data/pareto_frontier.json").read_text())
    head, body, cap = pareto_text(pf)
    T = p22.TAG
    # figure files and labels (untracked: not printed)
    for old, new in [("6", "7"), ("5", "6"), ("4", "5")]:
        for a, b in [(f"{{figures/fig{old}.png}}", f"{{figures/fig{new}.png}}"), (f"\\label{{fig:{old}}}", f"\\label{{fig:{new}}}"),
                     (f"{MARK}Figure {old}\n", f"{MARK}Figure {new}\n")]:
            assert s.count(a) == 1, a
            s = s.replace(a, b)
    anchor = "\\label{fig:3}\n\\end{figure}\n\n"
    assert s.count(anchor) == 1
    s = s.replace(anchor, anchor + rf"\pp{{\added[id={T}]{{Res-3b}}}}{{\added[id={T}]{{{head}}}}}" + "\n"
                  + rf"\added[id={T}]{{{body}}}" + "\n\n" + FIG4.replace("CAPTION", rf"\added[id={T}]{{{cap}}}"))
    lines = [  # (line start, [(old, new)]) on accepted text, tracked
        (r"\qquad For each penalty we also report",
         [("plane that a penalty's sweep points reach beyond a reference point at plain scVI's batch score and 0.10 below its biology "
           "score, divided by the batch gap that sets $b^*$ so that tasks share one scale.",
           "plane that a penalty's sweep points reach beyond a reference point at plain scVI's batch score and 0.10 below its biology "
           "score, with batch scores above the highest common batch score counted at it so that every penalty is measured over the "
           "same batch range, divided by the batch gap that sets $b^*$ so that tasks share one scale; it is the area under the "
           "penalty's Pareto frontier, the best biology among its strengths at or beyond each batch level (Figure 4; uncapped "
           "values in Supplementary Table S8).")]),
        (r"\pp{Res-5}{", [("(Figure 4c;", "(Supplementary Figure S8;")]),
        (r"\item Figure 1:", [("Second cut, after Figure 5.", "Second cut, after Figure 6.")]),
        (r"\pp{Res-6}{", [("(Figure 5)", "(Figure 6)")]),
        ("Across the real tasks, including the low-signal immune pair, good-strength widths", [("(Figure 5;", "(Figure 6;")]),
        (r"\pp{Res-7}{", [("(Figure 6b;", "(Figure 7b;")]),
        (r"\pp{Res-8}{", [("(Figure 6a;", "(Figure 7a;")]),
        (r"\item With the batch-conditioned decoder, a penalty's expected best", [("(Figure 6a)", "(Figure 7a)")]),
        (r"\item Figure S6, best of $k$", [("is Figure 6a.", "is Figure 7a.")]),
        (r"\item Figure 6:", [("Figure 6:", "Figure 7:")]),
        (r"\item Figure 5:", [("Figure 5:", "Figure 6:")]),
        (r"\item Figure 4:", [whole(s, r"\item Figure 4:", r"\item Figure 5: collapse rate up to the highest common batch score per "
                                    r"penalty, one panel per decoder, one point per real task with its 95\% interval and the median.")]),
        (r"\caption{Secondary summaries per penalty", [whole(s, r"\caption{Secondary summaries per penalty", "\\caption{" + CAP5 + "}")]),
    ]
    extra = collapse_edits(s)
    for start, pairs in lines + [(k, []) for k in extra if k not in dict(lines)]:
        s = tracked_line(s, start, pairs + extra.pop(start, []))
    assert not extra
    old5, new5 = res5_pair(s)
    hit = [l for l in s.split("\n") if accept(l).startswith(old5[:60])]
    assert len(hit) == 1 and accept(hit[0]).startswith(old5)
    line5 = rf"\replaced[id={T}]{{{new5.strip()}}}{{{p22.struck(old5.strip())}}} " + accept(hit[0])[len(old5):]
    assert " ".join(accept(line5).split()) == " ".join((new5 + accept(hit[0])[len(old5):]).split())
    s = s.replace(hit[0], line5, 1)
    item5 = [l for l in s.split("\n") if accept(l).startswith(r"\item Figure 5:")]
    assert len(item5) == 1
    s = s.replace(item5[0], rf"\item \added[id={T}]{{Figure 4: Pareto frontier per penalty and decoder, the median over the "
                  r"ranked real tasks of the best biology reachable at each scaled batch level, and the hypervolume up to the highest "
                  "common batch score per task.}\n" + item5[0], 1)
    s7 = [l for l in s.split("\n") if accept(l).startswith(r"\item Figure S7, trade-off curves")]
    assert len(s7) == 1
    s = s.replace(s7[0], s7[0] + "\n" + rf"\item \added[id={T}]{{Figure S8, local minus global batch removal at $b^*$ per "
                  "penalty, one panel per decoder, one point per real task and the median.}", 1)
    hd = rf"Figures S1--\replaced[id={T}]{{S7}}{{S6}}"
    assert s.count(hd) == 1
    s = s.replace(hd, rf"Figures S1--\replaced[id={T}]{{S8}}{{S6}}")
    (ROOT / "paper/supp/figures.tex").write_text((ROOT / "paper/supp/figures.tex").read_text() + FIGS8.replace("CAPTION", CAP6))
    edit("paper/supp/figures.tex", [("Supplementary Figures S1-S7", "Supplementary Figures S1-S8")])
    edit("paper/figure_scripts/make_supp_figures.sh", [("for n in 1 2 3 4 5 6 7; do", "for n in 1 2 3 4 5 6 7 8; do"),
                                                       ("and Figures S1-S7 (PNG", "and Figures S1-S8 (PNG")])
    edit("paper/supp_items.tex", [("and S12-S14 and Figures S1-S7", "and S12-S14 and Figures S1-S8"),
                                  ("S12--S14, Figures S1--S7)", "S12--S14, Figures S1--S8)")])
    w = r"\replaced[id=P9]{\replaced[id=P11]{Figures 1--6 are drawn from the sweep analysis"
    assert s.count(w) == 1
    i = s.index(w) + len(r"\replaced[id=P9]")
    j = group(s, i)[1]
    j = group(s, j)[1]  # end of the outer \replaced's second argument
    s = s[:j] + rf"\added[id={T}]{{ In Pass 23 Figure 4, the Pareto frontier, was added and the local-versus-global panel of the " \
        "former Figure 4 moved to Supplementary Figure S8, so Figures 1--7 are drawn from the sweep analysis.}" + s[j:]
    edit("paper/supp/figures.tex", [("is Figure 6a. Every", "is Figure 7a. Every")])
    edit("paper/supp/tableS7.tex", [("and of Figure~6). \\textbf{(a)}", "and of Figure~7). \\textbf{(a)}")])
    edit("paper/tools/make_supp_tables.py", [(r"and of Figure~6). \textbf{(a)}", r"and of Figure~7). \textbf{(a)}")])
    edit("paper/supp/note7.tex", [
        ("left out of the means of Figure 4c.", "left out of the medians of Supplementary Figure S8."),
        ("(Figure 6a; per-task", "(Figure 7a; per-task"),
        ("The hypervolume is reported per penalty, task and decoder; the analysis record also holds it with the reference point at "
         "biology 0, a sensitivity analysis not used in the paper.",
         "The paper reports the hypervolume up to the highest common batch score: each point's batch score is capped at "
         r"$b_{\mathrm{common}}$ before the rectangles are formed, so that every penalty is measured over the same batch range, from "
         r"plain scVI to the highest batch score all six reach. In the scaled batch $x=(B-b_0)/(b_{\mathrm{common}}-b_0)$ it is the "
         r"area from $x=0$ to 1 under the penalty's Pareto frontier, the highest biology score minus plain scVI's among its "
         r"strengths with scaled batch at least $x$, above $-0.10$; a penalty that keeps plain scVI's biology throughout scores 0.10 "
         r"(Figure 4, which shows the median frontier over the ranked real tasks). Without the cap, the hypervolume over all "
         r"strengths (HV$_{\mathrm{all}}$ in Supplementary Table S8) also counts batch removal beyond $b_{\mathrm{common}}$; the "
         rf"barycenter critic reaches $b_{{\mathrm{{common}}}}$ and no further in all {pf['n_real_pairs']} ranked real pairs, while "
         r"every other penalty reaches beyond it, so the uncapped value favours the penalties that reach furthest. The cap was "
         r"chosen after all scores had been read (Supplementary Note 10). Both values are reported per penalty, task and decoder; the "
         r"analysis record also holds the uncapped value with the reference point at biology 0, a sensitivity analysis not used in "
         r"the paper.")])
    edit("paper/supp/note10.tex", [
        ("(f) the hypervolume's name and the share of non-dominated strengths (2 November, above).",
         "(f) the hypervolume's name and the share of non-dominated strengths (2 November, above); (g) the Pareto frontier of "
         "Figure 4 and the cap of the hypervolume at the highest common batch score (Supplementary Note 7), chosen on 7 November "
         "2026 after all sweep scores and the uncapped hypervolume had been read; the uncapped hypervolume is reported beside it "
         "(Supplementary Table S8).")])
    return s


def collapse_text(cc):
    """Res-4 heading and paragraph: collapse rate in the common batch range (figure_data/collapse_common_range.json)."""
    pc = lambda v: f"{100 * v:.0f}"
    lab = dict(NAMES)
    med = {c: sorted(((cc[f"cond{c}.{a}"]["median"], a) for a, _ in NAMES), key=lambda x: -x[0]) for c in (0, 1)}
    def listing(c):
        nz = [f"{pc(v)}\\% for {lab[a]}" for v, a in med[c] if v > 0]
        z = [a for v, a in med[c] if v == 0]
        return ", ".join(nz[:-1]) + (" and " if not z else ", ") + nz[-1] + (f" and 0\\% for the other {len(z)}" if z else "")
    lst0 = listing(0).replace("the other 3", "the other three")
    d = cc["cond1.discriminator"]
    assert max(cc[f"cond1.{a}"]["max"] for a, _ in NAMES) == d["max"] and d["max_tasks"] == ["immune_hum_mou"]
    assert all(cc[f"cond1.{a}"]["max_tasks"] == ["atac_small"] for a, _ in NAMES if a != "discriminator")
    assert cc["cond0.barycenter"]["k_beyond"] == cc["cond1.barycenter"]["k_beyond"] == 0
    head = (r"The collapse rate counts how often a penalty's fits lose 0.10 or more biology (Figure 5; the share of "
            r"non-dominated strengths is in Supplementary Table S8).")
    body = (r"We count the fits at strengths whose mean batch score is at most the highest common batch score, the range of "
            r"Figure 4; rates over all strengths are in Supplementary Table S8 (Methods). With the batch-conditioned decoder, the "
            rf"median collapse rate over the six real tasks was {listing(1)}; with the unconditioned decoder it was {lst0}. The "
            rf"highest rate with the batch-conditioned decoder was the discriminator's, {pc(d['max'])}\% ({pc(d['max_lo'])}\%--"
            rf"{pc(d['max_hi'])}\%, nominal 95\% interval) on the human--mouse immune task; the other five penalties peaked on the "
            rf"small chromatin-accessibility task. MMD collapsed less often than the discriminator on "
            rf"{cc['cond1_mmd_below_discriminator']} of the {cc['cond1_n_tasks']} real tasks with that decoder, although it ranked "
            rf"above the discriminator at $b^*$ on only {cc['cond1_mmd_ranked_above_discriminator']} of the {cc['cond1_n_ranked']} "
            r"ranked tasks. Every collapse was a loss of biology; no fit failed. Over all strengths, "
            rf"{d['k_beyond']} of the discriminator's {d['k_all']} collapses with the batch-conditioned decoder and "
            rf"{cc['cond0.discriminator']['k_beyond']} of {cc['cond0.discriminator']['k_all']} with the unconditioned decoder lay "
            r"beyond the highest common batch score, against none of the barycenter critic's, whose strongest setting sets that "
            r"score. These summaries include the low-signal immune pair.")
    return head, body


def collapse_edits(s):
    """Pairs for the lines the collapse-rate change touches (user choice: main text in the common batch range)."""
    cc = json.loads((ROOT / "paper/figure_data/collapse_common_range.json").read_text())
    head, body = collapse_text(cc)
    pc = lambda v: f"{100 * v:.0f}"
    m, d = cc["cond1.mmd"], cc["cond1.discriminator"]
    assert min(cc[f"cond1.{a}"]["max"] for a, _ in NAMES) == m["max"]
    old_head = [accept(l) for l in s.split("\n") if accept(l).startswith(r"\pp{Res-4}{")]
    old_body = [accept(l).strip() for l in s.split("\n") if accept(l).startswith("The secondary summaries varied across real tasks")]
    assert len(old_head) == len(old_body) == 1
    return {
        r"\pp{Res-4}{": [(old_head[0][len(r"\pp{Res-4}{"):-1], head)],
        "The secondary summaries varied across real tasks": [(old_body[0], body)],
        r"\qquad For each penalty we also report": [
            ("the collapse rate is the collapsed share of a penalty's fits in a task and decoder, with exact binomial",
             "the collapse rate is the collapsed share of a penalty's fits in a task and decoder at strengths whose mean batch score "
             "is at most the highest common batch score (the range of Figure 4; all strengths in Supplementary Table S8), with "
             "exact binomial")],
        r"\item With the batch-conditioned decoder, the largest collapse rate": [
            ("With the batch-conditioned decoder, the largest collapse rate on a real task ranged from 27\\% (12\\%--46\\%) for MMD "
             "to 73\\% (54\\%--88\\%) for the discriminator",
             "Up to the highest common batch score, collapse rates with the batch-conditioned decoder peaked at "
             f"{pc(m['max'])}\\% ({pc(m['max_lo'])}\\%--{pc(m['max_hi'])}\\%) for MMD and {pc(d['max'])}\\% "
             f"({pc(d['max_lo'])}\\%--{pc(d['max_hi'])}\\%) for the discriminator")],
        r"\item The collapse-rate intervals are nominal": [
            ("(30; 33 for the discriminator, the reference critic and the pooled critic on the immune task with the "
             "batch-conditioned decoder, which have additional strengths; 36 where edge strengths were fitted)",
             f"({cc['n_common_min']} to {cc['n_common_max']} up to the highest common batch score; over all strengths 30, 33 for "
             "the discriminator, the reference critic and the pooled critic on the immune task with the batch-conditioned decoder, "
             "which have additional strengths, and 36 where edge strengths were fitted)"),
            ("it does not bound the coverage", "it covers only 30 and 36 fits and does not bound the coverage")],
    }


def collapse_supp():
    """Supplementary Notes 7 and 10 for the common-range collapse rate (after add_pareto, which writes Note 10 item g)."""
    cc = json.loads((ROOT / "paper/figure_data/collapse_common_range.json").read_text())
    edit("paper/supp/note7.tex", [
        ("Each rate has an exact binomial (Clopper--Pearson) 95\\% interval.",
         "Each rate has an exact binomial (Clopper--Pearson) 95\\% interval. The main text and Figure 5 count only the fits at "
         r"strengths whose mean batch score over the three seeds is at most $b_{\mathrm{common}}$, the range of Figure 4 "
         rf"({cc['n_common_min']} to {cc['n_common_max']} fits per real task and decoder pair), because many collapses lie beyond "
         r"it at strengths that only some penalties reach; Supplementary Table S8 gives the rate over all strengths beside it. "
         r"This restriction was chosen after all scores had been read (Supplementary Note 10)."),
        ("because the collapse rate does not use $b^*$.",
         r"because the collapse rate does not use $b^*$ (the common range uses $b_{\mathrm{common}}$, which every pair has).")])
    edit("paper/supp/note10.tex", [
        ("the uncapped hypervolume is reported beside it (Supplementary Table S8).",
         "the uncapped hypervolume is reported beside it (Supplementary Table S8); (h) the restriction of the main-text collapse "
         "rate to the same range, chosen on 7 November 2026 after all collapse rates had been read; the rate over all strengths "
         "is reported beside it (Supplementary Table S8).")])


def main():
    p = ROOT / "paper/outline.tex"
    s = p.read_text()
    for old, new in EDITS:
        assert s.count("\n" + old + "\n") == 1, old[:80]
        line = rewrite(old, new)
        assert " ".join(accept(line).split()) == " ".join((new or r"\item").split()), old[:80]
        s = s.replace("\n" + old + "\n", "\n" + line + "\n", 1)
    s = move_res2b(s)
    s = move_tables(s)
    s = move_figure3(s)
    s = add_pareto(s)
    collapse_supp()
    anchor = "\\definechangesauthor[name={Pass 22}, color=red]{P22}"
    assert s.count(anchor) == 1
    s = s.replace(anchor, "\\definechangesauthor[name={Pass 23}, color=blue]{P23}% Results walkthrough decisions; change log "
                  "paper/pass23_changelog.md\n" + anchor)
    p.write_text(s)


if __name__ == "__main__":
    main()
