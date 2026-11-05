"""Pass 12 edits (tracked id P12): fixes from the Pass 11 rigour review (findings F01-F08, F11, F12, F14, F15).
Edits paper/outline.tex, revision/reviewer_response.md and paper/lit_search_20261105.md. Every old text must occur exactly once
or the script stops. Every number it writes is a row of paper/pass12_numbers.csv, re-computed here from its source
(paper/tools/check_ledger.py) and checked against the row before it is used. Supplement blocks owned by branch supp-items
(Supplementary Note 1 body; the Table S6-S10 and Figure S1-S6 items) and the replication placeholders are not touched.
Run from the repository root, on the workstation (the sources are under the data root):  python paper/tools/pass12_edits.py
Then: python paper/tools/check_ledger.py"""
import csv, os, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import check_ledger as cl  # noqa: E402

ROOT = Path(os.environ.get("PASS12_ROOT") or cl.ROOT)  # where the edited files live (tests use a copy); sources resolve from the repo
rows = list(csv.DictReader(open(cl.ROOT / "paper/pass12_numbers.csv", newline="")))
raw = {}
for r in rows:
    raw[r["id"]] = cl.evaluate(cl.read_source(r["source"], cl.ROOT, cl.DATA_ROOT) if r["source"] else None, r["selector"], raw)
    got = cl.fmt_value(raw[r["id"]], r["fmt"], r["written"])
    assert cl.norm(got) == cl.norm(r["written"]), (r["id"], r["written"], got)
N = {r["id"]: r["written"] for r in rows}


def edit(path, pairs):
    p = ROOT / path
    s = p.read_text()
    for old, new in pairs:
        assert s.count(old) == 1, (path, s.count(old), old[:100])
        s = s.replace(old, new)
    p.write_text(s)


R = r"\replaced[id=P12]"
A = r"\added[id=P12]"
real = "real tasks"

outline = [
    # author definition
    (r"\definechangesauthor[name={Pass 10}, color=teal]{P10}",
     r"\definechangesauthor[name={Pass 10}, color=teal]{P10}" + "\n" +
     r"\definechangesauthor[name={Pass 12}, color=red]{P12}% rigour-review fixes; change log paper/pass12_changelog.md, numbers in paper/pass12_numbers.csv (checked by paper/tools/check_ledger.py)"),
    # F11: table numbers by label; the number 2 stays free
    (r"Tables 3 and 4 hold sweep values", r"Tables \ref{tab:3} and \ref{tab:4} hold sweep values"),
    (r"{Tables 1--2 hold real design facts}.",
     r"{Tables 1--2 hold real design facts}." + A + r"{ The number 2 is left free, so the tables print as Tables 1, 3 and 4 (renumbering is the first author's call), and the text cites them by label. Pass 12 (change log \texttt{paper/pass12\_changelog.md}) fixes the points of the Pass 11 rigour review; every abstract number is listed with its source in \texttt{paper/pass12\_numbers.csv} and re-read by \texttt{paper/tools/check\_ledger.py}.}"),
    ("% ---------------------------------------------------------------- Table 3\n\\begin{table}[t]",
     "% ---------------------------------------------------------------- Table 3\n"
     "\\setcounter{table}{2}% Pass 12: number 2 stays free (Table 2 removed in Pass 4), so this float prints as Table 3 and the next as Table 4\n"
     "\\begin{table}[t]"),
    (r"(Figure 2, Table 3)", r"(Figure 2, Table \ref{tab:3})"),
    (r"(Table 4, Figure 6b)", r"(Table \ref{tab:4}, Figure 6b)"),
    (r"\item Table 1: methods:", r"\item Table \ref{tab:1}: methods:"),
    (r"\item Table 3: \replaced{benchmark ranking", r"\item Table \ref{tab:3}: \replaced{benchmark ranking"),
    (r"\item Table 4: single-strength placement", r"\item Table \ref{tab:4}: single-strength placement"),
    # F02: abstract and Results label the binomial intervals as nominal
    (r"73\% (54\%--88\%) for the discriminator", r"73\% (54\%--88\%) for the discriminator" + A + r"{ (nominal 95\% intervals)}"),
    (r"with its exact interval, was as follows", r"with its exact " + A + r"{nominal 95\% }interval, was as follows"),
    (r"with exact binomial (Clopper--Pearson) 95\% intervals \citep{clopper1934use}.",
     r"with exact binomial (Clopper--Pearson) 95\% intervals \citep{clopper1934use}" + A + r"{, which are nominal because the fits of one seed are judged against one plain-scVI threshold and are not independent (Discussion)}."),
    (r"(b) Collapse rate in percentage points with exact 95\% intervals (lower is better)",
     r"(b) Collapse rate in percentage points with exact " + A + r"{nominal }95\% intervals (lower is better)"),
    # F04: abstract conclusions
    (r"For scVI users, the choice of strength mattered more", R + r"{Within these tasks, for scVI users}{For scVI users}, the choice of strength mattered more"),
    (r"mainly when several strengths were tried and scored, and scANVI, where cell-type labels exist, scored higher",
     r"mainly when several strengths were tried and scored" + A + r"{ on the same data, which makes that gain optimistic}, and scANVI, " + R +
     r"{which is trained with the cell-type labels that several biology metrics also use}{where cell-type labels exist}, scored higher"),
    # F14
    (r"so the adversary is the only correction", r"so the " + R + r"{alignment penalty}{adversary} is the only correction"),
    # F03: the sign-flip test and its per-task sensitivity analysis
    (r"by a one-sided sign-flip permutation test over pairs (all $2^{13}$ sign patterns, $p<0.05$).",
     r"by a one-sided sign-flip permutation test over pairs (all $2^{13}$ sign patterns, $p<0.05$)." + A +
     rf"{{ The sign-flip test treats the {N['E01']} pairs as exchangeable, although the two decoders of a task use the same task data and some tasks are related (two nested pairs), so its $p$-value is approximate. A sensitivity analysis that flips the signs per task, both decoders of a task together (all $2^{{{N['E02']}}}$ sign patterns over the {N['E03']} tasks), was added after the rule was fixed; it is reported beside the rule and is not part of it.}}"),
    (r"whether the agreement rule is met.}}",
     r"whether the agreement rule is met.}}" + "\n" + r"\item " + A +
     r"{\res{sensitivity analysis, not part of the rule: the permutation $p$ of the mean $\tau_b$ with signs flipped per task, both decoders of a task together.}}"),
    # F04: tiny differences from plain scVI in the single-strength placement
    (r"(median $+0.006$; Table 4); at each task's own best strength",
     r"(median $+0.006$; Table \ref{tab:4})" + A +
     rf"{{; {N['D05']} of these {N['D04']} differences were smaller than 0.005 in size (for example ${N['D01']}$ for the Sinkhorn divergence on pancreas, which Table \ref{{tab:4}} prints as a tie), against seed standard deviations of the penalty's selection score of {N['D06']} to {N['D07']} in those cells, so the counts mix small gains, small losses and near-ties}}"
     r"; at each task's own best strength"),
    (r"The immune task is low-signal with this decoder.}",
     r"The immune task is low-signal with this decoder.}" + A +
     rf"{{ Values are rounded to three decimals: the Sinkhorn divergence on pancreas ({N['D02']}) is {N['D03']} below plain scVI ({N['D08']}) and counts as not exceeding it.}}"),
    # F02 and F12: Discussion bullet on the collapse intervals
    (r"\added[id=P3]{The collapse-rate intervals treat a penalty's 30 fits in a task and decoder (36 where edge strengths were fitted) as independent",
     r"\added[id=P3]{" + R +
     rf"{{The collapse-rate intervals are nominal exact binomial 95\% intervals. They treat a penalty's fits in a task and decoder ({N['B01']}; {N['B02']} for the discriminator, the reference critic and the pooled critic on the immune task with the batch-conditioned decoder, which have additional strengths; {N['B03']} where edge strengths were fitted) as independent, although each seed's fits are judged against the same plain-scVI threshold and can collapse together, so their actual coverage may differ from 95\%. An illustrative simulation of this dependence is in Supplementary Note 7; it does not bound the coverage of the intervals reported here.}}"
     r"{The collapse-rate intervals treat a penalty's 30 fits in a task and decoder (36 where edge strengths were fitted) as independent"),
    (r"so the coverage of a nominal 0.95 may lie between 0.83 and 1.00 (Supplementary Note 7).}",
     r"so the coverage of a nominal 0.95 may lie between 0.83 and 1.00 (Supplementary Note 7).}}"),
    # F04: Conclusions
    (r"and plain scVI mainly when several strengths were tried;", r"and plain scVI mainly when several strengths were tried" + A + r"{ and scored on the same data};"),
    (r"Practical guidance for scVI users, within these tasks: (i) tune the strength rather than the penalty, and keep plain scVI with the batch-conditioned decoder as a candidate, since",
     R + r"{Observations for scVI users, from the scores on these eight tasks and not tested on independent data: (i) the strength mattered more than the choice of penalty, and plain scVI with the batch-conditioned decoder remained a strong candidate, since}{Practical guidance for scVI users, within these tasks: (i) tune the strength rather than the penalty, and keep plain scVI with the batch-conditioned decoder as a candidate, since}"),
    (r"and the best of six strengths in 30; (ii) start from the strengths chosen on other task families with that decoder, 0.1 to 100 across penalties (Results); (iii) check the biology directly at the chosen strength, because",
     r"and the best of six strengths in 30" + A + r"{, a count that is optimistic because the strengths were chosen and scored on the same data}; (ii) " + R +
     r"{the strengths chosen on other task families with that decoder lay between 0.1 and 100 across penalties (Results)}{start from the strengths chosen on other task families with that decoder, 0.1 to 100 across penalties (Results)}; (iii) " + R +
     r"{the biology was worth checking directly at the chosen strength}{check the biology directly at the chosen strength}, because"),
    (r"(iv) when cell-type labels are available, use scANVI.}",
     r"(iv) " + R + r"{when cell-type labels were available, scANVI scored higher than every penalty on most tasks, but it is trained with the labels that several biology metrics also use, so its biology scores are not label-free}{when cell-type labels are available, use scANVI}.}"),
    # F05 (minor): scope of 'no strength is chosen' in Supplementary Note 4
    (r"that was not run, so no strength is chosen from the data and the analysis reads every sweep strength;",
     r"that was not run, so " + R + r"{the strength-choice rules choose no strength}{no strength is chosen from the data} and the analysis reads every sweep strength" + A +
     r"{ (the replication's strengths, chosen from the sweep's batch scores, are described below)};"),
    # F08: no citation in the supplement
    (r"were given the same rank as unintegrated data \citep{luecken2022benchmarking}; mean rank over all real tasks",
     r"were given the same rank as unintegrated data" + R + r"{ (main-text reference)}{ [citation of the scIB benchmark]}; mean rank over all real tasks"),
    # F12 and F02: Supplementary Note 7
    (r"collapse (definition, \replaced{30 fits per penalty, task and decoder, 36 where edge strengths were fitted}{30 fits per cell}, exact binomial 95\% intervals,",
     r"collapse (definition, \replaced{" + R +
     rf"{{{N['B04']} fits per penalty, task and decoder, {N['B05']} for the discriminator, the reference critic and the pooled critic on the immune task with the conditioned decoder (additional strengths), {N['B06']} where edge strengths were fitted}}"
     r"{30 fits per penalty, task and decoder, 36 where edge strengths were fitted}}{30 fits per cell}, exact binomial 95\% intervals" + A +
     r"{, nominal because the fits of one seed are not independent},"),
    (r"failure types reported separately; coverage of the interval simulated for ten and twelve strengths with three seeds}",
     r"failure types reported separately; " + R +
     rf"{{an illustrative simulation of the interval's coverage (script \texttt{{paper/analysis\_pass10/coverage\_sim\_sweep.py}}, random seed {N['C01']}, {N['C02']} replicates per scenario). The estimand is the probability that the exact 95\% interval computed from {N['C03']} fits (ten strengths, three seeds) or {N['C04']} fits (twelve strengths) contains the mean collapse probability over the strength grid. In scenario A the fits are independent and the collapse probability rises with strength along a profile set by hand ({N['C05']} at six points, interpolated linearly to ten or twelve strengths). In scenario B each seed has one collapse threshold shared by its strengths, so a seed that collapses at one strength collapses at every higher one; the threshold position is drawn from a distribution set by hand ({N['C06']} over seven positions, the last meaning no collapse, interpolated and renormalised). A control with independent fits at a constant probability of {N['C07']} must cover at least 0.95. Coverage was at least {N['C08']} in scenario A, {N['C09']} ({N['C03']} fits) and {N['C10']} ({N['C04']} fits) in scenario B, and {N['C11']} and {N['C12']} in the control, with Monte Carlo standard errors of at most {N['C13']}. Both profiles are illustrative choices, not estimated from the sweep; other dependence structures give other coverage, and the {N['B05']}-fit design is not simulated, so the simulation shows that dependence can lower coverage below the nominal level but does not bound the coverage of the reported intervals}}"
     r"{coverage of the interval simulated for ten and twelve strengths with three seeds}}"),
]

response = [
    (r"Every number below is copied from the paper text, whose numbers are listed with their source files in `paper/pass9_numbers.csv`.",
     r"Every number below is copied from the paper text, whose numbers are listed with their source files in `paper/pass9_numbers.csv`" + "\n"
     r"and, from Pass 12, `paper/pass12_numbers.csv` (every abstract number; `paper/tools/check_ledger.py` re-reads the ledgers from" + "\n"
     r"their sources). Updated in Pass 12 (2026-11-05) after the Pass 11 rigour review. Table numbers are those the PDF prints: the" + "\n"
     r"number 2 is left free (the strength-range table was removed), so the ranking table is Table 3 and the placement table Table 4."),
    (r'seeds, checked against a prediction and an agreement rule fixed before any of these fits was scored (Methods, "Replication' + "\n"
     r'   across training seeds"; Res-2b).',
     r'seeds, checked against a prediction and an agreement rule fixed before any of these fits was scored (Methods, "Replication' + "\n"
     r'   across training seeds"; Res-2b).' + r" The rule's sign-flip test treats the 13 task and decoder pairs as exchangeable, although the" + "\n"
     r"   two decoders of a task share its data, so its $p$-value is approximate; a sensitivity analysis that flips the signs per task" + "\n"
     r"   (both decoders together), added after the rule was fixed, is reported beside it."),
    (r"the batch-conditioned discriminator, whose label-transfer recall fell to 0.326 against 0.797 for plain scVI. We did not analyse" + "\n"
     r"reference coverage: the reference batch is fixed by a rule (highest cell-type entropy), not varied (Disc-5).",
     r"the batch-conditioned discriminator, whose label-transfer recall fell to 0.326 against 0.797 for plain scVI. On references: the" + "\n"
     r"reference batch is fixed by a rule (highest cell-type entropy, so that the target does not favour a few cell types) and is not" + "\n"
     r"varied (Disc-5). We report how much of each task the chosen reference covers (Methods, penalties paragraph): it holds every cell" + "\n"
     r"type in five of the eight tasks, and 10 of 17 (lung), 14 of 19 (human--mouse immune) and 3 of 4 (simulation 2) in the others, so" + "\n"
     r"in those three tasks some cells of other batches have no cell of their type in the reference. We did not vary the reference or" + "\n"
     r"test whether its coverage explains a penalty's performance."),
    (r"best of six strengths in 30 of 36; scANVI, which uses cell-type labels, was exceeded only on pancreas (Res-8).",
     r"best of six strengths in 30 of 36, a count that is optimistic because the strengths were chosen and scored on the same data;" + "\n"
     r"scANVI, which is trained with the cell-type labels that several biology metrics also use, was exceeded only on pancreas (Res-8)."),
    (r"**Added.** Disc-3 and the Conclusions now give guidance supported by the numbers: tune the strength rather than the penalty,",
     r"**Added.** Disc-3 and the Conclusions now give observations from the scores on these eight tasks, not tested on independent" + "\n"
     r"data: the strength mattered more than the choice of penalty,"),
    (r"to six tries raised each penalty's mean selection score by 0.012 to 0.053; keep plain scVI with the batch-conditioned decoder as a" + "\n"
     r"candidate, since a penalty beat it mainly when several strengths were tried; start from the strengths chosen on other task" + "\n"
     r"families (0.1 to 100 across penalties with the batch-conditioned decoder); check biology directly at the chosen strength; and use" + "\n"
     r"scANVI when cell-type labels exist.",
     r"to six tries raised each penalty's mean selection score by 0.012 to 0.053; plain scVI with the batch-conditioned decoder remained" + "\n"
     r"a strong candidate, since a penalty beat it mainly when several strengths were tried and scored on the same data (an optimistic" + "\n"
     r"reading); the strengths chosen on other task families lay between 0.1 and 100 across penalties with the batch-conditioned" + "\n"
     r"decoder; the biology was worth checking directly at the chosen strength; and scANVI scored higher than every penalty on most" + "\n"
     r"tasks when cell-type labels existed, although it is trained with labels that several biology metrics also use."),
]

lit = [
    (r"- arXiv API: the request timed out (60 s) and was not repeated; arXiv preprints are indexed by OpenAlex and are covered through it.",
     r"- arXiv API: the request timed out (60 s) and was not repeated, so the direct arXiv search was not completed. The OpenAlex results" + "\n"
     r"  may include arXiv preprints that OpenAlex indexes (arXiv DOIs appear among the records below), but this does not show that every" + "\n"
     r"  relevant arXiv preprint was retrieved (corrected in Pass 12; the paper names only OpenAlex and Europe PMC for this repeat)."),
]

edit("paper/outline.tex", outline)
edit("revision/reviewer_response.md", response)
edit("paper/lit_search_20261105.md", lit)
print(len(outline), "outline edits,", len(response), "response edits,", len(lit), "search-log edit;", len(rows), "ledger rows checked")
