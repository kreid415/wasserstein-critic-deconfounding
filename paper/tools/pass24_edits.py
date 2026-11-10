"""Pass 24 edits (2026-11-09, tracked id P24): pre-submission edits from the two referee reviews (merged review sections A-C,
approved by the first author on 2026-11-09; Part b implements D1, re-reading label transfer and the direct biology check at
the first crossing). Change log paper/pass24_changelog.md; numbers in paper/pass24_numbers.csv.
D2: 'strength mattered more than penalty choice' becomes the narrower statement (similar biology at matched batch removal,
differences mainly at strong settings; 0.025 and 0.213 are ranges between the highest and lowest penalty). D3: Res-6 is prose
(counts once, 'scored above', near-tie caveat, no own-best count, scANVI and scvi-tools adversary sentences). D4: Figure 3 (dot
summary) moves to Supplementary Figure S7; the hypervolume panels of Figure 4 become Supplementary Figure S8; main figures are
renumbered 1-5 (frontier 3, collapse rate 4, placement 5). D5: abstract collapse rates are the Res-4 medians. D6-D8: limitation
sentences (aggregation, the barycenter critic sets b*, the cap and the collapse range, tested tool settings) and Supplementary
Table S15 (sensitivity settings, written by paper/tools/make_supp_tables.py). Section B fixes; section C cuts (detail moved to
Supplementary Notes 1, 4, 10 and 13, which already hold most of it); the abstract becomes three paragraphs; the cross-task mean
macro F1 leaves Notes 5 and 9 (Table S11: paper/tools/make_supp_supp-analysis.py) and is recorded in Note 10.
Each edited line of the outline is rewritten as word-level tracked changes against its printed (accepted) old text, as in
Passes 22 and 23; supplement files are edited in place. Every old text must occur exactly once or the script stops, and it
stops if the main text (Background to Conclusions, figures excluded) exceeds 5,500 words or the main captions 450.
Run from the repository root:  python paper/tools/pass24_edits.py   (PASS24_ROOT overrides the root of the edited files)
Then: python paper/tools/accept_changes.py ; python paper/tools/check_ledger.py"""
import os
import re
import sys
import subprocess
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pass22_edits as p22  # noqa: E402
import pass23_edits as p23  # noqa: E402  (its rewrite() and whole-line helpers)
from accept_changes import accept  # noqa: E402
from biology_check_table import summarize, table_lines  # noqa: E402

ROOT = Path(os.environ.get("PASS24_ROOT") or Path(__file__).resolve().parents[2])
p22.TAG = T = "P24"  # rewrite() marks its changes with this tag
MAIN_LIMIT, CAPTION_LIMIT = 5500, 450
MARK = p23.MARK


def words(t):
    """Printed words of LaTeX text (the abstract counter's convention: a formula is one word, citations are not words)."""
    t = re.sub(r"\\cite[pt]?\{[^}]*\}", " ", t)
    t = re.sub(r"\\hd\{[^}]*\}\{[^}]*\}|\\section\*?(\[[^\]]*\])?\{[^}]*\}|\\pp\{[^{}]*\}", " ", t)
    t = re.sub(r"\\(begin|end)\{[^}]*\}(\[[^\]]*\])?", " ", t)
    t = t.replace("\\%", "%").replace("---", "-").replace("--", "-")
    t = re.sub(r"\\[a-zA-Z]+\*?", " ", t).replace("~", " ")
    return re.sub(r"[{}]", "", t).split()


def budget(tex):
    """(main-text words, caption words) of the printed text: Background to the end of Conclusions, comments removed, figure
    environments counted apart (their captions only)."""
    body = accept(tex).split(r"\section[Background]{Background}", 1)[1]
    i = body.index(r"\hd{Conclusions}")
    body = re.sub(r"(?m)(?<!\\)%.*$", "", body[:body.index(r"\hd{", i + 1)])
    caps = re.findall(r"\\caption\{(.*?)\}\s*\\label", body, flags=re.S)
    body = re.sub(r"\\begin\{figure\}.*?\\end\{figure\}", " ", body, flags=re.S)
    return len(words(body)), sum(len(words(c)) for c in caps)


def find(s, key):
    """The one source line (not a TeX comment) whose printed text contains `key` (white space normalised: struck words leave
    double spaces)."""
    key = " ".join(key.split())
    hits = [ln for ln in s.split("\n") if not ln.startswith("%") and key in " ".join(accept(ln).split())]
    assert len(hits) == 1, (key, len(hits))
    return hits[0]


WHOLE = {  # rewritten throughout: one tracked replacement, so the new text is contiguous in the source (ledger anchors)
    r"\qquad \textbf{Penalty-strength sweep.}", "A replication on 895 new fits",
    "The penalties kept similar biology up to the matched level and differed at stronger settings",
    "We scaled batch removal per task and decoder", "We count the fits at strengths whose mean batch score",
    r"\caption{Pareto frontier of each penalty", "Label transfer predicts the cell types of each held-out batch"}


def rewrite_line(s, key, new=None, pairs=()):
    """The line containing `key` gets the printed text `new` (or its printed text with `pairs` applied), as tracked changes."""
    old = find(s, key)
    text = accept(old).strip()
    for a, b in pairs:
        assert text.count(a) == 1, (a, text.count(a))
        text = text.replace(a, b)
    new = new or text
    if key in WHOLE:
        cap = new.startswith(r"\caption{")
        a, b = (text[len(r"\caption{"):-1], new[len(r"\caption{"):-1]) if cap else (text, new)
        line = (r"\caption{" if cap else "") + rf"\replaced[id={T}]{{{b}}}{{{p22.struck(a)}}}" + ("}" if cap else "")
    else:
        line = p23.rewrite(old, new)
    assert " ".join(accept(line).split()) == " ".join(new.split()), key
    return s.replace(old, line, 1)


def drop_item(s, key):
    old = find(s, key)
    return s.replace(old, p22.rewrite(old, None), 1)


def add_after(s, key, *lines):
    old = find(s, key)
    return s.replace(old, "\n".join((old,) + lines), 1)


def edit(path, pairs):
    p = ROOT / path
    s = p.read_text()
    for old, new in pairs:
        assert s.count(old) == 1, (path, s.count(old), old[:100])
        s = s.replace(old, new)
    p.write_text(s)


# ------------------------------------------------------------------ abstract (D2, D3, D5; section C)
ABSTRACT = [
    r"\textbf{Background:} Single-cell batch correction often adds an alignment penalty (a discriminator, a Wasserstein-1 critic "
    r"or a distance between batches) to a generative model, but penalties have rarely been compared on one backbone. We asked "
    r"which penalty keeps the most biology at equal batch removal in one scVI (single-cell variational inference) model, and "
    r"whether any penalty improves on plain scVI or on standard tools.",
    r"\textbf{Results:} In an exploratory comparison on eight tasks of the single-cell integration benchmark, we added a "
    r"Jensen--Shannon discriminator, three Wasserstein-1 critics (reference, pooled, barycenter), maximum mean discrepancy (MMD) "
    r"or the Sinkhorn divergence to scVI, fitted each at ten strengths from 0.1 to 3,000 with the decoder given the batch label "
    r"or not and three seeds, and read biology at matched batch removal. The penalties were near-tied: 160 of 195 within-pair "
    r"biology differences were under 0.02. The pooled critic had the best mean rank with the batch-conditioned decoder (2.20 "
    r"over five real tasks) and MMD with the unconditioned decoder (2.17 over six); on five new seeds the four within-pair "
    r"differences of 0.04 or more kept their sign, but the Sinkhorn divergence and the pooled critic led. Up to the highest "
    r"batch score every penalty reached, the median collapse rate with the "
    r"batch-conditioned decoder was 34\% for the discriminator and 3\% for MMD. At one strength chosen on other task families, "
    r"the penalties scored above batch-conditioned plain scVI on 3 to 5 of six real tasks (median difference $+0.006$), and "
    r"Harmony, Scanorama and sysVI on 1, 0 and 1; scANVI, trained with cell-type labels, scored above every penalty on every "
    r"real task except pancreas.",
    r"\textbf{Conclusions:} On these tasks the penalties kept similar biology at matched batch removal and differed mainly at "
    r"strong settings: the range between the highest and the lowest penalty had a median of 0.025 at matched batch removal and "
    r"0.213 at each penalty's strongest setting. Adding a penalty to batch-conditioned scVI gave at most a small gain over plain "
    r"scVI."]


def abstract(s):
    """The numbered abstract (eight items) is struck and three paragraphs follow it; abstract_words.py counts both forms."""
    b = s.index(r"\begin{enumerate}", s.index(r"\hd{Abstract}"))
    e = s.index(r"\end{enumerate}", b)
    lines = s[b:e].split("\n")
    assert len(lines) == 10 and all(accept(ln).startswith(r"\item ") for ln in lines[1:-1]) and lines[-1] == ""
    struck = "\n".join([lines[0]] + [p22.rewrite(ln, None) for ln in lines[1:-1]]) + "\n"
    paras = [rf"\noindent \added[id={T}]{{{t}}}" for t in ABSTRACT]  # the space keeps \noindent apart from the accepted text
    return s[:b] + struck + "\\end{enumerate}\n" + "\\par\n".join(paras) + s[e + len(r"\end{enumerate}"):]


# ------------------------------------------------------------------ Background and Methods
BACKGROUND_TAIL = ("A benchmark at fixed strengths", "A replication with five new training seeds on the same tasks checks the "
                   "sweep's readings at matched batch removal (Methods).")

METHODS = {
    r"\qquad For a fair comparison all methods share one backbone": (
        r"\qquad Plain scVI and the six penalties share one backbone, scVI \citep{lopez2018deep}, implemented in the scvi-tools "
        r"python package \citep{gayoso2022python}; the standard tools use their own models (below). These fits use scvi-tools' "
        r"default settings and likelihood, with epochs, minibatch size and latent size equal within a task. The adversary update "
        r"steps and their optimiser follow each method's standard setting: the reference, pooled and barycenter critics take five "
        r"update steps per encoder step with a gradient penalty, as is usual for Wasserstein critics "
        r"\citep{gulrajani2017improved}, and the discriminator takes one (Supplementary Note 2). We use two settings: a "
        r"batch-conditioned setting in which the batch label is fed to the decoder, as scVI is usually run, and an unconditioned "
        r"setting in which no batch label reaches the encoder or decoder, so the alignment penalty is the only correction (Figure "
        r"1a). Hyperparameters and training details are in Supplementary Table S2 and Supplementary Note 2."),
    r"\qquad We benchmark the methods on eight integration tasks": (
        r"\qquad We benchmark the methods on eight integration tasks of the single-cell integration benchmark (scIB) "
        r"\citep{luecken2022benchmarking}, used as distributed with their batch and cell-type labels: six real datasets, four of "
        r"single-cell RNA sequencing (human pancreas, lung, immune cells, and joint human--mouse immune cells) and two of "
        r"chromatin accessibility (assay for transposase-accessible chromatin, ATAC; mouse brain, a small and a large dataset, as "
        r"gene-activity matrices), and two simulated datasets (Supplementary Table S1). RNA tasks use 2,000 highly variable genes "
        r"and the ATAC tasks all features; counts are used as distributed. The batch score is the mean of five scIB batch "
        r"metrics, among them the k-nearest-neighbor batch-effect test (kBET) \citep{buttner2019test}, and the biology score the "
        r"mean of the task's scIB biology metrics (Supplementary Note 3). The selection score is $0.4\,\mathrm{batch} + "
        r"0.6\,\mathrm{biology}$, the weights of scIB's overall score, but the metrics are averaged without scIB's min--max "
        r"scaling across methods, so a fit's scores do not depend on which other fits are scored."),
    r"\qquad \textbf{Penalty-strength sweep.}": (
        r"\qquad \textbf{Penalty-strength sweep.} We fit plain scVI and the six penalties on all eight tasks at ten strengths in "
        r"half-decade steps from 0.1 to 3,000, with both decoders and three seeds (100--102). Edge strengths were added in two "
        r"places (0.01 and 0.03 for the discriminator with the unconditioned decoder; 10,000 and 30,000 for the Sinkhorn "
        r"divergence with the batch-conditioned decoder), and three penalties were also fitted at 0.03 on the immune task with "
        r"the batch-conditioned decoder: 3,033 fits (Supplementary Note 4). A fit fails if its training loss or its latent "
        r"becomes non-finite; failed fits are counted and left out of seed means (Supplementary Table S5). The sweep was planned "
        r"as a pilot for a fixed-strength benchmark that was not run (Supplementary Notes 4 and 10), so every analysis reads all "
        r"strengths of the sweep, and the comparison is exploratory: it describes each penalty over its whole strength range on "
        r"these tasks and tests no hypothesis fixed in advance."),
    r"\qquad Strength has a different scale": (
        r"\qquad Strength has a different scale for each penalty, and a stronger penalty trades biology for batch removal along a "
        r"curve, as Li et al.\ showed for scVI \citep{li2022trade}, so we do not compare methods at a fixed strength. Instead, "
        r"for each task and decoder we set a batch score, $b^*$, halfway between plain scVI's mean batch score and the highest "
        r"mean batch score that every one of the six penalties reaches at a sweep strength with no failed fit, and read each "
        r"penalty's biology score there (Figure 1b; Supplementary Note 5). A task and decoder pair is low-signal when that gap "
        r"is under 0.02, so that $b^*$ sits less than 0.01 above plain scVI and cannot be placed reliably; the low-signal pairs, "
        r"the immune task and both simulations with the batch-conditioned decoder, are not ranked and are shown in separate "
        r"tables (gaps in Supplementary Note 5)."),
    r"\qquad \textbf{Replication across training seeds.}": (
        r"\qquad \textbf{Replication across training seeds.} A smaller replication replaced the planned fixed-strength benchmark. "
        r"For every ranked task and decoder pair (13 of 16) and every penalty, we refitted with five new seeds (0--4) the two "
        r"sweep strengths that bracket the first crossing of $b^*$ (next paragraph), with $b^*$ fixed at its sweep value, and "
        r"plain scVI: 855 planned fits, chosen from the sweep's batch scores alone (Supplementary Note 4). Biology at $b^*$ is "
        r"read on the new-seed means as in the sweep. Before any replication fit was scored, the sweep's readings and ranks were "
        r"written down as the prediction and the agreement rule was fixed (Supplementary Note 10): the four within-pair "
        r"differences of 0.04 or more must keep their sign, and the mean Kendall's rank correlation ($\tau_b$) between sweep and "
        r"replication ranks over the 13 pairs must be above zero by a one-sided sign-flip permutation test ($p<0.05$). The rule's "
        r"extension step, the test's assumptions and two later sensitivity analyses are in Supplementary Note 13."),
    r"\qquad The main analysis is a descriptive ranking.": (
        r"\qquad The main analysis is a descriptive ranking. For each task and decoder, a penalty's curve joins plain scVI and its "
        r"sweep strengths by seed-mean scores over the three seeds, and its biology score is read at $b^*$ by linear "
        r"interpolation where the curve first reaches $b^*$ (Figure 1b). A leave-one-seed-out standard error and rank range, with "
        r"the same seed left out of the penalty and of plain scVI and $b^*$ held fixed, describe how each reading moves with the "
        r"training seeds of the same data (Supplementary Note 6). Within each ranked (not low-signal) pair the six penalties are "
        r"ordered by biology at $b^*$ (rank 1 is best; ties share the average rank), and as in scIB "
        r"\citep{luecken2022benchmarking} the mean rank over the ranked real tasks is reported for each decoder; simulations are "
        r"ranked separately. The ranking has no significance test. Its rules were written for the planned fixed-strength "
        r"benchmark and applied to the sweep with the changes listed in Supplementary Note 10, before the sweep scores of five of "
        r"the eight tasks existed and after those of the first three had been read by the strength-choice rules (Supplementary "
        r"Notes 10 and 11)."),
    r"\qquad We place the penalties against plain scVI": (
        r"\qquad We place the penalties against plain scVI and against standard tools: scvi-tools' own adversarial classifier "
        r"(batch-conditioned decoder only), Harmony \citep{korsunsky2019fast}, Scanorama \citep{hie2019efficient}, sysVI "
        r"\citep{hrovatin2025integrating}, scANVI, which is trained with the cell-type labels \citep{xu2021probabilistic}, and "
        r"principal-component analysis (PCA) as the uncorrected reference, allowing for the outcome that no alignment penalty "
        r"improves on plain scVI. Harmony, Scanorama and sysVI run at six settings each, Harmony and Scanorama with 10 "
        r"dimensions, and scANVI and PCA at one; three settings at each tool's default size (Harmony at 50 principal components, "
        r"Scanorama at 100 dimensions and PCA at 50 components) are sensitivity checks (Supplementary Table S4 and Supplementary "
        r"Note 8). Placement is descriptive and compares each method at one setting not tuned on that task: for a penalty, the "
        r"strength with the best mean selection score on the real tasks of the other task families, shown beside the task's own "
        r"best strength as an optimistic bound (Supplementary Note 7), and for Harmony, Scanorama and sysVI the setting chosen by "
        r"the same rule (Supplementary Note 8)."),
    r"\qquad For each penalty we also report four descriptive views": (
        r"\qquad For each penalty we also report four descriptive views: collapse rate, hypervolume, share of non-dominated "
        r"strengths and label transfer (rules in Supplementary Notes 7 and 9). A fit collapses if it fails or its biology score is "
        r"at least 0.10 below the same seed's plain scVI; the collapse rate is the collapsed share of a penalty's fits in a task "
        r"and decoder at strengths whose mean batch score is at most the highest common batch score, with nominal exact binomial "
        r"(Clopper--Pearson) 95\% intervals \citep{clopper1934use}. The hypervolume \citep{zitzler1998multiobjective}, which Li "
        r"et al.\ used to compare trade-off fronts \citep{li2022trade}, is the area under a penalty's Pareto frontier (its best "
        r"biology at or beyond each batch level) above 0.10 below plain scVI's biology, up to the highest common batch score and "
        r"divided by the batch gap that sets $b^*$ (Figure 3; Supplementary Figure S8). In all 11 ranked real pairs the "
        r"barycenter critic's highest batch score is the highest common batch score, so this one penalty sets $b^*$, the cap of "
        r"the hypervolume and the range of the collapse rate; the cap and the range were chosen after the scores had been read "
        r"(Supplementary Note 10; uncapped values and rates over all strengths in Supplementary Table S8). The share of "
        r"non-dominated strengths, also from Li et al., is the fraction of a penalty's strengths that none of its other strengths "
        r"beats on both scores."),
    r"\qquad \emph{Use of large language models.}": (
        r"\qquad \emph{Use of large language models.} Claude (Anthropic), through the Claude Science and Claude Code "
        r"applications, and Codex (OpenAI) wrote and checked analysis, scoring and figure code; prepared, ran and monitored "
        r"computational jobs; assisted with literature searches; checked the numbers in the text against their source files; "
        r"reviewed the text; and drafted and edited text. The authors made or approved the study design and every decision about "
        r"what was run, measured and reported. All figures are drawn from the data by plotting scripts in the repository; no "
        r"image was made by a generative model. The authors checked all outputs and take responsibility for the content."),
}

PENALTY_PAIRS = [
    (r"the encoder minimizes scvi-tools' bounded fool loss, a Jensen--Shannon (JS) type objective \citep{goodfellow2014generative}.",
     r"the encoder minimizes scvi-tools' cross-entropy fool loss, a Jensen--Shannon (JS) type objective "
     r"\citep{goodfellow2014generative} that is not bounded above and is not the JS divergence itself (Supplementary Note 1)."),
    ("(the Shannon entropy of a batch's cell-type proportions, computed from the cell-type labels; ties go to the larger batch). "
     "The entropy rule prefers the batch whose cell types are most evenly represented, so that the "
     "target does not favour a few cell types, as a merely large batch, the other obvious choice, may do when a few cell types "
     "dominate it. The chosen reference holds every cell type in five of the eight tasks, and 10 of 17 (lung), 14 of 19 "
     "(human--mouse immune) and 3 of 4 (simulation 2) in the others, so in those three tasks some cells of other batches have no "
     "cell of their type in the reference.",
     "computed from the cell-type labels; in three tasks it lacks some cell types (Supplementary Note 1)."),
    (r"The barycenter critic \citep{cuturi2014fast} estimates an average distribution of the batches present in each minibatch, "
     "recomputed at every step, and aligns every batch to it.",
     r"The barycenter critic \citep{cuturi2014fast} aligns every batch to an average distribution of the batches present in "
     "each minibatch.")]

# ------------------------------------------------------------------ Results
RESULTS = {
    "The order of the penalties depended on the decoder:": (
        "The order of the penalties depended on the decoder. With the batch-conditioned decoder, the ranking rested on five real "
        "tasks: the pooled critic had the best mean rank, 2.20, followed by the Sinkhorn divergence, 2.40, and MMD and the "
        "discriminator had the two worst, 4.60 and 5.00. With the unconditioned decoder, all six real tasks contributed: MMD had the "
        "best mean rank, 2.17, and the Sinkhorn divergence the worst, 4.67."),
    "A replication on 895 new fits": (
        r"A replication on 895 new fits (seeds 0--4: the 855 planned fits and 40 of the extension step that the rule prescribes) "
        r"met its agreement rule (all four large sweep gaps kept their sign; mean Kendall's $\tau_b$ 0.456, $p=0.00049$), but "
        r"the first-ranked penalty changed in 7 of 13 pairs, and on the new seeds the Sinkhorn divergence led by mean rank with "
        r"the batch-conditioned decoder (2.20) and the pooled critic with the unconditioned decoder (2.67) (Supplementary Note "
        r"13; Supplementary Table S9)."),
    r"\pp{Res-3}{": (
        r"\pp{Res-3}{The penalties kept similar biology up to the matched level and differed mainly at their strongest fitted "
        r"settings (trade-off curves in Supplementary Figure S6, summarised in Supplementary Figure S7).}"),
    "The penalties kept similar biology up to the matched level and differed at stronger settings": (
        r"Over the 11 ranked real task and decoder pairs, the range of biology between the highest and the lowest penalty had a "
        r"median of 0.025 at $b^*$, against 0.213 at each penalty's strongest fitted setting, where the penalties reached "
        r"different batch scores. The barycenter critic reached the lowest biology score of the six penalties in all 12 real "
        r"task and decoder pairs, the low-signal immune pair included (0.287 to 0.516); on lung with the batch-conditioned "
        r"decoder its strongest setting ended below plain scVI on both scores (batch 0.466 and biology 0.397, against 0.633 and "
        r"0.722). With the unconditioned decoder, the chromatin-accessibility curves lost biology as batch removal increased, "
        r"whereas on pancreas every penalty raised both scores over part of its strength range."),
    "We scaled batch removal per task and decoder": (
        r"With batch removal scaled per task and decoder from plain scVI (0) through $b^*$ (0.5) to the highest batch score every "
        r"penalty reaches (1), the median frontier over the ranked real tasks stayed within 0.019 of plain scVI's biology up to "
        r"$b^*$ for every penalty and decoder. At the highest common batch score, the discriminator's median frontier with the "
        r"batch-conditioned decoder was 0.082 below plain scVI, against 0.006--0.036 for the other penalties; with the "
        r"unconditioned decoder, the six frontiers ended 0.007--0.029 below. The hypervolume up to that level had medians of "
        r"0.086--0.102 for every penalty and decoder except the discriminator with the batch-conditioned decoder, 0.063 "
        r"(Supplementary Figure S8; Supplementary Table S8). The barycenter critic's highest batch score was the highest common "
        r"batch score in all 11 ranked pairs, whereas every other penalty reached beyond it (Supplementary Note 7)."),
    "We count the fits at strengths whose mean batch score": (
        r"We count the fits at strengths whose mean batch score is at most the highest common batch score, strong settings whose "
        r"batch score fell back below it included; rates over all strengths are in Supplementary Table S8. With the "
        r"batch-conditioned decoder, the median collapse rate over the six real tasks ranged from 34\% for the discriminator to "
        r"3\% for MMD, and with the unconditioned decoder from 20\% for the barycenter critic to 0\% for three penalties. The "
        r"highest rate was the discriminator's with the batch-conditioned decoder, 56\% (31\%--78\%, nominal 95\% interval) on the "
        r"human--mouse immune task. MMD collapsed less often than the discriminator on 5 of the 6 real tasks with that decoder, "
        r"although it ranked above the discriminator at $b^*$ on only 3 of the 5 ranked tasks. Every collapse was a loss of "
        r"biology; no fit failed. Over all strengths, 67 of the discriminator's 91 collapses with the batch-conditioned decoder "
        r"and 47 of 59 with the unconditioned decoder lay beyond the highest common batch score, against none of the barycenter "
        r"critic's, whose highest batch score sets that limit. These summaries include the low-signal immune pair."),
    r"\pp{Res-6}{": (
        r"\pp{Res-6}{At one strength chosen on other task families, the penalties gave at most a small gain over "
        r"batch-conditioned plain scVI (Figure 5; Supplementary Table \ref{tab:4}).}"),
}
RES6 = (r"With the batch-conditioned decoder, the "
        r"penalties scored above plain scVI's selection score on 3 (discriminator), 4 (reference critic), 5 (pooled critic), 5 "
        r"(barycenter critic), 4 (MMD) and 4 (Sinkhorn divergence) of the six real tasks. The 36 differences from plain scVI ran "
        r"from $-0.023$ to $+0.045$ (median $+0.006$), and 11 of them were smaller than 0.005 in size, against seed standard "
        r"deviations of the penalty's selection score of 0.001 to 0.016 in those cells, so these counts mix small gains, small "
        r"losses and near-ties (Supplementary Table \ref{tab:4} prints the Sinkhorn divergence on pancreas, $-0.0004$, as a tie). "
        r"The transferred strengths were 0.1--0.3 for the discriminator, 0.3--3 for the reference and barycenter critics, 1 for "
        r"the pooled critic, 30 for MMD and 1--100 for the Sinkhorn divergence. The scvi-tools adversary scored above plain scVI "
        r"on pancreas, immune and the large chromatin-accessibility task. Harmony, Scanorama and sysVI at their transferred "
        r"settings scored above batch-conditioned plain scVI on 1, 0 and 1 of the six real tasks, and at each tool's default size "
        r"Harmony with 50 principal components and Scanorama with 100 dimensions on 3 and 1 (Supplementary Table S15), so these "
        r"results hold for the settings tested. scANVI, trained with the cell-type labels that several biology metrics also use, "
        r"scored above every transferred-strength penalty on every real task except pancreas. The unconditioned decoder, whose "
        r"plain scVI receives no batch information, is in Supplementary Table S7.")

CAPTIONS = {  # main-figure captions (section C: at most 450 words in total)
    r"\caption{Model and read-out.": (
        r"\caption{Model and read-out. (a) Plain scVI and the six penalties share one scVI encoder--decoder; the decoder receives "
        r"the batch label (batch-conditioned) or not (unconditioned), and the penalty acts on the 10-dimensional latent "
        r"embedding. The loss is the scVI loss plus the strength times the penalty; strength 0 is plain scVI. (b) ATAC small "
        r"with the unconditioned decoder, three of the six penalties: each curve joins plain scVI (star) and the sweep strengths "
        r"in increasing order, means over three seeds. Biology is read where the curve first reaches the matched batch removal "
        r"$b^*$ (dashed line), interpolating between neighbouring strengths (open circles; dashed arrows); later crossings are "
        r"not used.}"),
    r"\caption{Ranking of the six penalties": (
        r"\caption{Ranking of the six penalties by biology at matched batch removal $b^*$. (a) Biology score at $b^*$ (colour) "
        r"and rank within each task row (number) for each real task, batch-conditioned decoder above and unconditioned decoder "
        r"below; hatched row, the low-signal immune pair, not ranked. (b) Mean rank over the ranked real tasks (five with the "
        r"batch-conditioned decoder, six with the unconditioned decoder), penalties sorted from best to worst; dot, mean rank; "
        r"bar, lowest to highest rank over those tasks; rank 1 (best) at the right. Leave-one-seed-out standard errors and rank "
        r"ranges are in Supplementary Table S6; simulations in Supplementary Figure S1.}"),
    r"\caption{Pareto frontier of each penalty": (
        r"\caption{Pareto frontier of each penalty: median over the ranked real tasks (five with the batch-conditioned decoder, "
        r"a; six with the unconditioned decoder, b) of the best biology score reachable at each level of batch removal or beyond, "
        r"minus plain scVI's, from seed means. Batch removal is scaled per task from plain scVI (0) through $b^*$ (0.5, dashed "
        r"line) to the highest batch score every penalty reaches (1). The frontiers are assembled after scoring and are "
        r"optimistic; hypervolumes are in Supplementary Figure S8.}"),
    r"\caption{Collapse rate per penalty": (
        r"\caption{Collapse rate per penalty, batch-conditioned (a) and unconditioned decoder (b): the percentage of fits that "
        r"lose 0.10 or more biology against the same seed's plain scVI, at strengths whose mean batch score is at most the "
        r"highest batch score every penalty reaches (lower is better). One point per real task, in the order of Figure 2a, with "
        r"its nominal exact 95\% interval; black tick, median over the six real tasks, the ringed low-signal immune pair "
        r"included. Rates over all strengths are in Supplementary Table S8.}"),
    r"\caption{Placement against plain scVI": (
        r"\caption{Placement against plain scVI (star) and standard tools with the batch-conditioned decoder: batch against "
        r"biology score per real task at the strength or setting chosen on other task families (penalties, Harmony, Scanorama, "
        r"sysVI) or the fixed setting (scANVI, scvi-tools adversary, PCA with 10 dimensions). Selection scores are in "
        r"Supplementary Table \ref{tab:4}; the unconditioned decoder is in Supplementary Table S7.}"),
}


def renumber_figures(s):
    """D4: Figure 3 (the dot summary) leaves the main text for Supplementary Figure S7; Figures 4-6 become Figures 3-5. The
    figure scripts and files were renamed to match (paper/figure_scripts, make_figures.sh)."""
    i = s.index(MARK + "Figure 3\n")
    b = s.index(r"\begin{figure}", i)
    e = s.index(r"\end{figure}", b) + len(r"\end{figure}")
    assert "figures/fig3.png" in s[b:e] and "\\caption{Biology kept by each penalty" in accept(s[b:e])
    commented = "".join("%" + ln + "\n" for ln in s[b:e].split("\n"))
    note = rf"\noindent\deleted[id={T}]{{Figure 3 (biology relative to plain scVI) moved to Supplementary Figure S7.}}" + "\n"
    s = s[:i] + MARK + "Figure 3 (dot summary; Supplementary Figure S7 since Pass 24)\n" + note + commented + s[e:].lstrip("\n")
    for mark, new, a, b in [("Figure 4 (Pass 23: Pareto frontier)\n", "Figure 3 (Pareto frontier; Figure 4 in Pass 23)\n", 4, 3),
                            ("Figure 5\n", "Figure 4 (collapse rate; Figure 5 in Pass 23)\n", 5, 4),
                            ("Figure 6\n", "Figure 5 (placement; Figure 6 in Pass 23)\n", 6, 5)]:
        i = s.index(MARK + mark)
        e = s.index(r"\end{figure}", i)
        blk = s[i:e]
        assert blk.count(f"figures/fig{a}.png") == blk.count(f"\\label{{fig:{a}}}") == 1
        blk = blk.replace(MARK + mark, MARK + new).replace(f"figures/fig{a}.png", f"figures/fig{b}.png")
        s = s[:i] + blk.replace(f"\\label{{fig:{a}}}", f"\\label{{fig:{b}}}") + s[e:]
    return s


# ------------------------------------------------------------------ Discussion and Conclusions
DISCUSSION = {
    r"\pp{Disc-1}{": (
        r"\pp{Disc-1}{The penalty with the best mean rank depended on the decoder: the pooled critic with the batch-conditioned "
        r"decoder, closely followed by the Sinkhorn divergence, and MMD with the unconditioned decoder. Task-level orders varied, "
        r"many neighbouring biology readings were within a leave-one-seed-out standard error, and on new seeds of the same tasks "
        r"the replication kept the four large within-pair gaps but changed the leaders (Res-2b; Supplementary Note 13), so these "
        r"descriptive rankings do not establish a consistent ordering across tasks. The secondary summaries gave different orders "
        r"again: MMD had relatively low collapse rates despite ranking near the bottom with the batch-conditioned decoder. The "
        r"rankings are therefore descriptions of these tasks, implementations and score definitions.}"),
    r"\pp{Disc-3}{": (
        r"\pp{Disc-3}{For users of scVI, the penalties kept similar biology at matched batch removal and differed mainly at "
        r"strong settings. At $b^*$, 160 of the 195 within-pair differences in biology over the 13 ranked pairs were under 0.02, "
        r"and 4 were 0.04 or more. The range between the highest and the lowest penalty was much wider at each penalty's "
        r"strongest fitted setting (Res-3), but those settings reached different batch scores and depend on the top of the "
        r"strength grid. Added to batch-conditioned scVI, a penalty at the strength chosen on other task families gave at most a "
        r"small gain (Res-6). The comparisons with standard tools hold for the settings tested, and scANVI's use of the labels "
        r"limits a direct comparison with methods that do not train on them.}"),
    "Six real tasks in four families": (
        r"\item The six real tasks form four families, two of them nested pairs (the human--mouse immune task contains all immune "
        r"cells, and the large ATAC task contains the small one), and related tasks count as separate tasks in the mean rank."),
    r"\item The comparison is exploratory. The strength grid was fixed in advance": (
        r"\item The comparison is exploratory: the strengths of the placement and of the replication were chosen from sweep "
        r"scores, and the ranking rules were applied to fits that the strength-choice rules had already read (Methods). With "
        r"three seeds per strength, the leave-one-seed-out standard errors describe the spread over three repeats of the same "
        r"fits and carry no coverage guarantee (Supplementary Note 10). The replication refits the same eight tasks with new "
        r"seeds, not new data, and tests only the sign of the within-pair differences of 0.04 or more."),
    r"\item All fits use one backbone, scVI,": (
        r"\item The penalty fits use one backbone, scVI, with one fixed architecture and latent size (10 latent dimensions, one "
        r"hidden layer of 128 units), so the ranking may not hold for other backbones, architectures or latent sizes, or for "
        r"integrations at atlas scale. The biology checks, including label transfer, are exploratory: they were defined after the "
        r"sweep scores had been read."),
    r"\item The strength sweep is what Li et al.\ call scalarization": (
        r"\item The strength sweep is what Li et al.\ call scalarization \citep{li2022trade}: it describes the penalties as users "
        r"run them, not the best trade-off front that a multi-objective solver could reach for each penalty. Li et al.'s "
        r"adversarially trained mutual-information penalty (MINE) was not tested here."),
    r"\item One batch variable per task;": (
        r"\item Each task has one batch variable, and the ATAC tasks enter as gene-activity matrices. Counts are used as "
        r"distributed, including non-integer values in some batches, and scoring includes the training cells, as in scIB "
        r"(Supplementary Note 3). The Sinkhorn divergence at the smoothing used is not an estimate of W1, and the reference "
        r"critic's reference batch is chosen with the cell-type labels."),
    r"\item The collapse-rate intervals are nominal": (
        r"\item The collapse-rate intervals are nominal exact binomial 95\% intervals. They treat a penalty's fits in a task and "
        r"decoder (6 to 33 up to the highest common batch score) as independent, although each seed's fits are judged against the "
        r"same plain-scVI threshold and can collapse together, so their actual coverage may differ from 95\%; an illustrative "
        r"simulation is in Supplementary Note 7."),
    r"\item Not varied or tested:": (
        r"\item The network architecture and latent size, the Kullback--Leibler (KL) warm-up (which does not complete on the four "
        r"shorter tasks, while the penalty acts at full strength from the first step), the choice of reference batch and the "
        r"number of batches were held fixed. Effects of a shift in cell-type composition between batches were not tested."),
    r"\item On eight tasks of the single-cell integration benchmark, six alignment penalties": (
        r"\item On eight tasks of the single-cell integration benchmark, six alignment penalties added to one scVI model kept "
        r"similar biology at matched batch removal: the best mean rank was the pooled critic's with the batch-conditioned decoder "
        r"(2.20) and MMD's with the unconditioned decoder (2.17), but 160 of the 195 within-pair differences were under 0.02, so "
        r"the ranking does not separate most penalties, and on new training seeds the Sinkhorn divergence and the pooled critic "
        r"led instead. At the strength chosen on other task families with the batch-conditioned decoder, the penalties scored "
        r"above plain scVI on 3 to 5 of six real tasks (median difference $+0.006$), whereas Harmony, Scanorama and sysVI at the "
        r"settings tested did so on 1, 0 and 1 of six."),
}
CONCLUSIONS_PAIRS = [  # item (iii), the direct biology check, waits for D1 and is not edited
    ("(i) the strength mattered more than the choice of penalty (biology at $b^*$ differed between penalties by a median of 0.025, "
     "against 0.213 at each penalty's strongest fitted setting), and plain scVI with the batch-conditioned decoder remained a "
     "strong candidate, since at the strength chosen on other task families a penalty exceeded it on 3 to 5 of six real tasks, "
     "with a median difference of $+0.006$; (ii) the strengths chosen on other task families with that decoder lay between 0.1 "
     "and 100 across penalties (Results);",
     "(i) the penalties differed mainly at strong settings, not at matched batch removal (Res-3); (ii) plain scVI with the "
     "batch-conditioned decoder remained a strong candidate (Res-6);"),
    ("scANVI scored higher than every penalty on most tasks,", "scANVI scored higher than every penalty on every real task except "
     "pancreas (Res-6),")]
DISC4_NEW = [  # D6, D7 and D8, after the item on the exploratory design
    r"The ranking depends on how the scores are aggregated. The metrics are averaged without scIB's min--max scaling; in the "
    r"scaling checks, min--max scaling and z-scores put the Sinkhorn divergence first with the batch-conditioned decoder, and "
    r"z-scores put the discriminator first with the unconditioned decoder (Supplementary Figure S5).",
    r"One penalty sets the scale of the comparison: in all 11 ranked real pairs the barycenter critic's highest batch score is "
    r"the highest common batch score, so it fixes $b^*$, the cap of the hypervolume and the range of the collapse rate. The cap "
    r"and that range were chosen after the scores had been read; the uncapped hypervolumes and the collapse rates over all "
    r"strengths are in Supplementary Table S8 (Supplementary Note 10).",
    r"The conclusions about standard tools hold for the settings tested: Harmony and Scanorama ran with 10 dimensions, and at "
    r"each tool's default size their selection scores were above plain scVI on more tasks (Supplementary Table S15)."]
ABBREVIATIONS = (r"ATAC, assay for transposase-accessible chromatin; GAN, generative adversarial network; JS, Jensen--Shannon; "
                 r"kBET, k-nearest-neighbour batch-effect test; KL, Kullback--Leibler; MMD, maximum mean discrepancy; OT, "
                 r"optimal transport; PCA, "
                 r"principal-component analysis; RNA-seq, RNA sequencing; scIB, single-cell integration benchmark; scIB-E, extended "
                 r"single-cell integration benchmark; scVI, single-cell variational inference; SD, standard deviation; W1, "
                 r"Wasserstein-1; WGAN-GP, Wasserstein generative adversarial network with gradient penalty.")

# ------------------------------------------------------------------ supplement lists
STRENGTH_RULE = (r"of the two fitted strengths that bracket the first crossing of $b^*$, the one whose seed-mean batch score "
                 r"is nearer $b^*$ (ties to the smaller strength), or the upper one when the lower is plain scVI")
RES5 = (
    r"At first-crossing strengths (Methods), label-transfer macro F1 with the unconditioned decoder was below plain scVI's "
    r"for the discriminator on five of six real tasks (median difference $-0.002$), and lowest among the penalties on three. "
    r"With the batch-conditioned decoder its median difference was 0.000 over the five ranked tasks. The other penalties' "
    r"median differences were $-0.006$ to $+0.001$ with either decoder. Their label-transfer order did not follow their biology "
    r"order at $b^*$ (median Kendall $\tau_b=-0.067$ over the 11 ranked pairs). The selected embeddings were near $b^*$, "
    r"rather than exactly matched (batch-score distance at most 0.073, median 0.005); plain scVI uses its own unmatched embedding "
    r"(Supplementary Note 9). The direct check is exploratory and descriptive (Supplementary Note 12). Over the six real tasks, "
    r"mean neighbourhood purity minus plain scVI's was $-0.013$ for the discriminator with the batch-conditioned decoder and "
    r"$-0.001$ with the unconditioned decoder; the other penalties ranged from $-0.005$ to $+0.001$. Harmony, averaged over "
    r"its settings, differed from batch-conditioned plain scVI by $-0.141$, and scANVI, trained with the labels, by $+0.115$. "
    r"Rare types lost purity in 21 of 48 task, penalty and decoder comparisons. Marker-gene overlap was 0.390 to 0.399 for "
    r"every penalty and plain scVI, with differences at most 0.007. These readouts show small average changes at the first "
    r"crossing, without a consistent disadvantage for the discriminator across decoders. The earlier reading, which took some "
    r"penalties at strengths far past the first crossing, had shown larger losses for the discriminator (both readings and "
    r"per-cell-type detail in Supplementary Note 12).")
NOTE10_STRENGTH = (
    r"\item \emph{Strength rule changed after the first readings.} On 9 November 2026, after the first label-transfer and "
    r"direct-biology-check results had been read, the first author changed their strength rule from the failure-free strength "
    r"nearest $b^*$ on any crossing to the first-crossing rule: " + STRENGTH_RULE + r". This makes their strengths follow the "
    r"crossing used by the ranking; batch scores alone choose the strength. Both readings are given in Supplementary Note 12. "
    r"The earlier label-transfer table and summary are kept as \texttt{label\_transfer\_nearest\_any\_crossing.csv} and "
    r"\texttt{res5\_label\_transfer\_nearest\_any\_crossing.json} in \texttt{paper/figure\_data}; the earlier biology "
    r"tables are in \texttt{paper/figure\_data/biology\_check\_nearest\_any\_crossing/tables}. The new strengths and biology "
    r"outputs are \texttt{first\_crossing\_strength.csv} and \texttt{biology\_check\_first\_crossing/} there.")


def first_crossing_biology(s):
    s = rewrite_line(s, r"\qquad  Label transfer holds", r"\qquad Label transfer predicts each held-out batch's cell types "
                     r"from the other batches with a 15-nearest-neighbour classifier. For each penalty the rule is: " + STRENGTH_RULE +
                     r". This rule replaced the strength nearest $b^*$ on any crossing after the first results had been read "
                     r"(Supplementary Notes 9 and 10); accuracy never chooses the strength.")
    s = rewrite_line(s, r"\qquad A direct biology check reads", r"\qquad A direct biology check reads neighbourhood purity, "
                     r"the share of each cell's 15 nearest neighbours with its cell-type label, averaged within types; purity "
                     r"and label-transfer recall for rare types (under 1\% of a task's cells); marker preservation, the Jaccard "
                     r"overlap of each type's top 20 markers from the original and neighbour-averaged expression on the four RNA "
                     r"tasks; and label-transfer confusion. Penalties follow the label-transfer rule: " + STRENGTH_RULE +
                     r". This replaces the earlier strength nearest $b^*$ on any crossing (Supplementary Note 10); baselines use every "
                     r"setting. Differences are from plain scVI, with no new fits. The check is exploratory: its rules followed "
                     r"the sweep scores and its strength rule changed after its first results (Supplementary Note 12).")
    s = rewrite_line(s, r"\pp{Res-5}{", r"\pp{Res-5}{Label transfer and the direct biology check show small average changes "
                     r"from plain scVI at the first crossing (Supplementary Figure S3 and Supplementary Table S10).}")
    s = rewrite_line(s, "Label transfer predicts the cell types of each held-out batch", RES5)
    s = rewrite_line(s, r"\pp{Disc-2}{", pairs=[
        (r"Label-transfer scores did not follow the matched-biology ranking either, and their batch-score distances from $b^*$ "
         r"limit direct comparisons.", r"At first-crossing strengths, label transfer showed small median changes from plain "
         r"scVI and did not follow the matched-biology ranking; unequal batch-score distances from $b^*$ limit comparisons.")])
    s = rewrite_line(s, r"\item Observations for scVI users", pairs=[
        (r"the biology was worth checking directly at the chosen strength, because the discriminator lost the most neighbourhood "
         r"purity in the direct biology check ($-0.052$ against plain scVI, $-0.107$ for rare types, batch-conditioned decoder)",
         r"the direct biology check showed small average changes from plain scVI at first-crossing strengths, without a "
         r"consistent disadvantage for the discriminator across decoders (Res-5)")])
    s = rewrite_line(s, r"\item Repository github.com/kreid415", pairs=[("biology check at commit c4a793f", "biology check at commit 8073db6")])
    s = rewrite_line(s, r"\caption{Direct biology check", pairs=[
        ("Means over the six real tasks", "Penalties use the first-crossing strengths of Supplementary Note 9. Means over the six real tasks")])
    start = s.index("Method & Decoder & Purity diff.")
    end = s.index(r"\bottomrule", start)
    old_rows = s[start:end].splitlines()
    data_rows = [ln for ln in old_rows if " & " in accept(ln) and not ln.startswith("Method &")]
    assert len(data_rows) == 20
    for old, new in zip(data_rows, table_lines(summarize())):
        s = s.replace(old, p23.rewrite(old, new), 1)
    return s


SUPP_LIST_ITEMS = [  # (key of the line, new printed text or None to strike, lines added after it)
    ("Table S12, benchmark ranking (the former main-text Table 2)", [(" (the former main-text Table 2)", "")], []),
    ("Table S13, alignment methods (the former main-text Table 1)", [(" (the former main-text Table 1)", "")], []),
    ("Table S14, single-strength placement with the batch-conditioned decoder (the former main-text Table 3)",
     [(" (the former main-text Table 3)", "")],
     [r"\item \added[id=P24]{Table S15, standard tools at their default-size sensitivity settings: selection score per real "
      r"task for Harmony with 50 principal components, Scanorama with 100 dimensions and PCA with 50 components, beside "
      r"batch-conditioned plain scVI.}"]),
    ("Figure S4, per-metric strength curves", [("Figure S4, per-metric strength curves: each batch and biology metric against "
                                                "strength, per method and decoder.",
                                                "Figures S4a and S4b, per-metric strength curves: each batch and biology "
                                                "metric against strength, per method, with the batch-conditioned (S4a) and "
                                                "the unconditioned decoder (S4b).")], []),
    ("Figure S6, trade-off curves (the former main-text Figure 3)", [(" (the former main-text Figure 3)", "")],
     [r"\item \added[id=P24]{Figure S7, biology relative to plain scVI: biology score minus plain scVI's per penalty and decoder, "
      r"one dot per real task, at matched batch removal $b^*$ and at the strongest fitted setting.}",
      r"\item \added[id=P24]{Figure S8, hypervolume up to the highest common batch score per penalty, task and decoder, with "
      r"the median over the ranked real tasks.}"]),
    (r"\item Figure 1:", [(" First to move to the supplement if pages run short.", "")], []),
]
FIGURE_LIST = [  # main figures 3-5 after D4 (the Figure 6 item is struck first)
    (r"\item Figure 5: collapse rate", r"\item Figure 5: batch against biology score at single strength."),
    (r"\item Figure 4: Pareto frontier", r"\item Figure 4: collapse rate up to the highest common batch score per penalty, one "
                                         r"panel per decoder, one point per real task with its 95\% interval and the median."),
    (r"\item Figure 3: biology score minus", r"\item Figure 3: Pareto frontier per penalty and decoder, the median over the "
                                             r"ranked real tasks of the best biology reachable at each scaled batch level."),
]


def outline(s):
    s = abstract(s)
    s = rewrite_line(s, r"\qquad iMAP trained a critic", pairs=[("iDLC trains a WGAN-GP critic", "iDLC, a preprint, trains a "
                                                                                                  "WGAN-GP critic")])
    tail = accept(find(s, BACKGROUND_TAIL[0])).strip()
    s = rewrite_line(s, BACKGROUND_TAIL[0], pairs=[
        (tail[tail.index(BACKGROUND_TAIL[0]):], BACKGROUND_TAIL[1]),
        ("so differences between methods come from the alignment penalty as it is usually run.",
         "so differences between penalties come from each penalty as it is usually run.")])
    for key, new in list(METHODS.items()) + list(RESULTS.items()) + list(CAPTIONS.items()) + list(DISCUSSION.items()):
        s = rewrite_line(s, key, new)
    s = rewrite_line(s, r"\qquad Each method adds a fixed penalty", pairs=PENALTY_PAIRS)
    s = rewrite_line(s, r"\pp{Res-3b}{", pairs=[("(Figure 4)", "(Figure 3)")])
    s = rewrite_line(s, r"\pp{Res-4}{", pairs=[("(Figure 5;", "(Figure 4;")])
    s = rewrite_line(s, r"\item Observations for scVI users", pairs=CONCLUSIONS_PAIRS)
    s = first_crossing_biology(s)
    for key in (r"\item Rule: each penalty, and each of Harmony", r"\item At the strength chosen on real tasks of other families",
                r"\item The replication across training seeds met its agreement rule", r"\item When to add a penalty:",
                r"\item Figure 6:"):
        s = drop_item(s, key)
    s = add_after(s, r"\pp{Res-6}{At one strength chosen", rf"\added[id={T}]{{{RES6}}}")
    s = add_after(s, r"\item The comparison is exploratory: the strengths of the placement",
                  *[rf"\item \added[id={T}]{{{t}}}" for t in DISC4_NEW])
    s = renumber_figures(s)
    for key, pairs, after in SUPP_LIST_ITEMS:
        s = rewrite_line(s, key, pairs=pairs)
        if after:  # the new list items follow the edited one, whose key loses its history note
            s = add_after(s, re.sub(r" \(the former main-text [^)]*\)", "", key), *after)
    for key, new in FIGURE_LIST:
        s = rewrite_line(s, key, new)
    s = rewrite_line(s, r"\noindent Every supplementary note and table listed here", pairs=[
        ("Tables S6, S7, S9 and S12--S14 and Figures S1--S6, follow", "Tables S6, S7, S9 and S12--S15 and Figures S1--S8, follow")])
    old = r"S4--\replaced[id=P23]{S14}{S11}, Figures S1--S6}"
    assert s.count(old) == 1
    s = s.replace(old, rf"S4--\replaced[id={T}]{{S15}}{{S14}}, Figures S1--\replaced[id={T}]{{S8}}{{S6}}}}")
    decl = r"\hd{\replaced[id=P3]{Declarations}"
    assert s.count(decl) == 1
    s = s.replace(decl, rf"\hd{{\added[id={T}]{{List of abbreviations}}}}{{\added[id={T}]{{required by BMC}}}}" + "\n"
                  + rf"\noindent \added[id={T}]{{{ABBREVIATIONS}}}\par" + "\n\n" + decl)
    anchor = "\\definechangesauthor[name={Pass 23}, color=blue]{P23}"
    assert s.count(anchor) == 1
    return s.replace(anchor, "\\definechangesauthor[name={Pass 24}, color=pink]{P24}% referee reviews before submission; "
                     "change log paper/pass24_changelog.md\n" + anchor)


# ------------------------------------------------------------------ supplement
REPLICATION_RULE = (
    r" The reading is the sweep's: on the means over seeds 0--4, biology at $b^*$ by linear interpolation in batch score between "
    r"the two refitted strengths, walking up from plain scVI at the new seeds and taking the first crossing. The test is "
    r"one-sided over all $2^{13}$ sign patterns ($p<0.05$). It assumes that the 13 pairs' correlations are independent and "
    r"symmetric about zero when the two orders are unrelated; this holds only roughly, because the two decoders of a task use the "
    r"same task data and some tasks are related (two nested pairs), so its $p$-value is approximate. Two sensitivity analyses, "
    r"which flip the signs per task and per task family, were added after the rule was fixed and are reported beside it, not as "
    r"part of it (below). Because the rule tests only the sign of the within-pair differences of 0.04 or more and no precision "
    r"calculation was made, the replication is not set up to separate penalties whose readings differ by less. Sweep (seeds "
    r"100--102) and replication (seeds 0--4) readings are reported separately.")
REFERENCE_BATCH = (
    "\n\\noindent\\textit{Reference batch.} The reference critic's reference batch is the batch with the highest cell-type "
    "entropy, ties going to the larger batch. The entropy rule prefers the batch whose cell types are most evenly represented, so "
    "that the target does not favour a few cell types, as a merely large batch, the other obvious choice, may do when a few cell "
    "types dominate it. The chosen reference holds every cell type in five of the eight tasks, and 10 of 17 (lung), 14 of 19 "
    "(human--mouse immune) and 3 of 4 (simulation 2) in the others, so in those three tasks some cells of other batches have no "
    "cell of their type in the reference (\\texttt{paper/analysis\\_pass10/reference\\_coverage.json}).\n")
FIGS7_S8 = r"""
\begin{figure}[p]
\centering\renewcommand{\figurename}{Supplementary Figure}\renewcommand{\thefigure}{S7}%
\includegraphics[width=\textwidth]{figures/figS7.png}
\caption{Biology kept by each penalty relative to plain scVI, at the matched batch removal $b^*$ (filled dots) and at the penalty's strongest fitted setting (open dots), with the batch-conditioned (a) and the unconditioned decoder (b). One dot per real task: biology score minus plain scVI's with the same decoder (higher is better); grey line, no change. Filled dots are drawn for the ranked tasks only (the immune task with the batch-conditioned decoder is low-signal); open dots for all six real tasks. The full trade-off curves are in Supplementary Figure S6.}\label{fig:s7}
\end{figure}

\begin{figure}[p]
\centering\renewcommand{\figurename}{Supplementary Figure}\renewcommand{\thefigure}{S8}%
\includegraphics[width=\textwidth]{figures/figS8.png}
\caption{Hypervolume up to the highest common batch score per penalty, with the batch-conditioned decoder (a; five ranked real tasks) and the unconditioned decoder (b; six real tasks): the area under each task's Pareto frontier (Figure 3) above 0.10 below plain scVI's biology, from plain scVI to the highest batch score every penalty reaches (higher is better; dashed line, 0.10, plain scVI's biology kept throughout); dots, tasks; black tick, median. The axis shows every task. Values with and without the cap are in Supplementary Table S8 (Supplementary Note 7).}\label{fig:s8}
\end{figure}
"""
NOTE10_REMOVED = (
    r" The label-transfer analysis also had a cross-task summary, the mean macro F1 over the real tasks that are not low-signal, "
    r"per method and decoder; the first author removed it on 8 November 2026, after all scores had been read, because label "
    r"transfer is compared per task with plain scVI, and its last printed copies (Supplementary Note 9 and Table S11) were "
    r"deleted on 9 November 2026.")


def supplement():
    edit("paper/supp_items.tex", [
        ("Tables S6, S7, S9 and S12-S14 and Figures S1-S6", "Tables S6, S7, S9 and S12-S15 and Figures S1-S8"),
        ("(Note 1, Tables S6, S7, S9 and S12--S14, Figures S1--S6)", "(Note 1, Tables S6, S7, S9 and S12--S15, Figures S1--S8)"),
        ("\\input{supp/tableS14}\n", "\\input{supp/tableS14}\n\\clearpage\n\\input{supp/tableS15}\n")])
    edit("paper/supp/figures.tex", [
        ("Supplementary Figures S1-S6", "Supplementary Figures S1-S8"),
        (r"\caption{Scaling checks (exploratory, used for no conclusion).",
         r"\caption{Scaling checks (exploratory; the penalty with the best mean rank changes under some of them, Discussion)."),
        (r"\caption{Trade-off curves (the main-text Figure 3 until Pass 23): batch score", r"\caption{Trade-off curves: batch score"),
        (r"Figure 3 summarises these curves", r"Supplementary Figure S7 summarises these curves"),
        ("\\label{fig:s6}\n\\end{figure}\n", "\\label{fig:s6}\n\\end{figure}\n" + FIGS7_S8)])
    edit("paper/supp/note1.tex", [
        (r"minimises the bounded fool loss of scvi-tools, the cross-entropy between the prediction and the uniform distribution "
         r"over the other $K-1$ batches,",
         r"minimises the fool loss of scvi-tools, the cross-entropy between the prediction and the uniform distribution over the "
         r"other $K-1$ batches (a Jensen--Shannon-type objective; the loss is not bounded above and is not the Jensen--Shannon "
         r"divergence itself),"),
        ("No spectral normalisation is used.\n", "No spectral normalisation is used.\n" + REFERENCE_BATCH)])
    edit("paper/supp/note5.tex", [
        (r"Low-signal pairs are left out of every result that uses $b^*$:", r"Low-signal pairs are left out of these results that "
                                                                            r"use $b^*$:"),
        ("and the cross-task label-transfer means (Supplementary Note 9).", "and the label-transfer summaries (Supplementary Note 9)."),
        ("the collapse rate and the single-strength placement (Supplementary Note 7).",
         "the collapse rate and the single-strength placement (Supplementary Note 7). The direct biology check, which also reads the "
         "first-crossing strength (Supplementary Note 9), keeps them in its summaries over the six real tasks (Supplementary Note 12).")])
    edit("paper/supp/note6.tex", [
        ("they are exploratory and used for no conclusion.",
         "they are exploratory, and min--max scaling and z-scores change the penalty with the best mean rank, a limitation stated "
         "in the Discussion.")])
    edit("paper/supp/note7.tex", [
        ("Results per penalty, task and decoder are in Supplementary Table S8 and Figures 4--6.",
         "Results per penalty, task and decoder are in Supplementary Table S8, Figures 3 and 4 and Supplementary Figure S8; the "
         "placement is in Figure 5 and Supplementary Tables~\\ref{tab:4} and S7."),
        ("(Figure 4, which shows the median frontier over the ranked real tasks)",
         "(Figure 3 shows the median frontier over the ranked real tasks and Supplementary Figure S8 the hypervolumes)"),
        (r"reaches $b_{\mathrm{common}}$ and no further in all 11 ranked real pairs,",
         r"reaches $b_{\mathrm{common}}$ and no further in all 11 ranked real pairs (on the large chromatin-accessibility task "
         r"with the batch-conditioned decoder its highest batch score is 0.674, against 0.683 to 0.834 for the other penalties),"),
        (r"The main text and Figure 5 count only the fits at strengths whose mean batch score over the three seeds is at most "
         r"$b_{\mathrm{common}}$, the range of Figure 4 (6 to 33 fits per real task and decoder pair),",
         r"The main text and Figure 4 count only the fits at strengths whose mean batch score over the three seeds is at most "
         r"$b_{\mathrm{common}}$ (6 to 33 fits per real task and decoder pair), strong settings whose batch score fell back below "
         r"it included,"),
        ("Results are in Table~\\ref{tab:4} and Supplementary Table S7.", "Results are in Supplementary Tables~\\ref{tab:4} and S7.")])
    edit("paper/supp/note8.tex", [
        ("with the cell-type labels of all cells; it is the only method here that uses the labels.",
         "with the cell-type labels of all cells; it is the only method here trained with the labels (the reference critic's "
         "reference batch is also chosen with them, and several biology metrics use them; Supplementary Notes 1 and 3)."),
        ("and principal-component analysis with 50 components.",
         "and principal-component analysis with 50 components. Their selection scores are in Supplementary Table~S15: against "
         "plain scVI with the batch-conditioned decoder, Harmony with 50 principal components scored above it on 3 of the six "
         "real tasks, Scanorama with 100 dimensions on 1 and principal-component analysis with 50 components on 1, whereas "
         "Harmony and Scanorama at their transferred 10-dimension settings did so on 1 and 0 (Supplementary Table~\\ref{tab:4}).")])
    edit("paper/supp/note9.tex", [
        (r"the strength with no failed fit whose seed-mean batch score is nearest $b^*$, ties to the smaller strength, edge and "
         r"extension strengths included (Supplementary Note 5)",
         STRENGTH_RULE + r", using failure-free strengths, edge and extension strengths included (Supplementary Note 5). "
         r"This replaced the earlier strength nearest $b^*$ on any crossing after the first results had been read "
         r"(Supplementary Note 10)"),
        ("Harmony, Scanorama, sysVI, scANVI and PCA are not label-transferred.",
         "Harmony, Scanorama, sysVI, scANVI and PCA are not part of this analysis; the direct biology check reads their "
         "label-transfer recall and confusion (Supplementary Note 12)."),
        ("For each method and decoder, the cross-task summary is the mean macro F1 over the real tasks that are not low-signal "
         "(five with the batch-conditioned decoder, six with the unconditioned decoder). Low-signal pairs are tabulated and marked "
         "but not summarised;",
         "Label transfer is reported per task, against plain scVI with the same decoder; no mean over tasks is computed. Low-signal "
         "pairs are tabulated and marked and are left out of the per-task comparisons of the main text;"),
        (r"\input{supp/note9_table}",
         r"\noindent\textit{First-crossing results.} The discriminator's median macro-F1 difference from plain scVI was "
         r"0.000 over the five ranked real tasks with the batch-conditioned decoder and $-0.002$ over the six real tasks with "
         r"the unconditioned decoder (below plain scVI on five, lowest among penalties on three). Other penalties' median "
         r"differences ranged from $-0.006$ to $+0.001$. Kendall's $\tau_b$ with the biology order at $b^*$ had median "
         r"$-0.067$, range $-0.733$ to 0.600, over the 11 ranked real pairs. The selected penalties' batch-score distances "
         r"from $b^*$ had median 0.005 and maximum 0.073. Both strength readings are compared in Supplementary Note 12." +
         "\n\n" + r"\input{supp/note9_table}")])
    edit("paper/supp/note9.tex", [
        ("of the matched strength, " + STRENGTH_RULE, "at the first crossing: " + STRENGTH_RULE),
        (r"Note 10), with its three sweep seeds", r"Note 10). We use its three sweep seeds")])
    edit("paper/supp/note10.tex", [
        (r"\item \emph{Limitations.}", NOTE10_STRENGTH + "\n" + r"\item \emph{Limitations.}"),
        ("(g) the Pareto frontier of Figure 4 and the cap", "(g) the Pareto frontier of Figure 3 and the cap"),
        ('the ranking specification names them as exploratory checks used for no conclusion but does not define',
         'the ranking specification lists them as exploratory checks without a role in the conclusions and does not define'),  # D6: Figure S5 now informs a stated limitation
        ("rests on the single-strength placement (Supplementary Note 7).", "rests on the single-strength placement (Supplementary "
                                                                          "Note 7)." + NOTE10_REMOVED),
        (r"(\texttt{local\_minus\_global} in \texttt{paper/figure\_data/paper\_numbers.json};",
         r"(\texttt{local\_minus\_global} and \texttt{label\_transfer\_mean} in \texttt{paper/figure\_data/paper\_numbers.json};")])
    edit("paper/supp/note13.tex", [
        ("% paragraph of the main text (moved here in Pass 23) and paper/figure_data/replication_mean_ranks.json; do not edit.\n",
         "% paragraph of the main text (moved here in Pass 23) and paper/figure_data/replication_mean_ranks.json; Pass 24\n"
         "% (paper/tools/pass24_edits.py) adds the rule detail moved from the Methods. Do not edit by hand.\n"),
        ("everything else in this note is descriptive (Methods; Supplementary Notes 4 and 10).",
         "everything else in this note is descriptive (Supplementary Notes 4 and 10)." + REPLICATION_RULE),
        ("so it is approximate as well (Methods). With the signs", "so it is approximate as well (Rule, above). With the signs"),
        ("it was defined after the rule and is approximate as well (Methods).",
         "it was defined after the rule and is approximate as well (Rule, above)."),
        ("so their order is not tested by it (Methods).", "so their order is not tested by it (Rule, above).")])
    edit("paper/supp/tableS13.tex", [
        (" Equations in Supplementary Note 1;", " The discriminator, the reference critic and the pooled critic were also fitted at "
                                               "strength 0.03 on the immune task with the batch-conditioned decoder "
                                               "(Supplementary Note 4). Equations in Supplementary Note 1;")])
    first_crossing_supplement()


def first_crossing_supplement():
    edit("paper/supp/figures.tex", [
        ("F1 per cell type, method and task at the matched strength.",
         "F1 per cell type, method and task at the first-crossing strength."),
        (r"Penalties are at the strength whose seed-mean batch score is nearest the matched batch removal $b^*$, never chosen "
         r"by accuracy;", "For each penalty the rule is: " + STRENGTH_RULE + r", never chosen by accuracy (Supplementary Note 9);")])
    # Note 12 is newly edited in Part b; read the same fixed starting text as the other Pass 24 edits.
    source = subprocess.run(["git", "-C", str(Path(__file__).resolve().parents[2]), "show", "24f2665:paper/supp/note12.tex"],
                            capture_output=True, text=True, check=True).stdout
    source = source.replace(r"the failure-free sweep strength whose seed-mean batch score is nearest $b^*$, ties to the "
                            r"smaller strength; the choice uses batch scores only", STRENGTH_RULE +
                            r"; failure-free strengths, chosen from batch scores only")
    old = (r"The per-task, per-fit, per-type and confusion tables, the record (with the SHA-256 of every input) and the one-page "
           r"summary are in \texttt{docs/biology\_check/} on branch \texttt{biology-check}, with their checksums in "
           r"\texttt{docs/biology\_check/MANIFEST.sha256}.")
    new = (r"The first-crossing run used code commit 8073db6, which adds a strengths-file option to the same check. Its "
           r"per-task, per-fit, per-type and confusion tables, input checksums, record and summary are kept in "
           r"\texttt{paper/figure\_data/biology\_check\_first\_crossing/}, with its original \texttt{MANIFEST.sha256}. "
           r"The earlier \texttt{summary.csv} and \texttt{summary\_by\_method.csv} are kept in "
           r"\texttt{paper/figure\_data/biology\_check\_nearest\_any\_crossing/tables/}.")
    assert source.count(old) == 1
    source = source.replace(old, new)
    comparison = []
    import pandas as pd
    from biology_check_table import DATA, PEN, LABEL, signed
    old_rows = pd.read_csv(DATA / "biology_check_nearest_any_crossing/tables/summary_by_method.csv")
    for r in summarize()["rows"]:
        if r["method"] not in PEN:
            continue
        o = old_rows[(old_rows.task_group == "real") & (old_rows.method == r["method"]) & (old_rows.decoder == r["decoder"])].iloc[0]
        comparison.append(f"{LABEL[r['method']]} & {r['decoder']} & {signed(o.purity_diff)} & {signed(r['purity_diff'])} & "
                          f"{signed(o.rare_purity_diff)} & {signed(r['rare_purity_diff'])} " + r"\\")
    both = (r"\noindent\textit{Both strength readings.} On 9 November 2026, after the first results had been read, the first "
            r"author changed the strength rule for label transfer and this check (Supplementary Note 10). The earlier reading "
            r"chose the strength nearest $b^*$ on any crossing; some were far past the first crossing. The new rule is: " +
            STRENGTH_RULE + r". The table compares mean differences from plain scVI over the real tasks; rare types use only "
            r"the tasks that have them. The discriminator's larger losses in the earlier reading did not hold at the first "
            r"crossing across decoders." + "\n\n" +
            r"{\scriptsize\begin{tabular}{@{}llrrrr@{}}" + "\n" + r"\toprule" + "\n" +
            r"Penalty & Decoder & Purity, earlier & Purity, first & Rare purity, earlier & Rare purity, first \\" + "\n" +
            r"\midrule" + "\n" + "\n".join(comparison) + "\n" + r"\bottomrule\end{tabular}}" + "\n\n" +
            r"\noindent For label transfer, the discriminator's median macro-F1 difference from plain scVI changed from "
            r"$-0.006$ to 0.000 with the batch-conditioned decoder and from $-0.060$ to $-0.002$ with the unconditioned "
            r"decoder. The other penalties' median differences ranged from $-0.016$ to $+0.001$ earlier and from $-0.006$ "
            r"to $+0.001$ at the first crossing. These summaries leave out low-signal pairs (Supplementary Note 9)." + "\n\n" +
            r"\noindent\textit{Per-cell-type results at the first crossing.} Rare-type purity was below plain scVI in 21 of "
            r"48 task, penalty and decoder comparisons. For the lung task's 46 ionocytes, the batch-conditioned discriminator "
            r"at strength 0.3 had a purity difference of $+0.022$ and label-transfer recall 0.812, against 0.797 for plain "
            r"scVI. Its earlier large loss for these cells therefore does not describe its first-crossing strength. Full "
            r"per-type F1 is in Supplementary Figure S3; each task's \texttt{fit\_types.csv} and \texttt{confusion.csv} "
            r"tables in the new run give purity, recall, marker overlap and the types mistaken for each other.")
    source = source.replace("\n\\par\n\\endgroup", "\n\n" + both + "\n\\par\n\\endgroup")
    (ROOT / "paper/supp/note12.tex").write_text(source)


def main():
    p = ROOT / "paper/outline.tex"
    s = outline(p.read_text())
    n_main, n_cap = budget(s)
    print(f"main text {n_main} words (limit {MAIN_LIMIT}; Part a: limit 5700); main-figure captions {n_cap} words (limit {CAPTION_LIMIT})")
    assert n_main <= MAIN_LIMIT and n_cap <= CAPTION_LIMIT, (n_main, n_cap)
    p.write_text(s)
    supplement()


if __name__ == "__main__":
    main()
