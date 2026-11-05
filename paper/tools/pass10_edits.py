"""Pass 10 edits to paper/outline.tex (tracked id P10): real Figures 2-6 (paths, captions) and the \\tbd items that repository records answer.
Every old text must occur exactly once or the script stops. Writes paper/pass10_numbers.csv (one row per number written).
Re-run from the Pass 9 + figures-merge text:  python paper/tools/pass10_edits.py   (from the repository root).
Inputs read: paper/figure_captions_pass9.md, paper/analysis_pass10/coverage_sim_sweep.json, paper/analysis_pass10/reference_coverage.json."""
import csv, json, os, re, sys
from pathlib import Path

ROOT = Path(os.environ.get("PASS10_ROOT") or Path(__file__).resolve().parents[2])   # PASS10_ROOT: test copy of the repository
s = ""
ledger = []


def rep(old, new, why):
    global s
    assert s.count(old) == 1, (s.count(old), why, old[:90])
    s = s.replace(old, new)


def num(item, value, source, how):
    ledger.append(dict(item=item, value=value, source=source, how=how))
    return value


def match_brace(t, i):                      # t[i] == '{'; index of the matching '}'
    d = 0
    for j in range(i, len(t)):
        d += (t[j] == "{") - (t[j] == "}")
        if d == 0:
            return j
    raise ValueError("unbalanced")


def flatten(t):
    """Active (new) text of a caption with `changes` markup: \\replaced{a}{b} -> a, \\added{a} -> a, \\deleted{a} -> nothing."""
    out, i = "", 0
    pat = re.compile(r"\\(replaced|added|deleted)(\[[^\]]*\])?\{")
    while True:
        m = pat.search(t, i)
        if not m:
            return out + t[i:]
        out += t[i:m.start()]
        a_end = match_brace(t, m.end() - 1)
        a = flatten(t[m.end():a_end])
        i = a_end + 1
        if m.group(1) == "replaced":
            i = match_brace(t, i) + 1
        out += "" if m.group(1) == "deleted" else a


def main(root=ROOT):
    global s, ledger
    TEX = root / "paper" / "outline.tex"
    s = TEX.read_text()
    ledger = []
    # ---- numbers from committed result files
    cov = json.load(open(root / "paper/analysis_pass10/coverage_sim_sweep.json"))
    ref = json.load(open(root / "paper/analysis_pass10/reference_coverage.json"))
    c_ind = num("coverage, independent fits, 30 fits", f"{cov['NL10']['A_coverage']:.4f}", "paper/analysis_pass10/coverage_sim_sweep.json", "NL10.A_coverage")
    c30 = num("coverage, shared seed threshold, 30 fits", f"{cov['NL10']['B_coverage']:.2f}", "paper/analysis_pass10/coverage_sim_sweep.json", "NL10.B_coverage rounded to 2 decimals")
    c36 = num("coverage, shared seed threshold, 36 fits", f"{cov['NL12']['B_coverage']:.2f}", "paper/analysis_pass10/coverage_sim_sweep.json", "NL12.B_coverage rounded to 2 decimals")
    assert cov["NL10"]["A_coverage"] > 0.99 and cov["replicates"] == 20000
    full = [t for t in ref if t != "anndata" and ref[t]["types_in_reference"] == ref[t]["types_in_task"]]
    part = {t: (ref[t]["types_in_reference"], ref[t]["types_in_task"]) for t in ref if t not in ("anndata", *full)}
    assert len(full) == 5 and part == {"lung": (10, 17), "immune_hum_mou": (14, 19), "sim2": (3, 4)}, (full, part)
    for t, (a, b) in part.items():
        num(f"reference batch types, {t}", f"{a} of {b}", "paper/analysis_pass10/reference_coverage.json", "types_in_reference of types_in_task")

    # ---- preamble and header
    rep("\\usepackage{todonotes}", "\\definechangesauthor[name={Pass 10}, color=teal]{P10}% real Figures 2-6, \\tbd items answered from repository records; change log paper/pass10_changelog.md, numbers in paper/pass10_numbers.csv\n\\usepackage{todonotes}", "preamble")
    rep("Figures 1--6 are mocks (made on another branch); Tables 3 and 4 hold sweep values (Pass 9,",
        "Figure 1 is a placeholder schematic and Figures 2--6 are drawn from the sweep analysis (Pass 10, \\texttt{paper/figure\\_scripts}, change log \\texttt{paper/pass10\\_changelog.md}); Tables 3 and 4 hold sweep values (Pass 9,", "header")

    # ---- figures: paths, captions
    caps = {}
    for m in re.finditer(r"## Figure (\d)\n(.*?)(?=\n## |\Z)", (root / "paper/figure_captions_pass9.md").read_text(), re.S):
        caps[int(m.group(1))] = m.group(2).strip()
    assert sorted(caps) == [2, 3, 4, 5, 6]
    caps[2] = caps[2].replace("over the five leave-one-seed-out repeats", "over the three leave-one-seed-out repeats")
    caps[2] = caps[2].replace("A penalty whose curve does not reach $b^*$ takes the bottom rank.", "A penalty whose curve does not reach $b^*$ would take the bottom rank; every curve reached it here.")
    old6 = caps[6]
    k0 = old6.index("Each curve is drawn up to")
    k1 = old6.index("Horizontal line, plain scVI")
    caps[6] = (old6[:k0] + "Each curve is drawn up to the largest $k$ for which all six tasks have $k$ settings: $k=10$ for five penalties (the discriminator, reference critic and pooled critic have an eleventh setting on the immune task only) and $k=12$ for the Sinkhorn divergence, and $k=6$ for Harmony and sysVI. "
               "Scanorama has six settings on five tasks and five on the large chromatin-accessibility task (80 neighbours could not be fitted); it is drawn to $k=5$, and its $k=6$ value (open marker) is the mean of five tasks. "
               "The Results compare methods at $k=1$ and $k=6$, the number of settings of Harmony, Scanorama and sysVI; the penalty curves beyond $k=6$ are shown for completeness. " + old6[k1:])
    assert "Scanorama is drawn to $k=5$; its $k=6$ value (open marker) is the mean of five tasks because" not in caps[6]
    assert "The unconditioned decoder is in Table 4 and the supplement." in caps[6]
    caps[6] = caps[6].replace("The unconditioned decoder is in Table 4 and the supplement.", "The unconditioned decoder is in Supplementary Table S7.")
    for n in (2, 3, 4, 5, 6):
        inc = f"\\includegraphics[width=\\textwidth]{{figures/mock/fig{n}.png}}"
        rep(inc, f"\\includegraphics[width=\\textwidth]{{figures/fig{n}.png}}", f"fig{n} path")
        i = s.index(f"\\label{{fig:{n}}}")
        j = s.rindex("\\caption{", 0, i)
        e = match_brace(s, j + len("\\caption"))
        old = flatten(s[j + len("\\caption{"):e])
        old = old.replace("MOCK -- synthetic data. ", "")
        s = s[:j] + "\\caption{\\replaced[id=P10]{" + caps[n] + "}{MOCK -- synthetic data. " + old + "}}" + s[e + 1:]
    s, nsub = re.subn(r"% MOCK NEEDS REDRAW[^\n]*\n", "", s)
    assert nsub == 4, nsub

    # ---- \tbd items answered from repository records
    rep("Code and checksums: \\tbd{repository address and version tag at submission}.",
        "Code and checksums: \\replaced[id=P10]{github.com/kreid415/wasserstein-critic-deconfounding.}{\\tbd{repository address and version tag at submission}.}", "abstract repository")
    rep("In searches of OpenAlex, arXiv and Europe PMC run in October 2026, we found",
        "In searches of OpenAlex, arXiv and Europe PMC run in October 2026 and repeated in OpenAlex and Europe PMC on 5 November 2026 for works published since 1 October 2026, we found", "literature search")
    rep(" This absence is search-based, not exhaustive \\tbd{repeat the literature search before submission}.", " This absence is search-based, not exhaustive.", "gap tbd")
    rep(" A reference that contains every cell type gives each batch a target for all of its cell types, so no batch is pushed onto cell types the reference lacks \\tbd{confirm this justification}.",
        " The entropy rule prefers the batch whose cell types are most evenly represented, so that the target does not favour a few cell types. The chosen reference holds every cell type in five of the eight tasks, and 10 of 17 (lung), 14 of 19 (human--mouse immune) and 3 of 4 (simulation 2) in the others, so in those three tasks some cells of other batches have no cell of their type in the reference.", "reference")
    rep("are applied unchanged to the sweep \\tbd{confirm once the analysis code carries them; state whether any sweep score had been read when the ranking rules were fixed, since the strength-choice rules had already read the sweep's seed means} (Supplementary Note 10).",
        "are applied to the sweep with the changes listed in Supplementary Note 10: the three tasks fitted first had already been read by the strength-choice rules, whereas no score of the other five tasks existed when the ranking rules were written (22--23 October 2026).", "ranking rules")
    rep("\\tbd{redo the coverage simulation for ten strengths and three seeds; for the earlier design of six strengths and five seeds it gave 0.91 to 1.00 for a nominal 0.95}",
        f"in a synthetic simulation of the sweep design (20,000 replicates) the exact 95\\% interval covered the expected collapse rate in {c_ind} of replicates when the fits were independent, and in {c30} (30 fits) and {c36} (36 fits) when each seed had one threshold shared by its strengths, so the coverage of a nominal 0.95 may lie between {c36} and 1.00 (Supplementary Note 7)", "coverage")
    rep("\\tbd{version tag and commit at submission}; checksums in supplement.",
        "fitting and scoring code at commit 10a1ca5, sweep analysis at commit 8595a16, biology check at commit c4a793f, figure scripts and their inputs in \\texttt{paper/figure\\_scripts} and \\texttt{paper/figure\\_data}; \\tbd{release tag and commit of the submitted version}; checksums in supplement.", "availability")
    rep("\\tbd{Sinkhorn fits refitted after the solver correction: confirm for all eight tasks}",
        "all Sinkhorn fits of the eight tasks were made with the corrected solver of 13 October 2026 (commit 10a1ca5): the 180 fits of the three local tasks were refitted on 17--19 October 2026 and the five cluster tasks were fitted from 22 October 2026", "sinkhorn")
    rep("failure types reported separately}", "failure types reported separately; coverage of the interval simulated for ten and twelve strengths with three seeds}", "note 7 coverage")
    rep("when the sweep rules and the ranking-analysis rules were fixed, and which sweep scores had been read at that time \\tbd{dates for the sweep and the ranking analysis; which sweep scores the strength-choice rules had read before the ranking rules were fixed}",
        "when the sweep rules and the ranking-analysis rules were fixed, and which sweep scores had been read at that time (sweep rules: amendments of 4 and 16 October 2026, the last at commit 10a1ca5, the five further tasks added by the pilot extension decided on 21 October with those rules; ranking rules: the descriptive ranking design written on 22 and 23 October 2026, commits 065a486 and 5d01da0; by then the strength-choice rules had read the seed means of the small chromatin-accessibility, immune and simulation 1 tasks, and the other five tasks were fitted from 22 October and scored from 28 October; the analysis code for all eight tasks was written on 1 November, with a first run on all eight tasks that day for debugging, and the hypervolume and the share of non-dominated strengths were added on 2 November, after those scores had been read); changes from the ranking specification (three seeds instead of five, every fitted strength a point, $b^*$ computed from the sweep rows, at least two of three leave-one-seed-out readings)", "note 10 dates")
    rep("early-scored fits (19 fits of the planned fixed-strength benchmark were scored and not read; \\tbd{number of planned-benchmark fits made before it was dropped; they are not used})",
        "early-scored fits (160 fits of the planned fixed-strength benchmark were scored, 19 on 12 October and 141 on 20--21 October 2026, on the immune task with the batch-conditioned decoder, and are not used; about 572 fits were made before the benchmark was stopped on 21 October: 167 on the workstation and 405 on the cluster, the cluster fits not scored)", "note 10 fits")
    rep("planned fixed-strength benchmark not run, \\tbd{fits made for it before it was dropped}, unused; ",
        "planned fixed-strength benchmark not run, about 572 fits made for it before it was stopped on 21 October 2026 (167 on the workstation, 405 on the cluster; the cluster count is the repository record's, not re-counted), unused; ", "note 11 fits")
    num("planned-benchmark fits, workstation model files", "167", "tier12_v3/X1_local/models (file count, workstation, 2026-11-05)", "ls models | wc -l; 20 of them also in tier12_v2/X1_local")
    num("planned-benchmark fits, workstation score files", "160", "tier12_v3/X1_local/scores (file count; 13 dated 2026-10-20, 128 dated 2026-10-21, 19 symlinks to the 2026-10-12 scores)", "ls scores | wc -l")
    num("planned-benchmark latents, cluster", "405", "CONSTRAINTS.md line 658, docs/PREREG.md section 9 entry of 2026-10-22 item 1 (the user's count; not re-counted)", "as recorded")
    num("planned-benchmark fits, total", "about 572", "CONSTRAINTS.md line 658", "167 + 405 = 572")
    TEX.write_text(s)
    with open(root / "paper/pass10_numbers.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["item", "value", "source", "how"]); w.writeheader(); w.writerows(ledger)
    assert "\\tbd" in s and s.count("MOCK NEEDS REDRAW") == 0
    print("ok", len(ledger), "ledger rows;", s.count("\\tbd{"), "tbd macros remain (incl. definition and comments)")


if __name__ == "__main__":  # pragma: no cover  # entry point
    main()
