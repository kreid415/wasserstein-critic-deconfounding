"""Pass 29 edits (2026-11-10, tracked id P29): wording fixes after a review of the manuscript, asked for by the first author. No number
is recomputed and no analysis changes; every number the new text places in the main text is in paper/pass29_numbers.csv.
- paper/outline.tex, as tracked changes (id P29): the abstract says that batch removal was not matched at the strongest settings;
  Res-2 gives the count of near-tied within-pair differences (until now stated only in Disc-3), and Disc-3 points to Res-2 instead of
  repeating it; Res-2b adds the family-level p next to the pair-level p of the rule; Res-5 marks the pancreas near-tie, says that
  Harmony is averaged over its seven settings and states where the discriminator did worst, and Conclusions (iii) follows Res-5.
  Where a change falls inside the markup of an earlier pass, the P29 markup sits inside that pass's new text, as in Pass 26 to 28.
- paper/supp/note5.tex, note9.tex, note10.tex, note12.tex, paper/table1.tex and its generator paper/tools/make_table1.py, without
  tracked markup (both PDF builds input these files as they are): the matched strength is the first-crossing rule, the pancreas
  near-tie, the discriminator's purity loss with the batch-conditioned decoder, and "Nominal" instead of "Exact" 95% intervals.
Change log paper/pass29_changelog.md. It starts from the text committed at bec1ece (the Pass 28 result); every old string must occur
exactly once, and the main text and the main captions (the Table 1 caption included) must stay within the limits of Pass 24 (5,500
and 450 words), or the script stops before it writes anything.
Run from the repository root:  python paper/tools/pass29_edits.py   (PASS29_ROOT overrides the root of the edited files)
Then: python paper/tools/accept_changes.py"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_table1 import CAPTION  # noqa: E402
from pass24_edits import CAPTION_LIMIT, MAIN_LIMIT, budget, words  # noqa: E402

ROOT = Path(os.environ.get("PASS29_ROOT") or Path(__file__).resolve().parents[2])

AUTHOR = (r"\definechangesauthor[name={Pass 29}, color=orange]{P29}% wording fixes after the manuscript review (abstract, Res-2, "
          "Res-2b, Res-5, Disc-3, Conclusions); change log paper/pass29_changelog.md\n")
PAIRS = [  # (old, new) in paper/outline.tex, each old string once; a change inside earlier markup goes inside its new text
    (r"and 0.213 at each penalty's strongest setting.",                                         # abstract
     r"and 0.213 at each penalty's strongest setting\added[id=P29]{, where batch removal was not matched}."),
    (r"Most neighbouring readings were close. In 44 of the 55",                                 # Res-2
     r"\added[id=P29]{Over the 13 ranked pairs, 160 of the 195 within-pair differences in biology at $b^*$ (15 penalty pairs each) "
     r"were under 0.02 and 4 were 0.04 or more. }Most neighbouring readings were close. In 44 of the 55"),
    (r"matched batch removal \replaced[id=P24]{and differed mainly at strong settings. At $b^*$,}{the six penalties were near-tied:} "  # Disc-3
     r"160 of the 195 within-pair differences in biology over the 13 ranked pairs were under 0.02, and 4 were 0.04 or more. "
     r"\replaced[id=P24]{The range between}",
     r"matched batch removal \added[id=P29]{(Res-2) }\replaced[id=P24]{and differed mainly at strong settings.\deleted[id=P29]{ At "
     r"$b^*$,}}{the six penalties were near-tied:} \deleted[id=P29]{160 of the 195 within-pair differences in biology over the 13 "
     r"ranked pairs were under 0.02, and 4 were 0.04 or more. }\replaced[id=P24]{The range between}"),
    (r"mean Kendall's $\tau_b$ 0.456, $p=0.00049$), but the first-ranked",                     # Res-2b (the old text has "). The")
     r"mean Kendall's $\tau_b$ 0.456, $p=0.00049$\added[id=P29]{ over the 13 pairs, as the rule prescribes; $p=0.0156$ over the six "
     r"task families}), but the first-ranked"),
    (r"on five of six real tasks (median difference $-0.002$)",                                 # Res-5
     r"on five of six real tasks (\added[id=P29]{pancreas by less than 0.001; }median difference $-0.002$)"),
    (r"Harmony, averaged over its settings, differed",
     r"Harmony, averaged over its \added[id=P29]{seven }settings\added[id=P29]{ rather than read at one}, differed"),
    (r"These readouts show small average changes at the first crossing, without a consistent disadvantage for the discriminator "
     r"across decoders.",
     r"These readouts show small average changes at the first crossing\replaced[id=P29]{;}{, without a consistent disadvantage for} "
     r"the discriminator \replaced[id=P29]{had the largest purity loss with the batch-conditioned decoder and the lowest F1 on three "
     r"tasks with the unconditioned decoder}{across decoders}."),
    (r"\replaced[id=P24]{scVI at first-crossing strengths, without a consistent disadvantage}{scVI, $-0.107$} for "    # Conclusions (iii)
     r"\replaced[id=P24]{the discriminator across decoders (Res-5);}",
     r"\replaced[id=P24]{scVI at first-crossing strengths,\deleted[id=P29]{ without a consistent disadvantage}}{scVI, $-0.107$} "
     r"\replaced[id=P29]{and}{for} \replaced[id=P24]{the discriminator \replaced[id=P29]{had the largest purity loss with the "
     r"batch-conditioned decoder}{across decoders} (Res-5);}"),
]
SUPP = {  # untracked, in place: path -> [(old, new)]
    "paper/supp/note5.tex": [(r"the matched strength: among the strengths with no failed fit, the one whose seed-mean batch score is "
                              r"nearest $b^*$, ties to the smaller strength, edge and extension strengths included.",
                              r"the matched strength: of the two fitted strengths that bracket the first crossing of $b^*$, the one "
                              r"whose seed-mean batch score is nearer $b^*$ (ties to the smaller strength), or the upper one when the "
                              r"lower is plain scVI, among the strengths with no failed fit, edge and extension strengths included. "
                              r"This first-crossing rule replaced the earlier strength nearest $b^*$ on any crossing after the first "
                              r"results had been read (Supplementary Note 10).")],
    "paper/supp/note10.tex": [(r"the matched-strength table being read from those results;",
                               r"the matched-strength table being read from those results (this strength rule was later replaced by "
                               r"the first-crossing rule described below);")],
    "paper/supp/note9.tex": [(r"(below plain scVI on five, lowest among penalties on three)",
                              r"(below plain scVI on five, pancreas by less than 0.001; lowest among penalties on three)")],
    "paper/supp/note12.tex": [(r"The discriminator's larger losses in the earlier reading did not hold at the first crossing across "
                               r"decoders.",
                               r"The discriminator's larger losses in the earlier reading were not seen at the first crossing; with "
                               r"the batch-conditioned decoder its mean purity loss was still the largest among the penalties "
                               r"(Supplementary Table S10).")],
    "paper/table1.tex": [(r"Exact 95\% intervals", r"Nominal 95\% intervals")],
    "paper/tools/make_table1.py": [(r"Exact 95\% intervals", r"Nominal 95\% intervals")],
}


def replace_once(s, old, new):
    assert s.count(old) == 1, (s.count(old), old[:90])
    return s.replace(old, new)


def outline(s):
    s = replace_once(s, r"\definechangesauthor[name={Pass 28}", AUTHOR + r"\definechangesauthor[name={Pass 28}")
    for a, c in PAIRS:
        s = replace_once(s, a, c)
    return s


def main():
    text = outline((ROOT / "paper/outline.tex").read_text())
    n_main, n_cap = budget(text)
    n_cap += len(words(CAPTION))                      # the Table 1 caption is a main caption (Pass 27); "Exact" -> "Nominal" keeps its length
    assert n_main <= MAIN_LIMIT and n_cap <= CAPTION_LIMIT, (n_main, n_cap)
    edited = {"paper/outline.tex": text}
    for rel, pairs in SUPP.items():
        s = (ROOT / rel).read_text()
        for a, c in pairs:
            s = replace_once(s, a, c)
        edited[rel] = s
    for rel, s in edited.items():
        (ROOT / rel).write_text(s)
    print(f"main text {n_main} words (limit {MAIN_LIMIT}); main captions, Table 1 included, {n_cap} words (limit {CAPTION_LIMIT})")


if __name__ == "__main__":
    main()
