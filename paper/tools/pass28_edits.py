"""Pass 28 edits (2026-11-10, tracked id P28): Figure 2 panel a is coloured by the biology score at b* minus the median of the six
penalties in the same task row (paper/figure_scripts/fig2.py), by the first author's decision, so the colour follows the ranking within
a row instead of the differences between tasks. This script changes the text only: in paper/outline.tex, as tracked changes (id P28),
part (a) of the Figure 2 caption and part (a) of the "Figure 2" figure-list item describe the new colour. Each P28 change sits outside
the Pass 24 markup of the caption and inside the new text of the list item's \\replaced, as Pass 26 and Pass 27 did.
No number and no other text changes. Change log paper/pass28_changelog.md. It starts from the text committed at 6282edb (the Pass 27
result); every old string must occur exactly once, and the main text and the main captions (the Table 1 caption included) must stay
within the limits of Pass 24 (5,500 and 450 words), or the script stops before it writes anything.
Run from the repository root:  python paper/tools/pass28_edits.py   (PASS28_ROOT overrides the root of the edited files)
Then: python paper/tools/accept_changes.py"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_table1 import CAPTION  # noqa: E402
from pass24_edits import CAPTION_LIMIT, MAIN_LIMIT, budget, words  # noqa: E402

ROOT = Path(os.environ.get("PASS28_ROOT") or Path(__file__).resolve().parents[2])

AUTHOR = (r"\definechangesauthor[name={Pass 28}, color=magenta]{P28}% Figure 2a coloured by biology at $b^*$ minus the row median; "
          "change log paper/pass28_changelog.md\n")
PAIRS = [  # (old, new) in paper/outline.tex, each old string once
    (r"(a) Biology score at $b^*$ (colour) and rank \deleted[id=P24]{among the six penalties} within each task row "         # caption
     r"\replaced[id=P24]{(number)}{(number; tied penalties share the average rank)} for each real task,",
     r"(a) \replaced[id=P28]{Rank}{Biology score at $b^*$ (colour) and rank} \deleted[id=P24]{among the six penalties} within each "
     r"task row \replaced[id=P24]{(number)}{(number; tied penalties share the average rank)} for each real task,\added[id=P28]{ and "
     r"biology score at $b^*$ minus the median of the six penalties in that row (colour),}"),
    (r"(a) heatmap of biology score at $b^*$ (colour) and rank (number) for the six penalties in each real task,",            # figure list
     r"(a) heatmap of \replaced[id=P28]{rank (number) and of biology score at $b^*$ minus the row's median over the six penalties "
     r"(colour)}{biology score at $b^*$ (colour) and rank (number) for the six penalties} in each real task,"),
]


def replace_once(s, old, new):
    assert s.count(old) == 1, (s.count(old), old[:90])
    return s.replace(old, new)


def outline(s):
    s = replace_once(s, r"\definechangesauthor[name={Pass 27}", AUTHOR + r"\definechangesauthor[name={Pass 27}")
    for a, c in PAIRS:
        s = replace_once(s, a, c)
    return s


def main():
    text = outline((ROOT / "paper/outline.tex").read_text())
    n_main, n_cap = budget(text)
    n_cap += len(words(CAPTION))                      # the Table 1 caption is a main caption (Pass 27)
    assert n_main <= MAIN_LIMIT and n_cap <= CAPTION_LIMIT, (n_main, n_cap)
    (ROOT / "paper/outline.tex").write_text(text)
    print(f"main text {n_main} words (limit {MAIN_LIMIT}); main captions, Table 1 included, {n_cap} words (limit {CAPTION_LIMIT})")


if __name__ == "__main__":
    main()
