"""Pass 25 edits (2026-11-10, tracked id P25): Supplementary Figure S3 (label transfer) is redrawn as F1 minus plain scVI's with the same
decoder (paper/figure_scripts/figS3.py; plotted values in paper/figure_data/figS3_values.csv). This script changes the text only:
the Figure S3 caption and figure size in paper/supp/figures.tex, the one sentence of Supplementary Note 9 that says where the per-type F1
is, and the supplementary-list item of paper/outline.tex (word-level tracked changes, as in Passes 22-24; the supplement files are edited
in place, as in Pass 24). The main text, every number and the other figures are untouched. Change log paper/pass25_changelog.md.
It starts from the text committed at 1cea7ef (the Pass 24 result); every old text must occur exactly once or the script stops.
Run from the repository root:  python paper/tools/pass25_edits.py   (PASS25_ROOT overrides the root of the edited files)
Then: python paper/tools/accept_changes.py"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from accept_changes import accept  # noqa: E402
from pass24_edits import find  # noqa: E402

ROOT = Path(os.environ.get("PASS25_ROOT") or Path(__file__).resolve().parents[2])

AUTHOR = (r"\definechangesauthor[name={Pass 25}, color=purple]{P25}% Supplementary Figure S3 redrawn as F1 minus plain scVI's; "
          "change log paper/pass25_changelog.md\n")
ITEM_KEY = "Figure S3, label transfer: F1 per cell type"
ITEM = (r"Figure S3, label transfer: F1 minus plain scVI's with the same decoder, as macro F1 per penalty and real task and "
        r"per cell type, at the first-crossing strength; plain scVI's own F1 is printed.")
IMAGE = (r"\includegraphics[width=\textwidth,height=0.72\textheight,keepaspectratio]{figures/figS3.png}",
         r"\includegraphics[width=\textwidth,height=0.76\textheight,keepaspectratio]{figures/figS3.png}")
CAPTION_START = r"\caption{Label transfer: F1 per cell type, method and task at the first-crossing strength."
CAPTION = (
    r"\caption{\small Label transfer at the first-crossing strength: F1 minus plain scVI's with the same decoder, six real tasks. "
    r"Batch-conditioned decoder: plain scVI, six penalties and the scvi-tools adversary; unconditioned decoder: plain scVI (which "
    r"receives no batch information) and six penalties. "
    r"Each batch is held out in turn and its cells are predicted from the other batches by a 15-nearest-neighbour classifier; cell "
    r"types present in one batch only are dropped; a type's F1 is averaged over the held-out batches that contain it and over seeds "
    r"(three for plain scVI and the penalties, five for the scvi-tools adversary). "
    r"For each penalty the rule is: of the two fitted strengths that bracket the first crossing of $b^*$, the one whose seed-mean "
    r"batch score is nearer $b^*$ (ties to the smaller strength), or the upper one when the lower is plain scVI, never chosen by "
    r"accuracy (Supplementary Note 9); plain scVI and the scvi-tools adversary are unmatched. "
    r"\textbf{(a, b)} Macro F1 minus plain scVI's; one dot per real task, black tick the median over the ranked tasks; the immune "
    r"task with the batch-conditioned decoder is low-signal (open dot, left out of the median). "
    r"\textbf{(c)} F1 per cell type: grey column, plain scVI's own F1; coloured columns, each method's F1 minus plain scVI's (blue "
    r"above, red below; limits $\pm0.15$); types ordered by plain scVI's mean F1 over the two decoders. disc., discriminator; "
    r"ref., reference critic; bary., barycenter critic; Sinkh., Sinkhorn divergence; adv., scvi-tools adversary. "
    r"The simulations are not drawn; their values are in \texttt{paper/figure\_data/label\_transfer\_types.csv}. The plotted values "
    r"are in \texttt{figS3\_values.csv}, and the distances of the batch scores from $b^*$ in \texttt{label\_transfer.csv}, in the same "
    r"folder (Supplementary Note 9).}")
NOTE9 = ("F1 per cell type is in Supplementary Figure S3, and the per-type recall",
         r"F1 per cell type, as the difference from plain scVI's with plain scVI's own F1, is in Supplementary Figure S3 (values in "
         r"\texttt{paper/figure\_data/figS3\_values.csv}), and the per-type recall")


def edit(path, pairs):
    p = ROOT / path
    s = p.read_text()
    for old, new in pairs:
        assert s.count(old) == 1, (path, s.count(old), old[:100])
        s = s.replace(old, new)
    p.write_text(s)


def outline(s):
    anchor = r"\definechangesauthor[name={Pass 24}, color=pink]{P24}"
    assert s.count(anchor) == 1
    s = s.replace(anchor, AUTHOR + anchor)
    old = find(s, ITEM_KEY)
    acc = accept(old).strip()
    assert acc.startswith(r"\item ")
    return s.replace(old, r"\item \replaced[id=P25]{" + ITEM + "}{" + acc[len(r"\item "):] + "}", 1)  # one replacement keeps the tracked copy readable


def figures(s):
    lines = s.split("\n")
    hit = [i for i, ln in enumerate(lines) if ln.startswith(CAPTION_START)]
    assert len(hit) == 1 and lines[hit[0]].endswith(r"\label{fig:s3}"), hit
    head, tail = r"\caption{", r"}\label{fig:s3}"
    old_inner = lines[hit[0]][len(head):-len(tail)]
    assert CAPTION.startswith(head) and CAPTION.endswith("}")
    lines[hit[0]] = head + r"\replaced[id=P25]{" + CAPTION[len(head):-1] + "}{" + old_inner + "}" + tail  # tracked copy shows the caption change
    return "\n".join(lines)


def main():
    p = ROOT / "paper/outline.tex"
    p.write_text(outline(p.read_text()))
    edit("paper/supp/figures.tex", [IMAGE])
    p = ROOT / "paper/supp/figures.tex"
    p.write_text(figures(p.read_text()))
    edit("paper/supp/note9.tex", [NOTE9])


if __name__ == "__main__":
    main()
