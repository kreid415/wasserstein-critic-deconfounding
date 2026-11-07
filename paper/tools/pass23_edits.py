"""Pass 23 edits (2026-11-07, tracked id P23): decisions from the user's walkthrough of the Results, one result at a time.
Res-1: the fit-completion paragraph moves to the supplement (Supplementary Notes 5, 8 and 11 and Table S5 already hold every
number in it) and one sentence stays in the Results. Figure 2: the caption follows the redrawn figure (panel a has tasks as rows
and the decoders stacked; panel b stacks the decoders, sorts penalties best first and reverses the rank axis).
Each edited line is rewritten as word-level tracked changes against its printed (accepted) old text, as in Pass 22. Every old
line must occur exactly once or the script stops.
Run from the repository root:  python paper/tools/pass23_edits.py   (PASS23_ROOT overrides the root of the edited files)
Then: python paper/tools/accept_changes.py ; python paper/tools/check_ledger.py"""
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
]


def rewrite(old, new):
    """Captions are tracked inside \\caption{} so the caption command itself is never struck."""
    if new.startswith("\\caption{"):
        inner_old, _ = group(accept(old).strip(), len("\\caption"))
        inner_new, _ = group(new, len("\\caption"))
        return "\\caption{" + p22.tracked(inner_old, inner_new) + "}"
    return p22.rewrite(old, new)


def main():
    p = ROOT / "paper/outline.tex"
    s = p.read_text()
    for old, new in EDITS:
        assert s.count("\n" + old + "\n") == 1, old[:80]
        line = rewrite(old, new)
        assert " ".join(accept(line).split()) == " ".join(new.split()), old[:80]
        s = s.replace("\n" + old + "\n", "\n" + line + "\n", 1)
    anchor = "\\definechangesauthor[name={Pass 22}, color=red]{P22}"
    assert s.count(anchor) == 1
    s = s.replace(anchor, "\\definechangesauthor[name={Pass 23}, color=blue]{P23}% Results walkthrough decisions; change log "
                  "paper/pass23_changelog.md\n" + anchor)
    p.write_text(s)


if __name__ == "__main__":
    main()
