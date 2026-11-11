"""Pass 29: the edit script re-run in a temporary copy on the text it started from (the Pass 28 result, commit bec1ece) reproduces every
file it writes (paper/outline.tex, four supplement notes, Table 1 and its generator are never edited by hand); the printed text changes
only in the new author line and the eight wording fixes the first author asked for, and the supplement files take their fixes without
tracked markup (run from the repository root: python -m pytest paper/tools/test_pass29_edits.py)."""
import re
import runpy
import subprocess
import sys
from pathlib import Path

import pytest

TOOLS = Path(__file__).resolve().parent
REPO = TOOLS.parents[1]
sys.path.insert(0, str(TOOLS))
from abstract_words import count  # noqa: E402
from accept_changes import accept  # noqa: E402

START = "bec1ece"
BASE = "fd8c2fc"  # the Pass 29 result; later passes and hand edits change these files, so they are compared with this commit
MAIN = [  # (old, new) in the printed text of paper/outline.tex, the first author's wording of 2026-11-10
    (r"and 0.213 at each penalty's strongest setting.",
     r"and 0.213 at each penalty's strongest setting, where batch removal was not matched."),
    (r"Most neighbouring readings were close. In 44 of the 55",
     r"Over the 13 ranked pairs, 160 of the 195 within-pair differences in biology at $b^*$ (15 penalty pairs each) were under 0.02 and "
     r"4 were 0.04 or more. Most neighbouring readings were close. In 44 of the 55"),
    (r"kept similar biology at matched batch removal and differed mainly at strong settings. At $b^*$, 160 of the 195 within-pair "
     r"differences in biology over the 13 ranked pairs were under 0.02, and 4 were 0.04 or more. The range",
     r"kept similar biology at matched batch removal (Res-2) and differed mainly at strong settings. The range"),
    (r"mean Kendall's $\tau_b$ 0.456, $p=0.00049$)",
     r"mean Kendall's $\tau_b$ 0.456, $p=0.00049$ over the 13 pairs, as the rule prescribes; $p=0.0156$ over the six task families)"),
    (r"on five of six real tasks (median difference $-0.002$)",
     r"on five of six real tasks (pancreas by less than 0.001; median difference $-0.002$)"),
    (r"Harmony, averaged over its settings, differed", r"Harmony, averaged over its seven settings rather than read at one, differed"),
    (r"These readouts show small average changes at the first crossing, without a consistent disadvantage for the discriminator "
     r"across decoders.",
     r"These readouts show small average changes at the first crossing; the discriminator had the largest purity loss with the "
     r"batch-conditioned decoder and the lowest F1 on three tasks with the unconditioned decoder."),
    (r"at first-crossing strengths, without a consistent disadvantage for the discriminator across decoders (Res-5)",
     r"at first-crossing strengths, and the discriminator had the largest purity loss with the batch-conditioned decoder (Res-5)"),
]
SUPP = {  # path -> (old, new), edited in place
    "paper/supp/note5.tex": (
        r"the matched strength: among the strengths with no failed fit, the one whose seed-mean batch score is nearest $b^*$, ties to the "
        r"smaller strength, edge and extension strengths included.",
        r"the matched strength: of the two fitted strengths that bracket the first crossing of $b^*$, the one whose seed-mean batch score "
        r"is nearer $b^*$ (ties to the smaller strength), or the upper one when the lower is plain scVI, among the strengths with no "
        r"failed fit, edge and extension strengths included. This first-crossing rule replaced the earlier strength nearest $b^*$ on any "
        r"crossing after the first results had been read (Supplementary Note 10)."),
    "paper/supp/note10.tex": (
        r"the matched-strength table being read from those results;",
        r"the matched-strength table being read from those results (this strength rule was later replaced by the first-crossing rule "
        r"described below);"),
    "paper/supp/note9.tex": (
        r"(below plain scVI on five, lowest among penalties on three)",
        r"(below plain scVI on five, pancreas by less than 0.001; lowest among penalties on three)"),
    "paper/supp/note12.tex": (
        r"The discriminator's larger losses in the earlier reading did not hold at the first crossing across decoders.",
        r"The discriminator's larger losses in the earlier reading were not seen at the first crossing; with the batch-conditioned "
        r"decoder its mean purity loss was still the largest among the penalties (Supplementary Table S10)."),
    "paper/table1.tex": (r"Exact 95\% intervals", r"Nominal 95\% intervals"),
    "paper/tools/make_table1.py": (r"Exact 95\% intervals", r"Nominal 95\% intervals"),
}
EDITED = ["paper/outline.tex", *SUPP]


def committed(rel, commit=BASE):
    return subprocess.run(["git", "-C", str(REPO), "show", f"{commit}:{rel}"], capture_output=True, text=True, check=True).stdout


def test_pass29_script_reproduces_the_committed_files(tmp_path, monkeypatch, capsys):  # test-reason: runs pass29_edits.py, which no other test runs; catches a hand edit of a file it writes, a file written besides them, a printed change outside the author line and the eight fixes, a wording that differs from the first author's, struck text that is not the old wording, an author line out of place, tracked markup in a supplement file, and word counts that move
    start = {}
    for rel in EDITED:
        r = subprocess.run(["git", "-C", str(REPO), "show", f"{START}:{rel}"], capture_output=True, text=True)
        if r.returncode:
            pytest.skip(f"{START} not available")  # pragma: no cover  # shallow copies lack the history
        start[rel] = r.stdout
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / rel).write_text(r.stdout)
    monkeypatch.setenv("PASS29_ROOT", str(tmp_path))
    runpy.run_path(str(TOOLS / "pass29_edits.py"), run_name="__main__")
    for rel in EDITED:
        assert (tmp_path / rel).read_text() == committed(rel), rel
    assert sorted(p.relative_to(tmp_path).as_posix() for p in tmp_path.rglob("*") if p.is_file()) == sorted(EDITED)
    assert "main text 5488 words (limit 5500); main captions, Table 1 included, 422 words (limit 450)" in capsys.readouterr().out
    raw = committed("paper/outline.tex")
    assert len(count(raw)[0]) == 330                     # abstract, limit 350
    # the printed text is the bec1ece text with the eight fixes and the author line (directly before the Pass 28 line), nothing else
    lines = raw.split("\n")
    author = lines[[ln.startswith(r"\definechangesauthor[name={Pass 28}") for ln in lines].index(True) - 1]
    assert author.startswith(r"\definechangesauthor[name={Pass 29}") and "]{P29}% " in author
    assert author.endswith("; change log paper/pass29_changelog.md")
    want = accept(start["paper/outline.tex"]).replace(r"\definechangesauthor[name={Pass 28}", author + "\n" + r"\definechangesauthor[name={Pass 28}")
    for old, new in MAIN:
        assert want.count(old) == 1, old
        want = want.replace(old, new)
    assert accept(raw) == want
    text = " ".join(accept(raw).split())
    for old, new in MAIN:
        assert text.count(new) == 1 and text.count(old) == new.count(old), new   # the old Res-2 words stay, after the new sentence
    # struck text keeps the old wording: with the P29 changes rejected, the file is the bec1ece file again
    rejected = re.sub(r"\\added\[id=P29\]\{[^{}]*\}", "", raw.replace(author + "\n", ""))
    rejected = re.sub(r"\\deleted\[id=P29\]\{([^{}]*)\}", lambda m: m[1], rejected)
    rejected = re.sub(r"\\replaced\[id=P29\]\{[^{}]*\}\{([^{}]*)\}", lambda m: m[1], rejected)
    assert rejected == start["paper/outline.tex"]
    # supplement notes, Table 1 and its generator: the one replacement each, without tracked markup
    for rel, (old, new) in SUPP.items():
        s = committed(rel)
        assert start[rel].count(old) == 1 and s == start[rel].replace(old, new), rel
        assert not re.search(r"\\(replaced|added|deleted)\b", s), rel
