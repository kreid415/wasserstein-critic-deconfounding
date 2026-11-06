"""Pass 19 edits (2026-11-06, tracked id P19): the AI-tool statement in Methods, which BMC journals require ("Use of an LLM
should be properly documented in the Methods section"), replaces the Pass 3 placeholder and its \\tbd. Tools are named by
product and maker; model versions are not given, because several versions were used over the months of the project and no
record lists them all. No numbers are written. Every old text must occur exactly once or the script stops.
Run from the repository root:  python paper/tools/pass19_edits.py   (PASS19_ROOT overrides the root of the edited files)
Then: python paper/tools/accept_changes.py ; python paper/tools/check_ledger.py"""
import os
from pathlib import Path

ROOT = Path(os.environ.get("PASS19_ROOT") or Path(__file__).resolve().parents[2])


def edit(path, pairs):
    p = ROOT / path
    s = p.read_text()
    for old, new in pairs:
        assert s.count(old) == 1, (path, s.count(old), old[:100])
        s = s.replace(old, new)
    p.write_text(s)


OLD = (r"\added[id=P3]{\qquad Large language model tools were used for writing and checking analysis code, running and checking "
       r"computational jobs, and drafting and editing text. The authors checked all outputs and take responsibility for the content "
       r"\tbd{first author to confirm tools, versions and roles}.}")
NEW = (r"\added[id=P19]{\qquad \emph{Use of large language models.} Large language model tools were used in this work: Claude "
       r"(Anthropic), through the Claude Science and Claude Code applications, and Codex (OpenAI). They wrote and checked analysis, "
       r"scoring and figure code; prepared, ran and monitored computational jobs on the workstation and the cluster; assisted with "
       r"literature searches; checked the numbers in the text against their source files; reviewed the text independently; and "
       r"drafted and edited text. The study design and the decisions about what was run, measured and reported were made or "
       r"approved by the authors. All figures are drawn from the data by plotting scripts in the repository (no image was made "
       r"by a generative model), and every number in the paper is re-read from its source file by a script "
       r"(\texttt{paper/tools/check\_ledger.py}). The authors checked all outputs and take responsibility for the content.}")
edit("paper/outline.tex", [
    (OLD, NEW),
    ("\\definechangesauthor[name={Pass 13}, color=orange]{P13}",
     "\\definechangesauthor[name={Pass 19}, color=blue]{P19}% AI-tool statement in Methods; change log paper/pass19_changelog.md\n"
     "\\definechangesauthor[name={Pass 13}, color=orange]{P13}"),
])
