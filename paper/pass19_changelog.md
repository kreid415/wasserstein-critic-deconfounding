# Pass 19 change log (2026-11-06, tracked id P19, blue)

Branch `paper`, starting point Pass 18 (2ff8608). The user asked for the AI-tool statement to be filled in.

## Change

Methods, last paragraph before Results (`paper/tools/pass19_edits.py`, commit 4a3213e): the Pass 3 placeholder ("Large language
model tools were used for ... \tbd{first author to confirm tools, versions and roles}") is replaced by a statement headed
*Use of large language models*. It names the tools (Claude from Anthropic, used through the Claude Science and Claude Code
applications; Codex from OpenAI), what they did (wrote and checked analysis, scoring and figure code; prepared, ran and monitored
jobs on the workstation and the cluster; assisted with literature searches; checked numbers against source files; reviewed the
text independently; drafted and edited text), that the study design and the decisions about what was run, measured and reported
were made or approved by the authors, that no figure is an image from a generative model (all are drawn by plotting scripts in
`paper/figure_scripts/`), that every number is re-read by `paper/tools/check_ledger.py`, and that the authors checked all outputs
and take responsibility for the content (139 words).

Why here and in this form: BMC journals ask that "use of an LLM should be properly documented in the Methods section"; LLM
use for copy editing alone need not be declared, and LLMs are not authors. Model versions are not given: several versions were
used over the months of the project and no record lists them all; the first author may add them.

Six author items remain open: ethics wording, release tag and commit, competing interests, funding, authors' contributions,
acknowledgements.

## Checks

- `check_tests.py --base 2ff8608 --src paper/tools -- paper/tools`: OK, no WARN (82 passed, 2 skipped without the data root;
  the new test row re-runs `pass19_edits.py` from 2ff8608 and compares with 4a3213e).
- `check_ledger.py` on the workstation with the Pass 19 outline: ledger check OK, 0 failures.
- Abstract 328 words; 348 by the hyphen-split count (limit 350). `accept_changes.py --check`: up to date.
- Build (doc-build-gate, tectonic 0.16.9) from the saved zip extracted into an empty folder: tracked 64 pages, clean 55 pages, 13 of
  13 figures each, no failing log class (2 underfull boxes, as before); page-ink check flags only the page number at the bottom
  margin, as before.
- Pages viewed: every page that differs from the Pass 18 build (tracked 7-30, clean 6-18; the statement is on tracked page 7 and
  clean page 6, the rest is reflow). No supplementary heading at a page bottom.
- The response is unchanged.

## Re-run, from the repository root

```
git checkout 2ff8608 -- paper/outline.tex
python paper/tools/pass19_edits.py
python paper/tools/accept_changes.py
python paper/tools/check_ledger.py
python scripts/check_tests.py --base 2ff8608 --src paper/tools -- paper/tools
```
