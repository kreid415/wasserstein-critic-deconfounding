# Prose Introduction/Methods merge, 2026-10-22

Base: the first author's prose draft of Introduction and Methods (pasted 2026-10-22), merged into `paper/outline.tex` (previous bullet version: commit 8c89960).

| File | Made by | What |
|---|---|---|
| intro_tracked.tex, intro_changelog.md, new_refs.bib | Claude Code, claude-fable-5-1, effort high | Introduction with tracked corrections; change log with sources |
| methods_tracked.tex, methods_changelog.md | Claude Code, claude-fable-5-1, effort high | Methods with tracked corrections, three added shared paragraphs; change log, moved-to-Results sentences, Table 2 content, supplement map |
| results_supplement_outline.tex, topic_comparison.md | Claude Science sub-agent, claude-fable-5-1 | Results outline with analysis rules, supplement outline (Notes 1-12, Tables S1-S6), topic comparison |
| figures_tables.tex, make_mock_figures.py, ../figures/mock/fig1-6.png | Claude Science sub-agent, claude-fable-5-1 | Mock figures (synthetic data, numpy seed 20261022, MOCK watermark) and Tables 1-4 |

Integration edits in outline.tex: `\protect\cite...` inside struck-out text wrapped in `\mbox{}` (ulem); todo notes inside deletions shown as `\textit{todo: ...}`; Table 2 cells list the six strengths from `tier12_v3/rules/r1_pass2/r1_record.json` x1_grid (via methods_changelog.md section 3).
Build: `tectonic paper/outline.tex` (use `\usepackage[final]{changes}` for clean text). Regenerate mocks: `python paper/drafts_20261022/make_mock_figures.py`.
