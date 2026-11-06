# Pass 22 change log (2026-11-06, tracked id P22, red)

Branch `paper`, starting point Pass 21 (93a209c). The user asked for a pass that cuts fluff and keeps content.

## Scope and rules

Main text only: Background, Methods, Results, Discussion. Not edited: abstract, figure captions, tables, Conclusions,
Declarations, the AI-use paragraph, the supplement. Rules applied: every number, citation, cross-reference, caveat and hedge
stays; claim strength unchanged; terms of art unchanged; no em dashes. Cuts are filler openers ("It is important", "Despite their
promise"), announcements ("We first report which..."), restatements of a rule already stated nearby, and long lists regrouped
(the collapse-rate maxima are grouped by task instead of repeating the task after each penalty, values unchanged).

## Changes (`paper/tools/pass22_edits.py` with the new texts in `paper/tools/pass22_cut.json`; commits a18ffce, 145a36a)

- 28 paragraphs or list items rewritten; 1 list item deleted (Res-8 "Descriptive; best-of-k labelled optimistic.", which repeats
  the rule bullet and the closing sentence above it). Main-text words (Background to Discussion, captions excluded):
  9,407 -> 9,022 (-385).
- Moved, not dropped: the low-signal batch scores (0.638/0.657, 0.812/0.819, 0.656/0.665) now appear once, in Res-1, and Methods
  points to Results for them; the replication limits appear once, in Disc-4 (Disc-1 points there); a repeat of the per-task
  replication limit in the Disc-4 backbone bullet was removed (stated in the bullet above).
- The two sensitivity analyses of the replication (Methods) are now two sentences ("The first ... The second ...") instead of one
  sentence joined by semicolons.
- Each edited line is written as word-level tracked changes against its printed text; earlier tracked marks on those 29 lines
  are accepted first (changes cannot nest), so the tracked copy shows the Pass 22 cuts there but no longer the older marks
  on those lines. Struck citations are wrapped in \mbox, as in earlier passes, because the strikeout breaks on a bare citation.
- Paragraph-by-paragraph old and new text: artifact pass22_pairs.md.

## Independent check

A second model (Claude Sonnet sub-agent, read-only) compared all 28 old/new pairs sentence by sentence: no major findings, 9 minor.
Fixed: 'key' biological information (Background 1), 'have limitations that challenge their utility in practice' (Background 6),
'fixed penalty' and 'aims to align' (Methods), 'from different biological settings' (data paragraph), 'and not part of it' for
the reports beside the replication rule, the Res-1 heading back to a neutral signpost, the Res-1 pointer for low-signal pairs,
'suggesting' restored in Disc-2. Kept: the Disc-4 backbone bullet no longer repeats the same-tasks replication limit (stated in
the bullet above it).

## Number ledger

The cut changed the raw-source anchors of five ledger rows (one pointed into struck Pass 15 text; four spanned words that now
carry Pass 22 marks). `paper/pass22_numbers.csv` re-points them (P22E01-E03, P22D1-D2, column supersedes); the values are
re-read from the same sources. `check_ledger.py` lists the new ledger.

## Checks

- `check_ledger.py` on the workstation: 1,122 values re-read (1,117 + 5 re-pointed), ledger check OK, 0 failures.
- `check_tests.py --base 93a209c --src paper/tools -- paper/tools`: OK, no WARN; re-running `pass22_edits.py` on the 93a209c
  outline reproduces the committed one. Abstract 328 words (unchanged).
- Build (doc-build-gate, tectonic 0.16.9) from the saved zip extracted into an empty folder: tracked 63 pages, clean 54 pages,
  13 of 13 figures, no failing log class (2 underfull boxes, as before); no heading at a page bottom. Pages viewed: clean and
  tracked pages 1-18 (all main text).

## Re-run, from the repository root

```
git checkout 93a209c -- paper/outline.tex
python paper/tools/pass22_edits.py
python paper/tools/accept_changes.py
python paper/tools/check_ledger.py
python scripts/check_tests.py --base 93a209c --src paper/tools -- paper/tools
```
