# Pass 20 change log (2026-11-06, tracked id P20, blue)

Branch `paper`, starting point Pass 19 (f20fe27). The user asked for an ethics wording to be drafted and stated that there are no
competing interests.

## Changes (Declarations; `paper/tools/pass20_edits.py`, commit d28b747)

- **Ethics approval and consent to participate** (drafted; for the authors to confirm): "Not applicable. This study analyses only
  publicly available, previously published data: the integration tasks of the single-cell integration benchmark (figshare article
  12420968, version 8), which hold human and mouse single-cell data from earlier studies, and two simulated datasets. No new data
  were collected from human participants or animals." It replaces the Pass 3 sentence and its \tbd. BMC asks for "Not applicable"
  when a study involves no new human or animal data; the sentence after it says why. If the authors' institution issued a
  determination (for example an IRB "not human subjects research" letter), it should be named here.
- **Competing interests**: "The authors declare that they have no competing interests." (BMC's standard sentence), replacing the
  \tbd, as the user stated.

Four author items remain: release tag and commit, funding, authors' contributions, acknowledgements.

## Checks

- `check_tests.py --base f20fe27 --src paper/tools -- paper/tools`: OK, no WARN (new test row re-runs `pass20_edits.py` from
  f20fe27 and compares with d28b747).
- `check_ledger.py` on the workstation with the Pass 20 outline: ledger check OK, 0 failures. Abstract 328 words.
- Build (doc-build-gate, tectonic 0.16.9) from the saved zip extracted into an empty folder: tracked 64 pages, clean 55 pages, 13 of
  13 figures each, no failing log class (2 underfull boxes, as before).
- Pages viewed: every page that differs from the Pass 19 build (tracked 14-30, reflow after the Declarations; clean 11).

## Re-run, from the repository root

```
git checkout f20fe27 -- paper/outline.tex
python paper/tools/pass20_edits.py
python paper/tools/accept_changes.py
python paper/tools/check_ledger.py
python scripts/check_tests.py --base f20fe27 --src paper/tools -- paper/tools
```
