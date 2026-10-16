# Pre-tag display audit for tag prereg-tier12-v3 (run 2026-10-16)

**Question.** Since 2026-10-02 00:00 New York time, was any value of an X1 or X13 score (any file in the SI-79 hash list) shown in a session transcript or a text file, beyond the three displays in docs/PREREG.md section 9?
**Answer.** One line is flagged by the rule and is not a known display (candidate, below); `new_display_found` = true. Display 3 (X13_lung_scanvi_l0_c1_s0_8e6f5009, frame 4295f20e msg 117) was found again and is known.

**Input.** Repository HEAD 8f223c8 (hash list `experiments/x1_x13_score_hashes_20261015/score_file_hashes.csv`, 1,129 files). Data root (read-only) /home/kendall/experiment_data/wasserstein-critic-deconfounding. Python 3.11, conda env `python`. No score file is opened; only counts and digit-masked lines are stored.

**Files.** `audit_displayed_values.py` (copy of the 2026-10-16 script; only change: KNOWN also holds Display 3), `run_pretag_audit.py` (three steps), `audit_pretag.json` (all counts, candidates, the 83 manually read lines, all masked).

**Re-run** (from the repository root; the partial results go to ../partial, outside the repository, so they are not read as committed copies):
1. python: `import sys; sys.path.insert(0,'experiments/pretag_audit_20261016'); import run_pretag_audit as R; R.files_part()` (about 150 s)
2. repl tool (needs host.query; same sys.path line and import): `R.transcripts_part(host, '<your own frame id>')` (about 400 s, one query per message)
3. python: `R.finish('<your own frame id>', '<git rev-parse HEAD>', readings)`; `readings` maps "frame8:msg" to the reading of a candidate. Writes audit_pretag.json.
Rerun later: the cut-off moves, so later messages are read.

**Counts.**
- Cut-off: 2026-10-16 16:35:57 EDT. Sessions in scope 75, searched 74 (excluded: own session d7cc219b); 71 mention an identifier or folder.
- Messages read in full: 42,592 (3,321 mention an identifier or folder). Identifiers 565 (file stems of all 1,129 listed files, attempt logs included); folders 197.
- Files scanned: 26,561 (5,736 mention an identifier; 43 outside review/out and review/verify).
- Flagged lines by class (files + transcripts): copy of committed docs 3,056; committed repository file 600; committed notebook 472; ordinary word (failures/div/nan) 42; older than every listed file 15; known displays 1/2/3: 6/6/1 lines; unclassified 83.
- The 83 unclassified lines were read (masked): 1 candidate, 28 stored project-memory notes, 54 pilot/code/review lines with no listed X1/X13 identifier.
- Candidate: frame 1e3cd07e msg 45, 2026-10-12 23:17 New York, X13_atac_large_sysvi_l10_c1_s0_a617540b: a latent-file listing followed by that latent's fit-settings JSON (lr, eps, weight decay, KL constants). By its masked text these are training settings, not metrics; the rule flags it and the audit does not clear it. The lead decides.

**Limits.** Same as 2026-10-16: a value printed with no identifier, folder or metric name nearby can be missed; Codex sessions left only output files; what an agent read without printing is unknown; messages after the cut-off are not read.
