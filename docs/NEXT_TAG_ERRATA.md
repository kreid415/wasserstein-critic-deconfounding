# Errata to apply at the next tag

Text-only errors found in files that are frozen at tag `prereg-tier12-v2` (c88cce3). The running experiments
(A1, the filler scoring) use the tag checkout, so these are fixed only when the next tag is cut (e.g. the X1 tag after
R1), each in its own commit, with a test run showing behaviour is unchanged. Delete an entry once it is applied.

| Found | File | Error | Fix |
|---|---|---|---|
| 2026-10-04, background review 7fec098b | `scripts/prereg_rules.py`, module docstring | says `default_half_extensions()` "returns the A1 rows that fit the default half"; it returns the extension specs (experiment A1, task, cond, arm, lam), one per cell and side; the rows come from `extension_rows(M, specs)` | reword the docstring sentence to match the function's own docstring |
| 2026-10-04, background review 5909b10a | `tests/prereg/synth.py`, comment above `PROVENANCE` | says the dict is the scorer provenance "as score_scib_native.py writes it"; the scorer writes 7 columns (`score_host`, `cpu_model` too), the dict holds the 5 that `prereg_rules.PROVENANCE_COLS` checks | say "the 5 provenance columns prereg_rules checks (PROVENANCE_COLS); the scorer also writes score_host and cpu_model" |
