# Errata to apply at the next tag

Text-only errors found in files that are frozen at tag `prereg-tier12-v2` (c88cce3). The running experiments
(A1, the filler scoring) use the tag checkout, so these are fixed only when the next tag is cut (e.g. the X1 tag after
R1), each in its own commit, with a test run showing behaviour is unchanged. Delete an entry once it is applied.

| Found | File | Error | Fix |
|---|---|---|---|
| 2026-10-04, background review 7fec098b | `scripts/prereg_rules.py`, module docstring | says `default_half_extensions()` "returns the A1 rows that fit the default half"; it returns the extension specs (experiment A1, task, cond, arm, lam), one per cell and side; the rows come from `extension_rows(M, specs)` | reword the docstring sentence to match the function's own docstring |
| 2026-10-01, background review 81c25518 | `scripts/build_counts_pilot_manifest.py`, docstring line 13 | "old atac_small peaks: critics ~1, discriminator ~20" were computed with a 4-metric batch mean (ilisi, asw_batch, graph_conn, pcr) without saying so; kBET is NaN in all 6,305 rows of the old scored_all.csv (checked 2026-10-07), so it could not enter | add "(4-metric batch mean: kBET was not computed in the pre-revision scores)" |
| 2026-10-04, background review 5909b10a | `tests/prereg/synth.py`, comment above `PROVENANCE` | says the dict is the scorer provenance "as score_scib_native.py writes it"; the scorer writes 7 columns (`score_host`, `cpu_model` too), the dict holds the 5 that `prereg_rules.PROVENANCE_COLS` checks | say "the 5 provenance columns prereg_rules checks (PROVENANCE_COLS); the scorer also writes score_host and cpu_model" |
| 2026-10-10, background review 5e460344 | `docs/SPECS_missing_arms.md`, lines 327 and 335 | cite Harmony "Methods 5.4"; neither the PMC author manuscript (PMC6884693, sections unnumbered) nor the bioRxiv preprint (10.1101/461954) has a section 5.4. The content is right: the defaults incl. lambda=1 are in Online Methods > Analysis Details > Harmony Parameters, and lambda is the ridge penalty of Eq. 14 (Online Methods > Linear Mixture Model Correction). The 2018 preprint has no ridge term (its defaults list has no lambda) | replace "Methods 5.4" with those section names and "Eq. 14"; say the ridge lambda is from the published version |
