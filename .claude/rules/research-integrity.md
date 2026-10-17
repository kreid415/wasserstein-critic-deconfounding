# Research integrity (always loaded)

Translated from the user's Claude Science profile rules. These rules never trade accuracy for speed or cost.

1. Correct results the first time outrank speed and cost. Prefer the strongest reasonable setting plus an independent check over a cheaper shortcut. Do not lower model or effort to save budget.
2. Verify every claim against the primary source before suggesting or stating it (paper full text, GEO or BioStudies record, package source, committed code, data file) and say what you checked. Never replace checking with a "to verify" caveat.
3. Independent check before reporting: before any message that says done, fixed or verified, or that reports numbers, re-read the numbers from the files that hold them (never retype from memory or prose) and check each claim against the evidence. A subagent's verdict is verified by you before you repeat it.
4. Before submitting any job, sweep or new metric: write a short check list (config exists in the grid, expected row count from the recorded batch count, output path, seeds, environment, blinding status) and launch only when every item passes. Calibrate cost estimates on one full production-setting run. See `docs/claude/notes/pitfalls.md`.
5. Runner, launcher and metric code must fail loudly: a failed config writes a NaN row with a `failed` flag, never a silent drop; a filter that selects zero configs is an error.
6. Before delivering a document, deck or figure set, build it and check it against `CONSTRAINTS.md` (the ledger of standing rules, SI-01 to SI-92). When the user states or corrects a rule, add it to `CONSTRAINTS.md` with its source and find every file that still states the old version.
7. Blinding of the pre-registered experiments (`CONSTRAINTS.md` SI-78 and SI-79; `docs/PREREG.md`): a score is examined when a person reads its value or an analysis script uses it. Never open, print, summarise or copy any X1 or X13 score file, score row, latent, failures table, attempt record or status record; hashing and counting by status, with no arm and no score, are allowed as those rules state. Do not write X1 or X13 score values into any file. If unsure whether a file holds those values, do not read it.
8. Only report statistics the notes allow: the independent unit is the dataset; no bootstrap p at n = 6; no Nadeau-Bengio claim across datasets (`docs/claude/notes/statistics-and-claims.md`).
9. Never claim a verification you did not perform. If you cannot check something, say so and ask.
