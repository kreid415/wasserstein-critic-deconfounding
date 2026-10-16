# Git and reproducibility (always loaded)

Translated from the user's reproducibility and GitHub-backup rules.

1. All work must be reproducible from code in this GitHub repository. A result without committed code that regenerates it is incomplete.
2. Commit as work proceeds, in small commits with plain messages; do not leave one large dump for the end. Push to the remote with the `gh` CLI or the token supplied by the environment, except when the user has said not to push (check `docs/claude/STATUS.md` and the current session instructions first). Confirm repository name and visibility with the user before creating any new remote. Never push to a branch another session is working on.
3. Never commit credentials, tokens, keys, personal details or large raw data. Commit the fetch script instead of the data. The repository is public: add no email address except `kreid415@users.noreply.github.com` (the published author contact already in README.md and the manuscript template stays); git identity is user.name `kreid415`.
4. Every figure, table and number must trace to a script in the repository. Record, next to each result: the environment (lockfile or `conda env export`, plus the env name), random seeds, data provenance (accession, URL and file checksum, not just a file name), and the commit that produced it.
5. Keep a README with the exact end-to-end re-run commands, and a `MANIFEST.sha256` over output files when outputs are shared.
6. `.gitignore` hides `*.csv`, `*.tex`, `*.png`, `**/results/` and similar; use `git check-ignore -v <path>` before assuming a file is tracked, and `git add -f` only on purpose (see `docs/claude/notes/repo-conventions.md`).
7. Record each experiment, result, decision and correction as an entry in the lab notebook (`notebook/`, append-only, hash-chained; fix mistakes with a CORRECTION entry, never by editing) and keep the paper outline and its CHANGELOG in step when an entry changes the argument.
8. The reproducibility audit (every number traceable, no secrets, no large files) runs before wrap-up and before any tag.
