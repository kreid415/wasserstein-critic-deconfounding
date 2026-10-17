# First prompt for a Claude Code session resuming this project

Paste the block below as the first message. Start the session on branch `claude-code-migration` (or merge it into
`final-review-fixes` first and start there). Written 2026-10-16; after the merge has happened once, drop step 2.

```
You are resuming a pre-registered single-cell batch-correction study (Wasserstein critics vs discriminator adversaries). Read CLAUDE.md fully, then docs/claude/STATUS.md and docs/claude/LAUNCH_SINKHORN_STAGE1.md. Then, in this order:

1. `git fetch --all --tags`. Report the heads of final-review-fixes and claude-code-migration and confirm tag prereg-tier12-v3 = 10a1ca5.
2. If claude-code-migration is not yet merged: `git checkout final-review-fixes && git pull && git merge origin/claude-code-migration`, push, and stay on final-review-fixes. All later work, including every notebook or ledger entry, happens on that branch.
3. Confirm the environment: `which python` must resolve to the wcd-cpu env the start-up hook built (else use the full interpreter path the hook printed). Run, from the repo root, `python -m pytest -q tests/prereg tests/stage tests/analysis` and report the counts (expected 283 passed, 2 skipped).
4. Report, before changing anything: what the newest notebook entries and `git log` show that STATUS.md does not; what is running or paused; what the next experimental step is and what it needs from me.

Hard rules: never read an X1 or X13 score, latent, failures.csv or attempts file (CONSTRAINTS.md SI-78/79); do not change files frozen by tag prereg-tier12-v3 without asking; no cluster or GPU launch without my explicit go; ask me only about decisions that change what is run, measured or analysed (SI-92) and one at a time. This VM has no GPU and no project data: the Sinkhorn redo stage 1 is launched by me on the workstation with the commands in LAUNCH_SINKHORN_STAGE1.md; your first substantive task after the report is the stage-2 plan and preflight that LAUNCH_SINKHORN_STAGE1.md finding 3 asks for (STATUS step 5).
```
