---
name: session-handoff
description: Write and read a file-based handoff so a fresh Claude Code session can resume this project. Use when the user says to wrap up, hand off, or continue later; before ending a long session; when the context is getting long; or at the start of a session to find the resume point. The handoff lives in docs/claude/STATUS.md and is committed to git.
---
# Session handoff (file-based)

> **Helper loading (Claude Code).** `kernel.py` is plain Python (standard library only). Load it with
> `p = ".claude/skills/session-handoff/kernel.py"; exec(compile(open(p).read(), p, "exec"), globals())`.

A new session starts with only the git clone and CLAUDE.md. Nothing in memory, no notebook kernel, no earlier chat survives. The handoff file is therefore the single resume point: `docs/claude/STATUS.md` (committed). Keep it to about one page; every sentence should change what the next session does.

## When to write
- The user asks to wrap up, hand off, or continue later.
- A wave of jobs has just been launched or has just finished (the run state changes).
- The conversation is long (for example, you notice you are re-reading earlier material) or you are about to stop for any reason.

## When to read
At the start of every session: read `docs/claude/STATUS.md` first, then run `git log --oneline -15` and `git status`, then verify the handoff against the repository before acting (step 3 below).

## Write procedure
1. Fill `references/handoff-template.md`. Sections: Objective, State (done and verified / done but unverified / in progress / blocked), Live runs (host, job ids, a read-only command to check each), Decisions, Dead ends, Next steps (ordered, naming files, parameters, commands), Open questions for the user. Empty sections are worse than absent ones.
2. State what each claim rests on: commit hash, `file:line`, notebook id (NB-...), ledger id (SI-...). Read numbers back from result files; never retype them.
3. Run `handoff_check("docs/claude/STATUS.md")`. It reports: referenced repo paths that do not exist, commit hashes that do not resolve, strings that look like secrets, and missing required sections. Fix every item it reports.
4. Anything expensive to rebuild (a fitted model, a merged table) must exist as a file on a durable path before you stop, and the handoff must give the exact path or the command that regenerates it. Do not write data or secrets into the handoff.
5. Append a lab-notebook NOTE entry that points to the handoff (see `lab-notebook`), then commit `docs/claude/STATUS.md` with a message that starts with `handoff:`. Do not push unless the user has asked you to.
6. End the reply with the start message from `handoff_start_message(...)` (about 150 words, names the file and the single next action).

## Read procedure (start of a session)
1. Read `docs/claude/STATUS.md`. 2. `handoff_check(...)` on it: paths and commits must still resolve. 3. For every "done and verified" line, re-verify the cheapest one against the file it cites; if it fails, tell the user before continuing. 4. Ask the open questions together, once.

## Rules
- Blinded or embargoed results (see CLAUDE.md and `.claude/rules/`) never appear in the handoff.
- The handoff states what is known to be true. Mark anything unchecked as "unverified".
- Do not copy this file's content into CLAUDE.md; CLAUDE.md points to it.
