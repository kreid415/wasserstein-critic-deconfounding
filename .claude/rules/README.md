# Translation of the 22 Claude Science profile rows into Claude Code rules

Files here load every session (no `paths:` frontmatter). Total size is kept under 250 lines. Each row is a user-wide rule from the Claude Science profile memory (dump of 2026-10-16). Row ids are `mem_...` ids.

| Row id | Original rule (short) | What happened |
|---|---|---|
| mem_5007353a67fd | Concise responses | working-style.md items 1 and 2 |
| mem_71e8afc96fe4 | Cluster rules (storage, scratch purge, no account-wide cancel) | compute.md items 1 to 3; full skill at `.claude/skills/cluster-rules` |
| mem_2e7b05593cb6 | Reproducible from GitHub code; `reproducible` skill | git-and-reproducibility.md items 1 and 8; skill replaced by the audit step |
| mem_569c0582d22f | Reproducibility defaults (env, seeds, provenance, manifest) | git-and-reproducibility.md items 4 and 5; "artifact version_id" became file checksum and commit |
| mem_0dbddf2c5150 | Commit as you go, push, no secrets or large data | git-and-reproducibility.md items 2 and 3; push is conditional on the user not having said otherwise |
| mem_ccc7db104855 | Verify against the primary source | research-integrity.md item 2 |
| mem_2f42b49c471a | Routing: run sub-agents as matching specialist profiles | Rewritten as working-style.md items 8 and 9 (Task subagents in `.claude/agents/`) |
| mem_7de891a47aa0 | Workflow checks: preflight, claim check, document check, fail-loud, ledger | research-integrity.md items 3 to 6 and code-changes.md item 6; skill names dropped, checks kept as plain steps; "run-status-board" became on-request status checks (compute.md item 5) |
| mem_908b0c2b14fe | Biological diagram rule (bio-diagram skill, licensed icons) | working-style.md item 12 (principles kept; skill and profile routing dropped) |
| mem_58d6ce87d050 | Auto-offload to Codex over ssh | Dropped: no Codex host or ssh target exists in Claude Code cloud sessions; subagents replace offloading |
| mem_f9525a68baa2 | Top priority: correctness first, main agent on Opus | research-integrity.md item 1; model rule folded into working-style.md item 9 |
| mem_12e65d2ecfa5 | Session handoff size check and starter message | Rewritten as working-style.md item 11 (write `docs/claude/STATUS.md`, give a starter message); the frame-size measurement is dropped (no frame store) |
| mem_272901520453 | job-watch skill (validated on JHPCE) | compute.md item 5; script bundled at `.claude/skills/job-watch/scripts/jw.py` |
| mem_cb34fe676998 | PDF rule: text not images | working-style.md item 4; tool names replaced by pdftotext, pypdf, DOI fetch |
| mem_1b00d05a2cca | Refusal audit of 378 Claude Science frames (2026-10-09) | Dropped: an audit log, not a rule; it refers to Claude Science frame ids |
| mem_543b7c1363d3 | No-jargon rule | working-style.md item 3 |
| mem_f4a8dd9587dc | Ponytail rule (smallest change, review, audit) | code-changes.md items 1, 2, 7, 8; skills ported to `.claude/skills/ponytail*` |
| mem_7268c6d53324 | Fewest-tests rule | code-changes.md items 3 to 5 (`scripts/check_tests.py` exists in the repo) |
| mem_c4a72472d151 | Decision format (one per message, at most 120 words, cards) | working-style.md items 5 and 6; ask_user cards became plain-text options |
| mem_5837c38a8dfa | Delegate-first (more than 5 tool calls go to a Sonnet sub-agent) | working-style.md items 8 and 10; usage statistics dropped |
| mem_c1facf8d2f2e | Pin model claude-sonnet-5-5 on every delegation; Opus for judgment | working-style.md item 9 as `model: sonnet` vs opus; the literal model id is not hard-coded; the "check every result independently" part kept |
| mem_58ecb3bc5722 | Defer non-experiment decisions (SI-92) | working-style.md item 7 |

Safety and honesty content was kept in every translated rule; nothing was weakened. The biosecurity stop in working-style.md item 9 is carried over from the "hazardous-bio material" clause.
