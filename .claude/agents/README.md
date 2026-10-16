# Subagents (ported from Claude Science specialist profiles)

Each file is a Claude Code subagent (YAML frontmatter with `name`, `description`, `model`, optional `tools` and `skills`, then the system prompt). Claude Code starts one when the main session's task matches the `description`, or when you ask for it by name ("use the rigor-reviewer agent on ..."). The `skills:` list preloads the project skills in `.claude/skills/` for that agent.

## Mapping

| Claude Science profile | Claude Code file | model |
|---|---|---|
| RIGOR_REVIEWER | `.claude/agents/rigor-reviewer.md` | opus |
| SCRIBE | `.claude/agents/scribe.md` | sonnet |
| AUDITOR | `.claude/agents/auditor.md` | opus |
| SC_ANALYST | `.claude/agents/sc-analyst.md` | sonnet |
| ML_ANALYST | `.claude/agents/ml-analyst.md` | sonnet |
| RUNNER_ENGINEER | `.claude/agents/runner-engineer.md` | sonnet |
| CLUSTER_OPS | `.claude/agents/cluster-ops.md` | sonnet |
| LIT_REVIEWER | `.claude/agents/lit-reviewer.md` | sonnet |
| MANUSCRIPT_WRITER | `.claude/agents/manuscript-writer.md` | sonnet |
| FIGURE_EDITOR | `.claude/agents/figure-editor.md` | sonnet |
| EXTERNAL_REFEREE | `.claude/agents/external-referee.md` | opus |
| PRESENTER | `.claude/agents/presenter.md` | sonnet |

Models: `opus` for the three roles that return verdicts (rigor review and preflight GO/NO-GO, the reproducibility audit, referee reports); `sonnet` for the rest. Change the `model:` line if the available model names differ in your account.

## What changed in the port
- The scientific and rigor instructions are kept as written. Mechanics that only existed in Claude Science were replaced: the artifact store and `host.*` lineage became files in the repo plus `git log`; `ask_user` approval cards became "stop and return the question to the main session"; delegation between specialists became "name the next agent in your report"; the memory database became `CLAUDE.md`, `.claude/rules/` and `docs/claude/notes/`.
- Claude Code subagents cannot ask the user questions or start other subagents, so `cluster-ops` returns a submission proposal and only submits when the main session quotes the user's approval.
- Every agent carries the same "Operating in Claude Code" footer (repo records, citation of commit and note ids, no push, report format, blinding rule).
- `cluster-autoscout`, which `cluster-ops` used to choose a partition, is not ported; the agent now does read-only `sinfo`/`squeue` scans itself and keeps the storage and walltime rules inline.

## Profiles not ported

| Profile | Reason |
|---|---|
| OPERON | Claude Science's general-purpose coordinator (it carried the whole skill catalog); the Claude Code main session plays this role. |
| BOOKMARKER | Claude Science chat-bookmark helper; no equivalent need (system prompt is empty in the export). |
| ONBOARDING | Claude Science first-run onboarding; tied to its UI (system prompt is empty in the export). |
| REVIEWER | Claude Science background reviewer of conversation output (system prompt is empty in the export); replaced here by `rigor-reviewer` and `auditor`. |
| WORKSPACE_ADMIN | Administers the Claude Science daemon, skills and profiles (`science-daemon-ops`, `customize`, `skill-creator`); nothing to administer in Claude Code. |
| NGS_ENGINEER, SEQ_EVOLUTION, STAT_GENETICIST, VARIANT_INTERPRETER | Genomics profiles (variant calling, sequence evolution, GWAS, variant interpretation) not used by this project. |

Source of the prompts: the exported `agent_profiles.json` (21 profiles). Profiles whose export has an empty `systemPrompt` are listed above for that reason.
