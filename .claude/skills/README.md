# Project skills (ported from Claude Science)

Each folder is a Claude Code skill: `SKILL.md` (frontmatter `name` and `description`) plus optional `references/`, `scripts/` and `kernel.py`. Claude Code loads a skill when its description matches the task, or when an agent lists it under `skills:`.

## Helper loading
`kernel.py` files are plain Python helpers (prefixes such as `cg_`, `fl_`, `pf_`, `scpf_`, `rs_`, `nb_`, `si_`, `dg_`). In Claude Science they auto-loaded into the analysis kernel. In Claude Code, load one from Python or a script:

```python
p = ".claude/skills/claim-gate/kernel.py"
exec(compile(open(p).read(), p, "exec"), globals())
```

Load `experiment-preflight/kernel.py` before `experiment-preflight-singlecell/kernel.py`. `bio-diagram` also works as `sys.path.insert(0, ".claude/skills/bio-diagram/scripts"); import biodiagram`. `job-watch/scripts/jw.py` and `ponytail/scripts/check_tests.py` are command-line scripts.

## Rewrites made in the port
- Every `host.*` call, artifact-store step (`save_artifacts`, `version_of`, artifact ids), memory-tool step and Claude Science compute step was rewritten to a plain file, git or shell equivalent, or removed.
- `session-handoff` was rewritten as a file-based handoff to `docs/claude/STATUS.md`; the chat-size thresholds and the database queries were dropped.
- `reproducible`: provenance now comes from git and the files (`provenance_record(path, script)`), and the traceability column `artifact_version_id` became `artifact_path`.
- `rigor-review`: the ledger lives at `docs/rigor/rigor_ledger.json (created on the first rigor_save)`; the abstract triage that used the platform LLM became `overlap_prompts` / `overlap_apply` (the agent or a subagent rates each abstract).
- `doc-build-gate`: `dg_verify_artifact_refs` became `dg_verify_links` (relative links resolve to files). The check for leaked chat markers in build sources is kept (`DG_MARKER_RE`).
- `section-enrichment`: artifact-marker mapping removed.
- Tests that lived as Claude Science artifacts (`claim-gate_tests.py`, `tests/job-watch_tests.py`) were not part of the skill folders and are not included.

## Not ported
- `codex-offload`: offloads work to Codex through a Claude Science SSH compute target.
- `cluster-autoscout`: scans clusters through Claude Science compute providers (its storage and walltime rules were folded into `cluster-rules` and the `cluster-ops` agent).

Provenance: copied from the Claude Science skill registry (origin "personal"). Most also exist in the public repo kreid415/claude-science-skills (HEAD 50cfddf when ported); `ponytail`, `ponytail-review`, `ponytail-audit` and `section-enrichment` were only in the registry. `ponytail*` are adapted from github.com/DietrichGebert/ponytail (MIT, `ponytail/LICENSE.upstream`).
