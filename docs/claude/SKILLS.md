# Skills used by the ported agents

Covers every skill named in the `skillNames` of the 12 ported profiles (57 distinct skills). Agents preload only the ported ones (`skills:` in `.claude/agents/*.md`); the others are listed so you can install them if you want them.

Status key: **ported** = copied and rewritten into this repo; **imported-public** = third-party skill with a public repository, not copied; **Anthropic built-in** = not copied, Claude Code has its own equivalent; **dropped** = not usable in Claude Code, reason given.

| Skill | Used by agents | Status |
|---|---|---|
| `rigor-review` | rigor-reviewer | ported: `.claude/skills/rigor-review/` |
| `paper-lookup` | rigor-reviewer, scribe, runner-engineer, lit-reviewer, manuscript-writer, external-referee | imported-public: `K-Dense-AI/claude-scientific-skills` (folder `skills/paper-lookup`, checked at commit a1043ef). Install: copy that folder to `.claude/skills/paper-lookup/`, or add the repository as a plugin marketplace if you want the whole set |
| `literature-review` | rigor-reviewer, lit-reviewer | dropped: Claude Science-bundled (origin `anthropic`), no public source found; the `lit-reviewer` agent prompt carries the protocol. |
| `scientific-critical-thinking` | rigor-reviewer, lit-reviewer, external-referee | imported-public: `K-Dense-AI/claude-scientific-skills` (folder `skills/scientific-critical-thinking`, checked at commit a1043ef). Install: copy that folder to `.claude/skills/scientific-critical-thinking/`, or add the repository as a plugin marketplace if you want the whole set |
| `peer-review` | rigor-reviewer, external-referee | imported-public: `K-Dense-AI/claude-scientific-skills` (folder `skills/peer-review`, checked at commit a1043ef). Install: copy that folder to `.claude/skills/peer-review/`, or add the repository as a plugin marketplace if you want the whole set |
| `pdf-explore` | rigor-reviewer, scribe, auditor, runner-engineer, external-referee | dropped: Parses PDFs in the Claude Science kernel (`pdf_pages`, `pdf_scan`); in Claude Code read PDFs as text with `pdftotext` or the Read tool with page ranges. Anthropic-bundled, no public source found. |
| `reproducible` | rigor-reviewer, scribe, auditor, sc-analyst, ml-analyst, runner-engineer, figure-editor | ported: `.claude/skills/reproducible/` |
| `experiment-preflight` | rigor-reviewer, sc-analyst, ml-analyst, runner-engineer | ported: `.claude/skills/experiment-preflight/` |
| `experiment-preflight-singlecell` | rigor-reviewer, sc-analyst, runner-engineer | ported: `.claude/skills/experiment-preflight-singlecell/` |
| `claim-gate` | rigor-reviewer, sc-analyst, ml-analyst, runner-engineer, manuscript-writer | ported: `.claude/skills/claim-gate/` |
| `lab-notebook` | scribe, ml-analyst, runner-engineer | ported: `.claude/skills/lab-notebook/` |
| `paper-outline` | scribe | ported: `.claude/skills/paper-outline/` |
| `citation-management` | scribe, lit-reviewer, manuscript-writer | imported-public: `K-Dense-AI/claude-scientific-skills` (folder `skills/citation-management`, checked at commit a1043ef). Install: copy that folder to `.claude/skills/citation-management/`, or add the repository as a plugin marketplace if you want the whole set |
| `standing-instructions` | scribe, sc-analyst, ml-analyst, runner-engineer, manuscript-writer, figure-editor, presenter | ported: `.claude/skills/standing-instructions/` |
| `self-awareness` | auditor | dropped: Introspection of the Claude Science session database; no equivalent. Git history and the notebook are the record. |
| `pdf` | auditor | Anthropic built-in (not copied): public source `anthropics/skills` (folder `skills/pdf`, present in the clone checked 2026-10-16). Claude Code has its own document skills; install from that repository if the built-in is unavailable |
| `docx` | auditor | Anthropic built-in (not copied): public source `anthropics/skills` (folder `skills/docx`, present in the clone checked 2026-10-16). Claude Code has its own document skills; install from that repository if the built-in is unavailable |
| `ponytail-audit` | auditor | ported: `.claude/skills/ponytail-audit/` |
| `ponytail-review` | auditor | ported: `.claude/skills/ponytail-review/` |
| `scanpy` | sc-analyst | imported-public: `K-Dense-AI/claude-scientific-skills` (folder `skills/scanpy`, checked at commit a1043ef). Install: copy that folder to `.claude/skills/scanpy/`, or add the repository as a plugin marketplace if you want the whole set |
| `anndata` | sc-analyst | imported-public: `K-Dense-AI/claude-scientific-skills` (folder `skills/anndata`, checked at commit a1043ef). Install: copy that folder to `.claude/skills/anndata/`, or add the repository as a plugin marketplace if you want the whole set |
| `pathway-enrichment` | sc-analyst | imported-public: `K-Dense-AI/claude-scientific-skills` (folder `skills/pathway-enrichment`, checked at commit a1043ef). Install: copy that folder to `.claude/skills/pathway-enrichment/`, or add the repository as a plugin marketplace if you want the whole set |
| `scikit-learn` | sc-analyst, ml-analyst | imported-public: `K-Dense-AI/claude-scientific-skills` (folder `skills/scikit-learn`, checked at commit a1043ef). Install: copy that folder to `.claude/skills/scikit-learn/`, or add the repository as a plugin marketplace if you want the whole set |
| `statistical-analysis` | sc-analyst, ml-analyst | imported-public: `K-Dense-AI/claude-scientific-skills` (folder `skills/statistical-analysis`, checked at commit a1043ef). Install: copy that folder to `.claude/skills/statistical-analysis/`, or add the repository as a plugin marketplace if you want the whole set |
| `statsmodels` | sc-analyst | imported-public: `K-Dense-AI/claude-scientific-skills` (folder `skills/statsmodels`, checked at commit a1043ef). Install: copy that folder to `.claude/skills/statsmodels/`, or add the repository as a plugin marketplace if you want the whole set |
| `exploratory-data-analysis` | sc-analyst, ml-analyst | imported-public: `K-Dense-AI/claude-scientific-skills` (folder `skills/exploratory-data-analysis`, checked at commit a1043ef). Install: copy that folder to `.claude/skills/exploratory-data-analysis/`, or add the repository as a plugin marketplace if you want the whole set |
| `optimize-for-gpu` | sc-analyst, ml-analyst | imported-public: `K-Dense-AI/claude-scientific-skills` (folder `skills/optimize-for-gpu`, checked at commit a1043ef). Install: copy that folder to `.claude/skills/optimize-for-gpu/`, or add the repository as a plugin marketplace if you want the whole set |
| `fail-loud-pipelines` | sc-analyst, ml-analyst, runner-engineer, cluster-ops | ported: `.claude/skills/fail-loud-pipelines/` |
| `run-status-board` | ml-analyst, cluster-ops | ported: `.claude/skills/run-status-board/` |
| `pytorch-lightning` | ml-analyst | imported-public: `K-Dense-AI/claude-scientific-skills` (folder `skills/pytorch-lightning`, checked at commit a1043ef). Install: copy that folder to `.claude/skills/pytorch-lightning/`, or add the repository as a plugin marketplace if you want the whole set |
| `statistical-power` | ml-analyst | imported-public: `K-Dense-AI/claude-scientific-skills` (folder `skills/statistical-power`, checked at commit a1043ef). Install: copy that folder to `.claude/skills/statistical-power/`, or add the repository as a plugin marketplace if you want the whole set |
| `experimental-design` | ml-analyst | imported-public: `K-Dense-AI/claude-scientific-skills` (folder `skills/experimental-design`, checked at commit a1043ef). Install: copy that folder to `.claude/skills/experimental-design/`, or add the repository as a plugin marketplace if you want the whole set |
| `remote-compute-ssh` | runner-engineer, cluster-ops | dropped: Claude Science submit/harvest flow for its SSH targets (`submit_job`, `compute_done`); in Claude Code use `ssh`, `sbatch`, `sacct` directly (see `cluster-rules`, `job-watch`). |
| `cluster-autoscout` | cluster-ops | dropped: Claude Science compute-provider scan; not portable (rules folded into `cluster-rules` and the `cluster-ops` agent). |
| `compute-env-setup` | cluster-ops | dropped: Claude Science provider/image setup; replaced by `envs/*.yml` and `.claude/cloud-setup.sh` (see `docs/claude/CLOUD_SETUP.md`). |
| `job-watch` | cluster-ops | ported: `.claude/skills/job-watch/` |
| `cluster-rules` | cluster-ops | ported: `.claude/skills/cluster-rules/` |
| `scientific-writing` | manuscript-writer | imported-public: `K-Dense-AI/claude-scientific-skills` (folder `skills/scientific-writing`, checked at commit a1043ef). Install: copy that folder to `.claude/skills/scientific-writing/`, or add the repository as a plugin marketplace if you want the whole set |
| `deslop` | manuscript-writer | ported: `.claude/skills/deslop/` |
| `section-enrichment` | manuscript-writer | ported: `.claude/skills/section-enrichment/` |
| `academic-paper` | manuscript-writer | imported (origin `imported` in Claude Science), source unknown: not in `K-Dense-AI/claude-scientific-skills` or `anthropics/skills` (both cloned and searched); not copied |
| `venue-templates` | manuscript-writer | imported-public: `K-Dense-AI/claude-scientific-skills` (folder `skills/venue-templates`, checked at commit a1043ef). Install: copy that folder to `.claude/skills/venue-templates/`, or add the repository as a plugin marketplace if you want the whole set |
| `paper-narrative` | manuscript-writer, figure-editor | dropped: Claude Science-bundled (origin `anthropic`), no public source found; figure-set judgement is described in the `figure-editor` agent prompt. |
| `doc-build-gate` | manuscript-writer, figure-editor, presenter | ported: `.claude/skills/doc-build-gate/` |
| `figure-style` | figure-editor, presenter | dropped: Claude Science-bundled (origin `anthropic`), no public source found; its rules are partly covered by `doc-build-gate` figure QA and the `figure-editor` agent prompt. |
| `figure-composer` | figure-editor | dropped: Claude Science-bundled (origin `anthropic`), no public source found; see `figure-editor`. |
| `scientific-visualization` | figure-editor | imported-public: `K-Dense-AI/claude-scientific-skills` (folder `skills/scientific-visualization`, checked at commit a1043ef). Install: copy that folder to `.claude/skills/scientific-visualization/`, or add the repository as a plugin marketplace if you want the whole set |
| `matplotlib` | figure-editor | imported-public: `K-Dense-AI/claude-scientific-skills` (folder `skills/matplotlib`, checked at commit a1043ef). Install: copy that folder to `.claude/skills/matplotlib/`, or add the repository as a plugin marketplace if you want the whole set |
| `seaborn` | figure-editor | imported-public: `K-Dense-AI/claude-scientific-skills` (folder `skills/seaborn`, checked at commit a1043ef). Install: copy that folder to `.claude/skills/seaborn/`, or add the repository as a plugin marketplace if you want the whole set |
| `bio-diagram` | figure-editor | ported: `.claude/skills/bio-diagram/` |
| `academic-paper-reviewer` | external-referee | imported (origin `imported` in Claude Science), source unknown: not in `K-Dense-AI/claude-scientific-skills` or `anthropics/skills` (both cloned and searched); not copied |
| `scholar-evaluation` | external-referee | imported-public: `K-Dense-AI/claude-scientific-skills` (folder `skills/scholar-evaluation`, checked at commit a1043ef). Install: copy that folder to `.claude/skills/scholar-evaluation/`, or add the repository as a plugin marketplace if you want the whole set |
| `scientific-slides` | presenter | imported-public: `K-Dense-AI/claude-scientific-skills` (folder `skills/scientific-slides`, checked at commit a1043ef). Install: copy that folder to `.claude/skills/scientific-slides/`, or add the repository as a plugin marketplace if you want the whole set |
| `pptx` | presenter | Anthropic built-in (not copied): public source `anthropics/skills` (folder `skills/pptx`, present in the clone checked 2026-10-16). Claude Code has its own document skills; install from that repository if the built-in is unavailable |
| `pptx-posters` | presenter | imported-public: `K-Dense-AI/claude-scientific-skills` (folder `skills/pptx-posters`, checked at commit a1043ef). Install: copy that folder to `.claude/skills/pptx-posters/`, or add the repository as a plugin marketplace if you want the whole set |
| `latex-posters` | presenter | imported-public: `K-Dense-AI/claude-scientific-skills` (folder `skills/latex-posters`, checked at commit a1043ef). Install: copy that folder to `.claude/skills/latex-posters/`, or add the repository as a plugin marketplace if you want the whole set |
| `scientific-schematics` | presenter | imported-public: `K-Dense-AI/claude-scientific-skills` (folder `skills/scientific-schematics`, checked at commit a1043ef). Install: copy that folder to `.claude/skills/scientific-schematics/`, or add the repository as a plugin marketplace if you want the whole set |

## Ported but not named by any of the 12 profiles
`ponytail`, `session-handoff`. These are user-level skills that the project rules refer to (session handoff, code minimalism, standing rules). Skipped on purpose: `codex-offload` and `cluster-autoscout` (tied to Claude Science compute targets).

## Sources checked (2026-10-16)
- `kreid415/claude-science-skills` (public, HEAD 50cfddf): holds most of the personal skills.
- `K-Dense-AI/claude-scientific-skills` (HEAD a1043ef): `skills/<name>/SKILL.md`; the Claude Science copies carry `skill-author: K-Dense Inc.` metadata.
- `anthropics/skills`: has `skills/pdf`, `skills/docx`, `skills/pptx`.
- Skills with origin `anthropic` in Claude Science (`figure-style`, `figure-composer`, `paper-narrative`, `literature-review`, `pdf-explore`, `remote-compute-ssh`, `compute-env-setup`, `self-awareness`) were not found in either public repository.
