# Condensed project memory (docs/claude/notes)

Source: the Claude Science project memory dump of 2026-10-16 (166 project rows) condensed by topic for use in Claude Code. Every bullet ends with the memory row id(s) it came from and the evidence tag (observed, stated, inferred), written as `[mem_... · evidence]`. Validity tags on bullets: **[C]** still true at commit 10a1ca5 (`final-review-fixes`), **[A]** audited at an older commit (re-check), **[H]** historical, **[S]** superseded.

**Read this first.** The rows were written July to October 2026 and mostly describe the first revision waves (scCRAFT-derived backbone, then the "native backbone" rebuild, on the Rockfish and DSAI clusters). The project has since moved to a pre-registered Tier 1+2 rerun on an scVI backbone on JHPCE. For the present state read `CONSTRAINTS.md` (standing rules SI-01 to SI-92), `docs/PREREG.md`, `docs/PAPER_PLAN.md`, `notebook/` and `docs/claude/STATUS.md`. Blinding: no X1 or X13 score value is in these notes or may be added (SI-78, SI-79).

| File | Contents |
|---|---|
| `verified-code-facts.md` | Audited sign and wiring facts, metric conventions, primitives, harness flags, hyperparameter-selection corrections; things not to re-litigate |
| `pitfalls.md` | Footguns, dead ends, counting and cost-estimate mistakes, environment and tool gotchas |
| `statistics-and-claims.md` | Pseudo-replication, which p-values may and may not be reported, Nadeau-Bengio scope, n = 6 limits, narrowed or reversed claims |
| `experiment-history.md` | E0 to E9, the scCRAFT pivot, structure-collapse blocker, unconditioned re-runs, audits, pointer to the current Tier 1+2 programme (no X1 or X13 values) |
| `data-and-storage.md` | The 8 datasets, registry, figshare ids, where results and embeddings went, mirror rule |
| `compute-and-cluster.md` | Current JHPCE and local GPU setup, historical Rockfish and DSAI facts, environments, job handling |
| `repo-conventions.md` | Branches and tags, identity strings, tests, what `.gitignore` hides, resume keys |
| `manuscript.md` | Paper identity and source, title and venue history, writing conventions, Figure 1 design, reviewer items |

## Notes on the dump itself

- 17 rows are cut off with an ellipsis in the dump (mem_8c17cfd3aa4a, mem_2de87e3f5a01, mem_b4e19cd53bb7, mem_155c6c9ebae5, mem_d6f2179c3d8b, mem_30c3f99deb13, mem_f9897d84fc0d, mem_e93fef166370, mem_a1be44213db6, mem_1fd49efc28fb, mem_2bb545a9b0dc, mem_13ef95426e5b, mem_02f1fd226dd7, mem_6eb5dcdf165f, mem_c82657963b9d, mem_ca02d2e477c3, mem_d56e2434bcb7); only the visible text was condensed. Ask the previous session's memory store for the full text if a detail is needed.
- The dump's `artifact` list was empty. The memory store reported 822 rows on record for this project; the 166 rows here are the live set given to this track. Rows mem_92903abb214a (local mirroring rule) was added from a session recall block, not from the dump.
- Duplicates merged: mem_972f9518dee4 with mem_70a50730ade2; mem_1fd49efc28fb with mem_e93fef166370; mem_8f31ca8a260e with mem_9d9ee53fb67f; mem_b922a9c8ca6b with mem_6eb5dcdf165f; mem_f2d88d64b3fa, mem_aff813ce96b8 and mem_05aecef5b6fb are complementary, not duplicates.
- Corrections applied (newer row wins): mem_9d9ee53fb67f and mem_8f31ca8a260e correct mem_76f206699169 (clisi_weight 10, not 1.0); mem_a1be44213db6 corrects mem_81c35ab09d01 (lung E4 was complete; the earlier row is not in the dump); mem_30c3f99deb13 corrects an earlier "heavy tasks timed out" claim; mem_53e6fe2660e7 corrects an earlier idle-node claim; mem_fea42a1aa312 corrects a flat E4 count; mem_e6f94fa5423c supersedes mem_dc4da8fdab82, mem_2524c83cf5f8, mem_67d467a14bc1 and mem_d6f2179c3d8b; mem_76af659f9a75 is itself superseded by mem_76a6f44a21fc (not in the dump); mem_1f205c234555 supersedes the E2 n = 24 framing in mem_6eb5dcdf165f; mem_aff813ce96b8 and mem_9b3275546b2f correct the p = 0.004 and bootstrap p in mem_6eb5dcdf165f and mem_05aecef5b6fb.

## Rows dropped or reduced (Claude Science mechanics only)

| Row | Reason |
|---|---|
| mem_accc2e2b8914 | Only says raw `sbatch` jobs are invisible to the Claude Science job tracker; the plain fact (poll `sacct`/`squeue`) is kept in compute-and-cluster.md |
| mem_fc7227a2f697 | Claude Science upload tool path restriction (scratch root only); no plain statement left |
| mem_88c3a08d9c78 | 60-second command limit of the Claude Science cluster tool; not a cluster property |
| mem_d1eef306bc8a | Argument-length limit of the Claude Science cluster tool when pushing a manifest |
| mem_e93fef166370 | Claude Science download API return shape; only the "do not send files as base64 through stdout" lesson is kept (pitfalls.md) |
| mem_57167274b2e6 | Artifact pipeline rewrites image filenames in saved `.tex`; reduced to "the git repo is the compile source" |
| mem_8fe699da4716 | Artifact ids of the uploaded manuscript zip; reduced to the file locations in the repo |
| mem_7b20526ea60e | Artifact version id of a wrong `paper.tex`; reduced to one sentence ("different project, not this manuscript") |
