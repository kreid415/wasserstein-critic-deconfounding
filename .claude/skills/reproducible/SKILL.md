---
name: reproducible
description: Make a project's results auditable and verifiable by a human scientist, and keep the repository in a clean git-ready layout (data-gather, src, experiments, results, paper). Use when the user wants to audit, verify or fact-check results; trace a reported number to the file and script that produced it; produce a provenance record or verification bundle; prepare work for a reviewer; or organize a project for reproducibility. Triggers - "audit my results", "where did this number come from", "can a reviewer check this", "trace every figure to its source".
---
# Reproducible and auditable results

> **Helper loading (Claude Code).** `kernel.py` is plain Python (standard library only). Load it with
> `p = ".claude/skills/reproducible/kernel.py"; exec(compile(open(p).read(), p, "exec"), globals())`.

A result a human scientist cannot **trace**, **re-run**, **inspect** or **challenge** is asserted, not reproduced. In Claude Code the provenance sources are the git history, the committed scripts, the pinned environment files and the result files themselves. Never hand-type a reported number, seed or commit hash: read it back from the file or from `git`.

## The five capabilities

| Capability | Deliverable | The reviewer's question |
|---|---|---|
| **Trace** | `audit/traceability.csv` | "This figure says r=0.83. Which file and script produced it?" |
| **Re-run** | `audit/verify.py` | "If I run it again from the committed inputs, do I get the same numbers?" |
| **Inspect** | `audit/provenance/*.md` | "Which inputs, environment and seed went in?" |
| **Challenge** | `audit/DECISIONS.md` | "Why this threshold? Why were these samples dropped? Why this method?" |
| **Conclude** | `audit/FINDINGS.md` | "Did it hold up? What broke, what is unverifiable?" |

`audit/AUDIT.md` is the front door: coverage line, verdict, and links in reading order. **An audit must conclude.** `FINDINGS.md` lists every discrepancy (prose says 0.83, file says 0.81), untraceable claim, unpinned stochastic result and failed `verify.py` check, each with a severity. **Never resolve a discrepancy by editing one side to match the other**: report both values and hand the decision back to the scientist.

## Helpers (`kernel.py`)
- `provenance_record(path, script=None, root=".")` -> dict: sha256 of the result file, last git commit touching it, whether the working tree is dirty, the producing script (its commit and seed lines), and the environment files found (`conda-lock.yml`, `environment.y*ml`, `envs/*.yml`, `pyproject.toml`).
- `render_provenance_md(records)` -> markdown for one record or a list.
- `scan_seeds(code)` -> RNG-seed lines in source (a stochastic script with none is a red flag).
- `write_traceability(rows, path)` -> claim/value/artifact_path/source/verify CSV.
- `write_manifest(root, path)` -> `MANIFEST.sha256` (`sha256  relpath`, sorted), verifiable with `sha256sum -c`.

## Audit workflow
1. **Inventory claims.** Read the write-up under audit (manuscript, report, notebook entry) and list every load-bearing claim: each statistic, figure headline and table. The total is the **denominator**; report traced / re-run / unverifiable as a fraction in `AUDIT.md`. Tracing 3 of 10 claims in silence is cherry-picking.
2. **Trace.** For each claim find the file that carries the value (CSV/JSON under `results/` or `experiments/*/`) and the script that wrote it (`git log -- <file>`, the experiment README, the notebook entry). Write rows with `write_traceability`. The `value` column is read from the file, not from prose; disagreement is a finding.
3. **Inspect.** `render_provenance_md([provenance_record(f, script=s) for f, s in pairs])`. For external inputs record source URL or accession and a sha256. Two rules: **no figure without its source data** (every committed figure ships the CSV of the numbers behind it; extract them from plotting code into `results/tables/` and trace them like any claim); **capture the environment that actually ran** (`conda list --explicit` or the lockfile hash, not only a spec) and state where bit-exact reproduction is not guaranteed (GPU/CUDA kernels are often nondeterministic even with seeds pinned, so a seeded ML result may re-derive only to a tolerance).
4. **Re-run harness.** `audit/verify.py` regenerates the load-bearing numbers from committed inputs and asserts they match committed results within a stated tolerance, one `--check <name>` per traceable claim and `--check all`. Say why each tolerance is what it is (float noise vs genuinely stochastic step). Run it yourself before handing it over; a harness that does not pass is worse than none. Use the project's own library code path, not a re-implementation.
5. **Decisions.** `audit/DECISIONS.md`: every threshold (and nearby values if checked), every exclusion with the rule that excluded it, every method choice over an obvious alternative, every compromise forced by the data. State the rationale, not just the choice.
6. **Robustness on headline claims** (a check, not a re-derivation): a negative control (label shuffle), a second seed, a threshold sweep (about 20% either side of hand-chosen cutoffs). Cite the evidence in `FINDINGS.md`.
7. **Findings.** `audit/FINDINGS.md`, severity-tagged: **BLOCKER** (a reported number has no traceable source, or `verify.py` disagrees beyond tolerance); **CONCERN** (stochastic result with no pinned seed; conclusion flips under a second seed or threshold nudge; figure without source data); **NOTE** (cosmetic: rounding, missing unit). For each: expected, found (both values read from sources), where. Open with a coverage line ("14 claims; 13 traced, 12 re-run green, 1 untraceable") and a one-line verdict. A clean audit is a result; say so and let the coverage back it.
8. **AUDIT.md**, then **`write_manifest("audit")`** last so the checksums cover the final files; commit.

## Repo scaffold (project start)
```
<repo>/
  README.md  EXPERIMENTS.md  environment.yml  .gitignore
  data/gather/        one script per data source (fetch/build)
  src/                installable library imported by experiments (pip install -e .)
  experiments/NN_name/  source + run.sh + README.md (inputs, command, key numbers, commit)
  scripts/            gather_data.sh, submit_cluster.sh
  results/figures/  results/tables/   small regenerated outputs, committed
  audit/              AUDIT.md FINDINGS.md traceability.csv provenance/ DECISIONS.md verify.py MANIFEST.sha256
  paper/              main.tex, sections/, refs.bib (\graphicspath{{../results/figures/}})
```
`.gitignore` ignores bulk data and weights but KEEPS `results/` and `audit/`. Pin `environment.yml` to versions actually used. Before creating a GitHub remote or pushing, confirm repo name and visibility with the user. This repository already has its own layout (see CLAUDE.md); apply the audit bundle inside it rather than restructuring.

## Library, tests, lint
Shared logic lives in `src/` as pure, typed, documented functions (docstring with parameters, returns and an executed example); experiment scripts are thin wrappers that open with a numbered module docstring mapping the code flow and mark blocks `# === n. STAGE ===` with `# WHY:` / `# HOW:` comments. Standard libraries (numpy, pandas, scikit-learn, stdlib) before hand-rolled code. Tests: per-module unit tests on synthetic data with known structure, plus an end-to-end regression test reproducing a committed result through the library. Lint: `ruff check .` and `pytest` green before committing. Follow the project's FEWEST-TESTS rule when adding tests.
