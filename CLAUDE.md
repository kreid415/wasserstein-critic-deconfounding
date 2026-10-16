# CLAUDE.md: wasserstein-critic-deconfounding

Loaded at the start of every Claude Code session. The always-loaded rules in `.claude/rules/` add to it.
Built on 2026-10-16 from the Claude Science project (memory, notebook, ledger, profiles, skills, saved files).

## What this project is

- Question: do Wasserstein critics (adversaries that estimate an earth-mover distance between batches) remove
  single-cell batch effects better than discriminator adversaries (classifiers), at matched batch removal, on the
  scIB (single-cell integration benchmark) datasets with an scVI (single-cell variational inference) backbone?
- Status: revision of a submitted paper (journal manuscript BIOINF-2026-1434; preprint bioRxiv
  10.1101/2025.11.12.688061). The revision experiments are pre-registered: `docs/PREREG.md`, frozen by git tag
  `prereg-tier12-v3`, with the analysis fixed in `docs/ANALYSIS_SPEC.md` and `scripts/analysis_x1.py`.
- Main experiment X1: 8 datasets x 2 decoders x 5 seeds over the adversary arms. X13: standard-method baselines.
  A1/A2/A3: pilots that fix penalty strength, adversary input and standardisation before X1. R1-R4: the
  pre-registered decision rules that turn pilot results into X1 settings.

## Start every session here

1. `git fetch --all --tags`, then `git log --oneline -15 origin/final-review-fixes`: other sessions may have pushed.
2. Read `docs/claude/STATUS.md` (resume point and ordered next steps), then the newest page in `notebook/`.
   Where they differ, the notebook and git log are newer.
3. Read the active rules in `CONSTRAINTS.md` that bear on the task (ledger SI-01..SI-92; user's standing rules with
   their sources). Never contradict an active rule; when the user states a new rule, add it there (helper:
   `.claude/skills/standing-instructions/kernel.py`, `si_add`).

## Hard rules (break none of these)

1. **Blinding (SI-78, SI-79).** Never open, print, parse, summarise or copy any X1 or X13 score file, score row,
   latent, `failures.csv`, `attempts/*.jsonl` or status record, and never write X1 or X13 values into any file.
   A value shown to an AI agent must be disclosed in PREREG; an analysis script that uses a value counts as having
   examined it. Allowed: hashing files, and count-only tallies where a ledger rule permits them. `.claude/settings.json`
   denies reads of the blinded folders, but only for built-in file tools and simple `cat`/`head`-style commands, not
   `grep -r` or Python `open()`: the guard is partial, the rule is not. Never work around it. If unsure, do not read; ask.
2. **The tag freezes the code.** Do not move or delete tags `prereg-tier12-v1/-v2/-v3`. Any change under `src/`,
   `scripts/` or the frozen docs after the tag needs the user's OK and a PREREG section 9 amendment or notebook
   CORRECTION; launcher files under `experiments/` are not frozen.
3. **Cluster jobs need the user's explicit go** (show command, resources, walltime, outputs, job count first).
   Cancel only job ids you submitted. Read `.claude/rules/compute.md` first.
4. **Lab notebook is append-only and hash-chained** (`notebook/`; one page per day, New York time). Add entries with
   `.claude/skills/lab-notebook/kernel.py` (`nb_entry`, then `nb_index`; `nb_check('.')` must report ok). Fix
   mistakes with a CORRECTION entry, never by editing. Every experiment, result, decision and user rule gets one.
5. **Public repo.** No tokens, keys, cookies, journal correspondence or personal details in any commit. Git identity: `kreid415`,
   `kreid415@users.noreply.github.com`. Never `git add -A` or `git add .` (scratch files exist); add paths by name.
6. **Decisions.** Ask only about choices that change what is run, measured or analysed (SI-92); decide wording,
   documentation and tooling yourself and record them. One question per message, recommended option first
   (`.claude/rules/working-style.md`).
7. **Statistics.** The dataset is the independent unit. Do not report the pseudo-replicated p = 0.004, a bootstrap p
   at n = 6, or a Nadeau-Bengio correction across datasets (`docs/claude/notes/statistics-and-claims.md`).

## Branches and tags

| ref | role |
|---|---|
| `claude-code-migration` | this bundle (CLAUDE.md, `.claude/`, `docs/claude/`, `envs/`, `archive/`) on top of `final-review-fixes` @ 1ad44cb |
| `final-review-fixes` | live research branch (fixes, PREREG v3, analysis code, notebook) |
| `paper` | manuscript draft `paper/paper.tex`, `paper/refs.bib` (not merged into the research branch) |
| `main` | stale (2026-10-13, 5cb75c0); do not build on it until the user decides to merge |
| tags `prereg-tier12-v1/-v2/-v3` | pre-registration freezes; v3 (10a1ca5) is current, v2 = c88cce3 |

Notebook and ledger appends happen on one branch only, `final-review-fixes`. The Claude Science sessions have
finished and pushed (last commits 857234a, notebook entry NB-20261016-34, and 1ad44cb, the Sinkhorn-redo preflight GO);
this branch sits on 1ad44cb. First action of the first Claude Code session: `git checkout final-review-fixes && git pull
&& git merge origin/claude-code-migration`, push, and work on `final-review-fixes` (or short branches merged back into it).
This bundle adds new files and changes only the README download block, so the merge is clean.

## Repository map

| path | contents |
|---|---|
| `src/wcd_vae/wcd/` | model code: backbones (`backbones.py`, `scvi_backbone.py`), critic (`critic.py`), discriminator losses, MMD and Sinkhorn alignment (`alignment.py`), `training.py`, `primitives.py` |
| `scripts/run_stage.py` | stage runner for pilots and X1 (provenance checks, refusals, resume); usage at file top |
| `scripts/fit_paper_config.py`, `scripts/score_scib_native.py` | one fit; scIB scoring (kBET needs R) |
| `scripts/prereg_rules.py`, `freeze_a1_grid.py`, `freeze_matched_lambda.py`, `decide_a2_a3.py` | decision rules R1-R4 |
| `scripts/analysis_x1.py` | frozen X1 analysis (P1-P4); tests in `tests/analysis` |
| `scripts/check_tests.py` | fewest-tests check: every changed line run by a test (`--base REF --src scripts`) |
| `docs/PREREG.md`, `docs/ANALYSIS_SPEC.md`, `docs/PAPER_PLAN.md`, `docs/EQUIVALENT_COMMITS.json` | plan, frozen spec, paper plan, commits counted as result-identical |
| `docs/prereg/*.json` | rule decision records (R1, R4-a1, R2/R3) |
| `docs/review_final/`, `experiments/<check>/README.md` | review reports; one folder per check or run, each with its re-run commands |
| `manifests/` | fit manifests (one row per fit) |
| `cluster/` (`jhpce/`, `envspec/`) | JHPCE (Johns Hopkins cluster) job scripts, exact environment specs |
| `notebook/`, `CONSTRAINTS.md` | lab notebook (INDEX.md lists entries and open issues); standing-rule ledger |
| `legacy/`, `revision/`, `notebooks/`, `manuscript_tm/` | retired code (do not run), older revision material, figure notebook, old manuscript template |

## Environments and tests

- Cloud session: the SessionStart hook in `.claude/settings.json` runs `.claude/cloud-setup.sh` (CPU env `wcd-cpu`,
  about 90 s on first start). See `docs/claude/CLOUD_SETUP.md` for network domains and what the VM cannot run.
- Thread guards for every run: `KMP_AFFINITY=disabled OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=1 MKL_THREADING_LAYER=SEQUENTIAL`,
  `PYTHONPATH=src`.
- CPU tiers (measured 2026-10-16 with `wcd-cpu`): `python -m pytest -q tests/prereg tests/stage tests/analysis`
  (283 passed, 2 skipped); `python -m pytest -q tests/scvi tests/jhpce tests/test_legacy_retired.py tests/test_gate_task_split.py`
  (168 passed, 16 skipped).
- Workstation-only tiers: `tests/wcd_vae` (env `wcd-gpu`), `tests/scoring` and `tests/x13` (data, R/kBET, env `wcd-kbet`).
  Specs: `envs/README.md` (canonical exact specs are `cluster/envspec/`).
- Before calling code done: `python scripts/check_tests.py --base <commit before your change> --src scripts` exits 0.

## Machines and data

| machine | use |
|---|---|
| cloud VM (Claude Code app) | code, docs, CPU tests, rule scripts, git. Assume no GPU, no data, no JHPCE until checked |
| user's workstation (RTX 3080, 10 GB) | GPU fits, scoring, all data under `/home/kendall/experiment_data/wasserstein-critic-deconfounding` (`$D`); run Claude Code locally or via Remote Control |
| JHPCE (SLURM; L40S GPUs) | X1 cluster part; data on purged scratch, harvest and mirror to `$D` |

Data products, sizes and how each is regenerated: `docs/claude/DATA_LOCATIONS.md`. Long jobs run outside the session
(`sbatch`, or `nohup`/`tmux` with a log and done-marker); a command inside a session dies with it.

## Knowledge carried over from Claude Science

| where | what |
|---|---|
| `docs/claude/STATUS.md` | resume point, next steps, pending decisions, in-flight work at hand-over |
| `docs/claude/notes/` | 166 project memory rows condensed by topic, each bullet with its source id (`README.md` indexes them) |
| `.claude/rules/` | user's working rules (answers, decisions, integrity, code changes, git, compute) |
| `.claude/agents/` | 12 specialist subagents (rigor-reviewer, external-referee and auditor on opus; others sonnet); map in `README.md` |
| `.claude/skills/` | 20 of the user's skills, rewritten for Claude Code; `docs/claude/SKILLS.md` lists all skills and sources |
| `archive/claude-science/` | 429 of 522 saved Claude Science files (reviews, handoffs, tables, figures): historical snapshots, never authoritative (current versions are the repo files); `MANIFEST.csv` lists all 522 and why 93 were left out |
| `HANDOFF.md` (root) | stale (2026-08-08); superseded by `docs/claude/STATUS.md` |

Before a long session ends, update `docs/claude/STATUS.md`, commit, push, and give the user a starter message for the
next session.
