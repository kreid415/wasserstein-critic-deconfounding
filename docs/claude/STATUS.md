# STATUS: where to resume the research

Snapshot: branch `final-review-fixes` at 10a1ca5 (tag `prereg-tier12-v3`), 2026-10-16 about 17:55 New York time.
Written by the migration's Status track from the repo, the notebook and the Claude Science session records. Every fact
carries its source: commit, `file:line`, notebook entry (NB-yyyymmdd-nn, the dated lab-notebook entries in `notebook/`),
ledger rule (SI-nn, the numbered standing rules in `CONSTRAINTS.md`), or session id.
**Work may have landed after this snapshot. Before anything else: `git fetch`, `git log origin/final-review-fixes`, and read
the newest entries of `notebook/2026-10-16.md` (and any later page). Trust those over this file.**
Blinding (SI-78, SI-79): this file holds counts and status only. It contains no score from X1 (the main experiment,
8 datasets x 2 decoders x 5 seeds) or X13 (standard-method baselines). Do not open score files, latents, status records or
`attempts/*.jsonl` of those stages; an analysis script that reads a score counts as examining it.

## 1. State in one paragraph

The pre-registration is frozen again as version 3: tag `prereg-tier12-v3` (annotated, tag object ce056088ee) points to
10a1ca5 and is pushed (NB-20261016-34, commit 857234a). It freezes `docs/PREREG.md` v3 (section 9 amendments, SI-51 to SI-91),
`docs/ANALYSIS_SPEC.md`, `scripts/analysis_x1.py` with `tests/analysis`, `docs/EQUIVALENT_COMMITS.json`, and the code of every
stage. The user approved the final diff (NB-20261016-32); a display audit before the tag found no new display of a score
(NB-20261016-29, -30, -33); the tag was made before the Sinkhorn redo, as SI-87 requires. Done: the final-review fixes, the
X1 analysis code with its tests, three independent review rounds, the pilots A1 (penalty-strength pilot), the R1 extension,
A2 and A3 (adversary-input and standardisation pilots) for all arms except Sinkhorn, rules R1, R4-a1 (round 1 each). Not
done and blocking X1: the redo of the 234 Sinkhorn pilot fits (SI-51, SI-57) and the rules that depend on it. Running now:
a read-only pre-launch check of the Sinkhorn redo (section 5); no fit, scorer or cluster job is running. Paused since
2026-10-12 (NB-20261012-05): local X1 (19 of 1,125 rows scored) and filler scoring pass A (8 score files). Not started:
JHPCE X1 (cluster part, 1,800 fits), follow-ups X3/X6/X7/X8/X12/X15, the X1 analysis run, the results sections of the paper.

## 2. Next steps, in order

1. Orient (any machine, minutes). `git fetch --all --tags`; confirm `prereg-tier12-v3` -> 10a1ca5; read notebook pages from
   2026-10-16 onward; read `CONSTRAINTS.md` (active rules SI-01..SI-98). The notebook and ledger helpers are ported: `.claude/skills/lab-notebook/kernel.py` (`nb_entry`, `nb_index`,
   `nb_check`; `nb_check('.')` verified all 249 entries and their hash chain at 1ad44cb) and
   `.claude/skills/standing-instructions/kernel.py` (`si_add`, `si_retire`, `si_link`). Append entries only through them;
   never edit old entries.
2. Record the date GitHub shows for the tag (SI-79; NB-20261016-34 says it is still to be read). Why: an outside time stamp
   for the freeze. Local tag date: 2026-10-16T21:48:50Z. Where: new notebook entry; without `gh` use
   `curl -s https://api.github.com/repos/kreid415/wasserstein-critic-deconfounding/git/refs/tags/prereg-tier12-v3` and the
   tag object it points to (`tagger.date`); the events API keeps push events 90 days. Machine: any with GitHub access.
3. Fix the launchers that still name the v2 commit. `experiments/X1_local/run_X1_local.sh` (line 14 insists the checkout is
   c88cce3; it also pins the manifest hash MAN_SHA and the v2 checkout folder) and `experiments/X1_jhpce/x1_jhpce_command.py`
   (line 23, constant TAG_SHA = c88cce3, review item G-17) must point at the v3 tag; the 384 h and 576 h harness time limits of chained JHPCE jobs are unverified
   (G-15; `experiments/X1_jhpce/SUBMISSION_PLAN.md` still describes v2). Why: SI-54 says c88cce3 counts as v3 for every arm
   except Sinkhorn, so non-Sinkhorn fits may come from either, but Sinkhorn output made at c88cce3 is refused by
   `scripts/run_stage.py`. Machine: edit anywhere; run on the local machine. Show the user any change under `scripts/` or `src/`
   (frozen by the tag); launcher files under `experiments/` are not named in the tag message.
4. Sinkhorn redo, stage 1 (SI-57): refit the 180 A1 Sinkhorn fits with the corrected solver (tolerance 1e-4, iteration cap
   500,000, float64: SI-58, SI-61, SI-62). Why: the old Sinkhorn gradient was wrong (SI-51, NB-20261013-01). Preflight first
   (skills `.claude/skills/experiment-preflight` and `experiment-preflight-singlecell`; the pre-launch check record is committed in 1ad44cb at `experiments/preflight_sinkhorn_redo_20261016/`, verdict GO;
   launch sequence, conditions and the independent launch review: `docs/claude/LAUNCH_SINKHORN_STAGE1.md`). Where: `scripts/run_stage.py` (usage at its top), A1 manifest in `manifests/`, output under
   `$D/tier12*/`. Cost reference: the hardest-regime full-length fit took 7,778 s, 2.8x the pre-fix time (NB-20261013-11).
   Machine: the user's local RTX 3080 machine (a local run needs no cluster go, but report it). Not the cloud VM (no GPU, no data).
5. Re-run the rules in order, overwriting the record files in place (SI-87; old versions stay at the tags):
   (a) R1 for the Sinkhorn family and its 18 extension fits (`scripts/freeze_a1_grid.py`); (b) R4-a1 over all six arms
   (`scripts/freeze_matched_lambda.py --stage a1`), which can move non-Sinkhorn A2/A3 windows (NB-20261013-09), so refit
   any non-Sinkhorn A3 cell whose window moved; (c) the 36 Sinkhorn A3 fits; (d) R2 and R3 in full
   (`scripts/decide_a2_a3.py`). Records: `docs/prereg/r1_record.json`, `r1_record_round1.json`, `r4_a1_record.json`,
   `r4_a1_record_round1.json`, `r2r3_record.json`. The redo's notebook entry must list every decision for the other five arms
   that changed (SI-87). A Sinkhorn iteration-cap hit stops the stage until the user decides (SI-81 b); an NMI-0 case from
   scib is also the user's decision (SI-65, SI-81 a). Fits need the local GPU; the rule scripts run on CPU anywhere.
6. Identity proof. The proof that non-Sinkhorn fits and scores are bit-identical to c88cce3 was last run at c3fd92d
   (`experiments/integration_checks_20261013/README.md`; 59 of 59 non-Sinkhorn fits equal, 7 of 7 scores equal). Its own
   rule says it stands for a later commit if the diff changes no fit or score code. Checked for this snapshot:
   `git diff --stat c3fd92d prereg-tier12-v3 -- src scripts/fit_paper_config.py scripts/scvi_adversarial_plan.py
   scripts/score_scib_native.py` is empty, so no re-run is needed at the tag. `scripts/run_stage.py` did change after
   c3fd92d (start-up provenance checks only); `notebook/2026-10-14.md:340` records that it builds no fit or score command,
   environment variable or thread setting, so it is outside the proof's criterion. No fresh proof was run for this snapshot. Re-run the check after any later change to those
   files; if it is not empty, re-run `experiments/integration_checks_20261013/results/run_c3fd92d.sh` (about 35 min of fits plus 1 to 2 h of scoring) on the local machine.
7. Rule-free runs may start before step 5 ends because they depend on no rule (PREREG section 0): X1 arms `none` and
   `scvi_adv`, X13 baselines; including resuming filler scoring pass A (8 score files exist, not 1: NB-20261016-20; pass B
   done, 150). Resume with `--clear-foreign-claims` (stale claim files from the paused runs). Local machine, after preflight.
8. X1 local part: 1,125 rows of atac_small, immune, sim1 (`experiments/X1_local/run_X1_local.sh`; ledger tally 19 scored,
   1,097 need fits, 9 pending: NB-20261016-19). Starts after step 5 because the manifest rows need lambda, adversary input
   and standardisation from R1 and R2/R3; rebuild the resolved manifest and update MAN_SHA. 8 GPU lanes, 3 scorers during
   fits (the settings in that script). Machine: local RTX 3080.
9. X1 cluster part: 5 jobs, 1,800 fits (immune_hum_mou, atac_large, lung, sim2, pancreas; 360 fits each; 8.5 GPU-days
   predicted; `experiments/X1_jhpce/SUBMISSION_PLAN.md`). First re-run the L40S cross-host GPU gate at the new tag (review item
   CR2-01; `cluster/jhpce/gate_gpu.sh`). Every cluster submission needs the user's explicit go (project GO rule, 2026-10-03):
   show what runs, where, how long. Machine: JHPCE (partition gpu, account jhpce, one L40S per job); data and harvests stay on
   its scratch, latents are downloaded and scored locally with the tagged scorer (wcd-kbet environment, R with kBET).
10. Implement the 2026-10-17 follow-up design decisions (SI-93 to SI-98; NB-20261017-01 to NB-20261017-08; brief `docs/followup_design_decisions_20261017.md` on `final-review-fixes`). Why: these changes alter which cells the follow-up rows draw on, and R4-x1 (the rule that sets every follow-up's penalty strength from the X1 scores) must not run before they are committed. What to do, before R4-x1 and before anyone examines an X1 result (SI-68; under SI-78 a script that uses X1 scores counts as examining):
    (a) Rebuild the follow-up manifest and code: stratified nested X8 draw with N/16 cells per batch (SI-94); X7 low/matched/high penalty-strength window (SI-95) and X7 depleted-reference arm on pancreas and sim1 at 2 depletion levels (SI-96); X12 16-run test for `x12_runs=16` (SI-97); X6 spectral-norm discriminator arm with the critic's optimiser (SI-98); the I036 gradient-penalty fix (no gradient penalty on spectral-norm critics); unconditioned-decoder rows for X3, X6, X7, X8 and X12 (SI-93).
    (b) Commit them with a PREREG section 9 amendment and a new code tag after `prereg-tier12-v3`. `docs/PREREG.md` and `docs/ANALYSIS_SPEC.md` are frozen by the tag: the amendment is a new section, not an edit of frozen text.
    (c) Run an identity check that X1 fit paths give results identical to the v3 tag under the new tag (as `EQUIVALENT_COMMITS.json` did for SI-54).
    (d) Write and lock the follow-up analysis specification (P2 dose-response, P3, P4, R3 factor effects, X15) before R4-x1.
    (e) Re-estimate the cost: the cost model has no decoder term, so the +2,052 fits (about +570 GPU-hours) for unconditioned rows is provisional, and the depleted-reference arm is not yet specified.
    Nothing in this step blocks the Sinkhorn redo (step 4) or the X1 runs (steps 7 to 9). Machine: edit anywhere; the identity check needs the local GPU machine. An experiment preflight GO is still needed for each follow-up wave.
11. R4-x1 (`freeze_matched_lambda.py --stage x1`) after X1 is scored; then the follow-ups X3, X6, X7, X8, X12 (seeds 10 to 12)
    and X15. Their analyses are locked later, while X1 runs and before anyone examines an X1 result (SI-68;
    `docs/ANALYSIS_SPEC.md` section 0). X7 and X8 were rated NO-GO as designed (I015, I077, I078, D-06, D-07); step 10 implements the design fixes the user chose on 2026-10-17. The reporting script for no-pair step counts and unreached cells is written after the runs
    (SI-81 c).
12. X1 analysis, once, with `scripts/analysis_x1.py` (usage in its docstring; inputs: X1 manifest, extension manifests and score
    directories, one failures table covering every row, `docs/prereg/r1_record.json`, the R4-x1 record). Runs only after all X1
    rows are final; refuses to overwrite a record. It is CPU-only, so it can run on any machine that holds the score tables, but
    the run is the first examination of X1 scores (SI-78): do it only on the user's instruction. Tests, as last run (NB-20261016-31, fresh clone at 8f223c8,
    env scvi-api, `WCD_SRC=src`): `python scripts/check_tests.py --base 7fa19ea --src scripts cluster -- tests/analysis
    tests/prereg tests/scvi tests/stage tests/jhpce tests/test_legacy_retired.py` (exit 0; 445 passed, 18 skipped; 1,192 s).
13. Paper (branch `paper`, section 6): fill Results from the analysis record only after step 12; open paper decisions in section 3.

## 3. Pending user decisions and deferred ones

Rule for asking (SI-92, user 2026-10-16: discuss only decisions that matter for the experiment; defer the rest until after the
experiments are launched; and the user's directive the same day: decide minor details yourself, keep one defaults list, ask
only what would force a full re-run if wrong, or the cluster go). Wording, provenance, documentation and tooling choices: decide
and record; do not ask. Also: SI-67 and the plain-language rule (spell out every label and acronym), one decision per message.

Needed from the user to proceed (matter for the experiment):
1. Go for each cluster job: the JHPCE X1 jobs, and the L40S gate re-run (project GO rule). Nothing submitted yet.
2. Go to launch the Sinkhorn redo once its pre-launch check reports GO (a local run; report it; the user wanted launches soon).
3. Case-by-case decisions that arise during runs, by registered rule: each Sinkhorn iteration-cap hit (SI-81 b) and each scib
   NMI-0 case (SI-65, SI-81 a) stop the stage until the user decides whether it counts as a failed fit.
4. (Decided 2026-10-17, no longer pending.) The six follow-up design questions were answered: ledger entries SI-93 to SI-98 (`CONSTRAINTS.md`), notebook entries NB-20261017-01 to NB-20261017-06; the defaults list is NB-20261017-07 and what they require is NB-20261017-08. The decision brief is `docs/followup_design_decisions_20261017.md` on branch `final-review-fixes`.

Deferred (do not raise before the experiments are launched):
- Paper: journal, title, authors, figures; whether the contribution is worded as a "controlled decomposition"; whether research
  question R5 states a hypothesis; which data the premise check R0 uses; keep or drop the reviewer-pointer tag (CV-05);
  keep or cut the Supplement disclosures about this study's own pre-registration deviations (session ed231f45).
- Strip text naming the earlier submission and a withdrawn method from `paper/evidence.md` and hidden LaTeX comments in
  `paper/paper.tex` before submission (public repo).
- Wording-level fixes: ANALYSIS_SPEC.md:101 says atac_small and atac_large "share 3,618 cells"; verified (memory mem_08a34149a58f):
  all 11,270 atac_small cells are in atac_large and 3,618 of them (one 10x Genomics batch) have identical names. Also
  `docs/ANALYSIS_SPEC.md` lines 22 and 342-343 still say the tag "has not been made". `docs/ANALYSIS_SPEC.md` is frozen by the tag: ask the user how to
  record a post-tag text correction. `docs/NEXT_TAG_ERRATA.md` lists text fixes "to apply at the next tag"; its status after
  the v3 tag was not checked.
- Housekeeping: the user was to delete 8 old worktrees and 2 stray files (`rf_extract.txt`, `HANDOFF-final-review-fixes-4.md`,
  untracked) under the live code folder (Q18, handoff 14); the live checkout still showed both stray files at this snapshot.

## 4. Open issues (notebook INDEX "Open issues", plus items found while writing this)

`notebook/INDEX.md` lists 14 entries "not referenced by any later entry". None is of type ISSUE; all are RESULT, DECISION or
FINDING entries. One line each, with whether it is still live:
1. NB-20261003-21 (FINDING): review findings on lung valid mode and immune batch names checked against data. Closed fact, not live.
2. NB-20261013-13 (RESULT): reviews of the Sinkhorn fix and guards, PASS WITH FIXES. Superseded by later rounds (NB-20261014-20, NB-20261015-01, NB-20261016-13). Not live.
3. NB-20261014-20 (RESULT): rigour review, 47 findings, 20 to fix. Fixed in later commits. Not live.
4. NB-20261015-01 (RESULT): re-checks at 08f1e3c, 9 items. Fixed in round 3. Not live.
5. NB-20261013-12 (RESULT): non-Sinkhorn paths bit-identical to c88cce3 at 2b52542. Live as evidence (step 6).
6. NB-20261013-16 (RESULT): branch scorer reproduces stored production score rows. Live as evidence (step 6).
7. NB-20261002-25 (DECISION): kBET (batch-mixing metric) uses a fixed R seed. Standing rule, in force.
8. NB-20261002-26 (RESULT): kBET identical when one latent is scored twice. Standing fact.
9. NB-20261002-27 (RESULT): trajectory conservation changes biology score only on trajectory tasks. Standing fact.
10. NB-20261002-28 (DECISION): preflight GO for the scorer change. Historic.
11. NB-20261002-30 (RESULT): prereg-v2 tests pass on a fresh clone. Historic.
12. NB-20261013-09 (RESULT): R4-a1 can move non-Sinkhorn A2/A3 windows after the Sinkhorn refit. Live: handled in step 5 (SI-57).
13. NB-20261013-11 (RESULT): hardest-regime Sinkhorn fit far below the cap; 2.8x pre-fix time. Live as the cost input to step 4.
14. NB-20261002-29 (RESULT): wall time v2, 6,718 fits, 26.2 days on 8 lanes. Old estimate; not re-derived for v3. Treat as provisional.

Live items not in that list:
- SI-79 GitHub date of the tag not yet recorded (step 2).
- Launchers name c88cce3; G-15 time limit unknown; CR2-01 L40S gate not re-run at v3 (steps 3, 9).
- X1_local ledger: 9 rows pending, counted only (NB-20261016-19); `failures.csv` of X1_local has 0 data rows (NB-20261016-25).
- Known limitations stated in PREREG: iteration-cap and NMI-0 handled case by case (RC-M4, SI-65, SI-81); design underpowered
  for small effects (R1 power report); P3 (batch-count scaling) and P4 (reference choice) not specified as designed.
- Display disclosures: three displays of score values to an agent are listed in `docs/PREREG.md` section 9 (SI-88, SI-78 as
  amended); none counts as examined. Repeat the display audit before analysis if new sessions touched X1/X13 files.

## 5. In flight at the snapshot

Lead session cec0e8c3 (Claude Science chat "Launch Light-Dataset Experiment Wave"; processing, 188 messages). It started
as the confirming re-review of fb7edac and has since done the following, in order:
1. Re-review (RIGOR_REVIEWER, Opus): NOT READY; 15 of 16 earlier findings (RC-B1..RC-m10) fixed, RC-M2 partly, 6 new items
   CR-01..CR-06 (CR-01 major: three commit ids cited in PREREG did not exist; real ids da8ee06, 0100aa5, 4b02068;
   NB-20261016-27).
2. User answers (NB-20261016-24, -25, -22, -23, -26): SI-78 sentence reworded; failures.csv row-counted; 278 attempt logs
   hashed (SI-90); one test added for `scripts/check_tests.py` (SI-91); standing rule SI-92.
3. Fix pass (RUNNER_ENGINEER, Sonnet) applied as 3b9cf67, dcde2c5, a0ea5cd, 5f9bab0 (NB-20261016-28). The lead re-ran the
   scripts test tier at 8f223c8: 445 passed, 18 skipped, check_tests OK (NB-20261016-31).
4. User said yes to the tag (NB-20261016-32, 17:47); top-up display audit over 222 messages found 0 flagged lines
   (NB-20261016-33, repo 10a1ca5); tag made and pushed (NB-20261016-34, 17:54, commit 857234a; the preflight GO record followed as 1ad44cb).
5. Last action: dispatched a read-only pre-launch check of the Sinkhorn redo stage 1 to sub-agent 588cef60 (RUNNER_ENGINEER,
   Sonnet; processing, 54 messages at the snapshot): writes `experiments/preflight_sinkhorn_redo_20261016/PREFLIGHT.md`,
   `preflight.json`; launches nothing. Outcome: GO; the record was committed as 1ad44cb on `final-review-fixes`. The session
   then asked the user for the go and ended without an answer; nothing was launched.
Other recent sessions (all finished): 95c88c4c (round-3 patches, SI-89, audit redo; handoff 14), 83e5f00a (user's request to
cut pre-launch questions; produced the defaults rule above), d684be61 and 17c506be (round-3 fixes, 19-question diff review),
ed231f45 and 544c8a5f (paper intro, related work, references on branch `paper`).
Update (2026-10-16 18:45 New York): 857234a and 1ad44cb are pushed; `origin/final-review-fixes` = 1ad44cb. Session cec0e8c3
ended while asking the user for the go on Sinkhorn redo stage 1 (preflight GO in `experiments/preflight_sinkhorn_redo_20261016/`;
launcher `run_A1_sinkhorn_redo.sh` there, 180 fits, about 13 h on the RTX 3080). Nothing was launched. Expect later notebook
entries (NB-20261016-35 onward, or 2026-10-17 pages) only if the user ran something since.

## 6. Branches and tags

- `final-review-fixes`: the research branch; 10a1ca5 on GitHub at the snapshot, 289 commits ahead of `main`.
- `claude-code-migration`: this hand-over bundle, five commits on 1ad44cb (= 10a1ca5 + NB-20261016-34 + preflight GO record).
- Notebook and ledger appends happen on one branch only, `final-review-fixes`. The Claude Science sessions have
  finished and pushed (last commits 857234a, notebook entry NB-20261016-34, and 1ad44cb, the Sinkhorn-redo preflight GO);
  this branch sits on 1ad44cb. First action of the first Claude Code session: `git checkout final-review-fixes && git pull
  && git merge origin/claude-code-migration`, push, and work on `final-review-fixes` (or short branches merged back into it).
- `main`: stale (5cb75c0, 2026-10-13); an ancestor of `final-review-fixes`; do not work on it.
- `paper`: the manuscript outline (`paper/paper.tex`, `refs.bib`, `evidence.md`, disclosures); 1bc6304 (2026-10-15), 150 commits
  ahead of `main`, branched from 08f1e3c. Not yet merged with the later fixes; the paper's numbers must be re-read from the
  analysis records when they exist. Building the PDF needs a LaTeX toolchain.
- Merged historic branches (0 commits ahead of `main`, kept for history): amend-v2, fix-runner, fix-score-fit, integrate-arms2,
  integrate-fixes, integrate-gate-v2, integrate-jhpce, integrate-phase0, jhpce-gate-v2, jhpce-production, jhpce-setup,
  missing-arms, missing-arms-v2, prereg-rules, prereg-v2, stage-runner.
- `claude-code-migration`: this migration's branch (new).
- Tags: `prereg-tier12-v1` (129142b, 2026-10-03), `prereg-tier12-v2` (c88cce3, 2026-10-04; the code the A1-A3 pilots ran under, and the
  one the local X1 launcher and the JHPCE plan still name; result-identical to v3 for every arm but Sinkhorn per `docs/EQUIVALENT_COMMITS.json`, SI-54),
  `prereg-tier12-v3` (10a1ca5, 2026-10-16; the freeze the Sinkhorn redo and X1 run under; do not move or recreate it).

## 7. What the cloud VM can and cannot do

Can: read, search and edit the repo; run CPU tests that need no data (`tests/prereg`, `tests/analysis`, most of `tests/stage`;
15 tests are skipped without `PREPPED_DIR`); run the rule scripts and `scripts/analysis_x1.py` on score tables if the user
supplies them; edit `docs/` and `paper/`; use `git` and `gh` with the user's GitHub access.
Cannot, without something the VM does not have:
- Data: the prepared datasets (`prepped_scib`), latents, scores, ledgers and the 8-dataset counts live under
  `/home/kendall/experiment_data/wasserstein-critic-deconfounding` on the user's machine only. Nothing of that is in git (ignored:
  `**/results/`, `*.csv`, `*.tex`; force-add new result folders with `git add -f` and check `git ls-files`; mem_ba17295d3ef4).
- GPU: no local GPU is assumed; every fit (Sinkhorn redo, X1 local, fillers) and the identity proof need the user's RTX 3080
  machine, scoring needs its `wcd-kbet` environment (R with kBET, `R_LIBS`, `KMP_AFFINITY=disabled`).
- JHPCE: whether the VM can reach the cluster is unknown (not checked). Cluster jobs need the user's go and their credentials;
  scratch purges remove files, so environments live in `$HOME` and data on scratch (project cluster rules).
- Claude Science tools: no artifact store, memory database or `ask_user` cards (notebook/ledger helpers and profiles are ported to `.claude/`);
  the old artifacts cited in notebook links (`artifact:<uuid>`) are not reachable from git. Where an entry cites one, the
  committed copy under `experiments/` or `docs/` is the evidence.
- Environment names used in commands (scvi-api, wcd-gpu, wcd-kbet) are conda environments on the user's machine; rebuild
  recipes are in `environment.yaml`, `conda-lock.yml`, and the Env track's `envs/` folder.
First-day order on the cloud VM: steps 1, 2, 3 (edits only), housekeeping that does not touch frozen files; then hand the
user the exact local commands for steps 4 to 8 and wait for their go on cluster work.
