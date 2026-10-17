# Handoff: final-review fixes, part 3

**Session:** 2026-10-13 · about 380 messages · 0 folds · frame d202ca29-41bb-4597-ae51-8c6ea64d85be
**Status:** The Sinkhorn code round (precision check SI-58, cap SI-59, review fixes F1, F2, F4, F7, F8) is committed and independently re-reviewed (PASS WITH FIXES; the two "fix" findings are fixed). Six guard follow-ups are committed and mutation-tested. The SI-54 commit-equivalence code, the P1-P4 analysis code, the remaining PREREG amendments, the final identity proof and the user's diff review are not started. No tag. Nothing is running.

## Start message for the next chat
```
Continue from HANDOFF-final-review-fixes-3.md (artifact id bd2ef03f-f2a1-4a51-99ea-96dd78338c9a) in this project; read it before doing anything else.
Objective: finish the final-review fixes on branch final-review-fixes, show the user the diff, then tag and launch X1.
Next action: re-review notes R2-04/05/06/08, then SI-54 code (confirm the tag name first).
Open first: CONSTRAINTS.md; REREVIEW_SINKHORN_ROUND2.md
Lost with the old kernel: nothing to reload
Pending: no runs or sub-agents; re-review notes R2-03 to R2-09 open
Do not redo: SI-58 check (float64 kept); F1 NaN-only stop (NB-20261013-24)
Before reusing any identifier, number or quote from the previous chat (frame d202ca29-41bb-4597-ae51-8c6ea64d85be), search its archive.
```

## Objective
Benchmark Wasserstein critics against discriminator adversaries for single-cell batch correction (scVI backbone, scIB metrics) under the pre-registration (docs/PREREG.md, tag prereg-tier12-v2 = c88cce3). Finish the final pre-publication review's fixes on branch `final-review-fixes`, redo the Sinkhorn arm in rule order (SI-57), prove that everything else is unchanged, show the user the full diff (SI-52), then make a new tag and launch X1.

## State
Repository: clone `D/code/wcd_final-review-fixes` (D = /home/kendall/experiment_data/wasserstein-critic-deconfounding), branch final-review-fixes, HEAD b3dbabc, pushed to github kreid415/wasserstein-critic-deconfounding (check with `git ls-remote`), working tree clean. 21 commits since the session start (82263e5).
- Done, verified, re-reviewed: Sinkhorn round, commits 772d59b, 01b354f (preflight GO), daf1c78, 81f8ce4, 6b4aefc, 89d5e2c; README experiments/sinkhorn_fix_20261013.
  - SI-59: `SINKHORN_MAX_ITER = 500,000`; a cap hit still raises.
  - F1: each solve stops on a NaN marginal error (`not err >= tol`). NaN, +inf and -inf latents give a NaN error at the first iteration, so the loss is NaN and the training guard records 'diverged'. An infinite error with finite potentials does occur for finite latents of about 1e18 or more and must keep iterating (re-review R2-01; probe rows s = 1e20, 1e30); pinned by a test row (89d5e2c).
  - SI-58: float64 kept (NB-20261013-20). float32 met the criterion on the 598 hardest minibatches (1 - worst cosine 2.788e-07, worst relative value error 2.974e-06, no cap hit) but not on all 3,076 (3.17e-3 on collapsed R1-extension minibatches) and was no faster on the RTX 3080 at 128 cells (median time ratio 0.989-1.014). The float64 re-run is bit-identical to validation_12mb.csv.
  - F4: the reference test's gradient threshold 3e-4 pins SINKHORN_TOL; a dtype spy pins the float64 solve. F7/F8: docstrings (not a W1 estimate; cost gradient through the row point only).
- Done, verified, NOT yet independently reviewed: guard follow-ups G-01 (no-pair minibatch counter; `no_pair_steps` in the plan record only when > 0), G-02, G-03 (fitter end to end refuses an unread setting; coverage now measures subprocesses via `pyproject.toml` `[tool.coverage.run] patch = ["subprocess"]`), G-04 (table pins), G-05 (lam must stay 0 for none, scvi_adv, scANVI; 1,908 manifest rows checked), G-14 (harvest refuses a latent without history). Commits 96f4258, 4f1a570, 39aa51e, ca18e0d, 43a0f91; README experiments/guard_followups_20261013; 11 of 11 planted bugs caught. G-08 deliberately not done (scorer untouched, so the score identity proof stands).
- check_tests.py OK per tier: src with env wcd-gpu (tests/wcd_vae/test_divergences_v2.py), scripts + cluster with env scvi-api (tests/scvi guard, baseline, plan, missing-arms files and tests/jhpce/test_prod_job.py: 115 passed, 1 skipped). After 89d5e2c only test_divergences_v2.py was re-run (25 passed); its change is a comment plus a test row.
- fl_lint (fail-loud-pipelines) at medium: OK on the 10 files changed in the Sinkhorn round and guard follow-ups; one low finding (mutate.sh has no `set -e`, by design).
- Paused, untouched: local X1 (20 latents, 19 scored) and filler part A; JHPCE X1 not submitted. No runs, no sub-agents.

## Artifacts
- [REREVIEW_SINKHORN_ROUND2.md]({{artifact:8617428c-646a-4060-ac83-db82581de280}}) — independent re-review of 6b4aefc: claims checked, findings R2-01..R2-09.
- [rereview_sinkhorn_round2_evidence.tar.gz]({{artifact:f94b2587-97b7-4ce6-b2a3-0abfa9365e94}}) — the re-review's probes and outputs (inf-error regime, end-to-end guard probe, GPU microbenchmark).
- [si58_summary.md]({{artifact:0d875999-4e1e-4b92-a26a-12491ec79fe2}}) — SI-58 decision table (regression check, accuracy on the 598 and on all 3,076, GPU cost).
- [PREFLIGHT_si58.md]({{artifact:bec015f3-9c5a-4819-9b15-df45a4f1a6c7}}) — preflight record of the SI-58 check (GO, 24 items).
- [README_sinkhorn_fix_20261013.md]({{artifact:85c994e6-ef4f-4ee7-a94c-1852621d9d3a}}) — Sinkhorn fix: changes, evidence table with re-run commands, SI-58 section.
- [README_guard_followups_20261013.md]({{artifact:ecbf6332-8a95-46d1-af9e-e26bcc98854a}}) — guard follow-ups: finding, change, commit.
- [guard_mutation_results.tsv]({{artifact:e6369bd8-3ce1-463e-a22d-d94991a7251a}}) — 11 planted bugs and the test that caught each.
- [notebook_2026-10-13.md]({{artifact:82711001-bf74-4605-8900-41888b752f90}}) and [notebook_INDEX.md]({{artifact:fe606477-3149-4c02-9558-2e4568ea2f22}}) — lab notebook to NB-20261013-26.
- [CONSTRAINTS.md]({{artifact:61319a0e-6c23-4cd2-9134-1b391fbef8d7}}) — standing-instructions ledger, 59 rules, unchanged this session (also at the repo root).
- [review_findings_sinkhorn_and_guards_20261013.json]({{artifact:5ee08839-e3b8-42b6-904e-3346fabbc320}}) — first-round findings F1-F8 and G-01..G-18.
- [HANDOFF-final-review-fixes-2.md]({{artifact:9dbbabb4-eef1-4013-a0e3-3d8fd3df3731}}) — previous handoff (part 2).

## Kernel state (will be lost)
| in memory | rebuild cost | how to restore |
|---|---|---|
| skill helpers (si_*, nb_*, pf_*, fl_*, handoff_*) | trivial | load skills standing-instructions, lab-notebook, experiment-preflight, fail-loud-pipelines, session-handoff |
| `res` (first-round findings), `rr` (re-review result) | trivial | json.load the two review artifacts above |
| coverage venvs `dev/covenv` (wcd-gpu) and `dev/covenv_scvi` (scvi-api) in the workspace | cheap | `python -m venv --system-site-packages` on each env, then `pip install pytest-cov==7.1.0` |
| mutation worktree `wt_mut` in the workspace | none | `bash experiments/guard_followups_20261013/mutate.sh COMMIT DIR` re-creates it; run `git worktree prune` once the workspace is swept |

Git: .git/config is not writable; export GIT_AUTHOR_/GIT_COMMITTER_ NAME=wcd EMAIL=noreply@localhost and push with the token URL. `**/results/` is gitignored; result files are force-added. Envs: scvi-api (fits, tests/scvi, tests/jhpce, validation scripts), wcd-gpu (tests/wcd_vae with PYTHONPATH=src), wcd-kbet (scoring). Prefix KMP_AFFINITY=disabled OMP_NUM_THREADS=1. Review worktree D/code/wt_rereview_sinkhorn (detached at 6b4aefc) can stay.

## Decisions
- NB-20261013-17, corrected by -24 and -26 (agent, within SI-59): stop on NaN only, not on +inf. SI-59's literal wording ("a non-finite marginal error ends the loop at once") would return an unconverged gradient for huge finite latents (re-review: cosine 0.93-0.95 with 10 batches). Goes to the user at the diff review.
- NB-20261013-20 (agent, within SI-58): float64 kept. float32 met the user's criterion on the 598; it is kept out because it is no faster at 128 cells and fails on the full set. Re-review R2-06: say this plainly; the user may overrule.
- G-08 not done: error type only; leaving the scorer unchanged keeps the score identity proof.
- SI-54 implementation, proposed, NOT built: a committed `docs/EQUIVALENT_COMMITS.json` naming c88cce3 as result-identical to the new tag (named, e.g. prereg-tier12-v3, resolved with git) for every arm except sinkhorn. One helper in scripts/prereg_rules.py: always refuse sinkhorn rows at c88cce3; when rows mix commits, count covered-arm rows at c88cce3 as the tag commit, then require one commit. Use it in prereg_rules.check_provenance (pass each row's arm from outcomes()), run_stage.py's preflight and gate scorer-provenance checks (lines ~674, ~1085) and the gate's latent git_sha check (~1153). The JHPCE bundle must carry the tag ref (lesson RR-01). Alternative considered: pin sha256 hashes of the result-determining code files. Tag name and mechanism to confirm with the user.

## Dead ends
- float32 Sinkhorn solve: no speed-up at 128-cell minibatches (the call is bound by per-iteration kernel launches and the convergence test's host sync). Revisit only if minibatches reach about 512 cells (re-review R2-05: about 2.7x per iteration there).
- A float32 solve of latents above about 1e19 hangs: infinite cost diameter, `_eps_schedule` never ends (R2-07). Probe huge scales in float64 only.
- One pytest run cannot cover both tiers: the scvi-api env cannot import the wcd_vae package (no scib). A changed test file that is not run in a tier shows up as a WARN there.
- `pkill -f PATTERN` inside a bash cell kills the cell itself (from part 2).

## Next steps
1. Re-review notes: R2-05 (SINKHORN_DTYPE comment and README: "no speed-up at 128-cell minibatches"), R2-06 (NOTE entry rewording NB-20261013-20), R2-08 (sample sizes in the cap comment), R2-04 (optional assertion `SINKHORN_MAX_ITER == 500000`). R2-03 and R2-07 need no change.
2. SI-54 code as proposed under Decisions, after confirming the tag name with the user. Load fail-loud-pipelines first. Known-answer tests: all-c88cce3 input passes; c88cce3 non-Sinkhorn plus tag Sinkhorn passes; Sinkhorn at c88cce3 refused; two undeclared commits refused. Then a PREREG section 9 entry.
3. P1-P4 specification and frozen code with known-answer tests (I001, D-03, D-04/SI-56, D-08, D-09); decide whether not-evaluable cells leave the R2/R3 cell set.
4. Remaining PREREG section 9 entries (SI-51/SI-57; guards, including G-01's sentence naming the row groups that can hit a no-pair step: X7 reference-anchored MMD on pancreas and sim1, rarely X3 atac_small MMD, X8 sim2 MMD and X1 sim1 MMD and Sinkhorn; SI-58/59; SI-54; SI-56) and Methods notes (F6 validation scope, F7 "not a W1 estimate", G-07 sysVI likelihood labels). Then a RIGOR_REVIEWER review of all amendments plus the commits after 6b4aefc (R2-09).
5. Final identity proof at the candidate commit: experiments/integration_checks_20261013 fit_identity.sh (needed: G-01 changed the MMD function signatures) and score_identity.sh.
6. Show the user the full diff 5cb75c0..HEAD with the open questions below (SI-52). After sign-off: the new tag; x1_jhpce_command.py TAG_SHA (G-17); the 384/576 h harness-ceiling fallback (G-15); experiment-preflight with the single-cell pack; Sinkhorn redo per SI-57 as a cs-local host job; CR2-01 L40S check run (needs the user's go); resume local X1 and filler part A (--clear-foreign-claims); ask for the JHPCE go.

## Open questions for the user
- At the diff review: F1 stops on NaN only, not on every non-finite error as SI-59 is worded (why: Decisions); SI-58 keeps float64 although float32 met the criterion on the 598; the new tag's name and the SI-54 mechanism.
- Carried over: amendment wording (D-01, D-02 reporting rule, I091 note), the I042 allow-list difference from the review's suggestion, the refusal of legacy latent writers (I050), and the scib 1.1.7 NMI-0 edge case (loud error, not fixed).
