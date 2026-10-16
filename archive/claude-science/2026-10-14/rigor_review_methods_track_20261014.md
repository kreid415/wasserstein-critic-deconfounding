# Rigour review: methods-track fragments of paper/paper.tex (outline)

Reviewer: RIGOR_REVIEWER (independent check). Date: 2026-10-14. Repository pinned at 08f1e3c, read with `git show`, `git ls-tree`, `git log` only; data folder D read-only.

**Inputs reviewed.** methods_track.tar.gz (version 0d828fa2, sha256 b6ab6e85...): legend.tex, rq.tex, methods.tex, results.tex, discussion.tex, supplement.tex, the test wrapper and the build logs. lock_audit.csv (version 5242c490; 386 rows). Shared preamble (version bead4b1f).

**Verdict: PASS_WITH_FIXES**, on the condition that all 6 blocker replacements are applied. A blocker is not waived. The structure, the classification of X1 results, proposals and redesigns as pending, and almost every number and equation check out. The blockers are local text replacements.

**Counts.** 41 findings: 6 blockers, 16 fixes, 19 notes. 386 lock_audit items checked (every line of the six fragments, and every supplement table cell and equation). 10 coverage gaps.

## What was checked, and how

- **Ledger.** CONSTRAINTS.md at 08f1e3c, all 77 entries read. Checked by script: SI-42 is the only entry whose source is not the user; SI-17 carries an agent-derived sentence; SI-35 and SI-59 are retired. Every SI citation in the fragments was compared with its rule text. Uncited: SI-06, -12, -29, -49, -50, -67 (deliberate omissions in lock_audit; acceptable).
- **PREREG.** docs/PREREG.md compared byte for byte with `git show c88cce3:docs/PREREG.md`, by section and by section-9 entry. Tag prereg-tier12-v2 points to c88cce3, and the manifest of record has the same sha256 at both commits.
- **ANALYSIS_SPEC** (all 498 lines, including Appendix A) and **PAPER_PLAN** (all 298 lines): every user-decision section was separated from the agent proposals.
- **Code.** Each of the 99 distinct file:line citations was opened at 08f1e3c (scripts/, src/wcd_vae/wcd/, tests/), and the full training step, heads, critic loss, gradient penalty, barycenter, MMD, Sinkhorn, scorer and fitter code was read. Equations S-M1 to S-M18 were checked against that code.
- **Manifest of record.** 6,930 rows, sha256 6efbe422c8f09f98... recomputed. S-E1 (rows, tasks, arms with steps, decoders, seeds, strength source), S-E2, S-E3 and the X3, X6, X8 and X12 designs were recomputed from it with pandas.
- **Records.** docs/prereg/r1_record.json: all 8 grids and the power block. docs/prereg/r4_a1_record.json: targets of all 6 blocks.
- **Data.** D/prepped_scib obs of all 8 tasks (cells, batches, cell types, dpt_pseudotime). Name overlaps and barcode-within-batch overlaps. The counts layers of all 8 tasks were scanned for non-integer values per batch. Gene-activity rows of 200 matched ATAC pairs were compared.
- **Environment records.** cluster/envspec pip freezes and conda explicit files; docs/jhpce/gate_data/versions_*.json and gate_report_L40S/versions_fit_gpu.json; the live env wcd-kbet site-packages (read only); per-row provenance in D/tier12_v2/A1 (61 of 1,098 score rows sampled, one latent config).
- **Review and notebook files.** docs/review_final/REVIEW_FINAL.md; D/review/lead/verified_issues.csv (99 rows); D/review/rereview/design_fable/REVIEW_design_recheck.md (D-06, D-07); D/review/rigor_code_20261014 (K14, CF-02 and the import-closure output); notebook entries NB-20261004-20, NB-20261006-02, NB-20261012-01, -04, -05, NB-20261013-39, -40, NB-20261014-21 and -29 to -35.
- **Original-submission audit.** Project files original_manuscript_audit.md and original_manuscript_claims_audit.csv (linked from PAPER_PLAN; not in the repository).
- **Literature.** One fact about scIB (13 integration tasks), checked against the abstract and first Results paragraph of Luecken et al. 2022 (doi 10.1038/s41592-021-01336-8), retrieved by web search.
- **Build.** build.log and test log read: 0 errors, 3 underfull boxes; sources ASCII-only; no absolute paths or artifact markers. The build was not re-run.

## Specific items (a) to (i)

| Item | Verdict | Finding |
|---|---|---|
| (a) 'no fit or scoring code changed c3fd92d to 08f1e3c' | False as stated. run_stage.py (in the fit import closure) changed (CF-02). The EQUIVALENT_COMMITS validity condition is met; the carry-over is an agent judgment (NB-20261014-29), not re-run. | R-03 |
| (b) Software versions | PyTorch 2.13.0, scvi-tools 1.4.2, scib 1.1.7, R 4.5.2, kBET 0.99.6: correct, matching the recorded files and the per-row provenance. harmony-pytorch 0.1.7 is in no recorded environment file (the freeze lists harmonypy 2.0.0). | R-15, N-19 |
| (c) Final KL weights 0.20-0.61, S-E3 | Correct. Recomputed (E-1)/400 from the manifest epochs: 0.998, 0.613, 0.595, 0.203, 0.998, 0.233, 0.998, 0.998; X15 (E-1)/E 0.996 and 0.989; epochs equal the heuristic on the file cell counts. | none |
| (d) 18 non-integer batches | Correct. Recomputed from the counts layers: pancreas 4 (5,419 cells), lung 12 (22,771), immune 1, immune_hum_mou 1 (the same 1,022 Villani cells). That is 17 distinct batches. | N-09 |
| (e) Frozen X1 grids in S-E4 | Correct. All 6 non-Sinkhorn grids equal r1_record.json x1_grid (cond 1 = conditioned); js\|0 low-edge round 1 (0.03, 0.01) confirmed; Sinkhorn correctly pending. | none |
| (f) PREREG section-9 entries of 2026-10-03/04 byte-present at c88cce3 | 11 of 12 byte-identical. The 2026-10-03 entry has an appended 2026-10-13 correction. sec. 3 and sec. 5 also carry v3 brackets. The verification method used (git log -S on titles) cannot detect in-entry edits. | R-02 |
| (g) X12 resolution | III is correct (I = -ABD = -ACE = BCDE, from the 288 manifest rows). docs/paper_experiment_matrix.csv:14 still says IV. | N-18 |
| (h) Wilcoxon and Holm | 'descriptive' is wrongly attributed to PREREG sec. 5; the Holm family (4 tests, v2; 6, proposal) is not stated; the discussion lacks 'at the Holm-adjusted level'. | R-06, R-13 |
| (i) Glossary | X6 is wrong (no spectrally normalised pooled critic at 10 steps). X7, X8, P3 and P4 are shown as locked. Variant arms and several acronyms are undefined. SI-17's agent sentence is not mentioned. | R-04, R-08, R-09, N-01 to N-05 |

## Findings (severity-ordered)

Every replacement snippet below was compile-tested against the shared preamble: tectonic, env latex, 23 item snippets, 9 table rows and 6 prose snippets, 0 errors and 0 warnings. A `||` separates the parts of a multi-location replacement; the text before `->` names where a part goes.

| ID | Severity | Location | Problem (short) |
|---|---|---|---|
| R-01 | blocker | methods.tex:62; discussion.tex:17 | ATAC overlap given as 3,618 cells (name matches); all 11,270 atac_small cells are in atac_large |
| R-02 | blocker | legend.tex:6 | Legend says all of PREREG sec. 0-8 and the 10-03/04 entries are frozen v2; sec. 3, sec. 5 and the 10-03 entry carry v3 text |
| R-03 | blocker | methods.tex:145 | 'No fit or scoring code changed to 08f1e3c' is false: run_stage.py (fit import closure) changed; carry-over is an agent judgment |
| R-04 | blocker | legend.tex:26; methods.tex:118 | X6 described with a spectrally normalised pooled critic at 10 steps; the manifest has it at 1 and 5 only |
| R-05 | blocker | methods.tex:9; methods.tex:129 | 'All arms and neural baselines share epochs' is false (scANVI extra epochs; X3/X8 subsample epochs; X12) |
| R-06 | blocker | methods.tex:106; supplement.tex:95 | Wilcoxon 'descriptive' wrongly sourced to PREREG sec. 5; Holm family (4 vs 6) not stated |
| R-07 | fix | rq.tex:3 | 'No X1 result has been examined' asserted without source while its basis is pending |
| R-08 | fix | legend.tex:27; legend.tex:28 | X7/X8 glossary designs shown as locked (NO-GO as designed; redesign pending) |
| R-09 | fix | legend.tex:35 | P3/P4 definitions shown as locked (design-time; locked later per SI-68) |
| R-10 | fix | rq.tex:23; rq.tex:38 (also rq.tex:7, :30, :45) | R2/R4 question wording sourced only to PAPER_PLAN; depends on the redesigns |
| R-11 | fix | methods.tex:43; supplement.tex:202-203 | R2/R3 outcome marked pending although SI-66 plus the frozen R4 record fix it (defaults) |
| R-12 | fix | methods.tex:109 | Power figures marked pending although the frozen v2 record holds them |
| R-13 | fix | discussion.tex:18 | Discussion: Wilcoxon 'cannot reject' lacks 'at the Holm-adjusted level' |
| R-14 | fix | methods.tex:53 | '234 pilot fits are redone' treats a rule output as a fixed list (SI-57) |
| R-15 | fix | supplement.tex:273; supplement.tex:255 | harmony-pytorch 0.1.7 absent from the recorded environment files; rebuild evidence not cited |
| R-16 | fix | supplement.tex:298; supplement.tex:299 | Audit files cited as repository files; they are not in the repository |
| R-17 | fix | methods.tex:59 | 'The eight standard scIB tasks': scIB has 13 tasks; these are 8 of them |
| R-18 | fix | supplement.tex:56 (S-M9 importance weights) | SI-25 importance-weight definition missing |
| R-19 | fix | methods.tex:117 (or supplement S-E1 note) | X3 targets and dose-100 sizes (SI-32, SI-33, SI-41, manifest) missing |
| R-20 | fix | supplement.tex:75 (S-M12) | PREREG sec. 4 inheritance for subsampled and architecture variants missing; D-06 caveat missing |
| R-21 | fix | methods.tex:88 | Limits of the equivalence declaration missing |
| R-22 | fix | methods.tex:73 | Frozen rule on reporting scib fallback flags missing |
| N-01 | note | legend.tex:5 | SI-17's agent-derived sentence is not the user's |
| N-02 | note | legend.tex:50 | Family mapping is at prereg_rules.py:67-68 |
| N-03 | note | legend.tex:51 | Seeds: X15 from SI-44; CPU baselines at seed 0 only |
| N-04 | note | legend.tex:47 | Variant arms not in the glossary |
| N-05 | note | legend.tex (glossary); methods.tex:49-53, :68-69, :80, :142; supplement S-M4 | Undefined acronyms and labels for the stated reader |
| N-06 | note | methods.tex:33 | 'share one head' reads as shared weights |
| N-07 | note | methods.tex:29; supplement.tex:27; supplement.tex:198 | Barycenter: at most 10 iterations (stop threshold 1e-7) |
| N-08 | note | methods.tex:49; supplement.tex:194 | Discriminator weight decay 1e-6 omitted |
| N-09 | note | methods.tex:60; discussion.tex:15 | 18 batch-task pairs = 17 distinct batches (Villani in both immune tasks) |
| N-10 | note | supplement.tex:60; supplement.tex:63 | Two R1 clauses omitted in S-M10 |
| N-11 | note | methods.tex:87 | Freeze wording conflates tag (pre-A1) and per-stage commits |
| N-12 | note | rq.tex:15 | 'two-sided' should cite SI-74 |
| N-13 | note | rq.tex:17; results.tex:15-16 | P2 lines in rq/results lack the SI-75 open-item tag |
| N-14 | note | methods.tex:25; legend.tex:48 | 'Each arm adds lambda x term' does not hold for scvi_adv (lambda 0, kappa) |
| N-15 | note | legend.tex:30; methods.tex:119 | X12 'decoder type' ambiguous with conditioned/unconditioned |
| N-16 | note | methods.tex:134 | sysVI input and likelihood missing in the main text |
| N-17 | note | methods.tex:53, :62, :142, :145; supplement.tex:298, :308, :309 | Several bullets carry 3-5 claims |
| N-18 | note | methods.tex:119 (conflict to report) | X12 resolution III correct; experiment matrix still says IV |
| N-19 | note | supplement.tex:262-273 (S-E5 sources) | Cite the per-row provenance as the version record |

### R-01 (blocker) - methods.tex:62; discussion.tex:17

**Problem.** 'the ATAC tasks share 3,618 cells' is an undercount stated as a verified data fact. 3,618 is the number of identical cell NAMES (the 10x Genomics batch). Every one of atac_small's 11,270 cells is also an atac_large cell: same batch label, same barcode, same cell-type label; the names differ only in concatenation suffixes. The two files quantify the same cells separately, so no count row is identical. ANALYSIS_SPEC sec. 1 and sec. 16 item 6 and final-review I073 carry the same name-based count (source conflict to report).

**Evidence (08f1e3c unless stated).** D/prepped_scib/atac_small__scib.h5ad vs atac_large__scib.h5ad, obs read with h5py: matching on (batch, barcode = obs name before the first '-') gives 11,270 of 11,270 atac_small cells (Fang et al. - CEMBA180305_2B 3,750; Cusanovich et al. - WholeBrainA_62216 3,902; 10x Genomics 3,618), cell-type agreement 1.000; identical obs names 3,618 (the 10x batch); the other two batches carry different name suffixes ('-1-0' in atac_small vs '-0-0-0-0-0-1-0' in atac_large for Fang cells). layers['counts'] on 200 random matched pairs, 3,150 shared features (3,429 vs 3,580): median row correlation 0.872 (min 0.794), against 0.188 for different cells of the same batch and cell type; 0 of 200 rows identical. D/review/lead/verified_issues.csv I073 evidence: 'Task overlap measured on prepped obs names ... 3,618 atac_small cells are in atac_large'.

**Replacement** (parts separated by `||`):

```latex
methods.tex:62 -> \item Families: pancreas; lung; immune (immune, immune\_hum\_mou); ATAC (atac\_small, atac\_large) \src{SI-73}. immune\_hum\_mou contains all 33,506 immune cells; every one of the 11,270 atac\_small cells is also in atac\_large (same batch, barcode and cell-type label; 3,618 of them, the 10x Genomics batch, under identical names), with gene-activity values computed separately in each file \src{\file{D/prepped_scib} obs and counts layers}. Simulations are not pooled with real tasks \src{PREREG \S5}.  ||  discussion.tex:17 -> \item Six real tasks in four families, two of them nested: immune\_hum\_mou contains all immune cells, and atac\_large contains all atac\_small cells, re-quantified \src{SI-73; \file{D/prepped_scib} obs and counts layers}.
```

### R-02 (blocker) - legend.tex:6

**Problem.** States that PREREG sections 0-8 and the section-9 entries of 2026-10-03/04 are frozen at prereg-tier12-v2, checked by 'git log 08f1e3c ^c88cce3 per entry: no later commit'. False in part: at 08f1e3c sec. 3 and sec. 5 carry bracketed version-3 amendments, and the 2026-10-03 entry carries an appended correction dated 2026-10-13. The check used (git log -S on entry titles) only sees changes in how often a title string occurs, not edits inside an entry. Consequence: a reader takes the sec. 5 note (Holm over six results, an agent proposal) and the sec. 3 note as frozen v2 text. Task item (f).

**Evidence (08f1e3c unless stated).** Byte comparison of git show c88cce3:docs/PREREG.md with git show 08f1e3c:docs/PREREG.md (tag prereg-tier12-v2 -> c88cce3): sec. 0, 1, 2, 4, 6, 7, 8 identical. sec. 3 adds '[Amended 2026-10-13: a cell of a task x decoder flagged bstar_low_signal is not evaluable ... SI-56, SI-66.]' (PREREG.md:75). sec. 5 adds '[Amended 2026-10-14: the frozen X1 analysis applies Holm over six primary results ... the agent's proposal ...]' (PREREG.md:106). Of the 12 section-9 entries dated 2026-10-03/04, 11 are byte-identical; the 2026-10-03 entry (SI-38) appends '[Corrected 2026-10-13: R1 does depend on trajectory ... (finding I091).]' (PREREG.md:223-224). The status line and the Implementation paragraph also changed; the Disclosure paragraph is unchanged.

**Replacement** (parts separated by `||`):

```latex
\item PREREG \S n: section n of \file{docs/PREREG.md}, the pre-registered decision rules. Frozen at tag prereg-tier12-v2 (commit c88cce3) and byte-identical at 08f1e3c: \S0--\S2, \S4, \S6--\S8, the Disclosure paragraph, and 11 of the 12 section 9 entries dated 2026-10-03/04 \src{\file{docs/PREREG.md} at c88cce3 and 08f1e3c, compared}. At 08f1e3c, \S3, \S5 and the 2026-10-03 entry also carry bracketed notes dated 2026-10-13/14 (version 3): the \S3 note and the 2026-10-03 correction implement SI-56, SI-66 and SI-53; the \S5 note (Holm over six results) is an agent proposal. The section 9 entries dated 2026-10-13 and 2026-10-14 are version 3, which is not frozen.
```

### R-03 (blocker) - methods.tex:145

**Problem.** 'no fit or scoring code changed up to 08f1e3c' is false under the project's own definition. scripts/run_stage.py is a fit entry point inside the identity proof's import closure (code rigour review of 2026-10-14, K14), and it changed in 8351999 (CF-02). The validity condition of docs/EQUIVALENT_COMMITS.json is met: src/, fit_paper_config.py, scvi_adversarial_plan.py, score_scib_native.py and their imports did not change, and run_stage.py is not among them. The carry-over of the c3fd92d proof is the agent's judgment in NB-20261014-29; the proof was not re-run. The line also omits the evidence's limits: 1 CPU epoch per code path, 1 thread, one latent per scorer path. Task item (a).

**Evidence (08f1e3c unless stated).** git log c3fd92d..08f1e3c -- src scripts tests cluster: src/ none; scripts/run_stage.py (8351999 CF-02, 12 lines), scripts/decide_a2_a3.py (f3c3415), scripts/analysis_x1.py (ff88666, 8b90d79), cluster/jhpce/local_rehearsal.sh and prod_fit_job.sh (0110b63, comments only), tests. docs/EQUIVALENT_COMMITS.json valid_for_a_candidate_commit_if: 'git diff c3fd92d <candidate> over src/, scripts/fit_paper_config.py, scripts/scvi_adversarial_plan.py, scripts/score_scib_native.py and the files they import shows no change'. fit_paper_config.py imports scvi_adversarial_plan, fit_outcome; score_scib_native.py imports host_info (grep). D/review/rigor_code_20261014/REVIEW_CODE_RIGOR_20261014.md CF-02: 'run_stage.py is inside the identity proof's import closure (claim 6): after a change there, re-run probes/import_closure.py ... or repeat the proof'. NB-20261014-29: 'fit entry points: only scripts/run_stage.py changed ... the c3fd92d proof carries over; it was not re-run'.

**Replacement** (parts separated by `||`):

```latex
\item Identity checks at c3fd92d (CPU, 1 epoch, 1 thread, one fit per code path, one latent per scorer path): 59 of 59 non-Sinkhorn fit code paths and 7 of 7 rescored latents (32 columns) equal c88cce3 \src{\file{docs/EQUIVALENT_COMMITS.json}; NB-20261013-39, -40}. From c3fd92d to 08f1e3c no file under \file{src/} and none of \file{fit_paper_config.py}, \file{scvi_adversarial_plan.py}, \file{score_scib_native.py} or their imports changed, which is the validity condition of \file{docs/EQUIVALENT_COMMITS.json}; the stage runner \file{run_stage.py}, which is in the fit import closure, changed its start-up provenance check (CF-02, 8351999), and the proof was taken to carry over without a re-run \src{git log c3fd92d..08f1e3c; NB-20261014-29 (agent judgment)}. \pending{acceptance of the carry-over; tag commit, with the checks repeated if fit or score code changes}{diff review, tag prereg-tier12-v3 (SI-52)}
```

### R-04 (blocker) - legend.tex:26; methods.tex:118

**Problem.** 'the pooled critic at 1, 5 and 10 critic steps, with or without spectral normalisation' describes six pooled configurations. The manifest has five: the pooled critic at 1, 5 and 10 steps, and the spectrally normalised pooled critic (pooled_sn) at 1 and 5 steps only. supplement.tex:167 (S-E1) states it correctly.

**Evidence (08f1e3c unless stated).** manifests/paper_manifest_stock_pilot_u5_b10_v3.tsv, X6 rows by (arm, n_critic): discriminator 1, discriminator_r1 1, discriminator_sn 1, pooled 1 / 5 / 10, pooled_sn 1 / 5; 27 rows each, 216 in all.

**Replacement** (parts separated by `||`):

```latex
legend.tex:26 -> X6 & Regularisation and update budget: the discriminator plain, spectrally normalised or with an R1 gradient penalty (Mescheder et al.; not rule R1); the pooled critic at 1, 5 and 10 critic steps, and spectrally normalised at 1 and 5 steps. \src{manifest; SI-22} \\  ||  methods.tex:118 -> \item X6: discriminator plain, spectrally normalised (\unverified{miyato2018spectral}) or with an R1 gradient penalty ($\gamma$ 10; \unverified{mescheder2018training}); pooled critic at 1, 5 and 10 steps, and spectrally normalised at 1 and 5 steps \src{SI-22; manifest}.
```

### R-05 (blocker) - methods.tex:9; methods.tex:129

**Problem.** 'within a task all arms and neural baselines share epochs, minibatch and latent size' is false as stated. (i) scANVI adds min(10, max(2, round(E/3))) epochs after its scVI pretraining, a declared difference frozen in v2 (CR-12). (ii) Epochs are computed from the number of cells fitted, so a task's subsampled X3 and X8 rows use different epochs from its X1 rows. (iii) X12 varies latent size and minibatch by design. Within one experiment and task, every adversarial arm, plain scVI and sysVI do share them.

**Evidence (08f1e3c unless stated).** Manifest max_epochs by task and experiment: immune_hum_mou X1 82, X3 400, X8 400; lung X1 246, X8 400; X12 n_latent 10/30, batch_size 128/512. fit_paper_config.py:401 (scANVI epochs); PREREG sec. 9, 2026-10-04 (CR-12): 'scANVI adds min(10, max(2, round(E/3))) epochs after its scVI pretraining' (byte-identical at c88cce3); build_paper_manifest.py:114-115 (heuristic of n).

**Replacement** (parts separated by `||`):

```latex
methods.tex:9 -> \item Epochs by the scvi-tools heuristic $\min(400, \mathrm{round}(20000 \times 400/n))$ on the $n$ cells fitted, no early stopping; within an experiment and task every arm, plain scVI and sysVI share epochs, minibatch and latent size; declared exceptions: scANVI's extra epochs, X12's factorial levels, and the subsample-sized epochs of X3 and X8 \src{SI-04; \file{build_paper_manifest.py}:114; \file{scvi_adversarial_plan.py}:563; \file{fit_paper_config.py}:401; PREREG \S9, 2026-10-04 (CR-12); manifest}.  ||  methods.tex:129 -> \item Plain scVI = X1 arm none \src{manifest}; scANVI and sysVI share the arms' epochs, minibatch and latent size, and scANVI adds $\min(10, \max(2, \mathrm{round}(E/3)))$ epochs after its scVI pretraining \src{SI-04; \file{fit_paper_config.py}:401; PREREG \S9, 2026-10-04 (CR-12); manifest}.
```

### R-06 (blocker) - methods.tex:106; supplement.tex:95

**Problem.** 'is descriptive' is attributed to PREREG sec. 5, which does not say it. The word comes from ANALYSIS_SPEC sec. 4 and the 2026-10-13/14 amendments of PAPER_PLAN sec. 5, which are agent text, not user decisions. The 0.0125 threshold is unexplained: it is Holm's first threshold for the four tests P1-P4 of PREREG sec. 5. The competing six-test family (0.05/6 = 0.0083, ANALYSIS_SPEC sec. 5, a proposal) is not marked pending. Task item (h).

**Evidence (08f1e3c unless stated).** PREREG.md:106 (v2 sentence, unchanged): 'the frequentist companion (exact Wilcoxon on dataset means, Holm over P1-P4) cannot reject with 6 real tasks. The smallest two-sided p is 2/2^6 = 0.031, above Holm's first threshold 0.05/4 = 0.0125.' docs/prereg/r1_record.json power: holm_first_threshold 0.0125, wilcoxon_min_p_two_sided 0.03125, frequentist_can_reject false. ANALYSIS_SPEC.md:147-148 ('Frequentist companion: exact two-sided Wilcoxon signed-rank test on the d_t, descriptive') and :163-169 (multiplicity: 'the agent's proposal').

**Replacement** (parts separated by `||`):

```latex
methods.tex:106 -> \item Frequentist companion: exact two-sided Wilcoxon signed-rank test on the per-task differences, with Holm's correction over the four primary endpoints P1--P4; with 6 real tasks its smallest p, $2/2^6 = 0.031$, is above Holm's first threshold $0.05/4 = 0.0125$, so it cannot reject at the Holm-adjusted level \src{PREREG \S5; \file{docs/prereg/r1_record.json} power}. \pending{Holm over six primary results (first threshold 0.0083) and reporting the companion as descriptive}{diff review (ANALYSIS\_SPEC \S4, \S5)}  ||  supplement.tex:95 -> \item Frequentist companion: exact two-sided Wilcoxon signed-rank test on the $d_t$; smallest attainable p with 6 tasks $2/2^6 = 0.03125$, above Holm's first threshold 0.0125 for the four tests P1--P4 \src{PREREG \S5; \file{docs/prereg/r1_record.json} power}; \pending{Holm family of six (0.05/6 = 0.0083) and descriptive use}{diff review (ANALYSIS\_SPEC \S4, \S5, \S15 item 5)}.
```

### R-07 (fix) - rq.tex:3

**Problem.** 'All six are open: no X1 result has been examined.' is stated as fact with no source. Its basis is open: the definition of 'examined' is a proposal, and the user has not yet confirmed that nobody read an X1 adversarial-arm score. supplement.tex:309 marks the same item pending. lock_audit.csv row 'rq.tex | R0 status' calls it locked while noting that the confirmation is open.

**Evidence (08f1e3c unless stated).** PREREG sec. 9, entry of 2026-10-14 (disclosure): 19 X1 rows scored; 'The audit cannot show that nobody opened a score file outside those chat records: open item for the diff review'; 'Proposed definition ... (not yet approved)'. ANALYSIS_SPEC App. A.3 item 2. NB-20261012-05: 19 scored, 1 fitted not scored, 1,097 need fit.

**Replacement** (parts separated by `||`):

```latex
Each block gives the question, the hypothesis where it is locked, and the status. All six are open: 19 of the 3,000 X1 rows are scored (local run paused on 2026-10-12) and the X1 analysis has not been run \src{NB-20261012-05; NB-20261014-21}. \pending{that no X1 result has been examined (an audit of the chat records found none displayed or analysed)}{the user's confirmation and the definition of ``examined'' at the diff review (ANALYSIS\_SPEC App.~A.3, item 2)}
```

### R-08 (fix) - legend.tex:27; legend.tex:28

**Problem.** The X7 and X8 glossary entries present the designs as locked (plain text with a source). The shared definitions mark the X7/X8 designs NOT LOCKED: the final review rated them NO-GO as designed, and they are redesigned during X1 (SI-68). rq.tex and methods.tex already mark them pending, so the legend is inconsistent with them.

**Evidence (08f1e3c unless stated).** D/review/rereview/design_fable/REVIEW_design_recheck.md:11 ('NO-GO as designed until D-06 / D-07 are resolved'); SI-68 ('together with the X7/X8 redesign'); ANALYSIS_SPEC.md:36-37. Manifest facts checked: X7 = 18 references (pancreas 9, sim1 6, atac_small 3) x 4 arms x 3 seeds = 216; X8 n_cells per task immune_hum_mou 16,000, lung 6,000, sim2 3,000 at V = 2, 4, 8, 16.

**Replacement** (parts separated by `||`):

```latex
legend.tex:27 -> X7 & Reference choice, as designed in the manifest of record: every batch in turn as the reference (atac\_small, pancreas, sim1), for the reference critic, a fixed-reference critic, reference-anchored MMD and reference-anchored JS \src{manifest; SI-23}; \pending{redesign (NO-GO as designed: D-06, I089)}{during X1, before any X1 result is examined (SI-68)} \\  ||  legend.tex:28 -> X8 & Batch-count scaling, as designed in the manifest of record: V = 2, 4, 8 or 16 batches at a fixed total number of cells per task (immune\_hum\_mou 16,000, lung 6,000, sim2 3,000) \src{manifest}; \pending{redesign (NO-GO as designed: D-07, I015, I077, I078)}{during X1, before any X1 result is examined (SI-68)} \\
```

### R-09 (fix) - legend.tex:35

**Problem.** The P3 and P4 definitions are design-time PAPER_PLAN sec. 3 text shown as locked. SI-68 locks P3 and P4 later, with the X7/X8 redesign, and I089 says the P4 wording ('interaction of formulation and reference coverage') needs a design change before it can be claimed. rq.tex:24 and :39 mark the same wording pending.

**Evidence (08f1e3c unless stated).** SI-68 rule text ('Lock later ...: P3, P4 and the other follow-up analyses, together with the X7/X8 redesign'); PAPER_PLAN.md:98, :100 (P3, P4); verified_issues.csv I089 ('Word P4 as an association ...').

**Replacement** (parts separated by `||`):

```latex
P1 to P4 & Primary endpoints. P1: biology score at the matched batch score b*, reference critic minus discriminator, one result per decoder, with its trade-off area \src{SI-68; SI-70; SI-72}. P2: how often runs collapse \src{SI-75}, with the X3 dose-response specified later \src{SI-68}. P3 (batch-count scaling, X8) and P4 (reference choice, X7): \pending{definitions (PAPER\_PLAN \S3 proposes the change of the critic-minus-discriminator difference with batch count, and a formulation-by-coverage interaction)}{X7/X8 redesign, during X1 (SI-68)} \\
```

### R-10 (fix) - rq.tex:23; rq.tex:38 (also rq.tex:7, :30, :45)

**Problem.** The research-question wording is sourced only to PAPER_PLAN sec. 3, a design-time document, and shown as locked. For R2 and R4 the wording depends on the X8 and X7 redesigns. The R4 question is causal ('through its cell-type coverage'), which I089 says X7 as designed cannot support. For R0, R3 and R5 the topic is covered by SI-68 (ANALYSIS_SPEC sec. 0); add that source.

**Evidence (08f1e3c unless stated).** PAPER_PLAN.md:96-101 (R0-R5 rows, design-time); SI-68; ANALYSIS_SPEC.md:31-43 (sec. 0 table: P3 batch-count scaling, P4 reference choice, R0 premise, R3 window/factor effects, R5 placement); verified_issues.csv I089.

**Replacement** (parts separated by `||`):

```latex
rq.tex:23 -> \item Question: does the critic-minus-discriminator difference change with the number of batches? \src{topic: SI-68 (P3, batch-count scaling; ANALYSIS\_SPEC \S0)} \pending{exact wording (PAPER\_PLAN \S3, R2 row)}{X8 redesign, during X1 (SI-68)}  ||  rq.tex:38 -> \item Question: does the choice of reference batch matter more for the reference W1 critic than for reference-anchored JS or MMD with the same target? \src{topic: SI-68 (P4, reference choice; ANALYSIS\_SPEC \S0)} \pending{whether the question names cell-type coverage as the route (PAPER\_PLAN \S3, R4 row; I089: X7 as designed supports an association only)}{X7 redesign, during X1 (SI-68)}  ||  rq.tex:7, :30, :45 -> add 'SI-68; ANALYSIS\_SPEC \S0' to each \src{...}.
```

### R-11 (fix) - methods.tex:43; supplement.tex:202-203

**Problem.** The R2/R3 outcome for X1 is marked pending, but locked rules and a frozen record already fix it. SI-66 says that while a task x decoder block is flagged low-signal, criterion 1 fails and R2/R3 keep the defaults. The frozen rule-R4 record on A1 flags immune with the conditioned decoder. b_common is the minimum over arms, and Sinkhorn sets it in no block, so refitting the Sinkhorn arm can only lower b_common. The first run (NB-20261012-01) also kept both defaults. Only the formal re-run (SI-57) is pending.

**Evidence (08f1e3c unless stated).** docs/prereg/r4_a1_record.json targets: immune|1 b0 0.6382, b_common 0.6570 (barycenter), difference 0.0188, bstar_low_signal true; sim1|1 difference 0.0069 (set by mmd); b_common set by barycenter in 5 blocks and mmd in 1, never by sinkhorn. SI-66; SI-57 (non-Sinkhorn A1 fits and lambda = 0 fits are not refitted). NB-20261012-01: 'R2 ... keep mean; R3 ... keep off'.

**Replacement** (parts separated by `||`):

```latex
methods.tex:43 -> \item X1 keeps the defaults (posterior mean; standardisation off): the frozen rule-R4 record on A1 flags immune with the conditioned decoder as low-signal ($b_{\mathrm{common}} - b_0 = 0.0188$), so rules R2 and R3 cannot switch while that flag holds \src{SI-66; \file{docs/prereg/r4_a1_record.json}; NB-20261012-01}; Sinkhorn sets $b_{\mathrm{common}}$ in no block, so refitting it alone can only lower $b_{\mathrm{common}}$ \src{\file{docs/prereg/r4_a1_record.json} targets}. \pending{formal re-run of rules R2 and R3}{after the Sinkhorn redo (SI-57)}  ||  supplement.tex:202-203 -> 'posterior mean (rules R2/R3 cannot switch while immune, conditioned, is flagged low-signal; methods); \pending{formal rule R2 re-run}{after the Sinkhorn redo}' and likewise for standardisation.
```

### R-12 (fix) - methods.tex:109

**Problem.** 'underpowered-design statement and its figures' is marked pending, but the frozen v2 record already holds the power report defined by PREREG sec. 5 (v2): UNDERPOWERED, with its figures. These are verified facts. Only the recomputed estimate (P1 reading, evaluable blocks only) is pending.

**Evidence (08f1e3c unless stated).** docs/prereg/r1_record.json (status frozen, git_sha c88cce3) power: underpowered true; real_tasks D=6 se 0.0450, mde80 0.1307, power_at_delta 0.1044; families D=4 mde80 0.1818, power 0.0878; equivalence at 0.01 and 0.02 infeasible; sigma 0.0298 (12 df), tau 0.1094 (4 df); blocks atac_small|0/1, immune|0/1, sim1|0/1. PREREG sec. 5 (v2) defines the report and the flag.

**Replacement** (parts separated by `||`):

```latex
\item Power (PREREG \S5 rule; frozen record): the design is flagged UNDERPOWERED; for D = 6 real tasks SE 0.045, minimal detectable effect at 80\% power 0.131 and power at $\Delta = 0.02$ 0.104 (D = 4 families: 0.182 and 0.088); equivalence within $\pm 0.01$ or $\pm 0.02$ is unreachable; computed for the matched-strength contrast on the six A1 blocks, sim1 and the two low-signal blocks included \src{PREREG \S5; \file{docs/prereg/r1_record.json} power}. \pending{recomputed estimate with the P1 reading and evaluable blocks only}{rule R4 on A1 re-run after the Sinkhorn redo (SI-57)}
```

### R-13 (fix) - discussion.tex:18

**Problem.** 'The Wilcoxon companion cannot reject with 6 tasks' needs the qualifier 'at the Holm-adjusted level': unadjusted, the smallest p (0.031) is below 0.05. The power figures are frozen-record facts, not pending (see R-12).

**Evidence (08f1e3c unless stated).** PREREG.md:106 (v2 sentence and the 2026-10-14 bracket: 'unadjusted, 0.031 is below 0.05'); docs/prereg/r1_record.json power.

**Replacement** (parts separated by `||`):

```latex
\item The design is underpowered (minimal detectable effect 0.131 for 6 real tasks at 80\% power), and the Wilcoxon companion cannot reject at the Holm-adjusted level (smallest p 0.031 $>$ 0.0125) \src{PREREG \S5; \file{docs/prereg/r1_record.json} power}. \pending{recomputed power and minimal detectable effect}{rule R4 on A1 re-run (SI-57)}
```

### R-14 (fix) - methods.tex:53

**Problem.** 'its 234 pilot fits are redone in rule order' treats 234 as a fixed list. SI-57 narrows SI-51: the 180 A1 Sinkhorn fits are refitted, and rules R1 and R4 on the new fits decide the extension and A3 rows. 234 was the count before the fix. supplement.tex:308 states this correctly.

**Evidence (08f1e3c unless stated).** SI-57; NB-20261014-34 ('SI-51 is narrowed by SI-57: the Sinkhorn R1-extension and A3 refits (18 and 36 before the fix) are outputs of R1 and R4-a1 on the refitted A1 fits, not a fixed list'); PREREG sec. 9 Sinkhorn entry ('the numbers may change').

**Replacement** (parts separated by `||`):

```latex
\item Sinkhorn corrected before X1 (gradient counted once; solve converged at the final smoothing step) \src{SI-51}; its 180 A1 fits are refitted, then rules R1 and R4 on A1 decide which extension and A3 rows to refit (18 and 36 before the fix, 234 fits in all) \src{SI-51; SI-57}. Solve in float64, at most 500,000 final-step iterations with an error at the cap, immediate stop on a NaN marginal error \src{SI-61; SI-62}. \pending{tolerance ($10^{-4}$ L1 marginal error, \file{wcd/alignment.py}:56) and the outcome of a fit that keeps reaching the cap}{diff review, tag prereg-tier12-v3}
```

### R-15 (fix) - supplement.tex:273; supplement.tex:255

**Problem.** Task item (b). The Harmony version (harmony-pytorch 0.1.7) is not in any recorded environment file. The committed wcd-kbet freeze lists harmonypy 2.0.0, which the runner does not import. The source given is a code docstring plus the live env, so the committed environment record cannot rebuild the Harmony baseline. The rebuild sentence cites only the decision SI-48; the evidence that the rebuild matched is NB-20261004-20. All other S-E5 versions agree with the recorded files and with the per-row provenance of the A1 records.

**Evidence (08f1e3c unless stated).** cluster/envspec/local_wcd-kbet_pip_freeze.txt:34 harmonypy==2.0.0 (no harmony-pytorch); cluster/envspec/wcd-kbet.explicit.txt:211 harmonypy-2.0.0; env wcd-kbet site-packages holds harmony_pytorch-0.1.7.dist-info and harmonypy-2.0.0.dist-info; run_cpu_baselines.py:76 'from harmony import harmonize', :142 records md.version('harmony-pytorch') per latent. NB-20261004-20: 'pip freeze 90 identical + 34 declared CUDA substitutions'. Agreement checks: D/tier12_v2/A1 score rows scorer_versions {scib 1.1.7, R_kBET 0.99.6, scanpy 1.11.5, anndata 0.11.4, numpy 1.26.4}; A1 latent configs {scvi 1.4.2, torch 2.13.0+cu130, git_sha c88cce3}; versions_score_local.json R 4.5.2 (2025-10-31), kBET 0.99.6; versions_fit_gpu.json torch 2.13.0+cu126.

**Replacement** (parts separated by `||`):

```latex
supplement.tex:273 -> & Harmony (harmony-pytorch) & 0.1.7 in env wcd-kbet; missing from the committed freeze, which lists harmonypy 2.0.0 (not used) & \src{\file{run_cpu_baselines.py}:76, :142 (version written into each latent); \file{cluster/envspec/local_wcd-}\file{kbet_pip_freeze.txt}:34} \\  ||  supplement.tex:255 (last sentence) -> The JHPCE fitting environment was rebuilt with conda packages and pip versions equal to the local one, apart from 34 declared CUDA builds \src{SI-48; NB-20261004-20}; each latent records its scvi-tools and PyTorch versions and each score row its scorer versions \src{\file{fit_paper_config.py}:353--355; \file{score_scib_native.py}:38--41}.
```

### R-16 (fix) - supplement.tex:298; supplement.tex:299

**Problem.** The audit files are cited as \file{original_manuscript_audit.md} and \file{original_manuscript_claims_audit.csv}, but neither is in the repository at 08f1e3c or in D/. They are project files linked from PAPER_PLAN's Sources line, so the legend's file convention (repository file at 08f1e3c, or D/) cannot resolve them. The content is correct: D1-D9 match audit sec. 3 item 1, the claim counts 11/6/5/3 of 25 match the claims table, and '6 of 36 survive Holm' matches audit sec. 3 item 24.

**Evidence (08f1e3c unless stated).** git ls-tree -r --name-only 08f1e3c: no match for 'original_manuscript' or 'claims_audit'; find D (depth 4): none; docs/PAPER_PLAN.md:7 (Sources line links both files); project files original_manuscript_audit.md (version 0c9e00ea) and original_manuscript_claims_audit.csv (version 6c946526) read: sec. 3 item 1 'DEFECT LEGEND' D1-D9; status_now counts invalidated_by_bugs 11, untested 6, partially_supported 5, contradicted 3.

**Replacement** (parts separated by `||`):

```latex
supplement.tex:298 source bracket -> \src{PAPER\_PLAN \S1; the audit of the original submission named in PAPER\_PLAN's Sources (original\_manuscript\_audit.md, \S3 items 1 and 24; a project file outside the repository)}  ||  supplement.tex:299 source bracket -> \src{PAPER\_PLAN \S1; original\_manuscript\_claims\_audit.csv (project file outside the repository)}  ||  recommended: commit both files (e.g. docs/original_submission/) so \file{} paths resolve.
```

### R-17 (fix) - methods.tex:59

**Problem.** 'The eight standard scIB tasks' reads as if scIB defines eight tasks. The scIB benchmark has 13 integration tasks: 2 simulation, 5 scRNA-seq and 6 scATAC-seq. The project uses 8 of them, with its two ATAC tasks as gene-activity matrices.

**Evidence (08f1e3c unless stated).** Luecken et al. 2022, Nat Methods, doi 10.1038/s41592-021-01336-8 (abstract and Results first paragraph, retrieved by web search): '13 preprocessed integration tasks: two simulation tasks, five scRNA-seq tasks and six scATAC-seq tasks'. build_paper_manifest.py:54 (TASKS: the 8 used); docs/prepped_fingerprints_scib.json (ATAC source files *_atac_gene_activity.h5ad).

**Replacement** (parts separated by `||`):

```latex
\item Eight of the scIB benchmark's integration tasks (\unverified{luecken2022benchmarking}), no new datasets: six real (pancreas, lung, immune, immune\_hum\_mou, atac\_small, atac\_large; the ATAC tasks as gene-activity matrices) and two simulations (sim1, sim2) \src{SI-02; \file{build_paper_manifest.py}:54; \file{docs/prepped_fingerprints_scib.json}}; scIB's batch and label keys \src{SI-03}.
```

### R-18 (fix) - supplement.tex:56 (S-M9 importance weights)

**Problem.** The weight definition of SI-25, a locked user decision, is missing: the target is each batch's composition before depletion; w = 1/keep fraction for cells of the depleted types in the depleted batch and 1 otherwise; doses 50, 80 and 95% only. Only its realisation from the nested draws (SI-42, the lead's decision) is unconfirmed.

**Evidence (08f1e3c unless stated).** CONSTRAINTS.md SI-25 (rule text); SI-42 (source: NB-20261003-22 DECISION (lead)); PREREG sec. 9, 2026-10-04 X3 entry (byte-identical v2): 'w[b, y] = n_K0(b, y) / n_Kd(b, y) ... equal SI-25's formula when nothing is refilled'.

**Replacement** (parts separated by `||`):

```latex
\item Importance weights (X3, doses 50, 80 and 95\% only): target = each batch's composition before depletion; $w = 1/(\text{keep fraction})$ for cells of the depleted types in the depleted batch and 1 otherwise; every per-batch mean in the adversary and generator terms becomes the self-normalised mean $\sum_i w_i f(z_i)/\sum_i w_i$, with the same weights in discriminator, reference and pooled critics and MMD \src{SI-25}; the gradient penalty is unweighted \src{\file{wcd/adversarial.py}:91}. \pending{as implemented, $w[b, y] = n_{K_0}(b, y)/n_{K_d}(b, y)$ from the realised nested draws}{the user's confirmation of SI-42 (a lead-agent decision)}
```

### R-19 (fix) - methods.tex:117 (or supplement S-E1 note)

**Problem.** Locked X3 design facts are missing: the dose-100 cell counts of SI-41 and the depleted batch and cell types of each task (SI-32 and SI-33 for sim2 and atac_small; the manifest of record for pancreas and immune_hum_mou).

**Evidence (08f1e3c unless stated).** SI-41 (atac_small 8,402, pancreas 14,409, sim2 17,872, immune_hum_mou 20,000); manifest X3 extra.subsample: atac_small 'Fang et al. - CEMBA180305_2B' [Excitatory Neurons, Inhibitory Neurons] n_cells 8402; pancreas inDrop3 [alpha, acinar] 14409; immune_hum_mou MCA_BM_2 [Neutrophils, Monocyte progenitors] 20000; sim2 Batch3Sub1 [Group1] 17872; doses 0/50/80/95/100; iw 'depletion_oracle' on 288 rows.

**Replacement** (parts separated by `||`):

```latex
add after methods.tex:117 (or as an S-E1 note) -> \item X3 targets, cells per dose: atac\_small, Fang et al. - CEMBA180305\_2B, excitatory and inhibitory neurons, 8,402 \src{SI-33; SI-41}; pancreas, inDrop3, alpha and acinar cells, 14,409 \src{SI-41; manifest}; immune\_hum\_mou, MCA\_BM\_2, neutrophils and monocyte progenitors, 20,000 \src{SI-41; manifest}; sim2, Batch3Sub1, Group1, 17,872 \src{SI-32; SI-41}.
```

### R-20 (fix) - supplement.tex:75 (S-M12)

**Problem.** PREREG sec. 4 (v2) also has subsampled tasks (X3, X8) and architecture variants (X12) inherit the full task's matched strength; this rule is missing. The design-recheck caveat that inherited strengths do not put variants at matched batch removal (D-06, affecting X6, X7 and X12) is also missing.

**Evidence (08f1e3c unless stated).** PREREG.md:98 ('Subsampled tasks (X3, X8) and architecture variants (X12) inherit the full task's value. An arm without a mapping stops the script.'); REVIEW_design_recheck.md:39 (D-06: 'fix before follow-up launch (X7, X12, X6)').

**Replacement** (parts separated by `||`):

```latex
\item Variant arms inherit the matched strength of their base arm (discriminator\_sn, discriminator\_r1 and discriminator\_ref from the discriminator; reference\_fixed from the reference critic; pooled\_sn and the pooled critic at other step counts from the pooled critic; mmd\_ref from MMD); subsampled tasks (X3, X8) and architecture variants (X12) inherit the full task's value \src{PREREG \S4; \file{prereg_rules.py}:59}. An inherited strength does not put a variant at matched batch removal \src{D-06}; the follow-up analyses are locked later \src{SI-68}.
```

### R-21 (fix) - methods.tex:88

**Problem.** The two limits of the committed equivalence declaration are missing. Score rows carry the scorer's commit, not the fit's. Sinkhorn rows are refused only at c88cce3, so a stage made entirely at another commit is not refused. These limits are why every stage must be fitted and scored from one worktree pinned at the tag.

**Evidence (08f1e3c unless stated).** docs/EQUIVALENT_COMMITS.json 'meaning' ('Limits: score rows carry the scorer's commit, not the fit's ...; and not_for_arms rows are refused only at commit, so a stage made entirely at one other commit is not refused by this list'); PREREG sec. 9, 2026-10-04 entry (b) (v2: one worktree pinned at the tag).

**Replacement** (parts separated by `||`):

```latex
append to methods.tex:88 -> ; limits of the check: score rows carry the scorer's commit, not the fit's, and Sinkhorn rows are refused only at c88cce3, so each stage is fitted and scored from one worktree pinned at the tag \src{\file{docs/EQUIVALENT_COMMITS.json} meaning; PREREG \S9, 2026-10-04 (b)}
```

### R-22 (fix) - methods.tex:73

**Problem.** A frozen v2 reporting rule is missing: scib's silent fallbacks are flagged in every score row, and their counts per rule cell are reported with the results.

**Evidence (08f1e3c unless stated).** PREREG sec. 9, 2026-10-04 'scorer provenance and scib fallback flags (CR-05, CR-08)' (byte-identical v2): 'flags trajectory's root-cell fallback (value 0) and kBET labels forced to 1 or skipped ... their counts per rule cell are reported with the results; the rule scripts do not read them'.

**Replacement** (parts separated by `||`):

```latex
append to methods.tex:73 -> ; scib's silent fallbacks (trajectory root cell set to 0; kBET labels forced to 1 or skipped) are flagged in every score row and their counts per rule cell are reported with the results; no rule reads them \src{PREREG \S9, 2026-10-04 (CR-05, CR-08)}
```

### N-01 (note) - legend.tex:5

**Problem.** SI-17 contains a sentence marked as an agent-derived corollary ('on JHPCE each task runs on one pinned GPU model'), which is not the user's.

**Evidence (08f1e3c unless stated).** CONSTRAINTS.md SI-17 rule text: '(Agent-derived corollary, not a user statement: ...)'; programmatic check: SI-42 is the only entry whose source is not the user; 77 entries; SI-35, SI-59 retired.

**Replacement** (parts separated by `||`):

```latex
... (77 entries). Each quotes the user's answer, except SI-42, a lead-agent decision that is not locked, and the sentence of SI-17 marked as an agent-derived corollary. SI-35 and SI-59 are retired (replaced by SI-46 and SI-61).
```

### N-02 (note) - legend.tex:50

**Problem.** The family mapping is at prereg_rules.py:67-68 (TASK_FAMILY); line 66 is REAL_TASKS.

**Evidence (08f1e3c unless stated).** scripts/prereg_rules.py:66-68 at 08f1e3c.

**Replacement** (parts separated by `||`):

```latex
\src{SI-02; SI-73; \file{prereg_rules.py}:66--68}
```

### N-03 (note) - legend.tex:51

**Problem.** X15's seeds come from SI-44 (SI-26 does not list X15). The CPU baselines run once, at seed 0.

**Evidence (08f1e3c unless stated).** SI-26 (lists X3, X6, X7, X8, X12); SI-44 ('seeds 10-12'); manifest X13 harmony/scanorama/pca seed 0; SI-34, SI-46.

**Replacement** (parts separated by `||`):

```latex
seeds & A1--A3: 100--102; X1 and the neural X13 baselines (scANVI, sysVI): 0--4, Harmony, Scanorama and PCA once at seed 0; follow-ups X3, X6, X7, X8, X12, X15: 10--12. The three sets are disjoint. \src{SI-26; SI-44; SI-34; SI-46; manifest} \\
```

### N-04 (note) - legend.tex:47

**Problem.** The variant arms used in S-E1, S-M9 and S-M12 have no glossary entry.

**Evidence (08f1e3c unless stated).** scripts/scvi_adversarial_plan.py:12-24 (arm list); manifest arms of X6 and X7.

**Replacement** (parts separated by `||`):

```latex
add row -> variant arms & Follow-up versions of an arm: discriminator\_sn and pooled\_sn (spectrally normalised), discriminator\_r1 (R1 gradient penalty), reference\_fixed (reference cells held fixed), discriminator\_ref and mmd\_ref (reference-anchored JS and MMD) \src{manifest; \file{scvi_adversarial_plan.py}:12--24; SI-22; SI-23} \\
```

### N-05 (note) - legend.tex (glossary); methods.tex:49-53, :68-69, :80, :142; supplement S-M4

**Problem.** For the stated reader, these labels and acronyms are used without a definition: KL (Kullback-Leibler) warm-up, WGAN-GP, GP, POT, JHPCE, CR-nn, Tier 1/Tier 2, PCR, ASW, iLISI/cLISI, kBET, NMI, ARI, F1. 'R1 penalty' (Mescheder et al.) is a third meaning of R1, beside rule R1 and results section R1.

**Evidence (08f1e3c unless stated).** grep of the six fragments; NO-JARGON / SI-67.

**Replacement** (parts separated by `||`):

```latex
add glossary rows -> CR-nn & Finding nn of the code review of tag prereg-tier12-v1 (2026-10-03). \\ KL warm-up & The weight of the Kullback--Leibler term of the scVI loss rises linearly from 0 over the first 400 epochs. \\ Tier 1, Tier 2 & The design-time scope tiers; Tier 2 adds X3, X6 and X12 \src{SI-05; PAPER\_PLAN \S4}. \\  ||  instruction -> spell out at first use: PCR (principal-component regression), ASW (average silhouette width), iLISI and cLISI (integration and cell-type local inverse Simpson's index), kBET (k-nearest-neighbour batch-effect test), NMI (normalised mutual information), ARI (adjusted Rand index), WGAN-GP (Wasserstein GAN with gradient penalty), POT (Python Optimal Transport), JHPCE (the cluster); call the X6 term 'the R1 gradient penalty of Mescheder et al. (unrelated to rule R1)'.
```

### N-06 (note) - methods.tex:33

**Problem.** 'Discriminator and critics share one head' can be read as shared weights. They share one architecture, and every fit has its own head.

**Evidence (08f1e3c unless stated).** wcd/adversarial.py:23-52 (one class; instantiated per plan at scvi_adversarial_plan.py:236).

**Replacement** (parts separated by `||`):

```latex
\item Discriminator and critics use the same head architecture (latent, 128, 128, one output per batch; ReLU), one head per fit \src{\file{wcd/adversarial.py}:50, :76}; ...
```

### N-07 (note) - methods.tex:29; supplement.tex:27; supplement.tex:198

**Problem.** The barycenter runs at most 10 fixed-point iterations, with POT's stop threshold 1e-7, not exactly 10. SI-14's words are '10 fixed-point iterations'.

**Evidence (08f1e3c unless stated).** wcd/barycenter.py:72-73 (numItermax=int(n_iter), stopThr=stop_thr, stop_thr=1e-7 at :38).

**Replacement** (parts separated by `||`):

```latex
'... 10 iterations' -> '... at most 10 fixed-point iterations (POT stop threshold $10^{-7}$)'
```

### N-08 (note) - methods.tex:49; supplement.tex:194

**Problem.** The discriminator's Adam also has weight decay 1e-6; the frozen v2 text states it.

**Evidence (08f1e3c unless stated).** scvi_adversarial_plan.py:260 (weight_decay=self.weight_decay); PREREG sec. 9, 2026-10-04 CR-12 ('lr 1e-3, eps 0.01, weight decay 1e-6').

**Replacement** (parts separated by `||`):

```latex
Discriminator: Adam $10^{-3}$, $\epsilon$ 0.01, weight decay $10^{-6}$, 1 step
```

### N-09 (note) - methods.tex:60; discussion.tex:15

**Problem.** The 18 non-integer batches are 18 task-batch pairs and 17 distinct batches: the Villani batch is the same 1,022 cells in both immune tasks. S-D2 says so; the main text does not. Task item (d): counts verified.

**Evidence (08f1e3c unless stated).** Recomputed from D/prepped_scib layers['counts'] (any non-integer value per batch): pancreas 4 (celseq, celseq2, fluidigmc1, smarter; 5,419 cells), lung 12 (22,771), immune 1 (Villani, 1,022), immune_hum_mou 1 (Villani, 1,022), others 0; equals docs/prepped_fingerprints_scib.json count_status_as_distributed.

**Replacement** (parts separated by `||`):

```latex
18 batches in 4 tasks (17 distinct; the Villani batch is in both immune tasks) hold non-integer values
```

### N-10 (note) - supplement.tex:60; supplement.tex:63

**Problem.** S-M10 omits two clauses of PREREG sec. 2: sigma uses only grid points with at least 2 paired seeds; and with fewer than 6 points, padding starts above and skips a side that has run off the A1 grid.

**Evidence (08f1e3c unless stated).** PREREG.md:54, :65; prereg_rules.py:393-403, :448-455.

**Replacement** (parts separated by `||`):

```latex
'median over grid points (with at least 2 paired seeds) of ...'; '... plus points added alternately above and below, starting above and skipping a side that has run off the A1 grid'
```

### N-11 (note) - methods.tex:87

**Problem.** PREREG sec. 6 says two different things. The rules, prereg_rules.py, the three rule scripts and the manifest builder were tagged before the first A1 fit. Each decision record and resolved manifest is committed (not tagged) before the stage it governs.

**Evidence (08f1e3c unless stated).** PREREG.md:110-112 (sec. 6, v2).

**Replacement** (parts separated by `||`):

```latex
\item Freeze: this file, the rule code and the manifest builder are tagged before the first A1 fit; each rule writes a record with input hashes and commit, committed with its resolved manifest before the stage it governs \src{PREREG \S6}. ...
```

### N-12 (note) - rq.tex:15

**Problem.** 'two-sided' rests on SI-74, which declares a direction either way, or practical equivalence. SI-74 is not cited.

**Evidence (08f1e3c unless stated).** SI-74 rule text; PAPER_PLAN.md:94 ('Hypothesis (two-sided)').

**Replacement** (parts separated by `||`):

```latex
... tested once per decoder as P1, with its trade-off area \src{SI-01; SI-68; SI-70; SI-72; SI-74}.
```

### N-13 (note) - rq.tex:17; results.tex:15-16

**Problem.** P2 rests on SI-75, which ANALYSIS_SPEC App. A.3 item 1 questions after the recount. methods.tex:103 carries that pending tag; rq and results do not.

**Evidence (08f1e3c unless stated).** ANALYSIS_SPEC.md:488-490.

**Replacement** (parts separated by `||`):

```latex
append -> \pending{whether SI-75 stands after the pilot recount (8 of 54, not 8 of 45)}{diff review}
```

### N-14 (note) - methods.tex:25; legend.tex:48

**Problem.** 'Each arm adds lambda x (alignment term)' does not hold for scvi_adv. Its manifest lambda is 0, and its classifier loss is scaled by 1 - KL weight. A reader of the glossary could take lambda = 0 rows to be plain scVI.

**Evidence (08f1e3c unless stated).** manifest X1 scvi_adv lam 0 (40 rows); scvi_adversarial_plan.py:175-177 (scale_adversarial_loss 'auto').

**Replacement** (parts separated by `||`):

```latex
methods.tex:25 -> \item Each of the six adversarial arms adds ... ; scvi\_adv instead scales its classifier loss by $1 -$ KL weight (manifest $\lambda$ 0) \src{\file{scvi_adversarial_plan.py}:177; manifest}.
```

### N-15 (note) - legend.tex:30; methods.tex:119

**Problem.** 'decoder type' (an X12 factor) is scVI's default decoder versus LinearSCVI's linear decoder. It is not the conditioned/unconditioned decoder of the glossary.

**Evidence (08f1e3c unless stated).** manifest X12 decoder SCVI/LinearSCVI; scvi_adversarial_plan.py:516-519.

**Replacement** (parts separated by `||`):

```latex
'... minibatch size and decoder (scVI's default or LinearSCVI's linear decoder)'
```

### N-16 (note) - methods.tex:134

**Problem.** The main text does not state that sysVI is fitted on the file's normalised values with a Gaussian likelihood; only S-E2 does.

**Evidence (08f1e3c unless stated).** fit_paper_config.py:404-413.

**Replacement** (parts separated by `||`):

```latex
\item sysVI (\unverified{hrovatin2025integrating}): scvi-tools' SysVI on the file's normalised values (Gaussian likelihood), cycle weight ... \src{\file{fit_paper_config.py}:404, :413; manifest; PREREG \S9, 2026-10-04 (CR-12)}.
```

### N-17 (note) - methods.tex:53, :62, :142, :145; supplement.tex:298, :308, :309

**Problem.** These bullets carry 3 to 5 claims each, against the accepted one-claim-per-line outline style.

**Evidence (08f1e3c unless stated).** Fragment text.

**Replacement** (parts separated by `||`):

```latex
Split each into one bullet per claim; keep the source bracket with its claim.
```

### N-18 (note) - methods.tex:119 (conflict to report)

**Problem.** Task item (g). The fragment's 'resolution III' is right, but docs/paper_experiment_matrix.csv:14 still says 'resolution-IV'. List it among the source conflicts.

**Evidence (08f1e3c unless stated).** Recomputed from the 288 X12 manifest rows (8 runs; A=n_latent, B=adv_width, C=adv_lr, D=batch_size, E=decoder): D = -AB, E = -AC, defining relation I = -ABD = -ACE = BCDE; shortest word 3, so resolution III. Matches build_paper_manifest.py:259 and I028.

**Replacement** (parts separated by `||`):

```latex
No change to the fragment; add the conflict to the conflict list.
```

### N-19 (note) - supplement.tex:262-273 (S-E5 sources)

**Problem.** The strongest source for the versions actually used is the provenance stored with each latent and score row. It agrees with the gate-time files cited.

**Evidence (08f1e3c unless stated).** D/tier12_v2/A1/scores/*.csv scorer_versions (61 sampled rows, one distinct value); D/tier12_v2/A1/latents config (scvi 1.4.2, torch 2.13.0+cu130).

**Replacement** (parts separated by `||`):

```latex
add to the S-E5 caption -> Versions agree with the provenance written into every A1 latent and score row \src{\file{D/tier12_v2/A1} latents and scores}.
```


## Coverage gaps (locked information missing)

1. Power (PREREG sec. 5 rule, v2) and the frozen record's figures: UNDERPOWERED, SE 0.045, MDE 0.131 (D = 6), 0.182 (D = 4). These are verified facts, shown only as pending (R-12, R-13).
2. Weight definition of SI-25 (target = pre-depletion composition; w = 1/keep fraction; doses 50/80/95) is missing; only SI-42's realisation is unconfirmed (R-18).
3. X3 design facts: dose-100 cell counts (SI-41) and the depleted batch and cell types per task (SI-32, SI-33, manifest) (R-19).
4. PREREG sec. 4 (v2): subsampled tasks (X3, X8) and architecture variants (X12) inherit the full task's matched strength; edge cells that no stage row reads are flagged edge_not_extended; seed-separation checks. Also the D-06 caveat for inherited strengths (R-20).
5. Limits of the evidence: the identity proof (CPU, 1 epoch, 1 thread, one row per code path) and the committed equivalence declaration (scorer commit only; Sinkhorn refused only at c88cce3) (R-03, R-21).
6. PREREG sec. 9, 2026-10-04 (CR-05, CR-08), v2: scib fallback flags are recorded per score row and their counts are reported with the results (R-22).
7. Verified data fact: atac_small is nested in atac_large (all 11,270 cells, re-quantified); this bears on the dataset-family structure and on the pairs-first sensitivity analysis (R-01).
8. Glossary entries for the variant arms, CR-nn, KL warm-up, Tier 1/2, and the scIB metric acronyms (N-04, N-05).
9. PREREG sec. 8 (v2) items not carried: R1 is frozen before R4-a1 adds cell-specific A1 points; extension rows run on their task's machine (SI-17). Minor.
10. sysVI's input (normalised values) and likelihood (Gaussian) in the main Methods (N-16).

## Source conflicts to report to the user

1. ANALYSIS_SPEC.md:73 (sec. 1), its sec. 16 item 6, and final-review I073 give '3,618 shared ATAC cells', counted on cell names. Matching by batch and barcode, all 11,270 atac_small cells are in atac_large (R-01).
2. docs/paper_experiment_matrix.csv:14 calls X12 'resolution-IV'. Code, manifest and I028 give resolution III (N-18).
3. cluster/envspec/local_wcd-kbet_pip_freeze.txt omits harmony-pytorch 0.1.7, which the runner imports. The freeze is stale for the Harmony baseline (R-15).
4. lock_audit.csv contradicts itself. Row 'legend.tex | PREREG sN' uses a check (git log -S on entry titles) that cannot detect in-entry edits. Row 'rq.tex | R0 status' is marked locked while its own note says the confirmation is open. The rows for every finding above need their status and how_checked updated.

## Checked and correct (do not change)

- S-D1 and S-D2: every cell of both tables (cells, batches, keys, cell types, modality, organism, pseudotime, genes, source file, source MD5, counts hash) equals the prepared files and docs/prepped_fingerprints_scib.json.
- S-E1: every row equals the manifest. S-E2: every setting equals code or manifest, except the omitted discriminator weight decay (N-08). S-E3 and S-E4 are exact.
- Equations S-M1 to S-M18 match the cited code: generator objective, fool loss, critic estimate and its active heads, two-sided per-head gradient penalty with uniform partners, barycenter weights and support, MMD kernel (bandwidths sqrt(d) x {0.25, 0.5, 1, 2, 4}), Sinkhorn divergence (OT value from dual potentials, p = 1 cost, blur 0.05 sqrt(2d) = 0.224, annealing ratio 0.8, NaN-only stop, cap 500,000), standardisation (population SD, floor 1e-4), R1 penalty, stratified sampler, reference_fixed detach.
- Rules R1 to R4 as written match PREREG sec. 1-4 and prereg_rules.py, except the clauses in N-10 and R-20.
- SI-68 to SI-77 are represented faithfully, including the App. A.1 option texts for the jackknife variance (Q2-A), the area corner (Q5-A), the half-normal priors (Q6-A) and the P2 Agresti-Caffo/0.25/0.05 model (Q9-A). The ANALYSIS_SPEC proposals (sec. 2.3, 5, 8, 9, 10, 11 unnamed, 15) are correctly pending.
- X1 results, X7/X8 redesigns, follow-up analyses, the Sinkhorn grid, the tolerance and cap outcome, and SI-42 are correctly pending. S-X items 3-16 are correct as sourced (item 13 checked against ANALYSIS_SPEC App. A.1). Final-review counts (99 = 65 + 32 + 2; 17 fix before launch) are correct.

## Prior-work check

Not run. This output introduces no method, metric or contribution claim; the review scope is source fidelity of the outline. The one literature fact used (scIB has 13 integration tasks) was checked against the abstract and Results text of Luecken et al. 2022.

## Limits of this review

- Containment of atac_small in atac_large was counted on all 11,270 cells. The re-quantification evidence (correlations, no identical rows) rests on a random sample of 200 matched pairs; that part is provisional in magnitude, not in direction.
- NB-20261014-05, -07 and -09 were not opened. S-X item 13 ('pilot-informed') was checked against ANALYSIS_SPEC App. A.1, which quotes those entries.
- The identity proof and the import-closure probe were not re-run; their committed outputs and the rigour-review record were read.
- The LaTeX build of the six fragments was not re-run; the delivered logs were read. The replacement snippets were compiled separately (0 errors).
- analysis_x1.py was not checked against ANALYSIS_SPEC (not frozen; outside the outline's locked content).
- No project rigor_ledger artifact exists; earlier reviews keep their ledgers under D/review. The findings are delivered in this report and in the structured output for the lead to merge. No new ledger was started.
