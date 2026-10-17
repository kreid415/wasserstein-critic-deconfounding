# Re-check of methods-track fragments v2 (BIOINF-2026-1434 outline), 2026-10-14

**Verdict: PASS.** All 6 blockers and all 16 fixes from the first review are resolved. Each of the 19 adjusted texts keeps the substance of the recommended replacement. No changed or added line introduces a false locked statement, a wrong number or a misclassification. The 3 declined notes are acceptable, although the reason given for N-16 is inaccurate. 5 optional notes remain (section 4). Replacement texts are ASCII-only and compile with the shared preamble (tectonic, exit 0).

**Inputs.** v2 fragments `methods_track_v2.tar.gz` (version e0037fd8-3936-4d4d-9f7c-fee9610c55e7), `applied_fixes.csv` (7206dcd8-7573-4c85-bed8-1c63aee44719) and `lock_audit_v2.csv` (222250a4-08f1-4ad0-a936-3ce3bac9c71a). These were diffed against the v1 fragments (0d828fa2-c1cf-42ac-aab0-c0f5a701ed0b) and checked against the first review (eef1e505-33a8-47df-9633-9df9f21ee501). Sources were read at repository commit 08f1e3c (read-only), in the data folder `D/` and in installed POT 0.9.7.post1.

**Scope.** 88 changed or added lines (legend 16, research questions 7, methods 26, results 1, discussion 2, supplement 36). Unchanged lines were not re-reviewed. The v2 build log shows only 3 underfull-box warnings.

## 1. Resolution of first-review findings (44 rows of applied_fixes.csv)
The 44 rows are 22 applied, 19 adjusted and 3 declined. Three rows (GAP-09, LEN-1, LEN-2) are the author's extra changes, not findings.

| ID | Severity | Action | v2 location | Check |
|---|---|---|---|---|
| R-01 | blocker | adjusted | methods.tex:65; discussion.tex:17 | containment, labels, 10x names and differing counts kept; '0 of 200 sampled' matches the first-review sample |
| R-02 | blocker | applied | legend.tex:6 | text present verbatim or as instructed |
| R-03 | blocker | adjusted | methods.tex:151; methods.tex:152 | 'one latent per scorer path' dropped: it is in PREREG §9 version-3 text, not in the cited EQUIVALENT_COMMITS.json or NB-20261013-40 |
| R-04 | blocker | applied | legend.tex:26; methods.tex:122 | text present verbatim or as instructed |
| R-05 | blocker | adjusted | methods.tex:9; methods.tex:133 | shared-epochs statement moved to the Backbone line; baselines line keeps the scANVI exception |
| R-06 | blocker | applied | methods.tex:110; supplement.tex:96 | text present verbatim or as instructed |
| R-07 | fix | adjusted | rq.tex:3 | pending item kept; the chat-record audit and the open definition of 'examined' stay in the supplement (supplement:328-329) |
| R-08 | fix | applied | legend.tex:27; legend.tex:28 | text present verbatim or as instructed |
| R-09 | fix | applied | legend.tex:35 | text present verbatim or as instructed |
| R-10 | fix | adjusted | rq.tex:23; rq.tex:38; rq.tex:7; rq.tex:30; rq.tex:45 | R4-row pointer dropped from the pending text only |
| R-11 | fix | applied | methods.tex:43; supplement.tex:207; supplement.tex:208 | text present verbatim or as instructed |
| R-12 | fix | adjusted | methods.tex:113; supplement.tex:97 | main text keeps the UNDERPOWERED flag and both minimal detectable effects; full sentence in S-M15 (supplement:97) |
| R-13 | fix | applied | discussion.tex:18 | text present verbatim or as instructed |
| R-14 | fix | adjusted | methods.tex:53; methods.tex:54 | split in two; content unchanged |
| R-15 | fix | applied | supplement.tex:278; supplement.tex:260 | text present verbatim or as instructed |
| R-16 | fix | applied | supplement.tex:305; supplement.tex:317 | text present verbatim or as instructed |
| R-17 | fix | adjusted | methods.tex:60 | gene-activity phrase dropped; methods:62 still states it |
| R-18 | fix | applied | supplement.tex:56 | text present verbatim or as instructed |
| R-19 | fix | applied | supplement.tex:179 | text present verbatim or as instructed |
| R-20 | fix | applied | supplement.tex:75 | text present verbatim or as instructed |
| R-21 | fix | adjusted | supplement.tex:298; methods.tex:82 | moved to S-E6 (supplement:298); design pointer updated (methods:82) |
| R-22 | fix | adjusted | supplement.tex:299; methods.tex:70 | moved to S-E6 (supplement:299); evaluation pointer updated (methods:70) |
| N-01 | note | applied | legend.tex:5 | text present verbatim or as instructed |
| N-02 | note | applied | legend.tex:51 | text present verbatim or as instructed |
| N-03 | note | adjusted | legend.tex:52 | SI-36 added; consistent with the manifest (PCA seed 0 only) |
| N-04 | note | applied | legend.tex:48 | text present verbatim or as instructed |
| N-05 | note | adjusted | legend.tex:53; legend.tex:57 | glossary rows added (legend:53-57); two row-level wording notes below (V2-N1, V2-N2) |
| N-06 | note | adjusted | methods.tex:33 | 'same head architecture' carries the point (no shared weights) |
| N-07 | note | adjusted | methods.tex:29; supplement.tex:27; supplement.tex:203 | 'at most 10' is correct; source attribution note below (V2-N3) |
| N-08 | note | adjusted | methods.tex:49; supplement.tex:199 | generator and discriminator Adam settings are equal (lr 1e-3, eps 0.01, weight decay 1e-6) |
| N-09 | note | declined | methods.tex:61; discussion.tex:15 | declined; acceptable (see section 3) |
| N-10 | note | applied | supplement.tex:60; supplement.tex:63 | text present verbatim or as instructed |
| N-11 | note | adjusted | methods.tex:91 | matches PREREG §6 |
| N-12 | note | applied | rq.tex:15 | text present verbatim or as instructed |
| N-13 | note | declined | rq.tex:17; results.tex:15 | declined; acceptable (see section 3) |
| N-14 | note | adjusted | methods.tex:25; legend.tex:49 | scvi_adv exception stated; glossary row gives the 1 - KL weight scaling |
| N-15 | note | adjusted | legend.tex:30; methods.tex:123 | 'linear-or-default decoder' in methods, full wording in the glossary |
| N-16 | note | declined | methods.tex:138 | declined; acceptable for the main text, but the stated reason is inaccurate (V2-N4) |
| N-17 | note | applied | methods.tex:64; methods.tex:54; methods.tex:147; methods.tex:152; supplement.tex:307; supplement.tex:327; supplement.tex:329 | text present verbatim or as instructed |
| N-18 | note | applied | methods.tex:123 | fragment unchanged, as instructed; the conflict list it goes to is outside the fragments and was not seen |
| N-19 | note | adjusted | supplement.tex:260 | all 1,098 A1 latents and score rows re-read: one version set each, matching the S-E5 table |
| GAP-09 | extra | applied | supplement.tex:76 | text present verbatim or as instructed |
| LEN-1 | extra | applied | rq.tex:3 | text present verbatim or as instructed |
| LEN-2 | extra | applied | results.tex:3 | text present verbatim or as instructed |

Applied rows: the replacement is present verbatim, or as instructed where the replacement was an instruction (R-11 table rows, R-16, N-01, N-04, N-10, N-12, N-17, N-18). R-16's recommendation to commit the two audit files is not a fragment change. The fragments now label those files as project files outside the repository.

## 2. New or changed statements checked against sources (25 checks)
| Statement | v2 location | Source read | Result |
|---|---|---|---|
| CR-nn glossary row | legend:53 | docs/PREREG.md:167, :376 | row copies the joint 'CR-nn, RR-nn' definition (V2-N1) |
| KL warm-up glossary row | legend:54 | build_paper_manifest.py:104; fit_paper_config.py:120-124 | 400-epoch warm-up correct; 'fewer epochs end below 1' imprecise (V2-N2) |
| Tier 2 adds X3, X6, X12 | legend:55 | SI-05; PAPER_PLAN.md §4 table ('Tier 1 + 2 (adds X3, X6, X12)') | correct |
| metric name expansions | legend:56 | score_scib_native.py:51-53 | correct |
| other abbreviations, JHPCE with L40S GPUs | legend:57 | SI-39 | correct |
| scvi_adv: manifest lambda 0, loss scaled by 1 - KL weight | legend:49 | manifest (all scvi_adv rows lam 0); scvi_adversarial_plan.py:175-177 | correct |
| variant arms list | legend:48 | scvi_adversarial_plan.py:12-24; prereg_rules.py:59-61 | correct |
| X7 tasks atac_small, pancreas, sim1; four arms | legend:27 | manifest X7 rows | correct |
| X8 totals 16,000 / 6,000 / 3,000 | legend:28 | build_paper_manifest.py:111 | correct |
| X12 decoder: scVI default or LinearSCVI | legend:30 | scvi_adversarial_plan.py:516-519 | correct |
| seeds: neural X13 0-4; Harmony, Scanorama, PCA seed 0 | legend:52 | manifest X13 seeds by method; SI-36 | correct |
| epoch heuristic min(400, round(20000 x 400/n)) | methods:9 | build_paper_manifest.py:114-116 | correct |
| scANVI extra epochs min(10, max(2, round(E/3))) | methods:133 | fit_paper_config.py:401 | correct |
| barycenter: at most 10 iterations, stop threshold 1e-7 | methods:29; supplement:27, :203 | barycenter.py:38, :72-73 (unchanged c88cce3..08f1e3c); POT 0.9.7.post1 ot/lp/_barycenter_solvers.py:255 | correct; SI-14 attribution (V2-N3) |
| discriminator uses the same Adam settings as the generator | methods:49; supplement:199 | scvi_adversarial_plan.py:162, :245-260 | correct |
| Sinkhorn refit counts 180 + 18 + 36 = 234 | methods:53 | SI-57 (first review) | correct |
| power: SE 0.045, MDE 0.131 / 0.182, power 0.104 / 0.088, equivalence unreachable, sigma 0.0298 (12 df), tau 0.109 (4 df), six A1 blocks | methods:113; supplement:97; discussion:18 | docs/prereg/r1_record.json power | all values match (rounded) |
| R1 frozen before R4-a1 adds points; extension rows on the task's machine (GAP-09) | supplement:76 | PREREG §8, lines 138-139; SI-17 | correct |
| X3 targets and cells per dose | supplement:179 | SI-32; SI-33; SI-41 ('every dose ... the same cell count'); manifest | correct |
| JHPCE rebuild: conda equal, pip equal apart from 34 CUDA builds | supplement:260 | SI-48; notebook/2026-10-04.md NB-20261004-20 ('90 identical + 34 declared CUDA substitutions') | correct |
| latents record scvi-tools and PyTorch versions; score rows record scorer versions | supplement:260 | fit_paper_config.py:264-272, :353-355; score_scib_native.py:38-41 | correct |
| versions agree in all 1,098 A1 latents and score rows | supplement:260 | D/tier12_v2/A1: 1,098 latents all scvi 1.4.2, torch 2.13.0+cu130, RTX 3080; 1,098 score rows one scorer_versions set (scib 1.1.7, scanpy 1.11.5, anndata 0.11.4, numpy 1.26.4, R kBET 0.99.6) | correct |
| Harmony: harmony-pytorch 0.1.7; freeze lists harmonypy 2.0.0 | supplement:278 | run_cpu_baselines.py:76, :142; local_wcd-kbet_pip_freeze.txt:34 | correct |
| original-submission audit: §3 items 1 and 24; named in PAPER_PLAN Sources | supplement:305 | original_manuscript_audit.md (version 0c9e00ea) §3 items 1 and 24; PAPER_PLAN.md:7 | correct |
| sysVI input | supplement:191 (unchanged row, cited in the N-16 decline) | fit_paper_config.py:404-409 | input not stated in any fragment (V2-N4) |

## 3. Declined notes
- **N-09** ('18 batches in 4 tasks' hold non-integer values; 17 are distinct): acceptable. The count is of task-batch pairs and is literally true. S-D2 (supplement:136) states that the Villani batch is the same 1,022 cells in both immune tasks.
- **N-13** (no pending tag on SI-75 in research questions and results): acceptable. SI-75 is a locked user decision until the user changes it. The open re-decision is stated in methods:107 ('whether SI-75 stands after the pilot recount (8 of 54, not 8 of 45)') and in supplement:103.
- **N-16** (sysVI input in the main text): acceptable for the main text. The stated reason ('S-E2 states sysVI's normalised input') is inaccurate, so V2-N4 gives a supplement-only fix.
- The partial decline in **N-05** (no spelling-out at first use; glossary rows instead) is acceptable for an outline whose legend precedes every section.

## 4. Remaining findings (all notes, optional)
| ID | Location | Problem | Evidence |
|---|---|---|---|
| V2-N1 | legend.tex:53 | The CR-nn glossary row copies PREREG's joint 'CR-nn, RR-nn' row, so CR-nn (code-review finding) is defined as also covering re-review findings, which PREREG numbers RR-nn. | docs/PREREG.md:167 defines 'CR-nn, RR-nn' jointly; PREREG.md:376 uses 'lesson RR-01'. The fragments cite only CR-01, -05, -07, -08, -12 and -14. |
| V2-N2 | legend.tex:54 | 'so tasks with fewer epochs end below weight 1' implies that 400-epoch tasks reach weight 1. Every run ends below 1: the KL (Kullback-Leibler) weight at 0-based epoch e is e/400, so 400-epoch tasks end at 0.998. | fit_paper_config.py:123 ('KL weight at 0-based epoch e is e / n_epochs_kl_warmup'); S-E3 formula (E-1)/400; final weights 0.998 (E = 400) and 0.203-0.613 (E = 82, 94, 239, 246). |
| V2-N3 | methods.tex:29; supplement.tex:27, :203; lock_audit_v2.csv row 'M arms barycenter' | 'At most 10 iterations' is correct, but the source list attributes it to SI-14, whose rule says '10 fixed-point iterations per training step'. The SI-14 conflict the author reported is not marked in any fragment, and the audit row is still 'locked' with stale source lines :71-72. | barycenter.py:38 (n_iter=10, stop_thr=1e-7), :72-73 (numItermax, stopThr; no log argument, so iteration counts are not recorded); barycenter.py has no commits in c88cce3..08f1e3c; POT 0.9.7.post1 ot/lp/_barycenter_solvers.py:255 loops while the squared displacement > stopThr and iter_count < numItermax. |
| V2-N4 | supplement.tex:191 (S-E2 Likelihood row); N-16 decline | The N-16 decline is acceptable for the main text, but its reason is inaccurate. S-E2 gives only sysVI's Gaussian likelihood, and no fragment says that sysVI fits the file's normalised values, while methods:62 says fitting uses the counts layer. | supplement.tex:191 'ZINB (sysVI: Gaussian)'; grep of all six v2 fragments: the only 'normalised values' statement is methods:139 (Harmony, Scanorama, PCA); fit_paper_config.py:404-409 (SysVI.setup_anndata on the file, no layer; comment 'Gaussian likelihood on scIB's normalised X'). |
| V2-N5 | supplement.tex:283 | The S-E6 heading says 'fit provenance', but S-E6 now also holds a scoring statement (scib fallback flags, supplement:299), and the Evaluation pointer (methods:70) sends readers there. | supplement.tex:283, :299; methods.tex:70 '[S-E2, S-E6]'. |

Replacement texts (compile-tested):
- V2-N1: `CR-nn & A finding of the code review of tag prereg-tier12-v1 (2026-10-03); findings of the re-review of its fixes (2026-10-04) are numbered RR-nn. \src{\file{docs/PREREG.md}:167} \\`
- V2-N2: `KL warm-up & KL: Kullback--Leibler. The weight of the KL term of the scVI loss rises linearly from 0 over 400 epochs (scvi-tools default) and ends at $(E-1)/400$ for a task with $E \leq 400$ epochs: 0.998 at 400 epochs, 0.20--0.61 on the four shorter tasks [S-E3]. \src{\file{build_paper_manifest.py}:104; \file{fit_paper_config.py}:123} \\`
- V2-N3: `\item Barycenter critic (W1): head $k$ targets a Wasserstein-2 barycenter (average distribution; \unverified{cuturi2014fast}) of the minibatch's batches, recomputed per step, cold start, at most 10 iterations \src{SI-14 (10 iterations); \file{wcd/barycenter.py}:38, :72--73 (cap 10, stop threshold $10^{-7}$); manifest}.  ||  conflict list -> SI-14 says 10 fixed-point iterations; the frozen code (barycenter.py unchanged since c88cce3) caps at 10 and stops once the summed squared movement of the support is at most 1e-7, i.e. at a numerical fixed point; iteration counts are not recorded.  ||  lock_audit_v2.csv 'M arms barycenter' source -> SI-14; src/wcd_vae/wcd/barycenter.py@08f1e3c:38, :72-73`
- V2-N4: `Likelihood & ZINB (sysVI: Gaussian, on the file's normalised values) & \src{SI-10; \file{fit_paper_config.py}:404--409} \\`
- V2-N5: `\paragraph{S-E6 Hardware placement, fit and score provenance.}`

## 5. Not checked
- Lines unchanged since v1, per the task.
- The author's own check that 0 of all 11,270 matched ATAC count rows are identical. The fragment cites the 200-pair sample checked in the first review, so the larger check was not repeated.
- Facts inside my own first-review replacement texts were not re-derived; they were checked in the first review.
- lock_audit_v2.csv: only the 12 rows for the changed glossary, KL (Kullback-Leibler) warm-up, tier and barycenter items were read, not all 412 rows.
- The lead's conflict list, which should hold the N-18 conflict (docs/paper_experiment_matrix.csv:14 says resolution IV for X12; the design is resolution III) and the SI-14 conflict. It is outside the fragments and was not provided.
- Whether the barycenter loop actually stops before 10 iterations in practice. Fit records do not log iteration counts (barycenter.py:72-73 passes no log argument).
