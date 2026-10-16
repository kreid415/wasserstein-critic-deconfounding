# Rigour review, track 3 of 3: the amendments to the pre-registration documents

Reviewer: Rigor Reviewer. Date: 2026-10-14. Reviewed: branch `final-review-fixes`, detached at `1e615eb321097729c0fb4f117bc4e9fd1e0ac9a8`, all document changes since `5cb75c0`
(PREREG.md +94/-2 lines, ANALYSIS_SPEC.md new, 295 lines, PAPER_PLAN.md +29/-14, CONSTRAINTS.md +186 for SI-53 to SI-77, EQUIVALENT_COMMITS.json new, notebooks of 2026-10-13 and 2026-10-14).
Status of the tag: `prereg-tier12-v3` has not been made. Nothing here has been fixed by me; every item is a recommendation.

## 1. Verdict

**PASS_WITH_FIXES.** 13 fixes, 6 notes, 0 blockers. Of the 32 numbers quoted in PREREG, ANALYSIS_SPEC and PAPER_PLAN that could be recomputed, 23 reproduce as quoted and 9 reproduce but come with a wording or scope issue; none fails.
Two numbers quoted elsewhere do not reproduce as quoted: the pilot collapse counts put to the user in the SI-75 question (DOC-17) and the early and train-mode capture count in notebook entry NB-20261013-03 (236; the file has 268; DOC-19). The problems are in what the documents say about *when and by whom* things were decided and done, in
two inconsistencies between documents (Holm threshold, Sinkhorn redo), and in readability. All 13 fixes are edits to text or lists; none needs a new fit or analysis, and none changes a rule or an existing decision (I checked the R2/R3 decisions explicitly: they do not change).

The five items that matter most before the tag is made:

1. **DOC-01.** The new entries say the X1 analyses are frozen "before X1 runs". X1 had started (19 rows scored on 2026-10-12) and one X1 latent was rescored for the identity check. No document says so.
2. **DOC-02.** "All 234 Sinkhorn pilot fits are refitted" contradicts the rule order the user chose (SI-57): the extension and A3 rows depend on the redo.
3. **DOC-03 / DOC-04.** The header still says nothing was chosen after seeing pilot or X1 data; the entry on how low-signal blocks count in R2 and R3 (SI-56, SI-66) is dated before its implementation (2026-10-14) and omits that the R2/R3 record was made without the rule.
4. **DOC-05.** Holm threshold 0.0125 (four tests) in PREREG and PAPER_PLAN versus six tests (0.0083) in the spec and code. The conclusion does not change.
5. **DOC-11 / DOC-13.** The text does not meet the plain-language rule (SI-67), and ANALYSIS_SPEC is still a draft question sheet.

Severity: **blocker** = tagging as written would freeze a rule that cannot run or silently change a result (none found); **fix** = correct before the tag, no result changes; **note** = disclose, decide or tidy.

## 2. Terms used in this report

| Term | Meaning |
|---|---|
| PREREG.md / tag | The pre-registered decision rules; a tag is a git label that freezes the committed files at one moment (`prereg-tier12-v3` is planned) |
| A1, A2, A3 | Pilot experiments (3 tasks, 3 seeds): A1 sets each method's six penalty strengths; A2 tests whether the adversary sees latent means or samples; A3 tests latent standardisation |
| X1, X13 | The core benchmark (8 tasks, 5 seeds, 3,000 model fits) and the baseline integrators (408 rows) |
| R1 to R4 | PREREG rules: R1 six penalty strengths (lambda) per method family; R2 adversary input; R3 standardisation; R4 the matched lambda at equal batch removal. (PAPER_PLAN uses R1 to R5 for research questions; this report always means PREREG rules.) |
| b0, b_common, b* | Batch score of plain scVI (lambda = 0); the highest batch score every method reaches; the halfway target b0 + 0.5 (b_common - b0). A task-by-decoder block is **low-signal** (`bstar_low_signal`) when b_common - b0 < 0.02, i.e. too little room to compare methods |
| Decoder, conditioned | Whether scVI's decoder receives the batch label (conditioned) or not (unconditioned) |
| P1, area, P2 | Main endpoint (biology kept at equal batch removal, reference critic minus discriminator); co-primary frontier area; collapse rate (a run collapses when it fails or loses at least 0.10 biology score) |
| Holm | A multiple-testing correction: the smallest p-value is compared with 0.05 divided by the number of tests |
| Abbreviations | SD standard deviation; SE standard error; ROPE region of practical equivalence; W1 Wasserstein-1 distance (what the critics estimate); JS Jensen-Shannon divergence (what the discriminator estimates); MMD maximum mean discrepancy; scVI single-cell variational inference (the backbone model); scIB single-cell integration benchmarking (the scoring package); NMI normalised mutual information; HLCA Human Lung Cell Atlas; sim1, sim2 the two simulated tasks |
| SI-nn, I0nn, D-nn, G-nn, NB-date-nn | User decisions in CONSTRAINTS.md (each with the user's words); final-review issues; design re-check items; guard-review items; notebook entries |
| Labels used here | SI-02 new datasets removed; SI-51 fix Sinkhorn and redo; SI-54 and SI-60 declare the two commits result-identical and name the new tag; SI-55 PAPER_PLAN corrected to PREREG on grid edges; SI-56 P1 starts at lambda = 0 and low-signal blocks are not evaluable; SI-57 Sinkhorn redo in rule order; SI-61 hitting the Sinkhorn iteration cap stops the stage; SI-65 NMI-0 edge case decided case by case; SI-66 low-signal blocks count as not evaluable in R2/R3; SI-67 plain-language rule; SI-68 to SI-77 choices for the X1 analyses; I073 sim1 is a simulation yet enters the between-task SD; I091 R1 depends on trajectory conservation; D-10 PAPER_PLAN carries superseded statements; G-01 guard for minibatches with no pair of batches |

## 3. Findings

| ID | Severity | Where | What is wrong |
|---|---|---|---|
| DOC-01 | fix | PREREG.md:310 (entry heading), :315-316; ANALYSIS_SPEC.md:10-11, :40; PAPER_PLAN.md:164-166 | The documents say the X1 analyses are frozen before X1 runs, but X1 had already started and 19 scores existed |
| DOC-02 | fix | PREREG.md:261, :274-277; CONSTRAINTS.md:383 (SI-57) | 'All 234 Sinkhorn pilot fits are refitted' conflicts with the rule order the user chose in SI-57 |
| DOC-03 | fix | PREREG.md:3, :5, :145-147 | The unchanged header and section 9 lead-in still say nothing was chosen after pilot or X1 data and that every amendment precedes its stage |
| DOC-04 | fix | PREREG.md:297-309; ANALYSIS_SPEC.md:101-104 (section 2.5) | SI-56/SI-66 entry: wrong date order, undisclosed implementation gap, one flagged block omitted, misleading 'still fitted' |
| DOC-05 | fix | PREREG.md:106, :250-252; PAPER_PLAN.md:184; ANALYSIS_SPEC.md:155-157, :271-272; scripts/prereg_rules.py:52, :712; scripts/analysis_x1.py:479-482 | The Holm threshold is quoted for four tests while the amended spec and analysis code use six |
| DOC-06 | fix | PREREG.md:236-239; ANALYSIS_SPEC.md:55-56 | 'The frozen X1 grids stand' is not true for Sinkhorn; Sinkhorn figures in the I091 entry (final-review issue: R1 depends on trajectory) come from the discarded fits; the spec reads a record that will be superseded |
| DOC-07 | fix | PREREG.md:104, :245-248; PAPER_PLAN.md:192; ANALYSIS_SPEC.md:69-74 (section 2.2) | The quoted power figures are not the power of the endpoint the spec freezes |
| DOC-08 | fix | PREREG.md:240, :310-311; PAPER_PLAN.md:164, :184, :187, :191; ANALYSIS_SPEC.md:6-9, :187 | Agent proposals are presented as user decisions |
| DOC-09 | fix | PAPER_PLAN.md:122-128, :167, :189, :205-207, :214-223, :273-274 | PAPER_PLAN still states design facts the manifest and spec contradict, outside the section the pointer covers |
| DOC-11 | fix | PREREG.md:229-319 (esp. :264-296); ANALYSIS_SPEC.md (whole); PAPER_PLAN.md:164-173 | The amended text does not meet the plain-language rule (SI-67) |
| DOC-12 | fix | PREREG.md:261, :274-278, :250, :310, :315; PAPER_PLAN.md:165-166; ANALYSIS_SPEC.md:10, :248 | Planned steps are written in the present tense, and the freeze is described as done |
| DOC-13 | fix | ANALYSIS_SPEC.md:1-12, :29, :36-44, :48-50, :168-170, :291; PREREG.md:316 | ANALYSIS_SPEC is still a draft question sheet; the file to be frozen should hold only rules |
| DOC-15 | fix | docs/EQUIVALENT_COMMITS.json:8-9; PREREG.md:253-263, :281-284 | Evidence for 'result-identical' is cited loosely and the machine-readable list points at the earlier identity run |
| DOC-10 | note | ANALYSIS_SPEC.md:53-56; PREREG.md:92-99; scripts/prereg_rules.py:559-570; scripts/analysis_x1.py:359-391 | Open question: b* is computed from the stage's extended point set, but P1 uses only the six registered points |
| DOC-14 | note | PREREG.md:30, :269; CONSTRAINTS.md:412 (SI-61), :440 (SI-65) | Two contingencies have no registered procedure |
| DOC-16 | note | ANALYSIS_SPEC.md:130-136, :155-157, :161-170, :263-270; PREREG.md:243-245 | Statistical points an outside referee is likely to raise (none is an error; each needs a sentence) |
| DOC-17 | note | ANALYSIS_SPEC.md:118-123, :130-134, :161-170; CONSTRAINTS.md:516-522 (SI-75) | Pilot-informed choices are not labelled, and two pilot figures put to the user were off |
| DOC-18 | note | CONSTRAINTS.md:404-408 (SI-60), :425-430 (SI-63), :468-474 (SI-69); SI-51, SI-52 | Ledger housekeeping |
| DOC-19 | note | PREREG.md:265-266; PAPER_PLAN.md:74; notebook/2026-10-13.md:28-31; guard review (G-05) | Number provenance and wording |

### DOC-01 · fix · The documents say the X1 analyses are frozen before X1 runs, but X1 had already started and 19 scores existed

**Where:** PREREG.md:310 (entry heading), :315-316; ANALYSIS_SPEC.md:10-11, :40; PAPER_PLAN.md:164-166

**Problem:** The new entries say the X1 analyses are specified and frozen 'before X1 runs' (PREREG:310), 'before any X1 adversarial-arm score is examined' (PREREG:315, spec:10, PAPER_PLAN:165), and that each analysis is fixed before 'any X1 result exist' (spec:40). X1 is the core benchmark (3,000 model fits). Its local part started on 2026-10-12 at 12:40 and was stopped at 16:36 with 19 rows scored, 1 fitted but not scored and 1,097 not yet fitted (NB-20261012-04, -05). The lead's brief says 17 of the 19 are adversarial-arm rows (immune, conditioned decoder; discriminator, pooled and reference critics; one seed); I could not count them because the X1 folders are blinded to this review. About 250 plain-scVI and baseline 'filler' fits also existed (158 scored locally at the final review, per the lead's report); PREREG:22 lets such rows run while A1-A3 are in progress. Starting X1 after R1-R3 were decided (2026-10-12 11:43) is the registered order (stage table, PREREG:11-20) and the local start was the user's choice (NB-20261012-02, -04), so the run itself is not a deviation; what is missing is the disclosure. One of the 19 scores (discriminator, immune, conditioned decoder, seed 0, tag X1_immune_discriminator_lg1_c1_s0_122afe99) was recomputed twice for the identity check (NB-20261013-12, -16, -37, -40) and compared by script with its original row; the committed result lists only the machine name as different. No amended text mentions any of this. The user's SI-68 question (which analyses to lock) asked what to lock 'before X1 starts' and described locking as freezing 'before any X1 result exists'.

**Evidence:** outputs/timeline.txt; outputs/misc_counts.json (x1_latents_in_identity_check = 1 X1 tag; score_identity.differing_vs_production = score_host only); notebook/2026-10-12.md:33-51; the lead's final-review report to the user (read-only transcript) lists 'the 19 local X1 scores' as standing subject to the identity check, so the user was told, but the documents were not.

**Suggested fix:** Add one dated paragraph to PREREG section 9: when X1 started and stopped; rows scored, by arm, task and seed; that nobody read their values (state this only if true); that one rescored latent was compared by script only. Define 'examined' (suggested: a value of an X1 adversarial-arm score was read by a person or entered an analysis script). Change the heading to 'frozen before any X1 result is analysed' and repeat the wording at spec:10, spec:40 and PAPER_PLAN:165. Stating prior access to existing data, and tabulating deviations, is the standard practice (van den Akker et al. 2021; Lakens 2024; Willroth and Atherton 2024).

### DOC-02 · fix · 'All 234 Sinkhorn pilot fits are refitted' conflicts with the rule order the user chose in SI-57

**Where:** PREREG.md:261, :274-277; CONSTRAINTS.md:383 (SI-57)

**Problem:** PREREG:274-277 says all 234 Sinkhorn pilot fits (A1 180, R1 extension 18, A3 36) are refitted and rescored, and then, 'in rule order', that R1 produces the extension rows and R4 stage a1 'resolves' the A3 rows. Both cannot hold. The extension and A3 rows were chosen by rules applied to the discarded pre-fix Sinkhorn fits, so after the redo they can differ; SI-57 says to refit the 180 A1 fits, run R1 for the Sinkhorn family and its extension rows, run R4-a1, fit the A3 rows it resolves (including non-Sinkhorn cells whose window moved), then re-run R2 and R3. 234 is the number of Sinkhorn rows that existed before the fix (180 + 18 + 36 recomputed from the manifest), not a promised future count. 'No other fit is redone' (:277) is true only if fitting new non-Sinkhorn A3 rows for moved windows is not called a redo.

**Evidence:** outputs/misc_counts.json (sinkhorn_rows_in_manifest, sinkhorn_r1_extension_rows, sinkhorn_pilot_total); CONSTRAINTS.md:383-384; SI-51 wording 'redo all 234' (CONSTRAINTS.md SI-51).

**Suggested fix:** Reword: 'The 180 A1 Sinkhorn fits are refitted first. The R1 extension rows and the A3 Sinkhorn rows are then fitted as R1 and R4-a1 require on the new fits (18 and 36 rows existed before the redo; the numbers may change). Non-Sinkhorn A3 rows are fitted anew where their window moves; no existing non-Sinkhorn fit is redone.'

### DOC-03 · fix · The unchanged header and section 9 lead-in still say nothing was chosen after pilot or X1 data and that every amendment precedes its stage

**Where:** PREREG.md:3, :5, :145-147

**Problem:** PREREG:3 says 'nothing in this file is chosen after seeing A1, A2, A3 or X1 data' and :145 says each amendment is 'committed before the stage it affects'. Amendments since then change or add rules after pilot data had been seen: the rule for low-signal blocks in R2 and R3 (SI-56, SI-66) was decided on 2026-10-13 and implemented on 2026-10-14 (f3c3415), after R2 and R3 had first been run on the A2/A3 results on 2026-10-12 (NB-20261012-01), and it depends on a flag computed from A1 data (R2/R3 will be re-run after the Sinkhorn redo, SI-57, so 'committed before the stage' holds for the re-run, not for the first run); the R1 trajectory correction (I091) comes from A1 noise estimates; the Sinkhorn redo re-runs R1, R4-a1, R2 and R3 on new fits; the spec's prior scale 0.1, area corner and P2 margin were chosen after the pilot (DOC-17). Line 3 also still describes v2 (branch prereg-v2), and line 5 lists only the v2 rule files, not analysis_x1.py, ANALYSIS_SPEC.md or tests/analysis.

**Evidence:** outputs/timeline.txt; outputs/commit_dates.txt; report section 6 (table of what was seen before each amendment).

**Suggested fix:** Replace the sentence at line 3 by a v3 status paragraph with an amendment table: decided, committed, data seen before the decision (none, A1, A2/A3, X1 rows), result changed (yes/no). Section 6 of this report is a draft of that table. Update the file list at line 5.

### DOC-04 · fix · SI-56/SI-66 entry: wrong date order, undisclosed implementation gap, one flagged block omitted, misleading 'still fitted'

**Where:** PREREG.md:297-309; ANALYSIS_SPEC.md:101-104 (section 2.5)

**Problem:** The entry is dated 2026-10-13 but cites commits made on 2026-10-14 (f3c3415 at 07:50, ff88666 at 08:04). The notebook records that SI-56/SI-66 were 'recorded but never implemented' in the rule code until 07:50 on 2026-10-14 (NB-20261014-14, -15), so the R2/R3 record (made with the c88cce3 code on 2026-10-12) was produced without the rule. I re-applied the registered criteria to that record's cells: as registered, R2 had 12 of 12 cells evaluable (mean change in biology, alternative minus default, -0.0014) and R3 17 of 20 (+0.0010); with SI-66, R2 has 9 of 12 (-0.0042; the 'one evaluable cell per task and decoder' condition fails) and R3 13 of 20 (-0.0014; 14 needed). Neither decision changes: both keep the defaults, and would without SI-66 because the mean change is below the 0.01 a switch needs (criterion 2); but the record's coverage fields would differ if it were re-run. None of this is in the entry. The entry names only immune/conditioned as flagged; the frozen R4-a1 record flags sim1/conditioned too (b0 0.8120, b_common 0.8189, difference 0.0069, set by MMD), as spec section 2.5 says. 'A2 and A3 are still fitted and reported' (:307-308) reads as if fits were pending: all 288 A2/A3 rows were fitted and scored (the runner ended 2026-10-11 08:20, NB-20261012-02); only the 36 A3 Sinkhorn rows are to be redone. The argument that the flag cannot clear for immune/conditioned is correct: b_common is a minimum over arms, so the Sinkhorn redo can only lower it, and barycenter (which sets it) is not an A2/A3 arm. A consequence the entry states only in part: R2 and R3 can no longer switch in X1 whatever A2/A3 show, so the 36 Sinkhorn A3 rows and the full R2/R3 re-run (SI-57) cannot change X1.

**Evidence:** outputs/si66_effect.txt/.json (own re-implementation of PREREG section 3 on docs/prereg/r2r3_record.json; the repo script refuses the c88cce3 Sinkhorn A3 rows); outputs/misc_counts.json (r4a1_blocks); outputs/timeline.txt.

**Suggested fix:** Give both dates (decided 2026-10-13, implemented 2026-10-14). Add: 'The R2/R3 record made at c88cce3 predates this rule; applying it changes no decision (R2 ..., R3 ...).' Name both flagged blocks. Replace 'are still fitted' by 'exist (Sinkhorn A3 rows are redone, SI-57)'. Say whether the A3 Sinkhorn rows and the R2/R3 re-run are still run for the record or dropped.

### DOC-05 · fix · The Holm threshold is quoted for four tests while the amended spec and analysis code use six

**Where:** PREREG.md:106, :250-252; PAPER_PLAN.md:184; ANALYSIS_SPEC.md:155-157, :271-272; scripts/prereg_rules.py:52, :712; scripts/analysis_x1.py:479-482

**Problem:** Holm's method (a multiple-testing correction) compares the smallest p-value with 0.05 divided by the number of tests. PREREG section 5 and the 2026-10-13 entry use four tests (0.05/4 = 0.0125). The amended spec and analysis_x1.py use six (P1, area and P2, each per decoder; first threshold 0.00833). PAPER_PLAN:184 replaces 'Holm over P1-P4' by 'Holm over the six primary results' and keeps 'first threshold of 0.0125' in the same sentence. The committed record (docs/prereg/r1_record.json) stores 0.0125 because N_ENDPOINTS = 4 in prereg_rules.py. The smallest attainable exact two-sided Wilcoxon p-value with six datasets is 0.03125 (recomputed), above every candidate first threshold (0.0125, 0.00833, and 0.00625 for eight), so 'cannot reject' holds in all versions; but only after the Holm adjustment (unadjusted, 0.03125 < 0.05).

**Evidence:** outputs/power.json (wilcoxon_min_p_n6, holm_first_threshold 4/6/8); scripts/analysis_x1.py:479-482 (n_primary = 6); docs/prereg/r1_record.json power.holm_first_threshold.

**Suggested fix:** State one family size in every place; write '0.05/6 = 0.0083' in PREREG:106/:251 and PAPER_PLAN:184, say the companion 'cannot reject at the Holm-adjusted level', and note that the stored record shows the four-test value.

### DOC-06 · fix · 'The frozen X1 grids stand' is not true for Sinkhorn; Sinkhorn figures in the I091 entry (final-review issue: R1 depends on trajectory) come from the discarded fits; the spec reads a record that will be superseded

**Where:** PREREG.md:236-239; ANALYSIS_SPEC.md:55-56

**Problem:** The I091 entry says 'R1 stays as registered and the frozen X1 grids stand'. The Sinkhorn redo re-runs R1 for the Sinkhorn family, and the user was told 'the X1 manifest's Sinkhorn lambda grid will change after R1 is re-run; the other arms' grids won't' (lead transcript, read-only). The Sinkhorn figures in the I091 entry (bio-noise SD 0.059 / 0.013 and 'Sinkhorn changes to an extension') come from the discarded pre-fix Sinkhorn fits; my re-run of the committed script at 018c8f1 reproduces them exactly, and the entry's last sentence defers the Sinkhorn part to the refitted fits (SI-57). ANALYSIS_SPEC:55 takes the six lambda per arm from docs/prereg/r1_record.json (x1_grid), which still holds the superseded Sinkhorn grid; neither document says where the redone R1, R4-a1 and R2/R3 records will be committed or that the spec reads those.

**Evidence:** outputs/r1_without_trajectory_rerun.csv/.txt/_compare.txt (identical to the committed CSV); outputs/i091_summary.json; transcripts frame baa34fcd and ebf81ab2.

**Suggested fix:** Write 'the frozen X1 grids stand for every arm except Sinkhorn, whose grid is re-derived by R1 on the refitted fits'; label the Sinkhorn figures 'pre-fix'; name the files that will hold the redone records and have the spec read them.

### DOC-07 · fix · The quoted power figures are not the power of the endpoint the spec freezes

**Where:** PREREG.md:104, :245-248; PAPER_PLAN.md:192; ANALYSIS_SPEC.md:69-74 (section 2.2)

**Problem:** The figures reproduce exactly from the committed record (six real tasks: standard error 0.0450, minimum detectable effect at 80% power 0.1307, power at a 0.02 difference 0.1044; four families 0.1818 and 0.0878; within-block SD 0.0298, between-task SD 0.1094). They are not the power of the frozen P1. They use the matched-lambda contrast (PREREG:104), whereas P1 reads biology at the target batch score by interpolation (spec 2.2). They pool all six task-by-decoder pilot blocks, including the simulation sim1 and the two low-signal blocks the amendments call not evaluable. The between-task SD has 4 degrees of freedom (a crude chi-square interval runs from 0.066 to 0.314), and in the conditioned decoder the simulation sim1 supplies 67% of the squared spread of the three block means. Without sim1 (final-review issue I073: sim1 is a simulation yet enters the between-task SD): within-block SD 0.0329, between-task SD 0.0558 (2 degrees of freedom), SE 0.0236, minimum detectable effect 0.0684, power 0.182 (six tasks) and 0.137 (four families). The conclusion (underpowered; equivalence within +/-0.01 or 0.02 unreachable) holds either way.

**Evidence:** outputs/power.json and recompute_power.txt (scripts/recompute_power.py re-derives from docs/prereg/r1_record.json 'blocks'; sim1 removed for the second set).

**Suggested fix:** State in both documents that the figures are for the matched-lambda contrast on all six pilot blocks including sim1, add the without-sim1 set, and plan to recompute with the frozen P1 reading and only evaluable blocks when R4-a1 is re-run (SI-57).

### DOC-08 · fix · Agent proposals are presented as user decisions

**Where:** PREREG.md:240, :310-311; PAPER_PLAN.md:164, :184, :187, :191; ANALYSIS_SPEC.md:6-9, :187

**Problem:** PREREG:310-311 and PAPER_PLAN:164 attribute the X1 analyses to 'user decisions SI-68 to SI-77'. ANALYSIS_SPEC:6-9 says section 2.3 (not-evaluable rule), 5 (multiplicity), 8 (transfer lambda), 9-11 and 15 are the agent's proposals, to be shown at the diff review. Three PAPER_PLAN replacement markers (:184 Holm over six, :187 leave-one-family-out, :191 equal k) point to those sections, so a reader sees user-approved replacements that no user decision covers: CONSTRAINTS.md has no Holm-over-six or leave-one-family-out entry (SI-68 says only 'chosen on the other datasets'). PREREG:240 heads a four-part entry with 'user decision SI-55'; SI-55 covers part (a) only (grid edge). The heading of spec section 8 says 'DECIDED Q11, SI-77', but SI-77 decides only the selection score; the leave-one-family-out rule inside it is an agent proposal.

**Evidence:** ANALYSIS_SPEC.md:6-9; CONSTRAINTS.md:368-373 (SI-55), :460-466 (SI-68); PAPER_PLAN.md markers (grep of '[replaced').

**Suggested fix:** After the user approves the diff, record that approval as new ledger entries for the proposed sections and write 'user decisions SI-68 to SI-77 and the proposals approved at the diff review (SI-nn)'; narrow :240 to 'user decision SI-55 for (a)'.

### DOC-09 · fix · PAPER_PLAN still states design facts the manifest and spec contradict, outside the section the pointer covers

**Where:** PAPER_PLAN.md:122-128, :167, :189, :205-207, :214-223, :273-274

**Problem:** The matrix (section 4) gives A1 891 fits (manifest of record: 1,098), X1/X2 4,125 (3,000), X13 385 (408), X3 540 (828), X6 270 (216); X15 (84 rows) is absent. 'Equal configuration counts' (:126, :170, :205) is replaced only at :191. '9 real tasks, 7 families' (:207) versus six tasks in four families (spec:57-58). Pancreas is 14,890 cells and 8 batches (:214); the prepped file has 16,382 cells and 9 batch labels. immune_hum_mou's core key is 'species', V = 2 (:216), but the fitter uses 'batch' (23 labels). Lung is replaced by an HLCA subset and four new datasets are still listed (:219-223, :273) although SI-02 removed them (as :184 says), and 'Barycenter critic: drop or rebuild' (:274) is still listed as open although X1 contains 480 barycenter rows. The 'Scaling' sensitivities (:189: leave-one-method-out rescaling, z-score aggregation, composite without silhouette, Kendall tau) are neither in the spec nor marked replaced. The pointer (:167) says 'seven statements ... each marked [replaced]'; the page has eight markers (:178 twice, :181, :182, :184, :187, :190, :191) because the pointer joins the ROPE and the measurement-SD study. This is the part of final-review item D-10 (PAPER_PLAN carries superseded statements without markers) that the markers did not reach.

**Evidence:** outputs/misc_counts.json (experiment_row_counts_v3); outputs/check_task_facts.txt; docs/PAPER_PLAN.md at 1e615eb; review/rereview/design_fable/REVIEW_design_recheck.md:43 (D-10).

**Suggested fix:** Add a dated 'superseded by' line (manifest v3 counts, six tasks in four families, SI-02) at section 4 and 7, mark :189 and the new-dataset rows as replaced or dropped, and make the pointer's count match the markers.

### DOC-11 · fix · The amended text does not meet the plain-language rule (SI-67)

**Where:** PREREG.md:229-319 (esp. :264-296); ANALYSIS_SPEC.md (whole); PAPER_PLAN.md:164-173

**Problem:** SI-67 asks for text a genetics PhD student can follow without earlier messages, with labels and acronyms spelled out. PREREG section 9 lines 229-319 contain 29 mentions of SI-numbers, 20 of final-review ids, 6 of design ids, 3 of guard ids and 18 of commit hashes, mostly without a plain gloss; the guard entry (:281-296) carries 7 distinct final-review ids, 3 decision ids, the range 'G-01 to G-14' and 9 distinct commit hashes in one paragraph. The spec has 51 SI, 30 I and 17 D references. Unexplained terms: 'critic-free loss' and 'gradient graph' (:284-286), 'envelope theorem' and 'L1 marginal error' (:267-268), 'no_pair_steps', 'sigma_c' and 'delta_c' (:232-233), 'b_common', 'bstar_low_signal' (defined only in section 4), 'SE', 'tau'. Labels collide: R1-R5 are rules in PREREG but research questions in PAPER_PLAN (the spec uses both); 'cell' is task by decoder (spec:15) and also a single cell ('no reference cell', ':285', '128-cell minibatches' :273); 'family' is a divergence family (W1, JS, MMD, Sinkhorn) and a dataset family (immune, ATAC); 'D' is a difference and a number of datasets.

**Evidence:** outputs/label_counts.json (regex counts over the quoted line ranges at 1e615eb); spec lines 14-19 define only run, point, cell, B, C, b0, c0, b*.

**Suggested fix:** Give each id its meaning in the same sentence on first use per entry (or add a glossary table at the top of section 9), spell out the terms above, and use 'penalty-strength family' vs 'dataset family', 'task-by-decoder block' vs 'cell'.

### DOC-12 · fix · Planned steps are written in the present tense, and the freeze is described as done

**Where:** PREREG.md:261, :274-278, :250, :310, :315; PAPER_PLAN.md:165-166; ANALYSIS_SPEC.md:10, :248

**Problem:** Steps not yet done read as done: 'Sinkhorn rows are refitted and rescored at the tag' (:261), 'are refitted and rescored' (:274), 'is recomputed' (:250); the heading ':310 ... frozen before X1 runs'; 'freezes all of them' (:315); PAPER_PLAN:165 'both frozen by the tag'; spec :10 and :248 'frozen with the tag'. The tag prereg-tier12-v3 has not been made (notebook NB-20261014-19), the redo has not run, and the user has not yet reviewed the diff (SI-52).

**Evidence:** notebook/2026-10-14.md (NB-20261014-19); outputs/commit_dates.txt (1e615eb 'drafts no longer describe the tag as made').

**Suggested fix:** Use 'will be' for every step before the tag and the Sinkhorn redo; add 'planned' or 'done (date, commit)' at the start of each section 9 entry.

### DOC-13 · fix · ANALYSIS_SPEC is still a draft question sheet; the file to be frozen should hold only rules

**Where:** ANALYSIS_SPEC.md:1-12, :29, :36-44, :48-50, :168-170, :291; PREREG.md:316

**Problem:** The file is headed 'draft for the user's choices' (:1) with 'Status: DRAFT' (:3), keeps every unchosen option (e.g. B and C of Q1 at :43-44) and 12 lines with 'recommended'. Lines 9-11 say that after the answers the file and analysis_x1.py 'are committed and frozen with the tag', yet analysis_x1.py was already committed (ff88666, 2026-10-14 08:04). ':29 R3 factor effects | X6, X12 (A3 done)' is stale: A3 exists but its 36 Sinkhorn rows are to be redone (SI-57). There is no Q8 (multiplicity; the spec treats it as the agent's proposal) and the header does not say why. Spec:49-50 says the analysis runs after 'all 3,000 X1 rows'; PREREG:316 and analysis_x1.py (:352-358) also require the 408 X13 baseline rows.

**Evidence:** docs/ANALYSIS_SPEC.md at 1e615eb; outputs/commit_dates.txt.

**Suggested fix:** Before the tag: move options, reasons and open questions to a dated decision log, keep rules only, update the status line and :29, and explain the Q8 gap.

### DOC-15 · fix · Evidence for 'result-identical' is cited loosely and the machine-readable list points at the earlier identity run

**Where:** docs/EQUIVALENT_COMMITS.json:8-9; PREREG.md:253-263, :281-284

**Problem:** EQUIVALENT_COMMITS.json lists fit_identity.csv and score_identity.csv (first run, at 2b52542) as its evidence, not the c3fd92d files on which the PREREG entry relies (all 59 non-Sinkhorn fit rows identical, the 4 Sinkhorn rows differ as intended; 7 of 7 latents identical in 32 columns). PREREG:284 cites NB-20261013-39 for both fits and scores; the scores are NB-20261013-40. The evidence is narrow: one fit per code path at 1 epoch on CPU with one thread, and one latent per scorer path. I verified that no fit or scoring code differs between c3fd92d and 1e615eb (only analysis_x1.py is new and decide_a2_a3.py has 6 added lines), so the proof carries to the candidate. PREREG:257 says the checks are re-run at the candidate commit before it is tagged; if the diff review changes code, that will no longer be c3fd92d or 1e615eb.

**Evidence:** outputs/misc_counts.json (fit_identity*, score_identity*, diff_c3fd92d_to_head_fit_score_code, files_changed_c3fd92d_to_head); docs/EQUIVALENT_COMMITS.json; notebook/2026-10-13.md:462-485.

**Suggested fix:** List the c3fd92d files in the evidence field, cite NB-39 for fits and NB-40 for scores, state the design (one fit per code path, 1 epoch, CPU), and add a pre-tag step 'git diff c3fd92d <tag> over src, fit, plan and scoring files is empty, else re-run the identity check'.

### DOC-10 · note · Open question: b* is computed from the stage's extended point set, but P1 uses only the six registered points

**Where:** ANALYSIS_SPEC.md:53-56; PREREG.md:92-99; scripts/prereg_rules.py:559-570; scripts/analysis_x1.py:359-391

**Problem:** The spec reads b0, b_common and b* from the frozen R4 stage-x1 record. R4 takes b_common from the stage's own fits, which PREREG:92 defines as the family grid 'plus any R4 extension' (:94 says 'its grid points'; r4_stage uses all rows of the experiment), while P1 and the area use only the six registered points (PREREG:99, spec:55-56). If an extension raises an arm's best batch score, b* can lie beyond what that arm's six points reach, and the arm is then 'not evaluable (ii)' in P1. This cannot be checked now (no X1 extension rows exist; X1 results are blinded). Not a defect today; the spec should say that b* is computed after the extension rounds and how often 'not evaluable (ii)' is acceptable.

**Evidence:** scripts/prereg_rules.py:552-573 (r4_stage); PREREG.md:92-99; scripts/analysis_x1.py:359-391 (reads r4['targets']).

**Suggested fix:** Add one sentence to spec section 1 and a pre-stated report line: number of arm-by-task cells not evaluable because b* lies beyond the six points, by cause.

### DOC-14 · note · Two contingencies have no registered procedure

**Where:** PREREG.md:30, :269; CONSTRAINTS.md:412 (SI-61), :440 (SI-65)

**Problem:** (1) SI-65: if scib's NMI-0 edge case occurs, the loud error is kept and the case is 'decided case by case' - no rule says who decides or what is reported. (2) SI-61: hitting the 500,000-iteration Sinkhorn cap 'stops the stage'. PREREG:30 treats infrastructure errors as rerunnable, but a deterministic rerun hits the cap again, and P2 would silently exclude such a fit unless it is recorded as a failure. SI-65 is not cited anywhere in PREREG section 9.

**Evidence:** CONSTRAINTS.md SI-61, SI-65; PREREG.md:30, :269.

**Suggested fix:** Add one sentence each: who decides an NMI-0 case and that the case is listed in the results; whether a cap-hit fit is a failure type counted in P2 or stops the stage permanently.

### DOC-16 · note · Statistical points an outside referee is likely to raise (none is an error; each needs a sentence)

**Where:** ANALYSIS_SPEC.md:130-136, :155-157, :161-170, :263-270; PREREG.md:243-245

**Problem:** (1) P2 rates are taken over each family's own six points, whose placement R1 anchors to the pilot onset of collapse; the spec says this 'makes rates comparable' (:164-166), which holds only approximately. (2) The pooled Clopper-Pearson intervals treat the runs of one dataset and strength as independent. (3) Declarations use each posterior on its own, with no multiplicity control (an agent proposal, :155-157); the Wilcoxon companion gets Holm. (4) The family-level spread tau_f rests on four families, two with two members; its half-normal(0.1) prior may dominate. (5) The per-dataset standard errors come from five runs (leave-one-seed-out) and are treated as known. (6) P1 compares the reference critic at 5 critic steps with the discriminator at 1 step; the update-budget control (X6) exists only for the pooled critic (PREREG:243-245).

**Evidence:** manifest v3 X1 rows by arm and n_critic (outputs/misc_counts.json x1_n_critic); spec lines cited. No literature retrieved for these points (see section 9).

**Suggested fix:** State each limitation in the analysis section and, for (1) and (6), name the sensitivity analysis that addresses it or say none is planned.

### DOC-17 · note · Pilot-informed choices are not labelled, and two pilot figures put to the user were off

**Where:** ANALYSIS_SPEC.md:118-123, :130-134, :161-170; CONSTRAINTS.md:516-522 (SI-75)

**Problem:** The spec's prior scale 0.1 for the arm difference matches the pilot between-task SD (0.109), and the area corner (c0 - 0.10) is justified from pilot ranges (disclosed at :118-123), but the prior is not labelled pilot-informed; the 300 points behind the pilot ranges exclude Sinkhorn, unstated. P2's 5-point margin and 0.25 prior scale were decided after the user saw pilot collapse rates. Those rates were quoted as reference 10 of 54 and 15 of 54, discriminator 8 of 45 ('only 5 of its 6 strengths were in the pilot') and 17 of 54, differences +0.7 and -3.7 points. These reproduce only if the 9 lambda = 0.03 discriminator rows of the R1 extension are left out; with them the unconditioned figure is 8 of 54 and the difference +3.7 points, and all six strengths were in the pilot. The decision (SI-75) is unlikely to change (both differences are within 5 points) but its stated premise was off.

**Evidence:** outputs/pilot_ranges.json; outputs/p2_pilot_counts.json and recompute_p2_pilot_counts.txt (A1 scores + R1 extension scores, collapse = C at least 0.10 below the same seed's lambda = 0 run); question text in transcript frame 619cf68a (read-only).

**Suggested fix:** Add 'pilot-informed' labels to the prior scale, corner and P2 settings; correct the P2 premise in NB-20261014-09/SI-75 and tell the user.

### DOC-18 · note · Ledger housekeeping

**Where:** CONSTRAINTS.md:404-408 (SI-60), :425-430 (SI-63), :468-474 (SI-69); SI-51, SI-52

**Problem:** SI-60: source date 2026-10-14 but added 2026-10-13 (the user answered on 2026-10-13 at 20:03 New York time, which is 2026-10-14 in UTC). SI-63: 'none reads decoder, likelihood and cond' is ambiguous (the arm named none). SI-69: '(the reading R2 and R3 already use)' is attached to the leave-one-seed-out clause but describes the average-first reading (same slip in the question text). SI-52 ('frozen P1-P4 analysis code') and SI-51 ('Sinkhorn parts of R3') are not annotated as narrowed by SI-68 (P3/P4 locked later) and SI-57 (rule order). Otherwise SI-51 to SI-77 each have the user's quoted words, a date and a linked notebook decision (section 7).

**Evidence:** outputs/ledger_vs_notebook.json and check_ledger_notebook.txt.

**Suggested fix:** Correct the four wording items; add 'narrowed by SI-68 / SI-57' to SI-52 and SI-51.

### DOC-19 · note · Number provenance and wording

**Where:** PREREG.md:265-266; PAPER_PLAN.md:74; notebook/2026-10-13.md:28-31; guard review (G-05)

**Problem:** (1) '1,908 manifest rows carry lam = 0' (guard review) counts rows across 8 manifest files (1428 + 240 + 240); the manifest of record has 242 + 40 + 40 = 322 distinct rows, all lam 0 (conclusion unchanged). (2) PAPER_PLAN:74's ratio of the Sinkhorn divergence to exact W1 (0.53 to 0.89) comes from 5 pre-fix latents with 8 minibatches each in a reviewer log, not in the repository. (3) 'Twice the stated method's gradient' (PREREG:266) is the median for A1 only (norm ratio 2.00); A3 is 2.27 and the R1-extension rows 14.44 with median cosine 0.28 (notebook/2026-10-13.md:31). (4) NB-20261013-03 says 236 early/train-mode captures; the validation file has 192 + 76 = 268. (5) 'Rarely in X3 atac_small MMD, X8 sim2 MMD and X1 sim1 MMD and Sinkhorn' (:288-289) corresponds to expected shares of fits with a no-pair minibatch of 14%, 4.5% and 0.6% (model-based, not simulated runs).

**Evidence:** outputs/lam_audit_rerun.txt; outputs/misc_counts.json; outputs/g01_recompute.txt/g01_recomputed_groups.csv; review/sinkhorn_fix_review_20261013/logs/w1_gap.log (read only).

**Suggested fix:** Quote '322 distinct rows (1,908 counting repeats across manifest files)', give the denominators for the F7 range, say 'median 2.0 times on A1 (up to 14 times on the R1-extension rows)', and correct the capture count.


## 4. Claims checked

| # | Claim checked | Status | Evidence |
|---|---|---|---|
| 1 | PREREG section 9, I091 entry (:226-239): R1 depends on trajectory through its bio-noise SD; ratios 3.3 to 5.4; delta_c 0.10 to 0.13; three conditioned families move from frozen to an extension; five unchanged. | **partly** | Numbers reproduce: re-running experiments/final_review_amendments_20261013/r1_without_trajectory.py at 018c8f1 gives a CSV identical to the committed one; ratios 3.30 to 5.41, delta_c 0.102 to 0.127, 3 families change. The Sinkhorn figures come from discarded pre-fix fits and 'the frozen X1 grids stand' is not true for Sinkhorn (DOC-06). |
| 2 | PREREG 2026-10-13 entry (final review I006/D-01/D-02; SI-55): PAPER_PLAN corrected to PREREG on grid edge, P1 comparator, power, Holm. | **partly** | Discriminator n_critic = 1 in all 1,191 manifest rows and X6 pooled n_critic 1/5/10 (27 rows each) reproduce; power figures reproduce exactly; 0.031 > 0.0125 is right for four tests. Issues: power is not that of the frozen P1 (DOC-07); Holm family size differs across documents (DOC-05); the entry's 'user decision SI-55' covers part (a) only (DOC-08). |
| 3 | PREREG 2026-10-13 entry (SI-54, SI-60): fits and scores at c88cce3 are result-identical to the new tag for every arm except Sinkhorn; evidence = identity checks. | **confirmed** | all 59 non-Sinkhorn fit rows identical (63 rows in all), the 4 Sinkhorn rows differ as intended; 7 latents identical in 32 columns (flags only; scores not read); no fit or scoring code differs between c3fd92d and 1e615eb. Notes: citation, evidence list, narrow design (DOC-15). |
| 4 | PREREG 2026-10-13 Sinkhorn entry: gradient counted twice (fixed in 3cbaca4), tolerance 1e-4, float64, cap 500,000 raises, 234 = A1 180 + R1 extension 18 + A3 36, 3,076 validation minibatches, float32 criteria. | **partly** | Counts and constants reproduce: manifest A1 180 + R1 extension 18 + A3 36 = 234; 3,076 minibatches (atac_small 1,180, immune 1,068, sim1 828); TOL 1e-4, MAX_ITER 500000, float64; float32 meets the criterion on the 598 hardest and fails on all 3,076 (committed summary). 'All 234 ... refitted' conflicts with rule order (DOC-02); 'twice' is the A1 median only (DOC-19); GPU timings and validation values were read from committed CSVs, not re-run. |
| 5 | PREREG 2026-10-13 guards entry: none changes a completed fit or score; 59/59 and 7/7; no-pair minibatches expected in X7 pancreas and sim1 and rarely in X3 atac_small, X8 sim2, X1 sim1. | **confirmed** | Recomputed expected no-pair minibatches (hypergeometric, scVI sampler assumptions) reproduce the guard-review table: X7 pancreas mmd_ref 27 fits, 25.9 expected to hit; X7 sim1 18 of 18; X3 atac_small 6.7 of 48; X8 sim2 1.08 of 24; X1 sim1 0.36 of 60. 'Rarely' for X3 atac_small is 14% of fits. Model-based, not simulated runs (DOC-19). |
| 6 | PREREG 2026-10-13 SI-56/SI-66 entry: P1 starts at lambda = 0; low-signal cells not evaluable; R2/R3 keep defaults; the flag cannot clear. | **partly** | The logic holds for immune/conditioned (flagged, difference 0.0188 < 0.02, set by barycenter, not an A2/A3 arm). The entry omits sim1/conditioned (difference 0.0069), is dated before its implementation (f3c3415, 2026-10-14 07:50), and does not say that the R2/R3 record predates the rule (DOC-04). |
| 7 | PREREG 2026-10-14 entry and PAPER_PLAN:164-166 (commit de741a0): the tag freezes the X1 analyses before any X1 adversarial-arm score is examined; the script runs once after all rows are scored or failed. | **partly** | The wording is narrower than the spec's 'before any X1 result exist', which is good, but 19 X1 rows were scored before the freeze and one X1 latent was rescored and compared by script; 'examined' is undefined; the heading and spec:40 still say before X1 runs (DOC-01, DOC-12). |
| 8 | Nothing is described as done that is not done. | **partly** | True and checked: all 14 cited commit hashes exist with matching subjects; f3c3415 and ff88666 implement what is described; analysis_x1.py has functions for each analysis the entry lists and 17 known-answer test functions. Present-tense planned steps (Sinkhorn refit, tag, redone records) read as done (DOC-12); 'A2 and A3 are still fitted' (DOC-04). |
| 9 | Every number quoted in the amendments reproduces from committed inputs. | **partly** | All 32 numbers quoted in PREREG, ANALYSIS_SPEC and PAPER_PLAN that could be recomputed reproduce (23 as quoted, 9 with a wording or scope issue). In the wider change set two do not: the pilot collapse counts put to the user (DOC-17) and the capture count in NB-20261013-03 (236 versus 268, DOC-19); the 1,908 rows of the guard review count repeats across manifest files (322 distinct). Items that could not be recomputed are listed in report section 5 (blinded X1 data, GPU timings, reviewer-log ratios, sysVI latents). |
| 10 | ANALYSIS_SPEC is consistent with PREREG sections 1-5. | **partly** | Consistent: six registered lambda per arm, q = 0.5, delta_min 0.01, collapse bound 0.10, P1 contrast reference minus discriminator at 1 step, real tasks and families. Inconsistent or ambiguous: Holm family (DOC-05), b* from the extended point set (DOC-10), power estimator (DOC-07), label collisions (DOC-11). |
| 11 | PAPER_PLAN section 5 pointer and section 13 list match what the spec replaces. | **partly** | Eight [replaced] markers exist for a pointer that lists seven statements (the ROPE and the measurement-SD study are joined); three markers point to agent-proposed spec sections; unmarked conflicts remain outside section 5 (DOC-08, DOC-09). |
| 12 | The amendments disclose the SI-66 timing and that no existing R2/R3 result changes. | **partly** | True by my re-implementation (no decision changes; defaults stay), but neither fact nor the 2026-10-14 implementation date is stated in PREREG or the notebook summary (DOC-04). |
| 13 | The X1 early rows and the rescored X1 latent are disclosed in the documents. | **wrong** | Not in PREREG, ANALYSIS_SPEC or PAPER_PLAN; the facts are in the notebook (NB-20261012-04/-05, NB-20261013-12/-16/-37/-40) and the identity CSVs only (DOC-01). |
| 14 | The amendments were not driven by X1 outcome data. | **partly** | No amendment in the diff cites an X1 score, and I could not open X1 data (blinded) to check. Several are pilot-driven (I091, SI-56/66, prior scale, area corner, P2 settings): report section 6 tabulates what was seen before each. Not checkable: whether anyone read the 19 X1 scores. |
| 15 | Ledger entries SI-51 to SI-77 each have the user's words, a date and a notebook decision, and are no wider than what the user was asked. | **confirmed** | 27 entries (26 active, 1 retired); each has a quoted source, an added date and a linked notebook decision; 18 have identical text in the notebook and the rest are paraphrased or combined entries that say the same; rules match the question and option text in the lead transcripts. Minor items in DOC-18; SI-66 is the one entry where the user chose the option the agent did not recommend, and the entry says so. |
| 16 | The amendments meet the plain-language rule SI-67. | **wrong** | Id and acronym density, undefined terms and label collisions as counted in DOC-11; the spec's header defines only a few terms. |
| 17 | The stopped local X1 run had 19 scored rows, 17 of them adversarial-arm rows (immune, conditioned decoder, one seed). | **not_checkable** | 19 scored, 1 fitted not scored and 1,097 to fit are in NB-20261012-05; '17 adversarial' is in the lead's brief. The X1 folders are blinded to this review, so I did not recount. |
| 18 | Nobody read the values of the scored X1 rows before the analyses were frozen (the freeze is blind). | **not_checkable** | Only the lead and the user can confirm who looked at values. The committed identity result reports flags (which columns differ), not scores, and the notebook entries record counts; no amendment cites an X1 score (DOC-01). |
| 19 | The Holm example '0.031 > 0.0125, as section 5 states' is consistent across documents. | **partly** | Arithmetic correct for the four registered tests; the spec and code use six tests (0.0083); the conclusion (cannot reject) is unchanged (DOC-05). |

## 5. Numbers recomputed

All values below were recomputed from committed records, the manifest of record (v3, sha256 `6efbe422c8f09f9884001a84f7d7b363506f435803d238280e00bbb7b7e1cf24`, 6,930 rows) and pilot scores only. X1 scores were not opened. `numbers_recomputed.csv` holds the same table.

| ID | Quantity | Quoted in | Quoted | Recomputed | Status | How |
|---|---|---|---|---|---|---|
| N01 | Power, six real tasks: standard error | PREREG.md:246; PAPER_PLAN.md:192 | 0.045 | 0.0450 | reproduced | recompute_power.py |
| N02 | Minimum detectable effect at 80% power, six tasks | PREREG.md:246-247 | 0.131 | 0.1307 | reproduced | recompute_power.py |
| N03 | Power at a 0.02 difference, six tasks | PREREG.md:247 | 0.104 | 0.1044 | reproduced | recompute_power.py |
| N04 | Four families: minimum detectable effect and power | PREREG.md:247; PAPER_PLAN.md:192 | 0.182 and 0.088 | 0.1818 and 0.0878 | reproduced | recompute_power.py |
| N05 | Within-block SD and between-task SD | PREREG.md:248 | 0.030 and 0.109 | 0.0298 and 0.1094 | reproduced, wording issue | recompute_power.py (not the power of frozen P1: DOC-07) |
| N06 | Smallest exact two-sided Wilcoxon p, six datasets | PREREG.md:106, :251 | 0.031 | 0.03125 | reproduced | recompute_power.py |
| N07 | Holm first threshold, four tests | PREREG.md:106, :251; PAPER_PLAN.md:184 | 0.0125 | 0.0125 | reproduced, wording issue | recompute_power.py (spec and code use six: DOC-05) |
| N08 | Holm first threshold, six tests (spec section 15, analysis_x1.py) | ANALYSIS_SPEC.md:271-272 | (number not quoted) | 0.00833 | reviewer addition (not quoted) | recompute_power.py |
| N09 | Power without sim1 (final review I073): SE, minimum detectable effect, power | (not in the amendments) | (not quoted) | 0.0236, 0.0684, 0.182 | reviewer addition (not quoted) | recompute_power.py |
| N10 | Bio-noise SD ratio with / without trajectory, immune conditioned, six arms | PREREG.md:232-235 | 3.3 to 5.4 | 3.30 to 5.41 | reproduced | r1_without_trajectory.py at 018c8f1; i091_summary.py |
| N11 | delta_c range at n = 3 | PREREG.md:235 | 0.10 to 0.13 | 0.102 to 0.127 | reproduced | i091_summary.py |
| N12 | Conditioned families whose R1 grid changes without trajectory | PREREG.md:235-238 | 3 change, 5 unchanged | 3 change, 5 unchanged | reproduced, wording issue | re-run CSV identical to committed (Sinkhorn part from discarded fits: DOC-06) |
| N13 | Pilot: plain-scVI biology score range per cell | ANALYSIS_SPEC.md:120 | 0.67 to 0.90 | 0.6658 to 0.8966 | reproduced | recompute_pilot_ranges.py |
| N14 | Pilot: change in biology score, all 300 points | ANALYSIS_SPEC.md:121 | -0.59 to +0.23 | -0.5904 to +0.2310 | reproduced, wording issue | recompute_pilot_ranges.py (Sinkhorn excluded, unstated: DOC-17) |
| N15 | Pilot: middle half of the 300 points | ANALYSIS_SPEC.md:121 | -0.09 to 0.00 | -0.0936 to +0.0047 | reproduced | recompute_pilot_ranges.py |
| N16 | Pilot: per-cell middle half, extreme cells | ANALYSIS_SPEC.md:121 | -0.19 to +0.09 | -0.1877 to +0.0904 | reproduced | recompute_pilot_ranges.py |
| N17 | X1 manifest rows | ANALYSIS_SPEC.md:50 | 3,000 | 3,000 | reproduced | misc_counts.py |
| N18 | Sinkhorn pilot fits | PREREG.md:274 | 234 = 180 + 18 + 36 | 180 + 18 + 36 = 234 | reproduced, wording issue | misc_counts.py (rule order: DOC-02) |
| N19 | Discriminator rows with n_critic = 1 | PREREG.md:243-244 | every row | 1,191 rows, all n_critic 1 | reproduced | misc_counts.py |
| N20 | X6 pooled-critic rows at 1, 5, 10 steps | PREREG.md:244-245 | 1, 5, 10 | 27 rows each at 1, 5 and 10 | reproduced | misc_counts.py |
| N21 | Sinkhorn validation minibatches | PREREG.md:278-279 | 3,076 | 3,076 (atac_small 1,180, immune 1,068, sim1 828) | reproduced | misc_counts.py (committed CSV) |
| N22 | Batches in the validation tasks | PREREG.md:279 | 3, 10 and 6 | 3, 10 and 6 | reproduced | check_task_facts.py |
| N23 | float32 accuracy criterion | PREREG.md:272-274 | met on the 598 hardest, failed on all 3,076 | SI-58 accuracy criterion met by float32: 598 hardest: yes, all 3,076: NO | reproduced | committed summary read (not re-run) |
| N24 | Sinkhorn constants | PREREG.md:267-269 | tolerance 1e-4, cap 500,000, float64 | 1e-4, 500000, torch.float64 | reproduced | misc_counts.py (alignment.py) |
| N25 | Fit identity at c3fd92d | PREREG.md:282-283 | 59 of 59 non-Sinkhorn identical | 59 of 59 non-Sinkhorn identical; 0 of 4 Sinkhorn identical | reproduced | misc_counts.py (committed CSV) |
| N26 | Score identity at c3fd92d | PREREG.md:283-284 | 7 of 7 latents, 32 columns | 7 of 7 latents identical in 32 columns | reproduced, wording issue | misc_counts.py (flags only; citation NB-39 vs NB-40: DOC-15) |
| N27 | Fit or scoring code changed between c3fd92d and 1e615eb | (supports PREREG.md:257) | (not quoted) | empty unchanged | reviewer addition (not quoted) | misc_counts.py (git diff) |
| N28 | Commit hashes cited in section 9 that exist | PREREG.md section 9 | all | 14 of 14 | reproduced | misc_counts.py, commit_dates.py |
| N29 | Immune cells inside immune_hum_mou | ANALYSIS_SPEC.md:57-58 | all 33,506 | 33,506 of 33,506 | reproduced | check_task_facts.py |
| N30 | Cells shared by atac_small and atac_large | ANALYSIS_SPEC.md:58-59 | 3,618 | 3,618 | reproduced | check_task_facts.py |
| N31 | No-pair minibatches: X7 pancreas reference-anchored MMD | PREREG.md:287-288 | expected | 27 fits, 25.9 expected to hit | reproduced | recompute_g01.py (model) |
| N32 | No-pair minibatches: X7 sim1 reference-anchored MMD | PREREG.md:287-288 | expected | 18 of 18 fits | reproduced | recompute_g01.py (model) |
| N33 | No-pair minibatches 'rarely': X3 atac_small, X8 sim2, X1 sim1 (share of fits) | PREREG.md:288-289 | rarely | 14%, 4.5%, 0.6% | reproduced, wording issue | recompute_g01.py (model; DOC-19) |
| N34 | R2 and R3 decisions with and without SI-66 | PREREG.md:307 ('keep the defaults') | defaults kept | R2 12/12 to 9/12 evaluable, R3 17/20 to 13/20; no switch in either case | reproduced, wording issue | si66_effect.py (own re-implementation; DOC-04) |
| N35 | Blocks flagged low-signal in the frozen R4-a1 record | PREREG.md:304-305 | immune/conditioned: b0 0.6382, b_common 0.6570 | immune|1 0.6382 / 0.6570 (difference 0.0188); sim1|1 0.8120 / 0.8189 (difference 0.0069); other four blocks not flagged | reproduced, wording issue | misc_counts.py (r4_a1_record.json; sim1|1 omitted: DOC-04) |
| N36 | Manifest rows of arms none, scanvi, scvi_adv with lam = 0 | guard review G-05 (not in the amendments) | 1,908 | 1,428 + 240 + 240 counted across 8 files; 322 distinct rows in the manifest of record | reproduced, wording issue | lam_audit.py re-run; misc_counts.py (DOC-19) |
| N38 | Early and train-mode captures in the Sinkhorn validation (12 x 234 = 2,808 final minibatches plus captures) | notebook NB-20261013-03 (not in the three documents) | 236 | 192 + 76 = 268 (final 2,808; total 3,076) | not reproduced as quoted | misc_counts.py (DOC-19) |
| N37 | Pilot collapse counts put to the user (SI-75 question) | question text, not in the documents | reference 10/54, 15/54; discriminator 8/45, 17/54; +0.7 and -3.7 points | reproduce only without the R1 extension rows; with them discriminator 8/54 and difference +3.7 points (conditioned unchanged, -3.7) | not reproduced as quoted | recompute_p2_pilot_counts.py (DOC-17) |

**Could not be recomputed here** (each is read from a committed file or log and so is not independently confirmed):

- X1 row counts for the stopped local run (19 scored, 17 adversarial): the X1 folders are blinded to this review; counts come from the notebook and the lead's brief.
- Identity-check score values: only the flags (which columns differ) in the committed CSVs were read.
- Sinkhorn validation values, GPU timing ratios (0.989 to 1.014) and the float32 versus float64 agreement figures: committed CSVs and summaries read, not re-run (CPU-only rule).
- The ratio of the Sinkhorn divergence to exact W1 (0.53 to 0.89; PAPER_PLAN:74) and the gradient ratios (2.00, 2.27, 14.44): reviewer log and notebook only.
- The 150 sysVI filler latents: only the design count was read (the cluster output folder is blinded).
- G-01 counts are model-based expectations, not simulated runs; they assume the scVI sampler (no dropped last batch) and proportional batch composition of the training split.

## 6. What had been seen when

Events, from the notebook and git (New York time):

```
2026-10-04 17:42 -0400  c88cce3  tag prereg-tier12-v2 (rules, scripts, manifest frozen)
2026-10-13 07:45 -0400  5cb75c0  SI-51/SI-52 recorded (branch final-review-fixes starts)
2026-10-13 14:46 -0400  018c8f1  PREREG: I091 correction (SI-53), PAPER_PLAN corrected (SI-55, D-01, D-02)
2026-10-13 20:30 -0400  b2c68c8  PREREG: SI-54/SI-60 two commits declared result-identical
2026-10-13 21:22 -0400  6d0ca1f  ANALYSIS_SPEC draft (uses A1 pilot figures)
2026-10-14 06:47 -0400  56c0762  ANALYSIS_SPEC round 1 decided (SI-68..SI-72)
2026-10-14 06:58 -0400  990d37d  ANALYSIS_SPEC round 2 decided (SI-73..SI-77)
2026-10-14 07:50 -0400  f3c3415  decide_a2_a3.py implements SI-56/SI-66
2026-10-14 07:54 -0400  4d86a55  PREREG: Sinkhorn, guards, P1/SI-66, X1 analyses entries; PAPER_PLAN notes
2026-10-14 08:04 -0400  ff88666  scripts/analysis_x1.py committed
2026-10-14 08:48 -0400  de741a0  PREREG/PAPER_PLAN tag sentences reworded as commitment
2026-10-14 08:48 -0400  1e615eb  branch head reviewed

2026-10-12 11:43  NB-20261012-01  R2/R3 frozen at c88cce3 (A2/A3 pilot results read)
2026-10-12 12:40  NB-20261012-04  local X1 launched (1,125 fits)
2026-10-12 16:36  NB-20261012-05  local X1 stopped: 19 scored, 1 fitted not scored, 1,097 need fit
2026-10-13 03:28  NB-20261013-01  final review report: NOT READY
2026-10-13 11:58  NB-20261013-09  R4-a1 sensitivity on the A1/A2/A3 records
2026-10-13 14:50  NB-20261013-12  identity check 1: 7 latents rescored incl. local X1 immune discriminator
2026-10-13 17:08  NB-20261013-16  identity check 1 vs production score rows
2026-10-13 20:53  NB-20261013-35  SI-66 decided
2026-10-13 21:47  NB-20261013-40  identity check 2 at c3fd92d: same 7 latents rescored
2026-10-14 06:45  NB-20261014-01  SI-68..SI-72 decided (round 1)
2026-10-14 06:57  NB-20261014-07  SI-73..SI-77 decided (round 2)
2026-10-14 07:50  NB-20261014-14  ISSUE: SI-56/SI-66 never implemented in the rule code
2026-10-14 07:50  NB-20261014-15  SI-56/SI-66 implemented
```

For each amendment: the data that existed before the decision and whether any result or decision changed. This is the table I recommend for PREREG section 9 (DOC-03).

| # | Amendment (PREREG section 9) | Decided (user) | Committed | Pilot or X1 data seen before the decision | Result or decision changed? |
|---|---|---|---|---|---|
| 1 | R1 depends on trajectory (I091; SI-53) | 2026-10-13 11:46 | 018c8f1, 10-13 14:46 | A1 scores (bio-noise SDs) | Only as a sensitivity analysis: three conditioned families would extend; frozen grids stand except Sinkhorn |
| 2 | PAPER_PLAN corrected: grid edge, comparator, power, Holm (SI-55, D-01, D-02) | 2026-10-13 11:46 | 018c8f1 | A1 record (power figures) | No rule changes |
| 3 | Two commits declared result-identical (SI-54, SI-60) | 2026-10-13 11:46 and 20:03 | 8ce5a54 (10-13 20:10), b2c68c8 (20:30) | No X1 score cited; the identity check rescored one X1 adversarial latent by script | No |
| 4 | Sinkhorn fix and redo in rule order (SI-51, SI-57, SI-58, SI-61, SI-62) | 2026-10-13 07:45 to 20:48 | 3cbaca4 (09:56), 89d5e2c (19:46); text 4d86a55 (10-14 07:54) | Pilot Sinkhorn fits (review probes); no X1 score cited | Yes: Sinkhorn A1, R1, R4-a1, A3, R2/R3 are redone; the X1 Sinkhorn grid will change |
| 5 | Guards before the tag (SI-52, SI-63, SI-64; G-01 to G-14) | 2026-10-13 07:45 to 20:48 | c4be14b to c3fd92d, 10-13 10:09 to 21:00 | No score cited | No completed fit or score changes (59/59 fits, 7/7 latents) |
| 6 | P1 from lambda = 0; low-signal cells not evaluable (SI-56) | 2026-10-13 11:46 | f3c3415, 10-14 07:50 | A1 R4 record (flags come from pilot data) | No: R2/R3 keep defaults |
| 7 | Low-signal cells count as not evaluable in R2/R3 (SI-66) | 2026-10-13 20:53 | f3c3415, 10-14 07:50 | A2/A3 results (R2 and R3 were run on 10-12) | No decision changes (DOC-04) |
| 8 | X1 analyses specified (SI-68 to SI-77) | 2026-10-14 06:45 to 06:57 | 56c0762, 990d37d, ff88666 (08:04), de741a0, 1e615eb | A1 pilot ranges and collapse rates; 19 X1 rows existed | New rules, nothing existing changes |

## 7. Ledger against notebook (SI-51 to SI-77)

27 entries (26 active, 1 retired). Every entry has the user's quoted words, an added date and a linked notebook decision by the user; 18 entries have text identical to the notebook, the others are combined or paraphrased entries that say the same
(SI-51 to SI-56 in NB-20261013-02 and -08; SI-57 to SI-59 in NB-20261013-10, -14, -15). The rules match the question and option text the user saw (read from the lead transcripts, which I used read-only). Differences found:

1. **SI-60 (CONSTRAINTS.md:404-408).** Source date 2026-10-14, added 2026-10-13; the user answered on 2026-10-13 at 20:03 New York time (2026-10-14 in UTC). Harmless, but the two dates disagree.
2. **SI-69 (:468-474).** '(the reading R2 and R3 already use)' is attached to the leave-one-seed-out clause; it describes the average-first reading (the question text had the same slip).
3. **SI-63 (:425-430).** 'none reads decoder, likelihood and cond' is ambiguous: `none` is also the name of the plain-scVI arm.
4. **SI-51 and SI-52** are not annotated as narrowed by SI-57 (rule order instead of 'redo all 234') and SI-68 (P3/P4 locked later instead of 'frozen P1-P4 analysis code').
5. **SI-65 and SI-61 (cap hit stops the stage)** have no registered follow-up procedure (DOC-14); SI-65 is not cited in PREREG section 9.
6. **No Q8.** The spec's questions run Q1-Q7, Q9-Q11; the multiplicity rule (the missing Q8) is an agent proposal (DOC-08, DOC-13).
7. **SI-66** is the one entry where the user chose the option the agent did not recommend (its source quote has no '(Recommended)' tag); PREREG:302 records the choice, and the user was told R2/R3 could not switch while the flag holds.
8. **SI-75's premise** (pilot collapse counts quoted to the user) was off (DOC-17).
9. **NB-20261013-03** says 236 early and train-mode captures; the validation file has 192 + 76 = 268 (DOC-19).
10. **SI-59 is retired** (cap 500,000 with a stop on any non-finite error) and replaced by SI-61 (stop on a NaN error only); its retirement note says so and PREREG:269-271 follows SI-61.

## 8. Plain-language check (SI-67)

The amended text fails the rule in three ways.

1. **Labels without meaning.** PREREG section 9 lines 229-319 mention SI-numbers 29 times, final-review ids 20, design ids 6, guard ids 3 and commit hashes 18 times. ANALYSIS_SPEC: 51 SI, 30 I and 17 D references.
2. **Undefined terms.** 'critic-free loss', 'gradient graph', 'envelope theorem', 'L1 marginal error', 'no_pair_steps', 'sigma_c', 'delta_c', 'SE', 'tau'.
3. **Colliding words.** R1-R5 (PREREG rules versus PAPER_PLAN research questions), 'cell' (task by decoder; a single cell), 'family' (penalty-strength family; dataset family), 'D' (a difference; a number of datasets).

What works: the spec defines run, point, cell, B, C, b0, c0 and b* at its top (lines 14-19) and gives each decided question a plain reason.

Example rewrite of PREREG:284-287 (guard I033): *Some penalties (MMD, Sinkhorn) compare two batches inside each small training step. If a step contains cells from only one batch there is nothing to compare; the code now adds a zero penalty for that step instead of stopping the fit, and counts how often it happens (final-review issue I033, commit c4be14b; counts are reported with the results).*

## 9. Prior work and literature scope

No novelty check was run: the amendments claim no new method or result. Guidance retrieved (web search, two queries; abstract or summary pages only, full texts not read) to support the recommended fixes for DOC-01, DOC-03 and DOC-04:

- van den Akker O, Weston S, Campbell L, et al. (2021) Preregistration of secondary data analysis: a template and tutorial. Meta-Psychology 5:2625, https://doi.org/10.15626/MP.2020.2625. State prior knowledge of data that already exist.
- Willroth EC, Atherton OE (2024) Best laid plans: a guide to reporting preregistration deviations. Advances in Methods and Practices in Psychological Science, https://journals.sagepub.com/doi/10.1177/25152459231213802. A standard deviations table.
- Lakens D (2024) When and how to deviate from a preregistration. Collabra: Psychology 10(1):117094, https://online.ucpress.edu/collabra/article/10/1/117094/200749/When-and-How-to-Deviate-From-a-Preregistration. Deviations are acceptable if clearly indicated; line-numbered preregistrations help.

No literature was retrieved for the statistical points in DOC-16; they are stated as questions, not as prescriptions.

## 10. Checks run, limits and deviations

1. recompute_power.py: power, minimum detectable effect, SE, sigma-hat, tau-hat from docs/prereg/r1_record.json, with and without sim1; exact Wilcoxon minimum p; Holm thresholds for 4, 6 and 8 tests
2. recompute_pilot_ranges.py: plain-scVI C range, change-in-C ranges and percentiles over the 300 non-Sinkhorn pilot points (A1 scores)
3. r1_without_trajectory.py re-run at 018c8f1 (git archive) against the A1 and extension scores: CSV identical to the committed one; sigma_c ratios and delta_c range derived from it
4. misc_counts.py: manifest v3 row counts per experiment and arm, 234 = 180 + 18 + 36, n_critic values, sysVI settings, R4-a1 record blocks, Sinkhorn validation counts and constants, identity CSV counts, cited-commit existence, fit/score code diff c3fd92d to 1e615eb
5. recompute_g01.py: expected no-pair minibatches per experiment, task and arm (hypergeometric model of the scVI sampler)
6. lam_audit.py re-run in the clone: lam values per arm across manifest files
7. si66_effect.py: R2/R3 criteria of PREREG section 3 re-implemented on docs/prereg/r2r3_record.json with and without SI-66
8. recompute_p2_pilot_counts.py: pilot collapse counts quoted to the user, with and without the R1 extension rows
9. check_ledger_notebook.py: SI-51 to SI-77 entries compared with their notebook decisions (date, author, text)
10. timeline.py and commit_dates.py: decision and commit times; all cited commits exist
11. check_task_facts.py: batch counts, cell counts and overlaps of the eight tasks from the prepped obs tables
12. line-by-line reading of the diff 5cb75c0..1e615eb for PREREG.md, ANALYSIS_SPEC.md, PAPER_PLAN.md, CONSTRAINTS.md, EQUIVALENT_COMMITS.json, notebooks of 2026-10-12 to 2026-10-14; plus prereg_rules.py r4_stage and analysis_x1.py structure
13. web search for guidance on reporting preregistration deviations and on preregistering analyses of existing data (3 sources)

Limits and deviations from the brief:

- I worked in my own clone (`work/rv`, detached at 1e615eb, origin removed) because the sandbox masks the worktree's git pointers; the trajectory-free R1 check ran on a `git archive` copy of 018c8f1 (`work/rv_018c8f1`).
- Blinding was respected: I did not open `tier12_v2/X1_local/`, `jhpce_tier12/`, `review_sealed/`, the X1 runner logs or the two identity review folders; the identity CSVs read are the committed ones in `experiments/integration_checks_20261013/results/`, which hold only flags. Pilot scores (A1 and its R1 extension) and committed records were read; the A2/A3 decisions were checked from the committed record `docs/prereg/r2r3_record.json`.
- The repository script `decide_a2_a3.py` refuses the c88cce3 Sinkhorn A3 rows, so the SI-66 effect comes from my own re-implementation of PREREG section 3 on the record's cells, not from re-running the repo script.
- The `ponytail` skill is not in this profile's catalogue, so the minimal-change check was not run; I changed no project code.
- No rigor ledger exists in this project and I was told not to ask; I did not create one. `rigor_ledger_entries_DOC.json` holds the 19 findings in ledger format for merging.
- No sub-agents were used.

## 11. Files

- `REPORT.md` (this report); `numbers_recomputed.csv`; `rigor_ledger_entries_DOC.json` (findings in ledger format, for merging).
- `claim_receipt.json`: claim-gate receipt for the headline numbers (values read back from the output files with hashes).
- `scripts/`: `run_all.sh` (one command re-runs every check, about 80 seconds, one process at a time), the check scripts, `build_report.py`, `findings.py`, `report_template.md`.
- `outputs/`: every output the report cites (`*.txt`, `*.json`, `*.csv`); `outputs/superseded_first_run/` holds first-pass copies kept for audit.
- `rigor_docs_review_bundle.tar.gz`: scripts and outputs together.

Re-run (environment `scvi-api`, one process at a time): create the clone (`git clone <path of code/wcd_final-review-fixes> work/rv; git -C work/rv checkout --detach 1e615eb; git -C work/rv remote remove origin`), then `bash scripts/run_all.sh` (about 80 s, ends by running `python scripts/build_report.py`). The claim receipt is produced inside the session with the claim-gate helpers and is not part of `run_all.sh`. The scripts live in the review folder only; nothing was committed to the project repository (read-only rule).
