# Independent re-check, track 3 of 3: the document fixes

Reviewer: Rigor Reviewer. Date: 2026-10-14. Commit checked: `08f1e3c39ea449677cea7e34f6201925d18f4b84` (branch `final-review-fixes`, tree `bfc3238d`). Scope: the 18 document commits 38b65b0..fd9c8e3 (findings DOC-01 to DOC-19 except DOC-13, and the document parts of CF-03, CF-04, CF-06, CF-07), commit 4af5069 (ANALYSIS_SPEC.md reduced to rules, finding DOC-13), commit 06345ce, and notebook entries NB-20261014-20 to -22 and -29 to -35.

## 1. Answer

**Verdict: PASS_WITH_FIXES.** No blocker: nothing in the documents gives a wrong result or breaks a decision. Five things must change before the tag `prereg-tier12-v3` is made (DOC-R01 to DOC-R05); nine further points are notes.

| What | Result |
|---|---|
| Findings DOC-01 to DOC-19 of the first review | 16 closed; 3 closed in part (DOC-01, DOC-11, DOC-18); 0 not closed |
| Document parts of CF-03, CF-04, CF-06, CF-07 | all 4 closed |
| New points, DOC-R01 to DOC-R14 | 5 to fix before the tag, 9 notes, 0 blockers |
| Numbers recomputed from committed records, manifests, prepped tables and pilot scores | 73; 72 agree with the text; 1 is imprecise wording (DOC-R06) |
| Evidence scripts | 10 script files (01 to 09, with 03 and 03b); re-running them from a clean state reproduces every output file (only test timings differ) |

### 1.1 What must change before the tag

| Id | Problem in one sentence | Smallest fix |
|---|---|---|
| DOC-R01 | PREREG says an audit of every chat record found no X1 score value displayed or analysed, but the audit entry it rests on searched for reads of one folder (`tier12_v2/X1_local/scores`) and its outputs. | Say what was searched, in the notebook's words, or extend the search to the other places where X1 and X13 scores exist; keep the sentence that the audit cannot show that nobody opened a file. |
| DOC-R02 | The disclosure covers the 19 local X1 rows but not the 250 earlier filler fits of X1 (none, scvi_adv) and X13 (scANVI, sysVI) arms, 150 of which were scored by 2026-10-07 with metric values not inspected, nor the X13 row rescored in the identity check; 'examined' covers adversarial arms only although SI-68 says 'an X1 result'. | Add two or three sentences with counts and dates; widen 'examined' to every X1 and X13 score or say why not; say the 19 early rows are kept (SI-54). |
| DOC-R03 | ANALYSIS_SPEC.md:58-59 says the analysis 'never recomputes' b0, b_common, b* and the low-signal flag; section 15.1 and `analysis_x1.py:369` recompute them and stop if they differ. | Replace the sentence by the section 15.1 rule. |
| DOC-R04 | Numbers in PREREG, the spec and its appendix (no-pair expectations, P2 recount, pilot ranges, '0.011', 'coverage 0.91 to 1.00', F7 ratios) come from scripts that are only in review artifacts; NB-20261014-31 cites `scripts/recompute_g01.py`, which is not in the repository. | Commit one script and output per number group and cite the path; or label the number 'from the review's simulation, not reproducible from this repository'. |
| DOC-R05 | The status lines (PREREG.md:509, ANALYSIS_SPEC.md:11-12) and table row 8 name the first code and spec commits but not 8b90d79, 6aaa3ac, 56febe4, 4af5069, which changed the code by +162/-46 lines and the spec; row 5 omits SI-65; the spec says its sources are 'SI-01..SI-67'. | Name the later commits as revised after the independent review and not yet approved; add SI-65 to row 5; correct the range to SI-77. |

The verdict rule of the task: PASS needs every finding closed or properly deferred and no new fix or blocker. Items marked 'fix' are changes before the new tag; 'note' items are optional. Statuses used below: closed = the fix does what the finding needed and I verified it; partly = something remains (stated); deferred, not_closed and not_checkable did not occur at finding level.

## 2. Terms used in this report

| Term | Meaning |
|---|---|
| DOC-nn, CF-nn, AN-nn | Findings of the first independent review of 2026-10-14 (DOC = documents, CF = other code fixes, AN = X1 analysis code). CF findings are checked here only for their document parts. DOC-Rnn = new points found in this re-check. |
| SI-nn | Entry nn of the project's decision ledger, `CONSTRAINTS.md`. SI-68 to SI-77 are the user's decisions of 2026-10-14 on how the main experiment will be analysed. SI-67 is the user's rule that text must be readable by a genetics PhD student who has not seen earlier messages. |
| NB-yyyymmdd-nn | Entry nn of the lab notebook of that date (New York time); every entry carries a hash of the previous one. |
| PREREG, PAPER_PLAN, ANALYSIS_SPEC | `docs/PREREG.md` (the pre-registration: frozen rules R1 to R4 and dated amendment entries in its section 9), `docs/PAPER_PLAN.md` (the study plan), `docs/ANALYSIS_SPEC.md` (the rules for the main analysis; its Appendix A is the decision log). |
| X1, X13, A1 to A3, pilot | X1 is the main experiment (8 datasets called tasks, 2 decoder types, 5 seeds, 3,000 model fits). X13 fits standard integration methods for comparison. A1 to A3 are the pilot experiments that set the rules. |
| arm | One method in the comparison: none (plain scVI), scvi_adv, discriminator, reference, pooled, barycenter, mmd, sinkhorn (the six adversarial arms are discriminator, reference, pooled, barycenter, mmd, sinkhorn). |
| R1 to R4 | The four pre-registered decision rules (R1 sets the penalty-strength grid, R2 and R3 choose two defaults, R4 matches methods at one target batch-removal score b*). PAPER_PLAN uses R0 to R5 for research questions. |
| 'examined' | Proposed definition (PREREG.md:500-503, not yet approved by the user): a score value is examined when a person reads it or an analysis script receives it; integrity checks that print only counts or flags do not count. |
| tag prereg-tier12-v3 | The git tag that will freeze all rules; it has not been made. |
| diff review | The user's review of the full diff of this branch before the tag (SI-52). |
| filler fits | X1 (none, scvi_adv) and X13 (scANVI, sysVI) fits that depend on no rule and were run early on the cluster; their scores were computed on this machine. |

## 3. Result for each finding of the first review

For every finding: the status, the commits that fixed it, the new text with file and line at 08f1e3c, what I checked, and what remains.

**DOC-01 - partly.** Disclose the 19 X1 score files of 2026-10-12 and the rescored X1 latent; define 'examined'; stop saying 'before X1 runs'.  
Commits: 43fdc04 (PREREG); 4af5069 (spec status line); 06345ce; notebook NB-20261014-21 (828f3e7).  
New text: PREREG.md:483-503 (the local X1 run of 2026-10-12: 19 rows scored, 1 fitted but not scored, 1,097 still to fit; 17 adversarial-arm rows, all immune, conditioned decoder, seed 0, and 2 rows of the arms none and scvi_adv; three count-only integrity checks; the rescored row X1_immune_discriminator_lg1_c1_s0_122afe99, equality only; the limit of the audit), :500-503 (definition of 'examined', marked 'not yet approved'), heading :505-506 ('will be frozen before any X1 adversarial-arm score is examined'), status :509; ANALYSIS_SPEC.md:11-15; PAPER_PLAN.md:178, :185.  
Checked: Counts and wording agree with NB-20261014-21 and NB-20261012-04/-05. No score file was opened (blinding).  
Remaining: Disclosure is incomplete: DOC-R01 (the audit sentence is broader than the search the notebook entry describes) and DOC-R02 (earlier X1/X13 filler scores and the rescored X13 sysVI row are not disclosed; 'examined' covers adversarial arms only although SI-68 says 'an X1 result'); DOC-R14 (hash list, external time stamp). The user's confirmation that nobody read a value is correctly left open (PREREG.md:498-500).

**DOC-02 - closed.** State the Sinkhorn redo in rule order (SI-57), not as a fixed list of 234 refits.  
Commits: 3cfc3a7; table row 4 in 81e8017.  
New text: PREREG.md:397-413 (refit and rescore the 180 A1 fits first; then R1 with the extension rows it asks for, R4 stage a1, the A3 rows it resolves, R2 and R3 in full; the extension and A3 rows are 'not a fixed list: before the fix there were 18 and 36 of them'); table row 4 :209; :300; :313-314.  
Checked: Manifest of record: Sinkhorn A1 180 + R1 extension 18 + A3 36 = 234 (script 02, claim 34).

**DOC-03 - closed.** Replace 'nothing in this file is chosen after seeing data' by a status paragraph and a table of what existed when each amendment was decided.  
Commits: 81e8017.  
New text: PREREG.md:3 (version-3 status), :5 (file list), :154-158 (reading rule), :204-213 (table: decided / committed / data behind the decision / result changed).  
Checked: All 15 commit times and all 12 decision times in the table equal git log and the notebook (New York time) (script 02, claims 72-73); table has 6 columns in every row.  
Remaining: Small gaps: DOC-R05 (row 5 omits SI-65; row 8 omits later code commits), DOC-R09 (the commit column does not state its time zone).

**DOC-04 - closed.** SI-56/SI-66 entry: give both dates, name both flagged blocks, say the R2/R3 record predates the rule.  
Commits: 5227d0b; table rows 6-7 in 81e8017; d50348b.  
New text: PREREG.md:454-482 and table rows 6-7 (:211-212): decided 2026-10-13 (SI-56 11:46, SI-66 20:53); rule code f3c3415 (R2/R3, 2026-10-14 07:50) and ff88666 (P1, 08:04); flagged blocks immune/conditioned (b0 0.6382, b_common 0.6570) and sim1/conditioned (0.8120, 0.8189); the stored R2/R3 record was made at c88cce3 before the rule.  
Checked: Recomputed from the stored cells (script 02): R2 12/12 evaluable, -0.0014 without the marking, 9/12, -0.0042 with it; R3 17/20, +0.0010, and 13/20 (14 needed), -0.0014; both keep their defaults; b0 and b_common of the two blocks match the R4-a1 record; Sinkhorn sets b_common in no block.

**DOC-05 - closed.** Use one Holm family size (six primary results, first threshold 0.05/6 = 0.0083) everywhere.  
Commits: 38b65b0.  
New text: PREREG.md:342-343 (0.0125 for the four tests of section 5, 0.0083 for the six primary results), :106 (four-test value kept only where four tests are meant); PAPER_PLAN.md:196; ANALYSIS_SPEC.md section 5.  
Checked: 0.05/6 = 0.00833; smallest exact two-sided Wilcoxon p with six datasets = 2/2^6 = 0.03125 (scipy), above both thresholds and below 0.05.

**DOC-06 - closed.** 'The frozen X1 grids stand' holds for every arm except Sinkhorn; label the pre-fix Sinkhorn figures; name the redo records.  
Commits: 631cdfd.  
New text: PREREG.md:312-315 ('The frozen X1 grids stand for every arm except Sinkhorn'; r1_record.json 'still holds the superseded Sinkhorn grid'); names of the three redo records at :409-413, 'names proposed for the diff review'.  
Checked: r1_record.json has 8 grids, 6 of them non-Sinkhorn (script 02).  
Remaining: Approval of the three record names is left to the user (ANALYSIS_SPEC.md:496-497 lists only the R1 record name; see DOC-R13).

**DOC-07 - closed.** Say which contrast the power figures belong to (six pilot blocks incl. sim1) and add the without-sim1 set.  
Commits: f9ffec3; 4af5069 (spec part).  
New text: PREREG.md:328-340: figures are those of the matched-lambda contrast on all six pilot blocks, including sim1 and the two low-signal blocks, not the power of the frozen P1; without-sim1 set added; recompute planned after the Sinkhorn redo.  
Checked: power_without_sim1.py regenerates its JSON byte for byte (sha256 0be883be...); every quoted figure matches (SE 0.045, MDE 0.131, power 0.104; four families 0.182 / 0.088; without sim1 SE 0.0236, MDE 0.068, power 0.182; four families 0.095 / 0.137; sigma 0.030 with 12 df, tau 0.109 with 4 df; without sim1 sigma 0.033 / 8 df, tau 0.056 / 2 df; sim1 share 0.6661 = 67 %).  
Remaining: Risk when the power is recomputed: DOC-R12 (power_report stops with an error for some standard errors).

**DOC-08 - closed.** Only SI-68 to SI-77 are user decisions; label every agent proposal; narrow SI-55 to part (a).  
Commits: 43f7386; 06345ce; 4af5069 (spec part); 3ced541.  
New text: PREREG.md:319 (SI-55 for part (a)), :507, :513; ANALYSIS_SPEC.md:3-10; PAPER_PLAN.md:176-203, :217 (each replaced statement that is a proposal says 'the agent's proposal ... pending the diff review').  
Checked: Compared with the ledger lines SI-55, SI-68, SI-73, SI-75, SI-77 (section 4.5 of the report).  
Remaining: Two small inconsistencies: DOC-R07 and DOC-R08.

**DOC-09 - closed.** PAPER_PLAN: dated 'superseded' notes for the counts, tasks, new datasets, barycenter, scaling; marker count.  
Commits: 3ced541.  
New text: PAPER_PLAN.md:122 and :224 ('Superseded in part, 2026-10-14'), :180 (seven statements replaced, marked eight times), :217.  
Checked: Eight [replaced] markers found for seven statements, as :180 says. Manifest of record counts (A1 1,098; A2 108; A3 180; X1 3,000; X3 828; X6 216; X7 216; X8 504; X12 288; X13 408; X15 84; 6,930 rows, sha256 6efbe422...) and the eight tasks' cells and batches (pancreas 16,382 / 9 ... sim2 19,318 / 16) equal the code's TASKS table, the prepped tables and the fingerprints (script 02).

**DOC-10 - closed.** State that b* is computed after any extension round, and report how many arm-by-task cells are not evaluable because the six points never reach b*.  
Commits: 4af5069 (spec); fixer's AN-09 text.  
New text: ANALYSIS_SPEC.md:68-71.  
Checked: Read against PREREG section 4 and prereg_rules.r4_stage (all rows of the stage) and the six-point rule of P1.  
Remaining: A sentence two bullets earlier contradicts the code (DOC-R03).

**DOC-11 - partly.** Plain language (SI-67): glossary, labels explained at first use, 'penalty-strength family' vs 'dataset family', 'task-by-decoder block' vs 'cell'.  
Commits: fd9c8e3; 06345ce; 4af5069 (spec word list).  
New text: PREREG.md:159-200 (glossary of labels and terms), :185-187 (the two senses of 'family' and of 'cell'), :194 (R1-R4 are rules; PAPER_PLAN's R0-R5 are research questions); ANALYSIS_SPEC.md:18-27 (word list).  
Checked: Read every new paragraph of PREREG section 9, PAPER_PLAN section 5 and ANALYSIS_SPEC sections 0-16 as a reader without the earlier chat (section 4.9 of the report).  
Remaining: DOC-R09: the spec does not say that its R0/R3/R5 differ from R1-R4; scvi_adv and X13 are not defined; the glossary row 'A1, A2, A3, X1' reads as if X1 were a pilot; 'chat record', 'By file name', 'as recommended' are not explained.

**DOC-12 - closed.** Tense: 'will be' for every step not yet done; a status line at the start of each 2026-10-13 and 2026-10-14 entry.  
Commits: a828f41.  
New text: PREREG.md:154-158 (rule), status lines :298-300, :320, :347-348, :456-457, :509; tag 'has not been made' (:175, :517-518); redo, R4 stage-x1 record, X1 runs and approvals written as planned; ANALYSIS_SPEC.md:11-15; PAPER_PLAN.md:178.  
Checked: Scanned all three documents for every step named in the task (script 07; section 4.4 of the report).  
Remaining: Two slips: DOC-R10 (spec:62-67, present tense for a check that is planned and manual) and DOC-R05 (status lines name only the first code commits).

**DOC-13 - closed.** Reduce ANALYSIS_SPEC to rules; move unchosen options, reasons and open items to Appendix A.  
Commits: 4af5069.  
New text: ANALYSIS_SPEC.md:1-364 (rules), :365-498 (Appendix A: A.1 questions, A.2 why there is no Q8, A.3 open items).  
Checked: Script 01 (section 4.1 of the report): of 104 lines removed from sections 0-16 (103 non-blank), 66 reappear verbatim in Appendix A, 2 remain in the body re-wrapped, 35 are the rewritten title, status line, headings, two section-1 bullets and references; all 105 body edits read, none changes a rule; the four sensitivity variants are defined in the body; A.2 is accurate.  
Remaining: A.3 is not the complete list of open items (DOC-R13).

**DOC-14 - closed.** Say what happens at an NMI-0 case (SI-65) and at an iteration-cap hit (SI-61).  
Commits: 6746f9f.  
New text: PREREG.md:391-396 (cap hit: neither scored nor recorded as failed, the stage stops, no registered outcome; proposal: the stage stays stopped until the user decides whether it counts as a failed fit in P2) and :444-447 (NMI of 0: scib's error kept; proposal: the user decides each case and every case is listed with the results); both labelled 'Proposal for the diff review'.  
Checked: Compared with SI-61 and SI-65 in CONSTRAINTS.md.  
Remaining: The user's approval at the diff review (the two proposals are not in ANALYSIS_SPEC.md A.3; DOC-R13).

**DOC-15 - closed.** EQUIVALENT_COMMITS.json must name the c3fd92d identity files, not only the first run.  
Commits: 9b30d58.  
New text: docs/EQUIVALENT_COMMITS.json:8-13 (fit_identity_c3fd92d.csv, score_identity_c3fd92d.csv, run_c3fd92d.sh first; the 2b52542 files listed as superseded); valid_for_a_candidate_commit_if gives the pre-tag diff step; PREREG.md:355-365.  
Checked: JSON loads; commit_ref resolves to c88cce3c...; all evidence files exist; fit_identity_c3fd92d.csv has 63 rows, 59 of 59 non-Sinkhorn identical, 4 Sinkhorn differ; 0 files changed since c3fd92d in the criterion paths; tests/prereg/test_provenance_rule.py 13 passed. Score-identity file read for column names only.

**DOC-16 - closed.** State the limitations of the analysis next to the rules.  
Commits: 56febe4 (spec section 16); 4af5069.  
New text: ANALYSIS_SPEC.md:332-357 (seven limitations: standard errors treated as known; P1 depends on the lambda spacing; P2 comparability and independence; no multiplicity control for the Bayesian declarations; tau_f from four families; pairs-first independence; 5 versus 1 critic steps), each saying whether a sensitivity analysis exists.  
Checked: Manifest: n_critic 5 for reference, pooled and barycenter, 1 for the discriminator; the update-budget control X6 exists only for the pooled critic.  
Remaining: '0.011' (:340) and 'coverage 0.91 to 1.00' (:345) come from the first review's simulation; not re-run, not reproducible from the repository (DOC-R04).

**DOC-17 - closed.** Label the prior scale, corner and P2 settings 'pilot-informed'; correct the P2 premise quoted to the user.  
Commits: 4af5069; 44af4de (NB-20261014-22).  
New text: ANALYSIS_SPEC.md:135 (prior scale 0.1, between-task SD 0.109 with 4 df), :182-185 (P2 margin and prior scale 'chosen after the pilot collapse rates were shown'; one quoted rate miscounted; whether SI-75 stands is an open item), :435-436 (corner; the 300 points are the 5 non-Sinkhorn arms x 6 cells x 10 lambda), :460-463 (appendix: quoted 8 of 45 is 8 of 54, difference +3.7 not +0.7); notebook/2026-10-14.md:251-258.  
Checked: Script 04, pilot A1 and R1-extension scores, Sinkhorn dropped, collapse = C at least 0.10 below the same seed's lambda = 0 run: without the 9 lambda 0.03 rows reference 10/54, discriminator 8/45, +0.7 points; with them 8/54, +3.7; conditioned decoder 15/54 vs 17/54, -3.7; 0 of the 9 extension rows collapsed.  
Remaining: Whether SI-75 stands is left to the user (ANALYSIS_SPEC.md:184-185, :463; A.3 item 1).

**DOC-18 - partly.** Annotate SI-60 (UTC date), SI-63 ('none' is an arm), SI-69 (parenthetical), SI-51 and SI-52 (narrowed by SI-57 and SI-68) without changing the user's words.  
Commits: 33ab290 (NB-20261014-34).  
New text: notebook/2026-10-14.md:401-412 (all five annotations; the ledger tool cannot hold them, which the entry says).  
Checked: Annotation texts match CONSTRAINTS.md. SI-60 is the only entry among SI-51 to SI-77 whose source date differs from its added date (user answered 2026-10-13 20:03 New York time = 2026-10-14 UTC). CONSTRAINTS.md is identical to the artifact given and no commit touched it since 1e615eb.  
Remaining: A reader of CONSTRAINTS.md does not see the annotations and nothing in the ledger or the PREREG glossary points to NB-20261014-34. Smallest fix, if the user agrees: add NB-20261014-34 to the notebook field of SI-51, SI-52, SI-60, SI-63 and SI-69 (a field the ledger tool already renders).

**DOC-19 - closed.** Number provenance: gradient ratio, validation minibatches, no-pair shares, F7 ratios.  
Commits: bc9d592; 8e01f15.  
New text: PREREG.md:384-388 (gradient-norm ratios 2.00 on A1, 2.27 on A3, 14.4 on the extension rows, median cosine 0.28, 204 of 216), :404-409 (3,076 = 2,808 + 268; 268 = 192 + 76), :427-433 ('model-based estimate ... not simulated runs'); PAPER_PLAN.md:75-77 (F7 ratios 'provisional').  
Checked: Recomputed: ratios 2.000, 2.273, 14.439, cosine 0.281, 204 of 216 finite; validation_12mb.csv 2,808 / 192 / 76 / 3,076; no-pair figures (section 4.2 of the report).  
Remaining: F7 ratios cannot be recomputed here (log not in the repository; DOC-R11); '3,076 from the 234 fits' is imprecise (DOC-R06); the scripts behind the numbers are not committed (DOC-R04).

**CF-03 - closed.** (document part) README row G-05 and lam_audit.py: 322 / 382 / 1,908 with their meaning.  
Commits: 8e01f15.  
New text: experiments/guard_followups_20261013/README.md:10: 322 rows in the manifest of record (242 none, 40 scvi_adv, 40 scANVI), 382 distinct row tags in the 8 manifests with arm and lam columns, 1,908 = the sum of the rows over the 8 files.  
Checked: lam_audit.py prints exactly results/lam_audit.txt; 1,908 = 1,428 + 240 + 240; the 60 extra tags are X3 'none' rows of the first manifest, all listed in docs/x3_si41_tag_map.csv.

**CF-04 - closed.** (document part) PREREG sentence on the scope of SI-54: scorer commit only; 'not at c88cce3 and one commit'.  
Commits: f4b3b8f; 06345ce.  
New text: PREREG.md:361-376 (the evidence is narrow; two limits of the design: score rows carry the scorer's commit, so a Sinkhorn latent fitted at c88cce3 and rescored at the tag is caught only by the runner's start-up check; 'Sinkhorn rows must carry the new tag' is enforced as 'no Sinkhorn row at c88cce3 and one commit per input'); docs/EQUIVALENT_COMMITS.json.  
Checked: Matches SI-54's wording and the notebook entry NB-20261014-32; code behaviour is the code track's check.

**CF-06 - closed.** (document part) No-pair counts are tabulated by a reporting script after the runs; no code reads them yet.  
Commits: 8703cee.  
New text: PREREG.md:427-433 ('The counts will be tabulated ... by a short reporting script written after the runs. No code reads no_pair_steps yet ... proposal for the diff review').  
Checked: grep over scripts/ and src/: the counter is stats['no_pair'] in src/wcd_vae/wcd/alignment.py; the only mention of no_pair_steps is the record writer scripts/fit_paper_config.py:504-506; nothing reads it back.

**CF-07 - closed.** (document part) Pilot R2/R3 counts come from the frozen c88cce3 record; the current code reports the new counts.  
Commits: d50348b.  
New text: PREREG.md:473-480 and table row 7 (:212).  
Checked: Recomputed on the stored cells: 9 of 12 and 13 of 20 with the SI-66 marking, 12 of 12 and 17 of 20 without it (script 02, claims 50-53).

## 4. The eleven checks

### 4.1 DOC-13: the analysis specification reduced to rules (check 2)

Script `01_doc13_moved_lines.py` compares sections 0-16 of the spec before (fd9c8e3) and after (4af5069; the file at 08f1e3c is identical to 4af5069).

- The file went from 365 to 498 lines (104 lines removed, 237 added). Of the 103 non-blank removed lines, **66 reappear verbatim in Appendix A**, 2 remain in the body re-wrapped, and 35 are in neither place: the rewritten title, the status line, headings, two section-1 bullets, references, and three rewritten rule sentences (the multiplicity paragraph, one bullet of section 11, section 15.2). I read all 105 edits in the body; none changes a rule. Thirty residual word spans (201 words) are headings, the status line and references.
- The four sensitivity variants are still defined in the body: the per-run reading (section 2.4), the corner at (b0, 0) (section 3), pairs first and six independent datasets (section 4); section 11 lists all four.
- Appendix A.2 (why there is no Q8) is correct: in the first draft (6d0ca1f) the open question Q7 (line 143) comes before the multiplicity proposal (line 149), which comes before the open question Q9 (line 153), and no 'Q8' appears anywhere in the draft.
- Appendix A.3 is accurate but is not the complete list of open items (DOC-R13).

### 4.2 Numbers (check 3)

Script `02_numbers.py` recomputes 73 claims; 72 agree. The no-pair model is in scripts 03 and 03b, the P2 recount in 04. The pilot scores used are pilot A1 and the R1-extension scores only.

| Claim | In the documents | Recomputed |
|---|---|---|
| power_without_sim1.py regenerates its JSON | byte for byte | identical (sha256 0be883be...) |
| Six tasks: standard error / minimal detectable effect at 80 % power / power at a 0.02 difference | 0.045 / 0.131 / 0.104 | 0.04499 / 0.13066 / 0.10436 |
| Four families: minimal effect / power | 0.182 / 0.088 | 0.18178 / 0.08778 |
| Without sim1, six tasks: standard error / minimal effect / power | 0.0236 / 0.068 / 0.182 | 0.02355 / 0.06841 / 0.18234 |
| Without sim1, four families: minimal effect / power | 0.095 / 0.137 | 0.09517 / 0.13696 |
| Between-seed sigma and between-task tau (all blocks; without sim1) | 0.030 (12 df), 0.109 (4 df); 0.033 (8 df), 0.056 (2 df) | 0.0298, 0.1094; 0.0329, 0.0558 |
| sim1's share of the squared spread of the three conditioned block means | 67 % | 0.6661 |
| lam_audit.py; rows with lam = 0 for none, scvi_adv, scANVI | 322 = 242 + 40 + 40; 382 tags; 1,908 = 1,428 + 240 + 240 | identical to results/lam_audit.txt; 60 extra tags are X3 none rows in docs/x3_si41_tag_map.csv |
| Holm first threshold (six results; four results) | 0.0083; 0.0125 | 0.00833; 0.0125 |
| Smallest exact two-sided Wilcoxon p, six datasets | 0.03125 (2/2^6) | 0.03125 |
| Validation minibatches; early and training-mode captures | 3,076 = 2,808 + 268; 268 = 192 + 76 (NB-20261013-03 said 236) | 2,808 / 192 / 76 / 3,076; 268 |
| Gradient-norm ratio tagged / converged: A1, A3, R1 extension; cosine; finite | 2.00, 2.27, 14.4; 0.28; 204 of 216 | 2.000, 2.273, 14.439; 0.281; 204 of 216 |
| No-pair, X7 pancreas reference-anchored MMD: fits expected to meet at least one | 25.9 of 27 | 25.92 of 27 |
| No-pair, X7 sim1 | 18 of 18 | 18 of 18 |
| No-pair, X3 atac_small MMD | 6.673 of 48 (about 14 %) | 6.673 of 48 (13.9 %) |
| No-pair, X8 sim2 MMD | 1.076 of 24 (4.5 %) | 1.076 of 24 (4.5 %) |
| No-pair, X1 sim1 MMD and Sinkhorn (each) | 0.3615 of 60 (0.6 %) | 0.3615 of 60 (0.60 %) |
| P2 pilot recount, unconditioned decoder, discriminator | 8 of 54 with the nine lambda 0.03 rows; difference +3.7 points; 0 of 9 collapsed | 8/54; +3.7; 0 of 9 (and 8/45, +0.7 without them) |
| R2 / R3 on the stored cells without and with the SI-66 marking | 12/12, -0.0014; 9/12, -0.0042; 17/20, +0.0010; 13/20, -0.0014 (14 needed) | identical; both keep the defaults |
| Pilot ranges behind the area corner | plain scVI C 0.67-0.90; shifts -0.59 to +0.23; middle half -0.09 to 0.00; per cell -0.19 to +0.09; 300 points | 0.666-0.897; -0.590 to 0.231; -0.094 to 0.005; -0.188 to 0.090; 300 |
| Manifest of record | 6,930 rows: A1 1,098, A2 108, A3 180, X1 3,000, X3 828, X6 216, X7 216, X8 504, X12 288, X13 408, X15 84 | identical (sha256 6efbe422...) |
| Cells and batches of the eight tasks; immune inside immune_hum_mou; shared ATAC barcodes | PAPER_PLAN section 7; 33,506 of 33,506; 3,618 | identical to TASKS, prepped tables and fingerprints; 33,506 of 33,506; 3,618 |

**Assumption of the no-pair model about scvi-tools 1.4.2** (script `03b_no_pair_checks.py`, read from the installed source): the number of training cells is ceil(0.9 N); the partial last minibatch is kept (the data splitter's `drop_last` defaults to False and the training loader uses `shuffle=True, drop_last=self.drop_last`); the project never passes `drop_last`; all 1,734 critic-free manifest rows use batch size 128 and train_size 0.9; the stratified sampler is used only by the reference and pooled arms, so the MMD and Sinkhorn arms use the default loader. A Monte Carlo check of the sim1 case (12,097 cells, 10,888 training cells, 86 steps, last minibatch of 8 cells) gives a per-epoch probability of a minibatch without a pair of 1.48e-5 (296 of 20 million draws) against 1.513e-5 exact. The no-pair script is the first review's with two paths made arguments, so the model itself is not an independent implementation (declared in section 6).

**Not recomputed:** the F7 Sinkhorn-to-W1 ratios (log not in the repository; marked provisional in PAPER_PLAN.md:75-77, unqualified in NB-20261014-16, DOC-R11); the '0.011' spacing figure and the review-simulation coverage numbers (DOC-R04).

### 4.3 `prereg_rules.power_report` (check 4)

The fixer's report says the root search starts at 1.0 and that scipy's non-central t returns NaN (not a number) above a non-centrality of about 38, so the function fails when the standard error is below about 0.026.

- **Confirmed with a run** (script 05, scipy 1.17.1): with the without-sim1 standard error 0.02355 and 5 degrees of freedom the search raises `ValueError: The function value at x=1.0 is NaN; solver cannot continue`. It is loud, not silent. Starting at 0.01 gives the committed value 0.0684067435.
- **Imprecise**: the NaN values of `nct.cdf` are isolated, from non-centrality 42.15 (5 df) and 57.24 (3 df) upward (447 of 39,950 tested values between 0.5 and 400 for 5 df), not everything above 38. Of 230 (standard error, degrees of freedom) pairs between 0.003 and 0.06, 7 fail: six tasks 0.0175, 0.0185, 0.0215, 0.022, 0.0225, 0.0235; four families 0.0165. `nct.sf` gave no NaN anywhere on the tested range.
- **Could the recompute after the Sinkhorn redo hit it?** `freeze_a1_grid.py:62` calls `power_report` on all six blocks (standard error 0.04499, safe). The planned recompute uses only the evaluable blocks and the frozen P1 reading. With the committed pilot blocks, a proxy for that recompute (the four blocks not flagged low-signal) gives 0.02741, outside the failing values; the without-sim1 value 0.02355 fails. The redo changes the inputs, so a hit cannot be excluded.
- **Smallest fix** (not applied; file `08_power_report_fallback_fix.diff`): a helper `_mde80` that keeps the start at 1.0 and falls back to 0.01 only if the solver stops (15 lines added, 5 removed). Tested on a temporary clone (script 08): the 8 failing cases give finite results equal to an independent search started at 0.01; all 223 grid points that work today are bit-identical; all of `tests/prereg` passes (131 passed, 1 skipped). The one-line alternative (`hi = 0.01`) also works but changes `mde80` in the 15th digit (4.8e-15 and 8.9e-15 on the test's synthetic record), which fails the existing exact-equality test `test_decision_records_are_the_same[r1.json]`. Add one failing standard error to `test_power_report_on_synthetic_a1` (tests/prereg/test_rules.py:387). Details: DOC-R12.

### 4.4 Status and tense (check 5)

Nothing is described as done that is not, with two slips (DOC-R10, DOC-R05):

- The tag: 'has not been made' (PREREG.md:175; ANALYSIS_SPEC.md:13; PAPER_PLAN.md:178); 'will freeze' (PREREG.md:517-518).
- The Sinkhorn redo and its three records: 'will be refitted', 'will be committed', 'names proposed for the diff review' (PREREG.md:397-413); the single analysis run after every X1 and X13 row is scored or failed: 'Planned' (PREREG.md:510-511); the R4 stage-x1 record that it reads is made by a later stage (ANALYSIS_SPEC.md:54-59).
- The X1 runs: the local part was started 2026-10-12 and stopped the same day, 19 rows scored, 1 fitted, 1,097 still to fit (PREREG.md:486-490); the full runs are future.
- Approvals: proposals become rules 'only if the user approves them' and each approval 'will be recorded' (PREREG.md:516-519; ANALYSIS_SPEC.md:9-11).
- Every 2026-10-13 and 2026-10-14 entry begins with a status line (script 07 found all eight); dates and times of all 15 commits and 12 decisions in the amendment table equal git log and the notebook, in New York time; SI-60's UTC date is explained in NB-20261014-34.
- Slips: ANALYSIS_SPEC.md:62-67 (present tense for a planned manual check, DOC-R10); status lines that stop at the first code commits (DOC-R05).

### 4.5 Attribution (check 6)

Only SI-68 to SI-77 are called user decisions for the X1 analyses (PREREG.md:507, :513; ANALYSIS_SPEC.md:3-10; PAPER_PLAN.md:176). Older user decisions that the text cites by number (SI-38, 41, 43-46, 51-58, 60-64, 66) are genuine ledger entries. The user's words in `CONSTRAINTS.md` against the text:

- **SI-55**: 'X1 grid edge (I006): correct PAPER_PLAN to PREREG section 4 ...' decides the grid edge only. PREREG.md:319 now says 'user decision SI-55 for part (a)'. Correct.
- **SI-68**: 'Lock now: ... the comparison with standard tools (Harmony, Scanorama, sysVI, scANVI) when every method gets the same number of tries; and how wide each method's range of good penalty strengths is. Lock later, while X1 runs and before anyone looks at an X1 result: P3, P4 and the other follow-up analyses'. PREREG.md:521 and PAPER_PLAN.md:185 keep 'an X1 result' (any arm) for the follow-ups, as NB-20261014-33 decided. Correct. The same words show that 'equal numbers of tries' is the user's decision, which section 9 calls the agent's proposal (DOC-R07).
- **SI-73**: 'Combine the 6 real datasets with a family term ... exact calculation (no random sampling), so results reproduce digit for digit'. Section 4 of the spec adds the agent's choices (grid, cap, exclusion of a dataset with standard error below 1e-6, CPU caveat) under the heading 'user decision SI-73' (DOC-R08).
- **SI-75**: P2 margin of 5 percentage points and prior scales 0.25: matches spec section 6 (ANALYSIS_SPEC.md:180-185) and Q9; the premise quoted to the user was off and is now corrected there (DOC-17).
- **SI-77**: 'Selection score for the single-strength tables, the comparison with standard tools at equal numbers of tries, and the width of the range of good strengths: 0.4 x batch score + 0.6 x biology score (scIB's weights), without scIB's min-max scaling.' Spec section 8 says 'SI-77 decides only the selection score' for the transfer rule, which is right; the transfer rule itself ('chosen on the other datasets' narrowed to other families) is labelled the agent's proposal.

### 4.6 Disclosure (check 7)

Where an auditor would look (PREREG.md:483-503): the 19 files of 2026-10-12 (17 adversarial-arm rows, all immune, conditioned decoder, seed 0; 2 rows of none and scvi_adv), the three count-only integrity checks, the rescored latent (tag `X1_immune_discriminator_lg1_c1_s0_122afe99`, equality only), the audit and its limit, and the proposed definition of 'examined' are all stated, and the heading no longer says 'before X1 runs'. The amendment table (PREREG.md:204-213) is correct in its times (15 of 15, 12 of 12). The late implementation of SI-66 (decided 2026-10-13 20:53, implemented in f3c3415 on 2026-10-14 at 07:50) is stated in row 7 and in the entry (PREREG.md:454-457, :466). Gaps: DOC-R01, DOC-R02, DOC-R05, DOC-R14.

### 4.7 Notebook entries NB-20261014-20 to -22 and -29 to -35 (check 8)

| Entry | Claims checked | Result |
|---|---|---|
| NB-20261014-20 | 47 findings, 20 to fix, three tracks PASS WITH FIXES | confirmed (18 + 10 + 19 = 47; 5 + 2 + 13 = 20); analysis-track details not re-run |
| NB-20261014-21 | 19 = 17 + 2 files; runner and scorer log only tags and timings | logging claim confirmed at 828f3e7 (run_stage.py line is now :952); counts not checkable (blinded); scope of the audit: DOC-R01 |
| NB-20261014-22 | P2 recount 8 of 54, +3.7, 0 of 9 | confirmed (script 04) |
| NB-20261014-29 | 8 files +73/-22; tree 46a2df12; test counts | patch size and tree confirmed; test counts not re-run (code track) |
| NB-20261014-30 | designs for CF-01/CF-02 are agent proposals | confirmed (text); code consequences are the code track's |
| NB-20261014-31 | documents patch set +491/-131; power JSON; lam_audit; sim1 share; no-pair; scvi-tools | confirmed; cites a script path not in the repository (DOC-R04) |
| NB-20261014-32 | spec 365 to 498 lines; 35 other lines; A.2; counts; analysis_x1 stops on a missing lambda; no test reads the documents; pandoc | confirmed except pandoc (not re-run) |
| NB-20261014-33 | SI-68 scope kept | confirmed |
| NB-20261014-34 | annotations for SI-60, SI-63, SI-69, SI-51, SI-52 | confirmed; not visible from CONSTRAINTS.md (DOC-18) |
| NB-20261014-35 | 268 = 192 + 76 (NB-20261013-03 said 236) | confirmed |

The notebook hash chain is intact for all 186 entries; INDEX.md agrees; all 42 notebook entries and 35 ledger entries cited in the new text exist. The only cited paths that do not exist are the three proposed redo records (planned), the pattern `notebook/yyyy-mm-dd.md` and `scripts/recompute_g01.py` (DOC-R04).

### 4.8 Consistency after all changes (check 9)

- PREREG, PAPER_PLAN, ANALYSIS_SPEC and EQUIVALENT_COMMITS.json agree on: the tag status, the Holm family (six; 0.0083), the power figures (six tasks, with and without sim1), the Sinkhorn redo order, the SI-54 scope, the manifest counts and the attribution of SI-68 to SI-77. `CONSTRAINTS.md` is unchanged since the first review (same SHA-256 as the artifact; no commit touches it).
- Only documents changed between fd9c8e3 and HEAD outside the notebook; no test reads them (they appear in module docstrings only), so the test results for tree 46a2df12 carry over.
- Stale or contradictory after later commits: DOC-R03 (spec section 1 against section 15.1 and the code), DOC-R05 (status lines, table row 8, 'SI-01..SI-67'), DOC-R10 (spec tense), DOC-R07 (which sections are proposals).

### 4.9 Plain language (check 10)

The new PREREG glossary and status lines are readable by a genetics PhD student, and the two meanings of 'family' and of 'cell' are separated. The slips are listed in DOC-R09: the double use of R-labels in the spec, undefined scvi_adv and X13, the glossary row that groups X1 with the pilots, 'chat record' and 'By file name', 'fits' for one-epoch test refits in table row 5, 'as recommended', and a commit column without a time zone.

### 4.10 Open-ended: what a referee or pre-registration auditor would flag now (check 11)

The scope of the 'no value displayed' audit (DOC-R01); the filler scores that already exist (DOC-R02); numbers without committed scripts, which breaks the project's own reproducibility rule (DOC-R04); stale status lines (DOC-R05); an incomplete list of open decisions (DOC-R13); no outside time stamp, no hash list for the early score files and an audit done by the agent that ran the checks (DOC-R14).

## 5. New findings

Severity: fix = must change before the tag; note = worth knowing, optional. No blocker.

### DOC-R01 (fix): The audit sentence is broader than the search it rests on

- Where: docs/PREREG.md:493-497; notebook/2026-10-14.md:238-250 (NB-20261014-21, title and 'ran:' line); docs/ANALYSIS_SPEC.md:491-492
- Problem: PREREG says that an audit of every chat record of the project 'found no display or analysis of an X1 score value'. The notebook entry it rests on states what was searched: chat records for reads of tier12_v2/X1_local/scores (12 messages), the printed output of the scripts that read them, and two log statements. X1 and X13 scores also exist in other places (the cluster filler fits scored on this machine, the outputs of the identity checks), and a search keyed on one folder cannot see a value shown from another. The audit was made by the same agent that ran the checks (author: agent).
- Evidence: PREREG.md:493-497 against NB-20261014-21 'ran:'; NB-20261007-01 (150 filler score files exist); PREREG.md:358-359 (an X13 sysVI latent was rescored in the identity check).
- Suggested fix: State in PREREG what was searched, in the notebook's words, or extend the search to the other places where X1 and X13 scores exist and say so. Keep the sentence that the audit cannot show that nobody opened a file. Ask the user, or a reader who did not run the checks, to confirm.

### DOC-R02 (fix): The disclosure leaves out the other X1 and X13 scores that already exist

- Where: docs/PREREG.md:22, :358-359, :483-503; notebook NB-20261007-01, NB-20261012-02; CONSTRAINTS.md SI-68; docs/ANALYSIS_SPEC.md:45
- Problem: PREREG discloses the 19 local X1 rows of 2026-10-12 only. It does not say that 250 earlier fits of arms that depend on no rule (X1 arms none and scvi_adv; X13 baselines scANVI and sysVI; PREREG.md:22 lets them run first) exist and are being scored on this machine: pass B (lung, pancreas, sim2) was complete, 150 of 150 scored, on 2026-10-07 and states 'Metric values not inspected' (NB-20261007-01); pass A (atac_large, immune_hum_mou) was resumed on 2026-10-12 with 99 of 100 latents still to score (NB-20261012-02); later progress is not in the documents. The identity check also rescored one X13 sysVI latent (PREREG.md:358-359). The proposed definition of 'examined' covers X1 adversarial-arm scores only, but SI-68 says 'before anyone looks at an X1 result' (any arm), and the freeze covers the X13 analyses too (ANALYSIS_SPEC.md:45). Whether scvi_adv counts as an adversarial arm is not said, and nothing says that the 19 early rows stay X1 rows under SI-54.
- Evidence: Quoted lines above. The first review (DOC-01) counted 158 filler score files by 2026-10-12; I did not recount them because the folder is blinded.
- Suggested fix: Add two or three sentences to the disclosure: how many non-adversarial X1 and X13 scores exist, when they were scored, and that their values were checked only for finiteness and provenance (NB-20261007-01). Widen 'examined' to every X1 and X13 score, or say why none, scvi_adv and X13 are outside it. State that the 19 early rows are kept (SI-54). Define scvi_adv and X13 in the glossary.

### DOC-R03 (fix): The analysis rules contradict themselves on whether b0, b_common and b* are recomputed

- Where: docs/ANALYSIS_SPEC.md:58-59 against :281-285 (section 15.1) and scripts/analysis_x1.py:369
- Problem: Section 1 says that b0, b_common, b* and the low-signal flag 'are read from the frozen R4 stage-x1 record ... the analysis never recomputes them'. Section 15.1 says the script refuses a record whose b0, b_common, b* or flag differs from what prereg_rules.r4_stage computes from the same X1 rows, and analysis_x1.py:369 reruns P.r4_stage for exactly that purpose (rigour review AN-03). The code does the second thing. A registered rule that says the opposite of the registered code is a trap for the first reader of the tag.
- Evidence: ANALYSIS_SPEC.md:58-59 (quoted), :281-285; scripts/analysis_x1.py:369 ('the rule that wrote the record, rerun on these scores (AN-03)'). Script 07 output.
- Suggested fix: Replace the sentence at :58-59 by: the four quantities are read from the frozen R4 stage-x1 record; the analysis recomputes them from the X1 rows with prereg_rules.r4_stage and stops if they differ (section 15.1).

### DOC-R04 (fix): Numbers in the registered documents come from scripts that are not in the repository

- Where: docs/PREREG.md:427-433; notebook NB-20261014-31 (cites scripts/recompute_g01.py, which does not exist in the repository); NB-20261014-22 and docs/ANALYSIS_SPEC.md:461-463; ANALYSIS_SPEC.md:433-436, :340, :345, :336-338; docs/PAPER_PLAN.md:75-77
- Problem: The project rule is that every number is regenerated by committed code. The no-pair expectations (25.9 of 27 fits, 18 of 18, about 14 %, 4.5 %, 0.6 %), the P2 recount (8 of 54, +3.7 points) and the pilot ranges of the area corner (plain scVI C 0.67 to 0.90; shifts -0.59 to +0.23; 300 points) can be reproduced only from the first review's bundle or from my scripts. The spec's '0.011' and 'coverage 0.91 to 1.00' (the first review's simulations) and the F7 ratios cannot be reproduced from the repository at all.
- Evidence: 06_notebook_chain_and_references.txt (paths cited and missing: scripts/recompute_g01.py plus the three proposed redo records); scripts 02, 03 and 04 of the evidence bundle reproduce the first three groups; grep of experiments/analysis_numerics_20261014 finds no coverage or spacing computation.
- Suggested fix: Commit one script and its output per group (scripts 02 to 04 of the evidence bundle can be used after a look) and cite the path beside each number. For '0.011', 'coverage 0.91 to 1.00' and the F7 ratios either commit the review's scripts or write 'from the review's simulation; not reproducible from this repository'.

### DOC-R05 (fix): Status lines and the amendment table do not name later changes to the X1 analysis code and spec

- Where: docs/PREREG.md:509 and table rows 5 and 8 (:210, :213); docs/ANALYSIS_SPEC.md:11-12, :359-363
- Problem: Both status lines say the specification and code were committed in 56c0762, 990d37d and ff88666, with three test gaps closed in 858101b. Since then scripts/analysis_x1.py and tests/analysis changed by 162 added and 46 removed lines, almost all in 8b90d79 (rigour-review fixes AN-01 to AN-06 and others), and the spec changed in a1d497b, 56febe4 and 4af5069; a numerical-check folder was added in 6aaa3ac. None of these is named, and the table row for the X1 analyses has no row for them. Table row 5 omits SI-65 (decided 2026-10-13 20:48), although PREREG.md:444 gives its NMI-0 rule. The spec's last paragraph still lists 'CONSTRAINTS.md (SI-01..SI-67)' as the sources checked for 'this draft', although its decisions cite SI-68 to SI-77.
- Evidence: 07_text_scans.txt (commits that changed the code, tests or spec since 6d0ca1f, marked 'named' or 'NOT named'); git diff --shortstat ff88666..HEAD -- scripts/analysis_x1.py tests/analysis = 2 files, +162/-46.
- Suggested fix: Say in both status lines (and in table row 8) that the code and spec were revised after the independent review (8b90d79, 6aaa3ac, 56febe4, 4af5069) and are not yet approved by the user, or that the tag, not these commit ids, fixes the registered version. Add SI-65 to row 5. Change 'SI-01..SI-67' to 'SI-01..SI-77'.

### DOC-R06 (note): '3,076 minibatches from the 234 pre-fix fits' is imprecise

- Where: docs/PREREG.md:404-409; experiments/sinkhorn_fix_20261013/results/validation_12mb.csv
- Problem: The total (3,076 = 2,808 + 192 + 76) and the 268 early and training-mode captures are right, but the 192 early captures carry 160 separate replay tags (untrained, epoch 2, epoch 3), none of them among the 234 fits; the 76 training-mode captures come from 19 of the 234 fits.
- Evidence: Script 02, claim 27 (the only claim of the 73 not confirmed).
- Suggested fix: Write: 2,808 final-epoch minibatches from the 234 pre-fix fits (12 each), 76 training-mode minibatches from 19 of them, and 192 minibatches captured earlier in training in separate replays (160 tags).

### DOC-R07 (note): Attribution of sections 9 to 11 and of 'equal numbers of tries' is inconsistent

- Where: docs/PREREG.md:514-516 against docs/ANALYSIS_SPEC.md:7-10; docs/PAPER_PLAN.md:203, :217; docs/ANALYSIS_SPEC.md:214
- Problem: PREREG calls 'sections 9 to 11' of the spec proposals; the spec itself says sections 9 and 10 and only the sensitivity analyses of section 11 that no decision names (the user decided several: SI-38, SI-69, SI-71, SI-73). Section 9 and PAPER_PLAN replace 'equal configuration counts' by 'equal k' and call this the agent's proposal, although SI-68 and SI-77 already decide 'the comparison with standard tools ... when every method gets the same number of tries'; what is the agent's is the estimator (best-of-k curves) and the family composition.
- Evidence: Wording of SI-68 and SI-77 in CONSTRAINTS.md (section 4.5 of the report).
- Suggested fix: Align PREREG with the spec's list; in section 9 and PAPER_PLAN say that equal numbers of tries is the user's decision (SI-68, SI-77) and only the estimator is proposed.

### DOC-R08 (note): Agent's implementation choices sit under the heading 'user decision SI-73'

- Where: docs/ANALYSIS_SPEC.md:130-144; docs/ANALYSIS_SPEC.md:96-97
- Problem: Section 4 is headed 'user decision SI-73', but the 201 x 201 log grid, the cap 0.5, the lower end 1e-15, the exclusion of a dataset whose standard error is below 1e-6 (reason zero_spread) and 'on a different CPU type the last digits can differ' are the agent's choices from the rigour review (AN-01, AN-02). SI-73 says the exact calculation gives results that 'reproduce digit for digit'. The zero_spread exclusion removes a dataset from 'combine the 6 real datasets' in a case the user was not asked about; it is covered by the proposal label of section 2.3 only.
- Evidence: ANALYSIS_SPEC.md:136-144 against the SI-73 rule text.
- Suggested fix: Add 'details marked 'amended 2026-10-14' are the agent's proposals' under the section 4 heading, and say in SI-73's terms that digit-for-digit holds on the same input files and CPU type.

### DOC-R09 (note): Plain-language slips (SI-67)

- Where: docs/ANALYSIS_SPEC.md:18-27, :35-41, :214, :225; docs/PREREG.md:22, :195, :210, :490, :493
- Problem: (1) The spec uses R0, R3 and R5 for PAPER_PLAN's research questions next to R1 to R4 for the registered rules; the clash is explained only in PREREG.md:194, and the spec's word list does not point to it. (2) scvi_adv and X13 are not defined anywhere in the glossary, although the disclosure paragraph names them. (3) The glossary row 'A1, A2, A3, X1' begins 'Pilot experiments' and so reads as if X1 were a pilot. (4) 'chat record' and 'By file name' are not explained for an outside reader. (5) Table row 5 says 'No completed fit or score changes: 59 of 59 non-Sinkhorn fits ... identical'; these are one-epoch test refits of code paths, not the production fits (PREREG.md:355-357 says so). (6) 'all as recommended' (spec:4) means 'the option the agent recommended'. (7) The table's commit column does not state its time zone.
- Evidence: Quoted lines; script 07 output.
- Suggested fix: One sentence at spec:18-27 ('R0 to R5 in sections 0, 9 and 10 are PAPER_PLAN's research questions; R1 to R4 elsewhere are the registered rules'); two glossary rows (scvi_adv, X13); split the A1-A3 and X1 row; say 'logs of the agent's working sessions on this project'; write 'one-epoch test refits' in row 5; add 'New York time' to the commit column.

### DOC-R10 (note): A planned manual check is written in the present tense

- Where: docs/ANALYSIS_SPEC.md:62-67; docs/PREREG.md:411-413
- Problem: The spec says the Sinkhorn grid 'is re-derived', the analysis 'is given' the redone record, and the six non-Sinkhorn grids 'are checked against docs/prereg/r1_record.json first'. None has happened, and the last is a manual step: analysis_x1.py reads x1_grid from the record it is given (:392-394) and does not compare grids. PREREG.md:411-413 says 'will be checked'.
- Evidence: scripts/analysis_x1.py:392-394 (grid read from --r1-record; no comparison with the frozen record).
- Suggested fix: Write 'will be re-derived', 'will be given', 'are to be checked by hand ... (the script does not compare them)'; or add the comparison to the script.

### DOC-R11 (note): The F7 Sinkhorn-to-W1 ratios stand unqualified in one notebook entry and cannot be recomputed

- Where: notebook/2026-10-14.md:177 (NB-20261014-16); docs/PAPER_PLAN.md:75-77
- Problem: PAPER_PLAN marks the ratios 0.86-0.89 and 0.53-0.77 'provisional' (from the Sinkhorn review log, 5 pre-fix latents with 8 minibatches each, not in the repository). The notebook entry that first wrote them into PAPER_PLAN gives them without that qualification; the notebook is append-only. Nobody has recomputed them since the first review.
- Evidence: PAPER_PLAN.md:75-77; notebook/2026-10-14.md:177.
- Suggested fix: Add a one-line NOTE entry that the ratios are provisional; recompute them on the refitted latents as planned and commit the script.

### DOC-R12 (note): power_report stops with an error for some small standard errors; the power recompute after the Sinkhorn redo could hit it

- Where: scripts/prereg_rules.py:704-708; scripts/freeze_a1_grid.py:62; experiments/final_review_amendments_20261013/power_without_sim1.py (docstring)
- Problem: The root search for the smallest detectable effect starts at 1.0. With scipy 1.17.1, nct.cdf returns NaN at isolated large non-centralities (first at 42.15 for 5 degrees of freedom, 57.24 for 3; 447 of 39,950 tested values on 0.5 to 400 for 5 degrees of freedom), and the solver then stops with 'The function value at x=1.0 is NaN'. It is loud, not silent. It is not 'above about 38' and not for every standard error below 0.026: 7 of 230 (standard error, degrees of freedom) pairs on 0.003 to 0.06 fail (six tasks: 0.0175, 0.0185, 0.0215, 0.022, 0.0225, 0.0235; four families: 0.0165), including the without-sim1 value 0.02355. freeze_a1_grid.py:62 calls power_report on all six blocks (standard error 0.04499, safe). For the planned recompute on the blocks that are not flagged low-signal, the committed pilot blocks give a proxy of 0.02741 (outside the failing values), but the redo changes the inputs.
- Evidence: 05_power_report_nan_probe.txt; 08_power_report_fix_check.txt (the fix below: 8 failing cases give a finite result equal to an independent start-at-0.01 search; 223 of 223 working grid points bit-identical; tests/prereg 131 passed, 1 skipped).
- Suggested fix: Apply 08_power_report_fallback_fix.diff (a helper _mde80 that keeps the start at 1.0 and falls back to 0.01 only when the solver stops; 15 lines added, 5 removed; every result that works today stays bit-identical). The one-line alternative, 'hi = 0.01', also works but changes mde80 in the 15th digit (4.8e-15 and 8.9e-15 on the test's synthetic record) and then fails the exact-equality test test_decision_records_are_the_same[r1.json]. Add one failing standard error (for example 0.0175 with 5 degrees of freedom) to test_power_report_on_synthetic_a1. Correct 'above about 38' in the power_without_sim1.py docstring.

### DOC-R13 (note): No single list of everything the user is asked to decide at the diff review

- Where: docs/ANALYSIS_SPEC.md:486-498 (A.3); docs/PREREG.md:394-396, :446-447, :480-482
- Problem: A.3 lists five open items, covering six of the fix report's eight questions to the user and only the R1 record name of the three. The questions on the Sinkhorn A3 refits (PREREG.md:480-482) and on the NMI-0 and iteration-cap policies (PREREG.md:394-396, :446-447) appear only in PREREG; the names of the other two redo records only in PREREG.md:411-413.
- Evidence: Fix report section 4 (eight questions) against A.3 and PREREG.
- Suggested fix: Make A.3 (or one short list in PREREG section 9) the complete list of the eight questions, each with the file and line where it is explained.

### DOC-R14 (note): Suggestions a pre-registration auditor is likely to make

- Where: docs/PREREG.md:483-503; whole pre-registration
- Problem: (1) The 19 early X1 score files and the filler score files can be shown to be untouched only by hashes; a committed list of their SHA-256 values would let an auditor confirm later that the files were not changed or selectively read, without revealing a value. (2) The tag and the notebook chain are both created inside the same repository by the same agent; the documents name no outside time stamp (no push date, deposit or registry entry). (3) The audit of 'no value displayed' was made by the agent that ran the checks (see DOC-R01).
- Evidence: grep of the three documents for OSF, Zenodo, time stamp, push: no plan found; notebook chain verified internally (186 of 186 entries).
- Suggested fix: Record the hashes; push the tag to the project's GitHub remote (or deposit the registration) and record the remote's date next to the tag; ask the user to confirm the audit.

## 6. Limits of this check

- Not re-run: the '0.011' spacing figure (ANALYSIS_SPEC.md:340) and the review-simulation coverage numbers ('0.91 to 1.00', :345; 'conservative' / 'less than 95 %', :336-338); they come from the first review's analysis track and no script for them is in the repository.
- Not recomputed: the F7 Sinkhorn-to-W1 ratios (0.86-0.89, 0.53-0.77); the log is not in the repository. I only checked where they are marked provisional.
- Analysis-track and code-track numbers quoted in NB-20261014-20 and -29 (34 wrong declarations in 512 stress worlds; 421 passed / 18 skipped; 172 passed) were not re-run. Of the test suite I ran only tests/prereg (131 passed, 1 skipped, in a scratch clone with the suggested power_report fix) plus test_rules_unchanged.py (9 passed) and test_provenance_rule.py (13 passed) on the unpatched clone.
- Blinded material was not opened: the arm breakdown of the 19 early X1 rows and the filler-score counts (250 fits; 150 scored; the first review's 158) rely on NB-20261014-21, NB-20261007-01, NB-20261012-02 and the first review's text. The committed identity-check files in experiments/integration_checks_20261013/results were read for fit-identity counts and the score file for column names only; the identity score files under review/identity_* were not opened.
- The chat-record audit of NB-20261014-21 was not repeated.
- The no-pair model (03_no_pair_model.py) is the first review's script with two paths turned into arguments, so it is not an independent implementation; my independent parts are the reading of the installed scvi-tools 1.4.2 source and a Monte Carlo check of the sim1 case only. The other four figures rest on that model.
- The P2 recount drops Sinkhorn rows, as the task states; Sinkhorn collapse rates were not recounted.
- Pandoc rendering of the three documents (NB-20261014-32) was not re-run; only the column count of every row of the amendment table was checked.
- CPU only; one private clone and one scratch clone; scripts ran one at a time (cap of 2 processes respected).
- The project's rigor ledger was not edited because three re-checkers run in parallel; the new findings and status changes are provided as rigor_ledger_entries_recheck_docs_20261014.json for the lead to merge.

## 7. Files

- `RECHECK_DOCS_REPORT_20261014.md`: this report.
- `recheck_docs_20261014_evidence.tar.gz`: README.txt, run_all.sh, scripts 01 to 09, and their outputs (including `08_power_report_fallback_fix.diff`). Re-running `run_all.sh` takes about 2 minutes on one CPU thread.
- `rigor_ledger_entries_recheck_docs_20261014.json`: status changes for DOC-01 to DOC-19 and CF-03/-04/-06/-07 and the 14 new findings, in the format of the first review's ledger file, for the lead to merge.

Copies of all three are in `/home/kendall/experiment_data/wasserstein-critic-deconfounding/review/recheck_docs_20261014/`.
