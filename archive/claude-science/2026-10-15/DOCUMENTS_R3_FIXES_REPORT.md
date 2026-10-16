# Round 3, track "documents": fixes report

Branch `final-review-fixes`, base commit 5d06a27e39bd0337ac36b69f58703e26ab1e04a1. Private clone, branch `r3-documents`, 14 commits, head c97676c.
Changed: 27 files, +3,208 / -88 lines (docs/PREREG.md +184 / -74, docs/PAPER_PLAN.md +16 / -13, docs/EQUIVALENT_COMMITS.json +1 / -1; the rest is the new folder experiments/doc_numbers_20261015/ with 7 scripts, `run_all.sh`, `README.md` and 15 result files).
All six acceptance gates pass (section 4). `git am` of the series on a fresh clone of the bundle applies cleanly and changes only these 27 files.
No round-3 commit id is cited inside any document; changes are named by 2026-10-15 and ledger id.

## 1. Deliverables

| file | content |
|---|---|
| documents_r3_patches.tar.gz | `git format-patch 5d06a27..HEAD`, 14 patches, apply in file-name order |
| DOCUMENTS_R3_FIXES_REPORT.md | this report |
| documents_r3_evidence.tar.gz | the scripts and their outputs, the claim receipt (script, CSV, summary), the gate log, document builds, the standing-instructions verdicts, the deslop report, a blinding note |

## 2. Items A to N and the findings behind them

Action: fixed = the part assigned to this track is done (parts of the same finding in other tracks' files are in section 7); partly_fixed = part of what was assigned to me is not done (none); deferred = not my file. Line numbers are those of the head commit for documents I changed and of 5d06a27 for files I did not change.

| item | findings and ledger ids | what I checked (primary source) | action | commit | where to read it |
|---|---|---|---|---|---|
| A | DOC-R01 (audit sentence narrower than the search) | base PREREG.md:485, :493-501 (said every chat record; 'open item for the diff review'); NB-20261014-21, notebook/2026-10-14.md:238-243 (one folder: the agent's session logs for reads of tier12_v2/X1_local/scores, 12 messages; printed output of the scripts that read them; 2 log statements); NB-20261015-15, notebook/2026-10-15.md:166-171 (user: 'No, nobody read a value') | fixed | b327096 | PREREG.md:552-556 (audit as searched), :559 (user's statement, attributed to the user); no 'chat record' left (grep, 0 hits). ANALYSIS_SPEC.md:491-492 is open question 1 |
| B | DOC-R02, SI-78 (disclosure of the other existing X1 and X13 scores; definition of 'examined') | base PREREG.md:22, :358-359, :483-503; NB-20261006-01 (notebook/2026-10-06.md:4-6: 250 fits, pass A 100/100), NB-20261006-04 (:30: pass A scored 1 of 100), NB-20261007-01 (notebook/2026-10-07.md:3-6: pass B 150/150, values not inspected), NB-20261012-02 (notebook/2026-10-12.md:15-19: 99 latents to score), NB-20261012-05 (:46: 19 scored); counts of fits from the manifest of record by `filler_fit_counts.py`, none from a score file; S1 word for word replaces the drafted 'X1 adversarial-arm score' definition (base :500-503, :505, :518) and the freeze wording now says 'X1 or X13 score' in PREREG and PAPER_PLAN | fixed | b327096 | PREREG.md:22, :518-527 (250 = 100 + 150; pass B; pass A), :547-551 (definition, S1 at :548 and :619) |
| C | SI-79, DOC-R14 (hash list; outside time stamp; audit by the same agent) | SI-79 text (CONSTRAINTS.md:548-554); DOC-R14 points (1) hashes, (2) outside time stamp, (3) audit made by the agent that ran the checks (3 is covered by A) | fixed | b327096, c97676c | PREREG.md:528-531 (hash list folder named, hashes only), :545 (GitHub date recorded after the tag); line 5 lists the new folders. PREREG gives no file counts from the hash list (lead's instruction) |
| D | SI-81 (a) (b) (c), SI-80, NB-20261015-08, NB-20261015-07, NB-20261015-09, glossary row 'diff review, proposal' | ledger SI-80, SI-81 (CONSTRAINTS.md:556-571) and notebook/2026-10-15.md:76-110 | fixed | f963ea7, 57e047f, e656c90, 0460b13 | decided items carry date 2026-10-15 and ledger id; no 'proposal' or 'open' label on a decided item (gate 3); glossary row at PREREG.md:204. For SI-81 (b) I kept 'and the decision is listed with the results', which the ledger text omits (open question 6) |
| E | SI-82 | CONSTRAINTS.md:572-579 (four checks named) | fixed | cd9aef8 | PAPER_PLAN.md:204: reported, not pre-registered, used for no conclusion |
| F | SI-87, DOC-R10 | S2 and the lead's corrected ledger text (artifact b42c0723, which says the five files do not exist at prereg-tier12-v2); finding DOC-R10 (hand check, tense) | fixed | ef8ebcf | PREREG.md:425 (S2), :426-432; the sentence 'prereg-tier12-v2 holds these files' appears nowhere (gate 3). The clone's CONSTRAINTS.md still has the older SI-87 wording |
| G | DOC-R13 | questions file: 19 numbered rows; notebook/2026-10-15.md: 'question n of 19' | fixed | edb3d01 | PREREG.md:613-633; 19 body rows in the built HTML; column widths set so the PDF table fits the page |
| H | DOC-R04, DOC-R11 | scripts rerun from a fresh clone, every assertion held; all numbers listed by DOC-R04 reproduced (section 5); F7 ratios and '0.011' and 'coverage 0.91 to 1.00' cannot be recomputed from the repository | fixed for my files (the ANALYSIS_SPEC sentences for '0.011' and the coverage are open question 1) | 9a01443, f963ea7, ef8ebcf, 0460b13, 2947fd2, c97676c | experiments/doc_numbers_20261015/README.md; PREREG.md:446 (no-pair), :418 (minibatches), :218 (P2 recount); PAPER_PLAN.md:75-79 (F7: provisional, cannot be recomputed until the refitted latents) |
| I | DOC-R05, SI-65 | git log commit times in New York time for the 12 commit ids of rows 5 and 8 (c4be14b, c3fd92d, 6d0ca1f, 56c0762, 990d37d, a1d497b, ff88666, de741a0, 8b90d79, 6aaa3ac, 56febe4, 4af5069); ledger decision times from the notebook headers (06:45, 06:57, 09:37, 09:54, 20:48); SI-65 added to row 5 | fixed | 683f198 | PREREG.md:3 (status), :215 (row 5), :218 (row 8), :219 (new row 9) |
| J | DOC-R06 | experiments/sinkhorn_fix_20261013/results/validation_12mb.csv by `validation_minibatch_counts.py`: 3,076 = 2,808 + 192 + 76 | fixed | ef8ebcf, 9a01443 | PREREG.md:418-423 |
| K | DOC-R07 | base ANALYSIS_SPEC.md:7-10; base PAPER_PLAN.md:203, :217; SI-68, SI-77, SI-80 | fixed (PREREG and PAPER_PLAN parts; ANALYSIS_SPEC.md:214 is open question 1) | 57e047f | PREREG.md:588-595; PAPER_PLAN.md:206, :220 |
| L | DOC-R09 items 2, 3, 4, 5, 7 | each item against the glossary and table text at 5d06a27 | fixed (items 1 and 6 are in ANALYSIS_SPEC: deferred to the analysis track) | 88f56d4 (2, 3), 683f198 (5, 7), b327096 (4) | glossary rows scvi_adv :177, X13 :198, A1-A3 :196, X1 :197, agent roles :205; 'Counted from the names of the score files' at PREREG.md:514; row 5 :215; 'New York time' in the table header |
| M | CF-R04 | `scripts/prereg_rules.py` reads docs/EQUIVALENT_COMMITS.json; the JSON loads with the same 8 keys and unchanged values except `meaning`; tests/prereg 131 passed, 1 skipped | fixed | a1b8c53 | docs/EQUIVALENT_COMMITS.json:3; PREREG.md:382 |
| N | I091, D-01, D-02, DOC-03 wording | word-level diff of every changed hunk that names them: only the requested edits (row 2 'is' to 'was' plus the SI-80 date; the two clauses of row 3) | complied, nothing else changed | - | - |
| - | DOC-R03, DOC-R08 | in docs/ANALYSIS_SPEC.md (analysis track); not checked | deferred | - | - |
| - | DOC-R12 | `scripts/prereg_rules.py` power_report (code track); the user's decision SI-86 is stated in question 4 of the list | deferred | - | PREREG.md question 4 |

## 3. The lead's added facts (items B and C)

| fact | where it is written | source |
|---|---|---|
| identity check rescored 7 stored latents, among them one of the 19 early X1 rows (tag X1_immune_discriminator_lg1_c1_s0_122afe99) and one X13 sysVI row (X13_pancreas_sysvi_l10_c1_s0_60b085ea); both tags exist in the manifest of record (checked) | PREREG.md:213, :366-367, group (3) at :534 | NB-20261013-12, -16, -40; lead's message |
| two test runs of the X13 baseline runner on atac_small, 2026-10-02, before tag prereg-tier12-v1 (commit 129142b, 2026-10-03 15:07 New York time), each scoring 6 latents; not registered rows | group (4) | NB-20261002-09, -10 (19:40 on 2026-10-02) |
| probe outputs of the final review: 49 registered configurations, 42 from the reviewers' own fits on sim1, atac_small and pancreas at the code of 2026-10-12 with training settings unchecked, 7 rescoring 6 real X1 baseline latents of arm none; these are not production fits | group (5) | lead's message (not in the clone) |
| audit of 2026-10-15: two displays (2026-10-02 six runner-test runs; 2026-10-13 before 17:13, PCR_batch and cell-cycle conservation of one X1 discriminator run and one X13 sysVI run); no value of the 49 probe files in 56 transcripts and 26,722 review files; limits stated | PREREG.md:560-575 | lead's messages; NB-20261015-22, -23 (not in the clone) |
| user decision SI-88 (NB-20261015-25): the two displays are listed as shown to the agent only and do not count as examined under SI-78 | PREREG.md:574-575; amendment-table row 9; no open question is left on the classification | lead's message |

## 4. Acceptance gates

| gate | result | command or evidence |
|---|---|---|
| 1 claim receipt | 331 rows over 126 added or changed sentences: 151 quantities, 26 times, 101 dates, 53 labels; 0 coverage gaps; every date matched to a cited id or to a checked fact | `python ev/claim_receipt.py rv inputs/DIFF_REVIEW_QUESTIONS_draft.md ev/claim_receipt` (in the evidence tar); section 6 |
| 2 scripts from a fresh clone | env scvi-api (Python 3.11.15, numpy 2.4.6, scipy 1.17.1, pandas 2.3.3, anndata 0.12.19, scvi-tools 1.4.2): `bash experiments/doc_numbers_20261015/run_all.sh` in the clone made by `git am` finished in 47.8 s with every assertion held, and `git status` of the folder was empty afterwards, so all 15 result files are byte-identical to the committed ones | gates.log |
| 3 greps | `sinkhorn_redo`: 0 hits; no sentence says prereg-tier12-v2 holds the record files; no 'proposal' or 'open item' on an item decided on 2026-10-15; remaining occurrences below | gates.log |
| 4 shared sentences | S1 at PREREG.md:548 and :619; S2 at :425 and :626; S3 at :615 (grep -F, one physical line each; equal to the task's text) | gates.log |
| 5 standing instructions | `si_applicable`: 73 rules matched; `si_check_output` on the four files: ok, 0 violations, 0 pattern-checkable rules, all 73 manual: 24 followed, 49 not applicable (table in standing_instructions_verdicts.md); `si_scope_check` against 5d06a27: 27 changed, 27 in scope, 0 out of scope | evidence tar |
| 6 JSON and tests | `json.load` of docs/EQUIVALENT_COMMITS.json ok (exit 0); `WCD_SRC=src python -m pytest -q tests/prereg`: 131 passed, 1 skipped in 35 s | gates.log |
| git am check | `git am` of the 14 patches on a fresh clone of the bundle (SHA-256 0c7de7d507cba8c2e7b60760d81f30cc07b92eb2009bf91fb55b7c21dbe0915a, HEAD 5d06a27e39bd0337ac36b69f58703e26ab1e04a1): exit 0; `git diff --name-only 5d06a27 HEAD` lists only docs/PREREG.md, docs/PAPER_PLAN.md, docs/EQUIVALENT_COMMITS.json and 24 files under experiments/doc_numbers_20261015/; the resulting tree is identical to the tree of my clone; the same holds when the patches are applied from documents_r3_patches.tar.gz | gates.log |
| document builds | PREREG.md: HTML and PDF (45 pages) built from a byte-identical copy, no warnings; the amendment table has 9 body rows and the question table 19. PAPER_PLAN.md: pandoc build, no warnings (the source gate of the helper stops on 10 chat links `{{artifact:...}}` on lines 7 and 215, present since commit 589277b; I did not change them) | evidence tar, doc_build/ |

Gate 3, remaining occurrences of 'proposal' with the reason (PREREG.md unless stated):
:5 and :204 definitions (glossary 'diff review, proposal'; a proposal is text the agent suggested; becomes a rule only when the user approves it); :95 frozen text of version 1 ('the same b* is proposed for P1', commit 0c3ca24 of 2026-10-02; frozen rule text, not edited); :106, :203, :212, :218, :578, :588, :593, :622, :623 and PAPER_PLAN.md:177, :186, :199, :202, :206, :220 state that the agent's proposals were approved on 2026-10-15 (SI-80) or name the three process proposals as approved (SI-81); PAPER_PLAN.md:5 'Working title (proposal)' is the title, not a 2026-10-15 item. 'Open items': :127 (version 2 answers to the open items of version 1, 2026-10-02/03) and :135 (heading of section 8, whose items are version 2 limitations, none decided on 2026-10-15).

## 5. Numbers regenerated by committed scripts (DOC-R04)

| number | where quoted | script | result |
|---|---|---|---|
| 250 earlier X1 and X13 baseline fits | PREREG.md:22, :520 | filler_fit_counts.py | 250 = 100 (job A) + 150 (job B); 50 per task |
| 3,076 = 2,808 + 192 + 76; ratios 2.00, 2.27, 14.4; 204 of 216; cosine 0.28 | PREREG.md:396-398, :418-423 | validation_minibatch_counts.py | 3,076; 2.000, 2.273, 14.439; 204 of 216; 0.281 |
| P2 recount 8 of 54 (8 of 45 without the nine lambda 0.03 rows); +3.7 (+0.7) points | PREREG.md:218, :624 | p2_pilot_recount.py | reference 10/54, discriminator 8/54 (8/45); +3.7 (+0.7); conditioned decoder 15/54 and 17/54, -3.7 |
| pilot ranges (C 0.67 to 0.90; shifts -0.59 to +0.23; middle half -0.09 to 0.00; within cell -0.19 to +0.09; 300 points) | ANALYSIS_SPEC.md:127, :433-436 | pilot_ranges.py | 0.666 to 0.897; -0.590 to 0.231; -0.0936 to 0.0047; -0.188 to 0.090; 300 |
| no-pair expectations 25.9 of 27, 18 of 18, about 14 % of 48, 4.5 % of 24, 0.6 % of 60 | PREREG.md:446-451 | no_pair_expectations.py, no_pair_assumptions.py | 25.92; 18.00; 6.673 = 13.9 %; 1.076 = 4.5 %; 0.3615 = 0.60 %; Monte Carlo 1.48e-5 against exact 1.513e-5 per epoch |
| not regenerated: '0.011', 'coverage 0.91 to 1.00', 'conservative', F7 ratios 0.86-0.89 and 0.53-0.77 | ANALYSIS_SPEC.md:340, :345, :335-337; PAPER_PLAN.md:75-79 | none (inputs are not in the repository) | listed in the README; PAPER_PLAN says so; the ANALYSIS_SPEC sentences are open question 1 |

Found while rerunning from a fresh clone: `.gitignore` has `**/results/` and `*.csv`, so the first commit of the scripts had left their result files out of git, and one stored output (`validation_minibatch_counts.txt`) was older than its script (it printed `[np.int64(12)]`). I force-added the 15 result files and regenerated that one; the first commit now holds scripts and outputs.

## 6. Claim receipt (gate 1)

Rows by source (177 quantities and times): script output 35; manifest of record 8; notebook entry or question file 60; ledger or committed code 28; git commit time or notebook header time 19; lead's messages 27 (not verifiable in the clone). Lead-sourced numbers by location: PREREG.md:213 (two); PREREG.md:218 (two); PREREG.md:219 (two); PREREG.md:366 (two); PREREG.md:367 (two); PREREG.md:540 (13, 49, 5); PREREG.md:541 (42); PREREG.md:543 (6, 7); PREREG.md:560 (two); PREREG.md:562 (six); PREREG.md:565 (17:13); PREREG.md:566 (one, two); PREREG.md:567 (0, one); PREREG.md:568 (19, one); PREREG.md:570 (49, 5, 56); PREREG.md:571 (26,722); PREREG.md:574 (two); PREREG.md:620 (two).
Identifiers cited in added lines: 29 ledger ids (all in CONSTRAINTS.md except SI-88), 38 notebook ids (all in notebook/ except NB-20261015-21, -22, -23, -25), 26 commit ids (all commits in the clone), 6 finding ids (all in the findings file). SI-88 and NB-20261015-21, -22, -23, -25 come from the lead's messages and are not in the clone.
The full table (location, token, kind, source, command) is `claim_receipt.csv` in the evidence tar.

## 7. Open questions and proposed text for files that are not mine

1. docs/ANALYSIS_SPEC.md at 5d06a27 (analysis track): :13-14 'before any X1 adversarial-arm score is examined' becomes 'before any X1 or X13 score is examined', and ':14 "Examined" is defined, as a proposal the user has not yet approved' becomes 'defined by user decision SI-78 (2026-10-15)'; :388-389 keep the quoted old phrase and add that SI-78 replaced it; :491-492 (A.3 item 2) closed by SI-78, the user's statement NB-20261015-15 and SI-88; :64-65 and :496 replace the name `r1_record_sinkhorn_redo.json` by S2 (the redone record overwrites docs/prereg/r1_record.json in place) and close A.3 item 4 (SI-87); :340 and :345 (and :335-337) add 'from the review's simulation; not reproducible from this repository'; :214 as in DOC-R07; the A.3 items now decided by SI-78 to SI-88 are closed. Same phrase 'X1 adversarial-arm score' in scripts/analysis_x1.py:4 and tests/analysis/test_analysis_x1.py:3.
2. docs/PAPER_PLAN.md lines 7 and 215 hold 10 `{{artifact:...}}` chat links (present since 589277b, 2026-10-01); replace them by file names before the tag so the source can pass the document-build gate.
3. Lead: the clone's CONSTRAINTS.md and notebook lack SI-88 and NB-20261015-21, -22, -23, -25, and CONSTRAINTS.md still shows the older SI-87 wording ('v2 and v3'); PREREG cites them and follows the corrected wording.
4. Lead: experiments/x1_x13_score_hashes_20261015/ must exist before the tag, and its README must use the groups of PREREG.md:531-544 (PREREG gives no file counts from it).
5. Display 1 reads 'six X13 runner-test runs' (the lead's wording, kept) while group (4) says 'two test runs ... each scoring 6 latents' (notebook NB-20261002-09 and -10); the lead should check that both describe the same six scored latents in the hash-list README.
6. SI-81 (b): PREREG says the cap-hit decision 'is listed with the results'; the ledger text omits that clause (it was in the approved proposal text). Drop the clause if the ledger is the authority.
7. PREREG question 9 points to docs/ANALYSIS_SPEC.md for the number of cells that never reach b* (finding AN-R15); check the pointer after the analysis patch.
8. The README of experiments/doc_numbers_20261015/ cites docs/ANALYSIS_SPEC.md lines at 5d06a27; they move when the analysis patch is applied. Refresh them then.
9. Other tracks: results folders are ignored by `.gitignore` (`**/results/`, `*.csv`); new result files need `git add -f`.
10. A background reviewer finding on the lead's conversation (id 1ef0acbd-d82d-4197-8975-929784738d81, level warn) says the audit of 2026-10-15 flagged 355 file lines, only 6 of which were shown, so 'no value from the 49 probe files in any review file' rests partly on a sample. PREREG.md:570-573 repeats the audit's result and limits as the lead gave them and does not mention the sample. If the lead classifies the 355 lines, add one clause there saying how many were read; if not, add 'not every flagged line was read'.

## 8. Deviations and limits

- Skill `ponytail` is not in this profile's catalog; I applied its smallest-change aim by hand. No tests added: tests/ belongs to the other tracks; each script asserts the documented numbers and `run_all.sh` is their end-to-end check.
- Blinding: to find the pilot folders I listed the parent folders of the data root and of tier12_v2 (names only). I opened nothing under X1_local, jhpce_tier12, review_sealed, logs/, review/identity_*, the X13 baseline test folder, and did not open score_identity files or run any rule script on X1 or X13 scores.
- The README gives the SHA-256 of the 14 principal inputs and one digest per input ledger; the digest of each of the 1,098 + 36 pilot score files is in results/*_inputs.csv.
- Counts about the audit of 2026-10-15 and the review probes, the time 17:13 and the ids SI-88, NB-20261015-21, -22, -23, -25 rest on the lead's messages and are not verified.
- PREREG PDF checked as extracted text and on 2 table pages only (page 24: amendment-table row 9; page 42 of the final 45-page build: question 9 and 10). The long file path in the question-10 cell runs past the right margin on page 42; the same path is in amendment-table row 8, which I did not look at. The Markdown source is unaffected.
- Not touched: ANALYSIS_SPEC.md, scripts/, tests/, CONSTRAINTS.md, notebook/, README.md, src/; DOC-R03, DOC-R08, DOC-R12 and the ANALYSIS_SPEC parts of other findings are deferred.

## 9. Reproduce

`git clone` the bundle, `git am patches/*.patch`, then in env scvi-api `bash experiments/doc_numbers_20261015/run_all.sh` (47.8 s), `WCD_SRC=src python -m pytest -q tests/prereg`, `python ev/claim_receipt.py <clone> <questions file> <out prefix>` (evidence tar).
