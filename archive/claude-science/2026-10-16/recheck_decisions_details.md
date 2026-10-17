# Re-check of 2026-10-16: findings that need the user's decision (details)

Source: independent re-check (RIGOR_REVIEWER, Opus), report recheck_final_review_fixes_20261016.md (artifact version 8f59a6b6), verdict NOT READY: 2 blockers, 4 major, 10 minor. The lead checked both blockers against the committed files (file names, folder names and dates only; no score value read).

## Fixed without a decision (shown in the final diff)
- RC-B1 (blocker, confirmed): the 2026-10-15 display audit searched for 46 file stems, all from groups (3) to (5); none of the 177 files of groups (1) X1_local (19) and (2) fillers (158). Fix: re-run the digit-masked audit with all 827 stems and all score folders, covering every session up to the re-run; PREREG and the audit README state the identifier set and the cut-off time. Just before the tag the same audit is repeated for sessions after the cut-off; if it finds any display, the tag waits and you are told.
- RC-B2 (blocker, confirmed): PREREG:524-526 says 1 of 100 pass-A filler rows was scored and "the notebook records no later progress". The hash list has 8 pass-A score files, all immune_hum_mou: 1 dated 2026-10-06, 7 dated 2026-10-12 (the reviewer also named atac_large; that part is wrong). Fix: PREREG and PAPER_PLAN state 158 = pass B 150 + pass A 8; notebook CORRECTION.
- RC-M3: the audit cut-off (2026-10-15 10:33, fixers excluded) is stated, and handled by RC-B1's re-run.
- Minor RC-m1 to m8 and m10: SI-89 and the 2026-10-16 decisions added to amendment rows 5 and 9 and the Q3 row; status lines dated 2026-10-16 with SI-78 to SI-89; the SI-81 (c) clause in PREREG:454 and :623; the three identity-check rescoring times in New York time (the third, 2026-10-13 21:38-21:46, followed the 21:22 spec commit); the power probe script committed or its numbers removed; PAPER_PLAN points to groups (1)-(5); PREREG:22 sentence fixed; the blinding guard in doc_numbers/common.py also refuses review/out, review/verify and other score folders; a check that the end-of-run refusal uses the same status() call.

## Q-A. RC-M2: does a value shown to an AI agent count as "examined"?
- SI-78: "A score is examined when a person reads its value or an analysis script uses it." It does not say whether an agent is a person. SI-88 lists the two past displays as "shown to the agent only, not examined".
- Under the literal wording, Display 2 (the identity check's script compared old and new scores and printed them) could count as examined because a script used the values. The reviewer asks for one sentence that settles this.
- Agent's pick: yes from now on; SI-88 stays as an explicit, dated exception for the two past displays, with its reason. Then the pre-tag audit (RC-B1) is a check of the freeze, not only of disclosure.

## Q-B. RC-M4: register a default outcome for an iteration-cap hit and an NMI-0 case?
- Now (SI-61, SI-65, SI-81 (a), (b)): you decide each case after it occurs; the stage stops. P2 (collapse rate, critic minus discriminator, 5-point margin) is a registered comparison, and a case decided after the fit, knowing its arm, can move it.
- Option "defaults": a Sinkhorn fit that hits the 500,000-iteration cap, and a fit whose scoring hits the NMI-0 case (NMI = normalised mutual information, zero at every clustering resolution), are recorded as failed fits and count as collapses in P2. Any different decision you make is reported as a sensitivity result, and it must be made before that arm's P2 counts are seen. The stage still stops loudly at each case.
- Option "keep case by case": SI-65 and SI-81 unchanged; PREREG names this as a limitation.
- Agent's pick: defaults (a referee can check them; the stop and your review stay).

## Q-C. RC-M1: 8 local X1 rows are not accounted for
- NB-20261012-04 launched 1,125 local X1 fits; the ledger at the stop (NB-20261012-05, PREREG:511-512) gives 19 scored + 1 fitted not scored + 1,097 need fit = 1,117. Divergence records (which count as P2 collapses) are not in the SI-79 hash list.
- Option "count-only tally": a script reads the X1_local ledger's status column and prints only the total per status over all 8 rows (no arm, task or seed, no score); status records of every X1/X13 group are added to the hash list. Counts do not examine a score (SI-78), but a status such as 'diverged' is an outcome, so this is your call.
- Option "disclose only": PREREG states the gap of 8 rows as unexplained.
- Agent's pick: count-only tally without arm breakdown, plus hashing the status records.

## Q-D. RC-m9: datasets whose standard error is raised (SI-83)
- SI-83 keeps a dataset with zero spread and raises its standard error to 1e-6; with d = 0 it enters as a near-certain zero difference, which pulls the pooled result towards "practically equivalent". The `raised` list is recorded in the analysis output, but no report of it is registered.
- Option "report count": next to each declaration, report how many datasets were raised (ANALYSIS_SPEC and analysis_x1.py output; no rule changes).
- Option "count plus exploratory": also an exploratory, not pre-registered result without the raised datasets.
- Agent's pick: report count (no new analysis; you rejected exclusion under SI-83 because it biases results).
