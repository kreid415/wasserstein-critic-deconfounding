# Four open questions before tag prereg-tier12-v3: details (2026-10-16)

Branch final-review-fixes, HEAD 6f2436f (pushed). Line numbers are at this HEAD.

## Q1. SI-85 (refusing divergence records with no code version): Sinkhorn arm only, or every arm?
- Ledger SI-85 rule text: "The stage runner refuses a Sinkhorn divergence record that carries no git commit or was made from a checkout with uncommitted changes, as it refuses such results".
- The question the user answered (2026-10-15) also said "a Sinkhorn 'divergence record'".
- Code as committed (scripts/run_stage.py:444-451, `status()`): no arm condition. A record with status `diverged` and `git_dirty` not False is refused unless --allow-dirty; any record (any status) without `git_sha` is refused. This copies the existing latent rule (run_stage.py:426-427), which also has no arm condition.
- Option "Sinkhorn only": add `and row["arm"] == "sinkhorn"` to the two conditions at :447 and :449, and turn the non-Sinkhorn case (v1) of tests/stage/test_run_stage.py to accepted (code report section 10, item 1).
- Effect of "every arm": a divergence record of any arm (MMD, critics, discriminator) made from unknown or uncommitted code stops the stage at start-up, naming the file. Existing X1 records were not opened (blinding), so whether any would be refused is unknown; a refusal is loud, never silent.
- Agent's pick: every arm. A diverged fit counts as a collapse in P2 for every arm, so its code version matters equally; this also matches the latent rule. The ledger text would change to "a divergence record (any arm)".

## Q2. Should a latent with no git commit also be refused?
- Today (run_stage.py:426-427) a latent is refused if `git_dirty` is not False; a latent whose config has no `git_sha` is accepted. SI-85 covers status records only.
- Change: one line after :427, `if not cfg.get("git_sha"): raise InvalidOutput(f"{p}: the latent names no git_sha")`, and one more case in the stage test (code report section 10, item 2).
- Agent's pick: refuse. A latent with no commit cannot be traced to code, and status records are now held to that rule. Same blinding caveat as Q1: a refusal at resume is loud.

## Q3. SI-81 (c): does the tag freeze the reporting script, or only the promise to write it?
- SI-81 (c): the no-pair training-step counts and, per arm, the number of cells not evaluable because the arm never reaches b* (target batch score) "are tabulated by a reporting script written after the runs".
- docs/PREREG.md:453-454 already says: "No code reads no_pair_steps yet; the script affects no rule, so it can follow the tag." docs/ANALYSIS_SPEC.md:95 and :596 say the same without the last clause. Section 2.3 of ANALYSIS_SPEC fixes what "not evaluable" means, so the script only counts.
- Agent's pick: only the promise (keep PREREG:454 as it is; add "the script itself is not frozen by the tag" to ANALYSIS_SPEC section 1 and the ledger). Option "freeze": write the script before the tag and test it on synthetic records.

## Q4. SI-81 (b): keep "the decision is listed with the results" for iteration-cap hits?
- PREREG.md:406-408 (text since commit 6746f9f, 2026-10-14, before the review): "the stage stays stopped until the user decides whether such a fit is recorded as a failed fit; a fit recorded as failed counts as a collapse in P2, and the decision is listed with the results."
- Question 9 of the 2026-10-15 review listed (b) without the listing clause and said "Re-checked as stated in PREREG"; the user chose "Approve all three". The ledger SI-81 (b) omits the clause.
- Agent's pick: keep the clause in PREREG and add it to the ledger, so (b) matches (a), where each NMI-0 decision is listed. Option "drop": delete the clause from PREREG:407-408.
