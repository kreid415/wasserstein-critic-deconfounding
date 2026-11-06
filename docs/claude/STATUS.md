# STATUS: where the project stands (2026-11-06)

This file replaces the 2026-11-05 evening snapshot (b7c99d3; older ones in git history). Where this file and a newer change log
or branch disagree, trust the newer one. Numbers below are copied from the files named next to them.

## 1. What the paper is

An exploratory benchmark of six alignment penalties added to scVI (JS discriminator, reference, pooled and barycenter Wasserstein
critics, MMD, Sinkhorn divergence) against plain scVI and standard tools (scvi-tools adversary, scANVI, sysVI, Harmony, Scanorama,
PCA), read at matched batch removal over a penalty-strength sweep, with a replication across training seeds. Target journal:
**BMC Bioinformatics**, Research article (confirmed by the user 2026-11-06). The earlier version was rejected at Bioinformatics
(BIOINF-2026-1434); `revision/reviewer_response.md` answers those reviews. The outline already follows the BMC structure
(Background, Methods, Results, Discussion, Conclusions, Declarations with the BMC subheadings).

## 2. Experiments: all done, nothing left to run

- Sweep: 8 tasks, strengths 0.1-3,000 in half-decade steps plus edge strengths, 2 decoders, seeds 100-102; 3,033 fits analysed, 0
  failed. Final analysis commit 8595a16 (`sweep-paper`), independent placement check 3280bd4. Sweep code: tag `prereg-tier12-v3`
  = 10a1ca5 (tag name kept as an identifier).
- Baselines: 447 of 448 planned fits fitted and scored; Scanorama with 80 neighbours on the large ATAC task could not be fitted
  and is reported as missing.
- Replication across training seeds (branch `sweep-replication`, a475ea3): 895 fits (855 + 40 extension fits the committed rule
  requires), all scored. The agreement rule is met: 4 of 4 large-gap signs kept, mean Kendall tau-b 0.456 over 13 pairs,
  sign-flip p = 0.00049; before the extension step 3 of 4 (one undefined), tau-b 0.426. 77 of 78 readings reach b*. Sensitivity
  analyses defined after the rule: per task p = 0.0039, per task family p = 0.0156. Results in
  `experiments/sweep_replication/agreement_result_with_extension/`.
- Direct biology check (exploratory): branch `biology-check`, 83c452e; Res-5b, Supplementary Note 12, Table S10.

## 3. Paper: branch `paper`, Pass 18 (2ff8608)

- Text, figures (1-6, S1-S6) and the whole supplement are written: Supplementary Notes 1-12, Tables S1, S2 and S4-S11 (S3 removed
  with the dropped fixed-strength benchmark; S11 is the label-transfer table of Note 9, numbered in Pass 18).
- Checks at Pass 18 (`paper/pass18_changelog.md`): `paper/tools/check_ledger.py` 1,117 values re-read from their sources, 0
  failures (needs the data root, workstation); tests 83 passed; abstract 328 words (limit 350); tracked PDF 64 pages, clean PDF 55
  pages, 13 of 13 figures, no failing log class; every page of both PDFs viewed across Passes 17-18. Built PDFs in `paper/build/`;
  re-run commands in `paper/REPRODUCE.md`; checksums in `paper/MANIFEST.sha256`.
- Change history: `paper/passNN_changelog.md` (one per pass) and `paper/change_summary.md`.

## 4. Repository

- 2026-11-06: the figshare browser session cookie in the README download command was removed from the tips of the 27 branches that
  still had it (`main` included; one commit each, no history rewrite, by the user's decision). Old commits, 4 tags and pull-request
  refs 2-4 still contain it; invalidating the figshare session is the user's step. Local worktrees of those branches are one commit
  behind GitHub: `git pull` before working on them.
- `*.tex` and `*.csv` are git-ignored repo-wide: add paper sources and ledgers with `git add -f` (Pass 18 found the Pass 16 ledger
  had never been committed).

## 5. Next steps (all with the authors)

1. Fill the seven `\tbd` items: AI-tool statement, ethics wording, release tag and commit, competing interests, funding,
   authors' contributions, acknowledgements (kept empty on purpose until the first author's final draft).
2. Then: tag the release, write the tag and commit into the availability statement, rebuild both PDFs from the saved zip
   (doc-build-gate), re-run `check_ledger.py`, regenerate `paper/MANIFEST.sha256`.
3. Log out of all figshare sessions (or change the figshare password) so the cookie in old commits is useless.

Resume with: `git fetch --all --tags`; read `CONSTRAINTS.md` (branch `paper`; SI-78 on wording) and the newest
`paper/pass*_changelog.md`.
