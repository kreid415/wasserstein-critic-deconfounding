# STATUS: where the project stands (2026-11-05)

This file replaces the 2026-10-16 snapshot (still readable in git history at fe6776a). Where this file and a newer
notebook page or branch disagree, trust the newer one. Every number below was read from a file named next to it;
numbers marked "provisional" rest on a partial local copy.

## 1. What the paper is now

An exploratory benchmark of six alignment penalties added to scVI (JS discriminator, reference critic, pooled critic,
barycenter critic, MMD, Sinkhorn divergence) against plain scVI and standard tools (scvi-tools adversary, scANVI, sysVI,
Harmony, Scanorama, PCA). It is not a discriminator-versus-critic comparison. The planned fixed-strength benchmark was
dropped on 2026-11-02 and replaced by the sweep below plus a replication across training seeds. Submission target:
BMC Bioinformatics (resubmission, review BIOINF-2026-1434).

## 2. Penalty-strength sweep (done, analysed)

- Design: 8 tasks, 10 strengths in half-decade steps (0.1 to 3,000), 2 decoders (batch label given or not), seeds 100-102;
  3,024 fits including 96 edge-strength fits (12 per task), plus 9 extension fits on the immune task; 3,033 analysed, 0
  failed (`paper/outline.tex`, Res-1, Pass 7).
- Analysis: draft at `sweep_analysis/draft_8595a16`; final run at `sweep_analysis_final_20261105` (under the data root
  `/home/kendall/experiment_data/wasserstein-critic-deconfounding`). Paper numbers are traced in `paper/pass7_numbers.csv`.
- Code of the sweep: tag `prereg-tier12-v3` (the name is kept as an identifier) = 10a1ca5.

## 3. Replication across training seeds (scoring and agreement test in progress)

- 855 fits, seeds 0-4, at the two grid strengths that bracket the first crossing of b* for each of the 13 ranked
  task-and-decoder pairs. Prediction and agreement rule were committed before any replication score was read
  (commit 3a90d0f on branch `sweep-replication`; tag `replication-prediction-v1`).
- Scoring and the agreement test are being run today by another agent. Provisional: the workstation mirror
  `sweep_replication/jhpce_scores/scores` held 545 score files when this file was written; do not read this as the final count.
- Paper: Res-2b and Disc-1 hold placeholders for the agreement result.

## 4. Baselines (447 of 448 fitted and scored)

- Planned 448 fits: scvi-tools adversary 40, scANVI 40, sysVI 240, Harmony 56, Scanorama 56, PCA 16
  (`paper/pass4_changelog.md`).
- One setting cannot be fitted: Scanorama with 80 neighbours on atac_large. The local record
  (`sweep_baselines/attempts/X13_atac_large_scanorama_l80_c0_s0_f7f8daa2.jsonl`) shows two failed attempts: killed after
  7,585 s (exit -9), then a scipy error ("negative dimensions are not allowed") after 13,064 s. The 447 figure comes from
  the lead's brief; I did not recount it (the local mirror holds only 247 scored baseline fits, the rest are on JHPCE).
  The paper must report this setting as not fitted.

## 5. Biology check (in progress, exploratory)

- Branch `biology-check` at 639c302: rules written before any value is computed, but after the sweep scores were read, so the
  check is exploratory (stated in the paper's limitations). Inputs and manifests: `biology_check_20261105/`.
- Label transfer (Supplementary Note 9) is the part already in the outline; the biology-check tables and summary are not yet in the paper.

## 6. Paper, branch `paper`, Pass 8 (67019a8)

- Pass 8 (results-independent edits; log `paper/pass8_changelog.md`): update-step bullet moved from the Disc-4 limits list
  to Methods (critics take 5 steps with a gradient penalty, discriminator 1: standard settings, not a limitation);
  limitations bullet (one backbone and fixed architecture, not atlas scale, replication reuses the same 8 tasks, biology
  checks exploratory); MMD kernel bandwidths in Supplementary Note 1 (sigma = 0.79, 1.58, 3.16, 6.32, 12.65 for 10 latent
  dimensions, read from `alignment.py` at 10a1ca5); Li et al. 2022 framing (sweep = scalarization; MINE not tested);
  all-cells scoring note in Supplementary Note 3; Figure 4a label "hypervolume"; the words ruled out by SI-78, and the name of the dropped benchmark, are gone from active text.
- Build: tectonic, 25 pages, 6 of 6 figures, no undefined citations or references. All six figures are still MOCK
  (synthetic data); Figures 2-6 need real data and scripts.
- Remaining placeholders: 17 `\res{}` (results: baseline fit counts, replication results, single-strength tables,
  best-of-k, guidance and conclusions) and 23 `\tbd{}` (upper k, literature re-search, declarations, version tag,
  Sinkhorn refit confirmation, ranking-analysis dates, coverage simulation, and others), with line numbers in
  `paper/pass8_changelog.md` section 8; Tables 2 and 3 hold `xx.xx` cells.
- Open question for the first author: the Disc-5 sentence "Adversary steps reuse one minibatch" may read as an update-step limitation.

## 7. Next steps

1. Collect the replication scores and the agreement test from the other agent; fill Res-2b, Disc-1 and the abstract.
2. Fill the baseline `\res{}` with 447 of 448 and the Scanorama knn 80 / atac_large failure.
3. Add the biology-check tables and summary when that run finishes.
4. Replace the MOCK figures 2-6 with real figures and keep their scripts.
5. Resolve the `\tbd{}` list; then the reviewer response and the final PDF and LaTeX zip.

Resume with: `git fetch --all --tags`; read `CONSTRAINTS.md` (active rules, SI-78 on wording, SI-114 on the dropped Background paragraph)
and the newest `paper/pass*_changelog.md`.
