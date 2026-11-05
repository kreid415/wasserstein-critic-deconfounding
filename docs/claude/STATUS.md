# STATUS: where the project stands (2026-11-05, evening)

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
- Analysis: final run at `sweep_analysis_final_20261105` (code commit 8595a16 on `sweep-paper`, `paper_numbers.json` with 2,999 keys; under the
  data root `/home/kendall/experiment_data/wasserstein-critic-deconfounding`). The earlier draft is `sweep_analysis/draft_8595a16`. Paper numbers are
  traced in `paper/pass7_numbers.csv` and `paper/pass9_numbers.csv`.
- Code of the sweep: tag `prereg-tier12-v3` (the name is kept as an identifier) = 10a1ca5.

## 3. Replication across training seeds (all 855 fitted and scored; extension running)

- 855 fits, seeds 0-4, at the two grid strengths that bracket the first crossing of b* for each of the 13 ranked task-and-decoder pairs.
  Prediction and agreement rule were committed before any replication score was read (commit 3a90d0f, tag `replication-prediction-v1`).
- 855 of 855 fitted and scored (commit 803ed80 on branch `sweep-replication`, last 40 immune_hum_mou fits scored by Slurm job 36889227).
  Agreement tables: commit 61c440d (`replication_readings`, `replication_task_readings`, `replication_large_gaps`). Read the result from those
  files and `experiments/sweep_replication/`, not from this page.
- Extension: 40 further fits (the next sweep-grid strength above the highest replication strength for the 8 readings that did not reach b*, seeds 0-4), as
  the committed rule in `AGREEMENT.md` line 9 requires; the user decided on 2026-11-05 to run it. Run list commit 648f7e8; launchers commit f268ea7
  (15 fits on the workstation, 25 on the cluster as Slurm job 36899485). Running when this file was written; the count of scored extension fits is not known here.
- Paper: Res-2b, abstract item 7 and the Disc-1 bullet and closing sentence hold placeholders (6 `\res`) until the agreement result, with the
  extension, is written in by the agent that owns them.

## 4. Baselines (447 of 448 scored, 0 failed)

- Planned 448 fits: scvi-tools adversary 40, scANVI 40, sysVI 240, Harmony 56, Scanorama 56, PCA 16 (`paper/pass4_changelog.md`).
- One setting is not fitted: Scanorama with 80 neighbours on atac_large (two failed attempts in
  `sweep_baselines/attempts/X13_atac_large_scanorama_l80_c0_s0_f7f8daa2.jsonl`: killed after 7,585 s, then a scipy error after 13,064 s). The final sweep
  analysis reads 447 of 448 (`paper/figure_data/README.md`); the paper reports Scanorama with five of its six settings on that task (Table 4 note, Figure 6a open marker).

## 5. Sweep analysis and biology check (both done)

- Sweep analysis: final run, commit 8595a16; independent placement check at 3280bd4 (`experiments/sweep_checks/`).
- Biology check (exploratory, rules written after the sweep scores were read): branch `biology-check`, results commit 83c452e (run of 2026-11-05 at code
  commit c4a793f, 783 saved fits, no new fits); in the paper as Res-5b, Supplementary Note 12 and Table S10. Known fault: `docs/biology_check/summary.md`
  labels its seed SD as real-task values but averages it over all 8 tasks (`make_summary.py`, `seed_sd`); Table S10 uses the six-real-task value and was
  checked against `tables/summary.csv` in Pass 10. The summary page is not corrected.

## 6. Paper, branch `paper`, Pass 10 (717cc4c)

- Pass 9 (fbd41d7, decf7dc): sweep, placement and biology-check placeholders filled; reviewer response rewritten (replication section pending).
- Pass 10 (717cc4c, log `paper/pass10_changelog.md`, numbers `paper/pass10_numbers.csv`): branch `paper-figures` merged; Figures 2-6 are real (scripts
  `paper/figure_scripts/`, inputs `paper/figure_data/`), Figure 1 is still a placeholder schematic. Captions reconciled with the text; Figure 2b row labels and the
  Figure 3a b* label fixed. `\tbd` items answered from records: literature search repeated 2026-11-05, reference-batch cell-type coverage, coverage
  simulation for ten strengths and three seeds (0.88 and 0.83 with a shared seed threshold), Sinkhorn refit provenance, rule dates, planned-benchmark fit counts
  (about 572; the cluster share of 405 is the repository record's, not re-counted), availability commits.
- Build: tectonic 0.16.9, 29 pages, 6 of 6 figures, no undefined citations or references, 4 underfull boxes.
- Remaining placeholders: 6 `\res` (replication) and 7 active `\tbd` that only the authors can answer: LLM-tool statement, ethics wording, release tag and
  commit, competing interests, funding, authors' contributions, acknowledgements.

## 7. Next steps

1. Wait for the 40 extension fits, score them, and fill Res-2b, abstract item 7, Disc-1 and the response's replication section (agent that owns the agreement test).
2. First author: answer the seven `\tbd` questions (list in `paper/pass10_changelog.md` section 5); make the release tag after the replication text is in.
3. Run the final doc-build-gate build from the extracted zip, save the PDF and zip, and send the response.

Resume with: `git fetch --all --tags`; read `CONSTRAINTS.md` (active rules, SI-78 on wording, SI-114 on the dropped Background paragraph)
and the newest `paper/pass*_changelog.md`.
