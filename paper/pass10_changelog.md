# Pass 10 change log (author id P10, 2026-11-05)

Branch `paper`, file `paper/outline.tex`. Starting point: `origin/paper` at decf7dc (Pass 9) with branch `paper-figures` (416c165)
merged (merge commit befebcd; it touched only figure files, scripts, data and captions). Tracked changes carry `[id=P10]` (teal). As in
Passes 8 and 9, a placeholder that sits inside an earlier `\added` or `\replaced` argument is filled in place without new markup
(listed below as "in place"). The replication text (Res-2b, abstract item 7, the Disc-1 bullet and closing sentence, the response's
replication section) was not touched. Every edit is written by `paper/tools/pass10_edits.py` (each old text must occur exactly once,
otherwise it stops); numbers written by it are in `paper/pass10_numbers.csv`. `paper/tools/test_pass10_edits.py` re-runs the script on
the pre-edit text (befebcd) and checks the outline and the ledger come out identical.

## 1. Figures merged

- Figures 2-6 (`paper/figures/figN.png` and `.pdf`, scripts `paper/figure_scripts/`, inputs `paper/figure_data/` with SHA-256, captions
  `paper/figure_captions_pass9.md`). `\includegraphics` of Figures 2-6 now points to `figures/figN.png`; Figure 1 (placeholder schematic) stays
  at `figures/mock/fig1.png`. The "MOCK NEEDS REDRAW" source comments of Figures 2, 4, 5 and 6 are removed. Header note (line 42, in place):
  Figure 1 is a placeholder, Figures 2-6 are drawn from the sweep analysis.
- Captions of Figures 2-6 replaced (`\replaced[id=P10]`, the old caption shown struck through without its "MOCK" prefix). Changes to the proposed
  captions, each checked against `paper/figure_data`:
  1. Figure 2: "the five leave-one-seed-out repeats" corrected to three (the sweep has seeds 100-102; Methods and Supplementary Note 5 say three);
     the bottom-rank sentence now says every curve reached b* here (Supplementary Note 5: 0 of 96 combinations lacked a reading).
  2. Figure 3: no change; the script asserts that no real-task curve ends early because of failed fits (`n_fail` = 0), so the caption says so.
  3. Figure 6: curves are drawn to each method's number of settings (k = 10 for five penalties, 12 for the Sinkhorn divergence, 6 for Harmony and
     sysVI; Scanorama to k = 5 with an open marker at k = 6 that is the mean of five tasks, because 80 neighbours could not be fitted on the large
     chromatin-accessibility task); the discriminator, reference critic and pooled critic have an eleventh setting on the immune task only. The caption
     now states that the Results compare methods at k = 1 and k = 6 (the number of settings of Harmony, Scanorama and sysVI), as Res-8 and Supplementary
     Note 7 do, and that the curves beyond k = 6 are for completeness. Checked from `best_of_k_real_mean.csv` and `paper_numbers.json`: plain scVI 0.6477;
     at k = 1 only MMD (0.6528) is above it; first k above plain scVI: discriminator 4, reference 3, pooled 2, barycenter 2, MMD 1, Sinkhorn 2 (same as Res-8).
     "The unconditioned decoder is in Table 4 and the supplement" corrected to Supplementary Table S7 (Table 4 holds the batch-conditioned decoder only).
  4. Figures 4 and 5: no change (the ringed low-signal point and the "mean over five tasks" for hypervolume agree with Res-5 and Res-6).

## 2. Figure fixes

- Figure 2 panel b now has row labels (shared y axis with panel a; the spacer column was widened so the labels do not touch the colour-bar label).
- Figure 3 panel a, pancreas: the b* label moved from above the panel (it collided with the title) to the foot of the dashed line, inside the panel.
- Rebuilt with `bash paper/figure_scripts/make_figures.sh` (data SHA-256 check, then the five scripts, then `figures/MANIFEST.sha256`); every PNG viewed;
  `check_and_save` reports no text overlaps. The PNGs of Figures 4, 5 and 6 are byte-identical to the figure branch's; a rebuild gives different PDF bytes, so the branch's PDFs of Figures 4-6 were kept (the manifest lists the files as committed).

## 3. `\tbd` items answered from repository records (`paper/outline.tex` line numbers of Pass 10)

| item | answer | source |
|---|---|---|
| abstract, repository (line 54, `\replaced[id=P10]`) | repository address only; no tag in the abstract | |
| literature search (Introduction, in place) | repeated 2026-11-05 in OpenAlex and Europe PMC for works published since 2026-10-01; none compares the objectives; gap statement stands | `paper/lit_search_20261105.md`, `paper/evidence.md`; arXiv API timed out, not repeated |
| reference batch justification (Methods, in place) | the previous sentence ("a reference that contains every cell type") was not true for all tasks: the entropy-chosen reference holds every cell type on pancreas, immune, both chromatin-accessibility tasks and simulation 1, but 10 of 17 on lung, 14 of 19 on human-mouse immune, 3 of 4 on simulation 2. Sentence rewritten from the rationale in `wcd/data.py::select_reference_batch` (evenness of cell-type coverage) and these counts. The first author's `\todo{we need justification here}` note is left in place for the author to remove | `paper/analysis_pass10/reference_coverage.py` and `.json` (anndata 0.12.19, task files of `prepped_scib`) |
| ranking rules and sweep scores read (Methods, in place) | applied "with the changes listed in Supplementary Note 10" (not "unchanged"); the three tasks fitted first had been read by the strength-choice rules, the other five had no scores when the ranking rules were written | `docs/ANALYSIS_SPEC.md` commits 065a486, 5d01da0 (2026-10-22); `docs/SWEEP_ANALYSIS.md` section 3 (15 changes); score-file dates of `pilot5_jhpce/scores` (2026-10-28 to 10-30) and latents (2026-10-22 to 10-30) |
| collapse-interval coverage (Discussion limitations, in place) | 0.9995 (independent), 0.88 (30 fits), 0.83 (36 fits) with one threshold shared by a seed's strengths; so coverage of a nominal 0.95 lies between 0.83 and 1.00. Synthetic simulation, 20,000 replicates, seed 7, same two scenarios as the first review with the parameters stretched by linear interpolation from six to ten or twelve strengths and five to three seeds (my choice; the earlier design gave 0.91 to 1.00). Positive control: constant probability and independent fits cover 0.973 and 0.955 | `paper/analysis_pass10/coverage_sim_sweep.py` and `.json` |
| availability (line 251, in place) | commit 10a1ca5 (fitting and scoring), 8595a16 (sweep analysis), c4a793f (biology check), figure scripts and inputs; the release tag and commit of the submitted version stay `\tbd` | git |
| Sinkhorn refits (Supplementary Note 4, in place) | all Sinkhorn fits of the eight tasks used the corrected solver: 180 fits of the three local tasks refitted 17-19 October at 10a1ca5 (`tier12_v3/A1_sinkhorn`, ledger `git_sha` 10a1ca5, 180/180 scored; edge-strength refits 18, same commit); the five cluster tasks (330 Sinkhorn score files, `scorer_git_sha` 10a1ca5) fitted from 2026-10-22 from the bundle of 10a1ca5 (`PILOT5_BUILD.md` line 124, branch `pilot5-amend`: SI-54 refuses Sinkhorn rows at c88cce3) | ledgers and score files on the workstation; notebook NB-20261017-09, NB-20261019-01 |
| rule dates (Supplementary Note 10, in place) | sweep rules: amendments 2026-10-04 and 2026-10-16 (commit 10a1ca5); pilot extension to five tasks decided 2026-10-21; ranking rules 2026-10-22/23; sweep analysis code 2026-11-01, first all-task debugging run 2026-11-01 21:23, hypervolume and non-dominated share added 2026-11-02 (8595a16, after scores had been read) | git log; `sweep_analysis_dev/all8_debug` |
| planned-benchmark fit counts (Supplementary Notes 10 and 11, in place) | about 572 fits made before the benchmark was stopped on 2026-10-21: 167 on the workstation (all immune, batch-conditioned decoder; 160 with score files: 19 from 2026-10-12, 141 from 2026-10-20/21), 405 on the cluster (the repository's count; **not re-counted**, the cluster was not contacted). Earlier text said "19 scored and not read": changed to "scored ... and not used", because the repository records that no one read a value up to 2026-10-16 and could not verify it for 2026-10-17 to 2026-10-22 (`docs/PREREG.md` section 9, entry of 2026-10-22) | file counts on the workstation (`tier12_v3/X1_local`); `CONSTRAINTS.md` line 658 |

## 4. Biology-check seed SD

`summary.md` of branch `biology-check` labels its "seed SD" column as real-task values, but `make_summary.py` (`seed_sd`) averages `purity_diff_sd` over all
rows of `summary.csv`, i.e. all eight tasks. I recomputed from `tables/summary.csv`: Table S10 uses the six-real-task mean, and every Seed SD cell of Table S10
equals it (discriminator 0.030 and 0.028, reference 0.008 and 0.011, pooled 0.008 and 0.005, barycenter 0.007 and 0.007, MMD 0.006 and 0.005, Sinkhorn
0.008 and 0.008, plain scVI 0.000, scvi-tools adversary 0.009, scANVI 0.003, sysVI 0.023). The eight-task values in `summary.md` differ, for example for
discriminator (0.031 and 0.021), barycenter with the unconditioned decoder (0.027), pooled conditioned (0.010), reference conditioned (0.012) and sysVI (0.032).
**Table S10 is correct; `summary.md` is mislabelled** and is left unchanged on branch `biology-check` (not asked).

## 5. Still open (7 active `\tbd`, and the replication `\res` placeholders of Pass 9)

Author questions, also returned in `key_values.needs_user`: LLM tools statement (line 160), ethics wording (247), release tag and commit (251), competing
interests (255; the repository records "none declared" from the first author on 2026-10-20, commit e560e2b, but all authors must confirm), funding (256),
authors' contributions (257), acknowledgements (258).

## 6. Build

Built with tectonic 0.16.9 through doc-build-gate (`dg_check_sources` with 6 images, `dg_build_tex` with `engine="tectonic"`, `expected_figures=6`):
29 pages, 6 of 6 figures embedded, no undefined citations or references, no overfull boxes, 4 underfull boxes (badness 1210 to 5533; three are in the changed
abstract and availability paragraphs). The page-ink check lists ink within 17 pt of the bottom edge on 21 pages (page numbers) and at the right edge of page 4
(the first author's margin `\todo`), as in Pass 9. The saved zip was extracted into an empty directory and built there.
