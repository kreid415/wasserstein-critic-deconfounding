# Pass 8 change log (author id P8, 2026-11-05)

Branch `paper`, file `paper/outline.tex` (with `refs.bib`, unchanged). Starting point: `origin/paper` at 13d07c5 (Pass 7), fast-forwarded. All edits here are results-independent; no `\res{}` placeholder was filled. Standing rules checked against `CONSTRAINTS.md`: SI-78 (wording), SI-114 (no image-GAN paragraph; untouched), SI-09 and SI-10 (scVI backbone; unchanged).

## Edits

1. **Update-step bullet moved from the Disc-4 limits list to Methods (tracked).**
   - New sentence in the Methods paragraph on the fixed backbone (line 102, `\added[id=P8]`): each adversary is trained with its standard settings; the reference, pooled and barycenter critics take five update steps per encoder step with a gradient penalty (cited to Gulrajani et al. 2017, key `gulrajani2017improved`, already cited in the Methods), the discriminator takes one; the penalties are compared as they are normally run (Supplementary Note 2). It describes standard settings and is not a limitation.
   - Disc-4: the Pass 7 bullet is removed as a tracked deletion (line 223, macro `\droppeditem`, defined at line 20). The macro prints the struck-through text in the tracked build and nothing (no empty bullet) in a `[final]` build. `changes` cannot nest a deletion around Pass 7's `\replaced{}{}`, so the Pass 7 text sits inside the deletion and Pass 7's earlier wording (the old confound-style limitation) is kept as a `%` comment line above it.
   - No other limitations text mentions the 5-versus-1 update steps. Line 231 (Disc-5, "Not varied or tested") still ends with "Adversary steps reuse one minibatch"; that is a different implementation detail, not the 5-versus-1 count, and was left as is. **Flag for the first author:** delete that sentence if it reads as an update-step limitation.
2. **Limitations bullet in Disc-4 (line 224, `\added[id=P8]`):** one backbone (scVI) with one fixed architecture and latent size, 10 latent dimensions and one hidden layer of 128 units (reviewer R3.2); tasks of the scIB size, not atlas scale (R3.4); the replication across training seeds refits the same eight tasks, so it tests training-seed noise, not new datasets; the biology checks (including label transfer) are exploratory and were defined after the sweep scores had been read. Reviewer ids appear only in this log, not in the paper text. Update steps are not mentioned.
3. **MMD kernel, Supplementary Note 1 (line 258, `\added[id=P8]`).** Read from the code of tag `prereg-tier12-v3` (= 10a1ca5), `src/wcd_vae/wcd/alignment.py`: `BW_MULTIPLIERS = (0.25, 0.5, 1.0, 2.0, 4.0)` (line 49); `mmd_batch_others` (line 109; arm `"mmd"` at line 262) builds `kmat = mean over m of exp(-d2 / (2 d m^2))`, i.e. Gaussian kernels with sigma = sqrt(d) x m, d = number of latent dimensions. With d = 10 the five bandwidths are sigma = 0.79, 1.58, 3.16, 6.32, 12.65 (computed: sqrt(10) x multipliers = 0.7906, 1.5811, 3.1623, 6.3246, 12.6491). The loss is the mean over batches of the squared MMD of one batch against the pooled cells of the other batches, as a biased V-statistic. No call overrides the multipliers (`git grep` over src, scripts, configs and experiments at the tag), and `alignment.py` has no commit between the tag and `origin/final-review-fixes` (`BW_MULTIPLIERS` identical there). The note says the bandwidths are fixed in advance and not tuned, and that the MMD results hold for this kernel.
4. **Li et al. 2022 framing (Disc-4, line 225, `\added[id=P8]`; key `li2022trade`, already in `refs.bib`, DOI 10.1186/s12859-022-05003-3).** The strength sweep is what Li et al. call scalarization; the ranking describes penalties as users run them, not the best front a multi-objective solver could reach; MINE (their adversarially trained mutual-information penalty) is not among the six penalties and was not tested. Checked against the open-access full text: it calls the naive approach scalarization, uses Pareto MTL as the multi-objective method, and uses MINE as the batch-effect measure.
5. **Supplementary Note 3 (line 260, `\added[id=P8]`):** scores use all cells including training cells (scIB convention); Li et al. report held-out cells, "(main-text reference)". No citation in the supplement. Checked against the Li et al. full text: Fig. 2 shows candidates "on test set" and Table 1 reports means over 10 training-testing splits. I did not find a sentence that states outright on which cells the metrics were computed, so the held-out wording rests on those two statements and on your instruction.
6. **Figure 4a label hypervolume.** The caption (line 346) and the figure list already said hypervolume since Pass 6, with "trade-off area" kept as the old argument of the earlier `\replaced{}{}` (history, unchanged). What still said "trade-off area" was the mock image `paper/figures/mock/fig4.png` (panel title and x label). `paper/drafts_20261022/make_mock_figures.py` was edited (two strings) and the six mock figures regenerated: fig1-fig3, fig5 and fig6 are byte-identical to the committed PNGs (sha256 compared); only fig4 changed. It is still a MOCK figure with synthetic data.
7. **Wording.** Lines matching `pre-regist` before the edit: the old argument of the abstract `\replaced` (line 42), a `%` comment (line 103) and the old argument in Supplementary Note 2 (line 250). "Pre-registered comparison" became "Comparison with rules fixed in advance"; "pre-registered pilots" became "pilots whose switching rules were specified before scoring"; "the two pre-registered input pilots" became "the two input pilots with rules fixed in advance". The old argument of Supplementary Note 11 (line 259) also held "registration", "registered" (twice) and "registered plan"; these became "rules fixed in advance", "specified before" and "the plan". All of these were history text (old arguments or a comment), so the active text did not change there. The active text had no "registered"; it had "confirmatory" in 14 places (lines 36, 42, 81, 120, 123, 136, 215, 252, 259, 260 before the edit), all inside earlier `\added` or `\replaced` first arguments, so they were edited in place without new markup: "confirmatory benchmark at fixed strengths" became "benchmark at fixed strengths"; "planned confirmatory run" became "planned fixed-strength run"; "confirmatory-benchmark fits" became "planned-benchmark fits"; any other "confirmatory benchmark" became "planned fixed-strength benchmark". Check: a script that drops `\deleted` text, old arguments and comments finds 0 matches of `regist` and 0 of `onfirmator` in the active text. "Confirmatory" still occurs in struck-through (`\deleted`) history text and old arguments (it names the dropped benchmark, not this study); remove those too if you want a clean raw grep. Two `\tbd{}` texts in Supplementary Note 10 (line 268) and one in Note 12 (line 269) got the same wording change only; they remain unresolved placeholders.
8. **Placeholders left untouched** (active text of the tracked build; line numbers in the Pass 8 `outline.tex`). `\res{}` is result-dependent; `\tbd{}` is to be decided (some also wait on results: upper k, Sinkhorn refits, ranking-analysis dates). Also unresolved by design: the `xx.xx` cells of Tables 2 and 3 and all six MOCK figures.

| line | macro | text (first 110 characters) |
|---|---|---|
| 50 | `\res` | placement against plain scVI, Harmony, Scanorama, sysVI, scANVI at equal tries, including whether any penalty  |
| 51 | `\res` | whether the replication across training seeds agrees with the sweep: the sign of the four large within-pair di |
| 52 | `\res` | one-sentence guidance for scVI users. |
| 52 | `\tbd` | repository address and version tag at submission |
| 82 | `\tbd` | repeat the literature search before submission |
| 115 | `\tbd` | confirm this justification |
| 135 | `\res` | low-signal task--decoder pairs, with plain scVI's batch score, the highest batch score every penalty reaches a |
| 141 | `\tbd` | confirm once the analysis code carries them; state whether any sweep score had been read when the ranking rule |
| 156 | `\tbd` | first author to confirm tools, versions and roles |
| 160 | `\res` | baseline fits completed and failed per method, task and decoder; scoring remains incomplete. |
| 168 | `\res` | fits completed and failed of the 855 replication fits; plain scVI at seeds 0--4 beside the sweep's plain scVI  |
| 169 | `\res` | biology at $b^*$ on the new seeds per penalty, task and decoder beside the sweep reading, with the batch score |
| 170 | `\res` | the sign of each of the four large sweep gaps on the new seeds; Kendall's $\tau_b$ per pair and its mean over  |
| 171 | `\res` | share of sweep near-ties (differences under 0.02) whose replication difference stays within $\pm0.04$; the slo |
| 198 | `\res` | selection score per method, decoder, task at strength from other families\added{, with its difference from pla |
| 202 | `\tbd` | upper $k$: the tools have six settings, the penalties ten or twelve sweep strengths |
| 203 | `\res` | expected best selection score at $k=1$--\replaced[id=P4]{\tbd{upper $k$}}{6} per family and decoder\added[id=P |
| 203 | `\tbd` | upper $k$ |
| 210 | `\res` | whether the replication across training seeds met the agreement rule, and which of the four large gaps or pair |
| 213 | `\res` | guidance for scVI users, including the outcome that no alignment penalty improves on plain scVI. |
| 215 | `\res` | when a penalty is worth adding; strength range; whether a standard tool does better at equal tries. |
| 227 | `\tbd` | redo the coverage simulation for ten strengths and three seeds; for the earlier design of six strengths and fi |
| 236 | `\res` | main conclusion: which penalties keep the most biology at matched batch removal, and whether any beats plain s |
| 237 | `\res` | why it matters for scVI users: when a penalty is worth adding and at what strength. |
| 241 | `\tbd` | confirm wording |
| 245 | `\tbd` | version tag and commit at submission |
| 249 | `\tbd` | competing interests statement; the earlier outline said none declared, confirm with all authors |
| 250 | `\tbd` | funding statement |
| 251 | `\tbd` | authors' contributions statement |
| 252 | `\tbd` | acknowledgements |
| 261 | `\tbd` | Sinkhorn fits refitted after the solver correction: confirm for all eight tasks |
| 262 | `\tbd` | confirm from the analysis code whether a curve can still fail to reach $b^*$; if it cannot, the bottom-rank ru |
| 262 | `\tbd` | minimum number of the three readings |
| 262 | `\res` | low-signal pairs and their values (plain scVI batch score, highest common batch score, gap) |
| 264 | `\tbd` | upper $k$ |
| 268 | `\tbd` | dates for the sweep and the ranking analysis; which sweep scores the strength-choice rules had read before the |
| 268 | `\tbd` | number of planned-benchmark fits made before it was dropped; they are not used |
| 269 | `\tbd` | fits made for it before it was dropped |
| 292 | `\tbd` | upper $k$ |
| 364 | `\tbd` | upper $k$ |

Counts: 17 `\res` and 23 `\tbd` in active text (a `\tbd` nested inside a `\res` on line 203 is counted separately); two more `\tbd` sit in `%` comment lines.

## Build

Built with tectonic 0.16.9 (conda env `latex`) through doc-build-gate (`dg_check_sources`, `dg_build_tex`, `dg_page_ink_check`, `dg_render_pages`): 25 pages, 6 of 6 figures embedded, no undefined citations or references, no overfull boxes, one underfull box (badness 5533, below the 10,000 threshold). The page-ink check reports 17 findings (page 3: ink at the right edge and 17 pt from the bottom edge; later pages: ink 17 pt from the bottom edge, the page numbers, with a 1.7 cm margin). The identical list appears on the Pass 7 PDF built before the edits, so none is new. Pages viewed as images: 3, 10, 11, 12, 13, 14, 16, 18, 20, 23 (every page with an edit); the other 15 pages carry no Pass 8 change and were not viewed. A `[final]`-mode build (`\usepackage[final]{changes}`) already failed on Pass 7 ("Incomplete \iffalse" after line 77, in the Background `\replaced`); not touched. The `\droppeditem` macro was tested in both modes on a small file (tracked: struck-through bullet; final: no bullet).
