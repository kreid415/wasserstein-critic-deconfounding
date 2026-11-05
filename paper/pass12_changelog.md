# Pass 12 change log (author id P12, 2026-11-05)

Branch `paper`, starting point `origin/paper` at 0053042 (Pass 11). Fixes the confirmed points of the Pass 11 rigour review
(findings `pass11_rigour_findings.md` / `.json`, Codex RIGOR_REVIEWER, job 87559d47). Tracked changes carry `[id=P12]` (red).
Every text edit is written by `paper/tools/pass12_edits.py` (each old text must occur exactly once, otherwise it stops); every
number it writes is a row of `paper/pass12_numbers.csv`, re-computed from its source before it is used.
Not touched: the supplement blocks owned by branch `supp-items` (Supplementary Note 1 body; the Table S6-S10 and Figure S1-S6
items) and the replication placeholders (Res-2b results, abstract item 7, Disc-1 replication sentences), except one new `\res`
placeholder for the sign-flip sensitivity analysis (F03).

Re-run, from the repository root on the workstation (sources under the data root
`/home/kendall/experiment_data/wasserstein-critic-deconfounding`; Python of the `scvi-api` environment, pandas):

```
git checkout 0053042 -- paper/outline.tex revision/reviewer_response.md paper/lit_search_20261105.md   # Pass 11 text
python paper/tools/pass12_edits.py      # writes the three files
python paper/tools/check_ledger.py      # re-reads every ledger value from its source; exit 0 = OK
python -m pytest paper/tools            # includes the re-run of pass11_edits.py and pass12_edits.py on their starting text
```

## Findings and what changed

| finding | what was done |
|---|---|
| F01 (blocker) | `revision/reviewer_response.md`, R2.3: the false sentence "We did not analyse reference coverage" is replaced by the coverage the paper reports (every cell type in five of the eight tasks; 10 of 17 lung, 14 of 19 human-mouse immune, 3 of 4 simulation 2; Methods, penalties paragraph) and the entropy-rule rationale, and says what was not done (the reference was not varied; whether coverage explains performance was not tested). |
| F02 | Disc-4 bullet rewritten: the collapse intervals are nominal exact binomial 95% intervals; they assume independent fits although a seed's fits share one plain-scVI threshold; no numeric coverage range is stated for the study. Supplementary Note 7 now describes the simulation as an illustrative scenario: script, seed 7, 20,000 replicates, estimand, scenarios A and B and the control with their hand-set parameters (read from `coverage_sim_sweep.py`), coverage (A at least 0.999; B 0.88 and 0.83; control 0.97 and 0.95), Monte Carlo standard error at most 0.003, and that it does not bound the reported intervals; no citation. "Nominal" added to the interval labels in the abstract, Methods (secondary analyses), Res-4 and the Figure 4 caption. |
| F03 | Lead decision: the committed agreement rule is unchanged. Methods (replication paragraph) adds that the sign-flip test treats the 13 pairs as exchangeable although the two decoders of a task share its data and some tasks are related, so its p-value is approximate, and that a sensitivity analysis flipping signs per task (both decoders together, 2^8 patterns over the eight tasks), added after the rule, is reported beside it. New Res-2b placeholder `\res{...}` for that number (the replication pass fills it). Response, "What changed" item 4, says the same. |
| F04 (partly) | Abstract Conclusions: "Within these tasks"; the gain from several strengths is "tried and scored on the same data, which makes that gain optimistic"; scANVI "is trained with the cell-type labels that several biology metrics also use". Conclusions bullet 1: "and scored on the same data". Conclusions bullet 2 rephrased from instructions to observations from these eight tasks, not tested on independent data; optimism caveat on "best of six strengths in 30"; item (iv) states scANVI's label use. Res-7: 11 of the 36 differences from plain scVI are smaller than 0.005 (e.g. -0.0004 for the Sinkhorn divergence on pancreas), against seed SDs of the penalty's selection score of 0.001 to 0.016 in those cells. Table 4 caption: Sinkhorn on pancreas 0.6846 vs plain scVI 0.6850, counted as not exceeding it. Response R3.1 and R3.3 updated to match. Disc-3 unchanged (it already carries the optimism caveat). |
| F05 (minor) | Supplementary Note 4: "so no strength is chosen from the data" -> "so the strength-choice rules choose no strength", plus a pointer that the replication strengths were chosen from the sweep's batch scores. |
| F06 | `paper/pass12_numbers.csv` (machine-readable: id, location, written, source, selector, fmt, note): a row for every abstract number (A01-A44: design counts, mean ranks and task counts, collapse rates with interval limits, placement counts, the 13 ranked pairs, the four large gaps, 160 of 195, the 0.02 threshold, the Introduction's six losses of scIB-E), the numbers Pass 12 writes (B, C, D, E rows) and re-traces of Pass 10 numbers (R, P rows). Each row's location anchor must occur in `paper/outline.tex` and contain the written value. |
| F07 | Broken pointers fixed: `pass11_numbers.csv` sources now `paper/figure_data/paper_numbers.json:<key>` (and the generator line in `pass11_edits.py`); `pass10_numbers.csv` coverage rows are `<path>:<key>`; the workstation counts point to a dated listing `paper/analysis_pass12/planned_benchmark_files_20261105.tsv` (167 model entries, 147 files and 20 links; 160 score entries, 141 files and 19 links), the cluster count 405 and the total 572 to the commit-qualified record `git:98732ff:docs/PREREG.md` (section 9 amendment of 2026-10-22, recorded count, not re-counted); `pass9_numbers.csv`: the `<real task>_fit_types.csv` placeholder path is the tables directory, and the biology-check commit row records the full SHA the source holds. New `paper/tools/check_ledger.py` re-reads every ledger (pass7, 9, 10, 11, 12) and fails on a mismatch or an unresolvable pointer; tests in `paper/tools/test_check_ledger.py`. |
| F08 | Supplementary Note 6: `\citep{luecken2022benchmarking}` replaced by "(main-text reference)"; the main-text ranking paragraph keeps the citation. |
| F09 | Rejected by the lead (the tracked build is intended); no change. |
| F10 | Not in this pass (supplement bodies are added on branch `supp-items`). |
| F11 | Table numbers: the number 2 stays free (`\setcounter{table}{2}` before the ranking table), so the PDF prints Tables 1, 3, 4 and Supplementary Table S10, matching the text and the response; main-text references to Tables 1, 3 and 4 now use `\ref`. Renumbering to consecutive numbers is left to the first author; the supp-items lines that say "Table 4" (Table S7 and S10 items) stay correct under this choice. The response header states the printed numbering. |
| F12 | Collapse denominators: 30 fits; 33 for the discriminator, reference critic and pooled critic on the immune task with the batch-conditioned decoder; 36 where edge strengths were fitted (Disc-4 and Supplementary Note 7; from `tables/secondary.csv` collapse_n). Note 7 states that the 33-fit design is not simulated. |
| F13 | Branch `biology-check`: `experiments/biology_check/make_summary.py` takes the seed SD over the six real tasks (the same tasks as the purity column; it averaged all eight), with an assertion on the task set; `docs/biology_check/summary.md` regenerated; its seed SDs now equal the paper's Table S10 (e.g. discriminator 0.030 conditioned, 0.028 unconditioned). |
| F14 | Methods: "so the adversary is the only correction" -> "so the alignment penalty is the only correction". |
| F15 | `paper/lit_search_20261105.md`: the direct arXiv search was not completed; OpenAlex results may include indexed arXiv records, which does not show that every relevant arXiv preprint was retrieved. |

## Other changes

- `paper/tools/test_pass10_edits.py`: the re-run test compared the Pass 10 script's output with the working files, which Pass 11
  had already changed (the test failed at 0053042); it now compares with the Pass 10 commit 717cc4c.
- New `paper/tools/test_pass11_12_edits.py`: re-runs `pass11_edits.py` on 717cc4c (must give 0053042) and `pass12_edits.py` on
  0053042 (must give the Pass 12 files), and checks that each stops when run twice.
- Header note of `outline.tex`: Pass 12 entry; author definition `P12`.

## Checks run

- `python paper/tools/check_ledger.py`: pass7 191 values re-read, 65 pointers path only; pass9 80 and 435; pass10 3 and 7;
  pass11 10 and 0; pass12 90 and 0; 0 failures, "ledger check: OK". Rows with free-text selectors (pass7/pass9) are checked
  for their path only.
- `scripts/check_tests.py --base origin/paper --src paper/tools -- paper/tools`: 38 passed, OK, no WARN.
- Build: the LaTeX zip extracted into an empty directory and built with tectonic through doc-build-gate (see the commit message).

## Open

- The active abstract is about 483 words by a markup-stripping count (459 before this pass) against the 350-word limit of
  BMC Bioinformatics; it needs a cut by the first author or a later pass.
- Figure 4 panel (b) title still reads "exact 95% interval"; the caption now says nominal. Regenerating the figure is not part
  of this pass.
- Methods paragraph on the reference batch keeps the first author's `\todo{we need justification here}`.
