# Change summary: Pass 7 draft to Pass 15 (branch `paper`, 2026-11-06)

The Pass 7 draft had the sweep Results and Discussion written around result placeholders, mock Figures 1-6 and no supplement
bodies. Passes 8-15 filled every result from its source files (numbers listed in `paper/pass9_numbers.csv` to
`paper/pass15_numbers.csv`, re-read by `paper/tools/check_ledger.py`, 0 failures), added real figures and supplement items, the
replication result, and the fixes of two rigour reviews. Details: `paper/passNN_changelog.md`; response: `revision/reviewer_response.md`.

## Whole paper
- Results, Discussion and Conclusions filled from the sweep, the placement check and the direct biology check (Pass 9).
- Figures 2-6 drawn from the sweep analysis (Pass 10) and Figure 1 (model schematic and one real sweep example, Pass 11).
- Supplement: Note 1 (all loss equations), Tables S6, S7, S9, S10, Figures S1-S6 (supp-items branch, Passes 13-14). Notes 2-12 and
  Tables S1, S2, S4, S5, S8 are still described only (Pass 15 says so in the paper and the response).
- Abstract cut to the BMC limit: 328 words, 348 by a hyphen-split count (Passes 13 and 15); main tables numbered 1 to 3 (Pass 13).
- Clean copy with changes accepted, written by `paper/tools/accept_changes.py` (Pass 15).

## Reviewer 1
- **R1.1 local versus global removal:** Res-5 and Figure 4c report every penalty's local-minus-global reading at b* (Passes 9-10); the
  differences do not explain the ranking (Disc-2).
- **R1.2 disjoint support:** answered by the design; the response now says no support-overlap measurement was added (Pass 9).
- **R1 minor 2 annotation and marker genes:** label transfer (Res-5, Figure S3) and the direct biology check with marker-gene overlap
  (Res-5b, Table S10; Pass 9).
- **R1 minor 3 notation:** Supplementary Note 1 gives every loss with the per-gene likelihood and the discriminator loss on the
  posterior mean, the input every penalty reads (Pass 9; made consistent in the response and outline in Pass 15).

## Reviewer 2
- **R2.2 tuning and Pareto front:** strength sweep framed as Li et al.'s scalarization (Pass 8); trade-off curves, hypervolume and
  good-strength widths drawn from the data (Figures 3-5, Pass 10).
- **R2.3 which cell types collapse; references:** per-cell-type purity, rare types and label-transfer confusion (Pass 9); the
  reference coverage is reported and a false "not analysed" sentence removed (Pass 12); the reference-batch rule is justified (Pass 13).
- **R2.4 strength of claims:** claims scoped "within these tasks"; best-of-k gains called optimistic; scANVI's label use stated
  (Pass 12); collapse intervals called nominal, their simulation illustrative (Pass 12).
- **R2.5 reference design:** ranking at matched batch removal filled (Table 2, Pass 9).
- **R2.6 biological over-correction:** the direct biology check (Methods, Res-5b, Supplementary Note 12, Table S10; Pass 9).

## Reviewer 3
- **R3.1 significance; standard tools:** placement against plain scVI, Harmony, Scanorama, sysVI and scANVI at equal numbers of tries
  (Res-7, Res-8, Table 3, Figure 6, Table S7; Pass 9).
- **R3.2 robustness:** one-backbone limitation stated (Pass 8); replication across training seeds filled in (Pass 14): the rule
  was met only after its own extension step (40 fits), the result before that step is also reported, one reading did not reach
  b*. Pass 15 states that both sign-flip p-values are approximate and that no precision calculation was made for the replication.
- **R3.3 practical guidance:** Disc-3 and the Conclusions give observations tied to numbers, not tested on independent data (Pass 9).
- **R3.4 atlas-scale data, R3 minor 1-2:** stated as limits; label transfer and the biology check added (Pass 9).

## Still with the authors
Seven `\tbd` items: AI-tool statement, ethics wording, release tag, competing interests, funding, contributions, acknowledgements.
