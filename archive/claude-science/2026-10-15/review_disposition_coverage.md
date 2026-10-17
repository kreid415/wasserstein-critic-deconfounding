# Disposition of the coverage re-check of 2026-10-15 (diff d556bb1 -> 7ad3eb4)

Report: artifact version 7eeb12f5-e48a-431e-88e0-36e4d63e67f4 (RIGOR_REVIEWER). 0 blockers, 7 fixes, 6 notes.

| finding | severity | disposition | what was done |
|---|---|---|---|
| CV-01 | fix | applied | iDLC full text read (lead re-checked WGAN-GP discriminator, Sinkhorn generator term, reference batch, 5:1 steps, ablation without OT term, no MMD/JS arm); line rewritten, moved to Wasserstein critics, pending dropped; cited in both Gap bullets |
| CV-02 | fix | applied | Li et al. moved to Kernel and OT; preliminary scVI 0.3.0 and adversarially trained MINE stated; named in the Gap bullet with why it is not the JS/W1/MMD comparison |
| CV-03 | fix | applied | GLUE gloss names adversarial alignment of omics layers |
| CV-04 | fix | applied | title-block line no longer names BIOINF-2026-1434 (journal stays pending) |
| CV-05 | fix | not applied | \src{} tags are author-facing provenance, not paper text; the decision-letter pointer tells the author which reviewer comment R0 answers (needed for the response letter) |
| CV-06 | fix | applied | both pending markers name the unknown loss form; scStarCorrect marker notes StarGAN's WGAN-GP objective |
| CV-07 | fix | applied | evidence.md: iDLC entry from full text, line on iDLC 'not cited' updated, scCRAFT conflict marked resolved, abstract-only list extended, coverage works assessed against the gap claim |
| CV-08 | note | applied | Beaconet: per-batch network against the mixture of other batches, soft 1-Lipschitz penalty |
| CV-09 | note | applied | IMGG: L2 reconstruction loss and the authors' reason; 'never named Wasserstein' |
| CV-10 | note | applied | AIF: real-versus-reconstructed discriminator; decoding batch acts as a reference |
| CV-11 | note | applied | 'all strength grids derived from this study's pilot fits (A1--A3), none taken from other experiments' (SI-13: no cherry-picking from past experiments) |
| CV-12 | note | no change | recorded for the user: evidence.md and paper.tex comments on the public branch still name the earlier work |
| CV-13 | note | not applied | optional GLUE Related-work line; the user asked for the eight methods; GLUE is cited where the Introduction names it |
