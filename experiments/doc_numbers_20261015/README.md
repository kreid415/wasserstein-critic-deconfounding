# Regenerating numbers quoted in PREREG.md, PAPER_PLAN.md and ANALYSIS_SPEC.md (2026-10-15)

Why: finding DOC-R04 of the documents re-check of 2026-10-14 (the project rule is that every quoted number comes from committed code)
found numbers in `docs/PREREG.md`, `docs/ANALYSIS_SPEC.md` and `docs/PAPER_PLAN.md` that could be reproduced only from scripts outside
this repository: the first review's bundle, its `scripts/recompute_g01.py` (cited by notebook entry NB-20261014-31, never
committed) and the re-check's own evidence folder. Each script here recomputes one group of numbers from committed files, the
prepared datasets or the pilot scores, prints each value as the document quotes it, and asserts that it matches at the document's
rounding. Numbers that no committed code regenerates are listed at the end.

Blinding (SI-78: a score is examined when a person reads its value or a script uses it): the scripts read no X1 or X13 score.
`common.data_path` refuses every path that contains `X1_local`, `jhpce_tier12`, `review_sealed`, `identity`, `/logs/`,
`fillers_x1_x13` or `x13_cpu_baselines`; `python common.py` shows the refusal (`results/blinding_guard.txt`).

Labels used below (A1, A3, R1, X1, X3, X7, X8, X13, the batch score B, the biology score C) are defined in the glossary of `docs/PREREG.md`
(section 9, paragraph "Glossary"); P1 and P2 are defined in `docs/ANALYSIS_SPEC.md`.

## Run

    conda activate scvi-api     # Python 3.11.15, numpy 2.4.6, scipy 1.17.1, pandas 2.3.3, anndata 0.12.19, scvi-tools 1.4.2
    export WCD_DATA=/path/to/wasserstein-critic-deconfounding   # default: /home/kendall/experiment_data/wasserstein-critic-deconfounding
    bash experiments/doc_numbers_20261015/run_all.sh            # CPU only, one thread; about 1 minute; stops at the first failed assertion

`run_all.sh` writes `results/*.txt`. Every script also writes `results/<name>_inputs.csv` (path, bytes, SHA-256 of every file it
read; `$WCD_DATA` stands for the data folder), and the last line of its `.txt` gives the SHA-256 of that CSV. Re-running in a fresh
clone gives byte-identical files in `results/`.

## What each script regenerates

Line numbers are those of the commit that adds this file; for `docs/ANALYSIS_SPEC.md` they are the lines after the round-3
analysis fixes (commit 6925a4c).

| quoted number | quoted at | script | output |
|---|---|---|---|
| 250 earlier X1 and X13 baseline fits (X1 arms `none`, `scvi_adv`; X13 arms scANVI, sysVI; tasks atac_large, immune_hum_mou, lung, pancreas, sim2) | PREREG.md:22, :520 | `filler_fit_counts.py` | `results/filler_fit_counts.txt`: 250 manifest rows = 100 (cluster job A: atac_large, immune_hum_mou) + 150 (job B: lung, pancreas, sim2); 50 per task; X1 `none` 50, X1 `scvi_adv` 25, X13 scANVI 25, X13 sysVI 150 (the same arms have 400 rows on all 8 tasks) |
| 3,076 validation minibatches = 2,808 final-epoch (234 pre-fix fits x 12) + 192 early + 76 training-mode (19 fits); 160 replay tags, none among the 234 fits | PREREG.md:418-423 | `validation_minibatch_counts.py` | `results/validation_minibatch_counts.txt` |
| median gradient-norm ratios 2.00 (A1), 2.27 (A3), 14.4 (R1 extension); 204 of 216 finite; median cosine 0.28 | PREREG.md:396-398 | `validation_minibatch_counts.py` | same file (2.000, 2.273, 14.439; 204 of 216; 0.281) |
| P2 recount: discriminator with the unconditioned decoder collapsed in 8 of 54 pilot runs (8 of 45 without its nine lambda 0.03 runs); reference minus discriminator +3.7 points (+0.7 with 45) | PREREG.md:218 (amendment-table row 8), :624 (question 10); ANALYSIS_SPEC.md:559-561 | `p2_pilot_recount.py` | `results/p2_pilot_recount.txt`, `.json` (also prints reference 10 of 54, and the conditioned decoder 15 and 17 of 54, not quoted; 0 of the 9 lambda 0.03 runs collapsed) |
| pilot ranges: plain scVI biology score C 0.67 to 0.90 per cell; shift of C from it -0.59 to +0.23 over 300 points; middle half -0.09 to 0.00; within a cell -0.19 to +0.09 | ANALYSIS_SPEC.md:153-154, :530-532 | `pilot_ranges.py` | `results/pilot_ranges.txt` (unrounded: 0.666 to 0.897; -0.590 to 0.231; -0.0936 to 0.0047; -0.188 to 0.090) |
| first review's pipeline simulation (six scenarios): single-dataset intervals d ± 1.96 s covered 0.85 to 0.88; pooled 95% interval for Delta covered 0.998 to 1.000; 'practically equivalent' declared in none of 2,640 pooled worlds | ANALYSIS_SPEC.md:422-428 | `../analysis_numerics_20261014/first_review/run_simulations.sh` (runs `sim_pipeline.py`) | `../analysis_numerics_20261014/first_review/sim_pipeline_<scenario>.txt` (single-dataset coverage 0.853 to 0.883) |
| '0.011' (reading atac_small, conditioned decoder, on ten points instead of six; reference minus discriminator 0.017 and 0.028) | ANALYSIS_SPEC.md:430-433 | `../analysis_numerics_20261014/first_review/pilot_grid_reading.py` | `../analysis_numerics_20261014/first_review/pilot_grid_reading.csv` (d_shift 0.011039) |
| P2 clustering simulation: Clopper-Pearson coverage 0.9995 and 0.914; Agresti-Caffo 0.949 (paired) and 0.919 (independent) | ANALYSIS_SPEC.md:438-441 | `../analysis_numerics_20261014/first_review/p2_clustering_sim.py` | `../analysis_numerics_20261014/first_review/p2_clustering_sim.json` (0.9488, 0.9186) |
| fits expected to meet a minibatch with no pair of batches: 25.9 of 27 (X7 pancreas reference-anchored MMD), 18 of 18 (X7 sim1), about 14% of 48 (X3 atac_small MMD), 4.5% of 24 (X8 sim2 MMD), 0.6% of 60 (X1 sim1, MMD and Sinkhorn each) | PREREG.md:446-451 | `no_pair_expectations.py`, `no_pair_assumptions.py` | `results/no_pair_expectations.txt` (25.92; 18.00; 6.673 = 13.9%; 1.076 = 4.5%; 0.3615 = 0.60%), `results/no_pair_groups.csv` (every critic-free group), `results/no_pair_assumptions.txt` |

How the no-pair numbers are made: a calculation, not simulated runs. For each critic-free row of the manifest of record the script takes the
cells the row uses (after the row's own subsample, if it has one), the training size ceil(0.9 N), the number of minibatches per epoch (128
cells; the partial last minibatch is kept) and the batch sizes of those cells (obs table of the prepared dataset only; the training-set batch
counts are these counts scaled by n_train/N). It computes the exact probability that a minibatch drawn without replacement from the shuffled
training cells lies entirely inside one batch (for reference-anchored MMD: entirely inside the reference batch or entirely outside it), sums
it over the minibatches of an epoch, and takes 1 - exp(-epochs x expected such minibatches per epoch) as the chance that a fit meets at
least one (minibatches treated as independent); `results/no_pair_groups.csv` adds these chances over the fits of each group.
`no_pair_assumptions.py` reads the loader assumptions (train/validation split, shuffling, last minibatch kept) from the installed
scvi-tools 1.4.2 source and the project code, and checks one case by Monte Carlo (X1 sim1: 1.48e-5 per epoch against exact 1.513e-5;
0.3620 of 60 fits expected).

## Not regenerated by committed code

| number | quoted at | where it comes from |
|---|---|---|
| Sinkhorn-to-W1 ratios 0.86-0.89 and 0.53-0.77 (5 pre-fix latents, 8 minibatches each) | PAPER_PLAN.md:75-79 | the Sinkhorn review log, which is not in this repository; PAPER_PLAN marks them provisional until they are recomputed on the refitted latents |

A document that quotes these numbers should say that they cannot be reproduced from this repository.

## Inputs (SHA-256)

Digest of each script's input ledger (`results/<name>_inputs.csv`): `filler_fit_counts` ac6f723920881d43dcb6a51e669ceb79f20fa0d40c2c2bab57b502025a0410a5;
`validation_minibatch_counts` eea769c89977f0c0725aaed78cf4096e5821fa5e084eca99a36894adc1a1e5c9; `pilot_ranges` e1fc5d6c1f2f7880370d3121f449edb6a688248c9a3e4e318555b842156ad5d8 (1098 A1 score files);
`p2_pilot_recount` 45fc1469f4ebda645034c39e65835dc0fe18c1d619e1616c661ed95980b82362 (1098 A1 and 36 A1_ext_r1_round1 score files);
`no_pair_expectations` 02b29aa7ea96e8ea0a041dae2b882171a3a0a09c64bdeb71350e889f07f6649a; `no_pair_assumptions` dd6019ff171a81277d2b347f67dfcc46d2a555059160f887c0ccd7113bb0d918.

| input | bytes | SHA-256 |
|---|---|---|
| `manifests/paper_manifest_stock_pilot_u5_b10_v3.tsv` | 1,095,357 | `6efbe422c8f09f9884001a84f7d7b363506f435803d238280e00bbb7b7e1cf24` |
| `manifests/a1_extension_r1_round1.tsv` | 5,303 | `2839ff489e8af69b4586d773a1884a6437da8237dbb4319bc12fb65c86d3359a` |
| `docs/prereg/r1_record.json` | 384,376 | `0862f41e63aa045ae395b859fdcc6715070a9e9c632c2448a5f2f40bfa4a804f` |
| `experiments/sinkhorn_fix_20261013/results/validation_12mb.csv` | 1,100,979 | `9e9085fe896c509a29409cfc9ad16ba806515ad7cebe23bab118219e176fb3cc` |
| `scripts/fit_paper_config.py` | 33,591 | `84acaad553f701c84dec3ed95a4c72d5a1263f63b6f301747acdb57387753b0c` |
| `scripts/scvi_adversarial_plan.py` | 34,040 | `83d60502965a8c4d33190d3d2dd98ae9e04ecdafd3a9ce985993191f943184e5` |
| `$WCD_DATA/prepped_scib/atac_large__scib.h5ad` | 173,448,001 | `7bfddbe8e1d7c09637c61ed4974ce35b66463e115fe2bb9fc87552c52ac33269` |
| `$WCD_DATA/prepped_scib/atac_small__scib.h5ad` | 31,838,805 | `6ddc6fc25645bb9ab4dd2e4a5538731aaaa33f2232729baef8300a715a2d814a` |
| `$WCD_DATA/prepped_scib/immune__scib.h5ad` | 270,288,496 | `3d8596256dc6cf9d17b91e974aa27842711c4b1817d53b772d324914b74834e3` |
| `$WCD_DATA/prepped_scib/immune_hum_mou__scib.h5ad` | 639,100,504 | `60fa7788342ebcdd551e63e42017b0b398453151b640af5412074ee4c75ccdaa` |
| `$WCD_DATA/prepped_scib/lung__scib.h5ad` | 473,757,259 | `1b7ad36e535be6628bd3c9dc072552415cc0dac1ae698092a779119d1c31cf64` |
| `$WCD_DATA/prepped_scib/pancreas__scib.h5ad` | 335,958,306 | `d172898be6d237f6698d0723813644b2afc283276bb429e84debf4256bdeeaa9` |
| `$WCD_DATA/prepped_scib/sim1__scib.h5ad` | 502,773,807 | `5c25226fd56187f3826cd661dcabe9ffc00cefb14d5fd92cfd8a65932c9523f0` |
| `$WCD_DATA/prepped_scib/sim2__scib.h5ad` | 933,461,367 | `6e5533ccd18558eb645339e6488e08d20f5cd0d2bb7eb48ade98d0fbf78576ea` |
