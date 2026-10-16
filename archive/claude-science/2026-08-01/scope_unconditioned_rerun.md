# Scope: unconditioned re-run

**Status:** proposal for decision. No compute launched.

## Why a re-run is needed

Two independent defects invalidate the current native-backbone matrix.

**1. Mixing axis — decoder conditioning removes the batch effect before the adversary runs.**
`Poisson`, `NB`, `ZINB`, `LDVAE` default to `conditioned=True`; the decoder receives the
batch one-hot and reconstructs batch-specific expression from the label, so the encoder has
no incentive to encode batch in `z`. Measured on pancreas at lambda_adv = 0 (adversary
unopposed, z_dim 256):

| backbone | batch probe | majority | min loss_da | chance ln(9) |
|---|---|---|---|---|
| NB (conditioned) | 0.158 | 0.228 | 1.774 | 2.197 |
| NB_uncond | 0.404 | 0.209 | 1.047 | 2.197 |

Conditioned NB predicts batch *below* the majority baseline. Consequence: loss_da pinned at
exactly ln(V) on all 8 datasets, iLISI flat across a 100x lambda sweep (discriminator range
0.0008-0.0047), and every E4 design / E9 arm collapsing to identical values.

**2. Conservation axis — removing the triplet loss removed cluster compactness.**
Verified that unconditioned backbones do NOT fix this: ARI 0.030-0.052 and asw_celltype
0.496-0.497 across all four unconditioned configs. The latent still *encodes* cell type
(linear probe 0.724 vs 0.333 majority) but not *geometrically* -- and scIB's ARI/NMI/ASW all
require compact separable clusters, which the triplet loss used to enforce.

**These need separate fixes.** An unconditioned re-run addresses defect 1 only.

## What the re-run buys

A meaningful critic-vs-discriminator comparison on the mixing axis, plus a genuine finding:
*adversarial latent alignment is largely redundant with decoder batch-conditioning* --
supported by the probe table above and testable across all four unconditioned backbones.

## Cost (measured, not assumed)

Runtime fitted on two DSAI CPU anchors (16 cores): t(min) ~ 2.79e-6 * n_cells^1.57. The
exponent > 1 because the scIB metric suite (LISI, silhouette, kNN graph), not training,
dominates. Critic runs carry a 2.5x penalty (10 disc_iter vs 1).

| option | runs | CPU-hours | wall @6 parallel | wall @20 parallel |
|---|---|---|---|---|
| A. E1+E2+E4+E9+E5, 8 ds, 4 backbones | 955 | 1679 | 11.7 d | 3.5 d |
| B. E1+E2+E5, 8 ds, 4 backbones | 688 | 1174 | 8.2 d | 2.4 d |
| **C. E1+E2+E5, 6 light ds, 4 backbones** | **516** | **272** | **1.9 d** | **0.6 d** |
| D. E1+E5, 8 ds, NB_uncond only | 496 | 847 | 5.9 d | 1.8 d |

The two heavy datasets (atac_large 85k, cross-species 98k) account for ~77% of all cost:
dropping them takes the core scope from 1174 to 272 CPU-hours while retaining 75% of runs.

## Recommendation: option C first, heavy datasets second

1. **Run C now** (~2 days at the concurrency DSAI has been giving us). It answers the
   scientific question on 6 datasets and confirms the adversary engages before spending
   the expensive compute.
2. **Gate the heavy datasets on C's result.** If the effect is clear at 6 datasets, add
   atac_large + cross-species for completeness (+900 CPU-h). If C shows the adversary
   still doesn't engage, the heavy runs would have been wasted.

This is deliberately the reverse of the earlier "keep grinding" call on E1/E4/E9 -- that was
right when the only question was coverage, but here the *substrate itself* is unverified, so
cheap-first is the better sequencing.

## Decisions needed before launch

1. **Conservation axis.** Options: (a) restore the triplet loss as an optional term and
   report both with/without -- keeps scIB metrics meaningful and directly tests your
   aux-loss hypothesis; (b) keep pure native VAEs and switch conservation to probe-based
   metrics (kNN label accuracy), which measure what the latent actually has; (c) report
   mixing only. I recommend (a): it is the only option that makes the paper comparable to
   the submitted version and to other scIB-benchmarked methods.
2. **Backbone set.** All 4 unconditioned (Gaussian, NB_uncond, ZINB_uncond, LDVAE_uncond),
   or NB_uncond alone as primary with the rest as a generalization panel?
3. **Keep conditioned arms as a contrast?** They cost little at 6 datasets and they are the
   evidence for the redundancy finding -- I would keep them, reframed as a deliberate
   conditioning-vs-adversary comparison rather than as failures.

## Pre-flight guards

- `tests/wcd_vae/test_adversarial_engages.py` (committed, e84c8a2) -- 3 invariants.
- Add a **real-data** gate before the full fan-out: assert min(loss_da) < ln(V) - 0.15 on
  pancreas NB_uncond. The synthetic tests pass even when the real config fails, so this is
  the guard that actually binds.
- Log `loss_da` trajectory + a batch linear-probe accuracy per run, so a silent no-op is
  visible in the results CSV rather than only in a post-hoc audit.
