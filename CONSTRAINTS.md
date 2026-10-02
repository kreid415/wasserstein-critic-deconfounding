# Standing constraints
<!-- The user's standing rules for this project. Managed by the standing-instructions skill
     (si_add / si_retire / si_link). Entries are never deleted; retire them. Check every deliverable
     against the active entries before presenting it. -->

## SI-01 · active · method
- rule: don't run the discriminator at 10 update steps, that is not how they are trained and will result in failure
- source: "don't run the discriminator at 10 update steps, that is not how they are trained and will result in failure." (user, 2026-10-02)
- added: 2026-10-02
- keywords: d, i, s, c, r, i, m, i, n, a, t, o, r, n, _, c, r, i, t, i, c, u, p, d, a, t, e, s, t, e, p, s

## SI-02 · active · data
- rule: The datasets are the standard scIB datasets; no new datasets (the standard in literature and open problems is scIB)
- source: "The datasets are the standard scIB datasets now, why would we deviates? ... why would we add new datasets when the standard in literature and open problems is scIB?" (user, 2026-10-02)
- added: 2026-10-02
- keywords: d, a, t, a, s, e, t, d, a, t, a, s, e, t, s, t, a, s, k, t, a, s, k, s
- files: s, c, r, i, p, t, s, /, *, m, a, n, i, f, e, s, t, *, ., t, s, v
- check: {"forbid_literal": ["hlca_subset", "bmmc", "ding_pbmc", "cellbench"]}

## SI-03 · active · data
- rule: accept scIB standard keys
- source: "accept scIB standard keys." (user, 2026-10-02)
- added: 2026-10-02
- keywords: b, a, t, c, h, k, e, y, l, a, b, e, l, k, e, y, k, e, y, s

## SI-04 · active · method
- rule: match baselines and experiments to the same number of epochs, batch size, and latents
- source: "We will match baselines and experiments to the same number of epochs, batch size, and latents." (user, 2026-10-02)
- added: 2026-10-02
- keywords: e, p, o, c, h, s, b, a, t, c, h, s, i, z, e, l, a, t, e, n, t, b, a, s, e, l, i, n, e, b, a, s, e, l, i, n, e, s

## SI-05 · active · process
- rule: we will do tier 1 + 2 but we need to reduce wall time
- source: "we will do tier 1 + 2 but we need to reduce wall time." (user, 2026-10-02)
- added: 2026-10-02
- keywords: t, i, e, r, w, a, l, l, t, i, m, e, d, e, s, i, g, n, m, a, n, i, f, e, s, t

## SI-06 · active · process
- rule: We will discuss GPU once the compute wall time is finalized based on design
- source: "We will discuss GPU once the compute wall time is finalized based on design." (user, 2026-10-02)
- added: 2026-10-02
- keywords: G, P, U, c, l, u, s, t, e, r, c, o, m, p, u, t, e

## SI-07 · active · process
- rule: For the preprint, leave it for now and replace it with the next version when we have results
- source: "For the preprint, we will just leave it for now and replace it with the next version when we have results." (user, 2026-10-02)
- added: 2026-10-02
- keywords: p, r, e, p, r, i, n, t, b, i, o, R, x, i, v

## SI-08 · active · method
- rule: switch to scIB native scoring; the comparison is meaningless on a wrong loss
- source: "switch to scIB native. The comparison is meaningless on a wrong loss, so -1 month of time." (user, 2026-10-01)
- added: 2026-10-02
- keywords: s, c, o, r, e, s, c, o, r, i, n, g, m, e, t, r, i, c, m, e, t, r, i, c, s, s, c, I, B

## SI-09 · active · method
- rule: We only care about scVI as people use it
- source: "We only care about scVI as people use it" (user, 2026-10-01)
- added: 2026-10-02
- keywords: s, c, V, I, b, a, c, k, b, o, n, e, b, a, s, e, l, i, n, e

## SI-10 · active · method
- rule: Shared backbone for every arm and baseline: scvi-tools defaults (n_latent 10, 1 layer, ZINB, 90% train split, batch 128, heuristic epochs)
- source: "scvi-tools defaults (Recommended)" (user, ask_user answer to "Which scVI configuration should every arm and baseline share?", 2026-10-02)
- added: 2026-10-02
- keywords: b, a, c, k, b, o, n, e, n, _, l, a, t, e, n, t, s, c, V, I, c, o, n, f, i, g, u, r, a, t, i, o, n, m, a, n, i, f, e, s, t

## SI-11 · active · method
- rule: experiment with batch conditioned arms; review all code for fair comparison between the arms before running
- source: "before running, review all code to ensure fair comparison between the arms. We will also need to experiment with batch conditioned arms" (user, 2026-10-02)
- added: 2026-10-02
- keywords: c, o, n, d, i, t, i, o, n, e, d, f, a, i, r, a, r, m, s

## SI-12 · active · writing
- rule: take the previous paper and make it more rigorous and expand it as needed
- source: "we will take the previous paper and make it more rigorous and expand it as needed" (user, 2026-10-02)
- added: 2026-10-02
- keywords: p, a, p, e, r, o, u, t, l, i, n, e, s, e, c, t, i, o, n, s

## SI-13 · active · process
- rule: recreate all sweeps from base; reproducible from the baseline; no cherry-picking the best from past experiments
- source: "we need to recreate all sweeps from base because this needs to be reproducible from the baseline. we can't just cherry pick the best based on past experiments. This is the reproducible final run." (user, 2026-08-26)
- added: 2026-10-02
- keywords: s, w, e, e, p, l, a, m, b, d, a, g, r, i, d, r, e, u, s, e, p, a, s, t

## SI-14 · active · method
- rule: Barycenter target: cold start (init = minibatch cells), 10 fixed-point iterations per training step; no warm start
- source: "10 iterations" (user, ask_user answer to "Cold-start fixed-point iterations per training step for the barycenter target?", 2026-10-02)
- added: 2026-10-02
- keywords: b, a, r, y, c, e, n, t, e, r, b, a, r, y, _, i, t, e, r, w, a, r, m

## SI-15 · active · data
- rule: Fit the scIB files as distributed (the 'counts' layer exactly as scIB fed scVI); no strict-count main run
- source: "scIB files as distributed" (user, ask_user answer to "Which data should our fits use?", 2026-10-02)
- added: 2026-10-02
- keywords: c, o, u, n, t, s, v, a, l, i, d, s, t, r, i, c, t, n, o, n, -, i, n, t, e, g, e, r

## SI-16 · active · method
- rule: Tier 1+2 lambda design: A1 pilot (full 10-point grid on atac_small, immune, sim1; 3 seeds; both decoders) fixes a 6-point grid per arm family for X1; 5 seeds in both X1 blocks
- source: "A1 pilot, 5 uncond seeds (Recommended)" (user, ask_user answer to the lambda design question, 2026-10-02)
- added: 2026-10-02
- keywords: l, a, m, b, d, a, ,, g, r, i, d, ,, p, i, l, o, t, ,, A, 1, ,, s, e, e, d, s
