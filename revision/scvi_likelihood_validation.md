# scvi-tools likelihood validation (E2 backbones)

The rebuilt-E2 backbones attach our adversarial head to VAE decoders whose *generative
models* match scvi-tools. Because published scVI has no adversarial head, the decoder
architecture + head are our controlled additions; the reconstruction **likelihood** is what
must match scVI. We validate this numerically.

**Setup.** Shared random inputs (N=64 cells, G=40 genes): counts x, NB mean mu, inverse
dispersion theta, zero-inflation logits zi. Reference log-probabilities from scvi-tools
1.4.2 `NegativeBinomial(mu, theta)` and `ZeroInflatedNegativeBinomial(mu, theta, zi_logits)`,
summed over genes per cell.

**Result** (our `log_nb_positive` / `ZinbVAE._log_zinb` vs scvi-tools):

| Likelihood | max abs diff | mean abs diff | allclose(atol=1e-3) |
|---|---|---|---|
| Negative binomial | 0.000e+00 | 0.000e+00 | True |
| Zero-inflated NB | 6.10e-05 | 1.55e-05 | True |

NB is bit-identical; ZINB differs only at float32 rounding. The E2 backbones' generative
models are therefore faithful to the published scVI implementations. Reference: scvi-tools
1.4.2, torch 2.12.1 (env `scvi-ref`).
