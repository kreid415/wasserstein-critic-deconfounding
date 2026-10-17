# QA Audit — experiment code and reported claims

Scope: 3,327 lines across 9 modules (`src/wcd_vae/wcd/`) and 5 harnesses (`scripts/`),
plus the analysis that produced the reported E2 / lambda-sweep claims.
Method: empirical testing against ground truth and closed-form values, not code reading.
Repo state audited: `main` @ dfb5b48.

## Verdict

No correctness bugs in the numerical core. Two defects found in the experiment harness
(both fixed), and **one statistical error in a claim I reported to you** (not a code bug,
but the most consequential finding here).

---

## 1. Statistical error in the headline E2 claim — ACTION REQUIRED

I reported: *critic removes more residual batch signal, delta = -0.043, p = 0.004.*

That p-value is **pseudo-replicated**. It treats 24 (dataset x backbone) pairs as
independent observations, but the 4 unconditioned backbones are evaluated on the *same*
6 datasets. Datasets are the independent unit; backbones are repeated measures within them.

| analysis | n | mean delta | p |
|---|---|---|---|
| as I reported (dataset x backbone) | 24 | -0.0427 | **0.0038** |
| dataset-level (correct unit) | 6 | -0.0427 | **0.0878** |
| Wilcoxon, dataset-level | 6 | — | 0.1562 |

The effect size is identical; only the inference changes. Per-dataset deltas:

| dataset | delta | favors |
|---|---|---|
| atac_small | -0.1330 | critic |
| pancreas | -0.0443 | critic |
| sim2 | -0.0436 | critic |
| immune | -0.0393 | critic |
| sim1 | +0.0037 | discriminator |
| lung | +0.0005 | discriminator |

4 of 6 datasets favor the critic. Notably **lung and sim1 — the two that don't — are also
the datasets with no baseline batch effect to remove**, so they cannot show a removal
difference. That is a defensible argument for their exclusion, but it must be made
explicitly and pre-registered in the text, not applied post hoc.

**Correct reporting options:** (a) state n=6 datasets, p≈0.09, as a consistent trend with
4/6 sign agreement; or (b) fit a mixed-effects model with dataset as a random effect and
backbone as a repeated measure. **Do not report p=0.004.**

## 2. Harness defects found and fixed

**Silent config drop (fixed, dfb5b48).** A config raising inside the eval loop printed a
message and was dropped, leaving no trace in the CSV — so an incomplete wave was
indistinguishable from a complete one by row count. This is plausibly how earlier
"complete" claims went wrong. Failures now write a row with NaN metrics and a `failed`
column.

**Resume would make failures permanent (fixed, dfb5b48).** Once failed rows are recorded,
naive resume logic would treat them as done and never retry. `--resume` now skips rows with
`failed` set. Verified: successful seed skipped, failed seed retried.

## 3. Verified correct — numerical core

Each tested against an independent ground truth:

| component | test | result |
|---|---|---|
| `nb_log_likelihood` | vs `scipy.stats.nbinom.logpmf` | max abs diff 7.4e-07 |
| `gaussian_kl_to_standard_normal` | closed form | KL(0,1)=0.000000, KL(1,1)=0.500000 |
| `gaussian_sample` | variance semantics | var=4 -> std=2.0005 |
| `MultiClassCrossEntropy` | uniform logits == ln(V) | exact for V=2,4,9 |
| `seed_everything` | torch + numpy reproducibility | identical draws |
| `ilisi_graph` | mixed vs separated batches | 0.8297 vs 0.0000 |
| `clisi_graph` | separated vs mixed cell types | 0.0000 vs 0.8311 (lower=better confirmed) |
| `probe_metrics` | 95/5 imbalance, no signal | lift 0.000, majority 0.950 |
| `probe_metrics` | rare class n=2; single class | NaN, no crash, no fabrication |

**Critic (`critic.py`)** — Wasserstein loss -0.002 +/- 0.003 for identically distributed
batches; gradient penalty finite and non-negative across `reference`/`pooled`/`barycenter`;
all three input guards raise `ValueError` rather than passing silently.

**Adversarial sign convention** — correct for both heads. The critic's loss is pre-negated
inside `ReferenceWassersteinLoss`, and the generator's `-lambda * loss_da` composes with that
to minimise Wasserstein distance while the discriminator path maximises cross-entropy. I
traced this specifically because a double-negation here would invert the paper's mechanism.

**Optimizer wiring (`training.py`)** — barycenter anchors are excluded from the critic
optimizer and owned by the generator optimizer (correct: anchors approximate a Frechet mean
and must be pulled toward batches, not pushed away). `z.detach()` verified empirically to
leave encoder gradients unpolluted by the adversary loop.

**Reference modes (E4)** — `fixed` holds, `rotating` cycles deterministically as
`(base + epoch) % n_batches`, `joint` draws uniformly, discriminator correctly gets `None`.

**Resume keys** — match the row-write keys exactly in all three harnesses, including the
extra dimensions that would otherwise collide (E4 on `(ref_design, seed)`, E9 on
`(arm, seed)`). Empty-CSV crash is guarded.

## 4. Not bugs — examined and dismissed

**Warmup recomputes the adversary loss and discards it.** During warmup `opt_d.step()` is
skipped but the forward pass and gradient penalty still run. Measured cost: 5.9 ms per
minibatch, ~0.47 s over 5 warmup epochs against a 1.4 h run. Correctness-neutral, cost
negligible — code clarity issue only.

**`gaussian_sample(mu, var)` takes variance, not log-variance.** Not a bug, but an easy
caller-side error (passing logvar would silently inflate the sampling scale). All current
call sites pass variance correctly. Worth a docstring warning if the API is reused.

## 5. Lint

`ruff check`: 2 findings, 1 auto-fixed, 1 remaining (a `Path.mkdir` style suggestion).
12 files have `ruff format` drift — cosmetic only, not applied to avoid polluting the diff
during an audit.

## Test suite

30 passed, 1 skipped. The skip is the PAGA NaN-fallback test, which self-skips where
scanpy 1.9.8 + igraph 1.0.0 make PAGA non-functional and runs on the cluster.
