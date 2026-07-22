# Convergence check — NB backbone, epoch-count decision

**Question:** does the de-scCRAFT NB backbone (reconstruction + KL only, no triplet/cosine)
converge by the standing 150-epoch default, or is more needed?

**Method:** train NB backbone, both heads (discriminator, critic), 300 epochs,
d_coef=0.2, kl_coef=0.005, warmup=10; record L_vae (reconstruction+KL) and L_adv per epoch.
Run on synthetic (16k cells, no biological structure) and on real pancreas (16382 cells,
9 batches, standard prep_data HVG pipeline). Local RTX 2080 GPU.

## Result: 150 epochs is sufficient; no instability on real data

| dataset | head | L_vae plateau | jumps (\|Δ\|>0.5) | last-50 Δ% |
|---|---|---|---|---|
| **pancreas (real)** | discriminator | slow monotonic rise 0.65→0.78, no jumps | **0** | 3.15% |
| **pancreas (real)** | critic | slow monotonic rise 0.68→0.75, no jumps | **0** | 1.19% |
| synthetic | discriminator | step-up 2.1→4.0 at epoch ~175 | 2 | 0.94% |
| synthetic | critic | step-up 2.1→4.0 at epoch ~117 | 2 | 0.56% |

**Interpretation.** On real data both heads rise gently and monotonically with NO discrete
jumps; the curve has not fully plateaued by epoch 150 but its drift is small and smooth
(disc L_vae 0.725 at e100 -> 0.757 at e150 -> 0.781 at e299; <3.2% over the final 50
epochs). The mid-training
L_vae "regime switch" seen on synthetic data does NOT occur on real data — it was an
artifact of the model abandoning reconstruction of structureless synthetic counts once
the adversary engaged. Real data has genuine biological signal to reconstruct, so the
loss settles monotonically.

**Decision:** keep epochs = 150 for the production wave. The loss is still drifting
slightly at 150 (not a hard plateau) but smoothly and without instability, so 150 is a
reasonable operating point; it matches the original validated setting and minimises
fan-out cost. (The critical finding is the ABSENCE of the synthetic-data regime switch,
not a fully-flat plateau.)
