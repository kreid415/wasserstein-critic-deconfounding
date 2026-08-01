"""Regression tests: the adversarial head must actually train and exert pressure.

# WHY: A silent failure mode shipped a full experiment matrix -- the discriminator
#      sat pinned at ln(V) (exact chance for a V-way classifier) on every dataset,
#      so it emitted no gradient, iLISI was flat across a 100x lambda_adv sweep, and
#      every reference design / formulation arm collapsed to the same value. Nothing
#      in the metric suite flagged it. These tests assert the two invariants that
#      would have caught it at commit time.
"""

import numpy as np
import pytest
import torch

from wcd_vae.wcd.adversarial import Discriminator


def _separable_z(n=512, d=16, v=4, sigma=0.3, seed=0):
    """Latent codes trivially separable by batch label."""
    torch.manual_seed(seed)
    centers = torch.randn(v, d) * 6.0
    labels = torch.randint(0, v, (n,))
    return centers[labels] + torch.randn(n, d) * sigma, labels, v


def test_discriminator_head_learns_batch_from_separable_latent():
    """The CE head alone must beat chance on a perfectly separable latent.

    If this fails, the head/loss/optimiser wiring is broken independently of any
    training loop -- CE should collapse toward 0, never sit at ln(V).
    """
    z, labels, v = _separable_z()
    head = Discriminator(n_input=z.shape[1], domain_number=v, critic=False)
    opt = torch.optim.Adam(head.parameters(), lr=1e-3, betas=(0.5, 0.9))

    for _ in range(200):
        opt.zero_grad()
        loss, gp = head(z, labels)
        (loss + gp).backward()
        opt.step()

    chance = float(np.log(v))
    assert loss.item() < chance - 0.5, (
        f"discriminator stuck near chance: CE={loss.item():.4f} vs ln({v})={chance:.4f}"
    )


def test_discriminator_is_not_pinned_at_chance_in_training_loop():
    """End-to-end: with lambda_adv=0 the adversary should master batch prediction.

    lambda_adv=0 means the generator applies NO counter-pressure, so a working
    adversary must drive CE well below ln(V). Pinned-at-chance here is the exact
    signature of the shipped bug.
    """
    sc = pytest.importorskip("scanpy")
    ad = pytest.importorskip("anndata")
    pd = pytest.importorskip("pandas")
    from wcd_vae.wcd.training import train_integration_model

    rng = np.random.default_rng(0)
    n, g = 400, 300
    counts = rng.negative_binomial(5, 0.3, size=(n, g)).astype("float32")
    batch = np.array(["b0"] * (n // 2) + ["b1"] * (n // 2))
    counts[batch == "b1", :80] += 25  # strong, easily-detected batch effect

    adata = ad.AnnData(
        X=counts,
        obs=pd.DataFrame(
            {
                "batch": pd.Categorical(batch),
                "celltype": pd.Categorical([f"T{i % 3}" for i in range(n)]),
            }
        ),
    )
    adata.layers["counts"] = adata.X.copy()
    sc.pp.normalize_total(adata)
    sc.pp.log1p(adata)

    _, history = train_integration_model(
        adata, batch_key="batch", critic=False, d_coef=0.0, disc_iter=1,
        z_dim=16, epochs=25, warmup_epoch=2, batch_size=256, backbone="NB",
    )
    chance = float(np.log(2))
    best = min(history["loss_da"])
    assert best < chance - 0.15, (
        f"adversary never learned batch: min loss_da={best:.4f} vs chance ln(2)={chance:.4f}"
    )


def test_adversarial_pressure_changes_the_objective():
    """Raising lambda_adv must measurably change the adversarial loss.

    A no-op head produces the same loss_da regardless of lambda -- the flat-sweep
    signature that made E1/E4/E9 uninterpretable.
    """
    sc = pytest.importorskip("scanpy")
    ad = pytest.importorskip("anndata")
    pd = pytest.importorskip("pandas")
    from wcd_vae.wcd.training import train_integration_model

    rng = np.random.default_rng(1)
    n, g = 400, 300
    counts = rng.negative_binomial(5, 0.3, size=(n, g)).astype("float32")
    batch = np.array(["b0"] * (n // 2) + ["b1"] * (n // 2))
    counts[batch == "b1", :80] += 25

    def run(lam):
        adata = ad.AnnData(
            X=counts.copy(),
            obs=pd.DataFrame(
                {
                    "batch": pd.Categorical(batch),
                    "celltype": pd.Categorical([f"T{i % 3}" for i in range(n)]),
                }
            ),
        )
        adata.layers["counts"] = adata.X.copy()
        sc.pp.normalize_total(adata)
        sc.pp.log1p(adata)
        _, hist = train_integration_model(
            adata, batch_key="batch", critic=False, d_coef=lam, disc_iter=1,
            z_dim=16, epochs=25, warmup_epoch=2, batch_size=256, backbone="NB",
        )
        return hist["loss_da"][-1]

    da_off, da_on = run(0.0), run(1.0)
    assert abs(da_on - da_off) > 0.05, (
        f"lambda_adv has no effect on the adversarial loss: "
        f"loss_da(lambda=0)={da_off:.4f} vs loss_da(lambda=1)={da_on:.4f}"
    )
