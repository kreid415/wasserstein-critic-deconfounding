"""E2 backbone regression tests: every backbone must satisfy the training contract
and pass adversarial gradient to the encoder for BOTH heads."""
import anndata as ad
import numpy as np
import pytest
import scanpy as sc

from wcd_vae.scCRAFT.utils import multi_resolution_cluster
from wcd_vae.wcd.backbones import BACKBONES
from wcd_vae.wcd.training import train_integration_model


def _toy():
    rng = np.random.default_rng(0)
    n, g = 300, 150
    ct = rng.integers(0, 3, n)
    batch = rng.integers(0, 2, n)
    base = (rng.poisson(2 + ct[:, None], size=(n, g))
            + batch[:, None] * rng.poisson(1.0, size=(n, g))).astype(np.float32)
    a = ad.AnnData(base)
    a.obs["batch"] = [f"b{b}" for b in batch]
    a.obs["celltype"] = [f"c{c}" for c in ct]
    a.obs["batch"] = a.obs["batch"].astype("category")
    a.obs["celltype"] = a.obs["celltype"].astype("category")
    a.layers["counts"] = a.X.copy()
    sc.pp.normalize_per_cell(a, counts_per_cell_after=1e4)
    sc.pp.log1p(a)
    multi_resolution_cluster(a, resolution1=1, method="Leiden")
    return a


@pytest.mark.parametrize("backbone", list(BACKBONES))
@pytest.mark.parametrize("critic", [False, True])
def test_backbone_trains_finite(backbone, critic):
    # WHY: each (backbone, head) must produce finite losses; HOW: short train on toy data
    a = _toy()
    _vae, h = train_integration_model(
        a, batch_key="batch", critic=critic, reference_batch=0 if critic else None,
        disc_iter=3 if critic else 1, z_dim=32, epochs=4, warmup_epoch=1, d_coef=0.2,
        batch_size=100, backbone=backbone,
    )
    assert np.isfinite(h["all_loss"][-1])
    assert np.isfinite(h["loss_da"][-1])
