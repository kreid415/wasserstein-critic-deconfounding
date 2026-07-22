"""Formulation-space tests: the critic must train under all three alignment targets
(reference, pooled, barycenter), and the barycenter anchors must receive gradient and
move toward the data during training (they approximate the Frechet mean)."""
import anndata as ad
import numpy as np
import pytest
import scanpy as sc

from wcd_vae.scCRAFT.utils import multi_resolution_cluster
from wcd_vae.wcd.training import SCIntegrationModel, train_integration_model


def _toy():
    rng = np.random.default_rng(0)
    n, g = 300, 150
    ct = rng.integers(0, 3, n)
    batch = rng.integers(0, 3, n)  # 3 batches so reference + non-reference heads exist
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


@pytest.mark.parametrize("formulation", ["reference", "pooled", "barycenter"])
def test_formulation_trains_finite(formulation):
    a = _toy()
    ref_name = "b0" if formulation == "reference" else None
    ref_idx = 0 if formulation == "reference" else None
    _vae, hist = train_integration_model(
        a, disc_iter=3, z_dim=32, epochs=4, warmup_epoch=1, batch_size=128,
        critic=True, reference_batch=ref_idx, reference_batch_name_str=ref_name,
        formulation=formulation,
    )
    assert np.all(np.isfinite(hist["loss_da"])), f"{formulation}: non-finite adversarial loss"


def test_barycenter_anchors_move():
    a = _toy()
    m = SCIntegrationModel(a, "batch", z_dim=32, critic=True, reference_batch=None,
                           seed=0, formulation="barycenter")
    assert m.D_Z.anchors is not None
    a0 = m.D_Z.anchors.detach().clone()
    m.train_model(a, "batch", epochs=4, d_coef=0.2, kl_coef=0.005, triplet_coef=1,
                  cos_coef=20, warmup_epoch=1, disc_iter=3, batch_size=128)
    moved = float((m.D_Z.anchors.detach() - a0).abs().mean())
    assert moved > 1e-4, f"barycenter anchors did not move (moved={moved})"


def test_reference_only_has_no_anchors():
    a = _toy()
    for formu in ("reference", "pooled"):
        ref = 0 if formu == "reference" else None
        m = SCIntegrationModel(a, "batch", z_dim=32, critic=True, reference_batch=ref,
                               seed=0, formulation=formu)
        assert m.D_Z.anchors is None, f"{formu} should not allocate anchors"
