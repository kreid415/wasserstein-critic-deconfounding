"""wcd — authored contribution: reference-based Wasserstein critic, adversarial-head
dispatch, alternative backbones, training engine, evaluation metrics, and the
CV/Pareto experiment harness. Distinct from the upstream scCRAFT backbone (see
``wcd_vae.scCRAFT`` and scCRAFT/NOTICE)."""

from wcd_vae.wcd.adversarial import Discriminator
from wcd_vae.wcd.critic import (
    ReferenceWassersteinLoss,
    gradient_penalty,
    multi_class_gradient_penalty,
)
from wcd_vae.wcd.data import prep_data
from wcd_vae.wcd.evaluation import clisi_graph, compute_lisi, ilisi_graph
from wcd_vae.wcd.hyperparameter import run_comprehensive_nested_cv
from wcd_vae.wcd.training import obtain_embeddings, train_integration_model

__all__ = [
    "Discriminator",
    "ReferenceWassersteinLoss",
    "clisi_graph",
    "compute_lisi",
    "gradient_penalty",
    "ilisi_graph",
    "multi_class_gradient_penalty",
    "obtain_embeddings",
    "prep_data",
    "run_comprehensive_nested_cv",
    "train_integration_model",
]
