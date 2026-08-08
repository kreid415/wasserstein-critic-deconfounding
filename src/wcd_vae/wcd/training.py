"""Adversarial-deconfounding training engine — AUTHORED CONTRIBUTION (K. Reid).

# WHY: Train a VAE backbone against an adversarial head so the latent space mixes
#      batches while its reconstruction likelihood preserves biology. The backbone loss
#      is reconstruction + KL only; the adversarial head is the sole shared component
#      across backbones, which isolates the objective under study.
# HOW: Alternating optimisation — a stratified epoch sampler guarantees the reference
#      batch appears in every mini-batch (required for a stable critic gradient penalty);
#      the head takes ``disc_iter`` steps, then the encoder/generator takes one step
#      minimising L_backbone + lambda_adv * L_adv (lambda_adv = ``d_coef``, zero in warmup).
The backbone is injected, so this engine trains any ``wcd.backbones`` architecture
unchanged. Provenance: the training-loop scaffold was originally derived by modifying
scCRAFT's ``SCIntegrationModel`` (a single-dataloader, discriminator-only loop) and has
since been substantially rewritten — reference-batch handling, the stratified epoch
sampler, the per-batch adversarial step, critic support, the backbone-agnostic objective
(reconstruction + KL only), and removal of the triplet/cosine terms are authored. It is
not a clean-room reimplementation; the numerical primitives it calls (see
``wcd.primitives``) are.
"""

import copy
import time

import numpy as np
import pandas as pd
import scipy
from sklearn.decomposition import PCA
import torch
import torch.nn as nn
import torch.optim as optim

from wcd_vae.wcd.adversarial import Discriminator
from wcd_vae.wcd.backbones import build_backbone
from wcd_vae.wcd.primitives import inference_dataloader, init_batchnorm_weights, seed_everything


class SCIntegrationModel(nn.Module):
    def __init__(
        self,
        adata,
        batch_key,
        z_dim,
        critic,
        reference_batch,
        seed=None,
        backbone=None,
        formulation="reference",
    ):
        super().__init__()
        self.p_dim = adata.shape[1]
        self.z_dim = z_dim
        self.v_dim = np.unique(adata.obs[batch_key]).shape[0]
        # store head type + reference index so train_model can enable the E4 modes
        # (rotating/joint) even when the reference is given as an int, not a name.
        self.critic = critic
        self.reference_batch = reference_batch
        # formulation selects the critic alignment target: reference | pooled | barycenter.
        self.formulation = formulation

        # WHY: the adversarial head is held fixed while the z-producing backbone varies;
        #      HOW: build the named native VAE. NB (conditioned) is the primary backbone.
        self.VAE = build_backbone(backbone or "NB", self.p_dim, self.v_dim, self.z_dim)
        self.D_Z = Discriminator(
            n_input=self.z_dim,
            domain_number=self.v_dim,
            critic=critic,
            reference_batch=reference_batch,
            formulation=formulation,
        )

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.VAE.to(self.device)
        self.D_Z.to(self.device)

        if seed is not None:
            seed_everything(seed)
        self.VAE.apply(init_batchnorm_weights)
        self.D_Z.apply(init_batchnorm_weights)

    def _prepare_tensors(self, adata, batch_key, reference_batch_name_str=None):
        """
        One-time setup: Moves data to GPU and builds index maps for sampling.
        Determines definitive reference batch index based on provided name string.
        """
        # 1. Convert Feature Matrix
        if scipy.sparse.issparse(adata.X):
            X_tensor = torch.tensor(adata.X.toarray(), dtype=torch.float32)
        else:
            X_tensor = torch.tensor(adata.X, dtype=torch.float32)

        if "counts" in adata.layers:
            if scipy.sparse.issparse(adata.layers["counts"]):
                X_raw_tensor = torch.tensor(adata.layers["counts"].toarray(), dtype=torch.float32)
            else:
                X_raw_tensor = torch.tensor(adata.layers["counts"], dtype=torch.float32)
        else:
            # Fallback or error
            raise ValueError("Raw counts required for NB loss")

        # 2. Prepare Labels and Batch Indices
        unique_batches = adata.obs[batch_key].sort_values().unique()
        batch_map = {b: i for i, b in enumerate(unique_batches)}

        reference_batch_idx = None  # Default safe fallback
        if reference_batch_name_str is not None:
            if reference_batch_name_str in batch_map:
                reference_batch_idx = batch_map[reference_batch_name_str]
            else:
                # This happens if prep_data determined a largest batch, but subsequent
                # filtering in this step somehow removed it (unlikely but possible safety check).
                raise ValueError(
                    f"Reference batch name '{reference_batch_name_str}' not found in the current batch mapping."
                )

        batch_indices = np.array([batch_map[b] for b in adata.obs[batch_key]])

        # Create tensors (initially on CPU)
        batch_tensor = torch.tensor(batch_indices, dtype=torch.int64)

        # Move to device (e.g., GPU)
        X_tensor = X_tensor.to(self.device)
        batch_tensor = batch_tensor.to(self.device)
        X_raw_tensor = X_raw_tensor.to(self.device)

        data_dict = {
            "X": X_tensor,
            "X_raw": X_raw_tensor,
            "batch_labels": batch_tensor,
        }

        # Pre-calculate indices for each batch for fast sampling
        # batch_tensor is now on GPU, so 'idxs' will also be on GPU
        batch_indices_map = {}
        for i in range(len(unique_batches)):
            idxs = (batch_tensor == i).nonzero(as_tuple=True)[0]
            batch_indices_map[i] = idxs

        # Return the index along with the data
        return data_dict, batch_indices_map, reference_batch_idx

    def _sample_epoch_indices(self, batch_indices_map, sample_per_batch=512, batch_size=1024):
        """
        Generates stratified indices for one epoch.
        Ensures that EVERY mini-batch contains samples from all batches (including the reference),
        which is critical for the stability of the Wasserstein Critic's gradient penalty.
        """
        n_classes = len(batch_indices_map)

        if sample_per_batch is None:
            total_cells = sum(len(idxs) for idxs in batch_indices_map.values())
            # Distribute N samples evenly across K batches
            # We use max(1, ...) to handle edge cases with extremely small datasets/large K
            sample_per_batch = max(1, total_cells // n_classes)
        # 1. Collect balanced samples for each batch first
        batch_samples = {}
        for b_id, available_indices in batch_indices_map.items():
            n_avail = len(available_indices)

            if n_avail >= sample_per_batch:
                rand_perm = torch.randperm(n_avail, device=self.device)[:sample_per_batch]
                chosen = available_indices[rand_perm]
            else:
                rand_idx = torch.randint(n_avail, (sample_per_batch,), device=self.device)
                chosen = available_indices[rand_idx]

            batch_samples[b_id] = chosen

        # 2. Determine how many mini-batches (steps) we need to split these into
        total_samples = n_classes * sample_per_batch
        num_steps = (total_samples + batch_size - 1) // batch_size

        final_indices = []
        final_labels = []

        # 3. Construct the epoch mini-batch by mini-batch
        for step in range(num_steps):
            step_indices = []
            step_labels = []

            for b_id in batch_indices_map:
                # Calculate proportional slice for this step
                start = (step * sample_per_batch) // num_steps
                end = ((step + 1) * sample_per_batch) // num_steps

                # Get the slice of indices for this batch
                idxs = batch_samples[b_id][start:end]

                step_indices.append(idxs)
                step_labels.append(
                    torch.full((len(idxs),), b_id, device=self.device, dtype=torch.int64)
                )

            # Combine all batch representatives for this step
            mb_idxs = torch.cat(step_indices)
            mb_lbls = torch.cat(step_labels)

            # Shuffle ONLY within this mini-batch
            # This preserves the stratification while randomizing processing order
            perm = torch.randperm(len(mb_idxs), device=self.device)
            final_indices.append(mb_idxs[perm])
            final_labels.append(mb_lbls[perm])

        # Concatenate the stratified mini-batches
        return torch.cat(final_indices), torch.cat(final_labels)

    def _train_batch(self, batch_data, optimizers, params, warmup, reference_batch_idx=None):
        """
        Performs the forward and backward pass for a single mini-batch.
        """
        x, x_raw, v = batch_data
        opt_g, opt_d = optimizers
        d_coef, kl_coef, disc_iter = params

        batch_size = x.size(0)
        v_true = v
        v_one_hot = torch.zeros(batch_size, self.v_dim, device=self.device)
        v_one_hot.scatter_(1, v.unsqueeze(1), 1)

        # 1. VAE forward pass -> backbone loss L_backbone = reconstruction + KL.
        #    Each backbone owns its reconstruction likelihood; no shared auxiliary terms.
        reconst_loss, kl_divergence, z, _x_tilde = self.VAE(x, x_raw, v_one_hot, warmup)
        loss_vae = torch.mean(reconst_loss.mean() + kl_coef * kl_divergence.mean())

        # 2. Adversary (critic/discriminator) updates on detached z.
        for _ in range(disc_iter):
            opt_d.zero_grad()
            loss_d_z, gp = self.D_Z(z.detach(), v_true, reference_batch=reference_batch_idx)
            loss_d_z += gp
            if not warmup:
                loss_d_z.backward(retain_graph=True)
                opt_d.step()

        # 3. Encoder/generator update: minimize L_backbone + lambda_adv * L_adv
        #    (the framework objective; lambda_adv = d_coef, zeroed during warmup).
        opt_g.zero_grad()
        loss_da, gp = self.D_Z(z, v_true, reference_batch=reference_batch_idx)
        lam = 0.0 if warmup else d_coef
        all_loss = loss_vae - lam * loss_da
        all_loss.backward()
        opt_g.step()

        non_zero_mask = x_raw > 0
        if non_zero_mask.sum() > 0:
            reconst_loss_non_zero = reconst_loss[non_zero_mask].mean()
        else:
            reconst_loss_non_zero = torch.tensor(0.0, device=self.device)

        return (
            all_loss,
            loss_da,
            loss_vae,
            reconst_loss.mean(),
            reconst_loss_non_zero,
        )

    def train_model(
        self,
        adata,
        batch_key,
        epochs,
        d_coef,
        kl_coef,
        warmup_epoch,
        disc_iter,
        batch_size=1024,
        reference_batch_name_str=None,
        reference_mode="fixed",
        early_stopping=False,
        es_patience=5,
        es_check_every=10,
        es_holdout_frac=0.15,
        es_celltype_key=None,
        lr_g=1e-3,
        lr_d=1e-3,
    ):
        """Train the model.

        # WHY early stopping: a convergence sweep at 150/300/450 epochs (independent full
        #   runs, 3 datasets x both heads) showed MIXING converges but CONSERVATION
        #   DEGRADES with longer training -- pooled ARI 0.058 -> 0.015 -> 0.018 and linear
        #   cell-type probe lift 0.168 -> 0.132 -> 0.097, worst case sim2/critic ARI
        #   0.190 -> 0.0001. A fixed epoch budget is therefore an undisclosed
        #   REGULARISATION choice, and nested selection over epochs would simply pick the
        #   shortest run. Early stopping makes the stopping point data-driven per config.
        # HOW: monitor a HELD-OUT cell-type probe (a fold of cells excluded from the probe
        #   fit, not from training) every ``es_check_every`` epochs; keep the best
        #   weights and stop after ``es_patience`` checks without improvement. The
        #   monitored quantity is conservation because that is the axis that degrades;
        #   mixing is flat in epochs so it needs no guard.
        """
        training_history = {
            "all_loss": [],
            "loss_da": [],
            "loss_vae": [],
            "reconst_loss": [],
            "reconst_loss_non_zero": [],
        }
        # WHY separate: training_history is consumed as pd.DataFrame(history), so EVERY
        #   value must be a per-epoch list of equal length. The early-stopping trace is
        #   sampled every es_check_every epochs and the best-epoch summary is scalar, so
        #   putting them in the same dict makes it RAGGED and pandas raises
        #   "All arrays must be of the same length". They live in a sidecar dict instead.
        es_trace = {"es_epoch": [], "es_score": [], "es_best_epoch": None,
                    "es_best_score": None}

        # 1. Prepare Data (One-time GPU transfer)
        data_dict, batch_indices_map, reference_batch_idx = self._prepare_tensors(
            adata, batch_key, reference_batch_name_str
        )
        # WHY (E4 fix): experiments pass the reference as an INT index at construction
        #   (self.reference_batch), not as a name string, so _prepare_tensors returns
        #   reference_batch_idx=None and the epoch-level rotating/joint logic below never
        #   fired (all modes silently collapsed to the critic's constructor reference).
        #   Fall back to the model's stored int index so a reference IS active whenever
        #   this is the critic head, enabling the fixed/rotating/joint comparison.
        if reference_batch_idx is None and self.critic and getattr(self, "reference_batch", None) is not None:
            reference_batch_idx = int(self.reference_batch)

        # WHY (E4): isolate whether critic pathologies come from the REFERENCE DESIGN
        #           rather than the Wasserstein objective. HOW: three reference modes.
        #   "fixed"    - one reference batch for all of training (original behaviour).
        #   "rotating" - the reference batch cycles across batches each epoch, so no
        #                single batch is privileged as the alignment anchor.
        #   "joint"    - no privileged reference; each epoch draws a reference uniformly
        #                at random, approximating all-pairs alignment on average.
        n_batches_total = len(batch_indices_map)
        base_ref = reference_batch_idx if reference_batch_idx is not None else 0

        # WHY (barycenter formulation): the learnable anchors approximate the Frechet mean,
        #      so they must be MINIMISED with the generator (pulling the virtual centre toward
        #      the batches) and EXCLUDED from the critic step (which maximises distance and
        #      would otherwise push the anchors away adversarially).
        if self.D_Z.anchors is not None:
            anchor_ids = {id(self.D_Z.anchors)}
            critic_params = [p for p in self.D_Z.parameters() if id(p) not in anchor_ids]
            gen_params = list(self.VAE.parameters()) + [self.D_Z.anchors]
        else:
            critic_params = list(self.D_Z.parameters())
            gen_params = list(self.VAE.parameters())
        # lr is parameterised (default unchanged at 1e-3) so batch-size / learning-rate
        # interactions can be studied without altering any existing result. betas=(0.5,
        # 0.9) is the GAN convention: a low beta1 keeps the adversary responsive to a
        # generator that is moving under it.
        optimizer_d_z = optim.Adam(critic_params, lr=lr_d, betas=(0.5, 0.9))
        optimizer_g = optim.Adam(gen_params, lr=lr_g, betas=(0.5, 0.9))
        optimizers = (optimizer_g, optimizer_d_z)

        batch_size_loader = batch_size

        # [best_score, best_epoch, best_state_dict] and a strike counter for patience
        _es_best = [-np.inf, -1, None]
        _es_strikes = [0]
        print(f"Starting training on {self.device}...")

        for epoch in range(epochs):
            self.VAE.train()
            self.D_Z.train()

            # Track epoch accumulation
            epoch_vae = 0
            epoch_critic = 0
            epoch_gen_adv = 0
            epoch_total = 0
            epoch_reconst_non_zero = 0
            batch_count = 0

            # 2. Sample Indices for this Epoch
            train_idxs, train_v = self._sample_epoch_indices(
                batch_indices_map, sample_per_batch=None, batch_size=batch_size_loader
            )
            total_samples = len(train_idxs)

            warmup = epoch < warmup_epoch
            params = (d_coef, kl_coef, disc_iter)

            # E4: resolve the reference batch index used for THIS epoch.
            if reference_batch_idx is None:
                epoch_ref_idx = None  # discriminator head: no reference concept
            elif reference_mode == "rotating":
                epoch_ref_idx = (base_ref + epoch) % n_batches_total
            elif reference_mode == "joint":
                epoch_ref_idx = int(np.random.randint(n_batches_total))
            else:  # "fixed"
                epoch_ref_idx = reference_batch_idx

            # 3. Iterate Mini-batches
            for i in range(0, total_samples, batch_size_loader):
                end = min(i + batch_size_loader, total_samples)
                mb_idxs = train_idxs[i:end]

                # Slice directly from GPU tensors
                batch_data = (
                    data_dict["X"][mb_idxs],
                    data_dict["X_raw"][mb_idxs],
                    train_v[i:end],
                )

                all_loss, loss_da, loss_vae, reconst_loss, reconst_loss_non_zero = (
                    self._train_batch(batch_data, optimizers, params, warmup, epoch_ref_idx)
                )

                # Accumulate (handle tensors vs floats)
                epoch_total += all_loss.item()
                epoch_gen_adv += loss_da.item()
                epoch_vae += loss_vae.item()
                epoch_critic += reconst_loss.item()
                epoch_reconst_non_zero += reconst_loss_non_zero.item()
                batch_count += 1

            # Average losses for the epoch
            training_history["all_loss"].append(epoch_total / batch_count)
            training_history["loss_da"].append(epoch_gen_adv / batch_count)
            training_history["loss_vae"].append(epoch_vae / batch_count)
            training_history["reconst_loss"].append(epoch_critic / batch_count)
            training_history["reconst_loss_non_zero"].append(epoch_reconst_non_zero / batch_count)

            # ---- early stopping on a held-out cell-type probe ----
            if early_stopping and es_celltype_key is not None and not warmup:
                if (epoch + 1) % es_check_every == 0:
                    score = self._es_probe_score(
                        adata, es_celltype_key, data_dict, es_holdout_frac
                    )
                    es_trace["es_epoch"].append(epoch + 1)
                    es_trace["es_score"].append(score)
                    if score > _es_best[0] + 1e-4:
                        _es_best[0] = score
                        _es_best[1] = epoch + 1
                        _es_best[2] = copy.deepcopy(self.VAE.state_dict())
                        _es_strikes[0] = 0
                    else:
                        _es_strikes[0] += 1
                        if _es_strikes[0] >= es_patience:
                            print(f"[early-stop] no improvement for {es_patience} checks; "
                                  f"stopping at epoch {epoch + 1}, best was epoch "
                                  f"{_es_best[1]} (score {_es_best[0]:.4f})", flush=True)
                            break

        # restore the best-scoring weights rather than the last ones
        if early_stopping and _es_best[2] is not None:
            self.VAE.load_state_dict(_es_best[2])
            es_trace["es_best_epoch"] = _es_best[1]
            es_trace["es_best_score"] = _es_best[0]

        # attached, not merged: keeps pd.DataFrame(training_history) rectangular
        self.es_trace = es_trace
        return training_history

    def _es_probe_score(self, adata, celltype_key, data_dict, holdout_frac):
        """Held-out linear-probe accuracy for cell type on the current latent.

        # WHY: cheap (seconds) and directly measures the conservation axis that degrades.
        #      The holdout is over CELLS WITHIN THE PROBE FIT, so no training data is
        #      withheld from the VAE -- this monitors representation quality, it is not a
        #      model-selection split for the reported metrics.
        """
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import train_test_split

        self.VAE.eval()
        try:
            with torch.no_grad():
                zs = []
                X = data_dict["X"]
                for i in range(0, len(X), 4096):
                    _qm, _qv, z = self.VAE.encoder(X[i:i + 4096], warmup=False)
                    zs.append(z.cpu().numpy())
            Z = np.vstack(zs)
            y = adata.obs[celltype_key].astype(str).to_numpy()[: len(Z)]
            keep = pd.Series(y).groupby(y).transform("size").to_numpy() >= 4
            if keep.sum() < 40 or len(set(y[keep])) < 2:
                return float("nan")
            Ztr, Zte, ytr, yte = train_test_split(
                Z[keep], y[keep], test_size=holdout_frac, random_state=0, stratify=y[keep]
            )
            clf = LogisticRegression(max_iter=500).fit(Ztr, ytr)
            return float(clf.score(Zte, yte))
        except Exception:
            return float("nan")
        finally:
            self.VAE.train()


def train_integration_model(
    adata,
    disc_iter,
    batch_key="batch",
    reference_batch=None,
    reference_batch_name_str=None,
    z_dim=256,
    epochs=500,
    d_coef=0.2,
    kl_coef=0.005,
    warmup_epoch=5,
    critic=False,
    scale=None,
    flex_epochs=False,
    batch_size=1024,
    backbone=None,
    reference_mode="fixed",
    formulation="reference",
    early_stopping=False,
    es_celltype_key=None,
    es_patience=5,
    es_check_every=10,
    lr_g=1e-3,
    lr_d=1e-3,
):
    number_of_cells = adata.n_obs
    number_of_batches = np.unique(adata.obs[batch_key]).shape[0]

    # Default number of epochs
    if flex_epochs and number_of_cells > 100000:
        calculated_epochs = int(1.5 * number_of_cells / (number_of_batches * 512))
        # If the calculated value is larger than the default, use it instead
        if calculated_epochs > epochs:
            epochs = calculated_epochs

    model = SCIntegrationModel(
        adata=adata,
        batch_key=batch_key,
        z_dim=z_dim,
        critic=critic,
        reference_batch=reference_batch,
        backbone=backbone,
        formulation=formulation,
    )
    print(epochs)
    start_time = time.time()
    training_history = model.train_model(
        adata,
        batch_key=batch_key,
        epochs=epochs,
        d_coef=d_coef,
        kl_coef=kl_coef,
        warmup_epoch=warmup_epoch,
        disc_iter=disc_iter,
        reference_batch_name_str=reference_batch_name_str,
        batch_size=batch_size,
        reference_mode=reference_mode,
        early_stopping=early_stopping,
        es_celltype_key=es_celltype_key,
        es_patience=es_patience,
        es_check_every=es_check_every,
        lr_g=lr_g,
        lr_d=lr_d,
    )
    end_time = time.time()
    training_time = end_time - start_time
    print(f"Training completed in {training_time:.2f} seconds")
    model.VAE.eval()
    # Attach the early-stopping trace to the history so callers can record WHICH epoch
    # was actually selected. Without this the result row reports the requested epoch
    # budget (500) for every run, and a wave gives no way to tell a converged run from
    # one that stopped at epoch 60 -- see es_best_epoch in the result row.
    if getattr(model, "es_trace", None) is not None:
        training_history = dict(training_history)
        training_history["_es_trace"] = model.es_trace
    return model.VAE, training_history


def obtain_embeddings(adata, vae, dim=50, pca=True, seed=None):
    if seed is not None:
        seed_everything(seed)

    vae.eval()
    data_loader = inference_dataloader(adata)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    all_z = []
    all_indices = []

    for _, (x, indices) in enumerate(data_loader):
        x = x.to(device)
        _, _, z = vae.encoder(x, warmup=False)
        all_z.append(z)
        all_indices.extend(indices.tolist())

    all_z_combined = torch.cat(all_z, dim=0)
    all_indices_tensor = torch.tensor(all_indices)
    all_z_reordered = all_z_combined[all_indices_tensor.argsort()]
    all_z_np = all_z_reordered.cpu().detach().numpy()

    # Create anndata object with reordered embeddings
    adata.obsm["X_latent"] = all_z_np

    if pca:
        pca_model = PCA(n_components=dim)
        # Fit and transform the data
        x_latent_pca = pca_model.fit_transform(adata.obsm["X_latent"])
        # Store the PCA-reduced data back into adata.obsm
        adata.obsm["X_latent"] = x_latent_pca

    return adata
