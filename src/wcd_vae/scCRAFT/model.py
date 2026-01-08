import sys
import time

import numpy as np
import scipy
from sklearn.decomposition import PCA
import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812
import torch.optim as optim

from wcd_vae.scCRAFT.networks import VAE, Discriminator
from wcd_vae.scCRAFT.utils import (
    create_triplets,
    generate_adata_to_dataloader,
    set_seed,
    weights_init_normal,
)


# Main training class
class SCIntegrationModel(nn.Module):
    def __init__(self, adata, batch_key, z_dim, critic, seed, reference_batch):
        super().__init__()
        self.p_dim = adata.shape[1]
        self.z_dim = z_dim
        self.v_dim = np.unique(adata.obs[batch_key]).shape[0]

        self.VAE = VAE(p_dim=self.p_dim, v_dim=self.v_dim, latent_dim=self.z_dim)
        self.D_Z = Discriminator(
            n_input=self.z_dim,
            domain_number=self.v_dim,
            critic=critic,
            reference_batch=reference_batch,
        )

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.VAE.to(self.device)
        self.D_Z.to(self.device)

        if seed is not None:
            set_seed(seed)
        self.VAE.apply(weights_init_normal)
        self.D_Z.apply(weights_init_normal)

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
            X_raw_tensor = torch.tensor(
                adata.layers["counts"].toarray(), dtype=torch.float32
            )  # or adata.raw
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
        label1_tensor = torch.tensor(adata.obs["leiden1"].cat.codes.values, dtype=torch.int64)
        label2_tensor = torch.tensor(adata.obs["leiden2"].cat.codes.values, dtype=torch.int64)

        # Move to device (e.g., GPU)
        X_tensor = X_tensor.to(self.device)
        batch_tensor = batch_tensor.to(self.device)
        label1_tensor = label1_tensor.to(self.device)
        label2_tensor = label2_tensor.to(self.device)
        X_raw_tensor = X_raw_tensor.to(self.device)

        data_dict = {
            "X": X_tensor,
            "X_raw": X_raw_tensor,
            "batch_labels": batch_tensor,
            "l1": label1_tensor,
            "l2": label2_tensor,
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

            for b_id in batch_indices_map.keys():
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
        x, x_raw, v, labels_low, labels_high = batch_data
        opt_g, opt_d = optimizers
        d_coef, kl_coef, triplet_coef, cos_coef, disc_iter = params

        batch_size = x.size(0)
        v_true = v
        v_one_hot = torch.zeros(batch_size, self.v_dim, device=self.device)
        v_one_hot.scatter_(1, v.unsqueeze(1), 1)

        # 1. VAE Forward Pass
        reconst_loss, kl_divergence, z, x_tilde = self.VAE(x, x_raw, v_one_hot, warmup)

        loss_cos = (1 - torch.sum(F.normalize(x_tilde, p=2) * F.normalize(x, p=2), 1)).mean()
        loss_vae = torch.mean(reconst_loss.mean() + kl_coef * kl_divergence.mean())

        # 2. Discriminator Steps
        for _ in range(disc_iter):
            opt_d.zero_grad()
            loss_d_z, gp = self.D_Z(
                z.detach(), v_true, reference_batch=reference_batch_idx
            )  # D_Z handles 'critic' flag internally
            loss_d_z += gp
            loss_d_z.backward(retain_graph=True)
            opt_d.step()

        # 3. Generator/VAE Update
        opt_g.zero_grad()
        loss_da, gp = self.D_Z(z, v_true, reference_batch=reference_batch_idx)
        triplet_loss = create_triplets(z, labels_low, labels_high, v_true, margin=5)

        if warmup:
            all_loss = (
                -0 * loss_da
                + 1 * loss_vae
                + gp
                + triplet_coef * triplet_loss
                + cos_coef * loss_cos
            )
        else:
            all_loss = (
                -d_coef * loss_da
                + 1 * loss_vae
                + gp
                + triplet_coef * triplet_loss
                + cos_coef * loss_cos
            )

        all_loss.backward()
        opt_g.step()

        return all_loss, loss_da, triplet_loss, loss_vae

    def train_model(
        self,
        adata,
        batch_key,
        epochs,
        d_coef,
        kl_coef,
        triplet_coef,
        cos_coef,
        warmup_epoch,
        disc_iter,
        reference_batch_name_str=None,
    ):
        # 1. Prepare Data (One-time GPU transfer)
        data_dict, batch_indices_map, reference_batch_idx = self._prepare_tensors(
            adata, batch_key, reference_batch_name_str
        )

        optimizer_d_z = optim.Adam(self.D_Z.parameters(), lr=0.001, betas=(0.5, 0.9))
        optimizer_g = optim.Adam(self.VAE.parameters(), lr=0.001, betas=(0.5, 0.9))
        optimizers = (optimizer_g, optimizer_d_z)

        batch_size_loader = 1024

        print(f"Starting training on {self.device}...")

        for epoch in range(epochs):
            self.VAE.train()
            self.D_Z.train()

            # 2. Sample Indices for this Epoch
            train_idxs, train_v = self._sample_epoch_indices(
                batch_indices_map, sample_per_batch=None, batch_size=batch_size_loader
            )
            total_samples = len(train_idxs)

            warmup = epoch < warmup_epoch
            params = (d_coef, kl_coef, triplet_coef, cos_coef, disc_iter)

            # 3. Iterate Mini-batches
            for i in range(0, total_samples, batch_size_loader):
                end = min(i + batch_size_loader, total_samples)
                mb_idxs = train_idxs[i:end]

                # Slice directly from GPU tensors
                batch_data = (
                    data_dict["X"][mb_idxs],
                    train_v[i:end],
                    data_dict["l1"][mb_idxs],
                    data_dict["l2"][mb_idxs],
                )

                self._train_batch(batch_data, optimizers, params, warmup, reference_batch_idx)


def train_integration_model(
    adata,
    disc_iter,
    batch_key="batch",
    reference_batch=None,
    reference_batch_name_str=None,
    z_dim=256,
    epochs=150,
    d_coef=0.2,
    kl_coef=0.005,
    triplet_coef=1,
    cos_coef=20,
    warmup_epoch=50,
    critic=False,
    scale=None,
    flex_epochs=False,
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
    )
    print(epochs)
    start_time = time.time()
    model.train_model(
        adata,
        batch_key=batch_key,
        epochs=epochs,
        d_coef=d_coef,
        kl_coef=kl_coef,
        triplet_coef=triplet_coef,
        cos_coef=cos_coef,
        warmup_epoch=warmup_epoch,
        disc_iter=disc_iter,
        reference_batch_name_str=reference_batch_name_str,
    )
    end_time = time.time()
    training_time = end_time - start_time
    print(f"Training completed in {training_time:.2f} seconds")

    model.VAE.eval()
    return model.VAE


def obtain_embeddings(adata, vae, dim=50, pca=True, seed=None):
    if seed is not None:
        set_seed(seed)

    vae.eval()
    data_loader = generate_adata_to_dataloader(adata)
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
    adata.obsm["X_scCRAFT"] = all_z_np

    if pca:
        pca_model = PCA(n_components=dim)
        # Fit and transform the data
        x_sccraft_pca = pca_model.fit_transform(adata.obsm["X_scCRAFT"])
        # Store the PCA-reduced data back into adata.obsm
        adata.obsm["X_scCRAFT"] = x_sccraft_pca

    return adata
