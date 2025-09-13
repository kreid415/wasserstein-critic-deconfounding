from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import torch
import torch.utils.data as utils


def get_dataloader_from_adata(
    adata_concat,
    by: str,
    cell_label: str,
    test_size=0.2,
    batch_size: int = 128,
    num_workers: int = 8,
):
    # transform expression data into tensor
    data = adata_concat.X.toarray() if hasattr(adata_concat.X, "toarray") else adata_concat.X
    data_tensor = torch.tensor(data, dtype=torch.float32)

    # make domain labels
    domain_encoder = OneHotEncoder(sparse_output=False)
    domain_labels = domain_encoder.fit_transform(adata_concat.obs[by].to_numpy().reshape(-1, 1))
    domain_labels_tensor = torch.tensor(domain_labels, dtype=torch.float32)

    cell_encoder = OneHotEncoder(sparse_output=False)

    obs = (
        adata_concat.obs[cell_label].to_numpy()
        if hasattr(adata_concat.obs[cell_label], "to_numpy")
        else adata_concat.obs[cell_label]
    )
    cell_labels = cell_encoder.fit_transform(obs.reshape(-1, 1))
    cell_labels_tensor = torch.tensor(cell_labels, dtype=torch.float32)

    dataset = utils.TensorDataset(
        data_tensor.float(), domain_labels_tensor.float(), cell_labels_tensor.float()
    )

    train_set, test_set = train_test_split(dataset, test_size=test_size, random_state=42)
    train_loader, test_loader = (
        utils.DataLoader(
            dataset=train_set,
            num_workers=num_workers,
            batch_size=batch_size,
            shuffle=True,
        ),
        utils.DataLoader(test_set, num_workers=num_workers, batch_size=batch_size, shuffle=False),
    )

    return train_loader, test_loader, domain_encoder, cell_encoder
