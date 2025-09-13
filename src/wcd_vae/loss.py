#!/usr/bin/env
"""
# adopted from : https://github.com/caokai1073/uniPort/blob/43296b12f0e3927315ed5769c5a78f9f73c5c7f1/uniport/model/loss.py#L74
"""

import torch
from torch.distributions import Normal, kl_divergence


def kl_div(mu, var, weight=None):
    loss = kl_divergence(
        Normal(mu, var.sqrt()), Normal(torch.zeros_like(mu), torch.ones_like(var))
    ).sum(dim=1)

    return loss.mean()


def distance_matrix(pts_src: torch.Tensor, pts_dst: torch.Tensor, p: int = 2):
    """
    Returns the matrix of ||x_i-y_j||_p^p.

    Parameters
    ----------
    pts_src
        [R, D] matrix
    pts_dst
        C, D] matrix
    p
        p-norm

    Return
    ------
    [R, C] matrix
        distance matrix
    """
    x_col = pts_src.unsqueeze(1)
    y_row = pts_dst.unsqueeze(0)
    distance = torch.sum((torch.abs(x_col - y_row)) ** p, 2)
    return distance


def distance_gmm(
    mu_src: torch.Tensor, mu_dst: torch.Tensor, var_src: torch.Tensor, var_dst: torch.Tensor
):
    """
    Calculate a Wasserstein distance matrix between the gmm distributions with diagonal variances

    Parameters
    ----------
    mu_src
        [R, D] matrix, the means of R Gaussian distributions
    mu_dst
        [C, D] matrix, the means of C Gaussian distributions
    logvar_src
        [R, D] matrix, the log(variance) of R Gaussian distributions
    logvar_dst
        [C, D] matrix, the log(variance) of C Gaussian distributions

    Return
    ------
    [R, C] matrix
        distance matrix
    """
    std_src = var_src.sqrt()
    std_dst = var_dst.sqrt()
    distance_mean = distance_matrix(mu_src, mu_dst, p=2)
    distance_var = distance_matrix(std_src, std_dst, p=2)

    return distance_mean + distance_var + 1e-6


def unbalanced_ot(
    mu1,
    var1,
    mu2,
    var2,
    reg=0.1,
    reg_m=1.0,
    couple=None,
    device="cpu",
    idx_q=None,
    idx_r=None,
    query_weight=None,
    ref_weight=None,
):
    """
    Calculate a unbalanced optimal transport matrix between mini batches.

    Parameters
    ----------
    mu1
        mean vector of batch 1 from the encoder
    var1
        standard deviation vector of batch 1 from the encoder
    mu2
        mean vector of batch 2 from the encoder
    var2
        standard deviation vector of batch 2 from the encoder
    reg:
        Entropy regularization parameter in OT. Default: 0.1
    reg_m:
        Unbalanced OT parameter. Larger values means more balanced OT. Default: 1.0
    Couple
        prior information about weights between cell correspondence. Default: None
    device
        training device
    idx_q
        domain_id of query batch
    idx_r
        domain_id of reference batch
    query_weight
        reweighted vectors of query batch
    ref_weight
        reweighted vectors of reference batch

    Returns
    -------
    float
        minibatch unbalanced optimal transport loss
    matrix
        minibatch unbalanced optimal transport matrix
    """

    ns = mu1.size(0)
    nt = mu2.size(0)

    cost_pp = distance_gmm(mu1, mu2, var1, var2)

    if query_weight is None:
        p_s = torch.ones(ns, 1) / ns
    else:
        query_batch_weight = query_weight[idx_q]
        p_s = query_batch_weight / torch.sum(query_batch_weight)

    if ref_weight is None:
        p_t = torch.ones(nt, 1) / nt
    else:
        ref_batch_weight = ref_weight[idx_r]
        p_t = ref_batch_weight / torch.sum(ref_batch_weight)

    p_s = p_s.to(device)
    p_t = p_t.to(device)

    tran = torch.ones(ns, nt) / (ns * nt)
    tran = tran.to(device)

    dual = (torch.ones(ns, 1) / ns).to(device)
    f = reg_m / (reg_m + reg)

    for _m in range(10):
        cost = cost_pp * couple if couple is not None else cost_pp

        kernel = torch.exp(-cost / (reg * torch.max(torch.abs(cost)))) * tran
        b = p_t / (torch.t(kernel) @ dual)
        # dual = p_s / (kernel @ b)
        for _i in range(10):
            dual = (p_s / (kernel @ b)) ** f
            b = (p_t / (torch.t(kernel) @ dual)) ** f
        tran = (dual @ torch.t(b)) * kernel
    if torch.isnan(tran).sum() > 0:
        tran = (torch.ones(ns, nt) / (ns * nt)).to(device)

    d_fgw = (cost_pp * tran.detach().data).sum()

    return d_fgw, tran.detach()


# Wasserstein loss for the critic
def wasserstein_loss(y_pred, y_true):
    return torch.mean(y_true * y_pred)


def mmd_gaussian(mu1, var1, mu2, var2, bandwidth=1.0, device="cpu"):
    """
    Compute MMD between two sets of Gaussians with diagonal covariance
    under an RBF kernel with given bandwidth.

    Args:
        mu1, logvar1: tensors (batch_size, latent_dim)
        mu2, logvar2: tensors (batch_size, latent_dim)
        bandwidth: float

    Returns:
        scalar MMD loss
    """

    def kernel_expectation(mu_a, var_a, mu_b, var_b):
        # Computes E_{x ~ N(mu_a, var_a), y ~ N(mu_b, var_b)}[k(x,y)]
        var_sum = var_a + var_b + bandwidth  # Broadcasting sum + bandwidth
        diff = mu_a - mu_b
        exp_term = torch.exp(-0.5 * (diff**2) / var_sum)
        norm_term = torch.prod(bandwidth / var_sum, dim=1).sqrt()
        return norm_term * exp_term.prod(dim=1)

    # Pairwise expectations within each batch
    batch_size1 = mu1.size(0)
    batch_size2 = mu2.size(0)

    # Compute all pairwise expectations for batch1
    xx = 0.0
    for i in range(batch_size1):
        for j in range(batch_size1):
            xx += kernel_expectation(mu1[i], var1[i], mu1[j], var1[j])
    xx /= batch_size1 * batch_size1

    # Compute all pairwise expectations for batch2
    yy = 0.0
    for i in range(batch_size2):
        for j in range(batch_size2):
            yy += kernel_expectation(mu2[i], var2[i], mu2[j], var2[j])
    yy /= batch_size2 * batch_size2

    # Cross expectation
    xy = 0.0
    for i in range(batch_size1):
        for j in range(batch_size2):
            xy += kernel_expectation(mu1[i], var1[i], mu2[j], var2[j])
    xy /= batch_size1 * batch_size2

    mmd = xx + yy - 2 * xy
    return mmd
