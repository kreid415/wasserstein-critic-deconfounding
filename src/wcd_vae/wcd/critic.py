"""Per-batch Wasserstein critic: dual losses and gradient penalty -- AUTHORED (K. Reid).

The critic has one output head per batch. Head k estimates W1(P_k, T_k) through the
Kantorovich-Rubinstein dual, E_{P_k}[C_k] - E_{T_k}[C_k], with the 1-Lipschitz constraint
enforced by a per-head WGAN-GP penalty (Gulrajani et al. 2017). The target T_k depends on the
formulation:

  reference  : cells of the designated reference batch r (head r is skipped).
  pooled     : cells of all OTHER batches, P_{-k}.
               (Changed 2026-10-02, code review N4: the pool used to include batch k itself, so
               each head estimated (1 - pi_k) * W1(P_k, P_{-k}); with two batches the objective was
               exactly half the reference objective.)
  barycenter : target samples supplied by the caller -- the support of a free-support
               Wasserstein barycenter of the batch distributions (see wcd.barycenter). The
               target is detached: no optimiser moves it.

All heads are evaluated in ONE vectorised pass (one critic forward and one autograd call for
the gradient penalty, instead of one per head). It is the same mathematics as the per-head loop
it replaces -- per-head means, then the mean over active heads. tests/wcd_vae/test_critic_vectorised.py
checks loss and penalty values, and the critic-parameter gradients, against a per-head loop for
all three formulations, and checks the reference loss against the pre-2026-10-02 code. The aim
is to remove the cost term that grew linearly with the number of batches (2.4x the
discriminator's cost on 3 batches, 17.3x on 9, measured 2026-08).
"""
import torch
from torch.autograd import grad
import torch.nn as nn

FORMULATIONS = ("reference", "pooled", "barycenter")


def _check_reference(reference_batch, num_domains):
    if reference_batch is None or not (0 <= int(reference_batch) < num_domains):
        raise ValueError(
            f"Invalid reference index: {reference_batch}. The reference formulation needs the "
            "integer index of the reference batch."
        )
    return int(reference_batch)


class ReferenceWassersteinLoss(nn.Module):
    """Negative mean over active heads of E_{P_k}[C_k] - E_{T_k}[C_k] (the critic MINIMISES this).

    A head is active when its batch is present in the minibatch, it is not the reference head,
    and its target is non-empty.
    """

    def __init__(self, reference_class: int = -1, reduction: str = "mean", formulation: str = "reference"):
        super().__init__()
        if formulation not in FORMULATIONS:
            raise ValueError(f"Unknown formulation: {formulation}")
        self.reference_class = reference_class
        self.reduction = reduction
        self.formulation = formulation

    def forward(self, output, batch_ids, reference_batch=None, target_output=None):
        n, num_domains = output.shape
        counts = torch.bincount(batch_ids, minlength=num_domains)
        own = output.gather(1, batch_ids[:, None]).squeeze(1)                  # C_{b_i}(z_i)
        src_sum = output.new_zeros(num_domains).index_add_(0, batch_ids, own)  # sum over batch k of C_k
        src_mean = src_sum / counts.clamp_min(1).to(output.dtype)
        active = counts > 0

        if self.formulation == "reference":
            r = _check_reference(reference_batch if reference_batch is not None else self.reference_class,
                                 num_domains)
            mask_ref = batch_ids == r
            if not bool(mask_ref.any()):
                return output.new_zeros((), requires_grad=True)
            tgt_mean = output[mask_ref].mean(0)                                 # [V]
            active = active.clone()
            active[r] = False
        elif self.formulation == "pooled":
            n_other = (n - counts).to(output.dtype)                              # |P_{-k}| in the minibatch
            tgt_mean = (output.sum(0) - src_sum) / n_other.clamp_min(1.0)
            active = active & (n_other > 0)
        else:  # barycenter
            if target_output is None:
                raise ValueError("formulation='barycenter' requires target_output (critic scores of the barycenter support).")
            tgt_mean = target_output.mean(0)

        if not bool(active.any()):
            return output.new_zeros((), requires_grad=True)
        diff = (src_mean - tgt_mean)[active]
        if self.reduction == "sum":
            return -diff.sum()
        return -diff.mean()


def _draw_partners(batch_ids, active_rows, formulation, ref_rows=None, n_target=None):
    """Indices of a target partner for every active row (uniform within that row's target)."""
    device = batch_ids.device
    m = active_rows.numel()
    if formulation == "reference":
        return ref_rows[torch.randint(0, ref_rows.numel(), (m,), device=device)]
    if formulation == "barycenter":
        return torch.randint(0, n_target, (m,), device=device)
    # pooled: uniform over rows of OTHER batches. Sort rows by batch; for a row of batch k draw a
    # position u in [0, N - n_k) and skip batch k's contiguous block.
    n = batch_ids.numel()
    num_domains = int(batch_ids.max().item()) + 1
    order = torch.argsort(batch_ids, stable=True)
    counts = torch.bincount(batch_ids, minlength=num_domains)
    starts = torch.cumsum(counts, 0) - counts
    k = batch_ids[active_rows]
    n_other = (n - counts[k]).to(torch.float32)
    u = torch.floor(torch.rand(m, device=device) * n_other).long()
    j = u + counts[k] * (u >= starts[k]).long()
    return order[j]


def multi_class_gradient_penalty(
    critic,
    z,
    batch_ids,
    lambda_gp=10.0,
    reference_batch=None,
    formulation="reference",
    target_samples=None,
    num_domains=None,
    draws=None,
):
    """lambda_gp * mean over active heads of mean_{x in head k} (||grad C_k(x_hat)|| - 1)^2.

    x_hat = eps * z_i + (1 - eps) * t_i, where z_i is a cell of batch k and t_i is a random
    partner from head k's target (reference cells / other batches / barycenter support).
    ``draws`` = (partner_idx, eps) overrides the random draws (used by the equivalence test).
    """
    device = z.device
    if formulation not in FORMULATIONS:
        raise ValueError(f"Unknown formulation: {formulation}")
    if num_domains is None:
        num_domains = int(getattr(critic, "domain_number"))
    counts = torch.bincount(batch_ids, minlength=num_domains)
    n = z.shape[0]

    ref_rows, pool = None, z
    if formulation == "reference":
        r = _check_reference(reference_batch, num_domains)
        ref_rows = (batch_ids == r).nonzero(as_tuple=True)[0]
        if ref_rows.numel() == 0:
            return z.new_zeros(())
        row_ok = batch_ids != r
    elif formulation == "pooled":
        row_ok = (n - counts[batch_ids]) > 0
    else:
        if target_samples is None:
            raise ValueError("formulation='barycenter' requires target_samples.")
        pool = target_samples
        row_ok = torch.ones_like(batch_ids, dtype=torch.bool)

    rows = row_ok.nonzero(as_tuple=True)[0]
    if rows.numel() == 0:
        return z.new_zeros(())
    if draws is None:
        partner = _draw_partners(batch_ids, rows, formulation, ref_rows=ref_rows,
                                 n_target=(pool.shape[0] if formulation == "barycenter" else None))
        eps = torch.rand(rows.numel(), 1, device=device)
    else:
        partner, eps = draws
    z_rows = z[rows]
    z_hat = eps * z_rows + (1.0 - eps) * pool[partner]
    z_hat.requires_grad_(True)
    heads = batch_ids[rows]
    out = critic(z_hat, batch_ids=None)
    sel = out.gather(1, heads[:, None]).sum()
    g = grad(outputs=sel, inputs=z_hat, create_graph=True, retain_graph=True, only_inputs=True)[0]
    pen = (g.view(g.shape[0], -1).norm(2, dim=1) - 1.0) ** 2
    head_sum = pen.new_zeros(num_domains).index_add_(0, heads, pen)
    head_cnt = torch.bincount(heads, minlength=num_domains)
    act = head_cnt > 0
    return lambda_gp * (head_sum[act] / head_cnt[act].to(pen.dtype)).mean()
