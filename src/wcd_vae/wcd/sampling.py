"""Per-batch stratified minibatch sampler for the X8 critic arms -- AUTHORED (K. Reid), 2026-10-02.

Specification: docs/SPECS_missing_arms.md section 3 (signed off 2026-10-02, CONSTRAINTS.md SI-24). Precedent:
domain-stratified minibatches in domain-adversarial training (Ganin & Lempitsky 2015, ICML, Sec. 5: half of
each 128-cell batch from each domain). Replaces scvi-tools' BatchSampler(RandomSampler(train set), 128,
drop_last=False) for the training loader only.

  * every minibatch has exactly batch_size cells: m = batch_size // V from every batch, and the remainder
    batch_size - V m goes one extra cell each to that many batches drawn at random per step (exact equality
    when V divides batch_size; V = 2, 4, 8, 16 in X8);
  * within a batch, cells are drawn without replacement from a permutation of its cells; a used-up
    permutation is replaced by a new one, with the cells already in the current minibatch moved to its end
    so no cell appears twice in one minibatch; the cycles continue across epochs, so every cell of a batch
    is drawn once per cycle;
  * one epoch = ceil(n / batch_size) minibatches, the step count of the default sampler, so epochs and
    scvi-tools' epoch-based KL warm-up are unchanged;
  * a batch with fewer cells than its per-step quota (ceil(batch_size / V)) raises ValueError;
  * randomness: one seed per epoch drawn from torch's global RNG (as torch's RandomSampler does), so a fit
    is reproducible from scvi.settings.seed (the first epoch's generator draws the first permutations).
Yields lists of dataset POSITIONS (0..n-1), the convention of the sampler scvi-tools' AnnDataLoader passes to
torch's DataLoader with batch_size=None (one __getitem__ per minibatch).
"""
import math

import numpy as np
import torch
from torch.utils.data import Sampler


class StratifiedBatchSampler(Sampler):
    def __init__(self, groups, batch_size):
        groups = np.asarray(groups).ravel()
        if groups.size == 0:
            raise ValueError("stratified sampler: empty training set")
        self.batch_size = int(batch_size)
        self.codes, inv = np.unique(groups, return_inverse=True)
        self.members = [np.flatnonzero(inv == v) for v in range(len(self.codes))]
        self.n_groups = len(self.codes)
        if self.n_groups > self.batch_size:
            raise ValueError(f"stratified sampler: {self.n_groups} batches > batch size {self.batch_size}")
        self.m_base = self.batch_size // self.n_groups
        self.remainder = self.batch_size - self.n_groups * self.m_base
        self.quota = self.m_base + (1 if self.remainder else 0)
        small = [(self.codes[v], len(m)) for v, m in enumerate(self.members) if len(m) < self.quota]
        if small:
            raise ValueError(f"stratified sampler: batches with fewer training cells than the per-step quota "
                             f"{self.quota}: {small}")
        self.n = int(groups.size)
        self.steps = int(math.ceil(self.n / self.batch_size))
        self._perms, self._pos = None, None      # per-batch cycle state, carried across epochs

    def __len__(self):
        return self.steps

    def __iter__(self):
        seed = int(torch.empty((), dtype=torch.int64).random_().item())
        rng = np.random.default_rng(seed & 0xFFFFFFFFFFFFFFFF)
        if self._perms is None:
            self._perms = [rng.permutation(m) for m in self.members]
            self._pos = [0] * self.n_groups
        perms, pos = self._perms, self._pos
        for _ in range(self.steps):
            extra = (set(rng.choice(self.n_groups, self.remainder, replace=False).tolist())
                     if self.remainder else set())
            out = []
            for v in range(self.n_groups):
                need = self.m_base + (1 if v in extra else 0)
                take = perms[v][pos[v]:pos[v] + need]
                pos[v] += len(take)
                if len(take) < need:
                    fresh = rng.permutation(self.members[v])
                    seen = np.isin(fresh, take)
                    fresh = np.concatenate([fresh[~seen], fresh[seen]])
                    k = need - len(take)
                    take = np.concatenate([take, fresh[:k]])
                    perms[v], pos[v] = fresh, k
                out.append(take)
            yield np.concatenate(out).tolist()
