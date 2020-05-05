import numpy as np
import torch

from .base import LazyMultiStepMemory
from .segment_tree import SumTree, MinTree


class LazyPrioritizedMultiStepMemory(LazyMultiStepMemory):

    def __init__(self, capacity, state_shape, device, gamma=0.99,
                 multi_step=3, alpha=0.5, beta=0.4, beta_diff=0.001):
        super().__init__(capacity, state_shape, device, gamma, multi_step)

        self.alpha = alpha
        self.beta = beta
        self.beta_diff = beta_diff
        self.epsilon = 1e-4

        it_capacity = 1
        while it_capacity < capacity:
            it_capacity *= 2
        self.it_sum = SumTree(it_capacity)
        self.it_min = MinTree(it_capacity)

        self._cached = None
        self.max_p = 1.0

    def append(self, state, action, reward, next_state, done, error=None):
        # Calculate priority.
        if error is None:
            p = self.max_p
        else:
            print(error)
            p = error ** self.alpha

        # Store priority, which is done efficiently by SegmentTree.
        self.it_min[self._p] = p
        self.it_sum[self._p] = p

        super().append(state, action, reward, next_state, done)

    def _sample_idxes(self, batch_size):
        indices = np.empty((batch_size), dtype=np.int64)
        total_p = self.it_sum.sum(0, self._n)

        for i in range(batch_size):
            mass = np.random.rand() * total_p
            indices[i] = self.it_sum.find_prefixsum_idx(mass)

        # Anneal beta.
        self.beta = min(1. - self.epsilon, self.beta + self.beta_diff)
        return indices

    def sample(self, batch_size):
        assert self._cached is None, 'Update priorities before sampling.'

        self._cached = self._sample_idxes(batch_size)
        batch = self._sample(self._cached, batch_size)
        weights = self._calc_weights(self._cached)
        return batch, weights

    def _calc_weights(self, indices):
        weights = torch.empty((len(indices), 1), dtype=torch.float)
        p_min = self.it_min.min() / self.it_sum.sum()
        max_weight = (p_min * self._n) ** (-self.beta)

        for i, index in enumerate(indices):
            p_sample = self.it_sum[index] / self.it_sum.sum()
            weights[i] = (p_sample * self._n) ** (-self.beta) / max_weight

        return weights.to(self.device)

    def update_priority(self, errors):
        assert self._cached is not None

        errors = errors.detach().cpu().numpy().flatten()
        ps = errors ** self.alpha

        for index, p in zip(self._cached, ps):
            assert 0 <= index < self._n
            self.it_sum[index] = p
            self.it_min[index] = p

        self.max_p = max(self.max_p, np.max(ps))
        self._cached = None
