import numpy as np
import torch

from .base import LazyMultiStepMemory
from .segment_tree import SumTree, MinTree


class LazyPrioritizedMultiStepMemory(LazyMultiStepMemory):

    def __init__(self, capacity, state_shape, device, gamma=0.99,
                 multi_step=3, alpha=0.5, beta=0.4, beta_steps=2e5):
        super().__init__(capacity, state_shape, device, gamma, multi_step)

        self.alpha = alpha
        self.beta = beta
        self.beta_diff = (1.0 - beta) / beta_steps
        self.epsilon = 0.01

        it_capacity = 1
        while it_capacity < capacity:
            it_capacity *= 2
        self.it_sum = SumTree(it_capacity)
        self.it_min = MinTree(it_capacity)

        self._cached = None
        self.max_pa = 1.0

    def append(self, state, action, reward, next_state, done, p=None):
        # Calculate priority.
        if p is None:
            pa = self.max_pa
        else:
            pa = p ** self.alpha

        if self.multi_step != 1:
            self.buff.append(state, action, reward)

            if self.buff.is_full():
                state, action, reward = self.buff.get(self.gamma)
                self._append(state, action, reward, next_state, done, pa)

            if done:
                while not self.buff.is_empty():
                    state, action, reward = self.buff.get(self.gamma)
                    self._append(state, action, reward, next_state, done, pa)
        else:
            self._append(state, action, reward, next_state, done, pa)

    def _append(self, state, action, reward, next_state, done, pa):
        # Store priority, which is done efficiently by SegmentTree.
        self.it_min[self._p] = pa
        self.it_sum[self._p] = pa
        super()._append(state, action, reward, next_state, done)

    def _sample_idxes(self, batch_size):
        indices = np.empty((batch_size, ), dtype=np.int64)
        total_pa = self.it_sum.sum(0, self._n)

        for i in range(batch_size):
            mass = np.random.rand() * total_pa * (i + 1) / batch_size
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
        pi_min = self.it_min.min() / self.it_sum.sum()
        max_weight = (pi_min * self._n) ** (-self.beta)

        for i, index in enumerate(indices):
            pi = self.it_sum[index] / self.it_sum.sum()
            weights[i] = (pi * self._n) ** (-self.beta) / max_weight

        return weights.to(self.device)

    def update_priority(self, errors):
        assert self._cached is not None

        errors = errors.detach().cpu().numpy().flatten()
        pas = errors ** self.alpha

        for index, pa in zip(self._cached, pas):
            assert 0 <= index < self._n
            self.it_sum[index] = pa
            self.it_min[index] = pa
            self.max_pa = max(self.max_pa, pa)

        self._cached = None
