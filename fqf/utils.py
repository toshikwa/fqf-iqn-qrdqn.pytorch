from collections import deque
import numpy as np
import torch


def update_params(optim, loss, retain_graph=False):
    optim.zero_grad()
    loss.backward(retain_graph=retain_graph)
    optim.step()


def calculate_huber_loss(delta, kappa=1.0):
    return torch.where(
        delta.abs() < kappa,
        0.5 * delta.pow(2),
        kappa * (delta.abs() - 0.5 * kappa))


class RunningMeanStats:

    def __init__(self, n=10):
        self.n = n
        self.stats = deque(maxlen=n)

    def append(self, x):
        self.stats.append(x)

    def get(self):
        return np.mean(self.stats)
