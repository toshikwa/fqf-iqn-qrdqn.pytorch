import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


def weights_init_xavier(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0)


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class DQNBase(nn.Module):

    def __init__(self, num_channels, embedding_dim=7*7*64):
        super(DQNBase, self).__init__()

        self.embedding_dim = embedding_dim

        self.net = nn.Sequential(
            nn.Conv2d(num_channels, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            Flatten(),
        ).apply(weights_init_xavier)

    def forward(self, states):
        num_batches = states.shape[0]
        state_embedding = self.net(states)

        assert state_embedding.shape == (num_batches, self.embedding_dim)
        return state_embedding


class FractionProposalNetwork(nn.Module):

    def __init__(self, num_taus=32, embedding_dim=7*7*64):
        super(FractionProposalNetwork, self).__init__()

        self.num_taus = 32
        self.embedding_dim = embedding_dim

        self.net = nn.Sequential(
            nn.Linear(embedding_dim, num_taus)
        )

    def forward(self, state_embeddings):
        num_batches = state_embeddings.shape[0]

        # (num_batches, 1)
        tau_0 = torch.zeros(
            (num_batches, 1), dtype=state_embeddings.dtype,
            device=state_embeddings.device)

        # (num_batches, num_taus)
        taus_1_N = torch.cumsum(
            F.log_softmax(self.net(state_embeddings), dim=1).exp(), dim=1)

        # (num_batches, num_taus+1)
        taus = torch.cat((tau_0, taus_1_N), dim=1)

        # (num_batches, num_taus)
        hat_taus = (taus[:, :-1] + taus[:, 1:]) / 2.

        assert taus.shape == (num_batches, self.num_taus+1)
        assert hat_taus.shape == (num_batches, self.num_taus)
        return taus, hat_taus


class QuantileValueNetwork(nn.Module):

    def __init__(self, num_actions, num_cosines=64, embedding_dim=7*7*64):
        super(QuantileValueNetwork, self).__init__()

        self.num_actions = num_actions
        self.num_cosines = num_cosines
        self.embedding_dim = embedding_dim

        self.embeding_net = nn.Sequential(
            nn.Linear(num_cosines, embedding_dim),
            nn.ReLU()
        )
        self.output_net = nn.Sequential(
            nn.Linear(embedding_dim, num_actions),
        )

    def forward(self, state_embeddings, taus):
        assert state_embeddings.shape[1] == self.embedding_dim
        assert state_embeddings.shape[0] == taus.shape[0]

        num_batches = taus.shape[0]
        num_taus = taus.shape[1]

        # (num_batches, 1, embedding_dim)
        state_embeddings = state_embeddings.view(
            num_batches, 1, self.embedding_dim)

        # (num_batches, num_taus, embedding_dim)
        cos_embeddings = self.embed_taus(taus)

        # (num_batches * num_taus, embedding_dim)
        embeddings = (state_embeddings * cos_embeddings).view(
            num_batches * num_taus, self.embedding_dim)

        # (num_batches, num_taus, num_actions)
        quantile_values = self.output_net(embeddings).view(
            num_batches, num_taus, self.num_actions)

        return quantile_values

    def embed_taus(self, taus):
        num_batches = taus.shape[0]
        num_taus = taus.shape[1]

        # (1, 1, num_cosines)
        i_pi = np.pi * torch.arange(
            start=1, end=self.num_cosines+1, dtype=taus.dtype,
            device=taus.device).view(1, 1, self.num_cosines)

        # (num_batches * num_taus, num_cosines)
        cosines = torch.cos(
            taus.view(num_batches, num_taus, 1) * i_pi
            ).view(num_batches * num_taus, self.num_cosines)

        # (num_batches, num_taus, embedding_dim)
        cos_embeddings = self.embeding_net(cosines).view(
            num_batches, num_taus, self.embedding_dim)

        return cos_embeddings
