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

        self.net = nn.Sequential(
            nn.Conv2d(num_channels, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            Flatten(),
        ).apply(weights_init_xavier)

        self.embedding_dim = embedding_dim

    def forward(self, states):
        batch_size = states.shape[0]

        # Calculate embeddings of states.
        state_embedding = self.net(states)
        assert state_embedding.shape == (batch_size, self.embedding_dim)

        return state_embedding


class FractionProposalNetwork(nn.Module):

    def __init__(self, num_taus=32, embedding_dim=7*7*64):
        super(FractionProposalNetwork, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(embedding_dim, num_taus)
        )

        self.num_taus = num_taus
        self.embedding_dim = embedding_dim

    def forward(self, state_embeddings):

        batch_size = state_embeddings.shape[0]

        # Calculate probabilities q_i in the paper.
        log_probs = F.log_softmax(self.net(state_embeddings), dim=1)
        probs = log_probs.exp()
        assert probs.shape == (batch_size, self.num_taus)

        tau_0 = torch.zeros(
            (batch_size, 1), dtype=state_embeddings.dtype,
            device=state_embeddings.device)
        taus_1_N = torch.cumsum(probs, dim=1)

        # Calculate \tau_i (i=0,...,N).
        taus = torch.cat((tau_0, taus_1_N), dim=1)
        assert taus.shape == (batch_size, self.num_taus+1)

        # Calculate \hat \tau_i (i=0,...,N-1).
        hat_taus = (taus[:, :-1] + taus[:, 1:]) / 2.
        assert hat_taus.shape == (batch_size, self.num_taus)

        # Calculate entropies of the distributions.
        entropies = -(log_probs * probs).sum(dim=-1, keepdim=True)
        assert entropies.shape == (batch_size, 1)

        return taus, hat_taus, entropies


class QuantileValueNetwork(nn.Module):

    def __init__(self, num_actions, num_cosines=64, embedding_dim=7*7*64):
        super(QuantileValueNetwork, self).__init__()

        self.embeding_net = nn.Sequential(
            nn.Linear(num_cosines, embedding_dim),
            nn.ReLU()
        )
        self.output_net = nn.Sequential(
            nn.Linear(embedding_dim, 512),
            nn.ReLU(),
            nn.Linear(512, num_actions),
        )

        self.num_actions = num_actions
        self.num_cosines = num_cosines
        self.embedding_dim = embedding_dim

    def forward(self, state_embeddings, taus):
        assert state_embeddings.shape[1] == self.embedding_dim
        assert state_embeddings.shape[0] == taus.shape[0]

        # NOTE: Because variable taus correspond to either \tau or \hat \tau
        # in the paper, num_taus isn't neccesarily the same as fqf.num_taus.
        batch_size = taus.shape[0]
        num_taus = taus.shape[1]

        # Reshape into (batch_size, 1, embedding_dim).
        state_embeddings = state_embeddings.view(
            batch_size, 1, self.embedding_dim)

        # Calculate embeddings of taus.
        tau_embeddings = self.calculate_embedding_of_taus(taus)
        assert tau_embeddings.shape == (
            batch_size, num_taus, self.embedding_dim)

        # Calculate embeddings of states and taus.
        embeddings = (state_embeddings * tau_embeddings).view(
            batch_size * num_taus, self.embedding_dim)

        # Calculate quantile values.
        quantile_values = self.output_net(embeddings).view(
            batch_size, num_taus, self.num_actions)

        return quantile_values

    def calculate_embedding_of_taus(self, taus):
        batch_size = taus.shape[0]
        num_taus = taus.shape[1]

        # Calculate i * \pi (i=1,...,N).
        i_pi = np.pi * torch.arange(
            start=1, end=self.num_cosines+1, dtype=taus.dtype,
            device=taus.device).view(1, 1, self.num_cosines)

        # Calculate cos(i * \pi * \tau).
        cosines = torch.cos(
            taus.view(batch_size, num_taus, 1) * i_pi
            ).view(batch_size * num_taus, self.num_cosines)

        # Calculate embeddings of taus.
        tau_embeddings = self.embeding_net(cosines).view(
            batch_size, num_taus, self.embedding_dim)

        return tau_embeddings
