import os
import torch

from fqf_iqn.network import DQNBase, QuantileValueNetwork
from fqf_iqn.utils import grad_false


class IQN:

    def __init__(self, num_channels, num_actions, N=64, N_dash=64, K=32,
                 num_cosines=64, embedding_dim=7*7*64,
                 device=torch.device('cpu')):

        # Feature extractor.
        self.dqn_base = DQNBase(
            num_channels=num_channels, embedding_dim=embedding_dim).to(device)
        # Quantile Value Network.
        self.quantile_net = QuantileValueNetwork(
            num_actions=num_actions, num_cosines=num_cosines,
            embedding_dim=embedding_dim).to(device)
        # Target Network.
        self.target_net = QuantileValueNetwork(
            num_actions=num_actions, num_cosines=num_cosines,
            embedding_dim=embedding_dim).eval().to(device)

        # Copy parameters of the learning network to the target network.
        self.update_target()
        # Disable gradient calculations of the target network.
        grad_false(self.target_net)

        self.num_actions = num_actions
        self.N = N
        self.N_dash = N_dash
        self.K = K
        self.num_cosines = num_cosines
        self.embedding_dim = embedding_dim

    def calculate_q(self, state_embeddings):
        batch_size = state_embeddings.shape[0]

        # Calculate random fractions.
        taus = torch.rand(
            batch_size, self.K, dtype=state_embeddings.dtype,
            device=state_embeddings.device)

        # Calculate quantiles of random fractions.
        quantiles = self.quantile_net(state_embeddings, taus)
        assert quantiles.shape == (batch_size, self.K, self.num_actions)

        # Calculate expectations of values.
        q = quantiles.mean(dim=1)
        assert q.shape == (batch_size, self.num_actions)

        return q

    def update_target(self):
        self.target_net.load_state_dict(
            self.quantile_net.state_dict())

    def save(self, save_dir):
        torch.save(
            self.dqn_base.state_dict(),
            os.path.join(save_dir, 'dqn_base.pth'))
        torch.save(
            self.quantile_net.state_dict(),
            os.path.join(save_dir, 'quantile_net.pth'))
        torch.save(
            self.target_net.state_dict(),
            os.path.join(save_dir, 'target_net.pth'))

    def load(self, save_dir):
        self.dqn_base.load_state_dict(torch.load(
            os.path.join(save_dir, 'dqn_base.pth')))
        self.quantile_net.load_state_dict(torch.load(
            os.path.join(save_dir, 'quantile_net.pth')))
        self.target_net.load_state_dict(torch.load(
            os.path.join(save_dir, 'target_net.pth')))
