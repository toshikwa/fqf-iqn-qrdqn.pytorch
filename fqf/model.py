import os
import torch

from .network import DQNBase, FractionProposalNetwork, QuantileValueNetwork


def grad_false(network):
    for param in network.parameters():
        param.requires_grad = False


class FQF():

    def __init__(self, num_channels, num_actions, num_taus=32, num_cosines=64,
                 embedding_dim=7*7*64, device=torch.device('cpu')):

        # Feature extractor.
        self.dqn_base = DQNBase(
            num_channels=num_channels, embedding_dim=embedding_dim).to(device)
        # Fraction Proposal Network.
        self.fraction_net = FractionProposalNetwork(
            num_taus=num_taus, embedding_dim=embedding_dim).to(device)
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
        self.num_taus = num_taus
        self.num_cosines = num_cosines
        self.embedding_dim = embedding_dim

    def calculate_q(self, states):
        num_batches = states.shape[0]

        # Calculate state embeddings.
        state_embeddings = self.dqn_base(states)
        assert state_embeddings.shape == (num_batches, self.embedding_dim)

        # Calculate proposals of fractions.
        taus, hat_taus = self.fraction_net(state_embeddings)
        assert taus.shape == (num_batches, self.num_taus+1)
        assert hat_taus.shape == (num_batches, self.num_taus)

        # Calculate quantiles of proposed fractions.
        quantiles = self.quantile_net(state_embeddings, hat_taus)
        assert quantiles.shape == (
            num_batches, self.num_taus, self.num_actions)

        # Calculate expectations of fractions
        q = ((taus[:, 1:, None] - taus[:, :-1, None]) * quantiles).sum(dim=1)
        assert q.shape == (num_batches, self.num_actions)

        return q

    def calculate_gradients_of_tau(self, state_embeddings, taus, hat_taus):
        num_batches = state_embeddings.shape[0]

        with torch.no_grad():
            quantile_tau_i = self.quantile_net(
                state_embeddings, taus[:, 1:-1])
            assert quantile_tau_i.shape == (
                num_batches, self.num_taus-1, self.num_actions)

            quantile_hat_tau_i = self.quantile_net(
                state_embeddings, hat_taus[:, 1:])
            assert quantile_hat_tau_i.shape == (
                num_batches, self.num_taus-1, self.num_actions)

            quantile_hat_tau_i_1 = self.quantile_net(
                state_embeddings, hat_taus[:, :-1])
            assert quantile_hat_tau_i_1.shape == (
                num_batches, self.num_taus-1, self.num_actions)

        gradients =\
            2 * quantile_tau_i - quantile_hat_tau_i - quantile_hat_tau_i_1

        return gradients

    def update_target(self):
        self.target_net.load_state_dict(
            self.quantile_net.state_dict())

    def save(self, save_dir):
        torch.save(
            self.dqn_base.state_dict(),
            os.path.join(save_dir, 'dqn_base.pth'))
        torch.save(
            self.fraction_net.state_dict(),
            os.path.join(save_dir, 'fraction_net.pth'))
        torch.save(
            self.quantile_net.state_dict(),
            os.path.join(save_dir, 'quantile_net.pth'))
        torch.save(
            self.target_net.state_dict(),
            os.path.join(save_dir, 'target_net.pth'))

    def load(self, save_dir):
        self.dqn_base.load_state_dict(torch.load(
            os.path.join(save_dir, 'dqn_base.pth')))
        self.fraction_net.load_state_dict(torch.load(
            os.path.join(save_dir, 'fraction_net.pth')))
        self.quantile_net.load_state_dict(torch.load(
            os.path.join(save_dir, 'quantile_net.pth')))
        self.target_net.load_state_dict(torch.load(
            os.path.join(save_dir, 'target_net.pth')))
