from torch import nn

from fqf_iqn.network import DQNBase, CosineEmbeddingNetwork,\
    FractionProposalNetwork, QuantileNetwork, NoisyLinear


class FQF(nn.Module):

    def __init__(self, num_channels, num_actions, num_taus=32, num_cosines=32,
                 embedding_dim=7*7*64, dueling_net=False, noisy_net=False):
        super(FQF, self).__init__()

        # Feature extractor of DQN.
        self.dqn_net = DQNBase(num_channels=num_channels)
        # Fraction proposal network.
        self.fraction_net = FractionProposalNetwork(
            num_taus=num_taus, embedding_dim=embedding_dim)
        # Cosine embedding network.
        self.cosine_net = CosineEmbeddingNetwork(
            num_cosines=num_cosines, embedding_dim=embedding_dim,
            noisy_net=noisy_net)
        # Quantile network.
        self.quantile_net = QuantileNetwork(
            num_actions=num_actions, dueling_net=dueling_net,
            noisy_net=noisy_net)

        self.num_taus = num_taus
        self.num_channels = num_channels
        self.num_actions = num_actions
        self.num_cosines = num_cosines
        self.embedding_dim = embedding_dim
        self.dueling_net = dueling_net
        self.noisy_net = noisy_net

    def calculate_state_embeddings(self, states):
        return self.dqn_net(states)

    def calculate_fractions(self, states=None, state_embeddings=None):
        assert states is not None or state_embeddings is not None

        if state_embeddings is None:
            state_embeddings = self.dqn_net(states)

        taus, tau_hats, entropies = self.fraction_net(state_embeddings)

        return taus, tau_hats, entropies

    def calculate_quantiles(self, taus, states=None, state_embeddings=None):
        assert states is not None or state_embeddings is not None

        if state_embeddings is None:
            state_embeddings = self.dqn_net(states)

        tau_embeddings = self.cosine_net(taus)
        return self.quantile_net(state_embeddings, tau_embeddings)

    def calculate_q(self, taus=None, tau_hats=None, states=None,
                    state_embeddings=None):
        assert states is not None or state_embeddings is not None
        batch_size = states.shape[0] if states is not None\
            else state_embeddings.shape[0]

        if state_embeddings is None:
            state_embeddings = self.dqn_net(states)

        if taus is None or tau_hats is None:
            taus, tau_hats, _ = self.calculate_fractions(
                state_embeddings=state_embeddings)

        # Calculate quantiles.
        quantiles = self.calculate_quantiles(
            tau_hats, state_embeddings=state_embeddings)
        assert quantiles.shape == (batch_size, self.num_taus, self.num_actions)

        # Calculate expectations of value distribution.
        q = ((taus[:, 1:, None] - taus[:, :-1, None]) * quantiles).sum(dim=1)
        assert q.shape == (batch_size, self.num_actions)

        return q

    def reset_noise(self):
        if self.noisy_net:
            for m in self.modules():
                if isinstance(m, NoisyLinear):
                    m.reset_noise()
