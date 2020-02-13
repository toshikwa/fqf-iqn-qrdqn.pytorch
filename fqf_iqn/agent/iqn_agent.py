import os
import torch
from torch.optim import Adam

from fqf_iqn.network import DQNBase, QuantileValueNetwork
from fqf_iqn.utils import disable_gradients, update_params,\
    calculate_quantile_huber_loss
from .base_agent import BaseAgent


class IQNAgent(BaseAgent):

    def __init__(self, env, test_env, log_dir, num_steps=5*(10**7),
                 batch_size=32, N=64, N_dash=64, K=32, num_cosines=64,
                 kappa=1.0, lr=5e-5, memory_size=10**6, gamma=0.99,
                 multi_step=1, update_interval=4, target_update_interval=10000,
                 start_steps=50000, epsilon_train=0.01, epsilon_eval=0.001,
                 log_interval=50, eval_interval=250000, num_eval_steps=125000,
                 grad_cliping=5.0, cuda=True, seed=0):
        super(IQNAgent, self).__init__(
            env, test_env, log_dir, num_steps, batch_size, memory_size,
            gamma, multi_step, update_interval, target_update_interval,
            start_steps, epsilon_train, epsilon_eval, log_interval,
            eval_interval, num_eval_steps, grad_cliping, cuda, seed)

        # Feature extractor.
        self.dqn_base = DQNBase(
            num_channels=env.observation_space.shape[0]).to(self.device)
        # Quantile Value Network.
        self.quantile_net = QuantileValueNetwork(
            num_actions=self.num_actions, num_cosines=num_cosines
            ).to(self.device)
        # Target Network.
        self.target_net = QuantileValueNetwork(
            num_actions=self.num_actions, num_cosines=num_cosines
            ).eval().to(self.device)

        # Copy parameters of the learning network to the target network.
        self.update_target()
        # Disable calculations of gradients of the target network.
        disable_gradients(self.target_net)

        self.optim = Adam(
            list(self.dqn_base.parameters())
            + list(self.quantile_net.parameters()),
            lr=lr, eps=1e-2/batch_size)

        self.N = N
        self.N_dash = N_dash
        self.K = K
        self.num_cosines = num_cosines
        self.kappa = kappa

    def update_target(self):
        self.target_net.load_state_dict(
            self.quantile_net.state_dict())

    def exploit(self, state):
        # Act without randomness.
        state = torch.ByteTensor(
            state).unsqueeze(0).to(self.device).float() / 255.
        with torch.no_grad():
            state_embedding = self.dqn_base(state)
            action = self.calculate_q(
                state_embedding).argmax().item()
        return action

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

    def learn(self):
        self.learning_steps += 1

        if self.steps % self.target_update_interval == 0:
            self.update_target()

        states, actions, rewards, next_states, dones =\
            self.memory.sample(self.batch_size)

        # Calculate features of states.
        state_embeddings = self.dqn_base(states)

        quantile_loss = self.calculate_loss(
            state_embeddings, actions, rewards, next_states, dones)

        update_params(
            self.optim, quantile_loss,
            networks=[self.dqn_base, self.quantile_net],
            retain_graph=False, grad_cliping=self.grad_cliping)

        if self.learning_steps % self.log_interval == 0:
            self.writer.add_scalar(
                'loss/quantile_loss', quantile_loss.detach().item(),
                self.learning_steps)
            with torch.no_grad():
                mean_q = self.calculate_q(state_embeddings).mean()
            self.writer.add_scalar(
                'stats/mean_Q', mean_q, self.learning_steps)

    def calculate_loss(self, state_embeddings, actions, rewards, next_states,
                       dones):

        # Sample fractions.
        taus = torch.rand(
            self.batch_size, self.N, dtype=state_embeddings.dtype,
            device=state_embeddings.device)

        # Calculate quantile values of current states and all actions.
        current_s_quantiles = self.quantile_net(state_embeddings, taus)

        # Repeat current actions into (batch_size, N, 1).
        action_index = actions[..., None].expand(
            self.batch_size, self.N, 1)

        # Calculate quantile values of current states and current actions.
        current_sa_quantiles = current_s_quantiles.gather(
            dim=2, index=action_index).view(self.batch_size, self.N, 1)

        with torch.no_grad():
            # Calculate features of next states.
            next_state_embeddings = self.dqn_base(next_states)

            # Sample next fractions.
            tau_dashes = torch.rand(
                self.batch_size, self.N_dash, dtype=state_embeddings.dtype,
                device=state_embeddings.device)

            # Calculate quantile values of next states and all actions.
            next_s_quantiles = self.target_net(
                next_state_embeddings, tau_dashes)

            # Calculate next greedy actions.
            next_actions = torch.argmax(
                self.calculate_q(next_state_embeddings), dim=1
                ).view(self.batch_size, 1, 1)

            # Repeat next actions into (batch_size, num_taus, 1).
            next_action_index = next_actions.expand(
                self.batch_size, self.N_dash, 1)

            # Calculate quantile values of next states and next actions.
            next_sa_quantiles = next_s_quantiles.gather(
                dim=2, index=next_action_index).view(-1, 1, self.N_dash)

            # Calculate target quantile values.
            target_sa_quantiles = rewards[..., None] + (
                1.0 - dones[..., None]) * self.gamma_n * next_sa_quantiles
            assert target_sa_quantiles.shape == (
                self.batch_size, 1, self.N_dash)

        # TD errors.
        td_errors = target_sa_quantiles - current_sa_quantiles
        assert td_errors.shape == (self.batch_size, self.N, self.N_dash)

        # Calculate quantile huber loss.
        quantile_huber_loss = calculate_quantile_huber_loss(
            td_errors, taus, self.kappa)

        return quantile_huber_loss

    def save_models(self):
        torch.save(
            self.dqn_base.state_dict(),
            os.path.join(self.model_dir, 'dqn_base.pth'))
        torch.save(
            self.quantile_net.state_dict(),
            os.path.join(self.model_dir, 'quantile_net.pth'))
        torch.save(
            self.target_net.state_dict(),
            os.path.join(self.model_dir, 'target_net.pth'))

    def load_models(self):
        self.dqn_base.load_state_dict(torch.load(
            os.path.join(self.model_dir, 'dqn_base.pth')))
        self.quantile_net.load_state_dict(torch.load(
            os.path.join(self.model_dir, 'quantile_net.pth')))
        self.target_net.load_state_dict(torch.load(
            os.path.join(self.model_dir, 'target_net.pth')))
