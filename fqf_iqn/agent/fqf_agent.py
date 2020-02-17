import os
import torch
from torch.optim import Adam, RMSprop
# from torch.optim.lr_scheduler import MultiStepLR

from fqf_iqn.network import DQNBase, FractionProposalNetwork,\
    QuantileValueNetwork
from fqf_iqn.utils import disable_gradients, update_params,\
    calculate_quantile_huber_loss
from .base_agent import BaseAgent


class FQFAgent(BaseAgent):

    def __init__(self, env, test_env, log_dir, num_steps=5*(10**7),
                 batch_size=32, num_taus=32, num_cosines=64, ent_coef=0,
                 kappa=1.0, quantile_lr=5e-5, fraction_lr=2.5e-9,
                 memory_size=10**6, gamma=0.99, multi_step=1,
                 update_interval=4, target_update_interval=10000,
                 start_steps=50000, epsilon_train=0.01, epsilon_eval=0.001,
                 log_interval=50, eval_interval=250000, num_eval_steps=125000,
                 grad_cliping=5.0, cuda=True, seed=0):
        super(FQFAgent, self).__init__(
            env, test_env, log_dir, num_steps, batch_size, memory_size,
            gamma, multi_step, update_interval, target_update_interval,
            start_steps, epsilon_train, epsilon_eval, log_interval,
            eval_interval, num_eval_steps, grad_cliping, cuda, seed)

        # Feature extractor.
        self.dqn_base = DQNBase(
            num_channels=env.observation_space.shape[0]).to(self.device)
        # Fraction Proposal Network.
        self.fraction_net = FractionProposalNetwork(
            num_taus=num_taus).to(self.device)
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

        self.quantile_optim = Adam(
            list(self.dqn_base.parameters())
            + list(self.quantile_net.parameters()),
            lr=quantile_lr, eps=0.0003125)
        self.fraction_optim = RMSprop(
            self.fraction_net.parameters(),
            lr=fraction_lr, alpha=0.95, eps=0.00001)

        # We sweap the learning rate of Fraction Proposal Network from
        # 2.5e-5 to 2.5e-9 during training.
        # self.lr_sweeper = MultiStepLR(
        #     self.fraction_optim, gamma=0.1,
        #     milestones=[10**4//4, 4*10**4//4, 10**6//4, 10**7//4])

        self.num_taus = num_taus
        self.num_cosines = num_cosines
        self.ent_coef = ent_coef
        self.kappa = kappa

    def update_target(self):
        self.target_net.load_state_dict(
            self.quantile_net.state_dict())

    def exploit(self, state):
        # Act without randomness.
        state = torch.ByteTensor(
            state).unsqueeze(0).to(self.device).float() / 255.

        with torch.no_grad():
            # Calculate state embeddings.
            state_embedding = self.dqn_base(state)
            # Calculate proposals of fractions.
            tau, hat_tau, _ = self.fraction_net(state_embedding)
            # Calculate Q and get greedy action.
            action = self.calculate_q(
                state_embedding, tau, hat_tau).argmax().item()

        return action

    def calculate_q(self, state_embeddings, taus, hat_taus):
        batch_size = state_embeddings.shape[0]

        # Calculate quantiles of proposed fractions.
        quantiles = self.quantile_net(state_embeddings, hat_taus)
        assert quantiles.shape == (
            batch_size, self.num_taus, self.num_actions)

        # Calculate expectations of values.
        q = ((taus[:, 1:, None] - taus[:, :-1, None]) * quantiles).sum(dim=1)
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
        # Calculate fractions and entropies.
        taus, hat_taus, entropies = self.fraction_net(state_embeddings)

        fraction_loss = self.calculate_fraction_loss(
            state_embeddings, taus, hat_taus)

        quantile_loss = self.calculate_quantile_loss(
            state_embeddings, taus, hat_taus, actions, rewards,
            next_states, dones)

        # You can use entropy loss as a regularizer to prevent the distribution
        # from degenerating into a deterministic one, which happens rarely. We
        # don't use it by default.
        entropy_loss = -self.ent_coef * entropies.mean()

        update_params(
            self.fraction_optim, fraction_loss + entropy_loss,
            networks=[self.fraction_net], retain_graph=True,
            grad_cliping=self.grad_cliping)
        update_params(
            self.quantile_optim, quantile_loss + entropy_loss,
            networks=[self.dqn_base, self.quantile_net],
            retain_graph=False, grad_cliping=self.grad_cliping)

        # self.lr_sweeper.step()

        if self.learning_steps % self.log_interval == 0:
            self.writer.add_scalar(
                'loss/fraction_loss', fraction_loss.detach().item(),
                self.learning_steps)
            self.writer.add_scalar(
                'loss/quantile_loss', quantile_loss.detach().item(),
                self.learning_steps)
            self.writer.add_scalar(
                'loss/entropy_loss', entropy_loss.detach().item(),
                self.learning_steps)

            with torch.no_grad():
                mean_q = self.calculate_q(
                    state_embeddings, taus, hat_taus).mean()

            self.writer.add_scalar(
                'stats/mean_Q', mean_q, self.learning_steps)
            self.writer.add_scalar(
                'stats/mean_entropy_of_value_distribution',
                entropies.mean().detach().item(), self.learning_steps)
            self.writer.add_scalar(
                'time/mean_learning_time', self.learning_time.get(),
                self.learning_steps)

    def calculate_fraction_loss(self, state_embeddings, taus, hat_taus):

        gradient_of_taus = self.calculate_gradients_of_tau(
            state_embeddings, taus, hat_taus)

        fraction_loss = (
            gradient_of_taus.mean(dim=2) * taus[:, 1:-1]).sum(dim=1).mean()

        return fraction_loss

    def calculate_quantile_loss(self, state_embeddings, taus, hat_taus,
                                actions, rewards, next_states, dones):

        # Calculate quantile values of current states and all actions.
        current_s_quantiles = self.quantile_net(
            state_embeddings, hat_taus)

        # Repeat current actions into (batch_size, num_taus, 1).
        action_index = actions[..., None].expand(
            self.batch_size, self.num_taus, 1)

        # Calculate quantile values of current states and current actions.
        current_sa_quantiles = current_s_quantiles.gather(
            dim=2, index=action_index).view(self.batch_size, self.num_taus, 1)

        with torch.no_grad():
            # Calculate features of next states.
            next_state_embeddings = self.dqn_base(next_states)

            # Calculate fractions correspond to next states.
            next_taus, next_hat_taus, _ =\
                self.fraction_net(next_state_embeddings)

            # Calculate quantile values of next states and all actions.
            next_s_quantiles = self.target_net(
                next_state_embeddings, next_hat_taus)

            # Calculate next greedy actions.
            next_actions = torch.argmax(self.calculate_q(
                next_state_embeddings, next_taus, next_hat_taus), dim=1
                ).view(-1, 1, 1)
            assert next_actions.shape == (self.batch_size, 1, 1)

            # Repeat next actions into (batch_size, num_taus, 1).
            next_action_index = next_actions.expand(
                self.batch_size, self.num_taus, 1)

            # Calculate quantile values of next states and next actions.
            next_sa_quantiles = next_s_quantiles.gather(
                dim=2, index=next_action_index).view(-1, 1, self.num_taus)

            # Calculate target quantile values.
            target_sa_quantiles = rewards[..., None] + (
                1.0 - dones[..., None]) * self.gamma_n * next_sa_quantiles
            assert target_sa_quantiles.shape == (
                self.batch_size, 1, self.num_taus)

        # TD errors.
        td_errors = target_sa_quantiles - current_sa_quantiles
        assert td_errors.shape == (
            self.batch_size, self.num_taus, self.num_taus)

        # Calculate quantile huber loss.
        quantile_huber_loss = calculate_quantile_huber_loss(
            td_errors, hat_taus, self.kappa)

        return quantile_huber_loss

    def calculate_gradients_of_tau(self, state_embeddings, taus, hat_taus):

        batch_size = state_embeddings.shape[0]

        with torch.no_grad():
            quantile_tau_i = self.quantile_net(
                state_embeddings, taus[:, 1:-1])
            assert quantile_tau_i.shape == (
                batch_size, self.num_taus-1, self.num_actions)

            quantile_hat_tau_i = self.quantile_net(
                state_embeddings, hat_taus[:, 1:])
            assert quantile_hat_tau_i.shape == (
                batch_size, self.num_taus-1, self.num_actions)

            quantile_hat_tau_i_minus_1 = self.quantile_net(
                state_embeddings, hat_taus[:, :-1])
            assert quantile_hat_tau_i_minus_1.shape == (
                batch_size, self.num_taus-1, self.num_actions)

        gradients =\
            2*quantile_tau_i - quantile_hat_tau_i - quantile_hat_tau_i_minus_1

        return gradients

    def save_models(self, save_dir):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

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

    def load_models(self, save_dir):
        self.dqn_base.load_state_dict(torch.load(
            os.path.join(save_dir, 'dqn_base.pth')))
        self.fraction_net.load_state_dict(torch.load(
            os.path.join(save_dir, 'fraction_net.pth')))
        self.quantile_net.load_state_dict(torch.load(
            os.path.join(save_dir, 'quantile_net.pth')))
        self.target_net.load_state_dict(torch.load(
            os.path.join(save_dir, 'target_net.pth')))
