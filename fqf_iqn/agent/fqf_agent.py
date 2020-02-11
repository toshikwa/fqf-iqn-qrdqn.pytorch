import torch
from torch.optim import Adam, RMSprop

from fqf_iqn.model import FQF
from fqf_iqn.utils import update_params, calculate_quantile_huber_loss
from .base_agent import BaseAgent


class FQFAgent(BaseAgent):

    def __init__(self, env, test_env, log_dir, num_steps=5*(10**7),
                 batch_size=32, num_taus=32, num_cosines=64, ent_coef=1.0,
                 kappa=1.0, fraction_lr=2.5e-9, quantile_lr=5e-5,
                 memory_size=10**6, gamma=0.99, multi_step=1,
                 update_interval=4, target_update_interval=10000,
                 start_steps=50000, epsilon_train=0.01, epsilon_eval=0.001,
                 log_interval=50, eval_interval=250000, num_eval_steps=125000,
                 cuda=True, seed=0):
        super(FQFAgent, self).__init__(
            env, test_env, log_dir, num_steps, batch_size, memory_size,
            gamma, multi_step, update_interval, target_update_interval,
            start_steps, epsilon_train, epsilon_eval, log_interval,
            eval_interval, num_eval_steps, cuda, seed)

        # Fully parametrized Quantile Function.
        self.fqf = FQF(
            num_channels=self.env.observation_space.shape[0],
            num_actions=self.num_actions, num_taus=num_taus,
            num_cosines=num_cosines, device=self.device)

        self.fraction_optim = RMSprop(
            self.fqf.fraction_net.parameters(),
            lr=fraction_lr, eps=1e-2/batch_size)
        self.quantile_optim = Adam(
            list(self.fqf.dqn_base.parameters())
            + list(self.fqf.quantile_net.parameters()),
            lr=quantile_lr, eps=1e-2/batch_size)

        self.num_taus = num_taus
        self.num_cosines = num_cosines
        self.ent_coef = ent_coef
        self.kappa = kappa

    def exploit(self, state):
        # Act without randomness.
        state = torch.ByteTensor(
            state).unsqueeze(0).to(self.device).float() / 255.

        with torch.no_grad():
            # Calculate state embeddings.
            state_embedding = self.fqf.dqn_base(state)
            # Calculate proposals of fractions.
            tau, hat_tau, _ = self.fqf.fraction_net(state_embedding)
            # Calculate Q and get greedy action.
            action = self.fqf.calculate_q(
                state_embedding, tau, hat_tau).argmax().item()

        return action

    def learn(self):
        self.learning_steps += 1

        if self.steps % self.target_update_interval == 0:
            self.fqf.update_target()

        states, actions, rewards, next_states, dones =\
            self.memory.sample(self.batch_size)

        # Calculate features of states.
        state_embeddings = self.fqf.dqn_base(states)
        # Calculate fractions and entropies.
        taus, hat_taus, entropies = self.fqf.fraction_net(state_embeddings)

        fraction_loss = self.calculate_fraction_loss(
            state_embeddings, taus, hat_taus, actions)

        quantile_loss = self.calculate_quantile_loss(
            state_embeddings, taus, hat_taus, actions, rewards,
            next_states, dones)

        # We use entropy loss as a regularizer to prevent the distribution
        # from degenerating into a deterministic one.
        entropy_loss = -self.ent_coef * entropies.mean()

        update_params(self.fraction_optim, fraction_loss + entropy_loss, True)
        update_params(self.quantile_optim, quantile_loss + entropy_loss)

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
                curr_q = self.fqf.calculate_q(
                    state_embeddings, taus, hat_taus)
                mean_q = curr_q.mean(dim=0).sum()

            self.writer.add_scalar(
                'stats/mean_Q', mean_q, self.learning_steps)
            self.writer.add_scalar(
                'stats/entropy', entropies.mean().detach().item(),
                self.learning_steps)

    def calculate_fraction_loss(self, state_embeddings, taus, hat_taus,
                                actions):

        gradient_of_taus = self.fqf.calculate_gradients_of_tau_s(
            state_embeddings, taus, hat_taus)
        assert gradient_of_taus.shape == (
            self.batch_size, self.num_taus-1, self.num_actions)

        # action_index = actions[..., None].expand(
        #     self.batch_size, self.num_taus-1, 1)
        # gradient_of_taus = self.fqf.calculate_gradients_of_tau_sa(
        #     state_embeddings, taus, hat_taus, action_index)
        # assert gradient_of_taus.shape == (
        #     self.batch_size, self.num_taus-1, 1)

        fraction_loss = (
            gradient_of_taus * taus[:, 1:-1, None]).mean(dim=0).sum()

        return fraction_loss

    def calculate_quantile_loss(self, state_embeddings, taus, hat_taus,
                                actions, rewards, next_states, dones):

        # Calculate quantile values of current states and all actions.
        current_s_quantiles = self.fqf.quantile_net(
            state_embeddings, hat_taus)

        # Repeat current actions into (batch_size, num_taus, 1).
        action_index = actions[..., None].expand(
            self.batch_size, self.num_taus, 1)

        # Calculate quantile values of current states and current actions.
        current_sa_quantiles = current_s_quantiles.gather(
            dim=2, index=action_index).view(self.batch_size, 1, self.num_taus)

        with torch.no_grad():
            # Calculate features of next states.
            next_state_embeddings = self.fqf.dqn_base(next_states)

            # Calculate fractions correspond to next states.
            next_taus, next_hat_taus, _ =\
                self.fqf.fraction_net(next_state_embeddings)

            # Calculate quantile values of next states and all actions.
            next_s_quantiles = self.fqf.target_net(
                next_state_embeddings, hat_taus)

            # Calculate next greedy actions.
            next_actions = torch.argmax(self.fqf.calculate_q(
                next_state_embeddings, next_taus, next_hat_taus), dim=1
                ).view(-1, 1, 1)
            assert next_actions.shape == (self.batch_size, 1, 1)

            # Repeat next actions into (batch_size, num_taus, 1).
            next_action_index = next_actions.expand(
                self.batch_size, self.num_taus, 1)

            # Calculate quantile values of next states and next actions.
            next_sa_quantiles = next_s_quantiles.gather(
                dim=2, index=next_action_index).view(-1, self.num_taus, 1)

            # Calculate target quantile values.
            target_sa_quantiles = rewards[..., None] + (
                1.0 - dones[..., None]) * self.gamma_n * next_sa_quantiles
            assert target_sa_quantiles.shape == (
                self.batch_size, self.num_taus, 1)

        # TD errors.
        td_errors = target_sa_quantiles - current_sa_quantiles
        assert td_errors.shape == (
            self.batch_size, self.num_taus, self.num_taus)

        # Calculate quantile huber loss.
        quantile_huber_loss = calculate_quantile_huber_loss(
            td_errors, hat_taus, self.kappa)

        return quantile_huber_loss

    def save_models(self):
        self.fqf.save(self.model_dir)
