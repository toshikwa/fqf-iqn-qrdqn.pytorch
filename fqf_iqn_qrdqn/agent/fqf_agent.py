import torch
from torch.optim import Adam, RMSprop

from fqf_iqn_qrdqn.model import FQF
from fqf_iqn_qrdqn.utils import calculate_quantile_huber_loss, disable_gradients, evaluate_quantile_at_action, update_params

from .base_agent import BaseAgent


class FQFAgent(BaseAgent):

    def __init__(self, env, test_env, log_dir, num_steps=5*(10**7),
                 batch_size=32, N=32, num_cosines=64, ent_coef=0,
                 kappa=1.0, quantile_lr=5e-5, fraction_lr=2.5e-9,
                 memory_size=10**6, gamma=0.99, multi_step=1,
                 update_interval=4, target_update_interval=10000,
                 start_steps=50000, epsilon_train=0.01, epsilon_eval=0.001,
                 epsilon_decay_steps=250000, double_q_learning=False,
                 dueling_net=False, noisy_net=False, use_per=False,
                 log_interval=100, eval_interval=250000, num_eval_steps=125000,
                 max_episode_steps=27000, grad_cliping=None, cuda=True,
                 seed=0):
        super(FQFAgent, self).__init__(
            env, test_env, log_dir, num_steps, batch_size, memory_size,
            gamma, multi_step, update_interval, target_update_interval,
            start_steps, epsilon_train, epsilon_eval, epsilon_decay_steps,
            double_q_learning, dueling_net, noisy_net, use_per, log_interval,
            eval_interval, num_eval_steps, max_episode_steps, grad_cliping,
            cuda, seed)

        # Online network.
        self.online_net = FQF(
            num_channels=env.observation_space.shape[0],
            num_actions=self.num_actions, N=N,
            num_cosines=num_cosines, dueling_net=dueling_net,
            noisy_net=noisy_net).to(self.device)
        # Target network.
        self.target_net = FQF(
            num_channels=env.observation_space.shape[0],
            num_actions=self.num_actions, N=N,
            num_cosines=num_cosines, dueling_net=dueling_net,
            noisy_net=noisy_net, target=True).to(self.device)

        # Copy parameters of the learning network to the target network.
        self.update_target()
        # Disable calculations of gradients of the target network.
        disable_gradients(self.target_net)

        self.fraction_optim = RMSprop(
            self.online_net.fraction_net.parameters(),
            lr=fraction_lr, alpha=0.95, eps=0.00001)

        self.quantile_optim = Adam(
            list(self.online_net.dqn_net.parameters())
            + list(self.online_net.cosine_net.parameters())
            + list(self.online_net.quantile_net.parameters()),
            lr=quantile_lr, eps=1e-2/batch_size)

        # NOTE: The author said the training of Fraction Proposal Net is
        # unstable and value distribution degenerates into a deterministic
        # one rarely (e.g. 1 out of 20 seeds). So you can use entropy of value
        # distribution as a regularizer to stabilize (but possibly slow down)
        # training.
        self.ent_coef = ent_coef
        self.N = N
        self.num_cosines = num_cosines
        self.kappa = kappa

    def update_target(self):
        self.target_net.dqn_net.load_state_dict(
            self.online_net.dqn_net.state_dict())
        self.target_net.quantile_net.load_state_dict(
            self.online_net.quantile_net.state_dict())
        self.target_net.cosine_net.load_state_dict(
            self.online_net.cosine_net.state_dict())

    def learn(self):
        self.learning_steps += 1
        self.online_net.sample_noise()
        self.target_net.sample_noise()

        if self.use_per:
            (states, actions, rewards, next_states, dones), weights =\
                self.memory.sample(self.batch_size)
        else:
            states, actions, rewards, next_states, dones =\
                self.memory.sample(self.batch_size)
            weights = None

        # Calculate embeddings of current states.
        state_embeddings = self.online_net.calculate_state_embeddings(states)

        # Calculate fractions of current states and entropies.
        taus, tau_hats, entropies =\
            self.online_net.calculate_fractions(
                state_embeddings=state_embeddings.detach())

        # Calculate quantile values of current states and actions at tau_hats.
        current_sa_quantile_hats = evaluate_quantile_at_action(
            self.online_net.calculate_quantiles(
                tau_hats, state_embeddings=state_embeddings),
            actions)
        assert current_sa_quantile_hats.shape == (
            self.batch_size, self.N, 1)

        # NOTE: Detach state_embeddings not to update convolution layers. Also,
        # detach current_sa_quantile_hats because I calculate gradients of taus
        # explicitly, not by backpropagation.
        fraction_loss = self.calculate_fraction_loss(
            state_embeddings.detach(), current_sa_quantile_hats.detach(),
            taus, actions, weights)

        quantile_loss, mean_q, errors = self.calculate_quantile_loss(
            state_embeddings, tau_hats, current_sa_quantile_hats, actions,
            rewards, next_states, dones, weights)
        assert errors.shape == (self.batch_size, 1)

        entropy_loss = -self.ent_coef * entropies.mean()

        update_params(
            self.fraction_optim, fraction_loss + entropy_loss,
            networks=[self.online_net.fraction_net], retain_graph=True,
            grad_cliping=self.grad_cliping)
        update_params(
            self.quantile_optim, quantile_loss,
            networks=[
                self.online_net.dqn_net, self.online_net.cosine_net,
                self.online_net.quantile_net],
            retain_graph=False, grad_cliping=self.grad_cliping)

        if self.use_per:
            self.memory.update_priority(errors)

        if self.learning_steps % self.log_interval == 0:
            self.writer.add_scalar(
                'loss/fraction_loss', fraction_loss.detach().item(),
                4*self.steps)
            self.writer.add_scalar(
                'loss/quantile_loss', quantile_loss.detach().item(),
                4*self.steps)
            if self.ent_coef > 0.0:
                self.writer.add_scalar(
                    'loss/entropy_loss', entropy_loss.detach().item(),
                    4*self.steps)

            self.writer.add_scalar('stats/mean_Q', mean_q, 4*self.steps)
            self.writer.add_scalar(
                'stats/mean_entropy_of_value_distribution',
                entropies.mean().detach().item(), 4*self.steps)

    def calculate_fraction_loss(self, state_embeddings, sa_quantile_hats, taus,
                                actions, weights):
        assert not state_embeddings.requires_grad
        assert not sa_quantile_hats.requires_grad

        batch_size = state_embeddings.shape[0]

        with torch.no_grad():
            sa_quantiles = evaluate_quantile_at_action(
                self.online_net.calculate_quantiles(
                    taus=taus[:, 1:-1], state_embeddings=state_embeddings),
                actions)
            assert sa_quantiles.shape == (batch_size, self.N-1, 1)

        # NOTE: Proposition 1 in the paper requires F^{-1} is non-decreasing.
        # I relax this requirements and calculate gradients of taus even when
        # F^{-1} is not non-decreasing.

        values_1 = sa_quantiles - sa_quantile_hats[:, :-1]
        signs_1 = sa_quantiles > torch.cat([
            sa_quantile_hats[:, :1], sa_quantiles[:, :-1]], dim=1)
        assert values_1.shape == signs_1.shape

        values_2 = sa_quantiles - sa_quantile_hats[:, 1:]
        signs_2 = sa_quantiles < torch.cat([
            sa_quantiles[:, 1:], sa_quantile_hats[:, -1:]], dim=1)
        assert values_2.shape == signs_2.shape

        gradient_of_taus = (
            torch.where(signs_1, values_1, -values_1)
            + torch.where(signs_2, values_2, -values_2)
        ).view(batch_size, self.N-1)
        assert not gradient_of_taus.requires_grad
        assert gradient_of_taus.shape == taus[:, 1:-1].shape

        # Gradients of the network parameters and corresponding loss
        # are calculated using chain rule.
        if weights is not None:
            fraction_loss = ((
                (gradient_of_taus * taus[:, 1:-1]).sum(dim=1, keepdim=True)
            ) * weights).mean()
        else:
            fraction_loss = \
                (gradient_of_taus * taus[:, 1:-1]).sum(dim=1).mean()

        return fraction_loss

    def calculate_quantile_loss(self, state_embeddings, tau_hats,
                                current_sa_quantile_hats, actions, rewards,
                                next_states, dones, weights):
        assert not tau_hats.requires_grad

        with torch.no_grad():
            # NOTE: Current and target quantiles share the same proposed
            # fractions to reduce computations. (i.e. next_tau_hats = tau_hats)

            # Calculate Q values of next states.
            if self.double_q_learning:
                # Sample the noise of online network to decorrelate between
                # the action selection and the quantile calculation.
                self.online_net.sample_noise()
                next_q = self.online_net.calculate_q(states=next_states)
            else:
                next_state_embeddings =\
                    self.target_net.calculate_state_embeddings(next_states)
                next_q = self.target_net.calculate_q(
                    state_embeddings=next_state_embeddings,
                    fraction_net=self.online_net.fraction_net)

            # Calculate greedy actions.
            next_actions = torch.argmax(next_q, dim=1, keepdim=True)
            assert next_actions.shape == (self.batch_size, 1)

            # Calculate features of next states.
            if self.double_q_learning:
                next_state_embeddings =\
                    self.target_net.calculate_state_embeddings(next_states)

            # Calculate quantile values of next states and actions at tau_hats.
            next_sa_quantile_hats = evaluate_quantile_at_action(
                self.target_net.calculate_quantiles(
                    taus=tau_hats, state_embeddings=next_state_embeddings),
                next_actions).transpose(1, 2)
            assert next_sa_quantile_hats.shape == (
                self.batch_size, 1, self.N)

            # Calculate target quantile values.
            target_sa_quantile_hats = rewards[..., None] + (
                1.0 - dones[..., None]) * self.gamma_n * next_sa_quantile_hats
            assert target_sa_quantile_hats.shape == (
                self.batch_size, 1, self.N)

        td_errors = target_sa_quantile_hats - current_sa_quantile_hats
        assert td_errors.shape == (self.batch_size, self.N, self.N)

        quantile_huber_loss = calculate_quantile_huber_loss(
            td_errors, tau_hats, weights, self.kappa)

        return quantile_huber_loss, next_q.detach().mean().item(), \
            td_errors.detach().abs().sum(dim=1).mean(dim=1, keepdim=True)
