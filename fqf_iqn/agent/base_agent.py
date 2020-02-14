import os
from time import time
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from fqf_iqn.memory import DummyMultiStepMemory
from fqf_iqn.utils import RunningMeanStats


class BaseAgent:

    def __init__(self, env, test_env, log_dir, num_steps=5*(10**7),
                 batch_size=32, memory_size=10**6, gamma=0.99, multi_step=1,
                 update_interval=4, target_update_interval=10000,
                 start_steps=50000, epsilon_train=0.01, epsilon_eval=0.001,
                 log_interval=50, eval_interval=250000, num_eval_steps=125000,
                 grad_cliping=5.0, cuda=True, seed=0):

        self.env = env
        self.test_env = test_env

        torch.manual_seed(seed)
        np.random.seed(seed)
        self.env.seed(seed)
        self.test_env.seed(2**31-1-seed)
        # torch.backends.cudnn.deterministic = True  # It harms a performance.
        # torch.backends.cudnn.benchmark = False  # It harms a performance.

        self.device = torch.device(
            "cuda" if cuda and torch.cuda.is_available() else "cpu")

        # Replay memory which is memory-efficient to store stacked frames.
        self.memory = DummyMultiStepMemory(
            memory_size, self.env.observation_space.shape,
            self.device, gamma, multi_step)

        self.log_dir = log_dir
        self.model_dir = os.path.join(log_dir, 'model')
        self.summary_dir = os.path.join(log_dir, 'summary')
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        if not os.path.exists(self.summary_dir):
            os.makedirs(self.summary_dir)

        self.writer = SummaryWriter(log_dir=self.summary_dir)
        self.train_return = RunningMeanStats(log_interval)
        self.training_time = RunningMeanStats(log_interval)
        self.learning_time = RunningMeanStats(log_interval)

        self.steps = 0
        self.learning_steps = 0
        self.episodes = 0
        self.num_actions = self.env.action_space.n
        self.num_steps = num_steps
        self.batch_size = batch_size
        self.log_interval = log_interval
        self.eval_interval = eval_interval
        self.num_eval_steps = num_eval_steps
        self.gamma_n = gamma ** multi_step
        self.start_steps = start_steps
        self.epsilon_train = epsilon_train
        self.epsilon_eval = epsilon_eval
        self.update_interval = update_interval
        self.target_update_interval = target_update_interval
        self.grad_cliping = grad_cliping

    def run(self):
        while True:
            self.train_episode()
            if self.steps > self.num_steps:
                break

    def is_update(self):
        return self.steps % self.update_interval == 0\
            and self.steps >= self.start_steps

    def is_greedy(self, eval=False):
        if eval:
            return np.random.rand() < self.epsilon_eval
        else:
            return self.steps < self.start_steps\
                or np.random.rand() < self.epsilon_train

    def explore(self):
        # Act with randomness.
        action = self.env.action_space.sample()
        return action

    def exploit(self, state):
        raise NotImplementedError

    def learn(self):
        raise NotImplementedError

    def save_models(self):
        raise NotImplementedError

    def update_target(self):
        raise NotImplementedError

    def train_episode(self):
        self.episodes += 1
        episode_return = 0.
        episode_steps = 0
        t_time = time()

        done = False
        state = self.env.reset()

        while not done:
            if self.is_greedy(eval=False):
                action = self.explore()
            else:
                action = self.exploit(state)

            next_state, reward, done, _ = self.env.step(action)
            self.steps += 1
            episode_steps += 1
            episode_return += reward

            self.memory.append(
                state, action, reward, next_state, done)

            if self.is_update():
                l_time = time()
                self.learn()
                self.learning_time.append(time() - l_time)

            if self.steps % self.eval_interval == 0:
                e_time = time()
                self.evaluate()
                t_time += time() - e_time
                self.save_models()

            state = next_state

        # We log running mean of stats.
        self.training_time.append((time() - t_time) / episode_steps)
        self.train_return.append(episode_return)

        # We log evaluation results along with training frames = 4 * steps.
        if self.episodes % self.log_interval == 0:
            self.writer.add_scalar(
                'return/train', self.train_return.get(), 4 * self.steps)
            self.writer.add_scalar(
                'stats/mean_training_time', self.training_time.get(),
                4 * self.steps)

        print(f'Episode: {self.episodes:<4}  '
              f'episode steps: {episode_steps:<4}  '
              f'return: {episode_return:<5.1f}')

    def evaluate(self):
        num_episodes = 0
        num_steps = 0
        total_return = 0.0

        while True:
            state = self.test_env.reset()
            episode_return = 0.0
            done = False
            while not done:
                if self.is_greedy(eval=True):
                    action = self.explore()
                else:
                    action = self.exploit(state)

                next_state, reward, done, _ = self.test_env.step(action)
                num_steps += 1
                episode_return += reward
                state = next_state
                if done:
                    num_episodes += 1
                    total_return += episode_return

            if num_steps > self.num_eval_steps:
                break

        mean_return = total_return / num_episodes

        # We log evaluation results along with training frames = 4 * steps.
        self.writer.add_scalar(
            'return/test', mean_return, 4 * self.steps)
        print('-' * 60)
        print(f'Num steps: {self.steps:<5}  '
              f'return: {mean_return:<5.1f}')
        print('-' * 60)

    def __del__(self):
        self.env.close()
        self.test_env.close()
        self.writer.close()
