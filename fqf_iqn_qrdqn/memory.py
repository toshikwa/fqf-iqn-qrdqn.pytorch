from collections import deque
import numpy as np
import torch


class MultiStepBuff:
    keys = ["state", "action", "reward"]

    def __init__(self, maxlen=3):
        super(MultiStepBuff, self).__init__()
        self.maxlen = int(maxlen)
        self.memory = {
            key: deque(maxlen=self.maxlen)
            for key in self.keys
            }

    def append(self, state, action, reward):
        self.memory["state"].append(state)
        self.memory["action"].append(action)
        self.memory["reward"].append(reward)

    def get(self, gamma=0.99):
        assert len(self) == self.maxlen
        reward = self._multi_step_reward(gamma)
        state = self.memory["state"].popleft()
        action = self.memory["action"].popleft()
        _ = self.memory["reward"].popleft()
        return state, action, reward

    def _multi_step_reward(self, gamma):
        return np.sum([
            r * (gamma ** i) for i, r
            in enumerate(self.memory["reward"])])

    def __getitem__(self, key):
        if key not in self.keys:
            raise Exception(f'There is no key {key} in MultiStepBuff.')
        return self.memory[key]

    def reset(self):
        for key in self.keys:
            self.memory[key].clear()

    def __len__(self):
        return len(self.memory['state'])


class LazyMemory(dict):
    state_keys = ['state', 'next_state']
    np_keys = ['action', 'reward', 'done']
    keys = state_keys + np_keys

    def __init__(self, capacity, state_shape, device):
        super(LazyMemory, self).__init__()
        self.capacity = int(capacity)
        self.state_shape = state_shape
        self.device = device
        self.reset()

    def reset(self):
        self['state'] = []
        self['next_state'] = []

        self['action'] = np.empty((self.capacity, 1), dtype=np.int64)
        self['reward'] = np.empty((self.capacity, 1), dtype=np.float32)
        self['done'] = np.empty((self.capacity, 1), dtype=np.float32)

        self._n = 0
        self._p = 0

    def append(self, state, action, reward, next_state, done,
               episode_done=None):
        self._append(state, action, reward, next_state, done)

    def _append(self, state, action, reward, next_state, done):
        self['state'].append(state)
        self['next_state'].append(next_state)
        self['action'][self._p] = action
        self['reward'][self._p] = reward
        self['done'][self._p] = done

        self._n = min(self._n + 1, self.capacity)
        self._p = (self._p + 1) % self.capacity

        self.truncate()

    def truncate(self):
        while len(self) > self.capacity:
            del self['state'][0]
            del self['next_state'][0]

    def sample(self, batch_size):
        indices = np.random.randint(low=0, high=len(self), size=batch_size)
        return self._sample(indices, batch_size)

    def _sample(self, indices, batch_size):
        bias = -self._p if self._n == self.capacity else 0

        states = np.empty(
            (batch_size, *self.state_shape), dtype=np.uint8)
        next_states = np.empty(
            (batch_size, *self.state_shape), dtype=np.uint8)

        for i, index in enumerate(indices):
            _index = np.mod(index+bias, self.capacity)
            states[i, ...] = self['state'][_index]
            next_states[i, ...] = self['next_state'][_index]

        states = torch.ByteTensor(states).to(self.device).float() / 255.
        next_states = torch.ByteTensor(
            next_states).to(self.device).float() / 255.
        actions = torch.LongTensor(self['action'][indices]).to(self.device)
        rewards = torch.FloatTensor(self['reward'][indices]).to(self.device)
        dones = torch.FloatTensor(self['done'][indices]).to(self.device)

        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self['state'])

    def get(self):
        return dict(self)

    def load(self, memory):
        for key in self.state_keys:
            self[key].extend(memory[key])

        num_data = len(memory['state'])
        if self._p + num_data <= self.capacity:
            for key in self.np_keys:
                self[key][self._p:self._p+num_data] = memory[key]
        else:
            mid_index = self.capacity - self._p
            end_index = num_data - mid_index
            for key in self.np_keys:
                self[key][self._p:] = memory[key][:mid_index]
                self[key][:end_index] = memory[key][mid_index:]

        self._n = min(self._n + num_data, self.capacity)
        self._p = (self._p + num_data) % self.capacity
        self.truncate()
        assert self._n == len(self)


class LazyMultiStepMemory(LazyMemory):

    def __init__(self, capacity, state_shape, device, gamma=0.99,
                 multi_step=3):
        super(LazyMultiStepMemory, self).__init__(
            capacity, state_shape, device)

        self.gamma = gamma
        self.multi_step = int(multi_step)
        if self.multi_step != 1:
            self.buff = MultiStepBuff(maxlen=self.multi_step)

    def append(self, state, action, reward, next_state, done):
        if self.multi_step != 1:
            self.buff.append(state, action, reward)

            if len(self.buff) == self.multi_step:
                state, action, reward = self.buff.get(self.gamma)
                self._append(state, action, reward, next_state, done)

            if done:
                self.buff.reset()
        else:
            self._append(state, action, reward, next_state, done)
