from torch import nn

from fqf_iqn_qrdqn.network import NoisyLinear


class BaseModel(nn.Module):

    def __init__(self):
        super().__init__()

    def sample_noise(self):
        if self.noisy_net:
            for m in self.modules():
                if isinstance(m, NoisyLinear):
                    m.sample()
