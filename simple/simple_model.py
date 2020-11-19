import torch
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from torch import nn


class SimpleModel(TorchModelV2, nn.Module):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name, hidden_size):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)

        self.layers = nn.Sequential(
            nn.Linear(obs_space.shape[0], hidden_size),
            nn.LeakyReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU()
        )

        self.action_head = nn.Linear(hidden_size, num_outputs)
        self.value_head = nn.Linear(hidden_size, 1)

        self.value_out = None

        self.apply(SimpleModel._init_weights)

    @staticmethod
    def _init_weights(m):
        if type(m) == nn.Linear:
            torch.nn.init.xavier_normal_(m.weight, gain=nn.init.calculate_gain('leaky_relu'))
            m.bias.data.fill_(0)

    def forward(self, input_dict, state, seq_lens):
        state = input_dict["obs"]
        x = self.layers(state)
        self.value_out = self.value_head(x).squeeze(-1)
        return self.action_head(x), []

    def value_function(self):
        return self.value_out.detach()
