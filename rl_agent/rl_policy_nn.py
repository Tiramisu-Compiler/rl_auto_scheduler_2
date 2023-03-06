from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.framework import  try_import_torch
import numpy as np

torch, nn = try_import_torch()


class PolicyNN(TorchModelV2, nn.Module):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(
            self, obs_space, action_space, num_outputs, model_config, name
        )
        nn.Module.__init__(self)

        input_size = int(np.product(obs_space.shape))
        hidden_sizes = [512,512,256,64]
        output_size = 2
        
        self.layers = nn.ModuleList()
        sizes = [input_size] + hidden_sizes
        for i in range(len(sizes) - 1):
            layer = nn.Linear(sizes[i], sizes[i+1])
            nn.init.xavier_uniform_(layer.weight)
            self.layers.append(layer)
            self.layers.append(nn.BatchNorm1d(sizes[i+1]))
            self.layers.append(nn.Dropout(0.2))
        
        # Actions logits 
        self.logits_layer = nn.Linear(hidden_sizes[-1], output_size)
        nn.init.xavier_uniform_(self.logits_layer.weight)
        
        # Action value
        self.value_layer = nn.Linear(input_size, 1)
        nn.init.xavier_uniform_(self.value_layer.weight)

    def forward(self, input_dict, state, seq_lens):
        obs = input_dict["obs_flat"].float()
        self._last_flat_in = obs.reshape(obs.shape[0], -1)
        self.features = self._last_flat_in
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                self._last_flat_in = layer(self._last_flat_in)
            elif isinstance(layer, nn.BatchNorm1d):
                self._last_flat_in = layer(self._last_flat_in)
                self._last_flat_in = nn.functional.relu(self._last_flat_in)
            elif isinstance(layer, nn.Dropout):
                self._last_flat_in = layer(self._last_flat_in)
        # Output logits
        logits = self.logits_layer(self._last_flat_in)
        return logits, state

    def value_function(self):
        assert self.features is not None, "must call forward() first"
        value = self.value_layer(self.features)
        return value.squeeze(1)