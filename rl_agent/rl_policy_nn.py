from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.framework import try_import_torch

torch, nn = try_import_torch()


class PolicyNN(TorchModelV2, nn.Module):
    def __init__(
        self,
        obs_space,
        action_space,
        num_outputs,
        model_config,
        name,
        dropout_rate,
        policy_hidden_layers,
        vf_hidden_layers,
    ):
        TorchModelV2.__init__(
            self, obs_space, action_space, num_outputs, model_config, name
        )
        nn.Module.__init__(self)

        self.share_weights = model_config["vf_share_layers"]

        dropout = dropout_rate
        input_size = obs_space.original_space["embedding"].shape[0]

        # Policy network
        policy_hidden_sizes = policy_hidden_layers
        policy_output_size = num_outputs

        self.policy_layers = nn.ModuleList()
        policy_layers_sizes = [input_size] + policy_hidden_sizes
        for i in range(len(policy_layers_sizes) - 1):
            layer = nn.Linear(policy_layers_sizes[i], policy_layers_sizes[i + 1])
            nn.init.xavier_uniform_(layer.weight)
            self.policy_layers.append(layer)
            self.policy_layers.append(nn.BatchNorm1d(policy_layers_sizes[i + 1]))
            self.policy_layers.append(nn.Dropout(dropout))

        # Policy head
        self.logits_layer = nn.Linear(policy_layers_sizes[-1], policy_output_size)
        nn.init.xavier_uniform_(self.logits_layer.weight)

        # Value separate network
        value_hidden_sizes = vf_hidden_layers
        value_output_size = 1

        self.value_layers = nn.ModuleList()
        value_layers_sizes = [input_size] + value_hidden_sizes
        for i in range(len(value_layers_sizes) - 1):
            layer = nn.Linear(value_layers_sizes[i], value_layers_sizes[i + 1])
            nn.init.xavier_uniform_(layer.weight)
            self.value_layers.append(layer)
            self.value_layers.append(nn.BatchNorm1d(value_layers_sizes[i + 1]))
            self.value_layers.append(nn.Dropout(dropout))

        # Value head
        if self.share_weights:
            # If value function and policy weights are shared, use last weights of the policy network
            self.value_layer = nn.Linear(policy_layers_sizes[-1], value_output_size)
        else:
            self.value_layer = nn.Linear(value_layers_sizes[-1], value_output_size)
        nn.init.xavier_uniform_(self.value_layer.weight)

    def forward(self, input_dict, state, seq_lens):
        obs = input_dict["obs"]["embedding"].float()
        self._last_flat_in = obs.reshape(obs.shape[0], -1)
        self._features = self._last_flat_in
        for layer in self.policy_layers:
            if isinstance(layer, nn.Linear):
                self._last_flat_in = layer(self._last_flat_in)
            elif isinstance(layer, nn.BatchNorm1d):
                self._last_flat_in = layer(self._last_flat_in)
                self._last_flat_in = nn.functional.relu(self._last_flat_in)
            elif isinstance(layer, nn.Dropout):
                self._last_flat_in = layer(self._last_flat_in)
        # Output logits
        logits = self.logits_layer(self._last_flat_in)
        # Masking selected and restricted actions
        logits = logits - (1_000_000 * input_dict["obs"]["actions_mask"])
        return logits, state

    def value_function(self):
        if self.share_weights:
            assert self._last_flat_in is not None, "must call forward() first"
            value = self.value_layer(self._last_flat_in)
        else:
            assert self._features is not None, "must call forward() first"
            self._value_features = self._features
            for layer in self.value_layers:
                if isinstance(layer, nn.Linear):
                    self._value_features = layer(self._value_features)
                elif isinstance(layer, nn.BatchNorm1d):
                    self._value_features = layer(self._value_features)
                    self._value_features = nn.functional.relu(self._value_features)
                elif isinstance(layer, nn.Dropout):
                    self._value_features = layer(self._value_features)
            # Output the value
            value = self.value_layer(self._value_features)
        return value.squeeze(1)
