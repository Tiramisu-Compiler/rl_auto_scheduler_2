import numpy as np
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.torch.recurrent_net import RecurrentNetwork as TorchRNN
from ray.rllib.policy.rnn_sequencing import add_time_dimension
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_torch

torch, nn = try_import_torch()


class PolicyLSTM(TorchRNN, nn.Module):
    def __init__(
        self,
        obs_space,
        action_space,
        num_outputs,
        model_config,
        name,
        fc_size=1024,
        lstm_state_size=256,
        num_layers=1,
    ):
        nn.Module.__init__(self)
        super().__init__(obs_space, action_space, num_outputs, model_config, name)

        self.obs_size = obs_space.original_space["embedding"].shape[0]
        self.fc_size = fc_size
        self.lstm_state_size = lstm_state_size
        self.num_layers = num_layers

        self.shared_layers = nn.Sequential(
            nn.Linear(self.obs_size, self.fc_size),
            nn.SELU(),
            nn.Linear(self.fc_size, self.fc_size),
            nn.SELU(),
            nn.Linear(self.fc_size, self.fc_size),
            nn.SELU(),
            nn.Linear(self.fc_size, self.fc_size),
            nn.SELU(),
            nn.Linear(self.fc_size, self.fc_size),
            nn.SELU(),
            nn.Linear(self.fc_size, self.fc_size),
            nn.SELU(),
        )

        for model in self.shared_layers.children():
            if isinstance(model, nn.Linear):
                nn.init.xavier_uniform_(model.weight)

        self.lstm = nn.LSTM(
            self.fc_size, self.lstm_state_size, num_layers=num_layers, batch_first=True
        )
        for name, param in self.lstm.named_parameters():
            if "bias" in name:
                nn.init.constant_(param, 0.0)
            elif "weight" in name:
                nn.init.xavier_uniform_(param)

        # Actions branch
        self.action_network = nn.Sequential(
            nn.Linear(self.lstm_state_size, self.lstm_state_size),
            nn.SELU(),
            nn.Linear(self.lstm_state_size, self.lstm_state_size),
            nn.SELU(),
            nn.Linear(self.lstm_state_size, num_outputs),
        )

        for model in self.action_network.children():
            if isinstance(model, nn.Linear):
                nn.init.xavier_uniform_(model.weight)
        # self.action_hidden_layer = nn.Linear(self.lstm_state_size, self.lstm_state_size)
        # nn.init.xavier_uniform_(self.action_hidden_layer.weight)
        # self.action_branch = nn.Linear(self.lstm_state_size, num_outputs)
        # nn.init.xavier_uniform_(self.action_branch.weight)

        # Value branch

        self.value_network = nn.Sequential(
            nn.Linear(self.lstm_state_size, self.lstm_state_size),
            nn.SELU(),
            nn.Linear(self.lstm_state_size, self.lstm_state_size),
            nn.SELU(),
            nn.Linear(self.lstm_state_size, 1),
        )

        for model in self.value_network.children():
            if isinstance(model, nn.Linear):
                nn.init.xavier_uniform_(model.weight)

        # self.value_hidden_layer = nn.Linear(self.lstm_state_size, self.lstm_state_size)
        # nn.init.xavier_uniform_(self.value_hidden_layer.weight)
        # self.value_branch = nn.Linear(self.lstm_state_size, 1)
        # nn.init.xavier_uniform_(self.value_branch.weight)
        # Holds the current "base" output (before logits layer).
        self._features = None

    @override(ModelV2)
    def get_initial_state(self):
        h = [
            # The values of initial hidden states h_0 and c_0
            torch.tensor(np.zeros(self.lstm_state_size, np.float32)),
            torch.tensor(np.zeros(self.lstm_state_size, np.float32)),
        ]
        return h

    @override(ModelV2)
    def value_function(self):
        assert self._features is not None, "must call forward() first"
        # x = nn.functional.selu(self.value_hidden_layer(self._features))
        return torch.reshape(self.value_network(self._features), [-1])

    @override(ModelV2)
    def forward(
        self,
        input_dict,
        state,
        seq_lens,
    ):
        flat_inputs = input_dict["obs"]["embedding"].float()
        inputs = add_time_dimension(
            flat_inputs,
            seq_lens=seq_lens,
            framework="torch",
            time_major=False,
        )
        output, new_state = self.forward_rnn(inputs, state, seq_lens)
        logits = torch.reshape(output, [-1, self.num_outputs])
        # Applying the mask to actions
        logits = logits - (1_000_000 * input_dict["obs"]["actions_mask"])
        return logits, new_state

    @override(TorchRNN)
    def forward_rnn(self, inputs, state, seq_lens):
        x = self.shared_layers(inputs)
        self._features, [h, c] = self.lstm(
            x, [torch.unsqueeze(state[0], 0), torch.unsqueeze(state[1], 0)]
        )
        # x = nn.functional.selu(self.action_hidden_layer(self._features))
        action_out = self.action_network(self._features)
        return action_out, [torch.squeeze(h, 0), torch.squeeze(c, 0)]
