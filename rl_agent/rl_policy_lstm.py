from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.models.torch.recurrent_net import RecurrentNetwork as TorchRNN
from ray.rllib.utils.annotations import override
from ray.rllib.policy.rnn_sequencing import add_time_dimension
import numpy as np 

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
        num_layers = 1
    ):
        nn.Module.__init__(self)
        super().__init__(obs_space, action_space, num_outputs, model_config, name)

        self.obs_size = obs_space.original_space["embedding"].shape[0]
        self.fc_size = fc_size
        self.lstm_state_size = lstm_state_size
        self.num_layers = num_layers

        # Build the Module from fc + LSTM + 2xfc (action + value outs).
        self.fc1 = nn.Linear(self.obs_size, self.fc_size)
        self.lstm = nn.LSTM(self.fc_size, self.lstm_state_size, num_layers=num_layers ,batch_first=True)
        self.action_branch = nn.Linear(self.lstm_state_size, num_outputs)
        self.value_branch = nn.Linear(self.lstm_state_size, 1)
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
        return torch.reshape(self.value_branch(self._features), [-1])

    @override(ModelV2)
    def forward(
        self,
        input_dict,
        state,
        seq_lens,
    ) :
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
        x = nn.functional.relu(self.fc1(inputs))
        self._features, [h, c] = self.lstm(
            x, [torch.unsqueeze(state[0], 0),torch.unsqueeze(state[1], 0)]
        )
        action_out = self.action_branch(self._features)
        return action_out, [torch.squeeze(h, 0), torch.squeeze(c, 0)]

