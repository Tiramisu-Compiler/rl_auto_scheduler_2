import argparse
import ray, os
from ray.tune.registry import register_env
import ray.rllib.agents.ppo as ppo
from ray import air, tune
from env_api.core.services.converting_service import ConvertService
from env_api.tiramisu_api import TiramisuEnvAPIv1
from env_api.utils.config.config import Config
from rl_agent.rl_env import TiramisuRlAgent
from ray.rllib.models import ModelCatalog
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork as TorchFC
from ray.rllib.utils.framework import try_import_torch
from ray.tune.registry import get_trainable_cls
from ray.rllib.models.torch.misc import SlimFC

import gymnasium as gym, numpy as np
from gymnasium import spaces

torch, nn = try_import_torch()

ray.shutdown()


class TorchCustomModel(TorchModelV2, nn.Module):
    """Example of a PyTorch custom model that just delegates to a fc-net."""

    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(
            self, obs_space, action_space, num_outputs, model_config, name
        )
        nn.Module.__init__(self)

        self.torch_sub_model = SlimFC(in_size=2, out_size=num_outputs)
        self._value_branch = SlimFC(in_size=2, out_size=1)

    def forward(self, input_dict, state, seq_lens):
        encoded_tree = input_dict["obs"]
        print(encoded_tree["prog_tree"].shape)
        # print(ConvertService.decode_dict(encoded_tree["prog_tree"][0].numpy().tolist()))
        print(encoded_tree["prog_tree"][0].numpy())
        self._features = torch.tensor([[2, 4]], dtype=torch.float32)
        fc_out = self.torch_sub_model(self._features)
        return fc_out, state

    def value_function(self):
        return self._value_branch(self._features).squeeze(1)


parser = argparse.ArgumentParser()
parser.add_argument(
    "--run", type=str, default="PPO", help="The RLlib-registered algorithm to use."
)
parser.add_argument(
    "--framework",
    choices=["tf", "tf2", "torch"],
    default="tf",
    help="The DL framework specifier.",
)
parser.add_argument(
    "--as-test",
    action="store_true",
    help="Whether this script should be run as a test: --stop-reward must "
    "be achieved within --stop-timesteps AND --stop-iters.",
)
parser.add_argument(
    "--stop-iters", type=int, default=1, help="Number of iterations to train."
)
parser.add_argument(
    "--stop-timesteps", type=int, default=1, help="Number of timesteps to train."
)
parser.add_argument(
    "--stop-reward",
    type=float,
    default=-100,
    help="Reward at which we stop training.",
)
parser.add_argument(
    "--no-tune",
    action="store_true",
    help="Run without Tune using a manual train loop instead. In this case,"
    "use PPO without grid search and no TensorBoard.",
)
parser.add_argument(
    "--local-mode",
    action="store_true",
    help="Init Ray in local mode for easier debugging.",
)

ray.init(ignore_reinit_error=True, local_mode=False, num_cpus=28)


ENV = "TiramisuRlAgent"
register_env(ENV, lambda config: TiramisuRlAgent(config))

ModelCatalog.register_custom_model("my_model", TorchCustomModel)

args = parser.parse_args()

config = (
    # ppo.PPOConfig()
    get_trainable_cls(args.run)
    .get_default_config()
    .rollouts(num_rollout_workers=1)
    .framework("torch")
    .environment(ENV)
    .training(
        model={
            "custom_model": "my_model",
            "vf_share_layers": True,
        }
    )
    .resources(num_gpus=int(os.environ.get("RLLIB_NUM_GPUS", "0")))
)


stop = {
    "training_iteration": args.stop_iters,
    "timesteps_total": args.stop_timesteps,
    "episode_reward_mean": args.stop_reward,
}

# config.log_level = "DEBUG"
# config.timesteps_per_iteration = 2
# config.preprocessor_pref = None
# config.train_batch_size = 128
# config._disable_preprocessor_api = True
# config._disable_action_flattening = True


# config.lr = 1e-3
# algo = config.build()

tuner = tune.Tuner(
    args.run,
    param_space=config.to_dict(),
    run_config=air.RunConfig(stop=stop, local_dir="./ray_results"),
)

results = tuner.fit()

ray.shutdown()
