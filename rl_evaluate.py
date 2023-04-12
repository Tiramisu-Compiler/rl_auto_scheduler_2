import argparse
import os
import ray
import random
from ray.rllib.models import ModelCatalog
from env_api.core.services.compiling_service import CompilingService
from rl_agent.rl_env import TiramisuRlEnv
from ray.rllib.algorithms.ppo import PPO, PPOConfig
from ray.rllib.algorithms.algorithm_config import AlgorithmConfig
from ray.rllib.policy.policy import Policy
from config.config import Config
from rl_agent.rl_policy_nn import PolicyNN
from ray.rllib.algorithms.algorithm import Algorithm
from ray.air.checkpoint import Checkpoint
from ray.tune.registry import get_trainable_cls

from rllib_ray_utils.dataset_actor.dataset_actor import DatasetActor

parser = argparse.ArgumentParser()

parser.add_argument("--run",
                    type=str,
                    default="PPO",
                    help="The RLlib-registered algorithm to use.")
parser.add_argument(
    "--framework",
    choices=["tf", "tf2", "torch"],
    default="torch",
    help="The DL framework specifier.",
)
parser.add_argument(
    "--output-path",
    default="./workspace/",
    help="The DL framework specifier.",
)

if __name__ == "__main__":
    args = parser.parse_args()
    print(f"Running with following CLI options: {args}")
    ray.init()

    Config.init()
    Config.config.dataset.is_benchmark = True
    dataset_actor = DatasetActor.remote(Config.config.dataset)

    ModelCatalog.register_custom_model("policy_nn", PolicyNN)

    config = PPOConfig().framework(args.framework).environment(
        TiramisuRlEnv,
        env_config={
            "config": Config.config,
            "dataset_actor": dataset_actor
        }).rollouts(num_rollout_workers=0)
    config.explore = False

    config = config.to_dict()
    config["model"] = {
        "custom_model": "policy_nn",
        "vf_share_layers": Config.config.policy_network.vf_share_layers,
        "custom_model_config": {
            "policy_hidden_layers":
            Config.config.policy_network.policy_hidden_layers,
            "vf_hidden_layers": Config.config.policy_network.vf_hidden_layers,
            "dropout_rate": Config.config.policy_network.dropout_rate
        }
    }

    env = TiramisuRlEnv(config={
        "config": Config.config,
        "dataset_actor": dataset_actor
    })

    config = get_trainable_cls(args.run).get_default_config().environment(
        TiramisuRlEnv,
        env_config={
            "config": Config.config,
            "dataset_actor": dataset_actor,
        }).framework(args.framework).rollouts(
        num_rollout_workers=0,
        batch_mode="complete_episodes",
        enable_connectors=False).training(
        lr=Config.config.policy_network.lr,
        model={
            "custom_model": "policy_nn",
            "vf_share_layers": Config.config.policy_network.vf_share_layers,
            "custom_model_config": {
                            "policy_hidden_layers": Config.config.policy_network.policy_hidden_layers,
                            "vf_hidden_layers": Config.config.policy_network.vf_hidden_layers,
                            "dropout_rate": Config.config.policy_network.dropout_rate
            }
        }).resources(num_gpus=0).debugging(log_level="WARN")
    # Build the Algorithm instance using the config.
    # Restore the algo's state from the checkpoint.
    algo = config.build()
    algo.restore('checkpoints/checkpoint_000234')

    # trained_policy = Policy.from_checkpoint(
    #     "/scratch/sk10691/workspace/rl/benchmarks_rl/checkpoints/checkpoint_000234/policies/default_policy")
    # print(trained_policy.config)
    dataset_size = ray.get(dataset_actor.get_dataset_size.remote())
    for i in range(dataset_size):
        print(
            f"Running program {i} of {dataset_size}: {env.current_program}")
        observation, _ = env.reset()
        episode_done = False

        while not episode_done:
            action = algo.compute_single_action(
                observation=observation, explore=False)
            observation, reward, episode_done, _, _ = env.step(action)
        cpp_code = CompilingService.get_schedule_code(
            env.tiramisu_api.scheduler_service.schedule_object, env.tiramisu_api.scheduler_service.schedule_list)
        CompilingService.write_cpp_code(cpp_code, os.path.join(
            args.output_path, env.current_program))
