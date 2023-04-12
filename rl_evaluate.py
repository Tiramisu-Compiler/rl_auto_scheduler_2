import argparse, ray
from ray.rllib.models import ModelCatalog
from rl_agent.rl_env import TiramisuRlEnv
from ray.rllib.algorithms.ppo import PPO, PPOConfig
from ray.rllib.algorithms.algorithm_config import AlgorithmConfig
from rllib_ray_utils.dataset_actor.dataset_actor import DatasetActor, DatasetFormat
from rllib_ray_utils.dataset_actor import DatasetActor
from config.config import Config
from rl_agent.rl_policy_nn import PolicyNN
from ray.air.checkpoint import Checkpoint

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
if __name__ == "__main__":
    args = parser.parse_args()
    print(f"Running with following CLI options: {args}")
    ray.init()

    Config.init()
    Config.config.dataset.is_benchmark = True
    dataset_actor = DatasetActor.remote(
        dataset_path=Config.config.dataset.benchmark_path,
        use_dataset=True,
        path_to_save_dataset=Config.config.dataset.save_path,
        dataset_format=DatasetFormat.PICKLE,
    )

    ModelCatalog.register_custom_model("policy_nn", PolicyNN)

    config = PPOConfig().framework(args.framework).environment(
        TiramisuRlEnv,
        env_config={
            "config": Config.config,
            "dataset_actor": dataset_actor
        }).rollouts(num_rollout_workers=1)
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

    checkpoint = Checkpoint.from_directory(Config.config.ray.restore_checkpoint)
    ppo_agent = PPO(AlgorithmConfig.from_dict(config))
    ppo_agent.restore(checkpoint_path=checkpoint)

    env = TiramisuRlEnv(config={
        "config": Config.config,
        "dataset_actor": dataset_actor
    })

    for i in range(31):
        observation, _ = env.reset()
        episode_done = False
        while not episode_done:
            action = ppo_agent.compute_single_action(observation=observation,
                                                     explore=False,policy_id="default_policy")
            observation, reward, episode_done, _, _ = env.step(action)
        else:
            print()

    ray.shutdown()