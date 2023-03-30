import argparse, ray
from ray.rllib.models import ModelCatalog
from rl_agent.rl_env import TiramisuRlEnv
from ray.rllib.algorithms.ppo import PPO, PPOConfig
from ray.rllib.algorithms.algorithm_config import AlgorithmConfig
from env_api.tiramisu_api import TiramisuEnvAPI
from rllib_ray_utils.dataset_actor import DatasetActor
from env_api.utils.config.config import Config

from rl_agent.rl_policy_nn import PolicyNN

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
    ray.init(num_cpus=28)
    
    Config.init()
    tiramisu_api = TiramisuEnvAPI(local_dataset = False)
    dataset_actor = DatasetActor.remote(
        dataset_path=Config.config.dataset.offline,
        use_dataset=True,
        path_to_save_dataset=Config.config.dataset.save_path,
        dataset_format="PICKLE",
    )
    
    ModelCatalog.register_custom_model("policy_nn", PolicyNN)
    config = PPOConfig().framework(args.framework).environment(
        TiramisuRlEnv, env_config={"tiramisu_api": tiramisu_api,
                                   "dataset_actor": dataset_actor})
    config = config.to_dict()
    config["model"]["custom_model"] = "policy_nn"

    ppo_agent = PPO(AlgorithmConfig.from_dict(config))
    try : 
        ppo_agent.restore(
            checkpoint_path=
            "/scratch/dl5133/Dev/RL-Agent/tiramisu-env/ray_results/All-actions-punish-legality-beam-search-10m/PPO_TiramisuRlEnv_16641_00000_0_2023-03-27_17-12-56/checkpoint_000930"
        )
    except AssertionError as e :
        print(e)

    # env = ppo_agent.env_creator(config["env"])
    env = TiramisuRlEnv(config={"tiramisu_api": tiramisu_api,"dataset_actor": dataset_actor})

    for i in range(31):
        observation , _ = env.reset()
        episode_done = False
        while not episode_done :
            action = ppo_agent.compute_single_action(observation=observation, explore=False)
            observation, reward, episode_done, _ , _ = env.step(action)

    ray.shutdown()