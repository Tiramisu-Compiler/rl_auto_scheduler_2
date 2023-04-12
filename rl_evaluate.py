import argparse
import os
from pathlib import Path
import time
import ray
import random
from ray.rllib.models import ModelCatalog
from env_api.core.services.compiling_service import CompilingService
from rl_agent.rl_env import TiramisuRlEnv
from ray.rllib.algorithms.ppo import PPO, PPOConfig
from ray.rllib.algorithms.algorithm_config import AlgorithmConfig
from ray.rllib.policy.policy import Policy
from config.config import AutoSchedulerConfig, Config
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
parser.add_argument("--num-workers", default=-1, type=int)


@ray.remote
class BenchmarkActor:
    def __init__(self, config: AutoSchedulerConfig, args: dict, num_programs_to_do: int):

        self.config = config
        self.num_programs_to_do = num_programs_to_do

        self.env = TiramisuRlEnv(config={
            "config": config,
            "dataset_actor": dataset_actor
        })

        self.config = get_trainable_cls(args.run).get_default_config().environment(
            TiramisuRlEnv,
            env_config={
                "config": config,
                "dataset_actor": dataset_actor,
            }).framework(args.framework).rollouts(
            num_rollout_workers=0,
            batch_mode="complete_episodes",
            enable_connectors=False).training(
            lr=config.policy_network.lr,
            model={
                "custom_model": "policy_nn",
                "vf_share_layers": config.policy_network.vf_share_layers,
                "custom_model_config": {
                                "policy_hidden_layers": config.policy_network.policy_hidden_layers,
                                "vf_hidden_layers": config.policy_network.vf_hidden_layers,
                                "dropout_rate": config.policy_network.dropout_rate
                }
            }).resources(num_gpus=0).debugging(log_level="WARN")
        # Build the Algorithm instance using the config.
        # Restore the algo's state from the checkpoint.
        self.algo = self.config.build()
        self.algo.restore(config.ray.restore_checkpoint)
        self.num_programs_done = 0

    def explore_benchmarks(self):
        for i in range(self.num_programs_to_do):
            print(
                f"Running program {self.env.current_program}, num programs done: {self.num_programs_done} / {self.num_programs_to_do}")
            observation, _ = self.env.reset()
            episode_done = False

            while not episode_done:
                action = self.algo.compute_single_action(
                    observation=observation, explore=False)
                observation, reward, episode_done, _, _ = self.env.step(action)
            cpp_code = CompilingService.get_schedule_code(
                self.env.tiramisu_api.scheduler_service.schedule_object, self.env.tiramisu_api.scheduler_service.schedule_list)
            CompilingService.write_cpp_code(cpp_code, os.path.join(
                args.output_path, self.env.current_program))
            self.num_programs_done += 1

    def get_progress(self) -> float:
        return self.num_programs_done


if __name__ == "__main__":
    args = parser.parse_args()
    print(f"Running with following CLI options: {args}")
    ray.init()

    Config.init()
    Config.config.dataset.is_benchmark = True
    dataset_actor = DatasetActor.remote(Config.config.dataset)

    ModelCatalog.register_custom_model("policy_nn", PolicyNN)
    dataset_size = ray.get(dataset_actor.get_dataset_size.remote())

    num_workers = args.num_workers
    if (num_workers == -1):
        num_workers = int(ray.available_resources()['CPU'])

    # print(f"num workers: {num_workers}")

    num_programs_per_task = dataset_size // num_workers
    programs_remaining = dataset_size % num_workers

    Path(args.output_path).mkdir(parents=True, exist_ok=True)

    start_time = time.time()
    actors = []
    for i in range(num_workers):
        num_programs_to_do = num_programs_per_task

        if i == num_workers-1:
            num_programs_to_do += programs_remaining

        benchmark_actor = BenchmarkActor.remote(
            Config.config, args, num_programs_to_do)

        actors.append(benchmark_actor)

        benchmark_actor.explore_benchmarks.remote()
    
    while True:
        time.sleep(1)
        progress = ray.get([actor.get_progress.remote() for actor in actors])
        print(f"Progress: {sum(progress)} / {dataset_size}")
        if sum(progress) == dataset_size:
            break
    
    end_time = time.time()
    print(f"Total time: {end_time - start_time}")
