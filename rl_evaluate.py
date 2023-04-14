import argparse
import json
import os
from pathlib import Path
import time
import ray
import random
from ray.rllib.models import ModelCatalog
from env_api.core.services.compiling_service import CompilingService
from env_api.core.services.converting_service import ConvertService
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
    default="./workspace/schedules",
    help="The DL framework specifier.",
)
parser.add_argument("--num-workers", default=-1, type=int)


# Benchmark actor is used to explore schedules for benchmarks in a distributed way
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

    # explore schedules for benchmarks
    def explore_benchmarks(self):
        # store explored programs and their schedules
        explored_programs = {}

        # explore schedules for each program
        for i in range(self.num_programs_to_do):
            observation, _ = self.env.reset()
            print(
                f"Running program {self.env.current_program}, num programs done: {self.num_programs_done} / {self.num_programs_to_do}")

            episode_done = False

            # explore schedule for current program
            while not episode_done:
                # get action from policy
                action = self.algo.compute_single_action(
                    observation=observation, explore=False)
                # take action in environment and get new observation
                observation, reward, episode_done, _, _ = self.env.step(action)

            # when episode is done, write cpp code to file
            cpp_code = CompilingService.get_schedule_code(
                self.env.tiramisu_api.scheduler_service.schedule_object, self.env.tiramisu_api.scheduler_service.schedule_list)
            CompilingService.write_cpp_code(cpp_code, os.path.join(
                args.output_path, self.env.current_program))

            # store explored program and its schedule
            explored_programs[self.env.current_program] = {
                "schedule": ConvertService.build_sched_string(self.env.tiramisu_api.scheduler_service.schedule_list)
            }
            self.num_programs_done += 1

        return explored_programs

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
    explorations = []
    explored_programs = {}
    for i in range(num_workers):
        num_programs_to_do = num_programs_per_task

        if i == num_workers-1:
            num_programs_to_do += programs_remaining

        benchmark_actor = BenchmarkActor.remote(
            Config.config, args, num_programs_to_do)

        actors.append(benchmark_actor)

        explorations.append(benchmark_actor.explore_benchmarks.remote())

    print(len(explorations))
    while len(explorations) > 0:
        # Wait for actors to finish their exploration
        done, explorations = ray.wait(explorations)
        print(
            f"Done this iteration: {len(done)} / Remaining {len(explorations)}")
        # retrieve explored programs from actors that finished their exploration
        for actor in done:
            actor_programs = ray.get(actor)
            explored_programs.update(actor_programs)

        progress = ray.get([actor.get_progress.remote() for actor in actors])
        print(f"Progress: {sum(progress)} / {dataset_size}")

    end_time = time.time()
    print(f"Total time: {end_time - start_time}")

    # write explored programs to file
    with open(os.path.join(args.output_path, "explored_programs.json"), "w") as f:
        json.dump(explored_programs, f)
