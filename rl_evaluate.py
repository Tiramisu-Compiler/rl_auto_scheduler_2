import argparse
import json
import os
import time
from pathlib import Path

import ray

from config.config import Config
from rllib_ray_utils.dataset_actor.dataset_actor import DatasetActor
from rllib_ray_utils.evaluators.ff_evaluator import FFBenchmarkEvaluator
from rllib_ray_utils.evaluators.lstm_evaluator import LSTMBenchmarkEvaluator

parser = argparse.ArgumentParser()

parser.add_argument(
    "--run", type=str, default="PPO", help="The RLlib-registered algorithm to use."
)
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

if __name__ == "__main__":
    args = parser.parse_args()
    print(f"Running with following CLI options: {args}")
    ray.init()

    Config.init()
    Config.config.dataset.is_benchmark = True
    dataset_actor = DatasetActor.remote(Config.config.dataset)
    dataset_size = ray.get(dataset_actor.get_dataset_size.remote())
    num_workers = args.num_workers
    if num_workers == -1:
        num_workers = int(ray.available_resources()["CPU"])

    # print(f"num workers: {num_workers}")

    num_programs_per_task = dataset_size // num_workers
    programs_remaining = dataset_size % num_workers

    Path(args.output_path).mkdir(parents=True, exist_ok=True)

    start_time = time.time()
    actors = []
    explorations = []
    explored_programs = {}

    if Config.config.experiment.policy_model == "lstm":
        Benchmarker = LSTMBenchmarkEvaluator
    elif Config.config.experiment.policy_model == "ff":
        Benchmarker = FFBenchmarkEvaluator
    else:
        raise Exception("Unknown policy model")

    for i in range(num_workers):
        num_programs_to_do = num_programs_per_task

        if i == num_workers - 1:
            num_programs_to_do += programs_remaining

        benchmark_actor = Benchmarker.remote(
            Config.config, args, num_programs_to_do, dataset_actor
        )

        actors.append(benchmark_actor)

        explorations.append(benchmark_actor.explore_benchmarks.remote())

    print(len(explorations))
    while len(explorations) > 0:
        # Wait for actors to finish their exploration
        done, explorations = ray.wait(explorations)
        print(f"Done this iteration: {len(done)} / Remaining {len(explorations)}")
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
