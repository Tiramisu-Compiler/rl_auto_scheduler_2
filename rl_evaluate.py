import argparse
import json
import os
import time
from pathlib import Path

import grpc
import ray

from config.config import Config
from grpc_server.dataset_grpc_server.grpc_files import (
    tiramisu_function_pb2,
    tiramisu_function_pb2_grpc,
)
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
)
parser.add_argument("--num-workers", default=-1, type=int)

if __name__ == "__main__":
    args = parser.parse_args()
    print(f"Running with following CLI options: {args}")
    ray.init()

    Config.init()

    # check if args.output_path exists and create it if it doesn't
    output_path = Path(args.output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    # Empty the output directory
    for file in output_path.iterdir():
        file.unlink()

    # read the ip and port from the server_address file
    ip_and_port = ""
    while ip_and_port == "":
        with open("./server_address", "r") as f:
            ip_and_port = f.read()
    
    ip_and_port = ip_and_port.splitlines()[0]

    with grpc.insecure_channel(ip_and_port) as channel:
        stub = tiramisu_function_pb2_grpc.TiramisuDataServerStub(channel)
        response = stub.GetDatasetSize(tiramisu_function_pb2.Empty())
        dataset_size = response.size
        response = stub.GetListOfFunctions(tiramisu_function_pb2.Empty())
        function_names = response.names

    num_workers = args.num_workers
    if num_workers == -1:
        num_workers = int(ray.available_resources()["CPU"])

    print(f"num workers: {num_workers}")
    print(f"dataset size: {dataset_size}")

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

        start = i * num_programs_per_task
        end = start + num_programs_to_do
        benchmark_actor = Benchmarker.remote(
            Config.config,
            args,
            function_names[start:end],
        )

        actors.append(benchmark_actor)

        explorations.append(benchmark_actor.explore_benchmarks.remote())

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
