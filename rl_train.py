import argparse
import ray
import os
from ray import air, tune
from ray.rllib.models import ModelCatalog
from ray.rllib.utils.test_utils import check_learning_achieved
from ray.tune.logger import pretty_print
from rl_agent.rl_env import TiramisuRlEnv
from ray.rllib.algorithms.callbacks import MultiCallbacks
from config.config import Config

from rl_agent.rl_policy_nn import PolicyNN
from rl_agent.rl_policy_lstm import PolicyLSTM
from rllib_ray_utils.dataset_actor.dataset_actor import DatasetActor

from ray.rllib.algorithms.ppo import PPOConfig

parser = argparse.ArgumentParser()
parser.add_argument(
    "--num-workers",
    default=28,
    type=int,
    help="Number of workers to use for training",
)

parser.add_argument(
    "--num-gpus",
    default=0,
    type=int,
    help="Number of gpus",
)

parser.add_argument(
    "--framework",
    choices=["tf", "tf2", "torch"],
    default="torch",
    help="The DL framework specifier.",
)
parser.add_argument(
    "--as-test",
    action="store_true",
    help="Whether this script should be run as a test: --stop-reward must "
    "be achieved within --stop-timesteps AND --stop-iters.",
)

parser.add_argument(
    "--no-tune",
    default=False,
    action="store_true",
    help="Run without Tune using a manual train loop instead. In this case,"
    "use PPO without grid search and no TensorBoard.",
)
parser.add_argument(
    "--local-mode",
    default=False,
    action="store_true",
    help="Init Ray in local mode for easier debugging.",
)

if __name__ == "__main__":
    num_cpus = os.cpu_count()
    args = parser.parse_args()
    print(f"Running with following CLI options: {args}")
    # If num workers > 28 => means we are using more than 1 node.
    ray.init(address="auto") if args.num_workers >= num_cpus else ray.init()
    # Config.init() is necessary to load all env variables
    Config.init()
    # DatasetActor is the responsible class of syncronizing data between rollout-workers, TiramisuEnvAPI will read
    # data from this actor.
    dataset_actor = DatasetActor.remote(Config.config.dataset)

    match (Config.config.experiment.policy_model):
        case "lstm":
            ModelCatalog.register_custom_model("policy_nn", PolicyLSTM)
            model_custom_config = Config.config.lstm_policy.__dict__
        case "ff":
            ModelCatalog.register_custom_model("policy_nn", PolicyNN)
            model_custom_config = Config.config.policy_network.__dict__
    config = (
        PPOConfig().environment(
            TiramisuRlEnv,
            env_config={
                "config": Config.config,
                "dataset_actor": dataset_actor,
            },
        )
        .framework(args.framework)
        .callbacks(
            MultiCallbacks(
                [
                    # CustomMetricCallback
                ]
            )
        )
        .rollouts(
            num_rollout_workers=args.num_workers,
            batch_mode="complete_episodes",
            enable_connectors=False,
        )
        .training(
            lr=Config.config.experiment.lr,
            entropy_coeff=Config.config.experiment.entropy_coeff,
            vf_loss_coeff= Config.config.experiment.vf_loss_coeff,
            sgd_minibatch_size=Config.config.experiment.minibatch_size,
            train_batch_size=Config.config.experiment.train_batch_size,
            model={
                "custom_model": "policy_nn",
                "vf_share_layers": Config.config.experiment.vf_share_layers,
                "custom_model_config": model_custom_config,
            },
        ).resources(
            num_gpus=args.num_gpus
            # To train with execution on separate nodes
            # num_cpus_per_worker= Num of cpu in each node,
            # placement_strategy= "STRICT_SPREAD"
            )
        .debugging(log_level="WARN")
    )

    # Setting the stop conditions
    stop = {
        "training_iteration": Config.config.experiment.training_iteration,
        "timesteps_total": Config.config.experiment.timesteps_total,
        "episode_reward_mean": Config.config.experiment.episode_reward_mean,
    }

    if args.no_tune:
        print("Running manual train loop without Ray Tune.")
        # use fixed learning rate instead of grid search (needs tune)
        algo = config.build()
        # run manual training loop and print results after each iteration
        for _ in range(stop["training_iteration"]):
            result = algo.train()
            print(pretty_print(result))
            # stop training of the target train steps or reward are reached
            if (
                result["timesteps_total"] >= stop["timesteps_total"]
                or result["episode_reward_mean"] >= stop["episode_reward_mean"]
            ):
                break
        algo.stop()
    else:
        print("Training automatically with Ray Tune")
        try:
            tuner = tune.Tuner(
                "PPO",
                param_space=config.to_dict(),
                run_config=air.RunConfig(
                    name=Config.config.experiment.name,
                    stop=stop,
                    local_dir=Config.config.ray.results,
                    checkpoint_config=air.CheckpointConfig(
                        checkpoint_frequency=Config.config.experiment.checkpoint_frequency,
                        num_to_keep=Config.config.experiment.checkpoint_num_to_keep,
                        checkpoint_at_end=True,
                    ),
                    failure_config=air.FailureConfig(fail_fast=True),
                ),
            )

        except AssertionError as e:
            print(e)

        results = tuner.fit()

        if args.as_test:
            print("Checking if learning goals were achieved")
            check_learning_achieved(results, args.stop_reward)

    ray.shutdown()
