import argparse
import ray
from ray import air, tune
from ray.rllib.models import ModelCatalog
from ray.rllib.utils.test_utils import check_learning_achieved
from ray.tune.logger import pretty_print
from ray.tune.registry import get_trainable_cls
from rl_agent.rl_env import TiramisuRlEnv
from ray.rllib.algorithms.callbacks import MultiCallbacks
from config.config import Config

from rl_agent.rl_policy_nn import PolicyNN
from rl_agent.rl_policy_lstm import PolicyLSTM
from rllib_ray_utils.dataset_actor.dataset_actor import DatasetActor

from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.algorithms.pg import PGConfig

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
    "--run", type=str, default="PPO", help="The RLlib-registered algorithm to use."
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
    args = parser.parse_args()
    print(f"Running with following CLI options: {args}")
    # If num workers > 28 => means we are using more than 1 node.
    ray.init(address="auto") if args.num_workers > 208 else ray.init()
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
        get_trainable_cls(args.run)
        .get_default_config()
        .environment(
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
            vf_loss_coeff=1,
            sgd_minibatch_size=128,
            train_batch_size=4096,
            model={
                "custom_model": "policy_nn",
                "vf_share_layers": Config.config.experiment.vf_share_layers,
                "custom_model_config": model_custom_config,
            },
        )
        # .exploration(
        #     exploration_config={"type": "EpsilonGreedy"},
        # )
        .resources(num_gpus=args.num_gpus)
        .debugging(log_level="WARN")
    )

    # Setting the stop conditions
    stop = {
        "training_iteration": Config.config.experiment.training_iteration,
        "timesteps_total": Config.config.experiment.timesteps_total,
        "episode_reward_mean": Config.config.experiment.episode_reward_mean,
    }

    if args.no_tune:
        # manual training with train loop using PPO and fixed learning rate
        if args.run != "PPO":
            raise ValueError("Only support --run PPO with --no-tune.")
        print("Running manual train loop without Ray Tune.")
        # use fixed learning rate instead of grid search (needs tune)
        config.lr = 1e-3
        algo = config.build()
        # run manual training loop and print results after each iteration
        for _ in range(args.stop_iters):
            result = algo.train()
            print(pretty_print(result))
            # stop training of the target train steps or reward are reached
            if (
                result["timesteps_total"] >= args.stop_timesteps
                or result["episode_reward_mean"] >= args.stop_reward
            ):
                break
        algo.stop()
    else:
        print("Training automatically with Ray Tune")
        try:
            tuner = tune.Tuner(
                args.run,
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
