import argparse, ray
from ray import air, tune
from ray.rllib.models import ModelCatalog
from ray.rllib.utils.test_utils import check_learning_achieved
from ray.tune.logger import pretty_print
from ray.tune.registry import get_trainable_cls
import torch
from rl_agent.rl_env import TiramisuRlEnv
from ray.rllib.algorithms.ppo import PPOConfig
from rllib_ray_utils.custom_metrics_callback import CustomMetricCallback
from ray.rllib.algorithms.callbacks import MultiCallbacks
from env_api.tiramisu_api import TiramisuEnvAPI
from env_api.utils.config.config import Config

from rl_agent.rl_policy_nn import PolicyNN
from rllib_ray_utils.dataset_actor import DatasetActor

parser = argparse.ArgumentParser()
parser.add_argument(
        "--num-workers",
        default=-28,
        type=int,
        help="Number of workers to use for training",
    )
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
    "--as-test",
    action="store_true",
    help="Whether this script should be run as a test: --stop-reward must "
    "be achieved within --stop-timesteps AND --stop-iters.",
)
parser.add_argument("--stop-iters",
                    type=int,
                    default=500,
                    help="Number of iterations to train.")
parser.add_argument("--stop-timesteps",
                    type=int,
                    default=1_000_000,
                    help="Number of timesteps to train.")
parser.add_argument("--stop-reward",
                    type=float,
                    default=1,
                    help="Reward at which we stop training.")
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

    ray.init(address='auto') if args.num_workers > 28 else ray.init()

    Config.init()
    tiramisu_api = TiramisuEnvAPI(local_dataset = False)
    dataset_actor = DatasetActor.remote(
        dataset_path=Config.config.dataset.offline,
        use_dataset=True,
        path_to_save_dataset=Config.config.dataset.save_path,
        dataset_format="PICKLE",
    )

    ModelCatalog.register_custom_model("policy_nn", PolicyNN)
    config = (
        get_trainable_cls(args.run).get_default_config().environment(
            TiramisuRlEnv, env_config={
                "tiramisu_api": tiramisu_api,
                "dataset_actor": dataset_actor,
            }).framework(args.framework)
        # TODO : fix this callback
        .callbacks(MultiCallbacks([
        # CustomMetricCallback
        ])).rollouts(
            num_rollout_workers=args.num_workers-1,
            batch_mode="complete_episodes",
            enable_connectors=False).training(model={
                "custom_model": "policy_nn",
                "vf_share_layers": False,
            }).resources(num_gpus=0).debugging(log_level="WARN"))

    stop = {
        "training_iteration": args.stop_iters,
        "timesteps_total": args.stop_timesteps,
        "episode_reward_mean": args.stop_reward,
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
            if (result["timesteps_total"] >= args.stop_timesteps
                    or result["episode_reward_mean"] >= args.stop_reward):
                break
        algo.stop()
    else:
        # automated run with Tune and grid search and TensorBoard
        print("Training automatically with Ray Tune")
        try: 
            tuner = tune.Tuner.restore(
            path = "/scratch/dl5133/Dev/RL-Agent/tiramisu-env/ray_results/All-actions-new-exp2",
            
            resume_unfinished=True, resume_errored=True , restart_errored=True
            )
            # tuner = tune.Tuner(
            #     args.run,
            #     param_space=config.to_dict(),
            #     run_config=air.RunConfig(
            #         name="All-actions-new-exp2",
            #         stop=stop,
            #         local_dir=
            #         "/scratch/dl5133/Dev/RL-Agent/tiramisu-env/ray_results",
            #         checkpoint_config=air.CheckpointConfig(checkpoint_frequency=10,
            #                                             num_to_keep=3),
            #     ),
            # )
        except AssertionError as e :
            print(e)

        results = tuner.fit()

        if args.as_test:
            print("Checking if learning goals were achieved")
            check_learning_achieved(results, args.stop_reward)

    ray.shutdown()
