# Benchmark actor is used to explore schedules for benchmarks in a distributed way
import os

import ray
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.models import ModelCatalog

from config.config import AutoSchedulerConfig
from env_api.core.services.compiling_service import CompilingService
from rl_agent.rl_env import TiramisuRlEnv
from rl_agent.rl_policy_nn import PolicyNN


@ray.remote
class FFBenchmarkEvaluator:
    def __init__(
        self,
        config: AutoSchedulerConfig,
        args: dict,
        num_programs_to_do: int,
    ):
        self.config = config
        self.num_programs_to_do = num_programs_to_do
        self.args = args
        self.env = TiramisuRlEnv(config={"config": config})

        ModelCatalog.register_custom_model("policy_nn", PolicyNN)
        self.model_custom_config = config.policy_network.__dict__

        self.config_model = (
            PPOConfig()
            .framework(args.framework)
            .environment(
                TiramisuRlEnv,
                env_config={
                    "config": config,
                },
            )
            .framework(args.framework)
            .rollouts(
                num_rollout_workers=0,
                batch_mode="complete_episodes",
                enable_connectors=False,
            )
            .training(
                model={
                    "custom_model": "policy_nn",
                    "vf_share_layers": config.experiment.vf_share_layers,
                    "custom_model_config": self.model_custom_config,
                }
            )
            .resources(num_gpus=0)
            .debugging(log_level="WARN")
        )
        # Build the Algorithm instance using the config.
        # Restore the algo's state from the checkpoint.
        self.algo = self.config_model.build()
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
                f"Running program {self.env.current_program}, num programs done: {self.num_programs_done} / {self.num_programs_to_do}"
            )

            episode_done = False
            # explore schedule for current program
            while not episode_done:
                # get action from policy
                action = self.algo.compute_single_action(
                    observation=observation, explore=False
                )
                # take action in environment and get new observation
                observation, reward, episode_done, _, _ = self.env.step(action)

            (
                speedup,
                sched_str,
            ) = self.env.tiramisu_api.scheduler_service.get_current_speedup()
            tiramisu_prog = self.env.tiramisu_api.scheduler_service.schedule_object.prog
            optim_list = (
                self.env.tiramisu_api.scheduler_service.schedule_object.schedule_list
            )
            branches = self.env.tiramisu_api.scheduler_service.branches
            # when episode is done, write cpp code to file
            cpp_code = CompilingService.get_schedule_code(
                tiramisu_program=tiramisu_prog,
                optims_list=optim_list,
                branches=branches,
            )
            CompilingService.write_cpp_code(
                cpp_code, os.path.join(self.args.output_path, self.env.current_program)
            )

            # store explored program and its schedule
            explored_programs[self.env.current_program] = {
                "schedule": sched_str,
                "speedup_model": speedup,
            }
            self.num_programs_done += 1

        return explored_programs

    def get_progress(self) -> float:
        return self.num_programs_done
