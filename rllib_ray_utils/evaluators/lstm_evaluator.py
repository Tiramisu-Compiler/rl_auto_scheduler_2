# Benchmark actor is used to explore schedules for benchmarks in a distributed way
import os
from typing import List

import numpy as np
import ray
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.models import ModelCatalog

from config.config import AutoSchedulerConfig
from env_api.core.services.compiling_service import CompilingService
from rl_agent.rl_env import TiramisuRlEnv
from rl_agent.rl_policy_lstm import PolicyLSTM


@ray.remote
class LSTMBenchmarkEvaluator:
    def __init__(
        self,
        config: AutoSchedulerConfig,
        args: dict,
        programs_to_do: List[str],
    ):
        self.config = config
        self.programs_to_do = programs_to_do
        self.args = args
        self.env = TiramisuRlEnv(config={"config": config})

        ModelCatalog.register_custom_model("policy_nn", PolicyLSTM)
        self.model_custom_config = config.lstm_policy.__dict__

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

        restored_tuner = tune.Tuner.restore(config.ray.restore_checkpoint)
        result_grid = restored_tuner.get_results()
        best_result = result_grid.get_best_result("episode_reward_mean", "max")
        best_result
        best_checkpoint = None
        highest_reward = float("-inf")
        for checkpoint in best_result.best_checkpoints:
            episode_reward_mean = checkpoint[1]["episode_reward_mean"]
            if episode_reward_mean > highest_reward:
                highest_reward = episode_reward_mean
                best_checkpoint = checkpoint

        # Build the Algorithm instance using the config.
        # Restore the algo's state from the checkpoint.
        self.algo = self.config_model.build()
        self.algo.restore(best_checkpoint[0])
        self.num_programs_done = 0

    # explore schedules for benchmarks

    def explore_benchmarks(self):
        lstm_cell_size = self.model_custom_config["lstm_state_size"]
        init_state = state = [np.zeros([lstm_cell_size], np.float32) for _ in range(2)]
        # store explored programs and their schedules
        explored_programs = {}

        # explore schedules for each program
        for function_name in self.programs_to_do:
            observation, _ = self.env.reset(options={"function_name": function_name})
            print(
                f"Running program {self.env.current_program}, num programs done: {self.num_programs_done} / {len(self.programs_to_do)}"
            )

            episode_done = False
            state = init_state
            # explore schedule for current program
            while not episode_done:
                # get action from policy
                action, state_out, _ = self.algo.compute_single_action(
                    observation=observation, explore=False, state=state
                )
                # take action in environment and get new observation
                observation, reward, episode_done, _, _ = self.env.step(action)
                state = state_out

            (
                speedup,
                sched_str,
            ) = self.env.tiramisu_api.scheduler_service.get_current_speedup()
            # when episode is done, write cpp code to file
            tiramisu_prog = self.env.tiramisu_api.scheduler_service.schedule_object.prog
            optim_list = (
                self.env.tiramisu_api.scheduler_service.schedule_object.schedule_list
            )
            branches = self.env.tiramisu_api.scheduler_service.branches
            schedule_object = self.env.tiramisu_api.scheduler_service.schedule_object
            print(f"[Done] {tiramisu_prog.name} : {optim_list}")

            cpp_code = CompilingService.get_schedule_code(
                tiramisu_program=tiramisu_prog,
                optims_list=optim_list,
                branches=branches,
            )

            legality_code = CompilingService.get_legality_code(
                schedule_object=schedule_object,
                optims_list=optim_list,
                branches=branches,
            )
            CompilingService.write_cpp_code(
                cpp_code, os.path.join(self.args.output_path, self.env.current_program)
            )

            CompilingService.write_cpp_code(
                legality_code,
                os.path.join(
                    self.args.output_path, self.env.current_program + "_legality"
                ),
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
