from typing import List

import torch

from config.config import Config
from env_api.core.services.compiling_service import CompilingService
from env_api.core.services.converting_service import ConvertService
from env_api.scheduler.models.branch import Branch
from env_api.scheduler.models.schedule import Schedule
from env_api.utils.exceptions import ExecutingFunctionException

from ..models.multi_root_model import Model_Recursive_LSTM_v2

MAX_DEPTH = 6


class PredictionService:
    def __init__(self):
        # This model uses the tags instead of matrices
        self.tags_model = Model_Recursive_LSTM_v2()
        # Loading the weights
        self.tags_model.load_state_dict(
            torch.load(Config.config.tiramisu.tags_model_weights, map_location="cpu")
        )
        # Putting the model in evaluation mode to turn off Regularization layers.
        self.tags_model.eval()

    def get_predicted_speedup(
        self, comps_tensor, loops_tensor, expr_tensor, schedule_object
    ):
        tree_tensor = ConvertService.get_tree_representation(
            comps_tensor, loops_tensor, expr_tensor, schedule_object
        )
        with torch.no_grad():
            speedup, embedded_tensor = self.tags_model.forward(tree_tensor)
            return speedup.item(), embedded_tensor

    def get_real_speedup(self, schedule_object: Schedule, branches: List[Branch]):
        if "initial_execution" in schedule_object.prog.execution_times:
            # Original execution time of the program already exists in the dataset so we read the value directly
            initial_execution = schedule_object.prog.execution_times[
                "initial_execution"
            ]
        else:
            # We need to run the program to get the value
            initial_execution = CompilingService.execute_code(
                tiramisu_program=schedule_object.prog, optims_list=[], branches=branches
            )
            if initial_execution:
                schedule_object.prog.execution_times[
                    "initial_execution"
                ] = initial_execution
            else:
                raise ExecutingFunctionException

        if schedule_object.schedule_str in schedule_object.prog.execution_times:
            schedule_execution = schedule_object.prog.execution_times[
                schedule_object.schedule_str
            ]

        else:
            # We need to run the program to get the value
            schedule_execution = CompilingService.execute_code(
                tiramisu_program=schedule_object.prog,
                optims_list=schedule_object.schedule_list,
                branches=branches,
            )
            if schedule_execution:
                schedule_object.prog.execution_times[
                    schedule_object.schedule_str
                ] = schedule_execution
            else:
                raise ExecutingFunctionException

        return initial_execution / schedule_execution
