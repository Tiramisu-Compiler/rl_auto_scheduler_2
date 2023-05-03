from typing import List
from env_api.core.models.optim_cmd import OptimizationCommand
from env_api.core.models.tiramisu_program import TiramisuProgram
from env_api.core.services.compiling_service import CompilingService
from env_api.core.services.converting_service import ConvertService
from config.config import Config
from env_api.scheduler.models.schedule import Schedule
from ..models.tags_cost_model import Model_Recursive_LSTM_v3
import torch

MAX_DEPTH = 6


class SpeedupService:
    def __init__(self):
        # This attribute is used to determine whether we are using the cpu speedup or the speedup from the cost model
        if Config.config.tiramisu.env_type == "cpu":
            self.is_cpu_speedup = True
        else:
            self.is_cpu_speedup = False
        # This model uses the tags instead of matrices
        self.tags_model = Model_Recursive_LSTM_v3(
            input_size=890, loops_tensor_size=8)
        # Loading the weights
        self.tags_model.load_state_dict(
            torch.load(Config.config.tiramisu.tags_model_weights,
                       map_location="cpu")
        )
        # Putting the model in evaluation mode to turn off Regularization layers.
        self.tags_model.eval()

    def get_speedup_embedded_tensor(self, comps_tensor, loops_tensor, schedule_object, optims_list: List[OptimizationCommand] = []):
        tree_tensor = ConvertService.get_tree_representation(
            comps_tensor, loops_tensor, schedule_object
        )
        with torch.no_grad():
            speedup, embedded_tensor = self.tags_model.forward(tree_tensor)
            speedup = speedup.item()

        if self.is_cpu_speedup:
            speedup = self.get_cpu_speedup(schedule_object.prog, optims_list)

        return speedup, embedded_tensor

    def get_cpu_speedup(self, tiramisu_program: TiramisuProgram, optims_list: List[OptimizationCommand] = []):
        schedule_exec_times = CompilingService.get_cpu_exec_times(
            tiramisu_program=tiramisu_program, optims_list=optims_list)

        # Get the minimum execution time from the list
        min_schedule_exec_time = min(schedule_exec_times)

        # Get the initial execution time
        initial_exec_time = tiramisu_program.current_machine_initial_execution_time
        # Compute the speedup

        # TODO Remove print
        print(f"Program: {tiramisu_program.name}\nOptims: {optims_list}\nInitial Execution Time: {initial_exec_time}\nMin Schedule Execution Time: {min_schedule_exec_time}\nSpeedup: {initial_exec_time / min_schedule_exec_time}")
        return initial_exec_time / min_schedule_exec_time
