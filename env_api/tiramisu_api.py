import traceback
from env_api.data.data_service import DataSetService
from env_api.core.services.tiramisu_service import *
from env_api.scheduler.models.action import *
from env_api.scheduler.models.schedule import Schedule
from env_api.scheduler.services.scheduler_service import SchedulerService
import os
import torch

from config.config import Config


class TiramisuEnvAPI:
    def __init__(self, local_dataset=True):
        '''
        local_dataset : TiramisuEnvAPI has an internal dataset service to manage offline dataset stored in self.dataset_service
                        when this variable is False , it means we are using an external service to manage dataset or pure 
                        compilation.
        '''
        # Make the root to tiramisu root path explicit to the env in order to compile programs
        os.environ["TIRAMISU_ROOT"] = Config.config.tiramisu.tiramisu_path
        # The services of the environment
        self.scheduler_service: SchedulerService = SchedulerService()
        self.tiramisu_service: TiramisuService = TiramisuService()
        # Init database service with 2 paths :
        # - dataset_path : Contains the original program folder
        # - copy_path : Contains the path to store the chosen programs that are going to be optimized
        # This step of initializing the database service must be executed first in the init of tiramisu api
        self.dataset_service = DataSetService(
            dataset_path=Config.config.dataset.cpps_path,
            offline_path=Config.config.dataset.dataset_path if local_dataset else None)
        self.programs = None
        # The list of program names of the dataset
        self.programs = self.get_programs()

    # This method is used Outside of the RL env for independent testing , don't remove it in order 
    # to make tiramisu_api_tutorial work
    def get_programs(self):
        if self.programs == None:
            # If the offline dataset exists , get the program names from it
            if self.dataset_service.offline_dataset != None:
                self.programs = list(
                    self.dataset_service.offline_dataset.keys())
            # Else get them from the repository by calling system functions
            else:
                self.programs = os.listdir(self.dataset_service.dataset_path)
        return sorted(self.programs)

    def set_program(self, name: str, data: dict = None):
        print("Function : ", name)
        if data:
            tiramisu_prog = self.tiramisu_service.fetch_prog_offline(name=name,
                                                                     data=data)
        else:
            # Get the file path for the program with the given name
            file_path, exist_offline = self.dataset_service.get_file_path(name)
            # if exist_offline is True , then we can fetch the data from the offline dataset if the program name is saved there
            if (exist_offline):
                data = self.dataset_service.get_offline_prog_data(name=name)
                tiramisu_prog = self.tiramisu_service.fetch_prog_offline(
                    name=name, data=data)
            else:
                # Load the Tiramisu model from the file
                try:
                    tiramisu_prog = self.tiramisu_service.fetch_prog_compil(
                        path=file_path)
                except Exception as e:
                    if isinstance(e, LoopsDepthException):
                        print("Program has an unsupported loop level")
                    elif isinstance(e, NbAccessException):
                        print(
                            "Program has an unsupported number of access matrices"
                        )
                    print("Traceback of the error : " + 60 * "-")
                    print(traceback.print_exc())
                    print(80 * "-")
                    return None, None

        # Create a Schedule object for the Tiramisu model
        schedule = Schedule(tiramisu_prog)

        # Use the Scheduler service to set the schedule for the Tiramisu model
        comps_tensor, loops_tensor = self.scheduler_service.set_schedule(
            schedule_object=schedule)

        # Using the model to embed the program in a 180 sized vector
        with torch.no_grad():
            _, embedding_tensor = self.scheduler_service.prediction_service.get_speedup(
                comps_tensor, loops_tensor,
                self.scheduler_service.schedule_object)

        return embedding_tensor, self.scheduler_service.schedule_object.repr.action_mask

    # TODO : for all these actions we need to generalize over computations and not over shared iterators
    def parallelize(self, loop_level: int, env_id: int = None):
        # Create a Parallelization action with the given loop level
        parallelization = Parallelization(params=[loop_level], env_id=env_id)
        # Use the Scheduler service to apply the Parallelization action to the schedule
        return self.scheduler_service.apply_action(parallelization)

    def reverse(self, loop_level: int, env_id: int = None):
        # Create a Reversal action with given loop level
        reversal = Reversal(params=[loop_level], env_id=env_id)
        # Use the Scheduler service to apply the Reversal action to the schedule
        return self.scheduler_service.apply_action(reversal)

    def interchange(self,
                    loop_level1: int,
                    loop_level2: int,
                    env_id: int = None):
        # Create an Interchange action with given loop levels 1 and 2
        interchange = Interchange(params=[loop_level1, loop_level2],
                                  env_id=env_id)
        # Use the Scheduler service to apply the Interchange action to the schedule
        return self.scheduler_service.apply_action(interchange)

    def skew(self, loop_level1: int, loop_level2: int, env_id: int = None):
        # Create a skewing action for loop levels 1 and 2
        skewing = Skewing(params=[loop_level1, loop_level2], env_id=env_id)
        # Use the Scheduler to apply Skewing and return the speedup and legality
        return self.scheduler_service.apply_action(skewing)

    def fuse(self, loop_level: int, env_id: int = None):
        # Create a Fusion action with given loop level 1
        fusion = Fusion(params=[loop_level], env_id=env_id)
        # Use the Scheduler service to apply the Fusion action to the schedule
        return self.scheduler_service.apply_action(fusion)

    def tile2D(self,
               loop_level1: int,
               loop_level2: int,
               size_x: int,
               size_y: int,
               env_id: int = None):
        # Create a 2 dimensions Tiling action with given loop levels 1 and 2 , and 2D tile size (size_x,size_y)
        tiling2D = Tiling(params=[loop_level1, loop_level2, size_x, size_y],
                          env_id=env_id)
        # Use the Scheduler service to apply the Tiling action to the schedule
        return self.scheduler_service.apply_action(tiling2D)

    def tile3D(self,
               loop_level1: int,
               loop_level2: int,
               loop_level3: int,
               size_x: int,
               size_y: int,
               size_z: int,
               env_id: int = None):
        # Create a 3 dimensions Tiling action with given loop levels 1 , 2 and 3, and 3D tile size (size_x,size_y,size_z)
        tiling3D = Tiling(params=[
            loop_level1, loop_level2, loop_level3, size_x, size_y, size_z
        ],
            env_id=env_id)
        # Use the Scheduler service to apply the Tiling action to the schedule
        return self.scheduler_service.apply_action(tiling3D)

    def unroll(self, unrolling_factor: int, env_id: int = None):
        # Create an Unrolling action with given unrolling factor , the loop level is not given
        # because we suppose that unrollong innermost loop is more beneficial ,so this action is applied
        # on the innermost loop level
        unrolling = Unrolling(params=[unrolling_factor], env_id=env_id)
        # Use the Scheduler service to apply the Unrolling action to the schedule
        return self.scheduler_service.apply_action(unrolling)

    def save_legality_dataset(self, suffix: str = ""):
        self.dataset_service.store_offline_dataset(suffix=suffix)

    def get_schedules(self):
        return self.scheduler_service.schedule_object.prog.schedules_legality, self.scheduler_service.schedule_object.prog.schedules_solver
