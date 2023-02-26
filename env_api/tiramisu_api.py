from env_api.core.services.converting_service import ConvertService
from env_api.data.data_service import DataSetService
from env_api.core.services.tiramisu_service import *
from env_api.scheduler.models.action import *
from env_api.scheduler.models.schedule import Schedule
from env_api.scheduler.services.scheduler_service import SchedulerService
import os


class TiramisuEnvAPIv1:
    def __init__(self):
        # The services of the environment 
        self.dataset_service : DataSetService = None
        self.scheduler_service : SchedulerService = SchedulerService()
        self.tiramisu_service : TiramisuService = TiramisuService()
        # init database
        self.init_dataset_service(dataset_path = 'env_api/data/dataset/',copy_path = 'env_api/data/copy/')
        # a list of programs of the dataset 
        self.programs = os.listdir(self.dataset_service.dataset_path)
        

    def init_dataset_service(self,dataset_path : str,copy_path : str):
        self.dataset_service = DataSetService(dataset_path =dataset_path,copy_path =copy_path)

    def get_programs(self):
        if(self.programs == None ):
            self.programs = os.listdir(self.dataset_service.dataset_path)
        return self.programs

    def set_program(self,name : str):
        file_path = self.dataset_service.get_file_path(name)
        tiramisu_prog = self.tiramisu_service.get_tiramisu_model(path=file_path)  
        schedule =Schedule(tiramisu_prog)
        comps_tensor , loops_tensor =  self.scheduler_service.set_schedule(schedule_object=schedule)
        return ConvertService.get_tree_representation(comps_tensor , loops_tensor , schedule)

    def parallelize(self, loop_level: int):
        parallelization = Parallelization(params=[loop_level],
                                          name="Parallelization")
        return self.scheduler_service.apply_action(parallelization)
