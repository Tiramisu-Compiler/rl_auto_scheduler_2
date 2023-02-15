from data.data_service import DataSetService
from core.services.tiramisu_service import *
from scheduler.models.action import *
from scheduler.models.schedule import Schedule
from scheduler.services.scheduler_service import SchedulerService
import os


class TiramisuEnvAPIv1:
    def __init__(self):
        # The services of the environment 
        self.dataset_service : DataSetService = None
        self.scheduler_service : SchedulerService = SchedulerService()
        self.tiramisu_service : TiramisuService = TiramisuService()
        
        self.programs = None

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
        self.scheduler_service.set_schedule(schedule=schedule)

    def parallelize(self, loop_level: int):
        parallelization = Parallelization(params=[loop_level],
                                          name="Parallelization")
        return self.scheduler_service.apply_action(parallelization)

    def reverse(self, loop_level: int):
        reversal = Reversal(params=[loop_level], name="Reversal")
        return self.scheduler_service.apply_action(reversal)

    def interchange(self, loop1: int, loop2: int):
        interchange = Interchange(params=[loop1, loop2], name="Interchange")
        return self.scheduler_service.apply_action(interchange)