from env_api.core.models.optim_cmd import OptimizationCommand
from env_api.core.services.compiling_service import CompilingService
from env_api.core.services.converting_service import ConvertService
from env_api.scheduler.services.prediction_service import PredictionService
from ..models.schedule import Schedule
from ..models.action import *
import logging,numpy as np


class SchedulerService:
    def __init__(self):
        # An array that contains a list of optimizations that has been applied on the program
        # This list has objects of type `OptimizationCommand`
        self.schedule_list = []
        # The Schedule object contains all the informations of a program : annotatons , tree representation ...
        self.schedule_object: Schedule = None
        # The prediction service is an object that has a value estimator `get_speedup(schedule)` of the speedup that a schedule will have
        # This estimator is a recursive model that needs the schedule representation to give speedups
        self.prediction_service = PredictionService()

    def set_schedule(self,schedule_object : Schedule):
        '''
        The `set_schedule` function is called first in `tiramisu_api` to initialize the fields when a new program is fetched from the dataset.
        input : 
            - schedule_object : contains all the inforamtions on a program and the schedule 
        output : 
            - a tuple tensor that has the ready-to-use representaion that's going to represent the new optimized program (if any optim is applied) and serves as input to the cost and policy neural networks 
        '''
        self.schedule_object = schedule_object
        self.schedule_list = []
        return ConvertService.get_tree_representation(schedule_object)


    def get_annotations(self):
        '''
        output :
            - a dictionary containing the annotations of a program which is stored in `self.schedule_object.prog`
        '''
        return self.schedule_object.prog.annotations

    def get_schedule_dict(self):
        '''
        output : 
            - a dictionnary that contains the applied optimizations on a program in the form of tags
        '''
        return self.schedule_object.schedule_dict

    def apply_action(self, action: Action):
        '''
        input : 
            - an action that represents an optimization from the 7 types : Parallelization,Skewing,Interchange,Fusion,Reversal,Tiling,Unrolling
        output : 
            - speedup : float , representation : tuple(tensor) , legality_check : bool
        '''
        legality_check = (self.is_action_legal(action) == 1)
        tree_representation = ConvertService.get_tree_representation(self.schedule_object)
        speedup = 1.0
        if (legality_check):
            try:
                if isinstance(action, Parallelization):
                    self.apply_parallelization(loop_level=action.params[0])
                    self.schedule_object.is_parallelized = True
                    tree_representation = ConvertService.get_tree_representation(self.schedule_object)
                    speedup = self.prediction_service.get_speedup(tree_representation)
            except KeyError as e:
                logging.error(f"Key Error: {e}")
                legality_check = False        
        return speedup , tree_representation , legality_check

    def is_action_legal(self, action: Action):
        '''
        Checks the legality of action
        input : 
            - an action that represents an optimization from the 7 types : Parallelization,Skewing,Interchange,Fusion,Reversal,Tiling,Unrolling
        output : 
            - legality_check : int , if it is 1 it means it is legal
        '''
        optim_command = OptimizationCommand(action, self.schedule_object.comps)
        # Add the command to the array of schedule
        self.schedule_list.append(optim_command)
        # Check if the action is legal or no to be applied on self.schedule_object.prog
        try : 
            legality_check = int(CompilingService.compile_legality(schedule_object=self.schedule_object,optims_list=self.schedule_list))
        except ValueError as e :
            legality_check = 0
            print("Legality error :",e)
        if legality_check != 1:
            self.schedule_list.pop()
        return legality_check

    def apply_parallelization(self, loop_level):
        # Get any computation since we are using common iterators in a single root programs to apply action but #TODO : we need to fix this to support all cases
        computation = list(self.schedule_object.it_dict.keys())[0]
        # Getting the name of the iterator that points to the loop_level
        iterator = self.schedule_object.it_dict[computation][loop_level]['iterator']
        # Add the tag of parallelized loop level to the computations
        for comp in self.schedule_object.comps: 
            self.schedule_object.schedule_dict[comp]['parallelized_dim'] = iterator
        # Mask some actions after applying parallelization
        self.schedule_object.repr.action_mask[46] = 0
        self.schedule_object.repr.action_mask[47] = 0
        for i in range(56, 61):
            self.schedule_object.repr.action_mask[i] = 0
