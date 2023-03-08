from env_api.core.models.optim_cmd import OptimizationCommand
from env_api.core.services.compiling_service import CompilingService
from env_api.core.services.converting_service import ConvertService
from env_api.scheduler.services.prediction_service import PredictionService
from ..models.schedule import Schedule
from ..models.action import *
import logging


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

    def set_schedule(self, schedule_object: Schedule):
        """
        The `set_schedule` function is called first in `tiramisu_api` to initialize the fields when a new program is fetched from the dataset.
        input :
            - schedule_object : contains all the inforamtions on a program and the schedule
        output :
            - a tuple tensor that has the ready-to-use representaion that's going to represent the new optimized program (if any optim is applied) and serves as input to the cost and policy neural networks
        """
        self.schedule_object = schedule_object
        self.schedule_list = []
        return ConvertService.get_schedule_representation(schedule_object)

    def get_annotations(self):
        """
        output :
            - a dictionary containing the annotations of a program which is stored in `self.schedule_object.prog`
        """
        return self.schedule_object.prog.annotations

    def get_tree_tensor(self):
        repr_tensors = ConvertService.get_schedule_representation(self.schedule_object)
        return ConvertService.get_tree_representation(
            *repr_tensors, self.schedule_object
        )

    def get_schedule_dict(self):
        """
        output :
            - a dictionnary that contains the applied optimizations on a program in the form of tags
        """
        return self.schedule_object.schedule_dict

    def apply_action(self, action: Action):
        """
        input :
            - an action that represents an optimization from the 7 types : Parallelization,Skewing,Interchange,Fusion,Reversal,Tiling,Unrolling
        output :
            - speedup : float , representation : tuple(tensor) , legality_check : bool
        """
        # prog.schedules only has data when it is fetched from the offline dataset so no need to compile to get the legality
        # TODO : the data available is just for the parallelization action 
        if(self.schedule_object.prog.schedules):
            legality_check = self.schedule_object.prog.schedules['comp00P(L0)']
        else :
            legality_check = self.is_action_legal(action) == 1
        embedding_tensor = None
        speedup = 1.0
        if legality_check:
            try:
                if isinstance(action, Parallelization):
                    self.apply_parallelization(loop_level=action.params[0])
                    self.schedule_object.is_parallelized = True

                elif isinstance(action, Reversal):
                    self.apply_reversal(loop_level=action.params[0])
                    self.schedule_object.is_reversed = True
                
                elif isinstance(action,Interchange):
                    self.apply_interchange(loop_level1=action.params[0],loop_level2=action.params[1])
                    self.schedule_object.is_interchaged = True
                
                repr_tensors = ConvertService.get_schedule_representation(
                    self.schedule_object
                )
                speedup, embedding_tensor = self.prediction_service.get_speedup(
                    *repr_tensors, self.schedule_object
                )
            except KeyError as e:
                logging.error(f"This loop level: {e} doesn't exist")
                legality_check = False
                return speedup, embedding_tensor, legality_check

        return speedup, embedding_tensor, legality_check

    def is_action_legal(self, action: Action):
        """
        Checks the legality of action
        input :
            - an action that represents an optimization from the 7 types : Parallelization,Skewing,Interchange,Fusion,Reversal,Tiling,Unrolling
        output :
            - legality_check : int , if it is 1 it means it is legal, otherwise it is illegal
        """
        optim_command = OptimizationCommand(action, self.schedule_object.comps)
        # Add the command to the array of schedule
        self.schedule_list.append(optim_command)
        # Check if the action is legal or no to be applied on self.schedule_object.prog
        try:
            legality_check = int(
                CompilingService.compile_legality(
                    schedule_object=self.schedule_object, optims_list=self.schedule_list
                )
            )
        except ValueError as e:
            legality_check = 0
            print("Legality error :", e)
        if legality_check != 1:
            self.schedule_list.pop()
        return legality_check

    def apply_parallelization(self, loop_level):
        # Get any computation since we are using common iterators in a single root programs to apply action parallelization
        #  but #TODO : we need to fix this to support all cases
        computation = list(self.schedule_object.it_dict.keys())[0]
        # Getting the name of the iterator that points to the loop_level
        iterator = self.schedule_object.it_dict[computation][loop_level]["iterator"]
        # Add the tag of parallelized loop level to the computations
        for comp in self.schedule_object.comps:
            self.schedule_object.schedule_dict[comp]["parallelized_dim"] = iterator


    def apply_reversal(self, loop_level):
        # The tag representation is as follows:
        #         ['type_of_transformation', 'first_interchange_loop', 'second_interchange_loop', 'reversed_loop', 'first_skewing_loop', 'second_skewing_loop', 'first_skew_factor', 'second_skew_factor']
        #     Where the type_of_transformation tag is:
        #       - 0 for no transformation being applied
        #       - 1 for loop interchange
        #       - 2 for loop reversal
        #       - 3 for loop skewing
        transformation = [2, 0, 0, loop_level, 0, 0, 0, 0]
        # TODO : for now this action is applied to all comps because they share all the same loop levels , need to fix this to be applied on certain comps only
        for comp in self.schedule_object.comps : 
            self.schedule_object.schedule_dict[comp]["transformations_list"].append(transformation)
        

    def apply_interchange(self,loop_level1 : int,loop_level2 : int):
        # The tag representation is as follows:
        #         ['type_of_transformation', 'first_interchange_loop', 'second_interchange_loop', 'reversed_loop', 'first_skewing_loop', 'second_skewing_loop', 'first_skew_factor', 'second_skew_factor']
        #     Where the type_of_transformation tag is:
        #       - 0 for no transformation being applied
        #       - 1 for loop interchange
        #       - 2 for loop reversal
        #       - 3 for loop skewing
        transformation = [1, loop_level1, loop_level2, 0, 0, 0, 0, 0]
        # TODO : for now this action is applied to all comps because they share all the same loop levels , need to fix this to be applied on certain comps only
        for comp in self.schedule_object.comps : 
            self.schedule_object.schedule_dict[comp]["transformations_list"].append(transformation)