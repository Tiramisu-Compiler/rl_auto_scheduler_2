from config.config import Config
from env_api.core.models.optim_cmd import OptimizationCommand
from env_api.core.services.compiling_service import CompilingService
from env_api.core.services.converting_service import ConvertService
from env_api.scheduler.services.prediction_service import PredictionService
from env_api.utils.functions.fusion import transform_tree_for_fusion
from ..models.schedule import Schedule
from ..models.action import *
import logging


class SchedulerService:
    def __init__(self):
        # self.schedule_list is an array that contains a list of optimizations that has been applied on the program
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
        repr_tensors = ConvertService.get_schedule_representation(
            self.schedule_object)
        return ConvertService.get_tree_representation(*repr_tensors,
                                                      self.schedule_object)

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
        legality_check = self.is_action_legal(action) == 1
        embedding_tensor = None
        speedup = Config.config.experiment.legality_speedup
        actions_mask = self.schedule_object.repr.action_mask
        if legality_check:
            try:
                if isinstance(action, Parallelization):
                    self.apply_parallelization(loop_level=action.params[0])

                elif isinstance(action, Reversal):
                    self.apply_reversal(loop_level=action.params[0])
                    self.schedule_object.transformed += 1

                elif isinstance(action, Interchange):
                    self.apply_interchange(loop_level1=action.params[0],
                                           loop_level2=action.params[1])
                    self.schedule_object.transformed += 1

                #TODO : recheck if this is an efficient modeling
                elif isinstance(action, Tiling):
                    self.apply_tiling(params=action.params)

                elif isinstance(action, Fusion):
                    self.apply_fusion(loop_level=action.params[0],
                                      comps=action.comps)
                elif isinstance(action, Unrolling):
                    self.apply_unrolling(params=action.params)

                elif isinstance(action, Skewing):
                    self.apply_skewing(*action.params)
                    self.schedule_object.transformed += 1

                # repr_tensors contains 2 tensors , the 1st one is related to computations and the 2nd one is related to loops,
                # we need these 2 tensors for the input of the model.
                repr_tensors = ConvertService.get_schedule_representation(
                    self.schedule_object)
                speedup, embedding_tensor = self.prediction_service.get_speedup(
                    *repr_tensors, self.schedule_object)
                print(self.schedule_object.schedule_str)
                print("Speedup:", speedup)
            except KeyError as e:
                logging.error(f"This loop level: {e} doesn't exist")
                legality_check = False
            except AssertionError as e:
                print("%" * 50)
                print("Used more than 4 transformations of I,R,S")
                print(self.schedule_object.prog.name)
                print(self.schedule_object.schedule_str)
                print(action.params)
                print(action.name)
                print("%" * 50)
                legality_check = False

        actions_mask = self.schedule_object.update_actions_mask(
            action=action, applied=legality_check)
        legality_schedule = self.schedule_object.prog.schedules

        return speedup, embedding_tensor, legality_check, actions_mask, legality_schedule

    def is_action_legal(self, action: Action):
        """
        Checks the legality of action
        input :
            - an action that represents an optimization from the 7 types : Parallelization,Skewing,Interchange,Fusion,Reversal,Tiling,Unrolling
        output :
            - legality_check : int , if it is 1 it means it is legal, otherwise it is illegal
        """

        # Before checking legality from dataset or by compiling , we see if the iterators are included in the common iterators
        if (not isinstance(action, Unrolling)):
            num_iter = self.schedule_object.common_it.__len__()
            if isinstance(action, Tiling):
                # Becuase the second half of action.params contains tiling size, so we need only the first half of the vector
                params = action.params[:len(action.params) // 2]
            else:
                params = action.params
            for param in params:
                if param >= num_iter:
                    return 0

        if isinstance(action, Fusion):
            if (len(self.schedule_object.comps) <= 1):
                # If the program has a single computation , then fusion is illegal
                return 0
            requested_comps = [
                comp for comp in self.schedule_object.it_dict
                if action.params[0] in self.schedule_object.it_dict[comp]
            ]
            if (len(requested_comps) <= 1):
                # If there are many computations but , at the fusion loop level there is less than 2 computations
                # then fusion will be illegal
                return 0

        # TODO : remove this condition when we apply the new method
        elif isinstance(action, Unrolling):
            # In this case we unroll all the computations
            requested_comps = self.schedule_object.comps
            # We look for the last iterator of each computation and save it in the params
            unrolling_factor = action.params[0]
            action.params = {}
            for comp in self.schedule_object.it_dict:
                loop_level = len(self.schedule_object.it_dict[comp].keys()) - 1
                action.params[comp] = [loop_level, unrolling_factor]

        # For skewing action we need first to get the skewing params : a list of 2 int
        elif isinstance(action, Skewing):
            if (not self.schedule_object.prog.original_str):
                # Loading function code lines
                self.schedule_object.prog.load_code_lines()
            requested_comps = self.schedule_object.comps
            factors = CompilingService.call_skewing_solver(
                schedule_object=self.schedule_object,
                optim_list=self.schedule_list,
                params=action.params)
            if (factors == None):
                return 0
            else:
                action.params.extend(factors)
        else:
            requested_comps = self.schedule_object.comps
        # Assign the requested comps to the action
        # TODO : This field comps is recently added , propagate the use of it in all the actions methods
        action.comps = requested_comps
        optim_command = OptimizationCommand(action, requested_comps)
        # Add the command to the array of schedule
        self.schedule_list.append(optim_command)
        # Building schedule string
        schdule_str = ConvertService.build_sched_string(self.schedule_list)
        # Check if the action is legal or no to be applied on self.schedule_object.prog
        # prog.schedules only has data when it is fetched from the offline dataset so no need to compile to get the legality
        if (self.schedule_object.prog.schedules
                and (schdule_str in self.schedule_object.prog.schedules)):
            legality_check = int(
                self.schedule_object.prog.schedules[schdule_str])
        else:
            # To run the legality we need the original function code to generate legality code
            if (not self.schedule_object.prog.original_str):
                # Loading function code lines
                self.schedule_object.prog.load_code_lines()
            try:
                legality_check = int(
                    CompilingService.compile_legality(
                        schedule_object=self.schedule_object,
                        optims_list=self.schedule_list))
                # Saving the legality of the new schedule
                self.schedule_object.prog.schedules[schdule_str] = (
                    legality_check == 1)

            except ValueError as e:
                legality_check = 0
                print("Legality error :", e)
        if legality_check != 1:
            self.schedule_list.pop()
            schdule_str = ConvertService.build_sched_string(self.schedule_list)

        self.schedule_object.schedule_str = schdule_str
        return legality_check

    def apply_parallelization(self, loop_level):
        # Get any computation since we are using common iterators in a single root programs to apply action parallelization
        #  but #TODO : we need to fix this to support all cases
        computation = list(self.schedule_object.it_dict.keys())[0]
        # Getting the name of the iterator that points to the loop_level
        iterator = self.schedule_object.it_dict[computation][loop_level][
            "iterator"]
        # Add the tag of parallelized loop level to the computations
        for comp in self.schedule_object.comps:
            self.schedule_object.schedule_dict[comp][
                "parallelized_dim"] = iterator

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
        for comp in self.schedule_object.comps:
            self.schedule_object.schedule_dict[comp][
                "transformations_list"].append(transformation)

    def apply_interchange(self, loop_level1: int, loop_level2: int):
        # The tag representation is as follows:
        #         ['type_of_transformation', 'first_interchange_loop', 'second_interchange_loop', 'reversed_loop', 'first_skewing_loop', 'second_skewing_loop', 'first_skew_factor', 'second_skew_factor']
        #     Where the type_of_transformation tag is:
        #       - 0 for no transformation being applied
        #       - 1 for loop interchange
        #       - 2 for loop reversal
        #       - 3 for loop skewing
        transformation = [1, loop_level1, loop_level2, 0, 0, 0, 0, 0]
        # TODO : for now this action is applied to all comps because they share all the same loop levels , need to fix this to be applied on certain comps only
        for comp in self.schedule_object.comps:
            self.schedule_object.schedule_dict[comp][
                "transformations_list"].append(transformation)

    def apply_skewing(self, loop_level1: int, loop_level2: int, factor1: int,
                      factor2: int):
        # The tag representation is as follows:
        #         ['type_of_transformation', 'first_interchange_loop', 'second_interchange_loop', 'reversed_loop', 'first_skewing_loop', 'second_skewing_loop', 'first_skew_factor', 'second_skew_factor']
        #     Where the type_of_transformation tag is:
        #       - 0 for no transformation being applied
        #       - 1 for loop interchange
        #       - 2 for loop reversal
        #       - 3 for loop skewing
        transformation = [
            3, 0, 0, 0, loop_level1, loop_level2, factor1, factor2
        ]
        for comp in self.schedule_object.comps:
            self.schedule_object.schedule_dict[comp][
                "transformations_list"].append(transformation)

    def apply_tiling(self, params):
        if (len(params) == 4):
            # This is the 2d tiling , 4 params becuase it has 2 loop levels and 2 dimensions x,y
            for comp in self.schedule_object.comps:
                tiling_depth = 2  # Because it is 2D tiling
                tiling_factors = [str(params[-2]),
                                  str(params[-1])]  # size_x and size_y
                # iterators contains the names of the concerned 2 iterators
                iterators = self.schedule_object.it_dict[comp][
                    params[0]]["iterator"], self.schedule_object.it_dict[comp][
                        params[1]]["iterator"]
                tiling_dims = [*iterators]
        elif (len(params) == 6):
            # This is the 3d tiling , 6 params becuase it has 3 loop levels and 3 dimensions x,y,z
            for comp in self.schedule_object.comps:
                tiling_depth = 3  # Because it is 3D tiling
                tiling_factors = [
                    str(params[-3]),
                    str(params[-2]),
                    str(params[-1])
                ]  # size_x , size_y and size_z
                # iterators contains the name of the concerned 3 iterators
                iterators = self.schedule_object.it_dict[comp][
                    params[0]]["iterator"], self.schedule_object.it_dict[comp][
                        params[1]]["iterator"], self.schedule_object.it_dict[
                            comp][params[2]]["iterator"]
                tiling_dims = [*iterators]

        tiling_dict = {
            'tiling_depth': tiling_depth,
            'tiling_dims': tiling_dims,
            'tiling_factors': tiling_factors,
        }
        self.schedule_object.schedule_dict[comp]["tiling"] = tiling_dict

    def apply_fusion(self, loop_level, comps):
        # check if fusions are empty in schedule dict
        if not self.schedule_object.schedule_dict["fusions"]:
            self.schedule_object.schedule_dict["fusions"] = []
        # Form the new fusion field in schedule dict
        fusion = [*comps, loop_level]
        self.schedule_object.schedule_dict["fusions"].append(fusion)
        fused_tree = transform_tree_for_fusion(
            self.schedule_object.schedule_dict['tree_structure'],
            self.schedule_object.schedule_dict["fusions"])
        self.schedule_object.schedule_dict['tree_structure'] = fused_tree

    # TODO : change this function later
    def apply_unrolling(self, params):
        for comp in params:
            self.schedule_object.schedule_dict[comp]["unrolling_factor"] = str(
                params[comp][1])
