from typing import List

import torch
from config.config import Config
from env_api.core.models.tiramisu_program import TiramisuProgram
from env_api.core.services.converting_service import ConvertService
from env_api.scheduler.models.branch import Branch
from env_api.scheduler.services.legality_service import LegalityService
from env_api.scheduler.services.prediction_service import PredictionService
from env_api.utils.functions.fusion import transform_tree_for_fusion
from ..models.schedule import Schedule
from ..models.action import *
import logging


class SchedulerService:
    def __init__(self):
        # The Schedule object contains all the informations of a program : annotatons , tree representation ...
        self.schedule_object: Schedule = None
        # The branches generated from the program tree 
        self.branches : List[Branch] = []
        self.current_branch = 0
        # The prediction service is an object that has a value estimator `get_speedup(schedule)` of the speedup that a schedule will have
        # This estimator is a recursive model that needs the schedule representation to give speedups
        self.prediction_service = PredictionService()
        # A schedules-legality service
        self.legality_service = LegalityService()

    def set_schedule(self, schedule_object: Schedule):
        """
        The `set_schedule` function is called first in `tiramisu_api` to initialize the fields when a new program is fetched from the dataset.
        input :
            - schedule_object : contains all the inforamtions on a program and the schedule
        output :
            - a tuple of vectors that represents the main program and the current branch , in addition to their respective actions mask
        """
        self.schedule_object = schedule_object
        # We create the branches of the program
        self.create_branches()
        self.current_branch = 0
        main_comps , main_loops = ConvertService.get_schedule_representation(schedule_object)
        branch_comps , branch_loops = ConvertService.get_schedule_representation(self.branches[self.current_branch])
        # Using the model to embed the program and the branch in a 180 sized vector each
        with torch.no_grad():
            _, main_embed = self.prediction_service.get_speedup(
                main_comps, main_loops,schedule_object)
            _, branch_embed = self.prediction_service.get_speedup(
                branch_comps, branch_loops,self.branches[self.current_branch])
        
        return ([main_embed, branch_embed], 
                self.branches[self.current_branch].repr.action_mask
                )
    def get_current_speedup(self):
        repr_tensors = ConvertService.get_schedule_representation(
            self.schedule_object)
        speedup, embedding_tensor = self.prediction_service.get_speedup(
            *repr_tensors, self.schedule_object)
        return speedup , self.schedule_object.schedule_str

    def create_branches(self):
        self.branches.clear()
        for branch in self.schedule_object.branches : 
            program_data = {
                "program_annotation" : branch["annotations"],
                "schedules_legality" : {},
                "schedules_solver" : {}
            }
            new_branch = Branch(TiramisuProgram.from_dict(self.schedule_object.prog.name,
                                                                  data=program_data,
                                                                  original_str=""))
            new_branch.prog.load_code_lines(self.schedule_object.prog.original_str)
            self.branches.append(new_branch)
            
    def next_branch(self):
        self.current_branch += 1
        if (self.current_branch == len(self.branches)):
            return None
        main_comps , main_loops = ConvertService.get_schedule_representation(self.schedule_object)
        branch_comps , branch_loops = ConvertService.get_schedule_representation(self.branches[self.current_branch])
        # Using the model to embed the program and the branch in a 180 sized vector each
        with torch.no_grad():
            _, main_embed = self.prediction_service.get_speedup(
                main_comps, main_loops,self.schedule_object)
            _, branch_embed = self.prediction_service.get_speedup(
                branch_comps, branch_loops,self.branches[self.current_branch])
        
        return ([main_embed, branch_embed], 
                self.branches[self.current_branch].repr.action_mask
                )

    def apply_action(self, action: Action):
        """
        input :
            - an action that represents an optimization from the 7 types : Parallelization,Skewing,Interchange,Fusion,Reversal,Tiling,Unrolling
        output :
            - speedup : float , representation : tuple(tensor) , legality_check : bool
        """
        print("&"*50)
        print("Function : ", self.schedule_object.prog.name)
        legality_check = self.legality_service.is_action_legal(schedule_object=self.schedule_object,
                                                               branches=self.branches,
                                                               current_branch=self.current_branch,
                                                               action=action)
        
        print("Branch :" , self.current_branch + 1 , "/" , len(self.branches))
        print("Comps of the branch :", self.branches[self.current_branch].comps)
        print("Action :", action.name , *action.params)
        print("Legal ? :", legality_check)
        print("Comps of the action :",action.comps)
        print("Schedule :", self.schedule_object.schedule_str)

        embedding_tensor = None
        speedup = Config.config.experiment.legality_speedup
        if legality_check:
            try:
                if isinstance(action, Parallelization):
                    self.apply_parallelization(action=action)

                elif isinstance(action, Reversal):
                    self.apply_reversal(action=action)

                elif isinstance(action, Interchange):
                    self.apply_interchange(action=action)

                elif isinstance(action, Tiling):
                    self.apply_tiling(action=action)
                    
                elif isinstance(action, Unrolling):
                    self.apply_unrolling(action=action)

                elif isinstance(action, Skewing):
                    self.apply_skewing(action=action)

                # repr_tensors contains 2 tensors , the 1st one is related to computations and the 2nd one is related to loops,
                # we need these 2 tensors for the input of the model.
                main_repr_tensors = ConvertService.get_schedule_representation(
                    self.schedule_object)
                branch_repr_tensors = ConvertService.get_schedule_representation(
                    self.branches[self.current_branch])
                speedup, main_embedding_tensor = self.prediction_service.get_speedup(
                    *main_repr_tensors, self.schedule_object)
                _, branch_embedding_tensor = self.prediction_service.get_speedup(
                    *branch_repr_tensors, self.branches[self.current_branch])
                embedding_tensor = [main_embedding_tensor, branch_embedding_tensor]
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

        print("&"*50)

        return speedup, embedding_tensor, legality_check, self.branches[self.current_branch].repr.action_mask

    def apply_parallelization(self, action: Action):
        # Get any computation since we are using common iterators in a single root programs to apply action parallelization
        #  but #TODO : we need to fix this to support all cases
        computation = list(self.branches[self.current_branch].it_dict.keys())[0]
        # Getting the name of the iterator that points to the loop_level
        # action.params[0]] Represents the loop level 
        iterator = self.branches[self.current_branch].it_dict[computation][action.params[0]][
            "iterator"]
        # Add the tag of parallelized loop level to the computations
        for comp in action.comps:
            self.schedule_object.schedule_dict[comp][
                "parallelized_dim"] = iterator
            for branch in self.branches : 
                if (comp in branch.comps):
                    branch.schedule_dict[comp]["parallelized_dim"] = iterator
                    branch.update_actions_mask(action=action)


    def apply_reversal(self, action):
        # The tag representation is as follows:
        #         ['type_of_transformation', 'first_interchange_loop', 'second_interchange_loop', 'reversed_loop', 'first_skewing_loop', 'second_skewing_loop', 'first_skew_factor', 'second_skew_factor']
        #     Where the type_of_transformation tag is:
        #       - 0 for no transformation being applied
        #       - 1 for loop interchange
        #       - 2 for loop reversal
        #       - 3 for loop skewing
        transformation = [2, 0, 0, action.params[0] , 0, 0, 0, 0]

        for comp in action.comps:
            self.schedule_object.schedule_dict[comp][
                "transformations_list"].append(transformation)
            for branch in self.branches : 
                if (comp in branch.comps):
                    branch.schedule_dict[comp][
                "transformations_list"].append(transformation)
                    branch.transformed+=1
                    branch.update_actions_mask(action=action)

    def apply_interchange(self, action):
        # The tag representation is as follows:
        #         ['type_of_transformation', 'first_interchange_loop', 'second_interchange_loop', 'reversed_loop', 'first_skewing_loop', 'second_skewing_loop', 'first_skew_factor', 'second_skew_factor']
        #     Where the type_of_transformation tag is:
        #       - 0 for no transformation being applied
        #       - 1 for loop interchange
        #       - 2 for loop reversal
        #       - 3 for loop skewing
        
        transformation = [1, action.params[0], action.params[1], 0, 0, 0, 0, 0]

        for comp in action.comps:
            self.schedule_object.schedule_dict[comp][
                "transformations_list"].append(transformation)
            for branch in self.branches : 
                if (comp in branch.comps):
                    branch.schedule_dict[comp][
                "transformations_list"].append(transformation)
                    branch.transformed+=1
                    branch.update_actions_mask(action=action)

    def apply_skewing(self, action):
        # The tag representation is as follows:
        #         ['type_of_transformation', 'first_interchange_loop', 'second_interchange_loop', 'reversed_loop', 'first_skewing_loop', 'second_skewing_loop', 'first_skew_factor', 'second_skew_factor']
        #     Where the type_of_transformation tag is:
        #       - 0 for no transformation being applied
        #       - 1 for loop interchange
        #       - 2 for loop reversal
        #       - 3 for loop skewing
        transformation = [
            3, 0, 0, 0, action.params[0], action.params[1], action.params[2], action.params[3]
        ]
        for comp in action.comps:
            self.schedule_object.schedule_dict[comp][
                "transformations_list"].append(transformation)
            for branch in self.branches : 
                if (comp in branch.comps):
                    branch.schedule_dict[comp][
                "transformations_list"].append(transformation)
                    branch.transformed+=1
                    branch.update_actions_mask(action=action)

    def apply_tiling(self, action):
        params = action.params
        if (len(params) == 4):
            # This is the 2d tiling , 4 params becuase it has 2 loop levels and 2 dimensions x,y
            for comp in action.comps:
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
            for comp in action.comps:
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
        for comp in action.comps:
            self.schedule_object.schedule_dict[comp]["tiling"] = tiling_dict
            for branch in self.branches : 
                if (comp in branch.comps):
                    branch.schedule_dict[comp]["tiling"] = tiling_dict
                    branch.update_actions_mask(action=action)


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
    def apply_unrolling(self, action):
        for comp in action.comps:
            self.schedule_object.schedule_dict[comp]["unrolling_factor"] = str(action.params[1])
            self.branches[self.current_branch].schedule_dict[comp]["unrolling_factor"] = str(action.params[1])
            self.branches[self.current_branch].update_actions_mask(action=action)