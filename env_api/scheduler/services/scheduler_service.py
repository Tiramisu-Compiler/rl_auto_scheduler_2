import logging
from typing import List

import torch

from config.config import Config
from env_api.core.models.tiramisu_program import TiramisuProgram
from env_api.core.services.converting_service import ConvertService
from env_api.scheduler.models.branch import Branch
from env_api.scheduler.services.legality_service import LegalityService
from env_api.scheduler.services.prediction_service import PredictionService
from env_api.utils.data_preprocessors import (
    get_representation_template,
    get_schedule_representation,
    linear_diophantine_default,
)
from env_api.utils.exceptions import ExecutingFunctionException
from env_api.utils.functions.fusion import transform_tree_for_fusion

from ..models.action import *
from ..models.schedule import Schedule


class SchedulerService:
    def __init__(self):
        # The Schedule object contains all the informations of a program : annotatons , tree representation ...
        self.schedule_object: Schedule = None
        # The branches generated from the program tree
        self.branches: List[Branch] = []
        self.current_branch = 0
        # The prediction service is an object that has a value estimator `get_predicted_speedup(schedule)` of the speedup that a schedule will have
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
        # Re-init the index to the 1st branch
        self.current_branch = 0
        # Getting the representation of the main schedule and the branched schedule
        main_repr = get_schedule_representation(schedule_object)
        branch_repr = get_schedule_representation(self.branches[self.current_branch])
        # Using the model to embed the main program and the branch in a 180 sized vector for each
        with torch.no_grad():
            _, main_embed = self.prediction_service.get_predicted_speedup(
                *main_repr, schedule_object
            )
            _, branch_embed = self.prediction_service.get_predicted_speedup(
                *branch_repr, self.branches[self.current_branch]
            )

        return (
            [main_embed, branch_embed],
            self.branches[self.current_branch].repr.action_mask,
        )

    def get_current_speedup(self):
        repr_tensors = get_schedule_representation(self.schedule_object)
        speedup, _ = self.prediction_service.get_predicted_speedup(
            *repr_tensors, self.schedule_object
        )
        return speedup, self.schedule_object.schedule_str

    def create_branches(self):
        # Make sure to clear the branches of the previous function if there are ones
        self.branches.clear()
        for branch in self.schedule_object.branches:
            # Create a mock-up of a program from the data of a branch
            program_data = {
                "program_annotation": branch["program_annotation"],
                "schedules_legality": {},
                "schedules_solver": {},
            }
            # The Branch is an inherited class from Schedule, it has all its characteristics
            new_branch = Branch(
                TiramisuProgram.from_dict(
                    self.schedule_object.prog.name, data=program_data, original_str=""
                )
            )
            # The branch needs the original cpp code of the main function to calculate legality of schedules
            new_branch.prog.load_code_lines(self.schedule_object.prog.original_str)
            self.branches.append(new_branch)

    def next_branch(self):
        # Switch to the next branch to optimize it
        self.current_branch += 1
        if self.current_branch == len(self.branches):
            # This matks the finish of exploring the branches
            return None
        main_repr = get_schedule_representation(self.schedule_object)
        branch_repr = get_schedule_representation(self.branches[self.current_branch])
        # Using the model to embed the program and the branch in a 180 sized vector each
        with torch.no_grad():
            _, main_embed = self.prediction_service.get_predicted_speedup(
                *main_repr, self.schedule_object
            )
            _, branch_embed = self.prediction_service.get_predicted_speedup(
                *branch_repr, self.branches[self.current_branch]
            )

        return (
            [main_embed, branch_embed],
            self.branches[self.current_branch].repr.action_mask,
        )

    def apply_action(self, action: Action):
        """
        input :
            - an action that represents an optimization from the 7 types : Parallelization,Skewing,Interchange,Fusion,Reversal,Tiling,Unrolling
        output :
            - speedup : float , representation : tuple(tensor) , legality_check : bool
        """
        legality_check = self.legality_service.is_action_legal(
            schedule_object=self.schedule_object,
            branches=self.branches,
            current_branch=self.current_branch,
            action=action,
        )
        embedding_tensor = None
        speedup = Config.config.experiment.legality_speedup
        if legality_check:
            if Config.config.tiramisu.env_type == "execution":
                # We are going to get the speedup by execution
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

                    elif isinstance(action, Fusion):
                        self.apply_fusion(action=action)

                    speedup = self.prediction_service.get_real_speedup(
                        schedule_object=self.schedule_object, branches=self.branches
                    )

                    # After successfuly applying an action we get the new representation of the main schedule and the branch
                    main_repr_tensors = get_schedule_representation(
                        self.schedule_object
                    )
                    branch_repr_tensors = get_schedule_representation(
                        self.branches[self.current_branch]
                    )

                    # We mesure the speedup from the main schedule and we get the embeddings for both (main and branch)
                    (
                        _,
                        main_embedding_tensor,
                    ) = self.prediction_service.get_predicted_speedup(
                        *main_repr_tensors, self.schedule_object
                    )
                    (
                        _,
                        branch_embedding_tensor,
                    ) = self.prediction_service.get_predicted_speedup(
                        *branch_repr_tensors, self.branches[self.current_branch]
                    )

                    # We pach the 2 tensors to represent the program and the current branch
                    embedding_tensor = [main_embedding_tensor, branch_embedding_tensor]

                except ExecutingFunctionException as e:
                    # If the execution went wring remove it from the schedule list
                    self.schedule_object.schedule_list.pop()
                    # Rebuild the scedule string after removing the action
                    schdule_str = ConvertService.build_sched_string(
                        self.schedule_object.schedule_list
                    )
                    # Storing the schedule string to use it later
                    self.schedule_object.schedule_str = schdule_str

                    legality_check = False

            else:
                # Case where Config.config.tiramisu.env_type == "model"
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
                    # After successfuly applying an action we get the new representation of the main schedule and the branch
                    main_repr_tensors = get_schedule_representation(
                        self.schedule_object
                    )
                    branch_repr_tensors = get_schedule_representation(
                        self.branches[self.current_branch]
                    )

                    # We mesure the speedup from the main schedule and we get the embeddings for both (main and branch)
                    (
                        speedup,
                        main_embedding_tensor,
                    ) = self.prediction_service.get_predicted_speedup(
                        *main_repr_tensors, self.schedule_object
                    )
                    (
                        _,
                        branch_embedding_tensor,
                    ) = self.prediction_service.get_predicted_speedup(
                        *branch_repr_tensors, self.branches[self.current_branch]
                    )

                    # We pach the 2 tensors to represent the program and the current branch
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

        return (
            speedup,
            embedding_tensor,
            legality_check,
            self.branches[self.current_branch].repr.action_mask,
        )

    def apply_parallelization(self, action: Action):
        # Getting the first comp of the selected branch
        computation = list(self.branches[self.current_branch].it_dict.keys())[0]
        # Getting the name of the iterator that points to the loop_level
        # action.params[0]] Represents the loop level
        iterator = self.branches[self.current_branch].it_dict[computation][
            action.params[0]
        ]["iterator"]
        # Add the tag of parallelized loop level to the computations of the action
        for comp in action.comps:
            # Update main schedule
            self.schedule_object.schedule_dict[comp]["parallelized_dim"] = iterator
            for branch in self.branches:
                # Check for the branches that needs to be updated
                if comp in branch.comps:
                    # Update the schedule
                    branch.schedule_dict[comp]["parallelized_dim"] = iterator
                    # Update the actions mask
                    branch.update_actions_mask(action=action)

    def apply_reversal(self, action):
        # ['type_of_transformation', 'first_interchange_loop', 'second_interchange_loop',
        # 'reversed_loop', 'first_skewing_loop', 'second_skewing_loop', 'third_skewing_loop',
        # 'skew_parameter_1', 'skew_parameter_2', 'skew_parameter_3', 'skew_parameter_4',
        # 'skew_parameter_5', 'skew_parameter_6', 'skew_parameter_7', 'skew_parameter_8', 'skew_parameter_9']
        # Where the type_of_transformation tag is:
        # 0 for no transformation being applied
        # 1 for loop interchange
        # 2 for loop reversal
        # 3 for loop skewing
        transformation = [2, 0, 0, action.params[0], 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

        for comp in action.comps:
            # Update main schedule
            self.schedule_object.schedule_dict[comp]["transformations_list"].append(
                transformation
            )
            for branch in self.branches:
                # Check for the branches that needs to be updated
                if comp in branch.comps:
                    # Update the schedule
                    branch.schedule_dict[comp]["transformations_list"].append(
                        transformation
                    )
                    # For the affine transformations we must keep track of how many of them are applied
                    # inside the variable branch.transformed , the limit is 4
                    branch.transformed += 1
                    # Update the actions mask
                    branch.update_actions_mask(action=action)

    def apply_interchange(self, action):
        # ['type_of_transformation', 'first_interchange_loop', 'second_interchange_loop',
        # 'reversed_loop', 'first_skewing_loop', 'second_skewing_loop', 'third_skewing_loop',
        # 'skew_parameter_1', 'skew_parameter_2', 'skew_parameter_3', 'skew_parameter_4',
        # 'skew_parameter_5', 'skew_parameter_6', 'skew_parameter_7', 'skew_parameter_8', 'skew_parameter_9']
        # Where the type_of_transformation tag is:
        # 0 for no transformation being applied
        # 1 for loop interchange
        # 2 for loop reversal
        # 3 for loop skewing
        transformation = [
            1,
            action.params[0],
            action.params[1],
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
        ]

        for comp in action.comps:
            # Update main schedule
            self.schedule_object.schedule_dict[comp]["transformations_list"].append(
                transformation
            )
            for branch in self.branches:
                # Check for the branches that needs to be updated
                if comp in branch.comps:
                    # Update the schedule
                    branch.schedule_dict[comp]["transformations_list"].append(
                        transformation
                    )
                    # For the affine transformations we must keep track of how many of them are applied
                    # inside the variable branch.transformed , the limit is 4
                    branch.transformed += 1
                    # Update the actions mask
                    branch.update_actions_mask(action=action)

    def apply_skewing(self, action):
        # ['type_of_transformation', 'first_interchange_loop', 'second_interchange_loop',
        # 'reversed_loop', 'first_skewing_loop', 'second_skewing_loop', 'third_skewing_loop',
        # 'skew_parameter_1', 'skew_parameter_2', 'skew_parameter_3', 'skew_parameter_4',
        # 'skew_parameter_5', 'skew_parameter_6', 'skew_parameter_7', 'skew_parameter_8', 'skew_parameter_9']
        # Where the type_of_transformation tag is:
        # 0 for no transformation being applied
        # 1 for loop interchange
        # 2 for loop reversal
        # 3 for loop skewing
        x_1, x_2 = linear_diophantine_default(action.params[2], action.params[3])

        transformation = [
            3,
            0,
            0,
            0,
            action.params[0],
            action.params[1],
            0,
            action.params[2],
            action.params[3],
            x_1,
            x_2,
            0,
            0,
            0,
            0,
            0,
        ]

        for comp in action.comps:
            # Update main schedule
            self.schedule_object.schedule_dict[comp]["transformations_list"].append(
                transformation
            )
            for branch in self.branches:
                # Check for the branches that needs to be updated
                if comp in branch.comps:
                    # Update the schedule
                    branch.schedule_dict[comp]["transformations_list"].append(
                        transformation
                    )
                    # For the affine transformations we must keep track of how many of them are applied
                    # inside the variable branch.transformed , the limit is 4
                    branch.transformed += 1
                    # Update the actions mask
                    branch.update_actions_mask(action=action)

    def apply_tiling(self, action):
        params = action.params
        if len(params) == 4:
            # This is the 2d tiling , 4 params becuase it has 2 loop levels and 2 dimensions x,y
            for comp in action.comps:
                tiling_depth = 2  # Because it is 2D tiling
                tiling_factors = [str(params[-2]), str(params[-1])]  # size_x and size_y
                # iterators contains the names of the concerned 2 iterators
                iterators = (
                    self.schedule_object.it_dict[comp][params[0]]["iterator"],
                    self.schedule_object.it_dict[comp][params[1]]["iterator"],
                )
                tiling_dims = [*iterators]
        elif len(params) == 6:
            # This is the 3d tiling , 6 params becuase it has 3 loop levels and 3 dimensions x,y,z
            for comp in action.comps:
                tiling_depth = 3  # Because it is 3D tiling
                tiling_factors = [
                    str(params[-3]),
                    str(params[-2]),
                    str(params[-1]),
                ]  # size_x , size_y and size_z
                # iterators contains the name of the concerned 3 iterators
                iterators = (
                    self.schedule_object.it_dict[comp][params[0]]["iterator"],
                    self.schedule_object.it_dict[comp][params[1]]["iterator"],
                    self.schedule_object.it_dict[comp][params[2]]["iterator"],
                )
                tiling_dims = [*iterators]

        tiling_dict = {
            "tiling_depth": tiling_depth,
            "tiling_dims": tiling_dims,
            "tiling_factors": tiling_factors,
        }

        for comp in action.comps:
            # Update main schedule
            self.schedule_object.schedule_dict[comp]["tiling"] = tiling_dict
            for branch in self.branches:
                # Check for the branches that needs to be updated
                if comp in branch.comps:
                    # Update the branch schedule
                    branch.schedule_dict[comp]["tiling"] = tiling_dict
                    # Update the branch actions mask
                    branch.update_actions_mask(action=action)
                    # Update the additional loops
                    branch.additional_loops = tiling_depth

    def apply_fusion(self, action):
        pass

    def apply_unrolling(self, action):
        # Unrolling is always applied at the innermost level , so it includes only the computations from
        # one branch , no need to check if the action will update other branches besides the current one
        for comp in action.comps:
            # Update the main schedule
            self.schedule_object.schedule_dict[comp]["unrolling_factor"] = str(
                action.params[1]
            )
            # Update the branch schedule
            self.branches[self.current_branch].schedule_dict[comp][
                "unrolling_factor"
            ] = str(action.params[1])
            # Update the actions mask
            self.branches[self.current_branch].update_actions_mask(action=action)
