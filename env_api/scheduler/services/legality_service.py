import copy
import logging
from typing import List

from env_api.core.models.optim_cmd import OptimizationCommand
from env_api.core.services.compiling_service import CompilingService
from env_api.core.services.converting_service import ConvertService
from env_api.scheduler.models.action import *
from env_api.scheduler.models.schedule import Schedule


def str_to_int(str: str):
    try:
        return int(str)
    except ValueError:
        return None


class LegalityService:
    def __init__(self):
        """
        The legality service is responsible to evaluate the legality of tiramisu programs given a specific schedule
        """
        pass

    def is_action_legal(
        self,
        schedule_object: Schedule,
        branches: List[Schedule],
        current_branch: int,
        action: Action,
    ):
        """
        Checks the legality of action
        input :
            - an action that represents an optimization from the 7 types : Parallelization,Skewing,Interchange,Fusion,Reversal,Tiling,Unrolling
        output :
            - legality_check : bool
        """
        branches[current_branch].update_actions_mask(action=action, applied=False)

        if isinstance(action, Fusion):
            if len(action.params) != 2 or len(action.params[0]["iterators"]) != len(
                action.params[1]["iterators"]
            ):
                return False
        else:
            # Check first if the iterator(s) level(s) is(are) included in the current iterators
            # If not then the action is illegal by default
            exceeded_iterators = self.check_iterators(
                schedule_object,
                branches=branches,
                current_branch=current_branch,
                action=action,
            )
            if exceeded_iterators:
                return False

            # For the cost model we are only allowed to apply 4 affine transformations by branch
            # We verify that every branch doesn't exceed that amount
            legal_affine_trans = self.check_affine_transformations(
                branches=branches, action=action
            )
            if not legal_affine_trans:
                return False

        # The legality of Skewing is different than the others , we need to get the skewing params from the solver
        # If there are any , this means that skewing is legal , if the solver fails , it means that skewing is illegal
        if isinstance(action, Skewing):
            # check if results of skewing solver exist in the dataset
            schdule_str = ConvertService.build_sched_string(
                schedule_object.schedule_list
            )
            if schdule_str in schedule_object.prog.schedules_solver:
                factors = schedule_object.prog.schedules_solver[schdule_str]

            else:
                if not schedule_object.prog.original_str:
                    # Loading function code lines
                    schedule_object.prog.load_code_lines()
                # Call the skewing solver
                factors = CompilingService.call_skewing_solver(
                    schedule_object=schedule_object,
                    optim_list=schedule_object.schedule_list,
                    action=action,
                    branches=branches,
                )

                # Save the results of skewing solver in the dataset
                schedule_object.prog.schedules_solver[schdule_str] = factors
            if factors == None:
                # The solver fails to find solutions => illegal action
                return False
            else:
                # Adding the factors to the params
                action.params.extend(factors)
                # Assign the requested comps to the action
                optim_command = OptimizationCommand(action)
                # Add the command to the array of schedule
                schedule_object.schedule_list.append(optim_command)
                # Storing the schedule string
                schedule_object.schedule_str = ConvertService.build_sched_string(
                    schedule_object.schedule_list
                )
                return True

        # Assign the requested comps to the action
        optim_command = OptimizationCommand(action)
        # Add the command to the array of schedule
        schedule_object.schedule_list.append(optim_command)
        # Building schedule string
        schdule_str = ConvertService.build_sched_string(schedule_object.schedule_list)
        # Check if the action is legal or no to be applied on schedule_object.prog
        # prog.schedules_legality only has data when it is fetched from the offline dataset so no need to compile to get the legality
        if schdule_str in schedule_object.prog.schedules_legality:
            legality_check = int(schedule_object.prog.schedules_legality[schdule_str])
        else:
            # To run the legality we need the original function code to generate legality code
            if not schedule_object.prog.original_str:
                # Loading function code lines
                schedule_object.prog.load_code_lines()
            try:
                legality_check = int(
                    CompilingService.compile_legality(
                        schedule_object=schedule_object,
                        optims_list=schedule_object.schedule_list,
                        branches=branches,
                    )
                )

                # Saving the legality of the new schedule
                schedule_object.prog.schedules_legality[schdule_str] = (
                    legality_check == 1
                )

            except ValueError as e:
                legality_check = 0
                print("Legality error :", e)

        if legality_check != 1:
            # If the action is not legal , remove it from the schedule list
            schedule_object.schedule_list.pop()
            # Rebuild the scedule string after removing the action
            schdule_str = ConvertService.build_sched_string(
                schedule_object.schedule_list
            )
        # Storing the schedule string to use it later
        schedule_object.schedule_str = schdule_str
        return legality_check == 1

    def check_iterators(
        self,
        schedule_object: Schedule,
        branches: List[Schedule],
        current_branch: int,
        action: Action,
    ):
        loop_levels = []
        # Before checking legality from dataset or by compiling , we see if the iterators are included in the common iterators
        if isinstance(action, Unrolling):
            # We look for the last iterator of each computation and save it in the params
            unrolling_factor = action.params[0]

            innermost_iterator = list(
                branches[current_branch].prog.annotations["iterators"].keys()
            )[-1]

            lower_bound = branches[current_branch].prog.annotations["iterators"][
                innermost_iterator
            ]["lower_bound"]
            lower_bound_int = str_to_int(lower_bound)

            upper_bound = branches[current_branch].prog.annotations["iterators"][
                innermost_iterator
            ]["upper_bound"]

            upper_bound_int = str_to_int(upper_bound)

            if (
                lower_bound_int is not None
                and upper_bound_int is not None
                and abs(upper_bound_int - lower_bound_int) < unrolling_factor
            ):
                logging.error("Unrolling factor is bigger than the loop extent")
                return True

            loop_level = (
                len(branches[current_branch].common_it)
                - 1
                + branches[current_branch].additional_loops
            )
            action.params = copy.deepcopy([loop_level, unrolling_factor])
            action.comps = copy.deepcopy(branches[current_branch].comps)
            return False
        else:
            num_iter = branches[current_branch].common_it.__len__()
            if isinstance(action, Tiling):
                # First we verify if the tiling size is bigger than the loops extent
                # TODO : remove this strategy later
                tiling_size = max(action.params[len(action.params) // 2 :])
                # Becuase the second half of action.params contains tiling size, so we need only the first half of the vector
                loop_levels = action.params[: len(action.params) // 2]
                if loop_levels[-1] >= num_iter:
                    return True
                for i in loop_levels:
                    iterator = list(
                        branches[current_branch].prog.annotations["iterators"].keys()
                    )[i]
                    lower_bound = branches[current_branch].prog.annotations[
                        "iterators"
                    ][iterator]["lower_bound"]

                    lower_bound_int = str_to_int(lower_bound)

                    upper_bound = branches[current_branch].prog.annotations[
                        "iterators"
                    ][iterator]["upper_bound"]

                    upper_bound_int = str_to_int(upper_bound)

                    if (
                        lower_bound_int is not None
                        and upper_bound_int is not None
                        and abs(upper_bound_int - lower_bound_int) < tiling_size
                    ):
                        return True

            else:
                loop_levels = action.params
            # Checking if the big param is smaller than the number of existing iterators
            if loop_levels[-1] >= num_iter:
                return True

        # We have the current branch
        concerned_iterators = [
            branches[current_branch].common_it[it] for it in loop_levels
        ]
        concerned_comps = []
        # This part is for general Tiling to be applied and propagated through many branches
        if isinstance(action, Tiling):
            # We have the concerned iterators to be tiled in the current branch
            # 1st we check if the iterators are shared with other branches or not
            # we will build the following data structure to keep track of the branches that includes the iterators :
            # { 'i0' : [0,1], 'i1': [1]} where the lists contain ids of the branches
            it_dict = {it: [] for it in concerned_iterators}
            comp_dict = {}
            for index, branch in enumerate(branches):
                # print(f"Branch {index} : ", branch.common_it, branch.comps)
                for it in concerned_iterators:
                    if it in branch.common_it:
                        it_dict[it].append(index)
                        comp_dict[index] = copy.deepcopy(branch.comps)
                if concerned_iterators[0] in branch.common_it:
                    concerned_comps.extend(branch.comps)

            # After locating each iterator we now have to check if the branches has been already been tiled on that level
            # to do this we will fetch information form schedule_object.schedule_dict[comp]["tiling"], if it is {}, it means
            # no tiling was applied so far and we can tile with no problem
            for comp in concerned_comps:
                if schedule_object.schedule_dict[comp]["tiling"]:
                    # One of the branches of the block has been already tiled and can not be tiled again
                    return True
            # If no tiling exists in the block we can now apply our action
            match len(concerned_iterators):
                case 2:
                    # We have 2 possibilities : Whether the parent iterator is shared with other branches or not
                    # If it is not shared then no problem we have a 2D tiling to be applied on the current branch
                    # Else we need to apply 1D tiling to the parent iterator for all the comps in the other branches
                    # And a 2D tiling for the current branch
                    if len(it_dict[concerned_iterators[0]]) != len(
                        it_dict[concerned_iterators[1]]
                    ):
                        action.comps = []
                        for br in it_dict[concerned_iterators[0]]:
                            if br not in it_dict[concerned_iterators[1]]:
                                # All those comps should be 1D tiled with the first size
                                subtiling = Tiling(
                                    params=[loop_levels[0], action.params[2]],
                                    env_id=action.env_id,
                                    worker_id=action.worker_id,
                                )
                                subtiling.comps = comp_dict[br]
                                action.subtilings.append(subtiling)
                            else:
                                action.comps.extend(copy.deepcopy(branches[br].comps))
                    else:
                        # Those are the brances shared by all the iterators (the main branches to apply the action)
                        action.comps = copy.deepcopy(concerned_comps)
                case 3:
                    # We have 4 possibilities :
                    # - 1 The 2 parent iterators are not shared => No issue
                    # - 2 One and only one parent is shared (outermost level) => additional 1D Tiling for that iterator in the other branches
                    # - 3 parent iterators are shared with the same number of branches but innermost iterator is not => additional 2D Tiling for those iterators in the other branches
                    # - 4 parents are shared with a different number of branches for each parent iterator => 1D Tiling for the extra branches of the outer most iterator
                    #   + 2D Tiling for the middle iterator branches + 3D Tiling for the current branch
                    size_0 = len(it_dict[concerned_iterators[0]])
                    size_1 = len(it_dict[concerned_iterators[1]])
                    size_2 = len(it_dict[concerned_iterators[2]])

                    if size_0 != size_1:
                        if size_1 == size_2:
                            # This is the 2 nd case
                            # For all the branches in the 1st level we add a 1D tiling except for the branches shared with the children
                            for br in it_dict[concerned_iterators[0]]:
                                if br not in it_dict[concerned_iterators[1]]:
                                    subtiling = Tiling(
                                        params=[loop_levels[0], action.params[3]],
                                        env_id=action.env_id,
                                        worker_id=action.worker_id,
                                    )
                                    subtiling.comps = comp_dict[br]
                                    action.subtilings.append(subtiling)
                                else:
                                    # Those are the brances shared by all the iterators (the main branches to apply the action)
                                    action.comps.extend(
                                        copy.deepcopy(branches[br].comps)
                                    )
                        # else (size_1 != size_2)
                        else:
                            # This is the 4 th case
                            for br in it_dict[concerned_iterators[0]]:
                                if br not in it_dict[concerned_iterators[1]]:
                                    # 1D Tiling for all extra branches in the outermost level
                                    subtiling = Tiling(
                                        params=[loop_levels[0], action.params[3]],
                                        env_id=action.env_id,
                                        worker_id=action.worker_id,
                                    )
                                    subtiling.comps = comp_dict[br]
                                    action.subtilings.append(subtiling)
                                else:
                                    for br in it_dict[concerned_iterators[1]]:
                                        if br not in it_dict[concerned_iterators[2]]:
                                            # 2D Tiling for all extra branches in the middle level
                                            subtiling = Tiling(
                                                params=[
                                                    loop_levels[0],
                                                    loop_levels[1],
                                                    action.params[3],
                                                    action.params[4],
                                                ],
                                                env_id=action.env_id,
                                                worker_id=action.worker_id,
                                            )
                                            subtiling.comps = comp_dict[br]
                                            action.subtilings.append(subtiling)
                                        else:
                                            # Those are the brances shared by all the iterators (the main branches to apply the action)
                                            action.comps.extend(
                                                copy.deepcopy(branches[br].comps)
                                            )
                    # else (size_0 == size_1)
                    else:
                        if size_1 != size_2:
                            # This is the 3rd case
                            for br in it_dict[concerned_iterators[1]]:
                                if br not in it_dict[concerned_iterators[2]]:
                                    # 2D tiling for the extra non shared branches
                                    subtiling = Tiling(
                                        params=[
                                            loop_levels[0],
                                            loop_levels[1],
                                            action.params[3],
                                            action.params[4],
                                        ],
                                        env_id=action.env_id,
                                        worker_id=action.worker_id,
                                    )
                                    subtiling.comps = comp_dict[br]
                                    action.subtilings.append(subtiling)
                                else:
                                    action.comps.extend(
                                        copy.deepcopy(branches[br].comps)
                                    )
                        else:
                            # The case where the 3 iterators are shared with the same number of branches :
                            action.comps = copy.deepcopy(concerned_comps)
            # print("\nThe branches dictionary : ")
            # print(it_dict)
            # print("\nThe actions : ")
            # print(action)
            # print("\nThe comps dict : ")
            # print(comp_dict)
        else:
            # This part of code checks if the iterators of the current branch are all or none shared with the other branhes
            # If some are shared and other aren't we can't apply most of the transformations, and we return True to indicate
            # that the action is not feasible
            match len(concerned_iterators):
                case 1:
                    for branch in branches:
                        if concerned_iterators[0] in branch.common_it:
                            concerned_comps.extend(branch.comps)
                case 2:
                    for branch in branches:
                        if concerned_iterators[0] in branch.common_it:
                            if concerned_iterators[1] in branch.common_it:
                                concerned_comps.extend(branch.comps)
                            else:
                                # If for some branch , we have the parent iterator shared with current branch but
                                # the child is different , this means that only the parent is shared and the children are different
                                return True
                case 3:
                    for branch in branches:
                        if concerned_iterators[0] in branch.common_it:
                            if concerned_iterators[1] in branch.common_it:
                                if concerned_iterators[2] in branch.common_it:
                                    concerned_comps.extend(branch.comps)
                                else:
                                    return True
                            else:
                                return True
            action.comps = copy.deepcopy(concerned_comps)
        return False

    def check_affine_transformations(self, branches: List[Schedule], action: Action):
        if isinstance(action, AffineAction):
            for branch in branches:
                for comp in action.comps:
                    if comp in branch.comps and branch.transformed == 4:
                        return False
        return True
