import copy
import logging

import numpy as np

from config.config import Config
from env_api.core.models.tiramisu_program import TiramisuProgram
from env_api.scheduler.models.action import *
from env_api.scheduler.models.representation import Representation
from env_api.utils.data_preprocessors import (
    build_tree_structure,
    get_representation_template,
)


class Schedule:
    def __init__(self, program: TiramisuProgram):
        self.schedule_str = ""
        # A counter of the applied affine transformations
        self.transformed = 0
        self.prog = program
        # List of computations of the program
        self.comps = self.prog.comps
        # The repr object has the raw data of computations , loops , expressions as tensors
        self.repr: Representation = None
        # Iterators dictionnary
        self.it_dict = {}
        # List of branches of the program tree
        self.branches = []
        # A dictionnary that has the types of schedule applied on the program with their representation in the cost model
        self.schedule_dict = {}
        # List of common iterators
        self.common_it = []
        # self.schedule_list is an array that contains a list of optimizations that has been applied on the program
        # This list has objects of type `OptimizationCommand`
        self.schedule_list = []
        # Additional loops when Tiling is applied
        self.additional_loops = 0
        if (type(self).__name__) == "Schedule":
            self.__calculate_common_it()
            self.__init_schedule_dict_tags()
            self.__init_representation()
            self.__set_action_mask()
            self.__form_iterators_dict()
            self.__form_branches()
        else:
            self.__init_schedule_dict_tags()
            self.__init_representation()
            self.__set_branch_action_mask()
            self.__form_iterators_dict()

    def __calculate_common_it(self):
        if len(self.comps) != 1:  # Multi-computation program
            # comps_it is a list of lists of iterators of computations
            comps_it = []
            for comp in self.comps:
                comps_it.append(
                    self.prog.annotations["computations"][comp]["iterators"]
                )
            self.common_it = comps_it[0]
            for comp_it in comps_it[1:]:
                self.common_it = [it for it in comp_it if it in self.common_it]
        else:  # A single comp program
            self.common_it = self.prog.annotations["computations"][self.comps[0]][
                "iterators"
            ]

    def __init_schedule_dict_tags(self):
        self.schedule_dict["fusions"] = None
        for comp in self.prog.annotations["computations"]:
            self.schedule_dict[comp] = {
                "tiling": {},
                "unrolling_factor": None,
                "parallelized_dim": None,
                "shiftings": None,
                "transformations_list": [],
            }

        self.schedule_dict["tree_structure"] = {
            "roots": build_tree_structure(self.prog.annotations["iterators"])
        }

    def __init_representation(self):
        self.repr = Representation(
            *get_representation_template(self.prog.annotations, self.schedule_dict)
        )

    def __set_action_mask(self):
        self.repr.action_mask = np.ones(33)
        # Next action
        self.repr.action_mask[-1] = 0
        # Fusion Action
        self.repr.action_mask[-2] = 0

    def __set_branch_action_mask(self):
        self.repr.action_mask = np.zeros(33)
        # Next action
        self.repr.action_mask[-2] = 1

    def __form_iterators_dict(self):
        for comp in self.comps:
            comp_it_dict = {}
            iterators = list(self.prog.annotations["computations"][comp]["iterators"])
            for i in range(len(iterators)):
                comp_it_dict[i] = {}
                comp_it_dict[i]["iterator"] = iterators[i]
                comp_it_dict[i]["lower_bound"] = self.prog.annotations["iterators"][
                    iterators[i]
                ]["lower_bound"]
                comp_it_dict[i]["upper_bound"] = self.prog.annotations["iterators"][
                    iterators[i]
                ]["upper_bound"]
            self.it_dict[comp] = comp_it_dict

    def __form_branches(self):
        branches = []

        computations = copy.deepcopy(self.prog.annotations["computations"])
        it = {}
        for computation in computations:
            iterators = copy.deepcopy(
                self.prog.annotations["computations"][computation]["iterators"]
            )
            if iterators[-1] in it:
                it[iterators[-1]]["comps"].append(computation)
            else:
                it[iterators[-1]] = {"comps": [computation], "iterators": iterators}

        for iterator in it:
            branches.append(
                {
                    "comps": copy.deepcopy(it[iterator]["comps"]),
                    "iterators": copy.deepcopy(it[iterator]["iterators"]),
                }
            )

        for branch in branches:
            branch_annotations = {"computations": {}, "iterators": {}}
            for comp in branch["comps"]:
                branch_annotations["computations"][comp] = copy.deepcopy(
                    self.prog.annotations["computations"][comp]
                )
            # extract the branch specific iterators annotations
            for iterator in branch["iterators"]:
                branch_annotations["iterators"][iterator] = copy.deepcopy(
                    self.prog.annotations["iterators"][iterator]
                )

                if branch_annotations["iterators"][iterator]["computations_list"]:
                    branch_annotations["iterators"][iterator]["child_iterators"] = []

                if self.prog.annotations["iterators"][iterator]["parent_iterator"]:
                    # Making sure that the parent node has the actual node as the only child
                    # It may happen that the parent node has many children but in a branch it is only allowed
                    # to have a single child to form a straight-forward branch from top to bottom
                    parent = branch_annotations["iterators"][copy.copy(iterator)][
                        "parent_iterator"
                    ]
                    branch_annotations["iterators"][parent][
                        "child_iterators"
                    ] = copy.deepcopy([iterator])
                    branch_annotations["iterators"][parent]["computations_list"] = []
            branch["program_annotation"] = copy.deepcopy(branch_annotations)

        self.branches = branches

    def update_actions_mask(self, action: Action, applied: bool = True):
        # Whether an action is legal or not we should mask it to not use it again
        self.repr.action_mask[action.env_id] = 1

        if applied:
            # if the action is legal and applied we need to mask similar action when it comes
            # to Unrolling , skewing and parallelization because these action are applied once
            if isinstance(action, Parallelization):
                self.repr.action_mask[12:14] = 1
            elif isinstance(action, Tiling):
                self.repr.action_mask[14:26] = 1
            elif isinstance(action, Unrolling):
                self.repr.action_mask[26:31] = 1
            # The other case is for skewing , reversal and interchange
            # for these actions we are allowed to apply them in any order under the condition of not
            # surpassing 4 times of applying them.
            if self.transformed == 4:
                self.repr.action_mask[0:12] = 1

            if Config.config.experiment.beam_search_order:
                self.apply_beam_search_conditions(action=action)

    def apply_beam_search_conditions(self, action: Action):
        # The order of actions in beam search :
        # Fusion, [Interchange, reversal, skewing], parallelization, tiling, unrolling
        if isinstance(action, Parallelization):
            self.repr.action_mask[0:14] = 1

        elif isinstance(action, Tiling):
            self.repr.action_mask[0:26] = 1

        elif isinstance(action, Unrolling):
            self.repr.action_mask[0:31] = 1

    def unmask_actions(self):
        self.repr.action_mask = np.zeros(33)
        self.repr.action_mask[-2] = 1
