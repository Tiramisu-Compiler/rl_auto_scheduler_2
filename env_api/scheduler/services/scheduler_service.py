from env_api.core.models.optim_cmd import OptimizationCommand
from env_api.core.services.compiling_service import CompilingService
from env_api.core.services.converting_service import ConvertService
from env_api.scheduler.services.prediction_service import PredictionService
from ..models.schedule import Schedule
from ..models.action import *
import numpy as np
import logging


class SchedulerService:
    def __init__(self):
        self.schedule = []
        self.schedule_object: Schedule = None
        self.prediction_service = PredictionService()

    def set_schedule(self , schedule : Schedule):
        self.schedule_object = schedule
        self.schedule = []
        comps_tensor, loops_tensor = ConvertService.get_schedule_representation(self.schedule_object)
        x = comps_tensor
        batch_size, num_comps, __dict__ = x.shape
        x = x.view(batch_size * num_comps, -1)
        (first_part, vectors, third_part) = ConvertService.seperate_vector(x, num_transformations=4, pad=False)
        first_part = first_part.view(batch_size, num_comps, -1)
        third_part = third_part.view(batch_size, num_comps, -1)
        tree_tensor = (self.schedule_object.repr.prog_tree, first_part, vectors, third_part, loops_tensor, self.schedule_object.repr.comps_expr_tensor, self.schedule_object.repr.comps_expr_lengths)
        return tree_tensor

    def get_annotations(self):
        return self.schedule_object.prog.annotations

    def get_schedule_dict(self):
        return self.schedule_object.schedule_dict

    def get_representation(self):
        return self.schedule_object.repr

    def apply_action(self, action: Action):
        legality_check = self.is_action_legal(action) == 1
        speedup = 1.0
        representation = self.schedule_object.repr
        if (legality_check):
            try:
                if isinstance(action, Parallelization):
                    self.apply_parallelization(loop_level=action.params[0])
                    self.schedule_object.is_parallelized = True
                elif isinstance(action, Reversal):
                    self.apply_reversal(loop_level=action.params[0])
                    self.schedule_object.is_reversed = True
                elif isinstance(action, Interchange):
                    self.apply_interchange(loop1=action.params[0],
                                           loop2=action.params[1])
                    self.schedule_object.is_interchaged = True

                speedup = self.prediction_service.get_speedup(self.schedule_object)
                representation = self.schedule_object.repr

            except KeyError as e:
                logging.error(f"Key Error: {e}")
        
        return speedup , representation , legality_check

    def is_action_legal(self, action: Action):
        optim_command = OptimizationCommand(action, self.schedule_object.comps)
        # Add the command to the array of schedule
        self.schedule.append(optim_command)
        # Check if the action is legal or no to be applied on self.schedule_object.prog
        try : 
            legality_check = int(
                CompilingService.compile_legality(
                    tiramisu_program=self.schedule_object.prog,
                    optims_list=self.schedule,
                    first_comp=self.schedule_object.comps[0]))

        except ValueError as e :
            legality_check = 0
            print("Legality error :",e)
        
        if legality_check < 1:
            self.schedule.pop()
        return legality_check

    def apply_parallelization(self, loop_level):
        first_comp = list(self.schedule_object.it_dict.keys())[0]
        iterator = self.schedule_object.it_dict[first_comp][loop_level][
            'iterator']
        self.schedule_object.schedule_dict[first_comp][
            "parallelized_dim"] = iterator
        l_code = "L" + iterator

        self.schedule_object.repr["representation"][0][
            self.schedule_object.placeholders[first_comp][l_code +
                                                          "Parallelized"]] = 1

        iterators = list(
            self.schedule_object.prog.annotations["iterators"].keys())
        if self.schedule_object.it_dict[first_comp][loop_level][
                'iterator'] in iterators:
            loop_index = iterators.index(
                self.schedule_object.it_dict[first_comp][loop_level]
                ['iterator'])
        elif self.schedule_object.it_dict[first_comp][loop_level][
                'iterator'] in self.schedule_object.added_iterators:
            loop_index = len(self.schedule_object.prog.annotations['iterators']
                             ) + self.schedule_object.added_iterators.index(
                                 self.schedule_object.it_dict[first_comp]
                                 [loop_level]['iterator'])
        self.schedule_object.repr["loops_representation"][loop_index][10] = 1
        self.schedule_object.repr["action_mask"][46] = 0
        self.schedule_object.repr["action_mask"][47] = 0
        for i in range(56, 61):
            self.schedule_object.repr["action_mask"][i] = 0

    def apply_reversal(self, loop_level):

        for comp in self.schedule_object.comps:
            l_code = "L" + self.schedule_object.it_dict[comp][loop_level][
                'iterator']

            index_upper_bound = self.schedule_object.placeholders[comp][
                l_code + 'Interchanged'] - 1
            index_lower_bound = self.schedule_object.placeholders[comp][
                l_code + 'Interchanged'] - 2

            self.schedule_object.repr["representation"][
                self.schedule_object.comp_indic_dict[comp]][
                    self.schedule_object.placeholders[comp][l_code +
                                                            "Reversed"]] = 1

            tmp = self.schedule_object.repr["representation"][
                self.schedule_object.comp_indic_dict[comp]][index_lower_bound]
            self.schedule_object.repr["representation"][
                self.schedule_object.comp_indic_dict[comp]][
                    index_lower_bound] = self.schedule_object.repr[
                        "representation"][self.schedule_object.comp_indic_dict[
                            comp]][index_upper_bound]
            self.schedule_object.repr["representation"][
                self.schedule_object.
                comp_indic_dict[comp]][index_upper_bound] = tmp

        iterators = list(
            self.schedule_object.prog.annotations["iterators"].keys())
        if self.schedule_object.it_dict[comp][loop_level][
                'iterator'] in iterators:
            loop_index = iterators.index(
                self.schedule_object.it_dict[comp][loop_level]['iterator'])
        elif self.schedule_object.it_dict[comp][loop_level][
                'iterator'] in self.schedule_object.added_iterators:
            loop_index = len(
                self.schedule_object.prog.annotations['iterators']
            ) + self.schedule_object.added_iterators.index(
                self.schedule_object.it_dict[comp][loop_level]['iterator'])
        self.schedule_object.repr["loops_representation"][loop_index][11] = 1

        for i in range(48, 56):
            self.schedule_object.repr["action_mask"][i] = 0
        for i in range(56, 61):
            self.schedule_object.repr["action_mask"][i] = 0

        for comp in self.schedule_object.comps:
            dim = self.schedule_object.schedule_dict[comp]["dim"]
            reversal_matrix = np.eye(dim, dim)
            dim_index = loop_level
            reversal_matrix[dim_index, dim_index] = -1
            self.schedule_object.schedule_dict[comp][
                "transformation_matrices"].append(reversal_matrix)
            self.schedule_object.schedule_dict[comp][
                "transformation_matrix"] = reversal_matrix @ self.schedule_object.schedule_dict[
                    comp]["transformation_matrix"]

    def apply_interchange(self, loop1, loop2):
        for comp in self.schedule_object.comps:
            l_code = "L" + self.schedule_object.it_dict[comp][loop1]['iterator']
            self.schedule_object.repr["representation"][
                self.schedule_object.comp_indic_dict[comp]][
                    self.schedule_object.placeholders[comp][
                        l_code + "Interchanged"]] = 1
            l_code = "L" + self.schedule_object.it_dict[comp][loop2]['iterator']
            self.schedule_object.repr["representation"][
                self.schedule_object.comp_indic_dict[comp]][
                    self.schedule_object.placeholders[comp][
                        l_code + "Interchanged"]] = 1

        iterators = list(
            self.schedule_object.prog.annotations["iterators"].keys())
        if self.schedule_object.it_dict[comp][loop1]['iterator'] in iterators:
            loop_1 = iterators.index(
                self.schedule_object.it_dict[comp][loop1]['iterator'])
        elif self.schedule_object.it_dict[comp][loop1][
                'iterator'] in self.schedule_object.added_iterators:
            loop_1 = len(
                self.schedule_object.prog.annotations['iterators']
            ) + self.schedule_object.added_iterators.index(
                self.schedule_object.it_dict[comp][loop1]['iterator'])
        self.schedule_object.repr["loops_representation"][loop_1][2] = 1

        if self.schedule_object.it_dict[comp][loop2]['iterator'] in iterators:
            loop_2 = iterators.index(
                self.schedule_object.it_dict[comp][loop2]['iterator'])
        elif self.schedule_object.it_dict[comp][loop2][
                'iterator'] in self.schedule_object.added_iterators:
            loop_2 = len(
                self.schedule_object.prog.annotations['iterators']
            ) + self.schedule_object.added_iterators.index(
                self.schedule_object.it_dict[comp][loop2]['iterator'])
        self.schedule_object.repr["loops_representation"][loop_2][2] = 1

        for i in range(28):
            self.schedule_object.repr["action_mask"][i] = 0
        for i in range(56, 61):
            self.schedule_object.repr["action_mask"][i] = 0

        for comp in self.schedule_object.comps:
            dim = self.schedule_object.schedule_dict[comp]["dim"]
            interchange_matrix = np.eye(dim, dim)
            first_iter_index = loop1
            second_iter_index = loop2
            interchange_matrix[first_iter_index, first_iter_index] = 0
            interchange_matrix[second_iter_index, second_iter_index] = 0
            interchange_matrix[first_iter_index, second_iter_index] = 1
            interchange_matrix[second_iter_index, first_iter_index] = 1
            self.schedule_object.schedule_dict[comp][
                "transformation_matrices"].append(interchange_matrix)
            self.schedule_object.schedule_dict[comp][
                "transformation_matrix"] = interchange_matrix @ self.schedule_object.schedule_dict[
                    comp]["transformation_matrix"]