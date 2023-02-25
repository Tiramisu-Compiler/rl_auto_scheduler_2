import numpy as np
import json
from env_api.core.services.converting_service import ConvertService

_MAX_DEPTH = 6

class Schedule:
    def __init__(self, program):
        self.depth = 0
        self.schedule_str = ""
        self.is_interchaged = False
        self.is_tiled = False
        self.is_unrolled = False
        self.is_skewed = False
        self.is_parallelized = False
        self.is_reversed = False
        self.prog = program
        self.comps = self.prog.comp_name
        self.repr = {}
        self.templates = {}
        self.it_dict = {}
        self.schedule_dict = {}
        self.schedule_dict_tags = {}
        self.common_it = []
        self.__calculate_common_it()
        self.__init_schedule_dict()
        self.__init_schedule_dict_tags()
        self.__init_representation()
        self.__set_action_mask()
        self.__init_templates()
        self.__extend_representation()
        self.__form_iterators_dict()
        self.__create_program_tree()

    def __calculate_common_it(self):
        if len(self.comps) != 1:  # Multi-computation program
            # comps_it is a list of lists of iterators of computations
            self.comps_it = []
            for comp in self.comps:
                self.comps_it.append(
                    self.prog.annotations["computations"][comp]["iterators"]
                )
            self.common_it = self.comps_it[0]
            for comp_it in self.comps_it[1:]:
                self.common_it = [it for it in comp_it if it in self.common_it]
        else:  # A single comp program
            self.common_it = self.prog.annotations["computations"][self.comps[0]][
                "iterators"
            ]

    def __init_schedule_dict(self):
        self.schedule_dict["fusions"] = None
        for comp in self.comps:
            dim = len(self.prog.annotations["computations"][comp]["iterators"])
            self.schedule_dict[comp] = dict()
            self.schedule_dict[comp]["dim"] = dim
            self.schedule_dict[comp]["transformation_matrix"] = np.eye(
                dim, dim)
            self.schedule_dict[comp]["transformation_matrices"] = [
                np.eye(dim, dim)]
            self.schedule_dict[comp]["parallelized_dim"] = None
            self.schedule_dict[comp]["unrolling_factor"] = None
            self.schedule_dict[comp]["tiling"] = None
        self.schedule_dict["tree_structure"] = ConvertService.get_tree_structure(
            self.prog.annotations
        )


    def __init_schedule_dict_tags(self):
        self.schedule_dict_tags["fusions"] = None
        for comp in self.comps:
            self.schedule_dict_tags[comp] = {
                "tiling": {},
                "unrolling_factor": None,
                "parallelized_dim": None,
                "shiftings": None,
                "transformations_list": []
            }
        #TODO : Check for the multi root solution
        self.schedule_dict_tags["tree_structure"] = {
            "roots": [ConvertService.get_tree_structure(self.prog.annotations)]}

    def __init_representation(self):
        self.repr["representation"] = np.empty((0, 1052), np.float32)
        self.repr["loops_representation"] = np.empty((0, 26), np.float32)
        self.repr["child_list"] = np.empty((0, 11), np.float32)
        self.repr["has_comps"] = np.empty((0, 12), np.float32)
        self.repr["prog_tree"] = np.empty((0, 5000), np.float32)
        self.repr["computations_indices"] = np.empty((0, 5), np.float32)
    
    def __set_action_mask(self):
        match len(self.common_it):
            case 5:
                self.repr["action_mask"] = np.array(
                    [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,],
                    dtype=np.float32,
                )
            case 4:
                self.repr["action_mask"] = np.array(
                    [1,1,1,0,0,0,0,1,1,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,0,0,0,0,1,1,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,1,1,1,1,1,1,],
                    dtype=np.float32,
                )
            case 3:

                self.repr["action_mask"] = np.array(
                    [1,1,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,1,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,1,1,1,1,1,1,],
                    dtype=np.float32,
                )
            case 2:
                self.repr["action_mask"] = np.array(
                    [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,0,1,1,1,1,0,0,0,0,0,0,1,1,1,1,1,1,],
                    dtype=np.float32,
                )
            case 1:
                self.repr["action_mask"] = np.array(
                    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,0,0,1,0,1,0,0,0,0,0,0,0,1,1,1,1,1,1,],
                    dtype=np.float32,
                )
        if len(self.comps) == 1:
            np.put(self.repr["action_mask"], [56, 57, 58, 59, 60],[0, 0, 0, 0, 0])

            
    def __init_templates(self):
        templates = ConvertService.get_schedule_template(
             self.prog.annotations,
             self.schedule_dict,
             max_depth=_MAX_DEPTH - 1)
        (
            self.templates["prog_tree"],
            self.templates["comps_repr_templates_list"],
            self.templates["loops_repr_templates_list"],
            self.templates["comps_placeholders_indices_dict"],
            self.templates["loops_placeholders_indices_dict"],
        ) = templates

    def __extend_representation(self):
        # TODO : Check if it is necessary to have these values as attributes in schedule object 
        (self.prog_rep,
        self.comps_placeholders,
        self.comp_indic_dict) = ConvertService.get_computations_representation(self.prog.annotations)
        
        self.schedule_dict["fusions"] = []
        self.placeholders = self.comps_placeholders
        self.added_iterators = []

        for i in range(5):
            if i >= len(self.prog_rep):
                self.repr["representation"] = np.vstack(
                    [self.repr["representation"],
                     np.zeros(1052)])
            else:
                self.repr["representation"] = np.vstack([
                    self.repr["representation"],
                    np.array([self.prog_rep[i]], dtype=np.float32)
                ])
        iterators = list(self.prog.annotations["iterators"].keys())

        for i in range(len(iterators)):

            loop_repr = []
            loop_repr.append(
                self.prog.annotations['iterators'][iterators[i]]['lower_bound'])
            loop_repr.append(
                self.prog.annotations['iterators'][iterators[i]]['upper_bound'])
            loop_repr.extend([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
            loop_log_rep = list(np.log1p(loop_repr))
            loop_repr.extend(loop_log_rep)
            self.repr["loops_representation"] = np.vstack(
                [self.repr["loops_representation"],
                 np.array([loop_repr])])

            childs_indexes = [
                iterators.index(child) for child in
                self.prog.annotations['iterators'][iterators[i]]['child_iterators']
            ]
            if len(childs_indexes) != 11:
                for j in range(11 - len(childs_indexes)):
                    childs_indexes.append(-1)
            self.repr["child_list"] = np.vstack(
                [self.repr["child_list"],
                 np.array([childs_indexes])])

            if self.prog.annotations['iterators'][
                    iterators[i]]['computations_list'] != []:
                self.repr['has_comps'] = np.append(self.repr['has_comps'], 1)
            else:
                self.repr['has_comps'] = np.append(self.repr['has_comps'], 0)

            computations_list = list(self.prog.annotations['computations'].keys())
            loop_comps = [
                computations_list.index(comp)
                for comp in self.prog.annotations['iterators'][iterators[i]]
                ['computations_list']
            ]
            if len(loop_comps) != 5:
                for j in range(5 - len(loop_comps)):
                    loop_comps.append(-1)
            self.repr["computations_indices"] = np.vstack(
                [self.repr["computations_indices"],
                 np.array([loop_comps])])

        for i in range(15 - len(self.prog.annotations["iterators"])):
            loop_repr = np.full(26, -1)
            self.repr["loops_representation"] = np.vstack(
                [self.repr["loops_representation"], loop_repr])

        for i in range(12 - len(self.prog.annotations["iterators"])):
            self.repr["child_list"] = np.vstack(
                [self.repr["child_list"],
                 np.full(11, -1)])
            self.repr['has_comps'] = np.append(self.repr['has_comps'], 0)
            self.repr["computations_indices"] = np.vstack(
                [self.repr["computations_indices"],
                 np.full(5, -1)])

    def __form_iterators_dict(self):
        for comp in self.comps:
            comp_it_dict = {}
            iterators = list(
                self.prog.annotations["computations"][comp]["iterators"])

            for i in range(len(iterators)):
                comp_it_dict[i] = {}
                comp_it_dict[i]['iterator'] = iterators[i]
                comp_it_dict[i]['lower_bound'] = self.prog.annotations['iterators'][
                    iterators[i]]['lower_bound']
                comp_it_dict[i]['upper_bound'] = self.prog.annotations['iterators'][
                    iterators[i]]['upper_bound']

            self.it_dict[comp] = comp_it_dict
    
    def __create_program_tree(self):
        max_size = 5000
        string = json.dumps(self.templates["prog_tree"])
        padded_string = string + (max_size - len(string))*"_"
        self.repr["prog_tree"] = np.array(list(padded_string),"U1").view(np.float32)