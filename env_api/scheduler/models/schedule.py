import numpy as np
from env_api.core.services.converting_service import ConvertService
from env_api.scheduler.models.representation import Representation

_MAX_DEPTH = 6

class Schedule:
    def __init__(self, program):
        # TODO : fill this dict with the schedule applied on the comps
        self.schedule_str = {}
        self.is_interchaged = False
        self.is_tiled = False
        self.is_unrolled = False
        self.is_skewed = False
        self.is_parallelized = False
        self.is_reversed = False
        self.prog = program
        self.comps = self.prog.comp_name
        self.repr : Representation = None
        self.it_dict = {}
        self.schedule_dict = {}
        self.common_it = []
        self.__calculate_common_it()
        self.__init_schedule_dict_tags()
        self.__init_representation()
        self.__set_action_mask()
        self.__form_iterators_dict()


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



    def __init_schedule_dict_tags(self):
        self.schedule_dict["fusions"] = None
        for comp in self.comps:
            self.schedule_dict[comp] = {
                "tiling": {},
                "unrolling_factor": None,
                "parallelized_dim": None,
                "shiftings": None,
                "transformations_list": []
            }
        #TODO : ReCheck for the multi root solution
        self.schedule_dict["tree_structure"] = {
            "roots": [ConvertService.get_tree_structure(self.prog.annotations)]}

    def __init_representation(self):
        self.repr = Representation(*ConvertService.get_representation_template(self.prog.annotations,self.schedule_dict))
    
    def __set_action_mask(self):
        match len(self.common_it):
            case 5:
                self.repr.action_mask = np.array(
                    [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,],
                    dtype=np.float32,
                )
            case 4:
                self.repr.action_mask = np.array(
                    [1,1,1,0,0,0,0,1,1,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,0,0,0,0,1,1,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,1,1,1,1,1,1,],
                    dtype=np.float32,
                )
            case 3:
                self.repr.action_mask = np.array(
                    [1,1,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,1,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,1,1,1,1,1,1,],
                    dtype=np.float32,
                )
            case 2:
                self.repr.action_mask = np.array(
                    [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,0,1,1,1,1,0,0,0,0,0,0,1,1,1,1,1,1,],
                    dtype=np.float32,
                )
            case 1:
                self.repr.action_mask = np.array(
                    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,0,0,1,0,1,0,0,0,0,0,0,0,1,1,1,1,1,1,],
                    dtype=np.float32,
                )
        if len(self.comps) == 1:
            np.put(self.repr.action_mask, [56, 57, 58, 59, 60],[0, 0, 0, 0, 0])

    def __form_iterators_dict(self):
        for comp in self.comps:
            comp_it_dict = {}
            iterators = list(self.prog.annotations["computations"][comp]["iterators"])
            for i in range(len(iterators)):
                comp_it_dict[i] = {}
                comp_it_dict[i]['iterator'] = iterators[i]
                comp_it_dict[i]['lower_bound'] = self.prog.annotations['iterators'][
                    iterators[i]]['lower_bound']
                comp_it_dict[i]['upper_bound'] = self.prog.annotations['iterators'][
                    iterators[i]]['upper_bound']
            self.it_dict[comp] = comp_it_dict
    
