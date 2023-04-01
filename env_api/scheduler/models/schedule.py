import numpy as np
from env_api.core.services.converting_service import ConvertService
from env_api.scheduler.models.representation import Representation
from env_api.scheduler.models.action import *

class Schedule:
    def __init__(self, program):
        self.schedule_str = ""
        self.transformed = 0
        self.prog = program
        self.comps = self.prog.comps
        self.repr : Representation = None
        self.it_dict = {}
        self.branches = []
        self.schedule_dict = {}
        self.common_it = []
        self.__calculate_common_it()
        self.__init_schedule_dict_tags()
        self.__init_representation()
        self.__set_action_mask()
        self.__form_iterators_dict()
        self.__form_branches()


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
        self.repr.action_mask = np.zeros(27)

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
            
    def __form_branches(self):
        branchs = []
        iterators = self.prog.annotations["iterators"]
        for iterator in iterators.keys(): 
            if iterators[iterator]["computations_list"]:
                branchs.append({
                    "comps" : iterators[iterator]["computations_list"],
                    "iterators" : self.prog.annotations["computations"][iterators[iterator]["computations_list"][0]]["iterators"]
                })
        self.branches = branchs

    
    def update_actions_mask(self, action : Action,applied : bool,beam_search_order= False):
        # Whether an action is legal or not we should mask it to not use it again
        self.repr.action_mask[action.env_id] = 1

        if applied :
            # if the action is legal and applied we need to mask similar action when it comes 
            # to Unrilling , skewing and parallelization because these action are applied once 
            if isinstance(action, Unrolling) :
                self.repr.action_mask[4:7] = 1
            if isinstance(action, Tiling) :
                self.repr.action_mask[12:19] = 1
            if isinstance(action, Parallelization)  : 
                self.repr.action_mask[0:2] = 1
            # The other case is for skewing , reversal and interchange 
            # for these actions we are allowed to apply them in any order under the condition of not 
            # surpassing 4 times of applying them.
            if self.transformed == 4 :
                # Skewing
                self.repr.action_mask[2:4] = 1
                # Reversal
                self.repr.action_mask[7:12] = 1
                # Interchange
                self.repr.action_mask[19:26] = 1
                
            if beam_search_order : 
                self.apply_beam_search_conditions(action=action)

        return self.repr.action_mask
    
    def apply_beam_search_conditions(self, action : Action):
        # The order of actions in beam search :
        # Fusion, [Interchange, reversal, skewing], parallelization, tiling, unrolling
        if (isinstance(action,Unrolling) or isinstance(action,Tiling) or isinstance(action,Parallelization)):
            # Skewing
            self.repr.action_mask[2:4] = 1
            # Reversal
            self.repr.action_mask[7:12] = 1
            # Interchange
            self.repr.action_mask[19:26] = 1

            if (isinstance(action,Tiling)) : 
                # Parallelization
                self.repr.action_mask[0:2] = 1

            elif (isinstance(action,Unrolling)):
                # Parallelization
                self.repr.action_mask[0:2] = 1
                # Tiling
                self.repr.action_mask[12:19] = 1
