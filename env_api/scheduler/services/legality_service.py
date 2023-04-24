from env_api.core.models.optim_cmd import OptimizationCommand
from env_api.core.services.compiling_service import CompilingService
from env_api.core.services.converting_service import ConvertService
from env_api.scheduler.models.action import *
from env_api.scheduler.models.schedule import Schedule


class LegalityService:
    def __init__(self):
        '''
        The legality service is responsible to evaluate the legality of tiramisu programs given a specific schedule
        '''
        pass 

    def is_action_legal(self,schedule_object:Schedule, action: Action):
        """
        Checks the legality of action
        input :
            - an action that represents an optimization from the 7 types : Parallelization,Skewing,Interchange,Fusion,Reversal,Tiling,Unrolling
        output :
            - legality_check : bool
        """

        # Check first if the iterator(s) is(are) included in the available in the current iterators
        # If not then the action is illegal by default 
        exceeded_iterators = self.check_iterators(schedule_object=schedule_object,action=action)
        if exceeded_iterators : return False

        # TODO : remove this condition when we apply the new method
        if isinstance(action, Unrolling):
            # In this case we unroll all the computations
            action.comps = schedule_object.comps
            # We look for the last iterator of each computation and save it in the params
            unrolling_factor = action.params[0]
            action.params = {}
            for comp in schedule_object.it_dict:
                loop_level = len(schedule_object.it_dict[comp].keys()) - 1
                action.params[comp] = [loop_level, unrolling_factor]

        # For skewing action we need first to get the skewing params : a list of 2 int
        elif isinstance(action, Skewing):
            # construct the schedule string to check if it is legal or not
            schdule_str = ConvertService.build_sched_string(schedule_object.schedule_list)
            action.comps = schedule_object.comps
            # check if results of skewing solver exist in the dataset
            if schdule_str in schedule_object.prog.schedules_solver:
                factors = schedule_object.prog.schedules_solver[schdule_str]

            else:

                if (not schedule_object.prog.original_str):
                    # Loading function code lines
                    schedule_object.prog.load_code_lines()
                # Call the skewing solver
                factors = CompilingService.call_skewing_solver(
                    schedule_object=schedule_object,
                    optim_list=schedule_object.schedule_list,
                    params=action.params)

                # Save the results of skewing solver in the dataset
                schedule_object.prog.schedules_solver[schdule_str] = factors
            if (factors == None):
                return False
            else:
                action.params.extend(factors)
        else:
            action.comps = schedule_object.comps
        # Assign the requested comps to the action
        optim_command = OptimizationCommand(action)
        # Add the command to the array of schedule
        schedule_object.schedule_list.append(optim_command)
        # Building schedule string
        schdule_str = ConvertService.build_sched_string(schedule_object.schedule_list)
        # Check if the action is legal or no to be applied on schedule_object.prog
        # prog.schedules_legality only has data when it is fetched from the offline dataset so no need to compile to get the legality
        if schdule_str in schedule_object.prog.schedules_legality:
            legality_check = int(
                schedule_object.prog.schedules_legality[schdule_str])
        else:
            # To run the legality we need the original function code to generate legality code
            if (not schedule_object.prog.original_str):
                # Loading function code lines
                schedule_object.prog.load_code_lines()
            try:
                legality_check = int(
                    CompilingService.compile_legality(
                        schedule_object=schedule_object,
                        optims_list=schedule_object.schedule_list))

                # Saving the legality of the new schedule
                schedule_object.prog.schedules_legality[schdule_str] = (
                    legality_check == 1)

            except ValueError as e:
                legality_check = 0
                print("Legality error :", e)
        if legality_check != 1:
            schedule_object.schedule_list.pop()
            schdule_str = ConvertService.build_sched_string(schedule_object.schedule_list)
        schedule_object.schedule_str = schdule_str
        return legality_check == 1

    
    def check_iterators(self, schedule_object , action):
        included_iterator = True
        # Before checking legality from dataset or by compiling , we see if the iterators are included in the common iterators
        if (not isinstance(action, Unrolling)):
            num_iter = schedule_object.common_it.__len__()
            if isinstance(action, Tiling):
                # Becuase the second half of action.params contains tiling size, so we need only the first half of the vector
                params = action.params[:len(action.params) // 2]
            else:
                params = action.params
            for param in params:
                if param >= num_iter:
                    included_iterator = False 

        if isinstance(action, Fusion):
            if (len(schedule_object.comps) <= 1):
                # If the program has a single computation , then fusion is illegal
                included_iterator = False 
            action.comps = [
                comp for comp in schedule_object.it_dict
                if action.params[0] in schedule_object.it_dict[comp]
            ]
            if (len(action.comps) <= 1):
                # If there are many computations but , at the fusion loop level there is less than 2 computations
                # then fusion will be illegal
                included_iterator = False 
        
        return not included_iterator