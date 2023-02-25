import random
from env_api.core.services.converting_service import ConvertService
from env_api.scheduler.models.representation import Representation
from env_api.scheduler.services.prediction_service import PredictionService
from env_api.tiramisu_api import TiramisuEnvAPIv1
from env_api.utils.config.config import Config
from env_api.utils.exceptions import *


if __name__ == "__main__":
    # Init global config to run the env
    Config.init()
    tiramisu_api = TiramisuEnvAPIv1()
    # init database
    tiramisu_api.init_dataset_service(dataset_path = 'env_api/data/dataset/',copy_path = 'env_api/data/copy/')
    # Get a list of the program names in the database 
    programs =  tiramisu_api.get_programs()
    program ='function000031' #random.choice(programs)
    print("Selected function : ", program)
    # tiramisu_api.set_program creates all the necessary objects to do operations on a program
    tree_tensor = tiramisu_api.set_program(name=program)
    # After setting a program you can apply any action on it in any order
    #  and expect to get the speedup of the whole schedule, the representation
    #  and the result of legality check of the last operation
    # speedup , _ , legality = tiramisu_api.reverse(0)
    print(tiramisu_api.scheduler_service.get_schedule_dict())
    speedup , _ , legality = tiramisu_api.parallelize(0)
    print(speedup, legality)
    try : 
        # Select a program randomly
        # program = random.choice(programs)
        print("Selected function : ", program)
        # tiramisu_api.set_program creates all the necessary objects to do operations on a program
        tree_tensor = tiramisu_api.set_program(name=program)
        # After setting a program you can apply any action on it in any order
        #  and expect to get the speedup of the whole schedule, the representation
        #  and the result of legality check of the last operation
        speedup , _ , legality = tiramisu_api.parallelize(1)
        print(tiramisu_api.scheduler_service.get_schedule_dict())
        print(speedup, legality)
    except Exception as e :
        if isinstance(e , LoopsDepthException) : 
            print("Program has an unsupported loop level")
        elif isinstance(e , NbAccessException) :
            print("Program has an unsupported number of access matrices")
        else :
            print(e)



# Order of applying actions in beam search
# Fusion, [Interchange, reversal, skewing], parallelization, tiling, unrolling