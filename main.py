import random
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
    programs = tiramisu_api.get_programs()
    try : 
        # Select a program randomly
        program = random.choice(programs)
        print("Selected function : ", program)
        # tiramisu_api.set_program creates all the necessary objects to do operations on a program
        tiramisu_api.set_program(name=program)
        # After setting a program you can apply any action on it in any order
        #  and expect to get the speedup of the whole schedule, the representation
        #  and the result of legality check of the last operation
        print(tiramisu_api.scheduler_service.schedule_object.prog.annotations)
        speedup , representation , legality = tiramisu_api.reverse(loop_level= 0)
        print(speedup , legality)
        speedup , representation , legality = tiramisu_api.parallelize(loop_level= 1)
        print(speedup , legality)
    except Exception as e :
        if isinstance(e , LoopsDepthException) : 
            print("Program has an unsupported loop level")
        elif isinstance(e , NbAccessException) :
            print("Program has an unsupported number of access matrices")
        print(e)
    

    