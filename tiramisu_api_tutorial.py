import random
import traceback
from env_api.core.services.converting_service import ConvertService
from env_api.tiramisu_api import TiramisuEnvAPIv1
from env_api.utils.config.config import Config
from env_api.utils.exceptions import *


if __name__ == "__main__":
    # Init global config to run the Tiramisu env
    Config.init()
    tiramisu_api = TiramisuEnvAPIv1()
    # Get a list of the program names in the database 
    programs = tiramisu_api.get_programs()
    try : 
        # Select a program randomly for example program = "function000028"
        program : str = random.choice(programs)
        print("Selected function : ", program)
        # set_program(str) creates all the necessary objects to start doing operations on a program
        # it returns an encoded representation specific to the RL system 
        # This representation has a shape of : ((2000,), (1, 10, 922), (1, 32, 8), (1, 5, 64, 11), (16,)) of types (dtype('float32'), dtype('float32'), dtype('float32'), dtype('float32'), dtype('float32') respectively
        # These vectors has a fixed shape because they are padded, they are all numerical vectors that represents different aspects 
        # of the program like the program_tree , comps representation , loop representation .... , 
        # If you want to get the original tree structure from this representation you can call `TiramisuEnvAPIv1.get_decoded_rl_repr(*representation)`
        representation = tiramisu_api.set_program(name=program)
        # There is some programs that are not supported so we need to check our representation first 
        if(representation == None):
            # This means the program is unsupported you will see the source of error in the terminal when executing such programs 
            # We will use this None value of the representation in the RL system to reset the programs that has unsupported loop levels , access matrices ....
            pass
        else : 
            # After setting a program and checking if it is fully supported by our RL system, you can apply any action on it in any order
            # And expect to get the speedup of the whole schedule, the representation and the result of legality check of the last operation
            speedup , representation , legality = tiramisu_api.parallelize(loop_level= 0)
            print("Speedup : ",speedup ," ","Legality : ", legality)
    except Exception as e :
        print("Traceback of the error : " + 60 * "-")
        print(traceback.print_exc())
        print(80 * "-")
