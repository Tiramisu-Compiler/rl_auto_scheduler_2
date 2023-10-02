import random
import time
import traceback

from config.config import Config
from env_api.core.services.compiling_service import CompilingService
from env_api.tiramisu_api import TiramisuEnvAPI
from env_api.utils.exceptions import *

if __name__ == "__main__":
    start = time.time()
    # Init global config to run the Tiramisu env
    Config.init()
    tiramisu_api = TiramisuEnvAPI(local_dataset=True)
    # Get a list of the program names in the database
    programs = tiramisu_api.get_programs()
    try:
        # Select a program randomly for example program = "function025885"
        program: str = random.choice(programs)
        # program = "function1363707"
        print("Selected function : ", program)
        # set_program(str) creates all the necessary objects to start doing operations on a program
        # it returns an encoded representation specific to the RL system
        # This representation has a shape and type of torch.Size([180])
        embedding_tensor, actions_mask = tiramisu_api.set_program(name=program)
        # There is some programs that are not supported so we need to check our representation first
        if embedding_tensor == None:
            # This means the program is unsupported you will see the source of error in the terminal when executing such programs
            # We will use this None value of the representation in the RL system to reset the programs that has unsupported loop levels , access matrices ....
            pass
        else:
            # After setting a program and checking if it is fully supported by our RL system, you can apply any action on it in any order
            # And expect to get the speedup of the whole schedule, the representation and the result of legality check of the last operation
            # (
            #     speedup,
            #     embedding_tensor,
            #     legality,
            #     actions_mask,
            # ) = tiramisu_api.interchange(loop_level1=0,loop_level2=1, env_id=7)

            # (speedup, embedding_tensor,
            #  legality,actions_mask) = tiramisu_api.skew(loop_level1=0,loop_level2=1,env_id=2)

            # (speedup, embedding_tensor,
            #  legality,actions_mask) = tiramisu_api.skew(loop_level1=1,loop_level2=2,env_id=2)
            # (speedup, embedding_tensor, legality, actions_mask) = tiramisu_api.tile2D(
            #     loop_level1=1, loop_level2=2, size_x=32, size_y=128, env_id=4
            # )
            # (
            #     speedup,
            #     embedding_tensor,
            #     legality,
            #     actions_mask,
            # ) = tiramisu_api.parallelize(loop_level=0, env_id=1)
            (
                speedup,
                embedding_tensor,
                legality,
                actions_mask,
            ) = tiramisu_api.fuse(env_id=31)

            # tiramisu_api.scheduler_service.next_branch()
            # (speedup, embedding_tensor,
            #  legality,actions_mask) = tiramisu_api.tile2D(loop_level1=0,loop_level2=1,size_x=12,size_y=12,env_id=4)
            # (speedup, embedding_tensor, legality, actions_mask,
            # ) = tiramisu_api.unroll(unrolling_factor=16, env_id=7)

            # # (speedup, embedding_tensor,
            # # legality,actions_mask) = tiramisu_api.tile3D(loop_level1=0 , loop_level2=1,loop_level3=2,
            # #     size_x=128,size_y=128,size_z=128,env_id=17)
            print("Speedup : ", speedup, " ", "Legality : ", legality)

        print("Time : ", time.time() - start)
    except Exception as e:
        print("Traceback of the error : " + 60 * "-")
        print(traceback.print_exc())
        print(80 * "-")
