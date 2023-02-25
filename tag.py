import random
from env_api.scheduler.services.prediction_service import PredictionService
from env_api.tiramisu_api import TiramisuEnvAPIv1
from env_api.utils.config.config import Config
from env_api.utils.exceptions import *
from env_api.scheduler.models.json_to_tensor import *


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
        speedup , _ , legality = tiramisu_api.reverse(0)
        print(speedup, legality)
    except Exception as e :
        if isinstance(e , LoopsDepthException) : 
            print("Program has an unsupported loop level")
        elif isinstance(e , NbAccessException) :
            print("Program has an unsupported number of access matrices")
        else :
            print(e)

    (prog_tree,
    comps_repr_templates_list,
    loops_repr_templates_list,
    comps_placeholders_indices_dict,
    loops_placeholders_indices_dict,
    comps_expr_tensor,
    comps_expr_lengths) = get_representation_template(tiramisu_api.scheduler_service.schedule_object.prog.annotations,
                                tiramisu_api.scheduler_service.schedule_object.schedule_dict_tags,5)
    comps_tensor, loops_tensor =  get_schedule_representation(
                    tiramisu_api.scheduler_service.schedule_object.prog.annotations,
                    tiramisu_api.scheduler_service.schedule_object.schedule_dict_tags,
                    comps_repr_templates_list,
                    loops_repr_templates_list,
                    comps_placeholders_indices_dict,
                    loops_placeholders_indices_dict,
                    max_depth=5,)

    x = comps_tensor
    batch_size, num_comps, __dict__ = x.shape
    
    x = x.view(batch_size * num_comps, -1)
    
    (first_part, vectors, third_part) = seperate_vector(
            x, num_transformations=4, pad=False
        )
    first_part = first_part.view(batch_size, num_comps, -1)
    
    third_part = third_part.view(batch_size, num_comps, -1)
    
    tree_tensor = (prog_tree, first_part, vectors, third_part, loops_tensor, comps_expr_tensor, comps_expr_lengths)

    print(PredictionService().tags_model.forward(tree_tensor).item())

# Order of applying actions in beam search
# Fusion, [Interchange, reversal, skewing], parallelization, tiling, unrolling