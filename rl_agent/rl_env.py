import numpy as np, math,ray
import gymnasium as gym
from gymnasium import spaces
from ray.rllib.env.env_context import EnvContext

from config.config import Config
from rllib_ray_utils.dataset_actor import DatasetActor
from env_api.tiramisu_api import TiramisuEnvAPI


class TiramisuRlEnv(gym.Env):
    def __init__(self, config: EnvContext):
        Config.config = config["config"]
        # local_dataset=False => means that we are reading data from external source than the dataservice implemented in
        # TiramisuEnvAPI, this data is the annotations of a function + the leglaity of schedules
        self.tiramisu_api = TiramisuEnvAPI(local_dataset=False)
        self.dataset_actor : DatasetActor = config["dataset_actor"]
        # Define action and observation spaces
        self.action_space = spaces.Discrete(27)
        self.observation_space = spaces.Dict({
            "embedding":
            spaces.Box(-np.inf, np.inf, shape=(180, )),
            "actions_mask":
            spaces.Box(0, 1, shape=(27, ))
        })
        self.reset()

    def reset(self, seed=None, options={}):
        embedded_tensor = None
        # Select a program randomly
        while embedded_tensor == None:
            # There is some programs that has unsupported loop levels , acces matrices , ...
            # These programs are not supported yet so the embedded_tensor will be None
            # program = random.choice(self.tiramisu_api.programs)
            self.program_name , data = ray.get(self.dataset_actor.get_next_function.remote())
            # The shape of embedded_tensor : (180,)
            # Shape pf actions mask : (27,)
            embedded_tensor, actions_mask = self.tiramisu_api.set_program(
                name=self.program_name,
                data=data)
            
        self.state = {
            # Converting Tensor to numpy array
            "embedding": embedded_tensor.numpy(),
            "actions_mask": actions_mask
        }
        self.previous_speedup = self.reward = 1
        self.done = self.truncated = False
        self.info = {}
        self.steps = 0
        return self.state, self.info

    def step(self, action):
        self.steps += 1
        speedup, embedded_tensor, legality, actions_mask = self.apply_flattened_action(
            action=action)
        instant_speedup = 1
        if (legality and not self.done):
            self.state = {
                "embedding": embedded_tensor.numpy(),
                "actions_mask": actions_mask
            }
            # If the action is legal , we divide the speedup of new sequence {A_0 .. A_i+1} by the speedup of 
            # the previous Sequnce {A_0 .. A_i} to get the speedup of the action {A_i+1}
            instant_speedup = speedup / self.previous_speedup
            self.previous_speedup = speedup

        self.reward = math.log(instant_speedup, 4)

        if (self.steps == 20):
            self.done = True
        return self.state, self.reward, self.done, self.truncated, self.info

    def apply_flattened_action(self, action):
        if (action <= 1):
            # For parallelization 0 and 1
            speedup, embedded_tensor, legality, actions_mask,legality_schedule = self.tiramisu_api.parallelize(
                loop_level=action, env_id=action)
        elif (action <= 3):
            loop_level = action - 2
            # Skewing 0,1 and 1,2
            speedup, embedded_tensor, legality, actions_mask,legality_schedule = self.tiramisu_api.skew(
                loop_level1=loop_level,
                loop_level2=loop_level + 1,
                env_id=action)
        elif (action <= 6):
            # Unrolling 4 , 8 , 16
            factor = action - 2
            speedup, embedded_tensor, legality, actions_mask,legality_schedule = self.tiramisu_api.unroll(
                unrolling_factor=2**factor, env_id=action)
        elif (action <= 11):
            loop_level = action - 7
            # Reversal from 0 to 4
            speedup, embedded_tensor, legality, actions_mask,legality_schedule = self.tiramisu_api.reverse(
                loop_level=loop_level, env_id=action)
        elif (action <= 15):
            loop_level = action - 12
            # Tiling 2D from 0,1 to 3,4
            speedup, embedded_tensor, legality, actions_mask,legality_schedule = self.tiramisu_api.tile2D(
                loop_level1=loop_level,
                loop_level2=loop_level + 1,
                size_x=32,
                size_y=32,
                env_id=action)
        elif (action <= 18):
            loop_level = action - 16
            # Tiling 3d from 0,1,2 to 2,3,4
            speedup, embedded_tensor, legality, actions_mask,legality_schedule = self.tiramisu_api.tile3D(
                loop_level1=loop_level,
                loop_level2=loop_level + 1,
                loop_level3=loop_level + 2,
                size_x=32,
                size_y=32,
                size_z=32,
                env_id=action)
        elif (action <= 22):
            loop_level = action - 19
            # Interchange of loops 0,1 ... 4,5
            speedup, embedded_tensor, legality, actions_mask,legality_schedule = self.tiramisu_api.interchange(
                loop_level1=loop_level,
                loop_level2=loop_level + 1,
                env_id=action)
        elif (action <= 25):
            loop_level = action - 23
            # Interchange of loops (0,2) , (1,3) and (2,4)
            speedup, embedded_tensor, legality, actions_mask,legality_schedule = self.tiramisu_api.interchange(
                loop_level1=loop_level,
                loop_level2=loop_level + 2,
                env_id=action)
        else:
            # Exit case
            speedup, embedded_tensor, legality, actions_mask,legality_schedule = (1, None, True,
                                                                np.zeros(27),None)
            self.done = True
        
        # legality_schedule is of type dict, if it is not None, we update our dataset with new discovered legalities 
        # of the program
        if legality_schedule :
            self.dataset_actor.update_dataset.remote(
                self.program_name , legality_schedule
            )

        return speedup, embedded_tensor, legality, actions_mask
