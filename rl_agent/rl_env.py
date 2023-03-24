import random, numpy as np, math
import gymnasium as gym
from gymnasium import spaces
from ray.rllib.env.env_context import EnvContext

from env_api.utils.config.config import Config


class TiramisuRlEnv(gym.Env):
    def __init__(self, config: EnvContext):
        Config.init()
        # TODO : rechek this instance of tiramisu_api .(for the dataset)
        self.tiramisu_api = config["tiramisu_api"]
        # Define action and observation spaces
        self.action_space = spaces.Discrete(27)
        # self.observation_space = spaces.Box(-np.inf, np.inf, shape=(180, ))
        self.observation_space = spaces.Dict({
            "embedding":
            spaces.Box(-np.inf, np.inf, shape=(180, )),
            "actions_mask":
            spaces.Box(0, 1, shape=(27, ))
        })

        self.reset()

    def reset(self, seed=None, options={}, index=-1):
        # Select a program randomly
        embedded_tensor = None
        # while embedded_tensor == None:
        # There is some programs that has unsupported loop levels , acces matrices , ...
        # These programs are not supported yet so the embedded_tensor will be None
        if (index == -1):
            program = random.choice(self.tiramisu_api.programs)
        else:
            program = self.tiramisu_api.programs[
                index]  #random.choice(self.tiramisu_api.programs)
        # The shape of embedded_tensor : (180,)
        embedded_tensor, actions_mask = self.tiramisu_api.set_program(
            name=program)
        if embedded_tensor == None:
            self.state = {
                "embedding": np.ones(180),
                "actions_mask": np.zeros(27)
            }
            return self.state, {}
        # Converting to numpy array
        self.state = {
            "embedding": embedded_tensor.numpy(),
            "actions_mask": actions_mask
        }
        self.previous_speedup = 1
        self.reward = 1
        self.done = self.truncated = False
        self.info = {}
        self.steps = 0
        return self.state, self.info

    def step(self, action):
        self.steps += 1
        speedup, embedded_tensor, legality, actions_mask = self.apply_flattened_action(
            action=action)
        new_speedup = 1
        if (legality and not self.done):
            self.state = {
                "embedding": embedded_tensor.numpy(),
                "actions_mask": actions_mask
            }
            new_speedup = speedup / self.previous_speedup
            self.previous_speedup = speedup

        self.reward = math.log(new_speedup, 4)

        if (self.steps == 10):
            self.done = True
        return self.state, self.reward, self.done, self.truncated, self.info

    def apply_flattened_action(self, action):
        if (action <= 1):
            # For parallelization 0 and 1
            speedup, embedded_tensor, legality, actions_mask = self.tiramisu_api.parallelize(
                loop_level=action, env_id=action)
        elif (action <= 3):
            loop_level = action - 2
            # Skewing 0,1 and 1,2
            speedup, embedded_tensor, legality, actions_mask = self.tiramisu_api.skew(
                loop_level1=loop_level,
                loop_level2=loop_level + 1,
                env_id=action)
        elif (action <= 6):
            # Unrolling 4 , 8 , 16
            factor = action - 2
            speedup, embedded_tensor, legality, actions_mask = self.tiramisu_api.unroll(
                unrolling_factor=2**factor, env_id=action)
        elif (action <= 11):
            loop_level = action - 7
            # Reversal from 0 to 4
            speedup, embedded_tensor, legality, actions_mask = self.tiramisu_api.reverse(
                loop_level=loop_level, env_id=action)
        elif (action <= 15):
            loop_level = action - 12
            # Tiling 2D from 0,1 to 3,4
            speedup, embedded_tensor, legality, actions_mask = self.tiramisu_api.tile2D(
                loop_level1=loop_level,
                loop_level2=loop_level + 1,
                size_x=32,
                size_y=32,
                env_id=action)
        elif (action <= 18):
            loop_level = action - 16
            # Tiling 3d from 0,1,2 to 2,3,4
            speedup, embedded_tensor, legality, actions_mask = self.tiramisu_api.tile3D(
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
            speedup, embedded_tensor, legality, actions_mask = self.tiramisu_api.interchange(
                loop_level1=loop_level,
                loop_level2=loop_level + 1,
                env_id=action)
        elif (action <= 25):
            loop_level = action - 23
            # Interchange of loops (0,2) , (1,3) and (2,4)
            speedup, embedded_tensor, legality, actions_mask = self.tiramisu_api.interchange(
                loop_level1=loop_level,
                loop_level2=loop_level + 2,
                env_id=action)
        else:
            # Exit case
            speedup, embedded_tensor, legality, actions_mask = (1, None, True,
                                                                np.ones(27))
            self.done = True

        return speedup, embedded_tensor, legality, actions_mask
