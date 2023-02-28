import random, string, numpy as np
import gymnasium as gym
from gymnasium import spaces
from env_api.tiramisu_api import TiramisuEnvAPIv1
from env_api.utils.config.config import Config
from ray.rllib.env.env_context import EnvContext


class TiramisuRlAgent(gym.Env):
    def __init__(self, config: EnvContext):
        Config.init()
        self.tiramisu_api = TiramisuEnvAPIv1()
        # Define action and observation spaces
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Dict(
            {
                "prog_tree": spaces.Box(
                    -np.inf, np.inf, shape=(2000,), dtype=np.float32
                ),
                "comps": spaces.Box(
                    -np.inf, np.inf, shape=(1, 10, 922), dtype=np.float32
                ),
                "loops": spaces.Box(
                    -np.inf, np.inf, shape=(1, 32, 8), dtype=np.float32
                ),
                "comps_expr": spaces.Box(
                    -np.inf, np.inf, shape=(1, 5, 64, 11), dtype=np.float32
                ),
                "expr_loops": spaces.Box(
                    -np.inf, np.inf, shape=(16,), dtype=np.float32
                ),
            }
        )
        self.reset()

    def reset(self, seed=None, options={}):
        # Select a program randomly
        representation = None
        while representation == None:
            # There is some programs that has unsupported loop levels , acces matrices , ...
            # These programs are not supported yet so the representation will be None
            program = random.choice(self.tiramisu_api.programs)
            # The shape of representation is a tuple(((2000,), (1, 10, 922), (1, 32, 8), (1, 5, 64, 11), (16,)))
            representation = self.tiramisu_api.set_program(name=program)
        # Converting the tuple into the appropriate shape of our observation space
        self.state = self._tuple_to_dict(representation)
        self.reward = 0
        self.done = self.truncated = False
        self.info = {}
        return self.state, self.info
        # return obs

    def step(self, action):
        action =1 
        match action:
            case 0:
                speedup, representation, legality = self.tiramisu_api.parallelize(0)
                self.state = self._tuple_to_dict(representation)
                if legality:
                    self.reward = speedup
                else:
                    self.reward = 0.8
            case 1:
                # Exit case
                self.reward = 1.0
        self.done = True
        return self.state, self.reward, self.done, self.truncated, self.info

    def _tuple_to_dict(self,representation):
        return {
            "prog_tree": representation[0],
            "comps": representation[1],
            "loops": representation[2],
            "comps_expr": representation[3],
            "expr_loops": representation[4],
        }
