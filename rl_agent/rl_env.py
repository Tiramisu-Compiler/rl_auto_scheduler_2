import random, math, numpy as np
import gymnasium as gym
from gymnasium import spaces
from env_api.tiramisu_api import TiramisuEnvAPIv1
from env_api.utils.config.config import Config
from ray.rllib.env.env_context import EnvContext


class TiramisuRlEnv(gym.Env):
    def __init__(self, config: EnvContext):
        Config.init()
        self.tiramisu_api = TiramisuEnvAPIv1()
        # Define action and observation spaces
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(-np.inf,np.inf,shape=(180,))
        self.reset()

    def reset(self, seed=None, options={}):
        # Select a program randomly
        embedded_tensor = None
        while embedded_tensor == None:
            # There is some programs that has unsupported loop levels , acces matrices , ...
            # These programs are not supported yet so the embedded_tensor will be None
            program = random.choice(self.tiramisu_api.programs)
            # The shape of embedded_tensor : (180,)
            embedded_tensor = self.tiramisu_api.set_program(name=program)
        # Converting to numpy array
        self.state = embedded_tensor.numpy()
        self.reward = 1
        self.done = self.truncated = False
        self.info = {}
        return self.state, self.info
        # return obs

    def step(self, action):
        match action:
            case 0:
                speedup, embedded_tensor, legality = self.tiramisu_api.parallelize(0)
                if legality:
                    self.state = embedded_tensor.numpy()
                    self.reward = speedup
                else:
                    self.reward = 0.5
            case 1:
                # Exit case
                self.reward = 1.0
        self.done = True
        return self.state, math.log(self.reward,1.5), self.done, self.truncated, self.info
