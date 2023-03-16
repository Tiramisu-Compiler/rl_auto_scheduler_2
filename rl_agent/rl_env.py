import random, numpy as np,math
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
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(180,))
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

    def step(self, action):

        speedup , embedded_tensor,  legality = self.apply_action(action)
        if(legality and not self.done):
            self.state = embedded_tensor.numpy()

        self.reward = math.log(speedup,4)

        return self.state, self.reward, self.done, self.truncated, self.info


    def apply_action(self,action):
        if(action==0):
            speedup, embedded_tensor, legality = self.tiramisu_api.parallelize(action)
            self.done = True
        else:
            # Exit case
            speedup , embedded_tensor,  legality = (1 ,None, True)
            self.done = True

        
        return speedup, embedded_tensor, legality
