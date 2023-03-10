import random, numpy as np
import gymnasium as gym
from gymnasium import spaces
from ray.rllib.env.env_context import EnvContext


class TiramisuRlEnv(gym.Env):
    def __init__(self, config: EnvContext):
        
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
        match action:
            case 0:
                speedup, embedded_tensor, legality = self.tiramisu_api.parallelize(0)
                if legality:
                    self.state = embedded_tensor.numpy()
                    if(speedup >1):
                        self.reward = 1
                    else : 
                        self.reward = -1
                else :
                    self.reward = -1
            case 1:
                # Exit case
                speedup , legality = (1 , False)
                self.reward = 0

        # self.reward = self.calculate_reward(speedup=speedup,legality=legality)
        self.done = True

        return self.state, self.reward, self.done, self.truncated, self.info

    # def calculate_reward(self,speedup: float, legality: bool):

    #     return reward

