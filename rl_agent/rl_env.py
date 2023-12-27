import json
import math
import os
from collections import OrderedDict

import grpc
import gymnasium as gym
import numpy as np
import ray
import torch
from gymnasium import spaces
from ray.rllib.env.env_context import EnvContext

from config.config import Config
from env_api.tiramisu_api import TiramisuEnvAPI
from grpc_server.dataset_grpc_server.grpc_files import (
    tiramisu_function_pb2,
    tiramisu_function_pb2_grpc,
)


class TiramisuRlEnv(gym.Env):
    def __init__(self, config: EnvContext):
        Config.config = config["config"]
        # local_dataset=False => means that we are reading data from external source than the dataservice implemented in
        # TiramisuEnvAPI, this data is the annotations of a function + the leglaity of schedules
        self.tiramisu_api = TiramisuEnvAPI(local_dataset=False)
        # self.dataset_actor: DatasetActor = config["dataset_actor"]
        # Define action and observation spaces
        self.action_space = spaces.Discrete(36)
        space = {
            "embedding": spaces.Box(-np.inf, np.inf, shape=(362,)),
            "actions_mask": spaces.Box(0, 1, shape=(36,)),
        }
        space = OrderedDict(sorted(space.items()))
        self.observation_space = spaces.Dict(space)
        # The variable `self.worker_index` indexes which worker/actor is working on the chosen function, it will help us avoid problems during compiling,
        # by adding the index of the worker to the name of the worker in order to not interfer with the compilation of another node
        if isinstance(config, ray.rllib.env.env_context.EnvContext):
            # This the case of training
            self.worker_index = str(config.worker_index)
        else:
            # This is the case of evaluating
            self.worker_index = ""
        self.current_program = ""
        self.reset()

    def reset(self, seed=None, options={}):
        embedded_tensor = None
        # Select a program randomly
        while embedded_tensor == None:
            # There is some programs that has unsupported loop levels , acces matrices , ...
            # These programs are not supported yet so the embedded_tensor will be None

            # read the ip and port from the server_address file
            self.ip_and_port = ""
            while self.ip_and_port == "":
                with open("./server_address", "r") as f:
                    self.ip_and_port = f.read()
            function_name = (
                ""
                if options is None or "function_name" not in options
                else options["function_name"]
            )
            with grpc.insecure_channel(self.ip_and_port) as channel:
                stub = tiramisu_function_pb2_grpc.TiramisuDataServerStub(channel)
                response = stub.GetTiramisuFunction(
                    tiramisu_function_pb2.TiramisuFunctionName(
                        name=function_name
                    )  # You can also specify a function name like function550013
                )

            cpp = (
                response.cpp[1:-1]
                .replace('\\"', '"')
                .replace("\\n", "\n")
                .replace("\\t", "\t")
            )
            prog_infos = (
                response.name,
                json.loads(response.content),
                cpp,
                response.wrapper,
            )

            # The shape of embedded_tensor : (180,)
            # Shape of actions mask : (33,)
            embedded_tensor, actions_mask = self.tiramisu_api.set_program(*prog_infos)
            self.current_program = prog_infos[0]

        self.state = {
            # Converting Tensor to numpy array
            "embedding": self.preprocess_embeddings(embeddings=embedded_tensor),
            "actions_mask": actions_mask,
        }
        self.state = OrderedDict(sorted(self.state.items()))
        self.previous_speedup = self.reward = 1
        self.done = self.truncated = False
        self.info = {}
        self.action_index = 0
        return self.state, self.info

    def step(self, action):
        self.action_index += 1
        speedup, embedded_tensor, legality, actions_mask = self.apply_flattened_action(
            action=action
        )
        if speedup < 0:
            speedup = 0.01
        instant_speedup = 1
        if legality and not self.done:
            self.state = {
                "embedding": self.preprocess_embeddings(
                    embeddings=embedded_tensor, action=action
                ),
                "actions_mask": actions_mask,
            }

            # If the action is legal , we divide the speedup of new sequence {A_0 .. A_i+1} by the speedup of
            # the previous Sequnce {A_0 .. A_i} to get the speedup of the action {A_i+1}
            instant_speedup = speedup / self.previous_speedup
            self.previous_speedup = speedup

        self.reward = math.log(instant_speedup, 4)

        if self.action_index == 14:
            self.done = True

        # Update dataset on episode end
        if self.done:
            tiramisu_program_dict = (
                self.tiramisu_api.get_current_tiramisu_program_dict()
            )

            with grpc.insecure_channel(self.ip_and_port) as channel:
                stub = tiramisu_function_pb2_grpc.TiramisuDataServerStub(channel)
                response = stub.SaveTiramisuFunction(
                    tiramisu_function_pb2.TiramisuFunction(
                        name=self.current_program,
                        content=json.dumps(tiramisu_program_dict),
                    )  # You can also specify a function name like function550013
                )
            # self.dataset_actor.update_dataset.remote(
            #     self.current_program, tiramisu_program_dict
            # )

        return self.state, self.reward, self.done, self.truncated, self.info

    def apply_flattened_action(self, action):
        if action < 4:
            loop_level = action
            # Interchange of loops (0,1) (1,2) (2,3) (3,4)
            (
                speedup,
                embedded_tensor,
                legality,
                actions_mask,
            ) = self.tiramisu_api.interchange(
                loop_level1=loop_level,
                loop_level2=loop_level + 1,
                env_id=action,
                worker_id=self.worker_index,
            )
        elif action < 9:
            loop_level = action - 4
            # Reversal from 0 to 4
            (
                speedup,
                embedded_tensor,
                legality,
                actions_mask,
            ) = self.tiramisu_api.reverse(
                loop_level=loop_level, env_id=action, worker_id=self.worker_index
            )
        elif action < 12:
            loop_level = action - 9
            # Skewing 0,1 to 2,3
            speedup, embedded_tensor, legality, actions_mask = self.tiramisu_api.skew(
                loop_level1=loop_level,
                loop_level2=loop_level + 1,
                env_id=action,
                worker_id=self.worker_index,
            )
        elif action < 14:
            loop_level = action - 12
            # For parallelization 0 and 1
            (
                speedup,
                embedded_tensor,
                legality,
                actions_mask,
            ) = self.tiramisu_api.parallelize(
                loop_level=loop_level, env_id=action, worker_id=self.worker_index
            )
        elif action < 18:
            loop_level = action - 14

            speedup, embedded_tensor, legality, actions_mask = self.tiramisu_api.tile2D(
                loop_level1=loop_level,
                loop_level2=loop_level + 1,
                size_x=128,
                size_y=64,
                env_id=action,
                worker_id=self.worker_index,
            )
        elif action < 22:
            loop_level = action - 18

            speedup, embedded_tensor, legality, actions_mask = self.tiramisu_api.tile2D(
                loop_level1=loop_level,
                loop_level2=loop_level + 1,
                size_x=64,
                size_y=128,
                env_id=action,
                worker_id=self.worker_index,
            )
        elif action < 26:
            loop_level = action - 22

            speedup, embedded_tensor, legality, actions_mask = self.tiramisu_api.tile2D(
                loop_level1=loop_level,
                loop_level2=loop_level + 1,
                size_x=64,
                size_y=64,
                env_id=action,
                worker_id=self.worker_index,
            )
        elif action < 31:
            factor = action - 24
            speedup, embedded_tensor, legality, actions_mask = self.tiramisu_api.unroll(
                unrolling_factor=2**factor, env_id=action, worker_id=self.worker_index
            )
        elif action == 31:
            speedup, embedded_tensor, legality, actions_mask = self.tiramisu_api.fuse(
                env_id=action, worker_id=self.worker_index
            )
        
        elif action == 32:
            speedup, embedded_tensor, legality, actions_mask = self.tiramisu_api.addOne(
                env_id=action,
                worker_id=self.worker_index,
            )
        elif action == 33:
            speedup, embedded_tensor, legality, actions_mask = self.tiramisu_api.nextRow(
                env_id=action,
                worker_id=self.worker_index,
            )
        elif action == 34:
            speedup, embedded_tensor, legality, actions_mask = self.tiramisu_api.nextCol(
                env_id=action,
                worker_id=self.worker_index,
            )
        
        else:
            # Next case
            next_branch = self.tiramisu_api.scheduler_service.next_branch()
            if next_branch == None:
                speedup, embedded_tensor, legality, actions_mask = (
                    1,
                    None,
                    True,
                    np.zeros(33),
                )
                self.done = True
            else:
                speedup, embedded_tensor, legality, actions_mask = (
                    1,
                    next_branch[0],
                    True,
                    next_branch[1],
                )

        return speedup, embedded_tensor, legality, actions_mask

    def preprocess_embeddings(self, embeddings: torch.Tensor, action=-1):
        embeddings = torch.cat(
            (
                *embeddings,
                torch.tensor(
                    [
                        (
                            (self.tiramisu_api.scheduler_service.current_branch + 1)
                            / len(self.tiramisu_api.scheduler_service.branches)
                        ),
                        action,
                    ],
                    dtype=torch.float32,
                ),
            ),
            dim=0,
        )
        return embeddings.numpy()
