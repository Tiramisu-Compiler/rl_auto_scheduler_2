import os
import pickle
import random
from typing import Tuple

import numpy as np
from env_api.core.models.tiramisu_program import TiramisuProgram
from env_api.data.services.base_data_service import (
    BaseDataService,
)


class CompileDataService(BaseDataService):
    def __init__(
        self,
        dataset_path: str,
        cpps_path: str,
        path_to_save_dataset: str,
        shuffle: bool = False,
        seed: int = None,
        saving_frequency: int = 10000,
    ):
        super().__init__(
            dataset_path=dataset_path,
            path_to_save_dataset=path_to_save_dataset,
            shuffle=shuffle,
            seed=seed,
            saving_frequency=saving_frequency,
        )
        self.cpps_path = cpps_path
        self.cpps = {}

        print(
            f"reading dataset in compilation format: cpps location: {self.cpps_path}"
        )

        with open(self.dataset_path, "rb") as f:
            self.function_names = os.listdir(self.cpps_path)

        # Shuffle the dataset (can be used with random sampling turned off to get a random order)
        if self.shuffle:
            # Set the seed if specified (for reproducibility)
            if self.seed is not None:
                random.seed(self.seed)
            random.shuffle(self.function_names)

        self.dataset_size = len(self.function_names)

    # Returns next function name, function data, and function cpps
    def get_next_function(self, random=False) -> TiramisuProgram:
        if random:
            function_name = np.random.choice(self.function_names)
        # Choose the next function sequentially
        else:
            function_name = self.function_names[
                self.current_function_index % self.dataset_size
            ]
            self.current_function_index += 1

        print(
            f"Selected function with index: {self.current_function_index}, name: {function_name}"
        )
        file_name = function_name + "_generator.cpp"
        file_path = self.cpps_path + function_name + "/" + file_name

        tiramisu_program = self.tiramisu_service.fetch_prog_compil(file_path)

        self.dataset[function_name] = {
            'program_annotation': tiramisu_program.annotations,
            "schedules_legality_dict": {},
            "schedules_solver_results_dict": {},
        }

        return tiramisu_program
