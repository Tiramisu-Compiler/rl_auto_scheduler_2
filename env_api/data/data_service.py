from .data_gen.tiramisu_maker import generate_programs
import os, pickle


class DataSetService:
    def __init__(self,
                 dataset_path='env_api/data/dataset/',
                 offline_path=None):
        self.dataset_path = dataset_path
        self.offline_dataset = None
        if (offline_path != None):
            with open(offline_path, "rb") as file:
                self.offline_dataset = pickle.load(file)

    def generate_dataset(self, size):
        generate_programs(output_path=self.dataset_path,
                          first_seed=10,
                          nb_programs=size)

    def get_file_path(self, func_name):
        exist_offline = False
        # Check if the file exists in the offline dataset
        if (self.offline_dataset):
            # Checking the function name
            exist_offline = func_name in self.offline_dataset
            # We don't need to get the file path because we will fetch infos from the offline dataset
            if (exist_offline):
                return None, exist_offline
        # If the file is not in the dataset we compile it to get the annotations
        file_name = func_name + "_generator.cpp"
        file_path = self.dataset_path + func_name + "/" + file_name
        return file_path, exist_offline

    def get_offline_prog_data(self, name: str):
        return self.offline_dataset[name]