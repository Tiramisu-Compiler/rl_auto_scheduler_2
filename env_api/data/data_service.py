from .data_gen.tiramisu_maker import generate_programs
import os, pickle


# TODO : Recheck reading and writing to disk
class DataSetService:
    def __init__(self,
                 dataset_path='env_api/data/dataset/',
                 copy_path='env_api/data/copy/',
                 offline_path=None):
        self.dataset_path = dataset_path
        self.dataset_copy_path = copy_path
        self.offline_dataset = None
        if (offline_path != None):
            with open(offline_path, "rb") as file:
                self.offline_dataset = pickle.load(file)

    # TODO : REMOVE THIS METHOD
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
            return None, exist_offline
        file_name = func_name + "_generator.cpp"
        original_path = self.dataset_path + func_name + "/" + file_name
        target_path = "{}{}".format(self.dataset_copy_path, func_name)

        if not os.path.isdir(self.dataset_copy_path):
            os.mkdir(self.dataset_copy_path)
        if os.path.isdir(target_path):
            return (target_path + "/" + file_name, exist_offline)
        os.mkdir(target_path)
        os.system("cp {} {}".format(original_path,
                                    target_path + "/" + file_name))
        while not os.path.isfile(target_path + "/" + file_name):
            0
        return ("{}/{}".format(target_path, file_name), exist_offline)

    def get_offline_prog_data(self,name: str):
        return self.offline_dataset[name]