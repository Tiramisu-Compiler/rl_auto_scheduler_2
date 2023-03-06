from .data_gen.tiramisu_maker import generate_programs
import os
import subprocess


# TODO : Recheck reading and writing to disk
class DataSetService :
    def __init__(self,dataset_path = 'env_api/data/dataset/',copy_path = 'env_api/data/copy/'):
        self.dataset_path = dataset_path
        self.dataset_copy_path = copy_path
    
    # TODO : REMOVE THIS METHOD
    def generate_dataset(self,size):
        generate_programs(output_path=self.dataset_path, first_seed=10, nb_programs=size)

    def get_file_path(self, func_name):
        file_name = func_name + "_generator.cpp"
        original_path = self.dataset_path + func_name + "/" + file_name
        target_path = "{}{}".format(self.dataset_copy_path, func_name)

        if not os.path.isdir(self.dataset_copy_path):
            os.mkdir(self.dataset_copy_path)
        if os.path.isdir(target_path):
            return target_path + "/" + file_name   
        os.mkdir(target_path)
        os.system("cp {} {}".format(original_path, target_path + "/" + file_name ))
        while not os.path.isfile(target_path + "/" + file_name ):
            0
        return target_path + "/" + file_name



