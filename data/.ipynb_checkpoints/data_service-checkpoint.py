from .data_gen.tiramisu_maker import generate_programs
import os
import subprocess



class DataSetService :
    def __init__(self):
        self.dataset_path = 'data/dataset/'
        self.dataset_copy_path = 'data/copy/'
    
    def generate_dataset(self,size):
        generate_programs(output_path=self.dataset_path, first_seed=10, nb_programs=size)

    def get_file_path(self, func_name):
        file_name = func_name + "_generator.cpp"
        original_path = self.dataset_path + func_name + "/" + file_name
        target_path = "{}{}".format(self.dataset_copy_path, func_name)

        if not os.path.isdir(self.dataset_copy_path):
            p = subprocess.Popen(['mkdir',self.dataset_copy_path],stdout=subprocess.PIPE, shell=True).wait()
            p.kill()


        if os.path.isdir(target_path):
            return target_path + "/" + file_name   

        p = subprocess.Popen(['mkdir',target_path],stdout=subprocess.PIPE, shell=True).wait()
        p.kill()
        p = subprocess.Popen(['cp',original_path,target_path + "/" + file_name],stdout=subprocess.PIPE, shell=True).wait()
        p.kill()
        return target_path + "/" + file_name



