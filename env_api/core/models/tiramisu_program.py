import re
from pathlib import Path
from config.config import Config


class TiramisuProgram():
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.annotations = None
        self.comps = None
        self.name = None
        self.schedules_legality = {}
        self.schedules_solver = {}
        self.original_str = None
        if (file_path):
            self.load_code_lines()

    # Since there is no factory constructors in python, I am creating this class method to replace the factory pattern
    @classmethod
    def from_dict(cls, name: str, data: dict):
        # Initiate an instante of the TiramisuProgram class
        tiramisu_prog = cls(None)
        tiramisu_prog.name = name
        tiramisu_prog.annotations = data["program_annotation"]
        if (tiramisu_prog.annotations):
            tiramisu_prog.comps = list(
                tiramisu_prog.annotations["computations"].keys())
            tiramisu_prog.schedules_legality = data["schedules_legality_dict"]
            tiramisu_prog.schedules_solver = data["schedules_solver_results_dict"]
        # After taking the neccessary fields return the instance
        return tiramisu_prog

    def load_code_lines(self):
        '''
        This function loads the file code , it is necessary to generate legality check code and annotations
        '''
        if (self.name):
            # if self.name is None the program doesn't exist in the offline dataset but built from compiling
            # if self.name has a value than it is fetched from the dataset, we need the full path to read
            # the lines of the real function to execute legality code
            func_name = self.name
            file_name = func_name + "_generator.cpp"
            file_path = (Config.config.dataset.benchmark_cpp_files
                         if Config.config.dataset.is_benchmark else Config.
                         config.dataset.path) + func_name + "/" + file_name
            self.file_path = file_path
        else:
            file_path = self.file_path

        with open(file_path, 'r') as f:
            self.original_str = f.read()
        
        # Remove the wrapper include from the original string
        self.original_str = self.original_str.replace(f'#include "{func_name}_wrapper.h"', "")
        self.func_folder = ('/'.join(Path(file_path).parts[:-1])
                            if len(Path(file_path).parts) > 1 else '.') + '/'
        self.body = re.findall(r'(tiramisu::init(?s:.)+)tiramisu::codegen',
                               self.original_str)[0]
        self.name = re.findall(r'tiramisu::init\(\"(\w+)\"\);',
                               self.original_str)[0]
        self.comps = re.findall(r'computation (\w+)\(', self.original_str)
        self.code_gen_line = re.findall(r'tiramisu::codegen\({.+;',
                                        self.original_str)[0]
        buffers_vect = re.findall(r'{(.+)}', self.code_gen_line)[0]
        self.IO_buffer_names = re.findall(r'\w+', buffers_vect)
        self.buffer_sizes = []
        for buf_name in self.IO_buffer_names:
            sizes_vect = re.findall(r'buffer ' + buf_name + '.*{(.*)}',
                                    self.original_str)[0]
            self.buffer_sizes.append(re.findall(r'\d+', sizes_vect))