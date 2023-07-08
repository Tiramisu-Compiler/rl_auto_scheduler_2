import re
from pathlib import Path
import config.config as cfg


class TiramisuProgram():
    def __init__(self, code: str):
        self.annotations = None
        self.comps = None
        self.name = None
        self.schedules_legality = {}
        self.schedules_solver = {}
        self.original_str = None
        if (code):
            self.load_code_lines(original_str=code)

    # Since there is no factory constructors in python, I am creating this class method to replace the factory pattern
    @classmethod
    def from_dict(cls, name: str, data: dict, original_str: str = None):
        # Initiate an instante of the TiramisuProgram class
        tiramisu_prog = cls(None)
        tiramisu_prog.name = name
        tiramisu_prog.annotations = data["program_annotation"]
        if (tiramisu_prog.annotations):
            tiramisu_prog.comps = list(
                tiramisu_prog.annotations["computations"].keys())
            tiramisu_prog.schedules_legality = data["schedules_legality"]
            tiramisu_prog.schedules_solver = data["schedules_solver"]

        tiramisu_prog.load_code_lines(original_str)

        # After taking the neccessary fields return the instance
        return tiramisu_prog

    def load_code_lines(self, original_str: str = None):
        '''
        This function loads the file code , it is necessary to generate legality check code and annotations
        '''
        if original_str:
            self.original_str = original_str
        else :
            return

        self.body = re.findall(r'(tiramisu::init(?s:.)+)tiramisu::codegen',
                               self.original_str)[0]
        self.name = re.findall(r'tiramisu::init\(\"(\w+)\"\);',
                               self.original_str)[0]
        # Remove the wrapper include from the original string
        self.wrapper_str = f'#include "{self.name}_wrapper.h"'
        self.original_str = self.original_str.replace(
            self.wrapper_str, f"// {self.wrapper_str}")
        self.comps = re.findall(r'computation (\w+)\(', self.original_str)
        self.code_gen_line = re.findall(r'tiramisu::codegen\({.+;',
                                        self.original_str)[0]
        # buffers_vect = re.findall(r'{(.+)}', self.code_gen_line)[0]
        # self.IO_buffer_names = re.findall(r'\w+', buffers_vect)
        # self.buffer_sizes = []
        # for buf_name in self.IO_buffer_names:
        #     sizes_vect = re.findall(r'buffer ' + buf_name + '.*{(.*)}',
        #                             self.original_str)[0]
        #     self.buffer_sizes.append(re.findall(r'\d+', sizes_vect))
