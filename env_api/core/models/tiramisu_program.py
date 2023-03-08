import re
from pathlib import Path


class TiramisuProgram():
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.annotations = None
        self.comp_name = None
        self.name = None
        self.schedules = None
        if (file_path):
            with open(file_path, 'r') as f:
                self.original_str = f.read()
            self.func_folder = ('/'.join(Path(file_path).parts[:-1]) if
                                len(Path(file_path).parts) > 1 else '.') + '/'
            self.body = re.findall(r'(tiramisu::init(?s:.)+)tiramisu::codegen',
                                   self.original_str)[0]
            self.name = re.findall(r'tiramisu::init\(\"(\w+)\"\);',
                                   self.original_str)[0]
            self.comp_name = re.findall(r'computation (\w+)\(',
                                        self.original_str)
            self.code_gen_line = re.findall(r'tiramisu::codegen\({.+;',
                                            self.original_str)[0]
            buffers_vect = re.findall(r'{(.+)}', self.code_gen_line)[0]
            self.IO_buffer_names = re.findall(r'\w+', buffers_vect)
            self.buffer_sizes = []
            for buf_name in self.IO_buffer_names:
                sizes_vect = re.findall(r'buffer ' + buf_name + '.*{(.*)}',
                                        self.original_str)[0]
                self.buffer_sizes.append(re.findall(r'\d+', sizes_vect))

    # Since there is no factory constructors in python, I am creating this class method to replace the factory pattern
    @classmethod
    def from_dict(cls, name: str, data: dict):
        # Initiate an instante of the TiramisuProgram class
        tiramisu_prog = cls(None)
        tiramisu_prog.name = name
        tiramisu_prog.annotations = data["annotations"]
        tiramisu_prog.comp_name = list(
            data["annotations"]["computations"].keys())
        tiramisu_prog.schedules = data["schedules"]
        # After taking the neccessary fields return the instance
        return tiramisu_prog
