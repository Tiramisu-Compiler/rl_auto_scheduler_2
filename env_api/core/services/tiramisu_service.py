from ..models.tiramisu_program import TiramisuProgram
from .compiling_service import CompilingService
import json
from env_api.utils.exceptions import *


class TiramisuService:
    def __init__(self):
        pass

    def fetch_prog_compil(self, path: str):
        # This function takes a path and creates a tiramisu program from compiling the file to get the annotations and all the infos
        tiramisu_prog = TiramisuProgram(file_path=path)
        tiramisu_prog.annotations = self.get_annotations(tiramisu_prog)
        return tiramisu_prog

    def fetch_prog_offline(self,name:str,data:dict,original_str:str=None):
        # This function fetched all the data from an offline dataset
        tiramisu_prog = TiramisuProgram.from_dict(name=name,data=data,original_str=original_str)
        return tiramisu_prog
    
    def get_annotations(self, prog: TiramisuProgram):
        max_accesses = 15
        min_accesses = 1
        max_iterators = 5
        result = CompilingService.compile_annotations(prog)
        prog.annotations = json.loads(result)
        computations_dict = prog.annotations["computations"]
        # Making sure every computation doesn't exceed the limit of the cost model , if the model is updated change the conditions
        for comp_name in computations_dict:
            comp_dict = computations_dict[comp_name]
            if len(comp_dict["accesses"]) > max_accesses:
                raise NbAccessException
            if len(comp_dict["accesses"]) < min_accesses:
                raise NbAccessException
            if len(comp_dict["iterators"]) > max_iterators:
                raise LoopsDepthException
        return prog.annotations
