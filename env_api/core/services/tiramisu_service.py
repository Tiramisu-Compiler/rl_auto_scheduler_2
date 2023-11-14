import json

from env_api.utils.exceptions import *

from ..models.tiramisu_program import TiramisuProgram
from .compiling_service import CompilingService


class TiramisuService:
    def __init__(self):
        pass

    def fetch_prog_compil(self, code: str):
        tiramisu_prog = TiramisuProgram(code=code)
        tiramisu_prog.annotations = self.get_annotations(tiramisu_prog)
        return tiramisu_prog

    def fetch_prog_offline(
        self, name: str, data: dict, original_str: str = None, wrapper_obj: bytes = None
    ):
        # This function fetched all the data from an offline dataset
        tiramisu_prog = TiramisuProgram.from_dict(
            name=name, data=data, original_str=original_str, wrapper_obj=wrapper_obj
        )
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
