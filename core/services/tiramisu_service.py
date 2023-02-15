from ..models.tiramisu_program import TiramisuProgram
from .compiling_service import CompilingService
import json

_MAX_COMPS = 5

class TiramisuService():
    def __init__(self):
        pass

    def get_tiramisu_model(self,path : str):
        tiramisu_prog =  TiramisuProgram(file_path=path)
        tiramisu_prog.set_annotations(self.get_annotations(tiramisu_prog))
        return tiramisu_prog

    def get_annotations(self, prog : TiramisuProgram):
        max_accesses = 15
        min_accesses = 1
        max_iterators = 5
        result = CompilingService.compile_annotations(prog)
        prog.annotations = json.loads(result)
        computations_dict = prog.annotations["computations"]
        for comp_name in computations_dict:
            comp_dict = computations_dict[comp_name]
            if len(comp_dict["accesses"]) > max_accesses:
                raise NbAccessException
            if len(comp_dict["accesses"]) < min_accesses:
                raise NbAccessException
            if len(comp_dict["iterators"]) > max_iterators:
                raise LoopsDepthException
        return prog.annotations

class NbMatricesException(Exception):
    pass


class NbAccessException(Exception):
    pass


class LoopsDepthException(Exception):
    pass


