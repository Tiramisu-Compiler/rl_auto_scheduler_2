from ..models.tiramisu_program import TiramisuProgram

class TiramisuService():
    def __init__(self):
        pass

    def get_tiramisu_model(self,path):
        return TiramisuProgram(file_path=path)
    