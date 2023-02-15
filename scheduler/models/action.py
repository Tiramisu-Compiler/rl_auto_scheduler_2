from abc import ABC
class Action(ABC):
    def __init__(self, params : list ,name : str):
        self.params = params
        self.name = name

class Parallelization(Action):
    def __init__(self, params : list , name : str):
        super().__init__(params,name)

class Unrolling(Action):
    def __init__(self, params : list , name : str):
        super().__init__(params,name)

class Tiling(Action):
    def __init__(self, params : list , name : str):
        super().__init__(params,name)

class Reversal(Action):
    def __init__(self, params : list , name : str):
        super().__init__(params,name)

class Interchange(Action):
    def __init__(self, params : list , name : str):
        super().__init__(params,name)

class Skewing(Action):
    def __init__(self, params : list , name : str):
        super().__init__(params,name)

class Fusion(Action):
    def __init__(self, params : list , name : str):
        super().__init__(params,name)