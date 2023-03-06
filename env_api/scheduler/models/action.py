class Action():
    def __init__(self, params : list ,name : str):
        self.params = params
        self.name = name

class Parallelization(Action):
    def __init__(self, params : list):
        super().__init__(params,name="Parallelization")

class Unrolling(Action):
    def __init__(self, params : list):
        super().__init__(params,name="Unrolling")

class Tiling(Action):
    def __init__(self, params : list):
        super().__init__(params,name="Tiling")

class Reversal(Action):
    def __init__(self, params : list):
        super().__init__(params,name="Reversal")

class Interchange(Action):
    def __init__(self, params : list):
        super().__init__(params,name="Interchange")

class Skewing(Action):
    def __init__(self, params : list):
        super().__init__(params,name="Skewing")

class Fusion(Action):
    def __init__(self, params : list):
        super().__init__(params,name="Fusion")