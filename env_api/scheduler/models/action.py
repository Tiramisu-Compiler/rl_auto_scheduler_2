class Action():
    def __init__(self, params : list ,name : str,comps : list = [],env_id : int = None):
        self.params = params
        self.name = name
        self.comps = comps
        self.env_id = env_id

class AffineAction(Action):
    def __init__(self, params: list, name: str, comps: list = [], env_id: int = None):
        super().__init__(params, name, comps, env_id)

class Parallelization(Action):
    def __init__(self, params : list,env_id : int = None):
        super().__init__(params,name="Parallelization",env_id=env_id)

class Unrolling(Action):
    def __init__(self, params : list,env_id : int = None):
        super().__init__(params,name="Unrolling",env_id=env_id)

class Tiling(Action):
    def __init__(self, params : list,env_id : int = None):
        super().__init__(params,name="Tiling",env_id=env_id)

class Reversal(AffineAction):
    def __init__(self, params : list,env_id : int = None):
        super().__init__(params,name="Reversal",env_id=env_id)

class Interchange(AffineAction):
    def __init__(self, params : list,env_id : int = None):
        super().__init__(params,name="Interchange",env_id=env_id)

class Skewing(AffineAction):
    def __init__(self, params : list,env_id : int = None):
        super().__init__(params,name="Skewing",env_id=env_id)

class Fusion(Action):
    def __init__(self, params : list,env_id : int = None):
        super().__init__(params,name="Fusion",env_id=env_id)