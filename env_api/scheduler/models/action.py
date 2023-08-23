class Action:
    def __init__(
        self,
        params: list,
        name: str,
        comps: list = [],
        env_id: int = None,
        worker_id="",
    ):
        self.params = params
        self.name = name
        # List of computations concerned by this action
        self.comps = comps
        # The ID of the action inside the RL system, this ID helps in masking actions and matching RL with env_api
        self.env_id = env_id
        # In distributed training we want to know which worker is applying this action in order to distinguish the compilation
        # of the same function by different workers
        self.worker_id = worker_id


class AffineAction(Action):
    def __init__(
        self,
        params: list,
        name: str,
        comps: list = [],
        env_id: int = None,
        worker_id="",
    ):
        super().__init__(params, name, comps, env_id, worker_id)


class Reversal(AffineAction):
    def __init__(self, params: list, env_id: int = None, worker_id=""):
        super().__init__(params, name="Reversal", env_id=env_id, worker_id=worker_id)


class Interchange(AffineAction):
    def __init__(self, params: list, env_id: int = None, worker_id=""):
        super().__init__(params, name="Interchange", env_id=env_id, worker_id=worker_id)


class Skewing(AffineAction):
    def __init__(self, params: list, env_id: int = None, worker_id=""):
        super().__init__(params, name="Skewing", env_id=env_id, worker_id=worker_id)


class Parallelization(Action):
    def __init__(self, params: list, env_id: int = None, worker_id=""):
        super().__init__(
            params, name="Parallelization", env_id=env_id, worker_id=worker_id
        )


class Unrolling(Action):
    def __init__(self, params: list, env_id: int = None, worker_id=""):
        super().__init__(params, name="Unrolling", env_id=env_id, worker_id=worker_id)


class Tiling(Action):
    def __init__(self, params: list, env_id: int = None, worker_id=""):
        super().__init__(params, name="Tiling", env_id=env_id, worker_id=worker_id)


class Fusion(Action):
    def __init__(self, params: list, env_id: int = None, worker_id=""):
        super().__init__(params, name="Fusion", env_id=env_id, worker_id=worker_id)
