from typing import List


from env_api.core.services.compiling_service import CompilingService
from env_api.scheduler.models.schedule import Schedule
from env_api.scheduler.services.legality.base_service import BaseLegalityService


def str_to_int(str: str):
    try:
        return int(str)
    except ValueError:
        return None


class CompilingLegalityService(BaseLegalityService):
    def get_legality(self, schedule_object: Schedule, branches: List[Schedule]):
        """
        This function is used to get the legality of a schedule from the server
        """
        try:
            return int(
                CompilingService.compile_legality(
                    schedule_object=schedule_object,
                    optims_list=schedule_object.schedule_list,
                    branches=branches,
                )
            )
        except ValueError as e:
            print("Legality error :", e)
            return None
