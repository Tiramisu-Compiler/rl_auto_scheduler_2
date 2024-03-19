import logging
from typing import List

import torch

from config.config import Config
from env_api.core.services.converting_service import ConvertService
from env_api.scheduler.models.schedule import Schedule
from env_api.scheduler.services.legality.base_service import BaseLegalityService
from env_api.scheduler.services.legality.model.model import (
    Model_Recursive_LSTM_Legality,
)
from env_api.utils.data_preprocessors import get_schedule_representation


def str_to_int(str: str):
    try:
        return int(str)
    except ValueError:
        return None


class ModelLegalityService(BaseLegalityService):
    def __init__(self) -> None:
        super().__init__()

        self.model = Model_Recursive_LSTM_Legality(input_size=846)
        self.model.load_state_dict(
            torch.load(
                Config.config.legality.weights_path, map_location=torch.device("cpu")
            )
        )
        self.model.eval()

    def get_legality(self, schedule_object: Schedule, branches: List[Schedule]):
        repr_tensors = get_schedule_representation(schedule_object)

        tree_tensor = ConvertService.get_tree_representation(
            *repr_tensors, schedule_object
        )

        with torch.no_grad():
            legality_prediction = self.model.forward(tree_tensor)
            legality_prediction = legality_prediction.item()
            legality = (
                1 if legality_prediction > Config.config.legality.threshold else 0
            )
            print(f"Legality prediction: {legality_prediction} -> {legality}")
            return legality
