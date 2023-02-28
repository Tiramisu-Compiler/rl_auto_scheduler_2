from env_api.core.services.converting_service import ConvertService
from env_api.utils.config.config import Config
from ..models.tags_cost_model import Model_Recursive_LSTM_v3
import torch

MAX_DEPTH = 6

class PredictionService():

    def __init__(self):
        # This model uses the tags instead of matrices
        self.tags_model = Model_Recursive_LSTM_v3(input_size= 890,loops_tensor_size=8)
        # Loading the weights
        self.tags_model.load_state_dict(torch.load(Config.config.tiramisu.tags_model ,map_location="cpu"))
        # Putting the model in evaluation mode to turn off Regularization layers.
        self.tags_model.eval()

    def get_speedup(self,comps_tensor,loops_tensor,schedule_object):
        tree_tensor = ConvertService.get_tree_representation(comps_tensor,loops_tensor,schedule_object)
        with torch.no_grad():
            return  self.tags_model.forward(tree_tensor).item()