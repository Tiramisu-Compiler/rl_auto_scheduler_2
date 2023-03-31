from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal
import yaml, os


@dataclass
class TiramisuConfig:
    tiramisu_path: str = ""
    env_type: Literal["model", "cpu"] = "model"
    tags_model_weights: str = ""


@dataclass
class DatasetConfig:
    path: str = ""
    offline: str = ""
    save_path: str = ""
    is_benchmark: bool = False
    benchmark_cpp_files: str = ""
    benchmark_path: str = ""


@dataclass
class Ray:
    results: str = ""
    restore_checkpoint: str = ""


@dataclass
class Experiment:
    name: str = "test"
    checkpoint_frequency: int = 10
    checkpoint_num_to_keep: int = 10
    training_iteration: int = 500
    timesteps_total: int = 1000000
    episode_reward_mean: float = 2
    legality_speedup: float =  1.0

@dataclass
class PolicyNetwork:
    vf_share_layers: bool =  False
    policy_hidden_layers : List[int] = field(
        default_factory=lambda: [])
    vf_hidden_layers : List[int] = field(
        default_factory=lambda: [])
    dropout_rate: float = 0.2
    lr: float = 0.001


@dataclass
class AutoSchedulerConfig:

    tiramisu: TiramisuConfig
    dataset: DatasetConfig
    ray: Ray
    experiment:Experiment
    policy_network:PolicyNetwork

    def __post_init__(self):
        if isinstance(self.tiramisu, dict):
            self.tiramisu = TiramisuConfig(**self.tiramisu)
        if isinstance(self.dataset, dict):
            self.dataset = DatasetConfig(**self.dataset)
        if isinstance(self.ray, dict):
            self.ray = Ray(**self.ray)
        if isinstance(self.experiment, dict):
            self.experiment = Experiment(**self.experiment)
        if isinstance(self.policy_network, dict):
            self.policy_network = PolicyNetwork(**self.policy_network)


def read_yaml_file(path):
    with open(path) as yaml_file:
        return yaml_file.read()


def parse_yaml_file(yaml_string: str) -> Dict[Any, Any]:
    return yaml.safe_load(yaml_string)


def dict_to_config(parsed_yaml: Dict[Any, Any]) -> AutoSchedulerConfig:
    tiramisu = TiramisuConfig(**parsed_yaml["tiramisu"])
    dataset = DatasetConfig(**parsed_yaml["dataset"])
    ray = Ray(**parsed_yaml["ray"])
    experiment = Experiment(**parsed_yaml["experiment"])
    policy_network = PolicyNetwork(**parsed_yaml["policy_network"])
    return AutoSchedulerConfig(tiramisu, dataset, ray,experiment,policy_network)


class Config(object):
    config = None
    @classmethod
    def init(self, config_yaml="./config/config.yaml"):
        parsed_yaml_dict = parse_yaml_file(read_yaml_file(config_yaml))
        Config.config = dict_to_config(parsed_yaml_dict)
