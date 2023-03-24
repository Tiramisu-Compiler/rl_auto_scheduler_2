from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal
import yaml,os

@dataclass
class TiramisuConfig:
    tiramisu_path: str = ""
    env_type: Literal["model", "cpu"] = "model"
    tags_model :str = ""


@dataclass
class DatasetConfig:
    path: str = ""
    offline : str = ""
    benchmark : str = ""


@dataclass
class AutoSchedulerConfig:

    tiramisu: TiramisuConfig
    dataset: DatasetConfig

    def __post_init__(self):
        if isinstance(self.tiramisu, dict):
            self.tiramisu = TiramisuConfig(**self.tiramisu)
        if isinstance(self.dataset, dict):
            self.dataset = DatasetConfig(**self.dataset)

def read_yaml_file(path):
    with open(path) as yaml_file:
        return yaml_file.read()

def parse_yaml_file(yaml_string: str) -> Dict[Any, Any]:
    return yaml.safe_load(yaml_string)

def dict_to_config(parsed_yaml: Dict[Any, Any]) -> AutoSchedulerConfig:
    tiramisu = TiramisuConfig(**parsed_yaml["tiramisu"])
    dataset = DatasetConfig(**parsed_yaml["dataset"])
    return AutoSchedulerConfig(tiramisu,dataset)


class Config(object):
    config = None
    @classmethod
    def init(self,config_yaml = "env_api/utils/config/config.yaml"):
        parsed_yaml_dict = parse_yaml_file(read_yaml_file(config_yaml))
        Config.config = dict_to_config(parsed_yaml_dict)
        os.environ["TIRAMISU_ROOT"] = Config.config.tiramisu.tiramisu_path
