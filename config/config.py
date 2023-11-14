from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal

import yaml

# Enum for the dataset format


class DatasetFormat:
    # Both the CPP code and the data of the functions are loaded from PICKLE files
    PICKLE = "PICKLE"
    # We look for informations in the pickle files, if something is missing we get it from cpp files in a dynamic way
    HYBRID = "HYBRID"

    @staticmethod
    def from_string(s):
        if s == "PICKLE":
            return DatasetFormat.PICKLE
        elif s == "HYBRID":
            return DatasetFormat.HYBRID
        else:
            raise ValueError("Unknown dataset format")


@dataclass
class TiramisuConfig:
    tiramisu_path: str = ""
    env_type: Literal["model", "cpu"] = "model"
    tags_model_weights: str = ""
    is_new_tiramisu: bool = False
    workspace: str = "./workspace/"
    old_tiramisu_path: str = ""

    def __post_init__(self):
        if not self.is_new_tiramisu:
            self.tiramisu_path = self.old_tiramisu_path


@dataclass
class DatasetConfig:
    dataset_format: DatasetFormat = DatasetFormat.HYBRID
    cpps_path: str = ""
    dataset_path: str = ""
    save_path: str = ""
    shuffle: bool = False
    seed: int = None
    saving_frequency: int = 10000
    ip_address: str = "localhost"
    port: int = 50051

    def __init__(self, dataset_config_dict: Dict):
        self.dataset_format = DatasetFormat.from_string(
            dataset_config_dict["dataset_format"]
        )
        self.cpps_path = dataset_config_dict["cpps_path"]
        self.dataset_path = dataset_config_dict["dataset_path"]
        self.save_path = dataset_config_dict["save_path"]
        self.shuffle = dataset_config_dict["shuffle"]
        self.seed = dataset_config_dict["seed"]
        self.saving_frequency = dataset_config_dict["saving_frequency"]

        if "ip_address" in dataset_config_dict:
            self.ip_address = dataset_config_dict["ip_address"]

        if "port" in dataset_config_dict:
            self.port = dataset_config_dict["port"]

        if dataset_config_dict["is_benchmark"]:
            self.dataset_path = (
                dataset_config_dict["benchmark_dataset_path"]
                if dataset_config_dict["benchmark_dataset_path"]
                else self.dataset_path
            )
            self.cpps_path = (
                dataset_config_dict["benchmark_cpp_files"]
                if dataset_config_dict["benchmark_cpp_files"]
                else self.cpps_path
            )


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
    legality_speedup: float = 1.0
    beam_search_order: bool = False
    entropy_coeff: float = 0.0
    train_batch_size: int = 1024
    vf_loss_coeff: int = 1
    minibatch_size: int = 128
    lr: float = 0.001
    vf_share_layers: bool = False
    policy_model: str = ""


@dataclass
class PolicyNetwork:
    policy_hidden_layers: List[int] = field(default_factory=lambda: [])
    vf_hidden_layers: List[int] = field(default_factory=lambda: [])
    dropout_rate: float = 0.2


@dataclass
class LSTMPolicy:
    fc_size: int = 1024
    lstm_state_size: int = 256
    num_layers: int = 1


@dataclass
class EnvVars:
    TIRAMISU_ROOT: str = ""
    CONDA_ENV: str = ""
    LD_LIBRARY_PATH: str = "${CONDA_ENV}/lib:${TIRAMISU_ROOT}/3rdParty/Halide/build/src:${TIRAMISU_ROOT}/3rdParty/llvm/build/lib:${TIRAMISU_ROOT}/3rdParty/isl/build/lib:"


@dataclass
class AutoSchedulerConfig:
    tiramisu: TiramisuConfig
    dataset: DatasetConfig
    ray: Ray
    experiment: Experiment
    policy_network: PolicyNetwork
    lstm_policy: LSTMPolicy
    env_vars: EnvVars

    def __post_init__(self):
        if isinstance(self.tiramisu, dict):
            self.tiramisu = TiramisuConfig(**self.tiramisu)
        if isinstance(self.dataset, dict):
            self.dataset = DatasetConfig(self.dataset)
        if isinstance(self.ray, dict):
            self.ray = Ray(**self.ray)
        if isinstance(self.experiment, dict):
            self.experiment = Experiment(**self.experiment)
        if isinstance(self.policy_network, dict):
            self.policy_network = PolicyNetwork(**self.policy_network)
        if isinstance(self.lstm_policy, dict):
            self.lstm_policy = LSTMPolicy(**self.lstm_policy)


def read_yaml_file(path):
    with open(path) as yaml_file:
        return yaml_file.read()


def parse_yaml_file(yaml_string: str) -> Dict[Any, Any]:
    return yaml.safe_load(yaml_string)


def dict_to_config(parsed_yaml: Dict[Any, Any]) -> AutoSchedulerConfig:
    tiramisu = TiramisuConfig(**parsed_yaml["tiramisu"])
    dataset = DatasetConfig(parsed_yaml["dataset"])
    ray = Ray(**parsed_yaml["ray"])
    experiment = Experiment(**parsed_yaml["experiment"])
    policy_network = PolicyNetwork(**parsed_yaml["policy_network"])
    lstm_policy = LSTMPolicy(**parsed_yaml["lstm_policy"])
    env_vars = EnvVars(**parsed_yaml["env_vars"])
    env_vars.TIRAMISU_ROOT = tiramisu.tiramisu_path
    return AutoSchedulerConfig(
        tiramisu, dataset, ray, experiment, policy_network, lstm_policy, env_vars
    )


class Config(object):
    config = None

    @classmethod
    def init(self, config_yaml="./config/config.yaml"):
        parsed_yaml_dict = parse_yaml_file(read_yaml_file(config_yaml))
        Config.config = dict_to_config(parsed_yaml_dict)
