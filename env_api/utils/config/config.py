from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal
import yaml,os

@dataclass
class TiramisuConfig:
    tiramisu_path: str = ""
    env_type: Literal["model", "cpu"] = "cpu"
    tags_model :str = ""
    compile_tiramisu_cmd: str = 'printf "Compiling ${FILE_PATH}\n" >> ${FUNC_DIR}log.txt;\
        ${CXX} -I${TIRAMISU_ROOT}/3rdParty/Halide/include -I${TIRAMISU_ROOT}/include -I${TIRAMISU_ROOT}/3rdParty/isl/include  -Wl,--no-as-needed -ldl -g -fno-rtti   -lpthread -std=c++11 -O0 -o ${FILE_PATH}.o -c ${FILE_PATH};\
        ${CXX} -Wl,--no-as-needed -ldl -g -fno-rtti   -lpthread -std=c++11 -O0 ${FILE_PATH}.o -o ./${FILE_PATH}.out   -L${TIRAMISU_ROOT}/build  -L${TIRAMISU_ROOT}/3rdParty/Halide/lib  -L${TIRAMISU_ROOT}/3rdParty/isl/build/lib  -Wl,-rpath,${TIRAMISU_ROOT}/build:${TIRAMISU_ROOT}/3rdParty/Halide/lib:${TIRAMISU_ROOT}/3rdParty/isl/build/lib -ltiramisu -ltiramisu_auto_scheduler -lHalide -lisl'

    run_tiramisu_cmd: str = 'printf "Running ${FILE_PATH}.out\n">> ${FUNC_DIR}log.txt;\
        ./${FILE_PATH}.out>> ${FUNC_DIR}log.txt;'

    compile_wrapper_cmd = 'cd ${FUNC_DIR};\
            ${GXX} -shared -o ${FUNC_NAME}.o.so ${FUNC_NAME}.o;\
            ${CXX} -I${TIRAMISU_ROOT}/3rdParty/Halide/include -I${TIRAMISU_ROOT}/include -I${TIRAMISU_ROOT}/3rdParty/isl/include -Wl,--no-as-needed -ldl -g -fno-rtti -lpthread -std=c++11 -O3 -o ${FUNC_NAME}_wrapper ${FUNC_NAME}_wrapper.cpp ./${FUNC_NAME}.o.so -L${TIRAMISU_ROOT}/build  -L${TIRAMISU_ROOT}/3rdParty/Halide/lib  -L${TIRAMISU_ROOT}/3rdParty/isl/build/lib  -Wl,-rpath,${TIRAMISU_ROOT}/build:${TIRAMISU_ROOT}/3rdParty/Halide/lib:${TIRAMISU_ROOT}/3rdParty/isl/build/lib -ltiramisu -ltiramisu_auto_scheduler -lHalide -lisl'



@dataclass
class AutoSchedulerConfig:

    tiramisu: TiramisuConfig

    def __post_init__(self):
        if isinstance(self.tiramisu, dict):
            self.tiramisu = TiramisuConfig(**self.tiramisu)



def read_yaml_file(path):
    with open(path) as yaml_file:
        return yaml_file.read()


def parse_yaml_file(yaml_string: str) -> Dict[Any, Any]:
    return yaml.safe_load(yaml_string)


def dict_to_config(parsed_yaml: Dict[Any, Any]) -> AutoSchedulerConfig:
    tiramisu = TiramisuConfig(**parsed_yaml["tiramisu"])
    return AutoSchedulerConfig( tiramisu)


class Config(object):
    config = None
    @classmethod
    def init(self,config_yaml = "env_api/utils/config/config.yaml"):
        parsed_yaml_dict = parse_yaml_file(read_yaml_file(config_yaml))
        Config.config = dict_to_config(parsed_yaml_dict)
        os.environ["TIRAMISU_ROOT"] = Config.config.tiramisu.tiramisu_path
