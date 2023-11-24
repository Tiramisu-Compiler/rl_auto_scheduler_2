import copy
import logging
import os
import re
import subprocess
from typing import List

from config.config import Config
from env_api.core.models.optim_cmd import OptimizationCommand
from env_api.core.models.tiramisu_program import TiramisuProgram
from env_api.scheduler.models.action import Parallelization, Tiling, Unrolling
from env_api.scheduler.models.branch import Branch
from env_api.scheduler.models.schedule import Schedule


class CompilingService:
    @classmethod
    def compile_legality(
        cls,
        schedule_object: Schedule,
        optims_list: List[OptimizationCommand],
        branches: List[Branch],
    ):
        tiramisu_program = schedule_object.prog
        output_path = (
            Config.config.tiramisu.workspace
            + tiramisu_program.name
            + "legal"
            + optims_list[-1].action.worker_id
        )

        cpp_code = cls.get_legality_code(
            schedule_object=schedule_object, optims_list=optims_list, branches=branches
        )
        return cls.run_cpp_code(cpp_code=cpp_code, output_path=output_path)

    @classmethod
    def get_legality_code(
        cls,
        schedule_object: Schedule,
        optims_list: List[OptimizationCommand],
        branches: List[Branch],
    ):
        tiramisu_program = schedule_object.prog
        cpp_code = tiramisu_program.original_str
        updated_fusion = ""
        unrolling_legality = ""
        comps_dict = {}
        tiling_in_actions = False
        d = schedule_object.prog.annotations["computations"]
        for comp in d:
            comps_dict[comp] = copy.deepcopy(d[comp]["iterators"])
        # Add code to the original file to get legality result
        legality_check_lines = """
        prepare_schedules_for_legality_checks();
        perform_full_dependency_analysis();
        
        bool is_legal=true;"""
        for i in range(len(optims_list)):
            optim = optims_list[i]
            loop_levels_size = len(optim.action.params) // 2
            if not isinstance(optim.action, Unrolling):
                if isinstance(optim.action, Tiling):
                    tiling_in_actions = True
                    #  Add the tiling new loops to comps_dict
                    for impacted_comp in optim.action.comps:
                        for loop_index in optim.action.params[:loop_levels_size]:
                            comps_dict[impacted_comp].insert(
                                loop_levels_size + loop_index, f"t{loop_index}"
                            )

                    for subtiling in optim.action.subtilings:
                        loop_levels_size = len(subtiling.params) // 2
                        for impacted_comp in subtiling.comps:
                            for loop_index in subtiling.params[:loop_levels_size]:
                                comps_dict[impacted_comp].insert(
                                    loop_levels_size + loop_index, f"t{loop_index}"
                                )

                elif isinstance(optim.action, Parallelization):
                    legality_check_lines += (
                        """\n\tis_legal &= loop_parallelization_is_legal("""
                        + str(optim.params_list[0])
                        + """, {"""
                        + ",".join([f"&{comp}" for comp in optim.action.comps])
                        + """});\n"""
                    )
                legality_check_lines += optim.tiramisu_optim_str + "\n"
            else:
                unchanged = True
                comp = optim.action.comps[0]
                for op in optims_list[i + 1 :]:
                    if isinstance(op.action, Tiling) and (comp in op.comps):
                        tiling_in_actions = True
                        # TODO : Check the the params here
                        unrolling_legality += (
                            """\n\tis_legal &= loop_unrolling_is_legal("""
                            + str(optim.action.params[0] + 2)
                            + """, {"""
                            + ", ".join([f"&{comp}" for comp in optim.action.comps])
                            + """});\n"""
                        )
                        unrolling_legality += f"\n\t{comp}.unroll({optim.params_list[0]},{optim.params_list[1]});"
                        unchanged = False
                if unchanged:
                    unrolling_legality += (
                        """\n\tis_legal &= loop_unrolling_is_legal("""
                        + str(optim.action.params[0])
                        + """, {"""
                        + ", ".join([f"&{comp}" for comp in optim.action.comps])
                        + """});\n"""
                    )
                    unrolling_legality += f"\n\t{comp}.unroll({optim.params_list[0]},{optim.params_list[1]});"

        if tiling_in_actions:
            updated_fusion, cpp_code = cls.fuse_tiling_loops(
                code=cpp_code, comps_dict=comps_dict
            )
            legality_check_lines += """            
            clear_implicit_function_sched_graph();
"""

        legality_check_lines += f"""
            {updated_fusion}
            {unrolling_legality}
            prepare_schedules_for_legality_checks();
            is_legal &= check_legality_of_function();   
            std::cout << is_legal;
            """

        # Paste the lines responsable of checking legality of schedule in the cpp file
        cpp_code = cpp_code.replace(
            tiramisu_program.code_gen_line, legality_check_lines
        )
        return cpp_code

    @classmethod
    def compile_annotations(cls, tiramisu_program):
        # TODO : add getting tree structure object from executing the file instead of building it
        output_path = Config.config.tiramisu.workspace + tiramisu_program.name + "annot"
        # Add code to the original file to get json annotations

        if Config.config.tiramisu.is_new_tiramisu:
            get_json_lines = """
                auto ast = tiramisu::auto_scheduler::syntax_tree(tiramisu::global::get_implicit_function(), {});
                std::string program_json = tiramisu::auto_scheduler::evaluate_by_learning_model::get_program_json(ast);
                std::cout << program_json;
                """
        else:
            get_json_lines = """
            auto ast = tiramisu::auto_scheduler::syntax_tree(tiramisu::global::get_implicit_function(), {});
            std::string program_json = tiramisu::auto_scheduler::evaluate_by_learning_model::get_program_json(ast);
            std::cout << program_json;
                """
        # Paste the lines responsable of generating the program json tree in the cpp file
        cpp_code = tiramisu_program.original_str.replace(
            tiramisu_program.code_gen_line, get_json_lines
        )
        return cls.run_cpp_code(cpp_code=cpp_code, output_path=output_path)

    @classmethod
    def run_cpp_code(cls, cpp_code: str, output_path: str):
        if Config.config.tiramisu.is_new_tiramisu:
            # Making the tiramisu root path explicit to the env
            shell_script = [
                # Compile intermidiate tiramisu file
                "$CXX -I$TIRAMISU_ROOT/3rdParty/Halide/install/include -I$TIRAMISU_ROOT/include -I$TIRAMISU_ROOT/3rdParty/isl/include  -Wl,--no-as-needed -ldl -g -fno-rtti   -lpthread -fopenmp -std=c++17 -O0 -o {}.o -c -x c++ -".format(
                    output_path
                ),
                # Link generated file with executer
                "$CXX -Wl,--no-as-needed -ldl -g -fno-rtti -lpthread -fopenmp -std=c++17 -O0 {}.o -o {}.out   -L$TIRAMISU_ROOT/build  -L$TIRAMISU_ROOT/3rdParty/Halide/install/lib64  -L$TIRAMISU_ROOT/3rdParty/isl/build/lib  -Wl,-rpath,$TIRAMISU_ROOT/build:$TIRAMISU_ROOT/3rdParty/Halide/install/lib64:$TIRAMISU_ROOT/3rdParty/isl/build/lib -ltiramisu -ltiramisu_auto_scheduler -lHalide -lisl".format(
                    output_path, output_path
                ),
                # Run the program
                "{}.out".format(output_path),
                # Clean generated files
                "rm {}*".format(output_path),
            ]
        else:
            shell_script = [
                # Compile intermidiate tiramisu file
                "$CXX -I$TIRAMISU_ROOT/3rdParty/Halide/include -I$TIRAMISU_ROOT/include -I$TIRAMISU_ROOT/3rdParty/isl/include  -Wl,--no-as-needed -ldl -g -fno-rtti   -lpthread -std=c++11 -O0 -o {}.o -c -x c++ -".format(
                    output_path
                ),
                # Link generated file with executer
                "$CXX -Wl,--no-as-needed -ldl -g -fno-rtti -lpthread -std=c++11 -O0 {}.o -o {}.out   -L$TIRAMISU_ROOT/build  -L$TIRAMISU_ROOT/3rdParty/Halide/lib  -L$TIRAMISU_ROOT/3rdParty/isl/build/lib  -Wl,-rpath,$TIRAMISU_ROOT/build:$TIRAMISU_ROOT/3rdParty/Halide/lib:$TIRAMISU_ROOT/3rdParty/isl/build/lib -ltiramisu -ltiramisu_auto_scheduler -lHalide -lisl".format(
                    output_path, output_path
                ),
                # Run the program
                "{}.out".format(output_path),
                # Clean generated files
                "rm {}.out {}.o".format(output_path, output_path),
            ]
        try:
            # add env vars at top
            envs = []
            for key, val in Config.config.env_vars.__dict__.items():
                envs.append(f"export {key}={val}")
            shell_script = envs + shell_script
            compiler = subprocess.run(
                [" && ".join(shell_script)],
                input=cpp_code,
                capture_output=True,
                text=True,
                shell=True,
                check=True,
            )
            return compiler.stdout if compiler.stdout != "" else "0"
        except subprocess.CalledProcessError as e:
            logging.error(f"Process terminated with error code: {e.returncode}")
            logging.error(f"Error output: {e.stderr}")
            logging.error(f"Output:{e.stdout}")
            logging.error(cpp_code)
            logging.error(" && ".join(shell_script))
            return "0"
        except Exception as e:
            print(e)
            return "0"

    @classmethod
    def call_skewing_solver(cls, schedule_object, optim_list, action, branches):
        params = action.params
        legality_cpp_code = cls.get_legality_code(schedule_object, optim_list, branches)
        to_replace = re.findall(r"std::cout << is_legal;", legality_cpp_code)[0]
        header = """
        function * fct = tiramisu::global::get_implicit_function();\n"""
        legality_cpp_code = legality_cpp_code.replace(
            "is_legal &= check_legality_of_function();", ""
        )
        legality_cpp_code = legality_cpp_code.replace("bool is_legal=true;", "")
        legality_cpp_code = re.sub(
            r"is_legal &= loop_parallelization_is_legal.*\n", "", legality_cpp_code
        )
        legality_cpp_code = re.sub(
            r"is_legal &= loop_unrolling_is_legal.*\n", "", legality_cpp_code
        )

        solver_lines = (
            header
            + "\n\tauto auto_skewing_result = fct->skewing_local_solver({"
            + ", ".join([f"&{comp}" for comp in action.comps])
            + "}"
            + ",{},{},1);\n".format(*params)
        )

        solver_lines += """    
        std::vector<std::pair<int,int>> outer1, outer2,outer3;
        tie( outer1,  outer2,  outer3 )= auto_skewing_result;
        if (outer1.size()>0){
            std::cout << outer1.front().first;
            std::cout << ",";
            std::cout << outer1.front().second;
            std::cout << ",";
        }else {
            std::cout << "None,None,";
        }
        if(outer2.size()>0){
            std::cout << outer2.front().first;
            std::cout << ",";
            std::cout << outer2.front().second;
            std::cout << ",";
        }else {
            std::cout << "None,None,";
        }
        if(outer3.size()>0){
            std::cout << outer3.front().first;
            std::cout << ",";
            std::cout << outer3.front().second;
            std::cout << ",";
        }else {
            std::cout << "None,None,";
        }
        
            """

        solver_code = legality_cpp_code.replace(to_replace, solver_lines)
        output_path = (
            Config.config.tiramisu.workspace
            + schedule_object.prog.name
            + "skew_solver"
            + action.worker_id
        )
        result_str = cls.run_cpp_code(cpp_code=solver_code, output_path=output_path)
        if not result_str:
            return None
            # Refer to function run_cpp_code to see from where the "0" comes from
        elif result_str == "0":
            return None
        result_str = result_str.split(",")
        # Skewing Solver returns 3 solutions in form of tuples, the first tuple is for outer parallelism ,
        # second is for inner parallelism , and last one is for locality, we are going to use the first preferably
        # if availble , else , we are going to use the scond one if available, this policy of choosing factors may change
        # in later versions!
        # The compiler in our case returns a tuple of type : (fac0,fac1,fac2,fac3,fac4,fac5) each 2 factors represent the
        # solutions mentioned above
        if result_str[0] != "None":
            # Means we have a solution for outer parallelism
            fac1 = int(result_str[0])
            fac2 = int(result_str[1])
            return fac1, fac2
        if result_str[2] != "None":
            # Means we have a solution for inner parallelism
            fac1 = int(result_str[2])
            fac2 = int(result_str[3])
            return fac1, fac2
        else:
            return None

    @classmethod
    def fuse_tiling_loops(cls, code: str, comps_dict: dict):
        fusion_code = ""
        # This pattern will detect lines that looks like this :
        # ['comp00.then(comp01, i2)',
        # '.then(comp02, i1)',
        # '.then(comp03, i3)',
        # '.then(comp04, i1);']
        regex_first_comp = r"(\w+)\.then\("
        matching = re.search(regex_first_comp, code)

        if matching is None:
            return fusion_code, code

        # comps will contain all the computations that are fused together
        comps = [matching.group(1)]

        # regex rest of the thens
        regex_rest = r"\.then\(([\w]+),"
        # results will contain all the lines that match the regex
        for result in re.findall(regex_rest, code):
            comps.append(result)

        # levels indicates which loop level the 2 comps will be seperated in
        levels = []
        # updated_lines will contain new lines of code with the new seperated levels
        updated_lines = []
        # Defining intersection between comps' iterators
        for i in range(len(comps) - 1):
            level = 0
            while True:
                if comps_dict[comps[i]][level] == comps_dict[comps[i + 1]][level]:
                    if (
                        level + 1 == comps_dict[comps[i]].__len__()
                        or level + 1 == comps_dict[comps[i + 1]].__len__()
                    ):
                        levels.append(level)
                        break
                    level += 1
                else:
                    levels.append(level - 1)
                    break
            if levels[-1] == -1:
                updated_lines.append(f".then({comps[i+1]}, computation::root)")
            else:
                updated_lines.append(f".then({comps[i+1]},{levels[-1]})")

        updated_lines[0] = comps[0] + updated_lines[0]
        updated_lines[-1] = updated_lines[-1] + ";"

        for line in range(len(comps) - 1):
            # code = code.replace(results[line],"")
            fusion_code += updated_lines[line]
        return fusion_code, code

    @classmethod
    def get_schedule_code(
        cls,
        tiramisu_program: TiramisuProgram,
        optims_list: List[OptimizationCommand],
        branches: List[Branch],
    ):
        if not optims_list:
            return tiramisu_program.original_str

        cpp_code = tiramisu_program.original_str
        updated_fusion = ""
        unrolling_updated = ""
        comps_dict = {}
        tiling_in_actions = False
        d = tiramisu_program.annotations["computations"]
        for comp in d:
            comps_dict[comp] = copy.deepcopy(d[comp]["iterators"])
        # Add code to the original file to get legality result
        schedule_code = ""
        for i in range(len(optims_list)):
            optim = optims_list[i]
            loop_levels_size = len(optim.action.params) // 2
            if not isinstance(optim.action, Unrolling):
                if isinstance(optim.action, Tiling):
                    tiling_in_actions = True
                    #  Add the tiling new loops to comps_dict
                    for impacted_comp in optim.action.comps:
                        for loop_index in optim.action.params[:loop_levels_size]:
                            comps_dict[impacted_comp].insert(
                                loop_levels_size + loop_index, f"t{loop_index}"
                            )

                    for subtiling in optim.action.subtilings:
                        loop_levels_size = len(subtiling.params) // 2
                        for impacted_comp in subtiling.comps:
                            for loop_index in subtiling.params[:loop_levels_size]:
                                comps_dict[impacted_comp].insert(
                                    loop_levels_size + loop_index, f"t{loop_index}"
                                )
                schedule_code += optim.tiramisu_optim_str + "\n"
            else:
                unchanged = True
                comp = optim.action.comps[0]
                factor = optim.action.params[-1]
                for branch in branches:
                    if comp in branch.comps:
                        for c in branch.comps:
                            # unrolling_updated += comp +f".tag_unroll_level({len(branch.common_it) -1 + branch.additional_loops},{factor});\n"
                            unrolling_updated += f"\n\t{c}.unroll({len(branch.common_it) -1 + branch.additional_loops},{factor});"
                        unchanged = False
                if unchanged:
                    unrolling_updated += optim.tiramisu_optim_str + "\n"

        if tiling_in_actions:
            updated_fusion, cpp_code = cls.fuse_tiling_loops(
                code=cpp_code, comps_dict=comps_dict
            )

            schedule_code += f"""
                clear_implicit_function_sched_graph();
                {updated_fusion}
                """

        schedule_code += f"""
            {unrolling_updated}
            """

        # Add code gen line to the schedule code
        schedule_code += "\n\t" + tiramisu_program.code_gen_line + "\n"
        # Paste the lines responsable of checking legality of schedule in the cpp file
        cpp_code = tiramisu_program.original_str.replace(
            tiramisu_program.code_gen_line, schedule_code
        )
        cpp_code = cpp_code.replace(
            f"// {tiramisu_program.wrapper_str}", tiramisu_program.wrapper_str
        )
        return cpp_code

    @classmethod
    def write_cpp_code(cls, cpp_code: str, output_path: str):
        with open(output_path + ".cpp", "w") as f:
            f.write(cpp_code)

    @classmethod
    def execute_code(
        cls,
        tiramisu_program: TiramisuProgram,
        optims_list: List[OptimizationCommand],
        branches: List[Branch],
    ):
        execution_time = None

        cpp_code = cls.get_schedule_code(
            tiramisu_program=tiramisu_program,
            optims_list=optims_list,
            branches=branches,
        )

        logging.debug("cpp_code: \n %s", cpp_code)

        output_path = f"{Config.config.tiramisu.workspace}{tiramisu_program.name}"

        cpp_file_path = output_path + "_schedule.cpp"
        with open(cpp_file_path, "w") as file:
            file.write(cpp_code)

        wrapper_cpp, wrapper_h = tiramisu_program.build_wrappers()

        wrapper_cpp_path = output_path + "_wrapper.cpp"
        wrapper_h_path = output_path + "_wrapper.h"

        with open(wrapper_cpp_path, "w") as file:
            file.write(wrapper_cpp)

        with open(wrapper_h_path, "w") as file:
            file.write(wrapper_h)

        shell_script = [
            f"cd {Config.config.tiramisu.workspace}",
            # Compile intermidiate tiramisu file
            f"$CXX -I$TIRAMISU_ROOT/3rdParty/Halide/include -I$TIRAMISU_ROOT/include -I$TIRAMISU_ROOT/3rdParty/isl/include  -Wl,--no-as-needed -ldl -g -fno-rtti   -lpthread -std=c++11 -O0 -o {tiramisu_program.name}.o -c {tiramisu_program.name}_schedule.cpp",
            # Link generated file with executer
            f"$CXX -Wl,--no-as-needed -ldl -g -fno-rtti -lpthread -std=c++11 -O0 {tiramisu_program.name}.o -o {tiramisu_program.name}.out   -L$TIRAMISU_ROOT/build  -L$TIRAMISU_ROOT/3rdParty/Halide/lib  -L$TIRAMISU_ROOT/3rdParty/isl/build/lib  -Wl,-rpath,$TIRAMISU_ROOT/build:$TIRAMISU_ROOT/3rdParty/Halide/lib:$TIRAMISU_ROOT/3rdParty/isl/build/lib -ltiramisu -ltiramisu_auto_scheduler -lHalide -lisl",
            # Run the generator
            f"./{tiramisu_program.name}.out",
            # compile the wrapper
            f"$CXX -shared -o {tiramisu_program.name}.o.so {tiramisu_program.name}.o",
        ]

        if Config.config.tiramisu.is_new_tiramisu:
            shell_script = [
                f"cd {Config.config.tiramisu.workspace}",
                # Compile intermidiate tiramisu file
                f"$CXX -I$TIRAMISU_ROOT/3rdParty/Halide/install/include -I$TIRAMISU_ROOT/include -I$TIRAMISU_ROOT/3rdParty/isl/include  -Wl,--no-as-needed -ldl -g -fno-rtti   -lpthread -fopenmp -std=c++17 -O0 -o {tiramisu_program.name}.o -c {tiramisu_program.name}_schedule.cpp",
                # Link generated file with executer
                f"$CXX -Wl,--no-as-needed -ldl -g -fno-rtti -lpthread -fopenmp -std=c++17 -O0 {tiramisu_program.name}.o -o {tiramisu_program.name}.out   -L$TIRAMISU_ROOT/build  -L$TIRAMISU_ROOT/3rdParty/Halide/install/lib64  -L$TIRAMISU_ROOT/3rdParty/isl/build/lib  -Wl,-rpath,$TIRAMISU_ROOT/build:$TIRAMISU_ROOT/3rdParty/Halide/install/lib64:$TIRAMISU_ROOT/3rdParty/isl/build/lib -ltiramisu -ltiramisu_auto_scheduler -lHalide -lisl",
                # Run the generator
                f"./{tiramisu_program.name}.out",
                # compile the wrapper
                f"$CXX -shared -o {tiramisu_program.name}.o.so {tiramisu_program.name}.o",
            ]

        if tiramisu_program.wrapper_obj:
            # write object file to disk
            with open(
                os.path.join(
                    Config.config.tiramisu.workspace, f"{tiramisu_program.name}_wrapper"
                ),
                "wb",
            ) as f:
                f.write(tiramisu_program.wrapper_obj)

            # make it executable
            shell_script += [
                f"chmod +x {tiramisu_program.name}_wrapper",
            ]
        else:
            if Config.config.tiramisu.is_new_tiramisu:
                shell_script += [
                    f"$CXX -std=c++17 -fno-rtti -I$TIRAMISU_ROOT/include -I$TIRAMISU_ROOT/3rdParty/Halide/install/include -I$TIRAMISU_ROOT/3rdParty/isl/include/ -I$TIRAMISU_ROOT/benchmarks -L$TIRAMISU_ROOT/build -L$TIRAMISU_ROOT/3rdParty/Halide/install/lib64 -L$TIRAMISU_ROOT/3rdParty/isl/build/lib -o {tiramisu_program.name}_wrapper -ltiramisu -lHalide -ldl -lpthread -fopenmp -lm -Wl,-rpath,$TIRAMISU_ROOT/build {tiramisu_program.name}_wrapper.cpp ./{tiramisu_program.name}.o.so -ltiramisu -lHalide -ldl -lpthread -fopenmp -lm -lisl",
                ]
            else:
                shell_script += [
                    f"$CXX -std=c++11 -fno-rtti -I$TIRAMISU_ROOT/include -I$TIRAMISU_ROOT/3rdParty/Halide/include -I$TIRAMISU_ROOT/3rdParty/isl/include/ -I$TIRAMISU_ROOT/benchmarks -L$TIRAMISU_ROOT/build -L$TIRAMISU_ROOT/3rdParty/Halide/lib/ -L$TIRAMISU_ROOT/3rdParty/isl/build/lib -o {tiramisu_program.name}_wrapper -ltiramisu -lHalide -ldl -lpthread -lm -Wl,-rpath,$TIRAMISU_ROOT/build {tiramisu_program.name}_wrapper.cpp ./{tiramisu_program.name}.o.so -ltiramisu -lHalide -ldl -lpthread -lm -lisl",
                ]

        try:
            # add env vars at top
            envs = []
            for key, val in Config.config.env_vars.__dict__.items():
                envs.append(f"export {key}={val}")
            shell_script = envs + shell_script

            compiler = subprocess.run(
                [" && ".join(shell_script)],
                capture_output=True,
                text=True,
                shell=True,
                check=True,
            )
            run_script = [
                # cd to the workspace
                f"cd {Config.config.tiramisu.workspace}",
                # #  set the env variables
                "export DYNAMIC_RUNS=0",
                "export MAX_RUNS=5",
                "export NB_EXEC=5",
                # run the wrapper
                f"./{tiramisu_program.name}_wrapper"
                # # Clean generated files
                # f"rm {tiramisu_program.name}*",
            ]

            run_script = envs + run_script

            compiler = subprocess.run(
                [" ; ".join(run_script)],
                capture_output=True,
                text=True,
                shell=True,
                check=True,
            )

            subprocess.run(
                [f"rm {output_path}*"],
                capture_output=True,
                text=True,
                shell=True,
                check=True,
            )

            numbers = compiler.stdout.split(" ")[:-1]
            for i in range(len(numbers)):
                numbers[i] = float(numbers[i])
            if numbers:
                execution_time = min(numbers)
        except subprocess.CalledProcessError as e:
            logging.error(
                f"{tiramisu_program.name} : Process terminated with error code: {e.returncode}"
            )
            logging.error(f"Error output: {e.stderr}")
            logging.error(f"Output: {e.stdout}")
            logging.error(cpp_code)
            logging.error(shell_script)
            try:
                subprocess.run(
                    [f"rm {output_path}*"],
                    capture_output=True,
                    text=True,
                    shell=True,
                    check=True,
                )
            except:
                pass
            return None
        except Exception as e:
            pass

        return execution_time
