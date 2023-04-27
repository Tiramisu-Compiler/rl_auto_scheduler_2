import subprocess
import re
from typing import List
from env_api.scheduler.models.action import Parallelization, Unrolling
from env_api.scheduler.models.schedule import Schedule
from config.config import Config
from env_api.core.models.optim_cmd import OptimizationCommand


class CompilingService():
    @classmethod
    def compile_legality(cls, schedule_object: Schedule, optims_list: List[OptimizationCommand]):
        tiramisu_program = schedule_object.prog
        output_path = Config.config.tiramisu.workspace + tiramisu_program.name + 'legal'

        cpp_code = cls.get_legality_code(schedule_object=schedule_object,
                                         optims_list=optims_list)
        return cls.run_cpp_code(cpp_code=cpp_code, output_path=output_path)

    @classmethod
    def get_legality_code(cls, schedule_object: Schedule, optims_list: List[OptimizationCommand]):
        tiramisu_program = schedule_object.prog
        # Add code to the original file to get legality result
        legality_check_lines = '''\n\tprepare_schedules_for_legality_checks();\n\tperform_full_dependency_analysis();\n\tbool is_legal=true;'''
        for optim in optims_list:
            if isinstance(optim.action, Parallelization):
                legality_check_lines += '''\n\tis_legal &= loop_parallelization_is_legal(''' + str(
                    optim.params_list[0]
                ) + ''', {&''' + optim.action.comps[0] + '''});\n'''
            elif isinstance(optim.action, Unrolling):
                legality_check_lines += '''\n\tis_legal &= loop_unrolling_is_legal(''' + str(
                    optim.action.params[0]) + ''', {''' + ", ".join(
                        [f"&{comp}" for comp in optim.action.comps]) + '''});'''
            legality_check_lines += optim.tiramisu_optim_str + '\n'

        legality_check_lines += '''
            is_legal &= check_legality_of_function();   
            std::cout << is_legal;
            '''
        # Paste the lines responsable of checking legality of schedule in the cpp file
        cpp_code = tiramisu_program.original_str.replace(
            tiramisu_program.code_gen_line, legality_check_lines)
        return cpp_code

    @classmethod
    def compile_annotations(cls, tiramisu_program):
        # TODO : add getting tree structure object from executing the file instead of building it
        output_path = Config.config.tiramisu.workspace + tiramisu_program.name + 'annot'
        # Add code to the original file to get json annotations

        if Config.config.tiramisu.is_new_tiramisu:
            get_json_lines = '''
                auto ast = tiramisu::auto_scheduler::syntax_tree(tiramisu::global::get_implicit_function(), {});
                std::string program_json = tiramisu::auto_scheduler::evaluate_by_learning_model::get_program_json(ast);
                std::cout << program_json;
                '''
        else:
            get_json_lines = '''
                auto ast = tiramisu::auto_scheduler::syntax_tree(tiramisu::global::get_implicit_function());
                std::string program_json = tiramisu::auto_scheduler::evaluate_by_learning_model::get_program_json(ast);
                std::cout << program_json;
                '''
        # Paste the lines responsable of generating the program json tree in the cpp file
        cpp_code = tiramisu_program.original_str.replace(
            tiramisu_program.code_gen_line, get_json_lines)
        return cls.run_cpp_code(cpp_code=cpp_code, output_path=output_path)

    @classmethod
    def run_cpp_code(cls, cpp_code: str, output_path: str):
        if Config.config.tiramisu.is_new_tiramisu:
            # Making the tiramisu root path explicit to the env
            shell_script = [
                # Compile intermidiate tiramisu file
                "$CXX -I$TIRAMISU_ROOT/3rdParty/Halide/install/include -I$TIRAMISU_ROOT/include -I$TIRAMISU_ROOT/3rdParty/isl/include  -Wl,--no-as-needed -ldl -g -fno-rtti   -lpthread -std=c++17 -O0 -o {}.o -c -x c++ -"
                .format(output_path),
                # Link generated file with executer
                "$CXX -Wl,--no-as-needed -ldl -g -fno-rtti -lpthread -std=c++17 -O0 {}.o -o {}.out   -L$TIRAMISU_ROOT/build  -L$TIRAMISU_ROOT/3rdParty/Halide/install/lib64  -L$TIRAMISU_ROOT/3rdParty/isl/build/lib  -Wl,-rpath,$TIRAMISU_ROOT/build:$TIRAMISU_ROOT/3rdParty/Halide/install/lib64:$TIRAMISU_ROOT/3rdParty/isl/build/lib -ltiramisu -ltiramisu_auto_scheduler -lHalide -lisl"
                .format(output_path, output_path),
                # Run the program
                "{}.out".format(output_path),
                # Clean generated files
                "rm {}*".format(output_path)
            ]
        else:
            shell_script = [
                # Compile intermidiate tiramisu file
                "$CXX -I$TIRAMISU_ROOT/3rdParty/Halide/include -I$TIRAMISU_ROOT/include -I$TIRAMISU_ROOT/3rdParty/isl/include  -Wl,--no-as-needed -ldl -g -fno-rtti   -lpthread -std=c++11 -O0 -o {}.o -c -x c++ -"
                .format(output_path),
                # Link generated file with executer
                "$CXX -Wl,--no-as-needed -ldl -g -fno-rtti -lpthread -std=c++11 -O0 {}.o -o {}.out   -L$TIRAMISU_ROOT/build  -L$TIRAMISU_ROOT/3rdParty/Halide/lib  -L$TIRAMISU_ROOT/3rdParty/isl/build/lib  -Wl,-rpath,$TIRAMISU_ROOT/build:$TIRAMISU_ROOT/3rdParty/Halide/lib:$TIRAMISU_ROOT/3rdParty/isl/build/lib -ltiramisu -ltiramisu_auto_scheduler -lHalide -lisl"
                .format(output_path, output_path),
                # Run the program
                "{}.out".format(output_path),
                # Clean generated files
                "rm {}*".format(output_path)
            ]
        try:
            compiler = subprocess.run(["\n".join(shell_script)],
                                      input=cpp_code,
                                      capture_output=True,
                                      text=True,
                                      shell=True,
                                      check=True)
            return compiler.stdout if compiler.stdout != '' else "0"
        except subprocess.CalledProcessError as e:
            print("Process terminated with error code", e.returncode)
            print("Error output:", e.stderr)
            return "0"
        except Exception as e:
            print(e)
            return "0"

    @classmethod
    def call_skewing_solver(cls, schedule_object, optim_list, params):
        legality_cpp_code = cls.get_legality_code(schedule_object, optim_list)
        to_replace = re.findall(r'std::cout << is_legal;',
                                legality_cpp_code)[0]
        header = """
        function * fct = tiramisu::global::get_implicit_function();\n"""
        legality_cpp_code = legality_cpp_code.replace(
            "is_legal &= check_legality_of_function();", "")
        legality_cpp_code = legality_cpp_code.replace("bool is_legal=true;",
                                                      "")
        legality_cpp_code = re.sub(
            r'is_legal &= loop_parallelization_is_legal.*\n', "",
            legality_cpp_code)
        legality_cpp_code = re.sub(r'is_legal &= loop_unrolling_is_legal.*\n',
                                   "", legality_cpp_code)

        solver_lines = header + "\n\tauto auto_skewing_result = fct->skewing_local_solver({" + ", ".join(
            [f"&{comp}" for comp in schedule_object.comps
             ]) + "}" + ",{},{},1);\n".format(*params)

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
        output_path = Config.config.tiramisu.workspace + \
            schedule_object.prog.name + 'skew_solver'
        result_str = cls.run_cpp_code(cpp_code=solver_code,
                                      output_path=output_path)
        if not result_str:
            return None
            # Refer to function run_cpp_code to see from where the "0" comes from
        elif result_str == '0':
            return None
        result_str = result_str.split(",")
        # Skewing Solver returns 3 solutions in form of tuples, the first tuple is for outer parallelism ,
        # second is for inner parallelism , and last one is for locality, we are going to use the first preferably
        # if availble , else , we are going to use the scond one if available, this policy of choosing factors may change
        # in later versions!
        # The compiler in our case returns a tuple of type : (fac0,fac1,fac2,fac3,fac4,fac5) each 2 factors represent the
        # solutions mentioned above
        if (result_str[0] != "None"):
            # Means we have a solution for outer parallelism
            fac1 = int(result_str[0])
            fac2 = int(result_str[1])
            return fac1, fac2
        if (result_str[2] != "None"):
            # Means we have a solution for inner parallelism
            fac1 = int(result_str[2])
            fac2 = int(result_str[3])
            return fac1, fac2
        else:
            return None

    @classmethod
    def get_schedule_code(cls, schedule_object: Schedule):
        optims_list: List[OptimizationCommand] = schedule_object.schedule_list
        tiramisu_program = schedule_object.prog
        # Add code to the original file to get the schedule code
        schedule_code = ''
        for optim in optims_list:
            schedule_code += optim.tiramisu_optim_str + '\n'

        # Add code gen line to the schedule code
        schedule_code += '\n\t' + tiramisu_program.code_gen_line + '\n'
        # Paste the lines responsable of checking legality of schedule in the cpp file
        cpp_code = tiramisu_program.original_str.replace(
            tiramisu_program.code_gen_line, schedule_code)
        cpp_code = cpp_code.replace(
            f"// {tiramisu_program.wrapper_str}", tiramisu_program.wrapper_str)
        return cpp_code

    @classmethod
    def write_cpp_code(cls, cpp_code: str, output_path: str):
        with open(output_path + '.cpp', 'w') as f:
            f.write(cpp_code)
