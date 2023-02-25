import  subprocess


class CompilingService():
    @classmethod
    def compile_legality(cls,schedule_object, optims_list : list ,comps=None):
        tiramisu_program=schedule_object.prog
        first_comp=schedule_object.comps[0]
        output_path = tiramisu_program.func_folder+ tiramisu_program.name+ 'legal'
        # Add code to the original file to get legality result
        legality_check_lines = '''
            prepare_schedules_for_legality_checks();
            perform_full_dependency_analysis();
            bool is_legal=true;
            '''
        for optim in optims_list:
            if optim.type == 'Parallelization':
                legality_check_lines += '''
                is_legal &= loop_parallelization_is_legal(''' + str(optim.params_list[0]) + ''', {&''' + first_comp + '''});'''
            elif optim.type == 'Unrolling':
                legality_check_lines += '''is_legal &= loop_unrolling_is_legal(''' + str(optim.params_list[comps[0]][0]) + ''', {''' + ", ".join([f"&{comp}" for comp in comps]) + '''});'''
            legality_check_lines += optim.tiramisu_optim_str + '\n'    
        
        legality_check_lines += '''
            is_legal &= check_legality_of_function();   
            std::cout << is_legal;
            '''
        # Paste the lines responsable of checking legality of schedule in the cpp file
        cpp_code = tiramisu_program.original_str.replace(tiramisu_program.code_gen_line,legality_check_lines)

        return cls.run_cpp_code(cpp_code=cpp_code,output_path=output_path)   




    @classmethod 
    def compile_annotations(cls,tiramisu_program ):
        # TODO : add getting tree structure object from here 
        output_path = tiramisu_program.func_folder+ tiramisu_program.name+ 'annot'
        # Add code to the original file to get json annotations 
        get_json_lines = '''
            auto ast = tiramisu::auto_scheduler::syntax_tree(tiramisu::global::get_implicit_function());
            std::string program_json = tiramisu::auto_scheduler::evaluate_by_learning_model::get_program_json(ast);
            std::cout << program_json;
            '''
        # Paste the lines responsable of generating the program json tree in the cpp file
        cpp_code = tiramisu_program.original_str.replace(tiramisu_program.code_gen_line,get_json_lines)
        return cls.run_cpp_code(cpp_code=cpp_code,output_path=output_path)

    @classmethod
    def run_cpp_code(cls,cpp_code : str,output_path : str):
        shell_script = [
            # Compile intermidiate tiramisu file
            "$CXX -I$TIRAMISU_ROOT/3rdParty/Halide/include -I$TIRAMISU_ROOT/include -I$TIRAMISU_ROOT/3rdParty/isl/include  -Wl,--no-as-needed -ldl -g -fno-rtti   -lpthread -std=c++11 -O0 -o {}.o -c -x c++ -".format(output_path),
            # Link generated file with executer
            "$CXX -Wl,--no-as-needed -ldl -g -fno-rtti -lpthread -std=c++11 -O0 {}.o -o {}.out   -L$TIRAMISU_ROOT/build  -L$TIRAMISU_ROOT/3rdParty/Halide/lib  -L$TIRAMISU_ROOT/3rdParty/isl/build/lib  -Wl,-rpath,$TIRAMISU_ROOT/build:$TIRAMISU_ROOT/3rdParty/Halide/lib:$TIRAMISU_ROOT/3rdParty/isl/build/lib -ltiramisu -ltiramisu_auto_scheduler -lHalide -lisl".format(output_path,output_path),
            # Run the program
            "{}.out".format(output_path),
            # Clean generated files
            "rm {}*".format(output_path)
            ]
        try :
           
            compiler = subprocess.run(["\n".join(shell_script)], input=cpp_code, capture_output=True, text=True,shell=True,check=True)
            return compiler.stdout
        except subprocess.CalledProcessError as e:
            print("Process terminated with error code", e.returncode)
            print("Error output:", e.stderr)
            return "0"
        except Exception as e :
            print(e)
            return "0"