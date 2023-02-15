import subprocess
import time

from utils.config.config import Config

Config.init()

p = subprocess.Popen(["bash"], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

def run_command(p, command):
    p.stdin.write(command.encode())
    p.stdin.write(b'\n')
    p.stdin.flush()
    return p.stdout.readline().decode().strip()




cpp_code = """
#include <iostream>
#include <tiramisu/tiramisu.h> 
int main() {
    std::cout << "Hello, World!" << std::endl;
    return 0;
}
"""

# shell_script = """
# $CXX -I$TIRAMISU_ROOT/3rdParty/Halide/include -I$TIRAMISU_ROOT/include -I$TIRAMISU_ROOT/3rdParty/isl/include  -Wl,--no-as-needed -ldl -g -fno-rtti   -lpthread -std=c++11 -O0 -o hello.o -c -x c++ -
# """

compile_linker = """
$CXX -Wl,--no-as-needed -ldl -g -fno-rtti   -lpthread -std=c++11 -O0 {}.o -o ./{}.out   -L$TIRAMISU_ROOT/build  -L$TIRAMISU_ROOT/3rdParty/Halide/lib  -L$TIRAMISU_ROOT/3rdParty/isl/build/lib  -Wl,-rpath,$TIRAMISU_ROOT/build:$TIRAMISU_ROOT/3rdParty/Halide/lib:$TIRAMISU_ROOT/3rdParty/isl/build/lib -ltiramisu -ltiramisu_auto_scheduler -lHalide -lisl
""".format("func","func")

# # Compile the C++ code
# compiler = subprocess.Popen(["/bin/bash", "-c", shell_script], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
# compiler_output,error = compiler.communicate(input=cpp_code.encode('utf-8'))
# run_command(compiler,compile_linker)

# if compiler.returncode == 0:
#     # Run the compiled program
#     program = subprocess.Popen(["./hello.out"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
#     program_output = program.communicate()[0].decode("utf-8")
# else:
#     print("Compilation failed")
#     print(error.decode('utf-8'))

shell_script = [
    "$CXX -I$TIRAMISU_ROOT/3rdParty/Halide/include -I$TIRAMISU_ROOT/include -I$TIRAMISU_ROOT/3rdParty/isl/include  -Wl,--no-as-needed -ldl -g -fno-rtti   -lpthread -std=c++11 -O0 -o {}.o -c -x c++ -".format("func"),
    compile_linker,
    "./func.out",
    "rm func*"
]
compiler = subprocess.Popen(["/bin/bash", "-c", "\n".join(shell_script)], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
compiler_output,error = compiler.communicate(input=cpp_code.encode('utf-8'))
print(compiler.poll())
print(compiler_output.decode("utf-8"))
