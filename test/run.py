# import subprocess
# import time
# from utils.config.config import Config
# Config.init()

# cpp_code = """
# #include <iostream>
# #include <tiramisu/tiramisu.h> 
# int main() {
#     std::cout << "Hello, World!" << std::endl;
#     return 0;
# }
# """

# # shell_script = """
# # $CXX -I$TIRAMISU_ROOT/3rdParty/Halide/include -I$TIRAMISU_ROOT/include -I$TIRAMISU_ROOT/3rdParty/isl/include  -Wl,--no-as-needed -ldl -g -fno-rtti   -lpthread -std=c++11 -O0 -o hello.o -c -x c++ - 
# # """

# compile_linker = """
# $CXX -Wl,--no-as-needed -ldl -g -fno-rtti   -lpthread -std=c++11 -O0 hello.o -o ./hello.out   -L$TIRAMISU_ROOT/build  -L$TIRAMISU_ROOT/3rdParty/Halide/lib  -L$TIRAMISU_ROOT/3rdParty/isl/build/lib  -Wl,-rpath,$TIRAMISU_ROOT/build:$TIRAMISU_ROOT/3rdParty/Halide/lib:$TIRAMISU_ROOT/3rdParty/isl/build/lib -ltiramisu -ltiramisu_auto_scheduler -lHalide -lisl
# """

# shell_script = [
#     "$CXX -I$TIRAMISU_ROOT/3rdParty/Halide/include -I$TIRAMISU_ROOT/include -I$TIRAMISU_ROOT/3rdParty/isl/include  -Wl,--no-as-needed -ldl -g -fno-rtti   -lpthread -std=c++11 -O0 -o hello.o -c -x c++ -",
#     compile_linker,
#     "./hello.out"
# ]


# compiler = subprocess.run(["/bin/bash", "-c", "\n".join(shell_script)], input=cpp_code, capture_output=True, text=True)

# print(compiler.stdout)

import subprocess

cpp_code = """
#include <iostream>
int main() {
    std::cout << "Hello, World!" << std::endl;
    return 0;
}
"""

result = subprocess.run(["g++", "-x", "c++","-",], input=cpp_code, stdout=subprocess.PIPE, stderr=subprocess.PIPE, encoding="utf-8")
if result.returncode == 0:
    binary = result.stdout
    print(binary)
else:
    print("Compilation failed")
    print(result.stderr)