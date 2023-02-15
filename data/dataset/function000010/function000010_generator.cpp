#include <tiramisu/tiramisu.h> 
#include <tiramisu/auto_scheduler/evaluator.h>
#include <tiramisu/auto_scheduler/search_method.h>

using namespace tiramisu;

int main(int argc, char **argv){                
	tiramisu::init("function000010");
	var i0("i0", 0, 48), i1("i1", 0, 32), i2("i2", 0, 256);
	input input01("input01", {i1,i2}, p_float64);
	input input02("input02", {i0,i1,i2}, p_float64);
	computation comp00("comp00", {i0,i1,i2}, input01(i1, i2) + input02(i0, i1, i2));
	buffer buf00("buf00", {48,32}, p_float64, a_output);
	buffer buf01("buf01", {32,256}, p_float64, a_input);
	buffer buf02("buf02", {48,32,256}, p_float64, a_input);
	input01.store_in(&buf01);
	input02.store_in(&buf02);
	comp00.store_in(&buf00, {i0,i1});
	tiramisu::codegen({&buf00,&buf01,&buf02}, "function000010.o"); 
	return 0; 
}