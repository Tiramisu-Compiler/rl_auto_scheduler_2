#include <tiramisu/tiramisu.h> 
#include <tiramisu/auto_scheduler/evaluator.h>
#include <tiramisu/auto_scheduler/search_method.h>

using namespace tiramisu;

int main(int argc, char **argv){                
	tiramisu::init("function000023");
	var i0("i0", 0, 64), i1("i1", 0, 128), i2("i2", 0, 128);
	input icomp00("icomp00", {i1,i2}, p_float64);
	input input01("input01", {i2,i1,i0}, p_float64);
	input input02("input02", {i0,i1,i2}, p_float64);
	computation comp00("comp00", {i0,i1,i2}, input01(i2, i1, i0) + input02(i0, i1, i2));
	buffer buf00("buf00", {128,128}, p_float64, a_output);
	buffer buf01("buf01", {128,128,64}, p_float64, a_input);
	buffer buf02("buf02", {64,128,128}, p_float64, a_input);
	icomp00.store_in(&buf00);
	input01.store_in(&buf01);
	input02.store_in(&buf02);
	comp00.store_in(&buf00, {i1,i2});
	tiramisu::codegen({&buf00,&buf01,&buf02}, "function000023.o"); 
	return 0; 
}