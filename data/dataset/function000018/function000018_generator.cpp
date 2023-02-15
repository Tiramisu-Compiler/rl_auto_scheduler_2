#include <tiramisu/tiramisu.h> 
#include <tiramisu/auto_scheduler/evaluator.h>
#include <tiramisu/auto_scheduler/search_method.h>

using namespace tiramisu;

int main(int argc, char **argv){                
	tiramisu::init("function000018");
	var i0("i0", 0, 32), i1("i1", 0, 32), i2("i2", 0, 32);
	input icomp00("icomp00", {i0,i1,i2}, p_float64);
	input input01("input01", {i0,i1,i2}, p_float64);
	input input02("input02", {i0,i2}, p_float64);
	input input03("input03", {i0,i1,i2}, p_float64);
	computation comp00("comp00", {i0,i1,i2},  p_float64);
	comp00.set_expression((icomp00(i0, i1, i2) + input02(i0, i2) + input03(i0, i1, i2) - 3.250)*input01(i0, i1, i2));
	buffer buf00("buf00", {32,32,32}, p_float64, a_output);
	buffer buf01("buf01", {32,32,32}, p_float64, a_input);
	buffer buf02("buf02", {32,32}, p_float64, a_input);
	buffer buf03("buf03", {32,32,32}, p_float64, a_input);
	icomp00.store_in(&buf00);
	input01.store_in(&buf01);
	input02.store_in(&buf02);
	input03.store_in(&buf03);
	comp00.store_in(&buf00);
	tiramisu::codegen({&buf00,&buf01,&buf02,&buf03}, "function000018.o"); 
	return 0; 
}