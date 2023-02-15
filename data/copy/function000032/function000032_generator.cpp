#include <tiramisu/tiramisu.h> 
#include <tiramisu/auto_scheduler/evaluator.h>
#include <tiramisu/auto_scheduler/search_method.h>

using namespace tiramisu;

int main(int argc, char **argv){                
	tiramisu::init("function000032");
	var i0("i0", 0, 32), i1("i1", 0, 32);
	input icomp00("icomp00", {i1}, p_float64);
	input input01("input01", {i0,i1}, p_float64);
	computation comp00("comp00", {i0,i1},  p_float64);
	comp00.set_expression(icomp00(i1) - input01(i0, i1)/icomp00(i1));
	buffer buf00("buf00", {32}, p_float64, a_output);
	buffer buf01("buf01", {32,32}, p_float64, a_input);
	icomp00.store_in(&buf00);
	input01.store_in(&buf01);
	comp00.store_in(&buf00, {i1});
	tiramisu::codegen({&buf00,&buf01}, "function000032.o"); 
	return 0; 
}