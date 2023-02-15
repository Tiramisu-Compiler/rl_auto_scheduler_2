#include <tiramisu/tiramisu.h> 
#include <tiramisu/auto_scheduler/evaluator.h>
#include <tiramisu/auto_scheduler/search_method.h>

using namespace tiramisu;

int main(int argc, char **argv){                
	tiramisu::init("function000014");
	var i0("i0", 1, 97), i1("i1", 0, 32), i0_p1("i0_p1", 0, 98);
	input input01("input01", {i0_p1}, p_float64);
	computation comp00("comp00", {i0,i1}, input01(i0) + input01(i0 - 1) + input01(i0 + 1));
	buffer buf00("buf00", {97,32}, p_float64, a_output);
	buffer buf01("buf01", {98}, p_float64, a_input);
	input01.store_in(&buf01);
	comp00.store_in(&buf00);
	tiramisu::codegen({&buf00,&buf01}, "function000014.o"); 
	return 0; 
}