#include <tiramisu/tiramisu.h> 
#include <tiramisu/auto_scheduler/evaluator.h>
#include <tiramisu/auto_scheduler/search_method.h>

using namespace tiramisu;

int main(int argc, char **argv){                
	tiramisu::init("function000011");
	var i0("i0", 1, 321), i1("i1", 0, 128), i0_p1("i0_p1", 0, 322);
	input icomp00("icomp00", {i0_p1}, p_float64);
	computation comp00("comp00", {i0,i1},  p_float64);
	comp00.set_expression((icomp00(i0) + expr(2.030)*icomp00(i0 + 1))/icomp00(i0 - 1));
	buffer buf00("buf00", {322}, p_float64, a_output);
	icomp00.store_in(&buf00);
	comp00.store_in(&buf00, {i0});
	tiramisu::codegen({&buf00}, "function000011.o"); 
	return 0; 
}