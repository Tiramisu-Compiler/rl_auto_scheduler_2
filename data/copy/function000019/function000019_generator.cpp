#include <tiramisu/tiramisu.h> 
#include <tiramisu/auto_scheduler/evaluator.h>
#include <tiramisu/auto_scheduler/search_method.h>

using namespace tiramisu;

int main(int argc, char **argv){                
	tiramisu::init("function000019");
	var i0("i0", 1, 1025), i1("i1", 0, 1024), i0_p1("i0_p1", 0, 1026);
	input icomp00("icomp00", {i0_p1}, p_float64);
	computation comp00("comp00", {i0,i1},  p_float64);
	comp00.set_expression(expr(0.0) - expr(0.890)*icomp00(i0)*icomp00(i0)*icomp00(i0 + 1) + icomp00(i0) - icomp00(i0 - 1) + 3.010);
	buffer buf00("buf00", {1026}, p_float64, a_output);
	icomp00.store_in(&buf00);
	comp00.store_in(&buf00, {i0});
	tiramisu::codegen({&buf00}, "function000019.o"); 
	return 0; 
}