#include <tiramisu/tiramisu.h> 
#include <tiramisu/auto_scheduler/evaluator.h>
#include <tiramisu/auto_scheduler/search_method.h>

using namespace tiramisu;

int main(int argc, char **argv){                
	tiramisu::init("function000015");
	var i0("i0", 0, 192), i1("i1", 0, 192);
	input icomp00("icomp00", {i0}, p_float64);
	computation comp00("comp00", {i0,i1},  p_float64);
	comp00.set_expression(icomp00(i0)*icomp00(i0));
	buffer buf00("buf00", {192}, p_float64, a_output);
	icomp00.store_in(&buf00);
	comp00.store_in(&buf00, {i0});
	tiramisu::codegen({&buf00}, "function000015.o"); 
	return 0; 
}