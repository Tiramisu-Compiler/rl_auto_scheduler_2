#include <tiramisu/tiramisu.h> 
#include <tiramisu/auto_scheduler/evaluator.h>
#include <tiramisu/auto_scheduler/search_method.h>

using namespace tiramisu;

int main(int argc, char **argv){                
	tiramisu::init("function000030");
	var i0("i0", 0, 768), i1("i1", 0, 1280);
	input icomp00("icomp00", {i1}, p_float64);
	computation comp00("comp00", {i0,i1},  p_float64);
	comp00.set_expression(icomp00(i1)*icomp00(i1));
	buffer buf00("buf00", {1280}, p_float64, a_output);
	icomp00.store_in(&buf00);
	comp00.store_in(&buf00, {i1});
	tiramisu::codegen({&buf00}, "function000030.o"); 
	return 0; 
}