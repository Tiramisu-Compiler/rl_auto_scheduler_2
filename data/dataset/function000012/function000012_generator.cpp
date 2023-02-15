#include <tiramisu/tiramisu.h> 
#include <tiramisu/auto_scheduler/evaluator.h>
#include <tiramisu/auto_scheduler/search_method.h>

using namespace tiramisu;

int main(int argc, char **argv){                
	tiramisu::init("function000012");
	var i0("i0", 1, 129), i1("i1", 0, 64), i2("i2", 0, 128), i0_p1("i0_p1", 0, 130);
	input icomp00("icomp00", {i0_p1,i1,i2}, p_float64);
	computation comp00("comp00", {i0,i1,i2},  p_float64);
	comp00.set_expression(expr(5.940)*icomp00(i0, i1, i2) + icomp00(i0 - 1, i1, i2) + icomp00(i0 + 1, i1, i2));
	buffer buf00("buf00", {130,64,128}, p_float64, a_output);
	icomp00.store_in(&buf00);
	comp00.store_in(&buf00);
	tiramisu::codegen({&buf00}, "function000012.o"); 
	return 0; 
}