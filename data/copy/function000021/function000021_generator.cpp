#include <tiramisu/tiramisu.h> 
#include <tiramisu/auto_scheduler/evaluator.h>
#include <tiramisu/auto_scheduler/search_method.h>

using namespace tiramisu;

int main(int argc, char **argv){                
	tiramisu::init("function000021");
	var i0("i0", 1, 33), i1("i1", 0, 96), i2("i2", 0, 32), i2_p1("i2_p1", 0, 33), i0_p0("i0_p0", 0, 33);
	input icomp00("icomp00", {i1,i2}, p_float64);
	input input01("input01", {i0_p0,i1,i2_p1}, p_float64);
	input icomp01("icomp01", {i0_p0,i1,i2}, p_float64);
	computation comp00("comp00", {i0,i1,i2},  p_float64);
	comp00.set_expression(icomp00(i1, i2)*input01(i0, i1, i2 + 1));
	computation comp01("comp01", {i0,i1,i2},  p_float64);
	comp01.set_expression(expr(2.000)*icomp00(i1, i2) + icomp01(i0 - 1, i1, i2) + input01(i0, i1, i2) + 7.860);
	comp00.then(comp01, i2);
	buffer buf00("buf00", {96,32}, p_float64, a_output);
	buffer buf01("buf01", {33,96,33}, p_float64, a_input);
	buffer buf02("buf02", {33,96,32}, p_float64, a_output);
	icomp00.store_in(&buf00);
	input01.store_in(&buf01);
	icomp01.store_in(&buf02);
	comp00.store_in(&buf00, {i1,i2});
	comp01.store_in(&buf02);
	tiramisu::codegen({&buf00,&buf01,&buf02}, "function000021.o"); 
	return 0; 
}