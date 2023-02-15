#include <tiramisu/tiramisu.h> 
#include <tiramisu/auto_scheduler/evaluator.h>
#include <tiramisu/auto_scheduler/search_method.h>

using namespace tiramisu;

int main(int argc, char **argv){                
	tiramisu::init("function000033");
	var i0("i0", 0, 256), i1("i1", 1, 513), i2("i2", 1, 2049), i1_p0("i1_p0", 0, 513), i1_p1("i1_p1", 0, 514), i2_p1("i2_p1", 0, 2050);
	input input01("input01", {i1_p1,i2_p1}, p_float64);
	input icomp01("icomp01", {i0,i1_p0}, p_float64);
	computation comp00("comp00", {i0,i1,i2}, input01(i1, i2)/input01(i1, i2 - 1) + input01(i1, i2 + 1));
	computation comp01("comp01", {i0,i1,i2},  p_float64);
	comp01.set_expression(icomp01(i0, i1) - input01(i1, i2) + input01(i1, i2 - 1) + input01(i1, i2 + 1) + input01(i1 - 1, i2) - input01(i1 + 1, i2));
	comp00.then(comp01, i2);
	buffer buf00("buf00", {256,513}, p_float64, a_output);
	buffer buf01("buf01", {514,2050}, p_float64, a_input);
	input01.store_in(&buf01);
	icomp01.store_in(&buf00);
	comp00.store_in(&buf00, {i0,i1});
	comp01.store_in(&buf00, {i0,i1});
	tiramisu::codegen({&buf00,&buf01}, "function000033.o"); 
	return 0; 
}