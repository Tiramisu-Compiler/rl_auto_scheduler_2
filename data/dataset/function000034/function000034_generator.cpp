#include <tiramisu/tiramisu.h> 
#include <tiramisu/auto_scheduler/evaluator.h>
#include <tiramisu/auto_scheduler/search_method.h>

using namespace tiramisu;

int main(int argc, char **argv){                
	tiramisu::init("function000034");
	var i0("i0", 1, 257), i1("i1", 1, 257), i2("i2", 1, 257), i0_p0("i0_p0", 0, 257), i1_p0("i1_p0", 0, 257), i0_p1("i0_p1", 0, 258), i1_p1("i1_p1", 0, 258), i2_p1("i2_p1", 0, 258);
	input icomp00("icomp00", {i0_p0,i1_p0}, p_float64);
	input input01("input01", {i0_p1,i1_p1,i2_p1}, p_float64);
	computation comp00("comp00", {i0,i1,i2},  p_float64);
	comp00.set_expression(((icomp00(i0, i1)*input01(i0, i1, i2) + input01(i0, i1, i2 + 1)*input01(i0 + 1, i1, i2))*input01(i0, i1, i2 - 1)*input01(i0, i1 + 1, i2) - input01(i0, i1 - 1, i2)*input01(i0 - 1, i1, i2)*input01(i0 + 1, i1, i2))/input01(i0 + 1, i1, i2));
	buffer buf00("buf00", {257,257}, p_float64, a_output);
	buffer buf01("buf01", {258,258,258}, p_float64, a_input);
	icomp00.store_in(&buf00);
	input01.store_in(&buf01);
	comp00.store_in(&buf00, {i0,i1});
	tiramisu::codegen({&buf00,&buf01}, "function000034.o"); 
	return 0; 
}