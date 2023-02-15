#include <tiramisu/tiramisu.h> 
#include <tiramisu/auto_scheduler/evaluator.h>
#include <tiramisu/auto_scheduler/search_method.h>

using namespace tiramisu;

int main(int argc, char **argv){                
	tiramisu::init("function000027");
	var i0("i0", 0, 256), i1("i1", 1, 257), i1_p1("i1_p1", 0, 258);
	input input01("input01", {i0,i1_p1}, p_float64);
	input icomp00("icomp00", {i1_p1}, p_float64);
	input icomp01("icomp01", {i1_p1}, p_float64);
	computation comp00("comp00", {i0,i1}, expr(5.860)*input01(i0, i1)*input01(i0, i1 - 1) + expr(5.860)*input01(i0, i1 + 1));
	computation comp01("comp01", {i0,i1},  p_float64);
	comp01.set_expression(icomp00(i1)*icomp01(i1 + 1) - icomp00(i1 - 1)*icomp00(i1 + 1) + icomp01(i1) - icomp01(i1 - 1) + 1.780);
	comp00.then(comp01, i1);
	buffer buf00("buf00", {258}, p_float64, a_output);
	buffer buf01("buf01", {256,258}, p_float64, a_input);
	buffer buf02("buf02", {258}, p_float64, a_output);
	input01.store_in(&buf01);
	icomp00.store_in(&buf00);
	icomp01.store_in(&buf02);
	comp00.store_in(&buf00, {i1});
	comp01.store_in(&buf02, {i1});
	tiramisu::codegen({&buf00,&buf01,&buf02}, "function000027.o"); 
	return 0; 
}