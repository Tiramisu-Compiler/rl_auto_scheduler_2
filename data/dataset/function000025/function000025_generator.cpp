#include <tiramisu/tiramisu.h> 
#include <tiramisu/auto_scheduler/evaluator.h>
#include <tiramisu/auto_scheduler/search_method.h>

using namespace tiramisu;

int main(int argc, char **argv){                
	tiramisu::init("function000025");
	var i0("i0", 1, 321), i1("i1", 1, 1537), i0_p0("i0_p0", 0, 321), i1_p0("i1_p0", 0, 1537), i0_p1("i0_p1", 0, 322), i1_p1("i1_p1", 0, 1538);
	input input01("input01", {i0_p1,i1_p1}, p_float64);
	input icomp00("icomp00", {i0_p0,i1_p0}, p_float64);
	computation comp00("comp00", {i0,i1}, input01(i0, i1)*input01(i0 + 1, i1)*input01(i0 + 1, i1 + 1) + input01(i0, i1 - 1) - input01(i0, i1 + 1) + input01(i0 - 1, i1) + input01(i0 - 1, i1 - 1) - input01(i0 - 1, i1 + 1)*input01(i0 + 1, i1 - 1) - 3.750);
	computation comp01("comp01", {i0,i1}, icomp00(i0, i1));
	comp00.then(comp01, i1);
	buffer buf00("buf00", {321,1537}, p_float64, a_output);
	buffer buf01("buf01", {322,1538}, p_float64, a_input);
	buffer buf02("buf02", {1537}, p_float64, a_output);
	input01.store_in(&buf01);
	icomp00.store_in(&buf00);
	comp00.store_in(&buf00);
	comp01.store_in(&buf02, {i1});
	tiramisu::codegen({&buf00,&buf01,&buf02}, "function000025.o"); 
	return 0; 
}